"""Text windows drawn from the corpus's token files of plain texts
(docs/training.md, "Stages").

`stages` lists the window lengths in order, each {"block_size", "steps",
"batch_size"}: its updates and its windows per update, over every process
together (batch_size is global). The windows grow from one stage to the
next, inside one step and one optimizer schedule; each window length
divides the longest. `readings` (1 or more) is how many times the step
reads its training tokens: every stage runs readings times its `steps`
(`stages`), and eval_every counts the updates of one reading, so the step
is the step of one reading stretched readings times (docs/training.md,
"Readings").

The step walks its training tokens in readings, each a pass without
replacement (`_Walk`). A window of block_size + 1 tokens starts on the last
token of the window before it in the walk, so each window predicts
block_size tokens and the walk advances by block_size. Pass q cuts each
segment of the tokens (the whole of them; with --stream, each token file)
into slots of the longest window length, from an offset of its own, and
takes the slots in an order of its own; it reads them in groups of GROUP
slots, each group's windows in an order of their own. Every draw depends
on the order's seed, the step's number in the order, q and the window
lengths only. A stage that starts inside a group starts at the next group,
so no token is predicted twice in one pass; the tokens before a segment's
offset and after its last whole slot are not read in that pass. With
--stream, the windows are read from the shards in use from the segment
of the update's first window (wpt_corpus.shards.ShardStream).

prefix_tokens, when above 0, makes the training tokens the first
prefix_tokens of the corpus's, in the token files' order, and nothing after
them: the windows, the token files copied or streamed and the texts the
step learned from are those of that start. Each size of a model series
reads its first part of the series' one corpus (docs/model.md, "Model
series"). A corpus with fewer tokens is refused before any step runs;
`wpt train` gives a step no more than a corpus built at the order's size
holds (docs/training.md, "Prefix"), so such a corpus is read
whole.

Each process takes --micro-batch windows per pass (default: all of its
share of the batch, or in a stage the runner chose a number for, per_pass,
that number) and adds passes up until the batch is complete: pass a
of process r is slice (a * world_size + r) of the batch, so the batch
holds the same windows whatever the number of processes; with split
windows, every process takes every window and trains its slice of each.

Every eval_every updates, and at the first and last update of each stage,
the model is evaluated on eval_batches batches of windows at the stage's
window length, drawn at random places of the corpus's held-out portion
(the record's `held_out` token files, read whole), which no step trains
on. A stage's evaluation windows are the same at every evaluation, and
the same for any seed and any step of an order: they depend only on the
held-out tokens, the window length, the batch size and eval_batches. A
corpus without a held-out portion is evaluated on windows of the training
tokens drawn the same way, with --stream from the shards in use, and the
step says so in the run's record (`validation`: "training", else
"held_out").
"""

import bisect
import hashlib

import numpy as np
import torch

from wpt_corpus.shards import ShardStream, prefix_shards, token_file
from wpt_training import WINDOWS, Environment, Stage
from wpt_training import parts as windows_in_parts

GIVES = WINDOWS
KINDS = ("plain",)  # the kinds of texts it reads (depot.kinds)
SETTINGS = {"stages": [], "eval_every": 200, "eval_batches": 20, "prefix_tokens": 0, "readings": 1}
STAGE = ("block_size", "steps", "batch_size")


def check(settings):
    """ValueError on stages that cannot be: none, a field missing or not a
    positive whole number, eval_every or eval_batches below 1, or readings
    below 1."""
    if not settings["stages"]:
        raise ValueError("stages: at least one stage")
    for number, stage in enumerate(settings["stages"], 1):
        if not isinstance(stage, dict) or sorted(stage) != sorted(STAGE) \
                or not all(type(stage[key]) is int and stage[key] > 0 for key in STAGE):
            raise ValueError(f"stage {number}: {{\"block_size\", \"steps\", \"batch_size\"}}, each a whole number "
                             "above 0")
    for key in ("eval_every", "eval_batches"):
        if settings[key] < 1:
            raise ValueError(f"{key}: at least 1")
    if settings["prefix_tokens"] < 0:
        raise ValueError("prefix_tokens: 0 (every training token) or more")
    if settings["readings"] < 1:
        raise ValueError(f"readings {settings['readings']}: a whole number, 1 or more")
    longest = max(stage["block_size"] for stage in settings["stages"])
    for number, stage in enumerate(settings["stages"], 1):
        if longest % stage["block_size"]:
            raise ValueError(f"stage {number}: its window length {stage['block_size']} does not divide the longest, "
                             f"{longest}, so its windows cannot share the slots of a reading")


def stages(settings) -> list[dict]:
    """The stages the step runs: each stage of its settings with readings times its updates."""
    return [{**stage, "steps": stage["steps"] * settings["readings"]} for stage in settings["stages"]]


def times_read(settings, tokens) -> float:
    """How many times the stages read `tokens` training tokens: the tokens
    their windows predict over the tokens."""
    return sum(stage["steps"] * stage["batch_size"] * stage["block_size"] for stage in stages(settings)) / tokens


def shards(record, kinds) -> list[dict]:
    """The corpus's token files of the kinds the step reads, in order."""
    return [one for kind in kinds for one in record["shards"].get(kind, [])]


def check_corpus(settings, record, dock, kinds):
    """ValueError when the corpus holds fewer tokens of the kinds the step reads than the prefix it reads."""
    held = sum(one["tokens"] for one in shards(record, kinds))
    if settings["prefix_tokens"] > held:
        raise ValueError(f"the step reads the first {settings['prefix_tokens']:,} tokens; the corpus has {held:,} "
                         f"of {', '.join(kinds)}")


def context(settings) -> int:
    """The window length the step leaves the model reading at: the last stage's."""
    return settings["stages"][-1]["block_size"]


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") & (2**63 - 1)


def _starts(length, block_size, count, *seed_parts) -> list[int]:
    """Where count random windows of block_size + 1 tokens of length tokens start, the same on every machine."""
    generator = torch.Generator().manual_seed(_seed(*seed_parts))
    return torch.randint(length - block_size - 1, (count,), generator=generator).tolist()


def _gathered(tokens, starts, block_size):
    """The windows of block_size + 1 tokens that start at starts, as one tensor."""
    return torch.from_numpy(np.stack([np.asarray(tokens[s:s + block_size + 1], dtype=np.int64) for s in starts]))


def _windows(tokens, block_size, count, *seed_parts):
    """count random windows of block_size + 1 tokens, the same on every machine."""
    return _gathered(tokens, _starts(len(tokens), block_size, count, *seed_parts), block_size)


GROUP = 64  # the slots a pass reads together, each group's windows in an order of their own


class _Walk:
    """The places of the windows of a step's walk over its training tokens,
    the same on every machine (the module's docstring). segments: (start,
    length) of each part of the tokens a slot stays inside. unit: the
    longest window length, a slot's length in tokens predicted. seeds: the
    order's seed and the step's number."""

    def __init__(self, segments, unit, seeds):
        self.segments, self.unit, self.seeds = segments, unit, seeds
        self.starts = [0]  # where each pass starts in the walk, in tokens predicted; one more than the passes known
        self.layout = None  # (pass number, the start of each of its slots in the tokens in its order, its groups)
        self.orders = {}  # the order of the windows of the group in use, by (pass, group, window length)

    def _generator(self, number, segment):
        return torch.Generator().manual_seed(_seed(*self.seeds, "reading", number, self.unit, segment))

    def _offset(self, generator, length) -> int | None:
        """Where a segment of length tokens starts its slots, or None when no slot fits."""
        if length < self.unit + 1:
            return None
        return int(torch.randint(min(self.unit, length - self.unit - 1) + 1, (1,), generator=generator))

    def _counts(self, number) -> list[int]:
        """The slots of each segment in pass `number`."""
        found = []
        for index, (_, length) in enumerate(self.segments):
            offset = self._offset(self._generator(number, index), length)
            found.append(0 if offset is None else (length - offset - 1) // self.unit)
        return found

    def _pass(self, place) -> tuple[int, int]:
        """(the pass a place of the walk lies in, the place inside it)."""
        while self.starts[-1] <= place:
            self.starts.append(self.starts[-1] + sum(self._counts(len(self.starts) - 1)) * self.unit)
        number = bisect.bisect_right(self.starts, place) - 1
        return number, place - self.starts[number]

    def _groups(self, number) -> list[int]:
        """Where each group of pass `number` starts among its slots, then
        the pass's slots: a group is GROUP slots of one segment, or the
        segment's last ones."""
        groups, first = [], 0
        for count in self._counts(number):
            groups += range(first, first + count, GROUP)
            first += count
        return groups + [first]

    def _layout(self, number) -> np.ndarray:
        """Pass `number`'s slots, each its start in the tokens, in its order."""
        if self.layout is None or self.layout[0] != number:
            found = []
            for index, (start, length) in enumerate(self.segments):
                generator = self._generator(number, index)
                offset = self._offset(generator, length)
                if offset is not None:
                    order = torch.randperm((length - offset - 1) // self.unit, generator=generator).numpy()
                    found.append(start + offset + order.astype(np.int64) * self.unit)
            self.layout, self.orders = (number, np.concatenate(found), self._groups(number)), {}
        return self.layout[1]

    def aligned(self, place) -> int:
        """The first place at or after `place` where a group starts: where a stage starts."""
        number, inside = self._pass(place)
        groups = self._groups(number)
        later = groups[bisect.bisect_left(groups, -(-inside // self.unit))]
        return self.starts[number] + later * self.unit

    def start(self, place, block_size) -> int:
        """Where in the tokens the window at that place of the walk starts."""
        number, inside = self._pass(place)
        layout, groups = self._layout(number), self.layout[2]
        group = bisect.bisect_right(groups, inside // self.unit) - 1
        slots = layout[groups[group]:groups[group + 1]]
        each = self.unit // block_size
        key = (number, group, block_size)
        if key not in self.orders:
            generator = torch.Generator().manual_seed(_seed(*self.seeds, "group", number, group, block_size))
            self.orders = {key: torch.randperm(len(slots) * each, generator=generator).numpy()}
        drawn = int(self.orders[key][(inside - groups[group] * self.unit) // block_size])
        return int(slots[drawn // each]) + (drawn % each) * block_size


def batching(stages, micro_batch, sharing, first=1) -> list[dict]:
    """How each stage's batch splits, in order: its windows (the stage's
    batch_size), micro_batch (windows per pass in one process: the
    option's, or the process's whole share) and accumulation (passes each
    process adds up before an update). sharing: the processes that share out
    the windows (1 when each splits every window). first: the first
    stage's number. ValueError when a batch does not split evenly."""
    found = []
    for number, stage in enumerate(stages, first):
        micro = micro_batch or stage.batch_size // sharing
        if not micro or stage.batch_size % (micro * sharing):
            raise ValueError(f"stage {number}'s batch of {stage.batch_size} windows does not split into micro-batches "
                             f"of {micro} on {sharing} processes; choose sizes that divide it")
        found.append({"windows": stage.batch_size, "micro_batch": micro,
                      "accumulation": stage.batch_size // (micro * sharing)})
    return found


class TextWindows(Environment):
    def __init__(self, settings, data):
        self.settings, self.data, strategy = settings, data, data.strategy
        self.readings = settings["readings"]
        self.stage_list = tuple(Stage(stage["steps"], stage["batch_size"], stage["block_size"])
                                for stage in stages(settings))
        self.sharing = 1 if strategy.splits_windows else strategy.world_size
        self.batches = batching(self.stage_list, data.micro_batch, self.sharing)
        self.prefix = settings["prefix_tokens"] or None
        self.train_tokens = _token_array(data.dock, shards(data.record, data.kinds), strategy, data.stream_shards,
                                         self.prefix)
        longest = max(stage.block_size for stage in self.stage_list)
        if len(self.train_tokens) <= longest + 1:
            self.close()
            raise ValueError(f"the corpus has {len(self.train_tokens):,} tokens of {', '.join(data.kinds)}, fewer "
                             f"than one window of {longest}")
        self.tokens = len(self.train_tokens)
        self.index = 0
        self.held_out = None
        if isinstance(self.train_tokens, ShardStream):
            ends = self.train_tokens.ends
            segments = [(ends[i - 1] if i else 0, ends[i] - (ends[i - 1] if i else 0)) for i in range(len(ends))]
        else:
            segments = [(0, self.tokens)]
        if all(length < longest + 1 for _, length in segments):
            self.close()
            raise ValueError(f"no token file holds one window of {longest} tokens, so none can be streamed whole")
        self.walk = _Walk(segments, longest, (data.seed, data.number))
        self.stage_places = []
        held = data.record.get("held_out")
        self.held_out = None if held is None else _token_array(data.dock, held["shards"], strategy)
        if self.held_out is not None and len(self.held_out) <= longest + 1:
            self.close()
            raise ValueError(f"the held-out portion has {len(self.held_out):,} tokens, fewer than one window of "
                             f"{longest}: the validation reads its windows there")
        self.validation = "held_out" if self.held_out is not None else "training"
        data.log(f"{data.step}: validation on {len(self.held_out):,} tokens of the corpus's held-out portion, which "
                 "no step trains on" if self.held_out is not None else
                 f"{data.step}: validation on windows of the training tokens, as the corpus has no held-out portion: "
                 "the validation loss is measured on text the step trains on")

    @property
    def learned(self):
        """The tokens of the token files of its kinds it reads, from their start."""
        return {kind: len(self.train_tokens) for kind in self.data.kinds}

    def stage(self, index):
        self.index = index

    def per_pass(self, index, windows):
        """Stage index (from 0) takes `windows` windows per pass in each
        process: the runner's choice for it (wpt_training.memory).
        ValueError when the stage's batch does not split so."""
        self.batches[index] = batching(self.stage_list[index:index + 1], windows, self.sharing, index + 1)[0]

    def _used(self, step) -> int:
        """The tokens the windows of the updates before update `step` hold together."""
        used, first = 0, 1
        for stage in self.stage_list:
            done = min(max(step - first, 0), stage.steps)
            used += done * stage.batch_size * (stage.block_size + 1)
            first += stage.steps
        return used

    def _tokens_at(self, tokens, step):
        """The tokens update `step` draws its windows from: the array itself,
        or for a stream the part in use once the updates before it have used
        their tokens."""
        return tokens.at(self._used(step)) if hasattr(tokens, "at") else tokens

    def _split(self, batch):
        """This process's passes of one global batch of the current stage."""
        strategy, split = self.data.strategy, self.batches[self.index]
        shared, rank = (1, 0) if strategy.splits_windows else (strategy.world_size, strategy.rank)
        micro, found = split["micro_batch"], []
        for a in range(split["accumulation"]):
            start = (a * shared + rank) * micro
            part = batch[start:start + micro].to(strategy.device)
            found.append((part[:, :-1], part[:, 1:]))
        return found

    def _passes(self, tokens, step, *seed_parts):
        """This process's passes of one global batch of the current stage, its windows at random places."""
        stage = self.stage_list[self.index]
        return self._split(_windows(self._tokens_at(tokens, step), stage.block_size, stage.batch_size, *seed_parts))

    def _place(self, step) -> int:
        """Where update `step` starts in the step's walk over its tokens, in tokens predicted (_Walk)."""
        if not self.stage_places:  # where each stage starts: once its stage before it ends, at a group's start
            place = 0
            for stage in self.stage_list:
                self.stage_places.append(self.walk.aligned(place))
                place = self.stage_places[-1] + stage.steps * stage.batch_size * stage.block_size
        first = 1
        for stage, place in zip(self.stage_list, self.stage_places):
            if step < first + stage.steps:
                return place + (step - first) * stage.batch_size * stage.block_size
            first += stage.steps
        return place

    def batch(self, step, model, strategy):
        stage = self.stage_list[self.index]
        place = self._place(step)
        starts = [self.walk.start(place + i * stage.block_size, stage.block_size) for i in range(stage.batch_size)]
        if not isinstance(self.train_tokens, ShardStream):
            return self._split(_gathered(self.train_tokens, starts, stage.block_size))
        stream, total, found, base, joined = self.train_tokens, len(self.train_tokens), [], None, None
        for start in starts:  # the shards in use from a window's token file on, kept while they hold the next ones
            if joined is None or (start - base) % total + stage.block_size + 1 > len(joined):
                segment = bisect.bisect_right(stream.ends, start)
                base = stream.ends[segment - 1] if segment else 0
                joined = stream.at(base)
            found.append(np.asarray(joined[(start - base) % total:(start - base) % total + stage.block_size + 1],
                                    dtype=np.int64))
        return self._split(torch.from_numpy(np.stack(found)))

    def evaluates(self, step, first, last):
        return step % (self.settings["eval_every"] * self.readings) == 0 or step in (first, last)

    def evaluation_passes(self, step):
        """This process's passes of the current stage's evaluation batches:
        the same windows at every evaluation, for any seed and any step of
        an order, of the held-out tokens when the corpus has them."""
        if self.held_out is not None:
            block = self.stage_list[self.index].block_size
            return [one for i in range(self.settings["eval_batches"])
                    for one in self._passes(self.held_out, step, "held_out", block, i)]
        return [one for i in range(self.settings["eval_batches"])
                for one in self._passes(self.train_tokens, step, "eval", i)]

    @torch.no_grad()
    def evaluate(self, step, method, model, wrapped, strategy, segment):
        losses = [float(windows_in_parts.validation_loss(method, model, wrapped, x, y, strategy, segment))
                  for x, y in self.evaluation_passes(step)]
        value = strategy.mean(sum(losses) / len(losses))
        return value * (strategy.world_size if strategy.splits_windows else 1)  # split windows: the shares add up

    def close(self):
        if isinstance(self.train_tokens, ShardStream):
            self.train_tokens.close()


def _token_array(dock, files, strategy, stream_shards=None, prefix=None):
    """The training tokens of the token files files as an array read from
    disk on demand. One process per machine builds the file, checking every
    token file. With stream_shards, a ShardStream that keeps only that many
    token files on disk at once. With a prefix, the first prefix tokens,
    from the token files that hold them alone."""
    held = prefix_shards(files, prefix)
    if stream_shards:
        return ShardStream(dock, files, stream_shards, prefix)
    if strategy.is_local_main:
        token_file(dock, held)
    strategy.barrier()
    training = (np.memmap(token_file(dock, held), dtype="<u4", mode="r") if held else np.zeros(0, dtype="<u4"))
    return training[:prefix]


def start(settings, data):
    return TextWindows(settings, data)
