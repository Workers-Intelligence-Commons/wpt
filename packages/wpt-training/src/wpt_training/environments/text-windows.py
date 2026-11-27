"""Text windows drawn from the corpus's token files of plain texts
(docs/training.md, "Stages").

`stages` lists the window lengths in order, each {"block_size", "steps",
"batch_size"}: its updates and its windows per update, over every process
together (batch_size is global). The windows grow from one stage to the
next, inside one step and one optimizer schedule. Every update draws
batch_size windows of block_size + 1 tokens at random places of the
training tokens, seeded by the order's seed, the step's number in the
order and the update only; with
--stream, from the shards in use once the updates before it have used
their tokens (wpt_corpus.shards.ShardStream).

`readings` (1 or more) is how many times the step reads its training
tokens: every stage runs readings times its `steps` (`stages`), and
eval_every counts the updates of one reading, so the step is the step of
one reading stretched readings times (docs/training.md, "Readings"). With
readings above 1 and without --stream, a window whose place in the step's
walk over its tokens (the windows of every update, block_size + 1 tokens
each, one after the other) lies past the first pass over them comes from a
pass without replacement: pass q cuts the training tokens into windows
from a place of its own and takes each of them once, in an order of its
own, both drawn from the seed, the step's number, q and the window length
(`_pass`). A window within the first pass is the one a step of one reading
draws.

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
the model is evaluated on eval_batches batches of windows of the training
tokens, at the stage's window length, drawn apart from the training's.
A stage's evaluation windows are the same at every evaluation, and the
same for any seed and any step of an order: they depend only on the
training tokens, the window length, the batch size and eval_batches, so
two runs that read the same tokens through the same stages are measured
on the same windows. With --stream they are drawn from the shards in use,
which change as the run goes on.
"""

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


def stages(settings) -> list[dict]:
    """The stages the step runs: each stage of its settings with readings times its updates."""
    return [{**stage, "steps": stage["steps"] * settings["readings"]} for stage in settings["stages"]]


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


def _pass(length, block_size, *seed_parts) -> tuple[int, torch.Tensor]:
    """A pass over length tokens without replacement, the same on every
    machine: (the place its windows of block_size + 1 tokens start from, the
    order it takes them in: a permutation of the windows that fit one after
    the other from that place)."""
    size = block_size + 1
    generator = torch.Generator().manual_seed(_seed(*seed_parts))
    offset = int(torch.randint(min(block_size, length - size) + 1, (1,), generator=generator))
    return offset, torch.randperm((length - offset) // size, generator=generator)


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
        self.passes = {}  # the pass over the training tokens in use, by (its number, the window length): _order

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

    def batch(self, step, model, strategy):
        seeds = (self.data.seed, self.data.number, "train", step)
        if self.readings == 1 or isinstance(self.train_tokens, ShardStream):
            return self._passes(self.train_tokens, step, *seeds)
        return self._split(self._read_again(step, seeds))

    def _read_again(self, step, seeds):
        """The windows of update `step` of a step that reads its tokens more
        than once: a window whose place in the step's walk over its tokens
        lies within the first pass is the one a step of one reading draws,
        and one placed in pass q after it is the window its place gives in
        that pass's order (_pass). Where the windows that fit after the
        pass's place are fewer than the walk's places in the pass (one or
        two fewer), its last places take its first windows again."""
        stage, length = self.stage_list[self.index], len(self.train_tokens)
        size, used = stage.block_size + 1, self._used(step)
        starts = _starts(length, stage.block_size, stage.batch_size, *seeds)
        for i in range(stage.batch_size):
            place = used + i * size
            if place >= length:
                offset, order = self._order(place // length, stage.block_size)
                starts[i] = offset + int(order[(place % length) // size % len(order)]) * size
        return _gathered(self.train_tokens, starts, stage.block_size)

    def _order(self, number, block_size):
        """Pass `number` (from 1) over the training tokens in windows of block_size + 1 tokens (_pass)."""
        key = (number, block_size)
        if key not in self.passes:  # one pass at a time is in use
            self.passes = {key: _pass(len(self.train_tokens), block_size, self.data.seed, self.data.number,
                                      "reading", number, block_size)}
        return self.passes[key]

    def evaluates(self, step, first, last):
        return step % (self.settings["eval_every"] * self.readings) == 0 or step in (first, last)

    def evaluation_passes(self, step):
        """This process's passes of the current stage's evaluation batches:
        the same windows at every evaluation, for any seed and any step of
        an order."""
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
