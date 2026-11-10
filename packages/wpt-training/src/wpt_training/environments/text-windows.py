"""Text windows drawn from the corpus's token files of plain texts
(docs/training.md, "Stages").

`stages` lists the window lengths in order, each {"block_size", "steps",
"batch_size"}: its updates and its windows per update, over every process
together (batch_size is global). The windows grow from one stage to the
next, inside one step and one optimizer schedule. Every update draws
batch_size windows of block_size + 1 tokens at random places of the
training tokens, seeded by the recipe's seed, the step's number in the
recipe and the update only; with
--stream, from the shards in use once the updates before it have used
their tokens (wpt_store.shards.ShardStream).

prefix_tokens, when above 0, makes the training tokens the first
prefix_tokens of the corpus's, in the token files' order, and nothing after
them: the windows, the token files copied or streamed and the texts the
step learned from are those of that start. Each size of a model series
reads its first part of the series' one corpus (docs/model.md, "Model
series"). A corpus with fewer tokens is refused before any step runs.

Each process takes --micro-batch windows per pass (default: all of its
share of the batch) and adds passes up until the batch is complete: pass a
of process r is slice (a * world_size + r) of the batch, so the batch
holds the same windows whatever the number of processes; with split
windows, every process takes every window and trains its slice of each.

Every eval_every updates, and at the first and last update of each stage,
the model is evaluated on eval_batches batches of windows of the training
tokens, at the stage's window length, drawn apart from the training's.
"""

import hashlib

import numpy as np
import torch

from wpt_store.shards import ShardStream, prefix_shards, token_file
from wpt_training import WINDOWS, Environment, Stage
from wpt_training import parts as windows_in_parts

GIVES = WINDOWS
SETTINGS = {"stages": [], "eval_every": 200, "eval_batches": 20, "prefix_tokens": 0}
STAGE = ("block_size", "steps", "batch_size")


def check(settings):
    """ValueError on stages that cannot be: none, a field missing or not a
    positive whole number, or eval_every or eval_batches below 1."""
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


def check_corpus(settings, record, store):
    """ValueError when the corpus holds fewer tokens of plain text than the prefix the step reads."""
    if settings["prefix_tokens"] > record["total_tokens"]:
        raise ValueError(f"the step reads the first {settings['prefix_tokens']:,} tokens; the corpus has "
                         f"{record['total_tokens']:,}")


def context(settings) -> int:
    """The window length the step leaves the model reading at: the last stage's."""
    return settings["stages"][-1]["block_size"]


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") & (2**63 - 1)


def _windows(tokens, block_size, count, *seed_parts):
    """count random windows of block_size + 1 tokens, the same on every machine."""
    generator = torch.Generator().manual_seed(_seed(*seed_parts))
    starts = torch.randint(len(tokens) - block_size - 1, (count,), generator=generator).tolist()
    return torch.from_numpy(np.stack([np.asarray(tokens[s:s + block_size + 1], dtype=np.int64) for s in starts]))


def batching(stages, micro_batch, sharing) -> list[dict]:
    """How each stage's batch splits, in order: its windows (the stage's
    batch_size), micro_batch (windows per pass in one process: the
    option's, or the process's whole share) and accumulation (passes each
    process adds up before an update). sharing: the processes that share out
    the windows (1 when each splits every window). ValueError when a batch
    does not split evenly."""
    found = []
    for number, stage in enumerate(stages, 1):
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
        self.stage_list = tuple(Stage(stage["steps"], stage["batch_size"], stage["block_size"])
                                for stage in settings["stages"])
        self.batches = batching(self.stage_list, data.micro_batch, 1 if strategy.splits_windows else strategy.world_size)
        self.prefix = settings["prefix_tokens"] or None
        self.train_tokens = _token_array(data.store, data.record, strategy, data.stream_shards, self.prefix)
        longest = max(stage.block_size for stage in self.stage_list)
        if len(self.train_tokens) <= longest + 1:
            self.close()
            raise ValueError(f"the corpus has {len(self.train_tokens):,} tokens of plain text, fewer than one window "
                             f"of {longest}")
        self.tokens = len(self.train_tokens)
        self.index = 0

    @property
    def learned(self):
        """The tokens of the plain texts' token files it reads, from their start."""
        return {"text": len(self.train_tokens)}

    def stage(self, index):
        self.index = index

    def _tokens_at(self, tokens, step):
        """The tokens update `step` draws its windows from: the array itself,
        or for a stream the part in use once the updates before it have used
        their tokens."""
        if not hasattr(tokens, "at"):
            return tokens
        used, first = 0, 1
        for stage in self.stage_list:
            done = min(max(step - first, 0), stage.steps)
            used += done * stage.batch_size * (stage.block_size + 1)
            first += stage.steps
        return tokens.at(used)

    def _passes(self, tokens, step, *seed_parts):
        """This process's passes of one global batch of the current stage."""
        strategy, stage, split = self.data.strategy, self.stage_list[self.index], self.batches[self.index]
        batch = _windows(self._tokens_at(tokens, step), stage.block_size, stage.batch_size, self.data.seed,
                         self.data.number, *seed_parts)
        shared, rank = (1, 0) if strategy.splits_windows else (strategy.world_size, strategy.rank)
        micro, found = split["micro_batch"], []
        for a in range(split["accumulation"]):
            start = (a * shared + rank) * micro
            part = batch[start:start + micro].to(strategy.device)
            found.append((part[:, :-1], part[:, 1:]))
        return found

    def batch(self, step, model, strategy):
        return self._passes(self.train_tokens, step, "train", step)

    def evaluates(self, step, first, last):
        return step % self.settings["eval_every"] == 0 or step in (first, last)

    @torch.no_grad()
    def evaluate(self, step, method, model, wrapped, strategy, segment):
        losses = [float(windows_in_parts.validation_loss(method, model, wrapped, x, y, strategy, segment))
                  for i in range(self.settings["eval_batches"])
                  for x, y in self._passes(self.train_tokens, step, "eval", step, i)]
        value = strategy.mean(sum(losses) / len(losses))
        return value * (strategy.world_size if strategy.splits_windows else 1)  # split windows: the shares add up

    def close(self):
        if isinstance(self.train_tokens, ShardStream):
            self.train_tokens.close()


def _token_array(store, record, strategy, stream_shards=None, prefix=None):
    """The training tokens as an array read from disk on demand. One process
    per machine builds the file, checking every token file. With
    stream_shards, a ShardStream that keeps only that many token files on
    disk at once. With a prefix, the first prefix tokens, from the token
    files that hold them alone."""
    shards = prefix_shards(record["shards"], prefix)
    if stream_shards:
        return ShardStream(store, record["shards"], stream_shards, prefix)
    if strategy.is_local_main:
        token_file(store, shards)
    strategy.barrier()
    training = (np.memmap(token_file(store, shards), dtype="<u4", mode="r") if shards else np.zeros(0, dtype="<u4"))
    return training[:prefix]


def start(settings, data):
    return TextWindows(settings, data)
