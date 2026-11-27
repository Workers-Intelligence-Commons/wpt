"""wpt-eval's model as a gpu worker's job (docs/machines.md, "Evaluation"):
the part of an evaluation that needs the GPUs. The wpt manager runs the
tests, which hold the tokenizer, the dock, the sandboxes and the model
folder; the job holds the model and answers what the tests ask of it,
through the link. It deals only in token ids, as every job does.

The manager's spec holds the model's config, the weight files it offers
and the form of the attention it names (spec["kernels"], one of
wpt_model.kernels.FORMS, or None); the job decides the attention's form
first, as wpt train's does (docs/model.md, "Attention"), and a form that
cannot work here ends it with exit code 2 and the reason on its standard
error. It then downloads the files (checked against their sha256), loads the
model on the machine's best device in eval mode, tells the manager it is
ready, and then answers until the manager ends it:

    generator.new, writer.start, writer.write
                                 a seeded generator and the writers on the model
                                 (wpt_worker.writing)
    model.greedy                 the tokens the model writes greedily after each of several
                                 prompts of one length (greedy())
    model.losses                 the next-token loss of every token of several windows after
                                 the first (losses())
    model.ranks                  the rank of every token of several windows after the first
                                 among the model's scores at its place (ranks())
    eval.end                     the manager is done: the job ends with code 0

Served answers the same calls for a model in the caller's own process, with
the names the manager's side has (config, device, generator, writer,
greedy, losses and ranks): a test cannot tell where the model is.
"""

import threading

import numpy as np
from pathlib import Path

import torch
from safetensors.torch import load_file

from wpt_model import WPT, ModelConfig
from wpt_model.kernels import KernelUnavailable, kernels
from wpt_training.launch import REFUSED
from wpt_utils.device import describe, select_device
from wic_utils.lines import ERROR, say

from .job import call, download
from .writing import LocalWriter, Writers, Writing


class Served:
    """A model in this process: config (the model's config), device (a
    description of where it runs), generator() (a new seeded generator: its
    number), writer() (a writer on the model, as wpt_worker.writing.RemoteWriter
    is one on a gpu worker's), greedy(), losses() and ranks()."""

    def __init__(self, model):
        self.model, self.config = model, model.config
        self.device = describe(next(model.parameters()).device)
        self.writers = Writers(model)

    def generator(self, seed: int) -> int:
        return self.writers.new_generator(seed)

    def writer(self, prompt, rows=1, *, temperature, top_k, stops, routes=False, generator=None) -> LocalWriter:
        return LocalWriter(self.writers, prompt, rows, temperature=temperature, top_k=top_k, stops=stops,
                           routes=routes, generator=generator)

    def greedy(self, prompts: list[list[int]], steps: int) -> list[list[int]]:
        return greedy(self.model, prompts, steps)

    def losses(self, windows: list[list[int]]) -> list[np.ndarray]:
        return losses(self.model, windows)

    def ranks(self, windows: list[list[int]]) -> list[np.ndarray]:
        return ranks(self.model, windows)


@torch.no_grad()
def greedy(model, prompts: list[list[int]], steps: int) -> list[list[int]]:
    """The tokens model writes after each of prompts (token lists of one
    length), steps of them, always taking the likeliest."""
    ids = torch.tensor(prompts, device=next(model.parameters()).device)
    for _ in range(steps):
        logits = model(ids[:, -model.config.block_size:])[:, -1, :]
        ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return ids[:, len(prompts[0]):].tolist()


LOSS_CHUNK = 4096  # tokens read in one pass, over every row: the scores of one pass are LOSS_CHUNK x vocab_size


def batches(windows: list[list[int]]) -> list[list[int]]:
    """The windows' numbers in batches read together: by length, as many in
    a batch as keep the rows times the longest's tokens read within
    LOSS_CHUNK; a window longer than that alone."""
    found, current = [], []
    for number in sorted(range(len(windows)), key=lambda number: len(windows[number])):
        read = max(len(windows[number]) - 1, 1)  # the longest of the batch so far: they come by length
        if current and (len(current) + 1) * read > LOSS_CHUNK:
            found.append(current)
            current = []
        current.append(number)
    return found + [current] if current else found


@torch.no_grad()
def each_token(model, windows: list[list[int]], measure, dtype) -> list[np.ndarray]:
    """For each window (token ids), measure(scores, tokens) of each of its
    tokens after the first, from the model's scores read with only the
    window's earlier tokens before it: one fewer than the window, of dtype.
    The windows are read through the model's generation cache (docs/model.md,
    "Generation"), several as the rows of one cache (batches), each row its
    own number of tokens, at most LOSS_CHUNK tokens over the rows in one
    pass, a longer window LOSS_CHUNK tokens at a time, so the scores held at
    once stay LOSS_CHUNK x vocab_size whatever the windows' lengths."""
    device, found = next(model.parameters()).device, [np.zeros(0, dtype)] * len(windows)
    for batch in batches(windows):
        reads = [len(windows[number]) - 1 for number in batch]  # every token but the last is read
        cache, parts = model.new_cache(len(batch)), [[] for _ in batch]
        for start in range(0, max(reads), LOSS_CHUNK):
            counts = [min(max(read - start, 0), LOSS_CHUNK) for read in reads]
            ids = torch.zeros(len(batch), max(counts), dtype=torch.long, device=device)
            for row, (number, count) in enumerate(zip(batch, counts)):
                ids[row, :count] = torch.tensor(windows[number][start:start + count], dtype=torch.long)
            scores, _ = model.extend(ids, cache, counts)
            for row, (number, count) in enumerate(zip(batch, counts)):
                if count:
                    after = torch.tensor(windows[number][start + 1:start + count + 1], dtype=torch.long, device=device)
                    parts[row].append(measure(scores[row, :count].float(), after))
        for number, found_parts in zip(batch, parts):
            if found_parts:
                found[number] = torch.cat(found_parts).cpu().numpy().astype(dtype)
    return found


def losses(model, windows: list[list[int]]) -> list[np.ndarray]:
    """For each window, the next-token loss, in nats, of each of its tokens
    after the first (each_token): float32."""
    return each_token(model, windows, lambda scores, tokens: torch.nn.functional.cross_entropy(
        scores, tokens, reduction="none"), np.float32)


def ranks(model, windows: list[list[int]]) -> list[np.ndarray]:
    """For each window, the rank of each of its tokens after the first among
    the model's scores at its place (each_token): 1 when no token scores
    higher, int32."""
    return each_token(model, windows, lambda scores, tokens: 1 + (
        scores > scores.gather(1, tokens.unsqueeze(1))).sum(dim=1), np.int32)


def run(spec, link, work: Path) -> int:
    config = ModelConfig(**spec["model"])
    device, named = select_device(), spec["kernels"]
    if device.type == "cuda" or named:  # decided once, before the model loads (docs/model.md, "Attention")
        try:
            chosen = kernels(device, named)
        except KernelUnavailable as error:
            say(f"KernelUnavailable: {error}", ERROR)
            return REFUSED
        say("attention: " + (f"{chosen.NAME} kernels" if chosen else "PyTorch (--kernels pytorch)" if named
                             else "PyTorch (neither TileLang nor Triton is installed)"))
    state = {}
    for name, offer in spec["weights"].items():
        path = download(link, offer, work / "weights" / name)
        state.update(load_file(path))
        path.unlink()
    model = WPT(config)
    model.load_state_dict(state)
    del state
    model = model.to(device)
    model.eval()
    where, done = describe(device), threading.Event()
    writing = Writing(link)

    def answer(header, body):
        return {"tokens": greedy(model, header["prompts"], header["steps"])}, b""

    def windows_of(header, body):
        tokens, windows, start = np.frombuffer(body, dtype="<u4"), [], 0
        for length in header["lengths"]:
            windows.append(tokens[start:start + length].tolist())
            start += length
        return windows

    def scored(header, body):
        return {}, b"".join(found.astype("<f4").tobytes() for found in losses(model, windows_of(header, body)))

    def ranked(header, body):
        return {}, b"".join(found.astype("<u4").tobytes() for found in ranks(model, windows_of(header, body)))

    def end(header, body):
        done.set()

    with writing(model):
        link.handlers.update({"model.greedy": answer, "model.losses": scored, "model.ranks": ranked,
                              "eval.end": end})
        call(link, "ready", device=where)
        link.serve(until=done.is_set)
    return 0
