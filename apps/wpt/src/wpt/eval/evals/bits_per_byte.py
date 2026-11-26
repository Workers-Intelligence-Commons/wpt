"""Bits per byte: how well the model predicts texts no training read, the
held-out portion of its corpus (docs/data-selection.md, "The portions of a
corpus").

The test finds on the dock the corpus delivery the folder's provenance.json
names, and reads the token files of its held-out portion
(wpt.resources.held_out), each checked against its hashes. What it
measures depends on nothing but those files and the model's context, its
config's block_size B:

- the texts: each document once, in the order of their document ids, every
  one whose bytes still fit in MAX_BYTES; a text's bytes are those its
  tokens decode to, its own UTF-8 bytes, so the texts taken are the same
  whatever the vocabulary;
- the stream: END, then each text's tokens, which end with the END that
  closes it, as training reads documents one after the other;
- the windows: B + 1 tokens of the stream from token 0, B, 2B, ..., the
  last one shorter; the model gives the loss of each token of a window
  after its first, read with only the window's earlier tokens before it.
  Each window starts on the last token of the window before it, so every
  token of the stream but the first has its loss counted exactly once, and
  that first token is an END, which stands for no byte.

Bits per byte is the nats of every token counted, over ln 2, over the bytes
those tokens decode to: a token's bytes are those of tokenizer.decode on it
alone (at least one for a token of text, none for END), so the bytes
counted are every byte of the texts taken, the same for two vocabularies.
The loss of each END that closes a text is counted, with no byte: the
model pays for not knowing where a text ends. The test also gives the nats
per token counted, the spread of bits per byte between windows (the
standard deviation of each window's bits over its bytes, windows with no
byte left out) and the error of bits per byte from it (the standard error
of a ratio of two sums: the square root of sum((bits_i - r bytes_i)^2) /
(n (n - 1)), over the mean bytes of a window, r the bits per byte; none
under two windows), and bits per byte for each share's tag (the rows' pool),
each token counted in its text's share.

MAX_BYTES bounds the work: at most that many bytes of text, so at most
that many tokens of text plus one END per text and the first one, which a
GPU reads in a few minutes; it stands above wpt_corpus.roles.HELD_OUT_BYTES,
so the whole held-out portion is measured.

A model whose corpus has no held-out portion is refused (EvalError), and
so the run of wpt eval stops: one trained on a corpus the depot built under
corpora/ or before snapshots, on a delivery made without portions or for an
order whose roles leave the held-out portion out. So is a delivery the dock
no longer holds, and one an erasure reached (stale), whose texts are not to
be read again.
"""

import math
import tempfile

import numpy as np

from freight import api as freight
from wpt_corpus.files import DockFiles
from wpt_corpus.shards import read_verified, shard_file
from wpt_model.provenance import corpus_label, deliveries
from wpt_tokenizer import Tokenizer
from wpt import resources
from wpt.eval import EvalError, Result

MAX_BYTES = 4 * 1024 * 1024  # bytes of text measured at most, above the held-out portion's 2 MiB
CALL_TOKENS = 1 << 16  # tokens sent to the model in one call, about: the windows of a call are whole


def held_out(context) -> tuple[str, dict]:
    """(the corpus delivery's key, its held-out portion), or EvalError."""
    corpus = deliveries(context.provenance).get("corpus")
    if corpus is None:
        raise EvalError(f"{context.folder}: learned from {corpus_label(context.provenance)}, a corpus the freight did "
                        "not deliver, which has no held-out portion to measure it on")
    note = freight.delivery(corpus["key"], dock=context.dock)
    if note is None:
        raise EvalError(f"{context.folder}: the dock holds no corpus delivery {corpus['key']}, the one it learned from")
    for mark in ("stale", "deleted"):
        if mark in note:
            raise EvalError(f"{context.folder}: its corpus delivery {corpus['key']} is {mark}: {note[mark]}")
    found = resources.held_out(note)
    if found is None:
        raise EvalError(f"{context.folder}: its corpus delivery {corpus['key']} has no held-out portion (made without "
                        "portions, or for an order whose roles leave it out): nothing to measure it on")
    return corpus["key"], found


def token_bytes(tokenizer, tokens: np.ndarray) -> np.ndarray:
    """The bytes each token stands for, decoded alone: none for END."""
    unique, where = np.unique(tokens, return_inverse=True)
    sizes = np.array([len(tokenizer.decode([int(one)])) for one in unique], dtype=np.int64)
    return sizes[where.reshape(-1)]


def stream(rows, tokenizer, max_bytes: int = MAX_BYTES) -> dict:
    """The stream the test reads, from the rows (tokens, source, document,
    pool, pass) of the held-out token files: {tokens (END first), bytes
    (of each token), pools (the index in `names` of each token's share,
    -1 for the first END), names, texts (taken), left_out (texts that did
    not fit in max_bytes)}."""
    texts = {}
    for tokens, _, document, pool, _ in rows:
        texts.setdefault(document, (pool, np.asarray(tokens, dtype=np.int64)))  # a document repeated comes once
    names = sorted({pool for pool, _ in texts.values()})
    parts, sizes, pools = [np.array([Tokenizer.END])], [np.zeros(1, np.int64)], [np.full(1, -1)]
    taken = left_out = used = 0
    for document in sorted(texts):
        pool, tokens = texts[document]
        size = token_bytes(tokenizer, tokens)
        if used + int(size.sum()) > max_bytes:
            left_out += 1
            continue
        used += int(size.sum())
        taken += 1
        parts.append(tokens)
        sizes.append(size)
        pools.append(np.full(len(tokens), names.index(pool)))
    return {"tokens": np.concatenate(parts), "bytes": np.concatenate(sizes), "pools": np.concatenate(pools),
            "names": names, "texts": taken, "left_out": left_out}


def windows(length: int, block_size: int) -> list[tuple[int, int]]:
    """(start, end) of the windows of a stream of `length` tokens: block_size + 1 tokens from 0, block_size,
    2 block_size, ..., the last one shorter, each starting on the last token of the one before it."""
    return [(start, min(start + block_size + 1, length)) for start in range(0, length - 1, block_size)]


def measure(read: dict, block_size: int, losses) -> dict:
    """The figures of the stream `read` (as stream gives it) read in windows of block_size + 1 tokens, each
    window's losses from losses(list of windows) -> list of arrays, one fewer than each window."""
    cut = windows(len(read["tokens"]), block_size)
    nats = np.zeros(len(read["tokens"]), dtype=np.float64)  # the loss of each token, none for the first

    def send(group):
        found = losses([[int(token) for token in read["tokens"][start:end]] for start, end in group])
        for (start, end), values in zip(group, found):
            nats[start + 1:end] = np.asarray(values, dtype=np.float64)

    group = []
    for one in cut:
        group.append(one)
        if sum(end - start for start, end in group) >= CALL_TOKENS:
            send(group)
            group = []
    if group:
        send(group)

    bits, sizes = nats / math.log(2), read["bytes"]
    per_window = np.array([(bits[start + 1:end].sum(), sizes[start + 1:end].sum()) for start, end in cut]).reshape(-1, 2)

    def figures(mask) -> dict:
        tokens, size, total = int(mask[1:].sum()), int(sizes[mask].sum()), float(bits[mask].sum())
        return {"tokens": tokens, "bytes": size, "bits_per_byte": total / size if size else None,
                "nats_per_token": float(nats[mask].sum()) / tokens if tokens else None}

    whole = figures(np.ones(len(nats), dtype=bool))  # every token but the first
    rate, spread, error = whole["bits_per_byte"], None, None
    filled = per_window[per_window[:, 1] > 0]
    if len(filled) >= 2:
        spread = float(np.std(filled[:, 0] / filled[:, 1], ddof=1))
    if rate is not None and len(per_window) >= 2:
        n = len(per_window)
        residuals = per_window[:, 0] - rate * per_window[:, 1]
        error = float(math.sqrt((residuals ** 2).sum() / (n * (n - 1))) / per_window[:, 1].mean())
    by_share = {name: figures(read["pools"] == index) for index, name in enumerate(read["names"])}
    return {"block_size": block_size, "windows": len(cut), "texts": read["texts"], "texts_left_out": read["left_out"],
            **whole, "bits_per_byte_error": error, "bits_per_byte_spread": spread, "by_share": by_share}


def run(context) -> Result:
    key, portion = held_out(context)
    rows = []
    with tempfile.TemporaryDirectory(prefix="wpt-eval-held-out-") as folder:
        files = DockFiles(context.dock, folder)
        for shard in portion["shards"]:
            rows += read_verified(files, shard)
            files.release(shard_file(shard))
    read = stream(rows, context.tokenizer)
    data = {"delivery": key, "max_bytes": MAX_BYTES,
            **measure(read, context.model.config.block_size, context.model.losses)}
    if not data["bytes"]:
        return Result(data, f"The held-out portion of corpus delivery {key} holds no text: nothing measured.")
    error = "" if data["bits_per_byte_error"] is None else f" ± {data['bits_per_byte_error']:.4f}"
    lines = [f"{data['texts']:,} held-out texts of corpus delivery {key}, which no training read "
             f"({data['bytes']:,} bytes, {data['tokens']:,} tokens, {data['windows']:,} windows of "
             f"{data['block_size']:,} tokens): {data['bits_per_byte']:.4f}{error} bits per byte, "
             f"{data['nats_per_token']:.4f} nats per token."]
    if data["bits_per_byte_spread"] is not None:
        lines[0] += f" Between windows, bits per byte spread by {data['bits_per_byte_spread']:.4f}."
    if data["texts_left_out"]:
        lines[0] += f" {data['texts_left_out']:,} texts left out, past {MAX_BYTES:,} bytes."
    if len(data["by_share"]) > 1:
        lines += ["", "| share | bytes | tokens | bits per byte |", "| --- | --- | --- | --- |"]
        lines += [f"| {pool or '(no share)'} | {one['bytes']:,} | {one['tokens']:,} | "
                  + (f"{one['bits_per_byte']:.4f} |" if one["bits_per_byte"] is not None else "none |")
                  for pool, one in data["by_share"].items()]
    return Result(data, "\n".join(lines))
