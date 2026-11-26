"""Bits per byte: how well the model predicts texts no training read, the
held-out portion of its corpus (docs/data-selection.md, "The portions of a
corpus").

The test finds on the dock the corpus delivery the folder's provenance.json
names, and reads the token files of its held-out portion
(wpt.resources.held_out), each checked against its hashes. Each text is one
window: END, then the text's tokens and the END that closes it, each token
read with only the text's earlier tokens before it, as a document starts in
training and in wpt chat. The loss of every token, the closing END
included, adds up to the text's nats; bits per byte is the nats of every
text over ln 2 and over their bytes, the bytes their tokens decode to,
which are the texts' own UTF-8 bytes. It is also given for each share's
tag (the rows' pool), with nats per token.

A model whose corpus has no held-out portion is refused (EvalError), and
so the run of wpt eval stops: one trained on a corpus the depot built under
corpora/ or before snapshots, on a delivery made without portions or for an
order whose roles leave the held-out portion out. So is a delivery the dock
no longer holds, and one an erasure reached (stale), whose texts are not to
be read again.
"""

import math
import tempfile

from freight import api as freight
from wpt_corpus.files import DockFiles
from wpt_corpus.shards import read_verified, shard_file
from wpt_model.provenance import corpus_label, deliveries
from wpt_tokenizer import Tokenizer
from wpt import resources
from wpt.eval import EvalError, Result

CALL_TOKENS = 1 << 16  # tokens sent to the model in one call, about: the texts of a call are whole


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


def measure(context, rows) -> dict:
    """{pool: {texts, tokens, bytes, nats}} of the rows (tokens, source, document, pool, pass), the texts sent to
    the model about CALL_TOKENS tokens at a time."""
    found, waiting = {}, []

    def send():
        for (pool, ids, length), losses in zip(waiting, context.model.losses([ids for _, ids, _ in waiting])):
            held = found.setdefault(pool, {"texts": 0, "tokens": 0, "bytes": 0, "nats": 0.0})
            held["texts"] += 1
            held["tokens"] += len(ids) - 1
            held["bytes"] += length
            held["nats"] += float(losses.astype("float64").sum())
        waiting.clear()

    for tokens, _, _, pool, _ in rows:
        ids = [Tokenizer.END, *(int(token) for token in tokens)]  # the text's tokens end with END
        waiting.append((pool, ids, len(context.tokenizer.decode(ids[1:-1]))))
        if sum(len(one[1]) for one in waiting) >= CALL_TOKENS:
            send()
    if waiting:
        send()
    return found


def run(context) -> Result:
    key, portion = held_out(context)
    found = {}
    with tempfile.TemporaryDirectory(prefix="wpt-eval-held-out-") as folder:
        files = DockFiles(context.dock, folder)
        for shard in portion["shards"]:
            rows = read_verified(files, shard)
            files.release(shard_file(shard))
            for pool, held in measure(context, rows).items():
                into = found.setdefault(pool, {"texts": 0, "tokens": 0, "bytes": 0, "nats": 0.0})
                for name, value in held.items():
                    into[name] += value

    def figures(held: dict) -> dict:
        return {"texts": held["texts"], "tokens": held["tokens"], "bytes": held["bytes"],
                "bits_per_byte": held["nats"] / math.log(2) / held["bytes"] if held["bytes"] else None,
                "nats_per_token": held["nats"] / held["tokens"] if held["tokens"] else None}

    whole = {name: sum(held[name] for held in found.values()) for name in ("texts", "tokens", "bytes", "nats")}
    data = {"delivery": key, **figures(whole), "by_share": {pool: figures(held) for pool, held in sorted(found.items())}}
    if not whole["bytes"]:
        return Result(data, f"The held-out portion of corpus delivery {key} holds no text: nothing measured.")
    lines = [f"{data['texts']:,} held-out texts of corpus delivery {key}, which no training read "
             f"({data['bytes']:,} bytes, {data['tokens']:,} tokens): {data['bits_per_byte']:.4f} bits per byte, "
             f"{data['nats_per_token']:.4f} nats per token."]
    if len(found) > 1:
        lines += ["", "| share | texts | bytes | bits per byte |", "| --- | --- | --- | --- |"]
        lines += [f"| {pool or '(no share)'} | {one['texts']:,} | {one['bytes']:,} | {one['bits_per_byte']:.4f} |"
                  for pool, one in data["by_share"].items()]
    return Result(data, "\n".join(lines))
