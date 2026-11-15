"""What a corpus build of a dataset with one vocabulary at one size would
give, estimated from a small sample before anything is built: its tokens,
the bytes it would take in the store and the time its build would take.
Nothing is written.

The sample is at most POINTS texts of each library the dataset reads,
spread over the order its depot recorded the library's rows in
(Depot.library_spread), the same texts for the same depot state. Of them,
the texts the corpus may take (wpt_depot.order) are read and tokenized in
the sample's order as the build tokenizes them (a plain text with END
after it, a conversation through the chat template, a task as the JSON
text of its line), until SAMPLE_BYTES are tokenized; a text bigger than
BIG_TEXT counts by its bytes and is not tokenized. That bounds the work to
about SAMPLE_BYTES of reading and tokenizing, a few seconds.

Each sampled text stands for the texts of its library the depot's counts
give now (Depot.counts), over the texts sampled from that library: so the
bytes each share (or, without shares, every plain text) and the items may
take are estimated, a text two libraries hold counting twice. The tokens
come from those bytes at the sample's tokens per byte, plain texts and
items apart, and then the build's rules (wpt_depot.corpus) cut them: with
max_tokens, the items first, then each share its proportion of what they
leave; without it, each share trimmed to its proportion of the scarcest.
The store's bytes are the tokens at the bytes per token the sample's token
file takes, written as the build writes it but in memory, plus a row of
the texts table for each text taken at the bytes the sample's rows take,
plus the tasks' lines. The time is the bytes the build reads over the
sample's speed of reading and tokenizing, measured in one process, times
the processes the build tokenizes in (workers). It leaves out the pass
that orders the dataset (wpt_depot.order), which every build makes before
it tokenizes and which nothing here measures: order_texts says how many
texts that pass reads, the members of every library the dataset reads.

    estimate(depot, dataset, tokenizer, max_tokens, url, workers)   the estimate, as a dict
"""

import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store.shards import SCHEMA as SHARD_SCHEMA
from wpt_tokenizer import END, Tokenizer, chat

from .corpus import CORPUS_SCHEMAS, CREDITS, IN_PROCESS, _depots, depot_name, task_text
from .depot import Depot
from .order import TRAINING_KINDS
from .parts import scan

POINTS = 256  # texts sampled from each library, at most
SAMPLE_BYTES = 4 << 20  # bytes of texts the sample tokenizes, at most
BIG_TEXT = SAMPLE_BYTES // 16  # bytes above which a sampled text counts by its bytes alone


def estimate(depot: Depot, dataset: dict, tokenizer: Tokenizer, max_tokens: int | None, url: str | None = None,
             workers: int = 1) -> dict:
    """The estimate of the corpus of depot's dataset (its row) built with
    tokenizer at the size max_tokens (None: every text the rules take),
    whose build tokenizes in workers processes once it reads more than
    wpt_depot.corpus.IN_PROCESS bytes: {tokens, plain_tokens, item_tokens,
    texts, bytes, token_bytes, table_bytes, read_bytes, seconds, workers,
    order_texts, sample: {texts, tokenized, bytes, seconds}, warnings}.
    NotFound and NotADepot as reading the libraries meets them."""
    holder = depot_name(depot, dataset, url)
    names, opened = _depots(depot, dataset, holder)
    shares = dataset["shares"]
    tags = None if shares is None else [one["tag"] for one in shares]
    pools = [0] if shares is None else list(range(len(shares)))
    weight = {"plain": dict.fromkeys(pools, 0.0), "plain_texts": dict.fromkeys(pools, 0.0), "items": 0.0,
              "item_texts": 0.0, "task_bytes": 0.0}
    sampled = []  # (source name, id, revision, kind, pool, bytes)
    order_texts = 0  # the texts the pass that orders the dataset reads
    for name, ref in zip(names, dataset["libraries"]):
        source = opened[name]
        members = source.counts()["library"].get(ref["library"], 0)
        order_texts += members
        rows = source.library_spread(ref["library"], ref["change"], POINTS, members)
        if not rows:
            continue
        each = members / len(rows)  # the texts of the library one sampled text stands for
        for text, revision, kind, held_tags, size in rows:
            if kind is None:
                pool = 0 if tags is None else next((at for at, tag in enumerate(tags)
                                                    if tag in json.loads(held_tags)), None)
                if pool is None:
                    continue
                weight["plain"][pool] += each * size
                weight["plain_texts"][pool] += each
            elif kind in TRAINING_KINDS:
                pool = None
                weight["items"] += each * size
                weight["item_texts"] += each
                if kind == "task":
                    weight["task_bytes"] += each * size
            else:
                continue
            sampled.append((name, text, revision, kind, pool, size))

    measured = _tokenize(opened, sampled, tokenizer)
    plain_rate = measured["plain_tokens"] / measured["plain_bytes"] if measured["plain_bytes"] else 0.0
    item_rate = measured["item_tokens"] / measured["item_bytes"] if measured["item_bytes"] else plain_rate
    plain_rate = plain_rate or item_rate
    warnings = []
    available = {pool: weight["plain"][pool] * plain_rate for pool in pools}  # tokens each pool may take
    items = weight["items"] * item_rate
    if max_tokens is not None:
        items_taken = min(items, max_tokens)
        whole = sum(one["share"] for one in shares) if shares else 1.0
        proportion = {pool: (shares[pool]["share"] / whole if shares else 1.0) for pool in pools}
        plain = {pool: min(available[pool], (max_tokens - items_taken) * proportion[pool]) for pool in pools}
    elif shares is not None:
        items_taken = items
        whole = sum(one["share"] for one in shares)
        proportion = {pool: shares[pool]["share"] / whole for pool in pools}
        empty = [shares[pool]["tag"] for pool in pools if not weight["plain"][pool]]
        if empty:
            warnings.append(f"no sampled text has the tag {', '.join(empty)}: if the dataset holds none, the build is "
                            "refused; the estimate takes no plain text")
            plain = dict.fromkeys(pools, 0.0)
        else:
            scarcest = min(pools, key=lambda pool: available[pool] / proportion[pool])
            total = available[scarcest] / proportion[scarcest]
            plain = {pool: min(available[pool], total * proportion[pool]) for pool in pools}
    else:
        items_taken, plain = items, available
    if not sampled:
        warnings.append("the sample holds no text the corpus may take: the estimate is 0")
    elif not measured["texts"]:
        warnings.append(f"every sampled text is bigger than {BIG_TEXT:,} bytes, so none was tokenized: the "
                        "estimate of the tokens is 0")

    plain_tokens = sum(plain.values())
    taken_share = items_taken / items if items else 0.0  # the part of the items the size keeps
    plain_texts = sum(weight["plain_texts"][pool] * (plain[pool] / available[pool] if available[pool] else 0.0)
                      for pool in pools)
    texts = plain_texts + weight["item_texts"] * taken_share
    stored_tokens = plain_tokens + (items_taken - weight["task_bytes"] * taken_share * item_rate)
    token_bytes = stored_tokens * measured["bytes_per_token"] + weight["task_bytes"] * taken_share
    table_bytes = texts * measured["bytes_per_row"]
    read_bytes = (plain_tokens / plain_rate if plain_rate else 0.0) + weight["items"] * taken_share
    if max_tokens is not None:  # the items are read once more, to count their tokens first
        read_bytes += weight["items"]
    used = workers if read_bytes > IN_PROCESS else 1
    speed = measured["bytes"] / measured["seconds"] if measured["seconds"] else 0.0
    return {"tokens": round(plain_tokens + items_taken), "plain_tokens": round(plain_tokens),
            "item_tokens": round(items_taken), "texts": round(texts),
            "bytes": round(token_bytes + table_bytes), "token_bytes": round(token_bytes),
            "table_bytes": round(table_bytes), "read_bytes": round(read_bytes),
            "seconds": round(read_bytes / (speed * used)) if speed else None, "workers": used,
            "order_texts": order_texts,
            "sample": {"texts": len(sampled), "tokenized": measured["texts"], "bytes": measured["bytes"],
                       "seconds": round(measured["seconds"], 3)},
            "warnings": warnings}


def _tokenize(opened: dict[str, Depot], sampled: list[tuple], tokenizer: Tokenizer) -> dict:
    """The sample's texts read and tokenized in order as the build does,
    until SAMPLE_BYTES, timed: {texts, bytes, seconds, plain_tokens,
    plain_bytes, item_tokens, item_bytes, bytes_per_token (of the token
    file they would make), bytes_per_row (of their rows of the texts
    table)}."""
    found = {"texts": 0, "bytes": 0, "plain_tokens": 0, "plain_bytes": 0, "item_tokens": 0, "item_bytes": 0}
    shard_rows, credit_rows = [], []
    started = time.perf_counter()
    for name, text, revision, kind, pool, size in sampled:
        if size > BIG_TEXT:
            continue
        if found["bytes"] + size > SAMPLE_BYTES:
            break
        source = opened[name]
        row = source.listed_text(text, revision)
        content = "".join(value for _, value in scan(source.reader(), {text: row["parts"]}) if value is not None)
        if kind is None:
            tokens = np.asarray([*tokenizer.encode(content.encode("utf-8")), END], np.uint32)
        elif kind == "conversation":
            tokens = np.asarray(chat.encode(tokenizer, json.loads(content))[0], np.uint32)
        else:
            tokens = np.asarray(tokenizer.encode(task_text(json.loads(content)).encode("utf-8")), np.uint32)
        group = "plain" if kind is None else "item"
        found[f"{group}_tokens"] += len(tokens)
        found[f"{group}_bytes"] += size
        found["texts"] += 1
        found["bytes"] += size
        if kind != "task":
            shard_rows.append((tokens, name, text, "", 0))
        credit_rows.append({"depot": name, "id": text, "revision": revision, **{key: row[key] for key in CREDITS},
                            "tokens": 0})
    found["seconds"] = time.perf_counter() - started
    found["bytes_per_token"] = _written(_shard_table(shard_rows)) / max(sum(len(one[0]) for one in shard_rows), 1)
    found["bytes_per_row"] = (_written(pa.Table.from_pylist(credit_rows, schema=CORPUS_SCHEMAS["texts"]))
                              / max(len(credit_rows), 1))
    return found


def _shard_table(rows: list[tuple]) -> pa.Table:
    """The rows as a token file's table (wpt_store.shards)."""
    columns = {name: [] for name in SHARD_SCHEMA.names}
    for row in rows:
        for name, value in zip(SHARD_SCHEMA.names, row):
            columns[name].append(value)
    return pa.table(columns, schema=SHARD_SCHEMA)


def _written(table: pa.Table) -> int:
    """The bytes of the table as a Parquet file compressed with zstd, written in memory."""
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue().size
