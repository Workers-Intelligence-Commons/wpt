"""What a corpus build of a snapshot with a data section and one
vocabulary would give, estimated from a small sample before anything is
built: its tokens, the bytes it would take in the store and the time its
build would take. Nothing is written.

What the corpus may take comes from the build's census
(wpt_depot.corpus.census), the snapshot's summary read at once: the bytes
and texts each share (or, without shares, every plain text) may take, and
the items'. The sample turns those bytes into tokens: at most POINTS texts
of the snapshot, spread over the order of their export keys
(wpt_depot.order; for a snapshot of more than FOLLOWED texts, spread over
its numbers, so the estimate never works out every text's key), the same
texts for the same snapshot. Of them, the texts the corpus may take are
read and tokenized in the sample's order as the build tokenizes them (a
plain text with END after it, a conversation through the chat template, a
task as the JSON text of its line), until SAMPLE_BYTES are tokenized; a
text bigger than BIG_TEXT counts by its bytes and is not tokenized. That
bounds the work to about SAMPLE_BYTES of reading and tokenizing, a few
seconds. The tokens come from the census's bytes at the sample's tokens
per byte, plain texts and items apart, and then the build's rules
(wpt_depot.corpus) cut them: with total_tokens, the items their share of it
(their part of the bytes of the corpus without total_tokens), then each
share its proportion of what they leave; without it, each share trimmed to
its proportion of the scarcest.

With total_tokens, the tokens a share and the items get are a forecast
that follows the build's rule, for a snapshot of at most FOLLOWED texts:
the build takes the texts and pieces in the order of their export keys
and each part fills as wpt_depot.corpus.Fill says, a text that does not fit
what is left of its part left out, so a share whose texts are each longer
than its part at a small size stays empty. A share whose texts carry
several tags besides its own splits its part over them (wpt_depot.groups),
which changes which of its texts the build takes but not the part itself:
the forecast fills each share with its texts in order all the same, so its
figures are the share's, never its groups'. The forecast walks the whole
order, as the build reads it, keeps the kind, pool and bytes of the texts
the corpus may take, and fills each part with them, each text's tokens its
bytes at the sample's tokens per byte. For a bigger snapshot, each part is
its proportion of the size.
even_size is the smallest size (total_tokens, or one of a fixed ladder of
sizes above it, each a quarter more than the one before) at which every share's forecast part of
the plain text is within UNEVEN of its proportion, the rule the web app's
corpus page uses to say shares came out uneven; None without shares, or
when no size the walk can follow gets there.
The store's bytes are the tokens at the bytes per token the sample's token
file takes, written as the build writes it but in memory, plus a row of
the texts table for each text taken at the bytes the sample's rows take,
plus the tasks' lines. The time is the bytes the build reads over the
sample's speed of reading and tokenizing, measured in one process, times
the processes the build tokenizes in (workers). It leaves out ordering the
snapshot's texts (the export key of each, then a sort), the first pass of
a share that splits by group, which walks every text of the snapshot, and,
with total_tokens, the tokens it counts of the texts it chooses, which the
build tokenizes again as it writes them.

A conversation that quotes texts is taken only with every text it quotes
(wpt_depot.corpus; docs/data-selection.md, "Conversations that quote
texts"): a sampled conversation one of whose quoted texts (the depot's
table quotes) is no plain text of the snapshot a share takes, or that the
data section's kinds leave out, stands for no text, as the build leaves it
out: the items the census counts are cut by the part of the sampled items'
bytes such conversations hold. Whether a size or the trimming of the shares
leaves out a quoted text the snapshot holds, or an excerpt matches no
single place, the estimate cannot know without choosing the plain texts as
the build does: it counts such a conversation as taken, and a warning says
so when the sample holds one that quotes.

So what the estimate reads does not grow with the snapshot: its summary,
POINTS texts, the whole order of a snapshot of at most FOLLOWED texts, and
the quoted texts of the sampled conversations. It reads the depot only
through wpt_depot.api.

    estimate(depot, snapshot, data, tokenizer, workers)   the estimate, as a dict
"""

import hashlib
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store.shards import SCHEMA as SHARD_SCHEMA
from wpt_tokenizer import END, Tokenizer, chat

from . import api
from .corpus import (CORPUS_SCHEMAS, IN_PROCESS, Fill, _proportions, _tags, _tags_text, build_key, canonical, census,
                     entries, pool_of, scarcest_trim, task_text, walk)
from .depot import Depot
from .kinds import PLAIN

POINTS = 1024  # texts sampled from the snapshot, at most
SAMPLE_BYTES = 4 << 20  # bytes of texts the sample tokenizes, at most
BIG_TEXT = SAMPLE_BYTES // 16  # bytes above which a sampled text counts by its bytes alone
FOLLOWED = 50_000  # texts of a snapshot up to which the forecast walks its whole order
UNEVEN = 0.1  # how far a share's part of the plain text may be from its proportion and still count as even
STEPS = 40  # the sizes even_size tries, at most


def estimate(depot: Depot, snapshot: str, data: dict, tokenizer: Tokenizer, workers: int = 1) -> dict:
    """The estimate of the corpus of depot's snapshot whose id is snapshot
    built with the data section data (its kinds, shares and total_tokens,
    wpt_depot.corpus.canonical) and tokenizer, whose build tokenizes in
    workers processes once it reads more than wpt_depot.corpus.IN_PROCESS
    bytes: {build_key (with tokenizer's vocabulary), tokens, plain_tokens,
    item_tokens, item_share (the items' share of a size: their part of the
    bytes of the corpus without total_tokens), share_tokens (with shares:
    [{tag, tokens, part (the tokens its proportion of the size gives it)}],
    else None), forecast (with total_tokens: whether the walk closed every
    part, so the tokens follow the build's order rather than proportions;
    else None), even_size, texts (the plain texts and the items),
    plain_texts (as the corpus record's texts counts them), item_texts (the
    conversations and tasks it takes), items_held (those the snapshot holds
    that it may take), kinds_held ({kind: the texts of that kind the
    snapshot holds}), bytes, token_bytes, table_bytes, read_bytes, seconds,
    workers, sample: {texts, tokenized, bytes, seconds}, warnings}.
    Refused for a data section that cannot be built; NotFound without such
    a snapshot."""
    source = depot.address
    row = api.snapshot(source, snapshot)
    wanted = canonical(data)
    shares, kinds, max_tokens = wanted["shares"] or None, wanted["kinds"] or None, wanted["total_tokens"]
    tags = _tags(shares)
    pools = [0] if shares is None else list(range(len(shares)))
    counted = census(row["summary"], shares, kinds)
    if row["texts"] <= FOLLOWED:
        followed = _followed(walk(source, snapshot, tags, kinds))
        sample_entries = walk(source, snapshot, tags, kinds, spread=POINTS)
    else:
        sample_entries, followed = entries(api.sampled(source, snapshot, POINTS), tags, kinds), None
    sampled = []  # (id, revision, kind, pool, bytes)
    items_seen = {"texts": 0, "bytes": 0, "kept_texts": 0, "kept_bytes": 0}
    quoting = 0  # the sampled conversations that quote texts the snapshot holds, which a size may still leave out
    for entry in sample_entries:
        if not entry.taken:
            continue
        if entry.kind != PLAIN:
            items_seen["texts"] += 1
            items_seen["bytes"] += entry.bytes
            if entry.kind == "conversation":
                quoted = api.quoted(source, entry.id, entry.revision)
                if not all(_held(source, snapshot, one, tags, kinds) for one in quoted):
                    continue
                quoting += bool(quoted)
            items_seen["kept_texts"] += 1
            items_seen["kept_bytes"] += entry.bytes
        sampled.append((entry.id, entry.revision, entry.kind, entry.pool, entry.bytes))

    measured = _tokenize(source, sampled, tokenizer)
    plain_rate = measured["plain_tokens"] / measured["plain_bytes"] if measured["plain_bytes"] else 0.0
    item_rate = measured["item_tokens"] / measured["item_bytes"] if measured["item_bytes"] else plain_rate
    plain_rate = plain_rate or item_rate
    warnings = []
    kept = items_seen["kept_bytes"] / items_seen["bytes"] if items_seen["bytes"] else 1.0
    kept_texts = items_seen["kept_texts"] / items_seen["texts"] if items_seen["texts"] else 1.0
    item_bytes = counted["item_bytes"] * kept
    items_held = counted["items"] * kept_texts
    pool_bytes = counted["pool_bytes"]
    available = {pool: pool_bytes[pool] * plain_rate for pool in pools}  # tokens each pool may take
    items = item_bytes * item_rate
    item_share = item_bytes / (item_bytes + _full_plain(pool_bytes, shares)) if item_bytes else 0.0
    whole_share = sum(one["share"] for one in shares) if shares else 1.0
    proportion = {pool: (shares[pool]["share"] / whole_share if shares else 1.0) for pool in pools}
    share_parts, forecast, even_size, walked = None, None, None, None
    if max_tokens is not None:
        items_taken = min(items, max_tokens * item_share)
        share_parts = {pool: (max_tokens - items_taken) * proportion[pool] for pool in pools}
        plain = {pool: min(available[pool], share_parts[pool]) for pool in pools}
        if followed is not None:
            walked = _walk(followed, max_tokens, item_share, shares, plain_rate, item_rate)
            forecast = walked["complete"]
            if walked["items"] is not None:
                items_taken = walked["items"]
            for pool, tokens in walked["plain"].items():
                if tokens is not None:
                    plain[pool] = tokens
            if shares and any(proportion.values()):
                even_size = _even_size(followed, max_tokens, item_share, shares, plain_rate, item_rate, proportion)
        else:
            forecast = False
    elif shares is not None:
        items_taken = items
        empty = [shares[pool]["tag"] for pool in pools if not pool_bytes[pool]]
        if empty:
            warnings.append(f"no plain text of the snapshot has the tag {', '.join(empty)}: the build is refused; the "
                            "estimate takes no plain text")
            plain = dict.fromkeys(pools, 0.0)
        else:
            plain = {pool: size * plain_rate for pool, size in scarcest_trim(pool_bytes, shares).items()}
    else:
        items_taken, plain = items, available
    if quoting and (max_tokens is not None or shares is not None):
        warnings.append("the sample holds conversations that quote texts the snapshot holds: the estimate counts them "
                        "as taken, while the build leaves out one whose quoted text the size or the shares leave out, "
                        "or whose quote matches no single place")
    if not sampled:
        warnings.append("the sample holds no text the corpus may take: the estimate is 0")
    elif not measured["texts"]:
        warnings.append(f"every sampled text is bigger than {BIG_TEXT:,} bytes, so none was tokenized: the "
                        "estimate of the tokens is 0")

    plain_tokens = sum(plain.values())
    taken_share = items_taken / items if items else 0.0  # the part of the items the size keeps
    plain_texts = sum(counted["pool_texts"][pool] * (plain[pool] / available[pool] if available[pool] else 0.0)
                      for pool in pools)
    item_texts = items_held * taken_share
    if walked is not None:  # the walk counts the texts it takes, as the build does, where it closed their parts
        plain_texts = plain_texts if walked["plain_texts"] is None else walked["plain_texts"]
        item_texts = item_texts if walked["item_texts"] is None else walked["item_texts"]
    texts = plain_texts + item_texts
    task_bytes = counted["task_bytes"]
    stored_tokens = plain_tokens + (items_taken - task_bytes * taken_share * item_rate)
    token_bytes = stored_tokens * measured["bytes_per_token"] + task_bytes * taken_share
    table_bytes = texts * measured["bytes_per_row"]
    read_bytes = (plain_tokens / plain_rate if plain_rate else 0.0) + item_bytes * taken_share
    if max_tokens is not None:  # the items it takes are read once more, to count their tokens first
        read_bytes += item_bytes * taken_share
    used = workers if read_bytes > IN_PROCESS else 1
    speed = measured["bytes"] / measured["seconds"] if measured["seconds"] else 0.0
    share_tokens = None if shares is None else [
        {"tag": shares[pool]["tag"], "tokens": round(plain[pool]),
         "part": round(share_parts[pool] if share_parts is not None else plain[pool])} for pool in pools]
    vocabulary = data.get("vocabulary") or {}
    return {"build_key": build_key(row, data, vocabulary.get("sha256") or _sha256(tokenizer)),
            "tokens": round(plain_tokens + items_taken), "plain_tokens": round(plain_tokens),
            "item_tokens": round(items_taken), "item_share": item_share, "share_tokens": share_tokens,
            "forecast": forecast, "even_size": even_size, "texts": round(texts), "plain_texts": round(plain_texts),
            "item_texts": round(item_texts), "items_held": round(items_held),
            "kinds_held": {kind: one["texts"] for kind, one in counted["kinds"].items()},
            "bytes": round(token_bytes + table_bytes), "token_bytes": round(token_bytes),
            "table_bytes": round(table_bytes), "read_bytes": round(read_bytes),
            "seconds": round(read_bytes / (speed * used)) if speed else None, "workers": used,
            "sample": {"texts": len(sampled), "tokenized": measured["texts"], "bytes": measured["bytes"],
                       "seconds": round(measured["seconds"], 3)},
            "warnings": warnings}


def _sha256(tokenizer: Tokenizer) -> str:
    return hashlib.sha256(tokenizer.to_bytes()).hexdigest()


def _followed(walked) -> dict:
    """The texts the corpus may take in the whole order of a snapshot,
    walked (wpt_depot.corpus.walk), whatever it takes or not: {texts:
    [(kind, pool, bytes)] in order, whole: True}."""
    return {"texts": [(entry.kind, entry.pool, entry.bytes) for entry in walked if entry.taken], "whole": True}


def _walk(followed: dict, max_tokens: int, item_share: float, shares: list[dict] | None, plain_rate: float,
          item_rate: float) -> dict:
    """The build's rule at max_tokens (the size) over the texts followed (_followed),
    each text's tokens its bytes at the rates (a plain text one more, for
    the END after it): the items fill their part as one pool, then each
    share (pool 0 without shares) fills its proportion of what is left
    (wpt_depot.corpus.Fill). {items: their tokens, plain: {pool: its
    tokens}, item_texts and plain_texts: how many items and plain texts it
    takes, complete}: a value is None where the walk ran out of texts before
    that part was known (closed, or a text left out), unless the texts
    followed are every text, or, for the items, the snapshot holds none
    (item_share 0)."""
    texts, whole = followed["texts"], followed["whole"]
    items = Fill({0: int(max_tokens * item_share)})
    item_texts, items_left = 0, False
    for kind, _, size in texts:
        if kind == PLAIN:
            continue
        if items.done():
            break
        if items.take(0, round(size * item_rate)):
            item_texts += 1
        else:
            items_left = True
    plain = Fill(_proportions(shares, max_tokens - items.used[0]))
    left, plain_texts = set(), 0
    for kind, pool, size in texts:
        if kind != PLAIN:
            continue
        if plain.done():
            break
        if pool in plain.closed:
            continue
        if plain.take(pool, round(size * plain_rate) + 1):
            plain_texts += 1
        else:
            left.add(pool)
    found = {pool: (plain.used[pool] if whole or pool in plain.closed or pool in left else None)
             for pool in plain.budget}
    items_known = whole or items.done() or items_left or item_share == 0  # a snapshot without items gives them nothing
    plain_known = all(value is not None for value in found.values())
    return {"items": items.used[0] if items_known else None, "plain": found,
            "item_texts": item_texts if items_known else None, "plain_texts": plain_texts if plain_known else None,
            "complete": items_known and plain_known}


def _even_size(followed: dict, max_tokens: int, item_share: float, shares: list[dict], plain_rate: float,
               item_rate: float, proportion: dict[int, float]) -> int | None:
    """The smallest size, max_tokens itself or one of the ladder above it
    (1,000 tokens, then each a quarter more rounded up to a thousand: the
    same sizes from whatever size it starts, so two sizes asked give the
    same answer), whose walk (_walk) gives every share a part of the plain
    text within UNEVEN of its proportion; None when no size within STEPS
    steps gets there, or when the walk cannot follow one."""
    ladder = 1000
    sizes = [max_tokens]
    while len(sizes) <= STEPS:
        if ladder > max_tokens:
            sizes.append(ladder)
        ladder = -(-ladder * 5 // 4000) * 1000
    for size in sizes:
        walked = _walk(followed, size, item_share, shares, plain_rate, item_rate)
        if not walked["complete"]:
            return None
        got = sum(walked["plain"].values())
        if got and all(abs(walked["plain"][pool] / got - proportion[pool]) <= UNEVEN for pool in proportion):
            return size
    return None


def _full_plain(plain: dict[int, float], shares: list[dict] | None) -> float:
    """The bytes of plain text the corpus without total_tokens takes, from
    the bytes each pool may take: every one without shares, else each
    share trimmed as the build trims it (wpt_depot.corpus.scarcest_trim)."""
    return sum((plain if shares is None else scarcest_trim(plain, shares)).values())


def _held(source: str, snapshot: str, text: str, tags: list[str] | None, kinds: list[str] | None) -> bool:
    """Whether the snapshot whose id is snapshot, of the depot at the
    address source, holds text, at its revision as of the bound of its
    lowest part (wpt_depot.api.texts), as a plain text a share takes (every
    plain text without shares) and the kinds name, as a corpus reads it."""
    found = api.texts(source, snapshot, [text]).get(text)
    if found is None:
        return False
    return found["kind"] == PLAIN and pool_of(found["kind"], _tags_text(found["tags"]), tags, kinds)[1]


def _tokenize(source: str, sampled: list[tuple], tokenizer: Tokenizer) -> dict:
    """The sample's texts read and tokenized in order as the build does,
    until SAMPLE_BYTES, timed: {texts, bytes, seconds, plain_tokens,
    plain_bytes, item_tokens, item_bytes, bytes_per_token (of the token
    file they would make), bytes_per_row (of their rows of the texts
    table)}."""
    found = {"texts": 0, "bytes": 0, "plain_tokens": 0, "plain_bytes": 0, "item_tokens": 0, "item_bytes": 0}
    shard_rows, credit_rows = [], []
    name = api.about(source)["name"]
    started = time.perf_counter()
    for text, revision, kind, pool, size in sampled:
        if size > BIG_TEXT:
            continue
        if found["bytes"] + size > SAMPLE_BYTES:
            break
        [row] = api.credits(source, [(text, revision)])
        content = b"".join(api.content(source, text, revision))
        if kind == PLAIN:
            tokens = np.asarray([*tokenizer.encode(content), END], np.uint32)
        elif kind == "conversation":
            tokens = np.asarray(chat.encode(tokenizer, json.loads(content))[0], np.uint32)
        else:
            tokens = np.asarray(tokenizer.encode(task_text(json.loads(content)).encode("utf-8")), np.uint32)
        group = "plain" if kind == PLAIN else "item"
        found[f"{group}_tokens"] += len(tokens)
        found[f"{group}_bytes"] += size
        found["texts"] += 1
        found["bytes"] += size
        if kind != "task":
            shard_rows.append((tokens, name, text, "", 0))
        credit_rows.append({"depot": name, "id": text, "revision": revision, **row,
                            "tokens": 0, "piece": None})
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
