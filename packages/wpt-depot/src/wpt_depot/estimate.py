"""What a corpus build of a dataset with one vocabulary at one size would
give, estimated from a small sample before anything is built: its tokens,
the bytes it would take in the store and the time its build would take.
Nothing is written.

The sample is at most POINTS texts of each library the dataset reads,
spread over the order of their export keys (Depot.library_spread), the
same texts for the same depot state. Of them, the texts the corpus may
take (wpt_depot.order) are read and tokenized in the sample's order as the
build tokenizes them (a plain text with END
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
max_tokens, the items their share of it (their part of the estimated bytes
of the corpus without max_tokens), then each share its
proportion of what they leave; without it, each share trimmed to its
proportion of the scarcest.

With max_tokens, the tokens a share and the items get are a forecast that
follows the build's rule: the build takes the texts and pieces in the
order of their export keys (wpt_depot.order) and each part fills as
wpt_depot.corpus.Fill says, a text that does not fit what is left of its
part left out, so a share whose texts are each longer than its part at a
small size stays empty. The forecast walks the first FOLLOWED texts of that
order, as the build reads them, keeps the kind, pool and bytes of those the
corpus may take, and fills each part with them, each text's tokens its
bytes at the sample's tokens per byte. A part is known once it closes or
leaves a text out: past that, the build adds only texts smaller than what
is left of it. A part the walk does not know within those texts is left at
its proportion of the size. The items' part of the size and the items the dataset holds
come from the build's census (wpt_depot.corpus.census), as the build counts
them, rather than from the sample: the figures the dataset's revision
stores, read at once; when they are not known, they come from the sample's
weights instead, never from every text of the libraries, and a warning
says so.
even_size is the smallest size (max_tokens, or one of a fixed ladder of
sizes above it, each a quarter more than the one before) at which every share's forecast part of
the plain text is within UNEVEN of its proportion, the rule the web app's
corpus page uses to say shares came out uneven; None without shares, or
when no size the walk can follow gets there.
The store's bytes are the tokens at the bytes per token the sample's token
file takes, written as the build writes it but in memory, plus a row of
the texts table for each text taken at the bytes the sample's rows take,
plus the tasks' lines. The time is the bytes the build reads over the
sample's speed of reading and tokenizing, measured in one process, times
the processes the build tokenizes in (workers). figures says whether the
revision's figures are known: when they are not, the build first counts
every text of the dataset for its census, which the time leaves out, and
a warning says so.

So what the estimate reads does not grow with the dataset: POINTS texts of
each library, FOLLOWED texts of the order and the counts of the figures.

    estimate(depot, dataset, tokenizer, max_tokens, url, workers)   the estimate, as a dict
"""

import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store.shards import SCHEMA as SHARD_SCHEMA
from wpt_tokenizer import END, Tokenizer, chat

from .corpus import (CORPUS_SCHEMAS, CREDITS, IN_PROCESS, Fill, _depots, _proportions, _walked, census,
                     depot_name, revision_figures, scarcest_trim, task_text)
from .depot import Depot
from .order import TRAINING_KINDS, walk
from .parts import scan

POINTS = 256  # texts sampled from each library, at most
SAMPLE_BYTES = 4 << 20  # bytes of texts the sample tokenizes, at most
BIG_TEXT = SAMPLE_BYTES // 16  # bytes above which a sampled text counts by its bytes alone
FOLLOWED = 50_000  # texts at the start of the dataset's order the forecast walks, at most
UNEVEN = 0.1  # how far a share's part of the plain text may be from its proportion and still count as even
STEPS = 40  # the sizes even_size tries, at most


def estimate(depot: Depot, dataset: dict, tokenizer: Tokenizer, max_tokens: int | None, url: str | None = None,
             workers: int = 1) -> dict:
    """The estimate of the corpus of depot's dataset (its row) built with
    tokenizer at the size max_tokens (None: every text the rules take),
    whose build tokenizes in workers processes once it reads more than
    wpt_depot.corpus.IN_PROCESS bytes: {tokens, plain_tokens, item_tokens,
    item_share (the items' share of a size: their part of the bytes of the
    corpus without max_tokens), share_tokens (with shares: [{tag, tokens,
    part (the tokens its proportion of the size gives it)}], else None),
    forecast (with max_tokens: whether the walk closed every part, so the
    tokens follow the build's order rather than proportions; else None),
    even_size, texts (the plain texts and the items), plain_texts (as the
    corpus record's texts counts them), item_texts (the conversations and
    tasks it takes), items_held (those the dataset holds), bytes,
    token_bytes, table_bytes, read_bytes, seconds, workers, figures
    (whether the revision's figures are known), sample: {texts, tokenized,
    bytes, seconds}, warnings}.
    NotFound and NotADepot as reading the libraries meets them."""
    holder = depot_name(depot, dataset, url)
    names, opened = _depots(depot, dataset, holder)
    shares = dataset["shares"]
    tags = None if shares is None else [one["tag"] for one in shares]
    pools = [0] if shares is None else list(range(len(shares)))
    weight = {"plain": dict.fromkeys(pools, 0.0), "plain_texts": dict.fromkeys(pools, 0.0), "items": 0.0,
              "item_texts": 0.0, "task_bytes": 0.0}
    sampled = []  # (source name, id, revision, kind, pool, bytes)
    for name, ref in zip(names, dataset["libraries"]):
        source = opened[name]
        members = source.counts()["library"].get(ref["library"], 0)
        rows = source.library_spread(ref["library"], ref["change"], POINTS)
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
    figures = revision_figures(depot, dataset)["measure"] is not None
    if not figures:
        warnings.append("the figures of this revision of the dataset are not known, so its build first reads the "
                        "size of every text it holds, which takes long on a big dataset"
                        + (", and this estimate takes the items' part from its sample instead"
                           if max_tokens is not None else "")
                        + ": reading the dataset again on its page stores them for this depot's own libraries")
    available = {pool: weight["plain"][pool] * plain_rate for pool in pools}  # tokens each pool may take
    items = weight["items"] * item_rate
    item_share = weight["items"] / (weight["items"] + _full_plain(weight["plain"], shares)) if weight["items"] else 0.0
    whole = sum(one["share"] for one in shares) if shares else 1.0
    proportion = {pool: (shares[pool]["share"] / whole if shares else 1.0) for pool in pools}
    parts, forecast, even_size, walked = None, None, None, None
    items_held = weight["item_texts"]
    if max_tokens is not None:
        _, libraries = _walked(dataset, names, opened)
        if figures:  # the items' part as the build counts it, from the stored figures
            counted = census(depot, dataset, libraries)
            item_bytes, pool_bytes, items_held = counted["item_bytes"], counted["pool_bytes"], counted["items"]
        else:  # never every text of the dataset in a request: the sample's weights stand for them
            item_bytes, pool_bytes, items_held = weight["items"], weight["plain"], weight["item_texts"]
        item_share = (item_bytes / (item_bytes + _full_plain(pool_bytes, shares)) if item_bytes else 0.0)
        followed = _followed(libraries, tags)
        items_taken = min(items, max_tokens * item_share)
        parts = {pool: (max_tokens - items_taken) * proportion[pool] for pool in pools}
        plain = {pool: min(available[pool], parts[pool]) for pool in pools}
        walked = _walk(followed, max_tokens, item_share, shares, plain_rate, item_rate)
        forecast = walked["complete"]
        if walked["items"] is not None:
            items_taken = walked["items"]
        for pool, tokens in walked["plain"].items():
            if tokens is not None:
                plain[pool] = tokens
        if shares and any(proportion.values()):
            even_size = _even_size(followed, max_tokens, item_share, shares, plain_rate, item_rate, proportion)
    elif shares is not None:
        items_taken = items
        empty = [shares[pool]["tag"] for pool in pools if not weight["plain"][pool]]
        if empty:
            warnings.append(f"no sampled text has the tag {', '.join(empty)}: if the dataset holds none, the build is "
                            "refused; the estimate takes no plain text")
            plain = dict.fromkeys(pools, 0.0)
        else:
            plain = {pool: size * plain_rate for pool, size in scarcest_trim(weight["plain"], shares).items()}
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
    item_texts = weight["item_texts"] * taken_share
    if walked is not None:  # the walk counts the texts it takes, as the build does, where it closed their parts
        plain_texts = plain_texts if walked["plain_texts"] is None else walked["plain_texts"]
        item_texts = item_texts if walked["item_texts"] is None else walked["item_texts"]
    texts = plain_texts + item_texts
    stored_tokens = plain_tokens + (items_taken - weight["task_bytes"] * taken_share * item_rate)
    token_bytes = stored_tokens * measured["bytes_per_token"] + weight["task_bytes"] * taken_share
    table_bytes = texts * measured["bytes_per_row"]
    read_bytes = (plain_tokens / plain_rate if plain_rate else 0.0) + weight["items"] * taken_share
    if max_tokens is not None:  # the items it takes are read once more, to count their tokens first
        read_bytes += weight["items"] * taken_share
    used = workers if read_bytes > IN_PROCESS else 1
    speed = measured["bytes"] / measured["seconds"] if measured["seconds"] else 0.0
    share_tokens = None if shares is None else [
        {"tag": shares[pool]["tag"], "tokens": round(plain[pool]),
         "part": round(parts[pool] if parts is not None else plain[pool])} for pool in pools]
    return {"tokens": round(plain_tokens + items_taken), "plain_tokens": round(plain_tokens),
            "item_tokens": round(items_taken), "item_share": item_share, "share_tokens": share_tokens,
            "forecast": forecast, "even_size": even_size, "texts": round(texts), "plain_texts": round(plain_texts),
            "item_texts": round(item_texts), "items_held": round(items_held),
            "bytes": round(token_bytes + table_bytes), "token_bytes": round(token_bytes),
            "table_bytes": round(table_bytes), "read_bytes": round(read_bytes),
            "seconds": round(read_bytes / (speed * used)) if speed else None, "workers": used,
            "figures": figures,
            "sample": {"texts": len(sampled), "tokenized": measured["texts"], "bytes": measured["bytes"],
                       "seconds": round(measured["seconds"], 3)},
            "warnings": warnings}


def _followed(libraries: list[tuple], tags: list[str] | None) -> dict:
    """The texts the corpus may take among the first FOLLOWED texts of the
    dataset's order (wpt_depot.order.walk over libraries, as
    wpt_depot.corpus._walked gives them), whatever it takes or not: {texts:
    [(kind, pool, bytes)] in order, whole: whether they are every text}."""
    texts, read = [], 0
    for entry in walk(libraries, tags):
        if read == FOLLOWED:
            return {"texts": texts, "whole": False}
        read += 1
        if entry.taken:
            texts.append((entry.kind, entry.pool, entry.bytes))
    return {"texts": texts, "whole": True}


def _walk(followed: dict, max_tokens: int, item_share: float, shares: list[dict] | None, plain_rate: float,
          item_rate: float) -> dict:
    """The build's rule at max_tokens over the texts followed (_followed),
    each text's tokens its bytes at the rates (a plain text one more, for
    the END after it): the items fill their part as one pool, then each
    share (pool 0 without shares) fills its proportion of what is left
    (wpt_depot.corpus.Fill). {items: their tokens, plain: {pool: its
    tokens}, item_texts and plain_texts: how many items and plain texts it
    takes, complete}: a value is None where the walk ran out of texts before
    that part was known (closed, or a text left out), unless the texts
    followed are every text, or, for the items, the dataset holds none
    (item_share 0)."""
    texts, whole = followed["texts"], followed["whole"]
    items = Fill({0: int(max_tokens * item_share)})
    item_texts, items_left = 0, False
    for kind, _, size in texts:
        if kind is None:
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
        if kind is not None:
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
    items_known = whole or items.done() or items_left or item_share == 0  # a dataset without items gives them nothing
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
    """The bytes of plain text the corpus without max_tokens takes, from
    the bytes each pool may take: every one without shares, else each
    share trimmed as the build trims it (wpt_depot.corpus.scarcest_trim)."""
    return sum((plain if shares is None else scarcest_trim(plain, shares)).values())


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
