"""Building a dataset's corpus into the store: what training reads, which
says everything about what it holds even once the depots it read change or
vanish.

The build takes every text of the dataset's libraries, choosing nothing by
score, each library as it stood at the change number of its depot the
dataset names
(Depot.library_texts: its members then, each at its revision then, with
its content now), a text two libraries hold once. Duplicate texts are
kept. The texts go in a seeded order, the seed being the dataset's row
(without its id, revision and change), so that the build is
reproducible and the first part of the corpus is drawn from all of it:
every size of a model series reads the same corpus, the smaller sizes its
first part.

With shares, a plain text counts for the first share whose tag it has, and
a plain text that has none of their tags is left out. The build then trims
the shares to their proportions, in tokens: it keeps every text of the
scarcest share (the one with the fewest tokens relative to its proportion)
and, of each other share, the first texts of the seeded order that fit its
proportion of the scarcest share's size, skipping a text that no longer
fits and never repeating one. Items (conversations, tasks) are not plain
text and are all taken, whatever the shares. The vocabulary's sample
(`sample`) trims the same way, counting UTF-8 bytes, since the vocabulary
does not exist yet to count tokens.

A text is read from its parts (wpt_depot.parts), so a text bigger than
memory is read as a stream: a plain text is encoded a block at a time, each
block ending right before a line's first byte that is not whitespace, a
place where the tokenizer's chunks always break, so its tokens are those
of the whole text; only its tokens are held whole, the row of a token file
being one text. A plain text becomes tokens followed by END; a conversation becomes the
tokens of the chat template (wpt_tokenizer.chat), which start with END
and mark nothing: what the assistant wrote is told by the template's
special tokens. Both go into token files in the store's shard format
(wpt_store.shards), the plain texts' and the conversations' apart, each
row one text, its `source` the depot's address, its `document` the text's
id there, its `pool` the share's tag (empty without shares) and its
`pass` 0. A task is kept as one JSON line of tasks.jsonl. Items of any
other kind are no training data and are left out.

The corpus's tables are Parquet files (CORPUS_SCHEMAS): the dataset's row
(its id and revision included) copied with the time of the build; every
text taken with its credits, its tags and its scores (each with the
content version it was made from) copied and `tokens`, the position of its
first token over the token files of its own kind in order (a task's: its line in tasks.jsonl);
the licenses its texts name; the notices (ids carry their depot's name, so
they are copied as they are). Every text taken is in the corpus once
(shares_tokens in the record says how many tokens each share holds).

Everything goes under corpora/<dataset name>-<first 8 hex characters of the
corpus id>/ in the store, the record corpus.json last, so a build that
fails leaves no corpus. The corpus id covers the dataset's id and its
row (without its revision and change), the vocabulary and every file's
content (of each text: its depot, id, notice, first token and, when it has
any, its scores), so building the same corpus again finds it and writes
nothing,
and another dataset with the same content gets a corpus of its own (the
seed leaves the id out, so both draw their texts in the same order). The
record names the depot that holds the
dataset (its address as Depot.address gives it), the dataset's row (its
id and the revision built from included), the number of texts and
conversations it holds, and a fingerprint of each part a training step
reads, `fingerprints`: "text" (the vocabulary and the plain texts' token
files), "conversations" (the vocabulary and the conversations' token files)
and "tasks" (the vocabulary and tasks.jsonl), so a step's result is found
again only for the same part.

An admitted erasure of a text a corpus holds makes the corpus stale
(`mark_stale`): its record gains `stale`, the ids of the erasures, and
`wpt train` refuses it until the dataset is built again.

    records(store)                       every corpus the store holds, newest first
    find(store, corpus_id)               the corpus of that id (or of a start of it), None without one
    built_from(store, depot, dataset)    every corpus built for that depot's dataset (by id), newest first
    latest(store, depot, dataset)        the corpus last built for that depot's dataset (by id), None without one
    sample(depot, name, sample_bytes)    plain texts of a dataset, in the corpus's order, up to a number of bytes
    mark_stale(store, texts, erasures)   mark every corpus that holds one of these texts stale
"""

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store.names import as_record_name
from wpt_store.shards import write_shard
from wpt_tokenizer import END, chat
from wpt_utils.lines import say

from .depot import Depot, location
from .errors import Refused
from .parts import scan

SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per token file
WHITESPACE = b" \t\n\r\x0b\x0c"  # the bytes the tokenizer's chunks treat as whitespace (wpt_tokenizer)
WHITESPACE_BYTES = np.frombuffer(WHITESPACE, np.uint8)
FOLDER = "corpora"
RECORD = "corpus.json"
TRAINING_KINDS = ("conversation", "task")
TIME = pa.timestamp("us", tz="UTC")
NAMES = pa.list_(pa.string())
LIBRARY_REF = pa.struct([("depot", pa.string()), ("library", pa.string()), ("change", pa.int64())])
SHARE = pa.struct([("tag", pa.string()), ("share", pa.float64())])
SCORES = pa.list_(pa.struct([("name", pa.string()), ("value", pa.float64()), ("content_version", pa.int64())]))
CORPUS_SCHEMAS = {
    "dataset": pa.schema([pa.field("id", pa.string(), False), pa.field("revision", pa.int64(), False),
                          pa.field("name", pa.string(), False), pa.field("tokenizer", pa.string(), False),
                          pa.field("libraries", pa.list_(LIBRARY_REF), False), pa.field("shares", pa.list_(SHARE)),
                          pa.field("built_at", TIME, False)]),
    "texts": pa.schema([pa.field("depot", pa.string(), False), pa.field("id", pa.string(), False),
                        pa.field("kind", pa.string()), pa.field("title", pa.string()), pa.field("authors", NAMES),
                        pa.field("address", pa.string()), pa.field("licenses", NAMES, False),
                        pa.field("tags", NAMES, False), pa.field("notice", pa.string()),
                        pa.field("tokens", pa.int64(), False), pa.field("scores", SCORES)]),
    "licenses": pa.schema([pa.field("id", pa.string(), False), pa.field("name", pa.string(), False),
                           pa.field("text", pa.string(), False)]),
    "notices": pa.schema([pa.field("id", pa.string(), False), pa.field("text", pa.string(), False)]),
}


def build(depot: Depot, name: str, store, tokenizer, tokenizer_version: int | None = None,
          shard_tokens: int = SHARD_TOKENS, log=say) -> dict:
    """Build the corpus of depot's dataset name into store with tokenizer
    (wpt_tokenizer.Tokenizer, the vocabulary of the dataset's tokenizer
    recipe, at its version tokenizer_version, which the record keeps): the
    corpus's record. The failures of reading the libraries
    (NotFound, NotADepot) and shares that cannot be met
    (Refused) leave nothing in the store."""
    dataset = depot.dataset(name)
    taken, credits, left_out = _taken(depot, dataset)
    vocabulary = tokenizer.to_bytes()
    with tempfile.TemporaryDirectory(prefix="wpt-corpus-") as folder:
        work = Path(folder)
        held = _tokenized(taken, tokenizer, work)
        taken, left_out["trimmed"] = _shared(taken, dataset["shares"], lambda one: one["held"][1])
        texts = []
        text_shards = _Shards(work, "text", shard_tokens)
        conversation_shards = _Shards(work, "conversation", shard_tokens)
        tasks = []
        for one in taken:
            depot_address, row = one["depot"], one["row"]
            at, count = one["held"]
            if row["kind"] in (None, "conversation"):
                tokens = np.array(held["tokens"][at:at + count])
                shards = text_shards if row["kind"] is None else conversation_shards
                position = shards.add(tokens, depot_address, row["id"], one["pool"])
            else:
                position = len(tasks)
                content = bytes(held["tasks"][at:at + count]).decode("utf-8")
                tasks.append(json.dumps({"depot": depot_address, "id": row["id"], "content": json.loads(content)},
                                        ensure_ascii=False, sort_keys=True))
            texts.append({"depot": depot_address, "id": row["id"], "kind": row["kind"], "title": row["title"],
                          "authors": row["authors"], "address": row["address"], "licenses": row["licenses"],
                          "tags": row["tags"], "notice": row["notice"], "tokens": position,
                          "scores": row["scores"]})
        held = None  # its files close before the work folder goes
        text_shards.close()
        conversation_shards.close()
        named = sorted({license_id for one in texts for license_id in one["licenses"]})
        licenses = [credits[0][license_id] for license_id in named]
        named_notices = sorted({one["notice"] for one in texts if one["notice"] is not None})
        notice_rows = [{"id": number, "text": credits[1][number]} for number in named_notices]
        tasks_bytes = "".join(f"{line}\n" for line in tasks).encode("utf-8")
        corpus_id = _fingerprint({"dataset": _content(dataset), "dataset_id": dataset["id"],
                                  "vocabulary": hashlib.sha256(vocabulary).hexdigest(),
                                  "texts": [_text_entry(one) for one in texts],
                                  "text_shards": [s["token_sha256"] for s in text_shards.done],
                                  "conversation_shards": [s["token_sha256"] for s in conversation_shards.done],
                                  "tasks": hashlib.sha256(tasks_bytes).hexdigest(),
                                  "licenses": licenses, "notices": notice_rows})
        key = f"{FOLDER}/{as_record_name(name)}-{corpus_id[:8]}"
        known = store.get_json(f"{key}/{RECORD}")
        if known is not None and known["id"] == corpus_id and "stale" not in known:
            log(f"corpus {corpus_id} of {name} is already built")
            return known
        built_at = datetime.now(UTC)
        tables = {"dataset": [{**{k: v for k, v in dataset.items() if k != "change"}, "built_at": built_at}],
                  "texts": texts, "licenses": licenses, "notices": notice_rows}
        vocabulary_sha256 = hashlib.sha256(vocabulary).hexdigest()
        record = {
            "id": corpus_id, "depot": depot.address, "dataset": {k: v for k, v in dataset.items() if k != "change"},
            "built_at": built_at.isoformat(), "texts": len(texts),
            "conversations": sum(s["rows"] for s in conversation_shards.done),
            "tokenizer": {"recipe": dataset["tokenizer"], "recipe_version": tokenizer_version,
                          "vocab_size": tokenizer.vocab_size,
                          "vocabulary": store.put_named(f"{key}/vocabulary", vocabulary).reference()},
            "shards": [_put_shard(store, key, s) for s in text_shards.done],
            "conversation_shards": [_put_shard(store, key, s) for s in conversation_shards.done],
            "tables": {table: store.put_named(f"{key}/{table}.parquet", _parquet(work, table, found)).reference()
                       for table, found in tables.items()},
            "total_tokens": sum(s["tokens"] for s in text_shards.done),
            "shares_tokens": _shares_tokens(dataset, text_shards.pools),
            "left_out": left_out,
            "fingerprints": {
                "text": _fingerprint([vocabulary_sha256, [s["token_sha256"] for s in text_shards.done]]),
                "conversations": _fingerprint([vocabulary_sha256,
                                               [s["token_sha256"] for s in conversation_shards.done]]),
                "tasks": _fingerprint([vocabulary_sha256, hashlib.sha256(tasks_bytes).hexdigest()]),
            },
        }
        if tasks:
            record["tasks"] = {"file": store.put_named(f"{key}/tasks.jsonl", tasks_bytes).reference(),
                               "count": len(tasks)}
        record = store.put_json(f"{key}/{RECORD}", record)
    log(f"corpus {corpus_id} of {name}: {len(texts):,} texts, {record['total_tokens']:,} tokens of plain text in "
        f"{len(record['shards'])} token files, {len(record['conversation_shards'])} of conversations, "
        f"{len(tasks):,} tasks")
    return record


def sample(depot: Depot, name: str, sample_bytes: int) -> list[dict]:
    """The sample of depot's dataset name that trains a vocabulary: {depot,
    id, text} (the text as UTF-8 bytes) of plain texts the corpus would take
    (the dataset's libraries at their states, its shares trimmed by bytes),
    in the corpus's order, each whole and only while the bytes taken stay
    within sample_bytes: a text that no longer fits is passed over. Items
    are no sample."""
    dataset = depot.dataset(name)
    taken, _, _ = _taken(depot, dataset)
    taken, _ = _shared(taken, dataset["shares"], lambda one: one["row"]["bytes"])
    chosen, size = [], 0
    for one in taken:
        row = one["row"]
        if row["kind"] is None and size + row["bytes"] <= sample_bytes:
            chosen.append(one)
            size += row["bytes"]
    read = {}
    for source, entries in _by_depot(chosen):
        pieces = {}
        for text, value in scan(source.reader(), {text: one["row"]["parts"] for text, one in entries.items()}):
            if value is not None:
                pieces.setdefault(text, []).append(value)
            else:
                read[source.address, text] = "".join(pieces.pop(text, [])).encode("utf-8")
    return [{"depot": one["depot"], "id": one["row"]["id"], "text": read[one["depot"], one["row"]["id"]]}
            for one in chosen]


def _by_depot(taken: list[dict]):
    """(the depot read, {id: its entry}) for each depot the entries of taken come from."""
    found = {}
    for one in taken:
        found.setdefault(one["depot"], (one["source"], {}))[1][one["row"]["id"]] = one
    return found.values()


def _tokenized(taken: list[dict], tokenizer, work: Path) -> dict:
    """Read every text of taken in one pass over each depot's parts
    (wpt_depot.parts.scan) and keep what the corpus takes of it in files of
    work: a plain text's tokens followed by END, a conversation's tokens,
    a task's content as UTF-8 bytes. Each entry gains `held`, (where its
    tokens or bytes start in their file, how many there are). The files,
    read as arrays: {tokens, tasks}. A text's tokens are held whole in
    memory until it is read whole, and every text is read once."""
    paths = {"tokens": work / "tokens.bin", "tasks": work / "tasks.bin"}
    written = dict.fromkeys(paths, 0)
    with open(paths["tokens"], "wb") as tokens_file, open(paths["tasks"], "wb") as tasks_file:
        for source, entries in _by_depot(taken):
            reading = {}  # of each text begun: its encoder (a plain text) or its pieces (an item)
            for text, value in scan(source.reader(), {text: one["row"]["parts"] for text, one in entries.items()}):
                kind = entries[text]["row"]["kind"]
                if value is not None:
                    if kind is None:
                        reading.setdefault(text, _Encoder(tokenizer)).feed(value)
                    else:
                        reading.setdefault(text, []).append(value)
                    continue
                found = reading.pop(text, None)
                if kind is None:
                    data = np.concatenate([(found or _Encoder(tokenizer)).finish(), np.array([END], np.uint32)])
                elif kind == "conversation":
                    data = np.asarray(chat.encode(tokenizer, json.loads("".join(found or [])))[0], np.uint32)
                else:
                    data = "".join(found or []).encode("utf-8")
                    entries[text]["held"] = (written["tasks"], len(data))
                    tasks_file.write(data)
                    written["tasks"] += len(data)
                    continue
                entries[text]["held"] = (written["tokens"], len(data))
                data.astype("<u4").tofile(tokens_file)
                written["tokens"] += len(data)
    return {"tokens": np.memmap(paths["tokens"], "<u4", "r") if written["tokens"] else np.zeros(0, "<u4"),
            "tasks": np.memmap(paths["tasks"], np.uint8, "r") if written["tasks"] else np.zeros(0, np.uint8)}


class _Encoder:
    """The tokens of a plain text fed a piece at a time: each piece is
    encoded up to its last place where the tokenizer's chunks always break
    (_boundary), so the tokens are those of the whole text, and only the
    bytes after that place are held, whatever the text's lines."""

    def __init__(self, tokenizer):
        self.tokenizer, self.pending, self.last, self.tokens = tokenizer, [], None, []

    def feed(self, value: str):
        data = value.encode("utf-8")
        if not data:
            return
        cut = _boundary(data, self.last)
        if cut is None:
            self.pending.append(data)
        else:
            left = b"".join(self.pending) + data[:cut]
            if left:
                self.tokens.append(np.asarray(self.tokenizer.encode(left), np.uint32))
            self.pending = [data[cut:]]
        self.last = data[-1]

    def finish(self) -> np.ndarray:
        left = b"".join(self.pending)
        if left:
            self.tokens.append(np.asarray(self.tokenizer.encode(left), np.uint32))
        return np.concatenate(self.tokens) if self.tokens else np.zeros(0, np.uint32)


def _boundary(data: bytes, last: int | None) -> int | None:
    """The last place of data, read right after the bytes that end with
    last (None at a text's start), where a byte that is not whitespace is
    followed by whitespace (WHITESPACE): the tokenizer's chunk of that byte
    ends there whatever follows, so the bytes before it encode as they would
    within the whole text. None for no such place."""
    spaces = np.isin(np.frombuffer(data, np.uint8), WHITESPACE_BYTES)
    found = np.flatnonzero(~spaces[:-1] & spaces[1:])
    if len(found):
        return int(found[-1]) + 1
    if last is not None and last not in WHITESPACE and spaces[0]:
        return 0
    return None


def _taken(depot: Depot, dataset: dict) -> tuple[list[dict], tuple[dict, dict], dict]:
    """(the texts the corpus may take, each {depot, source: the depot read,
    row, pool}, in the corpus's order, before the shares trim them
    (_shared); ({license id: its row}, {notice id: its text}) of the depots
    read; what was left out: {"no share": count, "kinds": {kind: count}}).
    Each row is as Depot.library_texts gives it, without its text, which is
    read from its parts."""
    seen, taken, licenses, notices = set(), [], {}, {}
    no_share, other_kinds = 0, {}
    shares = dataset["shares"]
    opened = {depot.address: depot}
    for ref in dataset["libraries"]:
        where = depot.address if depot.reads_here(ref["depot"], ref["library"]) else str(location(ref["depot"]))
        if where not in opened:
            opened[where] = Depot.view(where)
        source = opened[where]
        for row in source.library_texts(ref["library"], ref["change"]):
            if (source.address, row["id"]) in seen:
                continue
            seen.add((source.address, row["id"]))
            if row["kind"] is not None and row["kind"] not in TRAINING_KINDS:
                other_kinds[row["kind"]] = other_kinds.get(row["kind"], 0) + 1
                continue
            pool = ""
            if shares is not None and row["kind"] is None:
                pool = next((share["tag"] for share in shares if share["tag"] in row["tags"]), None)
                if pool is None:
                    no_share += 1
                    continue
            taken.append({"depot": source.address, "source": source, "row": row, "pool": pool})
    for address, source in opened.items():
        for row in source.sql("SELECT id, name, text FROM licenses"):
            licenses.setdefault(row["id"], row)
        for row in source.sql("SELECT id, text FROM notices"):
            notices[row["id"]] = row["text"]
    seed = _fingerprint(_content(dataset))
    taken.sort(key=lambda one: hashlib.sha256(f"{seed}\0{one['depot']}\0{one['row']['id']}".encode()).digest())
    return taken, (licenses, notices), {"no share": no_share, "trimmed": 0, "kinds": dict(sorted(other_kinds.items()))}


def _shared(taken: list[dict], shares: list[dict] | None, size) -> tuple[list[dict], int]:
    """(taken trimmed to the shares, none trimmed without shares; how many
    texts the shares left out). size(one) is the size of a plain text that
    the shares count: its tokens, or its bytes."""
    if shares is None:
        return taken, 0
    kept = _trimmed(taken, shares, size)
    return kept, len(taken) - len(kept)


def _trimmed(taken: list[dict], shares: list[dict], size) -> list[dict]:
    """taken with its plain texts trimmed to the shares' proportions: every
    text of the scarcest share (the fewest sizes relative to its proportion)
    stays, and of each other share the first texts, in order, that fit its
    proportion of the scarcest share's total; a text that no longer fits is
    skipped, and none is repeated. Items stay. Refused when a share picks
    no text."""
    total_share = sum(share["share"] for share in shares)
    plain = [one for one in taken if one["row"]["kind"] is None]
    for one in plain:
        one["size"] = size(one)
    available = {share["tag"]: sum(one["size"] for one in plain if one["pool"] == share["tag"]) for share in shares}
    empty = [tag for tag, count in available.items() if not count]
    if empty:
        raise Refused(f"the shares cannot be met: no text of the libraries has the tag {', '.join(empty)}")
    proportion = {share["tag"]: share["share"] / total_share for share in shares}
    scarcest = min(proportion, key=lambda tag: available[tag] / proportion[tag])
    whole = available[scarcest] / proportion[scarcest]
    used = dict.fromkeys(proportion, 0)
    kept = []
    for one in taken:
        if one["row"]["kind"] is None and one["pool"] != scarcest:
            if used[one["pool"]] + one["size"] > proportion[one["pool"]] * whole:
                continue
            used[one["pool"]] += one["size"]
        kept.append(one)
    return kept


class _Shards:
    """Rows written as token files of at most about `size` tokens each."""

    def __init__(self, folder: Path, part: str, size: int):
        self.folder, self.part, self.size = folder, part, size
        self.batch, self.batch_tokens, self.position = [], 0, 0
        self.done, self.pools = [], {}

    def add(self, tokens, depot: str, text: int, pool: str) -> int:
        """Add one text's row: the position of its first token."""
        start = self.position
        self.batch.append((tokens, depot, str(text), pool, 0))
        self.batch_tokens += len(tokens)
        self.position += len(tokens)
        self.pools[pool] = self.pools.get(pool, 0) + len(tokens)
        if self.batch_tokens >= self.size:
            self._flush()
        return start

    def close(self):
        if self.batch:
            self._flush()

    def _flush(self):
        path = self.folder / f"{self.part}-{len(self.done):05d}.parquet"
        digest = write_shard(path, self.batch)
        self.done.append({"path": path, "token_sha256": digest, "rows": len(self.batch), "tokens": self.batch_tokens,
                          "sources": sorted({row[1] for row in self.batch})})
        self.batch, self.batch_tokens = [], 0


def _put_shard(store, key: str, shard: dict) -> dict:
    file = store.put_named(f"{key}/{shard['token_sha256']}.parquet", shard["path"])
    return {"key": file.key, "sha256": file.sha256, **{k: v for k, v in shard.items() if k != "path"}}


def _parquet(folder: Path, table: str, found: list[dict]) -> Path:
    path = folder / f"{table}.parquet"
    pq.write_table(pa.Table.from_pylist(found, schema=CORPUS_SCHEMAS[table]), path, compression="zstd")
    return path


def _text_entry(one: dict) -> list:
    """What the corpus id covers of a text of the corpus: its depot, id,
    notice and first token, then its scores when it has any."""
    return [one["depot"], one["id"], one["notice"], one["tokens"], *([one["scores"]] if one["scores"] else [])]


def _shares_tokens(dataset: dict, pools: dict) -> list[dict] | None:
    if dataset["shares"] is None:
        return None
    return [{"tag": share["tag"], "tokens": pools.get(share["tag"], 0)} for share in dataset["shares"]]


def _content(dataset: dict) -> dict:
    """What a dataset's row says of its corpus: the row without its id, revision and change."""
    return {k: v for k, v in dataset.items() if k not in ("id", "revision", "change")}


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def records(store) -> list[dict]:
    """The record of every corpus the store holds, each checked by the
    store, the last built first."""
    found = []
    for key in store.kind.list(f"{FOLDER}/", deep=False):
        if key.endswith("/") and (record := store.get_json(f"{key}{RECORD}")) is not None:
            found.append(record)
    return sorted(found, key=lambda record: record["built_at"], reverse=True)


def find(store, corpus_id: str) -> dict | None:
    """The record of the corpus whose id is corpus_id, or starts with it
    when only one does; None when none does. ValueError when several do."""
    found = [record for record in records(store) if record["id"].startswith(corpus_id)]
    if len(found) > 1:
        raise ValueError(f"{corpus_id!r} starts the ids of {len(found)} corpora: give more of the id")
    return found[0] if found else None


def built_from(store, depot: str, dataset: str) -> list[dict]:
    """The records of the corpora built from the dataset of id dataset of
    the depot at the address depot (a path or a file:// address, as
    Depot.open takes it), the last built first: a dataset's corpora stay
    its own through a rename, and a new dataset given an old name has none."""
    where = str(location(depot))
    return [record for record in records(store)
            if record["depot"] == where and record["dataset"].get("id") == dataset]


def latest(store, depot: str, dataset: str) -> dict | None:
    """The record of the corpus last built for the dataset of id dataset of
    the depot at the address depot (built_from); None when the store holds
    none."""
    return next(iter(built_from(store, depot, dataset)), None)


def mark_stale(store, texts, erasures) -> list[str]:
    """Mark stale every corpus of the store that holds one of the texts
    (ids), because the erasures (ids) were carried out: the ids of the
    corpora marked. A corpus already stale keeps its earlier erasures."""
    wanted = set(texts)
    marked = []
    for record in records(store):
        held = pq.read_table(store.path(record["tables"]["texts"]), columns=["id"])["id"].to_pylist()
        if wanted.isdisjoint(held):
            continue
        stale = sorted({*record.get("stale", {}).get("erasures", []), *erasures})
        store.put_json(f"{FOLDER}/{as_record_name(record['dataset']['name'])}-{record['id'][:8]}/{RECORD}",
                       {**record, "stale": {"erasures": stale}})
        marked.append(record["id"])
    return marked


def read_table(store, record: dict, table: str) -> pa.Table:
    """One of a corpus's tables (CORPUS_SCHEMAS), checked by the store."""
    return pq.read_table(store.path(record["tables"][table]))
