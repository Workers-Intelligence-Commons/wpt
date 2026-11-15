"""Building a dataset's corpus into the store: what training reads, which
says everything about what it holds even once the depots it read change or
vanish, and the sample of a dataset that trains a vocabulary.

The build is a stream. It reads the dataset's texts in their seeded order
(wpt_depot.order: every text of its libraries, each library as it stood at
the change number of its depot the dataset names, a text two libraries
hold once, choosing nothing by score) from an order file in its work
folder, tokenizes them in worker processes (wpt_utils.parallel), and takes
their results in that order. A token file (wpt_store.shards) is uploaded to
the store as soon as it is full, with the part of the texts table that
lists its texts, and its local copies deleted, so the memory and the local
disk the build takes stay bounded whatever the dataset's size: one token
file of each kind and one part of the texts table at a time, the order
file (about 50 bytes a text) and the tasks.jsonl it writes. After each
upload it hands its caller the state that resumes it (checkpoint): a build
started again with that state and the same work folder goes on from the
first text after the last token file uploaded, and writes the same bytes
as a build that never stopped.

A build is one dataset, one vocabulary and one size: max_tokens, a number
of tokens or none (every text the rules below take), which its caller
always names. The dataset's own max_tokens is never read here: it is only
the size the web app's build form starts at.

What it takes: a plain text counts for the first share whose tag it has,
and a plain text that has none of their tags is left out; items of a kind
other than conversation and task are no training data and are left out.
With max_tokens, the corpus's tokens, of plain texts, conversations and
tasks together, never pass it:

- the items (conversations and tasks) are taken first, in order, before
  the build writes anything: conversations stop at the first conversation
  whose tokens would pass what is left of max_tokens, and tasks at the
  first such task, never scanning on for a smaller one. A conversation
  counts the tokens of its chat template; a task, kept as a line of
  tasks.jsonl and never tokenized, counts the tokens of its content as the
  JSON text of that line (task_text);
- each share (or, without shares, every plain text) may take its
  proportion of what the items left of max_tokens, in tokens, and stops at
  its first text that would pass it, never scanning on for a smaller one.

Without max_tokens, every conversation and task is taken, and the plain
texts go in order until a limit stops them, a share at a time:

- with shares, the scarcest share (the one with the
  fewest UTF-8 bytes relative to its proportion) takes every text, and
  each other share stops at its first text whose bytes would pass its
  proportion of the scarcest share's bytes;
- with neither, every plain text.

A text is read from its parts (wpt_depot.parts), so a text bigger than
memory is read as a stream: a plain text is encoded a block at a time, each
block ending right before a line's first byte that is not whitespace, a
place where the tokenizer's chunks always break, so its tokens are those
of the whole text; only its tokens are held whole, the row of a token file
being one text. A plain text becomes tokens followed by END; a
conversation becomes the tokens of the chat template (wpt_tokenizer.chat),
which start with END and mark nothing: what the assistant wrote is told by
the template's special tokens. Both go into token files in the store's
shard format, the plain texts' and the conversations' apart, each row one
text, its `source` its depot's name (below), its `document` the text's id there,
its `pool` the share's tag (empty without shares) and its `pass` 0. A
token file closes once it holds SHARD_TOKENS tokens, and closes the other
kind's with it. A task is kept as one JSON line of tasks.jsonl.

The corpus's tables are Parquet files (CORPUS_SCHEMAS): the dataset's row
(its id and revision included, its max_tokens the build's size) copied with
the time of the build; every
text taken with its revision, its credits, its tags and its scores (each
with the content version it was made from) copied and `tokens`, the
position of its first token over the token files of its own kind in order
(a task's: its line in tasks.jsonl), in parts, one uploaded with each
token file, in the corpus's order; the licenses its texts name; the
notices (ids carry their depot's name, so they are copied as they are).
Every text taken is in the corpus once (shares_tokens in the record says
how many tokens each share holds).

A corpus names a depot by what no place of its file changes, and never
by a file's path: the depot that holds the dataset by its public address
(depot_name: the WPT_DEPOT_URL its caller passes as url, the upstream's web
address for a dataset a clone's pull brought, else the depot's name), and
another local depot by the name in its own settings (its about row's). That
name is the one the order's keys, the build's key and its erasures, the
token files' `source`, the texts table, tasks.jsonl and the corpus's copy
of the dataset's row carry. A build where two depots it reads would carry
one name is refused (Refused). Whatever reads the texts again finds the
other depots' files from that name by reading the dataset's libraries at
the revision the corpus took (sources). So the same depot state builds the
same corpus wherever its files are, and a clone builds its upstream's.

Everything goes under corpora/<dataset name>-<first 8 hex characters of the
build's key>/ in the store, the record corpus.json last, so a build that
fails leaves no record. The build's key covers what decides the corpus:
the depot's name, the dataset's id and its row as the corpus copies it
(without its revision, its change and the vocabulary it publishes, with
the build's size), the vocabulary's
sha256 and the erasures every depot read has carried out on a text of the
dataset's libraries, so the same depot state, dataset, vocabulary and size
build into the same folder, another vocabulary or size into another, and
building again a corpus whose files were deleted writes the same bytes
there. The corpus id covers the dataset's id and row, the vocabulary and
every file's content (each token file by its token hash, each part of the
texts table by a hash of its rows), so building the same corpus again
gives the same id. The record names the depot that holds the dataset (its
name), the dataset's row (its id and the revision built from included),
the vocabulary (its recipe, version, file and sha256), the size
(max_tokens, null for none), the number of texts and conversations it holds, the
size of every file, and a fingerprint of each part a training step reads,
`fingerprints`: "text" (the vocabulary and the plain texts' token files),
"conversations" (the vocabulary and the conversations' token files) and
"tasks" (the vocabulary and tasks.jsonl), so a step's result is found again
only for the same part. A corpus whose record exists and is fresh is not
built again (wpt_depot.corpora says what the store keeps of corpora).

An erasure carried out while the build ran is checked before the record is
written and again after: when it reached a text the corpus took, the
record is written stale and the build deletes no other corpus. A build
resumed, or about to write its record, whose uploaded files are no longer
in the store (another build's cap deleted its folder, which had no record
yet) starts over, or fails (FilesGone) without a record. Before its first
upload, the build hands the cap its estimated size (Room.expect).

    build(depot, name, store, tokenizer, ...)   the corpus of a dataset, its record
    depot_name(depot, dataset, url)             the name a corpus of the dataset records for its depot
    sources(depot, record)                      every depot a corpus read, by the name it records, opened
    sample(depot, name, sample_bytes, work)     the plain texts that train a vocabulary, as a stream
    sample_lines(texts, here)                   that stream as a depot's web app sends it
    remote_sample(url, dataset, sample_bytes)   that stream read from another depot's web app
"""

import collections
import hashlib
import json
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store.names import as_record_name
from wpt_store.shards import write_shard
from wpt_tokenizer import END, Tokenizer, chat
from wpt_utils.lines import say
from wpt_utils.parallel import Sizing, in_order, process_pool

from . import corpora
from .depot import Depot, location
from .errors import NotADepot, NotFound, Refused, UpstreamUnreachable
from .follow import Upstream
from .order import Order, seed_of
from .parts import scan

SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per token file
WHITESPACE = b" \t\n\r\x0b\x0c"  # the bytes the tokenizer's chunks treat as whitespace (wpt_tokenizer)
WHITESPACE_BYTES = np.frombuffer(WHITESPACE, np.uint8)
FOLDER, RECORD = corpora.FOLDER, corpora.RECORD
BATCH_BYTES = 8 << 20  # bytes of texts one worker's item reads and tokenizes
BATCH_TEXTS = 4096  # texts one worker's item holds at most
SAMPLE_TIMEOUT = 600  # seconds a read of a remote sample may wait for its next bytes
IN_PROCESS = 16 << 20  # bytes of texts the corpus may take below which the build tokenizes in its own process
TIME = pa.timestamp("us", tz="UTC")
NAMES = pa.list_(pa.string())
LIBRARY_REF = pa.struct([("depot", pa.string()), ("library", pa.string()), ("change", pa.int64())])
SHARE = pa.struct([("tag", pa.string()), ("share", pa.float64())])
SCORES = pa.list_(pa.struct([("name", pa.string()), ("value", pa.float64()), ("content_version", pa.int64())]))
CORPUS_SCHEMAS = {
    "dataset": pa.schema([pa.field("id", pa.string(), False), pa.field("revision", pa.int64(), False),
                          pa.field("name", pa.string(), False), pa.field("tokenizer", pa.string(), False),
                          pa.field("libraries", pa.list_(LIBRARY_REF), False), pa.field("shares", pa.list_(SHARE)),
                          pa.field("max_tokens", pa.int64()), pa.field("built_at", TIME, False)]),
    "texts": pa.schema([pa.field("depot", pa.string(), False), pa.field("id", pa.string(), False),
                        pa.field("revision", pa.int64(), False), pa.field("kind", pa.string()),
                        pa.field("title", pa.string()), pa.field("authors", NAMES),
                        pa.field("address", pa.string()), pa.field("licenses", NAMES, False),
                        pa.field("tags", NAMES, False), pa.field("notice", pa.string()),
                        pa.field("tokens", pa.int64(), False), pa.field("scores", SCORES)]),
    "licenses": pa.schema([pa.field("id", pa.string(), False), pa.field("name", pa.string(), False),
                           pa.field("text", pa.string(), False)]),
    "notices": pa.schema([pa.field("id", pa.string(), False), pa.field("text", pa.string(), False)]),
}
CREDITS = ("kind", "title", "authors", "address", "licenses", "tags", "notice", "scores")  # what a text's row copies


def build(depot: Depot, name: str, store, tokenizer: Tokenizer, tokenizer_version: int | None = None, *,
          dataset: dict | None = None, url: str | None = None, work: Path | None = None,
          checkpoint: dict | None = None, save=lambda state: None, progress=lambda state: None, room=None,
          max_tokens: int | None, shard_tokens: int = SHARD_TOKENS, recipe: str | None = None, log=say) -> dict:
    """Build the corpus of depot's dataset name (its current row, or the
    row given as dataset) into store with tokenizer (the vocabulary of the
    tokenizer recipe recipe, default the dataset's, at its version
    tokenizer_version, which the record keeps), at the size max_tokens (a
    number of tokens, None for none): the corpus's record. url: the depot's public address
    (depot_name), None when it has none.

    work: the folder the build keeps its order file and its local files in
    (default: a temporary folder, removed at the end); checkpoint: the state
    a former build in the same folder handed save, to go on from it; save:
    called with that state after each upload; progress: called with
    {stage, ...} as it goes, and may raise to stop the build; room: makes a
    wpt_depot.corpora.Room for the corpus's folder (default: no cap).

    The failures of reading the libraries (NotFound, NotADepot), shares
    that cannot be met (Refused), the cap's (TooBig, NoRoom) and files
    deleted meanwhile (FilesGone) leave no record; the files uploaded
    before stay, and the same build reuses those still there."""
    with ExitStack() as stack:
        if work is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="wpt-corpus-")))
        return _Build(depot, name, store, tokenizer, tokenizer_version, dataset, url, Path(work), checkpoint or {},
                      save, progress, room, shard_tokens, max_tokens, recipe, log).run()


def sample(depot: Depot, name: str, sample_bytes: int, work: Path, progress=lambda state: None
           ) -> Iterator[tuple[str, str, bytes]]:
    """The sample of depot's dataset name that trains a vocabulary, read as
    a stream in one pass over the corpus's order: (the depot that holds it,
    '' for depot and another depot's name as the corpus records it; its
    id; its text as UTF-8 bytes) of plain texts the
    corpus may take, each whole, each share (without shares, every plain
    text) taking its proportion of sample_bytes and stopping at its first
    text that would pass it. Items are no sample. The order file goes in the
    folder work, kept there for the next sample of the same libraries, shares
    and erasures (order_identity)."""
    dataset = depot.dataset(name)
    names, opened = _depots(depot, dataset, "")
    dataset = _named(dataset, names, "")
    order = _order(depot, dataset, names, opened, work, progress,
                   order_identity("", dataset, names, _erasures(dataset, names, opened)))
    sources = order.sources()
    try:
        shares = dataset["shares"]
        budget = _proportions(shares, sample_bytes)
        used = dict.fromkeys(budget, 0)
        closed = set()
        for entry in order.entries():
            if len(closed) == len(budget):
                break
            if entry.kind is not None or entry.pool in closed:
                continue
            if used[entry.pool] + entry.bytes > budget[entry.pool]:
                closed.add(entry.pool)
                continue
            used[entry.pool] += entry.bytes
            source_name = sources[entry.source]
            source = opened[source_name]
            row = source.listed_text(entry.id, entry.revision)
            pieces = [value for _, value in scan(source.reader(), {entry.id: row["parts"]}) if value is not None]
            yield source_name, entry.id, "".join(pieces).encode("utf-8")
    finally:
        order.close()


def sample_lines(texts: Iterator[tuple[str, str, bytes]]) -> Iterator[bytes]:
    """The sample texts (sample) as the sample route of a depot's web app
    sends them: for each text one JSON line {"depot", "text", "bytes"},
    depot as sample gives it, then the text's bytes; a last line {"end": true, "texts",
    "bytes"}. A line {"ordering": texts read} may come between texts."""
    count = total = 0
    for address, text, data in texts:
        if address is None:
            yield (json.dumps({"ordering": text}) + "\n").encode()
            continue
        yield (json.dumps({"depot": address, "text": text, "bytes": len(data)}) + "\n").encode() + data
        count, total = count + 1, total + len(data)
    yield (json.dumps({"end": True, "texts": count, "bytes": total}) + "\n").encode()


def remote_sample(url: str, dataset: str, sample_bytes: int, progress=lambda state: None
                  ) -> Iterator[tuple[str, str, bytes]]:
    """The sample of the dataset named so of the depot whose web app is at
    url, read through its sample route as sample gives it: (the depot's
    web address for its own texts, another depot's name as the route
    gives it; the text's id; its UTF-8 bytes). UpstreamUnreachable when it
    cannot be read or ends early."""
    base = Upstream(url).url
    route = f"{base}/api/datasets/{quote(dataset, safe='')}/sample?{urlencode({'bytes': sample_bytes})}"
    try:
        answer = urllib.request.urlopen(route, timeout=SAMPLE_TIMEOUT)
    except urllib.error.HTTPError as error:
        try:
            problems = "; ".join(json.loads(error.read())["problems"])
        except (ValueError, KeyError, TypeError):
            problems = error.reason
        raise UpstreamUnreachable(f"{base} answered {error.code} to the sample of {dataset}: {problems}") from None
    except OSError as error:
        raise UpstreamUnreachable(f"{base} cannot be reached for the sample of {dataset}: {error}") from None
    with answer:
        while True:
            line = answer.readline()
            if not line:
                raise UpstreamUnreachable(f"the sample of {dataset} from {base} ended before its last line")
            head = json.loads(line)
            if head.get("end"):
                return
            if "ordering" in head:
                progress({"stage": "ordering", "texts read": head["ordering"]})
                continue
            data = answer.read(head["bytes"])
            if len(data) != head["bytes"]:
                raise UpstreamUnreachable(f"the sample of {dataset} from {base} ended inside the text {head['text']}")
            yield head["depot"] or base, head["text"], data


def _proportions(shares: list[dict] | None, total: int) -> dict[int, int]:
    """{pool: its part of total}: each share's proportion of it, or all of it for the one pool without shares."""
    if shares is None:
        return {0: total}
    whole = sum(share["share"] for share in shares)
    return {at: int(total * share["share"] / whole) for at, share in enumerate(shares)}


def depot_name(depot: Depot, dataset: dict, url: str | None) -> str:
    """The name everything a corpus of the dataset (a row depot holds)
    records for depot, the same wherever its file is: the web address of
    depot's upstream when the dataset is a row the upstream owns (a clone's
    pull brought it), else url, depot's public address, else depot's name."""
    if depot.upstream_owned([dataset["id"]]):
        return depot.upstream()
    return url or depot.name()


def sources(depot: Depot, record: dict) -> dict[str, Depot]:
    """{the name of each depot a corpus's record says it read (_depots):
    that depot, opened}, depot being the one that holds its dataset; the
    files of the other depots are found again from the dataset's revision
    the record names."""
    return _depots(depot, record["dataset"], record["depot"])[1]


def _depots(depot: Depot, dataset: dict, holder: str) -> tuple[list[str], dict[str, Depot]]:
    """(for each library the dataset names, in order, the name of the depot
    it is read from: holder for depot (reads_here), another depot's own name
    (its about row's); {each such name: that depot, opened}). The libraries
    are read from the revision of the dataset that depot holds, the one the
    row dataset names, so a row whose libraries carry names (a corpus's
    copy, _named) finds its depots' files again. A file that revision
    names but no longer holds a depot (it moved since) is found by the
    depot's name among the files the dataset's latest revision names.
    NotFound when depot holds no such revision, or when a moved depot is
    in none of the latest revision's files; Refused when two depots would
    carry one name."""
    stored = depot.rows("SELECT libraries FROM datasets_revisions WHERE id = ? AND revision = ?",
                        (dataset["id"], dataset["revision"]))
    if not stored:
        raise NotFound(f"no revision {dataset['revision']} of the dataset {dataset['id']}")
    names, opened, by_file = [], {}, {}

    def view(path: str) -> Depot | None:
        if path not in by_file:
            try:
                by_file[path] = Depot.view(path)
            except NotADepot:
                by_file[path] = None
        return by_file[path]

    def moved(name: str) -> Depot:
        latest = depot.dataset_by_id(dataset["id"])["libraries"]
        for ref in latest:
            if not depot.reads_here(ref["depot"], ref["library"]):
                read = view(str(location(ref["depot"])))
                if read is not None and read.name() == name:
                    return read
        raise NotFound(f"the depot {name} the dataset {dataset['name']} read is in none of the files "
                       f"its latest revision names")

    for at, ref in enumerate(stored[0]["libraries"]):
        if depot.reads_here(ref["depot"], ref["library"]):
            name, read = holder, depot
        else:
            read = view(str(location(ref["depot"])))
            if read is None:
                given = dataset["libraries"][at]["depot"]  # a corpus's copy names the depot (_named)
                if given == ref["depot"]:
                    raise NotADepot(f"{ref['depot']} holds no depot")
                read = moved(given)
            name = read.name()
        if opened.setdefault(name, read).address != read.address:
            raise Refused(f"two depots the dataset {dataset['name']} reads carry the name {name}, "
                          f"{opened[name].address} and {read.address}: a corpus names each depot by its name")
        names.append(name)
    return names, opened


def _named(dataset: dict, names: list[str], holder: str) -> dict:
    """The dataset's row as its corpus copies it: each library of another
    depot named by that depot's name (names, from _depots), never by its
    file; a library read here as the row stores it."""
    libraries = [ref if name == holder else {**ref, "depot": name}
                 for ref, name in zip(dataset["libraries"], names, strict=True)]
    return {**dataset, "libraries": libraries}


def _erasures(dataset: dict, names: list[str], opened: dict[str, Depot]) -> dict[str, list[str]]:
    """{the name of each depot the dataset reads (_depots): the ids of the
    erasures it carried out that reached a text one of the dataset's
    libraries there holds or ever held}: an erasure of a text the dataset
    never read changes nothing of its corpus."""
    libraries = {}
    for name, ref in zip(names, dataset["libraries"]):
        libraries.setdefault(name, []).append(ref["library"])
    found = {}
    for name, held in libraries.items():
        found[name] = [one["id"] for one in opened[name].rows(
            "SELECT e.id FROM erasures e WHERE e.erased IS NOT NULL AND EXISTS (SELECT 1 FROM json_each(e.needles) n "
            "JOIN memberships m ON m.text = json_extract(n.value, '$.text') WHERE m.library IN "
            "(SELECT value FROM json_each(?))) ORDER BY e.id", (json.dumps(held),))]
    return found


def order_identity(holder: str, named: dict, names: list[str], erasures: dict) -> str:
    """What decides a dataset's order file: the name of the depot that holds
    the dataset, the row's libraries (as its corpus copies them, _named)
    and shares, each library's depot's name, and the erasures: never its
    name, tokenizer recipe or max_tokens, so every build of the dataset's
    libraries and shares reads the same order file."""
    return _fingerprint({"depot": holder, "libraries": named["libraries"], "shares": named["shares"],
                         "names": names, "erasures": erasures})


def _order(depot: Depot, dataset: dict, named: list[str], opened: dict[str, Depot], work: Path, progress,
           identity: str) -> Order:
    """The order file in work of the dataset (its row as its corpus copies
    it, _named), made when it is not whole, its sources the depots' names
    (named and opened, from _depots)."""
    work.mkdir(parents=True, exist_ok=True)
    sources, keys, refs = [], [], []
    for name, ref in zip(named, dataset["libraries"]):
        if name not in sources:
            sources.append(name)
            keys.append("" if opened[name] is depot else name)  # its name in the order's keys (wpt_depot.order)
        refs.append((sources.index(name), opened[name], ref))

    def members():
        for source, read, ref in refs:
            yield source, read.library_members(ref["library"], ref["change"])

    order = Order.made(work / "order.sqlite", members(), sources, seed_of(_content(dataset)), dataset["shares"],
                       identity, lambda count: progress({"stage": "ordering", "texts read": count}), keys)
    return order


# What one worker of the build holds: the vocabulary and the depots it read, opened once in each process
_HELD = {}


def _start_worker(vocabulary: bytes):
    _HELD.clear()
    _HELD["tokenizer"] = Tokenizer.from_bytes(vocabulary)
    _HELD["depots"] = {}


def _tokenized(batch: list[tuple]) -> list[tuple]:
    """For each (its depot's file, id, revision, kind, read) of batch: (its credits
    row, its tokens as an array, or a task's content as UTF-8 bytes, or
    None when read is False and it is not read), in order."""
    tokenizer, depots = _HELD["tokenizer"], _HELD["depots"]
    found = []
    for address, text, revision, kind, read in batch:
        if not read:
            found.append((None, None))
            continue
        source = depots.get(address)
        if source is None:
            source = depots[address] = Depot.view(address)
        row = source.listed_text(text, revision)
        encoder, pieces = (_Encoder(tokenizer), None) if kind is None else (None, [])
        for _, value in scan(source.reader(), {text: row["parts"]}):
            if value is None:
                break
            if encoder is not None:
                encoder.feed(value)
            else:
                pieces.append(value)
        if kind is None:
            data = np.concatenate([encoder.finish(), np.array([END], np.uint32)])
        elif kind == "conversation":
            data = np.asarray(chat.encode(tokenizer, json.loads("".join(pieces)))[0], np.uint32)
        else:
            data = "".join(pieces).encode("utf-8")
        found.append(({key: row[key] for key in CREDITS}, data))
    return found


def _counted(batch: list[tuple]) -> list[int]:
    """For each (its depot's file, id, revision, kind) of an item of batch: the
    tokens it takes in the corpus, in order: a conversation's tokens of the
    chat template, a task's tokens of its content as JSON text (task_text)."""
    tokenizer, depots = _HELD["tokenizer"], _HELD["depots"]
    found = []
    for address, text, revision, kind in batch:
        source = depots.get(address)
        if source is None:
            source = depots[address] = Depot.view(address)
        row = source.listed_text(text, revision)
        content = "".join(value for _, value in scan(source.reader(), {text: row["parts"]}) if value is not None)
        if kind == "conversation":
            found.append(len(chat.encode(tokenizer, json.loads(content))[0]))
        else:
            found.append(len(tokenizer.encode(task_text(json.loads(content)).encode("utf-8"))))
    return found


def task_text(content) -> str:
    """A task's content as the JSON text of its line in tasks.jsonl, whose
    tokens are what a task counts for the corpus's limit."""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


class _Build:
    """One build, as build() says: its state, the files it writes and uploads, and its stages."""

    def __init__(self, depot, name, store, tokenizer, version, dataset, url, work, checkpoint, save, progress, room,
                 shard_tokens, max_tokens, recipe, log):
        self.depot, self.store, self.tokenizer, self.version = depot, store, tokenizer, version
        self.dataset = dataset if dataset is not None else depot.dataset(name)
        self.recipe = recipe if recipe is not None else self.dataset["tokenizer"]
        self.limit = max_tokens
        if self.limit is not None and (not isinstance(self.limit, int) or isinstance(self.limit, bool)
                                       or self.limit < 1):
            raise Refused(f"a corpus's size is a whole number of tokens above 0, or none: {self.limit!r}")
        self.holder = depot_name(depot, self.dataset, url)
        self.names, self.opened = _depots(depot, self.dataset, self.holder)
        # what the corpus copies and its keys cover: the row, its max_tokens the build's size, without the
        # vocabulary the dataset publishes, which a build never reads (it is given one)
        self.named = {**{key: value for key, value in _named(self.dataset, self.names, self.holder).items()
                         if key != "vocabulary"}, "max_tokens": self.limit}
        self.work, self.state, self.save, self.progress, self.log = work, checkpoint, save, progress, log
        self.shard_tokens = shard_tokens
        self.vocabulary = tokenizer.to_bytes()
        self.vocabulary_sha256 = hashlib.sha256(self.vocabulary).hexdigest()
        self.erasures = _erasures(self.dataset, self.names, self.opened)
        self.limit_of = [self.vocabulary_sha256, self.limit]  # what the order's limit of the items is counted for
        self.key = self._key()
        self.folder = f"{FOLDER}/{as_record_name(self.dataset['name'])}-{self.key[:8]}"
        self.room = room(self.folder) if room is not None else corpora.Room(store, None, self.folder)

    def _key(self) -> str:
        """The build's key: what decides the corpus (the module says)."""
        return _fingerprint({"depot": self.holder, "dataset": self.dataset["id"],
                             "content": _content(self.named), "vocabulary": self.vocabulary_sha256,
                             "erasures": self.erasures})

    def run(self) -> dict:
        known = self.store.get_json(f"{self.folder}/{RECORD}")
        if known is not None and corpora.fresh(known):
            self.log(f"corpus {known['id']} of {self.dataset['name']} is already built")
            return known
        self.order = _order(self.depot, self.named, self.names, self.opened, self.work, self.progress,
                            order_identity(self.holder, self.named, self.names, self.erasures))
        self.sources = self.order.sources()
        try:
            if self.order.items_tokens(self.limit_of) is None:
                if self.limit is None:
                    self.order.limit_items([], 0, self.limit_of)
                else:
                    self._limit_items(self.limit)
            return self._stream()
        finally:
            self.order.close()

    def _stream(self) -> dict:
        state = self.state
        if state.get("key") == self.key and (gone := corpora.missing(self.store, self.folder, _uploaded(state))):
            self.log(f"{len(gone)} files this build uploaded are no longer in the store: it starts over")
            state = {}
        if state.get("key") != self.key:  # a state of another build starts nothing
            state = {"key": self.key, "position": 1, "shards": [], "conversation_shards": [], "texts": [],
                     "tasks": 0, "tasks_count": 0, "pools": {}, "used": {}, "closed": [], "taken": 0,
                     "conversations": 0, "text_position": 0, "conversation_position": 0}
        shares = self.dataset["shares"]
        limit = self.limit
        if limit is not None:  # the items took their tokens first (_limit_items); the shares split what is left
            budget, counted = _proportions(shares, limit - self.order.items_tokens(self.limit_of)), "tokens"
        elif shares is not None:
            budget, counted = self._byte_budget(shares), "bytes"
        else:
            budget, counted = {}, None
        used = {int(pool): value for pool, value in state["used"].items()}
        closed = set(state["closed"])
        tags = [""] if shares is None else [one["tag"] for one in shares]
        tasks_path = self.work / "tasks.jsonl"
        with open(tasks_path, "ab") as tasks_file:
            tasks_file.truncate(state["tasks"])
        self.order.con.execute("CREATE TABLE IF NOT EXISTS named (kind TEXT NOT NULL, id TEXT NOT NULL, "
                               "PRIMARY KEY (kind, id)) WITHOUT ROWID")
        writers = {"text": _Shards(self.work, "text", state["text_position"]),
                   "conversation": _Shards(self.work, "conversation", state["conversation_position"])}
        credits, named, credit_bytes = [], set(), [0]
        read_bytes = {"plain": 0, "items": 0}  # the bytes of the texts this run took: the size estimate's base
        last_item, count, plain = self.order.last_item(), self.order.count(), self.order.plain()

        def limited(entry, held_used: dict, held_closed: set, size: int) -> bool:
            """Whether a limit leaves the plain text out, size counted in its pool's unit; held_used and
            held_closed take it in (a pool closes at its first text that would pass its part)."""
            if entry.kind is not None or counted is None:
                return False
            if entry.pool in held_closed or held_used.get(entry.pool, 0) + size > budget[entry.pool]:
                held_closed.add(entry.pool)
                return True
            held_used[entry.pool] = held_used.get(entry.pool, 0) + size
            return False

        def batches():
            """The texts from the state's position on, in batches, each (entry, read): read is whether it is
            read, which a closed pool, or a limit in bytes reached, saves. The batches end once every pool
            is closed and no item follows."""
            ahead_used, ahead_closed = dict(used), set(closed)  # the limit in bytes, worked out ahead of the results
            batch, size = [], 0
            for entry in self.order.entries(state["position"]):
                if counted == "tokens":
                    ahead_closed = closed
                if counted is not None and len(ahead_closed) == len(budget) and entry.position > last_item:
                    break
                if counted == "bytes":
                    read = not limited(entry, ahead_used, ahead_closed, entry.bytes)
                else:
                    read = entry.kind is not None or entry.pool not in closed
                batch.append((entry, read))
                size += entry.bytes if read else 0
                if size >= BATCH_BYTES or len(batch) >= BATCH_TEXTS:
                    yield batch
                    batch, size = [], 0
            if batch:
                yield batch

        def upload(path: Path, key: str) -> dict:
            size = path.stat().st_size
            self.room.make(size)
            file = self.store.put_named(key, path)
            path.unlink()
            return {**file.reference(), "bytes": size}

        def estimate(local: int, final: bool) -> int:
            """The corpus's whole size in bytes, from the local files of its first upload: those files when the
            build ends with them, else their size times the bytes the corpus may take over the bytes they
            hold (the order is seeded, so its start stands for the whole)."""
            if final:
                return local
            plain_all = sum(self.order.pool_bytes().values())
            items = self.order.items_bytes()
            if counted == "bytes":
                plain = min(plain_all, sum(budget.values()))
            elif counted == "tokens":
                tokens = sum(state["pools"].values())
                plain = plain_all if not tokens else min(plain_all, int(sum(budget.values()) * read_bytes["plain"]
                                                                        / tokens))
            else:
                plain = plain_all
            return int(local * (plain + items) / max(read_bytes["plain"] + read_bytes["items"], 1))

        def checkpoint(position: int, final: bool = False):
            """Upload the token files and the texts' part begun, and hand the state that resumes after them;
            before a build's first upload, the cap's check of its estimated size (Room.expect)."""
            tasks_file.flush()
            ready = []  # (local path, key in the store, the list its reference joins, what the reference adds)
            for kind, kept in (("text", state["shards"]), ("conversation", state["conversation_shards"])):
                done = writers[kind].close()
                if done is not None:
                    ready.append((done.pop("path"), f"{self.folder}/{done['token_sha256']}.parquet", kept, done))
            if credits:
                number = len(state["texts"])
                path = self.work / f"texts-{number:05d}.parquet"
                pq.write_table(pa.Table.from_pylist(credits, schema=CORPUS_SCHEMAS["texts"]), path,
                               compression="zstd")
                ready.append((path, f"{self.folder}/texts-{number:05d}.parquet", state["texts"],
                              {"rows": len(credits), "text_bytes": credit_bytes[0],
                               "fingerprint": _fingerprint(credits)}))
                credits.clear()
                credit_bytes[0] = 0
            if ready and not (state["shards"] or state["conversation_shards"] or state["texts"]):
                self.room.expect(estimate(sum(path.stat().st_size for path, *_ in ready), final))
            for path, key, kept, added in ready:
                kept.append({**upload(path, key), **added})
            with self.order.con:
                self.order.con.executemany("INSERT OR IGNORE INTO named (kind, id) VALUES (?, ?)", sorted(named))
            named.clear()
            state.update(position=position, used={str(pool): value for pool, value in used.items()},
                         closed=sorted(closed), text_position=writers["text"].position,
                         conversation_position=writers["conversation"].position, tasks=tasks_path.stat().st_size)
            self.save(json.loads(json.dumps(state)))

        with ExitStack() as stack:
            tasks_file = stack.enter_context(open(tasks_path, "ab"))
            if _taken_bytes(self.order) > IN_PROCESS:
                sizing = Sizing("tokenizing workers")
                pool = stack.enter_context(process_pool(sizing, sizing.limit, _start_worker, (self.vocabulary,)))
                waiting = collections.deque()  # the batches handed to in_order, whose results come in their order

                def feeding():
                    for batch in batches():
                        waiting.append(batch)
                        yield self._items(batch)
                results = ((waiting.popleft(), found) for found in in_order(_tokenized, feeding(), pool, sizing))
            else:
                _start_worker(self.vocabulary)
                results = ((batch, _tokenized(self._items(batch))) for batch in batches())
            for batch, found in results:
                for (entry, read), (row, data) in zip(batch, found):
                    if counted is not None and entry.kind is None:
                        left = limited(entry, used, closed, len(data) if counted == "tokens" and read else entry.bytes)
                        if counted == "bytes" and left == read:
                            raise AssertionError(f"the limit in bytes left {entry.id} out on one pass and not the "
                                                 "other")
                        read = read and not left
                    if not read:
                        continue
                    source_name = self.sources[entry.source]
                    pool_tag = tags[entry.pool] if entry.kind is None else ""
                    if entry.kind == "task":
                        start = state["tasks_count"]
                        tasks_file.write(task_text({"depot": source_name, "id": entry.id,
                                                    "content": json.loads(data.decode("utf-8"))}).encode("utf-8")
                                         + b"\n")
                        state["tasks_count"] += 1
                    else:
                        kind = "text" if entry.kind is None else "conversation"
                        start = writers[kind].add(data, source_name, entry.id, pool_tag)
                        if entry.kind is None:
                            state["pools"][pool_tag] = state["pools"].get(pool_tag, 0) + len(data)
                        else:
                            state["conversations"] += 1
                    state["taken"] += 1
                    credits.append({"depot": source_name, "id": entry.id, "revision": entry.revision, **row,
                                    "tokens": start})
                    credit_bytes[0] += entry.bytes
                    read_bytes["plain" if entry.kind is None else "items"] += entry.bytes
                    named.update(("license", one) for one in row["licenses"])
                    if row["notice"] is not None:
                        named.add(("notice", row["notice"]))
                    if any(writer.full(self.shard_tokens) for writer in writers.values()):
                        checkpoint(entry.position + 1)
                state["position"] = batch[-1][0].position + 1
                self.progress({"stage": "reading", "position": batch[-1][0].position, "of": count,
                               "texts": state["taken"], "tokens": sum(state["pools"].values()),
                               "token files": len(state["shards"]) + len(state["conversation_shards"])})
            checkpoint(state["position"], final=True)
        self.progress({"stage": "finishing", "texts": state["taken"]})
        return self._finish(state, tasks_path, plain)

    def _items(self, batch) -> list[tuple]:
        return [(self.opened[self.sources[entry.source]].address, entry.id, entry.revision, entry.kind, read)
                for entry, read in batch]

    def _limit_items(self, limit: int):
        """Count the tokens of the items (conversations and tasks) in order,
        taking each until the first of its kind whose tokens would pass what
        is left of limit, and keep in the order file the items left out with
        the tokens of those taken (Order.limit_items)."""
        used, closed, left_out = 0, set(), []
        entries = self.order.items()

        def batches():
            batch, size = [], 0
            for entry in entries:
                batch.append(entry)
                size += entry.bytes
                if size >= BATCH_BYTES or len(batch) >= BATCH_TEXTS:
                    yield batch
                    batch, size = [], 0
            if batch:
                yield batch

        def items(batch):
            return [(self.opened[self.sources[entry.source]].address, entry.id, entry.revision, entry.kind)
                    for entry in batch]

        with ExitStack() as stack:
            if self.order.taken_bytes() - _taken_bytes(self.order) > IN_PROCESS:
                sizing = Sizing("counting workers")
                pool = stack.enter_context(process_pool(sizing, sizing.limit, _start_worker, (self.vocabulary,)))
                waiting = collections.deque()

                def feeding():
                    for batch in batches():
                        waiting.append(batch)
                        yield items(batch)
                results = ((waiting.popleft(), found) for found in in_order(_counted, feeding(), pool, sizing))
            else:
                _start_worker(self.vocabulary)
                results = ((batch, _counted(items(batch))) for batch in batches())
            done = 0
            for batch, found in results:
                for entry, tokens in zip(batch, found):
                    if entry.kind in closed or used + tokens > limit:
                        closed.add(entry.kind)
                        left_out.append(entry.position)
                    else:
                        used += tokens
                done += len(batch)
                self.progress({"stage": "counting items", "items": done, "tokens": used})
        self.order.limit_items(left_out, used, self.limit_of)

    def _byte_budget(self, shares: list[dict]) -> dict[int, int]:
        """{pool: the bytes it may take} without max_tokens: the scarcest
        share all of its bytes, each other its proportion of the scarcest's.
        Refused when a share has no text."""
        available = {at: 0 for at in range(len(shares))} | self.order.pool_bytes()
        empty = [shares[at]["tag"] for at, size in available.items() if not size]
        if empty:
            raise Refused(f"the shares cannot be met: no text of the libraries has the tag {', '.join(empty)}")
        whole = sum(share["share"] for share in shares)
        proportion = {at: share["share"] / whole for at, share in enumerate(shares)}
        scarcest = min(proportion, key=lambda at: available[at] / proportion[at])
        total = available[scarcest] / proportion[scarcest]
        return {at: available[at] if at == scarcest else int(proportion[at] * total) for at in proportion}

    def _finish(self, state: dict, tasks_path: Path, plain: int) -> dict:
        built_at = datetime.now(UTC)
        dataset = {k: v for k, v in self.named.items() if k != "change"}
        named = {kind: [one for (one,) in self.order.con.execute("SELECT id FROM named WHERE kind = ? ORDER BY id",
                                                                (kind,))] for kind in ("license", "notice")}
        licenses, notices = {}, {}
        for source_name in self.sources:
            read = self.opened[source_name]
            for row in read.rows("SELECT id, name, text FROM licenses WHERE id IN (SELECT value FROM json_each(?))",
                                 (json.dumps(named["license"]),)):
                licenses.setdefault(row["id"], row)
            for row in read.rows("SELECT id, text FROM notices WHERE id IN (SELECT value FROM json_each(?))",
                                 (json.dumps(named["notice"]),)):
                notices.setdefault(row["id"], {"id": row["id"], "text": row["text"]})
        license_rows = [licenses[one] for one in named["license"] if one in licenses]
        notice_rows = [notices[one] for one in named["notice"] if one in notices]
        tasks_sha256 = _sha256_file(tasks_path)
        plain_taken = sum(part["rows"] for part in state["texts"]) - state["conversations"] - state["tasks_count"]
        corpus_id = _fingerprint({"dataset": _content(self.named), "dataset_id": self.dataset["id"],
                                  "vocabulary": self.vocabulary_sha256,
                                  "texts": [part["fingerprint"] for part in state["texts"]],
                                  "text_shards": [s["token_sha256"] for s in state["shards"]],
                                  "conversation_shards": [s["token_sha256"] for s in state["conversation_shards"]],
                                  "tasks": tasks_sha256, "licenses": license_rows, "notices": notice_rows})

        def put(table: str, rows: list[dict]) -> dict:
            path = self.work / f"{table}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=CORPUS_SCHEMAS[table]), path, compression="zstd")
            size = path.stat().st_size
            self.room.make(size)
            reference = {**self.store.put_named(f"{self.folder}/{table}.parquet", path).reference(), "bytes": size}
            path.unlink()
            return reference

        self.room.make(len(self.vocabulary))
        vocabulary = {**self.store.put_named(f"{self.folder}/vocabulary", self.vocabulary).reference(),
                      "bytes": len(self.vocabulary)}
        left_out = {**self.order.left_out(), "trimmed": plain - plain_taken, "trimmed items": self.order.limited()}
        record = {
            "id": corpus_id, "depot": self.holder, "dataset": dataset, "folder": self.folder,
            "build": self.key, "erasures": self.erasures, "built_at": built_at.isoformat(),
            "texts": state["taken"], "conversations": state["conversations"],
            "tokenizer": {"recipe": self.recipe, "recipe_version": self.version, "sha256": self.vocabulary_sha256,
                          "vocab_size": self.tokenizer.vocab_size, "vocabulary": vocabulary},
            "max_tokens": self.limit,
            "shards": state["shards"], "conversation_shards": state["conversation_shards"],
            "tables": {"dataset": put("dataset", [{**dataset, "built_at": built_at}]), "texts": state["texts"],
                       "licenses": put("licenses", license_rows), "notices": put("notices", notice_rows)},
            "total_tokens": sum(s["tokens"] for s in state["shards"]),
            "shares_tokens": None if self.dataset["shares"] is None else [
                {"tag": share["tag"], "tokens": state["pools"].get(share["tag"], 0)}
                for share in self.dataset["shares"]],
            "left_out": left_out,
            "fingerprints": {
                "text": _fingerprint([self.vocabulary_sha256, [s["token_sha256"] for s in state["shards"]]]),
                "conversations": _fingerprint([self.vocabulary_sha256,
                                               [s["token_sha256"] for s in state["conversation_shards"]]]),
                "tasks": _fingerprint([self.vocabulary_sha256, tasks_sha256]),
            },
        }
        if state["tasks_count"]:
            size = tasks_path.stat().st_size
            self.room.make(size)
            record["tasks"] = {"file": {**self.store.put_named(f"{self.folder}/tasks.jsonl", tasks_path).reference(),
                                        "bytes": size}, "count": state["tasks_count"]}
        gone = corpora.missing(self.store, self.folder, [*_uploaded(state), *(
            [record["tasks"]["file"]] if "tasks" in record else []), vocabulary,
            *(one for one in record["tables"].values() if isinstance(one, dict))])
        if gone:
            raise corpora.FilesGone(f"{len(gone)} files of this build were deleted while it ran, {gone[0]} first: no "
                                    "record was written; build again")
        # an erasure carried out while the build ran may have reached a text it took after the build passed it:
        # checked before the record is written, and again after, for one carried out in between (an erasure
        # carried out once the record exists marks it itself, wpt_depot.corpora.mark_stale)
        now, hit = self._erased_since(self.erasures, record)
        if hit:
            record["stale"] = {"erasures": hit}
        record = self.store.put_json(f"{self.folder}/{RECORD}", record)
        _, hit = self._erased_since(now, record)
        if hit:
            record = corpora.marked_stale(self.store, record, hit)
        self.log(f"corpus {corpus_id} of {self.dataset['name']}: {state['taken']:,} texts, "
                 f"{record['total_tokens']:,} tokens of plain text in {len(record['shards'])} token files, "
                 f"{len(record['conversation_shards'])} of conversations, {state['tasks_count']:,} tasks")
        if not corpora.fresh(record):
            self.log(f"corpus {corpus_id} is stale already: the erasures {', '.join(record['stale']['erasures'])}, "
                     "carried out while it was built, reached a text it holds; build it again")
            return record
        corpora.supersede(self.store, record, self.log)
        return record

    def _erased_since(self, before: dict[str, list[str]], record: dict) -> tuple[dict[str, list[str]], list[str]]:
        """(the erasures every depot read has carried out now (_erasures),
        the ids of those not in before that reached a text the corpus holds)."""
        now = _erasures(self.dataset, self.names, self.opened)
        hit = []
        for source_name, found in now.items():
            new = set(found) - set(before.get(source_name, []))
            if not new:
                continue
            for row in self.opened[source_name].erasures():
                if row["id"] in new and corpora.holds(self.store, record, {one["text"] for one in row["needles"]}):
                    hit.append(row["id"])
        return now, sorted(hit)


def _uploaded(state: dict) -> list[dict]:
    """The references of the files a build's state says it uploaded."""
    return [*state["shards"], *state["conversation_shards"], *state["texts"]]


def _taken_bytes(order: Order) -> int:
    return sum(order.pool_bytes().values())


class _Shards:
    """Rows of one kind written as local token files, one at a time; the
    token positions go on over the files (position)."""

    def __init__(self, folder: Path, part: str, position: int):
        self.folder, self.part, self.position = folder, part, position
        self.batch, self.batch_tokens, self.pools = [], 0, {}

    def add(self, tokens, depot: str, text: str, pool: str) -> int:
        """Add one text's row: the position of its first token."""
        start = self.position
        self.batch.append((tokens, depot, str(text), pool, 0))
        self.batch_tokens += len(tokens)
        self.position += len(tokens)
        self.pools[pool] = self.pools.get(pool, 0) + len(tokens)
        return start

    def full(self, size: int) -> bool:
        return self.batch_tokens >= size

    def close(self) -> dict | None:
        """The token file of the rows added since the last one, written:
        {path, token_sha256, rows, tokens, sources}; None without a row."""
        if not self.batch:
            return None
        path = self.folder / f"{self.part}.parquet"
        digest = write_shard(path, self.batch)
        done = {"path": path, "token_sha256": digest, "rows": len(self.batch), "tokens": self.batch_tokens,
                "sources": sorted({row[1] for row in self.batch})}
        self.batch, self.batch_tokens = [], 0
        return done


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


def _content(dataset: dict) -> dict:
    """What a dataset's row says of its corpus: the row without its id, revision and change."""
    return {k: v for k, v in dataset.items() if k not in ("id", "revision", "change")}


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as held:
        while chunk := held.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()

