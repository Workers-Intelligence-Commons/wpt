"""Building a dataset's corpus into the store: what training reads, which
says everything about what it holds even once the depots it read change or
vanish, and the sample of a dataset that trains a vocabulary.

The build is a stream. It walks the dataset's texts in the order of their
export keys (wpt_depot.order: every text of its libraries, each library as
it stood at the change number of its depot the dataset names, a text two
libraries hold once, a long plain text as its pieces, choosing nothing by
score), reading only the texts it reaches, tokenizes them in worker processes (wpt_utils.parallel), and takes
their results in that order. With shares, a first pass walks the whole
order before it (wpt_depot.groups) to choose the texts of each share that
splits its budget over the tags of its texts (below), reading the rows of
the texts but never their contents, but for the texts whose tokens the
choice needs. A token file (wpt_store.shards) is uploaded to
the store as soon as it is full, with the part of the texts table that
lists its texts, and its local copies deleted, so the memory and the local
disk the build takes stay bounded whatever the dataset's size: one token
file of each kind and one part of the texts table at a time, the
tasks.jsonl it writes and the ids of the licenses and notices its texts
name (named.sqlite). After each upload it hands its caller the state that
resumes it (checkpoint): a build started again with that state and the
same work folder goes on from the first text after the last token file
uploaded, and writes the same bytes as a build that never stopped.

What a build needs of the whole dataset before it walks it (the bytes of
the plain texts of each share and of the items, which set the items' part
of a size and the shares' bytes without one, and the counts of what it
leaves out) is its census (census): the figures the dataset's revision
stores (revision_figures), or, when they are not known, the same figures
counted from every text of each library, which takes long on a big
dataset. Both count a text two libraries hold once for each, and a plain
text with several of the shares' tags once for each tag, so the same depot
state gives the same census either way.

A build is one dataset, one vocabulary and one size: max_tokens, a number
of tokens or none (every text the rules below take), which its caller
always names. The dataset's own max_tokens is never read here: it is only
the size the web app's build form starts at.

What it takes: a plain text counts for the first share whose tag it has,
and a plain text that has none of their tags is left out. Inside a share,
the budget splits over every tag its texts carry but the share's own,
each tag's part following the square root of what it holds (wpt_depot.groups): a share whose texts carry two such tags or more,
with a limit, takes the texts the groups choose, and every other share fills
as below. Items of a kind
other than conversation and task are no training data and are left out. A
plain text longer than wpt_depot.pieces.LENGTH characters is taken as its
pieces (wpt_depot.pieces), each at its own place of the order and counted
as a text of its own by every rule below, so a long text, such as a
repository the depot holds as one text, spreads over the corpus, and a
corpus of any size takes about its share of it. An item is never cut.

Every limit below is filled the same way (wpt_depot.fill.Fill), but for a
share that splits its budget by group: the texts come in order, a text that
fits what is left of its pool is taken, and one that would pass it is left
out while the walk goes on with the next. A pool closes once what is left
of it is at most a FULL-th of its budget, or once SKIPS of its texts in a
row did not fit, and the walk ends once every pool is closed, every share
that splits by group is past the last text it chose, and no item taken
comes later, so a nearly full corpus never reads the rest of the dataset
for its last tokens. A share that splits by group chooses its texts on
their exact tokens with max_tokens, which worker processes count ahead of
the choice, and on their bytes without; the stream then takes the texts it
chose, checking each against the share's budget again. The texts chosen
wait in the work folder (chosen.bin), their sha256 in the state that
resumes the build, so a build resumed takes the same texts; a file lost is
chosen again, the same.

With max_tokens, the corpus's tokens, of plain texts, conversations and
tasks together, never pass it, and a corpus of any size keeps the
dataset's mix of items and plain text:

- the items (conversations and tasks) have their natural share of
  max_tokens: the share of the UTF-8 bytes of the corpus without
  max_tokens (below: every item, and the plain texts the shares keep) that
  the items hold, rounded down to a whole token, so every size has the mix
  of the corpus of every token. Bytes, because they are known for every
  text before anything is tokenized. A share no text has counts for
  nothing there. Before the build writes anything, the items are counted in
  order and fill that part, as one pool (_limit_items). A conversation
  counts the tokens of its chat template; a task, kept as a line of
  tasks.jsonl and never tokenized, counts the tokens of its content as the
  JSON text of that line (task_text);
- each share (or, without shares, every plain text) fills its proportion
  of what the items left of max_tokens, in tokens.

So a smaller size takes the same texts as a larger one up to the first
text it leaves out, the start of what the larger one takes; past it, it
takes only what fits what is left of it. A share that splits by group takes
about the same part of each group at every size instead.

Without max_tokens, every conversation and task is taken, and the plain
texts go in order, a share at a time:

- with shares, the scarcest share (the one with the
  fewest UTF-8 bytes relative to its proportion) takes every text, and
  each other share fills its proportion of the scarcest share's bytes, a
  piece counting for its share of its text's bytes (wpt_depot.order);
- with neither, every plain text.

A text is read from its parts (wpt_depot.parts), so a text bigger than
memory is read as a stream: a plain text is encoded a block at a time, each
block ending right before a line's first byte that is not whitespace, a
place where the tokenizer's chunks always break, so its tokens are those
of the whole text; only its tokens are held whole. A piece is read from its
own characters alone (wpt_depot.pieces.read) and encoded as a text of its
own. A plain text or a piece becomes tokens followed by END; a
conversation becomes the tokens of the chat template (wpt_tokenizer.chat),
which start with END and mark nothing: what the assistant wrote is told by
the template's special tokens. Both go into token files in the store's
shard format, the plain texts' and the conversations' apart, each row one
text or one piece, its `source` its depot's name (below), its `document`
the text's id there or the piece's name (wpt_depot.pieces.name), its
`pool` the share's tag (empty without shares) and its `pass` 0. A
token file closes once it holds SHARD_TOKENS tokens, and closes the other
kind's with it. A task is kept as one JSON line of tasks.jsonl.

The corpus's tables are Parquet files (CORPUS_SCHEMAS): the dataset's row
(its id and revision included, its max_tokens the build's size) copied with
the time of the build; every
text taken with its revision, its credits, its tags and its scores (each
with the content version it was made from) copied, `piece` (null for a
whole text) and `tokens`, the
position of its first token over the token files of its own kind in order
(a task's: its line in tasks.jsonl), in parts, one uploaded with each
token file, in the corpus's order; a piece taken has a row of its own,
`piece` its number, with its text's credits; the licenses its texts name; the
notices (ids carry their depot's name, so they are copied as they are).
Every text taken is in the corpus once, whole or as some of its pieces,
each piece once (shares_tokens in the record says how many tokens each
share holds, and for each group of the share, its tag, null for the texts
that carry no other tag, how many texts it holds and their tokens, a text
counting for each of its groups).

A corpus names a depot by what no place of its file changes, and never
by a file's path: the depot that holds the dataset by its public address
(depot_name: the WPT_DEPOT_URL its caller passes as url, the upstream's web
address for a dataset a clone's pull brought, else the depot's name), and
another local depot by the name in its own settings (its about row's). That
name is the one the build's key and its erasures, the
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
sha256, the version of these rules (RULES) and the erasures every depot read has carried out on a text of the
dataset's libraries, so the same depot state, dataset, vocabulary and size
build into the same folder, another vocabulary or size into another, and
building again a corpus whose files were deleted writes the same bytes
there. The corpus id covers the dataset's id and row, the vocabulary and
every file's content (each token file by its token hash, each part of the
texts table by a hash of its rows), so building the same corpus again
gives the same id. The record names the depot that holds the dataset (its
name), the dataset's row (its id and the revision built from included),
the vocabulary (its recipe, version, file and sha256), the size
(max_tokens, null for none), the number of texts (a text taken as pieces
counting once), of pieces and of conversations it holds, what it left out
(left_out: plain texts no share takes, plain texts of which it took
nothing, items it left out, items of other kinds), the
tokens of its items (items_tokens: its conversations' and its tasks'), the
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
    sample(depot, name, sample_bytes)           the plain texts that train a vocabulary, as a stream
    census(depot, dataset, libraries)           what a corpus of the dataset may take, whatever its size
    sample_lines(texts, here)                   that stream as a depot's web app sends it
    remote_sample(url, dataset, sample_bytes)   that stream read from another depot's web app
"""

import collections
import hashlib
import json
import tempfile
import urllib.error
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlencode

import apsw
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store.names import as_record_name
from wpt_store.shards import write_shard
from wpt_tokenizer import END, Tokenizer, chat
from wpt_utils.lines import say
from wpt_utils.parallel import Sizing, in_order, process_pool
from wpt_utils import web

from . import corpora, counts, groups, pieces
from .depot import Depot, location
from .errors import NotADepot, NotFound, Refused, UpstreamUnreachable
from .fill import Fill
from .follow import Upstream
from .order import HIGHEST, LOWEST, TRAINING_KINDS, long_texts, walk
from .parts import Parted, scan

SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per token file
WHITESPACE = b" \t\n\r\x0b\x0c"  # the bytes the tokenizer's chunks treat as whitespace (wpt_tokenizer)
WHITESPACE_BYTES = np.frombuffer(WHITESPACE, np.uint8)
FOLDER, RECORD = corpora.FOLDER, corpora.RECORD
BATCH_BYTES = 8 << 20  # bytes of texts one worker's item reads and tokenizes
BATCH_TEXTS = 4096  # texts one worker's item holds at most
SAMPLE_TIMEOUT = 600  # seconds a read of a remote sample may wait for its next bytes
MEASURED = 100_000  # rows a census that counts libraries reads between two reports of its progress
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
                        pa.field("tokens", pa.int64(), False), pa.field("scores", SCORES),
                        pa.field("piece", pa.int64())]),
    "licenses": pa.schema([pa.field("id", pa.string(), False), pa.field("name", pa.string(), False),
                           pa.field("text", pa.string(), False)]),
    "notices": pa.schema([pa.field("id", pa.string(), False), pa.field("text", pa.string(), False)]),
}
RULES = 6  # the version of the rules of what a build takes, in the build's key: raised whenever they change
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

    work: the folder the build keeps its local files in
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


def sample(depot: Depot, name: str, sample_bytes: int) -> Iterator[tuple[str, str, bytes]]:
    """The sample of depot's dataset name that trains a vocabulary, read as
    a stream in one walk of the corpus's order (wpt_depot.order): (the
    depot that holds it, '' for depot and another depot's name as the
    corpus records it; its id; its text as UTF-8 bytes) of plain texts and
    pieces the corpus may take, a piece under its text's id, each share
    (without shares, every plain text) filling its proportion of
    sample_bytes as a corpus fills a budget (Fill, or the choice by group
    of wpt_depot.groups on bytes, a piece counting for its share of its
    text's bytes). Items are no sample. With shares, the first pass of that
    choice walks the whole order before the first text comes. The walk
    ends once every share has closed."""
    dataset = depot.dataset(name)
    names, opened = _depots(depot, dataset, "")
    sources, libraries = _walked(dataset, names, opened)
    shares = dataset["shares"]
    budget = _proportions(shares, sample_bytes)
    long = long_texts(libraries)
    chosen = _no_plan() if shares is None else groups.plan(libraries, _tags(shares), budget, long)
    picker = chosen.picker()
    fill = Fill({pool: limit for pool, limit in budget.items() if not picker.covers(pool)})
    for entry in walk(libraries, _tags(shares), long=long):
        if fill.done() and picker.done():
            break
        if not entry.taken or entry.kind is not None:
            continue
        if picker.covers(entry.pool):
            if not picker.pick(entry.pool)[1]:
                continue
        elif not fill.take(entry.pool, entry.bytes):
            continue
        source_name = sources[entry.source]
        source = opened[source_name]
        row = source.listed_text(entry.id, entry.revision)
        yield source_name, entry.id, _plain(source, entry.id, row["parts"], entry.piece).encode("utf-8")


def _no_plan() -> groups.Plan:
    """The plan of a dataset whose shares choose nothing by group (wpt_depot.groups)."""
    return groups.Plan({}, np.zeros(0, np.int64))


def _plain(source: Depot, text: str, parts: list, piece: int | None) -> str:
    """A plain text of source whose content version lists parts, whole, or its piece of that number."""
    if piece is not None:
        return pieces.read(Parted(source.reader(), text, parts), piece)
    return "".join(value for _, value in scan(source.reader(), {text: parts}) if value is not None)


def sample_lines(texts: Iterator[tuple[str, str, bytes]]) -> Iterator[bytes]:
    """The sample texts (sample) as the sample route of a depot's web app
    sends them: for each text one JSON line {"depot", "text", "bytes"},
    depot as sample gives it, then the text's bytes; a last line {"end": true, "texts",
    "bytes"}. A line {"waiting": true} may come between texts, for each
    (None, None, b"") of texts: the sample is slow to find its next text."""
    count = total = 0
    for address, text, data in texts:
        if address is None:
            yield (json.dumps({"waiting": True}) + "\n").encode()
            continue
        yield (json.dumps({"depot": address, "text": text, "bytes": len(data)}) + "\n").encode() + data
        count, total = count + 1, total + len(data)
    yield (json.dumps({"end": True, "texts": count, "bytes": total}) + "\n").encode()


def remote_sample(url: str, dataset: str, sample_bytes: int) -> Iterator[tuple[str, str, bytes]]:
    """The sample of the dataset named so of the depot whose web app is at
    url, read through its sample route as sample gives it: (the depot's
    web address for its own texts, another depot's name as the route
    gives it; the text's id; its UTF-8 bytes). UpstreamUnreachable when it
    cannot be read or ends early."""
    base = Upstream(url).url
    route = f"{base}/api/datasets/{quote(dataset, safe='')}/sample?{urlencode({'bytes': sample_bytes})}"
    try:
        answer = web.urlopen(route, timeout=SAMPLE_TIMEOUT)
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
            if head.get("waiting"):
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


def _walked(dataset: dict, names: list[str], opened: dict[str, Depot]) -> tuple[list[str], list[tuple]]:
    """(the names of the depots the dataset reads, each once, in the order
    of its libraries: what an Entry's source indexes; for each library, in
    order, (its source, its depot opened, its id, its change number), what
    wpt_depot.order.walk reads), from names and opened (_depots)."""
    sources, libraries = [], []
    for name, ref in zip(names, dataset["libraries"], strict=True):
        if name not in sources:
            sources.append(name)
        libraries.append((sources.index(name), opened[name], ref["library"], ref["change"]))
    return sources, libraries


def _tags(shares: list[dict] | None) -> list[str] | None:
    """The tags of a dataset's shares, in order; None without shares."""
    return None if shares is None else [one["tag"] for one in shares]


def census(depot: Depot, dataset: dict, libraries: list[tuple], progress=lambda state: None) -> dict:
    """What a corpus of depot's dataset (its row) may take whatever its
    size: {pool_bytes: {pool: the bytes of the plain texts of that pool
    (wpt_depot.order)}, plain: how many such texts, item_bytes and items:
    the bytes and number of the conversations and tasks, no_share: the
    plain texts no share takes, other: the items of the other kinds,
    texts: every text, known: whether they come from stored figures}, all
    from the figures of the dataset's libraries summed (counts.summed),
    which count a text two libraries hold once for each, and a plain text
    with several tags of the shares once for each of them (in each of
    those shares' pool), where the order takes it once, in its first
    share. The figures are those the dataset's revision stores
    (revision_figures), read at once, when every library's are known.
    Otherwise each library's are counted the same way from its rows (as
    libraries _walked gives them), which reads the row of each text and
    takes long on a big dataset (known False), progress getting {stage:
    "measuring", texts read} now and then; so both give the same census
    for the same depot state."""
    shares = dataset["shares"]
    tags = _tags(shares) or []
    measure = revision_figures(depot, dataset)["measure"]
    known = measure is not None
    if not known:
        read = [0]
        measure = counts.summed([_library_figures(one, tags, read, progress) for one in libraries], tags)
    if shares is None:
        pool_bytes, plain, no_share = {0: measure["plain"]["bytes"]}, measure["plain"]["texts"], 0
    else:
        pool_bytes = {at: one["bytes"] for at, one in enumerate(measure["shares"])}
        plain = sum(one["texts"] for one in measure["shares"])
        no_share = measure["unshared"]["texts"]
    return {"pool_bytes": pool_bytes, "plain": plain, "item_bytes": measure["items"]["bytes"],
            "items": measure["items"]["texts"], "no_share": no_share, "other": measure["other"]["texts"],
            "texts": measure["texts"], "known": known}


def _library_figures(library: tuple, tags: list[str], read: list[int], progress) -> dict:
    """The figures of one library (source, its depot opened, its id, its
    change number), as wpt_depot.counts.figures gives them for tags, its
    characters left at 0 (a census reads none), counted from its rows at
    that change (Depot.library_range); read[0] counts the rows read across
    calls, progress getting {stage: "measuring", texts read} every
    MEASURED of them."""
    _, held, name, change = library
    kinds = {kind: dict.fromkeys(counts.COLUMNS, 0) for kind in counts.KINDS}
    tagged = [dict.fromkeys(counts.COLUMNS, 0) for _ in tags]
    for _, _, _, kind, text_tags, size in held.library_range(name, change, LOWEST, HIGHEST):
        what = "plain" if kind is None else "items" if kind in TRAINING_KINDS else "other"
        into = [kinds[what]]
        if kind is None:
            held_tags = json.loads(text_tags or "[]")
            into += [tagged[at] for at, tag in enumerate(tags) if tag in held_tags]
        for one in into:
            one["texts"] += 1
            one["bytes"] += size or 0
        read[0] += 1
        if read[0] % MEASURED == 0:
            progress({"stage": "measuring", "texts read": read[0]})
    return kinds | {"tags": tagged}


# What one worker of the build holds: the vocabulary and the depots it read, opened once in each process
_HELD = {}


def _start_worker(vocabulary: bytes):
    _HELD.clear()
    _HELD["tokenizer"] = Tokenizer.from_bytes(vocabulary)
    _HELD["depots"] = {}


def _tokenized(batch: list[tuple]) -> list[tuple]:
    """For each (its depot's file, id, revision, kind, read, piece) of
    batch: (its credits row, its tokens as an array, or a task's content
    as UTF-8 bytes, or None when read is False and it is not read), in
    order; a piece (its number, None for a whole text) read and encoded
    alone (wpt_depot.pieces)."""
    tokenizer, depots = _HELD["tokenizer"], _HELD["depots"]
    found = []
    for address, text, revision, kind, read, piece in batch:
        if not read:
            found.append((None, None))
            continue
        source = depots.get(address)
        if source is None:
            source = depots[address] = Depot.view(address)
        row = source.listed_text(text, revision)
        encoder, held = (_Encoder(tokenizer), None) if kind is None else (None, [])
        if piece is not None:
            encoder.feed(pieces.read(Parted(source.reader(), text, row["parts"]), piece))
        else:
            for _, value in scan(source.reader(), {text: row["parts"]}):
                if value is None:
                    break
                if encoder is not None:
                    encoder.feed(value)
                else:
                    held.append(value)
        if kind is None:
            data = np.concatenate([encoder.finish(), np.array([END], np.uint32)])
        elif kind == "conversation":
            data = np.asarray(chat.encode(tokenizer, json.loads("".join(held)))[0], np.uint32)
        else:
            data = "".join(held).encode("utf-8")
        found.append(({key: row[key] for key in CREDITS}, data))
    return found


def _plain_counted(batch: list[tuple]) -> list[int]:
    """For each (its depot's file, id, revision, piece) of a plain text or
    a piece of batch: the tokens it takes in the corpus, END included, as
    _tokenized gives them, in order."""
    return [len(data) for _, data in _tokenized([(address, text, revision, None, True, piece)
                                                 for address, text, revision, piece in batch])]


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
            found.append(task_tokens(tokenizer, json.loads(content)))
    return found


def task_text(content) -> str:
    """A task's content as the JSON text of its line in tasks.jsonl, whose
    tokens are what a task counts for the corpus's limit."""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def task_tokens(tokenizer: Tokenizer, content) -> int:
    """The tokens a task's content counts for the corpus (task_text)."""
    return len(tokenizer.encode(task_text(content).encode("utf-8")))


def scarcest_trim(available: dict[int, float], shares: list[dict]) -> dict[int, float]:
    """{pool: what it may take} of the corpus without max_tokens, from what
    each pool of shares holds (bytes or tokens): the scarcest share all it
    holds, each other its proportion of the scarcest's, at most all it
    holds. A share that holds nothing takes nothing, and the others are
    measured among themselves."""
    held = [pool for pool in available if available[pool]]
    if not held:
        return dict.fromkeys(available, 0)
    whole = sum(share["share"] for share in shares)
    proportion = {pool: shares[pool]["share"] / whole for pool in available}
    scarcest = min(held, key=lambda pool: available[pool] / proportion[pool])
    total = available[scarcest] / proportion[scarcest]
    return {pool: available[pool] if pool == scarcest else min(available[pool], proportion[pool] * total)
            for pool in available}


def measure(depot: Depot, libraries: list[str], shares: list[dict]) -> dict:
    """What the libraries hold now (Depot.measure, each share's tag of
    shares, [{tag, share}], in order), with its mix (mixed)."""
    return mixed(depot.measure(libraries, [one["tag"] for one in shares]), shares)


def mixed(found: dict, shares: list[dict]) -> dict:
    """What libraries hold (as Depot.measure gives it for the tags of
    shares, [{tag, share}], in order) with the mix of the corpus of every
    token when shares are given (mix None otherwise): each share's row
    gains share, kept_bytes and kept_characters (what scarcest_trim keeps
    of it, its characters in the same proportion; a share of no proportion
    takes nothing, and the others are measured among themselves) and
    limits (true on the share that keeps all it holds while another keeps
    less); mix: {kept_bytes, kept_characters of plain text, items_share:
    the share of the bytes the conversations and tasks hold then}."""
    if not shares:
        return {**found, "mix": None}
    held = {at: one["bytes"] for at, one in enumerate(found["shares"]) if shares[at]["share"] > 0}
    kept = dict.fromkeys(range(len(shares)), 0) | (scarcest_trim(held, shares) if held else {})
    scarcest = [at for at in held if held[at] and kept[at] == held[at]
                and any(kept[other] < held[other] for other in held)]
    rows = [{**one, "share": shares[at]["share"], "kept_bytes": int(kept[at]),
             "kept_characters": round(one["characters"] * kept[at] / one["bytes"]) if one["bytes"] else 0,
             "limits": at in scarcest} for at, one in enumerate(found["shares"])]
    plain = sum(one["kept_bytes"] for one in rows)
    items = found["items"]["bytes"]
    return {**found, "shares": rows, "mix": {
        "kept_bytes": plain, "kept_characters": sum(one["kept_characters"] for one in rows),
        "items_share": items / (items + plain) if items + plain else 0}}


def revision_figures(depot: Depot, dataset: dict) -> dict:
    """What the revision of the dataset whose row is dataset reads, from
    the figures it stores (Depot.dataset_figures): {measure: as measure
    gives it for the dataset's libraries and shares, None when the figures
    of one of its libraries are not known; since: how many texts its
    libraries hold now minus that, None when the measure is not known or a
    library is not held here}."""
    found = depot.dataset_figures(dataset["id"], dataset["revision"])
    stored = [one["stored"] for one in found["libraries"]]
    if any(one is None for one in stored):
        return {"measure": None, "since": None}
    then = counts.summed(stored, found["tags"])
    now = [one["now"] for one in found["libraries"]]
    since = None if any(one is None for one in now) else counts.summed(now, found["tags"])["texts"] - then["texts"]
    return {"measure": mixed(then, dataset["shares"] or []), "since": since}


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
        self.sources, self.libraries = _walked(self.dataset, self.names, self.opened)
        # what the corpus copies and its keys cover: the row, its max_tokens the build's size, without the
        # vocabulary the dataset publishes, which a build never reads (it is given one)
        self.named = {**{key: value for key, value in _named(self.dataset, self.names, self.holder).items()
                         if key != "vocabulary"}, "max_tokens": self.limit}
        self.work, self.state, self.save, self.progress, self.log = work, checkpoint, save, progress, log
        self.shard_tokens = shard_tokens
        self.vocabulary = tokenizer.to_bytes()
        self.vocabulary_sha256 = hashlib.sha256(self.vocabulary).hexdigest()
        self.erasures = _erasures(self.dataset, self.names, self.opened)
        self.key = self._key()
        self.folder = f"{FOLDER}/{as_record_name(self.dataset['name'])}-{self.key[:8]}"
        self.room = room(self.folder) if room is not None else corpora.Room(store, None, self.folder)

    def _key(self) -> str:
        """The build's key: what decides the corpus (the module says)."""
        return _fingerprint({"depot": self.holder, "dataset": self.dataset["id"],
                             "content": _content(self.named), "vocabulary": self.vocabulary_sha256,
                             "rules": RULES, "erasures": self.erasures})

    def run(self) -> dict:
        known = self.store.get_json(f"{self.folder}/{RECORD}")
        if known is not None and corpora.fresh(known):
            self.log(f"corpus {known['id']} of {self.dataset['name']} is already built")
            return known
        self.census = census(self.depot, self.dataset, self.libraries, self.progress)
        if not self.census["known"]:
            self.log(f"the figures of the dataset {self.dataset['name']}'s revision {self.dataset['revision']} are "
                     f"not known: the build read the size of each of its {self.census['texts']:,} texts first")
        self.progress({"stage": "finding long texts"})
        self.long = long_texts(self.libraries)
        cut = sum(pieces.count(one.length) for one in self.long)
        self.walked = self.census["texts"] - len(self.long) + cut  # about what the walk gives, a piece counting once
        if self.long:
            self.log(f"{len(self.long):,} plain texts of {self.dataset['name']} are longer than "
                     f"{pieces.LENGTH:,} characters: each comes as its pieces, {cut:,} in all")
        self.work.mkdir(parents=True, exist_ok=True)
        named = apsw.Connection(str(self.work / "named.sqlite"))
        try:
            named.execute("CREATE TABLE IF NOT EXISTS named (kind TEXT NOT NULL, id TEXT NOT NULL, "
                          "PRIMARY KEY (kind, id)) WITHOUT ROWID")
            return self._stream(named)
        finally:
            named.close()

    def _stream(self, named_file) -> dict:
        state = self.state
        if state.get("key") == self.key and (gone := corpora.missing(self.store, self.folder, _uploaded(state))):
            self.log(f"{len(gone)} files this build uploaded are no longer in the store: it starts over")
            state = {}
        if state.get("key") != self.key:  # a state of another build starts nothing
            state = {"key": self.key, "after": None, "walked": 0, "items": None, "shards": [],
                     "conversation_shards": [], "texts": [], "tasks": 0, "tasks_count": 0, "pools": {}, "fill": None,
                     "taken": 0, "whole": 0, "pieces": 0, "conversations": 0, "text_position": 0,
                     "conversation_position": 0, "items_tokens": 0, "plan": None, "seen": 0, "covered": {},
                     "groups": {}}
            named_file.execute("DELETE FROM named")
        shares = self.dataset["shares"]
        limit = self.limit
        if state["items"] is None:  # with a limit, the items take their share of it first (_limit_items)
            state["items"] = self._limit_items(shares) if limit is not None else {
                "tokens": None, "texts": self.census["items"], "bytes": self.census["item_bytes"], "last": None,
                "skipped": []}
        items = state["items"]
        last_item = None if items["last"] is None else tuple(items["last"])
        skipped_items = {tuple(one) for one in items["skipped"]}

        def item_taken(entry) -> bool:
            """Whether the corpus takes the item: every one without a limit, else those _limit_items took."""
            place = tuple(entry.place())
            return limit is None or (last_item is not None and place <= last_item and place not in skipped_items)

        if limit is not None:  # the items took their share (_limit_items); the shares split what is left
            budget, counted = _proportions(shares, limit - items["tokens"]), "tokens"
        elif shares is not None:
            budget, counted = self._byte_budget(shares), "bytes"
        else:
            budget, counted = None, None
        chosen = self._plan(state, budget, counted, shares)
        fill = None if budget is None else Fill({pool: limit for pool, limit in budget.items()
                                                 if pool not in chosen.last}, state["fill"])
        tags = [""] if shares is None else [one["tag"] for one in shares]
        # each share's groups (wpt_depot.groups.groups_of): {share's tag: {group's tag: [texts, tokens]}}
        group_counts = {share: {tag: [texts, tokens] for tag, texts, tokens in found}
                        for share, found in state["groups"].items()}
        named_groups = {}  # (share's tag, a text's tags): its groups, a few kept
        tasks_path = self.work / "tasks.jsonl"
        with open(tasks_path, "ab") as tasks_file:
            tasks_file.truncate(state["tasks"])
        writers = {"text": _Shards(self.work, "text", state["text_position"]),
                   "conversation": _Shards(self.work, "conversation", state["conversation_position"])}
        credits, named, credit_bytes = [], set(), [0]
        read_bytes = {"plain": 0, "items": 0}  # the bytes of the texts this run took: the size estimate's base
        plain = self.census["plain"]

        def batches():
            """The texts the corpus takes from the state's place on, in batches of (entry, read, walked, place):
            read is whether it is read, which a closed pool, a limit in bytes that leaves it out, or a share's
            choice by group (wpt_depot.groups) saves; walked how many texts the walk passed up to it; place the
            entry's place among the plan's entries, None for an entry of a share the plan does not cover. With
            a limit, an item _limit_items did not take is passed, and the batches end once every pool is closed,
            every share the plan covers is past its last chosen text, and no item taken follows."""
            ahead = fill.copy() if counted == "bytes" else None  # the limit in bytes, worked out ahead of the results
            picker = chosen.picker(state["seen"])
            batch, size, walked = [], 0, state["walked"]
            for entry in walk(self.libraries, _tags(shares), state["after"], self.long):
                walked += 1
                if counted == "tokens" and fill.done() and picker.done() and (
                        last_item is None or tuple(entry.place()) > last_item):
                    break
                if not entry.taken or (entry.kind is not None and not item_taken(entry)):
                    continue
                place = None
                if entry.kind is not None or counted is None:
                    read = True
                elif picker.covers(entry.pool):
                    place, read = picker.pick(entry.pool)
                elif counted == "bytes":
                    read = ahead.take(entry.pool, entry.bytes)
                else:
                    read = entry.pool not in fill.closed
                batch.append((entry, read, walked, place))
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
            hold (the order is shuffled, so its start stands for the whole)."""
            if final:
                return local
            plain_all = sum(self.census["pool_bytes"].values())
            if counted == "bytes":
                plain_bytes = min(plain_all, sum(self.census["pool_bytes"].get(pool, 0) if value is None else value
                                                 for pool, value in budget.items()))
            elif counted == "tokens":
                tokens = sum(state["pools"].values())
                plain_bytes = plain_all if not tokens else min(plain_all, int(sum(budget.values())
                                                                              * read_bytes["plain"] / tokens))
            else:
                plain_bytes = plain_all
            return int(local * (plain_bytes + items["bytes"]) / max(read_bytes["plain"] + read_bytes["items"], 1))

        def checkpoint(after: list, walked: int, final: bool = False):
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
            with named_file:
                named_file.executemany("INSERT OR IGNORE INTO named (kind, id) VALUES (?, ?)", sorted(named))
            named.clear()
            state["groups"] = {share: [[tag, *counted_so] for tag, counted_so in found.items()]
                               for share, found in group_counts.items()}
            state.update(after=after, walked=walked, fill=None if fill is None else fill.state(),
                         text_position=writers["text"].position,
                         conversation_position=writers["conversation"].position, tasks=tasks_path.stat().st_size)
            self.save(json.loads(json.dumps(state)))

        with ExitStack() as stack:
            tasks_file = stack.enter_context(open(tasks_path, "ab"))
            if sum(self.census["pool_bytes"].values()) > IN_PROCESS:
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
                for (entry, read, walked, place), (row, data) in zip(batch, found):
                    if place is not None:  # a share that chose its texts by group: the chosen ones, within its budget
                        state["seen"] = place + 1
                        if read:
                            used = state["covered"].get(str(entry.pool), 0) + (
                                len(data) if counted == "tokens" else entry.bytes)
                            if used > budget[entry.pool]:
                                raise AssertionError(f"the share {tags[entry.pool]} chose {entry.id}, which passes "
                                                     "its budget")
                            state["covered"][str(entry.pool)] = used
                    elif counted is not None and entry.kind is None and (read or counted == "bytes"):
                        took = fill.take(entry.pool, len(data) if counted == "tokens" else entry.bytes)
                        if counted == "bytes" and took != read:
                            raise AssertionError(f"the limit in bytes left {entry.id} out on one pass and not the "
                                                 "other")
                        read = took
                    if not read:
                        continue
                    source_name = self.sources[entry.source]
                    pool_tag = tags[entry.pool] if entry.kind is None else ""
                    if entry.kind == "task":
                        start = state["tasks_count"]
                        content = json.loads(data.decode("utf-8"))
                        tasks_file.write(task_text({"depot": source_name, "id": entry.id,
                                                    "content": content}).encode("utf-8") + b"\n")
                        state["tasks_count"] += 1
                        if limit is None:  # with a limit, the items' tokens were counted already (_limit_items)
                            state["items_tokens"] += task_tokens(self.tokenizer, content)
                    else:
                        kind = "text" if entry.kind is None else "conversation"
                        document = entry.id if entry.piece is None else pieces.name(entry.id, entry.piece)
                        start = writers[kind].add(data, source_name, document, pool_tag)
                        if entry.kind is None:
                            state["pools"][pool_tag] = state["pools"].get(pool_tag, 0) + len(data)
                            if shares is not None:
                                held = named_groups.get((pool_tag, entry.tags))
                                if held is None:
                                    if len(named_groups) >= groups.INTERNED:
                                        named_groups.clear()
                                    held = named_groups[pool_tag, entry.tags] = groups.groups_of(entry.tags,
                                                                                                 pool_tag)
                                into = group_counts.setdefault(pool_tag, {})
                                for tag in held:
                                    counted_so = into.setdefault(tag, [0, 0])
                                    counted_so[0] += 1
                                    counted_so[1] += len(data)
                            if entry.piece is None:
                                state["whole"] += 1
                            else:
                                state["pieces"] += 1
                                named.add(("long text", json.dumps([source_name, entry.id])))
                        else:
                            state["conversations"] += 1
                            state["items_tokens"] += len(data)
                    state["taken"] += 1
                    credits.append({"depot": source_name, "id": entry.id, "revision": entry.revision, **row,
                                    "tokens": start, "piece": entry.piece})
                    credit_bytes[0] += entry.bytes
                    read_bytes["plain" if entry.kind is None else "items"] += entry.bytes
                    named.update(("license", one) for one in row["licenses"])
                    if row["notice"] is not None:
                        named.add(("notice", row["notice"]))
                    if any(writer.full(self.shard_tokens) for writer in writers.values()):
                        checkpoint(entry.place(), walked)
                state["after"], state["walked"] = batch[-1][0].place(), batch[-1][2]
                self.progress({"stage": "reading", "position": state["walked"], "of": self.walked,
                               "texts": state["taken"], "tokens": sum(state["pools"].values()),
                               "token files": len(state["shards"]) + len(state["conversation_shards"])})
            checkpoint(state["after"], state["walked"], final=True)
        self.progress({"stage": "finishing", "texts": state["taken"]})
        return self._finish(state, tasks_path, plain, named_file)

    def _items(self, batch) -> list[tuple]:
        return [(self.opened[self.sources[entry.source]].address, entry.id, entry.revision, entry.kind, read,
                 entry.piece) for entry, read, _, _ in batch]

    def _plan(self, state: dict, budget: dict | None, counted: str | None, shares: list[dict] | None
              ) -> groups.Plan:
        """The texts the shares choose by group (wpt_depot.groups), sizes
        counted as the budget counts them (tokens or bytes): read again
        from the work folder when the state names them and the file still
        holds them, else chosen again, the same for the same depot state,
        and kept there, its sha256 in the state."""
        if shares is None or budget is None:
            return _no_plan()
        path = self.work / "chosen.bin"
        known = state.get("plan")
        if known is not None:
            found = groups.Plan.load(path, {int(pool): last for pool, last in known["last"].items()},
                                     known["sha256"])
            if found is not None:
                return found
        self.progress({"stage": "choosing"})
        addresses = [self.opened[name].address for name in self.sources]
        with ExitStack() as stack:
            count = None
            if counted == "tokens" and sum(self.census["pool_bytes"].values()) > IN_PROCESS:
                sizing = Sizing("tokenizing workers")
                pool = stack.enter_context(process_pool(sizing, sizing.limit, _start_worker, (self.vocabulary,)))

                def count(items: list[tuple]) -> list[int]:
                    step = max(1, min(BATCH_TEXTS, -(-len(items) // (2 * sizing.limit))))
                    parts = (items[at:at + step] for at in range(0, len(items), step))
                    return [one for found in in_order(_plain_counted, parts, pool, sizing) for one in found]
            elif counted == "tokens":
                _start_worker(self.vocabulary)
                count = _plain_counted
            chosen = groups.plan(self.libraries, _tags(shares), budget, self.long, addresses, count, self.progress)
        sha256 = chosen.save(path)
        if known is not None and known["sha256"] != sha256:
            raise AssertionError("the shares chose other texts than this build chose before it stopped")
        state["plan"] = {"sha256": sha256, "last": {str(pool): last for pool, last in chosen.last.items()}}
        if chosen.last:
            self.log(f"the shares {', '.join(shares[pool]['tag'] for pool in sorted(chosen.last))} split their "
                     f"budgets over the tags of their texts, by the square root of what each holds: {len(chosen.chosen):,} texts chosen")
        return chosen

    def _limit_items(self, shares: list[dict] | None) -> dict:
        """The items (conversations and tasks) a build at max_tokens takes:
        their share of it (the items' part of the bytes of the corpus
        without max_tokens, from the census), filled by their tokens counted
        in the order, as one pool (Fill). {tokens, texts, bytes: of the
        items taken, last: the place of the last item taken (None: none),
        skipped: the places of the items before it left out}; the walk ends
        once the pool closes, or at the end of the dataset."""
        item_bytes = self.census["item_bytes"]
        found = {"tokens": 0, "texts": 0, "bytes": 0, "last": None, "skipped": []}
        if not self.census["items"]:
            return found
        plain = sum((self.census["pool_bytes"] if shares is None else self._byte_budget(shares, False)).values())
        part = self.limit * item_bytes // (item_bytes + plain) if item_bytes else 0
        fill = Fill({0: part})

        def batches():
            batch, size = [], 0
            for entry in walk(self.libraries, _tags(shares), long=self.long):
                if not entry.taken or entry.kind is None:
                    continue
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
            if item_bytes > IN_PROCESS:
                sizing = Sizing("counting workers")
                pool = stack.enter_context(process_pool(sizing, sizing.limit, _start_worker, (self.vocabulary,)))
                waiting = collections.deque()

                def feeding():
                    for batch in batches():
                        waiting.append(batch)
                        yield items(batch)
                results = ((waiting.popleft(), counted) for counted in in_order(_counted, feeding(), pool, sizing))
            else:
                _start_worker(self.vocabulary)
                results = ((batch, _counted(items(batch))) for batch in batches())
            done = 0
            skipped = []
            for batch, counted in results:
                for entry, tokens in zip(batch, counted):
                    if fill.done():
                        break
                    if not fill.take(0, tokens):
                        skipped.append(entry.place())
                        continue
                    found["tokens"] += tokens
                    found["texts"] += 1
                    found["bytes"] += entry.bytes
                    found["last"] = entry.place()
                    found["skipped"] += skipped
                    skipped = []
                done += len(batch)
                self.progress({"stage": "counting items", "items": done, "tokens": found["tokens"]})
                if fill.done():
                    break
        return found

    def _byte_budget(self, shares: list[dict], refuse: bool = True) -> dict[int, int | None]:
        """{pool: the bytes it may take} without max_tokens: the scarcest
        share all of its bytes, each other its proportion of the scarcest's
        (from the census). A share with no text is refused, or with refuse
        False takes nothing and the others are measured among themselves.
        With refuse, a share that may take all its bytes has no limit
        (None), so it takes every text."""
        available = {at: 0 for at in range(len(shares))} | self.census["pool_bytes"]
        empty = [shares[at]["tag"] for at, size in available.items() if not size]
        if empty and refuse:
            raise Refused(f"the shares cannot be met: no text of the libraries has the tag {', '.join(empty)}")
        trimmed = {at: int(size) for at, size in scarcest_trim(available, shares).items()}
        return trimmed if not refuse else {at: None if size >= available[at] else size
                                           for at, size in trimmed.items()}

    def _finish(self, state: dict, tasks_path: Path, plain: int, named_file) -> dict:
        built_at = datetime.now(UTC)
        dataset = {k: v for k, v in self.named.items() if k != "change"}
        named = {kind: [one for (one,) in named_file.execute("SELECT id FROM named WHERE kind = ? ORDER BY id",
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
        long_taken = next(named_file.execute("SELECT count(*) FROM named WHERE kind = 'long text'"))[0]
        plain_taken = state["whole"] + long_taken  # a text taken as pieces counts once
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
        items_taken = state["conversations"] + state["tasks_count"]
        left_out = {"no share": self.census["no_share"], "other kinds": self.census["other"],
                    "trimmed": max(0, plain - plain_taken), "trimmed items": max(0, self.census["items"] - items_taken)}
        record = {
            "id": corpus_id, "depot": self.holder, "dataset": dataset, "folder": self.folder,
            "build": self.key, "erasures": self.erasures, "built_at": built_at.isoformat(),
            "texts": state["taken"] - state["pieces"] + long_taken, "pieces": state["pieces"],
            "conversations": state["conversations"],
            "tokenizer": {"recipe": self.recipe, "recipe_version": self.version, "sha256": self.vocabulary_sha256,
                          "vocab_size": self.tokenizer.vocab_size, "vocabulary": vocabulary},
            "max_tokens": self.limit,
            "shards": state["shards"], "conversation_shards": state["conversation_shards"],
            "tables": {"dataset": put("dataset", [{**dataset, "built_at": built_at}]), "texts": state["texts"],
                       "licenses": put("licenses", license_rows), "notices": put("notices", notice_rows)},
            "total_tokens": sum(s["tokens"] for s in state["shards"]),
            "items_tokens": state["items_tokens"] if self.limit is None else state["items"]["tokens"],
            "shares_tokens": None if self.dataset["shares"] is None else [
                {"tag": share["tag"], "tokens": state["pools"].get(share["tag"], 0),
                 "groups": [{"tag": tag, "texts": texts, "tokens": tokens} for tag, texts, tokens in sorted(
                     state["groups"].get(share["tag"], []), key=lambda one: (one[0] is not None, one[0] or ""))]}
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
        self.log(f"corpus {corpus_id} of {self.dataset['name']}: {record['texts']:,} texts"
                 + (f" ({state['pieces']:,} pieces of {long_taken:,} of them)" if state["pieces"] else "") + ", "
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

