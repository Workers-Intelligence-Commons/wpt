"""Building a corpus onto the dock from a snapshot and an order's data
section: what training reads, which says everything about what it holds
even once the depot it read changes or vanishes. The freight
(packages/freight) chooses as this module does, and builds what the
deliveries hold; this module builds the corpora the tests compare it
with.

It builds a corpus without portions, the shape of the corpora under the
dock's corpora/: training reads all of it as its pretraining portion, with
no held-out one (wpt_corpus.roles). So it is the reference of a delivery
made without portions, and the corpora under corpora/ the tests train on;
the portions of a delivery are checked against the freight's own rules
(freight/test_portions.py).

A build reads one snapshot of a depot (depot.snapshots), each text at
its revision as of its bound, and a data section of the shape of the
corpus it builds (not that of an order, which names roles): the kinds
it takes (kinds; none named: every kind), its shares by tag (shares, [{tag,
share}] in order; none: every plain text, in the snapshot's own
proportions), its size (total_tokens, a number of tokens, or None for every
text the rules below take) and its vocabulary ({sha256, size}). The section
may also name the snapshot's change and digest, as an order does; a build
refuses a snapshot that reads another change or holds other texts than the
section says, and a vocabulary of another sha256 or size (check).

A build reads the depot only through depot.api, by the depot's address: the snapshot's row,
its listing, each text's content and credits, where a quote matches, the
licenses and notices, and the erasures carried out (docs/api/depot.md).

The build is a stream. It walks the snapshot's listing, its texts in the
order of their export keys (depot.api.listing, depot.order: every
text of the snapshot, a long plain text as its pieces, choosing nothing by
score), each under the data section's shares and kinds (walk, Entry),
reading only the texts it reaches, tokenizes them in worker processes
(wic_utils.parallel), and takes their results in that order. With shares, a first pass walks the whole
order before it (depot.groups) to choose the texts of each share that
splits its budget over the tags of its texts (below), reading the rows of
the texts but never their contents, but for the texts whose tokens the
choice needs. A token file (wpt_corpus.shards) is uploaded to
the dock as soon as it is full, with the part of the texts table that
lists its texts, and its local copies deleted, so the memory and the local
disk the build takes stay bounded whatever the snapshot's size: the order
(8 bytes a text), one token
file of each kind and one part of the texts table at a time, the
tasks.jsonl it writes, the ids of the licenses and notices its texts
name, the conversations it passed and the plain texts it took
(named.sqlite). After each upload it hands its caller the state that
resumes it (checkpoint), keyed by the build's key: a build started again
with that state and the same work folder goes on from the first text after
the last token file uploaded, and writes the same bytes as a build that
never stopped; a state of another key, or one from before an erasure that
reached the snapshot, starts over.

What a build needs of the whole snapshot before it walks it (the bytes of
the plain texts of each share and of the items, which set the items' part
of a size and the shares' bytes without one, and the counts of what it
leaves out) is its census (census), always read from the snapshot's
summary, never counted from its texts: the summary's kinds and its plain
tag sets give each share's texts and bytes exactly, a plain text counting
for the first share whose tag it has. So the census of a snapshot never
changes, an erasure after it included, and the same snapshot and data
section give the same plan.

What it takes: a plain text counts for the first share whose tag it has,
and a plain text that has none of their tags is left out. Inside a share,
the budget splits over every tag its texts carry but the share's own,
each tag's part following the square root of what it holds (depot.groups): a share whose texts carry two such tags or more,
with a limit, takes the texts the groups choose, and every other share fills
as below. Items of a kind
other than conversation and task are no training data and are left out,
and so is every text of a kind the data section does not name. A
plain text longer than depot.pieces.LENGTH characters is taken as its
pieces (depot.pieces), each at its own place of the order and counted
as a text of its own by every rule below, so a long text, such as a
repository the depot holds as one text, spreads over the corpus, and a
corpus of any size takes about its share of it. An item is never cut.

A conversation depends on every text its excerpts quote
(depot.quotes), which nobody declares: the corpus takes it only when
it takes each quoted text, a plain text of the snapshot,
and, of a text taken as pieces, every piece the passage lies in, and only
when each excerpt's selector matches exactly one place of the content
version it took. Otherwise the conversation is left out, as `quoted text
left out` or `quote unmatched` in left_out. So the walk passes the
conversations by, keeping their places in the work folder (held) with the
plain texts it takes (taken), and once it ends takes them in its order
(_resolve), HELD_PAGE at a time: the workers read and tokenize them, then
read each quoted content version once for all the excerpts of the page
that quote it, as its parts, never whole. Every
rule reads the snapshot alone, so the same snapshot builds the same corpus.

Every limit below is filled the same way (depot.fill.Fill), but for a
share that splits its budget by group: the texts come in order, a text that
fits what is left of its pool is taken, and one that would pass it is left
out while the walk goes on with the next. A pool closes once what is left
of it is at most a FULL-th of its budget, or once SKIPS of its texts in a
row did not fit, and the walk ends once every pool is closed, every share
that splits by group is past the last text it chose, and no item taken
comes later, so a nearly full corpus never reads the rest of the snapshot
for its last tokens. A share that splits by group chooses its texts on
their exact tokens with total_tokens, which worker processes count ahead of
the choice, and on their bytes without; the stream then takes the texts it
chose, checking each against the share's budget again. The texts chosen
wait in the work folder (chosen.bin), their sha256 in the state that
resumes the build, so a build resumed takes the same texts; a file lost is
chosen again, the same.

With total_tokens, the corpus's tokens, of plain texts, conversations and
tasks together, never pass it, and a corpus of any size keeps the
snapshot's mix of items and plain text:

- the items (conversations and tasks) have their natural share of
  total_tokens: the share of the UTF-8 bytes of the corpus without
  total_tokens (below: every item, and the plain texts the shares keep) that
  the items hold, rounded down to a whole token, so every size has the mix
  of the corpus of every token. Bytes, because they are known for every
  text before anything is tokenized. A share no text has counts for
  nothing there. Before the build writes anything, the items are counted in
  order and fill that part, as one pool (_limit_items). A conversation
  counts the tokens of its chat template; a task, kept as a line of
  tasks.jsonl and never tokenized, counts the tokens of its content as the
  JSON text of that line (task_text);
- each share (or, without shares, every plain text) fills its proportion
  of what the items left of total_tokens, in tokens.

So a smaller size takes the same texts as a larger one up to the first
text it leaves out, the start of what the larger one takes; past it, it
takes only what fits what is left of it. A share that splits by group takes
about the same part of each group at every size instead.

Without total_tokens, every conversation and task is taken, and the plain
texts go in order, a share at a time:

- with shares, the scarcest share (the one with the
  fewest UTF-8 bytes relative to its proportion) takes every text, and
  each other share fills its proportion of the scarcest share's bytes, a
  piece counting for its share of its text's bytes (depot.order);
- with neither, every plain text.

A text is read from its parts (depot.parts), so a text bigger than
memory is read as a stream: a plain text is encoded a block at a time, each
block ending right before a line's first byte that is not whitespace, a
place where the tokenizer's chunks always break, so its tokens are those
of the whole text; only its tokens are held whole. A piece is read from its
own characters alone (depot.pieces.read) and encoded as a text of its
own. A plain text or a piece becomes tokens followed by END; a
conversation becomes the tokens of the chat template (wpt_tokenizer.chat),
which start with END and mark nothing: what the assistant wrote is told by
the template's special tokens, and each excerpt is written as the passage
it quotes. Both go into token files in WPT's shard format (wpt_corpus.shards), kept by
kind (TOKEN_KINDS: plain and conversation), each token file holding texts
of one kind, each row one text or one piece, its `source` the name of the
depot that made the snapshot (its owner), its `document`
the text's id or the piece's name (depot.pieces.name), its
`pool` the share's tag (empty without shares) and its `pass` 0. A
token file closes once it holds SHARD_TOKENS tokens, and closes the other
kind's with it. A task, of kind task, is kept as one JSON line of
tasks.jsonl, never tokenized.

The corpus's tables are Parquet files (CORPUS_SCHEMAS): every
text taken with its revision, its credits, its tags as of the snapshot
(those the walk read at its bound, depot.order) and its scores (each
with the content version it was made from) copied, `piece` (null for a
whole text) and `tokens`, the
position of its first token over the token files of its own kind in order
(a task's: its line in tasks.jsonl), in parts, one uploaded with each
token file, in the corpus's order; a piece taken has a row of its own,
`piece` its number, with its text's credits; a conversation's row names the
texts it quotes (`quotes`), each in the corpus with its own row, credits,
licenses and notice; the licenses its texts name; the
notices (ids carry their depot's name, so they are copied as they are).
Every text taken is in the corpus once, whole or as some of its pieces,
each piece once (shares_tokens in the record says how many tokens each
share holds, and for each group of the share, its tag, null for the texts
that carry no other tag, how many texts it holds and their tokens, a text
counting for each of its groups).

The texts table, the token files and tasks.jsonl name the depot by the
snapshot's owner, the name in every one of its ids, never by a file's path
or an address: a snapshot travels to a clone with its owner's name, so a
clone and its upstream build the same bytes.

Everything goes under corpora/<build key>/ on the dock, the record
corpus.json last, so a build that fails leaves no record. The build's key
(build_key) covers what decides the corpus: the snapshot's id and digest,
the data section's kinds, shares and total_tokens in a canonical form
(canonical), the vocabulary's sha256 and the version of these rules
(RULES); never the address a request came from nor the code's commit,
which the record keeps. So the same snapshot, data section and vocabulary
build into the same folder wherever the depot's file is, on the depot that
made the snapshot as on a clone. A build whose folder holds a stale or
deleted record builds again over it, and deletes the files of the old
record the new one does not name. The corpus id covers the build's key and
every file's content (each token file by its token hash, each part of the
texts table by a hash of its rows), so building the same corpus again
gives the same id, and an erasure that cut a text it took gives another.

The record: {id, folder, depot (the address the caller names, the
order's), snapshot {id, name, change, digest}, order (its name, or
None), data (the data section as given), build_key, code_commit, rules,
erasures (the erasures the depot had carried out that reached a text of
the snapshot), built_at, texts (a text taken as pieces counting once),
pieces, conversations, tokenizer (its recipe and version when the caller
names them, sha256, vocab_size and the vocabulary file), vocabulary
({sha256, size}), max_tokens (total_tokens), its token files by kind
(shards: {kind: [token file]}, each kind of TOKEN_KINDS), tables (texts,
licenses, notices), total_tokens (of plain text), items_tokens,
shares_tokens, what it left out (left_out: plain texts no share takes,
plain texts of which it took nothing, items total_tokens left out,
conversations whose quoted text it left out, conversations whose quote is
unmatched, and the texts of each kind it does not take, "kinds not taken",
{kind: texts}), tasks, and a fingerprint of what it holds of each kind a
training step may read, `fingerprints`: "plain" (the vocabulary and the
plain texts' token files), "conversation" (the vocabulary and the
conversations' token files) and "task" (the vocabulary and tasks.jsonl),
so a step's result is found again only for the same kinds it reads
(docs/training.md, "Steps")}. A corpus whose record exists and is fresh is
not built again (freight.corpora says what the dock keeps of corpora, and
corpus_room the cap a build works under).

An erasure carried out while the build ran is checked before the record is
written and again after: when it reached a text the corpus took, the
record is written stale. A build resumed, or about to write its record,
whose uploaded files are no longer on the dock (another build's cap
deleted its folder, which had no record yet) starts over, or fails
(FilesGone) without a record. Before its first upload, the build hands the
cap its estimated size (Room.expect).

    build(depot, snapshot, data, dock, tokenizer, ...)    the corpus of a snapshot and data section, its record
    check(depot, snapshot, data, tokenizer)               the snapshot's row, refused when the section does not match
    canonical(data)                                       the kinds, shares and size of a data section, as keyed
    build_key(snapshot, data, vocabulary)                 the key of a build
    census(summary, shares, kinds)                        what a corpus of the snapshot may take, whatever its size
    walk(source, snapshot, shares, kinds, after, spread)  the snapshot's listing under a data section, as Entry
    entries(listed, shares, kinds)                        a listing given, the same way
    pool_of(kind, tags, shares, kinds)                    a text's pool and whether a corpus takes it
"""

import collections
import hashlib
import itertools
import json
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import apsw
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from depot import api, counts, groups, quotes
from depot.depot import Depot
from depot.errors import Refused
from depot.fill import Fill
from depot.kinds import PLAIN
from dock import api as dock_api
from freight import corpora
import corpus_room
from wpt_corpus.corpora import RULES as CORPUS_RULES
from wpt_corpus.corpora import fresh
from wic_utils.lines import say
from wic_utils.parallel import Sizing, in_order, process_pool
from wpt_corpus.shards import write_shard
from wpt_tokenizer import END, Tokenizer, chat

SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per token file
WHITESPACE = b" \t\n\r\x0b\x0c"  # the bytes the tokenizer's chunks treat as whitespace (wpt_tokenizer)
WHITESPACE_BYTES = np.frombuffer(WHITESPACE, np.uint8)
FOLDER, RECORD = corpora.FOLDER, corpora.RECORD
HELD_PAGE = 1000  # conversations the build reads from its work folder at a time once the walk ends
LEFT_OUT, UNMATCHED = "quoted text left out", "quote unmatched"  # why a conversation is left out (_resolve)
BATCH_BYTES = 8 << 20  # bytes of texts one worker's item reads and tokenizes
BATCH_TEXTS = 4096  # texts one worker's item holds at most
IN_PROCESS = 16 << 20  # bytes of texts a build reads in its own process before it starts worker processes (_Workers)
SHA256_HEX = 64  # hexadecimal characters of a sha256
TIME = pa.timestamp("us", tz="UTC")
NAMES = pa.list_(pa.string())
SCORES = pa.list_(pa.struct([("name", pa.string()), ("value", pa.float64()), ("content_version", pa.int64())]))
CORPUS_SCHEMAS = {
    "texts": pa.schema([pa.field("depot", pa.string(), False), pa.field("id", pa.string(), False),
                        pa.field("revision", pa.int64(), False), pa.field("kind", pa.string(), False),
                        pa.field("title", pa.string()), pa.field("authors", NAMES),
                        pa.field("address", pa.string()), pa.field("licenses", NAMES, False),
                        pa.field("tags", NAMES, False), pa.field("notice", pa.string()),
                        pa.field("tokens", pa.int64(), False), pa.field("scores", SCORES),
                        pa.field("piece", pa.int64()), pa.field("quotes", NAMES)]),
    "licenses": pa.schema([pa.field("id", pa.string(), False), pa.field("name", pa.string(), False),
                           pa.field("text", pa.string(), False)]),
    "notices": pa.schema([pa.field("id", pa.string(), False), pa.field("text", pa.string(), False)]),
}
RULES = CORPUS_RULES  # the version of the rules of what a build takes, in the build's key
# the kinds a corpus holds as token files, each kind's apart (a task is a line of tasks.jsonl, never tokenized)
TOKEN_KINDS = (PLAIN, "conversation")
TRAINING_KINDS = ("conversation", "task")  # the kinds beside plain a corpus takes
REBUILT = "built again over its stale record"  # why the files of a record a build replaces are deleted


def build(depot: Depot, snapshot: str, data: dict, dock, tokenizer: Tokenizer, *, address: str | None = None,
          order: str | None = None, tokenizer_recipe: str | None = None, tokenizer_version: int | None = None,
          code_commit: str | None = None, work: Path | None = None, checkpoint: dict | None = None,
          save=lambda state: None, progress=lambda state: None, room=None, shard_tokens: int = SHARD_TOKENS,
          log=say) -> dict:
    """Build the corpus of depot's snapshot whose id is snapshot with the
    data section data (the module says what it holds) into dock with
    tokenizer, whose sha256 and size the section names: the corpus's
    record. address: the depot's address the record names (the order's);
    order: the order's name, which the record names `recipe`, as the
    depot's corpus.json does; tokenizer_recipe and tokenizer_version: the
    depot's tokenizer recipe and version the vocabulary is of, when known;
    code_commit: the code's commit, recorded only.

    work: the folder the build keeps its local files in
    (default: a temporary folder, removed at the end); checkpoint: the state
    a former build in the same folder handed save, to go on from it; save:
    called with that state after each upload; progress: called with
    {stage, ...} as it goes, and may raise to stop the build; room: makes a
    corpus_room.Room for the corpus's folder (default: no cap).

    A snapshot or vocabulary the section does not match (check), shares
    that cannot be met (Refused), the cap's (TooBig, NoRoom) and files
    deleted meanwhile (FilesGone) leave no record; the files uploaded
    before stay, and the same build reuses those still there."""
    with ExitStack() as stack:
        if work is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="wpt-corpus-")))
        return _Build(depot, snapshot, data, dock, tokenizer, address, order, tokenizer_recipe, tokenizer_version,
                      code_commit, Path(work), checkpoint or {}, save, progress, room, shard_tokens, log).run()


def canonical(data: dict) -> dict:
    """What a data section says of its corpus, in the form the build's key
    covers: {kinds: the kinds named, sorted, each once ([]: every kind),
    shares: [{tag, share}] in order ([]: none), total_tokens: a number of
    tokens above 0, or None}. Refused for a section that cannot be built:
    kinds not a list of names, shares whose tags repeat or whose share is
    below 0 or not a number, or whose shares are all 0, a size that is not
    a whole number above 0."""
    kinds = data.get("kinds") or []
    if not isinstance(kinds, list) or not all(isinstance(one, str) and one for one in kinds):
        raise Refused(f"the data section's kinds are a list of kinds' names: {kinds!r}")
    shares = data.get("shares") or []
    if not isinstance(shares, list) or not all(isinstance(one, dict) and isinstance(one.get("tag"), str)
                                               and isinstance(one.get("share"), (int, float))
                                               and not isinstance(one.get("share"), bool) and one["share"] >= 0
                                               for one in shares):
        raise Refused(f"the data section's shares are a list of {{tag, share}}, each share a number of 0 or more: "
                      f"{shares!r}")
    tags = [one["tag"] for one in shares]
    if len(set(tags)) != len(tags):
        raise Refused(f"the data section's shares name a tag twice: {', '.join(tags)}")
    if shares and not sum(one["share"] for one in shares):
        raise Refused("the data section's shares are all 0: give one a share above 0, or give no shares")
    size = data.get("total_tokens")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise Refused(f"a corpus's size (total_tokens) is a whole number of tokens above 0, or none: {size!r}")
    return {"kinds": sorted(set(kinds)), "shares": [{"tag": one["tag"], "share": float(one["share"])}
                                                    for one in shares], "total_tokens": size}


def build_key(snapshot: dict, data: dict, vocabulary: str) -> str:
    """The key of the build of the snapshot (its row) with the data section
    data and the vocabulary of that sha256 (the module says what it covers)."""
    return _fingerprint({"snapshot": snapshot["id"], "digest": snapshot["digest"], "data": canonical(data),
                         "vocabulary": vocabulary, "rules": RULES})


def check(depot: Depot, snapshot: str, data: dict, tokenizer: Tokenizer) -> dict:
    """The row of depot's snapshot whose id is snapshot, once the data
    section matches it and tokenizer: the change and digest it names, when
    it names them, must be the snapshot's, and its vocabulary's sha256 and
    size those of tokenizer. Refused otherwise, saying both; NotFound
    without such a snapshot."""
    row = api.snapshot(depot.address, snapshot)
    change, digest = data.get("change"), data.get("digest")
    if (change is not None and change != row["at_change"]) or (digest is not None and digest != row["digest"]):
        raise Refused(f"snapshot {row['name']} of depot {row['owner']} reads change {row['at_change']} with digest "
                      f"{row['digest']}; the order says {change}, {digest}")
    vocabulary = data.get("vocabulary")
    if not isinstance(vocabulary, dict) or not isinstance(vocabulary.get("sha256"), str):
        raise Refused(f"the data section names its vocabulary as {{sha256, size}}: {vocabulary!r}")
    sha256 = hashlib.sha256(tokenizer.to_bytes()).hexdigest()
    if vocabulary["sha256"] != sha256 or vocabulary.get("size", tokenizer.vocab_size) != tokenizer.vocab_size:
        raise Refused(f"the vocabulary given has sha256 {sha256} and size {tokenizer.vocab_size}; the order says "
                      f"{vocabulary['sha256']}, {vocabulary.get('size')}")
    return row


def _erasures(source: str, snapshot: str) -> list[str]:
    """The ids of the erasures the depot at the address source has carried
    out that reached a text the snapshot whose id is snapshot holds: an
    erasure of a text it does not hold changes nothing of its corpus. A
    text counts when any part of the snapshot holds it, not only the last
    part of its owner."""
    found = api.erasures(source)["erasures"]
    held = set(api.holds(source, snapshot, sorted({text for one in found for text in one["texts"]})))
    return [one["id"] for one in found if any(text in held for text in one["texts"])]


class Entry(NamedTuple):
    """A text or a piece of a snapshot's listing (depot.api.listing)
    under a data section: its key (as an int, which orders places), owner,
    number, id, piece (None for a whole text), revision, kind, bytes, tags
    (as JSON text) and name (the text's id or the piece's name), and pool
    and taken (pool_of)."""
    key: int
    owner: str
    number: int
    id: str
    piece: int | None
    revision: int
    kind: str
    pool: int | None
    bytes: int
    taken: bool
    tags: str
    name: str

    def place(self) -> list:
        """The entry's place in the listing, [key, owner, number, piece (-1 for a whole text)], what a walk's
        after names."""
        return [self.key, self.owner, self.number, -1 if self.piece is None else self.piece]


def pool_of(kind: str, tags: str, shares: Sequence[str] | None, kinds: Sequence[str] | None = None
            ) -> tuple[int | None, bool]:
    """(pool, taken) of a text of that kind with tags (JSON text), under
    the tags of a data section's shares (None without shares) and the kinds
    it names (None or empty: every kind): pool, with shares, the index of
    the first share whose tag a plain text has, None when it has none;
    without shares, 0 for every plain text; None for an item; taken,
    whether a corpus takes it at all, that is, a text of a kind the section
    names that is a plain text with a pool or a text of one of
    TRAINING_KINDS (a text of any other kind is no training data)."""
    if kinds and kind not in kinds:
        return None, False
    if kind != PLAIN:
        return None, kind in TRAINING_KINDS
    if shares is None:
        return 0, True
    held = json.loads(tags or "[]")
    pool = next((at for at, tag in enumerate(shares) if tag in held), None)
    return pool, pool is not None


_TAGS_TEXT = {}  # a list of tags as a tuple: its JSON text, a few kept


def _tags_text(tags: list[str]) -> str:
    """A list of tags as JSON text, the form an entry keeps them in."""
    key = tuple(tags)
    found = _TAGS_TEXT.get(key)
    if found is None:
        if len(_TAGS_TEXT) >= groups.INTERNED:
            _TAGS_TEXT.clear()
        found = _TAGS_TEXT[key] = json.dumps(tags, ensure_ascii=False, separators=(",", ":"))
    return found


def entries(listed: Iterable[dict], shares: Sequence[str] | None, kinds: Sequence[str] | None = None
            ) -> Iterator[Entry]:
    """The entries of a listing (depot.api.listing), each as Entry
    under the tags of a data section's shares (None without shares) and the
    kinds it names (None or empty: every kind)."""
    for one in listed:
        tags = _tags_text(one["tags"])
        pool, taken = pool_of(one["kind"], tags, shares, kinds)
        yield Entry(int(one["key"]), one["owner"], one["number"], one["id"], one["piece"], one["revision"],
                    one["kind"], pool, one["bytes"], taken, tags, one["name"])


def walk(source: str, snapshot: str, shares: Sequence[str] | None, kinds: Sequence[str] | None = None,
         after: Sequence | None = None, spread: int | None = None) -> Iterator[Entry]:
    """The listing of the snapshot whose id is snapshot of the depot at the
    address source, from the place after on, as entries (Entry); spread:
    only that many places of it, spread evenly (depot.api.listing)."""
    if after is not None:
        after = [str(after[0]), *after[1:]]
    return entries(api.listing(source, snapshot, after, spread), shares, kinds)


def _no_plan() -> groups.Plan:
    """The plan of an order whose shares choose nothing by group (depot.groups)."""
    return groups.Plan({}, np.zeros(0, np.int64))


def _proportions(shares: list[dict] | None, total: int) -> dict[int, int]:
    """{pool: its part of total}: each share's proportion of it, or all of it for the one pool without shares."""
    if not shares:
        return {0: total}
    whole = sum(share["share"] for share in shares)
    return {at: int(total * share["share"] / whole) for at, share in enumerate(shares)}


def _tags(shares: list[dict] | None) -> list[str] | None:
    """The tags of an order's shares, in order; None without shares."""
    return [one["tag"] for one in shares] if shares else None


def census(summary: dict, shares: list[dict] | None, kinds: list[str] | None = None) -> dict:
    """What a corpus of a snapshot whose summary is summary
    (depot.snapshots) may take whatever its size, under the shares
    ([{tag, share}] in order, none: every plain text) and kinds (none:
    every kind) of a data section: {pool_bytes: {pool: the bytes of the
    plain texts of that pool (depot.order)}, pool_texts: {pool: how
    many such texts}, plain: how many in all, item_bytes and items: the
    bytes and number of the conversations and tasks (TRAINING_KINDS) the
    kinds name, task_bytes: the tasks' bytes, no_share: the plain texts no
    share takes, other: the texts of the kinds a corpus does not take or
    the data section does not name, kinds: {kind: {texts, bytes}} of each
    kind on its own, texts: every text}, from the summary alone: a plain
    text counts for the first share whose tag it has, as the order takes
    it."""
    allowed = set(kinds) if kinds else None
    sizes = {kind: {"texts": one[0], "bytes": one[2]} for kind, one in summary["kinds"].items()}

    def named(kind: str) -> bool:
        return allowed is None or kind in allowed
    taken_plain = named(PLAIN)
    tags = _tags(shares)
    pools = [0] if tags is None else list(range(len(tags)))
    pool_bytes, pool_texts, no_share = dict.fromkeys(pools, 0), dict.fromkeys(pools, 0), 0
    if taken_plain and tags is None:
        plain = sizes.get(PLAIN, {"texts": 0, "bytes": 0})
        pool_bytes[0], pool_texts[0] = plain["bytes"], plain["texts"]
    elif taken_plain:
        for joined, (texts, _, size) in summary["plain_tagsets"].items():
            held = set(joined.split("\0")) if joined else set()
            pool = next((at for at, tag in enumerate(tags) if tag in held), None)
            if pool is None:
                no_share += texts
            else:
                pool_bytes[pool] += size
                pool_texts[pool] += texts
    items = [kind for kind in TRAINING_KINDS if named(kind)]
    other = sum(one["texts"] for kind, one in sizes.items()
                if kind not in items and not (kind == PLAIN and taken_plain))
    return {"pool_bytes": pool_bytes, "pool_texts": pool_texts, "plain": sum(pool_texts.values()),
            "item_bytes": sum(sizes.get(kind, {"bytes": 0})["bytes"] for kind in items),
            "items": sum(sizes.get(kind, {"texts": 0})["texts"] for kind in items),
            "task_bytes": sizes.get("task", {"bytes": 0})["bytes"] if "task" in items else 0,
            "no_share": no_share, "other": other, "kinds": sizes,
            "texts": sum(one["texts"] for one in sizes.values())}


# What one worker of the build holds: the vocabulary (a worker reads the depot through depot.api, which opens it
# once in each process)
_HELD = {}


def _start_worker(vocabulary: bytes):
    _HELD.clear()
    _HELD["tokenizer"] = Tokenizer.from_bytes(vocabulary)


class _Workers:
    """Runs work on batches in order, in the build's own process for the
    first IN_PROCESS bytes of texts and on a pool of worker processes
    (started then, owned by stack) for the rest, so a build that reads
    little, such as a small size of a large snapshot, never pays for
    starting the workers. The results are those of work alone, whichever
    process ran it. pool: the pool once started, else None."""

    def __init__(self, stack: ExitStack, vocabulary: bytes, what: str):
        self.stack, self.vocabulary, self.what = stack, vocabulary, what
        self.pool = self.sizing = None
        _start_worker(vocabulary)

    def results(self, work, batches, items_of, bytes_of) -> Iterator[tuple]:
        """(batch, work(items_of(batch))) for each batch of batches, in
        order; bytes_of(batch) gives the bytes of texts it reads."""
        batches = iter(batches)
        done = 0
        while self.pool is None:
            batch = next(batches, None)
            if batch is None:
                return
            yield batch, work(items_of(batch))
            done += bytes_of(batch)
            if done > IN_PROCESS:
                self.sizing = Sizing(self.what)
                self.pool = self.stack.enter_context(process_pool(self.sizing, self.sizing.limit, _start_worker,
                                                                  (self.vocabulary,)))
        waiting = collections.deque()  # the batches handed to in_order, whose results come in their order

        def feeding():
            for batch in batches:
                waiting.append(batch)
                yield items_of(batch)
        for found in in_order(work, feeding(), self.pool, self.sizing):
            yield waiting.popleft(), found


def _tokenized(batch: list[tuple]) -> list[tuple]:
    """For each (its depot's address, id, revision, kind, read, piece) of
    batch: (its credits row, its tokens as an array, or a task's content
    as UTF-8 bytes, or None when read is False and it is not read), in
    order; a piece (its number, None for a whole text) read and encoded
    alone (depot.pieces). The depot is read through depot.api."""
    tokenizer = _HELD["tokenizer"]
    asked = [(address, text, revision, piece) for address, text, revision, _, read, piece in batch if read]
    read_ones = _read(asked)
    found = []
    for _, _, _, kind, read, _ in batch:
        if not read:
            found.append((None, None))
            continue
        row, value = next(read_ones)
        if kind == PLAIN:
            encoder = _Encoder(tokenizer)
            encoder.feed(value)
            data = np.concatenate([encoder.finish(), np.array([END], np.uint32)])
        elif kind == "conversation":
            data = np.asarray(chat.encode(tokenizer, json.loads(value))[0], np.uint32)
        else:
            data = value
        found.append((row, data))
    return found


def _read(asked: list[tuple]) -> Iterator[tuple[dict, bytes]]:
    """For each (its depot's address, id, revision, piece) of asked, in
    order: (its credits, depot.api.credits; its content as UTF-8
    bytes, depot.api.contents), each depot asked once for the credits
    and once for the contents of a run of entries of the same address."""
    for address, run in itertools.groupby(asked, key=lambda one: one[0]):
        run = [one[1:] for one in run]
        rows = api.credits(address, [(text, revision) for text, revision, _ in run])
        yield from zip(rows, api.contents(address, run))


def _conversations(batch: list[tuple]) -> list[tuple]:
    """For each (its depot's address, id, revision) of a conversation of batch:
    (its credits row, its tokens of the chat template, its excerpts as
    (the quoted text's id, the selector), in order), in order."""
    found = []
    for row, value in _read([(address, text, revision, None) for address, text, revision in batch]):
        conversation = json.loads(value)
        data = np.asarray(chat.encode(_HELD["tokenizer"], conversation)[0], np.uint32)
        found.append((row, data, [(block["source"], block["selector"]) for block in quotes.excerpts(conversation)]))
    return found


def _matched(batch: list[tuple]) -> list[list]:
    """For each (its depot's address, id, revision, whether the corpus took
    it as pieces, selectors) of a quoted plain text of batch: for each
    selector, in order, None when it matches no place or several of that
    content version, else the pieces its passage lies in (none for a text
    taken whole), as depot.api.matched finds them."""
    return [api.matched(address, text, revision, selectors, pieced)
            for address, text, revision, pieced, selectors in batch]


def _plain_counted(batch: list[tuple]) -> list[int]:
    """For each (its depot's address, id, revision, piece) of a plain text
    or a piece of batch: the tokens it takes in the corpus, END included,
    as _tokenized gives them, in order."""
    return [len(data) for _, data in _tokenized([(address, text, revision, PLAIN, True, piece)
                                                 for address, text, revision, piece in batch])]


def _counted(batch: list[tuple]) -> list[int]:
    """For each (its depot's address, id, revision, kind) of an item of batch: the
    tokens it takes in the corpus, in order: a conversation's tokens of the
    chat template, a task's tokens of its content as JSON text (task_text)."""
    tokenizer = _HELD["tokenizer"]
    found = []
    for (*_, kind), (_, value) in zip(batch, _read([(address, text, revision, None)
                                                     for address, text, revision, _ in batch])):
        content = json.loads(value)
        if kind == "conversation":
            found.append(len(chat.encode(tokenizer, content)[0]))
        else:
            found.append(task_tokens(tokenizer, content))
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


def mixed(found: dict, shares: list[dict]) -> dict:
    """Figures of what texts hold ({texts, characters, bytes, kinds,
    shares: one row for each tag of shares, [{tag, share}], in order,
    unshared}) with the mix of the corpus of every
    token when shares are given (mix None otherwise): each share's row
    gains share, kept_bytes and kept_characters (what scarcest_trim keeps
    of it, its characters in the same proportion; a share of no proportion
    takes nothing, and the others are measured among themselves) and
    limits (true on the share that keeps all it holds while another keeps
    less); mix: {kept_bytes, kept_characters of plain text, items_share:
    the share of the bytes the conversations and tasks hold then, the part
    of a size the build gives them together, kind_shares: {kind: its own
    share of those bytes} for each kind beside plain a corpus takes
    (TRAINING_KINDS)}."""
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
    items = counts.among(found["kinds"], TRAINING_KINDS)["bytes"]
    whole = items + plain
    return {**found, "shares": rows, "mix": {
        "kept_bytes": plain, "kept_characters": sum(one["kept_characters"] for one in rows),
        "items_share": items / whole if whole else 0,
        "kind_shares": {kind: counts.among(found["kinds"], [kind])["bytes"] / whole if whole else 0
                        for kind in TRAINING_KINDS}}}



class _Build:
    """One build, as build() says: its state, the files it writes and uploads, and its stages."""

    def __init__(self, depot, snapshot, data, dock, tokenizer, address, order, tokenizer_recipe, tokenizer_version,
                 code_commit, work, checkpoint, save, progress, room, shard_tokens, log):
        self.source, self.dock, self.tokenizer = depot.address, dock, tokenizer  # the depot, read through api
        self.data, self.address, self.order, self.code_commit = data, address, order, code_commit
        self.tokenizer_recipe, self.tokenizer_version = tokenizer_recipe, tokenizer_version
        self.snapshot = check(depot, snapshot, data, tokenizer)
        self.wanted = canonical(data)
        self.limit = self.wanted["total_tokens"]
        self.shares = self.wanted["shares"] or None
        self.kinds = self.wanted["kinds"] or None
        self.holder = self.snapshot["owner"]  # the depot's name every row of the corpus carries
        self.work, self.state, self.save, self.progress, self.log = work, checkpoint, save, progress, log
        self.shard_tokens = shard_tokens
        self.vocabulary = tokenizer.to_bytes()
        self.vocabulary_sha256 = hashlib.sha256(self.vocabulary).hexdigest()
        self.erasures = _erasures(self.source, snapshot)
        self.key = build_key(self.snapshot, data, self.vocabulary_sha256)
        self.folder = f"{FOLDER}/{self.key}"
        self.room = room(self.folder) if room is not None else corpus_room.Room(dock, None, self.folder)

    def run(self) -> dict:
        known = dock_api.get_json(self.dock, f"{self.folder}/{RECORD}")
        if known is not None and fresh(known):
            self.log(f"corpus {known['id']} of snapshot {self.snapshot['name']} is already built")
            return known
        self.replaced = known
        self.census = census(self.snapshot["summary"], self.shares, self.kinds)
        self.progress({"stage": "ordering", "texts": self.snapshot["texts"]})
        size = api.listing_size(self.source, self.snapshot["id"])
        self.walked = size["entries"]  # what the walk gives, a piece counting once
        if size["long_texts"]:
            self.log(f"{size['long_texts']:,} plain texts of snapshot {self.snapshot['name']} are longer than "
                     f"{size['piece_characters']:,} characters: each comes as its pieces, {size['pieces']:,} in all")
        self.work.mkdir(parents=True, exist_ok=True)
        named = apsw.Connection(str(self.work / "named.sqlite"))
        try:
            named.execute("CREATE TABLE IF NOT EXISTS named (kind TEXT NOT NULL, id TEXT NOT NULL, "
                          "PRIMARY KEY (kind, id)) WITHOUT ROWID")
            # the conversations the walk passed, by their place in it, and the plain texts it took (piece -1: whole)
            named.execute("CREATE TABLE IF NOT EXISTS held (walked INTEGER NOT NULL PRIMARY KEY, source TEXT NOT NULL, "
                          "id TEXT NOT NULL, revision INTEGER NOT NULL, bytes INTEGER NOT NULL, tags TEXT NOT NULL)")
            named.execute("CREATE TABLE IF NOT EXISTS taken (source TEXT NOT NULL, id TEXT NOT NULL, "
                          "piece INTEGER NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY (source, id, piece)) "
                          "WITHOUT ROWID")
            return self._stream(named)
        finally:
            named.close()

    def _stream(self, named_file) -> dict:
        state = self.state
        if state.get("key") == self.key and (gone := corpus_room.missing(self.dock, self.folder, _uploaded(state))):
            self.log(f"{len(gone)} files this build uploaded are no longer on the dock: it starts over")
            state = {}
        if state.get("key") == self.key and state.get("erasures") != self.erasures:
            self.log("an erasure reached a text of the snapshot since this build stopped: it starts over")
            state = {}
        if state.get("key") != self.key:  # a state of another build starts nothing
            state = {"key": self.key, "erasures": self.erasures, "after": None, "walked": 0, "items": None,
                     "shards": {kind: [] for kind in TOKEN_KINDS}, "positions": dict.fromkeys(TOKEN_KINDS, 0),
                     "texts": [], "tasks": 0, "tasks_count": 0, "pools": {}, "fill": None, "taken": 0, "whole": 0,
                     "pieces": 0, "conversations": 0, "items_tokens": 0, "plan": None, "seen": 0, "covered": {},
                     "groups": {}, "resolved": 0, "unquoted": {LEFT_OUT: 0, UNMATCHED: 0}, "dropped_tokens": 0}
            for table in ("named", "held", "taken"):
                named_file.execute(f"DELETE FROM {table}")
        shares = self.shares
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
        # each share's groups (depot.groups.groups_of): {share's tag: {group's tag: [texts, tokens]}}
        group_counts = {share: {tag: [texts, tokens] for tag, texts, tokens in found}
                        for share, found in state["groups"].items()}
        named_groups = {}  # (share's tag, a text's tags): its groups, a few kept
        tasks_path = self.work / "tasks.jsonl"
        with open(tasks_path, "ab") as tasks_file:
            tasks_file.truncate(state["tasks"])
        writers = {kind: _Shards(self.work, kind, state["positions"][kind]) for kind in TOKEN_KINDS}
        credits, named, credit_bytes = [], set(), [0]
        held_rows, taken_rows = [], []  # the conversations passed and the plain texts taken, since the last flush
        keep_taken = self.census["items"] > 0  # only a conversation reads which plain texts the corpus took
        read_bytes = {"plain": 0, "items": 0}  # the bytes of the texts this run took: the size estimate's base
        plain = self.census["plain"]

        def batches():
            """The texts the corpus takes from the state's place on, in batches of (entry, read, walked, place):
            read is whether it is read, which a closed pool, a limit in bytes that leaves it out, or a share's
            choice by group (depot.groups) saves; walked how many texts the walk passed up to it; place the
            entry's place among the plan's entries, None for an entry of a share the plan does not cover. With
            a limit, an item _limit_items did not take is passed, and the batches end once every pool is closed,
            every share the plan covers is past its last chosen text, and no item taken follows."""
            ahead = fill.copy() if counted == "bytes" else None  # the limit in bytes, worked out ahead of the results
            picker = chosen.picker(state["seen"])
            batch, size, walked = [], 0, state["walked"]
            for entry in walk(self.source, self.snapshot["id"], _tags(shares), self.kinds, state["after"]):
                walked += 1
                if counted == "tokens" and fill.done() and picker.done() and (
                        last_item is None or tuple(entry.place()) > last_item):
                    break
                if not entry.taken or (entry.kind != PLAIN and not item_taken(entry)):
                    continue
                place = None
                if entry.kind != PLAIN or counted is None:
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
            file = dock_api.put_from(self.dock, key, path)
            path.unlink()
            return {"key": file["key"], "sha256": file["sha256"], "bytes": size}

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

        def flush():
            """Keep in the work folder the ids named, the conversations passed and the plain texts taken so far:
            the walk gives them again, the same, after a stop."""
            with named_file:
                named_file.executemany("INSERT OR IGNORE INTO named (kind, id) VALUES (?, ?)", sorted(named))
                named_file.executemany("INSERT OR IGNORE INTO held (walked, source, id, revision, bytes, tags) "
                                       "VALUES (?, ?, ?, ?, ?, ?)", held_rows)
                named_file.executemany("INSERT OR IGNORE INTO taken (source, id, piece, revision) VALUES (?, ?, ?, ?)",
                                       taken_rows)
            named.clear()
            held_rows.clear()
            taken_rows.clear()

        def checkpoint(after: list, walked: int, final: bool = False):
            """Upload the token files and the texts' part begun, and hand the state that resumes after them;
            before a build's first upload, the cap's check of its estimated size (Room.expect)."""
            tasks_file.flush()
            ready = []  # (local path, key on the dock, the list its reference joins, what the reference adds)
            for kind in TOKEN_KINDS:
                done = writers[kind].close()
                if done is not None:
                    ready.append((done.pop("path"), f"{self.folder}/{done['token_sha256']}.parquet",
                                  state["shards"][kind], done))
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
            if ready and not (any(state["shards"].values()) or state["texts"]):
                self.room.expect(estimate(sum(path.stat().st_size for path, *_ in ready), final))
            for path, key, kept, added in ready:
                kept.append({**upload(path, key), **added})
            flush()
            state["groups"] = {share: [[tag, *counted_so] for tag, counted_so in found.items()]
                               for share, found in group_counts.items()}
            state.update(after=after, walked=walked, fill=None if fill is None else fill.state(),
                         positions={kind: writers[kind].position for kind in TOKEN_KINDS},
                         tasks=tasks_path.stat().st_size)
            self.save(json.loads(json.dumps(state)))

        with ExitStack() as stack:
            tasks_file = stack.enter_context(open(tasks_path, "ab"))
            workers = _Workers(stack, self.vocabulary, "tokenizing workers")

            def spread(work, items: list) -> list:
                """work's results for items, in order: on the workers, in parts, when the build started them."""
                if workers.pool is None:
                    return work(items)
                step = max(1, min(BATCH_TEXTS, -(-len(items) // (2 * workers.sizing.limit))))
                parts = (items[at:at + step] for at in range(0, len(items), step))
                return [one for found in in_order(work, parts, workers.pool, workers.sizing) for one in found]
            results = workers.results(_tokenized, batches(), self._items,
                                      lambda batch: sum(one[0].bytes for one in batch if one[1]))
            for batch, found in results:
                for (entry, read, walked, place), (row, data) in zip(batch, found):
                    if entry.kind == "conversation":  # taken once the walk knows every plain text taken (_resolve)
                        held_rows.append((walked, self.holder, entry.id, entry.revision, entry.bytes, entry.tags))
                        continue
                    if place is not None:  # a share that chose its texts by group: the chosen ones, within its budget
                        state["seen"] = place + 1
                        if read:
                            used = state["covered"].get(str(entry.pool), 0) + (
                                len(data) if counted == "tokens" else entry.bytes)
                            if used > budget[entry.pool]:
                                raise AssertionError(f"the share {tags[entry.pool]} chose {entry.id}, which passes "
                                                     "its budget")
                            state["covered"][str(entry.pool)] = used
                    elif counted is not None and entry.kind == PLAIN and (read or counted == "bytes"):
                        took = fill.take(entry.pool, len(data) if counted == "tokens" else entry.bytes)
                        if counted == "bytes" and took != read:
                            raise AssertionError(f"the limit in bytes left {entry.id} out on one pass and not the "
                                                 "other")
                        read = took
                    if not read:
                        continue
                    source_name = self.holder
                    pool_tag = tags[entry.pool] if entry.kind == PLAIN else ""
                    if entry.kind == "task":
                        start = state["tasks_count"]
                        content = json.loads(data.decode("utf-8"))
                        tasks_file.write(task_text({"depot": source_name, "id": entry.id,
                                                    "content": content}).encode("utf-8") + b"\n")
                        state["tasks_count"] += 1
                        if limit is None:  # with a limit, the items' tokens were counted already (_limit_items)
                            state["items_tokens"] += task_tokens(self.tokenizer, content)
                    else:
                        start = writers[PLAIN].add(data, source_name, entry.name, pool_tag)
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
                        if keep_taken:
                            taken_rows.append((source_name, entry.id, -1 if entry.piece is None else entry.piece,
                                               entry.revision))
                    state["taken"] += 1
                    credits.append({"depot": source_name, "id": entry.id, "revision": entry.revision, **row,
                                    "tags": json.loads(entry.tags), "tokens": start, "piece": entry.piece})
                    credit_bytes[0] += entry.bytes
                    read_bytes["plain" if entry.kind == PLAIN else "items"] += entry.bytes
                    named.update(("license", one) for one in row["licenses"])
                    if row["notice"] is not None:
                        named.add(("notice", row["notice"]))
                    if any(writer.full(self.shard_tokens) for writer in writers.values()):
                        checkpoint(entry.place(), walked)
                state["after"], state["walked"] = batch[-1][0].place(), batch[-1][2]
                self.progress({"stage": "reading", "position": state["walked"], "of": self.walked,
                               "texts": state["taken"], "tokens": sum(state["pools"].values()),
                               "token files": sum(len(kept) for kept in state["shards"].values())})
            flush()
            last = state["resolved"]
            while page := list(named_file.execute("SELECT walked, source, id, revision, bytes, tags FROM held "
                                                  "WHERE walked > ? ORDER BY walked LIMIT ?", (last, HELD_PAGE))):
                resolved = self._resolve([(source_name, text, revision) for _, source_name, text, revision, _, _ in page],
                                         named_file, spread)
                for (walked_at, source_name, text, revision, size, tags), (row, data, quoted, why) in zip(page,
                                                                                                         resolved):
                    state["resolved"] = last = walked_at
                    if why is not None:
                        state["unquoted"][why] += 1
                        state["dropped_tokens"] += len(data)
                        continue
                    start = writers["conversation"].add(data, source_name, text, "")
                    state["conversations"] += 1
                    state["items_tokens"] += len(data)
                    state["taken"] += 1
                    credits.append({"depot": source_name, "id": text, "revision": revision, **row,
                                    "tags": json.loads(tags), "tokens": start, "piece": None, "quotes": quoted})
                    credit_bytes[0] += size
                    read_bytes["items"] += size
                    named.update(("license", one) for one in row["licenses"])
                    if row["notice"] is not None:
                        named.add(("notice", row["notice"]))
                    if writers["conversation"].full(self.shard_tokens):
                        checkpoint(state["after"], state["walked"])
                self.progress({"stage": "taking conversations", "conversations": state["conversations"],
                               "left out": sum(state["unquoted"].values())})
            checkpoint(state["after"], state["walked"], final=True)
        self.progress({"stage": "finishing", "texts": state["taken"]})
        return self._finish(state, tasks_path, plain, named_file)

    def _items(self, batch) -> list[tuple]:
        """What the workers read of a batch (_tokenized): a conversation is read once the walk ends (_resolve)."""
        return [(self.source, entry.id, entry.revision, entry.kind,
                 read and entry.kind != "conversation", entry.piece) for entry, read, _, _ in batch]

    def _resolve(self, conversations: list[tuple], named_file, spread) -> list[tuple]:
        """For each (the name of its depot, id, revision) of a conversation of
        conversations, in order: (its credits row, its tokens of the chat
        template, the ids of the texts it quotes, why the corpus leaves it
        out or None), once the walk took every plain text it takes (the
        work folder's table taken). Each excerpt's text must be a plain text
        the corpus took from the same depot, and its selector must match
        exactly one place of the content version the corpus took
        (depot.quotes); of a text taken as pieces, the corpus must hold
        each piece the passage lies in. Otherwise it is left out, for its
        first excerpt that fails: LEFT_OUT, or UNMATCHED for a selector that
        matches no place or several. spread runs the reading on the
        workers: the conversations read and tokenized (_conversations), then
        each quoted content version read once for all the selectors on it
        (_matched)."""
        read = spread(_conversations, [(self.source, text, revision)
                                       for _, text, revision in conversations])
        taken = {}  # (depot name, id): its rows of taken, (piece, revision)
        asked = {}  # (depot name, id, revision, whether taken as pieces): {the selector as JSON: its place in asked}
        for (name, _, _), (_, _, excerpts) in zip(conversations, read):
            for source, selector in excerpts:
                if (name, source) not in taken:
                    taken[name, source] = list(named_file.execute(
                        "SELECT piece, revision FROM taken WHERE source = ? AND id = ?", (name, source)))
                rows = taken[name, source]
                if rows:
                    on = asked.setdefault((name, source, rows[0][1], rows[0][0] != -1), {})
                    on.setdefault(json.dumps(selector, sort_keys=True), len(on))
        jobs = list(asked.items())
        matched = spread(_matched, [(self.source, source, revision, pieced,
                                     [json.loads(one) for one in selectors])
                                    for (name, source, revision, pieced), selectors in jobs])
        lying = {(key, selector): found[at] for (key, selectors), found in zip(jobs, matched)
                 for selector, at in selectors.items()}
        resolved = []
        for (name, _, _), (row, data, excerpts) in zip(conversations, read):
            why = None
            for source, selector in excerpts:
                rows = taken[name, source]
                if not rows:
                    why = LEFT_OUT
                    break
                pieced = rows[0][0] != -1
                held = lying[(name, source, rows[0][1], pieced), json.dumps(selector, sort_keys=True)]
                if held is None:
                    why = UNMATCHED
                    break
                if pieced and not set(held) <= {piece for piece, _ in rows}:
                    why = LEFT_OUT
                    break
            resolved.append((row, data, list(dict.fromkeys(source for source, _ in excerpts)), why))
        return resolved

    def _plan(self, state: dict, budget: dict | None, counted: str | None, shares: list[dict] | None
              ) -> groups.Plan:
        """The texts the shares choose by group (depot.groups), sizes
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
            chosen = groups.plan(walk(self.source, self.snapshot["id"], _tags(shares), self.kinds), _tags(shares),
                                 budget, count, self.progress, self.source)
        sha256 = chosen.save(path)
        if known is not None and known["sha256"] != sha256:
            raise AssertionError("the shares chose other texts than this build chose before it stopped")
        state["plan"] = {"sha256": sha256, "last": {str(pool): last for pool, last in chosen.last.items()}}
        if chosen.last:
            self.log(f"the shares {', '.join(shares[pool]['tag'] for pool in sorted(chosen.last))} split their "
                     f"budgets over the tags of their texts, by the square root of what each holds: {len(chosen.chosen):,} texts chosen")
        return chosen

    def _limit_items(self, shares: list[dict] | None) -> dict:
        """The items (conversations and tasks) a build at total_tokens takes:
        their share of it (the items' part of the bytes of the corpus
        without total_tokens, from the census), filled by their tokens counted
        in the order, as one pool (Fill). {tokens, texts, bytes: of the
        items taken, last: the place of the last item taken (None: none),
        skipped: the places of the items before it left out}; the walk ends
        once the pool closes, or at the end of the snapshot."""
        item_bytes = self.census["item_bytes"]
        found = {"tokens": 0, "texts": 0, "bytes": 0, "last": None, "skipped": []}
        if not self.census["items"]:
            return found
        plain = sum((self.census["pool_bytes"] if not shares else self._byte_budget(shares, False)).values())
        part = self.limit * item_bytes // (item_bytes + plain) if item_bytes else 0
        fill = Fill({0: part})

        def batches():
            batch, size = [], 0
            for entry in walk(self.source, self.snapshot["id"], _tags(shares), self.kinds):
                if not entry.taken or entry.kind == PLAIN:
                    continue
                batch.append(entry)
                size += entry.bytes
                if size >= BATCH_BYTES or len(batch) >= BATCH_TEXTS:
                    yield batch
                    batch, size = [], 0
            if batch:
                yield batch

        def items(batch):
            return [(self.source, entry.id, entry.revision, entry.kind) for entry in batch]

        with ExitStack() as stack:
            results = _Workers(stack, self.vocabulary, "counting workers").results(
                _counted, batches(), items, lambda batch: sum(entry.bytes for entry in batch))
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
        """{pool: the bytes it may take} without total_tokens: the scarcest
        share all of its bytes, each other its proportion of the scarcest's
        (from the census). A share with no text is refused, or with refuse
        False takes nothing and the others are measured among themselves.
        With refuse, a share that may take all its bytes has no limit
        (None), so it takes every text."""
        available = {at: 0 for at in range(len(shares))} | self.census["pool_bytes"]
        empty = [shares[at]["tag"] for at, size in available.items() if not size]
        if empty and refuse:
            raise Refused(f"the shares cannot be met: no plain text of the snapshot {self.snapshot['name']} the "
                          f"order takes has the tag {', '.join(empty)}")
        trimmed = {at: int(size) for at, size in scarcest_trim(available, shares).items()}
        return trimmed if not refuse else {at: None if size >= available[at] else size
                                           for at, size in trimmed.items()}

    def _finish(self, state: dict, tasks_path: Path, plain: int, named_file) -> dict:
        built_at = datetime.now(UTC)
        named = {kind: [one for (one,) in named_file.execute("SELECT id FROM named WHERE kind = ? ORDER BY id",
                                                             (kind,))] for kind in ("license", "notice")}
        license_rows = api.licenses(self.source, named["license"])
        notice_rows = api.notices(self.source, named["notice"])
        tasks_sha256 = _sha256_file(tasks_path)
        long_taken = next(named_file.execute("SELECT count(*) FROM named WHERE kind = 'long text'"))[0]
        plain_taken = state["whole"] + long_taken  # a text taken as pieces counts once
        corpus_id = _fingerprint({"build": self.key, "vocabulary": self.vocabulary_sha256,
                                  "texts": [part["fingerprint"] for part in state["texts"]],
                                  "shards": {kind: [s["token_sha256"] for s in kept]
                                             for kind, kept in state["shards"].items()},
                                  "tasks": tasks_sha256, "licenses": license_rows, "notices": notice_rows})

        def put(table: str, rows: list[dict]) -> dict:
            path = self.work / f"{table}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=CORPUS_SCHEMAS[table]), path, compression="zstd")
            size = path.stat().st_size
            self.room.make(size)
            reference = dock_api.put_from(self.dock, f"{self.folder}/{table}.parquet", path)
            path.unlink()
            return reference

        self.room.make(len(self.vocabulary))
        vocabulary = dock_api.put(self.dock, f"{self.folder}/vocabulary", self.vocabulary)
        items_taken = state["conversations"] + state["tasks_count"]
        unquoted = sum(state["unquoted"].values())
        left_out = {"no share": self.census["no_share"],
                    "kinds not taken": self._not_taken(),
                    "trimmed": max(0, plain - plain_taken),
                    "trimmed items": max(0, self.census["items"] - items_taken - unquoted), **state["unquoted"]}
        record = {
            "id": corpus_id, "folder": self.folder, "depot": self.address,
            "snapshot": {"id": self.snapshot["id"], "name": self.snapshot["name"], "owner": self.holder,
                         "change": self.snapshot["at_change"], "digest": self.snapshot["digest"]},
            "recipe": self.order, "data": self.data, "build_key": self.key, "code_commit": self.code_commit,
            "rules": RULES, "erasures": self.erasures, "built_at": built_at.isoformat(),
            "texts": state["taken"] - state["pieces"] + long_taken, "pieces": state["pieces"],
            "conversations": state["conversations"],
            "tokenizer": {"recipe": self.tokenizer_recipe, "recipe_version": self.tokenizer_version,
                          "sha256": self.vocabulary_sha256, "vocab_size": self.tokenizer.vocab_size,
                          "vocabulary": vocabulary},
            "vocabulary": {"sha256": self.vocabulary_sha256, "size": self.tokenizer.vocab_size},
            "max_tokens": self.limit,
            "shards": state["shards"],
            "tables": {"texts": state["texts"], "licenses": put("licenses", license_rows),
                       "notices": put("notices", notice_rows)},
            "total_tokens": sum(s["tokens"] for s in state["shards"][PLAIN]),
            "items_tokens": (state["items_tokens"] if self.limit is None
                             else state["items"]["tokens"] - state["dropped_tokens"]),
            "shares_tokens": None if self.shares is None else [
                {"tag": share["tag"], "tokens": state["pools"].get(share["tag"], 0),
                 "groups": [{"tag": tag, "texts": texts, "tokens": tokens} for tag, texts, tokens in sorted(
                     state["groups"].get(share["tag"], []), key=lambda one: (one[0] is not None, one[0] or ""))]}
                for share in self.shares],
            "left_out": left_out,
            "fingerprints": {
                **{kind: _fingerprint([self.vocabulary_sha256, [s["token_sha256"] for s in kept]])
                   for kind, kept in state["shards"].items()},
                "task": _fingerprint([self.vocabulary_sha256, tasks_sha256]),
            },
        }
        if state["tasks_count"]:
            size = tasks_path.stat().st_size
            self.room.make(size)
            record["tasks"] = {"file": dock_api.put_from(self.dock, f"{self.folder}/tasks.jsonl", tasks_path),
                               "count": state["tasks_count"]}
        gone = corpus_room.missing(self.dock, self.folder, [*_uploaded(state), *(
            [record["tasks"]["file"]] if "tasks" in record else []), vocabulary,
            *(one for one in record["tables"].values() if isinstance(one, dict))])
        if gone:
            raise corpus_room.FilesGone(f"{len(gone)} files of this build were deleted while it ran, {gone[0]} first: no "
                                    "record was written; build again")
        # an erasure carried out while the build ran may have reached a text it took after the build passed it:
        # checked before the record is written, and again after, for one carried out in between (an erasure
        # carried out once the record exists marks it itself, freight.corpora.mark_stale)
        now, hit = self._erased_since(self.erasures, record)
        if hit:
            record["stale"] = {"erasures": hit}
        record = dock_api.put_json(self.dock, f"{self.folder}/{RECORD}", record)
        _, hit = self._erased_since(now, record)
        if hit:
            record = corpora.marked_stale(self.dock, record, hit)
        self.log(f"corpus {corpus_id} of snapshot {self.snapshot['name']}: {record['texts']:,} texts"
                 + (f" ({state['pieces']:,} pieces of {long_taken:,} of them)" if state["pieces"] else "") + ", "
                 f"{record['total_tokens']:,} tokens of plain text in {len(record['shards'][PLAIN])} token files, "
                 f"{len(record['shards']['conversation'])} of conversations, {state['tasks_count']:,} tasks")
        self._drop_replaced(record)
        if not fresh(record):
            self.log(f"corpus {corpus_id} is stale already: the erasures {', '.join(record['stale']['erasures'])}, "
                     "carried out while it was built, reached a text it holds; build it again")
        return record

    def _drop_replaced(self, record: dict):
        """Delete the files of the record this build replaced in its folder (a stale or deleted one) that the new
        record does not name."""
        if self.replaced is None:
            return
        kept = {one["key"] for one in corpora.files(record)}
        for one in corpora.files(self.replaced):
            if one["key"] not in kept and dock_api.has(self.dock, one["key"]):
                dock_api.delete(self.dock, one["key"])
                self.log(f"{one['key']} deleted: {REBUILT}")

    def _not_taken(self) -> dict[str, int]:
        """{kind: the texts of the snapshot of that kind the corpus does not take}: kinds no step reads and kinds
        the data section does not name."""
        allowed = self.kinds
        return {kind: one["texts"] for kind, one in self.census["kinds"].items() if one["texts"] and (
            (kind != PLAIN and kind not in TRAINING_KINDS) or (allowed is not None and kind not in allowed))}

    def _erased_since(self, before: list[str], record: dict) -> tuple[list[str], list[str]]:
        """(the erasures the depot has carried out now that reached a text of the snapshot (_erasures), the ids
        of those not in before that reached a text the corpus holds)."""
        now = _erasures(self.source, self.snapshot["id"])
        new = set(now) - set(before)
        hit = [one["id"] for one in api.erasures(self.source)["erasures"] if one["id"] in new
               and corpora.holds(self.dock, record, set(one["texts"]))]
        return now, sorted(hit)


def _uploaded(state: dict) -> list[dict]:
    """The references of the files a build's state says it uploaded."""
    return [*(one for kept in state["shards"].values() for one in kept), *state["texts"]]


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
    """The tokens of a plain text fed a part at a time (UTF-8 bytes or str): each part is
    encoded up to its last place where the tokenizer's chunks always break
    (_boundary), so the tokens are those of the whole text, and only the
    bytes after that place are held, whatever the text's lines."""

    def __init__(self, tokenizer):
        self.tokenizer, self.pending, self.last, self.tokens = tokenizer, [], None, []

    def feed(self, value: bytes | str):
        data = value.encode("utf-8") if isinstance(value, str) else value
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


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as held:
        while chunk := held.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()

