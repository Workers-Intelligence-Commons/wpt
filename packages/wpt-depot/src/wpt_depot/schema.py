"""The depot's tables in its SQLite file, with the key of each, and the
views that show the current state.

Every id but a change's is text made of the depot's name, a hyphen and a
number (wpt_depot.ids), so ids never collide across depots.

Every write is one change of the depot, numbered by the depot-wide change
number: a row of changes, with its kind and the UTC time it was made at
(none for a change made before format 4). What changes in place keeps every
version as a row of its own table, `<name>_revisions`, keyed by its id and
its revision (1, 2, 3...), and each row records the change number it was
made at; the highest revision is the current one, the only row of its id
whose `latest` is 1. A text's big part, its content, is stored as content
versions (1, 2, 3...), each a row of content_lists with the change number it
was made at: its kind, its length in characters and in UTF-8 bytes, and the
ordered list of its parts. Every content version has a kind, a row of the
table kinds (wpt_depot.kinds): the format its content is read in, which says
which tokens a training step takes from it. The kind plain is a text read as
it is; every other kind is JSON that matches the kind's JSON Schema. A
text's kind is the kind of its first content version, and every later
version keeps it. Each
part is a row of parts, its text stored once for its text, compressed with
zstd, keyed by the text's id and the part's hash, with the change number
that first stored it (wpt_depot.parts says where a content is cut). Each
revision of a text holds only the small fields and the number of the
content version it reads (content_version) and, on the current revision,
the status its decisions make (wpt_depot.policy), kept by every write.
Each revision also keeps the kind, length in characters and UTF-8 bytes of
the content version it reads (content_kind, content_length, content_bytes:
SIZES), copied from its row of content_lists by every write that makes a
revision or a content version, and by an erasure's cut, which changes a
version's length in place (sizes_sql), so the counts, a decision, a change
of licenses or tags, a snapshot's read and a build's walk read them from the
revision alone, never from content_lists. The
decisions on a text are rows of text_decisions, one for each decision and
text with the change the decision was made at, and the texts view gives a
text's decisions as a list read from text_decisions (decision_list). The
column decisions of texts_revisions is unused, kept because removing it
rewrites the whole table: every revision the current format writes holds '[]'
there, and a past revision upgraded from an older format keeps the list it
held. A text's tags are rows of taggings, never part of a revision: one
for each text and tag, with the change the text joined the tag at
(joined_change), the change it left it at (left_change, null while it has
it) and the change that last wrote the row (change), by which a pull picks
the rows it lacks (wpt_depot.taggings). The current revision of a text
holds a copy of the tags it has now in its column current_tags, a JSON
array in the order of their names, as tags_now reads them from taggings:
every write of taggings sets it again (wpt_depot.taggings, tags_sql), and a
new revision takes it from the one before, which drops its own (null), so
a read of the current texts (the texts view, the counts, a count step) finds
a text's tags on the row it reads already. A read at a change reads the tags
the text had then from taggings (tags_at). The column tags of
texts_revisions is unused in the same way as decisions: every revision the
current format writes holds '[]' there, and a past revision upgraded from an
older format keeps the tags it held, which the frozen reader of the legacy
tables reads (wpt_depot.legacy). The table counts holds how
many current texts have each status, tag, license, score name and
decision, every text and the texts of each tag and license split by status
(those of each tag and license also in characters and bytes), and what the
texts of each status hold by kind and, for plain texts, by set of tags, in
texts, characters and bytes (wpt_depot.counts). text_tags and
text_licenses list the current texts of each tag and of each license, each
row a name and the text's place in the numeric order of ids
(ids.order_sql), so the texts of one tag or license are read in that order
from their key alone. text_titles is an SQLite FTS5 table of the trigrams of
each current text's title, with title_owners, the slot of each depot name
the ids carry, which numbers its rows (wpt_depot.titles), so a plain search
of words of 3 characters or more reads only the texts whose title holds
them. A decision and a notice
record the change that wrote them. The tokenizer recipes and vocabularies
a depot kept are read and no longer written (only a pull brings an
upstream's rows of them): each recipe's revisions (tokenizers_revisions),
whose version rose only when what trained its vocabulary changed, the
snapshot and kinds of its sample (sample), or before snapshots a dataset
(dataset, and depot, the web address of the depot that held it, or null
for this one), its sample's bytes and its vocabulary's size; vocabularies,
at most one vocabulary for each version of a recipe, its merges kept as
text, with the change that wrote it; vocabulary_texts, by position, every
text it trained on, by its depot ('' for the depot that holds the recipe,
else the web address of the depot whose sample a vocabulary trained before
snapshots read) and id (wpt_depot.legacy reads them; `wpt
adopt-vocabularies` puts each vocabulary on the dock). numbers holds, for each kind of id, the
last number of the depot's own ids, so a number is never given twice, even
once an erasure removed its row.

The legacy tables (LEGACY) hold the libraries and datasets of a depot of an
older format, as it left them: no write of the depot touches them, and only
wpt_depot.legacy reads them, for the models trained before snapshots, with
the upgrade to format 13, which makes snapshots of them
(wpt_depot.conversion).
Which texts a library holds is memberships: the change a text joined at and
the change it left at (null while it is in). A locked library also has
pins: the exact revision of each text it locked, with the change that
locked it. Each row of both also stores its text's export key (export_key,
wpt_depot.ids). library_moves holds the change at which what each library
held last moved. A dataset's vocabulary, when set, is the sha256 of one of
the depot's vocabularies: the one it published, which training took its
corpus of. Each revision of a dataset stores, in figures, what each of its
libraries held at the change it reads it at, as the write that made the
revision copied it from the counts of its time.

A snapshot (wpt_depot.snapshots) is a name on a change number: the set of
texts that were admitted at that change, each read at its revision as of
it. A row of snapshots holds its id (from numbers, like every id), the
change that wrote it, its name, unique among the snapshots of the depot
whose name its id carries (the index snapshots_name), the change it reads
(at_change), its time, where it comes from (origin), an optional note, the
texts, characters and bytes it holds, the SHA-256 of what it holds (digest)
and a summary of it by kind, tag, plain tag set, license and owner, made
from the counts. Its texts are rows of snapshot_sets, one for each part and
each depot whose texts it holds (owner): the change of that depot's
numbering they are read at (bound), and their numbers (the digits of their
ids) as a portable roaring bitmap. A snapshot and its sets are written once
and never changed nor removed, an erasure included. The upgrade to format
13 made a snapshot of each locked library and of the dataset revisions
models were trained on (wpt_depot.conversion); legacy_snapshot_of names,
for each dataset revision it converted, the snapshot that holds what it
read, its origin naming the revision or, for a revision that read one
locked library alone, that library.

A list is JSON text: a text's licenses, authors, tags and scores, the
decisions (ids, in order) the texts view gives, a content version's parts, a library's problems, an
erasure request's decisions, needles and erased passages, a dataset's
libraries, shares and figures. A tokenizer recipe's sample is JSON text,
{"snapshot": its id, "kinds": [kind, ...]}. A list of names is a JSON array of strings, so a tag
test is `EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'book')`; a
part is {"hash": its SHA-256 in hexadecimal, "length": its characters}; a
score is {"name", "value", "content_version"}, in the order of their names
(null: none); an erased passage is {"length", "hash_a", "hash_b" (both in
hexadecimal), "at", "needle"}; a problem is {"texts", "problem",
"decision"}; a needle is {"text", "field", "start", "end", "version"}; a
dataset revision's figures are {"at": the change they were read at, in the
numbering of the depot that owns the dataset, "libraries": one entry for
each library it reads, in order, null when not known, else {"kinds":
{kind: {"texts", "characters", "bytes"}} of every kind of the depot, each
on its own, plain first, then the others in the order of their ids,
"tags": the same for the plain texts that have each tag of its shares, in
order}}. A
time is ISO 8601 text in UTC.

Every table has a declared primary key (KEYS): the sessions that record
each change (wpt_depot.database) need one to name a row. The index
texts_order holds the current texts in the numeric order of their ids
(ids.order_sql), so a page of texts in that order, the first or a far one,
reads only its own rows. The index texts_status holds them by status, then
in that order, so a page of the texts of one status reads only its own
rows. The
indexes memberships_key and pins_key hold each library's rows by their
export key, with every other column a read of the library needs (the text,
and the changes it joined and left, or the revision pinned), so a library's
texts whose key falls in a range are read in key order from the index alone,
without reading the other libraries' rows nor its own outside the range
(wpt_depot.legacy), and a library is counted from the
index alone. The indexes memberships_order, pins_order and memberships_held,
which ordered a library's texts by id and its open memberships by text, are
kept as an older format made them. The index text_decisions_order holds the rows of
text_decisions by decision, then in the numeric order of their texts' ids;
its table has no rowid, so the index also holds each row's text, and the
texts of one decision are read from the index alone, a page of them in
that order (filters.Filter.decision). The indexes of change numbers
(memberships_joined, memberships_left, content_lists_change,
text_decisions_change, pins_change, and texts_change) let a pull read the
rows past a change without reading a whole table (wpt_depot.follow). The
partial index content_lists_long holds the content versions of kind
plain longer than LONG characters, those a corpus build cuts into pieces
(wpt_depot.pieces), so a build finds the long texts it reads from them
without reading every text (wpt_depot.order, wpt_depot.legacy). The partial
index texts_length holds the current texts by status, then by the length in
characters of the content version each reads (content_length), then in the
numeric order of their ids, so a page of texts ordered by length, the
longest first or last, reads only its own rows whatever the depot's size
(Depot.search), and so do the texts of a snapshot read by length
(wpt_depot.sizes). The table quotes holds,
for each content version of a conversation, the ids of the texts its
excerpts quote (wpt_depot.quotes), written with the version itself by
every change that writes one (an add, a change of content, an erasure's
cut, a clone or a pull, each made here from the content), and its index
quotes_text gives the conversations that quote a text, so an erasure
finds them (Depot.quoting) without reading every conversation.

A depot that follows an upstream (wpt_depot.follow) keeps the change
numbers of the rows it received, which are those of the depot that made
each row, so pulls records how the numberings meet: for each pull, and for
the clone itself, the depot that pulled, its change number then, its
upstream's name and the upstream's change number the pull brought it up to.
A clone receives its upstream's pulls rows too, so a clone of a clone can
follow the numbers back through each depot (Depot.bounds).

In SQL (wpt_depot.database), `<name>` is a view of the current state
(VIEWS): the current revision of each id, and for texts the content of the
version their current revision reads. `contents` shows every content version with
its whole text (`text` for kind plain, `content` for every other kind), its parts joined in order.
A value in SQLite holds at most SQL_STRING bytes, so a query that reads the
text of a content version bigger than that fails, naming it; its text is
read from its parts (wpt_depot.parts). A query that reads no text never
reads a part.

The layout above has a number, its format (FORMAT), which each about row
records: the depot's format when the row was written, so the current row
holds the depot's format now. The code reads and writes a depot of its own
FORMAT only (Depot.view). The format also covers the schemas of the
kinds a new depot starts with (wpt_depot.kinds), which the code that builds
corpora and checks items reads: a change of one raises it. A depot of
format 3 or older is a folder of Delta Lake tables; `wpt depot upgrade`
brings it up (wpt_depot.upgrade).
"""

from . import ids, titles
from .kinds import PLAIN

FORMAT = 14  # the format of the layout below: what a depot this code reads and writes is written at

UNDER_REVIEW, ADMITTED, REFUSED = "under review", "admitted", "refused"
SQL_STRING = 1_000_000_000  # bytes: the longest value SQLite holds
LONG = 1 << 20  # characters: a plain content version longer than this is in the index content_lists_long

TABLES = {
    "about_revisions": """revision INTEGER NOT NULL, change INTEGER NOT NULL, name TEXT NOT NULL,
        maintainer TEXT NOT NULL, takedown_contact TEXT NOT NULL, upstream TEXT, format INTEGER NOT NULL""",
    "changes": "id INTEGER NOT NULL, kind TEXT NOT NULL, made_at TEXT",
    "numbers": "name TEXT NOT NULL, last INTEGER NOT NULL",
    "licenses": "id TEXT NOT NULL, name TEXT NOT NULL, text TEXT NOT NULL",
    "notices": "id TEXT NOT NULL, change INTEGER NOT NULL, text TEXT NOT NULL",
    "kinds": "id TEXT NOT NULL, meaning TEXT NOT NULL, schema TEXT NOT NULL",
    "decisions": f"""id TEXT NOT NULL, change INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('{UNDER_REVIEW}', '{ADMITTED}', '{REFUSED}')),
        name TEXT NOT NULL, email TEXT NOT NULL, accepted INTEGER NOT NULL CHECK (accepted = 1),
        made_at TEXT NOT NULL, reason TEXT""",
    "texts_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, licenses TEXT NOT NULL, notice TEXT, title TEXT, authors TEXT, tags TEXT NOT NULL,
        address TEXT, decisions TEXT NOT NULL, content_version INTEGER NOT NULL, scores TEXT, status TEXT,
        content_kind TEXT, content_length INTEGER, content_bytes INTEGER, current_tags TEXT""",
    "text_decisions": "text TEXT NOT NULL, decision TEXT NOT NULL, change INTEGER NOT NULL",
    "content_lists": """id TEXT NOT NULL, version INTEGER NOT NULL, change INTEGER NOT NULL, kind TEXT NOT NULL,
        parts TEXT NOT NULL, length INTEGER NOT NULL, bytes INTEGER NOT NULL""",
    "parts": "id TEXT NOT NULL, hash BLOB NOT NULL, change INTEGER NOT NULL, value BLOB NOT NULL",
    "libraries_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, name TEXT NOT NULL, forked_from TEXT, problems TEXT, locked_at TEXT""",
    "memberships": """library TEXT NOT NULL, text TEXT NOT NULL, joined_change INTEGER NOT NULL, left_change INTEGER,
        export_key INTEGER""",
    "pins": """library TEXT NOT NULL, text TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        export_key INTEGER""",
    "erasures_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, reason TEXT NOT NULL, decisions TEXT NOT NULL, needles TEXT NOT NULL, erased TEXT""",
    "datasets_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, name TEXT NOT NULL, tokenizer TEXT NOT NULL, libraries TEXT NOT NULL, shares TEXT,
        max_tokens INTEGER, vocabulary TEXT, figures TEXT""",
    "tokenizers_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL, depot TEXT, dataset TEXT NOT NULL,
        sample_bytes INTEGER NOT NULL, vocab_size INTEGER NOT NULL, note TEXT, sample TEXT""",
    "vocabularies": """recipe TEXT NOT NULL, version INTEGER NOT NULL, change INTEGER NOT NULL, sha256 TEXT NOT NULL,
        merges TEXT NOT NULL, texts INTEGER NOT NULL, bytes INTEGER, made_at TEXT NOT NULL""",
    "vocabulary_texts": """recipe TEXT NOT NULL, version INTEGER NOT NULL, position INTEGER NOT NULL,
        depot TEXT NOT NULL, text TEXT NOT NULL, change INTEGER NOT NULL""",
    "pulls": "depot TEXT NOT NULL, change INTEGER NOT NULL, upstream TEXT NOT NULL, upstream_change INTEGER NOT NULL",
    "counts": ("facet TEXT NOT NULL, name TEXT NOT NULL, texts INTEGER NOT NULL, characters INTEGER NOT NULL DEFAULT 0, "
               "bytes INTEGER NOT NULL DEFAULT 0"),
    "library_moves": "library TEXT NOT NULL, change INTEGER",
    "quotes": "conversation TEXT NOT NULL, version INTEGER NOT NULL, text TEXT NOT NULL",
    "text_tags": "tag TEXT NOT NULL, key TEXT NOT NULL",
    "taggings": "text TEXT NOT NULL, tag TEXT NOT NULL, joined_change INTEGER NOT NULL, left_change INTEGER, "
                "change INTEGER NOT NULL",
    "text_licenses": "license TEXT NOT NULL, key TEXT NOT NULL",
    titles.OWNERS: "owner TEXT NOT NULL, slot INTEGER NOT NULL",
    "snapshots": """id TEXT NOT NULL, change INTEGER NOT NULL, name TEXT NOT NULL, at_change INTEGER NOT NULL,
        made_at TEXT NOT NULL, origin TEXT NOT NULL, note TEXT, texts INTEGER NOT NULL, characters INTEGER NOT NULL,
        bytes INTEGER NOT NULL, digest TEXT NOT NULL, summary TEXT NOT NULL""",
    "snapshot_sets": """snapshot TEXT NOT NULL, part INTEGER NOT NULL, owner TEXT NOT NULL, bound INTEGER NOT NULL,
        bitmap BLOB NOT NULL, texts INTEGER NOT NULL""",
    "legacy_snapshot_of": "dataset TEXT NOT NULL, revision INTEGER NOT NULL, snapshot TEXT NOT NULL",
}
KEYS = {"about_revisions": ("revision",), "changes": ("id",), "numbers": ("name",), "licenses": ("id",),
        "notices": ("id",), "kinds": ("id",), "decisions": ("id",), "texts_revisions": ("id", "revision"),
        "content_lists": ("id", "version"), "parts": ("id", "hash"), "libraries_revisions": ("id", "revision"),
        "memberships": ("library", "text", "joined_change"), "pins": ("library", "text"),
        "erasures_revisions": ("id", "revision"), "datasets_revisions": ("id", "revision"),
        "pulls": ("depot", "change"), "text_decisions": ("text", "decision"), "counts": ("facet", "name"),
        "tokenizers_revisions": ("id", "revision"), "vocabularies": ("recipe", "version"),
        "vocabulary_texts": ("recipe", "version", "position"), "library_moves": ("library",),
        "quotes": ("conversation", "version", "text"),
        "text_tags": ("tag", "key"), "text_licenses": ("license", "key"), titles.OWNERS: ("owner",),
        "taggings": ("text", "tag", "joined_change"), "snapshots": ("id",), "snapshot_sets": ("snapshot", "part", "owner"),
        "legacy_snapshot_of": ("dataset", "revision")}
# the tables stored in the order of their key alone (SQLite's WITHOUT ROWID): small rows read by their key
WITHOUT_ROWID = frozenset({"text_decisions", "counts", "vocabulary_texts", "library_moves", "text_tags",
                           "text_licenses", "quotes", "taggings"})
# the current texts by status, then length, then id (format 14)
LENGTH_INDEX = (f"CREATE INDEX texts_length ON texts_revisions (status, content_length, {ids.order_sql()}) "
                "WHERE latest")
INDEXES = (
    "CREATE UNIQUE INDEX texts_latest ON texts_revisions (id) WHERE latest",
    f"CREATE INDEX texts_order ON texts_revisions ({ids.order_sql()}) WHERE latest",
    f"CREATE INDEX texts_status ON texts_revisions (status, {ids.order_sql()}) WHERE latest",
    LENGTH_INDEX,
    "CREATE UNIQUE INDEX libraries_latest ON libraries_revisions (id) WHERE latest",
    "CREATE UNIQUE INDEX erasures_latest ON erasures_revisions (id) WHERE latest",
    "CREATE UNIQUE INDEX datasets_latest ON datasets_revisions (id) WHERE latest",
    "CREATE UNIQUE INDEX tokenizers_latest ON tokenizers_revisions (id) WHERE latest",
    "CREATE INDEX memberships_held ON memberships (text) WHERE left_change IS NULL",
    "CREATE INDEX pins_text ON pins (text)",
    "CREATE INDEX memberships_key ON memberships (library, export_key, joined_change, left_change, text)",
    "CREATE INDEX pins_key ON pins (library, export_key, text, revision)",
    "CREATE INDEX texts_change ON texts_revisions (change)",
    "CREATE INDEX notices_text ON notices (text)",
    f"CREATE INDEX text_decisions_order ON text_decisions (decision, {ids.order_sql('text')})",
    f"CREATE INDEX memberships_order ON memberships (library, {ids.order_sql('text')}) WHERE left_change IS NULL",
    f"CREATE INDEX pins_order ON pins (library, {ids.order_sql('text')})",
    "CREATE INDEX memberships_joined ON memberships (joined_change)",
    "CREATE INDEX memberships_left ON memberships (left_change) WHERE left_change IS NOT NULL",
    "CREATE INDEX content_lists_change ON content_lists (change)",
    "CREATE INDEX text_decisions_change ON text_decisions (change)",
    "CREATE INDEX pins_change ON pins (change)",
    f"CREATE INDEX content_lists_long ON content_lists (id, version) WHERE kind = '{PLAIN}' AND length > {LONG}",
    "CREATE INDEX quotes_text ON quotes (text)",
    "CREATE INDEX taggings_change ON taggings (change)",
    f"CREATE UNIQUE INDEX snapshots_name ON snapshots ({ids.owner_sql('id')}, name)",
)
# the tables whose rows are revisions of an id, with the flag latest; <name> is the view of their current rows
REVISED = ("texts_revisions", "libraries_revisions", "erasures_revisions", "datasets_revisions",
           "tokenizers_revisions")
# the columns that hold JSON text, by name, in any table or view
JSON_COLUMNS = frozenset({"licenses", "authors", "tags", "decisions", "scores", "libraries", "parts", "problems",
                          "needles", "erased", "shares", "figures", "origin", "summary", "sample"})
# the tables of libraries and datasets, which no write of the depot touches any more: they keep what a depot of an
# older format held, and only wpt_depot.legacy and the upgrade to format 13 (wpt_depot.conversion) read them
LEGACY = ("libraries_revisions", "memberships", "pins", "library_moves", "datasets_revisions")


def ddl() -> list[str]:
    """The statements that make every table, with its key, every index and the title index."""
    return [f"CREATE TABLE {name} ({columns}, PRIMARY KEY ({', '.join(KEYS[name])})){table_options(name)}"
            for name, columns in TABLES.items()] + list(INDEXES) + [titles.CREATE]


def table_options(name: str) -> str:
    """What follows the columns of the table's statement: WITHOUT ROWID for those of WITHOUT_ROWID."""
    return " WITHOUT ROWID" if name in WITHOUT_ROWID else ""


def tags_now(text: str) -> str:
    """The tags the text whose id is the SQL expression text has now, as a
    JSON array in the order of their names (the rows of taggings it has not
    left)."""
    return (f"(SELECT json_group_array(g.tag ORDER BY g.tag) FROM taggings g WHERE g.text = {text} "
            "AND g.left_change IS NULL)")


def tags_at(text: str, change: str) -> str:
    """The tags the text whose id is the SQL expression text had at the
    change number the SQL expression change gives, as a JSON array in the
    order of their names: the rows of taggings it had joined by then and not
    yet left."""
    return (f"(SELECT json_group_array(g.tag ORDER BY g.tag) FROM taggings g WHERE g.text = {text} "
            f"AND g.joined_change <= {change} AND (g.left_change IS NULL OR g.left_change > {change}))")


def decision_list(text: str, bound: str | None = None) -> str:
    """The ids of the decisions on the text whose id is the SQL expression
    text, as a JSON array in the order they were made (the numeric order of
    their ids, wpt_depot.ids): its rows of text_decisions, those made at a
    change number up to the SQL expression bound when it is given."""
    within = "" if bound is None else f" AND l.change <= {bound}"
    return (f"(SELECT json_group_array(l.decision ORDER BY {ids.order_sql('l.decision')}) FROM text_decisions l "
            f"WHERE l.text = {text}{within})")


def tags_sql(chosen: str | None = None) -> str:
    """The statement that copies into the column current_tags of the
    current revision of each text whose id the query chosen gives (its first
    column; every text for None) the tags it has now (tags_now), writing
    only the revisions whose copy differs."""
    within = "" if chosen is None else f"id IN ({chosen}) AND "
    now = tags_now("texts_revisions.id")
    return f"UPDATE texts_revisions SET current_tags = {now} WHERE {within}latest AND current_tags IS NOT {now}"


# the columns of a revision of texts that copy the kind, length and bytes of the content version it reads
SIZES = ("content_kind", "content_length", "content_bytes")


def sizes_sql(chosen: str | None = None) -> str:
    """The statement that copies into SIZES of each revision of the texts
    whose ids the table chosen holds (its column id; every revision for
    None) the kind, length and bytes of the content version it reads, from
    its row of content_lists (null for a version the depot lacks), writing
    only the revisions whose SIZES differ from them."""
    found = ("(SELECT c.kind, c.length, c.bytes FROM content_lists c WHERE c.id = texts_revisions.id "
             "AND c.version = texts_revisions.content_version)")
    within = "" if chosen is None else f"id IN (SELECT id FROM {chosen}) AND "
    return (f"UPDATE texts_revisions SET ({', '.join(SIZES)}) = {found} "
            f"WHERE {within}({', '.join(SIZES)}) IS NOT {found}")


# a content version's whole text: its parts joined in order, refused past SQL_STRING bytes
JOINED = (f"(CASE WHEN c.bytes > {SQL_STRING} THEN content_too_big(c.id, c.version, c.bytes) "
          "ELSE coalesce((SELECT group_concat(part_text(p.value), '' ORDER BY j.key) FROM json_each(c.parts) j "
          "JOIN parts p ON p.id = c.id AND p.hash = unhex(j.value ->> 'hash')), '') END)")
TEXT_COLUMNS = (f"t.id, t.revision, t.change, CASE WHEN c.kind = '{PLAIN}' THEN {{joined}} END AS text, c.kind, "
                f"CASE WHEN c.kind <> '{PLAIN}' THEN {{joined}} END AS content, t.content_version, c.length, t.licenses, "
                "t.notice, t.title, t.authors, {tags} AS tags, t.address, {decisions} AS decisions, t.scores, t.status"
                ).format(joined=JOINED, tags="t.current_tags", decisions=decision_list("t.id"))
# a text at one revision with the list of parts of the version it reads and its bytes, never its text, and its tags
# now (the revision may be a past one, whose copy of its tags is not read: Depot.listed_text)
LISTED_COLUMNS = ("t.id, t.revision, t.change, c.kind, t.content_version, c.length, t.licenses, t.notice, t.title, "
                  f"t.authors, {tags_now('t.id')} AS tags, t.address, t.scores, c.parts, c.bytes")
# how a revision of texts (t) meets the content version it reads (c)
CONTENT_JOIN = "JOIN content_lists c ON c.id = t.id AND c.version = t.content_version"
VIEWS = {
    "about": "SELECT * FROM about_revisions ORDER BY revision DESC LIMIT 1",
    "texts": f"SELECT {TEXT_COLUMNS} FROM texts_revisions t {CONTENT_JOIN} WHERE t.latest",
    "contents": (f"SELECT c.id, c.version, c.change, CASE WHEN c.kind = '{PLAIN}' THEN {JOINED} END AS text, c.kind, "
                 f"CASE WHEN c.kind <> '{PLAIN}' THEN {JOINED} END AS content, c.parts, c.length, c.bytes "
                 "FROM content_lists c"),
    "erasures": ("SELECT id, revision, change, reason, decisions, needles, erased FROM erasures_revisions "
                 "WHERE latest"),
    "tokenizers": ("SELECT id, revision, change, name, version, depot, dataset, sample, sample_bytes, vocab_size, "
                   "note FROM tokenizers_revisions WHERE latest"),
}
# the view tokenizers over a file of format 12, whose tokenizers_revisions has no column sample: what the frozen
# reader of the legacy tables (wpt_depot.legacy) and the upgrade read there (database.Database.connect)
TOKENIZERS_12 = VIEWS["tokenizers"].replace(" sample,", " NULL AS sample,")
