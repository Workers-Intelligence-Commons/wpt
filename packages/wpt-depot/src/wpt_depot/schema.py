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
whose `latest` is 1. A text's big part (its plain text, or an item's kind
and JSON content) is stored as content versions (1, 2, 3...), each a row of
content_lists with the change number it was made at: its kind, its length
in characters and in UTF-8 bytes, and the ordered list of its parts. Each
part is a row of parts, its text stored once for its text, compressed with
zstd, keyed by the text's id and the part's hash, with the change number
that first stored it (wpt_depot.parts says where a content is cut). Each
revision of a text holds only the small fields and the number of the
content version it reads (content_version). Which texts a library holds is
memberships: the change a text joined at and the change it left at (null
while it is in). A locked library also has pins: the exact revision of each
text it locked, with the change that locked it. A decision and a notice
record the change that wrote them. numbers holds, for each kind of id, the
last number of the depot's own ids, so a number is never given twice, even
once an erasure removed its row.

A list is JSON text: a text's licenses, authors, tags, decisions (ids, in
order) and scores, a content version's parts, a library's problems, an
erasure request's decisions, needles and erased passages, a dataset's
libraries and shares. A list of names is a JSON array of strings, so a tag
test is `EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'book')`; a
part is {"hash": its SHA-256 in hexadecimal, "length": its characters}; a
score is {"name", "value", "content_version"}, in the order of their names
(null: none); an erased passage is {"length", "hash_a", "hash_b" (both in
hexadecimal), "at", "needle"}; a problem is {"texts", "problem",
"decision"}; a needle is {"text", "field", "start", "end", "version"}. A
time is ISO 8601 text in UTC.

Every table has a declared primary key (KEYS): the sessions that record
each change (wpt_depot.database) need one to name a row.

A depot that follows an upstream (wpt_depot.follow) keeps the change
numbers of the rows it received, which are those of the depot that made
each row, so pulls records how the numberings meet: for each pull, and for
the clone itself, the depot that pulled, its change number then, its
upstream's name and the upstream's change number the pull brought it up to.
A clone receives its upstream's pulls rows too, so a clone of a clone can
follow the numbers back through each depot (Depot.bounds).

In SQL (wpt_depot.database), `<name>` is a view of the current state
(VIEWS): the current revision of each id, and for texts the content of the
version their current revision reads and the libraries holding them now
(`libraries`, a JSON array). `contents` shows every content version with
its whole text (`text`, or an item's `content`), its parts joined in order.
A value in SQLite holds at most SQL_STRING bytes, so a query that reads the
text of a content version bigger than that fails, naming it; its text is
read from its parts (wpt_depot.parts). A query that reads no text never
reads a part.

The layout above has a number, its format (FORMAT), which each about row
records: the depot's format when the row was written, so the current row
holds the depot's format now. The code reads and writes a depot of its own
FORMAT only (Depot.view). A depot of format 3 or older is a folder of Delta
Lake tables; `wpt depot upgrade` brings it up (wpt_depot.upgrade).
"""

FORMAT = 4  # the format of the layout below: what a depot this code reads and writes is written at

UNDER_REVIEW, ADMITTED, REFUSED = "under review", "admitted", "refused"
SQL_STRING = 1_000_000_000  # bytes: the longest value SQLite holds

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
        address TEXT, decisions TEXT NOT NULL, content_version INTEGER NOT NULL, scores TEXT""",
    "content_lists": """id TEXT NOT NULL, version INTEGER NOT NULL, change INTEGER NOT NULL, kind TEXT,
        parts TEXT NOT NULL, length INTEGER NOT NULL, bytes INTEGER NOT NULL""",
    "parts": "id TEXT NOT NULL, hash BLOB NOT NULL, change INTEGER NOT NULL, value BLOB NOT NULL",
    "libraries_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, name TEXT NOT NULL, forked_from TEXT, problems TEXT, locked_at TEXT""",
    "memberships": "library TEXT NOT NULL, text TEXT NOT NULL, joined_change INTEGER NOT NULL, left_change INTEGER",
    "pins": "library TEXT NOT NULL, text TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL",
    "erasures_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, reason TEXT NOT NULL, decisions TEXT NOT NULL, needles TEXT NOT NULL, erased TEXT""",
    "datasets_revisions": """id TEXT NOT NULL, revision INTEGER NOT NULL, change INTEGER NOT NULL,
        latest INTEGER NOT NULL, name TEXT NOT NULL, tokenizer TEXT NOT NULL, libraries TEXT NOT NULL, shares TEXT""",
    "pulls": "depot TEXT NOT NULL, change INTEGER NOT NULL, upstream TEXT NOT NULL, upstream_change INTEGER NOT NULL",
}
KEYS = {"about_revisions": ("revision",), "changes": ("id",), "numbers": ("name",), "licenses": ("id",),
        "notices": ("id",), "kinds": ("id",), "decisions": ("id",), "texts_revisions": ("id", "revision"),
        "content_lists": ("id", "version"), "parts": ("id", "hash"), "libraries_revisions": ("id", "revision"),
        "memberships": ("library", "text", "joined_change"), "pins": ("library", "text"),
        "erasures_revisions": ("id", "revision"), "datasets_revisions": ("id", "revision"),
        "pulls": ("depot", "change")}
INDEXES = (
    "CREATE UNIQUE INDEX texts_latest ON texts_revisions (id) WHERE latest",
    "CREATE UNIQUE INDEX libraries_latest ON libraries_revisions (id) WHERE latest",
    "CREATE UNIQUE INDEX erasures_latest ON erasures_revisions (id) WHERE latest",
    "CREATE UNIQUE INDEX datasets_latest ON datasets_revisions (id) WHERE latest",
    "CREATE INDEX memberships_held ON memberships (text) WHERE left_change IS NULL",
    "CREATE INDEX pins_text ON pins (text)",
    "CREATE INDEX texts_change ON texts_revisions (change)",
    "CREATE INDEX notices_text ON notices (text)",
)
# the tables whose rows are revisions of an id, with the flag latest; <name> is the view of their current rows
REVISED = ("texts_revisions", "libraries_revisions", "erasures_revisions", "datasets_revisions")
# the columns that hold JSON text, by name, in any table or view
JSON_COLUMNS = frozenset({"licenses", "authors", "tags", "decisions", "scores", "libraries", "parts", "problems",
                          "needles", "erased", "shares"})


def ddl() -> list[str]:
    """The statements that make every table, with its key, and every index."""
    return [f"CREATE TABLE {name} ({columns}, PRIMARY KEY ({', '.join(KEYS[name])}))"
            for name, columns in TABLES.items()] + list(INDEXES)


# a content version's whole text: its parts joined in order, refused past SQL_STRING bytes
JOINED = (f"(CASE WHEN c.bytes > {SQL_STRING} THEN content_too_big(c.id, c.version, c.bytes) "
          "ELSE coalesce((SELECT group_concat(part_text(p.value), '' ORDER BY j.key) FROM json_each(c.parts) j "
          "JOIN parts p ON p.id = c.id AND p.hash = unhex(j.value ->> 'hash')), '') END)")
TEXT_COLUMNS = ("t.id, t.revision, t.change, CASE WHEN c.kind IS NULL THEN {joined} END AS text, c.kind, "
                "CASE WHEN c.kind IS NOT NULL THEN {joined} END AS content, t.content_version, c.length, t.licenses, "
                "t.notice, t.title, t.authors, t.tags, t.address, t.decisions, t.scores").format(joined=JOINED)
# a text at one revision with the list of parts of the version it reads and its bytes, never its text
LISTED_COLUMNS = ("t.id, t.revision, t.change, c.kind, t.content_version, c.length, t.licenses, t.notice, t.title, "
                  "t.authors, t.tags, t.address, t.decisions, t.scores, c.parts, c.bytes")
# how a revision of texts (t) meets the content version it reads (c)
CONTENT_JOIN = "JOIN content_lists c ON c.id = t.id AND c.version = t.content_version"
# the libraries holding the text t now, as a JSON array in the order of their ids
HELD = ("(SELECT json_group_array(m.library ORDER BY m.library) FROM memberships m "
        "WHERE m.text = t.id AND m.left_change IS NULL)")
VIEWS = {
    "about": "SELECT * FROM about_revisions ORDER BY revision DESC LIMIT 1",
    "texts": f"SELECT {TEXT_COLUMNS}, {HELD} AS libraries FROM texts_revisions t {CONTENT_JOIN} WHERE t.latest",
    "contents": (f"SELECT c.id, c.version, c.change, CASE WHEN c.kind IS NULL THEN {JOINED} END AS text, c.kind, "
                 f"CASE WHEN c.kind IS NOT NULL THEN {JOINED} END AS content, c.parts, c.length, c.bytes "
                 "FROM content_lists c"),
    "libraries": ("SELECT id, revision, change, name, forked_from, problems, locked_at FROM libraries_revisions "
                  "WHERE latest"),
    "erasures": ("SELECT id, revision, change, reason, decisions, needles, erased FROM erasures_revisions "
                 "WHERE latest"),
    "datasets": ("SELECT id, revision, change, name, tokenizer, libraries, shares FROM datasets_revisions "
                 "WHERE latest"),
}
