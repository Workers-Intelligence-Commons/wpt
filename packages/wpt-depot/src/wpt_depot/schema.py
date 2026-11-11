"""The depot's tables as Arrow schemas, with the key of each, the CHECK
constraints Delta Lake enforces, and the views that show the current state.
Delta Lake checks only NOT NULL (a field that is not nullable) and CHECK;
the app enforces keys and references itself.

Every id but the journal's is text made of the depot's name, a hyphen and
a number (wpt_depot.ids), so ids never collide across depots.

Every write is one change of the depot, numbered by the depot-wide change
number: the journal's id of its entry. What changes in place keeps every
version as a row of its own table (REVISED), keyed by its id and its
revision (1, 2, 3...), and each row records the change number it was made
at; the highest revision is the current one. A text's big part (its plain
text, or an item's kind and JSON content) is stored in contents, one row per
content version (1, 2, 3...), each with the change number it was made at,
and each revision of a text holds only the small fields and the number of
the content version it reads (content_version). A text's scores (SCORE) are
pairs of a name and a number, in the order of their names, each with the
content version it was made from; null means none. Which texts a library holds is
memberships: the change a text joined at and the change it left at (null
while it is in). A locked library also has pins: the exact revision of each
text it locked.

A depot that follows an upstream (wpt_depot.follow) keeps the change
numbers of the rows it received, which are the upstream's, so pulls
records how the two numberings meet: for each pull, and for the clone
itself, the depot that pulled, its change number then, its upstream's
name and the upstream's change number the pull brought it up to. A
clone receives its upstream's pulls rows too, so a clone of a clone can
follow the numbers back through each depot (Depot.bounds).

In SQL (Tables.connection), each revised table's rows are
`<name>_revisions`, and `<name>` is a view of the current state (CURRENT):
the highest revision of each id, and for texts the content of the version
their current revision reads and the libraries holding them now
(`libraries`). `contents` shows every content version.

The layout above has a number, its format (FORMAT), which each about row
records: the depot's format when the row was written, so the current row
holds the depot's format now. A depot whose about table has no format
column is of format 0. The code reads and writes a depot of its own FORMAT
only (Depot.view); a backup and a restore copy an older one as it is
(wpt_depot.backup).

`wpt depot upgrade` (Depot.upgrade) brings an older depot up to FORMAT. It
first finishes every change its journal holds undone, with the code of
FORMAT, on the layout the depot has then. Then it runs, in order, the step
of STEPS from the depot's format to the next, and records each format
reached as a new revision of the about row, a change of the journal, so an
upgrade cut short resumes after the last step it recorded. The about row a
change of the journal writes takes the columns the about table has then.

Every change to this layout raises FORMAT and adds the step from the format
before, with its test. A step names the columns it reads and writes itself,
never from SCHEMAS, since SCHEMAS is the layout of a later format. A step
that adds a column to a table without writing its rows again lists the
table in WIDENED. It
changes only the tables its layout change touches, and run again on the
layout it leaves it changes nothing more. A layout change keeps the code of
FORMAT able to apply an older format's undone journal changes
(wpt_depot.journal) on that older layout. A step that changes the journal
table keeps backups and restores of older formats working: they read the
journal's id and done, the about table's change and every table's storage
versions."""

import pyarrow as pa

FORMAT = 2  # the format of the layout below: what a depot this code reads and writes is written at

TIME = pa.timestamp("us", tz="UTC")
NAMES = pa.list_(pa.string())
# a place a request names: its text, the content version of that text it counts in, and in that version a field
# (an item's), a start and an end, or none of them for the whole text
NEEDLE = pa.struct([("text", pa.string()), ("field", pa.string()), ("start", pa.int64()), ("end", pa.int64()),
                    pa.field("version", pa.int64(), nullable=False)])
# an erased passage: its length, its two public hashes (wpt_depot.erasure), its place once cut (at: the characters
# of its string before it) and the place in the request's needles of the first needle it covers
ERASED = pa.struct([("length", pa.int64()), ("hash_a", pa.binary()), ("hash_b", pa.binary()), ("at", pa.int64()),
                    ("needle", pa.int64())])
PROBLEM = pa.struct([("texts", NAMES), ("problem", pa.string()), ("decision", pa.string())])
# a library a dataset reads: its depot's address, its id and the change number of that depot it is read at
LIBRARY_REF = pa.struct([("depot", pa.string()), ("library", pa.string()), ("change", pa.int64())])
SHARE = pa.struct([("tag", pa.string()), ("share", pa.float64())])
# a score of a text: its name (any name), its value, and the content version of the text it was made from
SCORE = pa.struct([("name", pa.string()), ("value", pa.float64()), ("content_version", pa.int64())])
SCORES = pa.list_(SCORE)
REVISION = (("revision", pa.int64()), ("change", pa.int64()))


def _schema(*fields) -> pa.Schema:
    """fields: (name, type) for a NOT NULL column, (name, type, True) for one that may be NULL."""
    return pa.schema([pa.field(name, kind, nullable=bool(rest and rest[0])) for name, kind, *rest in fields])


SCHEMAS = {
    "about": _schema(*REVISION, ("name", pa.string()), ("maintainer", pa.string()),
                     ("takedown_contact", pa.string()), ("upstream", pa.string(), True), ("format", pa.int64())),
    "journal": _schema(("id", pa.int64()), ("change", pa.string()), ("done", pa.bool_())),
    "licenses": _schema(("id", pa.string()), ("name", pa.string()), ("text", pa.string())),
    "notices": _schema(("id", pa.string()), ("text", pa.string())),
    "kinds": _schema(("id", pa.string()), ("meaning", pa.string()), ("schema", pa.string())),
    "decisions": _schema(("id", pa.string()), ("status", pa.string()), ("name", pa.string()), ("email", pa.string()),
                         ("accepted", pa.bool_()), ("made_at", TIME), ("reason", pa.string(), True)),
    "texts": _schema(("id", pa.string()), *REVISION, ("licenses", NAMES), ("notice", pa.string(), True),
                     ("title", pa.string(), True), ("authors", NAMES, True), ("tags", NAMES),
                     ("address", pa.string(), True), ("decisions", NAMES), ("content_version", pa.int64()),
                     ("scores", SCORES, True)),
    "contents": _schema(("id", pa.string()), ("version", pa.int64()), ("change", pa.int64()),
                        ("text", pa.string(), True), ("kind", pa.string(), True), ("content", pa.string(), True)),
    "libraries": _schema(("id", pa.string()), *REVISION, ("name", pa.string()), ("forked_from", pa.string(), True),
                         ("problems", pa.list_(PROBLEM), True), ("locked_at", TIME, True)),
    "memberships": _schema(("library", pa.string()), ("text", pa.string()), ("joined_change", pa.int64()),
                           ("left_change", pa.int64(), True)),
    "pins": _schema(("library", pa.string()), ("text", pa.string()), ("revision", pa.int64())),
    "erasures": _schema(("id", pa.string()), *REVISION, ("reason", pa.string()), ("decisions", NAMES),
                        ("needles", pa.list_(NEEDLE)), ("erased", pa.list_(ERASED), True)),
    "datasets": _schema(("id", pa.string()), *REVISION, ("name", pa.string()), ("tokenizer", pa.string()),
                        ("libraries", pa.list_(LIBRARY_REF)), ("shares", pa.list_(SHARE), True)),
    "pulls": _schema(("depot", pa.string()), ("change", pa.int64()), ("upstream", pa.string()),
                     ("upstream_change", pa.int64())),
}
TABLES = tuple(SCHEMAS)
KEYS = {"about": ("revision",), "journal": ("id",), "licenses": ("id",), "notices": ("id",), "kinds": ("id",),
        "decisions": ("id",), "texts": ("id", "revision"), "contents": ("id", "version"), "libraries": ("id", "revision"),
        "memberships": ("library", "text", "joined_change"), "pins": ("library", "text"),
        "erasures": ("id", "revision"), "datasets": ("id", "revision"),
        "pulls": ("depot", "change")}
# the tables whose rows are revisions; in SQL their rows are <name>_revisions and <name> is the current state
REVISED = ("about", "texts", "libraries", "erasures", "datasets")
# the columns that hold a change number, by table: the depot's change number is past every one of them
CHANGES = {"about": ("change",), "texts": ("change",), "contents": ("change",), "libraries": ("change",), "erasures": ("change",),
           "datasets": ("change",), "memberships": ("joined_change", "left_change"), "pulls": ("change",)}

UNDER_REVIEW, ADMITTED, REFUSED = "under review", "admitted", "refused"
CONSTRAINTS = {
    "decisions": {"status_known": f"status IN ('{UNDER_REVIEW}', '{ADMITTED}', '{REFUSED}')",
                  "accepted_only": "accepted"},
    "contents": {"text_or_item": "(kind IS NULL AND text IS NOT NULL AND content IS NULL) "
                                 "OR (kind IS NOT NULL AND text IS NULL AND content IS NOT NULL)"},
}


def sql_name(table: str) -> str:
    """The name a table's rows have in SQL: <name>_revisions for a revised table, its own name for any other."""
    return f"{table}_revisions" if table in REVISED else table


def latest(table: str) -> str:
    """A query of the current revision of each id of a revised table (about: its one current row)."""
    if table == "about":
        return "SELECT * FROM about_revisions ORDER BY revision DESC LIMIT 1"
    return f"SELECT * FROM {table}_revisions QUALIFY row_number() OVER (PARTITION BY id ORDER BY revision DESC) = 1"


# a text at one revision, with the content of the version it reads
TEXT_COLUMNS = ("t.id, t.revision, t.change, c.text, c.kind, c.content, t.content_version, t.licenses, t.notice, "
                "t.title, t.authors, t.tags, t.address, t.decisions, t.scores")
# how a revision of texts (t) meets the content version it reads (c)
CONTENT_JOIN = "JOIN contents c ON c.id = t.id AND c.version = t.content_version"
CURRENT = {
    **{table: latest(table) for table in REVISED},
    "texts": (f"SELECT {TEXT_COLUMNS}, coalesce(m.libraries, []::VARCHAR[]) AS libraries FROM ({latest('texts')}) t "
              f"{CONTENT_JOIN} LEFT JOIN (SELECT text, list(library ORDER BY library) AS libraries "
              "FROM memberships WHERE left_change IS NULL GROUP BY text) m ON m.text = t.id"),
}


def rows(table: str, found: list[dict]) -> pa.Table:
    """found as a table of the depot's table's schema."""
    return pa.Table.from_pylist(found, schema=SCHEMAS[table])


# the about table at format 1: format 0's columns, then format
ABOUT_1 = _schema(*REVISION, ("name", pa.string()), ("maintainer", pa.string()), ("takedown_contact", pa.string()),
                  ("upstream", pa.string(), True), ("format", pa.int64()))


def _add_format(tables):
    """Format 0 to 1: the about table gets the column format, every row it
    holds of format 0, in one commit that rewrites it; nothing when it has
    the column already."""
    if "format" in tables.columns("about"):
        return
    held = [name for name in ABOUT_1.names if name != "format"]
    found = tables.query(f"SELECT {', '.join(held)} FROM about_revisions")
    tables.rewrite("about", found.append_column("format", pa.array([0] * found.num_rows, pa.int64())).cast(ABOUT_1))


# the column the texts table gets at format 2: its scores, null for none
SCORES_2 = pa.field("scores", pa.list_(pa.struct([("name", pa.string()), ("value", pa.float64()),
                                                   ("content_version", pa.int64())])), nullable=True)


def _add_scores(tables):
    """Format 1 to 2: the texts table gets the column scores, null in every
    row it holds, which means no scores. Only the table's log changes (one
    commit that adds the column): no row is read or written again, whatever
    the number of texts, since a read fills the column with null in the
    data files written before it (wpt_depot.tables). Nothing when the table
    has the column already."""
    if SCORES_2.name not in tables.columns("texts"):
        tables.add_column("texts", SCORES_2)


# the step from each format to the next, by the format it starts from: a function over the depot's tables
# (wpt_depot.tables.Tables) that changes them to the next format's layout, and changes nothing more when run again
STEPS = {0: _add_format, 1: _add_scores}
# the tables a step widened in place (Tables.add_column): their data files do not all hold the same columns
WIDENED = ("texts",)
