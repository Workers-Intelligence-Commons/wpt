"""Bringing a depot of an older format up to the code's format
(schema.FORMAT), the only way a depot's format changes (`wpt depot
upgrade`).

A depot of format 3 or older is a folder of Delta Lake tables
(wpt_depot.legacy). The upgrade first runs, in that folder, the step from
its format to the next until it reaches format 3 (legacy.STEPS), each
recorded as it was in those formats. Then it writes a new SQLite file of
format 4 from the folder, which it never changes again: by default the
folder's own path with `.sqlite` added, or the file given (`--to`). It
refuses a folder whose journal holds a change never finished, since only
the code of its format knew how to finish it.

An SQLite file of format 4 or newer, the one given or the one just written
from a folder, is brought up in place by the step from its format to the
next (FILE_STEPS) until it reaches the code's format. A step changes the
layout and adds a revision of the about row at its new format under a
change of its own, kind 'upgrade'. A step that fits one transaction leaves
the file as it was when it is cut short, and runs again whole. A step that
rewrites every current text runs in several transactions: its first one
makes the table UNDER_WAY, which marks the depot as half upgraded, with
the new layout and the about revision; each next one handles one batch of
rows and records in UNDER_WAY where it stopped; the last one drops it. A
half-upgraded depot is at no format the code reads: every way in but the
upgrade refuses it (errors.HalfUpgraded), and the upgrade run again goes
on from the batch after the last one written, saying how far it is every
SAY_EVERY seconds. Before a step, the file
is marked to send a full copy at its next backup, under a new timeline
(wpt_depot.database), since the change files record no change of layout.
The step to format 5 builds the index texts_order (wpt_depot.schema) in
one statement, which reads every current text's row once and rewrites no
table; it says what it does before it starts and how long it took. The
step to format 6 makes, in its first transaction, the tables of the
tokenizer recipes, their vocabularies and the texts each vocabulary trained
on, and the column max_tokens of the datasets, all empty; then it writes
each current text's decisions as rows of text_decisions and its status,
BATCH rows of texts_revisions a transaction, then builds the index
texts_status and fills the counts of format 6 (COUNTS_6). The step to
format 7 adds, in its first transaction, the column export_key of
memberships and pins and the column vocabulary of the datasets, empty;
then it writes the export key (wpt_depot.ids.export_key) of every row of
those two tables, in that order (KEYED_7: every membership, then every
pin, by the text it names), BATCH rows a transaction, read and written by their rowid with
memory bounded by the batch, saying how many rows a second it writes; the
table upgrade_under_way keeps, for each table, the last rowid written, so a
run after a stop goes on in the table and at the row it reached; then it
builds the indexes memberships_key and pins_key, each in one
statement. The step to format 8 builds the index text_decisions_decision
(wpt_depot.schema) in one statement, which reads every row of
text_decisions once and rewrites no table, between a first transaction
that marks the depot half upgraded and a last one that drops the mark, so
a stop while it builds leaves a depot that only the upgrade opens; it says
what it does before it starts and how long it took. Format 8 is also the
format of a depot that may hold decisions that put texts back under review
(wpt_depot.policy), which the code of format 7 does not know: that code
refuses a depot of format 8. The step to format 9 marks the depot half
upgraded and adds the columns characters and bytes of counts, the column
figures of the dataset revisions and the table library_moves (MOVES_9) in
its first transaction; then it fills the counts of what each library holds by kind
and by tag (LIBRARY_KIND_9, LIBRARY_TAG_9): the memberships that have not
ended, the current revisions and the content versions are each read in
one sequential scan (SCANS_9), sorted by text id by SQLite in its
temporary folder, and merged (_merged_9), so the time grows with the rows
whatever order the file stores them in and no row is looked up through an
index. SQLite sorts the three scans before the first row comes back, so
its temporary folder (SQLITE_TMPDIR) holds the three sorts at once, about
SORTED_9 bytes a membership, which the step says before it starts. Every
BATCH memberships or so, at the end of a text, one transaction adds their
sums to counts, raises the change of library_moves of each library to
the latest change a text it holds among them was revised at, and records
that text's id in upgrade_under_way; a run after a stop scans only the
texts after it. It says how many memberships a second it counts every
SAY_EVERY seconds. Its last transaction raises library_moves to the latest
change each library's memberships joined or left at, read from the index
memberships_key alone (JOINED_OR_LEFT_9), or, in a depot that follows an
upstream, whose rows carry several numberings, makes every change of it
null; then it gives each dataset revision its figures (_figures_9), read
from the counts it filled, for each library they prove exact and null for
the others, and drops the mark. A library whose memberships and texts no
change touched after the change a revision reads it at holds what it held
then, so only those get figures, and an upgraded revision never holds
wrong ones. The step to format 10 marks the depot half upgraded and makes
the tables text_tags and text_licenses (TABLES_10) in its first
transaction; then it lists the current texts by tag and by license and
counts their score names and the splits of every text and of each tag,
license and library by status and by whether a locked library holds them
(LISTED_10, SCORES_10, SPLITS_10), reading the current
revisions in rowid order, BATCH rowids a transaction, each recording the
last rowid in upgrade_under_way, so a run after a stop goes on after it; it
says how many rows a second it reads every SAY_EVERY seconds. Then it
builds each index of INDEXES_10 in one statement, skipping one that
exists, so a run after a stop builds only those missing. Its last
transaction drops the index text_decisions_decision, which
text_decisions_order serves in its place (DROPPED_10), counts the pins
older than their text's current revision (NEWER_COUNTS_10) and drops the
mark.

A clone of those formats named its upstream by the upstream's folder
or bucket address; a clone of format 4 or newer follows its upstream over the
upstream's web app (wpt_depot.follow), so upgrading a clone takes that web
address (`--upstream`), which its about row then names, and is refused
without it.

The new file is written as `<file>.upgrading` and renamed once whole, so an
upgrade cut short (a crash, a full disk) leaves no depot at the file's
path. Run again, it goes on: each data file of each table goes into the
new file in one transaction that also records it in the table
upgrade_progress, so a data file is read once and a restart picks up after
the last one written. The last transaction makes the upgrade's change and
drops upgrade_progress; a file that holds that change (an about revision
of format 4) is whole, and a restart only renames it. A table's rows are read a batch at a time, so the
memory it takes does not grow with the depot. Lists become JSON text,
times ISO text, a part's text is compressed (wpt_depot.parts), and the
format 4 columns the folder lacks are made from it: a revision is the
current one when no higher revision of its id exists, a decision or a
notice records the journal entry that first named it (the upgrade's change
when none did), a pin the change that locked its library, each journal
entry becomes a row of changes (made at its decision's time, when it has
one), and numbers holds the highest number of each kind of the depot's own
ids, erasures' needles included. The upgrade itself is the last change, a
new revision of the about row at format 4. The new file's next backup
sends a full copy (wpt_depot.backup).
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from wpt_utils.lines import say

from . import ids, policy
from .database import Database, now
from .errors import DepotExists, OtherFormat, Refused
from .follow import Upstream
from .legacy import LAST, STEPS, Legacy
from .parts import pack
from .schema import FORMAT

UPGRADING = ".upgrading"  # what the name of the file being written adds to the new file's name
ROWS = 4096  # rows of a data file read at a time
PARTS_ROWS = 16  # rows of parts read at a time: each holds up to 64 MiB of text
WRITTEN = 4  # the format of the SQLite file written from a folder
# the index of format 5: the current texts in the numeric order of their ids (ids.order_sql)
ORDER = ("((CASE WHEN (rtrim(id, '0123456789') <> id "
               "AND rtrim(id, '0123456789') GLOB '*-') THEN substr(id, 1, length(rtrim(id, '0123456789')) - 1) ELSE "
               "id END) || char(1) || CASE WHEN (CASE WHEN (rtrim(id, '0123456789') <> id AND rtrim(id, '0123456789') "
               "GLOB '*-') THEN CAST(substr(id, length(rtrim(id, '0123456789')) + 1) AS INTEGER) END) IS NULL THEN "
               "'1' ELSE '0' || printf('%020d', (CASE WHEN (rtrim(id, '0123456789') <> id AND rtrim(id, "
               "'0123456789') GLOB '*-') THEN CAST(substr(id, length(rtrim(id, '0123456789')) + 1) AS INTEGER) "
               "END)) END || char(1) || id)")
TEXTS_ORDER = f"CREATE INDEX IF NOT EXISTS texts_order ON texts_revisions ({ORDER}) WHERE latest"
# format 6: the index of the current texts by status, then in the order of their ids, and its tables (a file
# written from a folder is made with the code's tables, so they may exist already, the column status too)
TEXTS_STATUS = f"CREATE INDEX IF NOT EXISTS texts_status ON texts_revisions (status, {ORDER}) WHERE latest"
TABLES_6 = ("CREATE TABLE IF NOT EXISTS text_decisions (text TEXT NOT NULL, decision TEXT NOT NULL, "
            "change INTEGER NOT NULL, PRIMARY KEY (text, decision)) WITHOUT ROWID",
            "CREATE TABLE IF NOT EXISTS counts (facet TEXT NOT NULL, name TEXT NOT NULL, texts INTEGER NOT NULL, "
            "PRIMARY KEY (facet, name)) WITHOUT ROWID",
            "CREATE TABLE IF NOT EXISTS tokenizers_revisions (id TEXT NOT NULL, revision INTEGER NOT NULL, "
            "change INTEGER NOT NULL, latest INTEGER NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL, depot TEXT, "
            "dataset TEXT NOT NULL, sample_bytes INTEGER NOT NULL, vocab_size INTEGER NOT NULL, note TEXT, "
            "PRIMARY KEY (id, revision))",
            "CREATE UNIQUE INDEX IF NOT EXISTS tokenizers_latest ON tokenizers_revisions (id) WHERE latest",
            "CREATE TABLE IF NOT EXISTS vocabularies (recipe TEXT NOT NULL, version INTEGER NOT NULL, "
            "change INTEGER NOT NULL, sha256 TEXT NOT NULL, merges TEXT NOT NULL, texts INTEGER NOT NULL, "
            "bytes INTEGER, made_at TEXT NOT NULL, PRIMARY KEY (recipe, version))",
            "CREATE TABLE IF NOT EXISTS vocabulary_texts (recipe TEXT NOT NULL, version INTEGER NOT NULL, "
            "position INTEGER NOT NULL, depot TEXT NOT NULL, text TEXT NOT NULL, change INTEGER NOT NULL, "
            "PRIMARY KEY (recipe, version, position)) WITHOUT ROWID")
# format 6: each facet of counts, with the query of its (name, texts) rows over the whole depot
COUNTS_6 = {
    "texts": "SELECT '', count(*) FROM texts_revisions WHERE latest",
    "status": "SELECT status, count(*) FROM texts_revisions WHERE latest GROUP BY status",
    "tag": "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.tags) j WHERE t.latest GROUP BY j.value",
    "license": "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.licenses) j WHERE t.latest "
               "GROUP BY j.value",
    "library": "SELECT library, count(*) FROM memberships WHERE left_change IS NULL GROUP BY library",
    "decision": "SELECT decision, count(*) FROM text_decisions GROUP BY decision",
}
# format 6 columns added to tables of format 5: (table, column, its declaration)
COLUMNS_6 = (("texts_revisions", "status", "TEXT"), ("datasets_revisions", "max_tokens", "INTEGER"))
# format 7 columns added to tables of format 6: (table, column, its declaration)
COLUMNS_7 = (("memberships", "export_key", "INTEGER"), ("pins", "export_key", "INTEGER"), ("datasets_revisions", "vocabulary", "TEXT"))
# format 7: the tables whose rows get their text's export key, in the order they are filled, and the column naming it
KEYED_7 = (("memberships", "text"), ("pins", "text"))
INDEXES_7 = (("memberships_key", "CREATE INDEX IF NOT EXISTS memberships_key ON "
                                 "memberships (library, export_key, joined_change, left_change, text)"),
             ("pins_key", "CREATE INDEX IF NOT EXISTS pins_key ON pins (library, export_key, text, revision)"))
# format 8: the index of text_decisions by decision
TEXT_DECISIONS_DECISION = "CREATE INDEX IF NOT EXISTS text_decisions_decision ON text_decisions (decision)"
# format 9: the columns added to counts, the facets of its new rows, the kinds of item a corpus takes, and the three
# scans the counts are filled from, each in the order of the texts' ids, after the text `{after}` names when given (+
# keeps SQLite from reading an index for that order, and NOT INDEXED from reading the rows through one)
COLUMNS_9 = ("characters", "bytes")
LIBRARY_KIND_9, LIBRARY_TAG_9 = "library kind", "library tag"
FACETS_9 = f"('{LIBRARY_KIND_9}', '{LIBRARY_TAG_9}')"
ITEMS_9 = ("conversation", "task")
SCANS_9 = ("SELECT text, library FROM memberships NOT INDEXED WHERE left_change IS NULL{after} ORDER BY +text",
           "SELECT id, content_version, tags, change FROM texts_revisions NOT INDEXED WHERE latest{after} ORDER BY +id",
           "SELECT id, version, kind, length, bytes FROM content_lists NOT INDEXED WHERE 1{after} ORDER BY +id, +version")
SCANNED_9 = ("text", "id", "id")  # the column of each scan of SCANS_9 that names the text
SORTED_9 = 100  # bytes of temporary files the three sorts of SCANS_9 take for each membership, about
# format 9: the table library_moves, and the latest change each library's memberships joined or left at, read from the
# index memberships_key alone
MOVES_9 = ("CREATE TABLE IF NOT EXISTS library_moves (library TEXT NOT NULL, change INTEGER, PRIMARY KEY (library)) "
           "WITHOUT ROWID")
JOINED_OR_LEFT_9 = ("SELECT library, max(max(joined_change, coalesce(left_change, 0))) FROM memberships "
                    "INDEXED BY memberships_key GROUP BY library")
KINDS_9 = ("plain", "items", "other")
# format 10: the listings of the current texts by tag and by license, each row a name and the text's place in the
# numeric order of ids (ORDER over the column text, ORDER_TEXT_10), the facets of counts it adds, and the indexes it
# builds, by name, in the order they are built
ORDER_TEXT_10 = ORDER.replace("id", "text")
TABLES_10 = ("CREATE TABLE IF NOT EXISTS text_tags (tag TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY (tag, key)) "
             "WITHOUT ROWID",
             "CREATE TABLE IF NOT EXISTS text_licenses (license TEXT NOT NULL, key TEXT NOT NULL, "
             "PRIMARY KEY (license, key)) WITHOUT ROWID")
SCORE_10, NEWER_10 = "score", "pinned newer"
# what one batch of current revisions, the rowids from :low (excluded) to :high, adds: the rows of each listing, and
# the count of each score name (NOT INDEXED: the rows are read in rowid order, from the table itself)
LISTED_10 = (("text_tags", "tag", "tags"), ("text_licenses", "license", "licenses"))
BATCH_ROWS_10 = "FROM texts_revisions t NOT INDEXED WHERE t.latest AND t.rowid > :low AND t.rowid <= :high"
SCORES_10 = f"SELECT s.value ->> 'name', count(*) {BATCH_ROWS_10.replace('WHERE', ', json_each(t.scores) s WHERE')} GROUP BY 1"
# the facets that split every text and the texts of each tag, license and library by status and by whether a locked
# library holds them, each row named `<status> <locked> <name>` (locked 1 or 0): what one batch adds to each, the
# locked test (LOCKED_10) read from the text's side through the index memberships_held
LOCKED_10 = ("EXISTS (SELECT 1 FROM memberships lm INDEXED BY memberships_held CROSS JOIN libraries_revisions ll "
             "ON ll.id = lm.library AND ll.latest AND ll.locked_at IS NOT NULL WHERE lm.text = t.id "
             "AND lm.left_change IS NULL)")
SPLIT_NAME_10 = f"t.status || ' ' || {LOCKED_10} || ' ' || "
SPLITS_10 = (
    ("texts split", f"SELECT {SPLIT_NAME_10}'', count(*) {BATCH_ROWS_10} GROUP BY 1"),
    ("tag split", f"SELECT {SPLIT_NAME_10}j.value, count(*) "
                  f"{BATCH_ROWS_10.replace('WHERE', ', json_each(t.tags) j WHERE')} GROUP BY 1"),
    ("license split", f"SELECT {SPLIT_NAME_10}j.value, count(*) "
                      f"{BATCH_ROWS_10.replace('WHERE', ', json_each(t.licenses) j WHERE')} GROUP BY 1"),
    ("library split", f"SELECT {SPLIT_NAME_10}m.library, count(*) "
                      + BATCH_ROWS_10.replace("WHERE", "CROSS JOIN memberships m INDEXED BY memberships_held "
                                                       "ON m.text = t.id AND m.left_change IS NULL WHERE")
                      + " GROUP BY 1"))
NEWER_COUNTS_10 = (f"INSERT INTO counts (facet, name, texts) SELECT '{NEWER_10}', p.library, count(*) FROM pins p "
                   "JOIN texts_revisions t ON t.id = p.text AND t.latest WHERE t.revision > p.revision "
                   "GROUP BY p.library")
INDEXES_10 = (
    ("memberships_order", f"CREATE INDEX IF NOT EXISTS memberships_order ON memberships (library, {ORDER_TEXT_10}) "
                          "WHERE left_change IS NULL"),
    ("pins_order", f"CREATE INDEX IF NOT EXISTS pins_order ON pins (library, {ORDER_TEXT_10})"),
    ("memberships_joined", "CREATE INDEX IF NOT EXISTS memberships_joined ON memberships (joined_change)"),
    ("memberships_left", "CREATE INDEX IF NOT EXISTS memberships_left ON memberships (left_change) "
                         "WHERE left_change IS NOT NULL"),
    ("content_lists_change", "CREATE INDEX IF NOT EXISTS content_lists_change ON content_lists (change)"),
    ("text_decisions_change", "CREATE INDEX IF NOT EXISTS text_decisions_change ON text_decisions (change)"),
    ("pins_change", "CREATE INDEX IF NOT EXISTS pins_change ON pins (change)"),
    ("text_decisions_order", f"CREATE INDEX IF NOT EXISTS text_decisions_order ON text_decisions (decision, "
                             f"{ORDER_TEXT_10})"))
# the index of format 9 that text_decisions_order serves in its place, dropped once that one is built
DROPPED_10 = "DROP INDEX IF EXISTS text_decisions_decision"
UNDER_WAY = "upgrade_under_way"  # the table that marks a depot whose upgrade has not ended, and how far it went
BATCH = 250_000  # rows a transaction of the upgrades to formats 6 (texts_revisions), 7 (memberships, pins) and 9
                 # (memberships) goes over
SAY_EVERY = 30  # seconds between two lines of progress
# each format 3 table, by its folder's name: the table of format 4 its rows go into
TARGETS = {"about": "about_revisions", "licenses": "licenses", "notices": "notices", "kinds": "kinds",
           "decisions": "decisions", "texts": "texts_revisions", "contents": "content_lists", "parts": "parts",
           "libraries": "libraries_revisions", "memberships": "memberships", "pins": "pins",
           "erasures": "erasures_revisions", "datasets": "datasets_revisions", "pulls": "pulls"}
LISTS = {"licenses", "authors", "tags", "decisions", "scores", "problems", "needles", "shares", "libraries"}
_ID = re.compile(r'"([A-Za-z0-9][A-Za-z0-9._-]*-[0-9]+)"')


def upgrade(address: str, to: str | None = None, upstream: str | None = None, log=say):
    """(the depot at address brought up to this code's format, the format it
    had). A depot of the code's format is left as it is. upstream: the web
    address of the upstream's web app, for a clone. log: where a step says
    what it does. OtherFormat for a newer format, which no code brings down;
    Refused for a folder whose journal holds a change never finished, for a
    clone without upstream and for upstream given to a depot that follows
    none; DepotExists when the new file's path holds a file already."""
    from .depot import Depot, format_of, location, other_format
    path = location(address)
    start = format_of(address)
    if start > FORMAT:
        raise OtherFormat(other_format(str(path), start))
    if not path.is_dir():
        if upstream is not None:
            raise Refused(f"{path} is a depot file already: --upstream is for the upgrade of a clone of format 3 "
                          "or older")
        db = Database(path)
        try:
            start = _format(db)  # a depot whose upgrade has not ended: the format it started from
        finally:
            db.close()
    else:
        path = _from_folder(path, to, upstream)
    _in_place(path, log)
    return Depot.open(str(path)), start


def _from_folder(path: Path, to: str | None, upstream: str | None) -> Path:
    """The SQLite file of format 4 written from the folder at path, as the module says."""
    from .depot import location
    legacy = Legacy(path)
    undone = legacy.undone()
    if undone:
        raise Refused(f"{path}: its journal holds {len(undone)} changes never finished (the first: {undone[0]}); "
                      "finish them with the wpt of the depot's format, by opening the depot once, then upgrade")
    followed = max(legacy.read("about").to_pylist(), key=lambda one: one["revision"]).get("upstream")
    if followed is None and upstream is not None:
        raise Refused(f"{path} follows no upstream: --upstream is for the upgrade of a clone")
    if followed is not None:
        if upstream is None:
            raise Refused(f"{path} is a clone of {followed}: a clone of format {FORMAT} follows its upstream over "
                          "the upstream's web app, so give its web address (--upstream https://...)")
        upstream = Upstream(upstream).url
    target = location(to) if to is not None else path.with_name(path.name + ".sqlite")
    if target.exists():
        raise DepotExists(f"{target} already exists: the upgrade writes a new file")
    while (reached := legacy.format()) < LAST:
        STEPS[reached](legacy)
        legacy.record_format(reached + 1)
    _write(legacy, target, upstream)
    return target


def _in_place(path: Path, log):
    """The SQLite file at path brought up to the code's format, a step at a time, as the module says."""
    db = Database(path)
    try:
        while (reached := _format(db)) < FORMAT:
            db.settle()
            db.mark_full_copy()
            FILE_STEPS[reached](db, log)
    finally:
        db.close()


def _format(db: Database) -> int:
    """The depot's format, or for a depot under an upgrade that has not
    ended, the format that upgrade starts from."""
    from .depot import half_upgraded
    con = db.reader()
    found = con.execute("SELECT format FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]
    return found - 1 if half_upgraded(con) else found


def _to_5(db: Database, log):
    """Format 4 to 5: the index texts_order."""
    texts = db.reader().execute("SELECT count(*) FROM texts_revisions WHERE latest").fetchall()[0][0]
    log(f"{db.path}: building the index texts_order over its {texts} current texts, in one statement that reads "
        "each of their rows once and writes no row of a table")
    began = time.monotonic()
    with db.bulk() as con:
        con.execute(TEXTS_ORDER)
        _new_format(con, 5)
    log(f"{db.path}: the index texts_order is built, in {time.monotonic() - began:.0f} seconds; the depot is at "
        "format 5")


def _new_format(con, reached: int):
    """The upgrade's own change: a new revision of the about row at the format reached."""
    number = con.execute("SELECT coalesce(max(id), 0) + 1 FROM changes").fetchall()[0][0]
    con.execute("INSERT INTO changes (id, kind, made_at) VALUES (?, 'upgrade', ?)", (number, now()))
    con.execute("INSERT INTO about_revisions (revision, change, name, maintainer, takedown_contact, upstream, format) "
                "SELECT revision + 1, ?, name, maintainer, takedown_contact, upstream, ? FROM about_revisions "
                "ORDER BY revision DESC LIMIT 1", (number, reached))


def _to_6(db: Database, log):
    """Format 5 to 6, in several transactions, as the module says: each
    text's decisions as rows of text_decisions, its stored status, the
    index texts_status and the counts."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('texts', 0)")
            for statement in TABLES_6:
                con.execute(statement)
            for table, column, declared in COLUMNS_6:
                if column not in {one[1] for one in con.execute(f"PRAGMA table_info({table})")}:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declared}")
            _new_format(con, 6)
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'texts'").fetchall()[0][0]
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    log(f"{db.path}: writing the decisions and the status of each current text, rows {after} to {last} of "
        f"texts_revisions, {BATCH} rows a transaction; run again after a stop, it goes on from the last one")
    began = said = time.monotonic()
    while after < last:
        upto = min(after + BATCH, last)
        with db.bulk() as con:
            con.execute("INSERT OR IGNORE INTO text_decisions (text, decision, change) "
                        "SELECT t.id, j.value, d.change FROM texts_revisions t, json_each(t.decisions) j "
                        "JOIN decisions d ON d.id = j.value WHERE t.latest AND t.rowid > ? AND t.rowid <= ?",
                        (after, upto))
            con.execute(f"UPDATE texts_revisions SET status = {policy.status_sql('decisions')}, decisions = '[]' "
                        "WHERE latest AND rowid > ? AND rowid <= ?", (after, upto))
            con.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'texts'", (upto,))
        after = upto
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: {after} of {last} rows of texts_revisions, {said - began:.0f} seconds")
    log(f"{db.path}: building the index texts_status over the current texts, in one statement")
    with db.bulk() as con:
        con.execute(TEXTS_STATUS)
    log(f"{db.path}: counting the texts of each status, tag, license, library and decision")
    with db.bulk() as con:
        _recount_6(con)
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the depot is at format 6, in {time.monotonic() - began:.0f} seconds")


def _recount_6(con):
    """The counts of format 6 made from the depot's rows (COUNTS_6)."""
    con.execute("DELETE FROM counts")
    for facet, query in COUNTS_6.items():
        con.execute(f"INSERT INTO counts (facet, name, texts) SELECT '{facet}', * FROM ({query})")
    con.execute("DELETE FROM counts WHERE texts = 0")


def _to_7(db: Database, log):
    """Format 6 to 7, in several transactions, as the module says: the
    export key of every row of memberships and pins, then their indexes."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.executemany(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES (?, 0)", [(t,) for t, _ in KEYED_7])
            for table, column, declared in COLUMNS_7:
                if column not in {one[1] for one in con.execute(f"PRAGMA table_info({table})")}:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declared}")
            _new_format(con, 7)
    began = time.monotonic()
    for table, text in KEYED_7:
        _keys(db, table, text, log)
    for name, statement in INDEXES_7:
        log(f"{db.path}: building the index {name}, in one statement")
        indexed = time.monotonic()
        with db.bulk() as con:
            con.execute(statement)
        log(f"{db.path}: the index {name} is built, in {time.monotonic() - indexed:.0f} seconds")
    with db.bulk() as con:
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the depot is at format 7, in {time.monotonic() - began:.0f} seconds")


def _to_8(db: Database, log):
    """Format 7 to 8, as the module says: the index text_decisions_decision,
    built while the depot is marked half upgraded."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            _new_format(con, 8)
    # the count kept in counts, since counting the table itself reads every row as the build does
    rows = db.reader().execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'decision'").fetchall()[0][0]
    log(f"{db.path}: building the index text_decisions_decision over its {rows} rows of text_decisions, in one "
        "statement that reads each row once and writes no row of a table; run again after a stop, it builds it again")
    began = time.monotonic()
    with db.bulk() as con:
        con.execute(TEXT_DECISIONS_DECISION)
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the index text_decisions_decision is built, in {time.monotonic() - began:.0f} seconds; the "
        "depot is at format 8")


def _keys(db: Database, table: str, text: str, log):
    """Write the export key of each row of table, from the text its column
    text names, BATCH rows a transaction from the rowid upgrade_under_way
    keeps for it, which each transaction moves on."""
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = ?", (table,)).fetchall()[0][0]
    last = con.execute(f"SELECT coalesce(max(rowid), 0) FROM {table}").fetchall()[0][0]
    if after >= last:
        return
    log(f"{db.path}: writing the export key of each row of {table}, rows {after} to {last}, {BATCH} rows a "
        "transaction; run again after a stop, it goes on from the last one")
    began = said = time.monotonic()
    written = 0
    while after < last:
        upto = min(after + BATCH, last)
        found = db.reader().execute(f"SELECT rowid, {text} FROM {table} WHERE rowid > ? AND rowid <= ?",
                                    (after, upto)).fetchall()
        with db.bulk() as con:
            con.executemany(f"UPDATE {table} SET export_key = ? WHERE rowid = ?",
                            [(ids.export_key(one), rowid) for rowid, one in found])
            con.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = ?", (upto, table))
        after, written = upto, written + len(found)
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: row {after} of {last} of {table}, {written / max(said - began, 1e-9):,.0f} rows "
                f"a second, {said - began:.0f} seconds")


def _to_9(db: Database, log):
    """Format 8 to 9, in several transactions, as the module says: the
    columns characters and bytes of counts, and the counts of what each
    library holds by kind and by tag, from one sorted merge of three
    sequential scans (_merged_9)."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after TEXT)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('memberships', NULL)")
            held = {one[1] for one in con.execute("PRAGMA table_info(counts)")}
            for column in COLUMNS_9:
                if column not in held:
                    con.execute(f"ALTER TABLE counts ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            if "figures" not in {one[1] for one in con.execute("PRAGMA table_info(datasets_revisions)")}:
                con.execute("ALTER TABLE datasets_revisions ADD COLUMN figures TEXT")
            con.execute(MOVES_9)
            con.execute("DELETE FROM library_moves")
            con.execute(f"DELETE FROM counts WHERE facet IN {FACETS_9}")
            _new_format(con, 9)
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'memberships'").fetchall()[0][0]
    total = con.execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'library'").fetchall()[0][0]
    log(f"{db.path}: counting what each library holds by kind and by tag, in characters and bytes: its {total:,} "
        "memberships that have not ended, the current revisions and the content versions, each read in one "
        "sequential scan sorted by text id, then merged, the sums written about every "
        f"{BATCH:,} memberships. SQLite sorts the three scans at once in its temporary folder (SQLITE_TMPDIR, "
        f"else /var/tmp, /usr/tmp or /tmp), which needs about {total * SORTED_9 / 1e9:,.1f} GB of free disk"
        + ("" if after is None else f"; going on after the text {after}, where the last run stopped"))
    began = said = time.monotonic()
    params = () if after is None else (after,)
    read = 0
    for sums, last, revised in _merged_9(*(con.execute(scan.format(after="" if after is None else f" AND {column} > ?"),
                                                       params) for scan, column in zip(SCANS_9, SCANNED_9))):
        with db.bulk() as written:
            written.executemany("INSERT INTO counts (facet, name, texts, characters, bytes) VALUES (?, ?, ?, ?, ?) "
                                "ON CONFLICT (facet, name) DO UPDATE SET texts = texts + excluded.texts, "
                                "characters = characters + excluded.characters, bytes = bytes + excluded.bytes",
                                [(facet, name, *by) for (facet, name), by in sums.items()])
            _moved_9(written, revised.items())
            written.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'memberships'", (last,))
        read += sum(by[0] for (facet, _), by in sums.items() if facet == LIBRARY_KIND_9)
        if time.monotonic() - said >= SAY_EVERY:
            said = time.monotonic()
            log(f"{db.path}: {read:,} memberships counted of {total:,}, the last text {last}, "
                f"{read / max(said - began, 1e-9):,.0f} rows a second, {said - began:.0f} seconds")
    counted = time.monotonic()
    log(f"{db.path}: {read:,} memberships counted, {read / max(counted - began, 1e-9):,.0f} rows a second, in "
        f"{counted - began:.0f} seconds; reading the latest change each library's memberships joined or left at "
        "from the index memberships_key, and the figures of each dataset revision the counts prove exact")
    with db.bulk() as written:
        _moved_9(written, written.execute(JOINED_OR_LEFT_9).fetchall())
        if written.execute("SELECT upstream FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]:
            written.execute("UPDATE library_moves SET change = NULL")  # its rows carry several numberings
        figured = _figures_9(written)
        written.execute(f"DROP TABLE {UNDER_WAY}")
    took = time.monotonic() - began
    log(f"{db.path}: {figured[0]:,} of {figured[1]:,} libraries read by a dataset revision have figures, in "
        f"{time.monotonic() - counted:.0f} seconds; the depot is at format 9, in {took:.0f} seconds")


def _moved_9(con, found):
    """library_moves raised to each (library, change) of found where it is lower, or made."""
    con.executemany("INSERT INTO library_moves (library, change) VALUES (?, ?) ON CONFLICT (library) "
                    "DO UPDATE SET change = max(change, excluded.change)", list(found))


def _figures_9(con) -> tuple[int, int]:
    """The figures of every dataset revision (the layout of format 9, as
    wpt_depot.schema says), read at the upgrade's change from the counts
    of format 9: those of a library this depot owns, read here, whose row
    of library_moves names a change at or before the change it is read at
    (the change that locked it, for a library locked then), or that has no
    row; null for any other. (libraries given figures, libraries read)."""
    name, at = con.execute("SELECT name, change FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0]
    given = read = 0
    for dataset, revision, libraries, shares in con.execute(
            "SELECT id, revision, libraries, shares FROM datasets_revisions").fetchall():
        tags = [one["tag"] for one in json.loads(shares or "[]")]
        found = []
        for ref in json.loads(libraries):
            read += 1
            library, change = ref["library"], ref["change"]
            if ref.get("depot") is not None or ids.owner(library) != name:
                found.append(None)
                continue
            state = con.execute("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                                "ORDER BY revision DESC LIMIT 1", (library, change)).fetchall()
            if state and state[0][0] is not None:
                change = con.execute("SELECT min(change) FROM libraries_revisions WHERE id = ? "
                                     "AND locked_at IS NOT NULL", (library,)).fetchall()[0][0]
            moved = con.execute("SELECT change FROM library_moves WHERE library = ?", (library,)).fetchall()
            if moved and (moved[0][0] is None or moved[0][0] > change):
                found.append(None)
                continue
            names = ([(LIBRARY_KIND_9, f"{library} {kind}") for kind in KINDS_9]
                     + [(LIBRARY_TAG_9, f"{library} {tag}") for tag in tags])
            sized = []
            for facet, row_name in names:
                row = con.execute("SELECT texts, characters, bytes FROM counts WHERE facet = ? AND name = ?",
                                  (facet, row_name)).fetchall()
                sized.append(dict(zip(("texts", "characters", "bytes"), row[0] if row else (0, 0, 0))))
            found.append(dict(zip(KINDS_9, sized)) | {"tags": sized[len(KINDS_9):]})
            given += 1
        con.execute("UPDATE datasets_revisions SET figures = ? WHERE id = ? AND revision = ?",
                    (json.dumps({"at": at, "libraries": found}, ensure_ascii=False, separators=(",", ":")),
                     dataset, revision))
    return given, read


def _merged_9(held, current, versions):
    """What the memberships held adds to the counts of format 9's new
    facets, from three streams in the order of the texts' ids (SCANS_9):
    held, (text, library) of each membership that has not ended; current,
    (id, content version, tags as JSON text, change) of each current
    revision; versions, (id, version, kind, length, bytes) of each content
    version. It goes through the three at once, as one sorted merge, and
    gives (sums: {(facet, name): [texts, characters, bytes]}, the id of the
    last text they cover, revised: {library: the latest change a text it
    holds among them was revised at}) each time about BATCH memberships are
    summed, at the end of a text, then the rest. Each membership adds to `<library>
    <kind>` (plain, items: ITEMS_9, other) and, for a plain text, to
    `<library> <tag>` for each of its tags; one of a text with no current
    revision or no content version counts for nothing."""
    sums, summed, seen, size, revised, change = {}, 0, None, None, {}, None
    text_row, version_row = next(current, None), next(versions, None)
    for text, library in held:
        if text != seen:
            if summed >= BATCH:
                yield sums, seen, revised
                sums, summed, revised = {}, 0, {}
            seen, size, change = text, None, None
            while text_row is not None and text_row[0] < text:
                text_row = next(current, None)
            if text_row is not None and text_row[0] == text:
                change = text_row[3]
                wanted = (text, text_row[1])
                while version_row is not None and (version_row[0], version_row[1]) < wanted:
                    version_row = next(versions, None)
                if version_row is not None and (version_row[0], version_row[1]) == wanted:
                    kind = version_row[2]
                    what = "plain" if kind is None else "items" if kind in ITEMS_9 else "other"
                    size = (what, version_row[3] or 0, version_row[4] or 0,
                            json.loads(text_row[2] or "[]") if kind is None else ())
        summed += 1
        if change is not None and revised.get(library, -1) < change:
            revised[library] = change
        if size is None:
            continue
        what, length, nbytes, tags = size
        for key in ((LIBRARY_KIND_9, f"{library} {what}"), *((LIBRARY_TAG_9, f"{library} {tag}") for tag in tags)):
            one = sums.get(key)
            if one is None:
                sums[key] = [1, length, nbytes]
            else:
                one[0] += 1
                one[1] += length
                one[2] += nbytes
    if sums or seen is not None:
        yield sums, seen, revised


def _to_10(db: Database, log):
    """Format 9 to 10, in several transactions, as the module says: the
    listings text_tags and text_licenses, the counts of score names and the
    splits, from the current revisions read in rowid order, BATCH rowids a
    transaction; then each index of INDEXES_10, one statement each; then
    the index text_decisions_decision dropped and the counts of pins whose
    text has a newer revision."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('texts', 0)")
            for statement in TABLES_10:
                con.execute(statement)
            for table, _, _ in LISTED_10:
                con.execute(f"DELETE FROM {table}")
            con.execute(f"DELETE FROM counts WHERE facet IN ('{SCORE_10}', '{NEWER_10}', "
                        + ", ".join(f"'{facet}'" for facet, _ in SPLITS_10) + ")")
            _new_format(con, 10)
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'texts'").fetchall()[0][0]
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    total = con.execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'texts'").fetchall()[0][0]
    log(f"{db.path}: listing the {total:,} current texts by tag and by license and counting their score names, "
        f"from the revisions read in rowid order, rows {after:,} to {last:,}, {BATCH:,} rowids a transaction"
        + ("" if after == 0 else "; going on after the last transaction the last run wrote"))
    began = said = time.monotonic()
    first = after
    while after < last:
        upto = min(after + BATCH, last)
        bounds = {"low": after, "high": upto}
        with db.bulk() as written:
            for table, name, column in LISTED_10:
                written.execute(f"INSERT OR IGNORE INTO {table} ({name}, key) SELECT j.value, {ORDER.replace('id', 't.id')} "
                                f"{BATCH_ROWS_10.replace('WHERE', f', json_each(t.{column}) j WHERE')}", bounds)
            added = [(SCORE_10, name, count) for name, count in written.execute(SCORES_10, bounds)]
            for facet, query in SPLITS_10:
                added += [(facet, name, count) for name, count in written.execute(query, bounds)]
            written.executemany("INSERT INTO counts (facet, name, texts) VALUES (?, ?, ?) ON CONFLICT (facet, name) "
                                "DO UPDATE SET texts = texts + excluded.texts", added)
            written.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'texts'", (upto,))
        after = upto
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: revision row {after:,} of {last:,}, {(after - first) / max(said - began, 1e-9):,.0f} rows a "
                f"second, {said - began:.0f} seconds")
    listed = time.monotonic()
    for name, statement in INDEXES_10:
        if db.reader().execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)).fetchall():
            continue
        log(f"{db.path}: building the index {name}, in one statement that reads its table once; run again after a "
            "stop, it builds it again")
        indexed = time.monotonic()
        with db.bulk() as written:
            written.execute(statement)
        log(f"{db.path}: the index {name} is built, in {time.monotonic() - indexed:.0f} seconds")
    with db.bulk() as written:
        written.execute(DROPPED_10)
        written.execute(f"DELETE FROM counts WHERE facet = '{NEWER_10}'")
        written.execute(NEWER_COUNTS_10)
        written.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the listings took {listed - began:.0f} seconds and the indexes {time.monotonic() - listed:.0f}; "
        f"the depot is at format 10, in {time.monotonic() - began:.0f} seconds")


# each format of an SQLite file below the code's: the step that brings it to the next
FILE_STEPS = {4: _to_5, 5: _to_6, 6: _to_7, 7: _to_8, 8: _to_9, 9: _to_10}


def _write(legacy: Legacy, target: Path, upstream: str | None):
    """Write the folder's tables into the new file target, as the module says."""
    working = target.with_name(target.name + UPGRADING)
    db = Database(working)
    if not working.exists():
        db.create()
    try:
        if not db.reader().execute("SELECT 1 FROM about_revisions WHERE format = ?", (WRITTEN,)).fetchall():
            _fill(db, legacy, upstream)
    finally:
        db.close()  # the last connection to close folds the WAL into the file and removes it
    for suffix in ("-wal", "-shm"):
        if working.with_name(working.name + suffix).exists():
            raise Refused(f"{working} is still open by another process")
    working.rename(target)
    waiting = working.with_name(working.name + ".waiting")
    if waiting.exists():
        waiting.rename(target.with_name(target.name + ".waiting"))
    Database(target).mark_full_copy()


def _fill(db: Database, legacy: Legacy, upstream: str | None):
    """Every data file not yet in, then the upgrade's own change."""
    with db.bulk() as con:
        con.execute("CREATE TABLE IF NOT EXISTS upgrade_progress (table_name TEXT NOT NULL, file TEXT NOT NULL, "
                    "PRIMARY KEY (table_name, file))")
    done = {(table, file) for table, file in db.reader().execute("SELECT table_name, file FROM upgrade_progress")}
    for table, into in TARGETS.items():
        rows = PARTS_ROWS if table == "parts" else ROWS
        for file in legacy.files(table) if table in legacy.exist() else []:
            name = Path(file).name
            if (table, name) in done:
                continue
            with db.bulk() as con:
                for batch in legacy.batches(table, file, rows):
                    _insert(con, into, [_converted(table, one) for one in batch])
                con.execute("INSERT INTO upgrade_progress VALUES (?, ?)", (table, name))
    with db.bulk() as con:
        _finish(con, legacy, upstream)
        con.execute("DROP TABLE upgrade_progress")


def _converted(table: str, row: dict) -> dict:
    """A row of a format 3 table as its format 4 table holds it."""
    found = {}
    for name, value in row.items():
        if table == "contents" and name == "parts":
            value = json.dumps([{"hash": one["hash"].hex(), "length": one["length"]} for one in value],
                               separators=(",", ":"))
        elif table == "erasures" and name == "erased" and value is not None:
            value = _json([{**one, "hash_a": one["hash_a"].hex(), "hash_b": one["hash_b"].hex()} for one in value])
        elif table == "parts" and name == "value":
            value = pack(value)
        elif name in LISTS:
            value = None if value is None else _json(value)
        elif isinstance(value, datetime):
            value = value.isoformat()
        elif isinstance(value, bool):
            value = int(value)
        found[name] = value
    if table in ("texts", "libraries", "erasures", "datasets"):
        found["latest"] = 0
    if table in ("decisions", "notices", "pins"):
        found["change"] = 0  # made from the journal and the libraries once every table is in (_finish)
    return found


def _insert(con, table: str, found: list[dict]):
    if found:
        names = list(found[0])
        con.executemany(f"INSERT OR IGNORE INTO {table} ({', '.join(names)}) "
                        f"VALUES ({', '.join(':' + one for one in names)})", found)


def _finish(con, legacy: Legacy, upstream: str | None):
    """Make the format 4 columns and tables the folder lacks, then the upgrade's own change."""
    for table in ("texts_revisions", "libraries_revisions", "erasures_revisions", "datasets_revisions"):
        con.execute(f"UPDATE {table} SET latest = 1 WHERE revision = "
                    f"(SELECT max(r.revision) FROM {table} r WHERE r.id = {table}.id)")
    con.execute("UPDATE pins SET change = coalesce((SELECT min(l.change) FROM libraries_revisions l "
                "WHERE l.id = pins.library AND l.locked_at IS NOT NULL), 0)")
    named = {one: None for (one,) in con.execute("SELECT id FROM decisions UNION ALL SELECT id FROM notices")}
    made = {}
    entries = []
    for batch in _journal(legacy):
        for entry, change in batch:
            text = json.loads(change)
            kind = text.get("kind", "change")
            decision = text.get("decision") if isinstance(text.get("decision"), dict) else None
            entries.append((entry, kind, decision.get("made_at") if decision else None))
            for one in _ID.findall(change):
                if one in named and named[one] is None:
                    named[one] = entry
    about = con.execute("SELECT change FROM about_revisions").fetchall()
    highest = con.execute("""SELECT max(n) FROM (
        SELECT max(change) AS n FROM about_revisions UNION ALL SELECT max(change) FROM texts_revisions
        UNION ALL SELECT max(change) FROM content_lists UNION ALL SELECT max(change) FROM libraries_revisions
        UNION ALL SELECT max(change) FROM erasures_revisions UNION ALL SELECT max(change) FROM datasets_revisions
        UNION ALL SELECT max(joined_change) FROM memberships UNION ALL SELECT max(left_change) FROM memberships
        UNION ALL SELECT max(change) FROM pulls)""").fetchall()[0][0] or 0
    number = max([highest, *(entry for entry, _, _ in entries)]) + 1
    for one, entry in named.items():
        made[one] = number if entry is None else entry
    con.executemany("UPDATE decisions SET change = ? WHERE id = ?", [(entry, one) for one, entry in made.items()])
    con.executemany("UPDATE notices SET change = ? WHERE id = ?", [(entry, one) for one, entry in made.items()])
    known = {entry for entry, _, _ in entries}
    con.executemany("INSERT OR IGNORE INTO changes (id, kind, made_at) VALUES (?, ?, ?)", entries)
    con.executemany("INSERT OR IGNORE INTO changes (id, kind, made_at) VALUES (?, 'about', NULL)",
                    [(change,) for (change,) in about if change not in known])
    current = con.execute("SELECT * FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0]
    columns = [one[1] for one in con.execute("PRAGMA table_info(about_revisions)")]
    row = dict(zip(columns, current))
    name = row["name"]
    con.execute("INSERT INTO changes (id, kind, made_at) VALUES (?, 'upgrade', ?)",
                (number, now()))
    con.execute("INSERT INTO about_revisions (revision, change, name, maintainer, takedown_contact, upstream, format) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", (row["revision"] + 1, number, name, row["maintainer"],
                                                 row["takedown_contact"], upstream, WRITTEN))
    own = ids.carries_sql(name)
    counted = {
        "texts": f"SELECT max({ids.number_sql()}) FROM (SELECT id FROM texts_revisions UNION SELECT j.value ->> 'text' "
                 "AS id FROM erasures_revisions e, json_each(e.needles) j) WHERE " + own,
        **{kind: f"SELECT max({ids.number_sql()}) FROM {table} WHERE {own}" for kind, table in (
            ("decisions", "decisions"), ("notices", "notices"), ("libraries", "libraries_revisions"),
            ("erasures", "erasures_revisions"), ("datasets", "datasets_revisions"))}}
    for kind, query in counted.items():
        last = con.execute(query).fetchall()[0][0]
        if last is not None:
            con.execute("INSERT INTO numbers (name, last) VALUES (?, ?)", (kind, last))


def _journal(legacy: Legacy):
    """(id, change) of every entry of the journal, a batch at a time."""
    for file in legacy.files("journal"):
        for batch in legacy.batches("journal", file, ROWS):
            yield [(one["id"], one["change"]) for one in batch]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_plain)


def _plain(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"{type(value).__name__} is no JSON value")
