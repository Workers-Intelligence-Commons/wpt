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
texts_status and fills the counts (wpt_depot.counts.recount). The step to
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
refuses a depot of format 8.

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

from . import counts, ids, policy
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
UNDER_WAY = "upgrade_under_way"  # the table that marks a depot whose upgrade has not ended, and how far it went
BATCH = 250_000  # rows a transaction of the upgrades to formats 6 (texts_revisions) and 7 (memberships, pins) goes over
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
        counts.recount(con)
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the depot is at format 6, in {time.monotonic() - began:.0f} seconds")


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


# each format of an SQLite file below the code's: the step that brings it to the next
FILE_STEPS = {4: _to_5, 5: _to_6, 6: _to_7, 7: _to_8}


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
