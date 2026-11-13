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

A clone of those formats named its upstream by the upstream's folder
or bucket address; a clone of format 4 follows its upstream over the
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
from datetime import datetime
from pathlib import Path

from . import ids
from .database import Database, now
from .errors import DepotExists, OtherFormat, Refused
from .follow import Upstream
from .legacy import LAST, STEPS, Legacy
from .parts import pack
from .schema import FORMAT

UPGRADING = ".upgrading"  # what the name of the file being written adds to the new file's name
ROWS = 4096  # rows of a data file read at a time
PARTS_ROWS = 16  # rows of parts read at a time: each holds up to 64 MiB of text
# each format 3 table, by its folder's name: the table of format 4 its rows go into
TARGETS = {"about": "about_revisions", "licenses": "licenses", "notices": "notices", "kinds": "kinds",
           "decisions": "decisions", "texts": "texts_revisions", "contents": "content_lists", "parts": "parts",
           "libraries": "libraries_revisions", "memberships": "memberships", "pins": "pins",
           "erasures": "erasures_revisions", "datasets": "datasets_revisions", "pulls": "pulls"}
LISTS = {"licenses", "authors", "tags", "decisions", "scores", "problems", "needles", "shares", "libraries"}
_ID = re.compile(r'"([A-Za-z0-9][A-Za-z0-9._-]*-[0-9]+)"')


def upgrade(address: str, to: str | None = None, upstream: str | None = None):
    """(the depot at address brought up to this code's format, the format it
    had). A depot of the code's format is left as it is. upstream: the web
    address of the upstream's web app, for a clone. OtherFormat for a
    newer format, which no code brings down; Refused for a folder whose
    journal holds a change never finished, for a clone without upstream
    and for upstream given to a depot that follows none; DepotExists when
    the new file's path holds a file already."""
    from .depot import Depot, format_of, location, other_format
    path = location(address)
    start = format_of(address)
    if start > FORMAT:
        raise OtherFormat(other_format(str(path), start))
    if not path.is_dir():
        if upstream is not None:
            raise Refused(f"{path} is of format {start} already: --upstream is for the upgrade of a clone")
        return Depot.open(address), start
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
    return Depot.open(str(target)), start


def _write(legacy: Legacy, target: Path, upstream: str | None):
    """Write the folder's tables into the new file target, as the module says."""
    working = target.with_name(target.name + UPGRADING)
    db = Database(working)
    if not working.exists():
        db.create()
    try:
        if not db.reader().execute("SELECT 1 FROM about_revisions WHERE format = ?", (FORMAT,)).fetchall():
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
                                                 row["takedown_contact"], upstream, FORMAT))
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
