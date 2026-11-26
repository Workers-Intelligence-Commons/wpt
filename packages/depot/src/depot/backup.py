"""The depot's backups, on the dock (DOCK, dock), under
backups/<the depot's name>/.

Every change of the depot leaves a change file in the folder next to its
file (depot.database): the rows it changed, as a session of SQLite
records them. `depot backup` (backup) sends each change file waiting
there onto the dock, under changes/, and deletes it from the folder once
the dock holds it; its first run for the file's timeline
(depot.database: a new depot, a clone, an upgrade and a restore each
start one), and a run asked for one (`--full`), also send a full copy of
the file, under full/, made with SQLite's backup into a temporary file
beside the depot's file while the depot goes on being written. Each run
then writes its record under runs/: its timeline, the full copy and the
change files it sent, each with its change number, the number of the
change before it, and the time it was made. How often it runs is not the
app's business: a timer of the machine runs it. The dock compresses what
it holds (dock), and the app never removes a backup.

`depot restore TO --at MOMENT` (restore) makes a new depot file at TO
from the backups of one timeline: the latest one by default; for a change
number, the latest one that reaches it; for a UTC time, the one that made
the latest change or full copy at or before it. From that timeline it takes
the last full copy at or before MOMENT, then every change file after it, in
order, up to MOMENT (the latest change sent, by default), each change
following the one before it. A change that does not apply exactly (a
conflict, a missing link) stops the restore, which leaves nothing at TO;
the file is written as `<TO>.restoring`, with the depot's own writer
settings (a deleted row's bytes overwritten), and renamed once whole. The
restored depot starts a timeline of its own, and its next backup sends a
full copy.

`depot backup --off` (depot.database: Database.backup_off) marks a
depot that nothing backs up: its writes record no change file and its
waiting change files are deleted. Its next backup turns the recording back
on before anything else, then sends a full copy.

A change file holds the rows its change removed as they were, so an
erased passage stays in the waiting folder until the next backup sends it,
and in the backups for good (docs/erasure.md).

The depot knows nothing of the dock: backup and restore take `dock`, the
place the backups go, which answers put_from(key, path) and put_json(key,
record) (each giving the reference {key, sha256} of what it stored),
get_json(key), folders(prefix) (the folders right under a key prefix) and
fetch_into(reference, path) (a stored file into a local file, checked).
The depot's command gives one over dock.api (depot_app.DockBackups).
"""

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import apsw

from .database import FULL_COPY_NEXT, WRITER, Database, now, parse_waiting
from .depot import Depot, format_of, location
from .errors import BackupRefused, DepotExists, NotFound, OtherFormat
from .schema import FORMAT

FOLDER = "backups"
RESTORING = ".restoring"
READ = 1 << 20  # bytes of a change file read at a time while it applies


def backup(depot: Depot, dock, full: bool = False) -> dict:
    """Send the depot's waiting change files, and a full copy when one is
    due, onto the dock: the run's record (the module's docstring)."""
    name = depot.name()
    base = f"{FOLDER}/{name}"
    db = depot.db
    if not db.recording():
        db.backup_on()  # before the full copy, so every change after the copy has its change file
        full = True
    db.settle()
    timeline = db.timeline()
    waiting = sorted(db.waiting_files(), key=lambda path: parse_waiting(path.name)[0])
    marker = db.waiting / FULL_COPY_NEXT
    full = full or marker.exists() or not any(one["timeline"] == timeline for one in _records(dock, name))
    made_at = now()
    stamp = _stamp(made_at)
    record = {"depot": name, "timeline": timeline, "made_at": made_at, "full": None, "changes": []}
    if full:
        with tempfile.TemporaryDirectory(prefix=f".{db.path.name}.backup-", dir=db.path.parent) as folder:
            copy = Path(folder) / "depot.sqlite"
            change = db.copy_to(copy)
            stored = dock.put_from(f"{base}/full/{change:012d}-{stamp}.sqlite", copy)
            record["full"] = {"key": stored["key"], "sha256": stored["sha256"], "change": change, "made_at": made_at}
    for path in waiting:
        number, previous, at = parse_waiting(path.name)
        stored = dock.put_from(f"{base}/changes/{path.name}", path)
        record["changes"].append({"key": stored["key"], "sha256": stored["sha256"], "number": number, "previous": previous,
                                  "made_at": at.isoformat()})
    record = dock.put_json(f"{base}/runs/{stamp}/record.json", record)
    for path in waiting:
        path.unlink()
    if full:
        marker.unlink(missing_ok=True)
    return record


def restore(dock, to: str, at: str | None = None, name: str | None = None) -> Depot:
    """A new depot at `to` made from the backups of the depot called name
    (the only one the dock holds backups of, by default) as of at: a
    change number, a UTC time (ISO 8601), or None for the latest change
    sent. DepotExists when `to` holds a file; NotFound without backups of
    that name; BackupRefused when no full copy is at or before the moment,
    a change is missing from the chain, or a change does not apply."""
    target = location(to)
    if target.exists():
        raise DepotExists(f"{target} already exists: a restore writes a new file")
    name = name or _only_name(dock)
    records = _records(dock, name)
    if not records:
        raise NotFound(f"the dock holds no backup of a depot called {name}")
    number, moment = _moment(at)
    records = _timeline(records, number, moment)
    fulls = [one["full"] for one in records if one["full"] is not None]
    changes = {}
    for one in records:
        for change in one["changes"]:
            changes.setdefault(change["number"], change)

    def before(one, key="change"):
        return one[key] <= number if number is not None else (moment is None or _time(one["made_at"]) <= moment)
    chosen = max((one for one in fulls if before(one)), key=lambda one: one["change"], default=None)
    if chosen is None:
        raise BackupRefused(f"the backups of {name} hold no full copy at or before {at}")
    chain, current = [], chosen["change"]
    for found in sorted(changes):
        change = changes[found]
        if found <= current:
            continue
        if not before(change, "number"):
            break
        if change["previous"] != current:
            raise BackupRefused(f"the backups of {name} lack the change after {current} (the next one sent is "
                                f"{found}, which follows {change['previous']}): nothing is restored")
        chain.append(change)
        current = found
    if number is not None and current != number:
        raise BackupRefused(f"the backups of {name} reach the change {current}, not {number}: nothing is restored")
    working = target.with_name(target.name + RESTORING)
    copies = target.with_name(target.name + RESTORING + "-copies")
    _remove(working, copies)
    try:
        copies.mkdir(parents=True)
        dock.fetch_into(chosen, working)
        found_format = format_of(str(working))
        if found_format != FORMAT:
            raise OtherFormat(f"the backups of {name} are of format {found_format}, and this depot restores format "
                              f"{FORMAT}")
        con = apsw.Connection(str(working))
        try:
            for pragma in WRITER:
                con.execute(f"PRAGMA {pragma}").fetchall()
            for change in chain:
                local = copies / f"{change['number']}"
                dock.fetch_into(change, local)
                with open(local, "rb") as file, con:
                    apsw.Changeset.apply(file.read, con, conflict=_conflict)
                local.unlink()
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        except apsw.Error as error:
            raise BackupRefused(f"a change of the backups of {name} does not apply: {error}") from None
        finally:
            con.close()
    except BaseException:
        _remove(working, copies)
        raise
    shutil.rmtree(copies, ignore_errors=True)
    working.rename(target)
    Database(target).mark_full_copy()
    return Depot.open(str(target))


def _remove(working: Path, copies: Path):
    """Remove what a restore cut short left: its file, the file's WAL, which
    SQLite would otherwise replay onto the next attempt's file, and its
    copies of the dock's files."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        working.with_name(working.name + suffix).unlink(missing_ok=True)
    shutil.rmtree(copies, ignore_errors=True)


def _timeline(records: list[dict], number: int | None, moment: datetime | None) -> list[dict]:
    """The records of the timeline a restore reads (the module's docstring), in order."""
    order = []
    for one in records:  # the timelines in the order of their latest run
        if one["timeline"] in order:
            order.remove(one["timeline"])
        order.append(one["timeline"])
    if number is not None:
        for timeline in reversed(order):
            held = [one for one in records if one["timeline"] == timeline]
            if any(one["full"] is not None and one["full"]["change"] == number for one in held) or any(
                    change["number"] == number for one in held for change in one["changes"]):
                return held
        raise BackupRefused(f"no timeline of the backups holds the change {number}: nothing is restored")
    chosen = order[-1]
    if moment is not None:
        made = [(_time(item["made_at"]), one["timeline"]) for one in records
                for item in ([one["full"]] if one["full"] is not None else []) + one["changes"]
                if _time(item["made_at"]) <= moment]
        if made:
            chosen = max(made)[1]
    return [one for one in records if one["timeline"] == chosen]


def _conflict(kind, change):
    raise BackupRefused(f"a change of the backups does not apply exactly ({_KINDS.get(kind, kind)} on {change.op} "
                        f"of {change.name})")


_KINDS = {apsw.SQLITE_CHANGESET_DATA: "a row that differs", apsw.SQLITE_CHANGESET_NOTFOUND: "a row missing",
          apsw.SQLITE_CHANGESET_CONFLICT: "a row there already", apsw.SQLITE_CHANGESET_CONSTRAINT: "a constraint",
          apsw.SQLITE_CHANGESET_FOREIGN_KEY: "a foreign key"}


def _records(dock, name: str) -> list[dict]:
    """The record of every backup run of the depot called name, in order."""
    found = []
    for run in dock.folders(f"{FOLDER}/{name}/runs/"):
        if (record := dock.get_json(f"{FOLDER}/{name}/runs/{run}/record.json")) is not None:
            found.append(record)
    return sorted(found, key=lambda one: one["made_at"])


def _only_name(dock) -> str:
    names = dock.folders(f"{FOLDER}/")
    if len(names) != 1:
        raise BackupRefused(f"the dock holds backups of {len(names)} depots ({', '.join(sorted(names)) or 'none'}): "
                            "name the one to restore")
    return names[0]


def _moment(at: str | None) -> tuple[int | None, datetime | None]:
    """(a change number, a time) from what --at says: one of them, or neither for the latest."""
    if at is None:
        return None, None
    if at.isdigit():
        return int(at), None
    try:
        found = datetime.fromisoformat(at)
    except ValueError:
        raise BackupRefused(f"--at takes a change number or a UTC time (ISO 8601), not {at!r}") from None
    return None, found if found.tzinfo else found.replace(tzinfo=UTC)


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _stamp(made_at: str) -> str:
    return datetime.fromisoformat(made_at).strftime("%Y%m%dT%H%M%S.%fZ")

