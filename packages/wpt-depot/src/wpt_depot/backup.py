"""Backups of a depot: a new full copy of its folder, every table with its
log and its kept older states, in a new folder of its own; and a restore,
which copies a backup as it is to a new address.

A backup is the depot as of one settled change (wpt_depot.journal): each
table is copied at its storage version as of that change, its log up to
that version and every data file that log adds. Delta Lake never changes
a written log entry or data file, so a write during the copy only adds
files the copied log does not name. Its folder, under the address it goes
to, is named after the depot, the UTC time and that change number:
`my-depot-20261110T153000Z-change1234`.

The storage versions as of the settled change: the journal's state is read
first, and the settled change taken from it; then every other table's
current version; then the journal's current state again. When it holds no
entry past the settled change, no change past it had begun when those
versions were read, so they hold every row numbered up to it and none
above. Otherwise (a change was being written, or one is left undone),
each table is taken at its last version committed before the journal
first held an entry past the settled change, and the journal at the
version just before it.

The app never cuts, vacuums or erases a backup, and lists none: an erasure
carried out later does not reach it, and a restore copies it as it is,
erased passages included. The address a backup or a restore goes to is a
folder or a bucket, as a depot's (wpt_depot.depot.location), and must
hold nothing: neither ever overwrites. Each copies the about table last,
its log after its data and newest entry first: Delta Lake reads a table
only from its first log entry or a checkpoint, and every entry after it
has landed by then, so a copy cut short holds no about table and is no
depot.

A backup and a restore copy a depot of the code's format or of an older
one (wpt_depot.schema) as it is, so the copy keeps its format, and an
older copy needs `wpt depot upgrade` before anything else reads it. They
refuse a newer format, which only a newer code reads. They read only what
every format holds: the journal's id and done, the about table's change
and name, and every table's storage versions.
"""

import json
import re
from datetime import UTC, datetime
from urllib.parse import unquote

import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .depot import Depot
from .errors import BackupCut, DepotExists, VersionGone
from .journal import Journal
from .schema import TABLES
from .tables import Tables

LOG = re.compile(r"(\d{20})\.(json|checkpoint(\.\d+\.\d+)?\.parquet)")  # a log entry or a checkpoint, by version


def backup(depot: Depot, to: str, settings: dict | None = None) -> Depot:
    """A new backup of depot in a new folder under the address `to`, as of
    its settled change, read with nothing written, of the depot's format.
    DepotExists when that folder holds anything; BackupCut when an erasure
    dropped what it was copying or the copy stopped; OtherFormat when the
    depot's format is newer than the code's."""
    depot.refuse_newer()
    versions = _versions(depot.tables)
    settled = depot.journal.settled(versions["journal"])
    name = f"{depot.name()}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-change{settled}"
    settings = settings or depot.settings
    target = Depot(Depot(to, settings).tables.uri(name), settings)
    source, root = depot.tables.filesystem()
    try:
        paths = [f"{table}/{path}" for table in versions
                 for path in _files(source, f"{root}/{table}", versions[table])]
    except (FileNotFoundError, VersionGone) as error:
        raise BackupCut(f"{depot.address}: an erasure dropped the storage states the backup was copying ({error}); "
                        f"{target.address} is no depot, back up again") from None
    _copy(source, root, target.tables, paths)
    return Depot.find(target.address, target.settings)


def restore(address: str, to: str, settings: dict | None = None) -> Depot:
    """A copy, at the address `to`, of the backup at address, as it is, of
    its format. DepotExists when `to` holds anything; NotADepot when address
    holds no depot; BackupCut when the copy stopped; OtherFormat when its
    format is newer than the code's."""
    found = Depot.find(address, settings)
    found.refuse_newer()
    target = Depot(to, settings or found.settings)
    source, root = found.tables.filesystem()
    paths = [info.path[len(root) + 1:] for info in source.get_file_info(pafs.FileSelector(root, recursive=True))
             if info.type == pafs.FileType.File]
    _copy(source, root, target.tables, paths)
    return Depot.find(target.address, target.settings)


def _versions(tables: Tables) -> dict[str, int]:
    """{table: its storage version as of the depot's settled change}."""
    first = tables.version("journal")
    held = set(tables.exist())
    settled = Journal(tables).settled(first)
    # a table a later format added (schema.ADDED) is missing from a depot of an older format
    versions = {name: tables.version(name) for name in TABLES if name != "journal" and name in held}
    history = sorted((one["version"], one["timestamp"]) for one in tables.table("journal").history())
    if _highest(tables, history[-1][0]) <= settled:
        return {**versions, "journal": first}
    low, high = 0, len(history) - 1  # the journal holds no entry past settled at low, and one at high
    if _highest(tables, history[low][0]) > settled:
        raise BackupCut(f"{tables.root}: an erasure dropped the depot's storage states as of its change {settled}")
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if _highest(tables, history[middle][0]) > settled else (middle, high)
    return {**{name: _before(tables, name, history[high][1]) for name in versions}, "journal": history[low][0]}


def _highest(tables: Tables, version: int) -> int:
    """The highest id the journal holds at that storage version."""
    return max(tables.read("journal", version, columns=["id"])["id"].to_pylist(), default=0)


def _before(tables: Tables, name: str, moment: int) -> int:
    """The last kept storage version of the table committed before moment
    (milliseconds since 1970, as Delta Lake records it)."""
    found = [one["version"] for one in tables.table(name).history() if one["timestamp"] < moment]
    if not found:
        raise BackupCut(f"{tables.root}: an erasure dropped the storage states of {name} the backup needs")
    return max(found)


def _files(system: pafs.FileSystem, root: str, version: int) -> list[str]:
    """The paths, in the table's folder root, of its log entries and
    checkpoints up to version and of every data file they add."""
    logs = sorted(f"_delta_log/{info.base_name}"
                  for info in system.get_file_info(pafs.FileSelector(f"{root}/_delta_log"))
                  if (found := LOG.fullmatch(info.base_name)) and int(found[1]) <= version)
    data = set()
    for log in logs:
        if log.endswith(".json"):
            with system.open_input_stream(f"{root}/{log}") as stream:
                actions = [json.loads(line) for line in stream.read().decode("utf-8").splitlines() if line]
            data.update(action["add"]["path"] for action in actions if "add" in action)
        else:
            added = pq.read_table(f"{root}/{log}", filesystem=system, columns=["add"]).column("add").to_pylist()
            data.update(one["path"] for one in added if one)
    return sorted(unquote(path) for path in data) + logs


def _copy(source: pafs.FileSystem, root: str, target: Tables, paths: list[str]):
    """Copy the files at paths under root in source to the same paths in
    target's folder, which must hold nothing: the about table last, each
    table's log after its data and newest entry first. DepotExists when the
    folder holds anything; BackupCut when a file could not be copied."""
    system, base = target.filesystem()
    if system.get_file_info(pafs.FileSelector(base, recursive=True, allow_not_found=True)):
        raise DepotExists(f"{target.root} already holds files: a backup or a restore never overwrites")
    try:
        # descending on (not about, not log, path): every other table, then the about table's data, then its log
        for path in sorted(paths, key=lambda one: (not one.startswith("about/"), "/_delta_log/" not in one, one),
                           reverse=True):
            system.create_dir(f"{base}/{path}".rsplit("/", 1)[0], recursive=True)
            pafs.copy_files(f"{root}/{path}", f"{base}/{path}", source_filesystem=source,
                            destination_filesystem=system)
    except OSError as error:
        raise BackupCut(f"the copy to {target.root} stopped ({error}); it holds no about table, so it is no depot: "
                        f"delete it and copy again") from None
