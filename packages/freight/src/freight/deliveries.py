"""The deliveries the freight leaves on the dock: their notes, the
delivery of a key, marking those an erasure reached stale, deleting a
delivery's files, the room a cap leaves, and a delivery adopted from files
made elsewhere.

Each delivery is a folder deliveries/<delivery key>/ of the dock
(freight.selection.delivery_key), its note delivery.json written last:

    deliveries/<delivery key>/
        delivery.json               the delivery note, written last
        items/part-NNNNN.parquet    the items taken, in order, with their credits
        licenses.parquet, notices.parquet
        files/...                   what the preparation wrote

A delivery is fresh while its note says neither `stale` (an erasure
reached an item it holds: the ids of the erasures) nor `deleted` (its files
are gone: when, and why). A deleted delivery keeps its note, so the same
request can run again and its id be compared with the one the note keeps.
A failed job leaves no note.

A cap (Room) bounds the bytes the deliveries' files take together, as
their notes give each file's size: past it, the job in progress deletes the
delivery made least recently, a file at a time, as it needs room, never
the one being made; a job that alone needs more than the cap fails
(TooBig), and so does one that finds nothing it may delete (NoRoom),
keeping what it uploaded. Before its first upload, a job estimates its
whole size (Room.expect), so a delivery bigger than the cap fails before
any other is deleted for it.

    notes(dock)                       every delivery note on the dock, newest first
    of_key(dock, key)                 the note of a delivery key, None without one
    find(dock, delivery_id)           the note of that id (or of a start of it), None without one
    files(note)                       every file a note names
    read_items(dock, note)            the items tables, as one table
    fetch(dock, note, folder)         a delivery's files copied into a local folder
    mark_stale(dock, texts, erasures) mark every fresh delivery that holds one of these texts stale
    delete(dock, note, why)           delete a delivery's files, its note kept and marked
    adopt(dock, key, files, result, about)   files made elsewhere put on the dock as a delivery
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store import PACKED
from wpt_store.names import File

from .selection import fingerprint

FOLDER = "deliveries"
NOTE = "delivery.json"
CAPPED = "the dock's cap needed its room"
ADOPTED = "adopted-"  # the start of the key of a delivery adopted from files made elsewhere


class DeliveryError(Exception):
    """What the dock refuses a job."""


class TooBig(DeliveryError):
    """The delivery being made needs more bytes than the dock's cap alone;
    what it uploaded stays."""


class NoRoom(DeliveryError):
    """The dock's cap is reached and nothing else may be deleted; what the job uploaded stays."""


class FilesGone(DeliveryError):
    """Files a job uploaded are no longer on the dock (another job's cap
    deleted its folder, which had no note yet): no note was written; the
    same request run again starts over."""


def folder_of(key: str) -> str:
    """The dock's folder of the delivery of that key."""
    return f"{FOLDER}/{key}"


def missing(dock, folder: str, uploaded: list[dict]) -> list[str]:
    """The keys of the files uploaded ({key, ...}) that the folder of the dock no longer holds, from one listing."""
    held = set(dock.kind.list(f"{folder}/", deep=True))
    return [one["key"] for one in uploaded if one["key"] not in held and one["key"] + PACKED not in held]


def fresh(note: dict) -> bool:
    return "stale" not in note and "deleted" not in note


def files(note: dict) -> list[dict]:
    """Every file of a delivery but its note, as the note names each ({key, sha256, bytes, ...})."""
    tables = note.get("tables", {})
    found = list(note.get("files", []))
    for table in tables.values():
        found += table if isinstance(table, list) else [table]
    return found


def notes(dock) -> list[dict]:
    """The note of every delivery on the dock, each checked by the dock, the last made first."""
    found = []
    for key in dock.kind.list(f"{FOLDER}/", deep=False):
        if key.endswith("/") and (note := dock.get_json(f"{key}{NOTE}")) is not None:
            found.append(note)
    return sorted(found, key=lambda note: note["made_at"], reverse=True)


def of_key(dock, key: str) -> dict | None:
    """The note of the delivery of that key, None when the dock holds none."""
    return dock.get_json(f"{folder_of(key)}/{NOTE}")


def find(dock, delivery_id: str) -> dict | None:
    """The note of the delivery whose id is delivery_id, or starts with it
    when only one does; None when none does. ValueError when several do."""
    found = [note for note in notes(dock) if note["id"].startswith(delivery_id)]
    if len(found) > 1:
        raise ValueError(f"{delivery_id!r} starts the ids of {len(found)} deliveries: give more of the id")
    return found[0] if found else None


def read_items(dock, note: dict) -> pa.Table:
    """The items tables of a delivery, each part checked by the dock, as one table in order."""
    parts = note.get("tables", {}).get("items", [])
    return pa.concat_tables([pq.read_table(dock.path(File.of(one))) for one in parts]) if parts else pa.table({})


def read_table(dock, note: dict, table: str) -> pa.Table:
    """One of a delivery's other tables (licenses, notices), checked by the dock."""
    return pq.read_table(dock.path(File.of(note["tables"][table])))


def fetch(dock, note: dict, folder: Path) -> Path:
    """Copy the files the preparation wrote into folder (each at its path
    under files/), each checked by the dock: the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for one in note.get("files", []):
        target = folder / one["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dock.path(File.of(one)), target)
        dock.release(File.of(one))
    return folder


def mark_stale(dock, texts, erasures, keep=lambda note: False) -> list[str]:
    """Mark stale every delivery of the dock whose files are kept and that
    holds one of the texts (ids) because the erasures (ids) were carried
    out: the keys of the deliveries marked. keep(note) True writes the
    erasures into the note instead and leaves it fresh. A delivery already
    stale keeps its earlier erasures; a deleted one is left as it is."""
    marked = []
    for note in notes(dock):
        if "deleted" in note or not holds(dock, note, texts):
            continue
        if keep(note):
            noted = sorted({*note.get("erasures_since", []), *erasures})
            dock.put_json(f"{folder_of(note['key'])}/{NOTE}", {**note, "erasures_since": noted})
            continue
        marked_stale(dock, note, erasures)
        marked.append(note["key"])
    return marked


def holds(dock, note: dict, texts) -> bool:
    """Whether the delivery holds one of the texts (ids), as its items tables list them."""
    wanted = set(texts)
    if not wanted:
        return False
    for part in note.get("tables", {}).get("items", []):
        ids = pq.read_table(dock.path(File.of(part)), columns=["id"])["id"].to_pylist()
        dock.release(File.of(part))
        if not wanted.isdisjoint(ids):
            return True
    return False


def marked_stale(dock, note: dict, erasures) -> dict:
    """The delivery's note marked stale because of the erasures (ids), with
    the erasures it was stale for already: the note as stored."""
    stale = sorted({*note.get("stale", {}).get("erasures", []), *erasures})
    return dock.put_json(f"{folder_of(note['key'])}/{NOTE}", {**note, "stale": {"erasures": stale}})


def delete(dock, note: dict, why: str) -> dict:
    """Delete every file of the delivery but its note, which says deleted
    from the first file on, and whole once the last file is gone: the note
    as stored."""
    if "deleted" not in note:
        note = mark_deleted(dock, note, why)
    for one in files(note):
        dock.delete(File.of(one))
    return mark_deleted(dock, note, note["deleted"]["why"], whole=True)


def mark_deleted(dock, note: dict, why: str, whole: bool = False) -> dict:
    """The delivery's note marked deleted for that reason, whole when none
    of its files is left: the note as stored. The time is the first mark's."""
    at = note.get("deleted", {}).get("at") or datetime.now(UTC).isoformat()
    return dock.put_json(f"{folder_of(note['key'])}/{NOTE}",
                         {**note, "deleted": {"at": at, "why": why, "whole": whole}})


def adopt(dock, key: str, made: dict[str, Path], result: dict, about: dict) -> dict:
    """Put files made elsewhere on the dock as the delivery of key, which
    starts with ADOPTED: made {path under files/: local file}, result the
    JSON a preparation would have answered, about what the note says of
    where they come from (`adopted`). The note as stored; the one already
    there when it is fresh."""
    if not key.startswith(ADOPTED):
        raise DeliveryError(f"an adopted delivery's key starts with {ADOPTED}: {key}")
    known = of_key(dock, key)
    if known is not None and fresh(known):
        return known
    folder = folder_of(key)
    listed = []
    for path, local in sorted(made.items()):
        file = dock.put_named(f"{folder}/files/{path}", local)
        listed.append({**file.reference(), "path": path, "bytes": Path(local).stat().st_size})
    note = {"key": key, "id": fingerprint({"key": key, "files": [[one["path"], one["sha256"]] for one in listed]}),
            "adopted": about, "result": result, "files": listed, "tables": {},
            "made_at": datetime.now(UTC).isoformat()}
    return dock.put_json(f"{folder}/{NOTE}", note)


class Room:
    """The room the cap (bytes, None: no cap) leaves on the dock for the
    delivery being made in the folder own: the files of every other
    delivery count, at the sizes their notes give, those of a delivery that
    keeps them and of one whose deletion is not whole, and so do the files
    of a folder of deliveries/ that holds no note (left by a job that
    failed or was cancelled: its request, run again, finds them gone and
    starts over). The files are listed once, when the first room is asked
    for, and only this Room deletes any while the job runs."""

    def __init__(self, dock, cap: int | None, own: str, log=lambda line: None):
        self.dock, self.cap, self.own, self.log = dock, cap, own, log
        self.used = 0  # the bytes the delivery being made uploaded so far
        self._kept = None  # {delivery key, or a folder without a note: (its note or None, [its files left])}
        self._said = set()  # the folders without a note whose deletion was said

    def _size(self, one: dict) -> int:
        if "bytes" in one:
            return one["bytes"]
        try:
            return self.dock.size(File.of(one))
        except Exception:  # a file the dock no longer holds takes no room
            return 0

    def _listed(self) -> dict:
        if self._kept is None:
            self._kept = {}
            for key in self.dock.kind.list(f"{FOLDER}/", deep=False):
                folder = key.rstrip("/")
                if not key.endswith("/") or folder == self.own:
                    continue
                note = self.dock.get_json(f"{key}{NOTE}")
                if note is None:
                    left = [{"key": one, "bytes": size or 0} for one, size in
                            sorted(self.dock.kind.list(key, deep=True).items())]
                    if left:
                        self._kept[folder] = (None, left)
                    continue
                if note.get("deleted", {}).get("whole", "deleted" in note):
                    continue
                left = [one for one in files(note) if "deleted" not in note or self.dock.has(File.of(one))]
                self._kept[note["key"]] = (note, [{**one, "bytes": self._size(one)} for one in left])
        return self._kept

    def taken(self) -> int:
        """The bytes the files of the other deliveries take now, as make counts them."""
        return sum(one["bytes"] for _, left in self._listed().values() for one in left)

    def expect(self, size: int):
        """TooBig when the delivery being made, estimated at size bytes in
        all, passes the cap on its own: called before its first upload, so
        nothing is deleted for a delivery that cannot fit."""
        if self.cap is not None and size > self.cap:
            raise TooBig(f"this delivery would take about {size:,} bytes, more than the dock's cap of {self.cap:,} "
                         "bytes on its own: raise the cap or ask for a smaller size; no delivery was deleted for it")

    def make(self, size: int):
        """Make room for size more bytes of the delivery being made: delete
        files of the others while the cap is passed, first those of a folder
        without a note, then those of a delivery whose deletion is not
        whole, then those of the delivery made least recently, a file at a
        time. TooBig when the delivery alone passes the cap; NoRoom when
        nothing more may be deleted."""
        if self.cap is None:
            self.used += size
            return
        if self.used + size > self.cap:
            raise TooBig(f"this delivery needs more than the dock's cap of {self.cap:,} bytes on its own "
                         f"({self.used + size:,} bytes so far): raise the cap or ask for a smaller size")
        kept = self._listed()
        taken = self.taken()
        while taken + self.used + size > self.cap:
            def rank(name):
                note = kept[name][0]
                if note is None:
                    return (0, "")
                return (1 if "deleted" in note else 2, note["made_at"])
            free = sorted((name for name, (_, left) in kept.items() if left), key=rank)
            if not free:
                raise NoRoom(f"the dock's cap of {self.cap:,} bytes is reached ({taken:,} bytes of other deliveries, "
                             f"{self.used + size:,} of this one) and nothing else may be deleted")
            note, left = kept[free[0]]
            one = left.pop(0)
            if note is None:
                if free[0] not in self._said:
                    self._said.add(free[0])
                    self.log(f"the files of {free[0]} deleted, a file at a time as this job needs room: no delivery "
                             "note names them")
                self.dock.kind.delete(one["key"])
            else:
                if "deleted" not in note:
                    note = mark_deleted(self.dock, note, CAPPED)
                    self.log(f"delivery {note['key']} deleted, a file at a time as this job needs room: {CAPPED}")
                self.dock.delete(File.of(one))
                if not left:
                    note = mark_deleted(self.dock, note, note["deleted"]["why"], whole=True)
            taken -= one["bytes"]
            kept[free[0]] = (note, left)
        self.used += size
