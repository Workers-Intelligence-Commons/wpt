"""The corpora the store keeps (wpt_depot.corpus builds them): their
records, the latest fresh one of a dataset, marking those an erasure
reached stale, deleting a corpus's files, and the room a cap leaves.

Each corpus is a folder corpora/<dataset name>-<8 hex characters>/ of the
store, its record corpus.json written last (folder_of). A corpus is fresh
while its record says neither `stale` (an admitted erasure reached a text
it holds: the ids of the erasures) nor `deleted` (its files are gone: when,
and why). A deleted corpus keeps its record, so the same build can be run
again and its id compared with the one the record keeps.

Only each dataset's latest corpus keeps its files: once a build is whole,
every other corpus of the same depot's dataset is deleted (superseded),
but one of a newer revision of the dataset than the build's.
A cap (Room) bounds the bytes the corpora's files take together, as the
records give each file's size: past it, the build in progress deletes the
corpus downloaded least recently, a file at a time, as it needs room,
never one being built or downloaded; a build that alone needs more than the
cap fails (TooBig), and so does one that finds nothing it may delete
(NoRoom), keeping what it uploaded. Before its first upload, a build
estimates its whole size (Room.expect), so a corpus bigger than the cap
fails before any other corpus is deleted for it.

    records(store)                       every corpus the store holds, newest first
    find(store, corpus_id)               the corpus of that id (or of a start of it), None without one
    built_from(store, depot, dataset)    every corpus built for that depot's dataset (by id), latest first
    latest(store, depot, dataset)        the corpus last built for that depot's dataset (by id), None without one
    latest_fresh(store, depot, dataset)  the same, of the fresh ones
    mark_stale(store, texts, erasures)   mark every fresh corpus that holds one of these texts stale
    holds(store, record, texts)          whether a corpus holds one of these texts
    delete(store, record, why)           delete a corpus's files, its record kept and marked
    supersede(store, record)             delete every other corpus of the record's dataset
"""

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
from wpt_store import PACKED
from wpt_store.names import File, as_record_name

from .errors import DepotError

FOLDER = "corpora"
RECORD = "corpus.json"
SUPERSEDED = "a newer corpus of the dataset replaced it"
CAPPED = "the store's cap needed its room"


class TooBig(DepotError):
    """The corpus being built needs more bytes than the store's cap alone;
    what it uploaded stays. Refused on the estimate its build makes before
    its first upload, no other corpus was deleted for it; refused later, when
    the estimate was too low, the corpora deleted for its room stay deleted."""


class NoRoom(DepotError):
    """The store's cap is reached and every other corpus is being built or
    downloaded, so none may be deleted; what the build uploaded stays."""


class FilesGone(DepotError):
    """Files a build uploaded are no longer in the store (another build's
    cap deleted its folder, which had no record yet): no record was written;
    the same build started again starts over."""


def missing(store, folder: str, uploaded: list[dict]) -> list[str]:
    """The keys of the files uploaded ({key, ...}) that the folder of the store no longer holds, from one listing."""
    held = set(store.kind.list(f"{folder}/", deep=True))
    return [one["key"] for one in uploaded if one["key"] not in held and one["key"] + PACKED not in held]


def folder_of(record: dict) -> str:
    """The store's folder of a corpus: the one its record names, or for a
    record that names none, corpora/<dataset name>-<first 8 hex characters of its id>."""
    return record.get("folder") or f"{FOLDER}/{as_record_name(record['dataset']['name'])}-{record['id'][:8]}"


def fresh(record: dict) -> bool:
    return "stale" not in record and "deleted" not in record


def files(record: dict) -> list[dict]:
    """Every file of a corpus but its record, as the record names each ({key, sha256} and maybe bytes)."""
    tables = record.get("tables", {})
    found = [*record.get("shards", []), *record.get("conversation_shards", [])]
    for table in tables.values():
        found += table if isinstance(table, list) else [table]
    if "vocabulary" in record.get("tokenizer", {}):
        found.append(record["tokenizer"]["vocabulary"])
    if "tasks" in record:
        found.append(record["tasks"]["file"])
    return found


def records(store) -> list[dict]:
    """The record of every corpus the store holds, each checked by the
    store, the last built first."""
    found = []
    for key in store.kind.list(f"{FOLDER}/", deep=False):
        if key.endswith("/") and (record := store.get_json(f"{key}{RECORD}")) is not None:
            found.append(record)
    return sorted(found, key=lambda record: record["built_at"], reverse=True)


def find(store, corpus_id: str) -> dict | None:
    """The record of the corpus whose id is corpus_id, or starts with it
    when only one does; None when none does. ValueError when several do."""
    found = [record for record in records(store) if record["id"].startswith(corpus_id)]
    if len(found) > 1:
        raise ValueError(f"{corpus_id!r} starts the ids of {len(found)} corpora: give more of the id")
    return found[0] if found else None


def built_from(store, depot: str, dataset: str) -> list[dict]:
    """The records of the corpora built from the dataset of id dataset of
    the depot of that name (wpt_depot.corpus.depot_name), latest first: those of the dataset's newest
    revision first, each revision's last built first, so a corpus of an
    older revision built again never comes before one of a newer revision.
    A dataset's corpora stay its own through a rename, and a new dataset
    given an old name has none."""
    found = [record for record in records(store)
             if record["depot"] == depot and record["dataset"].get("id") == dataset]
    return sorted(found, key=lambda record: (record["dataset"].get("revision", 0), record["built_at"]), reverse=True)


def latest(store, depot: str, dataset: str) -> dict | None:
    """The record of the corpus last built for the dataset of id dataset of
    the depot of that name (built_from); None when the store holds
    none."""
    return next(iter(built_from(store, depot, dataset)), None)


def latest_fresh(store, depot: str, dataset: str) -> dict | None:
    """The record of the fresh corpus last built for that dataset; None without one."""
    return next((record for record in built_from(store, depot, dataset) if fresh(record)), None)


def read_table(store, record: dict, table: str) -> pa.Table:
    """One of a corpus's tables (wpt_depot.corpus.CORPUS_SCHEMAS), each
    file checked by the store: the texts table is the concatenation of its
    parts, in order."""
    found = record["tables"][table]
    if isinstance(found, list):
        return pa.concat_tables([pq.read_table(store.path(one)) for one in found]) if found else pa.table({})
    return pq.read_table(store.path(found))


def mark_stale(store, texts, erasures) -> list[str]:
    """Mark stale every corpus of the store whose files are kept and that
    holds one of the texts (ids), because the erasures (ids) were carried
    out: the ids of the corpora marked. A corpus already stale keeps its
    earlier erasures. A deleted corpus is left as it is: its rebuild
    compares its id."""
    marked = []
    for record in records(store):
        if "deleted" in record or not holds(store, record, texts):
            continue
        marked_stale(store, record, erasures)
        marked.append(record["id"])
    return marked


def holds(store, record: dict, texts) -> bool:
    """Whether the corpus holds one of the texts (ids), as its texts table lists them."""
    wanted = set(texts)
    if not wanted:
        return False
    parts = record["tables"]["texts"]
    for part in parts if isinstance(parts, list) else [parts]:
        ids = pq.read_table(store.path(part), columns=["id"])["id"].to_pylist()
        store.release(part)
        if not wanted.isdisjoint(ids):
            return True
    return False


def marked_stale(store, record: dict, erasures) -> dict:
    """The corpus's record marked stale because of the erasures (ids), with
    the erasures it was stale for already: the record as stored."""
    stale = sorted({*record.get("stale", {}).get("erasures", []), *erasures})
    return store.put_json(f"{folder_of(record)}/{RECORD}", {**record, "stale": {"erasures": stale}})


def delete(store, record: dict, why: str) -> dict:
    """Delete every file of the corpus but its record, which says deleted
    from the first file on, and whole once the last file is gone: the record
    as stored."""
    if "deleted" not in record:
        record = mark_deleted(store, record, why)
    for one in files(record):
        store.delete(File.of(one))
    return mark_deleted(store, record, record["deleted"]["why"], whole=True)


def mark_deleted(store, record: dict, why: str, whole: bool = False) -> dict:
    """The corpus's record marked deleted for that reason, whole when none
    of its files is left: the record as stored. The time is the first
    mark's."""
    at = record.get("deleted", {}).get("at") or datetime.now(UTC).isoformat()
    return store.put_json(f"{folder_of(record)}/{RECORD}",
                          {**record, "deleted": {"at": at, "why": why, "whole": whole}})


def supersede(store, record: dict, log) -> list[str]:
    """Delete the files of every other corpus built from the record's
    dataset (its depot and id) that keeps them, but those of a newer
    revision of the dataset (a corpus of an older revision built again
    leaves the newer ones as they are): the ids deleted."""
    gone = []
    revision = record["dataset"].get("revision", 0)
    for other in built_from(store, record["depot"], record["dataset"]["id"]):
        if (other["id"] != record["id"] and "deleted" not in other
                and other["dataset"].get("revision", 0) <= revision):
            delete(store, other, SUPERSEDED)
            gone.append(other["id"])
            log(f"corpus {other['id']} deleted: {SUPERSEDED}")
    return gone


class Room:
    """The room the cap (bytes, None: no cap) leaves in the store for the
    corpus being built in the folder own: the files of every other corpus
    count, at the sizes their records give, those of a corpus that keeps
    them and of one whose deletion is not whole, and so do the files of a
    folder of corpora/ that holds no record (left by a build that failed,
    was cancelled or was never resumed: its build, started again, finds them
    gone and starts over, wpt_depot.corpus). last_download(corpus id) gives
    the time (seconds since the epoch, 0 for never) the corpus was last
    downloaded, busy() the ids of the corpora being downloaded now. The
    files are listed once, when the first room is asked for, and only this
    Room deletes any while the build runs."""

    def __init__(self, store, cap: int | None, own: str, last_download=lambda corpus: 0.0, busy=lambda: set(),
                 log=lambda line: None):
        self.store, self.cap, self.own = store, cap, own
        self.last_download, self.busy, self.log = last_download, busy, log
        self.used = 0  # the bytes the corpus being built uploaded so far
        # {corpus id, or the folder of one without a record: (its record or None, [its files left, each with
        # its size])}
        self._kept = None
        self._said = set()  # the folders without a record whose deletion was said

    def _size(self, one: dict) -> int:
        if "bytes" in one:
            return one["bytes"]
        try:
            return self.store.size(File.of(one))
        except Exception:  # a file the store no longer holds takes no room
            return 0

    def _listed(self) -> dict:
        if self._kept is None:
            self._kept = {}
            for key in self.store.kind.list(f"{FOLDER}/", deep=False):
                folder = key.rstrip("/")
                if not key.endswith("/") or folder == self.own:
                    continue
                record = self.store.get_json(f"{key}{RECORD}")
                if record is None:
                    left = [{"key": one, "bytes": size or 0} for one, size in
                            sorted(self.store.kind.list(key, deep=True).items())]
                    if left:
                        self._kept[folder] = (None, left)
                    continue
                if record.get("deleted", {}).get("whole", "deleted" in record):
                    continue
                left = [one for one in files(record)
                        if "deleted" not in record or self.store.has(File.of(one))]
                self._kept[record["id"]] = (record, [{**one, "bytes": self._size(one)} for one in left])
        return self._kept

    def expect(self, size: int):
        """TooBig when the corpus being built, estimated at size bytes in
        all, passes the cap on its own: called before its first upload, so
        nothing is deleted for a corpus that cannot fit."""
        if self.cap is not None and size > self.cap:
            raise TooBig(f"this corpus would take about {size:,} bytes, more than the store's cap of {self.cap:,} "
                         "bytes on its own: raise the cap or set the dataset's max_tokens; no corpus was deleted "
                         "for it")

    def make(self, size: int):
        """Make room for size more bytes of the corpus being built: delete
        files of the others while the cap is passed, first those of a
        folder without a record, then those of a corpus whose deletion is not
        whole, then those of the corpus downloaded least recently, a file at
        a time. TooBig when the corpus alone passes the cap; NoRoom when
        nothing more may be deleted."""
        if self.cap is None:
            self.used += size
            return
        if self.used + size > self.cap:
            raise TooBig(f"this corpus needs more than the store's cap of {self.cap:,} bytes on its own "
                         f"({self.used + size:,} bytes so far): raise the cap or set the dataset's max_tokens")
        kept = self._listed()
        taken = sum(one["bytes"] for _, left in kept.values() for one in left)
        while taken + self.used + size > self.cap:
            busy = self.busy()

            def rank(name):
                record = kept[name][0]
                if record is None:
                    return (0, 0.0, "")
                return (1 if "deleted" in record else 2, self.last_download(name), record["built_at"])
            free = sorted((name for name, (record, left) in kept.items() if left and name not in busy), key=rank)
            if not free:
                raise NoRoom(f"the store's cap of {self.cap:,} bytes is reached ({taken:,} bytes of other corpora, "
                             f"{self.used + size:,} of this one) and every other corpus is being downloaded: build "
                             "again once a download ended")
            record, left = kept[free[0]]
            one = left.pop(0)
            if record is None:
                if free[0] not in self._said:
                    self._said.add(free[0])
                    self.log(f"the files of {free[0]} deleted, a file at a time as this build needs room: no corpus "
                             "record names them")
                self.store.kind.delete(one["key"])
            else:
                if "deleted" not in record:
                    record = mark_deleted(self.store, record, CAPPED)
                    self.log(f"corpus {record['id']} deleted, a file at a time as this build needs room: {CAPPED}")
                self.store.delete(File.of(one))
                if not left:
                    record = mark_deleted(self.store, record, record["deleted"]["why"], whole=True)
            taken -= one["bytes"]
            kept[free[0]] = (record, left)
        self.used += size
