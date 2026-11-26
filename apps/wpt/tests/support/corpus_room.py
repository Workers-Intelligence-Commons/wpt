"""The cap the tests' reference build (corpus_build) works under on the
dock's corpora/: the room it leaves, what it refuses, and the deletion of
a corpus's files, its record kept and marked. The freight's own cap on
its deliveries is freight.deliveries.Room.

A cap (Room) bounds the bytes the corpora's files take together, as the
records give each file's size: past it, a build deletes the corpus
downloaded least recently, a file at a time, as it needs room, never one
being built or downloaded; a build that alone needs more than the cap fails
(TooBig), and so does one that finds nothing it may delete (NoRoom).

    missing(dock, folder, uploaded)      the uploaded files a folder no longer holds
    delete(dock, record, why)            delete a corpus's files, its record kept and marked
    Room(dock, cap, own, ...)            the room a cap leaves for the corpus being built
"""

from datetime import UTC, datetime

from dock import api as dock_api
from freight.corpora import FOLDER, RECORD, files, folder_of

CAPPED = "the dock's cap needed its room"


class CorpusError(Exception):
    """What a corpus's room refuses."""


class TooBig(CorpusError):
    """The corpus being built needs more bytes than the dock's cap alone;
    what it uploaded stays. Refused on the estimate its build makes before
    its first upload, no other corpus was deleted for it; refused later, when
    the estimate was too low, the corpora deleted for its room stay deleted."""


class NoRoom(CorpusError):
    """The dock's cap is reached and every other corpus is being built or
    downloaded, so none may be deleted; what the build uploaded stays."""


class FilesGone(CorpusError):
    """Files a build uploaded are no longer on the dock (another build's
    cap deleted its folder, which had no record yet): no record was written;
    the same build started again starts over."""


def missing(dock, folder: str, uploaded: list[dict]) -> list[str]:
    """The keys of the files uploaded ({key, ...}) that the folder of the dock no longer holds, from one listing."""
    held = {one["key"] for one in dock_api.find(dock, f"{folder}/")}
    return [one["key"] for one in uploaded if one["key"] not in held]


def delete(dock, record: dict, why: str) -> dict:
    """Delete every file of the corpus but its record, which says deleted
    from the first file on, and whole once the last file is gone: the record
    as stored."""
    if "deleted" not in record:
        record = mark_deleted(dock, record, why)
    for one in files(record):
        dock_api.delete(dock, one["key"])
    return mark_deleted(dock, record, record["deleted"]["why"], whole=True)


def mark_deleted(dock, record: dict, why: str, whole: bool = False) -> dict:
    """The corpus's record marked deleted for that reason, whole when none
    of its files is left: the record as stored. The time is the first
    mark's."""
    at = record.get("deleted", {}).get("at") or datetime.now(UTC).isoformat()
    return dock_api.put_json(dock, f"{folder_of(record)}/{RECORD}",
                             {**record, "deleted": {"at": at, "why": why, "whole": whole}})


class Room:
    """The room the cap (bytes, None: no cap) leaves on the dock for the
    corpus being built in the folder own: the files of every other corpus
    count, at the sizes their records give, those of a corpus that keeps
    them and of one whose deletion is not whole, and so do the files of a
    folder of corpora/ that holds no record (left by a build that failed,
    was cancelled or was never resumed: its build, started again, finds them
    gone and starts over). last_download(corpus id) gives
    the time (seconds since the epoch, 0 for never) the corpus was last
    downloaded, busy() the ids of the corpora being downloaded now. The
    files are listed once, when the first room is asked for, and only this
    Room deletes any while the build runs."""

    def __init__(self, dock, cap: int | None, own: str, last_download=lambda corpus: 0.0, busy=lambda: set(),
                 log=lambda line: None):
        self.dock, self.cap, self.own = dock, cap, own
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
            return dock_api.size(self.dock, one["key"])
        except Exception:  # a file the dock no longer holds takes no room
            return 0

    def _listed(self) -> dict:
        if self._kept is None:
            self._kept = {}
            for name in dock_api.folders(self.dock, f"{FOLDER}/"):
                folder = f"{FOLDER}/{name}"
                if folder == self.own:
                    continue
                record = dock_api.get_json(self.dock, f"{folder}/{RECORD}")
                if record is None:
                    left = dock_api.find(self.dock, f"{folder}/")
                    if left:
                        self._kept[folder] = (None, left)
                    continue
                if record.get("deleted", {}).get("whole", "deleted" in record):
                    continue
                left = [one for one in files(record)
                        if "deleted" not in record or dock_api.has(self.dock, one["key"])]
                self._kept[record["id"]] = (record, [{**one, "bytes": self._size(one)} for one in left])
        return self._kept

    def taken(self) -> int:
        """The bytes the files of the other corpora take now, as make counts them."""
        return sum(one["bytes"] for _, left in self._listed().values() for one in left)

    def evicted(self, size: int) -> list[dict] | None:
        """What a build of size bytes in all would delete, as make would
        choose it, deleting nothing: [{corpus (its id, or the folder of
        files no record names), bytes}], each in the order make takes them,
        [] when it fits; None when the corpus alone passes the cap or
        nothing it may delete makes room (the build would fail)."""
        if self.cap is None:
            return []
        if size > self.cap:
            return None
        kept, busy = self._listed(), self.busy()
        over = self.taken() + size - self.cap
        chosen = []

        def rank(name):
            record = kept[name][0]
            if record is None:
                return (0, 0.0, "")
            return (1 if "deleted" in record else 2, self.last_download(name), record["built_at"])
        for name in sorted((name for name, (_, left) in kept.items() if left and name not in busy), key=rank):
            if over <= 0:
                break
            taken = sum(one["bytes"] for one in kept[name][1])
            freed = 0
            for one in kept[name][1]:
                if over - freed <= 0:
                    break
                freed += one["bytes"]
            chosen.append({"corpus": name, "bytes": freed, "whole": freed == taken})
            over -= freed
        return None if over > 0 else chosen

    def expect(self, size: int):
        """TooBig when the corpus being built, estimated at size bytes in
        all, passes the cap on its own: called before its first upload, so
        nothing is deleted for a corpus that cannot fit."""
        if self.cap is not None and size > self.cap:
            raise TooBig(f"this corpus would take about {size:,} bytes, more than the dock's cap of {self.cap:,} "
                         "bytes on its own: raise the cap or build it at a smaller size (max_tokens); no corpus "
                         "was deleted for it")

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
            raise TooBig(f"this corpus needs more than the dock's cap of {self.cap:,} bytes on its own "
                         f"({self.used + size:,} bytes so far): raise the cap or build it at a smaller size "
                         "(max_tokens)")
        kept = self._listed()
        taken = self.taken()
        while taken + self.used + size > self.cap:
            busy = self.busy()

            def rank(name):
                record = kept[name][0]
                if record is None:
                    return (0, 0.0, "")
                return (1 if "deleted" in record else 2, self.last_download(name), record["built_at"])
            free = sorted((name for name, (record, left) in kept.items() if left and name not in busy), key=rank)
            if not free:
                raise NoRoom(f"the dock's cap of {self.cap:,} bytes is reached ({taken:,} bytes of other corpora, "
                             f"{self.used + size:,} of this one) and every other corpus is being downloaded: build "
                             "again once a download ended")
            record, left = kept[free[0]]
            one = left.pop(0)
            if record is None:
                if free[0] not in self._said:
                    self._said.add(free[0])
                    self.log(f"the files of {free[0]} deleted, a file at a time as this build needs room: no corpus "
                             "record names them")
                dock_api.delete(self.dock, one["key"])
            else:
                if "deleted" not in record:
                    record = mark_deleted(self.dock, record, CAPPED)
                    self.log(f"corpus {record['id']} deleted, a file at a time as this build needs room: {CAPPED}")
                dock_api.delete(self.dock, one["key"])
                if not left:
                    record = mark_deleted(self.dock, record, record["deleted"]["why"], whole=True)
            taken -= one["bytes"]
            kept[free[0]] = (record, left)
        self.used += size
