"""The corpora the dock keeps under corpora/, built by the depot's build
before the freight delivered corpora: their records, the corpus of a build
key, marking those an erasure reached stale, deleting a corpus's files, and
the room a cap leaves. WPT reads them through freight.api; nothing here
builds one.

Each corpus is a folder corpora/<build key>/ of the dock (a corpus built
from a dataset before snapshots: corpora/<dataset name>-<8 hex
characters>/), its record corpus.json written last (folder_of). A corpus is
fresh while its record says neither `stale` (an admitted erasure reached a
text it holds: the ids of the erasures) nor `deleted` (its files are gone:
when, and why), and holds its token files by kind (by_kind; a record of
another layout is never fresh: training cannot read it). A deleted corpus
keeps its record.

A corpus of a snapshot was built for a build key: its snapshot, its data
section in the form canonical gives, its vocabulary and the version of the
selection rules then, RULES. A corpus built from a dataset before snapshots
was built for a triple (triple): its dataset, its vocabulary (by sha256) and
its size; those are found by their dataset (built_from, latest,
latest_fresh), and a corpus of a build key belongs to no dataset.
A cap (Room) bounds the bytes the corpora's files take together, as the
records give each file's size: past it, a build deletes the corpus
downloaded least recently, a file at a time, as it needs room, never one
being built or downloaded; a build that alone needs more than the cap fails
(TooBig), and so does one that finds nothing it may delete (NoRoom).

    records(store)                       every corpus the dock holds, newest first
    find(store, corpus_id)               the corpus of that id (or of a start of it), None without one
    of_key(store, key)                   the corpus of a build key, None without one
    canonical(data), RULES               a data section in the form a build key covers, and the rules' version
    triple(record)                       a dataset's corpus: (its dataset's id, its vocabulary's sha256, its size)
    built_from(store, depot, dataset)    every corpus built for that depot's dataset (by id), latest first
    latest(store, depot, dataset)        the corpus last built for that depot's dataset (by id), None without one
    latest_fresh(store, depot, dataset)  the same, of the fresh ones
    read_table(store, record, table)     one of a corpus's tables
    mark_stale(store, texts, erasures)   mark every fresh corpus that holds one of these texts stale
    holds(store, record, texts)          whether a corpus holds one of these texts
    delete(store, record, why)           delete a corpus's files, its record kept and marked
"""

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
from dock.api import PACKED, File, as_record_name

from .selection import Refused

FOLDER = "corpora"
RECORD = "corpus.json"
CAPPED = "the dock's cap needed its room"
ANY = "any"  # a size that keeps the corpora of every size (built_from)
RULES = 8  # the version of the selection rules the build of a corpus of a build key followed, which its key covers


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


def missing(store, folder: str, uploaded: list[dict]) -> list[str]:
    """The keys of the files uploaded ({key, ...}) that the folder of the dock no longer holds, from one listing."""
    held = set(store.kind.list(f"{folder}/", deep=True))
    return [one["key"] for one in uploaded if one["key"] not in held and one["key"] + PACKED not in held]


def folder_of(record: dict) -> str:
    """The dock's folder of a corpus: the one its record names, or for a
    record that names none, corpora/<dataset name>-<first 8 hex characters of its id>."""
    return record.get("folder") or f"{FOLDER}/{as_record_name(record['dataset']['name'])}-{record['id'][:8]}"


def fresh(record: dict) -> bool:
    return "stale" not in record and "deleted" not in record and by_kind(record)


def by_kind(record: dict) -> bool:
    """Whether the record holds its token files by kind ({kind: [token file]})."""
    return isinstance(record.get("shards", {}), dict)


def token_files(record: dict) -> list[dict]:
    """Every token file of a corpus, of every kind, as its record names each; a list of a record of another layout
    as it is."""
    shards = record.get("shards", {})
    return [one for kept in shards.values() for one in kept] if isinstance(shards, dict) else list(shards)


def files(record: dict) -> list[dict]:
    """Every file of a corpus but its record, as the record names each ({key, sha256} and maybe bytes)."""
    tables = record.get("tables", {})
    found = [*token_files(record), *record.get("conversation_shards", [])]
    for table in tables.values():
        found += table if isinstance(table, list) else [table]
    if "vocabulary" in record.get("tokenizer", {}):
        found.append(record["tokenizer"]["vocabulary"])
    if "tasks" in record:
        found.append(record["tasks"]["file"])
    return found


def triple(record: dict) -> tuple[str | None, str, int | None]:
    """What a corpus was built for: (its dataset's id, None for a corpus of
    a build key, the sha256 of its vocabulary, its size: max_tokens, None
    for none). A record that names no size of its own was built at its
    dataset's max_tokens, and one that names no vocabulary sha256 of its own
    has its vocabulary file's."""
    dataset = record.get("dataset") or {}
    size = record["max_tokens"] if "max_tokens" in record else dataset.get("max_tokens")
    vocabulary = record["tokenizer"].get("sha256") or record["tokenizer"]["vocabulary"]["sha256"]
    return dataset.get("id"), vocabulary, size


def canonical(data: dict) -> dict:
    """What a data section says of its corpus, in the form a build key
    covers: {kinds: the kinds named, sorted, each once ([]: every kind),
    shares: [{tag, share}] in order ([]: none), total_tokens: a number of
    tokens above 0, or None}. Refused for a section no corpus was built
    for: kinds not a list of names, shares whose tags repeat or whose share
    is below 0 or not a number, or whose shares are all 0, a size that is
    not a whole number above 0."""
    kinds = data.get("kinds") or []
    if not isinstance(kinds, list) or not all(isinstance(one, str) and one for one in kinds):
        raise Refused(f"the data section's kinds are a list of kinds' names: {kinds!r}")
    shares = data.get("shares") or []
    if not isinstance(shares, list) or not all(isinstance(one, dict) and isinstance(one.get("tag"), str)
                                               and isinstance(one.get("share"), (int, float))
                                               and not isinstance(one.get("share"), bool) and one["share"] >= 0
                                               for one in shares):
        raise Refused(f"the data section's shares are a list of {{tag, share}}, each share a number of 0 or more: "
                      f"{shares!r}")
    tags = [one["tag"] for one in shares]
    if len(set(tags)) != len(tags):
        raise Refused(f"the data section's shares name a tag twice: {', '.join(tags)}")
    if shares and not sum(one["share"] for one in shares):
        raise Refused("the data section's shares are all 0: give one a share above 0, or give no shares")
    size = data.get("total_tokens")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise Refused(f"a corpus's size (total_tokens) is a whole number of tokens above 0, or none: {size!r}")
    return {"kinds": sorted(set(kinds)), "shares": [{"tag": one["tag"], "share": float(one["share"])}
                                                    for one in shares], "total_tokens": size}


def records(store) -> list[dict]:
    """The record of every corpus the dock holds, each checked by the
    dock, the last built first."""
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


def of_key(store, key: str) -> dict | None:
    """The record of the corpus of that build key (a key canonical, RULES, the snapshot and the vocabulary made), None when the dock holds none."""
    return store.get_json(f"{FOLDER}/{key}/{RECORD}")


def built_from(store, depot: str, dataset: str, vocabulary: str | None = None, size=ANY) -> list[dict]:
    """The records of the corpora built from the dataset of id dataset of
    the depot of that name (depot.legacy.depot_name), latest first: those of the dataset's newest
    revision first, each revision's last built first, so a corpus of an
    older revision built again never comes before one of a newer revision.
    A dataset's corpora stay its own through a rename, and a new dataset
    given an old name has none. vocabulary (a sha256) and size (max_tokens,
    None for none) keep only the corpora built for them (triple); by
    default, every vocabulary and size."""
    found = [record for record in records(store)
             if record["depot"] == depot and (record.get("dataset") or {}).get("id") == dataset
             and (vocabulary is None or triple(record)[1] == vocabulary)
             and (size is ANY or triple(record)[2] == size)]
    return sorted(found, key=lambda record: (record["dataset"].get("revision", 0), record["built_at"]), reverse=True)


def latest(store, depot: str, dataset: str, vocabulary: str | None = None, size=ANY) -> dict | None:
    """The record of the corpus last built for the dataset of id dataset of
    the depot of that name (built_from, which says what vocabulary and size
    keep); None when the dock holds none."""
    return next(iter(built_from(store, depot, dataset, vocabulary, size)), None)


def latest_fresh(store, depot: str, dataset: str, vocabulary: str | None = None, size=ANY) -> dict | None:
    """The record of the fresh corpus last built for that dataset (and
    vocabulary and size, as built_from says); None without one."""
    return next((record for record in built_from(store, depot, dataset, vocabulary, size) if fresh(record)), None)


def read_table(store, record: dict, table: str) -> pa.Table:
    """One of a corpus's tables (texts, licenses, notices), each
    file checked by the dock: the texts table is the concatenation of its
    parts, in order."""
    found = record["tables"][table]
    if isinstance(found, list):
        return pa.concat_tables([pq.read_table(store.path(one)) for one in found]) if found else pa.table({})
    return pq.read_table(store.path(found))


def mark_stale(store, texts, erasures) -> list[str]:
    """Mark stale every corpus of the dock whose files are kept and that
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
        except Exception:  # a file the dock no longer holds takes no room
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
