"""A depot's snapshots: a name on a change number, the set of texts that
were admitted at that change, each read at its revision as of it
(wpt_depot.schema, tables snapshots and snapshot_sets).

A snapshot holds its texts as sets of numbers, the digits that end their
ids, one set for each part and each depot whose texts it holds (its owner:
a text's id is `<owner>-<number>`, wpt_depot.ids), stored as portable
roaring bitmaps. A set also has a bound: the change number, in the
numbering of its owner, its texts are read at, each at
`max(revision) WHERE change <= bound`, with the tags it had then (its rows
of taggings that cover the bound, schema.tags_at), so a text retagged after
the snapshot is read with the tags it had in it. A snapshot made here (create) has
one part, part 0, whose bound is the snapshot's change (at_change) for the
depot's own texts and, in a depot that follows an upstream, the change of
each other depot's numbering its pulls had brought by then (Depot.bounds).
A snapshot of several parts reads each number in the lowest part that
holds it, with that part's bound.

Making one (create) never holds a read transaction longer than a page of
the index texts_status takes to read, so daily writes, and an erasure's
checkpoint, which waits for every reader, are never held back:

1. It notes the depot's change C0 and reads the ids of the admitted
   current texts from the index texts_status, which holds them in the
   numeric order of their ids (ids.order_sql), PAGE ids a read
   transaction, each page after the last id of the page before. The index
   alone gives each id, so no revision row is read.
2. In one short read transaction it notes the change C1 and reads again
   the current status of every text that moved in between: the texts with a
   revision written in (C0, C1] (index texts_change), those with a decision
   linked in (C0, C1] (index text_decisions_change) and those named by an
   erasure written in (C0, C1]; each is put in or taken out of its owner's
   set. A pull in between brings rows of other numberings, which these
   indexes cannot find by C0, so the reading starts over (at most TRIES
   times). In the same transaction the summary is read from the counts
   (counts.of_status), which every write keeps exact, so no text is read,
   and refused unless the counts hold as many admitted texts as the sets.
   The snapshot reads C1: every status it read is the one stored at C1.
3. One short write transaction, a change of kind `snapshot`, writes its row
   and its sets, the name checked again.

A snapshot and its sets are written once and never changed nor removed:
an erasure leaves them as they are. A text an erasure removed whole has
no row left, so a read skips it and counts it (texts); a text it cut is
read cut.

The digest is the SHA-256, in hexadecimal, of these bytes, which depend
on the numbers alone, never on how a roaring library lays them out:

    b"wpt-snapshot-1\\n"
    the snapshot's at_change in decimal, b"\\n"
    then for each set, in the order of (part, owner), the owner's name
    compared by its UTF-8 bytes:
        part in decimal, b"\\n"
        owner's name in UTF-8, b"\\n"
        bound in decimal, b"\\n"
        how many numbers it holds in decimal, b"\\n"
        its numbers in increasing order, each 4 bytes, an unsigned
        integer in little-endian order

Decimal numbers are written in ASCII digits with no sign, no leading zero
and no separator (0 is "0"). A number above 2**32 - 1 cannot be held:
create refuses a depot that holds one admitted.

The summary (JSON) is {"version": 1, "kinds": {kind: [texts, characters,
bytes]}, "tags": {tag: [...]}, "plain_tagsets": {the tags of a plain text
sorted and joined by NUL ("" for none): [...]}, "licenses": {license:
[...]}, "owners": {owner: texts}}, each name in order: what the texts held
at the snapshot's change, characters and bytes those of the content
version each read then. count reads it alone.

The reads: snapshots and snapshot give the rows; sets the parts with their
bitmaps, checked against the digest once and then kept in this process
(the rows never change); texts the texts in the order of (owner, number),
a page at a time; erased_since what the erasures carried out after a
snapshot's bounds cut or removed of its texts.

Before one is made: next_name gives the name the pages offer (the year and
month, then the same with b, c, ... once taken), and estimate what making
one now takes (its texts, seconds and bytes), from the counts alone.
"""

import hashlib
import itertools
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from pyroaring import BitMap, FrozenBitMap

from . import counts, ids
from .database import now
from .errors import NotFound, Refused
from .filters import TEXTS_KEY
from .schema import ADMITTED, tags_at

MAGIC = b"wpt-snapshot-1\n"  # the first bytes the digest reads
PAGE = 50_000  # ids a read transaction of the scan reads
TRIES = 3  # how many times create reads again after a pull came in between
LARGEST = (1 << 32) - 1  # the largest number a set holds
KEPT = 8  # snapshots whose sets this process keeps read
READ_BATCH = 2_000  # ids texts reads in one statement
# the speed estimate takes for create: admitted texts read a second, measured at about 700,000 on 2M texts, and the
# seconds of the rest (catching up, the summary, writing)
RATE, FIXED_SECONDS = 500_000, 1.0
CONTAINER = 1 << 16  # the numbers one container of a roaring bitmap holds
SUMMARY_ROW_BYTES = 40  # the bytes of JSON a name of the summary takes besides its own (its three sizes, quotes)
COLUMNS = "id, change, name, at_change, made_at, origin, note, texts, characters, bytes, digest, summary"
LISTED = "id, change, name, at_change, made_at, origin, note, texts, characters, bytes, digest"
# the admitted current texts after a place in the numeric order of ids, from the index texts_status alone: a page of
# their places (ids.order_sql: the owner, char(1), '0' and the number in 20 digits, char(1), the id) given back, for
# each owner, as its numbers joined by commas, with how many, how many have a number, and the last place; SQLite does
# the work of each row, so the scan holds Python's lock for a few calls a page
PLACES = (f"SELECT {ids.order_sql()} AS k FROM texts_revisions INDEXED BY texts_status "
          f"WHERE status = '{ADMITTED}' AND latest AND {ids.order_sql()} > ? ORDER BY {ids.order_sql()} LIMIT ?")
SCAN = ("SELECT substr(k, 1, instr(k, char(1)) - 1), "
        "group_concat(CAST(substr(k, instr(k, char(1)) + 2, 20) AS INTEGER), ','), count(*), "
        "sum(substr(k, instr(k, char(1)) + 1, 1) = '0'), max(k) "
        f"FROM ({PLACES}) GROUP BY 1")
# the ids of the texts that moved in (?, ?]: a revision written, a decision linked
MOVED = ("SELECT id FROM texts_revisions INDEXED BY texts_change WHERE change > :low AND change <= :high "
         "UNION SELECT text FROM text_decisions INDEXED BY text_decisions_change WHERE change > :low "
         "AND change <= :high")
# each text asked ([id, bound] pairs as JSON) at its revision as of its bound, with the content version it reads and
# its kind, length and bytes, which the revision keeps (schema.SIZES)
AT_BOUND = ("WITH a AS MATERIALIZED (SELECT key, value ->> 0 AS text, value ->> 1 AS bound FROM json_each(?)) "
            "SELECT a.key, t.id, t.revision, t.change, t.content_version, t.content_kind, t.content_length, "
            "t.content_bytes, t.title, "
            f"{tags_at('t.id', 'a.bound')}, t.licenses FROM a CROSS JOIN texts_revisions t "
            "ON t.id = a.text AND t.revision = ("
            f"SELECT max(r.revision) FROM texts_revisions r INDEXED BY {TEXTS_KEY} WHERE r.id = a.text "
            "AND r.change <= a.bound) WHERE t.content_kind IS NOT NULL ORDER BY a.key")

# the same for the texts whose current revision is the one at their bound (its change at or below it), the most of
# them: each found by its place in the numeric order of ids ([place (ids.order_sql), bound] pairs as JSON) in the
# index texts_order, so the texts of a page, which follow that order, are read from neighbouring places of it
CURRENT_AT_BOUND = ("WITH a AS MATERIALIZED (SELECT key, value ->> 0 AS place, value ->> 1 AS bound "
                    "FROM json_each(?)) "
                    "SELECT a.key, t.id, t.revision, t.change, t.content_version, t.content_kind, t.content_length, "
                    "t.content_bytes, t.title, "
                    f"{tags_at('t.id', 'a.bound')}, t.licenses FROM a "
                    "CROSS JOIN texts_revisions t INDEXED BY texts_order "
                    f"ON {ids.order_sql('t.id')} = a.place AND t.latest AND t.change <= a.bound "
                    "WHERE t.content_kind IS NOT NULL")

_kept: OrderedDict = OrderedDict()  # (depot file, snapshot id, digest): _Read, the KEPT read last
_kept_lock = threading.Lock()


class _Read:
    """A snapshot's sets as read: parts, (part, owner, bound, bitmap) in the
    order of (part, owner); by_owner, {owner: [(part, bound, bitmap)] in
    the order of parts}; and each owner's numbers of every part together,
    made when first asked (union)."""

    def __init__(self, parts: list[tuple[int, str, int, FrozenBitMap]]):
        self.parts = parts
        self.by_owner: dict[str, list[tuple[int, int, FrozenBitMap]]] = {}
        for part, owner, bound, bitmap in parts:
            self.by_owner.setdefault(owner, []).append((part, bound, bitmap))
        self._unions = {}

    def union(self, owner: str) -> FrozenBitMap:
        if owner not in self._unions:
            found = self.by_owner[owner]
            self._unions[owner] = found[0][2] if len(found) == 1 else FrozenBitMap.union(*(one[2] for one in found))
        return self._unions[owner]

    def bound(self, owner: str, number: int) -> tuple[int, int]:
        """(part, bound) of the lowest part that holds the number of the owner."""
        for part, bound, bitmap in self.by_owner[owner]:
            if number in bitmap:
                return part, bound
        raise KeyError(number)


def check_name(name: str):
    """Refused unless name can name a snapshot: letters, digits, '.', '_'
    and '-', beginning with a letter or a digit."""
    if not isinstance(name, str) or not ids.NAME.fullmatch(name):
        raise Refused(f"a snapshot's name is made of letters, digits, '.', '_' and '-', beginning with a letter or a "
                      f"digit: {name!r}")


def digest_of(sets: Iterable[tuple[int, str, int, Iterable[int]]], at_change: int) -> str:
    """The digest (the module says how it is made) of the sets, (part,
    owner, bound, numbers) each, the numbers a bitmap or any iterable of
    distinct numbers, of a snapshot that reads at_change."""
    found = hashlib.sha256(MAGIC)
    found.update(f"{at_change}\n".encode())
    for part, owner, bound, numbers in sorted(sets, key=lambda one: (one[0], one[1].encode())):
        if isinstance(numbers, (BitMap, FrozenBitMap)):
            values = np.frombuffer(numbers.to_array(), dtype=np.uint32)
        else:
            values = np.unique(np.fromiter(numbers, dtype=np.uint64))
            if values.size and values[-1] > LARGEST:
                raise Refused(f"the number {int(values[-1])} is above {LARGEST}, which no snapshot holds")
            values = values.astype(np.uint32)
        found.update(f"{part}\n{owner}\n{bound}\n{values.size}\n".encode())
        found.update(values.astype("<u4", copy=False).tobytes())
    return found.hexdigest()


def create(depot, name: str, note: str | None = None, progress: Callable[[str], None] | None = None,
           read: Callable[[int], None] | None = None, page: int = PAGE, measured: dict | None = None) -> dict:
    """A new snapshot of the depot called name, made as the module says,
    with an optional note: its row (snapshot). progress, called with the
    name of each stage it starts ("reading texts", "catching up",
    "writing"); read, called after each page of the scan with how many ids
    it read so far, no read transaction open; measured, when given, receives
    {"pages", "longest_read" (seconds, the longest read transaction),
    "moved" (texts read again), "tries", "started" (C0), "at_change" (C1)}.
    Refused when the name is not one (check_name) or a snapshot of the
    depot has it already, when a pull keeps coming in between (TRIES), when
    an admitted text's number is above LARGEST, and when the counts do not
    hold as many admitted texts as were read (`wpt depot recount` makes them
    again)."""
    check_name(name)
    own = depot.name()
    if _named(depot.db.reader(), own, name):
        raise Refused(f"the depot has a snapshot named {name} already")
    progress = progress or (lambda stage: None)
    stats = {"pages": 0, "longest_read": 0.0, "moved": 0, "tries": 0}
    for _ in range(TRIES):
        stats["tries"] += 1
        progress("reading texts")
        sets, started = _scan(depot, page, read, stats)
        progress("catching up")
        caught = _catch_up(depot, sets, started, stats)
        if caught is not None:
            break
    else:
        raise Refused(f"the depot pulled from its upstream while the snapshot read its texts, {TRIES} times in a row: "
                      "make it again later")
    at_change, summary, bounds = caught
    for owner, bitmap in sets.items():
        bitmap.run_optimize()
    held = {owner: bitmap for owner, bitmap in sorted(sets.items()) if bitmap}
    summary["owners"] = {owner: len(bitmap) for owner, bitmap in held.items()}
    total = sum(len(bitmap) for bitmap in held.values())
    counted = sum(sizes[0] for sizes in summary["kinds"].values())
    if counted != total:
        raise Refused(f"the depot's counts hold {counted} admitted texts but {total} were read: run `wpt depot "
                      "recount`, then make the snapshot again")
    parts = []
    for owner, bitmap in held.items():
        if owner != own and owner not in bounds:
            raise Refused(f"the depot holds admitted texts of {owner}, a depot its pulls never brought rows of")
        parts.append((0, owner, at_change if owner == own else bounds[owner], bitmap))
    digest = digest_of(parts, at_change)
    sized = [sum(sizes[at] for sizes in summary["kinds"].values()) for at in (1, 2)]
    progress("writing")
    with depot.db.write("snapshot") as (con, number):
        if _named(con, own, name):
            raise Refused(f"the depot has a snapshot named {name} already")
        snapshot = depot._id(depot._take(con, "snapshots"))
        con.execute(f"INSERT INTO snapshots ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (snapshot, number, name, at_change, now(), _json({"kind": "made"}), note, total, sized[0],
                     sized[1], digest, _json({"version": 1, **summary})))
        con.executemany("INSERT INTO snapshot_sets (snapshot, part, owner, bound, bitmap, texts) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [(snapshot, part, owner, bound, bitmap.serialize(), len(bitmap))
                         for part, owner, bound, bitmap in parts])
    if measured is not None:
        measured.update(stats, started=started, at_change=at_change)
    return snapshot_by_id(depot, snapshot)


def _scan(depot, page: int, read, stats: dict) -> tuple[dict[str, BitMap], int]:
    """({owner: the numbers of its admitted current texts}, the change C0
    noted before), read from the index texts_status a page a read
    transaction (each one statement, read whole)."""
    con = depot.db.reader()
    started = con.execute("SELECT coalesce(max(id), 0) FROM changes").fetchall()[0][0]
    sets: dict[str, BitMap] = {}
    after, done = "", 0
    while True:
        began = time.monotonic()
        found = con.execute(SCAN, (after, page)).fetchall()
        stats["longest_read"] = max(stats["longest_read"], time.monotonic() - began)
        stats["pages"] += 1
        read_now = sum(one[2] for one in found)
        for owner, joined, many, numbered, last in found:
            if numbered != many:
                raise Refused(f"the depot holds an admitted text of {owner} with no number, which no snapshot holds")
            numbers = np.fromstring(joined, dtype=np.int64, sep=",")
            if numbers.size != many:
                raise Refused(f"the numbers of the admitted texts of {owner} after {after!r} could not be read")
            if numbers.max() > LARGEST:
                raise Refused(f"the text {owner}-{int(numbers.max())} has a number above {LARGEST}, which no snapshot "
                              "holds")
            sets.setdefault(owner, BitMap()).update(numbers.astype(np.uint32))
            after = max(after, last)
        done += read_now
        if read_now and read is not None:
            read(done)
        if read_now < page:
            return sets, started


def _catch_up(depot, sets: dict[str, BitMap], started: int, stats: dict) -> tuple[int, dict, dict] | None:
    """In one read transaction: the change C1, every text that moved since
    started put in or taken out of sets by its current status, the summary
    from the counts and the bounds of each depot at C1, as (C1, summary,
    bounds); None when a pull came in between, sets then left half caught
    up."""
    con = depot.db.reader()
    began = time.monotonic()
    con.execute("BEGIN")
    try:
        at_change = con.execute("SELECT coalesce(max(id), 0) FROM changes").fetchall()[0][0]
        if con.execute("SELECT 1 FROM changes WHERE id > ? AND id <= ? AND kind = 'pull' LIMIT 1",
                       (started, at_change)).fetchall():
            return None
        moved = {one for (one,) in con.execute(MOVED, {"low": started, "high": at_change})}
        for (needles,) in con.execute("SELECT needles FROM erasures_revisions WHERE change > ? AND change <= ?",
                                      (started, at_change)).fetchall():
            moved.update(one["text"] for one in json.loads(needles))
        admitted = {one for (one,) in con.execute(
            "SELECT t.id FROM json_each(?) j CROSS JOIN texts_revisions t ON t.id = j.value AND t.latest "
            f"WHERE t.status = '{ADMITTED}'", (_json(sorted(moved)),))}
        for text in moved:
            owner, number = ids.owner(text), ids.number(text)
            if number is None:
                continue
            if text in admitted:
                if number > LARGEST:
                    raise Refused(f"the text {text} has a number above {LARGEST}, which no snapshot holds")
                sets.setdefault(owner, BitMap()).add(number)
            elif owner in sets:
                sets[owner].discard(number)
        summary = counts.of_status(con, ADMITTED)
        bounds = depot.bounds(depot.name(), at_change)
    finally:
        con.execute("COMMIT")
        stats["longest_read"] = max(stats["longest_read"], time.monotonic() - began)
    stats["moved"] += len(moved)
    return at_change, summary, bounds


def _named(con, owner: str, name: str) -> bool:
    return bool(con.execute(f"SELECT 1 FROM snapshots WHERE {ids.owner_sql('id')} = ? AND name = ?",
                            (owner, name)).fetchall())


def snapshots(depot) -> list[dict]:
    """Every snapshot of the depot, those of each owner after those of the
    owners before it, each owner's in the order they were made, with its
    owner, without its summary."""
    return depot._rows(f"SELECT {LISTED}, {ids.owner_sql('id')} AS owner FROM snapshots "
                       f"ORDER BY {ids.order_sql('id')}")


def snapshot(depot, name: str, owner: str | None = None) -> dict:
    """The snapshot called name of the depot owner (this depot by
    default), with its owner and summary. NotFound when it has none."""
    owner = depot.name() if owner is None else owner
    found = depot._rows(f"SELECT {COLUMNS}, {ids.owner_sql('id')} AS owner FROM snapshots "
                        f"WHERE {ids.owner_sql('id')} = ? AND name = ?", (owner, name))
    if not found:
        raise NotFound(f"no snapshot {name} of {owner}")
    return found[0]


def owner_named(depot, address: str, public: str | None) -> str | None:
    """The name of the depot whose snapshots a recipe naming the depot by
    address means here: this depot's own when address is its public
    address public (else, without one, its name), its upstream's when it is
    the address of the upstream it follows; None for another address."""
    if address == (public or depot.name()):
        return depot.name()
    if address == depot.upstream():
        found = depot._rows("SELECT upstream FROM pulls WHERE depot = ? ORDER BY change DESC LIMIT 1",
                            (depot.name(),))
        return found[0]["upstream"] if found else None
    return None


def snapshot_by_id(depot, snapshot_id: str) -> dict:
    """The snapshot whose id is snapshot_id, as snapshot gives it. NotFound when there is none."""
    found = depot._rows(f"SELECT {COLUMNS}, {ids.owner_sql('id')} AS owner FROM snapshots WHERE id = ?",
                        (snapshot_id,))
    if not found:
        raise NotFound(f"no snapshot {snapshot_id}")
    return found[0]


def sets(depot, snapshot_id: str) -> list[tuple[int, str, int, FrozenBitMap]]:
    """The snapshot's sets, (part, owner, bound, bitmap) in the order of
    (part, owner): read once, checked against its digest, then kept
    (KEPT). Refused when they do not match the digest."""
    return _read(depot, snapshot_id).parts


def _read(depot, snapshot_id: str) -> _Read:
    con = depot.db.reader()
    found = con.execute("SELECT digest, at_change FROM snapshots WHERE id = ?", (snapshot_id,)).fetchall()
    if not found:
        raise NotFound(f"no snapshot {snapshot_id}")
    digest, at_change = found[0]
    key = (str(depot.db.path), snapshot_id, digest)
    with _kept_lock:
        if key in _kept:
            _kept.move_to_end(key)
            return _kept[key]
    parts = [(part, owner, bound, FrozenBitMap.deserialize(bitmap)) for part, owner, bound, bitmap in con.execute(
        "SELECT part, owner, bound, bitmap FROM snapshot_sets WHERE snapshot = ? ORDER BY part, owner",
        (snapshot_id,)).fetchall()]
    parts.sort(key=lambda one: (one[0], one[1].encode()))
    if digest_of(parts, at_change) != digest:
        raise Refused(f"the sets of the snapshot {snapshot_id} do not match its digest: the depot's file is damaged")
    read = _Read(parts)
    with _kept_lock:
        _kept[key] = read
        while len(_kept) > KEPT:
            _kept.popitem(last=False)
    return read


def texts(depot, snapshot_id: str, kinds: Sequence[str] | None = None, after: str | None = None,
          limit: int = 1000) -> dict:
    """A page of the snapshot's texts, in the order of (owner, number):
    {texts: up to limit rows {id, part, revision, change, content_version,
    kind, length, bytes, title, tags, licenses}, each at its revision as of
    its part's bound, the kind that of the content version that revision
    reads, the tags those it had at that bound; erased: how many numbers the page went past that have no row
    left (texts an erasure removed whole); after: the id to give as after
    for the next page, None after the last}. kinds: only texts of those
    kinds. after: the page starts after that id."""
    read = _read(depot, snapshot_id)
    wanted = None if kinds is None else set(kinds)
    con = depot.db.reader()
    rows, erased = [], 0
    start_owner, start_number = (None, None) if after is None else (ids.owner(after), ids.number(after))
    for owner in sorted(read.by_owner, key=str.encode):
        if start_owner is not None and owner.encode() < start_owner.encode():
            continue
        union = read.union(owner)
        if owner == start_owner and start_number is not None:
            numbers = union.iter_equal_or_larger(start_number + 1)
        else:
            numbers = iter(union)
        while True:
            batch = []
            for number in numbers:
                batch.append(number)
                if len(batch) >= READ_BATCH:
                    break
            if not batch:
                break
            asked = [(ids.make(owner, number), *read.bound(owner, number)) for number in batch]
            found = _at_bounds(con, owner, batch, asked)
            for at, (text, part, _) in enumerate(asked):
                row = found.get(at)
                if row is None:
                    erased += 1
                    continue
                if wanted is not None and row[4] not in wanted:
                    continue
                rows.append(dict(zip(("id", "revision", "change", "content_version", "kind", "length", "bytes",
                                      "title", "tags", "licenses"), row)) | {"part": part})
                rows[-1]["tags"] = json.loads(rows[-1]["tags"])
                rows[-1]["licenses"] = json.loads(rows[-1]["licenses"])
                if len(rows) >= limit:
                    return {"texts": rows, "erased": erased, "after": text}
    return {"texts": rows, "erased": erased, "after": None}


def _at_bounds(con, owner: str, numbers: list[int], asked: list[tuple[str, int, int]]) -> dict[int, list]:
    """{position in asked: the row of AT_BOUND less its key} of the texts
    asked, (id, part, bound) each, of the owner's numbers: first those
    whose current revision is the one at their bound (CURRENT_AT_BOUND),
    then the others by AT_BOUND; a text with no row left is left out."""
    head = f"{owner}\x01"
    found = {at: row for at, *row in con.execute(CURRENT_AT_BOUND, (_json(
        [[f"{head}0{number:020d}\x01{text}", bound] for number, (text, _, bound) in zip(numbers, asked)]),))}
    rest = [at for at in range(len(asked)) if at not in found]
    if rest:
        for at, *row in con.execute(AT_BOUND, (_json([[asked[at][0], asked[at][2]] for at in rest]),)):
            found[rest[at]] = row
    return found


def count(depot, snapshot_id: str, kinds: Sequence[str] | None = None) -> dict:
    """{texts, characters, bytes} of the snapshot's texts of the kinds
    named (every kind by default), from its summary alone."""
    found = depot.db.reader().execute("SELECT summary ->> '$.kinds' FROM snapshots WHERE id = ?",
                                      (snapshot_id,)).fetchall()
    if not found:
        raise NotFound(f"no snapshot {snapshot_id}")
    held = json.loads(found[0][0])  # the kinds alone, not the whole summary, which its tag sets make large
    chosen = held if kinds is None else {kind: held.get(kind, [0, 0, 0]) for kind in kinds}
    return dict(zip(("texts", "characters", "bytes"), (sum(sizes[at] for sizes in chosen.values())
                                                       for at in range(3))))


def next_name(depot, today, taken: Iterable[str] = ()) -> str:
    """The name the pages offer for the next snapshot of the depot made on
    the date today: its year and month (2026-11), else the first of
    2026-11b, 2026-11c, ..., 2026-11z, 2026-11ba, ... that neither one of
    the depot's own snapshots nor taken (names being made) has."""
    own = depot.name()
    used = {row["name"] for row in snapshots(depot) if row["owner"] == own} | set(taken)
    base = f"{today.year:04d}-{today.month:02d}"
    if base not in used:
        return base
    for number in itertools.count(2):  # b, c, ..., z, aa, ab, ...: the letters of number in bijective base 26
        letters, left = "", number
        while left:
            left, at = divmod(left - 1, 26)
            letters = chr(ord("a") + at) + letters
        if f"{base}{letters}" not in used:
            return f"{base}{letters}"


def estimate(depot) -> dict:
    """What making a snapshot now takes, from the counts alone: {texts (the
    admitted texts now), seconds (at RATE, plus FIXED_SECONDS), bytes (its
    sets and summary: set_bytes, summary_bytes), at: what change it reads}.
    The sets' bytes are an upper bound from how many texts and how far
    their numbers spread: each container of CONTAINER numbers the depot
    handed out holds its texts as 2 bytes each, at most 8 KiB."""
    con = depot.db.reader()
    texts = (con.execute("SELECT texts FROM counts WHERE facet = ? AND name = ?",
                         (counts.STATUS, ADMITTED)).fetchall() or [(0,)])[0][0]
    handed = (con.execute("SELECT last FROM numbers WHERE name = 'texts'").fetchall() or [(0,)])[0][0]
    containers = max(1, -(-max(handed, texts) // CONTAINER)) if texts else 0
    per = texts / containers if containers else 0
    set_bytes = int(containers * (min(2 * per, 8192) + 8))
    low, high = f"{ADMITTED} ", f"{ADMITTED}!"
    summary_bytes = 0
    for facet in (counts.KIND, counts.SPLIT_TAG, counts.PLAIN_TAGSET, counts.SPLIT_LICENSE):
        rows, names = con.execute("SELECT count(*), coalesce(sum(length(name)), 0) FROM counts WHERE facet = ? "
                                  "AND name >= ? AND name < ?", (facet, low, high)).fetchone()
        summary_bytes += names + rows * (SUMMARY_ROW_BYTES - len(low))
    return {"texts": texts, "seconds": round(FIXED_SECONDS + texts / RATE, 1), "bytes": set_bytes + summary_bytes,
            "set_bytes": set_bytes, "summary_bytes": summary_bytes,
            "at": "the change current when it finishes: texts admitted or refused meanwhile are read as they are then",
            "basis": f"approximate: {RATE:,} admitted texts read a second, plus {FIXED_SECONDS:g} second; the bytes "
                     "of its sets at most 2 per text"}


def erased_since(depot, snapshot_id: str) -> dict:
    """What the erasures carried out after the snapshot's bounds did to its
    texts: {texts: how many of its texts they cut or removed, erasures:
    their ids}. An erasure counts when its carrying out (the change of its
    revision that holds what it erased, in the numbering of the depot that
    owns it) is above every bound the snapshot has for that depot, or the
    snapshot reads no text of that depot."""
    read = _read(depot, snapshot_id)
    bounds = {}
    for _, owner, bound, _ in read.parts:
        bounds[owner] = max(bound, bounds.get(owner, bound))
    touched, named = set(), []
    for row in depot.erasures():
        if row["erased"] is None:
            continue
        owner = ids.owner(row["id"])
        if owner in bounds and row["change"] <= bounds[owner]:
            continue
        held = {needle["text"] for needle in row["needles"] if _holds(read, needle["text"])}
        if held:
            touched |= held
            named.append(row["id"])
    return {"texts": len(touched), "erasures": named}


def _holds(read: _Read, text: str) -> bool:
    owner, number = ids.owner(text), ids.number(text)
    return owner in read.by_owner and number is not None and number in read.union(owner)


def last_erasure(depot) -> int:
    """A number that rises with every erasure revision the depot gains,
    written here or brought by a pull (0 without one): what erased_since's
    answer depends on besides the snapshot. It is the highest rowid of the
    erasure revisions, not their highest change, since a pull brings
    revisions that keep their upstream's change numbers, which can be lower
    than this depot's own."""
    return depot.db.reader().execute("SELECT coalesce(max(rowid), 0) FROM erasures_revisions").fetchone()[0]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
