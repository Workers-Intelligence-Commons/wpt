"""The order in which the freight reads a snapshot's texts (depot.snapshots),
which they are given as its listing (depot.api.listing, the entries of
a walk): every text the snapshot holds, by its export key (depot.ids.export_key),
but for its long plain texts, each of which comes as its pieces
(depot.pieces), each piece at the export key of its name. The key is
made from a text's id alone, or a piece's name, so the order is fixed and
shuffled at once: it depends on no order, size, vocabulary or depot file,
a text and each of its pieces keep their place whatever the snapshot's
other texts, and the pieces of a long text spread over the whole order, so
the first part of the order holds about its share of each long text.

An Order (of) is made in memory from the snapshot's sets, never from a
table of stored keys: each owner's numbers taken from its sets in the order
of their parts, a number held by several parts kept in the lowest one (first
part wins), with that part's bound; the export key of each id worked out;
the long texts found (below) and their pieces added; then everything sorted
by (key, owner, number, piece), a whole text's piece being WHOLE. The
sort holds one int64 an entry: the key's high bits over the entry's
position in its low ones, sorted in place; entries whose high bits are the
same (a few hundred pairs among 22 million) are then put in the order of
their whole key, owner, number and piece. Once sorted, an Order holds 8
bytes an entry (number, 4; set and piece, 2 each) and no key, 10 when a
text has more than 32,768 pieces (more than 34 billion characters), whose
numbers take 4 bytes each: an int32 holds the number of any piece of any
text a depot holds, since a content version's list of parts is one SQLite
value (schema.SQL_STRING bytes, about 90 a part) and a part holds at most
parts.SIZES.maximum bytes, so a text holds fewer than 8 × 10^14 characters,
fewer than 800 million pieces. The type changes no place of the order. A
walk works out each
entry's key as it reads it, and a place in the order is found by a binary
search over keys worked out the same way. While it is made, an entry costs
16 bytes at most, beside a few megabytes for the keys of CHUNK entries.

A text comes as pieces when its revision as of its bound reads a plain
content version longer than depot.pieces.LENGTH characters. Those are
found from the depot's index of long content versions
(content_lists_long), whose read grows with the depot's long contents,
never with the snapshot's texts: each such version's text, when the
snapshot holds it, is read at its bound.

A walk (walk) reads the order from a place on, PAGE entries at a time:
each whole text's revision as of its bound, its kind (of the content version
that revision reads), its tags as of its bound (schema.tags_at) and bytes,
in one statement a page. A number
with no row left (an erasure removed the text whole) is passed by. Nothing
past the last entry a walk gives is read, and it starts after any place it
gave before (after), which is how a build goes on from its last upload.

Each Entry is a text or a piece with its key, owner (the depot whose
numbering the id is in), number, id, piece (its number, None for a whole
text), revision, kind (depot.kinds), bytes (its content's UTF-8 bytes;
a piece's, its share of its text's bytes, split evenly among its pieces,
since they are known only once it is read) and tags (its tags as JSON
text; a piece's, its text's).

    of(depot, snapshot)                       the order of a snapshot's texts, as an Order
    sampled(depot, snapshot, points)          some of its texts, spread over its numbers, in order
    walk(order, after)                        its texts and pieces, in order, as Entry
"""

import bisect
import json
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from . import ids, pieces, snapshots
from .filters import TEXTS_KEY
from .ids import export_key
from .kinds import PLAIN
from .schema import LONG, tags_at

PAGE = 2_000  # whole texts a walk reads in one statement
CHUNK = 1 << 16  # entries whose keys are worked out, or whose sorted place is read, at a time
LOW_BITS = 1  # the fewest low bits of a sorted int64 that hold an entry's position (tests raise it to force ties)
WHOLE = -1  # what a whole text's piece holds where a piece's holds its number
# each text asked ([id, bound] pairs as JSON) at its revision as of its bound: (its place among those asked,
# revision, kind and length and bytes of the content version it reads, which the revision keeps (schema.SIZES), its
# tags as of its bound)
AT_BOUND = ("WITH a AS MATERIALIZED (SELECT key, value ->> 0 AS text, value ->> 1 AS bound FROM json_each(?)) "
            f"SELECT a.key, t.revision, t.content_kind, t.content_length, t.content_bytes, "
            f"{tags_at('t.id', 'a.bound')} FROM a CROSS JOIN "
            "texts_revisions t ON t.id = a.text AND t.revision = (SELECT max(r.revision) FROM "
            f"texts_revisions r INDEXED BY {TEXTS_KEY} WHERE r.id = a.text AND r.change <= a.bound) "
            "WHERE t.content_kind IS NOT NULL")
LONG_VERSIONS = (f"SELECT DISTINCT id FROM content_lists INDEXED BY content_lists_long WHERE kind = '{PLAIN}' "
                 f"AND length > {LONG}")


class Entry(NamedTuple):
    key: int
    owner: str
    number: int
    id: str
    piece: int | None
    revision: int
    kind: str
    bytes: int
    tags: str

    def place(self) -> list:
        """The entry's place in the order, [key, owner, number, piece (WHOLE for a whole text)], what a walk's
        after names."""
        return [self.key, self.owner, self.number, WHOLE if self.piece is None else self.piece]


class Long(NamedTuple):
    """A long text whose pieces a walk gives: its revision, tags (JSON text), length, bytes and number of pieces."""
    revision: int
    tags: str
    length: int
    bytes: int
    count: int


class Order:
    """A snapshot's texts and pieces in order, as the module says: sets,
    [(owner, bound)] each set an entry may name; the arrays numbers,
    owner_sets (each entry's set) and piece (WHOLE for a whole text), one
    item an entry; long, {(set, number): Long} of the texts that come as
    pieces; depot, the depot read, and address, its file."""

    def __init__(self, depot, sets: list[tuple[str, int]], numbers: np.ndarray, owner_sets: np.ndarray,
                 piece: np.ndarray, long: dict):
        self.depot, self.address, self.sets = depot, depot.address, sets
        self.numbers, self.owner_sets, self.piece, self.long = numbers, owner_sets, piece, long

    def __len__(self) -> int:
        return len(self.numbers)

    def whole(self) -> int:
        """How many texts the order holds, a text that comes as pieces counted once."""
        return int(np.count_nonzero(self.piece == WHOLE)) + len(self.long)

    def text(self, at: int) -> str:
        """The id of the text of the entry at that position."""
        return ids.make(self.sets[int(self.owner_sets[at])][0], int(self.numbers[at]))

    def place(self, at: int) -> tuple:
        """The entry at that position as (key, owner's UTF-8 bytes, number, piece): what the order sorts by."""
        owner, number, piece = self.sets[int(self.owner_sets[at])][0], int(self.numbers[at]), int(self.piece[at])
        text = ids.make(owner, number)
        return export_key(text if piece == WHOLE else pieces.name(text, piece)), owner.encode(), number, piece

    def start(self, after: Sequence | None) -> int:
        """The position of the first entry after the place after (Entry.place), 0 without one."""
        if after is None:
            return 0
        key, owner, number, piece = after
        return bisect.bisect_right(range(len(self)), (key, owner.encode(), number, piece), key=self.place)

    def subset(self, positions: Sequence[int]) -> "Order":
        """The order of the entries at those positions alone, in order."""
        at = np.asarray(sorted(positions), np.int64)
        return Order(self.depot, self.sets, self.numbers[at], self.owner_sets[at], self.piece[at], self.long)

    def spread(self, points: int) -> list[int]:
        """At most points positions spread evenly over the order: every one when it holds no more."""
        if len(self) <= points:
            return list(range(len(self)))
        return sorted({int(at) for at in np.linspace(0, len(self) - 1, points)})


def _ranks(owners: list[str]) -> list[int]:
    """Each owner's rank among the owners ordered by their UTF-8 bytes."""
    ordered = {owner: at for at, owner in enumerate(sorted(set(owners), key=str.encode))}
    return [ordered[owner] for owner in owners]


def of(depot, snapshot: str) -> Order:
    """The order of the texts of depot's snapshot whose id is snapshot, as the module says."""
    parts = snapshots.sets(depot, snapshot)
    sets, found, held = [], [], {}
    for _, owner, bound, bitmap in parts:  # in the order of (part, owner): a number keeps its lowest part
        seen = held.get(owner)
        fresh = bitmap if seen is None else bitmap - seen
        held[owner] = bitmap if seen is None else seen | bitmap
        found.append(np.frombuffer(fresh.to_array(), np.uint32) if len(fresh) else np.zeros(0, np.uint32))
        sets.append((owner, bound))
    starts = [0]
    for count in map(len, found):
        starts.append(starts[-1] + count)
    numbers = np.concatenate(found) if found else np.zeros(0, np.uint32)
    found = None  # each set's array freed: the numbers are all in one now
    base = len(numbers)
    long = _long(depot, sets, held, numbers, starts)
    cut, extra = [], []  # the positions of the long texts' whole entries; (key, number, set, piece) of their pieces
    for (index, number), one in long.items():
        low, high = starts[index], starts[index + 1]
        cut.append(low + int(np.searchsorted(numbers[low:high], number)))
        text = ids.make(sets[index][0], number)
        extra.extend((export_key(pieces.name(text, n)), number, index, n) for n in range(one.count))
    size = base + len(extra)
    bits = max(LOW_BITS, (size - 1).bit_length())
    low_mask = (1 << bits) - 1
    high_mask = np.int64(~low_mask)
    keyed = np.empty(size, np.int64)  # each entry's key's high bits over its position
    for index, (owner, _) in enumerate(sets):
        prefix = ids.prefix(owner)
        for low in range(starts[index], starts[index + 1], CHUNK):
            high = min(low + CHUNK, starts[index + 1])
            keys = np.fromiter((export_key(f"{prefix}{number}") for number in numbers[low:high].tolist()), np.int64,
                               high - low)
            keyed[low:high] = (keys & high_mask) | np.arange(low, high, dtype=np.int64)
    for at, (key, *_) in enumerate(extra):
        keyed[base + at] = (key & int(high_mask)) | (base + at)
    keyed.sort()
    ranks = _ranks([owner for owner, _ in sets])
    extra_numbers = np.array([number for _, number, _, _ in extra], np.uint32)
    extra_sets = np.array([index for _, _, index, _ in extra], np.int16 if len(sets) < (1 << 15) else np.int32)
    extra_pieces = np.array([n for _, _, _, n in extra], _piece_type(long))
    bounds = np.array(starts[1:], np.int64)

    def entry(position: int) -> tuple:
        """The entry at that position before the sort, as the sort orders it: (key, owner's rank, number, piece)."""
        if position >= base:
            key, number, index, n = extra[position - base]
            return key, ranks[index], number, n
        index = int(np.searchsorted(bounds, position, "right"))
        number = int(numbers[position])
        return export_key(ids.make(sets[index][0], number)), ranks[index], number, WHOLE
    for first, last in _ties(keyed, bits):
        positions = (keyed[first:last] & low_mask).tolist()
        positions.sort(key=entry)
        keyed[first:last] = (keyed[first:last] & high_mask) | np.array(positions, np.int64)
    dropped = np.array(sorted(cut), np.int64)
    kept = size - len(dropped)
    sorted_numbers = np.empty(kept, np.uint32)
    for start, end, positions in _positions(keyed, low_mask, dropped):
        inside = positions < base
        chunk = np.empty(len(positions), np.uint32)
        chunk[inside] = numbers[positions[inside]]
        chunk[~inside] = extra_numbers[positions[~inside] - base]
        sorted_numbers[start:end] = chunk
    numbers = None  # its memory freed before the sets and pieces take theirs
    owner_sets = np.empty(kept, extra_sets.dtype)
    piece = np.empty(kept, extra_pieces.dtype)
    for start, end, positions in _positions(keyed, low_mask, dropped):
        inside = positions < base
        chunk_sets = np.empty(len(positions), extra_sets.dtype)
        chunk_sets[inside] = np.searchsorted(bounds, positions[inside], "right")
        chunk_sets[~inside] = extra_sets[positions[~inside] - base]
        owner_sets[start:end] = chunk_sets
        chunk_piece = np.full(len(positions), WHOLE, extra_pieces.dtype)
        chunk_piece[~inside] = extra_pieces[positions[~inside] - base]
        piece[start:end] = chunk_piece
    return Order(depot, sets, sorted_numbers, owner_sets, piece, long)


def _piece_type(long: dict) -> type:
    """The type of the pieces' numbers of an order whose long texts are long ({(set, number): Long}), as the
    module says: int16 while every text has at most 32,768 pieces, else int32."""
    return np.int16 if max((one.count for one in long.values()), default=0) <= 1 << 15 else np.int32


def _ties(keyed: np.ndarray, bits: int) -> Iterator[tuple[int, int]]:
    """Each run [first, last) of two or more neighbours of the sorted keyed whose bits above the low bits are the
    same, in order, read CHUNK at a time."""
    first, last = None, 0
    for low in range(0, max(len(keyed) - 1, 0), CHUNK):
        high = min(low + CHUNK + 1, len(keyed))
        part = keyed[low:high] >> bits
        for at in np.flatnonzero(part[1:] == part[:-1]).tolist():
            at += low
            if first is None or at > last - 1:
                if first is not None:
                    yield first, last
                first = at
            last = at + 2
    if first is not None:
        yield first, last


def _positions(keyed: np.ndarray, low_mask: int, dropped: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
    """The positions the sorted keyed holds, without those dropped, CHUNK at a time, each chunk with the place it
    takes among those kept: (start, end, positions)."""
    start = 0
    for low in range(0, len(keyed), CHUNK):
        positions = keyed[low:low + CHUNK] & low_mask
        if len(dropped):
            positions = positions[~np.isin(positions, dropped)]
        yield start, start + len(positions), positions
        start += len(positions)


def sampled(depot, snapshot: str, points: int) -> Order:
    """At most points texts of depot's snapshot whose id is snapshot, in
    the order of their export keys, each whole: those at evenly spaced
    ranks among its numbers (each owner's numbers in increasing order, the
    owners in the order of their sets, a number kept in its lowest part), so
    the same for the same snapshot, made without the export key of every
    text: what an estimate reads of a big snapshot."""
    parts = snapshots.sets(depot, snapshot)
    sets, fresh, held = [], [], {}
    for _, owner, bound, bitmap in parts:
        seen = held.get(owner)
        fresh.append(bitmap if seen is None else bitmap - seen)
        held[owner] = bitmap if seen is None else seen | bitmap
        sets.append((owner, bound))
    sizes = np.array([len(one) for one in fresh], np.int64)
    total = int(sizes.sum())
    ranks = range(total) if total <= points else sorted({int(at) for at in np.linspace(0, total - 1, points)})
    starts = np.concatenate([[0], np.cumsum(sizes)])
    owner_sets, numbers = [], []
    for rank in ranks:
        index = int(np.searchsorted(starts, rank, "right")) - 1
        owner_sets.append(index)
        numbers.append(fresh[index][rank - int(starts[index])])
    keys = np.array([export_key(ids.make(sets[index][0], number)) for index, number in zip(owner_sets, numbers)],
                    np.int64)
    numbers = np.array(numbers, np.uint32)
    owner_sets = np.array(owner_sets, np.int16 if len(sets) < (1 << 15) else np.int32)
    ranks_of = np.array(_ranks([owner for owner, _ in sets]), np.int32)
    order = np.lexsort((numbers, ranks_of[owner_sets] if len(sets) else owner_sets, keys))
    return Order(depot, sets, numbers[order], owner_sets[order], np.full(len(keys), WHOLE, np.int16), {})


def _long(depot, sets, held, numbers, starts) -> dict:
    """{(set, number): Long} of the texts of the order whose revision as of
    their bound reads a long plain content version, found from the index
    content_lists_long."""
    con = depot.db.reader()
    asked = []  # (set, number, id, bound)
    for (text,) in con.execute(LONG_VERSIONS).fetchall():
        owner, number = ids.owner(text), ids.number(text)
        if number is None or owner not in held or number not in held[owner]:
            continue
        for index, (named, bound) in enumerate(sets):
            low, high = int(starts[index]), int(starts[index + 1])
            if named == owner and low < high:
                at = low + int(np.searchsorted(numbers[low:high], number))
                if at < high and int(numbers[at]) == number:
                    asked.append((index, number, text, bound))
                    break
    found = {}
    for start in range(0, len(asked), PAGE):
        page = asked[start:start + PAGE]
        for at, revision, kind, length, size, tags in con.execute(
                AT_BOUND, (json.dumps([[text, bound] for _, _, text, bound in page]),)).fetchall():
            if kind == PLAIN and length > LONG:
                index, number, _, _ = page[at]
                found[index, number] = Long(revision, tags or "[]", length, size, pieces.count(length))
    return found


def walk(order: Order, after: Sequence | None = None) -> Iterator[Entry]:
    """The texts and pieces of the order, from the place after on (every
    one without it), each as Entry; a number whose text has no row left is
    passed by."""
    con = order.depot.db.reader()
    for start in range(order.start(after), len(order), PAGE):
        end = min(start + PAGE, len(order))
        numbers = order.numbers[start:end].tolist()
        sets = order.owner_sets[start:end].tolist()
        piece = order.piece[start:end].tolist()
        asked = [at for at in range(len(numbers)) if piece[at] == WHOLE]
        rows = {}
        if asked:
            question = [[ids.make(order.sets[sets[at]][0], numbers[at]), order.sets[sets[at]][1]] for at in asked]
            for place, revision, kind, _, size, tags in con.execute(AT_BOUND, (json.dumps(question),)).fetchall():
                rows[asked[place]] = (revision, kind, size, tags or "[]")
        for at in range(len(numbers)):
            owner = order.sets[sets[at]][0]
            text = ids.make(owner, numbers[at])
            if piece[at] == WHOLE:
                found = rows.get(at)
                if found is None:
                    continue
                revision, kind, size, tags = found
                yield Entry(export_key(text), owner, numbers[at], text, None, revision, kind, size, tags)
            else:
                one = order.long[sets[at], numbers[at]]
                number = piece[at]
                size = one.bytes // one.count + (number < one.bytes % one.count)
                yield Entry(export_key(pieces.name(text, number)), owner, numbers[at], text, number, one.revision,
                            PLAIN, size, one.tags)

