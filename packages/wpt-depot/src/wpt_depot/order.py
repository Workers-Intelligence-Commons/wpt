"""The order in which a corpus build, the estimate of a build and the
sample that trains a vocabulary read a dataset's texts: every text of its
libraries, each library as it stood at the change number of its depot the
dataset names, by its export key (wpt_depot.ids.export_key, which its depot
stores with each membership and pin), the order an export reads too
(wpt_depot.exports), but for its long plain texts, each of which comes as
its pieces (wpt_depot.pieces), each piece at the export key of its name.
The key is made from a text's id alone, or a piece's name, so the order is
fixed and shuffled at once: it depends on no dataset, size, vocabulary or
depot file, a text and each of its pieces keep their place whatever the
dataset's other texts, and the pieces of a long text spread over the whole
order, so the first part of the order holds about its share of each long
text.

A walk (walk) reads each library as its own rows by key, from the index
memberships_key (pins_key for a library locked then) as Depot.library_range
gives them, PAGE rows at a time, each row's revision and content version
looked up as it is read, and merges the libraries' reads with the pieces by
(key, the place of the text's depot among the depots the dataset reads, id,
piece): texts of one key go by their depot's place, then by their id, a
whole text before a piece. A text that two libraries of one depot hold
comes once, at its revision in the first of those libraries the dataset
names. Nothing past the last text a walk gives is read, so its first texts
come at once whatever the dataset's size, and it starts after any text it
gave before (after: that text's place), which is how a build goes on from
its last upload.

The pieces need the long texts before the walk starts (long_texts): those
each library holds, read from the depot's index of long content versions
(Depot.library_long), whose read grows with the depot's long contents,
never with a library's texts. A long text comes as pieces when its revision
in the first library of its depot that holds it is long; a library's row of
such a text is then passed. The walk holds the export key, text and number
of every piece in arrays, 16 bytes a piece, and twice that while it sorts
them.

Each Entry is a text or a piece with its key, source (the index of its depot
among the depots the dataset reads), id, piece (its number, None for a whole
text), revision, kind (None: a plain text), pool (with shares: the index of
the first share whose tag a plain text has, None when it has none; without
shares, 0 for every plain text; None for an item), bytes (its content's
UTF-8 bytes; a piece's, its share of its text's bytes, split evenly among
its pieces, since they are known only once it is read), taken: whether a
corpus takes it at all, that is, a plain text with a pool or a conversation
or a task (an item of another kind is no training data), and tags (its
tags as JSON text; a piece's, its text's).

    walk(libraries, tags, after, long)   the texts and pieces of the libraries, in order, as Entry
    long_texts(libraries)                the long texts whose pieces a walk gives
"""

import heapq
import json
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from . import pieces
from .ids import export_key

TRAINING_KINDS = ("conversation", "task")
PAGE = 2_000  # rows of one library read in one statement
LOWEST, HIGHEST = -(1 << 63), (1 << 63) - 1  # the first and last export keys
WHOLE = -1  # what a whole text's place holds where a piece's holds its number


class Entry(NamedTuple):
    key: int
    source: int
    id: str
    piece: int | None
    revision: int
    kind: str | None
    pool: int | None
    bytes: int
    taken: bool
    tags: str

    def place(self) -> list:
        """The text's place in the order, [key, source, id, piece (WHOLE for a whole text)], what a walk's after
        names."""
        return [self.key, self.source, self.id, WHOLE if self.piece is None else self.piece]


class Long(NamedTuple):
    """A long text whose pieces a walk gives: its source, id, revision, tags (JSON text), length and bytes."""
    source: int
    id: str
    revision: int
    tags: str
    length: int
    bytes: int


def pool_of(kind: str | None, tags: str, shares: Sequence[str] | None) -> tuple[int | None, bool]:
    """(pool, taken) of a text of that kind with tags (JSON text), under
    the tags of a dataset's shares (None without shares), as Entry says."""
    if kind is not None:
        return None, kind in TRAINING_KINDS
    if shares is None:
        return 0, True
    held = json.loads(tags)
    pool = next((at for at, tag in enumerate(shares) if tag in held), None)
    return pool, pool is not None


def long_texts(libraries: Sequence[tuple]) -> list[Long]:
    """The long texts of the libraries (as walk takes them) that a walk
    gives as pieces, as the module says, in the order of their (source,
    id)."""
    found = {}
    for at, (source, depot, library, change) in enumerate(libraries):
        earlier = [one for one in libraries[:at] if one[0] == source]
        for key, text, revision, tags, length, size in depot.library_long(library, change):
            if (source, text) in found:
                continue
            # an earlier library of the depot that holds it holds a revision that is not long: a whole text
            if any(text in (row[1] for row in held.library_range(name, then, key, key))
                   for _, held, name, then in earlier):
                continue
            found[source, text] = Long(source, text, revision, tags, length, size)
    return [found[place] for place in sorted(found)]


def walk(libraries: Sequence[tuple], shares: Sequence[str] | None, after: Sequence | None = None,
         long: list[Long] | None = None) -> Iterator[Entry]:
    """The texts and pieces of libraries, for each library the dataset
    names in order (source, its depot opened, library id, change number),
    in the order the module says, each as Entry under the tags of the
    dataset's shares (None without shares); only those after the place
    after when it is given. long: the long texts (long_texts), read from
    the libraries when not given."""
    after = None if after is None else tuple(after)
    long = long_texts(libraries) if long is None else long
    cut = {(one.source, one.id) for one in long}
    reads = [_library(source, depot, library, change, after) for source, depot, library, change in libraries]
    last = None
    # heapq.merge gives rows of one place in the order of the reads: the first library's revision first
    for key, source, text, piece, revision, kind, tags, size in heapq.merge(*reads, _pieces(long, after),
                                                                           key=lambda row: row[:4]):
        if (key, source, text, piece) == last or (piece == WHOLE and (source, text) in cut):
            continue
        last = key, source, text, piece
        pool, taken = pool_of(kind, tags, shares)
        yield Entry(key, source, text, None if piece == WHOLE else piece, revision, kind, pool, size, taken,
                    tags or "[]")


def _pieces(long: list[Long], after: tuple | None) -> Iterator[tuple]:
    """The rows of the pieces of the long texts, in the order of their (key,
    source, id, number), as _library gives a text's (its kind None, its
    bytes its share of its text's)."""
    counts = np.array([pieces.count(one.length) for one in long], np.int64)
    total = int(counts.sum())
    owners = np.repeat(np.arange(len(long), dtype=np.int32), counts)
    numbers = (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)).astype(np.int32)
    keys = np.fromiter((export_key(pieces.name(long[owner].id, int(number)))
                        for owner, number in zip(owners, numbers)), np.int64, total)
    order = np.argsort(keys, kind="stable")  # long is in the order of (source, id), each text's pieces by number
    keys, owners, numbers = keys[order], owners[order], numbers[order]
    del order
    first = 0 if after is None else int(np.searchsorted(keys, after[0]))
    for start in range(first, total, PAGE):  # a page of plain ints at a time
        end = start + PAGE
        for key, owner, number in zip(keys[start:end].tolist(), owners[start:end].tolist(),
                                      numbers[start:end].tolist()):
            one, count = long[owner], int(counts[owner])
            row = (key, one.source, one.id, number, one.revision, None, one.tags,
                   one.bytes // count + (number < one.bytes % count))
            if after is None or row[:4] > after:
                yield row


def _library(source: int, depot, library: str, change: int, after: tuple | None) -> Iterator[tuple]:
    """The rows of one library, in the order of their (key, source, id),
    from its depot a page at a time: (key, source, id, WHOLE, revision,
    kind, tags as JSON text, bytes). A page ends before the rows of its
    last key, which the next page starts with, so the texts of one key are
    sorted by id together."""
    low = LOWEST if after is None else after[0]
    while True:
        rows = list(depot.library_range(library, change, low, HIGHEST, PAGE))
        if not rows:
            return
        ended = len(rows) < PAGE
        if not ended:
            top = rows[-1][0]
            rows = ([row for row in rows if row[0] != top] if rows[0][0] != top
                    else list(depot.library_range(library, change, top, top)))
        found = sorted(((key, source, text, WHOLE, revision, kind, tags, size)
                        for key, text, revision, kind, tags, size in rows), key=lambda row: row[:3])
        for row in found:
            if after is None or row[:4] > after:
                yield row
        high = found[-1][0]
        if ended or high == HIGHEST:
            return
        low = high + 1
