"""The order in which a corpus build, the estimate of a build and the
sample that trains a vocabulary read a dataset's texts: every text of its
libraries, each library as it stood at the change number of its depot the
dataset names, by its export key (wpt_depot.ids.export_key, which its depot
stores with each membership and pin), the order an export reads too
(wpt_depot.exports). The key is made from a text's id alone, so the order
is fixed and shuffled at once: it depends on no dataset, size, vocabulary
or depot file, a text keeps its place whatever the dataset's other texts,
and a smaller size of a dataset's revision takes the first part of what a
larger one takes.

A walk (walk) reads each library as its own rows by key, from the index
memberships_key (pins_key for a library locked then) as Depot.library_range
gives them, PAGE rows at a time, each row's revision and content version
looked up as it is read, and merges the libraries' reads by (key, the place
of the text's depot among the depots the dataset reads, id): texts of one
key go by their depot's place, then by their id. A text that two libraries
of one depot hold comes once, at its revision in the first of those
libraries the dataset names. Nothing past the last text a walk gives is
read, so its first texts come at once whatever the dataset's size, and it
starts after any text it gave before (after: that text's place), which is
how a build goes on from its last upload.

Each Entry is a text with its key, source (the index of its depot among
the depots the dataset reads), id, revision, kind (None: a plain text),
pool (with shares: the index of the first share whose tag a plain text
has, None when it has none; without shares, 0 for every plain text; None
for an item), bytes (its content's UTF-8 bytes) and taken: whether a
corpus takes it at all, that is, a plain text with a pool or a
conversation or a task (an item of another kind is no training data).

    walk(libraries, tags, after)   the texts of the libraries, in order, as Entry
"""

import heapq
import json
from collections.abc import Iterator, Sequence
from typing import NamedTuple

TRAINING_KINDS = ("conversation", "task")
PAGE = 2_000  # rows of one library read in one statement
LOWEST, HIGHEST = -(1 << 63), (1 << 63) - 1  # the first and last export keys


class Entry(NamedTuple):
    key: int
    source: int
    id: str
    revision: int
    kind: str | None
    pool: int | None
    bytes: int
    taken: bool

    def place(self) -> list:
        """The text's place in the order, [key, source, id], what a walk's after names."""
        return [self.key, self.source, self.id]


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


def walk(libraries: Sequence[tuple], shares: Sequence[str] | None, after: Sequence | None = None) -> Iterator[Entry]:
    """The texts of libraries, for each library the dataset names in
    order (source, its depot opened, library id, change number), in the
    order the module says, each as Entry under the tags of the dataset's
    shares (None without shares); only those after the place after
    ([key, source, id]) when it is given."""
    after = None if after is None else tuple(after)
    reads = [_library(source, depot, library, change, after) for source, depot, library, change in libraries]
    last = None
    # heapq.merge gives rows of one place in the order of the reads: the first library's revision first
    for key, source, text, revision, kind, tags, size in heapq.merge(*reads, key=lambda row: row[:3]):
        if (key, source, text) == last:
            continue
        last = key, source, text
        pool, taken = pool_of(kind, tags, shares)
        yield Entry(key, source, text, revision, kind, pool, size, taken)


def _library(source: int, depot, library: str, change: int, after: tuple | None) -> Iterator[tuple]:
    """The rows of one library, in the order of their (key, source, id),
    from its depot a page at a time: (key, source, id, revision, kind,
    tags as JSON text, bytes). A page ends before the rows of its last key,
    which the next page starts with, so the texts of one key are sorted by
    id together."""
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
        found = sorted(((key, source, text, revision, kind, tags, size)
                        for key, text, revision, kind, tags, size in rows), key=lambda row: row[:3])
        for row in found:
            if after is None or row[:3] > after:
                yield row
        high = found[-1][0]
        if ended or high == HIGHEST:
            return
        low = high + 1
