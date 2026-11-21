"""A library read as one strip of text: its texts one after another, in
the numeric order of their ids (ids.order_sql), each its current content
version for an open library, the one of the revision it pinned for a
locked one; an item's content is its JSON.

A place in the strip is a text and a character in it (an anchor), read
from the library's index of that order (memberships_order, pins_order),
so a read at any place reads only the rows it shows, whatever the
library's size. read gives the characters after an anchor, or before it,
cut from the parts of the texts it reaches (wpt_depot.parts): a text of
any size is never read whole.

A position counts the characters of the strip before a place, from 0. A
library's map (build) holds the characters before every EVERY-th text,
made in one walk of the library's rows that reads no text, so locate
turns a position into an anchor by reading at most EVERY rows. A map is
made for one state of the library (state), and a library that moved since
its map was made reads on through its anchors, only its positions off.
"""

import bisect
import json
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

from . import counts, ids
from .errors import NotFound
from .parts import Parted, Reader

EVERY = 4096  # texts between two marks of a map
BATCH = 64  # rows a read asks the index for at a time
MOST_TEXTS = 200  # texts one read shows at most
LINE = 32_768  # characters a read goes past its count, or before its start, at most, to end or start with a line
WALKED = 100_000  # rows a walk reads in one statement, so no read stays open for the whole walk
CACHED = 256 << 20  # characters of unpacked parts kept for the next reads, the least recently read dropped first
_PLACE = ids.order_sql("h.text")


class Row(NamedTuple):
    place: str  # the text's place in the numeric order of ids (ids.order_sql)
    text: str
    title: str | None
    kind: str | None
    length: int  # characters
    parts: str  # the content version's parts, as JSON text


@dataclass(frozen=True)
class Map:
    """What a walk of a library found: state, the library's state it read
    (state); texts and characters, of the whole strip; marks, (place, text,
    characters before it) of every EVERY-th text from the first."""
    state: str
    texts: int
    characters: int
    marks: list[tuple[str, str, int]]

    def as_json(self) -> dict:
        return {"state": self.state, "texts": self.texts, "characters": self.characters,
                "marks": [list(one) for one in self.marks]}

    @classmethod
    def of_json(cls, found: dict) -> "Map":
        return cls(found["state"], found["texts"], found["characters"], [tuple(one) for one in found["marks"]])


def _locked(depot, library: str) -> bool:
    """Whether the library is locked. NotFound without such a library."""
    found = depot.rows("SELECT locked_at FROM libraries WHERE id = ?", (library,))
    if not found:
        raise NotFound(f"no library {library}")
    return found[0]["locked_at"] is not None


def state(depot, library: str) -> str:
    """A name of what the library holds now, the same while it holds the
    same texts at the same versions: a locked library's never changes; an
    open one's is the change at which what it holds last moved
    (wpt_depot.counts), or the depot's change when that is not known."""
    if _locked(depot, library):
        return "locked"
    found = depot.db.reader().execute("SELECT change FROM library_moves WHERE library = ?", (library,)).fetchall()
    moved = found[0][0] if found else 0
    return f"open-{moved}" if moved is not None else f"depot-{depot.change()}"


def _query(locked: bool, columns: str, where: str, order: str, limit: int | None) -> str:
    if locked:
        held = "pins h INDEXED BY pins_order CROSS JOIN texts_revisions t ON t.id = h.text AND t.revision = h.revision"
        member = ""
    else:
        held = ("memberships h INDEXED BY memberships_order "
                "CROSS JOIN texts_revisions t INDEXED BY texts_latest ON t.id = h.text AND t.latest")
        member = "AND h.left_change IS NULL "
    return (f"SELECT {columns} FROM {held} CROSS JOIN content_lists c ON c.id = t.id AND c.version = t.content_version "
            f"WHERE h.library = :library {member}{where} ORDER BY {_PLACE} {order}"
            + ("" if limit is None else f" LIMIT {int(limit)}"))


def rows(depot, library: str, place: str = "", forward: bool = True, inclusive: bool = False,
         locked: bool | None = None) -> Iterator[Row]:
    """The library's texts after place (before it, backward), that place
    too when inclusive, in the strip's order (backward: the other way
    round), BATCH at a time from the index."""
    locked = _locked(depot, library) if locked is None else locked
    sign = (">=" if inclusive else ">") if forward else ("<=" if inclusive else "<")
    con = depot.db.reader()
    while True:
        query = _query(locked, f"{_PLACE}, h.text, t.title, c.kind, c.length, c.parts",
                       f"AND {_PLACE} {sign} :place " if place else "", "" if forward else "DESC", BATCH)
        found = [Row(*one) for one in con.execute(query, {"library": library, "place": place})]
        yield from found
        if len(found) < BATCH:
            return
        place, sign = found[-1].place, (">" if forward else "<")


def place_of(depot, text: str) -> str:
    """A text's place in the numeric order of ids, as the index holds it."""
    return depot.db.reader().execute(f"SELECT {ids.order_sql(':text')}", {"text": text}).fetchone()[0]


def build(depot, library: str, progress: Callable[[int], None] | None = None, every: int = EVERY) -> Map:
    """The library's map, read in one walk of its rows in the strip's order,
    which reads no text, WALKED rows a statement, each its own read, so the
    walk never holds one read open from start to end: progress, when given,
    gets how many texts it walked, every EVERY of them."""
    named = state(depot, library)
    locked = named == "locked"
    marks, texts, characters, place = [], 0, 0, None
    while True:
        query = _query(locked, f"{_PLACE}, h.text, c.length", "" if place is None else f"AND {_PLACE} > :place ",
                       "", WALKED)
        found = depot.db.reader().execute(query, {"library": library, "place": place}).fetchall()
        for place, text, length in found:
            if texts % every == 0:
                marks.append((place, text, characters))
                if progress and texts:
                    progress(texts)
            texts += 1
            characters += length
        if len(found) < WALKED:
            return Map(named, texts, characters, marks)


def locate(depot, library: str, found: Map, position: int) -> tuple[str, int] | None:
    """The anchor (text, character in it) of a position of the strip, as the
    map counts: the end of the last text for a position at or past its end;
    None for a library that holds no text. Reads at most EVERY rows; for a
    library that moved since its map was made, at most twice that, then
    gives the place it reached."""
    if not found.marks:
        return None
    position = max(0, position)
    mark = max(0, bisect.bisect_right([before for _, _, before in found.marks], position) - 1)
    place, _, before = found.marks[mark]
    final = None
    for walked, row in enumerate(rows(depot, library, place, inclusive=True)):
        if position < before + row.length:
            return row.text, position - before
        before += row.length
        final = row
        if walked >= 2 * EVERY:  # the library moved since the map was made: its place here, at most this far
            break
    return (final.text, final.length) if final else None


def last(depot, library: str) -> tuple[str, int] | None:
    """The anchor at the end of the strip: the last text and its length; None for an empty library."""
    for row in rows(depot, library, forward=False, place="\U0010ffff"):
        return row.text, row.length
    return None


class _Cached(Reader):
    """A Reader that keeps the unpacked parts it read, at most CACHED
    characters of them, for the reads after it: reads of a few KB in a
    part of MBs unpack the part once while the strip is read on."""

    def __init__(self, connection):
        super().__init__(connection)
        self.kept: OrderedDict[tuple[str, bytes], str] = OrderedDict()
        self.size = 0
        self.lock = threading.Lock()

    def values(self, text: str, hashes) -> dict[bytes, str]:
        wanted, found = list(dict.fromkeys(hashes)), {}
        with self.lock:
            for one in wanted:
                if (text, one) in self.kept:
                    self.kept.move_to_end((text, one))
                    found[one] = self.kept[(text, one)]
        missing = [one for one in wanted if one not in found]
        if missing:
            read = super().values(text, missing)
            found |= read
            with self.lock:
                for one, value in read.items():
                    if (text, one) not in self.kept:
                        self.kept[(text, one)] = value
                        self.size += len(value)
                while self.size > CACHED and self.kept:
                    self.size -= len(self.kept.popitem(last=False)[1])
        return found


_readers: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()  # a depot's database object: its cached reader
_readers_lock = threading.Lock()


def _reader(depot) -> _Cached:
    """The depot's cached reader, one for each depot object, reading through the depot's connection of each thread."""
    with _readers_lock:
        found = _readers.get(depot.db)
        if found is None:
            found = _readers[depot.db] = _Cached(depot.db.reader)
        return found


def _piece(depot, row: Row, start: int, end: int, forward: bool, snap: bool) -> dict:
    """The row's text from start to end, the end moved on to the end of its
    line (forward) or the start back to the start of its line (backward),
    and with snap the start back to the start of its line too, each by at
    most LINE characters."""
    if end <= start:
        return {"id": row.text, "title": row.title, "kind": row.kind, "length": row.length, "start": start, "text": ""}
    content = Parted(_reader(depot), row.text, json.loads(row.parts))
    low = max(0, start - LINE) if (snap or not forward) and start > 0 else start
    high = min(row.length, end + LINE) if forward else end
    read = content[low:high]
    if low < start:  # the start of start's line: after the last line end before it
        line = read.rfind("\n", 0, start - low)
        start = low + line + 1 if line >= 0 else 0 if low == 0 else start
    if forward and end < row.length and read[end - low - 1] != "\n":  # the end of end's line, unless it ends one
        line = read.find("\n", end - low)
        end = low + line + 1 if line >= 0 else end
    return {"id": row.text, "title": row.title, "kind": row.kind, "length": row.length, "start": start,
            "text": read[start - low:end - low]}


def read(depot, library: str, text: str | None, offset: int, characters: int, forward: bool = True,
         inclusive: bool = False, most_texts: int = MOST_TEXTS) -> dict:
    """At most characters characters of the strip after the anchor (text,
    offset), before it when not forward, and at most most_texts texts:
    {pieces: each {id, title, kind, length, start, text}, the text's
    characters from start, in the strip's order; edge: whether the read
    reached the strip's end (its start, backward)}. A read ends with a line:
    the line it would cut is read on to its end (backward, back to its
    start), when that end is at most LINE characters away. A text with no
    characters shows as a piece with none. After an anchor at the end of its
    text the read starts with the next text, and before one at its start
    with the text before it, unless inclusive: then (forward only) the
    anchor's text shows from the start of offset's line (LINE characters
    before it at most) even at its end. A text the library no
    longer holds reads on from its place; no text (None) reads from the
    strip's start, or its end backward."""
    locked = _locked(depot, library)
    place = place_of(depot, text) if text is not None else ("" if forward else "\U0010ffff")
    pieces, left = [], characters
    found = rows(depot, library, place, forward, inclusive=text is not None, locked=locked)
    for row in found:
        if forward:
            start = min(offset, row.length) if row.text == text else 0
            if row.text == text and start >= row.length and not inclusive:
                continue
            end = min(row.length, start + left)
        else:
            end = min(offset, row.length) if row.text == text else row.length
            if row.text == text and end <= 0:
                continue
            start = max(0, end - left)
        pieces.append(piece := _piece(depot, row, start, end, forward, inclusive and row.text == text))
        start, end = piece["start"], piece["start"] + len(piece["text"])
        left -= end - start
        cut = (end < row.length) if forward else (start > 0)
        if cut or left <= 0 or len(pieces) >= most_texts:
            edge = False
            if not cut and left <= 0:  # a read that ends on a text's edge: whether another text follows
                edge = next(rows(depot, library, row.place, forward, locked=locked), None) is None
            break
    else:
        edge = True
    if not forward:
        pieces.reverse()
    return {"pieces": pieces, "edge": edge}


def total(depot, library: str) -> dict:
    """{texts, characters} the library holds now, from the depot's counts at
    once (wpt_depot.counts), every kind together."""
    held = counts.figures(depot.db.reader(), library, ())
    return {"texts": sum(one["texts"] for one in held["kinds"].values()),
            "characters": sum(one["characters"] for one in held["kinds"].values())}
