"""A content version as an ordered list of parts, each stored once.

Every content version of a text (its plain text, or an item's JSON content)
is the UTF-8 bytes of one string cut into parts, the same rule for every
text: a text shorter than Sizes.minimum is one part, an empty one none. A
part is known by the SHA-256 of its bytes (its hash), and the depot stores
each part once for its text, keyed by the text's id and that hash
(wpt_depot.schema): a version that holds a part a version before it held
reuses it, and no part is shared between two texts.

Where a cut falls depends only on the bytes around it, so an edit changes
only the parts around it and a new version stores only those (split):

- A cut falls only right after a line end (a byte \\n), never before
  Sizes.minimum bytes into a part. The line end of a line of L bytes
  (counted from the end of the line before it) is a cut when a hash of the
  16 bytes that end with it, scaled to the range from 0 to 1, falls below
  L / (Sizes.target - Sizes.minimum), so a part holds about Sizes.target
  bytes whatever the length of its lines.
- A part never holds more than Sizes.maximum bytes: one that reaches it
  without such a cut ends at its last line end, or, with none, at the last
  character that fits.

So a part ends with a line end but where a line runs past Sizes.maximum,
and the last part, which ends where the text ends.

A part's text is stored compressed with zstd (pack, unpack). Reading
goes through Reader, which fetches parts by hash, and Parted, a content
version read lazily: its length, a slice of its characters or its parts in
order, each part fetched only when needed (per-part lengths come with its
list). A reader of many texts at once (the corpus build) uses scan.
"""

import bisect
import hashlib
import json
from collections.abc import Iterable, Iterator
from compression import zstd
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


class Sizes(NamedTuple):
    minimum: int  # bytes: no cut before this many bytes of a part
    target: int  # bytes: the size a part has on average
    maximum: int  # bytes: the most a part holds


SIZES = Sizes(minimum=1 << 20, target=4 << 20, maximum=64 << 20)
BLOCK = 16 << 20  # characters read from a file at a time
WINDOW = 16  # the bytes that end at a line end which the cut's hash reads
FETCHED = 64 << 20  # characters of parts a Reader fetches in one query, about
LEVEL = 3  # the zstd level a part's stored value is compressed at
_MIX = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x94D049BB133111EB))


class Part(NamedTuple):
    hash: bytes  # SHA-256 of its UTF-8 bytes
    value: str
    length: int  # characters
    bytes: int


@dataclass(frozen=True)
class FileText:
    """A plain text read from a UTF-8 file as a stream, its line ends read
    as Path.read_text reads them (\\r\\n and \\r become \\n), never whole in
    memory: what a text bigger than memory, or than one string, is added
    from."""
    path: Path

    def blocks(self) -> Iterator[str]:
        with open(self.path, encoding="utf-8", newline=None) as file:
            while block := file.read(BLOCK):
                yield block


Source = str | FileText


def encoded(source: Source) -> Iterator[bytes]:
    """The UTF-8 bytes of a content, in blocks."""
    if isinstance(source, str):
        yield source.encode("utf-8")
        return
    for block in source.blocks():
        yield block.encode("utf-8")


def split(blocks: Iterable[bytes], sizes: Sizes | None = None, before: bytes = b"", line: int = -1
          ) -> Iterator[bytes]:
    """The parts of the bytes given in blocks, in order, cut as the module
    says: the same parts however the bytes are cut into blocks. Given the
    WINDOW - 1 bytes before a cut of a longer text (before, its start
    padded with zeros) and the place of the last line end before that cut,
    counted from it (line, negative; -1 at a text's start or right after a
    line end), it cuts the rest of that text as cutting the whole text
    would from there."""
    sizes = sizes or SIZES
    scale = (1 << 32) // max(1, sizes.target - sizes.minimum)
    pending = b""
    before = (bytes(WINDOW - 1) + before)[-(WINDOW - 1):]  # the bytes just before pending, zeros at the text's start
    # line: the place of the last line end before pending, counted from pending's start (negative)
    scanned = 0  # the bytes of pending whose line ends were read already
    blocks = iter(blocks)
    ended = False
    while True:
        cut = _cut(pending, scanned, line, before, sizes, scale)
        if cut is None:
            scanned = len(pending)
            if len(pending) > sizes.maximum:
                cut = _forced(pending, sizes.maximum)
            elif not ended:
                block = next(blocks, None)
                if block is None:
                    ended = True
                else:
                    pending += block
                continue
            elif pending:
                cut = len(pending)
            else:
                return
        part = pending[:cut]
        yield part
        before = (before + part)[-(WINDOW - 1):]
        newline = part.rfind(b"\n")
        line = (newline if newline >= 0 else line) - cut
        pending, scanned = pending[cut:], 0


def _cut(pending: bytes, scanned: int, line: int, before: bytes, sizes: Sizes, scale: int) -> int | None:
    """The end of the part pending begins, when one of its line ends from
    scanned on is a cut by the hash rule; None otherwise. line: the place of
    the last line end before pending, counted from its start."""
    view = np.frombuffer(pending, np.uint8)
    ends = np.flatnonzero(view[scanned:] == 10) + scanned
    if not len(ends):
        return None
    previous = pending.rfind(b"\n", 0, scanned)
    starts = np.concatenate([[previous if previous >= 0 else line], ends[:-1]])  # the line end before each
    lines = (ends - starts).astype(np.uint64)  # each line's bytes, its line end included
    size = ends + 1
    chosen = (size >= sizes.minimum) & (size <= sizes.maximum)
    if not chosen.any():
        return None
    ends, lines = ends[chosen], lines[chosen]
    padded = before + pending
    words = np.ndarray(shape=(len(padded) - 7,), dtype="<u8", buffer=padded, strides=(1,))
    hashes = _mixed(words[ends], words[ends + 8])  # the 16 bytes that end at each line end
    hit = np.flatnonzero((hashes >> np.uint64(32)) < lines * np.uint64(scale))
    return int(ends[hit[0]]) + 1 if len(hit) else None


def _mixed(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    value = (first * _MIX[0]) ^ (second * _MIX[1])
    value ^= value >> np.uint64(31)
    value *= _MIX[2]
    return value ^ (value >> np.uint64(29))


def _forced(pending: bytes, maximum: int) -> int:
    """Where a part that ran past maximum bytes with no cut ends: after its
    last line end within them, or, with none, after the last character that
    ends within them."""
    found = pending.rfind(b"\n", 0, maximum)
    if found >= 0:
        return found + 1
    end = maximum
    while end > 0 and pending[end] & 0xC0 == 0x80:  # a byte inside a character
        end -= 1
    return end


def pack(value: str) -> bytes:
    """A part's text as the parts table stores it: its UTF-8 bytes compressed with zstd."""
    return zstd.compress(value.encode("utf-8"), LEVEL)


def unpack(stored: bytes) -> str:
    """A part's text from the value the parts table stores."""
    return zstd.decompress(stored).decode("utf-8")


def made(data: bytes) -> Part:
    """The part of these bytes."""
    value = data.decode("utf-8")
    return Part(hashlib.sha256(data).digest(), value, len(value), len(data))


def parted(source: Source, sizes: Sizes | None = None) -> Iterator[Part]:
    """The parts of a content, in order, each with its hash and lengths."""
    sizes = sizes or SIZES
    if isinstance(source, str):
        data = source.encode("utf-8")
        if len(data) <= sizes.minimum:  # no cut falls before a part's least bytes: one part, or none when empty
            if data:
                yield Part(hashlib.sha256(data).digest(), source, len(source), len(data))
            return
        source = None
    for data in split([data] if source is None else encoded(source), sizes):
        yield made(data)


class Listed(NamedTuple):
    """A content version as its list of parts names it: [(hash, length)], and its length and bytes."""
    parts: list[tuple[bytes, int]]
    length: int
    bytes: int


def listed(source: Source, sizes: Sizes | None = None) -> Listed:
    """The list of a content's parts, read without keeping their text."""
    found, length, size = [], 0, 0
    for part in parted(source, sizes):
        found.append((part.hash, part.length))
        length += part.length
        size += part.bytes
    return Listed(found, length, size)


class Reader:
    """Reads parts of texts from the depot's parts table through a
    connection (wpt_depot.database), which `connection()` gives each time
    it is asked: the one of the thread that reads."""

    def __init__(self, connection):
        self.connection = connection

    def values(self, text: str, hashes: Iterable[bytes]) -> dict[bytes, str]:
        """{hash: its text} of the parts of the text named."""
        wanted = list(dict.fromkeys(hashes))
        if not wanted:
            return {}
        found = self.connection().execute(
            "SELECT hash, value FROM parts WHERE id = ? AND hash IN (SELECT unhex(value) FROM json_each(?))",
            (text, json.dumps([one.hex() for one in wanted]))).fetchall()
        values = {bytes(hash_): unpack(value) for hash_, value in found}
        missing = [one for one in wanted if one not in values]
        if missing:
            raise LookupError(f"the depot holds no part {missing[0].hex()} of the text {text}")
        return values

    def stream(self, text: str, parts: list[tuple[bytes, int]]) -> Iterator[str]:
        """The text of each part listed, in order, fetched FETCHED characters or so at a time."""
        at = 0
        while at < len(parts):
            batch, size = [], 0
            while at < len(parts) and (not batch or size + parts[at][1] <= FETCHED):
                batch.append(parts[at][0])
                size += parts[at][1]
                at += 1
            values = self.values(text, batch)
            for hash_ in batch:
                yield values[hash_]


def scan(reader: Reader, wanted: dict[str, list]) -> Iterator[tuple[str, str | None]]:
    """The texts of wanted ({id: the list of its parts, as Parted takes
    it}), read through reader one text after the other, in the order of
    wanted: (id, the text of its next part) for each part of a text in its
    list's order, then (id, None) once the text is whole. Only a few parts
    are held at a time. LookupError when a part listed is missing."""
    for text, found in wanted.items():
        yield from ((text, value) for value in reader.stream(text, Parted.listed(found)))
        yield text, None


class Parted:
    """A content version of a text read lazily from its parts (`parts`: the
    list [(hash, length)] its row holds): its length in characters, a slice
    of its characters (fetching only the parts the slice reaches), its parts'
    texts in order when iterated, the whole of it as str()."""

    def __init__(self, reader: Reader, text: str, parts: list):
        self.reader, self.text = reader, text
        self.parts = self.listed(parts)
        self.starts = [0]
        for _, length in self.parts:
            self.starts.append(self.starts[-1] + length)

    @staticmethod
    def listed(parts: list) -> list[tuple[bytes, int]]:
        """[(hash, length)] of a list of parts: {hash, length} dicts or pairs."""
        return [(_hash(one["hash"]), one["length"]) if isinstance(one, dict) else (_hash(one[0]), one[1])
                for one in parts]

    def __len__(self) -> int:
        return self.starts[-1]

    def __getitem__(self, key: slice) -> str:
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("a content version is sliced with no step")
        if start >= stop:
            return ""
        first = bisect.bisect_right(self.starts, start) - 1
        last = bisect.bisect_left(self.starts, stop)  # the parts first..last-1 hold the slice
        chosen = self.parts[first:last]
        values = self.reader.values(self.text, [hash_ for hash_, _ in chosen])
        joined = "".join(values[hash_] for hash_, _ in chosen)
        return joined[start - self.starts[first]:stop - self.starts[first]]

    def __iter__(self) -> Iterator[str]:
        return self.reader.stream(self.text, self.parts)

    def __str__(self) -> str:
        return "".join(self)


def _hash(value) -> bytes:
    """A part's hash, given as bytes or as hexadecimal text (a list read from JSON)."""
    return bytes.fromhex(value) if isinstance(value, str) else bytes(value)
