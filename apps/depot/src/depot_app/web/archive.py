"""A tar archive of stored files, made as it is sent and never stored.

Its bytes are a function of the names, the sizes and one time alone: every
header is written here, with mode 0644, owner and group 0 and no owner
names, so its length is known before a byte is read and any span of it can
be sent again on its own (one Range of a resumed download). A name past
ustar's 100 bytes or a size past its 8 GiB goes into a POSIX pax extended
header before the file's own header, which Python's tarfile, GNU tar, bsdtar
and 7-Zip read.

layout(entries, mtime) reads no file: it places each header, file and
padding; span(pieces, start, end, read) yields the bytes of [start, end),
reading the files' bytes through read(index, at, length) only there.
"""

from bisect import bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass

BLOCK = 512
END = 2 * BLOCK  # the two zero blocks that end an archive
PAX_NAME = b"././@PaxHeader"  # the name of a pax extended header, as Python's tarfile writes it
NAME_BYTES = 100
MOST_SIZE = 8 ** 11  # the first size the 11 octal digits of ustar's size field cannot hold: 8 GiB


@dataclass(frozen=True)
class Piece:
    """At offset in the archive, length bytes: data when it is written here,
    else the bytes of the file of that index (0 for the first entry)."""
    offset: int
    length: int
    data: bytes | None = None
    index: int | None = None


def _octal(value: int, width: int) -> bytes:
    return b"%0*o\0" % (width - 1, value)


def _header(name: bytes, size: int, mtime: int, kind: bytes) -> bytes:
    fields = [name[:NAME_BYTES].ljust(NAME_BYTES, b"\0"), _octal(0o644, 8), _octal(0, 8), _octal(0, 8),
              _octal(size, 12), _octal(mtime, 12), b" " * 8, kind, b"\0" * 100, b"ustar\x0000",
              b"\0" * 32, b"\0" * 32, _octal(0, 8), _octal(0, 8), b"\0" * 155]
    block = b"".join(fields).ljust(BLOCK, b"\0")
    checksum = b"%06o\0 " % sum(block)
    return block[:148] + checksum + block[156:]


def _record(key: str, value: bytes) -> bytes:
    """One pax record, "LENGTH KEY=VALUE\\n", LENGTH counting itself."""
    rest = b" " + key.encode() + b"=" + value + b"\n"
    length = len(rest) + 1
    while len(str(length).encode() + rest) != length:
        length += 1
    return str(length).encode() + rest


def _padding(size: int) -> int:
    return -size % BLOCK


def headers(name: str, size: int, mtime: int) -> bytes:
    """The header blocks of one regular file: a pax extended header first
    when its name or size does not fit ustar's fields."""
    encoded = name.encode("utf-8")
    records = b""
    if len(encoded) > NAME_BYTES or not name.isascii():
        records += _record("path", encoded)
    if size >= MOST_SIZE:
        records += _record("size", str(size).encode())
    own = _header(encoded, 0 if size >= MOST_SIZE else size, mtime, b"0")
    if not records:
        return own
    extended = _header(PAX_NAME, len(records), mtime, b"x") + records + b"\0" * _padding(len(records))
    return extended + own


def layout(entries: list[tuple[str, int]], mtime: int) -> tuple[list[Piece], int]:
    """(the pieces of the archive of these (name, size) entries in this
    order, its length in bytes), each file's time mtime (whole seconds)."""
    pieces, at = [], 0
    for index, (name, size) in enumerate(entries):
        written = headers(name, size, mtime)
        pieces.append(Piece(at, len(written), data=written))
        at += len(written)
        if size:
            pieces.append(Piece(at, size, index=index))
            at += size
        if _padding(size):
            pieces.append(Piece(at, _padding(size), data=b"\0" * _padding(size)))
            at += _padding(size)
    pieces.append(Piece(at, END, data=b"\0" * END))
    return pieces, at + END


def span(pieces: list[Piece], start: int, end: int, read: Callable[[int, int, int], bytes],
         chunk: int) -> Iterator[bytes]:
    """The archive's bytes from start to end (excluded), at most chunk bytes
    of a file read at a time through read(index, at, length)."""
    first = bisect_right([one.offset for one in pieces], start) - 1
    for piece in pieces[max(first, 0):]:
        if piece.offset >= end:
            return
        low, high = max(start, piece.offset) - piece.offset, min(end, piece.offset + piece.length) - piece.offset
        if piece.data is not None:
            yield piece.data[low:high]
            continue
        while low < high:
            length = min(chunk, high - low)
            data = read(piece.index, low, length)
            if len(data) != length:
                raise OSError(f"a file of the archive gave {len(data)} bytes where {length} were expected")
            yield data
            low += length
