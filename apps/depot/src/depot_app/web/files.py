"""What the routes that send a corpus's files (depot_app.web.legacy, the
frozen dataset downloads) use: what each file of a corpus is, a name fit
for a downloaded file, the one Range of bytes a request asks for, and a
stream that leaves its guard once it is closed."""

import re

from .server import Request

CHUNK = 8 << 20  # bytes of a stored file read at a time to send it
ARCHIVE = "everything.tar"  # the last part of the address of a corpus's archive of every file
ARCHIVE_FORMAT = "wpt-tar-1"  # mixed into the archive's ETag: raise it when archive.py changes the bytes it writes
SIZING = 16  # dock calls at once to learn the sizes of a corpus's files: each waits on the dock, not the machine
RANGE = re.compile(r"bytes=(\d*)-(\d*)")
UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")  # what a downloaded file's name replaces by a hyphen
JSON_TYPE = "application/json"  # the media type of a vocabulary file, the JSON of wpt_tokenizer's merges


class Guarded:
    """The chunks of a stream whose guard (a context manager entered
    already) is left once, when the server closes the stream, sent or not;
    closed (default: nothing) is called then too."""

    def __init__(self, chunks, guard, closed=lambda: None):
        self.chunks, self.guard, self.closed, self.left = chunks, guard, closed, False

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        return next(self.chunks)

    def close(self):
        if not self.left:
            self.left = True
            try:
                self.closed()
                getattr(self.chunks, "close", lambda: None)()
            finally:
                self.guard.__exit__(None, None, None)


def slug(text: str) -> str:
    """text as a part of a downloaded file's name: ASCII letters, digits, '.', '_' and '-', the rest each run a hyphen."""
    return UNSAFE.sub("-", text).strip("-") or "unnamed"


def roles(record: dict) -> dict:
    """{key: what the file is} of each file of the corpus: "KIND tokens",
    "table NAME", "vocabulary" or "tasks"."""
    found = {one["key"]: f"{kind} tokens" for kind, kept in record.get("shards", {}).items() for one in kept}
    for name, table in record.get("tables", {}).items():
        found |= {one["key"]: f"table {name}" for one in (table if isinstance(table, list) else [table])}
    if "vocabulary" in record.get("tokenizer", {}):
        found[record["tokenizer"]["vocabulary"]["key"]] = "vocabulary"
    if "tasks" in record:
        found[record["tasks"]["file"]["key"]] = "tasks"
    return found


def ranged(request: Request, size: int, etag: str | None = None):
    """(start, end, status) of the bytes to send of size bytes: one Range of them (206), else all (200),
    all as well when If-Range names another ETag than etag, or any with no etag; an answer of 416 for a
    Range that cannot be met."""
    asked = request.headers.get("Range")
    condition = request.headers.get("If-Range")
    if not asked or (condition is not None and condition.strip() != etag):
        return 0, size, 200
    match = RANGE.fullmatch(asked.strip())
    if match is None or (not match[1] and not match[2]):
        return 416, {"problems": [f"the Range {asked} is not one range of bytes"]}, None
    if match[1]:
        start, end = int(match[1]), min(int(match[2]) + 1 if match[2] else size, size)
    else:
        start, end = max(size - int(match[2]), 0), size
    if start >= end:
        return 416, {"problems": [f"the Range {asked} is outside the file's {size:,} bytes"]}, None
    return start, end, 206
