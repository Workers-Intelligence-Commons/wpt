"""The texts of a built corpus as files to download, made when asked and
never stored: one part for each part of the corpus's texts table
(wpt_depot.corpus), its texts in the corpus's order, each read from its
depot at the revision the corpus took.

A part comes in two forms, each the same bytes every time it is made for
the same corpus, by the same versions of Python's zstd and of pyarrow:

- "jsonl.zst": one JSON line per text, {"depot", "id", "revision", "kind",
  "text"} (an item's text is its JSON content), compressed with zstd at
  LEVEL as one frame; a text is read and written a part at a time, so a
  text of any size streams through;
- "parquet": a Parquet file of the same columns (TEXT_SCHEMA), zstd
  compressed, in row groups of at most GROUP_TEXTS texts or about
  GROUP_BYTES bytes of text, each group sent once written; a text is held
  whole while its group is written.

A text an erasure removed since the build cannot be read: NotFound stops
the part (an erasure makes the corpus stale first, so a fresh corpus never
meets one).

    FORMS                                        the forms, by name
    MAKER                                        the versions a part's bytes depend on
    part(depot, store, record, part, form)       the bytes of a part, as a stream
"""

import json
from collections.abc import Iterator
from compression import zstd

import pyarrow as pa
import pyarrow.parquet as pq

from .corpus import sources
from .depot import Depot
from .parts import scan

LEVEL = 6  # the zstd level of a part in jsonl.zst
GROUP_TEXTS = 4096  # texts in one row group of a part in parquet, at most
GROUP_BYTES = 64 << 20  # characters of text after which a row group of a part in parquet ends
TEXT_SCHEMA = pa.schema([pa.field("depot", pa.string(), False), pa.field("id", pa.string(), False),
                         pa.field("revision", pa.int64(), False), pa.field("kind", pa.string()),
                         pa.field("text", pa.large_string(), False)])
FORMS = {"jsonl.zst": "application/zstd", "parquet": "application/vnd.apache.parquet"}
# what makes a part's bytes beside the corpus: a size or sha256 kept for a part holds only for the same MAKER
MAKER = f"pyarrow {pa.__version__}, zstd {zstd.zstd_version}"


def _rows(depot: Depot, store, record: dict, part: int) -> Iterator[tuple[dict, Iterator[str]]]:
    """For each text of the part, in order: (its row without its text, its text's pieces in order)."""
    reference = record["tables"]["texts"][part]
    table = pq.read_table(store.path(reference), columns=["depot", "id", "revision", "kind"])
    store.release(reference)
    opened = sources(depot, record)
    for row in table.to_pylist():
        source = opened[row["depot"]]
        listed = source.listed_text(row["id"], row["revision"])
        pieces = (value for _, value in scan(source.reader(), {row["id"]: listed["parts"]}) if value is not None)
        yield row, pieces


def _jsonl(rows) -> Iterator[bytes]:
    compressor = zstd.ZstdCompressor(level=LEVEL)
    for row, pieces in rows:
        head = json.dumps(row, ensure_ascii=False)[:-1] + ', "text": "'
        out = compressor.compress(head.encode("utf-8"))
        if out:
            yield out
        for piece in pieces:
            out = compressor.compress(json.dumps(piece, ensure_ascii=False)[1:-1].encode("utf-8"))
            if out:
                yield out
        out = compressor.compress(b'"}\n')
        if out:
            yield out
    yield compressor.flush()


class _Sink:
    """A file a Parquet writer writes to, whose bytes are taken as they come."""

    def __init__(self):
        self.held, self.position, self.closed = [], 0, False

    def write(self, data) -> int:
        data = bytes(data)
        self.held.append(data)
        self.position += len(data)
        return len(data)

    def tell(self) -> int:
        return self.position

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def taken(self) -> bytes:
        found = b"".join(self.held)
        self.held.clear()
        return found


def _parquet(rows) -> Iterator[bytes]:
    sink = _Sink()
    writer = pq.ParquetWriter(pa.PythonFile(sink, mode="w"), TEXT_SCHEMA, compression="zstd")
    group, size = [], 0
    for row, pieces in rows:
        text = "".join(pieces)
        group.append({**row, "text": text})
        size += len(text)
        if len(group) >= GROUP_TEXTS or size >= GROUP_BYTES:
            writer.write_table(pa.Table.from_pylist(group, schema=TEXT_SCHEMA))
            group, size = [], 0
            yield sink.taken()
    if group:
        writer.write_table(pa.Table.from_pylist(group, schema=TEXT_SCHEMA))
    writer.close()
    yield sink.taken()


def part(depot: Depot, store, record: dict, part: int, form: str) -> Iterator[bytes]:
    """The bytes of the part numbered part (from 0) of the corpus's texts in
    that form (FORMS), as they are made: depot is the depot that holds the
    dataset, which the record names; the texts of another depot are read
    from its file, found again from the dataset's revision the record
    names (wpt_depot.corpus.sources). KeyError for a
    form FORMS does not name, IndexError for a part the corpus has not."""
    if form not in FORMS:
        raise KeyError(form)
    record["tables"]["texts"][part]  # noqa: B018 - IndexError before anything is sent
    rows = _rows(depot, store, record, part)
    return _jsonl(rows) if form == "jsonl.zst" else _parquet(rows)
