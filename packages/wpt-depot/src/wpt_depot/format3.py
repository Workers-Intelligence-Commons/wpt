"""The step from format 2 to format 3 (wpt_depot.legacy.STEPS), on the
folder of Delta Lake tables: every
content version stored as the ordered list of its parts (wpt_depot.parts).

At format 2 a row of contents holds its version's whole text (text, or an
item's content). The step cuts each into parts, stores each part once per
text in a new table parts, and replaces the rows of contents with rows that
hold the list of their parts, their length in characters and in UTF-8
bytes. It never holds the table in memory: it reads the data files of
contents in batches of about BATCH_BYTES, ROWS rows at a time, and stores
each batch's parts in one commit of parts that records the batch's number
as a Delta Lake transaction of the app APP. The new rows of contents are
written last, in one commit that replaces the table, read from the lists
each batch kept in the folder LISTS of the depot's folder, or cut again
from the old rows when that folder lost them; then the folder goes, and
every older state of contents, which holds every text a second time, is
dropped. Until then the depot holds each text twice, in contents and in
parts, so the step needs about as much free space as the texts take.

A step cut short (a crash, a failed write) runs again from the batch after
the last one parts recorded, so no part is stored twice: a part of a text
with several versions, which two batches may both hold, is stored by the
first. Run again once contents holds lists, it changes nothing but the
older states of contents and the folder LISTS it had not removed yet. Its
layouts and the way it writes their files are its own (the columns it
reads at format 2, the tables format 3 lays out and their writer
properties), never those of a later format, and it reads the tables' files
itself, through pyarrow.
"""

import hashlib

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from deltalake import ColumnProperties, WriterProperties

from .parts import parted

READ = ("id", "version", "change", "text", "kind", "content")  # the columns of contents the step reads, at format 2
PART_3 = pa.struct([pa.field("hash", pa.binary(), nullable=False), pa.field("length", pa.int64(), nullable=False)])
CONTENTS_3 = pa.schema([pa.field("id", pa.string(), False), pa.field("version", pa.int64(), False),
                        pa.field("change", pa.int64(), False), pa.field("kind", pa.string(), True),
                        pa.field("parts", pa.list_(PART_3), False), pa.field("length", pa.int64(), False),
                        pa.field("bytes", pa.int64(), False)])
PARTS_3 = pa.schema([pa.field("id", pa.string(), False), pa.field("hash", pa.binary(), False),
                     pa.field("change", pa.int64(), False), pa.field("value", pa.string(), False)])
CONSTRAINT_2 = "text_or_item"  # the CHECK constraint of contents at format 2, over its columns text and content
APP = "wpt-format-3"  # the Delta Lake transaction of parts that records the last batch stored
BATCH_BYTES = 256 << 20  # bytes of data files of contents a batch reads, about
ROWS = 64  # rows of contents read into memory at a time
WRITTEN = 32 << 20  # bytes of parts in one record batch written
LISTS = "format-3-lists"  # the folder of the depot's folder that keeps each batch's lists until contents is replaced
# how the step writes parts: the value without statistics or dictionary, a page's size checked every 16 rows
PARTS_WRITTEN_3 = WriterProperties(compression="SNAPPY", write_batch_size=16, column_properties={
    "value": ColumnProperties(statistics_enabled="NONE", dictionary_enabled=False)})
CONTENTS_WRITTEN_3 = WriterProperties(compression="SNAPPY")


def to_parts(tables):
    """Format 2 to 3 over the depot's tables (wpt_depot.legacy.Legacy)."""
    system, folder = _folder(tables)
    if "parts" in tables.columns("contents"):
        _remove(system, folder)
        _drop_replaced(tables)
        return
    if "parts" not in tables.exist():
        tables.create_table("parts", PARTS_3)
    contents = tables.table("contents")
    batches = _batches(contents)
    stored = tables.table("parts").transaction_version(APP)
    start = 0 if stored is None else stored + 1
    many = _many(tables, contents.file_uris())
    seen = _seen(tables, tables.table("parts").file_uris(), many) if start else set()
    system.create_dir(folder, recursive=True)
    for number in range(start, len(batches)):
        lists = []
        tables.append_stream("parts", _parts(tables, batches[number], lists, many, seen), PARTS_3, PARTS_WRITTEN_3,
                             transaction=(APP, number))
        _keep(system, folder, number, batches[number], lists)
    contents.alter.drop_constraint(CONSTRAINT_2, raise_if_not_exists=False)
    tables.rewrite("contents", pa.RecordBatchReader.from_batches(CONTENTS_3, _lists(tables, batches, system, folder)),
                   CONTENTS_WRITTEN_3)
    _remove(system, folder)
    _drop_replaced(tables)


def _batches(table) -> list[list[str]]:
    """The data files of contents in batches of about BATCH_BYTES, in the
    order of their names: the same batches each time the step runs."""
    actions = pa.table(table.get_add_actions(flatten=True))
    by_name = dict(zip((path.rsplit("/", 1)[-1] for path in actions.column("path").to_pylist()),
                       actions.column("size_bytes").to_pylist()))
    sizes = {file: by_name[file.rsplit("/", 1)[-1]] for file in table.file_uris()}
    found, batch, size = [], [], 0
    for file in sorted(sizes):
        if batch and size + sizes[file] > BATCH_BYTES:
            found.append(batch)
            batch, size = [], 0
        batch.append(file)
        size += sizes[file]
    return found + ([batch] if batch else [])


def _rows(tables, files: list[str]):
    """The rows of contents in the files, ROWS at a time, as lists of dicts."""
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=ROWS, columns=list(READ)):
            yield batch.to_pylist()


def _many(tables, files: list[str]) -> set[str]:
    """The ids of the texts with more than one content version, read from
    the files of contents: only their parts can be held by two batches."""
    if not files:
        return set()
    ids = ds.dataset(files, format="parquet").to_table(columns=["id"])
    counted = ids.group_by("id").aggregate([("id", "count")])
    return set(counted.filter(pc.greater(counted["id_count"], 1))["id"].to_pylist())


def _seen(tables, files: list[str], many: set[str]) -> set[tuple[str, bytes]]:
    """(id, hash) of the parts stored already, in the files of parts, of the texts named."""
    if not many or not files:
        return set()
    found = ds.dataset(files, format="parquet").to_table(columns=["id", "hash"],
                                                         filter=pc.field("id").isin(sorted(many)))
    return set(zip(found["id"].to_pylist(), found["hash"].to_pylist()))


def _cut(row: dict):
    """(the row of contents at format 3 but its parts, every part of its
    text): a row of contents at format 2 cut into parts."""
    value = row["text"] if row["kind"] is None else row["content"]
    found = list(parted(value))
    return ({"id": row["id"], "version": row["version"], "change": row["change"], "kind": row["kind"],
             "parts": [{"hash": one.hash, "length": one.length} for one in found],
             "length": sum(one.length for one in found), "bytes": sum(one.bytes for one in found)}, found)


def _parts(tables, files: list[str], lists: list[dict], many: set[str], seen: set):
    """The rows of parts the files' rows add, in record batches, each part
    of a text once (seen: those of the texts of many stored already); the
    rows of contents at format 3 go into lists."""
    pending, size = [], 0
    for rows in _rows(tables, files):
        for row in rows:
            listed, found = _cut(row)
            lists.append(listed)
            own = seen if row["id"] in many else set()
            for one in found:
                if (row["id"], one.hash) in own:
                    continue
                own.add((row["id"], one.hash))
                pending.append({"id": row["id"], "hash": one.hash, "change": row["change"], "value": one.value})
                size += one.bytes
                if size >= WRITTEN:
                    yield pa.RecordBatch.from_pylist(pending, schema=PARTS_3)
                    pending, size = [], 0
    if pending:
        yield pa.RecordBatch.from_pylist(pending, schema=PARTS_3)


def _folder(tables) -> tuple[pafs.FileSystem, str]:
    """(the file system of the depot's folder, the path in it of the folder
    LISTS that keeps the lists of the batches between runs of the step)."""
    return pafs.LocalFileSystem(), f"{tables.root.as_posix()}/{LISTS}"


def _remove(system: pafs.FileSystem, folder: str):
    if system.get_file_info(folder).type != pafs.FileType.NotFound:
        system.delete_dir(folder)


def _named(folder: str, number: int, files: list[str]) -> str:
    return f"{folder}/{number}-{hashlib.sha256(chr(0).join(files).encode()).hexdigest()[:16]}.parquet"


def _keep(system: pafs.FileSystem, folder: str, number: int, files: list[str], lists: list[dict]):
    path = _named(folder, number, files)
    pq.write_table(pa.Table.from_pylist(lists, schema=CONTENTS_3), path + ".part", filesystem=system)
    system.move(path + ".part", path)


def _lists(tables, batches: list[list[str]], system: pafs.FileSystem, folder: str):
    """The rows of contents at format 3, batch after batch: those a batch
    kept, or its rows cut again when its file is gone."""
    for number, files in enumerate(batches):
        path = _named(folder, number, files)
        if system.get_file_info(path).type == pafs.FileType.File:
            yield from pq.read_table(path, filesystem=system).cast(CONTENTS_3).to_batches()
            continue
        for rows in _rows(tables, files):
            yield pa.RecordBatch.from_pylist([_cut(row)[0] for row in rows], schema=CONTENTS_3)


def _drop_replaced(tables):
    """Drop the older states of contents when its folder keeps files its
    current state does not list (the rows at format 2, each text whole)."""
    if tables.table("contents").vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=True):
        tables.drop_history(["contents"])
