"""A depot of format 3 or older: a folder of Delta Lake tables, one folder
per table, which `wpt depot upgrade` brings up to the code's format
(wpt_depot.upgrade). Nothing but the upgrade reads or writes it.

Legacy reaches the tables through delta-rs and pyarrow, as the upgrade
needs them: the format the folder holds, the data files of a table's
current state, and the steps from each format up to 3, each of which
changes the folder in place to the next format's layout and changes
nothing more when it runs again (STEPS). Each step records the format it
reached as a new revision of the about table, numbered as a done entry of
the folder's journal, the change log every write of these formats went
through first: an entry is a JSON object whose `kind` names the change, and
a change left undone has its rows written in some tables and not in others.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import CommitProperties, DeltaTable, Transaction, WriterProperties, write_deltalake
from deltalake.schema import Field

from .errors import NotADepot

# the tables of a depot of format 3, by the name of their folder
TABLES = ("about", "journal", "licenses", "notices", "kinds", "decisions", "texts", "contents", "parts", "libraries",
          "memberships", "pins", "erasures", "datasets", "pulls")
ADDED = {"parts": 3}  # the tables a format added, with that format: a depot of an older format lacks them
KEPT = "interval 36500 days"  # how long a table keeps older states: until drop_history
RETENTION = {"delta.logRetentionDuration": KEPT, "delta.deletedFileRetentionDuration": KEPT}
NONE_KEPT = {"delta.logRetentionDuration": "interval 0 seconds",
             "delta.deletedFileRetentionDuration": "interval 0 seconds"}
WRITTEN = WriterProperties(compression="SNAPPY")
TIME = pa.timestamp("us", tz="UTC")


class Legacy:
    """The Delta Lake tables of the depot in the folder path."""

    def __init__(self, path: Path):
        self.root = Path(path)

    def uri(self, name: str) -> str:
        return str(self.root / name)

    def exist(self) -> list[str]:
        """The names of the tables the folder holds."""
        return [name for name in TABLES if DeltaTable.is_deltatable(self.uri(name))]

    def check(self):
        """NotADepot when a table is missing (but one a later format added) or the about table holds no row."""
        held = self.exist()
        missing = [name for name in TABLES if name not in held and name not in ADDED]
        if missing:
            raise NotADepot(f"{self.root} is not a depot: no table {', '.join(missing)}")
        if not self.table("about").to_pyarrow_dataset().count_rows():
            raise NotADepot(f"{self.root} is not a depot: its about table holds no row")

    def table(self, name: str) -> DeltaTable:
        if not DeltaTable.is_deltatable(self.uri(name)):
            raise NotADepot(f"{self.root} holds no table {name}")
        return DeltaTable(self.uri(name))

    def columns(self, name: str) -> list[str]:
        """The names of the columns the table's current state has."""
        return pa.schema(self.table(name).schema()).names

    def files(self, name: str) -> list[str]:
        """The data files of the table's current state, in the order of their names."""
        return sorted(self.table(name).file_uris())

    def read(self, name: str, columns: list[str] | None = None) -> pa.Table:
        """Every row of a small table (the columns named, when given)."""
        return self.table(name).to_pyarrow_dataset().to_table(columns=columns)

    def batches(self, name: str, file: str, rows: int):
        """The rows of one data file of the table, rows at a time, as lists
        of dicts in the table's current layout: a column the file lacks (one
        a later format added in place) reads as None."""
        names = self.columns(name)
        for batch in pq.ParquetFile(file).iter_batches(batch_size=rows):
            held = batch.to_pylist()
            yield [{column: one.get(column) for column in names} for one in held]

    def format(self) -> int:
        """The format of the depot's layout: its current about row's, or 0 when the about table has no column format."""
        self.check()
        about = self.read("about")
        if "format" not in about.column_names:
            return 0
        found = max(about.to_pylist(), key=lambda one: one["revision"])
        return found["format"]

    def undone(self) -> list[int]:
        """The ids of the journal's entries not done, in order: changes an older code never finished."""
        found = self.read("journal", columns=["id", "done"]).to_pylist()
        return sorted(one["id"] for one in found if not one["done"])

    def create_table(self, name: str, schema: pa.Schema):
        DeltaTable.create(self.uri(name), schema=schema, mode="error", configuration=RETENTION)

    def append_stream(self, name: str, found, schema: pa.Schema, properties: WriterProperties = WRITTEN,
                      transaction: tuple[str, int] | None = None):
        """Add the record batches of found in one commit; with transaction (an
        app's name and a number), the commit records that number as the app's
        Delta Lake transaction."""
        committed = None if transaction is None else CommitProperties(app_transactions=[Transaction(*transaction)])
        write_deltalake(self.table(name), pa.RecordBatchReader.from_batches(schema, found), mode="append",
                        writer_properties=properties, commit_properties=committed)

    def add_column(self, name: str, column: pa.Field):
        """Add a column that may be null: one commit that changes only the table's log."""
        self.table(name).alter.add_columns([Field.from_arrow(column.with_nullable(True))])

    def rewrite(self, name: str, found, properties: WriterProperties = WRITTEN):
        """Replace every row of the table with found (a table or a
        RecordBatchReader), its columns becoming found's, in one commit."""
        write_deltalake(self.table(name), found, mode="overwrite", schema_mode="overwrite",
                        writer_properties=properties)

    def drop_history(self, names):
        """Remove every older state of the tables named: its log before a new
        checkpoint and every file its current state does not list."""
        for name in names:
            self.table(name).alter.set_table_properties(NONE_KEPT)
            table = self.table(name)
            table.create_checkpoint()
            table.cleanup_metadata()
            table.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False, full=True)
            self.table(name).alter.set_table_properties(RETENTION)

    def record_format(self, reached: int):
        """Record the format a step reached: a done entry of the journal, and
        a new revision of the about row numbered by it, in the about table's
        layout (format 1 and on)."""
        journal = self.read("journal", columns=["id"])["id"].to_pylist()
        about = self.read("about").to_pylist()
        number = max([*journal, *(one["change"] for one in about)], default=0) + 1
        entry = json.dumps({"kind": "about", "values": {"format": reached}}, sort_keys=True)
        write_deltalake(self.table("journal"), pa.table({"id": pa.array([number], pa.int64()),
                                                         "change": pa.array([entry], pa.string()),
                                                         "done": pa.array([True], pa.bool_())}),
                        mode="append", writer_properties=WRITTEN)
        current = max(about, key=lambda one: one["revision"])
        row = {**current, "revision": current["revision"] + 1, "change": number, "format": reached}
        write_deltalake(self.table("about"), pa.Table.from_pylist([row], schema=pa.schema(self.table("about").schema())),
                        mode="append", writer_properties=WRITTEN)


# the about table at format 1: format 0's columns, then format
ABOUT_1 = pa.schema([pa.field("revision", pa.int64(), False), pa.field("change", pa.int64(), False),
                     pa.field("name", pa.string(), False), pa.field("maintainer", pa.string(), False),
                     pa.field("takedown_contact", pa.string(), False), pa.field("upstream", pa.string(), True),
                     pa.field("format", pa.int64(), False)])


def add_format(legacy: Legacy):
    """Format 0 to 1: the about table gets the column format, every row it
    holds of format 0, in one commit that rewrites it; nothing when it has
    the column already."""
    if "format" in legacy.columns("about"):
        return
    found = legacy.read("about", columns=[name for name in ABOUT_1.names if name != "format"])
    legacy.rewrite("about", found.append_column("format", pa.array([0] * found.num_rows, pa.int64())).cast(ABOUT_1))


# the column the texts table gets at format 2: its scores, null for none
SCORES_2 = pa.field("scores", pa.list_(pa.struct([("name", pa.string()), ("value", pa.float64()),
                                                   ("content_version", pa.int64())])), nullable=True)


def add_scores(legacy: Legacy):
    """Format 1 to 2: the texts table gets the column scores, null in every
    row it holds, which means no scores. Only the table's log changes: no
    row is read or written again. Nothing when the table has the column
    already."""
    if SCORES_2.name not in legacy.columns("texts"):
        legacy.add_column("texts", SCORES_2)


def _to_parts(legacy: Legacy):
    from .format3 import to_parts
    to_parts(legacy)


# the step from each format to the next, by the format it starts from
STEPS = {0: add_format, 1: add_scores, 2: _to_parts}
LAST = 3  # the format the last step reaches: the newest format of a folder of Delta Lake tables

