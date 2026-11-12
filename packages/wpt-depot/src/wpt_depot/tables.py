"""The depot's tables on disk: one Delta Lake table per name of
schema.TABLES, each in its own folder under the depot's folder, written
through delta-rs and read as SQL through DuckDB.

Every older state of a table stays readable (Delta Lake's time travel)
until drop_history removes them all: the tables keep their log and their
replaced files for KEPT, and only drop_history shortens that, for the time
it runs. The depot's history is its revision rows and change numbers
(wpt_depot.schema), never these states: only a pull reads them, to read
its upstream again as it read it.

In SQL, a revised table's rows are `<name>_revisions` and `<name>` is the
view of its current state (schema.CURRENT).

Reads go through DuckDB's own Parquet reader over the files a table's
current (or named) state lists, never through delta-rs's Arrow reader,
whose file system keeps the Python interpreter from exiting once it read
rows. That reader is right for these tables because only this app writes
them and delta-rs writes no deletion vectors and no column mapping: the
listed files hold exactly the table's rows. Each view gives every column
of the table's schema (schema.SCHEMAS), null where its files lack it (a
table of an older format's layout), and a table a later format added
(schema.ADDED) reads as empty in a depot that lacks it. The files of a table a format step
widened in place (schema.WIDENED, Tables.add_column) do not all hold the
same columns, since the files written before the step lack the column it
added: its view matches each file's columns by name, which reads every
file's footer before the query runs. Every other table's files hold the
same columns, and its view reads them by the first file's. A query opens only the
tables it reads, as DuckDB's parser names them, since opening a table
replays its log and lists its files, and a query that reads no text
(_reads_text) sees the views of schema.LIGHT, which never join the parts
of a content version (schema.CURRENT). Strings come out of DuckDB as Arrow
large strings, so a value past 2 GiB, or a batch of values past it,
reaches Python whole.

Every write adds at least one data file and one log entry, which every
later read of the table opens again. delta-rs sums a table's log up in a
checkpoint every 100 commits by itself, so a read replays at most 100 log
entries after it. A checkpoint also records every file the table ever
replaced, for as long as KEPT keeps older states, so opening a table
still grows with the number of files replaced before, merged files
included.

Every COMPACT_EVERY storage versions, a write also merges the table's
data files smaller than COMPACT_SIZE into one (Tables._compact), so a
read opens at most about COMPACT_EVERY small files plus the merged ones,
whatever the number of writes before. The merged files stay with the
older states, as every replaced file does. The parts table, which holds
each content, and contents, which lists its parts, are never merged
(UNCOMPACTED), since replaced files of parts would keep a second copy of
each content: their file counts grow by one at every write that adds
contents.

The column that holds a text's content (a part's value) is written with
no statistics and no dictionary (UNMEASURED): a data file's footer and the
table's log would otherwise hold the first characters of its smallest and
largest value, a second copy of a text's start that only the data file
should hold. The Parquet writer closes a page once it passes its page
size limit, and it checks that limit every PAGE_ROWS rows: a page of parts
holds at most PAGE_ROWS parts past the limit, so the largest part
(wpt_depot.parts.SIZES) times PAGE_ROWS stays within the 2 GiB a Parquet
page holds.

The folder is a path or, for a depot in a bucket, an s3:// address; the
bucket's settings (wpt_depot.settings.bucket_settings) go to delta-rs as
its storage options and to DuckDB as an S3 secret.
"""

import json
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import pyarrow as pa
import pyarrow.fs as pafs
from deltalake import ColumnProperties, CommitProperties, DeltaTable, Transaction, WriterProperties, write_deltalake
from deltalake.exceptions import DeltaError, TableNotFoundError
from deltalake.schema import Field
from wpt_utils.lines import WARNING, say

from .errors import NotADepot, VersionGone
from .schema import ADDED, CONSTRAINTS, CURRENT, KEYS, LIGHT, SCHEMAS, TABLES, TEXT_NAMES, WIDENED, sql_name
from .settings import DEFAULT_REGION

KEPT = "interval 36500 days"  # how long a table keeps older states: until drop_history
RETENTION = {"delta.logRetentionDuration": KEPT, "delta.deletedFileRetentionDuration": KEPT}
# the columns written without statistics, by table: those that hold a text's content
UNMEASURED = {"parts": ("value",), "journal": ("change",)}
# the rows after which the writer checks a page's size, by table: a page of parts ends at most 16 parts of at
# most 64 MiB past its size limit
PAGE_ROWS = {"parts": 16}
STREAMED_ROWS = 65536  # the rows of a batch stream gives of a table PAGE_ROWS does not name
COMPACT_EVERY = 100  # storage versions of a table between two compactions (Tables._compact)
COMPACT_SIZE = 1 << 20  # bytes: a data file smaller than this is merged again at each compaction
UNCOMPACTED = ("contents", "parts")  # merging parts would keep a second copy of each content in the files it replaced
NONE_KEPT = {"delta.logRetentionDuration": "interval 0 seconds",
             "delta.deletedFileRetentionDuration": "interval 0 seconds"}


class Tables:
    """The tables of the depot in the folder root (a Path), or in the bucket
    address root (an s3:// string) with these bucket settings."""

    def __init__(self, root, settings: dict | None = None):
        self.root = root if isinstance(root, str) else Path(root)
        self.settings = settings
        self._database = None  # the DuckDB database every connection is a cursor of (_connect)
        self._starting = threading.Lock()

    def uri(self, name: str) -> str:
        return f"{self.root}/{name}" if isinstance(self.root, str) else str(self.root / name)

    def options(self) -> dict | None:
        """delta-rs's storage options for the bucket; None for a folder."""
        if not isinstance(self.root, str):
            return None
        found = self.settings
        options = {"AWS_ACCESS_KEY_ID": found["access_key_id"], "AWS_SECRET_ACCESS_KEY": found["secret_access_key"],
                   "AWS_REGION": found.get("region", DEFAULT_REGION), "aws_conditional_put": "etag"}
        if "session_token" in found:
            options["AWS_SESSION_TOKEN"] = found["session_token"]
        if "endpoint_url" in found:
            options["AWS_ENDPOINT_URL"] = found["endpoint_url"]
            if found["endpoint_url"].startswith("http://"):
                options["AWS_ALLOW_HTTP"] = "true"
        return options

    def filesystem(self) -> tuple[pafs.FileSystem, str]:
        """(the file system that holds the folder, as pyarrow reaches it, and
        the folder's path in it): for a bucket, S3 with the bucket's settings."""
        if not isinstance(self.root, str):
            return pafs.LocalFileSystem(), self.root.as_posix()
        found = self.settings
        endpoint = urlparse(found["endpoint_url"]) if "endpoint_url" in found else None
        system = pafs.S3FileSystem(access_key=found["access_key_id"], secret_key=found["secret_access_key"],
                                   session_token=found.get("session_token"),
                                   region=found.get("region", DEFAULT_REGION),
                                   endpoint_override=endpoint.netloc if endpoint else None,
                                   scheme=endpoint.scheme if endpoint else "https")
        return system, self.root.removeprefix("s3://")

    def exist(self) -> list[str]:
        """The names of the tables the folder holds."""
        return [name for name in TABLES if DeltaTable.is_deltatable(self.uri(name), self.options())]

    def create(self):
        """Every table, empty, with its constraints."""
        for name in TABLES:
            self.create_table(name, SCHEMAS[name], CONSTRAINTS.get(name))

    def create_table(self, name: str, schema: pa.Schema, constraints: dict | None = None):
        """One table, empty, of that layout, with its constraints (a step of
        an upgrade makes the tables its format adds)."""
        DeltaTable.create(self.uri(name), schema=schema, mode="error", configuration=RETENTION,
                          storage_options=self.options())
        if constraints:
            self.table(name).alter.add_constraint(constraints)

    def table(self, name: str, version: int | None = None) -> DeltaTable:
        """The table's current state, or its state at version. NotADepot
        when it is missing; VersionGone when the version is no longer kept."""
        try:
            return DeltaTable(self.uri(name), version=version, storage_options=self.options())
        except TableNotFoundError:
            if version is not None and DeltaTable.is_deltatable(self.uri(name), self.options()):
                raise VersionGone(f"{self.root}: the state {version} of {name} is no longer kept") from None
            raise NotADepot(f"{self.root} holds no table {name}") from None
        except DeltaError as error:
            if version is not None:
                raise VersionGone(f"{self.root}: the state {version} of {name} is no longer kept ({error})") from None
            raise

    def version(self, name: str) -> int:
        return self.table(name).version()

    def connection(self, reading: str | None = None) -> duckdb.DuckDBPyConnection:
        """A DuckDB connection where each table's current rows are a view
        under its SQL name (schema.sql_name), and each revised table's
        current state a view under its own name (schema.CURRENT): only the
        views the query `reading` names, with those they read, when given.
        A view costs the opening of its table's current state, so a query
        opens only the tables it reads; one that reads no text sees the
        views of schema.LIGHT instead (_reads_text)."""
        con = self._connect()
        views = CURRENT if reading is None or _reads_text(con, reading) else {**CURRENT, **LIGHT}
        names = _read_by(con, reading, views)
        for name in TABLES:
            if names is None or sql_name(name) in names:
                try:
                    table = self.table(name)
                except NotADepot:
                    if name not in ADDED:
                        raise
                    table = None  # a table a later format added, which this depot lacks yet: no row
                self._view(con, name, table, sql_name(name))
        for name, query in views.items():
            if names is None or name in names:
                con.execute(f'CREATE TEMP VIEW "{name}" AS {query}')
        return con

    def query(self, sql: str, params=None) -> pa.Table:
        """The rows of a DuckDB query over the tables (connection), with
        params for its `?` placeholders."""
        con = self.connection(sql)
        try:
            return con.sql(sql, params=params).to_arrow_table()
        finally:
            con.close()

    def read(self, name: str, version: int | None = None, columns: list[str] | None = None) -> pa.Table:
        """Every row of a table, at its current state or at version, in its
        schema now (only the columns named, when given): a column that state
        lacks (an older format's layout, wpt_depot.schema) is null and may
        be, and so is a field a struct of an older state lacks."""
        wanted = SCHEMAS[name] if columns is None else pa.schema([SCHEMAS[name].field(one) for one in columns])
        table = self.table(name, version)
        held = set(pa.schema(table.schema()).names)
        schema = pa.schema([one if one.name in held else one.with_nullable(True) for one in wanted])
        con = self._connect()
        try:
            self._view(con, name, table)
            listed = ", ".join(f'"{one}"' if one in held else f'NULL AS "{one}"' for one in schema.names)
            return con.sql(f'SELECT {listed} FROM "{name}"').to_arrow_table().cast(schema)
        finally:
            con.close()

    def stream(self, name: str, version: int | None = None, where: str = "true", params=None
               ) -> Iterator[pa.RecordBatch]:
        """The rows of a table of the code's layout, at its current state or
        at version, where the SQL condition where holds (with params for its
        `?` placeholders), as record batches of its schema, at most
        PAGE_ROWS rows each when the table names a number: a table bigger
        than memory read a few rows at a time. The table is opened before
        this returns, so the batches can feed a write of another table
        (append_stream), during which no table can be opened."""
        table = self.table(name, version)
        con = self._connect()
        try:
            self._view(con, name, table)
            found = con.sql(f'SELECT * FROM "{name}" WHERE {where}', params=params).to_arrow_reader(
                PAGE_ROWS.get(name, STREAMED_ROWS))
        except BaseException:
            con.close()
            raise

        def batches():
            try:
                for batch in found:
                    yield batch.cast(SCHEMAS[name])
            finally:
                con.close()
        return batches()

    @staticmethod
    def _view(con, name: str, table: DeltaTable | None, as_name: str | None = None):
        """The rows of the table name, in this state (None: no row), as the
        view as_name (name by default), with every column of SCHEMAS[name]:
        a column its files lack (of an older format's layout, or, for a
        table of WIDENED, written before the column was added) is null in
        their rows."""
        as_name = as_name or name
        layout = f"{as_name}__layout"
        con.register(layout, SCHEMAS[name].empty_table())
        files = [] if table is None else table.file_uris()
        if files:
            listed = ", ".join("'" + file.replace("'", "''") + "'" for file in files)
            by_name = ", union_by_name = true" if name in WIDENED else ""
            con.execute(f'CREATE TEMP VIEW "{as_name}" AS SELECT * FROM "{layout}" UNION ALL BY NAME '
                        f'SELECT * FROM read_parquet([{listed}]{by_name})')
        else:
            con.execute(f'CREATE TEMP VIEW "{as_name}" AS SELECT * FROM "{layout}"')

    def append(self, name: str, found: pa.Table):
        """Add the rows in one commit, reading none of the rows the table
        holds, so its cost does not grow with them: for rows whose keys the
        table cannot hold yet. Adding them again adds them twice."""
        if found.num_rows:
            table = self.table(name)
            write_deltalake(table, found, mode="append", writer_properties=_properties(name))
            self._compact(name, table)

    def append_stream(self, name: str, found, transaction: tuple[str, int] | None = None,
                      schema: pa.Schema | None = None, properties: WriterProperties | None = None):
        """Add the rows of a stream (a RecordBatchReader, or record batches
        of schema, the table's by default) in one commit, as append does,
        holding only a part of them in memory at a time, its files written
        with properties (the table's by default); with transaction (an app's
        name and a number), the commit also records that number as the app's
        Delta Lake transaction. A step of an upgrade passes its own schema
        and properties, never the code's current layout."""
        table = self.table(name)
        if not isinstance(found, pa.RecordBatchReader):
            found = pa.RecordBatchReader.from_batches(schema or SCHEMAS[name], found)
        committed = None if transaction is None else CommitProperties(app_transactions=[Transaction(*transaction)])
        write_deltalake(table, found, mode="append", writer_properties=properties or _properties(name),
                        commit_properties=committed)
        self._compact(name, table)

    def insert(self, name: str, found: pa.Table):
        """Add the rows whose key the table does not hold yet: adding the
        same rows again changes nothing. A merge, which reads every row of
        the table."""
        if not found.num_rows:
            return
        table = self.table(name)
        table.merge(found, predicate=_same(name), source_alias="source", target_alias="target",
                    writer_properties=_properties(name)).when_not_matched_insert_all().execute()
        self._compact(name, table)

    def upsert(self, name: str, found: pa.Table):
        """Put the rows: each replaces the row of the same key, or is added."""
        if not found.num_rows:
            return
        table = self.table(name)
        (table.merge(found, predicate=_same(name), source_alias="source", target_alias="target",
                     writer_properties=_properties(name))
         .when_matched_update_all().when_not_matched_insert_all().execute())
        self._compact(name, table)

    def merge_update(self, name: str, source: pa.Table, updates: dict, predicate: str | None = None):
        """For each row of source, update the row of the same key with
        updates (SQL over target.* and source.*), where predicate holds."""
        if not source.num_rows:
            return
        table = self.table(name)
        (table.merge(source, predicate=_same(name), source_alias="source", target_alias="target",
                     writer_properties=_properties(name))
         .when_matched_update(updates, predicate=predicate).execute())
        self._compact(name, table)

    def columns(self, name: str) -> list[str]:
        """The names of the columns the table's current state has, which an
        older format of the depot's layout may lack (wpt_depot.schema)."""
        return self.layout(name).names

    def layout(self, name: str) -> pa.Schema:
        """The schema of the table's current state: schema.SCHEMAS[name], or
        an older format's layout of it."""
        return pa.schema(self.table(name).schema())

    def add_column(self, name: str, column: pa.Field):
        """Add a column that may be null to the table (a step of an upgrade,
        wpt_depot.schema.STEPS): one commit that changes only its log, no
        data file, so its cost does not grow with the rows it holds. Every
        row it holds then reads the column as null."""
        self.table(name).alter.add_columns([Field.from_arrow(column.with_nullable(True))])

    def rewrite(self, name: str, found, properties: WriterProperties | None = None):
        """Replace every row of the table with found (a table or a
        RecordBatchReader), its columns becoming found's (a step of an
        upgrade, wpt_depot.schema.STEPS), its files written with properties
        (the table's by default): one commit, so the table holds either its
        old rows or these. Its settings and constraints stay."""
        write_deltalake(self.table(name), found, mode="overwrite", schema_mode="overwrite",
                        writer_properties=properties or _properties(name))

    def update(self, name: str, updates: dict, predicate: str):
        table = self.table(name)
        table.update(updates, predicate=predicate, writer_properties=_properties(name))
        self._compact(name, table)

    def delete(self, name: str, predicate: str):
        table = self.table(name)
        table.delete(predicate, writer_properties=_properties(name))
        self._compact(name, table)

    @staticmethod
    def _compact(name: str, table: DeltaTable):
        """After a write to the table name, whose state is table now: at
        every COMPACT_EVERY-th storage version, its data files smaller than
        COMPACT_SIZE merged in one commit that changes no row. A write that
        ends on another version, or a table of UNCOMPACTED, is left as it is.
        A merge that fails is reported as a warning and leaves the write as
        it was committed."""
        if name not in UNCOMPACTED and table.version() % COMPACT_EVERY == 0:
            try:
                table.optimize.compact(target_size=COMPACT_SIZE, writer_properties=_properties(name))
            except (DeltaError, OSError) as error:
                # the write is committed already, and a merge that fails commits nothing: the
                # write stands, and the table's small files wait for its next COMPACT_EVERY-th version
                say(f"Merging the small files of the table {name} failed, its rows unchanged: {error}", WARNING)

    def drop_history(self, names=None):
        """Remove every older state of the tables named (every table the
        depot has by default): its log before a new checkpoint and every
        file its current state does not list."""
        for name in self.exist() if names is None else names:
            self.table(name).alter.set_table_properties(NONE_KEPT)
            table = self.table(name)
            table.create_checkpoint()
            table.cleanup_metadata()
            table.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False, full=True)
            self.table(name).alter.set_table_properties(RETENTION)

    def bare(self) -> duckdb.DuckDBPyConnection:
        """A DuckDB connection that reaches the tables' files (the bucket's
        settings included) and holds no view: for a reader that names the
        files it reads itself, as a step of an upgrade does."""
        return self._connect()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """A DuckDB connection that gives times in UTC, as the tables hold
        them: a cursor of one in-memory database these tables start once,
        since starting a database costs far more than a cursor. Its views
        are temporary, so they are its own."""
        with self._starting:
            if self._database is None:
                self._database = self._start()
        return self._database.cursor()

    def _start(self) -> duckdb.DuckDBPyConnection:
        """An in-memory DuckDB database that gives times in UTC, writes what
        does not fit in memory into a folder of its own under the system's
        temporary folder (never the folder the command runs from) and, for
        a bucket, reads it with the bucket's settings."""
        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")
        con.execute("SET arrow_large_buffer_size = true")
        con.execute("SET temp_directory = ?", [str(Path(tempfile.gettempdir()) / f"wpt-depot-{os.getpid()}")])
        if isinstance(self.root, str):
            found = self.settings
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
            endpoint = urlparse(found["endpoint_url"]) if "endpoint_url" in found else None
            parts = {"TYPE": "S3", "KEY_ID": found["access_key_id"], "SECRET": found["secret_access_key"],
                     "REGION": found.get("region", DEFAULT_REGION)}
            if "session_token" in found:
                parts["SESSION_TOKEN"] = found["session_token"]
            if endpoint is not None:
                parts.update(ENDPOINT=endpoint.netloc, URL_STYLE="path", USE_SSL="true" if endpoint.scheme == "https" else "false")
            listed = ", ".join(f"{key} '{str(value).replace(chr(39), chr(39) * 2)}'" for key, value in parts.items())
            con.execute(f"CREATE SECRET ({listed})")
        return con


def _read_by(con: duckdb.DuckDBPyConnection, sql: str | None, views: dict = CURRENT) -> set[str] | None:
    """The names of the views the query sql reads, directly or through one
    of views (schema.CURRENT, or with schema.LIGHT), as DuckDB's parser
    finds them; None (every view) for no query, or one the parser refuses,
    so that running it reports its own error."""
    if sql is None:
        return None
    found, waiting = set(), {sql}
    while waiting:
        tree = _tree(con, waiting.pop())
        if tree is None:
            return None
        for name in _base_tables(tree) - found:
            found.add(name)
            if name in views:
                waiting.add(views[name])
    return found


def _tree(con: duckdb.DuckDBPyConnection, sql: str) -> dict | None:
    """The parse tree of sql (json_serialize_sql), None when the parser refuses it."""
    tree = json.loads(con.execute("SELECT json_serialize_sql(?)", [sql]).fetchall()[0][0])
    return None if tree["error"] else tree


def _reads_text(con: duckdb.DuckDBPyConnection, sql: str) -> bool:
    """Whether the query sql may read a column of a view that holds a text
    (schema.TEXT_NAMES): it names one, but through a table other than
    texts and contents (as licenses.text), or a table or subquery as a
    whole, or it has a `*` but one of such another table's (as l.*). True
    when the parser refuses it."""
    tree = _tree(con, sql)
    if tree is None:
        return True
    tables = {}  # each name and alias of a table or subquery of the query: its table's name (None: a subquery)
    for node in _nodes(tree):
        if node.get("type") in ("BASE_TABLE", "SUBQUERY", "TABLE_FUNCTION"):
            name = node.get("table_name", "").lower() or None
            for one in (name, node.get("alias", "").lower()):
                if one:
                    tables[one] = name

    def other(qualifier: str) -> bool:  # names a table that is not texts or contents
        return tables.get(qualifier) not in (None, "texts", "contents")
    for node in _nodes(tree):
        if node.get("class") == "STAR" and not other(node.get("relation_name", "").lower()):
            return True
        if node.get("class") == "COLUMN_REF":
            names = [one.lower() for one in node["column_names"]]
            if len(names) == 1 and (names[0] in TEXT_NAMES or names[0] in tables):
                return True
            if len(names) > 1 and not other(names[0]) and any(one in TEXT_NAMES for one in names[1:]):
                return True
    return False


def _nodes(node):
    """Every object of a parse tree, depth first."""
    if isinstance(node, list):
        for one in node:
            yield from _nodes(one)
    elif isinstance(node, dict):
        yield node
        for one in node.values():
            yield from _nodes(one)


def _base_tables(node) -> set[str]:
    """The names, in lower case as SQL does not tell cases apart, of every
    table a parse tree of json_serialize_sql reads."""
    if isinstance(node, list):
        return set().union(*map(_base_tables, node))
    if not isinstance(node, dict):
        return set()
    own = {node["table_name"].lower()} if node.get("type") == "BASE_TABLE" else set()
    return own.union(*map(_base_tables, node.values()))


def _same(name: str) -> str:
    """The merge predicate that matches a source row to the target row of the same key."""
    return " AND ".join(f"target.{key} = source.{key}" for key in KEYS[name])


def _properties(name: str) -> WriterProperties:
    """How a table's data files are written: compressed with Snappy, the
    columns of UNMEASURED without statistics or dictionary, at most
    PAGE_ROWS rows to a page when the table names a number."""
    return WriterProperties(compression="SNAPPY", write_batch_size=PAGE_ROWS.get(name), column_properties={
        column: ColumnProperties(statistics_enabled="NONE", dictionary_enabled=False)
        for column in UNMEASURED.get(name, ())})
