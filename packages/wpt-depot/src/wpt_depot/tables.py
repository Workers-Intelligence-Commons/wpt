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
listed files hold exactly the table's rows.

The folder is a path or, for a depot in a bucket, an s3:// address; the
bucket's settings (wpt_depot.settings.bucket_settings) go to delta-rs as
its storage options and to DuckDB as an S3 secret.
"""

from pathlib import Path
from urllib.parse import urlparse

import duckdb
import pyarrow as pa
import pyarrow.fs as pafs
from deltalake import DeltaTable
from deltalake.exceptions import DeltaError, TableNotFoundError

from .errors import NotADepot, VersionGone
from .schema import CONSTRAINTS, CURRENT, KEYS, SCHEMAS, TABLES, sql_name
from .settings import DEFAULT_REGION

KEPT = "interval 36500 days"  # how long a table keeps older states: until drop_history
RETENTION = {"delta.logRetentionDuration": KEPT, "delta.deletedFileRetentionDuration": KEPT}
NONE_KEPT = {"delta.logRetentionDuration": "interval 0 seconds",
             "delta.deletedFileRetentionDuration": "interval 0 seconds"}


class Tables:
    """The tables of the depot in the folder root (a Path), or in the bucket
    address root (an s3:// string) with these bucket settings."""

    def __init__(self, root, settings: dict | None = None):
        self.root = root if isinstance(root, str) else Path(root)
        self.settings = settings

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
            DeltaTable.create(self.uri(name), schema=SCHEMAS[name], mode="error", configuration=RETENTION,
                              storage_options=self.options())
            if name in CONSTRAINTS:
                self.table(name).alter.add_constraint(CONSTRAINTS[name])

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

    def connection(self) -> duckdb.DuckDBPyConnection:
        """A DuckDB connection where each table's current rows are a view
        under its SQL name (schema.sql_name), and each revised table's
        current state a view under its own name (schema.CURRENT)."""
        con = self._connect()
        for name in TABLES:
            self._view(con, name, self.table(name), sql_name(name))
        for name, query in CURRENT.items():
            con.execute(f'CREATE VIEW "{name}" AS {query}')
        return con

    def query(self, sql: str, params=None) -> pa.Table:
        """The rows of a DuckDB query over every table (connection), with
        params for its `?` placeholders."""
        con = self.connection()
        try:
            return con.sql(sql, params=params).to_arrow_table()
        finally:
            con.close()

    def read(self, name: str, version: int | None = None, columns: list[str] | None = None) -> pa.Table:
        """Every row of a table, at its current state or at version, in its
        schema now (only the columns named, when given): a field a struct of
        an older state lacks is null."""
        schema = SCHEMAS[name] if columns is None else pa.schema([SCHEMAS[name].field(one) for one in columns])
        con = self._connect()
        try:
            self._view(con, name, self.table(name, version))
            listed = ", ".join(f'"{one}"' for one in schema.names)
            return con.sql(f'SELECT {listed} FROM "{name}"').to_arrow_table().cast(schema)
        finally:
            con.close()

    @staticmethod
    def _view(con, name: str, table: DeltaTable, as_name: str | None = None):
        """The rows of the table name, in this state, as the view as_name (name by default)."""
        as_name = as_name or name
        files = table.file_uris()
        if files:
            listed = ", ".join("'" + file.replace("'", "''") + "'" for file in files)
            con.execute(f'CREATE VIEW "{as_name}" AS SELECT * FROM read_parquet([{listed}])')
        else:
            con.register(as_name, SCHEMAS[name].empty_table())

    def insert(self, name: str, found: pa.Table):
        """Add the rows whose key the table does not hold yet: adding the
        same rows again changes nothing."""
        if not found.num_rows:
            return
        self.table(name).merge(found, predicate=_same(name), source_alias="source",
                               target_alias="target").when_not_matched_insert_all().execute()

    def upsert(self, name: str, found: pa.Table):
        """Put the rows: each replaces the row of the same key, or is added."""
        if not found.num_rows:
            return
        (self.table(name).merge(found, predicate=_same(name), source_alias="source", target_alias="target")
         .when_matched_update_all().when_not_matched_insert_all().execute())

    def merge_update(self, name: str, source: pa.Table, updates: dict, predicate: str | None = None):
        """For each row of source, update the row of the same key with
        updates (SQL over target.* and source.*), where predicate holds."""
        if not source.num_rows:
            return
        (self.table(name).merge(source, predicate=_same(name), source_alias="source", target_alias="target")
         .when_matched_update(updates, predicate=predicate).execute())

    def update(self, name: str, updates: dict, predicate: str):
        self.table(name).update(updates, predicate=predicate)

    def delete(self, name: str, predicate: str):
        self.table(name).delete(predicate)

    def drop_history(self, names=TABLES):
        """Remove every older state of the tables named: its log before a
        new checkpoint and every file its current state does not list."""
        for name in names:
            self.table(name).alter.set_table_properties(NONE_KEPT)
            table = self.table(name)
            table.create_checkpoint()
            table.cleanup_metadata()
            table.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False, full=True)
            self.table(name).alter.set_table_properties(RETENTION)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """A DuckDB connection that gives times in UTC, as the tables hold
        them, and for a bucket reads it with the bucket's settings."""
        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")
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


def _same(name: str) -> str:
    """The merge predicate that matches a source row to the target row of the same key."""
    return " AND ".join(f"target.{key} = source.{key}" for key in KEYS[name])
