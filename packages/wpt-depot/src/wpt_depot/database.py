"""The depot's SQLite file: its connections, its SQL functions and views,
and the write that records each change.

A depot is one SQLite file in WAL mode. Several processes may open it:
each reads at any time, and SQLite's lock lets one write at a time, the
others waiting for it (WAIT), however long it writes. Inside a process, one
connection writes, under a lock its threads share, and each thread reads
through connections of its own.

Every connection sees the tables by their names (wpt_depot.schema) and the
views of the current state (schema.VIEWS), made on it as temporary views,
with two functions they use: part_text, a part's stored value as its text,
and content_too_big, which refuses a content version too big for one value
of SQLite. A person's SQL (the query of `wpt depot sql`, a condition
selecting texts) runs on a connection of its own that opens the file
read-only and whose authorizer lets a statement only read (READS): a
statement that writes, attaches another database, runs a pragma or loads an
extension is refused, and a query is one statement (one_statement). Its
time is bounded when `deadline` is set (a time.monotonic() value): past it,
the statement is interrupted.

A write (Database.write) is one change: one transaction that begins by
taking the write lock and its change number, the next after every change
of the depot (or after `past`, for a pull), and adds its row to changes.
SQLite's session records every row the transaction changes; before COMMIT
the session's changeset goes to a file of the waiting folder (WAITING,
beside the depot's file), named by the change number, the number of the
change before it and the change's time (waiting_name), with `.pending` at
its end, and the file and the folder are synced. Once the transaction is
committed the file is renamed without `.pending`: it is final. A crash
between the two leaves a pending file; the next write (or backup) of any
process keeps it, renamed, when its change is in the depot, and drops it
otherwise. It does so holding the write lock, which no other process holds
between writing its pending file and committing.

A failure inside a write rolls the whole transaction back: a refused or
failed change writes nothing. A deleted row's bytes are overwritten in the
file (secure_delete), so an erased passage leaves no copy in the file's free
pages, and after an erasure the WAL is folded into the file and emptied
(checkpoint). The changesets keep what each change wrote and deleted, an
erased passage included, until a backup sends them away (wpt_depot.backup).

A file a command makes whole (a new depot, a clone, an upgrade, a restore)
holds the marker FULL_COPY_NEXT in its waiting folder: its next backup
sends a full copy, which every change after it builds on. It also gets a
new timeline (TIMELINE, a random id in the waiting folder), which its
backups record: a depot restored to an earlier moment numbers its changes
on from there again, and its timeline keeps them apart from the changes
of the same numbers the depot it came from made.
"""

import contextlib
import os
import re
import secrets
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import apsw
import apsw.ext

from .errors import BadQuery, DepotExists, NotADepot, TimeLimitReached
from .parts import unpack
from .schema import SQL_STRING, VIEWS, ddl

WAIT = 24 * 3600 * 1000  # milliseconds a write waits for the write lock another process holds
WAITING = ".waiting"  # what the waiting folder's name adds to the depot file's name
PENDING, FINAL = ".pending", ".changeset"
FULL_COPY_NEXT = "full-copy-next"  # the marker of a file made whole: its next backup sends a full copy
TIMELINE = "timeline"  # the file of the waiting folder that holds the id of the file's timeline
WRITER = ("synchronous = FULL", "secure_delete = ON", "temp_store = FILE")  # the pragmas of a connection that writes
STEPS = 10_000  # SQLite steps between two checks of a person's statement's time limit
# what a person's statement may do: read rows, call functions, recurse in a common table expression
READS = {apsw.SQLITE_SELECT, apsw.SQLITE_READ, apsw.SQLITE_FUNCTION, apsw.SQLITE_RECURSIVE}
REFUSED_FUNCTIONS = {"load_extension"}
_NAME = re.compile(r"(\d+)-(\d+)-(\d{8}T\d{6}\.\d{6}Z)")


def waiting_name(number: int, previous: int, made_at: str) -> str:
    """The name, without its ending, of the file of the change number, made
    right after the change previous, at made_at (ISO 8601 in UTC)."""
    return f"{number:012d}-{previous:012d}-{datetime.fromisoformat(made_at):%Y%m%dT%H%M%S.%fZ}"


def parse_waiting(name: str) -> tuple[int, int, datetime] | None:
    """(change number, the change before it, its time) of a change file's
    name (with or without its ending), None for another name."""
    found = _NAME.match(name)
    if found is None:
        return None
    return (int(found[1]), int(found[2]),
            datetime.strptime(found[3], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=UTC))


def now() -> str:
    return datetime.now(UTC).isoformat()


def sync_folder(folder: Path):
    """Put the folder's entries (a file made, renamed or removed) on the disk; nothing where folders cannot be
    opened (Windows)."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    handle = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def rows(cursor) -> list[dict]:
    """The rows a cursor gives, as dicts by column name."""
    found, names = [], None
    for one in cursor:
        names = names or [column[0] for column in cursor.get_description()]
        found.append(dict(zip(names, one)))
    return found


class Database:
    """The depot's SQLite file at path; readonly: every connection opens it
    read-only (Depot.view)."""

    def __init__(self, path: Path, readonly: bool = False):
        self.path = Path(path)
        self.waiting = self.path.with_name(self.path.name + WAITING)
        self.readonly = readonly
        self.deadline: float | None = None  # when a person's statement is interrupted, a time.monotonic() value
        self._lock = threading.RLock()
        self._local = threading.local()
        self._writer = None

    # Connections

    def create(self):
        """A new file holding every table, empty, and its waiting folder
        with FULL_COPY_NEXT. DepotExists when the path holds a file."""
        if self.path.exists():
            raise DepotExists(f"{self.path} already exists: a new depot is a new file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = apsw.Connection(str(self.path))
        try:
            con.execute("PRAGMA journal_mode = WAL").fetchall()
            with con:
                for statement in ddl():
                    con.execute(statement)
        finally:
            con.close()
        self.mark_full_copy()

    def mark_full_copy(self):
        """Mark the file as made whole: a new timeline, and a full copy at its next backup."""
        self.waiting.mkdir(exist_ok=True)
        _write_synced(self.waiting / TIMELINE, secrets.token_hex(16))
        (self.waiting / FULL_COPY_NEXT).touch()
        sync_folder(self.waiting)

    def timeline(self) -> str:
        """The id of the file's timeline; a new one, with a full copy at the
        next backup, when the waiting folder holds none."""
        path = self.waiting / TIMELINE
        if not path.is_file():
            self.mark_full_copy()
        return path.read_text(encoding="utf-8").strip()

    def connect(self, person: bool = False, pragmas: tuple[str, ...] = ()) -> apsw.Connection:
        """A new connection with the functions and the views; person: for a
        person's SQL, read-only and authorized to read only. The pragmas run
        before the views are made, since changing temp_store drops every
        temporary view."""
        if not self.path.is_file():
            raise NotADepot(f"{self.path} holds no depot: a depot is one SQLite file")
        if person or self.readonly:
            con = apsw.Connection(f"file:{quote(self.path.as_posix())}?mode=ro",
                                  flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
        else:
            con = apsw.Connection(str(self.path))
        con.set_busy_timeout(WAIT)
        con.create_scalar_function("part_text", _part_text, 1, deterministic=True)
        con.create_scalar_function("content_too_big", _too_big, 3, deterministic=True)
        try:
            for pragma in pragmas:
                con.execute(f"PRAGMA {pragma}").fetchall()
            for name, query in VIEWS.items():
                con.execute(f"CREATE TEMP VIEW {name} AS {query}")
        except apsw.SQLError as error:
            con.close()
            raise NotADepot(f"{self.path} is not a depot of this layout: {error}") from None
        except apsw.NotADBError:
            con.close()
            raise NotADepot(f"{self.path} is not an SQLite file") from None
        if person:
            con.authorizer = _authorize
            con.set_progress_handler(self._past_deadline, STEPS)
        return con

    def _past_deadline(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def reader(self) -> apsw.Connection:
        """The connection this thread reads through: the writer while this
        thread writes (it sees what the write wrote so far), else the
        thread's own."""
        writing = getattr(self._local, "writing", None)
        if writing is not None:
            return writing
        if getattr(self._local, "reader", None) is None:
            self._local.reader = self.connect()
        return self._local.reader

    def person(self) -> apsw.Connection:
        """The connection this thread runs a person's SQL on."""
        if getattr(self._local, "person", None) is None:
            self._local.person = self.connect(person=True)
        return self._local.person

    def query(self, sql: str, params=None) -> list[dict]:
        """The rows of the depot's own query, on reader()."""
        return rows(self.reader().execute(sql, params))

    def person_query(self, sql: str, params=None) -> Iterator[tuple]:
        """The rows of a query that holds a person's SQL, as tuples, on
        person(): BadQuery when it is not one statement or SQLite refuses
        it."""
        con = self.person()
        one_statement(con, sql)
        try:
            cursor = con.execute(sql, params)
            yield from cursor
        except apsw.InterruptError:
            raise TimeLimitReached("the query ran past its time limit and was stopped") from None
        except (apsw.SQLError, apsw.AuthError, apsw.ConstraintError, apsw.RangeError, apsw.MismatchError,
                apsw.TooBigError, apsw.ReadOnlyError) as error:
            raise BadQuery(f"{type(error).__name__}: {error}") from None

    def person_rows(self, sql: str, params=None) -> list[dict]:
        """person_query's rows as dicts by column name."""
        names = person_columns(self.person(), sql)
        return [dict(zip(names, one)) for one in self.person_query(sql, params)]

    # Writing

    @contextlib.contextmanager
    def write(self, kind: str, past: int = 0) -> Iterator[tuple[apsw.Connection, int]]:
        """One change of that kind: (the writer connection, its change
        number), in a transaction that holds the write lock and is
        recorded (the module's docstring); committed when the block ends,
        rolled back whole when it raises."""
        with self._lock:
            con = self._write_connection()
            con.execute("BEGIN IMMEDIATE")
            session = pending = None
            try:
                self._settle(con)
                previous = con.execute("SELECT coalesce(max(id), 0) FROM changes").fetchall()[0][0]
                number, made_at = max(previous, past) + 1, now()
                session = apsw.Session(con, "main")
                session.attach(None)
                con.execute("INSERT INTO changes (id, kind, made_at) VALUES (?, ?, ?)", (number, kind, made_at))
                self._local.writing = con
                try:
                    yield con, number
                finally:
                    self._local.writing = None
                pending = self.waiting / (waiting_name(number, previous, made_at) + PENDING)
                self.waiting.mkdir(exist_ok=True)
                with open(pending, "wb") as file:
                    session.changeset_stream(file.write)
                    file.flush()
                    os.fsync(file.fileno())
                sync_folder(self.waiting)
                con.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(apsw.Error):
                    con.execute("ROLLBACK")
                if pending is not None:
                    pending.unlink(missing_ok=True)
                raise
            finally:
                if session is not None:
                    session.close()
            _final(pending)

    @contextlib.contextmanager
    def bulk(self) -> Iterator[apsw.Connection]:
        """A transaction of the writer that records no change: the rows a
        command writes into a file it is making whole (a clone, an upgrade),
        which holds FULL_COPY_NEXT."""
        with self._lock:
            con = self._write_connection()
            con.execute("BEGIN IMMEDIATE")
            self._local.writing = con
            try:
                yield con
                con.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(apsw.Error):
                    con.execute("ROLLBACK")
                raise
            finally:
                self._local.writing = None

    def settle(self):
        """Keep or drop the pending files of the waiting folder (the
        module's docstring), holding the write lock, writing no change."""
        if self.readonly:
            return
        with self._lock:
            con = self._write_connection()
            con.execute("BEGIN IMMEDIATE")
            try:
                self._settle(con)
            finally:
                con.execute("ROLLBACK")

    def _settle(self, con: apsw.Connection):
        if not self.waiting.is_dir():
            return
        for path in sorted(self.waiting.glob("*" + PENDING)):
            found = parse_waiting(path.name)
            held = found is not None and con.execute("SELECT 1 FROM changes WHERE id = ?", (found[0],)).fetchall()
            if held:
                _final(path)
            else:
                path.unlink(missing_ok=True)
                sync_folder(self.waiting)

    def checkpoint(self):
        """Fold the WAL into the file and empty it, so no page an erasure
        overwrote stays in the WAL; it waits for the readers that hold it."""
        with self._lock:
            self._write_connection().execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()

    def _write_connection(self) -> apsw.Connection:
        if self.readonly:
            raise NotADepot(f"{self.path} is opened to read only")
        if self._writer is None:
            self._writer = self.connect(pragmas=WRITER)
        return self._writer

    def waiting_files(self) -> list[Path]:
        """The final change files of the waiting folder, in the order of their change numbers."""
        if not self.waiting.is_dir():
            return []
        return sorted((path for path in self.waiting.glob("*" + FINAL) if parse_waiting(path.name)),
                      key=lambda path: parse_waiting(path.name)[0])

    def copy_to(self, target: Path) -> int:
        """A consistent full copy of the file into target, a new file, made
        with SQLite's backup in one step while writers go on: the number of
        the last change it holds."""
        source = self.connect()
        try:
            copy = apsw.Connection(str(target))
            try:
                with copy.backup("main", source, "main") as step:
                    step.step(-1)
                return copy.execute("SELECT coalesce(max(id), 0) FROM changes").fetchall()[0][0]
            finally:
                copy.close()
        finally:
            source.close()

    def close(self):
        for one in (self._writer, getattr(self._local, "reader", None), getattr(self._local, "person", None)):
            if one is not None:
                one.close()
        self._writer = self._local.reader = self._local.person = None


def one_statement(con: apsw.Connection, sql: str):
    """BadQuery unless sql is one statement that reads (a SELECT, WITH or VALUES)."""
    try:
        found = apsw.ext.query_info(con, sql)
    except apsw.Error as error:
        raise BadQuery(f"{type(error).__name__}: {error}") from None
    if found.query_remaining and found.query_remaining.strip().strip(";").strip():
        raise BadQuery("a query is one statement")
    if not found.is_readonly or found.is_explain:
        raise BadQuery("a query is one statement that reads")


def person_columns(con: apsw.Connection, sql: str) -> list[str]:
    """The names of the columns a query gives, without running it."""
    one_statement(con, sql)
    return [one[0] for one in apsw.ext.query_info(con, sql).description or ()]


def _write_synced(path: Path, text: str):
    """Write a small file whole: under another name, synced, then renamed into place."""
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def _final(pending: Path | None):
    """Rename a pending change file to its final name, once its change is
    committed; nothing when another process renamed it first."""
    if pending is None:
        return
    with contextlib.suppress(FileNotFoundError):
        pending.rename(pending.with_name(pending.name.removesuffix(PENDING) + FINAL))
    sync_folder(pending.parent)


def _part_text(value) -> str | None:
    return None if value is None else unpack(value)


def _too_big(text, version, size):
    raise BadQuery(f"the content version {version} of {text} is {size} bytes, more than the {SQL_STRING} a value "
                   "holds in SQLite: read it from its parts, as wpt depot show does")


def _authorize(action, first, second, database, trigger):
    if action == apsw.SQLITE_FUNCTION and (second or "").lower() in REFUSED_FUNCTIONS:
        return apsw.SQLITE_DENY
    return apsw.SQLITE_OK if action in READS else apsw.SQLITE_DENY
