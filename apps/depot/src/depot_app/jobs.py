"""The queue of a depot's jobs: the long work the web app runs for anyone
(making a snapshot, a write over a selection of texts, ordering a snapshot
by length), each in a process of its own, so the web app keeps the other
cores and answers while a job runs.

The jobs run in three lanes, one job at a time in each (LANES): the write
lane runs the writes over many texts (decisions, licenses, scores, tags),
short and each holding the depot's write lock while it writes; the heavy
lane makes the snapshots, each holding the write lock for one short
transaction at its end; the read lane runs the reads that make what a page
shows, kept in the jobs' folder and written nowhere in the depot (a
snapshot's order by length, depot_app.web.snapshots). So a snapshot never
holds back a decision over many texts nor the strip of a snapshot, and a
write run in the request waits only for the write lane (writing).

The queue is a small SQLite file beside the depot, <depot file>.jobs,
which nothing backs up: it holds each job's kind and lane, what it does in
words, what it is for (its key: a sizes job's snapshot id; its target: what
it makes), the operation it runs and its arguments, its state, its progress
and its answer, and the jobs' other state (depot_app.downloads). Each
lane's runner takes the oldest queued job of its lane, marks it running and
runs it in a spawned process; it looks at the queue when a job is added in
its own process, and every POLL seconds. A job's state is queued, running,
done (its answer), refused (the failure that stopped it, with its HTTP
status; what it is refused writes nothing), failed (the process ended
without an answer, or with a failure the routes do not name) or cancelled.
At most one snapshot job is queued or running, and each snapshot id has at
most one sizes job (Busy otherwise). Only the command line cancels a job
(depot jobs cancel): a queued one never runs, and a running one's
process is stopped.

A job survives a restart of the web app: a queued job stays queued, and a
job the restart cut short runs again from the start (a snapshot is read
again from the start, so it reads the change current when it finishes; a
write cut short wrote nothing), at most RESTARTS times, then fails with
that reason.

A queue file of an older layout is brought up when it is opened
(connected). One whose jobs name a dataset has its jobs' dataset column
become key and each job gets its lane; every write that was queued or
running fails: one of an operation that no longer exists (GONE), and any
other (WRITES_BEFORE), since its selection was pickled with the libraries
it named, which a selection no longer holds, so run now it would reach
more texts than the person chose. A job of a kind the web app no longer
runs (KINDS_GONE: a vocabulary's training, a corpus's build) that was
queued or running fails (KIND_GONE), and the tables of what only those
jobs kept (the checkpoints of builds, when each corpus was last
downloaded) go.

A small write runs in the request itself (record) and is kept as a job that
has ended, so every write answers the same way.
"""

import importlib
import io
import json
import multiprocessing
import pickle
import secrets
import sqlite3
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from depot import Depot, errors
from depot.depot import location

RESTARTS = 3  # restarts of the web app a running job lives through before it fails
KEPT = 1000  # jobs that ended the queue keeps; the older ones go
POLL = 0.5  # seconds between two looks of the runner at the queue and at a running job
PROGRESS_EVERY = 1.0  # seconds between two writes of a job's progress
LANES = {"write": "write", "snapshot": "heavy", "sizes": "read"}  # by kind
KINDS = tuple(LANES)
KINDS_GONE = ("vocabulary", "corpus")  # the kinds of jobs the web app ran once and runs no more
# why a job of one of those kinds fails when it was queued or running
KIND_GONE = ("the web app no longer trains vocabularies nor builds corpora: the freight makes them (`wpt "
             "build-resources`); nothing was done")
OPEN = ("queued", "running")
ENDED = ("done", "refused", "failed", "cancelled")
# the operations of writes that no longer exist: a job of one queued or running when the queue file is brought up fails
GONE = (":fork_library", ":report_problem", ":put_texts", ":take_out_texts", ":lock_library")
# why any other write queued or running when the queue file is brought up fails
WRITES_BEFORE = ("this write was queued before the depot's upgrade, when a selection could name libraries, which no "
                 "longer exist: run now it could reach more texts than were chosen; nothing was written, choose the "
                 "texts again")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, number INTEGER NOT NULL, kind TEXT NOT NULL, lane TEXT NOT NULL, title TEXT NOT NULL,
    key TEXT, recipe TEXT, target TEXT, operation TEXT NOT NULL, arguments BLOB NOT NULL,
    state TEXT NOT NULL, stages TEXT NOT NULL, progress TEXT, answer TEXT, problem TEXT, status INTEGER,
    created REAL NOT NULL, started REAL, ended REAL, restarts INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS jobs_open ON jobs (state, number);
DROP TABLE IF EXISTS checkpoints;
CREATE TABLE IF NOT EXISTS erased_since (snapshot TEXT PRIMARY KEY, last_erasure INTEGER NOT NULL, found TEXT NOT NULL);
DROP TABLE IF EXISTS downloads;
CREATE TABLE IF NOT EXISTS streams (id TEXT PRIMARY KEY, corpus TEXT, visitor TEXT NOT NULL, kind TEXT NOT NULL,
    started REAL NOT NULL);
DROP TABLE IF EXISTS exports;
DROP TABLE IF EXISTS export_parts;
CREATE TABLE IF NOT EXISTS exported_parts (export TEXT NOT NULL, parts INTEGER NOT NULL, part INTEGER NOT NULL,
    format TEXT NOT NULL, maker TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, left_out TEXT NOT NULL,
    PRIMARY KEY (export, parts, part, format, maker));
CREATE TABLE IF NOT EXISTS export_texts (export TEXT PRIMARY KEY, texts INTEGER NOT NULL);
DROP TABLE IF EXISTS marking;
"""


class Busy(errors.DepotError):
    """A snapshot job is queued or running already, or the snapshot a sizes
    job: nothing was queued."""


def jobs_file(address: str) -> Path:
    """The queue's file of the depot at address: <depot file>.jobs beside it."""
    path = location(address)
    return path.with_name(path.name + ".jobs")


def work_folder(address: str) -> Path:
    """The folder the jobs of the depot at address work in: <depot file>.work beside it."""
    path = location(address)
    return path.with_name(path.name + ".work")


@contextmanager
def connected(path: Path, write: bool = False):
    """A connection to the queue's file, its tables made; with write, one
    transaction that holds the file's write lock from its start."""
    con = sqlite3.connect(path, timeout=60, isolation_level=None)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA busy_timeout = 60000")
        con.executescript(SCHEMA)
        if {"target", "key", "lane"} - _columns(con, "jobs") or con.execute(
                f"SELECT 1 FROM jobs WHERE kind IN ({', '.join('?' * len(KINDS_GONE))}) AND state IN "
                "('queued', 'running') LIMIT 1", KINDS_GONE).fetchone():
            _brought_up(con)
        if write:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")
        else:
            yield con
    finally:
        con.close()


def _columns(con, table: str) -> set[str]:
    return {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}


def _brought_up(con):
    """Bring a queue file made before up to SCHEMA, in one transaction that
    looks again once it holds the write lock (another connection may have
    done it meanwhile): a jobs table without target gets it; one with
    dataset has it renamed key, gets lane, and every write that was queued
    or running fails (GONE, WRITES_BEFORE); every job of KINDS_GONE queued
    or running fails (KIND_GONE)."""
    con.execute("BEGIN IMMEDIATE")
    try:
        columns = _columns(con, "jobs")
        if "target" not in columns:
            con.execute("ALTER TABLE jobs ADD COLUMN target TEXT")
        if "dataset" in columns:
            con.execute("ALTER TABLE jobs RENAME COLUMN dataset TO key")
            for gone in GONE:
                con.execute("UPDATE jobs SET state = 'failed', status = 422, ended = ?, problem = ? "
                            "WHERE kind = 'write' AND state IN ('queued', 'running') AND operation LIKE ?",
                            (time.time(), "this operation no longer exists: nothing was written", f"%{gone}"))
            con.execute("UPDATE jobs SET state = 'failed', status = 422, ended = ?, problem = ? "
                        "WHERE kind = 'write' AND state IN ('queued', 'running')", (time.time(), WRITES_BEFORE))
        if "lane" not in _columns(con, "jobs"):
            con.execute("ALTER TABLE jobs ADD COLUMN lane TEXT NOT NULL DEFAULT 'heavy'")
            con.execute("UPDATE jobs SET lane = 'write' WHERE kind = 'write'")
        con.execute(f"UPDATE jobs SET state = 'failed', status = 422, ended = ?, problem = ? WHERE kind IN "
                    f"({', '.join('?' * len(KINDS_GONE))}) AND state IN ('queued', 'running')",
                    (time.time(), KIND_GONE, *KINDS_GONE))
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")


def _named(operation) -> str:
    return f"{operation.__module__}:{operation.__qualname__}"


# names of modules a queue file may hold that the code has under another name: each read as the second
MOVED = (("wpt.depot", "depot_app"), ("wpt_depot", "depot"), ("wpt_store", "dock"))


def _moved(module: str) -> str:
    for old, new in MOVED:
        if module == old or module.startswith(old + "."):
            return new + module[len(old):]
    return module


class _Unpickler(pickle.Unpickler):
    """Reads arguments pickled with a module of MOVED under its own name."""

    def find_class(self, module, name):
        return super().find_class(_moved(module), name)


def _operation(name: str):
    module, _, function = name.partition(":")
    return getattr(importlib.import_module(_moved(module)), function)


def _shown(row) -> dict:
    """A job's row as the routes and the command line give it."""
    return {"id": row["id"], "kind": row["kind"], "lane": row["lane"], "title": row["title"], "key": row["key"],
            "recipe": row["recipe"], "target": json.loads(row["target"] or "null"), "state": row["state"],
            "stages": json.loads(row["stages"]), "progress": json.loads(row["progress"] or "null"),
            "answer": json.loads(row["answer"] or "null"), "problem": row["problem"], "status": row["status"],
            "created": row["created"], "started": row["started"], "ended": row["ended"],
            "restarts": row["restarts"]}


class Job:
    """What a running job's operation gets beside its arguments (as job=):
    the queue's file, the job's id, the folder it works in and its key."""

    def __init__(self, path: Path, job: str, work: Path, key: str | None):
        self.path, self.id, self.work, self.key = path, job, work, key


def _work(path: str, address: str, job: str, statuses: dict):
    """A job's process: the operation run over the depot at address, its
    progress, then its answer or its failure, written to the queue's file
    while the job is still running there (a cancelled job keeps its state)."""
    path = Path(path)
    with connected(path) as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
    args, kwargs = _Unpickler(io.BytesIO(row["arguments"])).load()
    stages, last = json.loads(row["stages"]), [0.0]

    def progress(value):
        """A stage's name (text), kept with its time, or the job's progress now (a dict), kept now and then."""
        now = time.time()
        if isinstance(value, str):
            stages.append({"stage": value, "at": now})
        elif now - last[0] < PROGRESS_EVERY:
            return
        last[0] = now
        with connected(path, write=True) as con:
            con.execute("UPDATE jobs SET stages = ?, progress = coalesce(?, progress) WHERE id = ? AND state = 'running'",
                        (json.dumps(stages), None if isinstance(value, str) else json.dumps(value), job))

    def ended(state: str, answer=None, problem=None, status=None):
        with connected(path, write=True) as con:
            con.execute("UPDATE jobs SET state = ?, answer = ?, problem = ?, status = ?, stages = ?, ended = ? "
                        "WHERE id = ? AND state = 'running'",
                        (state, json.dumps(answer), problem, status, json.dumps(stages), time.time(), job))

    try:
        operation = _operation(row["operation"])
        if row["kind"] != "write":
            work = work_folder(address)
            kwargs = {**kwargs, "job": Job(path, job, work, row["key"])}
        answer = operation(Depot.open(address), *args, progress=progress, **kwargs)
    except Exception as error:
        status = next((code for kind, code in statuses.items() if isinstance(error, kind)), None)
        if status is None:
            traceback.print_exc()
            ended("failed", problem=f"{type(error).__name__}: {error}", status=500)
        else:
            ended("refused", problem=f"{type(error).__name__}: {error}", status=status)
    else:
        ended("done", answer=answer)


class Jobs:
    """The queue of the depot at address. statuses: the HTTP status of
    each failure an operation may meet (depot_app.web.routes.STATUSES), by
    class; another failure fails the job."""

    def __init__(self, address: str, statuses: dict | None = None):
        self.address = address
        self.path = jobs_file(address)
        self.statuses = statuses or {errors.DepotError: 422}
        self._context = multiprocessing.get_context("spawn")
        self._runners = []
        self._stop = threading.Event()
        # set by add() and stop(), so a lane's runner takes a job queued in this process at once instead of at its
        # next look at the queue (POLL), which a job queued by another process still waits for
        self._wake = {lane: threading.Event() for lane in dict.fromkeys(LANES.values())}
        with connected(self.path):
            pass

    # Adding and reading

    def add(self, kind: str, title: str, operation, *args, key: str | None = None,
            target: dict | None = None, **kwargs) -> dict:
        """Queue a job of that kind (KINDS), in its lane, that does what
        title says by running operation(depot, *args, progress=...,
        **kwargs), a function the job's process imports by name (and job=,
        a Job, for a kind that is not a write), for the key (a sizes job's
        snapshot id), target saying what it makes (JSON): its state. Busy when a snapshot job is open, for a snapshot job, or
        a sizes job of the key, for a sizes job."""
        if kind not in KINDS:
            raise ValueError(f"no kind of job {kind!r}")
        job = secrets.token_hex(8)
        with connected(self.path, write=True) as con:
            if kind == "snapshot" and con.execute(
                    "SELECT 1 FROM jobs WHERE kind = 'snapshot' AND state IN ('queued', 'running')").fetchone():
                raise Busy("a snapshot is being made already: see the jobs page")
            if kind == "sizes" and con.execute(
                    "SELECT 1 FROM jobs WHERE kind = ? AND key = ? AND state IN ('queued', 'running')",
                    (kind, key)).fetchone():
                raise Busy(f"the snapshot {key} is being ordered by length already")
            number = con.execute("SELECT coalesce(max(number), 0) + 1 FROM jobs").fetchone()[0]
            con.execute("INSERT INTO jobs (id, number, kind, lane, title, key, recipe, target, operation, arguments, "
                        "state, stages, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', '[]', ?)",
                        (job, number, kind, LANES[kind], title, key, None,
                         None if target is None else json.dumps(target), _named(operation),
                         pickle.dumps((args, kwargs)), time.time()))
        self._wake[LANES[kind]].set()
        return self.state(job)

    def open_job(self, kind: str, key: str | None = None) -> dict | None:
        """The job of that kind queued or running for the key (a sizes job's
        snapshot id; any key for a snapshot job), None without one."""
        with connected(self.path) as con:
            row = con.execute("SELECT * FROM jobs WHERE kind = ? AND (? OR key = ?) AND state IN ('queued', "
                              "'running') ORDER BY number LIMIT 1", (kind, kind == "snapshot", key)).fetchone()
        return None if row is None else _shown(row)

    def last_ended(self, kind: str, key: str) -> dict | None:
        """The latest job of that kind for the key (a sizes job's snapshot
        id) that ended, whatever its state, None without one."""
        with connected(self.path) as con:
            row = con.execute("SELECT * FROM jobs WHERE kind = ? AND key = ? AND state NOT IN ('queued', 'running') "
                              "ORDER BY number DESC LIMIT 1", (kind, key)).fetchone()
        return None if row is None else _shown(row)

    def record(self, kind: str, title: str, depot: Depot, operation, *args) -> dict:
        """Run operation(depot, *args, progress=...) in this process, kept
        as a job of that kind that ends before this returns: its state. A
        failure that statuses does not name is kept as failed and raised."""
        job = secrets.token_hex(8)
        stages = []
        started = time.time()

        def keep(state, answer=None, problem=None, status=None):
            with connected(self.path, write=True) as con:
                number = con.execute("SELECT coalesce(max(number), 0) + 1 FROM jobs").fetchone()[0]
                con.execute("INSERT INTO jobs (id, number, kind, lane, title, operation, arguments, state, stages, "
                            "answer, problem, status, created, started, ended) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (job, number, kind, LANES[kind], title, _named(operation), pickle.dumps(((), {})), state,
                             json.dumps(stages), json.dumps(answer), problem, status, started, started, time.time()))
        try:
            answer = operation(depot, *args, progress=lambda stage: stages.append({"stage": stage, "at": time.time()}))
        except Exception as error:
            status = next((code for one, code in self.statuses.items() if isinstance(error, one)), None)
            if status is None:
                keep("failed", problem="the write failed: nothing was written", status=500)
                raise
            keep("refused", problem=f"{type(error).__name__}: {error}", status=status)
        else:
            keep("done", answer=answer)
        self._forget()
        return self.state(job)

    def state(self, job: str) -> dict:
        """The state of the job: {id, kind, lane, title, key, recipe, target,
        state, stages: [{stage, at}], progress, answer, problem, status,
        created, started, ended, restarts}, times in seconds since the
        epoch. NotFound for a job the queue does not hold."""
        with connected(self.path) as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
        if row is None:
            raise errors.NotFound(f"no job {job}")
        return _shown(row)

    def listed(self, limit: int = 100) -> list[dict]:
        """Every job queued or running, in the order they run, then the
        limit latest that ended, the latest first."""
        with connected(self.path) as con:
            open_rows = con.execute("SELECT * FROM jobs WHERE state IN ('queued', 'running') ORDER BY "
                                    "state = 'queued', number").fetchall()
            ended_rows = con.execute("SELECT * FROM jobs WHERE state NOT IN ('queued', 'running') ORDER BY number DESC "
                                     "LIMIT ?", (limit,)).fetchall()
        return [_shown(row) for row in [*open_rows, *ended_rows]]

    def writing(self) -> bool:
        """Whether a job of the write lane is queued or running: the jobs of
        the heavy lane hold the depot's write lock for no more than a short
        transaction, so a write run in the request waits for none of them."""
        with connected(self.path) as con:
            return con.execute("SELECT 1 FROM jobs WHERE lane = 'write' AND state IN ('queued', 'running')"
                               ).fetchone() is not None

    def erased_since(self, snapshot: str, last_erasure: int) -> dict | None:
        """What erased_kept kept for the snapshot (its id) while the depot's
        erasures stood at last_erasure (depot.snapshots.last_erasure);
        None when nothing was kept then."""
        with connected(self.path) as con:
            found = con.execute("SELECT found FROM erased_since WHERE snapshot = ? AND last_erasure = ?",
                                (snapshot, last_erasure)).fetchone()
        return None if found is None else json.loads(found["found"])

    def erased_kept(self, snapshot: str, last_erasure: int, found: dict):
        """Keep what erasures did to the snapshot (its id) while the depot's erasures stood at last_erasure."""
        with connected(self.path, write=True) as con:
            con.execute("INSERT OR REPLACE INTO erased_since (snapshot, last_erasure, found) VALUES (?, ?, ?)",
                        (snapshot, last_erasure, json.dumps(found)))

    def wait(self, job: str, seconds: float) -> dict:
        """The state of the job once it ended, or after seconds."""
        deadline = time.monotonic() + seconds
        while True:
            found = self.state(job)
            if found["state"] in ENDED or time.monotonic() >= deadline:
                return found
            time.sleep(0.05)

    def cancel(self, job: str) -> dict:
        """Cancel a queued or running job: its state as it was. NotFound
        for a job the queue does not hold; Refused for one that ended."""
        with connected(self.path, write=True) as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
            if row is None:
                raise errors.NotFound(f"no job {job}")
            if row["state"] not in OPEN:
                raise errors.Refused(f"the job {job} ended already ({row['state']})")
            con.execute("UPDATE jobs SET state = 'cancelled', problem = 'cancelled from the command line', "
                        "ended = ? WHERE id = ?", (time.time(), job))
        return _shown(row)

    def _forget(self):
        with connected(self.path, write=True) as con:
            con.execute("DELETE FROM jobs WHERE state NOT IN ('queued', 'running') AND number <= "
                        "(SELECT number FROM jobs WHERE state NOT IN ('queued', 'running') ORDER BY number DESC "
                        "LIMIT 1 OFFSET ?)", (KEPT,))

    # The runner

    def start(self):
        """Take up the jobs a stop cut short (again, or failed past
        RESTARTS restarts), forget the streams the stop cut, and start a
        runner thread for each lane, which runs the lane's queued jobs one at
        a time until stop()."""
        with connected(self.path, write=True) as con:
            con.execute("DELETE FROM streams")
            for row in con.execute("SELECT id, restarts FROM jobs WHERE state = 'running'").fetchall():
                if row["restarts"] >= RESTARTS:
                    con.execute("UPDATE jobs SET state = 'failed', status = 500, ended = ?, problem = ? WHERE id = ?",
                                (time.time(), f"the web app stopped {row['restarts'] + 1} times while it ran", row["id"]))
                else:
                    con.execute("UPDATE jobs SET state = 'queued', restarts = restarts + 1 WHERE id = ?", (row["id"],))
        for lane in dict.fromkeys(LANES.values()):
            runner = threading.Thread(target=self._run, args=(lane,), name=f"jobs-{lane}", daemon=True)
            runner.start()
            self._runners.append(runner)

    def stop(self):
        """Stop the runners, the running jobs' processes with them; the jobs stay running, so the next start takes
        them up."""
        self._stop.set()
        for wake in self._wake.values():
            wake.set()
        for runner in self._runners:
            runner.join()

    def _next(self, lane: str) -> str | None:
        """Mark the oldest queued job of the lane running: its id; None when none is queued."""
        with connected(self.path, write=True) as con:
            row = con.execute("SELECT id FROM jobs WHERE state = 'queued' AND lane = ? ORDER BY number LIMIT 1",
                              (lane,)).fetchone()
            if row is None:
                return None
            con.execute("UPDATE jobs SET state = 'running', started = ? WHERE id = ?", (time.time(), row["id"]))
            return row["id"]

    def _run(self, lane: str):
        while not self._stop.is_set():
            try:
                job = self._next(lane)
            except sqlite3.Error as error:
                print(f"depot web: the jobs file cannot be read: {error}", file=sys.stderr)
                job = None
            if job is None:
                self._wake[lane].wait(POLL)
                self._wake[lane].clear()
                continue
            # not a daemon: stop() ends the process instead
            process = self._context.Process(target=_work, args=(str(self.path), self.address, job, self.statuses))
            process.start()
            while process.is_alive():
                process.join(POLL)
                if self._stop.is_set() or self.state(job)["state"] == "cancelled":
                    process.terminate()
                    process.join()
            if self._stop.is_set():
                return
            with connected(self.path, write=True) as con:
                con.execute("UPDATE jobs SET state = 'failed', status = 500, ended = ?, problem = ? "
                            "WHERE id = ? AND state = 'running'",
                            (time.time(), f"the job's process ended before it answered (exit code {process.exitcode})",
                             job))
            self._forget()
