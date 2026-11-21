"""The queue of a depot's jobs: the long work the web app runs for anyone
(training a tokenizer recipe's vocabulary, building a dataset's corpus, a
write over a selection of texts), one job at a time, in a process of its
own, so the web app keeps the other cores and answers while a job runs.

The queue is a small SQLite file beside the depot, <depot file>.jobs,
which nothing backs up: it holds each job's kind, what it does in words,
what it is for (its dataset or recipe, and its target: what a corpus job
builds, its vocabulary and size), the operation it runs and its arguments,
its state, its progress and its answer, and the jobs' other state
(wpt.depot.downloads). The web app's runner takes the oldest queued job,
marks it running and runs it in a spawned process. A job's state is queued, running,
done (its answer), refused (the failure that stopped it, with its HTTP
status; what it is refused writes nothing), failed (the process ended
without an answer, or with a failure the routes do not name) or cancelled.
Each dataset has at most one corpus job queued or running, and each
tokenizer recipe at most one vocabulary job (Busy otherwise). Only the
command line cancels a job (wpt depot jobs cancel): a queued one never
runs, and a running one's process is stopped.

A job survives a restart of the web app: a queued job stays queued, and a
job the restart cut short runs again from the start (a corpus build goes
on from its last token file uploaded, wpt.depot.shared.build_corpus; a
write cut short wrote nothing), at most RESTARTS times, then fails with
that reason.

A small write runs in the request itself (record) and is kept as a job that
has ended, so every write answers the same way.
"""

import importlib
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

from wpt_depot import Depot, errors
from wpt_depot.depot import location

RESTARTS = 3  # restarts of the web app a running job lives through before it fails
KEPT = 1000  # jobs that ended the queue keeps; the older ones go
POLL = 0.5  # seconds between two looks of the runner at the queue and at a running job
PROGRESS_EVERY = 1.0  # seconds between two writes of a job's progress
KINDS = ("vocabulary", "corpus", "write")
OPEN = ("queued", "running")
ENDED = ("done", "refused", "failed", "cancelled")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, number INTEGER NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
    dataset TEXT, recipe TEXT, target TEXT, operation TEXT NOT NULL, arguments BLOB NOT NULL,
    state TEXT NOT NULL, stages TEXT NOT NULL, progress TEXT, answer TEXT, problem TEXT, status INTEGER,
    created REAL NOT NULL, started REAL, ended REAL, restarts INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS jobs_open ON jobs (state, number);
CREATE TABLE IF NOT EXISTS checkpoints (dataset TEXT PRIMARY KEY, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS downloads (corpus TEXT PRIMARY KEY, last REAL NOT NULL);
CREATE TABLE IF NOT EXISTS streams (id TEXT PRIMARY KEY, corpus TEXT, visitor TEXT NOT NULL, kind TEXT NOT NULL,
    started REAL NOT NULL);
DROP TABLE IF EXISTS exports;
DROP TABLE IF EXISTS export_parts;
CREATE TABLE IF NOT EXISTS exported_parts (export TEXT NOT NULL, parts INTEGER NOT NULL, part INTEGER NOT NULL,
    format TEXT NOT NULL, maker TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, left_out TEXT NOT NULL,
    PRIMARY KEY (export, parts, part, format, maker));
CREATE TABLE IF NOT EXISTS export_texts (export TEXT PRIMARY KEY, texts INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS marking (erasure TEXT PRIMARY KEY, since REAL NOT NULL);
"""


class Busy(errors.DepotError):
    """The dataset has a corpus job queued or running already, or the
    recipe a vocabulary job: nothing was queued."""


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
        if "target" not in {row["name"] for row in con.execute("PRAGMA table_info(jobs)")}:  # a file made before it
            try:
                con.execute("ALTER TABLE jobs ADD COLUMN target TEXT")
            except sqlite3.OperationalError as error:  # another connection added it meanwhile
                if "duplicate column" not in str(error):
                    raise
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


def _named(operation) -> str:
    return f"{operation.__module__}:{operation.__qualname__}"


def _operation(name: str):
    module, _, function = name.partition(":")
    return getattr(importlib.import_module(module), function)


def _shown(row) -> dict:
    """A job's row as the routes and the command line give it."""
    return {"id": row["id"], "kind": row["kind"], "title": row["title"], "dataset": row["dataset"],
            "recipe": row["recipe"], "target": json.loads(row["target"] or "null"), "state": row["state"],
            "stages": json.loads(row["stages"]), "progress": json.loads(row["progress"] or "null"),
            "answer": json.loads(row["answer"] or "null"), "problem": row["problem"], "status": row["status"],
            "created": row["created"], "started": row["started"], "ended": row["ended"],
            "restarts": row["restarts"]}


class Job:
    """What a running job's operation gets beside its arguments (as job=):
    the queue's file, the job's id and the folder it works in, and its
    dataset's checkpoint."""

    def __init__(self, path: Path, job: str, work: Path, dataset: str | None):
        self.path, self.id, self.work, self.dataset = path, job, work, dataset

    def checkpoint(self) -> dict | None:
        """The state the last build of the job's dataset saved, None without one."""
        with connected(self.path) as con:
            found = con.execute("SELECT state FROM checkpoints WHERE dataset = ?", (self.dataset,)).fetchone()
        return None if found is None else json.loads(found["state"])

    def save(self, state: dict | None):
        """Keep the state that resumes the job's dataset's build; None forgets it."""
        with connected(self.path, write=True) as con:
            if state is None:
                con.execute("DELETE FROM checkpoints WHERE dataset = ?", (self.dataset,))
            else:
                con.execute("INSERT OR REPLACE INTO checkpoints (dataset, state) VALUES (?, ?)",
                            (self.dataset, json.dumps(state)))


def _work(path: str, address: str, job: str, statuses: dict):
    """A job's process: the operation run over the depot at address, its
    progress, then its answer or its failure, written to the queue's file
    while the job is still running there (a cancelled job keeps its state)."""
    path = Path(path)
    with connected(path) as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
    args, kwargs = pickle.loads(row["arguments"])
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
            kwargs = {**kwargs, "job": Job(path, job, work, row["dataset"])}
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
    each failure an operation may meet (wpt.depot.web.routes.STATUSES), by
    class; another failure fails the job."""

    def __init__(self, address: str, statuses: dict | None = None):
        self.address = address
        self.path = jobs_file(address)
        self.statuses = statuses or {errors.DepotError: 422}
        self._context = multiprocessing.get_context("spawn")
        self._runner = None
        self._stop = threading.Event()
        with connected(self.path):
            pass

    # Adding and reading

    def add(self, kind: str, title: str, operation, *args, dataset: str | None = None, recipe: str | None = None,
            target: dict | None = None, **kwargs) -> dict:
        """Queue a job of that kind (KINDS) that does what title says by
        running operation(depot, *args, progress=..., **kwargs), a function
        the job's process imports by name (and job=, a Job, for a kind that
        is not a write), for the dataset or the recipe, target saying what
        it makes (JSON): its state. Busy when the dataset has a corpus job
        open, or the recipe a vocabulary job."""
        if kind not in KINDS:
            raise ValueError(f"no kind of job {kind!r}")
        job = secrets.token_hex(8)
        with connected(self.path, write=True) as con:
            if kind == "corpus" and con.execute("SELECT 1 FROM jobs WHERE kind = 'corpus' AND dataset = ? AND state IN "
                                                "('queued', 'running')", (dataset,)).fetchone():
                raise Busy(f"the dataset {dataset} has a corpus job queued or running already: see the jobs page")
            if kind == "vocabulary" and con.execute("SELECT 1 FROM jobs WHERE kind = 'vocabulary' AND recipe = ? AND "
                                                    "state IN ('queued', 'running')", (recipe,)).fetchone():
                raise Busy(f"the tokenizer recipe {recipe} has a vocabulary job queued or running already")
            number = con.execute("SELECT coalesce(max(number), 0) + 1 FROM jobs").fetchone()[0]
            con.execute("INSERT INTO jobs (id, number, kind, title, dataset, recipe, target, operation, arguments, "
                        "state, stages, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', '[]', ?)",
                        (job, number, kind, title, dataset, recipe, None if target is None else json.dumps(target),
                         _named(operation), pickle.dumps((args, kwargs)), time.time()))
        return self.state(job)

    def open_job(self, kind: str, dataset: str | None = None, recipe: str | None = None) -> dict | None:
        """The job of that kind queued or running for the dataset or the recipe, None without one."""
        column, value = ("dataset", dataset) if kind == "corpus" else ("recipe", recipe)
        with connected(self.path) as con:
            row = con.execute(f"SELECT * FROM jobs WHERE kind = ? AND {column} = ? AND state IN ('queued', 'running') "
                              "ORDER BY number LIMIT 1", (kind, value)).fetchone()
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
                con.execute("INSERT INTO jobs (id, number, kind, title, operation, arguments, state, stages, answer, "
                            "problem, status, created, started, ended) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (job, number, kind, title, _named(operation), pickle.dumps(((), {})), state,
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
        """The state of the job: {id, kind, title, dataset, recipe, target,
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
        """Whether a write job is queued or running: the jobs of the other
        kinds hold no write lock of the depot while they run."""
        with connected(self.path) as con:
            return con.execute("SELECT 1 FROM jobs WHERE kind = 'write' AND state IN ('queued', 'running')"
                               ).fetchone() is not None

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
        RESTARTS restarts), forget the streams the stop cut, and start the
        runner thread, which runs the queued jobs one at a time until stop()."""
        with connected(self.path, write=True) as con:
            con.execute("DELETE FROM streams")
            for row in con.execute("SELECT id, restarts FROM jobs WHERE state = 'running'").fetchall():
                if row["restarts"] >= RESTARTS:
                    con.execute("UPDATE jobs SET state = 'failed', status = 500, ended = ?, problem = ? WHERE id = ?",
                                (time.time(), f"the web app stopped {row['restarts'] + 1} times while it ran", row["id"]))
                else:
                    con.execute("UPDATE jobs SET state = 'queued', restarts = restarts + 1 WHERE id = ?", (row["id"],))
        self._runner = threading.Thread(target=self._run, name="jobs", daemon=True)
        self._runner.start()

    def stop(self):
        """Stop the runner, a running job's process with it; the job stays running, so the next start takes it up."""
        self._stop.set()
        if self._runner is not None:
            self._runner.join()

    def _next(self) -> str | None:
        """Mark the oldest queued job running: its id; None when none is queued."""
        with connected(self.path, write=True) as con:
            row = con.execute("SELECT id FROM jobs WHERE state = 'queued' ORDER BY number LIMIT 1").fetchone()
            if row is None:
                return None
            con.execute("UPDATE jobs SET state = 'running', started = ? WHERE id = ?", (time.time(), row["id"]))
            return row["id"]

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._next()
            except sqlite3.Error as error:
                print(f"wpt depot web: the jobs file cannot be read: {error}", file=sys.stderr)
                job = None
            if job is None:
                self._stop.wait(POLL)
                continue
            # not a daemon: a corpus build starts worker processes of its own, which a daemon may not have;
            # stop() ends the process instead
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
