"""The freight's queue: the requests it was given, run one at a time in
the order they came, each in a process of its own that has the whole
machine (docs/api/freight.md, "The queue").

The queue is a small SQLite file in the freight's own folder, its home
(FREIGHT_HOME): queue.sqlite, which nothing backs up. It holds each job's
request, the delivery key it was given when queued, the depot it reads and
the dock it delivers on, its state, its progress and its answer (the
delivery note), the checkpoints of the jobs by delivery key, and the mark
of the erasures the freight saw last in each depot (freight.api.check).
The jobs work in home/work/<delivery key>/.

A job's state is queued, running, done (its answer), refused (the request
cannot be met as it is; nothing was delivered), failed (the job's process
ended without an answer, or with a failure no refusal names) or
cancelled. Each delivery key has at most one job queued or running
(Busy otherwise).

One process at a time runs the queue (the lock home/worker.lock): it takes
the oldest queued job, marks it running and runs it in a spawned process,
looking at the queue every POLL seconds. A job a stop cut short (its runner
gone) is queued again when a runner starts, and goes on from its
checkpoint, at most RESTARTS times; then it fails with that reason. A
cancelled job never runs, and a running one's process is stopped.
"""

import json
import multiprocessing
import secrets
import sqlite3
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from wpt_utils.files import try_lock_file, unlock_file
from wpt_utils.lines import say

RESTARTS = 3  # stops of the runner a running job lives through before it fails
KEPT = 1000  # jobs that ended the queue keeps; the older ones go
POLL = 0.2  # seconds between two looks of the runner at the queue and at a running job
PROGRESS_EVERY = 1.0  # seconds between two writes of a job's progress
OPEN = ("queued", "running")
ENDED = ("done", "refused", "failed", "cancelled")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, number INTEGER NOT NULL, key TEXT NOT NULL, request TEXT NOT NULL, depot TEXT NOT NULL,
    dock TEXT NOT NULL, cap INTEGER, state TEXT NOT NULL, stages TEXT NOT NULL, progress TEXT, answer TEXT,
    problem TEXT, created REAL NOT NULL, started REAL, ended REAL, restarts INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS jobs_open ON jobs (state, number);
CREATE TABLE IF NOT EXISTS checkpoints (key TEXT PRIMARY KEY, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS marks (depot TEXT PRIMARY KEY, mark INTEGER NOT NULL);
"""


class QueueError(Exception):
    """What the queue refuses."""


class Busy(QueueError):
    """The delivery key has a job queued or running already: nothing was queued."""


class NoJob(QueueError):
    """The queue holds no such job."""


@contextmanager
def connected(home: Path, write: bool = False):
    """A connection to the queue's file in home, its tables made; with
    write, one transaction that holds the file's write lock from its start."""
    home.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(home / "queue.sqlite", timeout=60, isolation_level=None)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA busy_timeout = 60000")
        con.executescript(SCHEMA)
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


def _shown(row) -> dict:
    """A job's row as the api and the command line give it."""
    return {"id": row["id"], "key": row["key"], "request": json.loads(row["request"]), "state": row["state"],
            "stages": json.loads(row["stages"]), "progress": json.loads(row["progress"] or "null"),
            "answer": json.loads(row["answer"] or "null"), "problem": row["problem"], "created": row["created"],
            "started": row["started"], "ended": row["ended"], "restarts": row["restarts"]}


class Queue:
    """The queue of the freight whose home is home."""

    def __init__(self, home: str | Path):
        self.home = Path(home)
        with connected(self.home):
            pass

    def work_folder(self, key: str) -> Path:
        return self.home / "work" / key

    # Adding and reading

    def add(self, request: dict, key: str, depot: str, dock: str, cap: int | None) -> dict:
        """Queue the request, whose delivery key is key, to read the depot
        at that address and deliver on the dock of that URL: its state.
        Busy when the key has a job queued or running."""
        job = secrets.token_hex(8)
        with connected(self.home, write=True) as con:
            found = con.execute("SELECT id FROM jobs WHERE key = ? AND state IN ('queued', 'running')",
                                (key,)).fetchone()
            if found is not None:
                raise Busy(f"the delivery {key} has a job queued or running already: {found['id']}")
            con.execute("INSERT INTO jobs (id, number, key, request, depot, dock, cap, state, stages, created) VALUES "
                        "(?, (SELECT coalesce(max(number), 0) + 1 FROM jobs), ?, ?, ?, ?, ?, 'queued', '[]', ?)",
                        (job, key, json.dumps(request), depot, dock, cap, time.time()))
        return self.state(job)

    def answered(self, request: dict, key: str, depot: str, dock: str, note: dict) -> dict:
        """Keep a request whose delivery the dock holds fresh as a job done at once, its answer that note."""
        job = secrets.token_hex(8)
        now = time.time()
        with connected(self.home, write=True) as con:
            con.execute("INSERT INTO jobs (id, number, key, request, depot, dock, state, stages, answer, created, "
                        "started, ended) VALUES (?, (SELECT coalesce(max(number), 0) + 1 FROM jobs), ?, ?, ?, ?, "
                        "'done', '[]', ?, ?, ?, ?)", (job, key, json.dumps(request), depot, dock, json.dumps(note),
                                                      now, now, now))
        self._forget()
        return self.state(job)

    def state(self, job: str) -> dict:
        """The state of the job: {id, key, request, state, stages: [{stage, at}], progress, answer, problem,
        created, started, ended, restarts}, times in seconds since the epoch. NoJob without one."""
        with connected(self.home) as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
        if row is None:
            raise NoJob(f"no job {job}")
        return _shown(row)

    def listed(self, limit: int = 100) -> list[dict]:
        """Every job queued or running, in the order they run, then the limit latest that ended, the latest first."""
        with connected(self.home) as con:
            open_rows = con.execute("SELECT * FROM jobs WHERE state IN ('queued', 'running') ORDER BY "
                                    "state = 'queued', number").fetchall()
            ended_rows = con.execute("SELECT * FROM jobs WHERE state NOT IN ('queued', 'running') ORDER BY number DESC "
                                     "LIMIT ?", (limit,)).fetchall()
        return [_shown(row) for row in [*open_rows, *ended_rows]]

    def cancel(self, job: str) -> dict:
        """Cancel a queued or running job: its state as it was. NoJob for a
        job the queue does not hold; QueueError for one that ended."""
        with connected(self.home, write=True) as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
            if row is None:
                raise NoJob(f"no job {job}")
            if row["state"] not in OPEN:
                raise QueueError(f"the job {job} ended already ({row['state']})")
            con.execute("UPDATE jobs SET state = 'cancelled', problem = 'cancelled', ended = ? WHERE id = ?",
                        (time.time(), job))
        return _shown(row)

    def checkpoint(self, key: str) -> dict | None:
        """The state the last job of that delivery key saved, None without one."""
        with connected(self.home) as con:
            found = con.execute("SELECT state FROM checkpoints WHERE key = ?", (key,)).fetchone()
        return None if found is None else json.loads(found["state"])

    def save(self, key: str, state: dict | None):
        """Keep the state that resumes the job of that delivery key; None forgets it."""
        with connected(self.home, write=True) as con:
            if state is None:
                con.execute("DELETE FROM checkpoints WHERE key = ?", (key,))
            else:
                con.execute("INSERT OR REPLACE INTO checkpoints (key, state) VALUES (?, ?)", (key, json.dumps(state)))

    def mark(self, depot: str) -> int:
        """The mark of the erasures the freight saw last in the depot at that address, 0 for none."""
        with connected(self.home) as con:
            found = con.execute("SELECT mark FROM marks WHERE depot = ?", (depot,)).fetchone()
        return 0 if found is None else found["mark"]

    def keep_mark(self, depot: str, mark: int):
        with connected(self.home, write=True) as con:
            con.execute("INSERT OR REPLACE INTO marks (depot, mark) VALUES (?, ?)", (depot, mark))

    def _forget(self):
        with connected(self.home, write=True) as con:
            con.execute("DELETE FROM jobs WHERE state NOT IN ('queued', 'running') AND number <= "
                        "(SELECT number FROM jobs WHERE state NOT IN ('queued', 'running') ORDER BY number DESC "
                        "LIMIT 1 OFFSET ?)", (KEPT,))

    # The runner

    def run(self, until=lambda: False, stop=lambda: False) -> bool:
        """Run the queued jobs, one at a time, while this process holds the
        runner's lock, until until() says so once no job runs, or stop() at
        any time (the running job's process stopped, the job left running so
        the next runner takes it up): False when another process runs the
        queue."""
        lock = try_lock_file(self.home / "worker.lock")
        if lock is None:
            return False
        try:
            self._taken_up()
            context = multiprocessing.get_context("spawn")
            while not stop():
                job = self._next()
                if job is None:
                    if until():
                        return True
                    time.sleep(POLL)
                    continue
                process = context.Process(target=_work, args=(str(self.home), job))
                process.start()
                while process.is_alive():
                    process.join(POLL)
                    if stop() or self.state(job)["state"] == "cancelled":
                        process.terminate()
                        process.join()
                if stop():
                    return True
                with connected(self.home, write=True) as con:
                    con.execute("UPDATE jobs SET state = 'failed', ended = ?, problem = ? WHERE id = ? AND "
                                "state = 'running'", (time.time(), "the job's process ended before it answered "
                                                                   f"(exit code {process.exitcode})", job))
                self._forget()
            return True
        finally:
            unlock_file(lock)

    def _taken_up(self):
        """Queue again the jobs a stop of the runner cut short, or fail those past RESTARTS restarts."""
        with connected(self.home, write=True) as con:
            for row in con.execute("SELECT id, restarts FROM jobs WHERE state = 'running'").fetchall():
                if row["restarts"] >= RESTARTS:
                    con.execute("UPDATE jobs SET state = 'failed', ended = ?, problem = ? WHERE id = ?",
                                (time.time(), f"the runner stopped {row['restarts'] + 1} times while it ran",
                                 row["id"]))
                else:
                    con.execute("UPDATE jobs SET state = 'queued', restarts = restarts + 1 WHERE id = ?", (row["id"],))

    def _next(self) -> str | None:
        """Mark the oldest queued job running: its id; None when none is queued."""
        with connected(self.home, write=True) as con:
            row = con.execute("SELECT id FROM jobs WHERE state = 'queued' ORDER BY number LIMIT 1").fetchone()
            if row is None:
                return None
            con.execute("UPDATE jobs SET state = 'running', started = ? WHERE id = ?", (time.time(), row["id"]))
            return row["id"]


def _work(home: str, job: str):
    """A job's process: the request run (freight.job.Job), its progress,
    then its answer or its failure, written to the queue while the job is
    still running there (a cancelled job keeps its state)."""
    from wpt_store import open_store
    from wpt_utils.programs import code

    from .deliveries import DeliveryError
    from .exchange import PreparationFailed
    from .job import Job
    from .selection import Refused

    queue = Queue(home)
    with connected(queue.home) as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
    stages, last = json.loads(row["stages"]), [0.0]

    def progress(value: dict):
        now = time.time()
        if value.get("stage") != (stages[-1]["stage"] if stages else None):
            stages.append({"stage": value.get("stage"), "at": now})
        elif now - last[0] < PROGRESS_EVERY:
            return
        last[0] = now
        with connected(queue.home, write=True) as con:
            con.execute("UPDATE jobs SET stages = ?, progress = ? WHERE id = ? AND state = 'running'",
                        (json.dumps(stages), json.dumps(value), job))

    def ended(state: str, answer=None, problem=None):
        with connected(queue.home, write=True) as con:
            con.execute("UPDATE jobs SET state = ?, answer = ?, problem = ?, stages = ?, ended = ? "
                        "WHERE id = ? AND state = 'running'",
                        (state, json.dumps(answer), problem, json.dumps(stages), time.time(), job))

    key = row["key"]
    try:
        dock = open_store(row["dock"], env_file=".env")
        found = Job(json.loads(row["request"]), row["depot"], dock, queue.work_folder(key),
                    checkpoint=queue.checkpoint(key), save=lambda state: queue.save(key, state), progress=progress,
                    cap=row["cap"], code_commit=code()["commit"]).run()
    except (Refused, DeliveryError, PreparationFailed) as error:
        ended("refused" if isinstance(error, Refused) else "failed", problem=f"{type(error).__name__}: {error}")
    except Exception as error:  # noqa: BLE001 every other failure fails the job, with its trace in the log
        say(traceback.format_exc(), "ERROR")
        ended("failed", problem=f"{type(error).__name__}: {error}")
    else:
        if found["key"] != key:
            ended("failed", problem=f"the request was queued for the delivery {key} and made {found['key']}: its "
                                    "program changed its version meanwhile")
            return
        queue.save(key, None)
        ended("done", answer=found)
