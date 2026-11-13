"""A person's SQL, run by `wpt depot web` in processes of their own: the
query of POST /api/sql and the condition of a filter (search, passages).

A reader process opens the depot's file read-only (Depot.view): every
statement of a person runs on a connection that opens the file with
mode=ro and whose authorizer lets it only read (wpt_depot.database), so a
statement that writes, attaches another file, runs a pragma or loads an
extension is refused. Each job sets the connection's deadline SECONDS from
when the reader takes it, and SQLite stops the statement there. The
process's memory is capped at MEMORY bytes more than it holds once started
(on Linux, through the limit the system puts on a process's data), and the
server waits for its answer at most SECONDS from the request's start,
waiting for a free reader included: then it kills the process and answers
TimeLimitReached. A reader that dies (past its memory) answers ReaderLost.
Either way the server's own process goes on, and the next request gets a
new reader. At most READERS run at a time, and at most KEPT of those that
answered wait for the next request. A query's answer holds at most
ANSWER_ROWS rows and ANSWER_BYTES bytes of JSON, counted as its rows are
read.
"""

import json
import multiprocessing
import sys
import threading
import time
import weakref

from wpt_depot import Depot, errors
from wpt_depot.errors import TimeLimitReached

SECONDS = 60  # a request's whole time, under the 100 seconds a proxy in front waits
MEMORY = 2 << 30  # bytes of data a reader process may hold more than it holds once started
READERS = 2  # reader processes running at a time
KEPT = 1  # idle reader processes kept for the next requests
ANSWER_ROWS = 100_000  # rows a query's answer holds at most
ANSWER_BYTES = 16 << 20  # bytes of JSON a query's answer holds at most


class ReaderLost(errors.DepotError):
    """The process that ran the reads ended before it answered: it went past MEMORY."""


class AnswerTooLarge(errors.DepotError):
    """A query's answer holds more than ANSWER_ROWS rows or ANSWER_BYTES bytes of JSON."""


def _sql(depot: Depot, query: str, most_rows: int, most_bytes: int) -> dict:
    """{columns, rows} of the query; AnswerTooLarge past most_rows rows or
    most_bytes bytes of JSON, counted as the rows are read."""
    columns, found = depot.answer(query)
    rows, size = [], 0
    for one in found:
        if len(rows) == most_rows:
            raise AnswerTooLarge(f"the query's answer holds more than {most_rows} rows: narrow it, or add LIMIT and "
                                 "OFFSET")
        row = dict(zip(columns, one))
        size += len(json.dumps(row, default=_plain))
        if size > most_bytes:
            raise AnswerTooLarge(f"the query's answer holds more than {most_bytes} bytes of JSON: narrow it, or "
                                 "add LIMIT and OFFSET")
        rows.append(row)
    return {"columns": columns, "rows": rows}


def _plain(value):
    return value.hex() if isinstance(value, bytes) else str(value)


def _held() -> int:
    """The bytes of data this process holds now, as the system's limit counts them (Linux)."""
    with open("/proc/self/status", encoding="utf-8") as status:
        found = next(line for line in status if line.startswith("VmData:"))
    return int(found.split()[1]) * 1024


def _serve(address: str, memory: int, answer: tuple[int, int], seconds: float, connection):
    """A reader process: the depot at address viewed, its data capped (on
    Linux) at memory bytes more than it holds once started, then each job
    the connection sends answered within seconds, a query's answer bounded
    by answer (rows, bytes), until it closes. The limits come as arguments,
    since the process imports this module anew. It first reads once with
    each kind of job, a read that matches nothing, so the cap is set on top
    of what reading holds."""
    depot = Depot.view(address)
    depot.sql("SELECT 1")
    depot.search("false")
    depot.passages("false")
    if sys.platform == "linux":
        import resource
        held = _held()
        resource.setrlimit(resource.RLIMIT_DATA, (held + memory, held + memory))
    jobs = {"sql": lambda query: _sql(depot, query, *answer), "search": depot.search, "passages": depot.passages}
    while True:
        try:
            job, args = connection.recv()
        except EOFError:
            return
        depot.db.deadline = time.monotonic() + seconds
        try:
            connection.send(("done", jobs[job](*args)))
        except errors.DepotError as error:
            connection.send(("refused", type(error).__name__, str(error)))
        except MemoryError:
            connection.send(("refused", ReaderLost.__name__, f"the reads went past {memory} bytes of memory"))
        finally:
            depot.db.deadline = None


class Reader:
    """One reader process and the connection to it."""

    def __init__(self, context, address: str):
        self.connection, theirs = context.Pipe()
        self.process = context.Process(target=_serve, args=(address, MEMORY, (ANSWER_ROWS, ANSWER_BYTES), SECONDS,
                                                            theirs), daemon=True)
        self.process.start()
        theirs.close()

    def kill(self):
        self.process.kill()
        self.process.join()
        self.connection.close()


def _stop(readers: list[Reader]):
    for reader in readers:
        reader.kill()


class Readers:
    """The reader processes of the depot at address."""

    def __init__(self, address: str):
        self.address = address
        self._context = multiprocessing.get_context("spawn")
        self._running = threading.BoundedSemaphore(READERS)
        self._kept: list[Reader] = []
        self._lock = threading.Lock()
        weakref.finalize(self, _stop, self._kept)  # the kept readers end with the routes that use them

    def run(self, job: str, *args):
        """The answer of job (sql, search or passages) over the depot with
        args, within SECONDS: the depot's failure as it is; TimeLimitReached
        or ReaderLost when the reader was stopped or ended."""
        deadline = time.monotonic() + SECONDS
        if not self._running.acquire(timeout=SECONDS):
            raise TimeLimitReached(f"no reader was free within the limit of {SECONDS:g} seconds")
        try:
            reader = self._take()
            try:
                reader.connection.send((job, args))
                if not reader.connection.poll(max(0.0, deadline - time.monotonic())):
                    reader.kill()
                    raise TimeLimitReached(f"the reads ran past their limit of {SECONDS:g} seconds and were stopped")
                answer = reader.connection.recv()
            except (EOFError, OSError):
                reader.kill()
                raise ReaderLost(f"the process of the reads ended before it answered: they went past {MEMORY} "
                                 "bytes of memory") from None
            self._keep(reader)
        finally:
            self._running.release()
        if answer[0] == "done":
            return answer[1]
        kinds = {one.__name__: one for one in (ReaderLost, AnswerTooLarge)}
        kind = kinds.get(answer[1]) or getattr(errors, answer[1], errors.DepotError)
        raise kind(answer[2])

    def _take(self) -> Reader:
        with self._lock:
            while self._kept:
                reader = self._kept.pop()
                if reader.process.is_alive():
                    return reader
                reader.kill()
        return Reader(self._context, self.address)

    def _keep(self, reader: Reader):
        with self._lock:
            if len(self._kept) < KEPT:
                self._kept.append(reader)
                return
        reader.kill()
