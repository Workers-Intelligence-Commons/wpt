"""A person's SQL, run by `wpt depot web` in processes of their own: the
query of POST /api/sql and the condition of a filter (search, passages).

DuckDB checks an interruption only between batches of rows, so a single
large operation of one query can run on, and take memory, past any limit
set inside the server's process. A reader process holds its own view of
the depot (Depot.view, under the same DuckDB lock on the depot's folder),
its memory capped at MEMORY bytes more than it holds once started (on
Linux, through the limit the system puts on a process's data), and the
server waits for its answer at most SECONDS from the request's start,
waiting for a free reader included: then it kills the process and answers
TimeLimitReached. A reader that dies (past its memory) answers ReaderLost.
Either way the server's own process goes on, and the next request gets a
new reader. At most READERS run at a time, and at most KEPT of those that
answered wait for the next request, since starting one costs about half
a second.
"""

import json
import multiprocessing
import sys
import threading
import time
import weakref

from wpt_depot import Depot, errors, tables

SECONDS = 60  # a request's whole time, under the 100 seconds a proxy in front waits
MEMORY = 2 << 30  # bytes of data a reader process may hold more than it holds once started
READERS = 2  # reader processes running at a time
KEPT = 1  # idle reader processes kept for the next requests
ANSWER_ROWS = 100_000  # rows a query's answer holds at most
ANSWER_BYTES = 16 << 20  # bytes of JSON a query's answer holds at most


class TimeLimitReached(errors.DepotError):
    """The reads of a request ran past SECONDS; their process was killed."""


class ReaderLost(errors.DepotError):
    """The process that ran the reads ended before it answered: it went past MEMORY."""


class AnswerTooLarge(errors.DepotError):
    """A query's answer holds more than ANSWER_ROWS rows or ANSWER_BYTES bytes of JSON."""


def _sql(depot: Depot, query: str, most_rows: int, most_bytes: int) -> dict:
    """{columns, rows} of the query; AnswerTooLarge past most_rows rows or
    most_bytes bytes of JSON, found a thousand rows at a time."""
    found = depot.sql(query)
    if found.num_rows > most_rows:
        raise AnswerTooLarge(f"the query's answer holds {found.num_rows} rows, more than {most_rows}: narrow it, "
                             "or add LIMIT and OFFSET")
    rows, size = [], 0
    for batch in found.to_batches(max_chunksize=1000):
        part = batch.to_pylist()
        size += len(json.dumps(part, default=str))
        if size > most_bytes:
            raise AnswerTooLarge(f"the query's answer holds more than {most_bytes} bytes of JSON: narrow it, or "
                                 "add LIMIT and OFFSET")
        rows += part
    return {"columns": found.column_names, "rows": rows}


def _held() -> int:
    """The bytes of data this process holds now, as the system's limit counts them (Linux)."""
    with open("/proc/self/status", encoding="utf-8") as status:
        found = next(line for line in status if line.startswith("VmData:"))
    return int(found.split()[1]) * 1024


def _serve(address: str, settings: dict | None, memory: int, answer: tuple[int, int], connection):
    """A reader process: the depot at address viewed, its data capped (on
    Linux) at memory bytes more than it holds once started, then each job
    the connection sends answered, a query's answer bounded by answer
    (rows, bytes), until it closes. The limits come as arguments, since the
    process imports this module anew.

    The system's limit counts the memory a process reserves to write, and
    the threads of DuckDB and delta-rs reserve gigabytes they mostly never
    use: so the process first starts them, with one read of each kind that
    matches nothing, and the cap is set on top of what it then holds."""
    tables.MEMORY_LIMIT = memory // 2  # DuckDB writes to its temporary folder before the cap stops it
    depot = Depot.view(address, settings)
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
        try:
            connection.send(("done", jobs[job](*args)))
        except errors.DepotError as error:
            connection.send(("refused", type(error).__name__, str(error)))
        except MemoryError:
            connection.send(("refused", ReaderLost.__name__, f"the reads went past {memory} bytes of memory"))


class Reader:
    """One reader process and the connection to it."""

    def __init__(self, context, address: str, settings: dict | None):
        self.connection, theirs = context.Pipe()
        self.process = context.Process(target=_serve, args=(address, settings, MEMORY,
                                                            (ANSWER_ROWS, ANSWER_BYTES), theirs), daemon=True)
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

    def __init__(self, address: str, settings: dict | None):
        self.address, self.settings = address, settings
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
        return Reader(self._context, self.address, self.settings)

    def _keep(self, reader: Reader):
        with self._lock:
            if len(self._kept) < KEPT:
                self._kept.append(reader)
                return
        reader.kill()
