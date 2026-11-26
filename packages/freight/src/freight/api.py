"""What code outside the freight asks of it (docs/api/freight.md): queue
a request, wait for its job, find a delivery again, work out a delivery's
key, mark what an erasure reached, and adopt files made elsewhere as a
delivery.

Every function takes and gives only values JSON and byte streams can
carry: the freight's home as a path, the depot as its address and the dock
as its URL, each read from the settings when not given (settings), and a
file's content as bytes. None takes a function, and none hands out a
connection, a row object or a path into the freight's own files, so each
can become a route without its callers changing.

    settings(home, depot, dock, cap)        the freight's settings, those not given read from the environment
    key(request, version)                   the delivery key of a request, its program answering that version
    canonical(selection)                    a request's selection in the form its delivery key covers
    request(request, ...)                   queue a request: its job (done at once when its delivery is on the dock)
    wait(job, ...)                          the job once it ended, the queue run here while no other process does
    work(...)                               run the queue until stopped, or for a number of seconds
    job(job), jobs(), cancel(job)           a job's state, every job's, cancelling one
    delivery(key), deliveries_on()          a delivery's note, every note on the dock
    items(key)                              the rows of a delivery's items tables
    check(...)                              mark what the erasures since the last check did to the deliveries
                                            and to the corpora under corpora/
    adopt(key, files, result, about, ...)   files made elsewhere put on the dock as a delivery, with its items
    ADOPTED                                 the start of the key of an adopted delivery
    Refused, Busy, SettingMissing           what a call refuses: a request that cannot be met, a key with a job
                                            queued or running already, a setting named nowhere
"""

import os
import tempfile
import time
from pathlib import Path

from wic_utils.env import read_env_file

from . import deliveries, erasures
from .jobs import ENDED, Busy, Queue
from .job import Job
from .selection import Refused, delivery_key
from .selection import canonical as canonical_selection

ENV_FILE = ".env"
ADOPTED = deliveries.ADOPTED
__all__ = ["ADOPTED", "Busy", "Refused", "SettingMissing", "adopt", "cancel", "canonical", "check", "deliveries_on",
           "delivery", "items", "job", "jobs", "key", "request", "settings", "settings_dock", "settings_home", "wait",
           "work"]


class SettingMissing(Exception):
    """A setting the freight needs is not set."""


def settings(home: str | None = None, depot: str | None = None, dock: str | None = None,
             cap: int | None = None, environ=None) -> dict:
    """{home, depot, dock, cap}: each given, else read from the environment,
    else from the .env file of the working folder: FREIGHT_HOME (the
    freight's own folder: its queue and the jobs' work), DEPOT (the
    depot it reads, by its address), DOCK (the dock it delivers on, by
    its URL) and FREIGHT_CAP (the bytes the deliveries may take on the dock
    together; none by default). SettingMissing for a home, depot or dock
    named nowhere."""
    environ = os.environ if environ is None else environ
    file = read_env_file(ENV_FILE)

    def setting(name):
        return environ.get(name) or file.get(name)
    found = {"home": home or setting("FREIGHT_HOME"), "depot": depot or setting("DEPOT"),
             "dock": dock or setting("DOCK")}
    for name, variable in (("home", "FREIGHT_HOME"), ("depot", "DEPOT"), ("dock", "DOCK")):
        if not found[name]:
            raise SettingMissing(f"{variable} is not set: put it in the environment or in .env")
    if cap is None and setting("FREIGHT_CAP"):
        cap = int(setting("FREIGHT_CAP"))
    return {**found, "cap": cap}


def key(request: dict, version: str) -> str:
    """The delivery key of request when its preparation's program answers
    version (freight.selection.delivery_key): worked out from the request
    alone, with no depot, dock or program at hand."""
    preparation = request.get("preparation") or {}
    return delivery_key(request["snapshot"], request.get("selection") or {},
                        {"name": preparation["name"], "version": version, "settings": preparation.get("settings"),
                         "inputs": preparation.get("inputs")})


def canonical(selection: dict) -> dict:
    """A request's selection in the form its delivery key covers
    (freight.selection.canonical): {kinds sorted, each once; shares in
    order, each share a float; size None or {unit, amount}}. Refused for a
    selection that cannot be met."""
    return canonical_selection(selection)


def request(request: dict, *, home: str | None = None, depot: str | None = None, dock: str | None = None,
            cap: int | None = None) -> dict:
    """Queue request (docs/api/freight.md, "A request"): its job's state.
    The erasures carried out since the last check are marked first
    (check). The request is checked against the depot (its snapshot, change
    and digest), the dock (its inputs) and the program (its version, which
    the delivery key covers): a request whose delivery the dock holds,
    fresh, is a job done at once, its answer that delivery's note. Refused
    for a request that cannot be met; Busy when its key has a job queued or
    running."""
    found = settings(home, depot, dock, cap)
    queue = Queue(found["home"])
    store = found["dock"]
    erasures.check(queue, found["depot"], store)
    with tempfile.TemporaryDirectory(prefix="probe-", dir=_made(Path(found["home"]))) as folder:
        planned = Job(request, found["depot"], store, Path(folder))
        delivery = planned.plan()
    known = deliveries.of_key(store, delivery)
    if known is not None and deliveries.fresh(known):
        return queue.answered(request, delivery, found["depot"], found["dock"], known)
    return queue.add(request, delivery, found["depot"], found["dock"], found["cap"])


def wait(job: str, *, home: str | None = None, seconds: float | None = None) -> dict:
    """The state of the job once it ended, or after seconds (None: no
    limit), which a job this process runs is never stopped for: while no
    other process runs the queue, this one runs it, the jobs queued before
    this one first."""
    queue = Queue(settings_home(home))
    deadline = None if seconds is None else time.monotonic() + seconds

    def ended() -> bool:
        return queue.state(job)["state"] in ENDED

    def over() -> bool:
        return ended() or (deadline is not None and time.monotonic() >= deadline)
    while not over():
        if not queue.run(until=over, stop=ended):
            time.sleep(0.2)
    return queue.state(job)


def work(*, home: str | None = None, seconds: float | None = None) -> bool:
    """Run the queue until the process is stopped, or for seconds (None: no
    limit; a job running then is cut short and goes on from its checkpoint
    when the queue runs again): False when another process runs it."""
    deadline = None if seconds is None else time.monotonic() + seconds
    return Queue(settings_home(home)).run(stop=lambda: deadline is not None and time.monotonic() >= deadline)


def job(job: str, *, home: str | None = None) -> dict:
    """The state of the job: {id, key, request, state, stages, progress, answer, problem, created, started, ended,
    restarts}."""
    return Queue(settings_home(home)).state(job)


def jobs(*, home: str | None = None, limit: int = 100) -> list[dict]:
    """Every job queued or running, in the order they run, then the limit latest that ended."""
    return Queue(settings_home(home)).listed(limit)


def cancel(job: str, *, home: str | None = None) -> dict:
    """Cancel a queued or running job: its state as it was."""
    return Queue(settings_home(home)).cancel(job)


def delivery(key: str, *, dock: str | None = None) -> dict | None:
    """The note of the delivery of that key, None when the dock holds none."""
    return deliveries.of_key(settings_dock(dock), key)


def deliveries_on(*, dock: str | None = None) -> list[dict]:
    """The note of every delivery on the dock, the last made first."""
    return deliveries.notes(settings_dock(dock))


def items(key: str, *, dock: str | None = None) -> list[dict]:
    """The rows of the items tables of the delivery of that key, in order."""
    store = settings_dock(dock)
    note = deliveries.of_key(store, key)
    if note is None:
        raise deliveries.DeliveryError(f"the dock holds no delivery {key}")
    return deliveries.read_items(store, note).to_pylist()


def check(*, home: str | None = None, depot: str | None = None, dock: str | None = None) -> dict:
    """Ask the depot which erasures it carried out since the mark the
    freight saw last there, and write what each did into every delivery
    whose items hold a text it reached, as its note's on_erasure says:
    {mark (the depot's mark now), erasures (their ids), stale (the keys of
    the deliveries marked stale), noted (those whose note took an erasure
    in and stay fresh), corpora (the ids of the corpora under corpora/,
    built before the freight, marked stale)}. A job asks the same before it runs."""
    found = settings(home, depot, dock)
    return erasures.check(Queue(found["home"]), found["depot"], found["dock"])


def adopt(key: str, files: dict[str, bytes], result: dict, about: dict, *, on_erasure: str = deliveries.STALE,
          items: list[dict] | None = None, dock: str | None = None) -> dict:
    """Put files made elsewhere ({path under files/: its bytes}) on the dock as the delivery of key, which starts
    with `adopted-`, its items table the rows items gives (each with the `depot` and `id` of a text), an erasure
    that reaches one of its items doing what on_erasure says ("stale" or "note"): its note
    (freight.deliveries.adopt)."""
    return deliveries.adopt(settings_dock(dock), key, files, result, about, on_erasure, items)


def settings_home(home: str | None) -> str:
    environ_home = home or os.environ.get("FREIGHT_HOME") or read_env_file(ENV_FILE).get("FREIGHT_HOME")
    if not environ_home:
        raise SettingMissing("FREIGHT_HOME is not set: put it in the environment or in .env")
    return environ_home


def settings_dock(dock: str | None) -> str:
    found = dock or os.environ.get("DOCK") or read_env_file(ENV_FILE).get("DOCK")
    if not found:
        raise SettingMissing("DOCK is not set: put it in the environment or in .env")
    return found


def _made(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder
