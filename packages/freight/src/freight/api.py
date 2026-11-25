"""What code outside the freight asks of it (docs/api/freight.md): queue
a request, wait for its job, find a delivery again, work out a delivery's
key, mark what an erasure reached, and adopt files made elsewhere as a
delivery.

Every function takes and gives only values JSON can carry: the freight's
home as a path, the depot as its address and the dock as its URL, each
read from the settings when not given (settings). None hands out a
connection, a row object or a path into the freight's own files, so each
can become a route without its callers changing.

    settings(home, depot, dock, cap)        the freight's settings, those not given read from the environment
    key(request, version)                   the delivery key of a request, its program answering that version
    request(request, ...)                   queue a request: its job (done at once when its delivery is on the dock)
    wait(job, ...)                          the job once it ended, the queue run here while no other process does
    work(...)                               run the queue until stopped
    job(job), jobs(), cancel(job)           a job's state, every job's, cancelling one
    delivery(key), deliveries()             a delivery's note, every note on the dock
    items(key)                              the rows of a delivery's items tables
    check(...)                              mark stale every delivery an erasure reached since the last check
    adopt(key, files, result, about)        files made elsewhere put on the dock as a delivery
"""

import os
import tempfile
import time
from pathlib import Path

from wpt_depot import api as depot_api
from wpt_store import open_store, read_env_file
from wpt_utils.lines import say

from . import deliveries
from .jobs import ENDED, Queue
from .job import Job
from .selection import delivery_key

ENV_FILE = ".env"


class SettingMissing(Exception):
    """A setting the freight needs is not set."""


def settings(home: str | None = None, depot: str | None = None, dock: str | None = None,
             cap: int | None = None, environ=None) -> dict:
    """{home, depot, dock, cap}: each given, else read from the environment,
    else from the .env file of the working folder: FREIGHT_HOME (the
    freight's own folder: its queue and the jobs' work), WPT_DEPOT (the
    depot it reads, by its address), WPT_STORE (the dock it delivers on, by
    its URL) and FREIGHT_CAP (the bytes the deliveries may take on the dock
    together; none by default). SettingMissing for a home, depot or dock
    named nowhere."""
    environ = os.environ if environ is None else environ
    file = read_env_file(ENV_FILE)

    def setting(name):
        return environ.get(name) or file.get(name)
    found = {"home": home or setting("FREIGHT_HOME"), "depot": depot or setting("WPT_DEPOT"),
             "dock": dock or setting("WPT_STORE")}
    for name, variable in (("home", "FREIGHT_HOME"), ("depot", "WPT_DEPOT"), ("dock", "WPT_STORE")):
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


def request(request: dict, *, home: str | None = None, depot: str | None = None, dock: str | None = None,
            cap: int | None = None, log=say) -> dict:
    """Queue request (docs/api/freight.md, "A request"): its job's state.
    The request is checked first against the depot (its snapshot, change
    and digest), the dock (its inputs) and the program (its version, which
    the delivery key covers): a request whose delivery the dock holds,
    fresh, is a job done at once, its answer that delivery's note. Refused
    for a request that cannot be met; Busy when its key has a job queued or
    running."""
    found = settings(home, depot, dock, cap)
    queue = Queue(found["home"])
    store = _dock(found["dock"])
    with tempfile.TemporaryDirectory(prefix="probe-", dir=_made(Path(found["home"]))) as folder:
        planned = Job(request, found["depot"], store, Path(folder), log=log)
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


def work(*, home: str | None = None, stop=lambda: False) -> bool:
    """Run the queue until stop() says so: False when another process runs it."""
    return Queue(settings_home(home)).run(stop=stop)


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
    return deliveries.of_key(_dock(settings_dock(dock)), key)


def deliveries_on(*, dock: str | None = None) -> list[dict]:
    """The note of every delivery on the dock, the last made first."""
    return deliveries.notes(_dock(settings_dock(dock)))


def items(key: str, *, dock: str | None = None) -> list[dict]:
    """The rows of the items tables of the delivery of that key, in order."""
    store = _dock(settings_dock(dock))
    note = deliveries.of_key(store, key)
    if note is None:
        raise deliveries.DeliveryError(f"the dock holds no delivery {key}")
    return deliveries.read_items(store, note).to_pylist()


def check(*, home: str | None = None, depot: str | None = None, dock: str | None = None) -> dict:
    """Ask the depot which erasures it carried out since the mark the
    freight saw last there, and mark stale every fresh delivery whose items
    hold a text one of them reached: {mark (the depot's mark now), erasures
    (their ids), marked (the keys of the deliveries marked)}."""
    found = settings(home, depot, dock)
    queue = Queue(found["home"])
    since = queue.mark(found["depot"])
    answer = depot_api.erasures(found["depot"], since)
    marked = []
    store = _dock(found["dock"])
    for one in answer["erasures"]:
        marked += deliveries.mark_stale(store, one["texts"], [one["id"]])
    queue.keep_mark(found["depot"], answer["mark"])
    return {"mark": answer["mark"], "erasures": [one["id"] for one in answer["erasures"]],
            "marked": sorted(set(marked))}


def adopt(key: str, files: dict[str, str], result: dict, about: dict, *, dock: str | None = None) -> dict:
    """Put files made elsewhere ({path under files/: local path}) on the dock as the delivery of key, which starts
    with `adopted-`: its note (freight.deliveries.adopt)."""
    return deliveries.adopt(_dock(settings_dock(dock)), key, {path: Path(local) for path, local in files.items()},
                            result, about)


def settings_home(home: str | None) -> str:
    environ_home = home or os.environ.get("FREIGHT_HOME") or read_env_file(ENV_FILE).get("FREIGHT_HOME")
    if not environ_home:
        raise SettingMissing("FREIGHT_HOME is not set: put it in the environment or in .env")
    return environ_home


def settings_dock(dock: str | None) -> str:
    found = dock or os.environ.get("WPT_STORE") or read_env_file(ENV_FILE).get("WPT_STORE")
    if not found:
        raise SettingMissing("WPT_STORE is not set: put it in the environment or in .env")
    return found


def _dock(url: str):
    """The dock of that URL, its keys read from the environment or .env."""
    return open_store(url, env_file=ENV_FILE)


def _made(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder
