"""What the two controllers of `depot`, its subcommands and its web
routes, share: the depot DEPOT names, the words of the confirmation a
decision needs, rows made fit for JSON, the check that an estimate a person
confirmed still holds, and an erasure carried out with the corpora it makes
stale."""

import threading
from datetime import date, datetime
from pathlib import Path

from depot import Declaration, Depot, pull
from depot.schema import ADMITTED
from depot.settings import depot_address, depot_url
from dock.api import DockSettingError, dock_from_settings

from . import corpora

# what a person confirms with every decision: the subcommands' --accept and the web pages' button say it
CONFIRMATION = ("I read the requirements, I accept them, and what I submit meets them. My name and email are "
                "recorded on this decision and are public; nobody verifies them.")
# the page the confirmation's "requirements" means
REQUIREMENTS = "https://github.com/Workers-Intelligence-Commons/wpt/blob/main/docs/requirements.md"
# how far a cost may pass the one a person confirmed and still hold (unchanged): its share, then its unit of slack
TOLERANCE = 0.1


def env_file() -> Path:
    """The .env file the settings fall back on: the current folder's."""
    return Path.cwd() / ".env"


def unchanged(shown, now: dict, costs: dict[str, float], same=()) -> list[str]:
    """What makes the estimate now differ from the one a person confirmed
    (shown, as the route gave it): each key of same whose value changed,
    and each cost of costs ({key: its slack, in its unit}) now above the
    shown one by more than TOLERANCE of it plus its slack; [] when the
    confirmed estimate still holds. A cost now lower always holds."""
    if not isinstance(shown, dict):
        return ["the estimate shown was not given back with the confirmation"]
    found = []
    for key in same:
        if shown.get(key) != now.get(key):
            found.append(f"{key} was {shown.get(key)!r} and is now {now.get(key)!r}")
    for key, slack in costs.items():
        was, then = shown.get(key), now.get(key)
        if then is None:
            continue
        if not isinstance(was, (int, float)) or isinstance(was, bool):
            found.append(f"the estimate shown gives no {key}")
        elif then > was * (1 + TOLERANCE) + slack:
            found.append(f"{key} was {was:,} and is now {then:,}")
    return found


def opened() -> Depot:
    """The depot DEPOT names, the change files a crash left pending kept
    or dropped (depot.database). NoDepotSetting without the setting;
    NotADepot, UnsupportedAddress for an address that holds no depot."""
    return Depot.open(depot_address(env_file=env_file()))


def public_url() -> str | None:
    """The depot's public address DEPOT_URL gives (depot.settings.depot_url); None without it."""
    return depot_url(env_file=env_file())


def viewed(address: str) -> Depot:
    """The depot at address, to read."""
    return Depot.view(address)


def corpora_store(log):
    """The dock DOCK names, where the corpora are; None, said in the
    log, when it is not set: no corpus of this machine can be marked."""
    try:
        return dock_from_settings(env_file())
    except DockSettingError as error:
        log(f"no corpus is marked stale: {error}")
        return None


def decide_erasures(depot: Depot, requests: list[str], status: str, declaration: Declaration, reason: str | None,
                    log, downloads=None) -> dict[str, bool]:
    """depot.decide_erasures, then every corpus of the dock that holds a
    text the carried-out erasures cut or removed marked stale (said in the
    log): whether each was carried out, by id. The dock is found before
    any erasure is carried out, since carrying one out cannot be undone.
    With downloads (the web app's depot_app.downloads.Downloads), an
    admission is kept as being marked before it is carried out, and the
    marking runs in the background (settle_marks)."""
    store = corpora_store(log)
    if downloads is None:
        carried_out = depot.decide_erasures(requests, status, declaration, reason)
        done = [request for request, carried in carried_out.items() if carried]
        if done:
            stale(depot, store, done, log)
        return carried_out
    if status == ADMITTED:
        downloads.marking(requests)
    try:
        return depot.decide_erasures(requests, status, declaration, reason)
    finally:
        settle_marks(depot, downloads, log)


def settle_marks(depot: Depot, downloads, log) -> threading.Thread | None:
    """The erasures downloads keeps as being marked: those never carried out
    leave it at once; the corpora holding a text of the others are marked
    stale in a thread of their own, which takes them out of it once done
    (a failure is said in the log, and they stay until the next call, at
    the web app's next start or next erasure decided): that thread, None
    without one."""
    pending = downloads.pending()
    if not pending:
        return None
    carried = {row["id"] for row in depot.erasures() if row["id"] in pending and row["erased"] is not None}
    downloads.marked([one for one in pending if one not in carried])
    if not carried:
        return None

    def mark():
        try:
            stale(depot, corpora_store(log), sorted(carried), log)
        except Exception as error:
            log(f"the corpora holding a text of the erasures {', '.join(sorted(carried))} are not marked stale yet "
                f"({type(error).__name__}: {error}): no corpus is sent until they are")
            return
        downloads.marked(sorted(carried))
    thread = threading.Thread(target=mark, name="marking", daemon=True)
    thread.start()
    return thread


def pulled(depot: Depot, log) -> list[str]:
    """pull(depot), then every corpus of the dock holding a text the erasures
    it carried out cut or removed marked stale (said in the log): the
    erasures."""
    store = corpora_store(log)
    brought = pull(depot)
    if brought:
        stale(depot, store, brought, log)
    return brought


def stale(depot: Depot, store, erasures: list[str], log):
    if store is None:
        return
    texts = sorted({needle["text"] for row in depot.erasures() if row["id"] in erasures
                    for needle in row["needles"]})
    marked = corpora.mark_stale(store, texts, erasures)
    log(f"{len(marked)} corpora marked stale: nothing downloads them, and wpt train refuses them, "
        "until they are built again" if marked else "no corpus of the dock holds a text of the erasure")


def jsonable(value):
    """value with every time as ISO 8601 text and every byte string as
    hexadecimal text, the rest as it is: what a row of the depot's tables
    becomes in JSON."""
    if isinstance(value, dict):
        return {key: jsonable(one) for key, one in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(one) for one in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return value
