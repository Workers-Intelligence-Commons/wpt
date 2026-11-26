"""What the two controllers of `depot`, its subcommands and its web
routes, share: the depot DEPOT names, the words of the confirmation a
decision needs, rows made fit for JSON, and the check that an estimate a
person confirmed still holds."""

from datetime import date, datetime
from pathlib import Path

from depot import Depot
from depot.settings import depot_address, depot_url

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
