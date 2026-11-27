"""Which depot to work on, and the address it is known by.

DEPOT names the depot to work on, from the environment, else from a
.env file, as DOCK names the dock: the path of its SQLite file, or a
file:// address. DEPOT_URL, read the same way, is that depot's public
address, the web address of its web app, which its web app gives as the
depot's address (GET /api/about, what an order's data section names it by)
and which names it in the corpora built from a dataset before snapshots,
whose records an upgrade to format 13 reads (depot.upgrade).
"""

import os

from wic_utils.env import read_env_file

from .errors import BadDepotSetting, NoDepotSetting

SETTING = "DEPOT"
URL_SETTING = "DEPOT_URL"


def depot_address(environ=None, env_file=None) -> str:
    """The depot's address DEPOT gives, in environ (default: the
    process's environment), else in env_file. NoDepotSetting when neither
    sets it."""
    environ = os.environ if environ is None else environ
    found = environ.get(SETTING) or (read_env_file(env_file).get(SETTING) if env_file is not None else None)
    if not found:
        raise NoDepotSetting(f"set {SETTING} to the depot's file, in the environment or in .env")
    return found


def depot_url(environ=None, env_file=None) -> str | None:
    """The depot's public address DEPOT_URL gives, in environ (default:
    the process's environment), else in env_file, without a closing slash;
    None when neither sets it. BadDepotSetting for a value that is not an
    http:// or https:// address."""
    environ = os.environ if environ is None else environ
    found = environ.get(URL_SETTING) or (read_env_file(env_file).get(URL_SETTING) if env_file is not None else None)
    if not found:
        return None
    if not found.startswith(("http://", "https://")):
        raise BadDepotSetting(f"{URL_SETTING}={found!r}: give the web address of the depot's web app (https://...)")
    return found.rstrip("/")
