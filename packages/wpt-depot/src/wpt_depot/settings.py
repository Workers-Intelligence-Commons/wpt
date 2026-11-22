"""Which depot to work on, and the address it is known by.

WPT_DEPOT names the depot to work on, from the environment, else from a
.env file, as WPT_STORE names the store: the path of its SQLite file, or a
file:// address. WPT_DEPOT_URL, read the same way, is that depot's public
address, the web address of its web app, which names it in what a corpus
built here records (wpt.depot.shared.build_corpus) and in the corpora built
from a dataset before (wpt_depot.legacy.depot_name).
"""

import os

from wpt_store.settings import read_env_file

from .errors import BadDepotSetting, NoDepotSetting

SETTING = "WPT_DEPOT"
URL_SETTING = "WPT_DEPOT_URL"


def depot_address(environ=None, env_file=None) -> str:
    """The depot's address WPT_DEPOT gives, in environ (default: the
    process's environment), else in env_file. NoDepotSetting when neither
    sets it."""
    environ = os.environ if environ is None else environ
    found = environ.get(SETTING) or (read_env_file(env_file).get(SETTING) if env_file is not None else None)
    if not found:
        raise NoDepotSetting(f"set {SETTING} to the depot's file, in the environment or in .env")
    return found


def depot_url(environ=None, env_file=None) -> str | None:
    """The depot's public address WPT_DEPOT_URL gives, in environ (default:
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
