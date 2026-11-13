"""Which depot to work on.

WPT_DEPOT names the depot to work on, from the environment, else from a
.env file, as WPT_STORE names the store: the path of its SQLite file, or a
file:// address.
"""

import os

from wpt_store.settings import read_env_file

from .errors import NoDepotSetting

SETTING = "WPT_DEPOT"


def depot_address(environ=None, env_file=None) -> str:
    """The depot's address WPT_DEPOT gives, in environ (default: the
    process's environment), else in env_file. NoDepotSetting when neither
    sets it."""
    environ = os.environ if environ is None else environ
    found = environ.get(SETTING) or (read_env_file(env_file).get(SETTING) if env_file is not None else None)
    if not found:
        raise NoDepotSetting(f"set {SETTING} to the depot's file, in the environment or in .env")
    return found
