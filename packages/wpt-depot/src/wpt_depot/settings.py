"""Which depot to work on, and how to reach a depot in a bucket.

WPT_DEPOT names the depot to work on, from the environment, else from a
.env file, as WPT_STORE names the store: a folder, or a bucket
(s3://bucket/prefix). A depot in a bucket reads its own settings, never the
store's: WPT_DEPOT_ACCESS_KEY_ID and WPT_DEPOT_SECRET_ACCESS_KEY (required),
and WPT_DEPOT_SESSION_TOKEN, WPT_DEPOT_ENDPOINT_URL (a non-AWS endpoint) and
WPT_DEPOT_REGION (default us-east-1). It reads no AWS_* setting, and every
depot in a bucket that a command opens uses the same ones.
"""

import os

from wpt_store.settings import read_env_file

from .errors import NoDepotSetting, NoBucketSetting

SETTING = "WPT_DEPOT"
BUCKET_SETTINGS = {"access_key_id": "WPT_DEPOT_ACCESS_KEY_ID", "secret_access_key": "WPT_DEPOT_SECRET_ACCESS_KEY",
                   "session_token": "WPT_DEPOT_SESSION_TOKEN", "endpoint_url": "WPT_DEPOT_ENDPOINT_URL",
                   "region": "WPT_DEPOT_REGION"}
REQUIRED = ("access_key_id", "secret_access_key")
DEFAULT_REGION = "us-east-1"


def depot_address(environ=None, env_file=None) -> str:
    """The depot's address WPT_DEPOT gives, in environ (default: the
    process's environment), else in env_file. NoDepotSetting when neither
    sets it."""
    environ = os.environ if environ is None else environ
    found = environ.get(SETTING) or (read_env_file(env_file).get(SETTING) if env_file is not None else None)
    if not found:
        raise NoDepotSetting(f"set {SETTING} to the depot's folder, in the environment or in .env")
    return found


def bucket_settings(environ=None, env_file=None) -> dict:
    """{name: value} of the bucket settings (BUCKET_SETTINGS) set in
    environ (default: the process's environment), else in env_file.
    NoBucketSetting when a required one is not set."""
    environ = os.environ if environ is None else environ
    from_file = read_env_file(env_file) if env_file is not None else {}
    found = {name: value for name, variable in BUCKET_SETTINGS.items()
             if (value := environ.get(variable) or from_file.get(variable))}
    missing = [BUCKET_SETTINGS[name] for name in REQUIRED if name not in found]
    if missing:
        raise NoBucketSetting(f"a depot in a bucket needs {' and '.join(missing)} (in the environment or in .env); "
                              "it reads no AWS_* setting and not the store's")
    return found
