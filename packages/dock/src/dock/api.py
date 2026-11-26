"""What code outside the dock asks of it (docs/api/dock.md): the three
gestures, find, fetch and put, on a dock named by its URL, the URL the
settings name, and the dock as an object for the callers that read and
write through it.

The three gestures take and give only values JSON and byte streams can
carry, so each can become a route without its callers changing. The dock
as an object (open_dock, dock_from_settings) is what WPT's training, its
records and the depot's backups still work through; docs/api/dock.md lists
what it offers.

    url(env_file, environ)            the dock's URL that DOCK names
    find(dock, prefix)                every file whose key starts with prefix: [{key, bytes}]
    fetch(dock, key, sha256)          a file's bytes, checked against its sha256 when it is given
    put(dock, key, data)              data stored under key: {key, sha256, bytes}
    open_dock(url), dock_from_settings(env_file)
                                      the dock as an object (Dock)
    File, NAME, PACKED, check_name, as_record_name, check_record_name
                                      the names a dock gives its files and records
    DockError, DockSettingError, MissingFile, CorruptFile, MixedRecord
                                      what a dock refuses
"""

import os

from wic_utils.env import read_env_file

from . import PACKED, Dock, dock_from_settings, local_folder, open_dock
from .errors import CorruptFile, DockError, DockSettingError, MissingFile, MixedRecord
from .names import NAME, File, as_record_name, check_name, check_record_name

SETTING = "DOCK"
__all__ = ["NAME", "PACKED", "SETTING", "CorruptFile", "Dock", "DockError", "DockSettingError", "File", "MissingFile",
           "MixedRecord", "as_record_name", "check_name", "check_record_name", "dock_from_settings", "fetch", "find", "local_folder",
           "open_dock", "put", "url"]

_OPENED = {}  # a dock's URL: the dock opened for it, once in each process


def url(env_file=".env", environ=None) -> str:
    """The dock's URL DOCK gives, in environ (default: the process's
    environment), else in env_file. DockSettingError when neither sets it."""
    environ = os.environ if environ is None else environ
    found = environ.get(SETTING) or read_env_file(env_file).get(SETTING)
    if not found:
        raise DockSettingError(f"{SETTING} is not set: put {SETTING}=file:///path or s3://bucket/prefix in .env")
    return found


def _dock(dock: str) -> Dock:
    if dock not in _OPENED:
        _OPENED[dock] = open_dock(dock, env_file=".env")
    return _OPENED[dock]


def find(dock: str, prefix: str) -> list[dict]:
    """Every file of the dock whose key starts with prefix, in the order of
    their keys: {key, bytes}, bytes the size it takes on the dock (a file
    kept compressed counts its compressed size). A key is given without the
    `.zst` the dock adds to a file it keeps compressed."""
    found = {}
    for key, size in _dock(dock).kind.list(prefix).items():
        if size is None or key.rsplit("/", 1)[-1].startswith("."):
            continue
        found[key.removesuffix(PACKED)] = size
    return [{"key": key, "bytes": size} for key, size in sorted(found.items())]


def fetch(dock: str, key: str, sha256: str | None = None) -> bytes:
    """The bytes of the file under key, checked against sha256 when it is
    given (a record's reference to a file gives it). MissingFile when the
    dock holds no such file, CorruptFile when it does not match."""
    held = _dock(dock)
    if sha256 is not None:
        return held.get_bytes(File(key, sha256))
    data = held.whole(key)
    if data is None:
        raise MissingFile(f"{key} is not on the dock")
    return data


def put(dock: str, key: str, data: bytes) -> dict:
    """data stored under key, replacing any earlier file: {key, sha256,
    bytes}, the reference that fetches it back checked."""
    stored = _dock(dock).put_named(key, bytes(data))
    return {**stored.reference(), "bytes": len(data)}
