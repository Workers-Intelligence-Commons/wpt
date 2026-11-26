"""What the tests read of a dock, named by its URL, through dock.api, a
file named by a record's reference to it ({key, sha256}); and the folder
of a local dock, for a test that damages or removes its files itself."""

from pathlib import Path
from urllib.parse import unquote, urlparse

from dock import api


def fetch(dock: str, reference: dict) -> bytes:
    """The file's bytes, checked against its sha256."""
    return api.fetch(dock, reference["key"], reference["sha256"])


def has(dock: str, reference) -> bool:
    """Whether the dock holds the file (a reference, or its key)."""
    return api.has(dock, reference if isinstance(reference, str) else reference["key"])


def into(dock: str, reference: dict, path) -> Path:
    """The file fetched into the local file path, checked: path."""
    api.fetch_into(dock, reference["key"], path, reference["sha256"])
    return Path(path)


def folder(dock: str) -> Path:
    """The folder of a local dock's file:// URL."""
    return Path(unquote(urlparse(dock).path))
