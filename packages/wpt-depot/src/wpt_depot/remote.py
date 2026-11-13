"""A remote: a dataset written out as JSON, which points to where its data
comes from, so anyone can build its corpus without the depot that made it.
A library with no address (null) is one the depot that reads the remote
holds: the depot that made it, or any clone of it.
It holds the dataset's current row but its id, revision and change, each
library with the change number of its depot it is read at:

    {"name": "...", "tokenizer": "...",
     "libraries": [{"depot": "<address>" or null, "library": "<depot name>-3", "change": 12}, ...],
     "shares": [{"tag": "en", "share": 0.3}, ...] or null}
"""

import json

from .errors import BadRemote

FIELDS = ("name", "tokenizer", "libraries", "shares")


def to_remote(dataset: dict) -> dict:
    """The remote of a datasets row."""
    return {name: dataset[name] for name in FIELDS}


def to_json(remote: dict) -> str:
    return json.dumps(remote, ensure_ascii=False, indent=2) + "\n"


def from_json(text: str) -> dict:
    """The remote a JSON text holds, checked (from_remote). BadRemote when it is not one."""
    try:
        found = json.loads(text)
    except ValueError as error:
        raise BadRemote(f"a remote is JSON: {error}") from None
    return from_remote(found)


def from_remote(found) -> dict:
    """A datasets row without its id, from a remote. BadRemote unless it has
    exactly the remote's fields, each of its type."""
    if not isinstance(found, dict) or set(found) != set(FIELDS):
        raise BadRemote(f"a remote is an object with exactly {', '.join(FIELDS)}")
    if not isinstance(found["name"], str) or not isinstance(found["tokenizer"], str):
        raise BadRemote("a remote's name and tokenizer are text")
    libraries = found["libraries"]
    if not isinstance(libraries, list) or not all(
            isinstance(one, dict) and set(one) == {"depot", "library", "change"}
            and (one["depot"] is None or isinstance(one["depot"], str))
            and isinstance(one["library"], str) and _integer(one["change"]) for one in libraries):
        raise BadRemote("a remote's libraries are objects of a depot's address or null, a library id and a change number")
    shares = found["shares"]
    if shares is not None and (not isinstance(shares, list) or not all(
            isinstance(one, dict) and set(one) == {"tag", "share"} and isinstance(one["tag"], str)
            and isinstance(one["share"], (int, float)) and not isinstance(one["share"], bool) for one in shares)):
        raise BadRemote("a remote's shares are null, or objects of a tag and a share")
    return {"name": found["name"], "tokenizer": found["tokenizer"], "libraries": [dict(one) for one in libraries],
            "shares": None if shares is None else [{"tag": one["tag"], "share": float(one["share"])} for one in shares]}


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
