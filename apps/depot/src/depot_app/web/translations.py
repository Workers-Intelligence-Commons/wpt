"""The routes behind the Translate button of the Texts page: the language
pairs that Firefox's translation models cover, and each model file, named
by its sha256.

The translation itself runs in the browser (apps/depot/ui/src/translation/),
so no text reaches this server for it and none leaves the machine. Nothing
of a translation is written to the depot.

The models are Mozilla's, under MPL-2.0, listed in Firefox's Remote
Settings with the sha256 of every file. That service lets a page read the
list, but its file server refuses Chromium browsers, so this server
fetches a file the first time a page asks for it, checks it against its
sha256, keeps it on the dock (cache/files/<sha256>) and serves it from
there. It fetches only the files the list names. A file never changes
under its sha256, so browsers are told to keep it for a year.
"""

import hashlib
import json
import re
import threading
import time

from depot import errors
from dock import api as dock_api
from wic_utils.web import urlopen

from .. import shared
from .server import Route, Stream

RECORDS = "https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models/records"
ATTACHMENTS = "https://firefox-settings-attachments.cdn.mozilla.net/"
LISTED_FOR = 3600  # seconds the list of models is kept before it is read again
TIMEOUT = 120  # seconds one request to Mozilla may take, a model file of about 40 MB included
KEPT = "public, max-age=31536000, immutable"  # a model file named by its sha256 never changes
VOCABULARIES = ({"vocab"}, {"srcvocab", "trgvocab"})  # one vocabulary, or one for each side
VERSION = re.compile(r"(\d+)\.(\d+)(?:a(\d+))?")


class ModelsUnreachable(Exception):
    """Mozilla's list or file could not be read, or a file did not match its sha256."""


def version_key(version: str) -> tuple:
    """A model version's order: 1.0a1 before 1.0, before 1.1, before 2.0."""
    found = VERSION.fullmatch(version)
    if found is None:
        return (-1,)
    major, minor, alpha = found.groups()
    return (int(major), int(minor), 0 if alpha else 1, int(alpha or 0))


def pairs_of(records: list[dict]) -> list[dict]:
    """The newest complete model of each language pair among Remote Settings'
    records, leaving out those served to some Firefox builds only (a
    filter_expression): [{from, to, version, files: {part: {sha256, size}}}],
    a part being model, lex, and vocab or srcvocab and trgvocab."""
    versions: dict[tuple, dict] = {}
    for record in records:
        if record.get("filter_expression") or not record.get("attachment"):
            continue
        key = (record["fromLang"], record["toLang"], record["version"])
        versions.setdefault(key, {})[record["fileType"]] = {
            "sha256": record["attachment"]["hash"], "size": record["attachment"]["size"],
            "location": record["attachment"]["location"]}
    best: dict[tuple, tuple] = {}
    for (source, target, version), files in versions.items():
        complete = {"model", "lex"} <= set(files) and any(needed <= set(files) for needed in VOCABULARIES)
        current = best.get((source, target))
        if complete and (current is None or version_key(version) > version_key(current[0])):
            best[(source, target)] = (version, files)
    return [{"from": source, "to": target, "version": version, "files": files}
            for (source, target), (version, files) in sorted(best.items())]


def translation_routes(refusals) -> list[Route]:
    """The two routes; refusals wraps a handler so that it answers each
    failure with its status (depot_app.web.routes.refusals)."""
    listed = {"at": 0.0, "pairs": None}
    listing = threading.Lock()
    fetching: dict[str, threading.Lock] = {}  # one lock per file, so two pages never fetch the same file at once
    fetching_lock = threading.Lock()

    def pairs() -> list[dict]:
        with listing:
            if listed["pairs"] is None or time.time() - listed["at"] > LISTED_FOR:
                try:
                    with urlopen(RECORDS, timeout=TIMEOUT) as answer:
                        records = json.loads(answer.read())["data"]
                except (OSError, ValueError, KeyError) as error:
                    if listed["pairs"] is None:
                        raise ModelsUnreachable(f"Mozilla's list of translation models: {error}") from None
                    records = None  # the list read before stays in use
                if records is not None:
                    listed.update(at=time.time(), pairs=pairs_of(records))
            return listed["pairs"]

    @refusals
    def listed_pairs(values, body):
        return 200, {"pairs": [{**pair, "files": {part: {"sha256": file["sha256"], "size": file["size"]}
                                                  for part, file in pair["files"].items()}}
                               for pair in pairs()]}

    @refusals
    def model_file(values, body):
        sha256 = values["sha256"]
        location = next((file["location"] for pair in pairs() for file in pair["files"].values()
                         if file["sha256"] == sha256), None)
        if location is None:
            raise errors.NotFound(f"no translation model file {sha256} in Mozilla's list")
        dock, key = dock_api.url(shared.env_file()), dock_api.content_key(sha256)
        with fetching_lock:
            lock = fetching.setdefault(sha256, threading.Lock())
        with lock:
            if not dock_api.has(dock, key):
                try:
                    with urlopen(ATTACHMENTS + location, timeout=TIMEOUT) as answer:
                        data = answer.read()
                except OSError as error:
                    raise ModelsUnreachable(f"the translation model file {sha256}: {error}") from None
                if hashlib.sha256(data).hexdigest() != sha256:
                    raise ModelsUnreachable(f"the translation model file {sha256} came with other bytes; nothing kept")
                dock_api.put_content(dock, data)
        data = dock_api.fetch(dock, key, sha256)
        return 200, Stream("application/octet-stream", iter([data]), len(data), {"Cache-Control": KEPT})

    return [
        Route("GET", "/api/translations", listed_pairs, "the language pairs Firefox's translation models cover, "
              "each with its newest model's version and files (model, lex, and vocab or srcvocab and trgvocab: "
              "sha256 and size), read from Mozilla's Remote Settings; the Texts page translates a plain text in the "
              "browser with them, and nothing of a translation is written",
              answer={"type": "object", "required": ["pairs"],
                      "properties": {"pairs": {"type": "array", "items": {
                          "type": "object", "required": ["from", "to", "version", "files"]}}}}),
        Route("GET", "/api/translations/files/{sha256}", model_file, "one file of a translation model that "
              "GET /api/translations lists, by its sha256: fetched from Mozilla the first time, checked, then kept "
              "on the dock and served from it, for browsers to keep a year", answer={"type": "string", "format": "binary"}),
    ]
