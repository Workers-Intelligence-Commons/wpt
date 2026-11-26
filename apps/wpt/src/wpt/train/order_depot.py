"""The depot an order of the legacy dataset shape names, read without
writing to it: the data part names the depot by its public address and a
dataset of it (by id or name). An order of a snapshot reads no depot: its
corpus is a delivery on the dock (wpt.resources).

WPT never opens a depot's file: the depot is read through its web app,
at the address the data part names when it is a web address (http:// or
https://), with the frozen dataset and tokenizer routes `depot web` serves
(docs/depot-web.md, "Models trained before snapshots").

An order of a dataset finds the latest fresh corpus of one dataset,
vocabulary and size (wpt_corpus.corpora.triple), built from any revision
of the dataset: the WebDepot reads it from the listing of the dataset's
downloads, which names it for each triple, and reads that corpus's record
from the dock.

    opened(data)   the depot, as a WebDepot; Unreadable when it cannot be read
    download_page(address, dataset)   the address of the dataset's listing of corpora in the depot's web app
"""

import json
import urllib.error
from pathlib import Path
from urllib.parse import quote

from wpt_corpus import corpora
from wic_utils import web
from wpt_tokenizer import Tokenizer

TIMEOUT = 100  # seconds a request to a depot's web app may take


class Unreadable(Exception):
    """The depot cannot be read here: it is not named by a web address, its
    web app cannot be reached, or what it answered cannot be read."""


class Missing(Exception):
    """The depot holds no dataset or tokenizer recipe of that id or name."""


def opened(data: dict, env_file: Path | None = None):
    """The depot data["depot"] names, for its dataset data["dataset"]: a
    WebDepot for a web address. Unreadable for any other address."""
    if data["depot"].startswith(("http://", "https://")):
        return WebDepot(data["depot"])
    raise Unreadable(f"the depot {data['depot']!r} cannot be read here: an order of a dataset names its depot by "
                     "the web address of its web app, or the corpus is given by its id")


def download_page(address: str, dataset: str) -> str:
    """The address of the listing of the dataset's corpora (its id) in the web app of the depot named address: the
    frozen route of the legacy datasets (depot_app.web.legacy), since the web app has no page of datasets."""
    return f"{address.rstrip('/')}/api/datasets/{quote(dataset, safe='')}/download"


class WebDepot:
    """A depot read through its web app at the web address url."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _get(self, route: str) -> bytes:
        try:
            with web.urlopen(f"{self.url}{route}", timeout=TIMEOUT) as answer:
                return answer.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise Missing(f"{self.url} holds nothing at {route}") from None
            raise Unreadable(f"{self.url} answered {error.code} to {route}") from None
        except OSError as error:
            raise Unreadable(f"{self.url} cannot be reached: {error}") from None

    def _json(self, route: str):
        try:
            return json.loads(self._get(route))
        except ValueError as error:
            raise Unreadable(f"{self.url} answered {route} with what is not JSON: {error}") from None

    def dataset(self, dataset: str) -> dict:
        """The current row of the dataset of that id or name. Missing without one."""
        found = [row for row in self._json("/api/datasets") if dataset in (row.get("id"), row.get("name"))]
        if not found:
            raise Missing(f"no dataset {dataset!r} at {self.url}")
        return next((row for row in found if row.get("id") == dataset), found[0])

    def built(self, store, address: str, dataset: dict, vocabulary: str, size: int | None) -> dict | None:
        """The record of the latest fresh corpus of the dataset (its row),
        the vocabulary of that sha256 and that size, through the listing of
        the dataset's downloads: the record, read from the dock, of the
        corpus it names as the triple's latest fresh one; None when it names
        none.
        Unreadable when the listing cannot be made (an erasure being
        checked, no dock) or the dock does not hold the corpus it names."""
        found = self._json(f"/api/datasets/{quote(dataset['id'], safe='')}/download")
        if found.get("state") != "listed":
            raise Unreadable(f"{self.url} cannot list the corpora of the dataset {dataset['id']} now: "
                             f"{found.get('state')} {found.get('problem') or found.get('erasures') or ''}".rstrip())
        named = next((one["latest_fresh"] for one in found["corpora"]
                      if one["vocabulary"]["sha256"] == vocabulary and one["max_tokens"] == size), None)
        if named is None:
            return None
        record = corpora.find(store, named["id"])
        if record is None:
            raise Unreadable(f"{self.url} names the corpus {named['id']} for that dataset, vocabulary and size, "
                             f"which the dock {store} does not hold")
        return record

    def vocab_size(self, recipe: str) -> tuple[int, bool]:
        """(the size of the vocabulary of the current version of the
        tokenizer recipe named so, or the size it asks for before it is
        trained; whether it is trained), through the frozen tokenizer routes
        (depot_app.web.legacy). Missing without the recipe."""
        name = quote(recipe, safe="")
        found = self._json(f"/api/tokenizers/{name}")
        version = found["recipe"]["version"]
        if not found.get("trained"):
            return found["recipe"]["vocab_size"], False
        merges = self._get(f"/api/tokenizers/{name}/vocabularies/{version}/file")
        return Tokenizer.from_bytes(merges).vocab_size, True
