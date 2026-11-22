"""The depot a recipe's data part names, read without writing to it: the
data part names the depot by its public address, and either a snapshot of
it (by name, with the change and digest it reads) or, in a recipe written
before snapshots, a dataset of it (by id or name).

The depot is read from its file when WPT_DEPOT (environment or .env) names
a depot that the address names, else through its web app when the address
is a web address (http:// or https://), with the routes `wpt depot web`
serves. For a snapshot, the file names it when the address is the depot's
own public address (WPT_DEPOT_URL, else its name) or the address of the
upstream it follows (a clone holds its upstream's snapshots); the snapshot
is then the one of that name whose owner is the depot the address names.
For a dataset, the file names it when the dataset's corpora name that
address (wpt_depot.legacy.depot_name).

A recipe of a snapshot finds its corpus by the build's key
(wpt_depot.corpus.build_key), which training works out from the snapshot's
row; the readers below only give that row (snapshot). A recipe of a
dataset finds the latest fresh corpus of one dataset, vocabulary and size
(wpt_depot.corpora.triple), built from any revision of the dataset: a
FileDepot looks for it in the store; a WebDepot reads it from the listing
of the dataset's downloads, which names it for each triple (latest_fresh),
and reads that corpus's record from the store.

    opened(data)   the depot, as one of the two readers below; Unreadable when neither reads it
    download_page(address, dataset)   the address of the dataset's listing of corpora in the depot's web app
"""

import json
import urllib.error
from pathlib import Path
from urllib.parse import quote

from wpt_depot import Depot, corpora, snapshots
from wpt_depot.legacy import Legacy, depot_name
from wpt_depot.errors import DepotError, NotFound
from wpt_depot.settings import depot_address, depot_url
from wpt_utils import web
from wpt_tokenizer import Tokenizer

TIMEOUT = 100  # seconds a request to a depot's web app may take


class Unreadable(Exception):
    """The depot cannot be read here: no WPT_DEPOT names it, its web app
    cannot be reached, or what it answered cannot be read."""


class Missing(Exception):
    """The depot holds no snapshot, dataset or tokenizer recipe of that id or name."""


def opened(data: dict, env_file: Path | None = None):
    """The depot data["depot"] names: a FileDepot when WPT_DEPOT names a
    depot that address names (for a snapshot: its own address or its
    upstream's; for a dataset: the one its corpora name it by), else a
    WebDepot for a web address. Unreadable when neither reads it."""
    try:
        address = depot_address(env_file=env_file)
    except DepotError:
        address = None
    if address is not None:
        try:
            local = FileDepot(Depot.view(address), depot_url(env_file=env_file))
            if (local.owner(data["depot"]) is not None if "snapshot" in data
                    else local.names(data["depot"], data["dataset"])):
                return local
        except (DepotError, OSError):
            pass
    if data["depot"].startswith(("http://", "https://")):
        return WebDepot(data["depot"])
    raise Unreadable(f"the depot {data['depot']!r} cannot be read here: set WPT_DEPOT to its file (and WPT_DEPOT_URL "
                     "to its public address when it has one), or name it by the web address of its web app")


def download_page(address: str, dataset: str) -> str:
    """The address of the listing of the dataset's corpora (its id) in the web app of the depot named address: the
    frozen route of the legacy datasets (wpt.depot.web.legacy), since the web app has no page of datasets."""
    return f"{address.rstrip('/')}/api/datasets/{quote(dataset, safe='')}/download"


class FileDepot:
    """A depot read from its file; url: its public address, None without one."""

    def __init__(self, depot: Depot, url: str | None):
        self.depot, self.url = depot, url

    def owner(self, address: str) -> str | None:
        """The name of the depot whose snapshots address names here: this
        depot's own for its public address (else its name), its upstream's
        for the address of the upstream it follows; None for another."""
        return snapshots.owner_named(self.depot, address, self.url)

    def snapshot(self, address: str, name: str) -> dict:
        """The row of the snapshot called name of the depot address names
        ({id, name, owner, at_change, digest, ...}). Missing without one."""
        owner = self.owner(address)
        if owner is None:
            raise Missing(f"the depot {address!r} is neither this one nor the upstream it follows")
        try:
            return self.depot.snapshot(name, owner)
        except NotFound:
            raise Missing(f"no snapshot {name!r} of the depot {address!r} ({owner})") from None

    def names(self, address: str, dataset: str) -> bool:
        """Whether the corpora of the dataset (its id or name) name this
        depot address; of a dataset it does not hold, whether its own corpora would."""
        try:
            row = self.dataset(dataset)
        except Missing:
            return (self.url or self.depot.name()) == address
        return depot_name(self.depot, row, self.url) == address

    def dataset(self, dataset: str) -> dict:
        """The current row of the dataset of that id or name, read from the
        legacy tables (wpt_depot.legacy). Missing without one."""
        try:
            return Legacy(self.depot.address).dataset_named(dataset)
        except NotFound:
            raise Missing(f"no dataset {dataset!r}") from None

    def built(self, store, address: str, dataset: dict, vocabulary: str, size: int | None) -> dict | None:
        """The record of the latest fresh corpus of the dataset (its row),
        the vocabulary of that sha256 and that size, among the corpora of
        the depot address names that the store holds; None without one."""
        return corpora.latest_fresh(store, address, dataset["id"], vocabulary, size)

    def vocab_size(self, recipe: str) -> tuple[int, bool]:
        """(the size of the vocabulary of the current version of the
        tokenizer recipe named so, or the size it asks for before it is
        trained; whether it is trained). Missing without the recipe."""
        try:
            found = self.depot.tokenizer(recipe)
        except NotFound:
            raise Missing(f"no tokenizer recipe {recipe!r}") from None
        trained = self.depot.vocabulary(found["id"], found["version"])
        if trained is None:
            return found["vocab_size"], False
        return Tokenizer.from_bytes(trained["merges"].encode("utf-8")).vocab_size, True


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

    def snapshot(self, address: str, name: str) -> dict:
        """As FileDepot.snapshot, through the listing of the snapshots
        (GET /api/snapshots, each row with its owner) and the depot's own
        name (GET /api/about): the depot at this web address owns the
        snapshots address names."""
        owner = self._json("/api/about")["about"]["name"]
        listed = self._json("/api/snapshots")
        rows = listed.get("snapshots", []) if isinstance(listed, dict) else listed
        found = [row for row in rows if row.get("owner") == owner and row.get("name") == name]
        if not found:
            raise Missing(f"no snapshot {name!r} of the depot {address!r} ({owner}) at {self.url}")
        return found[0]

    def dataset(self, dataset: str) -> dict:
        """The current row of the dataset of that id or name. Missing without one."""
        found = [row for row in self._json("/api/datasets") if dataset in (row.get("id"), row.get("name"))]
        if not found:
            raise Missing(f"no dataset {dataset!r} at {self.url}")
        return next((row for row in found if row.get("id") == dataset), found[0])

    def built(self, store, address: str, dataset: dict, vocabulary: str, size: int | None) -> dict | None:
        """As FileDepot.built, through the listing of the dataset's
        downloads: the record, read from the store, of the corpus it names
        as the triple's latest fresh one; None when it names none.
        Unreadable when the listing cannot be made (an erasure being
        checked, no store) or the store does not hold the corpus it names."""
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
                             f"which the store {store} does not hold")
        return record

    def vocab_size(self, recipe: str) -> tuple[int, bool]:
        """As FileDepot.vocab_size, through the tokenizer routes."""
        name = quote(recipe, safe="")
        found = self._json(f"/api/tokenizers/{name}")
        version = found["recipe"]["version"]
        if not found.get("trained"):
            return found["recipe"]["vocab_size"], False
        merges = self._get(f"/api/tokenizers/{name}/vocabularies/{version}/file")
        return Tokenizer.from_bytes(merges).vocab_size, True
