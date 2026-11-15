"""The depot a recipe's data part names, read without writing to it: the
data part names the depot by its public address, as a corpus records it
(wpt_depot.corpus.depot_name), and its dataset by id or name.

The depot is read from its file when WPT_DEPOT (environment or .env) names
a depot that the address names (the depot's name in its corpora is that
address), else through its web app when the address is a web address
(http:// or https://), with the routes `wpt depot web` serves.

Each reader finds the corpus a recipe trains on (built): the latest fresh
corpus of one dataset, vocabulary and size (wpt_depot.corpora.triple),
built from any revision of the dataset. A FileDepot looks for it in the
store; a WebDepot reads it from the listing of the dataset's downloads,
which names it for each triple (latest_fresh), and reads that corpus's
record from the store.

    opened(data)   the depot, as one of the two readers below; Unreadable when neither reads it
    download_page(address, dataset)   the address of the dataset's download page in the depot's web app
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from wpt_depot import Depot, corpora
from wpt_depot.corpus import depot_name
from wpt_depot.errors import DepotError, NotFound
from wpt_depot.settings import depot_address, depot_url
from wpt_tokenizer import Tokenizer

TIMEOUT = 100  # seconds a request to a depot's web app may take


class Unreadable(Exception):
    """The depot cannot be read here: no WPT_DEPOT names it, its web app
    cannot be reached, or what it answered cannot be read."""


class Missing(Exception):
    """The depot holds no dataset, or no tokenizer recipe, of that id or name."""


def opened(data: dict, env_file: Path | None = None):
    """The depot data["depot"] names: a FileDepot when WPT_DEPOT names a
    depot whose corpora of data["dataset"] name it so, else a WebDepot for a
    web address. Unreadable when neither reads it."""
    try:
        address = depot_address(env_file=env_file)
    except DepotError:
        address = None
    if address is not None:
        try:
            local = FileDepot(Depot.view(address), depot_url(env_file=env_file))
            if local.names(data["depot"], data["dataset"]):
                return local
        except (DepotError, OSError):
            pass
    if data["depot"].startswith(("http://", "https://")):
        return WebDepot(data["depot"])
    raise Unreadable(f"the depot {data['depot']!r} cannot be read here: set WPT_DEPOT to its file (and WPT_DEPOT_URL "
                     "to its public address when it has one), or name it by the web address of its web app")


def download_page(address: str, dataset: str) -> str:
    """The address of the download page of the dataset (its id) in the web app of the depot named address."""
    return f"{address.rstrip('/')}/#/download/{quote(dataset, safe='')}"


class FileDepot:
    """A depot read from its file; url: its public address, None without one."""

    def __init__(self, depot: Depot, url: str | None):
        self.depot, self.url = depot, url

    def names(self, address: str, dataset: str) -> bool:
        """Whether the corpora of the dataset (its id or name) name this
        depot address; of a dataset it does not hold, whether its own corpora would."""
        try:
            row = self.dataset(dataset)
        except Missing:
            return (self.url or self.depot.name()) == address
        return depot_name(self.depot, row, self.url) == address

    def dataset(self, dataset: str) -> dict:
        """The current row of the dataset of that id or name. Missing without one."""
        for find in (self.depot.dataset_by_id, self.depot.dataset):
            try:
                return find(dataset)
            except NotFound:
                continue
        raise Missing(f"no dataset {dataset!r}")

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
            with urllib.request.urlopen(f"{self.url}{route}", timeout=TIMEOUT) as answer:
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
