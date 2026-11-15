"""What the two controllers of `wpt depot`, its subcommands and its web
routes, share: the depot WPT_DEPOT names, the words of the confirmation a
decision needs, rows made fit for JSON, the store and its cap for corpora,
a vocabulary trained and a corpus built from the depot's tokenizer recipes,
a recipe imported from a file, the corpora built from a dataset, and an
erasure carried out with the corpora it makes stale."""

import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

from wpt_depot import Declaration, Depot, corpora, pull
from wpt_depot.corpus import build, depot_name, remote_sample, sample
from wpt_depot.errors import NotFound, Refused
from wpt_depot.schema import ADMITTED
from wpt_depot.settings import depot_address, depot_url
from wpt_store import StoreError, StoreSettingError, read_env_file, store_from_settings
from wpt_tokenizer import Tokenizer

# what a person confirms with every decision: the subcommands' --accept and the web pages' button say it
CONFIRMATION = ("I read the requirements, I accept them, and what I submit meets them. My name and email are "
                "recorded on this decision and are public; nobody verifies them.")
# the page the confirmation's "requirements" means
REQUIREMENTS = "https://github.com/Workers-Intelligence-Commons/wpt/blob/main/docs/requirements.md"
CAP = "WPT_CORPORA_CAP"  # the setting of the bytes the store's corpora may take together
UNITS = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
RECIPE_FIELDS = ("sample_bytes", "vocab_size")  # what a recipe file gives that the depot's recipe keeps as it is


class NoVocabulary(Exception):
    """The dataset's tokenizer recipe is not in the depot, or has no
    vocabulary of its current version; nothing was built."""


class NoSample(Exception):
    """The recipe's dataset holds no plain text that fits its sample_bytes;
    nothing was trained."""


class BadSetting(Exception):
    """A setting holds what it cannot: nothing was done."""


def env_file() -> Path:
    """The .env file the settings fall back on: the current folder's."""
    return Path.cwd() / ".env"


def setting(name: str) -> str | None:
    """The setting of that name: the environment's, else the .env file's; None when neither sets it."""
    return os.environ.get(name) or read_env_file(env_file()).get(name) or None


def corpora_cap() -> int | None:
    """The bytes WPT_CORPORA_CAP lets the store's corpora take together (a
    number of bytes, or one with a unit, KB, MB, GB or TB); None, no cap,
    when it is not set. BadSetting for another value."""
    found = setting(CAP)
    if found is None:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?B?)\s*", found.upper())
    if not match:
        raise BadSetting(f"{CAP}={found!r}: give bytes, or a number with KB, MB, GB or TB (3TB)")
    return int(float(match[1]) * UNITS[match[2]])


def opened() -> Depot:
    """The depot WPT_DEPOT names, the change files a crash left pending kept
    or dropped (wpt_depot.database). NoDepotSetting without the setting;
    NotADepot, UnsupportedAddress for an address that holds no depot."""
    return Depot.open(depot_address(env_file=env_file()))


def public_url() -> str | None:
    """The depot's public address WPT_DEPOT_URL gives (wpt_depot.settings.depot_url); None without it."""
    return depot_url(env_file=env_file())


def corpus_depot(depot: Depot, dataset: dict) -> str:
    """The name the corpora of depot's dataset (its row) record for depot
    (wpt_depot.corpus.depot_name), with the public address WPT_DEPOT_URL gives."""
    return depot_name(depot, dataset, public_url())


def viewed(address: str) -> Depot:
    """The depot at address, to read."""
    return Depot.view(address)


def corpora_store(log):
    """The store WPT_STORE names, where the corpora are; None, said in the
    log, when it is not set: no corpus of this machine can be marked."""
    try:
        return store_from_settings(env_file())
    except StoreSettingError as error:
        log(f"no corpus is marked stale: {error}")
        return None


def corpora_of(depot: Depot, dataset: str) -> tuple[list[dict] | None, str | None]:
    """(the records of the corpora the store WPT_STORE names built from
    depot's dataset of that id, the last built first, through its renames,
    or None when the store cannot be read; why it cannot, or None)."""
    try:
        return corpora.built_from(store_from_settings(env_file()), corpus_depot(depot, depot.dataset_by_id(dataset)),
                                  dataset), None
    except (StoreSettingError, StoreError, OSError) as error:
        return None, f"{type(error).__name__}: {error}"


def decide_erasures(depot: Depot, requests: list[str], status: str, declaration: Declaration, reason: str | None,
                    log, downloads=None) -> dict[str, bool]:
    """depot.decide_erasures, then every corpus of the store that holds a
    text the carried-out erasures cut or removed marked stale (said in the
    log): whether each was carried out, by id. The store is found before
    any erasure is carried out, since carrying one out cannot be undone.
    With downloads (the web app's wpt.depot.downloads.Downloads), an
    admission is kept as being marked before it is carried out, and the
    marking runs in the background (settle_marks)."""
    store = corpora_store(log)
    if downloads is None:
        carried_out = depot.decide_erasures(requests, status, declaration, reason)
        done = [request for request, carried in carried_out.items() if carried]
        if done:
            stale(depot, store, done, log)
        return carried_out
    if status == ADMITTED:
        downloads.marking(requests)
    try:
        return depot.decide_erasures(requests, status, declaration, reason)
    finally:
        settle_marks(depot, downloads, log)


def settle_marks(depot: Depot, downloads, log) -> threading.Thread | None:
    """The erasures downloads keeps as being marked: those never carried out
    leave it at once; the corpora holding a text of the others are marked
    stale in a thread of their own, which takes them out of it once done
    (a failure is said in the log, and they stay until the next call, at
    the web app's next start or next erasure decided): that thread, None
    without one."""
    pending = downloads.pending()
    if not pending:
        return None
    carried = {row["id"] for row in depot.erasures() if row["id"] in pending and row["erased"] is not None}
    downloads.marked([one for one in pending if one not in carried])
    if not carried:
        return None

    def mark():
        try:
            stale(depot, corpora_store(log), sorted(carried), log)
        except Exception as error:
            log(f"the corpora holding a text of the erasures {', '.join(sorted(carried))} are not marked stale yet "
                f"({type(error).__name__}: {error}): no corpus is sent until they are")
            return
        downloads.marked(sorted(carried))
    thread = threading.Thread(target=mark, name="marking", daemon=True)
    thread.start()
    return thread


def pulled(depot: Depot, log) -> list[str]:
    """pull(depot), then every corpus of the store holding a text the erasures
    it carried out cut or removed marked stale (said in the log): the
    erasures."""
    store = corpora_store(log)
    brought = pull(depot)
    if brought:
        stale(depot, store, brought, log)
    return brought


def stale(depot: Depot, store, erasures: list[str], log):
    if store is None:
        return
    texts = sorted({needle["text"] for row in depot.erasures() if row["id"] in erasures
                    for needle in row["needles"]})
    marked = corpora.mark_stale(store, texts, erasures)
    log(f"{len(marked)} corpora marked stale: their datasets give nothing to download, and wpt train refuses them, "
        "until they are built again" if marked else "no corpus of the store holds a text of the erasure")


def jsonable(value):
    """value with every time as ISO 8601 text and every byte string as
    hexadecimal text, the rest as it is: what a row of the depot's tables
    becomes in JSON."""
    if isinstance(value, dict):
        return {key: jsonable(one) for key, one in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(one) for one in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return value


def vocabulary_of(depot: Depot, name: str, version: int | None = None) -> tuple[dict, Tokenizer]:
    """(the depot's tokenizer recipe named so, the tokenizer of the
    vocabulary of its version, its current one by default). NoVocabulary
    when the depot holds no such recipe, or no vocabulary of that version."""
    try:
        recipe = depot.tokenizer(name)
    except NotFound:
        raise NoVocabulary(f"the depot holds no tokenizer recipe {name}: make it, or import it with "
                           "`wpt depot tokenizer import`") from None
    version = recipe["version"] if version is None else version
    found = depot.vocabulary(recipe["id"], version)
    if found is None:
        raise NoVocabulary(f"the tokenizer recipe {name} has no vocabulary of its version {version}: train it "
                           f"with `wpt depot tokenizer train {name}`")
    return recipe, Tokenizer.from_bytes(found["merges"].encode("utf-8"))


def build_corpus(depot: Depot, dataset: str, log, *, store=None, record: dict | None = None, work: Path | None = None,
                 checkpoint: dict | None = None, save=lambda state: None, progress=lambda state: None,
                 room=None) -> dict:
    """Build the corpus of depot's dataset of that id, as it is now, with the
    vocabulary of the current version of its tokenizer recipe, into store
    (default: the store WPT_STORE names): the corpus's record
    (wpt_depot.corpus.build, which says what work, checkpoint, save,
    progress and room do). With record, a corpus built before: the same
    dataset row and vocabulary again, and the record of the build, which
    says `changed` when its id is not record's (an erasure carried out
    since changed what it holds). NoVocabulary without that vocabulary; the
    depot's and the store's failures as they are."""
    store = store if store is not None else store_from_settings(env_file())
    if record is None:
        row = depot.dataset_by_id(dataset)
        recipe, tokenizer = vocabulary_of(depot, row["tokenizer"])
        version = recipe["version"]
    else:
        row, version = record["dataset"], record["tokenizer"]["recipe_version"]
        recipe, tokenizer = vocabulary_of(depot, record["tokenizer"]["recipe"], version)
    built = build(depot, row["name"], store, tokenizer, version, dataset=row, url=public_url(), work=work,
                  checkpoint=checkpoint, save=save, progress=progress, room=room, log=log)
    if record is not None and built["id"] != record["id"]:
        why = (f"corpus {record['id']} was built again as {built['id']}: an erasure carried out since changed what "
               "it holds")
        log(why)
        return {**built, "changed": why}
    return built


def sample_of(depot: Depot, recipe: dict, work: Path, progress=lambda state: None
              ) -> Iterator[tuple[str, str, bytes]]:
    """The sample of the dataset a tokenizer recipe of depot names: (the
    depot vocabulary_texts keeps, '' for depot, another local depot's name
    (wpt_depot.corpus.sample), or a depot's web address from its sample
    route; the text's id; its UTF-8 bytes),
    read in one pass, from depot (wpt_depot.corpus.sample) or through
    another depot's web app (wpt_depot.corpus.remote_sample)."""
    if recipe["depot"] is None:
        name = depot.dataset_by_id(recipe["dataset"])["name"]
        yield from sample(depot, name, recipe["sample_bytes"], work, progress)
    else:
        yield from remote_sample(recipe["depot"], recipe["dataset"], recipe["sample_bytes"], progress)


def train_vocabulary(depot: Depot, name: str, log, *, work: Path | None = None, progress=lambda state: None) -> str:
    """Train the vocabulary of the current version of depot's tokenizer
    recipe named so, on its sample (sample_of) read once, and keep it in
    the depot with the list of the texts it trained on: its sha256. The
    list waits in a file of work (default: a temporary folder) while the
    vocabulary trains. Refused when that version has a vocabulary already
    (the depot refuses a second one); NoSample when the sample is empty;
    the depot's failures as they are."""
    recipe = depot.tokenizer(name)
    if depot.vocabulary(recipe["id"], recipe["version"]) is not None:
        raise Refused(f"the tokenizer recipe {name} has a vocabulary of its version {recipe['version']} already: a "
                      "change of what trains it makes a new version")
    with tempfile.TemporaryDirectory(prefix="wpt-vocabulary-", dir=work) as folder:
        listed = Path(folder) / "texts.jsonl"
        counted = {"texts": 0, "bytes": 0}
        with open(listed, "w", encoding="utf-8") as held:
            def texts():
                for address, text, data in sample_of(depot, recipe, Path(folder) / "order", progress):
                    held.write(json.dumps([address, text]) + "\n")
                    counted["texts"] += 1
                    counted["bytes"] += len(data)
                    if counted["texts"] % 10_000 == 0:
                        progress({"stage": "reading the sample", "texts": counted["texts"],
                                  "bytes": counted["bytes"], "of": recipe["sample_bytes"]})
                    yield data
            log(f"training the vocabulary of {name} version {recipe['version']} ({recipe['vocab_size']:,} tokens) "
                f"on a sample of up to {recipe['sample_bytes']:,} bytes")
            tokenizer = Tokenizer.train(texts(), recipe["vocab_size"],
                                        merged=lambda done: progress({"stage": "merging", "merges": done,
                                                                      "of": recipe["vocab_size"]}))
        if not counted["texts"]:
            raise NoSample(f"the dataset of the tokenizer recipe {name} holds no plain text within "
                           f"{recipe['sample_bytes']:,} bytes")

        def read_back():
            with open(listed, encoding="utf-8") as held:
                for line in held:
                    address, text = json.loads(line)
                    yield address, text
        progress({"stage": "keeping", "texts": counted["texts"]})
        sha = depot.add_vocabulary(recipe["id"], recipe["version"], tokenizer.to_bytes(), read_back(),
                                   counted["bytes"])
    log(f"vocabulary {sha} of {name} version {recipe['version']}: {tokenizer.vocab_size:,} tokens, trained on "
        f"{counted['texts']:,} texts ({counted['bytes']:,} bytes)")
    return sha


def import_tokenizer(depot: Depot, name: str, given: dict, dataset: str, lock: dict | None = None,
                     vocabulary: bytes | None = None, store=None) -> dict:
    """Make depot's tokenizer recipe named so from a recipe file's fields
    (given: its sample_bytes and vocab_size, and its note when it has one),
    trained on depot's dataset (its name or id), and keep the vocabulary a
    lock file names (lock: its texts and vocabulary_sha256, the vocabulary
    read from store) or the vocabulary file given, as the vocabulary of its
    version 1: {recipe: its id, vocabulary: its sha256 or None}. Refused
    for a lock of another recipe's fields, and as new_tokenizer and
    add_vocabulary refuse; MissingFile when the store does not hold the
    lock's vocabulary."""
    missing = [field for field in RECIPE_FIELDS if field not in given]
    if missing:
        raise Refused(f"the recipe file gives no {', '.join(missing)}")
    if lock is not None and vocabulary is not None:
        raise Refused("give a lock or a vocabulary file, not both")
    if lock is not None and any(lock["recipe"].get(field) != given[field] for field in RECIPE_FIELDS):
        raise Refused("the lock was made for other sample_bytes or vocab_size than the recipe file's: its vocabulary "
                      "is not this recipe's")
    texts: list[tuple[str, str]] = []
    if lock is not None:
        vocabulary = store.get_bytes(lock["vocabulary_sha256"])
        texts = [("" if depot.is_here(one["depot"]) else one["depot"], one["id"]) for one in lock["texts"]]
    if vocabulary is not None:
        Tokenizer.from_bytes(vocabulary)  # a vocabulary file it cannot read is refused before anything is written
    recipe = depot.new_tokenizer(name, None, dataset, given["sample_bytes"], given["vocab_size"], given.get("note"))
    sha = None
    if vocabulary is not None:
        sha = depot.add_vocabulary(recipe, 1, vocabulary, texts, None)
    return {"recipe": recipe, "vocabulary": sha}
