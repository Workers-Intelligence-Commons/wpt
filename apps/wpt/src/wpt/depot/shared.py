"""What the two controllers of `wpt depot`, its subcommands and its web
routes, share: the depot WPT_DEPOT names, the words of the confirmation a
decision needs, rows made fit for JSON, the store and its cap for corpora,
the repository checkout WPT_REPOSITORY names, whose model recipes the web
app builds, a vocabulary trained from the depot's tokenizer recipes, a
corpus built from a snapshot, the check that an estimate a person confirmed
still holds, a recipe imported from a file, and an erasure carried out with
the corpora it makes stale."""

import json
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

from wpt_depot import Declaration, Depot, corpora, pull, snapshots
from wpt_depot.corpus import build, sample
from wpt_depot.errors import NotFound, Refused
from wpt_recipes import NotFound as NoRecipes, Recipes
from wpt_utils.programs import MissingTool, program
from wpt_depot.schema import ADMITTED
from wpt_depot.settings import depot_address, depot_url
from wpt_store import StoreSettingError, read_env_file, store_from_settings
from wpt_tokenizer import Tokenizer

# what a person confirms with every decision: the subcommands' --accept and the web pages' button say it
CONFIRMATION = ("I read the requirements, I accept them, and what I submit meets them. My name and email are "
                "recorded on this decision and are public; nobody verifies them.")
# the page the confirmation's "requirements" means
REQUIREMENTS = "https://github.com/Workers-Intelligence-Commons/wpt/blob/main/docs/requirements.md"
CAP = "WPT_CORPORA_CAP"  # the setting of the bytes the store's corpora may take together
UNITS = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
RECIPE_FIELDS = ("sample_bytes", "vocab_size")  # what a recipe file gives that the depot's recipe keeps as it is
REPOSITORY = "WPT_REPOSITORY"  # the setting of the repository checkout whose recipes the web app lists and builds
# how far a cost may pass the one a person confirmed and still hold (unchanged): its share, then its unit of slack
TOLERANCE = 0.1


class NoVocabulary(Exception):
    """The vocabulary a build names is not in the depot (not trained, or its
    tokenizer recipe is not there); nothing was built."""


class NoSample(Exception):
    """The depot has no snapshot to read the sample of, or the snapshot holds
    no plain text that fits the recipe's sample_bytes; nothing was trained."""


class BadSetting(Exception):
    """A setting holds what it cannot: nothing was done."""


class NoRepository(Exception):
    """WPT_REPOSITORY names no repository checkout holding data/recipes:
    no recipe can be read."""


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


def repository() -> Recipes:
    """The recipes of the repository checkout WPT_REPOSITORY names (its
    root folder). NoRepository when it is not set, or names no folder
    holding data/recipes."""
    found = setting(REPOSITORY)
    if found is None:
        raise NoRepository(f"{REPOSITORY} is not set: set it to the folder of a checkout of the WPT repository to "
                           "list its recipes")
    root = Path(found).expanduser().resolve()
    try:
        recipes = Recipes.find(root)
    except NoRecipes:
        recipes = None
    if recipes is None or recipes.root != root:
        raise NoRepository(f"{REPOSITORY} names no checkout of the WPT repository (no data/recipes in it)")
    return recipes


def checkout(recipes: Recipes) -> dict:
    """{commit: the commit the checkout of recipes is at, None when git
    cannot tell; dirty: whether it has changes not committed, None then}."""
    try:
        git = program("git")
        commit = subprocess.run([git, "-C", str(recipes.root), "rev-parse", "HEAD"], capture_output=True,
                                check=True).stdout.decode().strip()
        dirty = subprocess.run([git, "-C", str(recipes.root), "status", "--porcelain"], capture_output=True,
                               check=True).stdout.strip() != b""
    except (MissingTool, OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty if commit else None}


def snapshot_named(depot: Depot, address: str, name: str) -> dict:
    """The row of the snapshot called name of the depot a recipe names by
    address: this depot's own when address is its public address
    (WPT_DEPOT_URL, else its name), its upstream's when it is the address
    of the upstream it follows (wpt_depot.snapshots.owner_named). NotFound
    otherwise, or without such a snapshot."""
    owner = snapshots.owner_named(depot, address, public_url())
    if owner is None:
        raise NotFound(f"the recipe names the depot {address!r}, which is neither this depot "
                       f"({public_url() or depot.name()}) nor the upstream it follows")
    return depot.snapshot(name, owner)


def unchanged(shown, now: dict, costs: dict[str, float], same=()) -> list[str]:
    """What makes the estimate now differ from the one a person confirmed
    (shown, as the route gave it): each key of same whose value changed,
    and each cost of costs ({key: its slack, in its unit}) now above the
    shown one by more than TOLERANCE of it plus its slack; [] when the
    confirmed estimate still holds. A cost now lower always holds."""
    if not isinstance(shown, dict):
        return ["the estimate shown was not given back with the confirmation"]
    found = []
    for key in same:
        if shown.get(key) != now.get(key):
            found.append(f"{key} was {shown.get(key)!r} and is now {now.get(key)!r}")
    for key, slack in costs.items():
        was, then = shown.get(key), now.get(key)
        if then is None:
            continue
        if not isinstance(was, (int, float)) or isinstance(was, bool):
            found.append(f"the estimate shown gives no {key}")
        elif then > was * (1 + TOLERANCE) + slack:
            found.append(f"{key} was {was:,} and is now {then:,}")
    return found


def opened() -> Depot:
    """The depot WPT_DEPOT names, the change files a crash left pending kept
    or dropped (wpt_depot.database). NoDepotSetting without the setting;
    NotADepot, UnsupportedAddress for an address that holds no depot."""
    return Depot.open(depot_address(env_file=env_file()))


def public_url() -> str | None:
    """The depot's public address WPT_DEPOT_URL gives (wpt_depot.settings.depot_url); None without it."""
    return depot_url(env_file=env_file())


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
    log(f"{len(marked)} corpora marked stale: nothing downloads them, and wpt train refuses them, "
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


def vocabulary_by_sha256(depot: Depot, sha256: str) -> tuple[dict, int, Tokenizer]:
    """(the tokenizer recipe of depot's vocabulary of that sha256, the
    recipe's version it is of, its tokenizer). NoVocabulary when the depot
    holds no such vocabulary: it is not trained."""
    found = depot.rows("SELECT recipe, version, merges FROM vocabularies WHERE sha256 = ? ORDER BY recipe, version",
                       (sha256,))
    if not found:
        raise NoVocabulary(f"the depot holds no vocabulary of sha256 {sha256}: train it first, on its tokenizer "
                           "recipe's page or with `wpt depot tokenizer train`")
    return (depot.tokenizer_by_id(found[0]["recipe"]), found[0]["version"],
            Tokenizer.from_bytes(found[0]["merges"].encode("utf-8")))


def build_corpus(depot: Depot, snapshot: str | None, data: dict | None, log, *, vocabulary: str | None = None,
                 store=None, record: dict | None = None, work: Path | None = None, checkpoint: dict | None = None,
                 save=lambda state: None, progress=lambda state: None, room=None, recipe: str | None = None,
                 code_commit: str | None = None) -> dict:
    """Build the corpus of depot's snapshot of that id with the data
    section data (its kinds, shares and total_tokens,
    wpt_depot.corpus.canonical; its vocabulary, when it names none, the
    one given) and the vocabulary of that sha256, into store (default: the
    store WPT_STORE names): the corpus's record (wpt_depot.corpus.build,
    which says what work, checkpoint, save, progress and room do). With
    record, a corpus built before: its snapshot, data section and
    vocabulary again (snapshot, data and vocabulary unread), and the record
    of the build, which says `changed` when its id is not record's (an
    erasure carried out since changed what it holds). recipe and
    code_commit: the model recipe's name and the code's commit the record
    names. The record names the depot data names (the recipe's, which on
    a clone is its upstream), else this depot's public address or name.
    NoVocabulary without that vocabulary; the depot's and the
    store's failures as they are."""
    store = store if store is not None else store_from_settings(env_file())
    if record is not None:
        snapshot, data, vocabulary = record["snapshot"]["id"], record["data"], record["vocabulary"]["sha256"]
    if vocabulary is None:
        vocabulary = ((data or {}).get("vocabulary") or {}).get("sha256")
    if vocabulary is None:
        raise NoVocabulary("a build names its vocabulary by its sha256: nothing was built")
    tokenizer_recipe, version, tokenizer = vocabulary_by_sha256(depot, vocabulary)
    data = {**(data or {})}
    data.setdefault("vocabulary", {"sha256": vocabulary, "size": tokenizer.vocab_size})
    # the record names the depot the recipe names, the upstream's address when a clone builds
    built = build(depot, snapshot, data, store, tokenizer, address=data.get("depot") or public_url() or depot.name(),
                  recipe=recipe, code_commit=code_commit, tokenizer_recipe=tokenizer_recipe["name"],
                  tokenizer_version=version,
                  work=work, checkpoint=checkpoint, save=save, progress=progress, room=room, log=log)
    if record is not None and built["id"] != record["id"]:
        why = (f"corpus {record['id']} was built again as {built['id']}: an erasure carried out since changed what "
               "it holds")
        log(why)
        return {**built, "changed": why}
    return built


def sample_of(depot: Depot, recipe: dict) -> Iterator[tuple[str, str, bytes]]:
    """The sample a tokenizer recipe of depot trains on: (the depot
    vocabulary_texts keeps, '' for depot; the text's id; its UTF-8 bytes),
    read in one pass, from the plain texts of the snapshot its sample names
    (wpt_depot.corpus.sample). NoSample for a recipe that names no sample
    (needs_snapshot: a revision written before snapshots, which named a
    dataset)."""
    if recipe["needs_snapshot"]:
        raise NoSample(f"the tokenizer recipe {recipe['name']} names a dataset, written before snapshots: an edit "
                       "naming a snapshot gives it a sample; nothing was trained")
    yield from sample(depot, recipe["sample"]["snapshot"], recipe["sample_bytes"])


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
        listed_once = set()  # a long text's pieces come apart in the sample: the list names the text once
        with open(listed, "w", encoding="utf-8") as held:
            def texts():
                for address, text, data in sample_of(depot, recipe):
                    if (address, text) not in listed_once:
                        listed_once.add((address, text))
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
            raise NoSample(f"the snapshot the tokenizer recipe {name} reads holds no plain text within "
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


def import_tokenizer(depot: Depot, name: str, given: dict, snapshot: str, kinds: list[str] | None = None,
                     lock: dict | None = None, vocabulary: bytes | None = None, store=None) -> dict:
    """Make depot's tokenizer recipe named so from a recipe file's fields
    (given: its sample_bytes and vocab_size, and its note when it has one),
    its sample read from the texts of those kinds (plain by default) of
    depot's snapshot (its id or the name of one of depot's own), and keep the vocabulary a
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
    recipe = depot.new_tokenizer(name, snapshot, kinds, given["sample_bytes"], given["vocab_size"], given.get("note"))
    sha = None
    if vocabulary is not None:
        sha = depot.add_vocabulary(recipe, 1, vocabulary, texts, None)
    return {"recipe": recipe, "vocabulary": sha}
