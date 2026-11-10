"""What the two controllers of `wpt depot`, its subcommands and its web
routes, share: the depot WPT_DEPOT names, the words of the confirmation a
decision needs, rows made fit for JSON, a dataset's corpus built with
the vocabulary its tokenizer recipe locked, the corpora built from a
dataset, that vocabulary trained, and an erasure carried out with the
corpora it makes stale."""

from datetime import date, datetime
from pathlib import Path

from wpt_depot import Declaration, Depot, pull
from wpt_depot.corpus import build, built_from, mark_stale, sample
from wpt_depot.errors import NoBucketSetting
from wpt_depot.settings import bucket_settings, depot_address
from wpt_recipes import InvalidRecipe, Recipes
from wpt_store import StoreError, StoreSettingError, store_from_settings
from wpt_tokenizer import Tokenizer

# what a person confirms with every decision: the subcommands' --accept and the web pages' button say it
CONFIRMATION = ("I read the requirements, I accept them, and what I submit meets them. My name and email are "
                "recorded on this decision and are public; nobody verifies them.")
# the page the confirmation's "requirements" means
REQUIREMENTS = "https://github.com/Workers-Intelligence-Commons/wpt/blob/main/docs/requirements.md"


class NoVocabulary(Exception):
    """The dataset's tokenizer recipe has no lock of its version, or the
    store does not hold the vocabulary its lock names; nothing was built."""


class TokenizerLocked(Exception):
    """The tokenizer recipe's lock is of the recipe's version already: a
    locked vocabulary is never trained again; nothing was trained."""


class NoSample(Exception):
    """The recipe's dataset holds no plain text that fits its sample_bytes;
    nothing was trained."""


def env_file() -> Path:
    """The .env file the settings fall back on: the current folder's."""
    return Path.cwd() / ".env"


def bucket() -> dict | None:
    """The settings of depots in buckets (WPT_DEPOT_*, environment or .env);
    None when they are not set."""
    try:
        return bucket_settings(env_file=env_file())
    except NoBucketSetting:
        return None


def opened() -> Depot:
    """The depot WPT_DEPOT names, its unfinished journal changes applied.
    NoDepotSetting without the setting; NotADepot, UnsupportedAddress for
    an address that holds no depot."""
    return Depot.open(depot_address(env_file=env_file()), bucket())


def viewed(address: str) -> Depot:
    """The depot at address, to read."""
    return Depot.view(address, bucket())


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
        return built_from(store_from_settings(env_file()), depot.address, dataset), None
    except (StoreSettingError, StoreError, OSError) as error:
        return None, f"{type(error).__name__}: {error}"


def decide_erasures(depot: Depot, requests: list[str], status: str, declaration: Declaration, reason: str | None,
                    log) -> dict[str, bool]:
    """depot.decide_erasures, then every corpus of the store that holds a
    text the carried-out erasures cut or removed marked stale (said in the
    log): whether each was carried out, by id. The store is found before
    any erasure is carried out, since carrying one out cannot be undone."""
    store = corpora_store(log)
    carried_out = depot.decide_erasures(requests, status, declaration, reason)
    done = [request for request, carried in carried_out.items() if carried]
    if done:
        stale(depot, store, done, log)
    return carried_out


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
    texts = sorted({needle["text"] for row in depot.erasures().to_pylist() if row["id"] in erasures
                    for needle in row["needles"]})
    marked = mark_stale(store, texts, erasures)
    log(f"{len(marked)} corpora marked stale: wpt train refuses them until their datasets are built again"
        if marked else "no corpus of the store holds a text of the erasure")


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


def build_corpus(depot: Depot, dataset: str, log) -> dict:
    """Build the corpus of depot's dataset into the store WPT_STORE names,
    with the vocabulary the lock of the dataset's tokenizer recipe names
    (the repository's recipes, found from the current folder): the corpus's
    record (wpt_depot.corpus.build). NoVocabulary without that lock or that
    vocabulary; the recipes' and the store's failures as they are."""
    name = depot.dataset(dataset)["tokenizer"]
    recipes = Recipes.find()
    recipe, lock = recipes.recipe("tokenizer", name), recipes.generated("lock", name)
    if lock is None or lock["recipe"].get("version") != recipe["version"]:
        raise NoVocabulary(f"the tokenizer recipe {name} has no lock of its version {recipe['version']}")
    store = store_from_settings(env_file())
    vocabulary = lock["vocabulary_sha256"]
    if not store.has(vocabulary):
        raise NoVocabulary(f"the store {store} does not hold the vocabulary {vocabulary} that the lock of the "
                           f"tokenizer recipe {name} names")
    return build(depot, dataset, store, Tokenizer.from_bytes(store.get_bytes(vocabulary)), recipe["version"], log=log)


def train_vocabulary(name: str, log) -> dict:
    """Train the vocabulary of the tokenizer recipe name (the repository's
    recipes, found from the current folder) on the plain texts of its
    dataset in its depot, up to its sample_bytes (wpt_depot.corpus.sample),
    put it in the store WPT_STORE names, and write the recipe's lock: the
    lock. InvalidRecipe for a recipe whose depot or dataset is empty;
    TokenizerLocked when the lock is of the recipe's version already;
    NoSample when the sample is empty; the depot's, the recipes' and the
    store's failures as they are."""
    recipes = Recipes.find()
    recipe = recipes.recipe("tokenizer", name)
    if found := recipes.problems("tokenizer", name):
        raise InvalidRecipe(found[0])
    lock = recipes.generated("lock", name)
    if lock is not None and lock["recipe"].get("version") == recipe["version"]:
        raise TokenizerLocked(f"the tokenizer recipe {name} is locked at its version {recipe['version']} already: "
                              "raise the version to train a new vocabulary")
    store = store_from_settings(env_file())
    texts = sample(viewed(recipe["depot"]), recipe["dataset"], recipe["sample_bytes"])
    if not texts:
        raise NoSample(f"the dataset {recipe['dataset']} of {recipe['depot']} holds no plain text within "
                       f"{recipe['sample_bytes']:,} bytes")
    log(f"training the vocabulary of {name} version {recipe['version']} on {len(texts):,} texts "
        f"(vocab_size {recipe['vocab_size']:,})")
    tokenizer = Tokenizer.train((one["text"] for one in texts), recipe["vocab_size"])
    lock = {"recipe": recipe, "texts": [{"depot": one["depot"], "id": one["id"]} for one in texts],
            "vocabulary_sha256": store.put_bytes(tokenizer.to_bytes())}
    log(f"wrote {recipes.write_generated('lock', name, lock)}: commit it so every machine uses this vocabulary, "
        "and keep the store that holds it")
    return lock
