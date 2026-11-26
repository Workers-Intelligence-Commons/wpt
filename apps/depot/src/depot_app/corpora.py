"""The corpora the dock keeps under corpora/, as the depot's web app and
commands read them: the frozen dataset routes list and send those built
from a dataset (depot_app.web.legacy), and an upgrade from a format
before snapshots reads their records (depot.upgrade). The depot only reads
them: the freight marks stale those an erasure reached (freight.erasures).
The depot reaches the dock through dock.api alone (docs/architecture.md,
"The parts").

    records(store)                       every corpus the dock holds, newest first
    built_from(store, depot, dataset)    every corpus built for that depot's dataset (by id), latest first
    folder_of, fresh, by_kind, token_files, files, triple
                                         what a record says: its folder, fresh, its files, what it was built for
"""

from dock.api import as_record_name

FOLDER = "corpora"
RECORD = "corpus.json"
ANY = "any"  # a size that keeps the corpora of every size (built_from)


def folder_of(record: dict) -> str:
    """The dock's folder of a corpus: the one its record names, or for a
    record that names none, corpora/<dataset name>-<first 8 hex characters of its id>."""
    return record.get("folder") or f"{FOLDER}/{as_record_name(record['dataset']['name'])}-{record['id'][:8]}"


def fresh(record: dict) -> bool:
    return "stale" not in record and "deleted" not in record and by_kind(record)


def by_kind(record: dict) -> bool:
    """Whether the record holds its token files by kind ({kind: [token file]})."""
    return isinstance(record.get("shards", {}), dict)


def token_files(record: dict) -> list[dict]:
    """Every token file of a corpus, of every kind, as its record names each; a list of a record of another layout
    as it is."""
    shards = record.get("shards", {})
    return [one for kept in shards.values() for one in kept] if isinstance(shards, dict) else list(shards)


def files(record: dict) -> list[dict]:
    """Every file of a corpus but its record, as the record names each ({key, sha256} and maybe bytes), in the
    order the frozen dataset routes list them."""
    tables = record.get("tables", {})
    found = [*token_files(record), *record.get("conversation_shards", [])]
    for table in tables.values():
        found += table if isinstance(table, list) else [table]
    if "vocabulary" in record.get("tokenizer", {}):
        found.append(record["tokenizer"]["vocabulary"])
    if "tasks" in record:
        found.append(record["tasks"]["file"])
    return found


def triple(record: dict) -> tuple[str | None, str, int | None]:
    """What a corpus was built for: (its dataset's id, None for a corpus of
    a build key, the sha256 of its vocabulary, its size: max_tokens, None
    for none). A record that names no size of its own was built at its
    dataset's max_tokens, and one that names no vocabulary sha256 of its own
    has its vocabulary file's."""
    dataset = record.get("dataset") or {}
    size = record["max_tokens"] if "max_tokens" in record else dataset.get("max_tokens")
    vocabulary = record["tokenizer"].get("sha256") or record["tokenizer"]["vocabulary"]["sha256"]
    return dataset.get("id"), vocabulary, size


def records(store) -> list[dict]:
    """The record of every corpus the dock holds, each checked by the
    dock, the last built first."""
    found = []
    for key in store.kind.list(f"{FOLDER}/", deep=False):
        if key.endswith("/") and (record := store.get_json(f"{key}{RECORD}")) is not None:
            found.append(record)
    return sorted(found, key=lambda record: record["built_at"], reverse=True)


def built_from(store, depot: str, dataset: str, vocabulary: str | None = None, size=ANY) -> list[dict]:
    """The records of the corpora built from the dataset of id dataset of
    the depot of that name (depot.legacy.depot_name), latest first: those of the dataset's newest
    revision first, each revision's last built first, so a corpus of an
    older revision built again never comes before one of a newer revision.
    A dataset's corpora stay its own through a rename, and a new dataset
    given an old name has none. vocabulary (a sha256) and size (max_tokens,
    None for none) keep only the corpora built for them (triple); by
    default, every vocabulary and size."""
    found = [record for record in records(store)
             if record["depot"] == depot and (record.get("dataset") or {}).get("id") == dataset
             and (vocabulary is None or triple(record)[1] == vocabulary)
             and (size is ANY or triple(record)[2] == size)]
    return sorted(found, key=lambda record: (record["dataset"].get("revision", 0), record["built_at"]), reverse=True)

