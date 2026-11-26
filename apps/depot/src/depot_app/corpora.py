"""The corpora the dock keeps under corpora/, as the depot's web app and
commands read them: the frozen dataset routes list and send those built
from a dataset (depot_app.web.legacy), an upgrade from a format before
snapshots reads their records (depot.upgrade), and an erasure carried out
marks stale those that hold a text it reached. The freight keeps the same
reading of them for WPT (freight.corpora); the depot reaches the dock
through dock.api alone (docs/architecture.md, "The parts").

    records(store)                       every corpus the dock holds, newest first
    built_from(store, depot, dataset)    every corpus built for that depot's dataset (by id), latest first
    folder_of, fresh, by_kind, token_files, files, triple
                                         what a record says: its folder, fresh, its files, what it was built for
    read_table(store, record, table)     one of a corpus's tables
    mark_stale(store, texts, erasures)   mark every corpus that holds one of these texts stale
"""

import pyarrow as pa
import pyarrow.parquet as pq
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
    """Every file of a corpus but its record, as the record names each ({key, sha256} and maybe bytes)."""
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


def read_table(store, record: dict, table: str) -> pa.Table:
    """One of a corpus's tables (texts, licenses, notices), each
    file checked by the dock: the texts table is the concatenation of its
    parts, in order."""
    found = record["tables"][table]
    if isinstance(found, list):
        return pa.concat_tables([pq.read_table(store.path(one)) for one in found]) if found else pa.table({})
    return pq.read_table(store.path(found))


def mark_stale(store, texts, erasures) -> list[str]:
    """Mark stale every corpus of the dock whose files are kept and that
    holds one of the texts (ids), because the erasures (ids) were carried
    out: the ids of the corpora marked. A corpus already stale keeps its
    earlier erasures. A deleted corpus is left as it is: its rebuild
    compares its id."""
    marked = []
    for record in records(store):
        if "deleted" in record or not holds(store, record, texts):
            continue
        marked_stale(store, record, erasures)
        marked.append(record["id"])
    return marked


def holds(store, record: dict, texts) -> bool:
    """Whether the corpus holds one of the texts (ids), as its texts table lists them."""
    wanted = set(texts)
    if not wanted:
        return False
    parts = record["tables"]["texts"]
    for part in parts if isinstance(parts, list) else [parts]:
        ids = pq.read_table(store.path(part), columns=["id"])["id"].to_pylist()
        store.release(part)
        if not wanted.isdisjoint(ids):
            return True
    return False


def marked_stale(store, record: dict, erasures) -> dict:
    """The corpus's record marked stale because of the erasures (ids), with
    the erasures it was stale for already: the record as stored."""
    stale = sorted({*record.get("stale", {}).get("erasures", []), *erasures})
    return store.put_json(f"{folder_of(record)}/{RECORD}", {**record, "stale": {"erasures": stale}})
