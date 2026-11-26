"""The corpora the dock keeps under corpora/, made by the depot before
the freight delivered corpora: finding them and marking stale those an
erasure reached. Nothing writes a new one, and nothing here reads what a
corpus holds beyond the ids of its texts and the files its record names:
WPT reads the rest (wpt_corpus.corpora).

Each corpus is a folder corpora/<folder>/ of the dock (a corpus made
from a dataset before snapshots: corpora/<dataset name>-<8 hex
characters>/), its record corpus.json written last (folder_of). Its record
names each of its files as {key, sha256} (files), and its texts table
lists the ids of its texts. A record says `stale` once an admitted erasure
reached a text it holds (the ids of the erasures), and `deleted` once its
files are gone (when, and why). A deleted corpus keeps its record.

    records(dock)                        every corpus the dock holds, newest first
    files(record)                        every file a corpus's record names
    mark_stale(dock, texts, erasures)    mark every corpus that holds one of these texts stale
    holds(dock, record, texts)           whether a corpus holds one of these texts

The dock is named by its URL, and read and written through dock.api.
"""

import io

import pyarrow.parquet as pq
from dock import api

FOLDER = "corpora"
RECORD = "corpus.json"


def folder_of(record: dict) -> str:
    """The dock's folder of a corpus: the one its record names, or for a
    record that names none, corpora/<dataset name>-<first 8 hex characters of its id>."""
    return record.get("folder") or f"{FOLDER}/{api.as_record_name(record['dataset']['name'])}-{record['id'][:8]}"


def files(record: dict) -> list[dict]:
    """Every file of a corpus but its record, as the record names each: every
    {key, sha256} the record holds, at any depth, in the record's order
    (and maybe bytes)."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("key"), str) and isinstance(value.get("sha256"), str):
                found.append(value)
                return
            for one in value.values():
                walk(one)
        elif isinstance(value, list):
            for one in value:
                walk(one)
    walk(record)
    return found


def records(dock: str) -> list[dict]:
    """The record of every corpus the dock holds, each checked by the
    dock, the last built first."""
    found = []
    for name in api.folders(dock, f"{FOLDER}/"):
        if (record := api.get_json(dock, f"{FOLDER}/{name}/{RECORD}")) is not None:
            found.append(record)
    return sorted(found, key=lambda record: record["built_at"], reverse=True)


def mark_stale(dock: str, texts, erasures) -> list[str]:
    """Mark stale every corpus of the dock whose files are kept and that
    holds one of the texts (ids), because the erasures (ids) were carried
    out: the ids of the corpora marked. A corpus already stale keeps its
    earlier erasures. A deleted corpus is left as it is: its rebuild
    compares its id."""
    marked = []
    for record in records(dock):
        if "deleted" in record or not holds(dock, record, texts):
            continue
        marked_stale(dock, record, erasures)
        marked.append(record["id"])
    return marked


def holds(dock: str, record: dict, texts) -> bool:
    """Whether the corpus holds one of the texts (ids), as its texts table lists them."""
    wanted = set(texts)
    if not wanted:
        return False
    parts = record["tables"]["texts"]
    for part in parts if isinstance(parts, list) else [parts]:
        ids = pq.read_table(io.BytesIO(api.fetch(dock, part["key"], part["sha256"])), columns=["id"])["id"].to_pylist()
        if not wanted.isdisjoint(ids):
            return True
    return False


def marked_stale(dock: str, record: dict, erasures) -> dict:
    """The corpus's record marked stale because of the erasures (ids), with
    the erasures it was stale for already: the record as stored."""
    stale = sorted({*record.get("stale", {}).get("erasures", []), *erasures})
    return api.put_json(dock, f"{folder_of(record)}/{RECORD}", {**record, "stale": {"erasures": stale}})
