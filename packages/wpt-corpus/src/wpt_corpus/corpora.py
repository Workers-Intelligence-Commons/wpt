"""WPT's reading of the corpora the dock keeps under corpora/, built
before the freight delivered corpora: what each record says of its tokens
and vocabulary, and the corpus an order of the legacy shapes trains on.
Nothing here builds or changes one: the freight marks them stale after an
erasure (freight.corpora), and these readers only read the dock.

Each corpus is a folder corpora/<build key>/ of the dock (a corpus built
from a dataset before snapshots: corpora/<dataset name>-<8 hex
characters>/), its record corpus.json. A corpus is fresh while its record
says neither `stale` (an admitted erasure reached a text it holds: the ids
of the erasures) nor `deleted` (its files are gone: when, and why), and
holds its token files by kind (by_kind; a record of another layout is
never fresh: training cannot read it).

A corpus of a snapshot was built for a build key: its snapshot, its data
section in the form canonical gives, its vocabulary and the version of the
selection rules then, RULES. A corpus built from a dataset before snapshots
was built for a triple (triple): its dataset, its vocabulary (by sha256)
and its size; those are found by their dataset (built_from, latest_fresh),
and a corpus of a build key belongs to no dataset.

    records(dock)                        every corpus the dock holds, newest first
    find(dock, corpus_id)                the corpus of that id (or of a start of it), None without one
    canonical(data), RULES               a data section in the form a build key covers, and the rules' version
    fresh(record), by_kind(record)       whether a corpus can train, whether it lists its token files by kind
    triple(record)                       what a corpus was built for: (its dataset's id, its vocabulary's sha256,
                                         its size)
    built_from(dock, depot, dataset)     every corpus built for that depot's dataset (by id), latest first
    latest_fresh(dock, depot, dataset)   the fresh one built last
    rows(dock, record, table)            the rows of one of a corpus's tables, of the portions it trains on

dock is the dock's URL, read through dock.api.
"""

import io

import pyarrow as pa
import pyarrow.parquet as pq
from dock import api

FOLDER = "corpora"
RECORD = "corpus.json"
ANY = "any"  # a size that keeps the corpora of every size (built_from)
RULES = 8  # the version of the selection rules a corpus of a build key was built with, which its key covers


class Refused(ValueError):
    """A data section no corpus was built for."""


def fresh(record: dict) -> bool:
    return "stale" not in record and "deleted" not in record and by_kind(record)


def by_kind(record: dict) -> bool:
    """Whether the record holds its token files by kind ({kind: [token file]})."""
    return isinstance(record.get("shards", {}), dict)


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


def canonical(data: dict) -> dict:
    """What a data section says of its corpus, in the form a build key
    covers: {kinds: the kinds named, sorted, each once ([]: every kind),
    shares: [{tag, share}] in order ([]: none), total_tokens: a number of
    tokens above 0, or None}. Refused for a section no corpus was built
    for: kinds not a list of names, shares whose tags repeat or whose share
    is below 0 or not a number, or whose shares are all 0, a size that is
    not a whole number above 0."""
    kinds = data.get("kinds") or []
    if not isinstance(kinds, list) or not all(isinstance(one, str) and one for one in kinds):
        raise Refused(f"the data section's kinds are a list of kinds' names: {kinds!r}")
    shares = data.get("shares") or []
    if not isinstance(shares, list) or not all(isinstance(one, dict) and isinstance(one.get("tag"), str)
                                               and isinstance(one.get("share"), (int, float))
                                               and not isinstance(one.get("share"), bool) and one["share"] >= 0
                                               for one in shares):
        raise Refused(f"the data section's shares are a list of {{tag, share}}, each share a number of 0 or more: "
                      f"{shares!r}")
    tags = [one["tag"] for one in shares]
    if len(set(tags)) != len(tags):
        raise Refused(f"the data section's shares name a tag twice: {', '.join(tags)}")
    if shares and not sum(one["share"] for one in shares):
        raise Refused("the data section's shares are all 0: give one a share above 0, or give no shares")
    size = data.get("total_tokens")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise Refused(f"a corpus's size (total_tokens) is a whole number of tokens above 0, or none: {size!r}")
    return {"kinds": sorted(set(kinds)), "shares": [{"tag": one["tag"], "share": float(one["share"])}
                                                    for one in shares], "total_tokens": size}


def records(dock) -> list[dict]:
    """The record of every corpus the dock holds, each checked by the
    dock, the last built first."""
    found = []
    for name in api.folders(dock, f"{FOLDER}/"):
        if (record := api.get_json(dock, f"{FOLDER}/{name}/{RECORD}")) is not None:
            found.append(record)
    return sorted(found, key=lambda record: record["built_at"], reverse=True)


def find(dock, corpus_id: str) -> dict | None:
    """The record of the corpus whose id is corpus_id, or starts with it
    when only one does; None when none does. ValueError when several do."""
    found = [record for record in records(dock) if record["id"].startswith(corpus_id)]
    if len(found) > 1:
        raise ValueError(f"{corpus_id!r} starts the ids of {len(found)} corpora: give more of the id")
    return found[0] if found else None


def built_from(dock, depot: str, dataset: str, vocabulary: str | None = None, size=ANY) -> list[dict]:
    """The records of the corpora built from the dataset of id dataset of
    the depot of that name, latest first: those of the dataset's newest
    revision first, each revision's last built first, so a corpus of an
    older revision built again never comes before one of a newer revision.
    vocabulary (a sha256) and size (max_tokens, None for none) keep only the
    corpora built for them (triple); by default, every vocabulary and
    size."""
    found = [record for record in records(dock)
             if record["depot"] == depot and (record.get("dataset") or {}).get("id") == dataset
             and (vocabulary is None or triple(record)[1] == vocabulary)
             and (size is ANY or triple(record)[2] == size)]
    return sorted(found, key=lambda record: (record["dataset"].get("revision", 0), record["built_at"]), reverse=True)


def latest_fresh(dock, depot: str, dataset: str, vocabulary: str | None = None, size=ANY) -> dict | None:
    """The record of the fresh corpus last built for that dataset (and
    vocabulary and size, as built_from says); None without one."""
    return next((record for record in built_from(dock, depot, dataset, vocabulary, size) if fresh(record)), None)


def rows(dock, record: dict, table: str) -> list[dict]:
    """The rows of one of a corpus's tables (texts, licenses, notices), in
    order, each file checked by the dock: the texts table is the
    concatenation of its parts. A record that names the portions training
    reads (`trains_on`: a corpus delivery cut into portions, as
    wpt.resources.corpus_record gives it) gives only the texts of those
    portions, never one of its held-out portion."""
    found = record["tables"][table]
    if isinstance(found, list):
        read = pa.concat_tables([_table(dock, one) for one in found]) if found else pa.table({})
    else:
        read = _table(dock, found)
    found = read.to_pylist()
    if table == "texts" and "trains_on" in record:
        found = [one for one in found if one["portion"] in record["trains_on"]]
    return found


def _table(dock, one: dict) -> pa.Table:
    """A Parquet file a record names by its reference {key, sha256}, checked by the dock."""
    return pq.read_table(io.BytesIO(api.fetch(dock, one["key"], one["sha256"])))
