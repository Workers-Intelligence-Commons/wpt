"""What a model trained before snapshots reads of its depot, frozen: the
libraries and datasets of format 12 and older, read with plain SQL from
the tables that held them (libraries_revisions, memberships, pins,
datasets_revisions), and the tokenizer recipes and vocabularies those
models were trained with (tokenizers_revisions, vocabularies,
vocabulary_texts). Every read the dataset routes and training make of
those tables is here, and this module never writes: it opens the depot's
file read-only, of format 12 or of a later format that keeps the tables,
and gives what the dataset routes of `depot web` gave then
(depot_app.web.legacy serves them, docs/training.md says who reads them).

A dataset is a list of libraries, each read at a change number of the
depot that owns it (in a clone, a library of the upstream at the
upstream's number): an open library holds the texts that had joined it and
not left it then, each at its latest revision numbered within what the
depot held then (bounds, Depot.bounds); a library locked then holds the
revisions it pinned, read at the change that locked it. A dataset reads
only libraries of the depot that holds it (reads_here); a revision that
names another depot's library is refused where it is read.

The texts of a dataset come in the order of their export keys (the column
export_key of memberships and pins, depot.ids.export_key), a text two
libraries hold once, at its revision in the first of them the dataset
names. Two things read that order: the export of a dataset's current
revision in numbered parts (Export, as its own docstring says), and the
sample of a dataset that trains a vocabulary (sample), each plain text of a
share filling that share's part of the sample's bytes, a plain text longer
than depot.pieces.LENGTH characters as its pieces, each at the export
key of its name, as a corpus of then took them.

What the routes said of a dataset that could still change then is frozen
now: a dataset is never read again (read_again_shows is False), and its
figures say what the revision stored, never what its libraries hold now
(since is None).

    Legacy(address)                        the legacy tables of the depot at address, read-only
    Legacy.datasets()                      every dataset's current row, with upstream and here
    Legacy.dataset_by_id(id), dataset(name)  one dataset's current row
    Legacy.remote(name)                    a dataset's remote: the REMOTE fields of its row
    Legacy.figures(id, revision)           what a revision stored of each library it reads
    Legacy.library_range(...)              a library's texts at a change, by export key
    Legacy.tokenizer(name), vocabulary_file(name, version)  a tokenizer recipe and its vocabularies
    Export(legacy, dataset, url)           the export of a dataset's current revision in parts
    sample(legacy, name, sample_bytes)     the sample of a dataset that trains a vocabulary
    depot_name(depot, dataset, url)        the name an export and a corpus of a dataset record for its depot
"""

import hashlib
import heapq
import itertools
import json
import math
from collections.abc import Iterator, Sequence
from compression import zstd
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import counts, groups, ids, pieces, quotes
from .depot import Depot
from .errors import NotFound, Refused
from .fill import Fill
from .ids import EXPORT_SEED, export_key
from .kinds import PLAIN
from .parts import Parted, scan
from .schema import CONTENT_JOIN, LISTED_COLUMNS, LONG, tags_now

LEFT_OUT, UNMATCHED = "quoted text left out", "quote unmatched"  # why a conversation is left out of an export
TRAINING_KINDS = ("conversation", "task")  # the kinds beside plain a dataset's corpus took

# a text at one revision with the columns Depot.listed_text gives, its tags those its revision holds: what a revision
# held before format 13, the revisions these tables name all written before it (a later one holds none)
LISTED = LISTED_COLUMNS.replace(f"{tags_now('t.id')} AS tags", "t.tags")
# the fields of a dataset's row its remote holds (Legacy.remote): what `depot dataset import` read
REMOTE = ("name", "tokenizer", "libraries", "shares", "max_tokens")
# the columns of a dataset's current row, as the view datasets of format 12 gave them
DATASET = "id, revision, change, name, tokenizer, libraries, shares, max_tokens, vocabulary"
TEXTS_KEY = "sqlite_autoindex_texts_revisions_1"
MEMBERSHIPS_KEY = "sqlite_autoindex_memberships_1"
PINS_KEY = "sqlite_autoindex_pins_1"
# the revision of the text h.text a library held at the change :change, the change numbers of each depot's rows it
# held then given by :bounds (Depot.bounds), and whether the membership h was open then
REVISION_THEN = (f"(SELECT max(r.revision) FROM texts_revisions r INDEXED BY {TEXTS_KEY} WHERE r.id = h.text "
                 f"AND r.change <= (SELECT b.value FROM json_each(:bounds) b WHERE b.key = {ids.owner_sql('r.id')}))")
MEMBER_THEN = "AND h.joined_change <= :change AND (h.left_change IS NULL OR h.left_change > :change) "
CONTENT_JOIN = "JOIN content_lists c ON c.id = t.id AND c.version = t.content_version"

PART_TEXTS = 200_000  # texts of the dataset's libraries per part, about: near 1 GB of text at 5 KB a text
SPACE = 1 << 64  # how many keys there are
LOWEST, HIGHEST = -(1 << 63), (1 << 63) - 1  # the first and last export keys
LEVEL = 6  # the zstd level of a part in jsonl.zst
GROUP_TEXTS = 4096  # texts in one row group of a part in parquet, at most
GROUP_BYTES = 64 << 20  # characters of text after which a row group of a part in parquet ends
TEXT_SCHEMA = pa.schema([pa.field("depot", pa.string(), False), pa.field("id", pa.string(), False),
                         pa.field("revision", pa.int64(), False), pa.field("kind", pa.string(), False),
                         pa.field("text", pa.large_string(), False)])
FORMS = {"jsonl.zst": "application/zstd", "parquet": "application/vnd.apache.parquet"}
# what makes a part's bytes beside the export: a size or sha256 kept for a part holds only for the same MAKER
ROWS = 2  # the version of what a part's rows hold
MAKER = f"rows {ROWS}, pyarrow {pa.__version__}, zstd {zstd.zstd_version}"
PAGE = 2_000  # rows of one library a walk of the sample reads in one statement
WHOLE = -1  # what a whole text's place holds where a piece's holds its number


class Legacy:
    """The legacy tables of the depot in the SQLite file at address (a path
    or a file:// address), read through connections that open it read-only,
    whatever its format: nothing here checks the format, so a file of format
    12 reads as one of a later format that keeps the tables."""

    def __init__(self, address: str):
        self.depot = Depot(address, readonly=True)
        self.address = self.depot.address

    # What the rest of the depot says, read through the depot's own reads (no legacy table)

    def name(self) -> str:
        return self.depot.name()

    def upstream(self) -> str | None:
        return self.depot.upstream()

    def upstream_owned(self, row_ids: Sequence[str]) -> list[str]:
        return self.depot.upstream_owned(row_ids)

    def reads_here(self, address: str | None, library: str) -> bool:
        """Whether a library a dataset names (its depot's address, None for
        this depot, and its id) is one of this depot's rows: its address
        leads here (in any form, as Depot.place compares them), or, in a
        depot that follows an upstream, it is the upstream's web address and
        the id is one the upstream owns, a library a pull brought here."""
        place = self.depot.place(address)
        if place is None:
            return True
        return (self.upstream() is not None and place == self.depot.place(self.upstream())
                and bool(self.upstream_owned([library])))

    def listed_text(self, text: str, revision: int) -> dict:
        """One text at one revision as Depot.listed_text gives it, but its
        tags those its revision holds (LISTED), read in a file of format 12
        too. NotFound when the depot no longer holds it."""
        found = self.depot.rows(f"SELECT {LISTED} FROM texts_revisions t {CONTENT_JOIN} WHERE t.id = ? "
                                "AND t.revision = ?", (text, revision))
        if not found:
            raise NotFound(f"the depot holds no revision {revision} of the text {text}")
        return found[0]

    def reader(self):
        return self.depot.reader()

    def rows(self, sql: str, params=None) -> list[dict]:
        return self.depot.rows(sql, params)

    def bounds(self, depot: str, change: int) -> dict[str, int]:
        return self.depot.bounds(depot, change)

    def _con(self):
        return self.depot.db.reader()

    # Datasets

    def datasets(self) -> list[dict]:
        """Every dataset's current row, in the order of their ids, with
        upstream (1 when the upstream owns it, else 0) and here: the ids of
        the libraries it reads that this depot holds and reads, each once."""
        found = self.rows(f"SELECT {DATASET}, {self.depot.upstream_sql()} AS upstream FROM datasets_revisions "
                          "WHERE latest ORDER BY id")
        held = self.libraries()
        return [self.here(row, held) for row in found]

    def here(self, row: dict, held: dict) -> dict:
        """A dataset's row with here: the ids of the libraries it reads from
        this depot (reads_here) that this depot holds (held: libraries)."""
        return {**row, "here": list(dict.fromkeys(
            ref["library"] for ref in row["libraries"]
            if ref["library"] in held and self.reads_here(ref["depot"], ref["library"])))}

    def dataset(self, name: str) -> dict:
        """The current row of the dataset named so. NotFound without one."""
        found = self.rows(f"SELECT {DATASET} FROM datasets_revisions WHERE latest AND name = ?", (name,))
        if not found:
            raise NotFound(f"no dataset {name}")
        return found[0]

    def dataset_by_id(self, dataset: str) -> dict:
        """The current row of the dataset of that id. NotFound without one."""
        found = self.rows(f"SELECT {DATASET} FROM datasets_revisions WHERE latest AND id = ?", (dataset,))
        if not found:
            raise NotFound(f"no dataset {dataset}")
        return found[0]

    def dataset_named(self, dataset: str) -> dict:
        """The current row of the dataset of that id, else of that name. NotFound without one."""
        try:
            return self.dataset_by_id(dataset)
        except NotFound:
            return self.dataset(dataset)

    def revision(self, dataset: str, revision: int, columns: Sequence[str]) -> dict | None:
        """Those columns of one revision of the dataset, None without it."""
        found = self.rows(f"SELECT {', '.join(columns)} FROM datasets_revisions WHERE id = ? AND revision = ?",
                          (dataset, revision))
        return found[0] if found else None

    def remote(self, name: str) -> dict:
        """The dataset's remote: the REMOTE fields of its current row."""
        found = self.dataset(name)
        return {key: found[key] for key in REMOTE}

    def read_again_shows(self, dataset: str) -> bool:
        """Whether reading the dataset again would change what it reads:
        never, since nothing reads a dataset again. NotFound without it."""
        self.dataset_by_id(dataset)
        return False

    def libraries(self) -> dict[str, dict]:
        """{id: {id, name, locked_at}} of every library this depot holds, as it is now."""
        return {one["id"]: one for one in self.rows(
            "SELECT id, name, locked_at FROM libraries_revisions WHERE latest")}

    def library(self, library: str) -> dict:
        """The current row of the library. NotFound without one."""
        found = self.rows("SELECT id, revision, change, name, forked_from, problems, locked_at "
                          "FROM libraries_revisions WHERE latest AND id = ?", (library,))
        if not found:
            raise NotFound(f"no library {library}")
        return found[0]

    def figures(self, dataset: str, revision: int | None = None) -> dict:
        """What each library a revision of the dataset of that id reads (its
        current revision by default) held at the change it reads it at, as
        the revision stores it in its column figures: {tags: those of the
        revision's shares, in order, libraries: for each library, in order,
        {stored: its figures ({kinds: {kind: {texts, characters, bytes}},
        tags: [the same for each tag]}), None when not
        known}}. A stored figure is not known when the write of the revision
        could not prove it, or when an erasure carried out after the figures
        were read may have reached a text the library held: one whose text
        the library holds or held, one that removed a text whole (which
        leaves no row naming its libraries), and one of a depot whose rows
        the dataset's depot did not hold at the figures' change. NotFound
        without such a revision."""
        found = self.rows("SELECT libraries, shares, figures FROM datasets_revisions WHERE id = :id AND "
                          + ("latest" if revision is None else "revision = :revision"),
                          {"id": dataset, "revision": revision})
        if not found:
            raise NotFound(f"no {'current' if revision is None else f'revision {revision} of the'} dataset {dataset}")
        row = found[0]
        tags = [one["tag"] for one in row["shares"] or []]
        figures = row["figures"] or {"at": None, "libraries": [None] * len(row["libraries"])}
        con = self._con()
        with con:  # one read transaction: the stored figures and the erasures of one state
            reached = _reached(con, ids.owner(dataset), figures["at"])
            libraries = []
            for ref, one in zip(row["libraries"], figures["libraries"], strict=True):
                if one is not None and (reached is None or any(
                        _held_ever(con, ref["library"], text) for text in reached)):
                    one = None
                libraries.append({"stored": one})
        return {"tags": tags, "libraries": libraries}

    def revision_figures(self, dataset: dict) -> dict:
        """What the revision of the dataset whose row is dataset reads, from
        the figures it stores (figures): {measure: as the corpus build mixed
        gives it for the dataset's shares, None when the figures of one of
        its libraries are not known; since: None, libraries no longer move}."""
        found = self.figures(dataset["id"], dataset["revision"])
        stored = [one["stored"] for one in found["libraries"]]
        if any(one is None for one in stored):
            return {"measure": None, "since": None}
        return {"measure": mixed(_summed(stored, found["tags"]), dataset["shares"] or []), "since": None}

    # What a library held at a change

    def _locked_then(self, library: str, change: int) -> bool:
        """Whether the library was locked at that change. NotFound without
        such a library, or when it did not exist yet at that change."""
        self.library(library)
        state = self._con().execute("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                                    "ORDER BY revision DESC LIMIT 1", (library, change)).fetchall()
        if not state:
            raise NotFound(f"the library {library} did not exist yet at the change {change}")
        return state[0][0] is not None

    def _lock_change(self, library: str) -> int | None:
        """The change at which the library was locked, in the numbering of the depot that owns it; None if never."""
        found = self._con().execute("SELECT change FROM libraries_revisions WHERE id = ? AND locked_at IS NOT NULL "
                                    "ORDER BY revision LIMIT 1", (library,)).fetchall()
        return found[0][0] if found else None

    def _held(self, library: str, change: int, pins: str, memberships: str) -> tuple[str, str, str, int]:
        """(the table of the library's rows, the revision of a row, the condition of a row held, the change
        read at) of the library at that change: its pins at its lock change when it was locked then, else its
        memberships open then."""
        if self._locked_then(library, change):
            return f"pins h INDEXED BY {pins}", "h.revision", "", self._lock_change(library)
        return f"memberships h INDEXED BY {memberships}", REVISION_THEN, MEMBER_THEN, change

    def library_range(self, library: str, change: int, low: int, high: int, limit: int | None = None
                      ) -> Iterator[tuple]:
        """The texts the library held at that change whose export key is
        from low to high, both included, in the order of their keys, the
        first limit of them (None: all): (export key, id, revision, kind,
        tags as JSON text, bytes) of each, read as that range of the index
        memberships_key (pins_key for a library locked then), each row's
        revision and content version looked up after it (CROSS JOIN fixes
        that order)."""
        held, revision, when, change = self._held(library, change, "pins_key", "memberships_key")
        cursor = self._con().cursor().execute(
            f"SELECT h.export_key, t.id, t.revision, c.kind, t.tags, c.bytes FROM {held} "
            f"CROSS JOIN texts_revisions t ON t.id = h.text AND t.revision = {revision} {CONTENT_JOIN} "
            f"WHERE h.library = :library AND h.export_key BETWEEN :low AND :high {when}ORDER BY h.export_key"
            + ("" if limit is None else f" LIMIT {int(limit)}"),
            {"library": library, "change": change, "low": low, "high": high,
             "bounds": json.dumps(self.bounds(ids.owner(library), change))})
        yield from cursor

    def library_long(self, library: str, change: int) -> list[tuple]:
        """The plain texts the library held at that change whose content
        version is longer than schema.LONG characters, in no set order:
        (export key, id, revision, tags as JSON text, length, bytes), read
        from the partial index content_lists_long."""
        held, revision, when, change = self._held(library, change, PINS_KEY, MEMBERSHIPS_KEY)
        return self._con().execute(
            "SELECT h.export_key, t.id, t.revision, t.tags, c.length, c.bytes FROM content_lists c INDEXED BY "
            f"content_lists_long CROSS JOIN {held} ON h.library = :library AND h.text = c.id {when}"
            f"CROSS JOIN texts_revisions t ON t.id = h.text AND t.revision = {revision} "
            f"WHERE c.kind = '{PLAIN}' AND c.length > {LONG} AND t.content_version = c.version",
            {"library": library, "change": change, "bounds": json.dumps(self.bounds(ids.owner(library), change))}
        ).fetchall()

    def library_count(self, library: str, change: int) -> int:
        """How many texts the library held at that change, counted from its
        rows of the index memberships_key (pins_key for a library locked then) alone."""
        con = self._con()
        if self._locked_then(library, change):
            return con.execute("SELECT count(*) FROM pins INDEXED BY pins_key WHERE library = ?",
                               (library,)).fetchall()[0][0]
        return con.execute("SELECT count(*) FROM memberships INDEXED BY memberships_key WHERE library = ? "
                           "AND joined_change <= ? AND (left_change IS NULL OR left_change > ?)",
                           (library, change, change)).fetchall()[0][0]

    # Tokenizer recipes and their vocabularies, which `wpt train` of a dataset's order read

    def tokenizer(self, name: str) -> dict:
        """The tokenizer recipe named so as the depot kept it: {recipe: its
        current revision, revisions: every revision, the latest first,
        vocabularies: each version's vocabulary without its file, the latest
        version first, trained: whether its current version has one}.
        NotFound without such a recipe."""
        found = self.depot.rows("SELECT * FROM tokenizers WHERE name = ?", (name,))
        if not found:
            raise NotFound(f"no tokenizer recipe {name}")
        recipe = found[0]
        vocabularies = self.depot.rows("SELECT recipe, version, change, sha256, texts, bytes, made_at FROM vocabularies "
                                       "WHERE recipe = ? ORDER BY version DESC", (recipe["id"],))
        return {"recipe": recipe,
                "revisions": self.depot.rows("SELECT * FROM tokenizers_revisions WHERE id = ? ORDER BY revision DESC",
                                             (recipe["id"],)),
                "vocabularies": vocabularies,
                "trained": any(one["version"] == recipe["version"] for one in vocabularies)}

    def vocabulary_file(self, name: str, version: int) -> tuple[str, bytes]:
        """(the sha256, the file) of the vocabulary of that version of the
        tokenizer recipe named so. NotFound without the recipe or without a
        vocabulary of that version."""
        recipe = self.tokenizer(name)["recipe"]
        found = self._con().execute("SELECT sha256, merges FROM vocabularies WHERE recipe = ? AND version = ?",
                                    (recipe["id"], version)).fetchall()
        if not found:
            raise NotFound(f"the tokenizer recipe {name} has no vocabulary of its version {version}")
        return found[0][0], found[0][1].encode("utf-8")

    def vocabulary_named(self, sha256: str) -> dict:
        """{sha256, recipe: the name of its tokenizer recipe, version} of the
        depot's vocabulary of that sha256; recipe and version None when the
        depot holds none."""
        found = self._con().execute(
            "SELECT r.name, v.version FROM vocabularies v JOIN tokenizers_revisions r ON r.id = v.recipe AND r.latest "
            "WHERE v.sha256 = ? ORDER BY v.recipe, v.version LIMIT 1", (sha256,)).fetchall()
        recipe, version = found[0] if found else (None, None)
        return {"sha256": sha256, "recipe": recipe, "version": version}


def _reached(con, owner: str, at: int | None) -> list[str] | None:
    """The ids of the texts whose passages the erasures cut that the depot
    named owner did not hold carried out at its change at (bounds: one of
    another depot's numbering counts as held once a pull at or before at
    brought its carrying out), None when one of them removed a text whole,
    is of a depot whose rows owner held none of at at, or at is None."""
    if at is None:
        return None
    pulls = con.execute("SELECT depot, change, upstream, upstream_change FROM pulls").fetchall()
    held, depot, change = {}, owner, at
    while depot not in held:
        held[depot] = change
        last = max((one for one in pulls if one[0] == depot and one[1] <= change), key=lambda one: one[1], default=None)
        if last is None:
            break
        depot, change = last[2], last[3]
    texts = set()
    for erasure, carried, needles in con.execute(
            "SELECT e.id, (SELECT min(r.change) FROM erasures_revisions r WHERE r.id = e.id "
            "AND r.erased IS NOT NULL), e.needles FROM erasures_revisions e WHERE e.latest AND e.erased IS NOT NULL"
    ).fetchall():
        bound = held.get(ids.owner(erasure))
        if bound is None:
            return None
        if carried <= bound:
            continue
        for needle in json.loads(needles):
            if needle["start"] is None and needle["field"] is None:
                return None
            texts.add(needle["text"])
    return sorted(texts)



def _summed(found: Sequence[dict], tags: Sequence[str]) -> dict:
    """The figures of several libraries (each as a revision stores them,
    for tags) as one: {texts, characters, bytes of them all; kinds: {kind:
    {texts, characters, bytes}} of every kind any of them names, plain
    first; shares: for each tag of tags, in order, {tag, texts, characters,
    bytes}; unshared: the plain texts left once the shares are taken away,
    when tags are given (None otherwise)}, each library counted apart."""
    empty = dict.fromkeys(counts.COLUMNS, 0)
    kinds = {PLAIN: dict(empty)}
    shares = [dict(empty) for _ in tags]
    for one in found:
        pairs = [*((kinds.setdefault(kind, dict(empty)), sized) for kind, sized in one["kinds"].items()),
                 *zip(shares, one["tags"], strict=True)]
        for into, read in pairs:
            for key in counts.COLUMNS:
                into[key] += read[key]
    unshared = {key: max(0, kinds[PLAIN][key] - sum(one[key] for one in shares)) for key in counts.COLUMNS}
    return {key: sum(one[key] for one in kinds.values()) for key in counts.COLUMNS} | {
        "kinds": kinds, "shares": [{"tag": tag, **shares[at]} for at, tag in enumerate(tags)],
        "unshared": unshared if tags else None}

def _held_ever(con, library: str, text: str) -> bool:
    """Whether the library holds or held the text: a membership or a pin of it, read by their keys."""
    return bool(con.execute("SELECT 1 FROM memberships WHERE library = ? AND text = ? UNION ALL "
                            "SELECT 1 FROM pins WHERE library = ? AND text = ? LIMIT 1",
                            (library, text, library, text)).fetchall())


def depot_name(depot, dataset: dict, url: str | None) -> str:
    """The name everything an export and a corpus of the dataset (a row
    depot, a Depot or a Legacy, holds) recorded for depot, the same wherever
    its file is: the web address of depot's upstream when the dataset is a
    row the upstream owns (a clone's pull brought it), else url, depot's
    public address, else depot's name."""
    if depot.upstream_owned([dataset["id"]]):
        return depot.upstream()
    return url or depot.name()


def _depots(legacy: Legacy, dataset: dict, holder: str) -> tuple[list[str], dict[str, Legacy]]:
    """(holder, the name an export records for the depot, once for each
    library the dataset names; {holder: legacy}). A dataset reads only
    libraries the depot holds and reads (reads_here), as the revision the
    row dataset names stores them. NotFound when the depot holds no such
    revision; Refused when it names a library of another depot."""
    stored = legacy.revision(dataset["id"], dataset["revision"], ["libraries"])
    if stored is None:
        raise NotFound(f"no revision {dataset['revision']} of the dataset {dataset['id']}")
    for ref in stored["libraries"]:
        if not legacy.reads_here(ref["depot"], ref["library"]):
            raise Refused(f"the dataset {dataset['name']} reads the library {ref['library']} of another depot, "
                          f"{ref['depot']}: an export reads only the libraries of the depot that holds its dataset")
    return [holder] * len(stored["libraries"]), {holder: legacy}


def _erasures(dataset: dict, names: list[str], opened: dict[str, Legacy]) -> dict[str, list[str]]:
    """{the name of each depot the dataset reads (_depots): the ids of the
    erasures it carried out that reached a text one of the dataset's
    libraries there holds or ever held}."""
    libraries = {}
    for name, ref in zip(names, dataset["libraries"]):
        libraries.setdefault(name, []).append(ref["library"])
    found = {}
    for name, held in libraries.items():
        found[name] = [one["id"] for one in opened[name].rows(
            "SELECT e.id FROM erasures_revisions e WHERE e.latest AND e.erased IS NOT NULL AND EXISTS (SELECT 1 FROM "
            "json_each(e.needles) n JOIN memberships m ON m.text = json_extract(n.value, '$.text') WHERE m.library IN "
            "(SELECT value FROM json_each(?))) ORDER BY e.id", (json.dumps(held),))]
    return found


# A dataset's export

def parts_for(texts: int) -> int:
    """How many parts an export of a dataset whose libraries hold texts texts has: one per PART_TEXTS, at least one."""
    return max(1, math.ceil(texts / PART_TEXTS))


def bounds(number: int, parts: int) -> tuple[int, int]:
    """The keys of part number of parts: (the first, the one after the last)."""
    return LOWEST + number * SPACE // parts, LOWEST + (number + 1) * SPACE // parts


class Export:
    """The export of the current revision of a dataset (its row) of the
    depot legacy reads, whose depot url names (depot_name): every text of
    the revision a corpus took then, split into numbered parts, made when
    asked and never stored.

    The texts are those of the dataset's libraries in order (the module
    says how), of them the plain texts a share takes (without shares, every
    plain text) and the conversations and tasks, with no limit. A
    conversation that quotes texts (depot.quotes) is kept only when
    each excerpt's text is a plain text the export holds from the same
    depot and its selector matches exactly one place of the revision the
    export holds; otherwise it is left out as LEFT_OUT or UNMATCHED and
    counted in the part it would be in (part's left_out). The export has
    parts_for(texts) parts, texts being how many texts its libraries held
    at their change numbers (texts), and part n holds the texts whose key
    falls in the n-th of that many equal ranges of the keys (bounds), each
    library read as that range of its own rows by key and the reads merged.

    A part comes in two forms, each the same bytes every time it is made
    for the same export (identity: the dataset's revision and the erasures
    its depot carried out), by the same MAKER: "jsonl.zst", one JSON line
    per text, {"depot", "id", "revision", "kind", "text"} (an item's text is
    its JSON content), compressed with zstd at LEVEL as one frame; and
    "parquet", a Parquet file of the same columns (TEXT_SCHEMA), zstd
    compressed, in row groups of at most GROUP_TEXTS texts or about
    GROUP_BYTES characters, each group sent once written."""

    def __init__(self, legacy: Legacy, dataset: dict, url: str | None):
        self.legacy, self.dataset = legacy, dataset
        self.holder = depot_name(legacy, dataset, url)
        self.names, self.opened = _depots(legacy, dataset, self.holder)
        self.identity = _fingerprint({"seed": EXPORT_SEED, "depot": self.holder, "dataset": dataset["id"],
                                      "revision": dataset["revision"], "libraries": dataset["libraries"],
                                      "erasures": _erasures(dataset, self.names, self.opened)})

    def texts(self) -> int:
        """How many texts the dataset's libraries held at their change
        numbers, each library counted apart (a text two hold counts twice,
        and so does one no share takes): what sets the number of parts. A
        library's count is the one the revision stores in its figures
        (Legacy.figures), or, where that figure is not known, counted from
        the library's index."""
        stored = self.legacy.figures(self.dataset["id"], self.dataset["revision"])["libraries"]
        return sum(self.opened[name].library_count(ref["library"], ref["change"]) if one["stored"] is None
                   else sum(sized["texts"] for sized in one["stored"]["kinds"].values())
                   for name, ref, one in zip(self.names, self.dataset["libraries"], stored, strict=True))

    def members(self, number: int, parts: int) -> Iterator[tuple[str, str, int, str]]:
        """The texts of part number of parts, in key order (texts of one key
        in the order of their ids, then of their depots' names), a text two
        libraries hold once, at its revision in the first of them: (the name
        a corpus records for its depot, id, revision, kind)."""
        low, high = bounds(number, parts)

        def read(name: str, ref: dict) -> Iterator[tuple]:
            for key, text, revision, kind, held_tags, _ in self.opened[name].library_range(
                    ref["library"], ref["change"], low, high - 1):
                yield key, name, text, revision, kind, held_tags

        reads = [read(name, ref) for name, ref in zip(self.names, self.dataset["libraries"])]
        # heapq.merge keeps the libraries' order among rows of one key, so the first library's revision comes first
        for _, rows in itertools.groupby(heapq.merge(*reads, key=lambda row: row[0]), key=lambda row: row[0]):
            found = {}
            for _, name, text, revision, kind, held_tags in rows:
                if (name, text) in found:
                    continue
                found[name, text] = (revision, kind, self._taken(kind, held_tags))
            yield from ((name, text, revision, kind) for (text, name), (revision, kind, taken)
                        in sorted(((text, name), value) for (name, text), value in found.items()) if taken)

    def _taken(self, kind: str, held_tags: str) -> bool:
        """Whether the export takes a text of that kind and tags (JSON text) in
        the first library holding it: a plain text a share takes (every plain
        text without shares), an item of a kind a corpus trains on."""
        if kind != PLAIN:
            return kind in TRAINING_KINDS
        shares = self.dataset["shares"]
        return shares is None or any(one["tag"] in json.loads(held_tags) for one in shares)

    def _held(self, name: str, text: str) -> int | None:
        """The revision at which the export holds text as a plain text of
        the depot named name, None when it does not hold it so: read from
        the first of the dataset's libraries in that depot holding it."""
        key = export_key(text)
        for library, ref in zip(self.names, self.dataset["libraries"]):
            if library != name:
                continue
            for _, held, revision, kind, held_tags, _ in self.opened[name].library_range(
                    ref["library"], ref["change"], key, key):
                if held == text:
                    return revision if kind == PLAIN and self._taken(kind, held_tags) else None
        return None

    def _left_out(self, name: str, conversation: dict) -> str | None:
        """Why the export leaves out a conversation of the depot named name
        (LEFT_OUT, UNMATCHED), or None when it holds it."""
        source = self.opened[name]
        for block in quotes.excerpts(conversation):
            revision = self._held(name, block["source"])
            if revision is None:
                return LEFT_OUT
            quoted = source.listed_text(block["source"], revision)
            if len(quotes.find(Parted(source.reader(), block["source"], quoted["parts"]), block["selector"])) != 1:
                return UNMATCHED
        return None

    def _rows(self, number: int, parts: int, left_out: dict[str, int]) -> Iterator[tuple[dict, Iterator[str]]]:
        """For each text of the part, in order: (its row without its text, its
        text's pieces in order); a conversation is read whole, and one the
        export leaves out is counted in left_out instead."""
        for name, text, revision, kind in self.members(number, parts):
            source = self.opened[name]
            listed = source.listed_text(text, revision)
            texts = (value for _, value in scan(source.reader(), {text: listed["parts"]}) if value is not None)
            if kind == "conversation":
                whole = "".join(texts)
                why = self._left_out(name, json.loads(whole))
                if why is not None:
                    left_out[why] += 1
                    continue
                texts = iter([whole])
            yield {"depot": name, "id": text, "revision": revision, "kind": kind}, texts

    def part(self, number: int, parts: int, form: str, left_out: dict[str, int] | None = None) -> Iterator[bytes]:
        """The bytes of part number (from 0) of parts in that form (FORMS),
        as they are made. left_out, when given, ends up holding how many
        conversations of the part the export left out, by why (LEFT_OUT,
        UNMATCHED), once the bytes were all read. KeyError for a form
        FORMS does not name, IndexError for a part outside parts."""
        if form not in FORMS:
            raise KeyError(form)
        if not 0 <= number < parts:
            raise IndexError(number)
        if left_out is None:
            left_out = {}
        left_out.update({LEFT_OUT: 0, UNMATCHED: 0})
        rows = self._rows(number, parts, left_out)
        return _jsonl(rows) if form == "jsonl.zst" else _parquet(rows)


def _jsonl(rows) -> Iterator[bytes]:
    compressor = zstd.ZstdCompressor(level=LEVEL)
    for row, texts in rows:
        head = json.dumps(row, ensure_ascii=False)[:-1] + ', "text": "'
        out = compressor.compress(head.encode("utf-8"))
        if out:
            yield out
        for piece in texts:
            out = compressor.compress(json.dumps(piece, ensure_ascii=False)[1:-1].encode("utf-8"))
            if out:
                yield out
        out = compressor.compress(b'"}\n')
        if out:
            yield out
    yield compressor.flush()


class _Sink:
    """A file a Parquet writer writes to, whose bytes are taken as they come."""

    def __init__(self):
        self.held, self.position, self.closed = [], 0, False

    def write(self, data) -> int:
        data = bytes(data)
        self.held.append(data)
        self.position += len(data)
        return len(data)

    def tell(self) -> int:
        return self.position

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def taken(self) -> bytes:
        found = b"".join(self.held)
        self.held.clear()
        return found


def _parquet(rows) -> Iterator[bytes]:
    sink = _Sink()
    writer = pq.ParquetWriter(pa.PythonFile(sink, mode="w"), TEXT_SCHEMA, compression="zstd")
    group, size = [], 0
    for row, texts in rows:
        text = "".join(texts)
        group.append({**row, "text": text})
        size += len(text)
        if len(group) >= GROUP_TEXTS or size >= GROUP_BYTES:
            writer.write_table(pa.Table.from_pylist(group, schema=TEXT_SCHEMA))
            group, size = [], 0
            yield sink.taken()
    if group:
        writer.write_table(pa.Table.from_pylist(group, schema=TEXT_SCHEMA))
    writer.close()
    yield sink.taken()


# A dataset's sample, which trains a vocabulary

class _Entry(NamedTuple):
    """A text or a piece of the order of a dataset, as depot.groups reads one: its key, id, piece (its
    number, None for a whole text), revision, kind, pool (pool_of), bytes (a piece's: its share of its text's),
    taken and tags (JSON text)."""
    key: int
    id: str
    piece: int | None
    revision: int
    kind: str
    pool: int | None
    bytes: int
    taken: bool
    tags: str


class _Long(NamedTuple):
    """A long text whose pieces the walk gives: its id, revision, tags (JSON text), length and bytes."""
    id: str
    revision: int
    tags: str
    length: int
    bytes: int


def _long_texts(libraries: Sequence[tuple]) -> list[_Long]:
    """The long texts of the libraries ((legacy, library, change) each)
    that the walk gives as pieces, in the order of their ids: a text whose
    revision in the first library holding it is long."""
    found = {}
    for at, (legacy, library, change) in enumerate(libraries):
        for key, text, revision, tags, length, size in legacy.library_long(library, change):
            if text in found:
                continue
            # an earlier library that holds it holds a revision that is not long: a whole text
            if any(text in (row[1] for row in held.library_range(name, then, key, key))
                   for held, name, then in libraries[:at]):
                continue
            found[text] = _Long(text, revision, tags, length, size)
    return [found[text] for text in sorted(found)]


def _walk(libraries: Sequence[tuple], shares: Sequence[str] | None, long: list[_Long]) -> Iterator[_Entry]:
    """The texts and pieces of the libraries ((legacy, library, change)
    each, in the dataset's order) by (export key, id, piece), a whole text
    before a piece, a text two libraries hold once at its revision in the
    first of them, a long text (long) as its pieces, each as _Entry under
    the tags of the dataset's shares (None without shares)."""
    cut = {one.id for one in long}
    reads = [_library(legacy, library, change) for legacy, library, change in libraries]
    last = None
    # heapq.merge gives rows of one place in the order of the reads: the first library's revision first
    for key, text, piece, revision, kind, tags, size in heapq.merge(*reads, _pieces(long), key=lambda row: row[:3]):
        if (key, text, piece) == last or (piece == WHOLE and text in cut):
            continue
        last = key, text, piece
        pool, taken = pool_of(kind, tags, shares)
        yield _Entry(key, text, None if piece == WHOLE else piece, revision, kind, pool, size, taken, tags or "[]")


def _pieces(long: list[_Long]) -> Iterator[tuple]:
    """The rows of the pieces of the long texts, in the order of their (key,
    id, number), as _library gives a text's (its kind plain, its bytes its
    share of its text's)."""
    counted = np.array([pieces.count(one.length) for one in long], np.int64)
    total = int(counted.sum())
    owners = np.repeat(np.arange(len(long), dtype=np.int32), counted)
    numbers = (np.arange(total) - np.repeat(np.cumsum(counted) - counted, counted)).astype(np.int32)
    keys = np.fromiter((export_key(pieces.name(long[owner].id, int(number)))
                        for owner, number in zip(owners, numbers)), np.int64, total)
    order = np.argsort(keys, kind="stable")  # long is in the order of ids, each text's pieces by number
    for at in order.tolist():
        owner, number = int(owners[at]), int(numbers[at])
        one, count = long[owner], int(counted[owner])
        yield (int(keys[at]), one.id, number, one.revision, PLAIN, one.tags,
               one.bytes // count + (number < one.bytes % count))


def _library(legacy: Legacy, library: str, change: int) -> Iterator[tuple]:
    """The rows of one library at that change, in the order of their (key,
    id), PAGE at a time: (key, id, WHOLE, revision, kind, tags as JSON
    text, bytes). A page ends before the rows of its last key, which the
    next page starts with, so the texts of one key are sorted by id together."""
    low = LOWEST
    while True:
        rows = list(legacy.library_range(library, change, low, HIGHEST, PAGE))
        if not rows:
            return
        ended = len(rows) < PAGE
        if not ended:
            top = rows[-1][0]
            rows = ([row for row in rows if row[0] != top] if rows[0][0] != top
                    else list(legacy.library_range(library, change, top, top)))
        found = sorted(((key, text, WHOLE, revision, kind, tags, size)
                        for key, text, revision, kind, tags, size in rows), key=lambda row: row[:2])
        yield from found
        high = found[-1][0]
        if ended or high == HIGHEST:
            return
        low = high + 1


def _proportions(shares: list[dict] | None, total: int) -> dict[int, int]:
    """{pool: its part of total}: each share's proportion of it, or all of it for the one pool without shares."""
    if shares is None:
        return {0: total}
    whole = sum(share["share"] for share in shares)
    return {at: int(total * share["share"] / whole) for at, share in enumerate(shares)}


def _plain(legacy: Legacy, text: str, parts: list, piece: int | None) -> str:
    """A plain text whose content version lists parts, whole, or its piece of that number."""
    if piece is not None:
        return pieces.read(Parted(legacy.reader(), text, parts), piece)
    return "".join(value for _, value in scan(legacy.reader(), {text: parts}) if value is not None)


def sample(legacy: Legacy, name: str, sample_bytes: int) -> Iterator[tuple[str, str, bytes]]:
    """The sample of the dataset named so (its name, or its id) that
    trains a vocabulary, read as a stream in one walk of the dataset's
    order: ('', which stands for the depot; its id; its text as UTF-8 bytes)
    of plain texts and pieces a corpus may take, a piece under its text's
    id, each share (without shares, every plain text) filling its
    proportion of sample_bytes as a corpus filled a budget (Fill, or the
    choice by group of depot.groups on bytes, a piece counting for its
    share of its text's bytes). Items are no sample. With shares, the first
    pass of that choice walks the whole order before the first text comes.
    The walk ends once every share has closed."""
    dataset = legacy.dataset_named(name)
    _depots(legacy, dataset, "")
    libraries = [(legacy, ref["library"], ref["change"]) for ref in dataset["libraries"]]
    shares = dataset["shares"]
    tags = None if shares is None else [one["tag"] for one in shares]
    budget = _proportions(shares, sample_bytes)
    long = _long_texts(libraries)
    chosen = (groups.Plan({}, np.zeros(0, np.int64)) if shares is None
              else groups.plan(_walk(libraries, tags, long), tags, budget))
    picker = chosen.picker()
    fill = Fill({pool: limit for pool, limit in budget.items() if not picker.covers(pool)})
    for entry in _walk(libraries, tags, long):
        if fill.done() and picker.done():
            break
        if not entry.taken or entry.kind != PLAIN:
            continue
        if picker.covers(entry.pool):
            if not picker.pick(entry.pool)[1]:
                continue
        elif not fill.take(entry.pool, entry.bytes):
            continue
        row = legacy.listed_text(entry.id, entry.revision)
        yield "", entry.id, _plain(legacy, entry.id, row["parts"], entry.piece).encode("utf-8")


def pool_of(kind: str, tags: str, shares: Sequence[str] | None, kinds: Sequence[str] | None = None
            ) -> tuple[int | None, bool]:
    """(pool, taken) of a text of that kind with tags (JSON text), under
    the tags of a data section's shares (None without shares) and the kinds
    it names (None or empty: every kind): pool, with shares, the index of
    the first share whose tag a plain text has, None when it has none;
    without shares, 0 for every plain text; None for an item; taken,
    whether a corpus takes it at all, that is, a text of a kind the section
    names that is a plain text with a pool or a text of one of
    TRAINING_KINDS (a text of any other kind is no training data)."""
    if kinds and kind not in kinds:
        return None, False
    if kind != PLAIN:
        return None, kind in TRAINING_KINDS
    if shares is None:
        return 0, True
    held = json.loads(tags or "[]")
    pool = next((at for at, tag in enumerate(shares) if tag in held), None)
    return pool, pool is not None


def scarcest_trim(available: dict[int, float], shares: list[dict]) -> dict[int, float]:
    """{pool: what it may take} of the corpus without max_tokens, from what
    each pool of shares holds (bytes or tokens): the scarcest share all it
    holds, each other its proportion of the scarcest's, at most all it
    holds. A share that holds nothing takes nothing, and the others are
    measured among themselves."""
    held = [pool for pool in available if available[pool]]
    if not held:
        return dict.fromkeys(available, 0)
    whole = sum(share["share"] for share in shares)
    proportion = {pool: shares[pool]["share"] / whole for pool in available}
    scarcest = min(held, key=lambda pool: available[pool] / proportion[pool])
    total = available[scarcest] / proportion[scarcest]
    return {pool: available[pool] if pool == scarcest else min(available[pool], proportion[pool] * total)
            for pool in available}


def mixed(found: dict, shares: list[dict]) -> dict:
    """Figures of what texts hold ({texts, characters, bytes, kinds,
    shares: one row for each tag of shares, [{tag, share}], in order,
    unshared}) with the mix of the corpus of every
    token when shares are given (mix None otherwise): each share's row
    gains share, kept_bytes and kept_characters (what scarcest_trim keeps
    of it, its characters in the same proportion; a share of no proportion
    takes nothing, and the others are measured among themselves) and
    limits (true on the share that keeps all it holds while another keeps
    less); mix: {kept_bytes, kept_characters of plain text, items_share:
    the share of the bytes the conversations and tasks hold then, the part
    of a size the build gives them together, kind_shares: {kind: its own
    share of those bytes} for each kind beside plain a corpus takes
    (TRAINING_KINDS)}."""
    if not shares:
        return {**found, "mix": None}
    held = {at: one["bytes"] for at, one in enumerate(found["shares"]) if shares[at]["share"] > 0}
    kept = dict.fromkeys(range(len(shares)), 0) | (scarcest_trim(held, shares) if held else {})
    scarcest = [at for at in held if held[at] and kept[at] == held[at]
                and any(kept[other] < held[other] for other in held)]
    rows = [{**one, "share": shares[at]["share"], "kept_bytes": int(kept[at]),
             "kept_characters": round(one["characters"] * kept[at] / one["bytes"]) if one["bytes"] else 0,
             "limits": at in scarcest} for at, one in enumerate(found["shares"])]
    plain = sum(one["kept_bytes"] for one in rows)
    items = counts.among(found["kinds"], TRAINING_KINDS)["bytes"]
    whole = items + plain
    return {**found, "shares": rows, "mix": {
        "kept_bytes": plain, "kept_characters": sum(one["kept_characters"] for one in rows),
        "items_share": items / whole if whole else 0,
        "kind_shares": {kind: counts.among(found["kinds"], [kind])["bytes"] / whole if whole else 0
                        for kind in TRAINING_KINDS}}}


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
