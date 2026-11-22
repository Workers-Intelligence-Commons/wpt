"""What the step to format 13 (wpt_depot.upgrade) makes of the libraries
and datasets of format 12: the snapshots that stand for them, worked out
by reading the legacy tables (schema.LEGACY), and checked against what the
legacy read (wpt_depot.legacy) gives. Nothing here writes: plan and measure
read through any connection, read-only or not, so `wpt depot upgrade
--check` runs them on a file it never changes, and the upgrade writes what
they give.

What converts (plan):

- each locked library this depot owns becomes a snapshot of one part: the
  texts it pinned, read at the change L that locked it (its first revision
  with locked_at), with the bounds of every depot's numbering this depot
  held at L (Depot.bounds); at_change is L. Its name is the library's name
  made a snapshot's name (sanitized: every character a name cannot hold
  becomes '-', the marks it cannot begin with are dropped, an empty name
  becomes `library`), with `-<the library's number>` added when another
  snapshot of the depot has it;
- each dataset revision named by a corpus record (named) and each
  dataset's current revision becomes a snapshot with one part per library
  it reads, in its order, each read as the legacy read reads it: a library
  locked at the change the revision reads it at, by its pins at L; another,
  by the texts that had joined it and not left it at that change (the change
  the revision was written at when the ref names none), each at its latest
  revision within what the depot held then. Its name is the dataset's name
  sanitized (`dataset` when empty), with `-r<revision>` for a revision
  other than the current one and `-<n>` (2, 3, ...) on a clash. A revision
  whose only library is one that converts as a locked library reuses that
  library's snapshot (reuses). A revision that reads a library whose rows
  this depot does not hold, of another depot, or that did not exist yet at
  the change it is read at, is not converted, and plan says why.

Open libraries are not converted: their rows stay in the legacy tables.

What a snapshot holds (measure): each part's texts that the legacy read
gives now, as roaring bitmaps of their numbers by owner, each owner's
bound that of the part's bounds; a text an erasure removed whole before
the upgrade has no row and is in no set. A text several parts hold is read
in the first of them, as a dataset read it in the first library holding it.
For every text of every part, the revision the legacy read gives (the pin
for a library locked then) must equal the revision a snapshot read gives
(the highest revision at or below the owner's bound): measure counts those
that differ and names the first EXAMPLES. The summary and the totals are
those of the snapshot read, each text counted once; the digest is
snapshots.digest_of's.

Every part is read in pages of PAGE texts, each one statement, so a check
on a depot in use never holds a long read transaction.
"""

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
from pyroaring import BitMap

from . import ids
from .depot import _bounds
from .kinds import PLAIN
from .snapshots import LARGEST, digest_of

PAGE = 50_000  # texts of one part a statement reads
EXAMPLES = 10  # texts whose revisions differ that measure names
TEXTS_KEY = "sqlite_autoindex_texts_revisions_1"
MEMBERSHIPS_KEY = "sqlite_autoindex_memberships_1"
PINS_KEY = "sqlite_autoindex_pins_1"
# the revision of the text h.text a library held at the change it is read at, {bound} the change of its owner's
# rows it held then (_bound): the legacy read's own expression (wpt_depot.legacy.REVISION_THEN) with the bound
# written out instead of looked up in a JSON object for each revision row
LEGACY_THEN = (f"(SELECT max(r.revision) FROM texts_revisions r INDEXED BY {TEXTS_KEY} WHERE r.id = h.text "
               "AND r.change <= {bound})")
# the revision a snapshot reads of the text {col}: its highest revision at or below its owner's bound {bound}
SNAPSHOT_THEN = (f"(SELECT max(s.revision) FROM texts_revisions s INDEXED BY {TEXTS_KEY} WHERE s.id = {{col}} "
                 "AND s.change <= {bound})")
# a pinned revision, when the depot still holds it
PINNED = f"(SELECT p.revision FROM texts_revisions p INDEXED BY {TEXTS_KEY} WHERE p.id = h.text AND p.revision = h.revision)"
OPEN_THEN = "h.joined_change <= :change{k} AND (h.left_change IS NULL OR h.left_change > :change{k})"
NAME_MARKS = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class Part:
    """A part of a converted snapshot: the library, whether it is read by
    its pins (locked when read), the change it is read at (L for a locked
    one, in the numbering of the depot that owns it) and the bounds of every
    depot's numbering then."""
    library: str
    locked: bool
    change: int
    bounds: dict[str, int]


@dataclass
class Planned:
    """A snapshot the step makes: its name, origin (schema: snapshots.origin),
    at_change and parts; for a dataset revision, its row (name, shares,
    max_tokens, vocabulary, tokenizer), why it converts ('current' or the
    ids of the corpora naming it) and the library whose snapshot it reuses,
    if any."""
    name: str
    origin: dict
    at_change: int
    parts: list[Part]
    row: dict | None = None
    why: str = ""
    reuses: str | None = None


@dataclass
class Measured:
    """What a planned snapshot holds (the module says how it is read)."""
    sets: list[tuple[int, str, int, BitMap]]
    summary: dict
    texts: int
    characters: int
    bytes: int
    digest: str
    rows: int  # rows of the legacy tables read
    differ: int  # texts whose two revisions differ
    examples: list[str] = field(default_factory=list)  # 'id: legacy revision R, snapshot revision S'
    seconds: float = 0.0
    parts: list[dict] = field(default_factory=list)  # {library, locked, texts, differ} of each part


def sanitized(name: str, empty: str) -> str:
    """name made a snapshot's name, as the module says."""
    return NAME_MARKS.sub("-", name).lstrip("._-") or empty


def _owner(con) -> str:
    return con.execute("SELECT name FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]


def _lock_change(con, library: str) -> int | None:
    found = con.execute("SELECT change FROM libraries_revisions WHERE id = ? AND locked_at IS NOT NULL "
                        "ORDER BY revision LIMIT 1", (library,)).fetchall()
    return found[0][0] if found else None


def plan(con, named: dict[tuple[str, int], list[str]], reads_here=None) -> tuple[list[Planned], list[Planned],
                                                                               list[str]]:
    """(the locked libraries' snapshots, the dataset revisions' snapshots,
    why each dataset revision left out is), as the module says. named:
    {(dataset id, revision): ids of the corpora whose record names it}.
    reads_here(address, library): whether a library a dataset names by a
    depot's address is one of this depot's rows (wpt_depot.legacy.Legacy);
    without it, only refs that name no address read here. Names are given
    in this order: libraries by id, then datasets by id and revision, each
    avoiding the names of the depot's own snapshots not made by the step
    to format 13 and those given before it."""
    own = _owner(con)
    taken = {name for name, origin in con.execute(
        f"SELECT name, origin FROM snapshots WHERE {ids.owner_sql('id')} = ?", (own,)).fetchall()
        if json.loads(origin).get("kind") not in ("library", "dataset")} if _has(con, "snapshots") else set()
    libraries = []
    for library, name in con.execute(
            f"SELECT id, name FROM libraries_revisions WHERE latest AND locked_at IS NOT NULL "
            f"ORDER BY {ids.order_sql('id')}").fetchall():
        if ids.owner(library) != own:
            continue
        lock = _lock_change(con, library)
        chosen = sanitized(name, "library")
        if chosen in taken:
            chosen = f"{chosen}-{ids.number(library)}"
        chosen = _free(chosen, taken)
        taken.add(chosen)
        libraries.append(Planned(chosen, {"kind": "library", "library": library, "name": name, "lock_change": lock},
                                 lock, [Part(library, True, lock, _bounds(con, own, lock))]))
    locked = {one.origin["library"] for one in libraries}
    wanted = {}
    for dataset, revision in con.execute("SELECT id, revision FROM datasets_revisions WHERE latest").fetchall():
        wanted.setdefault((dataset, revision), []).append("current")
    for key, corpora in named.items():
        wanted.setdefault(key, []).extend(corpora)
    datasets, left = [], []
    for dataset, revision in sorted(wanted, key=lambda one: (ids.order_key(one[0]), one[1])):
        found = con.execute("SELECT name, change, latest, libraries, shares, max_tokens, vocabulary, tokenizer "
                            "FROM datasets_revisions WHERE id = ? AND revision = ?", (dataset, revision)).fetchall()
        if not found:
            left.append(f"revision {revision} of the dataset {dataset}, which a corpus names: the depot holds no such "
                        "revision")
            continue
        name, written, latest, refs, shares, max_tokens, vocabulary, tokenizer = found[0]
        if ids.owner(dataset) != own:
            continue
        parts, why = [], None
        for ref in json.loads(refs):
            library, change = ref["library"], ref.get("change")
            change = written if change is None else change
            if ref.get("depot") is not None and (reads_here is None or not reads_here(ref["depot"], library)):
                why = f"it reads the library {library} of another depot, {ref['depot']}"
                break
            state = con.execute("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                                "ORDER BY revision DESC LIMIT 1", (library, change)).fetchall()
            if not state:
                held = con.execute("SELECT 1 FROM libraries_revisions WHERE id = ? LIMIT 1", (library,)).fetchall()
                why = (f"it reads the library {library}, which did not exist yet at the change {change}" if held else
                       f"it reads the library {library}, whose rows this depot does not hold")
                break
            if state[0][0] is not None:
                lock = _lock_change(con, library)
                parts.append(Part(library, True, lock, _bounds(con, ids.owner(library), lock)))
            else:
                parts.append(Part(library, False, change, _bounds(con, ids.owner(library), change)))
        label = f"revision {revision} of the dataset {name} ({dataset})"
        if why is not None:
            left.append(f"{label}: {why}")
            continue
        chosen = sanitized(name, "dataset") + ("" if latest else f"-r{revision}")
        reuses = parts[0].library if len(parts) == 1 and parts[0].locked and parts[0].library in locked else None
        if reuses is None:
            chosen = _free(chosen, taken)
            taken.add(chosen)
        else:
            chosen = next(one.name for one in libraries if one.origin["library"] == reuses)
        datasets.append(Planned(chosen, {"kind": "dataset", "dataset": dataset, "revision": revision, "name": name},
                                written, parts, {"name": name, "shares": json.loads(shares) if shares else None,
                                                 "max_tokens": max_tokens, "vocabulary": vocabulary,
                                                 "tokenizer": tokenizer},
                                ", ".join(wanted[dataset, revision]), reuses))
    return libraries, datasets, left


def _free(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    number = 2
    while f"{name}-{number}" in taken:
        number += 1
    return f"{name}-{number}"


def _has(con, table: str) -> bool:
    return bool(con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchall())


def _bound(part: Part, col: str) -> str:
    """The change of the part's bounds for the owner of the text {col}
    (null for an owner it has no bound for), as SQL: a CASE over the
    owners, the numbers written out."""
    if not part.bounds:
        return "NULL"
    whens = " ".join(f"WHEN '{owner.replace(chr(39), chr(39) * 2)}' THEN {int(change)}"
                     for owner, change in sorted(part.bounds.items()))
    return f"(CASE {ids.owner_sql(col)} {whens} END)"


def _source(part: Part, k: int) -> str:
    """The rows (text, o, n, legacy, snapshot) of the part's library
    after :after, in the order of their ids, the first :page: o and n the
    text's owner and number, legacy the revision the legacy read gives
    (null when it gives none), snapshot the one a snapshot read gives."""
    snapshot = SNAPSHOT_THEN.format(bound=_bound(part, "h.text"), col="h.text")
    text = f"h.text AS text, {ids.owner_sql('h.text')} AS o, {ids.number_sql('h.text')} AS n"
    if part.locked:
        return (f"SELECT {text}, {PINNED} AS legacy, {snapshot} AS snapshot FROM pins h INDEXED BY {PINS_KEY} "
                f"WHERE h.library = :library{k} AND h.text > :after ORDER BY h.text LIMIT :page")
    legacy = LEGACY_THEN.format(bound=_bound(part, "h.text"))
    return (f"SELECT {text}, {legacy} AS legacy, {snapshot} AS snapshot FROM memberships h "
            f"INDEXED BY {MEMBERSHIPS_KEY} WHERE h.library = :library{k} AND h.text > :after AND "
            f"{OPEN_THEN.format(k=k)} ORDER BY h.text LIMIT :page")


def _held(part: Part, k: int, col: str) -> str:
    """Whether the part's set holds the text {col}, as SQL."""
    if part.locked:
        return (f"EXISTS (SELECT 1 FROM pins h INDEXED BY {PINS_KEY} WHERE h.library = :library{k} AND h.text = {col} "
                f"AND {PINNED} IS NOT NULL)")
    return (f"EXISTS (SELECT 1 FROM memberships h INDEXED BY {MEMBERSHIPS_KEY} WHERE h.library = :library{k} "
            f"AND h.text = {col} AND {OPEN_THEN.format(k=k)} AND {LEGACY_THEN.format(bound=_bound(part, 'h.text'))} "
            "IS NOT NULL)")


def _page_sql(parts: list[Part], k: int) -> str:
    earlier = " OR ".join(_held(parts[at], at, "x.text") for at in range(k)) or "0"
    return (f"WITH x AS MATERIALIZED ({_source(parts[k], k)}) "
            f"SELECT x.o, x.legacy IS NOT NULL AS held, ({earlier}) AS first_elsewhere, "
            f"coalesce(c.kind, '{PLAIN}') AS kind, t.tags, t.licenses, "
            "group_concat(CASE WHEN x.legacy IS NOT NULL THEN x.n END, ','), count(*), "
            "count(CASE WHEN x.legacy IS NOT NULL THEN x.n END), "
            "sum(x.legacy IS NOT x.snapshot), group_concat(CASE WHEN x.legacy IS NOT x.snapshot THEN x.text || ': legacy "
            "revision ' || coalesce(x.legacy, 'none') || ', snapshot revision ' || coalesce(x.snapshot, 'none') END, "
            "char(10)), count(t.id), coalesce(sum(c.length), 0), coalesce(sum(c.bytes), 0), max(x.text) "
            f"FROM x LEFT JOIN texts_revisions t INDEXED BY {TEXTS_KEY} ON t.id = x.text AND t.revision = x.snapshot "
            "LEFT JOIN content_lists c ON c.id = t.id AND c.version = t.content_version GROUP BY 1, 2, 3, 4, 5, 6")


def measure(con, planned: Planned, page: int = PAGE) -> Measured:
    """What the planned snapshot holds, read through con (the module says how)."""
    began = time.monotonic()
    params = {}
    for k, part in enumerate(planned.parts):
        params |= {f"library{k}": part.library, f"change{k}": part.change}
    sets, groups, rows, differ, examples, described = [], {}, 0, 0, [], []
    sets_of = {}  # a JSON array as text: its distinct values in order (_distinct)
    for k, part in enumerate(planned.parts):
        numbers: dict[str, list[np.ndarray]] = {}
        after, part_texts, part_differ = "", 0, 0
        statement = _page_sql(planned.parts, k)
        while True:
            found = con.execute(statement, params | {"after": after, "page": page}).fetchall()
            read = 0
            for (owner, held, elsewhere, kind, tagset, licenses, joined, count, numbered, wrong, named, contents,
                 characters, size, last) in found:
                read += count
                after = max(after, last)
                part_differ += wrong
                if named and len(examples) < EXAMPLES:
                    examples += named.split("\n")[:EXAMPLES - len(examples)]
                if not held:
                    continue
                values = np.fromstring(joined, dtype=np.int64, sep=",") if joined else np.zeros(0, np.int64)
                if numbered != count or values.size != count:
                    raise ValueError(f"a text of {owner} that the library {part.library} holds has no number")
                if values.size and values.max() > LARGEST:
                    raise ValueError(f"the text {owner}-{int(values.max())} has a number above {LARGEST}, which no "
                                     "snapshot holds")
                numbers.setdefault(owner, []).append(values)
                part_texts += count
                if not elsewhere and contents:
                    key = (owner, kind, _distinct(tagset, sets_of), _distinct(licenses, sets_of))
                    sums = groups.setdefault(key, [0, 0, 0])
                    sums[0] += contents
                    sums[1] += characters
                    sums[2] += size
            rows += read
            if read < page:
                break
        for owner in sorted(numbers, key=str.encode):
            bitmap = BitMap(np.concatenate(numbers[owner]).astype(np.uint32))
            bitmap.run_optimize()
            sets.append((k, owner, part.bounds[owner], bitmap))
        differ += part_differ
        described.append({"library": part.library, "locked": part.locked, "change": part.change,
                          "texts": part_texts, "differ": part_differ})
    summary = _summary(groups)
    texts = sum(sizes[0] for sizes in summary["kinds"].values())
    characters = sum(sizes[1] for sizes in summary["kinds"].values())
    size = sum(sizes[2] for sizes in summary["kinds"].values())
    return Measured(sets, summary, texts, characters, size, digest_of(sets, planned.at_change), rows, differ,
                    examples, time.monotonic() - began, described)


def _distinct(listed: str | None, known: dict) -> tuple[str, ...]:
    """The distinct values of a JSON array given as text (None for none),
    in order, kept in known by the text so each one is read once."""
    if listed not in known:
        known[listed] = tuple(sorted(set(json.loads(listed or "[]"))))
    return known[listed]


def _summary(groups: dict[tuple, list[int]]) -> dict:
    """The summary (wpt_depot.snapshots) of the texts counted once, grouped by (owner, kind, tags, licenses), the
    tags and the licenses each a tuple of distinct values in order."""
    found = {"kinds": {}, "tags": {}, "plain_tagsets": {}, "licenses": {}, "owners": {}}

    def add(into: dict, name: str, sizes: list[int]):
        held = into.setdefault(name, [0, 0, 0])
        for at, value in enumerate(sizes):
            held[at] += value

    for (owner, kind, tags, licenses), sizes in groups.items():
        add(found["kinds"], kind, sizes)
        for tag in tags:
            add(found["tags"], tag, sizes)
        if kind == PLAIN:
            add(found["plain_tagsets"], "\0".join(tags), sizes)
        for one in licenses:
            add(found["licenses"], one, sizes)
        found["owners"][owner] = found["owners"].get(owner, 0) + sizes[0]
    return {"version": 1, **{key: dict(sorted(value.items())) for key, value in found.items()}}


def set_bytes(measured: Measured) -> int:
    """The bytes the snapshot's sets take, serialized as the step writes them."""
    return sum(len(bitmap.serialize()) for _, _, _, bitmap in measured.sets)


def recipe_data(planned: Planned, measured: Measured, depot: str, vocab_size: int | None) -> dict:
    """The data section of a model recipe (wpt-recipes, the snapshot shape)
    that builds what the dataset revision built: its snapshot, change and
    digest, the kinds a dataset's corpus took, its shares, its max_tokens as
    total_tokens and its vocabulary (sha256 '' and size 0 when it named
    none: a vocabulary has to be chosen)."""
    row = planned.row or {}
    found = {"depot": depot, "snapshot": planned.name, "change": planned.at_change, "digest": measured.digest,
             "total_tokens": row.get("max_tokens"), "vocabulary": {"sha256": row.get("vocabulary") or "",
                                                                   "size": vocab_size or 0},
             "kinds": ["conversation", PLAIN, "task"]}
    if row.get("shares"):
        found["shares"] = [{"tag": one["tag"], "share": one["share"]} for one in row["shares"]]
    return found
