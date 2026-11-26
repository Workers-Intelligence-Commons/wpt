"""What the step to format 13 (depot.upgrade) makes of the libraries
and datasets of format 12: the snapshots that stand for them, worked out
by reading the legacy tables (schema.LEGACY), and checked against what the
legacy read (depot.legacy) gives. Nothing here writes: plan and measure
read through any connection, read-only or not, so `depot upgrade
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

How it is read (measure_all): every planned snapshot at once, in one
sorted merge of four sequential scans (SCANS), each sorted by text id by
SQLite in its temporary folder: every revision of every text, every
content version, the pins of the locked libraries read and the
memberships of the open ones, each sort holding only the columns the
merge needs. The revisions the snapshots read are kept in a table of the
connection's temporary database (CHOSEN_TABLE), then read back in the
order of their rowids with their tags and licenses (CHOSEN), a forward
read of texts_revisions that reads each of its pages once. Each table is
read in the order of the file, whatever how many snapshots and parts, and
no row is looked up at random through an index: on a file far bigger than
the memory, each lookup is a random read of the disk, and a read by
lookups takes one for each revision of each text of each part. The read
holds one read transaction for its whole length, so a depot in use
meanwhile keeps in its WAL what is written until it ends. The sorts and
the table take about SORTED bytes of temporary files for each revision
and each content version.
"""

import json
import re
import time
from array import array
from dataclasses import dataclass, field

import numpy as np
from pyroaring import BitMap

from . import ids
from .depot import _bounds
from .kinds import PLAIN
from .snapshots import LARGEST, digest_of

EXAMPLES = 10  # texts whose revisions differ that measure names
# the four sequential scans measure merges, each sorted by text id by SQLite in its temporary folder: every revision
# of every text, every content version, the pins of the locked libraries read and the memberships of the open ones
# (their ids a JSON array as the parameter); NOT INDEXED and the unary plus keep SQLite from reading a table through
# an index, whose order is not the file's
SCANS = ("SELECT id, revision, change, content_version, rowid FROM texts_revisions NOT INDEXED "
         "ORDER BY +id, +revision",
         "SELECT id, version, kind, length, bytes FROM content_lists NOT INDEXED ORDER BY +id, +version",
         "SELECT text, library, revision FROM pins NOT INDEXED WHERE library IN (SELECT value FROM json_each(?)) "
         "ORDER BY +text, +library",
         "SELECT text, library, joined_change, left_change FROM memberships NOT INDEXED WHERE library IN "
         "(SELECT value FROM json_each(?)) ORDER BY +text, +library, +joined_change")
# the revisions the snapshots read, kept in a table of the connection's temporary database while the merge runs: the
# rowid of the revision, the snapshots that read it (an index into a list the merge keeps), the kind, length and bytes
# of its content version; then read back in the order of the rowids with each revision's text, tags and licenses,
# which the sort of SCANS leaves out (a forward read of texts_revisions, each page once)
CHOSEN_TABLE = ("CREATE TEMP TABLE measure_chosen (revision INTEGER NOT NULL, snapshots INTEGER NOT NULL, "
                "kind TEXT NOT NULL, length INTEGER NOT NULL, bytes INTEGER NOT NULL)")
CHOSEN = ("WITH c AS MATERIALIZED (SELECT * FROM temp.measure_chosen ORDER BY revision) "
          "SELECT c.snapshots, c.kind, c.length, c.bytes, t.id, t.tags, t.licenses FROM c "
          "CROSS JOIN texts_revisions t ON t.rowid = c.revision")
CHOSEN_BATCH = 100_000  # rows of measure_chosen written at once
# bytes of the temporary files the sorts of SCANS and the table of the revisions read take at once for each revision
# of a text and each content version, about (bench/large.py's depot: 1.38 GB at most for 18,322,341 revisions and
# 4,624,673 content versions)
SORTED = 70
# the pragmas of the connection while measure reads: a page cache of 64 MiB, which each of the four sorts also takes
# as the most memory a run it sorts holds before it writes it to a temporary file (about 256 MiB in all), and no
# thread to help the sorts, whose merges would then write their runs to temporary files once more (bench/large.py's
# depot: 2.4 GB of temporary files written with no thread, 5.5 GB with one, for 6 seconds more out of 70)
MEASURE_PRAGMAS = (("cache_size", -(64 << 10)), ("threads", 0))
SAY_EVERY = 30  # seconds between two lines of progress
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
    depot's address is one of this depot's rows (depot.legacy.Legacy);
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


def key_of(planned: Planned):
    """The key a planned snapshot's measure is known by: its library's id, or (dataset id, revision)."""
    origin = planned.origin
    return origin["library"] if origin["kind"] == "library" else (origin["dataset"], origin["revision"])


def measure_all(con, planned: list[Planned], log=None) -> dict:
    """What each planned snapshot holds, read through con in one sorted
    merge of the four scans of SCANS, whatever how many snapshots and parts:
    {key_of(planned): Measured}. log, when given, is told the progress every
    SAY_EVERY seconds. Its seconds are the merge's, shared out among the
    snapshots by the rows each part read."""
    began = said = time.monotonic()
    parts, index = [], {}  # each distinct part read once: (library, locked, change, bounds)
    for one in planned:
        for part in one.parts:
            index.setdefault((part.library, part.locked, part.change, tuple(sorted(part.bounds.items()))),
                             len(index))
            if len(parts) < len(index):
                parts.append(part)
    reads = [[index[(part.library, part.locked, part.change, tuple(sorted(part.bounds.items())))]
              for part in one.parts] for one in planned]
    locked = sorted({part.library for part in parts if part.locked})
    opened = sorted({part.library for part in parts if not part.locked})
    by_library = {}
    for at, part in enumerate(parts):
        by_library.setdefault((part.library, part.locked), []).append(at)
    # what each part read: rows, texts held, revisions that differ, examples, numbers by owner
    rows, held_count, differ, examples = [0] * len(parts), [0] * len(parts), [0] * len(parts), [[] for _ in parts]
    numbers: list[dict[str, array]] = [{} for _ in parts]
    groups = [{} for _ in planned]  # each snapshot's sums by (owner, kind, tags, licenses)
    sets_of = {}  # a JSON array as text: its distinct values in order (_distinct)
    saved, cursors = {}, []
    sets_at, chosen = {}, []  # each list of snapshots that read a revision: its index; rows for measure_chosen
    for pragma, value in MEASURE_PRAGMAS:
        saved[pragma] = con.execute(f"PRAGMA {pragma}").fetchall()[0][0]
        con.execute(f"PRAGMA {pragma} = {value}")
    try:
        con.execute("DROP TABLE IF EXISTS temp.measure_chosen")
        con.execute(CHOSEN_TABLE)
        cursors = [con.cursor() for _ in SCANS]
        revisions = cursors[0].execute(SCANS[0])
        contents = cursors[1].execute(SCANS[1])
        pins = cursors[2].execute(SCANS[2], (json.dumps(locked),))
        members = cursors[3].execute(SCANS[3], (json.dumps(opened),))
        revision_row, content_row = next(revisions, None), next(contents, None)
        pin_row, member_row = next(pins, None), next(members, None)
        if log is not None:
            log(f"the four scans read and sorted in {time.monotonic() - began:.0f} seconds; merging them")
        done = 0
        while pin_row is not None or member_row is not None:
            text = (member_row[0] if pin_row is None else pin_row[0] if member_row is None
                    else min(pin_row[0], member_row[0]))
            pinned, joined = {}, {}
            while pin_row is not None and pin_row[0] == text:
                pinned[pin_row[1]] = pin_row[2]
                pin_row = next(pins, None)
            while member_row is not None and member_row[0] == text:
                joined.setdefault(member_row[1], []).append(member_row[2:])
                member_row = next(members, None)
            while revision_row is not None and revision_row[0] < text:
                revision_row = next(revisions, None)
            held_revisions = []  # (revision, change, content version, rowid), by revision
            while revision_row is not None and revision_row[0] == text:
                held_revisions.append(revision_row[1:])
                revision_row = next(revisions, None)
            while content_row is not None and content_row[0] < text:
                content_row = next(contents, None)
            versions = {}
            while content_row is not None and content_row[0] == text:
                versions[content_row[1]] = content_row[2:]
                content_row = next(contents, None)
            owner, number = ids.owner(text), ids.number(text)
            latest = {}  # bound: the revision at or below it (the row), None for none

            def at(bound):
                if bound is None:
                    return None
                if bound not in latest:
                    latest[bound] = next((one for one in reversed(held_revisions) if one[1] <= bound), None)
                return latest[bound]

            found = {}  # part: [(legacy revision, the snapshot's revision row)] of each row it read of the text
            for library, revision in pinned.items():
                legacy = revision if any(one[0] == revision for one in held_revisions) else None
                for part_at in by_library.get((library, True), ()):
                    found[part_at] = [(legacy, at(parts[part_at].bounds.get(owner)))]
            for library, spans in joined.items():
                for part_at in by_library.get((library, False), ()):
                    change = parts[part_at].change
                    row = at(parts[part_at].bounds.get(owner))
                    found[part_at] = [(None if row is None else row[0], row) for start, end in spans
                                      if start <= change and (end is None or end > change)]
            for part_at, read in found.items():
                part = parts[part_at]
                rows[part_at] += len(read)
                for legacy, row in read:
                    snapshot = None if row is None else row[0]
                    if legacy != snapshot:
                        differ[part_at] += 1
                        if len(examples[part_at]) < EXAMPLES:
                            examples[part_at].append(f"{text}: legacy revision {legacy or 'none'}, snapshot revision "
                                                     f"{snapshot or 'none'}")
                    if legacy is None:
                        continue
                    if number is None:
                        raise ValueError(f"a text of {owner} that the library {part.library} holds has no number")
                    if number > LARGEST:
                        raise ValueError(f"the text {owner}-{number} has a number above {LARGEST}, which no snapshot "
                                         "holds")
                    numbers[part_at].setdefault(owner, array("q")).append(number)
                    held_count[part_at] += 1
            readers = {}  # the rowid of each revision the snapshots read: the snapshots that read it, each once a row
            for one, read_parts in enumerate(reads):
                for part_at in read_parts:
                    read = found.get(part_at)
                    if not read:
                        continue
                    kept = [row for legacy, row in read if legacy is not None]
                    for row in kept:
                        if row is not None:
                            readers.setdefault(row, []).append(one)
                    if kept:
                        break  # a text several parts hold is read in the first of them
            for row, which in readers.items():
                content = versions.get(row[2])
                kind = PLAIN if content is None or content[0] is None else content[0]
                key = tuple(which)
                if key not in sets_at:
                    sets_at[key] = len(sets_at)
                chosen.append((row[3], sets_at[key], kind, 0 if content is None else content[1] or 0,
                               0 if content is None else content[2] or 0))
            if len(chosen) >= CHOSEN_BATCH:
                con.executemany("INSERT INTO temp.measure_chosen VALUES (?, ?, ?, ?, ?)", chosen)
                chosen.clear()
            done += 1
            if log is not None and time.monotonic() - said >= SAY_EVERY:
                said = time.monotonic()
                log(f"{done:,} texts of the libraries read, the last {text}, {done / max(said - began, 1e-9):,.0f} "
                    f"texts a second, {said - began:.0f} seconds")
        for cursor in cursors:
            cursor.close(force=True)
        cursors = []
        con.executemany("INSERT INTO temp.measure_chosen VALUES (?, ?, ?, ?, ?)", chosen)
        chosen.clear()
        if log is not None:
            log(f"the texts merged in {time.monotonic() - began:.0f} seconds; reading the tags and licenses of the "
                "revisions the snapshots read, in the order of the file")
        snapshots_of = list(sets_at)
        for at_set, kind, length, size, text, tags, licenses in con.execute(CHOSEN):
            key = (ids.owner(text), kind, _distinct(tags, sets_of), _distinct(licenses, sets_of))
            for one in snapshots_of[at_set]:
                sums = groups[one].setdefault(key, [0, 0, 0])
                sums[0] += 1
                sums[1] += length
                sums[2] += size
    finally:
        for cursor in cursors:  # the scans not read to their end, whose read would stay open
            cursor.close(force=True)
        con.execute("DROP TABLE IF EXISTS temp.measure_chosen")
        for pragma, value in saved.items():
            con.execute(f"PRAGMA {pragma} = {value}")
    seconds = time.monotonic() - began
    total = max(1, sum(rows[part_at] for read_parts in reads for part_at in read_parts))
    measured = {}
    for one, (planned_one, read_parts) in enumerate(zip(planned, reads)):
        sets, described, said_examples = [], [], []
        for k, (part, part_at) in enumerate(zip(planned_one.parts, read_parts)):
            for owner in sorted(numbers[part_at], key=str.encode):
                bitmap = BitMap(np.frombuffer(numbers[part_at][owner], dtype=np.int64).astype(np.uint32))
                bitmap.run_optimize()
                sets.append((k, owner, part.bounds[owner], bitmap))
            said_examples += examples[part_at][:EXAMPLES - len(said_examples)]
            described.append({"library": part.library, "locked": part.locked, "change": part.change,
                              "texts": held_count[part_at], "differ": differ[part_at]})
        summary = _summary(groups[one])
        read = sum(rows[part_at] for part_at in read_parts)
        measured[key_of(planned_one)] = Measured(
            sets, summary, sum(sizes[0] for sizes in summary["kinds"].values()),
            sum(sizes[1] for sizes in summary["kinds"].values()), sum(sizes[2] for sizes in summary["kinds"].values()),
            digest_of(sets, planned_one.at_change), read, sum(differ[part_at] for part_at in read_parts), said_examples,
            seconds * read / total, described)
    return measured


def _distinct(listed: str | None, known: dict) -> tuple[str, ...]:
    """The distinct values of a JSON array given as text (None for none),
    in order, kept in known by the text so each one is read once."""
    if listed not in known:
        known[listed] = tuple(sorted(set(json.loads(listed or "[]"))))
    return known[listed]


def _summary(groups: dict[tuple, list[int]]) -> dict:
    """The summary (depot.snapshots) of the texts counted once, grouped by (owner, kind, tags, licenses), the
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
    """The data section of a model order (wpt-orders, the snapshot shape)
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
