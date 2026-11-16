"""The depot's counts: how many current texts it holds, and of them how many
have each status, each tag, each license and each score name, how many each
library holds now, how many each decision is on and how many pins of each
locked library are older than their text's current revision, with what each
library holds by kind and by tag in characters and bytes, kept in the table
counts (wpt_depot.schema) so that a page reads them at once whatever the
depot's size; and the listings of the current texts of each tag and of each
license (LISTINGS: the tables text_tags and text_licenses), from which a page
reads the texts of one of them in the numeric order of ids.

This module is the only code that writes those tables. Each write of the
depot keeps them right in its own transaction: it measures what the texts it
changes add to the counts before and after it changes them (changing), or
what the memberships it adds or ends add (held), and shifts the counts by
the difference (shift); changing also takes the texts out of the listings
before the change (unlist) and puts them back as they are after it
(enlist). recount makes the table again from the depot's
rows, which `wpt depot recount` runs to repair it and a clone runs once it
has copied its upstream.

A row is (facet, name, texts, characters, bytes): facet TEXTS with the name
'' (every current text), STATUS with a status (wpt_depot.policy), TAG with
a tag, LICENSE with a license id a text names, LIBRARY with a library id
(its memberships that have not ended, locked libraries included) and
DECISION with a decision id (its rows of text_decisions), SCORE with a
score name, NEWER with a locked library id (its pins whose text has a newer
current revision), whose characters and bytes stay 0; the facets of SPLIT,
which split the texts of TEXTS, TAG, LICENSE and LIBRARY by status and by
whether a locked library holds them (split_name): SPLIT_TEXTS, SPLIT_TAG,
SPLIT_LICENSE and SPLIT_LIBRARY with the name `<status> <locked> <name>`,
locked 1 or 0 and name the facet's own ('' for every text), so a filter of
one status, one tag, license or library, and whether a locked library holds
the text, is counted from a few rows read by their key (split); and the
facets of SIZED, which say what each library
holds, each text at its current revision, its characters and UTF-8 bytes
those of the content version it reads (content_lists): LIBRARY_KIND with
the name `<library id> <kind>`, kind plain (a plain text), items (a
conversation or a task, the items a corpus takes, TRAINING_KINDS) or other
(an item of another kind), and LIBRARY_TAG with the name `<library id>
<tag>`, the plain texts of the library that have the tag. A library id
holds no space, so the name's first space ends it. A text with several tags
counts once for each of them. A count that falls to 0 has no row.

Keys of a delta (shift): (facet, name) for its texts, (facet, name,
"characters") and (facet, name, "bytes") for the other two columns.

The table library_moves, which this module alone writes too, holds for each
library the change at which what it holds (its rows of HELD) last moved:
shift records the change being written (the highest id of changes, the row
each write adds first, wpt_depot.database) for every library whose rows it
moves. Its change is null when that change is not known, and a library
with no row has never held a text. A library whose row names a change at
or before a change C holds now what it held at C, each text at its
revision then, so its counts now are its figures at C (Depot._figures).
fill records, for every library that holds or held a text, the latest
change any of its memberships joined or left at, or any text it holds now
was revised at, which is at or after its last move; in a depot that
follows an upstream, whose rows carry the numberings of several depots,
it records null instead.
"""

import contextlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence

from .filters import in_locked
from .ids import order_sql
from .order import TRAINING_KINDS
from .schema import ADMITTED, REFUSED, UNDER_REVIEW

TEXTS, STATUS, TAG, LICENSE, LIBRARY, DECISION = "texts", "status", "tag", "license", "library", "decision"
SCORE, NEWER = "score", "pinned newer"
LIBRARY_KIND, LIBRARY_TAG = "library kind", "library tag"
COUNTED = (TEXTS, STATUS, TAG, LICENSE, LIBRARY, DECISION, SCORE, NEWER)  # the facets of counts of texts alone (read)
SPLIT_TEXTS, SPLIT_TAG, SPLIT_LICENSE, SPLIT_LIBRARY = "texts split", "tag split", "license split", "library split"
SPLITS = {TEXTS: SPLIT_TEXTS, TAG: SPLIT_TAG, LICENSE: SPLIT_LICENSE, LIBRARY: SPLIT_LIBRARY}  # each facet's split
SPLIT = tuple(SPLITS.values())  # the facets split by status and by whether a locked library holds the text
SIZED = (LIBRARY_KIND, LIBRARY_TAG)  # the facets with characters and bytes
FACETS = (*COUNTED, *SPLIT, *SIZED)
HELD = (LIBRARY, *SIZED)  # the facets of what libraries hold (library_moves), measured from rows of memberships
ROWS = (*HELD, SPLIT_LIBRARY)  # the facets measured from rows of memberships
KINDS = ("plain", "items", "other")  # the kinds of LIBRARY_KIND
COLUMNS = ("texts", "characters", "bytes")

# what each facet adds of the texts of a table of ids `{chosen}` (column id): (facet name, count) rows; each reads
# from the chosen texts' side (CROSS JOIN fixes that order), each text by its key
_MEASURES = {
    TEXTS: "SELECT '', count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest",
    STATUS: "SELECT t.status, count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest "
            "GROUP BY t.status",
    TAG: "SELECT j.value, count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest, "
         "json_each(t.tags) j GROUP BY j.value",
    LICENSE: "SELECT j.value, count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest, "
             "json_each(t.licenses) j GROUP BY j.value",
    DECISION: "SELECT l.decision, count(*) FROM {chosen} c CROSS JOIN text_decisions l ON l.text = c.id "
              "GROUP BY l.decision",
    SCORE: "SELECT s.value ->> 'name', count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id "
           "AND t.latest, json_each(t.scores) s GROUP BY 1",
    NEWER: "SELECT p.library, count(*) FROM {chosen} c CROSS JOIN pins p ON p.text = c.id CROSS JOIN texts_revisions t "
           "ON t.id = c.id AND t.latest WHERE t.revision > p.revision GROUP BY p.library",
    # {locked}: whether a locked library holds the text t (_locked)
    SPLIT_TEXTS: "SELECT t.status || ' ' || {locked} || ' ', count(*) FROM {chosen} c CROSS JOIN texts_revisions t "
                 "ON t.id = c.id AND t.latest GROUP BY 1",
    SPLIT_TAG: "SELECT t.status || ' ' || {locked} || ' ' || j.value, count(*) FROM {chosen} c CROSS JOIN "
               "texts_revisions t ON t.id = c.id AND t.latest, json_each(t.tags) j GROUP BY 1",
    SPLIT_LICENSE: "SELECT t.status || ' ' || {locked} || ' ' || j.value, count(*) FROM {chosen} c CROSS JOIN "
                   "texts_revisions t ON t.id = c.id AND t.latest, json_each(t.licenses) j GROUP BY 1",
}
# what the rows of memberships a query `{rows}` gives (columns library and text) hold, in one pass that looks up each
# row's current revision and content version once: (library, its kind as LIBRARY_KIND names it, NULL for a text with
# no current revision or no content version, the tags as JSON text of a plain text, NULL for another, the text's
# status, NULL with no current revision, whether a locked library holds it ({locked}, _locked), count, sum of
# characters, sum of bytes) rows, few since they group by library, kind, set of tags, status and locked
_HELD = ("SELECT h.library, CASE WHEN c.id IS NULL THEN NULL WHEN c.kind IS NULL THEN 'plain' WHEN c.kind IN ("
         + ", ".join(f"'{one}'" for one in TRAINING_KINDS) + ") THEN 'items' ELSE 'other' END, "
         "CASE WHEN c.id IS NOT NULL AND c.kind IS NULL THEN t.tags END, t.status, {locked}, count(*), sum(c.length), "
         "sum(c.bytes) FROM ({rows}) h LEFT JOIN texts_revisions t ON t.id = h.text AND t.latest "
         "LEFT JOIN content_lists c ON c.id = t.id AND c.version = t.content_version GROUP BY 1, 2, 3, 4, 5")
# LIBRARY and SPLIT_LIBRARY alone, with no lookup of a content version: (library, status, locked, count) rows
_LIBRARIES = ("SELECT h.library, t.status, {locked}, count(*) FROM ({rows}) h "
              "LEFT JOIN texts_revisions t ON t.id = h.text AND t.latest GROUP BY 1, 2, 3")
# the memberships that have not ended of the texts of a table of ids `{chosen}`, each text's read through the index
# memberships_held (by text), never a library's rows
_CHOSEN_ROWS = ("SELECT m.library, m.text FROM {chosen} x CROSS JOIN memberships m INDEXED BY memberships_held "
                "ON m.text = x.id AND m.left_change IS NULL")
# the same over every text of the depot, for the facets of COUNTED
_WHOLE = {
    TEXTS: "SELECT '', count(*) FROM texts_revisions WHERE latest",
    STATUS: "SELECT status, count(*) FROM texts_revisions WHERE latest GROUP BY status",
    TAG: "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.tags) j WHERE t.latest GROUP BY j.value",
    LICENSE: "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.licenses) j WHERE t.latest "
             "GROUP BY j.value",
    LIBRARY: "SELECT library, count(*) FROM memberships WHERE left_change IS NULL GROUP BY library",
    DECISION: "SELECT decision, count(*) FROM text_decisions GROUP BY decision",
    SCORE: "SELECT s.value ->> 'name', count(*) FROM texts_revisions t, json_each(t.scores) s WHERE t.latest GROUP BY 1",
    NEWER: "SELECT p.library, count(*) FROM pins p JOIN texts_revisions t ON t.id = p.text AND t.latest "
           "WHERE t.revision > p.revision GROUP BY p.library",
    SPLIT_TEXTS: "SELECT t.status || ' ' || {locked} || ' ', count(*) FROM texts_revisions t WHERE t.latest GROUP BY 1",
    SPLIT_TAG: "SELECT t.status || ' ' || {locked} || ' ' || j.value, count(*) FROM texts_revisions t, "
               "json_each(t.tags) j WHERE t.latest GROUP BY 1",
    SPLIT_LICENSE: "SELECT t.status || ' ' || {locked} || ' ' || j.value, count(*) FROM texts_revisions t, "
                   "json_each(t.licenses) j WHERE t.latest GROUP BY 1",
    SPLIT_LIBRARY: "SELECT t.status || ' ' || {locked} || ' ' || m.library, count(*) FROM memberships m CROSS JOIN "
                   "texts_revisions t ON t.id = m.text AND t.latest WHERE m.left_change IS NULL GROUP BY 1",
}
# the stored listings (LISTINGS): for each facet listed, its table, the column of the name and the column of the
# current revision that lists the names (a JSON array)
LISTINGS = {TAG: ("text_tags", "tag", "tags"), LICENSE: ("text_licenses", "license", "licenses")}
# the rows of a listing of the texts of a table of ids `{chosen}`, at their current revisions
_LISTED = ("SELECT DISTINCT j.value, " + order_sql("t.id") + " FROM {chosen} c CROSS JOIN texts_revisions t "
           "ON t.id = c.id AND t.latest, json_each(t.{column}) j")


def measure(con, chosen: str, facets: Iterable[str]) -> Counter:
    """The delta (the module says its keys) of what the current texts whose
    ids the table chosen holds (its column id) add to the counts of each
    facet named."""
    facets = tuple(facets)
    found = Counter()
    rows = _CHOSEN_ROWS.format(chosen=chosen)
    if any(facet in SIZED for facet in facets):
        found.update({key: by for key, by in held(con, rows).items() if key[0] in facets})
    elif LIBRARY in facets or SPLIT_LIBRARY in facets:
        for library, status, locked, count in con.execute(_LIBRARIES.format(rows=rows,
                                                                            locked=_locked(con, "h.text"))):
            if LIBRARY in facets:
                found[LIBRARY, library] += count
            if SPLIT_LIBRARY in facets and status is not None:
                found[SPLIT_LIBRARY, split_name(status, locked, library)] += count
    locked = _locked(con, "t.id")
    for facet in facets:
        if facet not in ROWS:
            _add(found, facet, con.execute(_MEASURES[facet].format(chosen=chosen, locked=locked)))
    return found


def split_name(status: str, locked, name: str) -> str:
    """The name of a row of a facet of SPLIT: the texts of that status, held
    by a locked library or not (locked), of that name of the facet split."""
    return f"{status} {int(locked)} {name}"


def split(con, facet: str, name: str) -> dict[tuple[str, bool], int]:
    """{(status, locked): how many texts} of the name of a facet of SPLITS
    (TEXTS with the name ''), read from its rows of the split facet by their
    keys; a pair with no row is left out."""
    keys = {split_name(status, locked, name): (status, bool(locked))
            for status in (UNDER_REVIEW, ADMITTED, REFUSED) for locked in (0, 1)}
    return {keys[one]: count for one, count in con.execute(
        "SELECT name, texts FROM counts WHERE facet = ? AND name IN (SELECT value FROM json_each(?))",
        (SPLITS[facet], json.dumps(list(keys))))}


def _locked(con, text: str) -> str:
    """Whether a locked library holds the text whose id the SQL expression
    text gives, as an SQL expression of 1 or 0: 0 itself while no library is
    locked, so a depot with none looks nothing up."""
    if con.execute("SELECT 1 FROM libraries_revisions WHERE latest AND locked_at IS NOT NULL LIMIT 1").fetchall():
        return in_locked(text)
    return "0"


def held(con, rows: str, params=()) -> Counter:
    """The delta of what the rows of memberships that the query rows gives
    (columns library and text; params its parameters) add to the counts of
    HELD and of SPLIT_LIBRARY, as if none of them had ended: what a write that adds those rows
    shifts the counts by, or, negated, one that ends them. The query runs
    once, and each of its rows is looked up once."""
    found = Counter()
    for library, what, tags, status, locked, count, length, nbytes in con.execute(
            _HELD.format(rows=rows, locked=_locked(con, "h.text")), params):
        found[LIBRARY, library] += count
        if status is not None:
            found[SPLIT_LIBRARY, split_name(status, locked, library)] += count
        if what is None:
            continue
        names = [(LIBRARY_KIND, f"{library} {what}")]
        names += [(LIBRARY_TAG, f"{library} {tag}") for tag in json.loads(tags or "[]")]
        for facet, name in names:
            found[facet, name] += count
            found[facet, name, "characters"] += length or 0
            found[facet, name, "bytes"] += nbytes or 0
    return found


def _add(found: Counter, facet: str, rows):
    for name, count, *sizes in rows:
        if count:
            found[facet, name] += count
            if sizes:
                found[facet, name, "characters"] += sizes[0] or 0
                found[facet, name, "bytes"] += sizes[1] or 0


def shift(con, delta: dict):
    """Add delta (the module says its keys) to the counts, a row whose texts
    fall to 0 removed."""
    rows = {}
    for key, by in delta.items():
        if by:
            rows.setdefault(key[:2], [0, 0, 0])[COLUMNS.index(key[2]) if len(key) == 3 else 0] += by
    moved = [(facet, name, *by) for (facet, name), by in rows.items() if any(by)]
    if not moved:
        return
    libraries = sorted({name.split(" ", 1)[0] if facet in SIZED else name for facet, name, *_ in moved if facet in HELD})
    con.executemany("INSERT INTO library_moves (library, change) SELECT ?, max(id) FROM changes WHERE true "
                    "ON CONFLICT (library) DO UPDATE SET change = excluded.change", [(one,) for one in libraries])
    con.executemany("INSERT INTO counts (facet, name, texts, characters, bytes) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (facet, name) DO UPDATE SET texts = texts + excluded.texts, "
                    "characters = characters + excluded.characters, bytes = bytes + excluded.bytes", moved)
    con.executemany("DELETE FROM counts WHERE facet = ? AND name = ? AND texts = 0",
                    [(facet, name) for facet, name, *_ in moved])


@contextlib.contextmanager
def changing(con, chosen: str, facets: Iterable[str]) -> Iterator[None]:
    """The counts of the facets named kept right across the block, which
    changes the texts whose ids the table chosen holds: what they add is
    measured before and after it, and the counts shifted by the
    difference. The table must name the same texts at both ends, and no
    write inside the block may shift the same facets itself."""
    facets = tuple(facets)
    before = measure(con, chosen, facets)
    unlist(con, chosen, facets)
    yield
    after = measure(con, chosen, facets)
    after.subtract(before)
    shift(con, after)
    enlist(con, chosen, facets)


def enlist(con, chosen: str, facets: Iterable[str]):
    """Add to the listings of the facets named among LISTINGS the rows of
    the texts whose ids the table chosen holds, at their current
    revisions."""
    for facet in facets:
        if facet in LISTINGS:
            table, name, column = LISTINGS[facet]
            con.execute(f"INSERT OR IGNORE INTO {table} ({name}, key) "
                        + _LISTED.format(chosen=chosen, column=column))


def unlist(con, chosen: str, facets: Iterable[str]):
    """Take out of the listings of the facets named among LISTINGS the rows
    of the texts whose ids the table chosen holds, as their current
    revisions list them: each row read by its key."""
    for facet in facets:
        if facet in LISTINGS:
            table, name, column = LISTINGS[facet]
            con.execute(f"DELETE FROM {table} WHERE ({name}, key) IN ("
                        + _LISTED.format(chosen=chosen, column=column) + ")")


def recount(con, known: bool):
    """The counts made again from the depot's rows, every one of them (fill
    makes those of SIZED and library_moves; known: whether the depot's rows
    carry its own numbering alone, False in a depot that follows an
    upstream)."""
    con.execute("DELETE FROM counts")
    locked = _locked(con, "t.id")
    for facet in (*COUNTED, *SPLIT):
        con.execute(f"INSERT INTO counts (facet, name, texts) SELECT '{facet}', * FROM "
                    f"({_WHOLE[facet].format(locked=locked)})")
    con.execute("DELETE FROM counts WHERE texts = 0")
    for table, name, column in LISTINGS.values():
        con.execute(f"DELETE FROM {table}")
        con.execute(f"INSERT OR IGNORE INTO {table} ({name}, key) SELECT j.value, {order_sql('t.id')} "
                    f"FROM texts_revisions t, json_each(t.{column}) j WHERE t.latest")
    fill(con, known)


# what fill reads, each in the order of the texts' ids (+ keeps SQLite from reading an index for that order, and NOT
# INDEXED from reading the rows through one): every scan is sequential, and SQLite's sorter orders the rows
HELD_SCAN = "SELECT text, library FROM memberships NOT INDEXED WHERE left_change IS NULL ORDER BY +text"
CURRENT_SCAN = "SELECT id, content_version, tags, change FROM texts_revisions NOT INDEXED WHERE latest ORDER BY +id"
# the latest change each library's memberships joined or left at, read from the index memberships_key alone, which
# holds both columns in the order of the libraries
JOINED_OR_LEFT = ("SELECT library, max(max(joined_change, coalesce(left_change, 0))) FROM memberships "
                  "INDEXED BY memberships_key GROUP BY library")
VERSIONS_SCAN = "SELECT id, version, kind, length, bytes FROM content_lists NOT INDEXED ORDER BY +id, +version"
EVERY = 250_000  # memberships merged gives the sums of at once, about


def fill(con, known: bool):
    """The counts of SIZED made from every membership that has not ended,
    the rows of those facets the table held before replaced: three
    sequential scans (HELD_SCAN, CURRENT_SCAN, VERSIONS_SCAN) summed by
    merged, so the time it takes grows with the depot's rows, whatever
    order they are stored in; and library_moves made again as the module
    says (known: whether the depot's rows carry its own numbering alone),
    from merged and one read of the index memberships_key
    (JOINED_OR_LEFT)."""
    con.execute(f"DELETE FROM counts WHERE facet IN ('{LIBRARY_KIND}', '{LIBRARY_TAG}')")
    moves = dict(con.execute(JOINED_OR_LEFT).fetchall())
    for sums, _, revised in merged(con.execute(HELD_SCAN), con.execute(CURRENT_SCAN), con.execute(VERSIONS_SCAN)):
        shift(con, delta(sums))
        for library, change in revised.items():
            moves[library] = max(moves.get(library, 0), change)
    con.execute("DELETE FROM library_moves")
    con.executemany("INSERT INTO library_moves (library, change) VALUES (?, ?)",
                    [(library, change if known else None) for library, change in moves.items()])


def merged(held: Iterator, current: Iterator, versions: Iterator,
           every: int = EVERY) -> Iterator[tuple[dict, str, dict]]:
    """What the memberships held adds to the counts of SIZED, from three
    streams in the order of the texts' ids: held, (text, library) of each
    membership that has not ended; current, (id, content version, tags as
    JSON text, change) of each current revision; versions, (id, version,
    kind, length, bytes) of each content version, by id then version. It
    goes through the three at once, as one sorted merge, and gives (sums:
    {(facet, name): [texts, characters, bytes]}, the id of the last text
    they cover, revised: {library: the latest change a text it holds among
    them was revised at}) each time about every memberships are summed, at
    the end of a text, so the sums given cover the texts up to that id and
    none after; then the rest. A membership of a text with no current
    revision or no content version counts for nothing."""
    sums, summed, seen, size, revised, change = {}, 0, None, None, {}, None
    text_row, version_row = next(current, None), next(versions, None)
    for text, library in held:
        if text != seen:
            if summed >= every:
                yield sums, seen, revised
                sums, summed, revised = {}, 0, {}
            seen, size, change = text, None, None
            while text_row is not None and text_row[0] < text:
                text_row = next(current, None)
            if text_row is not None and text_row[0] == text:
                change = text_row[3]
                wanted = (text, text_row[1])
                while version_row is not None and (version_row[0], version_row[1]) < wanted:
                    version_row = next(versions, None)
                if version_row is not None and (version_row[0], version_row[1]) == wanted:
                    kind = version_row[2]
                    what = "plain" if kind is None else "items" if kind in TRAINING_KINDS else "other"
                    size = (what, version_row[3] or 0, version_row[4] or 0,
                            json.loads(text_row[2] or "[]") if kind is None else ())
        summed += 1
        if change is not None and revised.get(library, -1) < change:
            revised[library] = change
        if size is None:
            continue
        what, length, nbytes, tags = size
        for key in ((LIBRARY_KIND, f"{library} {what}"), *((LIBRARY_TAG, f"{library} {tag}") for tag in tags)):
            one = sums.get(key)
            if one is None:
                sums[key] = [1, length, nbytes]
            else:
                one[0] += 1
                one[1] += length
                one[2] += nbytes
    if sums or seen is not None:
        yield sums, seen, revised


def delta(sums: dict) -> Counter:
    """The sums merged gives as a delta (the module says its keys)."""
    found = Counter()
    for (facet, name), by in sums.items():
        for column, value in zip(COLUMNS, by):
            found[(facet, name) if column == "texts" else (facet, name, column)] += value
    return found


def read(con) -> dict[str, dict[str, int]]:
    """{facet: {name: count}} of every count of COUNTED, each facet present."""
    found = {facet: {} for facet in COUNTED}
    for facet, name, count in con.execute("SELECT facet, name, texts FROM counts WHERE facet IN "
                                          f"({', '.join(repr(one) for one in COUNTED)}) ORDER BY facet, name"):
        found[facet][name] = count
    return found


def unmoved(con, library: str, change: int) -> bool:
    """Whether the library holds now what it held at the change number
    change of this depot's numbering, as library_moves says (the module
    says how): no row, or a row whose change is known and at or before
    change."""
    found = con.execute("SELECT change FROM library_moves WHERE library = ?", (library,)).fetchall()
    return not found or (found[0][0] is not None and found[0][0] <= change)


def summed(found: Sequence[dict], tags: Sequence[str]) -> dict:
    """The figures of several libraries (each as figures gives it, for
    tags) as one: {texts, characters, bytes of them all; plain, items,
    other: each {texts, characters, bytes}; shares: for each tag of tags,
    in order, {tag, texts, characters, bytes}; unshared: the plain texts
    left once the shares are taken away, when tags are given (None
    otherwise)}, each library counted apart."""
    empty = dict.fromkeys(COLUMNS, 0)
    kinds = {kind: dict(empty) for kind in KINDS}
    shares = [dict(empty) for _ in tags]
    for one in found:
        for into, read in [*((kinds[kind], one[kind]) for kind in KINDS), *zip(shares, one["tags"], strict=True)]:
            for key in COLUMNS:
                into[key] += read[key]
    unshared = {key: max(0, kinds["plain"][key] - sum(one[key] for one in shares)) for key in COLUMNS}
    return {key: sum(one[key] for one in kinds.values()) for key in COLUMNS} | kinds | {
        "shares": [{"tag": tag, **shares[at]} for at, tag in enumerate(tags)],
        "unshared": unshared if tags else None}


def figures(con, library: str, tags: Sequence[str]) -> dict:
    """What the library holds, from the counts of SIZED: {plain, items,
    other: each {texts, characters, bytes}, tags: [the same for the plain
    texts that have each tag of tags, in order]}. It reads only those rows,
    by their key."""
    names = [(LIBRARY_KIND, f"{library} {kind}") for kind in KINDS] + [(LIBRARY_TAG, f"{library} {tag}") for tag in tags]
    found = []
    for facet, name in names:
        row = con.execute("SELECT texts, characters, bytes FROM counts WHERE facet = ? AND name = ?",
                          (facet, name)).fetchall()
        found.append(dict(zip(COLUMNS, row[0] if row else (0, 0, 0))))
    return dict(zip(KINDS, found)) | {"tags": found[len(KINDS):]}
