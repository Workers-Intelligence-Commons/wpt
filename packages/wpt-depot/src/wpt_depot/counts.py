"""The depot's counts: how many current texts it holds, and of them how many
have each status, each tag, each license and each score name, and how many
each decision is on, with what the texts of each status hold by kind, by
plain tag set, by tag and by license, in characters and bytes, kept in the
table counts (wpt_depot.schema) so that a page reads them at once whatever
the depot's size; and the listings of the current texts of each tag and of
each license (LISTINGS: the tables text_tags and text_licenses), from which
a page reads the texts of one of them in the numeric order of ids, and the
index of their titles (TITLE, wpt_depot.titles), which a plain search reads.

This module is the only code that writes those tables, and the index through
wpt_depot.titles. Each write of the depot keeps them right in its own
transaction: it measures what the texts it changes add to the counts before
and after it changes them (changing), and shifts the counts by the
difference (shift), except a decision, which changes statuses alone and
moves each text's rows from its old status to its new one in one pass
(moving); changing also takes the texts out of the listings
before the change (unlist) and puts them back as they are after it
(enlist), the title index too when TITLE is named among the facets. recount
makes the table again from the depot's rows, which `wpt depot recount` runs
to repair it and a clone runs once it has copied its upstream.

A row is (facet, name, texts, characters, bytes): facet TEXTS with the name
'' (every current text), STATUS with a status (wpt_depot.policy), TAG with
a tag, LICENSE with a license id a text names, DECISION with a decision id
(its rows of text_decisions) and SCORE with a score name, whose characters
and bytes stay 0; the facets of SPLIT, which split the texts of TEXTS, TAG
and LICENSE by status (split_name): SPLIT_TEXTS, SPLIT_TAG and
SPLIT_LICENSE with the name `<status> <name>`, name the facet's own (''
for every text), so a filter of one status and one tag or license is
counted from a few rows read by their key (split), the rows of SPLIT_TAG
and SPLIT_LICENSE with the characters and bytes of the content versions
their texts read; and the facets of STATUSED, which say what the current
texts of each status hold, in texts, characters and bytes: KIND with the
name `<status> <kind>`, and PLAIN_TAGSET with the name `<status> <tags>`,
the plain texts whose set of tags is exactly tags, a JSON array of them,
each once, in order (TAGS), so what the texts of a status hold by kind,
tag, set of tags and license is read from a few ranges of rows, and a
snapshot's summary is made without reading a text (of_status). A text with several tags
counts once for each of them. A text's tags are read from its rows of
taggings it has not left (wpt_depot.taggings), its other fields from its
current revision, the kind, characters and bytes of its content version
included (schema.SIZES): no count reads content_lists. A count that falls to 0 has no row.

Keys of a delta (shift): (facet, name) for its texts, (facet, name,
"characters") and (facet, name, "bytes") for the other two columns.
"""

import contextlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator

from . import titles
from .ids import order_sql
from .kinds import PLAIN
from .schema import ADMITTED, REFUSED, UNDER_REVIEW, sizes_sql, tags_sql

TEXTS, STATUS, TAG, LICENSE, DECISION, SCORE = "texts", "status", "tag", "license", "decision", "score"
KIND, PLAIN_TAGSET = "kind", "plain tagset"
COUNTED = (TEXTS, STATUS, TAG, LICENSE, DECISION, SCORE)  # the facets of counts of texts alone (read)
SPLIT_TEXTS, SPLIT_TAG, SPLIT_LICENSE = "texts split", "tag split", "license split"
SPLITS = {TEXTS: SPLIT_TEXTS, TAG: SPLIT_TAG, LICENSE: SPLIT_LICENSE}  # each facet's split
SPLIT = tuple(SPLITS.values())  # the facets split by status
STATUSED = (KIND, PLAIN_TAGSET)  # what the texts of each status hold by kind and by plain tag set, sized
BY_STATUS = (*SPLIT, *STATUSED)  # every facet whose rows move when a text's status does
# the facets with characters and bytes of the texts themselves, which move when a text's content version does
TEXT_SIZED = (*STATUSED, SPLIT_TAG, SPLIT_LICENSE)
FACETS = (*COUNTED, *SPLIT, *STATUSED)
COLUMNS = ("texts", "characters", "bytes")

# what each facet adds of the texts of a table of ids `{chosen}` (column id): (facet name, count) rows; each reads
# from the chosen texts' side (CROSS JOIN fixes that order), each text by its key
_MEASURES = {
    TEXTS: "SELECT '', count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest",
    STATUS: "SELECT t.status, count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest "
            "GROUP BY t.status",
    TAG: "SELECT g.tag, count(*) FROM {chosen} c CROSS JOIN taggings g ON g.text = c.id AND g.left_change IS NULL "
         "GROUP BY g.tag",
    LICENSE: "SELECT j.value, count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest, "
             "json_each(t.licenses) j GROUP BY j.value",
    DECISION: "SELECT l.decision, count(*) FROM {chosen} c CROSS JOIN text_decisions l ON l.text = c.id "
              "GROUP BY l.decision",
    SCORE: "SELECT s.value ->> 'name', count(*) FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id "
           "AND t.latest, json_each(t.scores) s GROUP BY 1",
    SPLIT_TEXTS: "SELECT t.status || ' ', count(*) FROM {chosen} c CROSS JOIN texts_revisions t "
                 "ON t.id = c.id AND t.latest GROUP BY 1",
}
# the tags the text of the current revision t has now, a JSON array of its tags, each once, in order: the copy the
# revision holds (schema.tags_sql); for a plain text, the same text names its row of PLAIN_TAGSET (_sized_names)
TAGS = "t.current_tags"
# what the texts of a table of ids `{chosen}` add to the facets of TEXT_SIZED, in one pass that looks up each text's
# current revision once, which holds the kind, length and bytes of the content version it reads (schema.SIZES):
# (status, its kind, NULL for no content version, its tags and licenses as JSON text, count, characters, bytes) rows
_TEXT_SIZED = (f"SELECT t.status, t.content_kind, {TAGS}, t.licenses, count(*), coalesce(sum(t.content_length), 0), "
               "coalesce(sum(t.content_bytes), 0) "
               "FROM {chosen} x CROSS JOIN texts_revisions t ON t.id = x.id AND t.latest GROUP BY 1, 2, 3, 4")
# the same over every text of the depot, for the facets of COUNTED
_WHOLE = {
    TEXTS: "SELECT '', count(*) FROM texts_revisions WHERE latest",
    STATUS: "SELECT status, count(*) FROM texts_revisions WHERE latest GROUP BY status",
    TAG: "SELECT tag, count(*) FROM taggings WHERE left_change IS NULL GROUP BY tag",
    LICENSE: "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.licenses) j WHERE t.latest "
             "GROUP BY j.value",
    DECISION: "SELECT decision, count(*) FROM text_decisions GROUP BY decision",
    SCORE: "SELECT s.value ->> 'name', count(*) FROM texts_revisions t, json_each(t.scores) s WHERE t.latest GROUP BY 1",
    SPLIT_TEXTS: "SELECT t.status || ' ', count(*) FROM texts_revisions t WHERE t.latest GROUP BY 1",
}
# the same for the facets of TEXT_SIZED, with characters and bytes (SUMS), from the current revisions alone
SUMS = "count(*), coalesce(sum(t.content_length), 0), coalesce(sum(t.content_bytes), 0)"
_WHOLE_SIZED = {
    SPLIT_TAG: f"SELECT t.status || ' ' || g.tag, {SUMS} FROM texts_revisions t "
               "JOIN taggings g ON g.text = t.id AND g.left_change IS NULL WHERE t.latest GROUP BY 1",
    SPLIT_LICENSE: f"SELECT t.status || ' ' || j.value, {SUMS} "
                   "FROM texts_revisions t, json_each(t.licenses) j WHERE t.latest GROUP BY 1",
    KIND: f"SELECT t.status || ' ' || t.content_kind, {SUMS} FROM texts_revisions t WHERE t.latest "
          "AND t.content_kind IS NOT NULL GROUP BY 1",
    PLAIN_TAGSET: f"SELECT t.status || ' ' || {TAGS}, {SUMS} FROM texts_revisions t "
                  f"WHERE t.latest AND t.content_kind = '{PLAIN}' GROUP BY 1",
}
# the stored listings (LISTINGS): for each facet listed, its table, the column of the name, and the rows of the
# listing of the texts of a table of ids `{chosen}` now (_LISTED): a text's tags from its rows of taggings, its
# licenses from the column of its current revision that lists them (a JSON array)
LISTINGS = {TAG: ("text_tags", "tag"), LICENSE: ("text_licenses", "license")}
# the index of the current texts' titles (wpt_depot.titles), named among the facets of changing, enlist and unlist like
# a listing, with no count; LISTED: every listing and the index, what a write that adds or removes texts names
TITLE = "title"
LISTED = (*LISTINGS, TITLE)
_LISTED = {TAG: ("SELECT DISTINCT g.tag, " + order_sql("c.id") + " FROM {chosen} c CROSS JOIN taggings g "
                 "ON g.text = c.id AND g.left_change IS NULL"),
           LICENSE: ("SELECT DISTINCT j.value, " + order_sql("t.id") + " FROM {chosen} c CROSS JOIN texts_revisions t "
                     "ON t.id = c.id AND t.latest, json_each(t.licenses) j")}
# the same over every current text, for recount
_LISTED_WHOLE = {TAG: f"SELECT tag, {order_sql('text')} FROM taggings WHERE left_change IS NULL",
                 LICENSE: f"SELECT j.value, {order_sql('t.id')} FROM texts_revisions t, json_each(t.licenses) j "
                          "WHERE t.latest"}


def measure(con, chosen: str, facets: Iterable[str]) -> Counter:
    """The delta (the module says its keys) of what the current texts whose
    ids the table chosen holds (its column id) add to the counts of each
    facet named."""
    facets = tuple(facets)
    found = Counter()
    for facet in facets:
        if facet in _MEASURES:
            _add(found, facet, con.execute(_MEASURES[facet].format(chosen=chosen)))
    if any(facet in TEXT_SIZED for facet in facets):
        _text_sized(con, chosen, facets, found)
    return found


def _text_sized(con, chosen: str, facets: tuple[str, ...], found: Counter):
    """Add to found what the texts of the table chosen add to the facets of
    TEXT_SIZED named among facets, from one pass (_TEXT_SIZED)."""
    for status, kind, tags, licenses, count, length, nbytes in con.execute(_TEXT_SIZED.format(chosen=chosen)):
        if status is not None:
            _add_sized(found, status, _sized_names(facets, kind, tags, licenses), count, length, nbytes)


def _sized_names(facets: tuple[str, ...], kind, tags, licenses) -> list[tuple[str, str]]:
    """(facet, name less its status) of each row of the facets of
    TEXT_SIZED named among facets that a text of that kind (None for no
    content version), tags and licenses (JSON arrays as text, as TAGS
    gives them) counts in; a plain text's tags as they are name its row of
    PLAIN_TAGSET."""
    names = []
    if KIND in facets and kind is not None:
        names.append((KIND, kind))
    if PLAIN_TAGSET in facets and kind == PLAIN:
        names.append((PLAIN_TAGSET, tags))
    for facet, listed in ((SPLIT_TAG, tags), (SPLIT_LICENSE, licenses)):
        if facet in facets:
            names += [(facet, one) for one in json.loads(listed or "[]")]
    return names


def _add_sized(found: Counter, status: str, names: list[tuple[str, str]], count: int, length: int, nbytes: int):
    """Add to found count texts holding length characters and nbytes bytes
    (each may be negative) in the rows of the status of names
    (_sized_names)."""
    for facet, name in names:
        key = split_name(status, name)
        found[facet, key] += count
        found[facet, key, "characters"] += length
        found[facet, key, "bytes"] += nbytes


def moving(con, chosen: str, new: str) -> Counter:
    """The delta of STATUS and of every facet of BY_STATUS when each current
    text whose id the table chosen holds (its column id) takes the status
    the SQL expression new over its current revision (t) gives: one pass
    that moves each text's rows from its old status to its new one, since a
    status change leaves its kind, tags, licenses and content as they are.
    Read before the statuses are written."""
    query = (f"SELECT t.status, {new}, t.content_kind, {TAGS}, t.licenses, count(*), "
             "coalesce(sum(t.content_length), 0), coalesce(sum(t.content_bytes), 0) "
             "FROM {chosen} x CROSS JOIN texts_revisions t ON t.id = x.id AND t.latest "
             f"WHERE t.status IS NOT {new} GROUP BY 1, 2, 3, 4, 5").replace("{chosen}", chosen)
    found = Counter()
    for old, now, kind, tags, licenses, count, length, nbytes in con.execute(query):
        names = _sized_names(TEXT_SIZED, kind, tags, licenses)
        for status, sign in ((old, -1), (now, 1)):
            found[STATUS, status] += sign * count
            found[SPLIT_TEXTS, split_name(status, "")] += sign * count
            _add_sized(found, status, names, sign * count, sign * length, sign * nbytes)
    return found


def split_name(status: str, name: str) -> str:
    """The name of a row of a facet of SPLIT: the texts of that status, of
    that name of the facet split."""
    return f"{status} {name}"


def split(con, facet: str, name: str) -> dict[str, int]:
    """{status: how many texts} of the name of a facet of SPLITS (TEXTS
    with the name ''), read from its rows of the split facet by their keys;
    a status with no row is left out."""
    keys = {split_name(status, name): status for status in (UNDER_REVIEW, ADMITTED, REFUSED)}
    return {keys[one]: count for one, count in con.execute(
        "SELECT name, texts FROM counts WHERE facet = ? AND name IN (SELECT value FROM json_each(?))",
        (SPLITS[facet], json.dumps(list(keys))))}


def of_kinds(con, statuses: Iterable[str], kinds: Iterable[str]) -> dict[str, int]:
    """{status: how many current texts of that status are of one of the
    kinds} for each status given, read from the rows of KIND by their keys
    (`<status> <kind>`); a status with no row counts 0."""
    statuses, kinds = list(dict.fromkeys(statuses)), list(dict.fromkeys(kinds))
    keys = {f"{status} {kind}": status for status in statuses for kind in kinds}
    found = dict.fromkeys(statuses, 0)
    for name, count in con.execute("SELECT name, texts FROM counts WHERE facet = ? AND name IN "
                                   "(SELECT value FROM json_each(?))", (KIND, json.dumps(list(keys)))):
        found[keys[name]] += count
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
    con.executemany("INSERT INTO counts (facet, name, texts, characters, bytes) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (facet, name) DO UPDATE SET texts = texts + excluded.texts, "
                    "characters = characters + excluded.characters, bytes = bytes + excluded.bytes", moved)
    con.executemany("DELETE FROM counts WHERE facet = ? AND name = ? AND texts = 0",
                    [(facet, name) for facet, name, texts, *_ in moved if texts <= 0])  # a row that rose stays above 0


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
    revisions, and to the title index their titles when TITLE is named."""
    for facet in facets:
        if facet == TITLE:
            titles.enlist(con, chosen)
        elif facet in LISTINGS:
            table, name = LISTINGS[facet]
            con.execute(f"INSERT OR IGNORE INTO {table} ({name}, key) " + _LISTED[facet].format(chosen=chosen))


def unlist(con, chosen: str, facets: Iterable[str]):
    """Take out of the listings of the facets named among LISTINGS the rows
    of the texts whose ids the table chosen holds, as their current
    revisions list them: each row read by its key; and out of the title
    index their titles when TITLE is named."""
    for facet in facets:
        if facet == TITLE:
            titles.unlist(con, chosen)
        elif facet in LISTINGS:
            table, name = LISTINGS[facet]
            con.execute(f"DELETE FROM {table} WHERE ({name}, key) IN (" + _LISTED[facet].format(chosen=chosen) + ")")


def recount(con):
    """The counts made again from the depot's rows, every one of them, with
    the listings and the title index, after the kind, length and bytes each
    revision keeps of its content version are copied again from
    content_lists where they differ (schema.sizes_sql), and the tags each
    current revision copies from taggings (schema.tags_sql)."""
    con.execute(sizes_sql())
    con.execute(tags_sql())
    con.execute("DELETE FROM counts")
    for facet in FACETS:
        if facet in _WHOLE_SIZED:
            con.execute(f"INSERT INTO counts (facet, name, texts, characters, bytes) SELECT '{facet}', * FROM "
                        f"({_WHOLE_SIZED[facet]})")
        else:
            con.execute(f"INSERT INTO counts (facet, name, texts) SELECT '{facet}', * FROM ({_WHOLE[facet]})")
    con.execute("DELETE FROM counts WHERE texts = 0")
    for facet, (table, name) in LISTINGS.items():
        con.execute(f"DELETE FROM {table}")
        con.execute(f"INSERT OR IGNORE INTO {table} ({name}, key) {_LISTED_WHOLE[facet]}")
    titles.refill(con)


def read(con) -> dict[str, dict[str, int]]:
    """{facet: {name: count}} of every count of COUNTED, each facet present."""
    found = {facet: {} for facet in COUNTED}
    for facet, name, count in con.execute("SELECT facet, name, texts FROM counts WHERE facet IN "
                                          f"({', '.join(repr(one) for one in COUNTED)}) ORDER BY facet, name"):
        found[facet][name] = count
    return found


def among(kinds: dict, chosen: Iterable[str]) -> dict:
    """{texts, characters, bytes} of the kinds chosen of kinds ({kind:
    {texts, characters, bytes}}) together, a kind
    kinds does not name counting 0."""
    chosen = set(chosen)
    return {key: sum(sized[key] for kind, sized in kinds.items() if kind in chosen) for key in COLUMNS}


def of_status(con, status: str) -> dict[str, dict[str, list[int]]]:
    """What the current texts of the status hold, from the counts alone:
    {kinds: {kind: [texts, characters, bytes]}, tags: {tag: [...]} of
    every text that has the tag, plain_tagsets: {tags sorted and joined by
    NUL: [...]} of the plain texts that have exactly that set of tags (''
    for none), licenses: {license: [...]}}, each name in order. Each facet's
    rows of that status are one range of names (`<status> `), read by
    their key."""
    low, high = f"{status} ", f"{status}!"  # every name that begins with the status and a space
    found = {"kinds": {}, "tags": {}, "plain_tagsets": {}, "licenses": {}}
    for facet, into in ((KIND, "kinds"), (SPLIT_TAG, "tags"), (PLAIN_TAGSET, "plain_tagsets"),
                        (SPLIT_LICENSE, "licenses")):
        for name, *sizes in con.execute("SELECT name, texts, characters, bytes FROM counts WHERE facet = ? "
                                        "AND name >= ? AND name < ? ORDER BY name", (facet, low, high)):
            rest = name[len(low):]
            if facet == PLAIN_TAGSET:
                rest = "\0".join(json.loads(rest))
            held = found[into].setdefault(rest, [0, 0, 0])
            for at, value in enumerate(sizes):
                held[at] += value
    return {key: dict(sorted(value.items())) for key, value in found.items()}
