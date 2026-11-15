"""The depot's counts: how many current texts it holds, and of them how many
have each status, each tag and each license, how many each library holds
now and how many each decision is on, kept in the table counts
(wpt_depot.schema) so that a page reads them at once whatever the depot's
size.

This module is the only code that writes that table. Each write of the
depot keeps it right in its own transaction: it measures what the texts it
changes add to the counts before and after it changes them (changing), or,
when it knows the difference itself, shifts the counts by it (shift).
recount makes the table again from the depot's rows, which `wpt depot
recount` runs to repair it and the upgrade to format 6 runs to fill it.

A row is (facet, name, texts): facet TEXTS with the name '' (every current
text), STATUS with a status (wpt_depot.policy), TAG with a tag, LICENSE with
a license id a text names, LIBRARY with a library id (its memberships that
have not ended, locked libraries included) and DECISION with a decision id
(its rows of text_decisions). A count that falls to 0 has no row.
"""

import contextlib
from collections import Counter
from collections.abc import Iterable, Iterator

TEXTS, STATUS, TAG, LICENSE, LIBRARY, DECISION = "texts", "status", "tag", "license", "library", "decision"
FACETS = (TEXTS, STATUS, TAG, LICENSE, LIBRARY, DECISION)

# what each facet adds of the texts of a table of ids `{chosen}` (column id): (facet name, count) rows
_MEASURES = {
    TEXTS: "SELECT '', count(*) FROM {chosen} c JOIN texts_revisions t ON t.id = c.id AND t.latest",
    STATUS: "SELECT t.status, count(*) FROM {chosen} c JOIN texts_revisions t ON t.id = c.id AND t.latest "
            "GROUP BY t.status",
    TAG: "SELECT j.value, count(*) FROM {chosen} c JOIN texts_revisions t ON t.id = c.id AND t.latest, "
         "json_each(t.tags) j GROUP BY j.value",
    LICENSE: "SELECT j.value, count(*) FROM {chosen} c JOIN texts_revisions t ON t.id = c.id AND t.latest, "
             "json_each(t.licenses) j GROUP BY j.value",
    LIBRARY: "SELECT m.library, count(*) FROM {chosen} c JOIN memberships m ON m.text = c.id AND m.left_change IS NULL "
             "GROUP BY m.library",
    DECISION: "SELECT l.decision, count(*) FROM {chosen} c JOIN text_decisions l ON l.text = c.id GROUP BY l.decision",
}
# the same over every text of the depot
_WHOLE = {
    TEXTS: "SELECT '', count(*) FROM texts_revisions WHERE latest",
    STATUS: "SELECT status, count(*) FROM texts_revisions WHERE latest GROUP BY status",
    TAG: "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.tags) j WHERE t.latest GROUP BY j.value",
    LICENSE: "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.licenses) j WHERE t.latest "
             "GROUP BY j.value",
    LIBRARY: "SELECT library, count(*) FROM memberships WHERE left_change IS NULL GROUP BY library",
    DECISION: "SELECT decision, count(*) FROM text_decisions GROUP BY decision",
}


def measure(con, chosen: str, facets: Iterable[str]) -> Counter:
    """{(facet, name): count} of what the current texts whose ids the table
    chosen holds (its column id) add to the counts of each facet named."""
    found = Counter()
    for facet in facets:
        for name, count in con.execute(_MEASURES[facet].format(chosen=chosen)):
            if count:
                found[facet, name] += count
    return found


def shift(con, delta: dict):
    """Add delta ({(facet, name): difference}) to the counts, a row that
    falls to 0 removed."""
    moved = [(facet, name, by) for (facet, name), by in delta.items() if by]
    if not moved:
        return
    con.executemany("INSERT INTO counts (facet, name, texts) VALUES (?, ?, ?) "
                    "ON CONFLICT (facet, name) DO UPDATE SET texts = texts + excluded.texts", moved)
    con.executemany("DELETE FROM counts WHERE facet = ? AND name = ? AND texts = 0",
                    [(facet, name) for facet, name, _ in moved])


@contextlib.contextmanager
def changing(con, chosen: str, facets: Iterable[str]) -> Iterator[None]:
    """The counts of the facets named kept right across the block, which
    changes the texts whose ids the table chosen holds: what they add is
    measured before and after it, and the counts shifted by the
    difference. The table must name the same texts at both ends."""
    facets = tuple(facets)
    before = measure(con, chosen, facets)
    yield
    after = measure(con, chosen, facets)
    after.subtract(before)
    shift(con, after)


def recount(con):
    """The counts made again from the depot's rows, every one of them."""
    con.execute("DELETE FROM counts")
    for facet in FACETS:
        con.execute(f"INSERT INTO counts (facet, name, texts) SELECT '{facet}', * FROM ({_WHOLE[facet]})")
    con.execute("DELETE FROM counts WHERE texts = 0")


def read(con) -> dict[str, dict[str, int]]:
    """{facet: {name: count}} of every count, each facet present."""
    found = {facet: {} for facet in FACETS}
    for facet, name, count in con.execute("SELECT facet, name, texts FROM counts ORDER BY facet, name"):
        found[facet][name] = count
    return found
