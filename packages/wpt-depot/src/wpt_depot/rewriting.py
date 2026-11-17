"""A change of the content of many plain texts (Depot.change_content_many),
made inside the change's transaction over temporary tables of its
connection, so its cost in memory stays that of one text whatever the
number of texts it names.

A change names each text on one JSON object: {"id": ID, "text": T, "was":
V}, T the text's new plain text and V, optional, the content version its
writer read: the line changes the text only while that is still its
current content version, so a line written from an earlier reading never
undoes a later change. Python reads the lines one at a time, checks the
shape of each, and writes it into the temporary table `content_changes`
(id, text, was); the rules across texts are then queries. The depot then
reads the lines back one at a time with each text's current content
(`current`), stores each new content version as `wpt depot content` does,
and records each text it changes in the temporary table `rewritten` (id,
its content version before, the new one), which the new revisions read.

Only plain texts: an item's content keeps its kind's rules, and changes
one item at a time (Depot.change_content). One change names at most MOST
texts, as a change of credits does (wpt_depot.crediting): a larger change
is sent as several.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

from .errors import NotFound, Refused

SHAPE = ('a change is {"id": ID, "text": T, "was": V}: T the new plain text, V the content version the line was '
         'written from, optional')
MOST = 350_000  # the texts one change names at most
BATCH = 1000  # the lines written into the temporary table at a time
TABLES = ("content_changes", "content_now", "rewritten", "rewritten_ids", "rewritten_group")


class Rewritten(NamedTuple):
    """What a change of the content of many texts did: each decision it
    recorded (its id, its reason, how many texts it is on), how many texts
    changed, how many were left out because their content is that already,
    how many because their content version is no longer the one their
    line's was names, and the first of those by id."""
    decisions: list[tuple[str, str, int]]
    changed: int
    unchanged: int
    stale: int
    stale_ids: list[str]


def open_changes(con, changes: str | Path | Iterable[dict]):
    """The temporary table content_changes, filled from changes: a
    JSON-lines file (blank lines skipped) or the lines as objects (a web
    request's). Refused for a file that cannot be read or is not JSON
    lines, for a change of another shape (_checked), or for more than MOST
    lines."""
    con.execute("CREATE TEMP TABLE content_changes (id TEXT NOT NULL, text TEXT NOT NULL, was INTEGER)")
    batch, count = [], 0
    for one in _lines(changes):
        count += 1
        if count > MOST:
            raise Refused(f"a change of content names at most {MOST:,} texts: split the lines into several "
                          "changes")
        batch.append(_checked(one))
        if len(batch) == BATCH:
            con.executemany("INSERT INTO temp.content_changes VALUES (?, ?, ?)", batch)
            batch = []
    if batch:
        con.executemany("INSERT INTO temp.content_changes VALUES (?, ?, ?)", batch)
    con.execute("CREATE INDEX temp.content_changes_id ON content_changes (id)")


def close(con):
    """Drop the temporary tables of a change of content."""
    for name in TABLES:
        con.execute(f"DROP TABLE IF EXISTS temp.{name}")


def _lines(changes) -> Iterator[tuple[object, str]]:
    """(each change as an object, its line as shown in a refusal)."""
    if not isinstance(changes, (str, Path)):
        for one in changes:
            yield one, json.dumps(one, ensure_ascii=False)
        return
    try:
        with Path(changes).open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    try:
                        yield json.loads(line), line.strip()
                    except ValueError as error:
                        raise Refused(f"the changes of content are not JSON lines: {error}") from None
    except (OSError, UnicodeDecodeError) as error:
        raise Refused(f"the changes of content are not JSON lines: {error}") from None


def _checked(given: tuple[object, str]) -> tuple[str, str, int | None]:
    """A change as the row of content_changes. Refused for a change that is
    not of the shape the module says: a key other than id, text and was, an
    id or a text that is not text, a was that is not a content version (a
    whole number from 1)."""
    one, line = given
    problem = None
    if not isinstance(one, dict):
        problem = "is not a JSON object"
    elif not set(one) <= {"id", "text", "was"}:
        problem = "holds a key other than id, text and was"
    elif not isinstance(one.get("id"), str):
        problem = "has no id that is text"
    elif not isinstance(one.get("text"), str):
        problem = "has no text that is text"
    elif one.get("was") is not None and (isinstance(one["was"], bool) or not isinstance(one["was"], int)
                                         or one["was"] < 1):
        problem = "has a was that is not a content version, a whole number from 1"
    if problem:
        shown = line if len(line) <= 200 else line[:200] + "..."
        raise Refused(f"a change of content {problem} ({SHAPE}): {shown}")
    return one["id"], one["text"], one.get("was")


def check(con, upstream_sql: str, named: int) -> tuple[int, list[str]]:
    """Refused or NotFound for the first rule the changes break, each a
    query: no line, a text named twice, a text the depot does not hold, an
    item; (how many texts named the upstream owns, upstream_sql over id
    holding, and the first `named` of them by id), for the caller to
    refuse: (0, []) when none is."""
    if not con.execute("SELECT 1 FROM temp.content_changes LIMIT 1").fetchall():
        raise Refused("no text is named: no content changes")
    twice = con.execute("SELECT id FROM temp.content_changes GROUP BY id HAVING count(*) > 1 ORDER BY id "
                        "LIMIT 1").fetchall()
    if twice:
        raise Refused(f"a change names each text once, not {twice[0][0]} several times")
    missing = con.execute("SELECT count(*), min(c.id) FROM temp.content_changes c WHERE NOT EXISTS "
                          "(SELECT 1 FROM texts_revisions t WHERE t.id = c.id AND t.latest)").fetchall()
    if missing[0][0]:
        raise NotFound(f"{missing[0][0]} of the texts named are no text of the depot (the first: {missing[0][1]})")
    items = con.execute("SELECT count(*), min(c.id) FROM temp.content_changes c JOIN texts_revisions t "
                        "ON t.id = c.id AND t.latest JOIN content_lists l ON l.id = t.id "
                        "AND l.version = t.content_version WHERE l.kind IS NOT NULL").fetchall()
    if items[0][0]:
        raise Refused(f"a change of the content of many texts takes plain texts only, and {items[0][0]} of the "
                      f"texts named are items (the first: {items[0][1]}): an item's content changes one item at "
                      "a time")
    total = con.execute(f"SELECT count(*) FROM temp.content_changes WHERE {upstream_sql}").fetchall()[0][0]
    if total:
        return total, [one for (one,) in con.execute(
            f"SELECT id FROM temp.content_changes WHERE {upstream_sql} ORDER BY id LIMIT {int(named)}").fetchall()]
    return 0, []


def current(con) -> Iterator[tuple[str, str, int | None, int, list[tuple[bytes, int]], int]]:
    """(id, the new text, the was its line names, the text's current
    content version, that version's parts as (hash, length), the text's
    last content version) of each line, one at a time. What the texts hold
    now is copied first into the temporary table content_now, so the
    content versions the caller writes as it reads are never read back."""
    con.execute("CREATE TEMP TABLE content_now (id TEXT PRIMARY KEY, version INTEGER NOT NULL, parts TEXT NOT NULL, "
                "last INTEGER NOT NULL)")
    con.execute("INSERT INTO temp.content_now SELECT c.id, t.content_version, l.parts, "
                "(SELECT max(v.version) FROM content_lists v WHERE v.id = c.id) FROM temp.content_changes c "
                "JOIN texts_revisions t ON t.id = c.id AND t.latest "
                "JOIN content_lists l ON l.id = t.id AND l.version = t.content_version")
    for text, new, was, version, parts, last in con.cursor().execute(
            "SELECT c.id, c.text, c.was, n.version, n.parts, n.last FROM temp.content_changes c "
            "JOIN temp.content_now n ON n.id = c.id"):
        yield text, new, was, version, [(bytes.fromhex(one["hash"]), one["length"]) for one in json.loads(parts)], last
