"""A change of the credits of many texts (Depot.edit_credits_many), made
inside the change's transaction over temporary tables of its connection, so
its cost in memory stays the same whatever the number of texts it names.

A change names each text on one JSON object: {"id": ID, "title": T,
"authors": [A], "tags": [T], "address": A, "notice": N}, each credit
optional; a credit given replaces the text's, null clears it, and a credit
left out stays as it is. A line may add "was": {CREDIT: VALUE, ...}, the
values its writer read: the line changes the text only while each credit
it names still holds that value (null: none), so a line written from an
earlier reading never undoes a later change. Python reads the lines one at
a time, checks the shape of each, and writes it into the temporary table
`credit_changes` (id, given: the credits given, was: the credits expected,
both as JSON text, tags each once); the rules across texts are then
queries. Each text's new credits are made in Python as the rows are read,
and the texts whose credits change go into the temporary table `credited`
(id, the five credits as the new revision stores them, the notice by its
id, the tags as the text has them from then on, in their order, and revised:
whether a credit other than the tags changes), which the new revisions read,
a text whose tags alone change getting no revision, only its rows of
taggings (wpt_depot.taggings).

One change names at most MOST texts: one write holds the depot's lock for
the time it takes, about three times a decision on that many texts (half
of it writing the new revisions), so a larger change is sent as several,
each its own decision. A change of titles takes about a third longer, as
the title index takes the old titles out and puts the new ones in
(wpt_depot.titles).
"""

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from .errors import NotFound, Refused

CREDITS = ("title", "authors", "tags", "address", "notice")  # a text's credits, in the order a reason names them
LISTS = ("authors", "tags")
SHAPE = ('a change is {"id": ID, "title": T, "authors": [A], "tags": [T], "address": A, "notice": N, '
         '"was": {CREDIT: VALUE}}, each credit optional, null clearing it, "was" optional')
MOST = 350_000  # the texts one change names at most
BATCH = 1000  # the lines written into a temporary table at a time


def open_changes(con, changes: str | Path | Iterable[dict]):
    """The temporary table credit_changes, filled from changes: a
    JSON-lines file (blank lines skipped) or the lines as objects (a web
    request's). Refused for a file that cannot be read or is not JSON
    lines, for a change of another shape (_checked), or for more than MOST
    lines."""
    con.execute("CREATE TEMP TABLE credit_changes (id TEXT NOT NULL, given TEXT NOT NULL, was TEXT NOT NULL)")
    batch, count = [], 0
    for one in _lines(changes):
        count += 1
        if count > MOST:
            raise Refused(f"a change of credits names at most {MOST:,} texts: split the lines into several "
                          "changes, each its own decision")
        batch.append(_checked(one))
        if len(batch) == BATCH:
            con.executemany("INSERT INTO temp.credit_changes VALUES (?, ?, ?)", batch)
            batch = []
    if batch:
        con.executemany("INSERT INTO temp.credit_changes VALUES (?, ?, ?)", batch)
    con.execute("CREATE INDEX temp.credit_changes_id ON credit_changes (id)")


def close(con):
    """Drop the temporary tables of a change of credits."""
    for name in ("credit_changes", "credited", "credited_ids"):
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
                        raise Refused(f"the changes of credits are not JSON lines: {error}") from None
    except (OSError, UnicodeDecodeError) as error:
        raise Refused(f"the changes of credits are not JSON lines: {error}") from None


def _checked(given: tuple[object, str]) -> tuple[str, str, str]:
    """A change as the row of credit_changes: (id, the credits given as
    JSON, the credits expected as JSON, tags each once in both). Refused for
    a change that is not of the shape the module says: a key other than id,
    the credits and was, a was that is not an object of credits, a title,
    address or notice that is not text or is empty, authors or tags that are
    not a list of words that are not empty."""
    one, line = given
    problem = None
    if not isinstance(one, dict):
        problem = "is not a JSON object"
    elif not set(one) <= {"id", *CREDITS, "was"}:
        problem = f"holds a key other than id, {', '.join(CREDITS)} and was"
    elif not isinstance(one.get("id"), str):
        problem = "has no id that is text"
    elif not isinstance(one.get("was", {}), dict) or not set(one.get("was", {})) <= set(CREDITS):
        problem = f"has a was that is not an object of {', '.join(CREDITS)}"
    else:
        problem = _shape(one) or _shape(one.get("was", {}), "was: ")
    if problem:
        shown = line if len(line) <= 200 else line[:200] + "..."
        raise Refused(f"a change of credits {problem} ({SHAPE}): {shown}")
    return one["id"], _json(_credits(one)), _json(_credits(one.get("was", {})))


def _shape(credits: dict, within: str = "") -> str | None:
    """What is wrong with the credits of an object, or None."""
    for name in CREDITS:
        value = credits.get(name)
        if value is None:
            continue
        if name in LISTS:
            if not isinstance(value, list) or any(not isinstance(each, str) or not each.strip() for each in value):
                return f"has {within}{name} that are not a list of words that are not empty"
        elif not isinstance(value, str) or not value.strip():
            return f"has {within}a {name} that is not text, or is empty"
    return None


def _credits(one: dict) -> dict:
    """The credits an object names, tags each once."""
    credits = {name: one[name] for name in CREDITS if name in one}
    if "tags" in credits:
        credits["tags"] = list(dict.fromkeys(credits["tags"] or []))
    return credits


def check(con, upstream_sql: str, named: int) -> tuple[int, list[str]]:
    """Refused or NotFound for the first rule the changes break, each a
    query; (how many texts named the upstream owns, upstream_sql over id
    holding, and the first `named` of them by id), for the caller to
    refuse: (0, []) when none is."""
    if not con.execute("SELECT 1 FROM temp.credit_changes LIMIT 1").fetchall():
        raise Refused("no text is named: no credits change")
    twice = con.execute("SELECT id FROM temp.credit_changes GROUP BY id HAVING count(*) > 1 ORDER BY id "
                        "LIMIT 1").fetchall()
    if twice:
        raise Refused(f"a change names each text once, not {twice[0][0]} several times")
    missing = con.execute("SELECT count(*), min(c.id) FROM temp.credit_changes c WHERE NOT EXISTS "
                          "(SELECT 1 FROM texts_revisions t WHERE t.id = c.id AND t.latest)").fetchall()
    if missing[0][0]:
        raise NotFound(f"{missing[0][0]} of the texts named are no text of the depot (the first: {missing[0][1]})")
    total = con.execute(f"SELECT count(*) FROM temp.credit_changes WHERE {upstream_sql}").fetchall()[0][0]
    if total:
        return total, [one for (one,) in con.execute(
            f"SELECT id FROM temp.credit_changes WHERE {upstream_sql} ORDER BY id LIMIT {int(named)}").fetchall()]
    return 0, []


def credit(con, notice: Callable[[str], str]) -> tuple[int, int, list[str]]:
    """The temporary table credited, and (how many texts change, how many
    are left out, the names of the credits that change on at least one
    text, in the order of CREDITS). notice gives the id of a notice's text,
    stored once. Each text's new credits are made in Python, one text at a
    time, as the rows are read. A text is left out when its credits would
    stay as they are, or when a credit its line's was names no longer holds
    that value."""
    con.execute("CREATE TEMP TABLE credited (id TEXT PRIMARY KEY, title TEXT, authors TEXT, tags TEXT NOT NULL, "
                "address TEXT, notice TEXT, revised INTEGER NOT NULL)")
    changed = left_out = 0
    names, batch = set(), []
    cursor = con.cursor().execute(f"SELECT c.id, c.given, c.was, n.text, t.title, t.authors, t.current_tags, "
                                  "t.address, t.notice FROM temp.credit_changes c JOIN texts_revisions t ON "
                                  "t.id = c.id AND t.latest LEFT JOIN notices n ON n.id = t.notice")
    for text, given, was, notice_text, *old in cursor:
        given = json.loads(given)
        now = dict(zip(CREDITS, old))
        if was != "{}" and not _holds(json.loads(was), now, notice_text):
            left_out += 1
            continue
        new = dict(now)
        moved = []
        for name, value in given.items():
            if name == "notice" and value is not None:
                value = notice(value)
            if name in LISTS:
                before = json.loads(now[name]) if now[name] is not None else None
                if name == "tags":
                    value, before = sorted(value or []), before or []
                if value != before:
                    new[name] = _json(value)
                    moved.append(name)
            elif value != now[name]:
                new[name] = value
                moved.append(name)
        if not moved:
            left_out += 1
            continue
        changed += 1
        names.update(moved)
        batch.append((text, *(new[name] for name in CREDITS), any(name != "tags" for name in moved)))
        if len(batch) == BATCH:
            con.executemany("INSERT INTO temp.credited VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            batch = []
    if batch:
        con.executemany("INSERT INTO temp.credited VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
    return changed, left_out, [name for name in CREDITS if name in names]


def _holds(was: dict, now: dict, notice_text: str | None) -> bool:
    """Whether each credit was names has that value now (the notice by its text, tags in any order)."""
    for name, value in was.items():
        current = notice_text if name == "notice" else now[name]
        if name in LISTS and current is not None:
            current = json.loads(current)
        if name == "tags":
            value, current = sorted(value or []), sorted(current or [])
        if value != current:
            return False
    return True


def _json(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
