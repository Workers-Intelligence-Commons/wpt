"""A change of the scores of many texts (Depot.edit_scores), made inside
the change's transaction over temporary tables of its connection, so its
cost in memory stays the same whatever the number of texts it names.

A change names each text on one JSON object: {"id": ID, "scores": [{"name":
N, "value": V, "content_version": C}], "remove": [N]}, scores and remove
each optional. Python reads the lines one at a time, checks the shape of
each, and writes it into the temporary table `changes` (id, scores, remove:
each name trimmed, as JSON text); every rule across texts is then one
query. The texts whose scores change go into the temporary table `scored`
(id, scores: its new scores, those it had less those removed or set again,
plus those set, in the order of their names, null for none), which the
new revisions read.
"""

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from .errors import NotFound, Refused

SHAPE = ('a change is {"id": ID, "scores": [{"name": N, "value": V, "content_version": C}], "remove": [N]}')
KEYS = {"id", "scores", "remove"}
SCORE_KEYS = {"name", "value", "content_version"}
BATCH = 1000  # the lines written into the temporary table at a time


def open_changes(con, changes: str | Path | Iterable[dict]):
    """The temporary table changes, filled from changes: a JSON-lines file
    (blank lines skipped) or the lines as objects (a web request's).
    Refused for a file that cannot be read or is not JSON lines, or for a
    change of another shape (_checked)."""
    con.execute("CREATE TEMP TABLE changes (id TEXT NOT NULL, scores TEXT NOT NULL, remove TEXT NOT NULL)")
    batch = []
    for one in _lines(changes):
        batch.append(_checked(one))
        if len(batch) == BATCH:
            con.executemany("INSERT INTO temp.changes VALUES (?, ?, ?)", batch)
            batch = []
    if batch:
        con.executemany("INSERT INTO temp.changes VALUES (?, ?, ?)", batch)
    con.execute("CREATE INDEX temp.changes_id ON changes (id)")


def close(con):
    """Drop the temporary tables of a change of scores."""
    for name in ("changes", "scored", "scored_ids"):
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
                        yield json.loads(line, parse_constant=_no_constant), line.strip()
                    except ValueError as error:
                        raise Refused(f"the changes of scores are not JSON lines: {error}") from None
    except (OSError, UnicodeDecodeError) as error:
        raise Refused(f"the changes of scores are not JSON lines: {error}") from None


def _no_constant(name: str):
    raise ValueError(f"{name} is no JSON number")


def _checked(given: tuple[object, str]) -> tuple[str, str, str]:
    """A change as the row of changes: (id, its scores as JSON, its names to
    remove as JSON, each name trimmed and each removal once). Refused for a
    change that is not of the shape the module says, with a value that is
    not a finite number, a content version that is not a whole number from
    1, or an empty name."""
    one, line = given
    problem = None
    if not isinstance(one, dict):
        problem = "is not a JSON object"
    elif not set(one) <= KEYS:
        problem = "holds a key other than id, scores and remove"
    elif not isinstance(one.get("id"), str):
        problem = "has no id that is text"
    elif not isinstance(one.get("scores") or [], list):
        problem = "has scores that are no list"
    elif not isinstance(one.get("remove") or [], list):
        problem = "has remove that is no list"
    else:
        for score in one.get("scores") or []:
            if not isinstance(score, dict) or set(score) != SCORE_KEYS:
                problem = "has a score that is not {name, value, content_version}"
            elif not isinstance(score["name"], str) or not score["name"].strip():
                problem = "has a score with no name"
            elif (isinstance(score["value"], bool) or not isinstance(score["value"], (int, float))
                  or not math.isfinite(score["value"])):
                problem = "has a score whose value is not a finite number"
            elif (isinstance(score["content_version"], bool) or not isinstance(score["content_version"], int)
                  or score["content_version"] < 1):
                problem = "has a score whose content version is not a whole number from 1"
            if problem:
                break
        if not problem and any(not isinstance(name, str) or not name.strip() for name in one.get("remove") or []):
            problem = "has a name to remove that is not text, or is empty"
    if problem:
        shown = line if len(line) <= 200 else line[:200] + "..."
        raise Refused(f"a change of scores {problem} ({SHAPE}): {shown}")
    scores = [{"name": score["name"].strip(), "value": float(score["value"]),
               "content_version": score["content_version"]} for score in one.get("scores") or []]
    remove = list(dict.fromkeys(name.strip() for name in one.get("remove") or []))
    return one["id"], _json(scores), _json(remove)


def check(con, upstream_sql: str, named: int) -> tuple[int, list[str]]:
    """Refused or NotFound for the first rule the changes break, each a
    query; (how many texts named the upstream owns, upstream_sql over id
    holding, and the first `named` of them by id), for the caller to
    refuse: (0, []) when none is."""
    if not con.execute("SELECT 1 FROM temp.changes LIMIT 1").fetchall():
        raise Refused("no text is named: nothing to score")
    twice = con.execute("SELECT id FROM temp.changes GROUP BY id HAVING count(*) > 1 ORDER BY id LIMIT 1").fetchall()
    if twice:
        raise Refused(f"a change names each text once, not {twice[0][0]} several times")
    found = con.execute("""SELECT c.id FROM temp.changes c WHERE
        (SELECT count(*) - count(DISTINCT s.value ->> 'name') FROM json_each(c.scores) s) > 0
        OR EXISTS (SELECT 1 FROM json_each(c.scores) s, json_each(c.remove) r WHERE s.value ->> 'name' = r.value)
        ORDER BY c.id LIMIT 1""").fetchall()
    if found:
        raise Refused(f"the change of {found[0][0]} sets or removes a score more than once")
    missing = con.execute("SELECT count(*), min(c.id) FROM temp.changes c WHERE NOT EXISTS "
                          "(SELECT 1 FROM texts_revisions t WHERE t.id = c.id AND t.latest)").fetchall()
    if missing[0][0]:
        raise NotFound(f"{missing[0][0]} of the texts named are no text of the depot (the first: {missing[0][1]})")
    total = con.execute(f"SELECT count(*) FROM temp.changes WHERE {upstream_sql}").fetchall()[0][0]
    if total:
        return total, [one for (one,) in con.execute(
            f"SELECT id FROM temp.changes WHERE {upstream_sql} ORDER BY id LIMIT {int(named)}").fetchall()]
    late = con.execute("""SELECT c.id, s.value ->> 'name', s.value ->> 'content_version' AS version,
        (SELECT max(v.version) FROM content_lists v WHERE v.id = c.id) AS last
        FROM temp.changes c, json_each(c.scores) s WHERE version > last ORDER BY c.id LIMIT 1""").fetchall()
    if late:
        text, name, version, last = late[0]
        raise Refused(f"the text {text} has no content version {version} (its last is {last}), which the score "
                      f"{name} names")
    return 0, []


def score(con) -> tuple[int, int, list[str], list[str]]:
    """The temporary table scored, and (how many texts change, how many are
    left out, the names of the scores set, the names of the scores
    removed). Each text's new scores are made in Python, one text at a
    time, as the rows are read."""
    con.execute("CREATE TEMP TABLE scored (id TEXT PRIMARY KEY, scores TEXT)")
    changed = left_out = 0
    set_names, removed, batch = set(), set(), []
    cursor = con.cursor().execute("SELECT c.id, c.scores, c.remove, t.scores FROM temp.changes c "
                                  "JOIN texts_revisions t ON t.id = c.id AND t.latest")
    for text, given, remove, old in cursor:
        given, remove, old = json.loads(given), set(json.loads(remove)), json.loads(old) if old else []
        names = {one["name"] for one in given}
        kept = [one for one in old if one["name"] not in remove and one["name"] not in names] + given
        new = sorted(kept, key=lambda one: one["name"]) or None
        if (new or []) == old:
            left_out += 1
            continue
        changed += 1
        set_names.update(one["name"] for one in new or [] if one not in old)
        removed.update(one["name"] for one in old if one["name"] not in {n["name"] for n in new or []})
        batch.append((text, _json(new)))
        if len(batch) == BATCH:
            con.executemany("INSERT INTO temp.scored VALUES (?, ?)", batch)
            batch = []
    if batch:
        con.executemany("INSERT INTO temp.scored VALUES (?, ?)", batch)
    return changed, left_out, sorted(set_names), sorted(removed)


def _json(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
