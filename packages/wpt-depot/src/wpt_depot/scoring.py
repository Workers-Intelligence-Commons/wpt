"""A change of the scores of many texts (Depot.edit_scores) as SQL over a
DuckDB connection to the depot's tables, so its cost in memory stays the
same whatever the number of texts it names: the changes are read where they
are (an Arrow table) or a few lines at a time into a temporary table of
DuckDB (a JSON-lines file), every check is one query, and
the new revision rows come out of one query as a stream.

A change names each text on one JSON object: {"id": ID, "scores": [{"name":
N, "value": V, "content_version": C}], "remove": [N]}, scores and remove
each optional. The queries read it as the view `given` (one column `json`),
then as the view `changes` (id, scores, remove, each name trimmed), then as
the view `scored`: each text named, at its current revision, with its new
scores (`new_scores`: its scores less those removed or set again, plus those
set, in the order of their names, null for none) and whether they differ
from its scores now (`changed`)."""

import json
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow as pa

from .errors import NotFound, Refused
from .schema import SCHEMAS

# how the depot reads the scores and the names of a change once its shape is checked
_TYPED = ("""json_transform(json -> 'scores', '[{"name": "VARCHAR", "value": "DOUBLE", "content_version": "BIGINT"}]')""",
          """json_transform(json -> 'remove', '["VARCHAR"]')""")
# the JSON types a value and a content version may have
_NUMBERS, _WHOLE = "('BIGINT', 'UBIGINT', 'DOUBLE')", "('BIGINT', 'UBIGINT')"
FILE_LINES = 256  # the lines of a file of changes Python holds at a time (open_changes)


def lines(changes: Sequence[dict]) -> pa.Table:
    """Changes given as objects (a web request's) as the Arrow table edit_scores reads: one JSON text per row."""
    return pa.table({"json": pa.array([json.dumps(one, ensure_ascii=False) for one in changes], pa.string())})


def open_changes(con: duckdb.DuckDBPyConnection, changes: str | Path | pa.Table):
    """The views given and changes over changes: a JSON-lines file (blank
    lines skipped) or an Arrow table with a column json. Refused for a file
    that cannot be read or is not JSON lines, or for a change of another
    shape (shape). The depot's DuckDB reaches no file outside its folder
    (wpt_depot.tables), so Python reads the file, FILE_LINES lines at a
    time, into a temporary table of the connection, which DuckDB keeps on
    disk past its memory."""
    if isinstance(changes, pa.Table):
        con.register("given_table", changes)
    else:
        try:
            con.register("given_lines", _file_lines(Path(changes)))
            con.execute("CREATE TEMP TABLE given_table AS SELECT json FROM given_lines")
        except (OSError, UnicodeDecodeError, duckdb.Error) as error:
            raise Refused(f"the changes of scores are not JSON lines: {error}") from None
    con.execute("CREATE TEMP VIEW given AS SELECT json::JSON AS json FROM given_table")
    con.execute(f"""CREATE TEMP VIEW changes AS SELECT json ->> 'id' AS id,
        coalesce(list_transform({_TYPED[0]}, lambda s: {{'name': trim(s.name), 'value': s.value,
                                                         'content_version': s.content_version}}), []) AS scores,
        coalesce(list_distinct(list_transform({_TYPED[1]}, lambda r: trim(r))), []) AS remove FROM given""")
    try:
        _shape(con)
    except duckdb.Error as error:
        raise Refused(f"the changes of scores are not JSON lines: {error}") from None


def _file_lines(path: Path) -> pa.RecordBatchReader:
    """The lines of the file that are not blank, read as UTF-8 a few at a
    time: one string per row in a column json. OSError when it cannot be
    opened."""
    found = path.open(encoding="utf-8")

    def batches():
        with found:
            batch = []
            for line in found:
                if line.strip():
                    batch.append(line)
                if len(batch) == FILE_LINES:
                    yield pa.record_batch([pa.array(batch, pa.string())], names=["json"])
                    batch = []
            if batch:
                yield pa.record_batch([pa.array(batch, pa.string())], names=["json"])
    return pa.RecordBatchReader.from_batches(pa.schema([("json", pa.string())]), batches())


def _shape(con: duckdb.DuckDBPyConnection):
    """Refused for the first change that is not of the shape the module
    says, with a value that is not a finite number, a content version that
    is not a whole number from 1, or an empty name."""
    found = con.sql("""SELECT json, CASE
        WHEN json_type(json) <> 'OBJECT' THEN 'is not a JSON object'
        WHEN NOT list_has_all(['id', 'scores', 'remove'], json_keys(json))
            THEN 'holds a key other than id, scores and remove'
        WHEN coalesce(json_type(json, '$.id'), '') <> 'VARCHAR' THEN 'has no id that is text'
        WHEN coalesce(json_type(json, '$.scores'), 'NULL') NOT IN ('ARRAY', 'NULL') THEN 'has scores that are no list'
        WHEN coalesce(json_type(json, '$.remove'), 'NULL') NOT IN ('ARRAY', 'NULL') THEN 'has remove that is no list'
        END AS problem FROM given WHERE problem IS NOT NULL LIMIT 1""").fetchall()
    if not found:
        found = con.sql(f"""SELECT json, CASE
            WHEN json_type(s) <> 'OBJECT' OR len(json_keys(s)) <> 3
                 OR NOT list_has_all(json_keys(s), ['name', 'value', 'content_version'])
                THEN 'has a score that is not {{name, value, content_version}}'
            WHEN json_type(s, '$.name') <> 'VARCHAR' OR trim(s ->> 'name') = '' THEN 'has a score with no name'
            WHEN json_type(s, '$.value') NOT IN {_NUMBERS} OR NOT isfinite(TRY_CAST(s ->> 'value' AS DOUBLE))
                THEN 'has a score whose value is not a finite number'
            WHEN json_type(s, '$.content_version') NOT IN {_WHOLE} OR TRY_CAST(s ->> 'content_version' AS BIGINT) < 1
                THEN 'has a score whose content version is not a whole number from 1'
            END AS problem
            FROM (SELECT json, unnest(json_extract(json, '$.scores[*]')) AS s FROM given
                  WHERE json_type(json, '$.scores') = 'ARRAY')
            WHERE problem IS NOT NULL LIMIT 1""").fetchall()
    if not found:
        found = con.sql("""SELECT json, 'has a name to remove that is not text, or is empty' FROM
            (SELECT json, unnest(json_extract(json, '$.remove[*]')) AS r FROM given
             WHERE json_type(json, '$.remove') = 'ARRAY')
            WHERE json_type(r) <> 'VARCHAR' OR trim(r ->> '$') = '' LIMIT 1""").fetchall()
    if found:
        line, problem = found[0]
        shown = line if len(line) <= 200 else line[:200] + "..."
        raise Refused(f"a change of scores {problem} (a change is {{\"id\": ID, \"scores\": [{{\"name\": N, \"value\": "
                      f"V, \"content_version\": C}}], \"remove\": [N]}}): {shown}")


def check(con: duckdb.DuckDBPyConnection, upstream_sql: str, named: int) -> tuple[int, list[str]]:
    """Refused or NotFound for the first rule the changes break, each a
    query; (how many texts named the upstream owns, upstream_sql over id
    holding, and the first `named` of them by id), for the caller to
    refuse: (0, []) when none is."""
    if not con.sql("SELECT 1 FROM changes LIMIT 1").fetchall():
        raise Refused("no text is named: nothing to score")
    twice = con.sql("SELECT id FROM changes GROUP BY id HAVING count(*) > 1 ORDER BY id LIMIT 1").fetchall()
    if twice:
        raise Refused(f"a change names each text once, not {twice[0][0]} several times")
    found = con.sql("""SELECT id FROM (SELECT id, list_transform(scores, lambda s: s.name) AS names, remove FROM changes)
        WHERE len(list_distinct(names)) < len(names) OR len(list_intersect(names, remove)) > 0
        ORDER BY id LIMIT 1""").fetchall()
    if found:
        raise Refused(f"the change of {found[0][0]} sets or removes a score more than once")
    missing = con.sql("SELECT count(*), min(c.id) FROM changes c ANTI JOIN texts_revisions t ON t.id = c.id").fetchall()
    if missing[0][0]:
        raise NotFound(f"{missing[0][0]} of the texts named are no text of the depot (the first: {missing[0][1]})")
    total = con.sql(f"SELECT count(*) FROM changes WHERE {upstream_sql}").fetchall()[0][0]
    if total:
        return total, [one for (one,) in con.sql(
            f"SELECT id FROM changes WHERE {upstream_sql} ORDER BY id LIMIT {int(named)}").fetchall()]
    late = con.sql("""SELECT c.id, c.s.name, c.s.content_version, v.last
        FROM (SELECT id, unnest(scores) AS s FROM changes) c
        JOIN (SELECT id, max(version) AS last FROM content_lists GROUP BY id) v ON v.id = c.id
        WHERE c.s.content_version > v.last ORDER BY c.id LIMIT 1""").fetchall()
    if late:
        text, name, version, last = late[0]
        raise Refused(f"the text {text} has no content version {version} (its last is {last}), which the score "
                      f"{name} names")
    return 0, []


def score(con: duckdb.DuckDBPyConnection) -> tuple[int, int, list[str], list[str]]:
    """The view scored, and (how many texts change, how many are left out,
    the names of the scores set, the names of the scores removed)."""
    con.execute("""CREATE TEMP VIEW scored AS SELECT *, new_scores IS DISTINCT FROM scores AS changed FROM (
        SELECT t.*, CASE WHEN len(kept) = 0 THEN NULL ELSE list_sort(kept) END AS new_scores FROM (
            SELECT t.*, list_concat(list_filter(coalesce(t.scores, []), lambda s: NOT list_contains(c.remove, s.name)
                AND NOT list_contains(list_transform(c.scores, lambda g: g.name), s.name)), c.scores) AS kept
            FROM (SELECT * FROM texts_revisions WHERE id IN (SELECT id FROM changes)
                  QUALIFY row_number() OVER (PARTITION BY id ORDER BY revision DESC) = 1) t
            JOIN changes c ON c.id = t.id) t)""")
    changed, left_out = con.sql("SELECT count(*) FILTER (WHERE changed), count(*) FILTER (WHERE NOT changed) "
                                "FROM scored").fetchall()[0]
    set_names = [one for (one,) in con.sql(
        "SELECT DISTINCT s.name FROM (SELECT unnest(new_scores) AS s, scores FROM scored WHERE changed) "
        "WHERE NOT list_contains(coalesce(scores, []), s) ORDER BY 1").fetchall()]
    removed = [one for (one,) in con.sql(
        "SELECT DISTINCT s.name FROM (SELECT unnest(scores) AS s, new_scores FROM scored WHERE changed) "
        "WHERE NOT list_contains(list_transform(coalesce(new_scores, []), lambda n: n.name), s.name) "
        "ORDER BY 1").fetchall()]
    return changed, left_out, set_names, removed


def revisions(con: duckdb.DuckDBPyConnection, number: int, decision: str) -> pa.RecordBatchReader:
    """The new revision of every text scored changes, made at the change
    number and carrying the decision, as a stream of rows of texts."""
    schema = SCHEMAS["texts"]
    columns = ", ".join(
        {"revision": "revision + 1 AS revision", "change": "$number AS change",
         "decisions": "list_append(decisions, $decision) AS decisions", "scores": "new_scores AS scores"}.get(
            name, name) for name in schema.names)
    found = con.sql(f"SELECT {columns} FROM scored WHERE changed",
                    params={"number": number, "decision": decision}).to_arrow_reader()
    return pa.RecordBatchReader.from_batches(schema, (batch.cast(schema) for batch in found))
