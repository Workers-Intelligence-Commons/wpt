"""The title index: the titles of the depot's current texts in an SQLite
FTS5 table (TABLE), split into trigrams, so that a plain search of words of
at least SHORTEST characters reads only the texts whose title holds each of
them, from the index, never every text (Depot.search).

The table stores no title, only the trigrams of each (content='', with no
sizes and no positions: columnsize=0, detail=none), which is the smallest
form FTS5 keeps. A text's row in it is numbered by its id: its number
plus its owner's slot times SPAN (row_sql), the slot being the owner's row
of the table title_owners (each depot name the ids of the current texts
carry, depot.ids.owner, numbered from 0 in the order they first came),
so the rows of one owner come in the numeric order of their ids, which is
the order of their keys (ids.order_sql), and a page reads them in that
order from the index, after its place (window). An id that is not exactly
its owner, a hyphen and its number written without leading zeros, or whose
number reaches SPAN, has no row number: a write that would list one is
refused (Unlisted), so the index never misses a titled text.

A word matches a title holding it in any case: the trigram tokenizer folds
the case of every letter, and LIKE that of ASCII letters only, so the texts
the index gives for a word are those a LIKE on the title matches, and
perhaps a few more, which the search's condition then leaves out
(depot.filters). A word of fewer than SHORTEST characters has no
trigram, and no index row can stand for it.

Only depot.counts writes the index, as it writes the listings of tags
and licenses, in the same transaction as the change: unlist takes out the
rows of texts as their current revisions title them, before a change, and
enlist puts them back as they are after it; refill makes it again from
every current revision. Each statement writes its rows in the order of
their row numbers: FTS5 writes the terms it holds in memory out to a new
segment of the index whenever a row comes with a number below the one
before it in the same transaction, and the ids' own order (their text) is
not that one.
"""

import json

from . import ids
from .errors import Refused

TABLE = "text_titles"
OWNERS = "title_owners"
CREATE = (f"CREATE VIRTUAL TABLE {TABLE} USING fts5(title, content='', columnsize=0, detail=none, "
          "tokenize='trigram')")
SHORTEST = 3  # the characters of a trigram: a shorter word has none
SPAN = 1 << 44  # row numbers of one owner: its slot's range holds every number below it


class Unlisted(Refused):
    """A text to list has an id the index cannot number (the module says
    which): nothing is written."""


def row_sql(column: str) -> str:
    """The index's row number of the text whose id is in column, as SQL:
    null for an owner with no slot yet."""
    return (f"((SELECT o.slot FROM {OWNERS} o WHERE o.owner = {ids.owner_sql(column)}) * {SPAN} "
            f"+ {ids.number_sql(column)})")


def _numbered(column: str) -> str:
    """Whether the id in column has a row number, as the module says, as SQL."""
    number = ids.number_sql(column)
    return (f"({number} IS NOT NULL AND {number} < {SPAN} "
            f"AND {column} = {ids.owner_sql(column)} || '-' || {number})")


def _titled(chosen: str) -> str:
    """The current revisions with a title of the texts whose ids the table chosen holds (column id), each by its key."""
    return f"FROM {chosen} c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest WHERE t.title IS NOT NULL"


def enlist(con, chosen: str):
    """Add to the index the titles of the current revisions of the texts
    whose ids the table chosen holds. Unlisted for an id with no row
    number."""
    _add(con, _titled(chosen))


def unlist(con, chosen: str):
    """Take out of the index the titles of the texts whose ids the table
    chosen holds, as their current revisions title them."""
    con.execute(f"INSERT INTO {TABLE} ({TABLE}, rowid, title) SELECT 'delete', {row_sql('t.id')}, t.title "
                + _titled(chosen) + " ORDER BY 2")


def refill(con):
    """The index made again from every current revision."""
    con.execute(f"INSERT INTO {TABLE} ({TABLE}) VALUES ('delete-all')")
    con.execute(f"DELETE FROM {OWNERS}")
    _add(con, "FROM texts_revisions t WHERE t.latest AND t.title IS NOT NULL")


def add_rows(con, rows: str, params: dict | None = None):
    """Add to the index the titles of the revisions rows names: the FROM
    and WHERE clauses of a query over texts_revisions named t, each a
    current revision with a title (the upgrade's batches). Unlisted for an
    id with no row number."""
    _add(con, rows, params)


def _add(con, rows: str, params: dict | None = None):
    params = params or {}
    odd = con.execute(f"SELECT t.id {rows} AND NOT {_numbered('t.id')} LIMIT 1", params).fetchall()
    if odd:
        raise Unlisted(f"the title index numbers a text by its depot's name, a hyphen and its number, which "
                       f"{odd[0][0]} is not: no title is listed")
    con.execute(f"INSERT INTO {OWNERS} (owner, slot) SELECT n.owner, "
                f"(SELECT coalesce(max(slot), -1) FROM {OWNERS}) + row_number() OVER (ORDER BY n.owner) "
                f"FROM (SELECT DISTINCT {ids.owner_sql('t.id')} AS owner {rows}) n "
                f"WHERE n.owner NOT IN (SELECT owner FROM {OWNERS})", params)
    con.execute(f"INSERT INTO {TABLE} (rowid, title) SELECT {row_sql('t.id')}, t.title {rows} ORDER BY 1", params)


def match(words: list[str]) -> str | None:
    """The FTS5 query that gives the texts whose title may hold each word
    of SHORTEST characters or more (every trigram of each), None when no
    word is that long."""
    trigrams = dict.fromkeys(word[at:at + SHORTEST] for word in words if len(word) >= SHORTEST
                             for at in range(len(word) - SHORTEST + 1))
    if not trigrams:
        return None
    return " AND ".join('"' + one.replace('"', '""') + '"' for one in trigrams)


class Part:
    """The texts of one owner whose title the query matches, as keys
    (ids.order_sql) in the order of the index: rows reads them all;
    window(compare, direction, size) reads at most size of them after a
    place, as Depot.search's window does (depot.depot._window)."""

    def __init__(self, owner: str, slot: int, query: str):
        self.owner, self.base, self.query = owner, slot * SPAN, query
        number = f"(rowid - {self.base})"
        self.key = (f"{_literal(owner)} || char(1) || '0' || printf('%020d', {number}) || char(1) || "
                    f"{_literal(owner + '-')} || {number}")
        self.rows = (f"SELECT {self.key} AS key FROM {TABLE} WHERE {TABLE} MATCH {_literal(query)} "
                     f"AND rowid >= {self.base} AND rowid < {self.base + SPAN}")

    def window(self, compare: str | None, direction: str, size: int) -> tuple[str, int]:
        """(the SQL of at most size keys after the place, in the order asked,
        how many times it names the place's parameter), the place's bound
        turned into one of the index's row numbers so the index reads from
        it (bound)."""
        low, high, uses = f">= {self.base}", f"< {self.base + SPAN}", 0
        if compare is not None:
            bound, uses = self.bound(), 4
            if compare == ">":
                low = f"> max({self.base - 1}, {bound})"
            elif compare == ">=":
                low = f">= max({self.base}, {bound})"
            elif compare == "<":
                high = f"< min({self.base + SPAN}, {bound})"
            else:
                high = f"<= min({self.base + SPAN - 1}, {bound})"
        return (f"SELECT {self.key} AS key FROM {TABLE} WHERE {TABLE} MATCH {_literal(self.query)} "
                f"AND rowid {low} AND rowid {high} ORDER BY rowid{direction} LIMIT {int(size)}"), uses

    def bound(self) -> str:
        """The row number a place (one parameter, a key, named four times)
        stands at for this owner, as SQL: just before the owner's range
        when the place's owner comes before it, just past it when it comes
        after it or when the place is an id of the owner with no number
        (which comes after every numbered one); else the place's number in
        the owner's range."""
        head, tail = _literal(self.owner + "\x01"), _literal(self.owner + "\x02")
        width = len(self.owner)
        before, beyond = self.base - 1, self.base + SPAN
        return (f"(CASE WHEN ? < {head} THEN {before} WHEN ? >= {tail} THEN {beyond} "
                f"WHEN substr(?, {width + 2}, 1) <> '0' THEN {beyond} "
                f"ELSE {self.base} + CAST(substr(?, {width + 3}, 20) AS INTEGER) END)")


def parts(con, query: str) -> list[Part]:
    """The parts of the index a query reads: one for each owner."""
    return [Part(owner, slot, query) for owner, slot in con.execute(f"SELECT owner, slot FROM {OWNERS} ORDER BY owner")]


def count_sql(query: str, most: int | None = None) -> tuple[str, tuple]:
    """(the SQL of how many texts the index gives for a query, as its column
    n, at most most (all when None), its parameters)."""
    limit = "" if most is None else f" LIMIT {int(most)}"
    return f"SELECT count(*) AS n FROM (SELECT 1 FROM {TABLE} WHERE {TABLE} MATCH ?{limit})", (query,)


def outside_sql(query: str, named: list[str]) -> tuple[str, tuple]:
    """(the SQL of how many of the texts whose ids are named the depot holds
    and the index does not give for the query, as its column n, its
    parameters)."""
    return (f"SELECT count(*) AS n FROM json_each(?) x CROSS JOIN texts_revisions t ON t.id = x.value "
            f"AND t.latest WHERE NOT EXISTS (SELECT 1 FROM {TABLE} WHERE {TABLE} MATCH ? "
            f"AND rowid = {row_sql('t.id')})", (json.dumps(named), query))


def _literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"
