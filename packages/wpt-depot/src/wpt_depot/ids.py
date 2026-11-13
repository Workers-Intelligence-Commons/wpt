"""The ids of a depot's rows. Every id but the journal's is the depot's name,
a hyphen and a number (`gutenberg-12`), so ids never collide across depots:
a clone follows its upstream and adds rows of its own at once, and a corpus
or a dataset can name rows of several depots by their ids alone.

An id carries a depot's name when it is exactly that name, a hyphen and
digits: `wpt-eu-3` carries the name `wpt-eu`, never `wpt`, so names that
begin alike never share an id. The number counts up from 1 among the ids
that carry the depot's own name; an id of another depot (a row a clone
received) never takes part.
"""

import re

from .errors import Refused

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def check_name(name: str):
    """Refused unless name can begin an id: letters, digits, '.', '_' and
    '-', beginning with a letter or a digit."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise Refused(f"a depot's name is made of letters, digits, '.', '_' and '-', beginning with a letter or "
                      f"a digit: {name!r}")


def prefix(name: str) -> str:
    """What every id of the depot called name begins with."""
    return f"{name}-"


def make(name: str, number: int) -> str:
    return f"{prefix(name)}{number}"


def carries(name: str, one: str) -> bool:
    """Whether the id one carries the depot's name name."""
    return re.fullmatch(re.escape(name) + "-[0-9]+", one) is not None


def carries_sql(name: str, column: str = "id") -> str:
    """carries as an SQL condition over a column of ids (a name holds no
    character GLOB reads as a pattern, check_name)."""
    return f"({column} GLOB '{name}-[0-9]*' AND substr({column}, {len(name) + 2}) NOT GLOB '*[^0-9]*')"


def owner(one: str) -> str:
    """The name of the depot that made the row of the id one: what comes
    before its last hyphen and digits (one itself when it has none)."""
    return re.sub(r"-[0-9]+\Z", "", one)


def number(one: str) -> int | None:
    """The number after the last hyphen of the id one, None when it has none."""
    found = re.search(r"-([0-9]+)\Z", one)
    return None if found is None else int(found[1])


def _numbered(column: str) -> str:
    """Whether the id in column ends with a hyphen and digits, as SQL."""
    return f"(rtrim({column}, '0123456789') <> {column} AND rtrim({column}, '0123456789') GLOB '*-')"


def owner_sql(column: str = "id") -> str:
    """owner as an SQL expression over a column of ids."""
    return (f"(CASE WHEN {_numbered(column)} THEN substr({column}, 1, length(rtrim({column}, '0123456789')) - 1) "
            f"ELSE {column} END)")


def number_sql(column: str = "id") -> str:
    """number as an SQL expression over a column of ids."""
    return (f"(CASE WHEN {_numbered(column)} THEN CAST(substr({column}, length(rtrim({column}, '0123456789')) + 1) "
            "AS INTEGER) END)")


def order_sql(column: str = "id") -> str:
    """An SQL ORDER BY list that puts ids in their numeric order: by the
    depot's name, then by number, so trial-2 comes before trial-10. An id
    with no number after its last hyphen comes after those of its name."""
    return f"{owner_sql(column)}, {number_sql(column)} NULLS LAST, {column}"


def order_key(one: str) -> tuple:
    """The same order as order_sql, as a key of Python's sort."""
    found = number(one)
    return owner(one), found is None, found or 0, one
