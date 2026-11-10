"""A filter of the depot's texts, and the SQL condition over the texts table
it stands for, the condition a decision, a put, a fork or a report takes.

A filter holds any of: the ids of chosen texts, tags (a text has every one),
licenses (a text names one of them), statuses (the text has one of them,
wpt_depot.policy), libraries (one of them holds the text), whether a locked
library holds it, and a free SQL condition of the person's own. A text matches when it matches every part
given; a filter with none matches every text.
"""

from dataclasses import dataclass

from . import policy
from .errors import Refused
from .schema import ADMITTED, REFUSED, UNDER_REVIEW

STATUSES = (UNDER_REVIEW, ADMITTED, REFUSED)
# the ids of the locked libraries, as one list
LOCKED_LIBRARIES = "(SELECT coalesce(list(id), []) FROM libraries WHERE locked_at IS NOT NULL)"
# whether a locked library holds the text: over a row of texts, with the depot's libraries table as `libraries`
IN_LOCKED = f"(len(list_intersect(libraries, {LOCKED_LIBRARIES})) > 0)"


@dataclass(frozen=True)
class Filter:
    ids: tuple[str, ...] | None = None  # None: any text; empty: no text
    tags: tuple[str, ...] = ()
    licenses: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    libraries: tuple[str, ...] = ()
    locked: bool | None = None
    where: str | None = None


def condition(found: Filter) -> str:
    """The SQL condition over the texts table that matches the texts the
    filter matches. Refused for a status that is none of STATUSES."""
    parts = []
    if found.ids is not None:
        parts.append(f"list_contains({strings(found.ids)}, id)" if found.ids else "false")
    if found.tags:
        parts.append(f"list_has_all(tags, {strings(found.tags)})")
    if found.licenses:
        parts.append(f"list_has_any(licenses, {strings(found.licenses)})")
    if found.statuses:
        unknown = [one for one in found.statuses if one not in STATUSES]
        if unknown:
            raise Refused(f"a text's status is one of {', '.join(STATUSES)}, not {unknown[0]!r}")
        parts.append(f"list_contains({strings(found.statuses)}, {policy.status_sql('decisions')})")
    if found.libraries:
        parts.append(f"list_has_any(libraries, {strings(found.libraries)})")
    if found.locked is not None:
        parts.append(IN_LOCKED if found.locked else f"NOT {IN_LOCKED}")
    if found.where is not None and found.where.strip():
        parts.append(f"({found.where})")
    return " AND ".join(parts) or "true"


def literal(text: str) -> str:
    """text as an SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def strings(values) -> str:
    """values as an SQL list of string literals."""
    return "[" + ", ".join(literal(one) for one in values) + "]::VARCHAR[]"
