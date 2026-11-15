"""A filter of the depot's texts, and the SQL condition over the texts view
it stands for, the condition a decision, a put, a fork or a report takes.

A filter holds any of: the ids of chosen texts, tags (a text has every one),
licenses (a text names one of them), statuses (the text has one of them,
the status the depot stores by wpt_depot.policy), libraries (one of them
holds the text), whether a locked library holds it, a free SQL condition of
the person's own, the ids of texts left out (unticked), and the change
number a page was read at (change). A text matches when it matches every
part given; a filter with none matches every text.

With a change number C, a text matches only when nothing changed it after
C (changed_since): its current revision was made at C or before, no
decision was made on it after C, it joined or left no library after C, no
library that holds it was locked or renamed after C, and no pull brought
rows into the depot after C (a pull brings rows numbered by the upstream,
which can be lower than C). So an operation on the texts a
page showed at C never reaches a text that changed since, even one that
came to match the filter only after C; such texts are left out, and the
operation says how many. A list of ids (chosen or unticked) is written as
one JSON string, which SQLite reads as one value whatever its length.
"""

import json
from dataclasses import dataclass

from .errors import Refused
from .schema import ADMITTED, REFUSED, UNDER_REVIEW

STATUSES = (UNDER_REVIEW, ADMITTED, REFUSED)
# whether a locked library holds the text: over a row of the texts view named texts
IN_LOCKED = ("EXISTS (SELECT 1 FROM memberships m JOIN libraries_revisions l ON l.id = m.library AND l.latest "
             "AND l.locked_at IS NOT NULL WHERE m.text = texts.id AND m.left_change IS NULL)")


@dataclass(frozen=True)
class Filter:
    ids: tuple[str, ...] | None = None  # None: any text; empty: no text
    tags: tuple[str, ...] = ()
    licenses: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    libraries: tuple[str, ...] = ()
    locked: bool | None = None
    where: str | None = None
    unticked: tuple[str, ...] = ()
    change: int | None = None


def condition(found: Filter, latest: int | None = None) -> str:
    """The SQL condition over the texts view, named texts, that matches the
    texts the filter matches. latest: the depot's change number now, when
    known; a filter's change number at or past it leaves out no text, so
    its condition is left out. Refused for a status that is none of
    STATUSES."""
    parts = []
    if found.ids is not None:
        parts.append(f"id IN {listed(found.ids)}" if found.ids else "false")
    for tag in found.tags:
        parts.append(f"EXISTS (SELECT 1 FROM json_each(tags) WHERE value = {literal(tag)})")
    if found.licenses:
        parts.append(f"EXISTS (SELECT 1 FROM json_each(licenses) WHERE value IN {strings(found.licenses)})")
    if found.statuses:
        unknown = [one for one in found.statuses if one not in STATUSES]
        if unknown:
            raise Refused(f"a text's status is one of {', '.join(STATUSES)}, not {unknown[0]!r}")
        chosen = list(dict.fromkeys(found.statuses))
        parts.append(f"status = {literal(chosen[0])}" if len(chosen) == 1 else f"status IN {strings(chosen)}")
    if found.libraries:
        parts.append("EXISTS (SELECT 1 FROM memberships m WHERE m.text = texts.id AND m.left_change IS NULL "
                     f"AND m.library IN {strings(found.libraries)})")
    if found.locked is not None:
        parts.append(IN_LOCKED if found.locked else f"NOT {IN_LOCKED}")
    if found.where is not None and found.where.strip():
        parts.append(f"({found.where})")
    if found.unticked:
        parts.append(f"id NOT IN {listed(found.unticked)}")
    if found.change is not None and (latest is None or found.change < latest):
        parts.append(f"NOT {changed_since(found.change)}")
    return " AND ".join(parts) or "true"


def changed_since(change: int) -> str:
    """Whether something changed the text after the change number change,
    as the module says, as an SQL condition over a row of the texts view
    named texts."""
    number = int(change)
    # the memberships of a text, ended or not, are read through their key (library, text, ...), so the test
    # names every library: no index serves memberships by text alone
    return (f"(texts.change > {number} "
            f"OR EXISTS (SELECT 1 FROM text_decisions l WHERE l.text = texts.id AND l.change > {number}) "
            "OR EXISTS (SELECT 1 FROM memberships m WHERE m.text = texts.id "
            "AND m.library IN (SELECT r.id FROM libraries_revisions r WHERE r.latest) "
            f"AND (m.joined_change > {number} OR m.left_change > {number})) "
            "OR EXISTS (SELECT 1 FROM memberships m JOIN libraries_revisions r ON r.id = m.library AND r.latest "
            f"WHERE m.text = texts.id AND m.left_change IS NULL AND r.change > {number}) "
            f"OR EXISTS (SELECT 1 FROM pulls p WHERE p.change > {number}))")


def literal(text: str) -> str:
    """text as an SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def strings(values) -> str:
    """values as a parenthesized SQL list of string literals, for IN."""
    return "(" + ", ".join(literal(one) for one in values) + ")"


def listed(values) -> str:
    """values as a subquery over one JSON string literal, for IN: one value
    for SQLite to read whatever the number of values."""
    return f"(SELECT value FROM json_each({literal(json.dumps(list(values), ensure_ascii=False))}))"
