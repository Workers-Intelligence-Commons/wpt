"""The app's policy: how the decisions on a text or an erasure combine, and
the status they make of it. The depot stores every decision in order, and
each text's status by this rule (a column of its current revision, kept up
to date by every write that changes it); this module is the one place that
says what decisions mean, the depot applying it through the functions
below.

Only the decisions from the latest start over on count. A text starts over
when its licenses change or its content changes: each is recorded as a
decision 'under review' whose reason the depot writes in one fixed form
(license_change, content_change), and every decision before it no longer
counts. Of the decisions that count,
a text or an erasure is admitted when at least one admits it and none
refuses it; 'under review' decisions count for neither. It is refused when
one of them refuses it, and under review while none admits or refuses it.
Only an admitted text trains and joins a library, and only an admitted
erasure is carried out.
"""

from . import ids
from .schema import ADMITTED, REFUSED, UNDER_REVIEW

# how the reason of a license change, and of a content change, starts; only the depot writes a reason that starts so
LICENSE_CHANGE = "licenses changed from "
CONTENT_CHANGE = "content changed from version "
STARTS_OVER = (LICENSE_CHANGE, CONTENT_CHANGE)


def license_change(old, new) -> str:
    """The reason the depot writes on the decision that records a change of
    licenses from the list old to the list new."""
    return f"{LICENSE_CHANGE}{', '.join(old)} to {', '.join(new)}"


def content_change(old: int, new: int) -> str:
    """The reason the depot writes on the decision that records a change of
    content from the content version old to the new one."""
    return f"{CONTENT_CHANGE}{old} to version {new}"


def scores_change(set_names: list[str], removed: list[str]) -> str:
    """The reason the depot writes on the decision that records a change of
    scores: the names of the scores set and of those removed. It never
    starts over (starts_over)."""
    parts = ([f"set {', '.join(set_names)}"] if set_names else []) + (
        [f"removed {', '.join(removed)}"] if removed else [])
    return f"scores changed: {'; '.join(parts)}"


def starts_over(reason: str | None) -> bool:
    """Whether a decision's reason is in the form of license_change or
    content_change: the decisions before it no longer count."""
    return reason is not None and reason.startswith(STARTS_OVER)


def counted(decisions: list[dict]) -> list[dict]:
    """Of decisions in order, each with its status and reason, those that
    count: from the latest start over on, or every one without one."""
    start = max((at for at, one in enumerate(decisions) if starts_over(one["reason"])), default=0)
    return decisions[start:]


def admitted(decisions: list[dict]) -> bool:
    """Whether decisions in order, each with its status and reason, admit
    what they decide."""
    statuses = [one["status"] for one in counted(decisions)]
    return ADMITTED in statuses and REFUSED not in statuses


def status(decisions: list[dict]) -> str:
    """What decisions in order, each with its status and reason, make of
    what they decide: admitted, refused or under review."""
    statuses = [one["status"] for one in counted(decisions)]
    return REFUSED if REFUSED in statuses else ADMITTED if ADMITTED in statuses else UNDER_REVIEW


def _since(decisions: str) -> str:
    """The place in the JSON list of decision ids of the latest start over, 0 without one."""
    return (f"(SELECT coalesce(max(o.key), 0) FROM json_each({decisions}) o JOIN decisions c ON c.id = o.value "
            f"WHERE {_over('c.reason')})")


def _having(decisions: str, status: str) -> str:
    return (f"EXISTS (SELECT 1 FROM json_each({decisions}) h JOIN decisions d ON d.id = h.value "
            f"WHERE d.status = '{status}' AND h.key >= {_since(decisions)})")


def status_sql(decisions: str) -> str:
    """The same rule as status, as an SQL expression over a column of
    decision ids in order (a JSON array), with the depot's decisions table
    as `decisions`."""
    return (f"(CASE WHEN {_having(decisions, REFUSED)} THEN '{REFUSED}' WHEN {_having(decisions, ADMITTED)} "
            f"THEN '{ADMITTED}' ELSE '{UNDER_REVIEW}' END)")


def combined_sql(old: str, status: str, starts_over: bool) -> str:
    """The status of a text whose status was old once one more decision of
    that status is made on it, a start over (starts_over) or not: the same
    rule as status, from the status alone, as an SQL expression over a
    column of statuses old."""
    if starts_over or status == REFUSED:
        return f"'{status}'"
    if status == ADMITTED:
        return f"(CASE WHEN {old} = '{REFUSED}' THEN '{REFUSED}' ELSE '{ADMITTED}' END)"
    return old


def _over(reason: str) -> str:
    return " OR ".join(f"substr({reason}, 1, {len(one)}) = '{one}'" for one in STARTS_OVER)


def linked_status_sql(text: str, bound: str | None = None) -> str:
    """The same rule as status, as an SQL expression over the decisions on
    the text whose id is the SQL expression text, in the depot's table
    text_decisions (the decisions made at a change number up to the SQL
    expression bound, when it is given), in the order of their ids."""
    def within(alias: str) -> str:
        return f"{alias}.text = {text}" + ("" if bound is None else f" AND {alias}.change <= {bound}")
    since = (f"(SELECT max({ids.order_sql('o.decision')}) FROM text_decisions o JOIN decisions c ON c.id = o.decision "
             f"WHERE {within('o')} AND ({_over('c.reason')}))")

    def having(status: str) -> str:
        return (f"EXISTS (SELECT 1 FROM text_decisions h JOIN decisions d ON d.id = h.decision WHERE {within('h')} "
                f"AND d.status = '{status}' AND coalesce({ids.order_sql('h.decision')} >= {since}, 1))")
    return (f"(CASE WHEN {having(REFUSED)} THEN '{REFUSED}' WHEN {having(ADMITTED)} THEN '{ADMITTED}' "
            f"ELSE '{UNDER_REVIEW}' END)")
