"""The app's policy: how the decisions on a text or an erasure combine. The
depot stores every decision in order; it is this module, never the depot,
that says what they mean.

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
    """The place in the list of the latest start over, 0 without one."""
    over = " OR ".join(f"starts_with(c.reason, '{one}')" for one in STARTS_OVER)
    return (f"(SELECT coalesce(max(list_position({decisions}, c.id)), 0) FROM decisions c "
            f"WHERE list_contains({decisions}, c.id) AND ({over}))")


def _having(decisions: str, status: str) -> str:
    return (f"EXISTS (SELECT 1 FROM decisions d WHERE list_contains({decisions}, d.id) AND d.status = '{status}' "
            f"AND list_position({decisions}, d.id) >= {_since(decisions)})")


def admitted_sql(decisions: str) -> str:
    """The same rule as admitted, as a DuckDB condition over a column of
    decision ids in order, with the depot's decisions table as `decisions`."""
    return f"({_having(decisions, ADMITTED)} AND NOT {_having(decisions, REFUSED)})"


def status_sql(decisions: str) -> str:
    """The same rule as status, as a DuckDB expression over a column of
    decision ids in order, with the depot's decisions table as `decisions`."""
    return (f"(CASE WHEN {_having(decisions, REFUSED)} THEN '{REFUSED}' WHEN {_having(decisions, ADMITTED)} "
            f"THEN '{ADMITTED}' ELSE '{UNDER_REVIEW}' END)")
