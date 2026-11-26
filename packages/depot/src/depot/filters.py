"""A filter of the depot's texts, and the SQL condition over the texts view
it stands for, the condition a decision, a change of licenses or tags and
the other operations on many texts take.

A filter holds any of: the ids of chosen texts, tags (a text has every one),
licenses (a text names one of them), kinds (the content version its
current revision reads is of one of them, depot.kinds), statuses (the text has one of them,
the status the depot stores by depot.policy), a decision it is on, the words of a plain search,
a free SQL condition of the person's own, the ids of texts left out
(unticked), and the change number a page was read at (change). A text
matches when it matches every part given; a filter with none matches every
text.

The texts of a decision are found through the index of text_decisions by
decision (depot.schema) in one of two ways. Read from the decision's
rows, the condition reads every row of the decision, and only those, each
time it runs: what a query over every match needs, and what a page or a
count step needs for a decision on few texts, whose page then sorts them
all. Tested on each text, the condition looks up only the texts the query
reaches, so a page or a count step stays bounded by its own rows for a
decision on many texts, where reading every row of the decision at each
page or step would grow with the decision (Depot.condition chooses).

A plain search's words are split at spaces, and a text matches when each
one is in its title, or is its id (a word in the form of an id, is_id), or,
with in_text, is in its text itself too (a plain text's text, an item's
JSON content). A word matches in a title or a text in any case: SQLite's
LIKE ignores the case of ASCII letters, and a % or _ in a
word is that character, not a wildcard. With whole_words, a word matches
only where no letter, digit or underscore stands right before or after it
(the SQL function has_word, depot.database), so "rivers" never matches
"drivers"; it ignores the case of ASCII letters as LIKE does. On the
title, LIKE picks the candidates first; on the text, put together from its
parts, the function reads it alone, so each text is put together once. A
page reads the texts a search may match without in_text from the title
index and the ids named (depot.titles, Depot.search), never every
text, when a word has 3 characters or more.

A text has a tag when it has a row of taggings for it that it has not left
(depot.taggings), looked up by its key.

With a change number C, a text matches only when nothing changed it after
C (changed_since): its current revision was made at C or before, no
decision was made on it after C (a change of its tags links one to it), and
no pull brought rows into the depot
after C (a pull brings rows numbered by the upstream,
which can be lower than C). So an operation on the texts a
page showed at C never reaches a text that changed since, even one that
came to match the filter only after C; such texts are left out, and the
operation says how many. A list of ids (chosen or unticked) is written as
one JSON string, which SQLite reads as one value whatever its length.
"""

import json
from dataclasses import dataclass

from . import ids
from .errors import Refused
from .schema import ADMITTED, REFUSED, UNDER_REVIEW

STATUSES = (UNDER_REVIEW, ADMITTED, REFUSED)


# the index of the key of texts_revisions (id, revision), which SQLite names, for the queries that name the index they
# read (KEY_INDEXES: each with its table)
TEXTS_KEY = "sqlite_autoindex_texts_revisions_1"
KEY_INDEXES = {TEXTS_KEY: "texts_revisions"}


@dataclass(frozen=True)
class Filter:
    ids: tuple[str, ...] | None = None  # None: any text; empty: no text
    tags: tuple[str, ...] = ()
    licenses: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    where: str | None = None
    unticked: tuple[str, ...] = ()
    change: int | None = None
    decision: str | None = None  # a decision the text is on
    search: str | None = None  # the words of a plain search
    in_text: bool = False  # the search reads the text itself too
    whole_words: bool = False  # each word of the search matches only as a whole word
    kinds: tuple[str, ...] = ()  # the kinds the text's current content version may be of


def condition(found: Filter, latest: int | None = None, decision_rows: bool = True) -> str:
    """The SQL condition over the texts view, named texts, that matches the
    texts the filter matches. latest: the depot's change number now, when
    known; a filter's change number at or past it leaves out no text, so
    its condition is left out. decision_rows: the texts of the filter's
    decision are read from the decision's rows, else tested one by one, as
    the module says. Refused for a status that is none of STATUSES."""
    parts = []
    if found.ids is not None:
        parts.append(f"id IN {listed(found.ids)}" if found.ids else "false")
    for tag in found.tags:
        parts.append(f"EXISTS (SELECT 1 FROM taggings g WHERE g.text = texts.id AND g.tag = {literal(tag)} "
                     "AND g.left_change IS NULL)")
    if found.licenses:
        parts.append(f"EXISTS (SELECT 1 FROM json_each(licenses) WHERE value IN {strings(found.licenses)})")
    if found.kinds:
        chosen = list(dict.fromkeys(found.kinds))
        parts.append(f"kind = {literal(chosen[0])}" if len(chosen) == 1 else f"kind IN {strings(chosen)}")
    if found.statuses:
        unknown = [one for one in found.statuses if one not in STATUSES]
        if unknown:
            raise Refused(f"a text's status is one of {', '.join(STATUSES)}, not {unknown[0]!r}")
        chosen = list(dict.fromkeys(found.statuses))
        parts.append(f"status = {literal(chosen[0])}" if len(chosen) == 1 else f"status IN {strings(chosen)}")
    if found.decision is not None:
        decision = literal(found.decision)
        if decision_rows:
            parts.append(f"id IN (SELECT l.text FROM text_decisions l WHERE l.decision = {decision})")
        else:
            parts.append(f"EXISTS (SELECT 1 FROM text_decisions l WHERE l.text = texts.id AND l.decision = {decision})")
    for word in words(found.search):
        parts.append(_word(word, found.in_text, found.whole_words))
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
    return (f"(texts.change > {number} "
            f"OR EXISTS (SELECT 1 FROM text_decisions l WHERE l.text = texts.id AND l.change > {number}) "
            f"OR EXISTS (SELECT 1 FROM pulls p WHERE p.change > {number}))")


def words(search: str | None) -> list[str]:
    """The words of a plain search: what is typed, split at spaces."""
    return (search or "").split()


def is_id(word: str) -> bool:
    """Whether a word of a plain search is in the form of an id: a depot's
    name, a hyphen and a number written without leading zeros
    (depot.ids)."""
    number = ids.number(word)
    return number is not None and ids.NAME.fullmatch(ids.owner(word)) is not None and word == ids.make(
        ids.owner(word), number)


def _word(word: str, in_text: bool, whole: bool) -> str:
    """Whether a text holds the word of a plain search in its title, or
    the word is its id, or with in_text the text holds it, as the module
    says: an SQL condition over the texts view."""
    pattern = literal("%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    def has(column):
        return f"has_word({column}, {literal(word)})"

    column = "coalesce(title, '')"
    test = f"{column} LIKE {pattern} ESCAPE '\\'"
    # LIKE first, which picks the candidates the function then reads
    found = [f"({test} AND {has(column)})" if whole else test]
    if is_id(word):
        found.append(f"id = {literal(word)}")
    if in_text:
        # the text is put together from its parts each time it is read, so a whole word reads it once, through
        # the function alone, which finds the word wherever LIKE would
        column = "coalesce(text, content, '')"
        found.append(has(column) if whole else f"{column} LIKE {pattern} ESCAPE '\\'")
    return f"({' OR '.join(found)})"


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
