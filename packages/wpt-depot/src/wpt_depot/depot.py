"""A depot: texts and structured items, the decisions on them, libraries,
erasures and datasets, in the tables of one SQLite file (wpt_depot.schema).

Depot is the only code that changes a depot's tables. Each operation is one
change: one transaction under the next change number, recorded for the
backups (wpt_depot.database). It checks the whole change against the
depot's rules inside it, so a refused or failed change writes nothing.

The depot's state is numbered by its change number (Depot.change): every
write raises it, and every revision row and membership records the change
number it was made at. What changes in place keeps its old versions as
revision rows of its own table, never overwritten or deleted but by an
erasure: a text's credits, licenses and decisions, a library's name, lock
and problems, a dataset, the depot's maintainer and takedown contact, an
erasure request's decisions and what it erased.

The rules (the app's policy, wpt_depot.policy, says how decisions combine):

- Every id is the depot's name, a hyphen and a number (wpt_depot.ids).
- A new text or item comes with one decision 'under review' and joins no
  library. Its text, or an item's kind and content, is its content version
  1, stored as its parts (wpt_depot.parts): a plain text may come from a
  file read as a stream (FileText), so no text is ever too big to add. An
  item's content matches its kind's schema, as the kinds table
  stores it. Every license a text names is a
  row of licenses (one of them is enough to allow it), and its notice is
  stored once.
- An operation on texts (a decision, a put, a take-out, a fork's texts
  left out, a report) takes them as a selection: an SQL condition over the texts
  view (the current state), matched when the operation runs, or the exact
  ids of chosen texts, every one of which must be a text of the depot.
- A decision on texts is appended to every text selected, as a new
  revision, with an optional public reason. Admitting is refused when a
  text it covers has no allowed license (wpt_depot.licenses). A refusal
  takes each of them out of every open library. Depot.blocked says, for
  given texts, which ones an admission cannot cover and why; a decision
  applies the same check. A person's reason never takes the forms the depot
  keeps for license and content changes (wpt_depot.policy).
- A text's licenses change as a new revision, its id kept: the change is
  recorded as a decision 'under review' whose reason the depot writes,
  naming the old and new lists (one decision per distinct old list), and it
  takes the text out of every open library, since the decisions before it
  no longer count (wpt_depot.policy). Every new license is a row of licenses.
- A text's content (its plain text, or an item's content, which still
  matches its kind's schema) changes as a new content version stored next
  to the old ones, and a new revision of the text that reads it, its id
  kept. The new version stores only the parts its text holds that no
  version of the text held before. The change is recorded as a decision 'under review' made by the
  person's declaration, whose reason the depot writes naming the old and
  new versions; the decisions before it no longer count (wpt_depot.policy),
  so the text leaves every open library. A content that changes nothing
  writes nothing. A locked library keeps reading the content version of
  the revision it pinned.
- A text's credits (title, authors, tags, address, notice) change as a new
  revision, its id kept: the change is recorded as a decision 'under
  review' made by the person's declaration, whose reason the depot writes,
  naming each credit's old and new value. The decisions before it keep
  counting, so the text keeps its status and its libraries.
- A text's scores (each a name, a number and the content version it was
  made from) change as a new revision of every text a change names, their
  ids kept: one decision 'under review' made by the person's declaration
  covers them all, whose reason the depot writes naming the scores set and
  removed. The decisions before it keep counting, as for credits. A score
  set under a name the text has replaces it; a text whose scores stay as
  they are is left out, and a change that leaves out every text writes
  nothing. The change reads its lines a few at a time into a temporary
  table and is checked and made by queries over it (wpt_depot.scoring),
  so its cost in memory does not grow with the texts it names. Every other change of a text (its content, licenses, credits,
  decisions) keeps its scores as they are, each still naming the content
  version it was made from.
- Only admitted texts join a library. An open library reads the current
  revision of each text it holds, and can be read as it stood at any past
  change number of the depot that owns it (Depot.library_texts); a text
  of another depot it holds, in a clone, is read as the pulls up to that
  change brought it (Depot.bounds). Locking a library pins the revision
  of each text it holds: it reads those revisions for good, whatever
  revisions its texts take later, takes no new text and loses none but by
  an erasure. A text taken out of an open library leaves it and keeps its
  status, admitted included. Its name is not what it holds, so it can be
  renamed, open or locked. A fork is a new open library holding the same
  texts that are admitted now, less those selected to leave out when it is
  given any.
- An erasure request is decided like a text, and several requests can be
  decided together, each checked before any is decided. Each needle names
  a place in one content version of its text, which the request records
  (its current one when the request is asked, unless it names one). Once
  admitted, the app carries it out at once (wpt_depot.erasure): every content version of the texts its needles
  name is cut in place, the passage at the place named and every
  occurrence of it in each version, so the cut reaches every revision: of
  a plain text, each part that holds a character cut is replaced by its
  cut text, a new part, and each version lists the new parts instead (an
  item is cut at a field of its JSON, every cut version is validated
  against its kind's schema again and its parts cut again); a part no
  version lists any longer goes. A text named whole is removed with every
  revision, content version, part, membership and pin; its
  `erased` gets each passage's length and hashes, and the file keeps no
  copy of what it removed (wpt_depot.database). The hashes are public and
  keyless (wpt_depot.erasure), so any depot or clone can match an erased
  passage.
- A problem with a library, locked or not, is reported on the library
  (`problems`): the texts concerned, the problem in words, and the
  decision that records who reported it, 'under review' like an add. It
  changes no membership and no text, so a locked library takes it too.
- A dataset names libraries, of this depot or another, each with a
  change number of the depot that owns it, which it reads (a clone naming
  one of its upstream's libraries by its own address names it at the
  upstream change its pulls brought), and optional ordered shares by tag.
  Its name, libraries and shares change as a new revision, its id kept,
  with the checks of a new one; a library it read already keeps its change
  number. Exported, a dataset is a remote (wpt_depot.remote).
- The depot's maintainer and takedown contact change as a new revision;
  its name never does, since it begins every id. A library's name, a
  dataset and the depot's facts change with no declaration: they are no
  decision on a text.
- The depot's format (wpt_depot.schema) changes only through
  Depot.upgrade (wpt_depot.upgrade); opening or reading a depot of another
  format than the code's is refused (OtherFormat).
- A depot that follows an upstream (a clone, wpt_depot.follow) only reads
  the rows its upstream owns: those whose id does not carry the depot's
  own name, that is, is not exactly the name, a hyphen and digits
  (wpt_depot.ids). Every operation that would change one is refused,
  writing nothing (BelongsToUpstream): a decision on, a license change of,
  a content change of or an edit of the credits or scores of an upstream text, a request to erase one,
  a decision on an erasure request that is the upstream's or names an
  upstream text, renaming, locking, reporting
  on, putting texts into or taking texts out of an upstream library,
  editing an upstream dataset. An operation on several texts is refused whole when one of
  them is the upstream's, and Depot.blocked names them. The depot's own
  rows stay fully changeable: its own library may hold upstream texts,
  reading their current revisions while it is open and the revisions it
  pinned once it is locked, and a fork of an upstream library is a
  library of its own. An upstream text a pull brings refused, or with its
  licenses or content changed, leaves the depot's own open libraries at the
  pull's change, as a refusal or such a change made here would
  (wpt_depot.follow). A dataset reads one of the upstream's libraries
  here whether it names it by this depot's address or by the upstream's
  (Depot.reads_here).
"""

import bisect
import functools
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import apsw

from . import erasure, ids, kinds, licenses, policy, scoring
from .database import Database, person_columns
from .erasure import Needle
from .errors import BelongsToUpstream, DepotExists, Locked, NotADepot, NotFound, OtherFormat, Refused, UnsupportedAddress
from .filters import IN_LOCKED
from .parts import WINDOW, FileText, Parted, Reader, Source, listed, made, pack, parted, split
from .schema import (ADMITTED, CONTENT_JOIN, FORMAT, JSON_COLUMNS, LISTED_COLUMNS, REFUSED, UNDER_REVIEW)

# texts an operation works on: an SQL condition over the texts view, matched when the operation runs, or the exact
# ids of chosen texts
Selection = str | Sequence[str]
# why a decision cannot cover a text (Depot.blocked): none of its licenses is allowed (an admission only), or it is
# a row the upstream of a clone owns (any decision)
NO_ALLOWED_LICENSE = "no allowed license"
BELONGS_TO_UPSTREAM = "belongs to the upstream"
# how many ids a refusal of upstream rows names before it gives the count of the rest
NAMED = 20
# what a clone can do instead of changing an upstream library: a fork is a library of its own
FORK = ", or fork it into a library of this depot"
SINGULAR = {"texts": "text", "libraries": "library", "erasure requests": "erasure request", "datasets": "dataset"}
# the credits of a text that change as a new revision (Depot.edit_credits), in the order its reason names them
CREDITS = ("title", "authors", "tags", "address", "notice")
# what the rows of a search can be ordered by: each an ORDER BY list over the searched rows
ORDERS = {"id": [ids.order_sql()], "title": ["title"], "kind": ["kind"], "status": ["status"], "tags": ["tags"],
          "licenses": ["licenses"], "libraries": ["libraries"], "length": ["length"]}
# the columns of the texts view but those that hold a text: what a query reads that never reads a text's parts
FIELDS = ("id, revision, change, kind, content_version, length, licenses, notice, title, authors, tags, address, "
          "decisions, scores, libraries")


class _Nothing(Exception):
    """Raised inside a write that ends with nothing to write: the write
    rolls back and the operation answers value."""

    def __init__(self, value):
        super().__init__()
        self.value = value


def _answers(method):
    """The method, answering a _Nothing's value when one ends its write (rolled back, nothing written)."""
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except _Nothing as nothing:
            return nothing.value
    return wrapped


@dataclass(frozen=True)
class Declaration:
    """Who decides: a declared legal name and email, and the confirmation
    that they read the requirements, accept them, and that what they
    submit meets them. Recorded on each decision, public, never verified."""
    name: str
    email: str
    accepted: bool


@dataclass(frozen=True)
class About:
    name: str
    maintainer: str
    takedown_contact: str
    upstream: str | None = None


@dataclass(frozen=True)
class NewText:
    """A plain text (text: a string, or a FileText read as a stream) or a
    structured item (kind and its JSON content), with its credits. notice
    is the notice's text, stored once."""
    text: Source | None = None
    kind: str | None = None
    content: str | None = None
    licenses: tuple[str, ...] = ()
    notice: str | None = None
    title: str | None = None
    authors: tuple[str, ...] | None = None
    tags: tuple[str, ...] = ()
    address: str | None = None


@dataclass
class Depot:
    """The depot in the SQLite file `address` (a path or a file://
    address), opened with open() or view() or made with new(); readonly:
    every connection opens the file read-only (view)."""
    address: str
    readonly: bool = False
    db: Database = field(init=False)

    def __post_init__(self):
        self.db = Database(location(self.address), self.readonly)
        self.address = str(self.db.path)
        self._name = None
        self._upstream = ...  # not read yet; None: it follows no upstream
        self._names = {}

    # Opening

    @classmethod
    def new(cls, address: str, about: About) -> "Depot":
        """An empty depot in a new file at address, with its about row
        (change 1) and the starting kinds (wpt_depot.kinds). DepotExists
        when the address holds a file; Refused when its name cannot begin an
        id (wpt_depot.ids)."""
        ids.check_name(about.name)
        if location(address).exists():
            raise DepotExists(f"{location(address)} already exists: a new depot is a new file")
        depot = cls(address)
        depot.db.create()
        with depot.db.write("new") as (con, number):
            _insert(con, "kinds", kinds.starting_rows())
            _insert(con, "about_revisions", [{"revision": 1, "change": number, **vars(about), "format": FORMAT}])
        return depot

    @classmethod
    def view(cls, address: str) -> "Depot":
        """The depot at address, to read: every connection opens its file
        read-only, so nothing is written. NotADepot when the address holds
        none; OtherFormat when its format is not this code's
        (schema.FORMAT)."""
        found = format_of(address)
        depot = cls(address, readonly=True)
        if found != FORMAT:
            raise OtherFormat(other_format(depot.address, found))
        return depot

    @classmethod
    def open(cls, address: str) -> "Depot":
        """The depot at address, to write: the change files a crash left
        pending kept or dropped first (wpt_depot.database). NotADepot when
        the address holds none; OtherFormat when its format is not this
        code's (schema.FORMAT)."""
        found = format_of(address)
        depot = cls(address)
        if found != FORMAT:
            raise OtherFormat(other_format(depot.address, found))
        depot.db.settle()
        return depot

    @classmethod
    def upgrade(cls, address: str, to: str | None = None, upstream: str | None = None) -> tuple["Depot", int]:
        """(the depot at address brought up to this code's format, the
        format it had), as wpt_depot.upgrade.upgrade does."""
        from .upgrade import upgrade
        return upgrade(address, to, upstream)

    def format(self) -> int:
        """The format of the depot's layout (wpt_depot.schema): its current about row's."""
        return self._one("SELECT format FROM about")

    def change(self) -> int:
        """The depot's change number: the number of its latest change, past
        every change number its own rows record."""
        return self._one("SELECT coalesce(max(id), 0) FROM changes")

    def about(self) -> dict:
        return self._rows("SELECT * FROM about")[0]

    def name(self) -> str:
        """The depot's name, which begins its ids."""
        if self._name is None:
            self._name = self.about()["name"]
        return self._name

    def upstream(self) -> str | None:
        """The web address of the depot this one follows, None when it follows none."""
        if self._upstream is ...:
            self._upstream = self.about()["upstream"]
        return self._upstream

    def upstream_owned(self, row_ids: Sequence[str]) -> list[str]:
        """Of the ids of rows (texts, libraries, erasure requests, datasets),
        those the upstream owns, each once, in the order given: in a depot
        that follows an upstream, every id that does not carry this depot's
        own name (wpt_depot.ids.carries); in any other depot, none."""
        if self.upstream() is None:
            return []
        return [one for one in dict.fromkeys(row_ids) if not ids.carries(self.name(), one)]

    def upstream_sql(self, column: str = "id") -> str:
        """The same rule as upstream_owned, as an SQL condition over a column of ids."""
        if self.upstream() is None:
            return "false"
        return f"NOT {ids.carries_sql(self.name(), column)}"

    def _refuse_upstream(self, what: str, row_ids: Sequence[str], instead: str = "", total: int | None = None):
        """BelongsToUpstream when one of the rows named is the upstream's:
        what names their kind in the plural ('texts'), instead what else
        can be done, after asking the upstream; total, when given, how many
        of the rows the upstream owns, of which row_ids are the first."""
        theirs = self.upstream_owned(row_ids)
        if theirs:
            total = len(theirs) if total is None else total
            listed = ", ".join(theirs[:NAMED]) + (f" and {total - NAMED} more" if total > NAMED else "")
            subject = (f"the {SINGULAR[what]} {listed} belongs" if total == 1
                       else f"{total} of the {what} belong ({listed})")
            raise BelongsToUpstream(f"{subject} to the upstream {self.upstream()}: this depot follows it and changes "
                                    f"none of its rows, so nothing is written; ask the upstream{instead}")

    def edit_about(self, maintainer: str | None = None, takedown_contact: str | None = None):
        """Change the depot's maintainer, its takedown contact, or both (None:
        kept), as a new revision of its about row. Its name never changes: it
        begins every id. Refused for an empty value or when neither is given."""
        values = {name: value for name, value in (("maintainer", maintainer),
                                                  ("takedown_contact", takedown_contact)) if value is not None}
        if not values:
            raise Refused("give a maintainer, a takedown contact or both")
        _filled(**values)
        with self._write("about") as (con, number):
            self._revise_about(con, number, {name: value.strip() for name, value in values.items()})

    def _revise_about(self, con, number: int, values: dict):
        current = self._rows("SELECT * FROM about")[0]
        _insert(con, "about_revisions", [{**current, **values, "revision": current["revision"] + 1,
                                          "change": number}])

    # Reading

    def sql(self, query: str) -> list[dict]:
        """The rows of one query that reads (wpt_depot.database) over the
        depot's tables and views, by their names (wpt_depot.schema), each a
        dict of SQLite's values (JSON text, bytes and numbers as they are).
        BadQuery for anything else."""
        columns, found = self.answer(query)
        return [dict(zip(columns, one)) for one in found]

    def rows(self, query: str, params=None) -> list[dict]:
        """The rows of a query the app itself writes (never a person's SQL,
        which goes through sql), each list column read from its JSON as
        Python lists and objects, a part's hash and an erased passage's
        hashes as bytes."""
        return self._rows(query, params)

    def answer(self, query: str) -> tuple[list[str], Iterator[tuple]]:
        """(the column names, the rows as tuples, given as they are read) of
        one query that reads. BadQuery for anything else."""
        return person_columns(self.db.person(), query), self.db.person_query(query)

    def show(self, text: str) -> dict:
        """{text: its current row, status (wpt_depot.policy), notice: its
        text, decisions: their rows in order, each with `counts`: whether
        the policy counts it (none before the latest license or content
        change, wpt_depot.policy),
        libraries: the rows of the libraries holding it, each with upstream:
        whether the upstream of this clone owns it, revisions: how many
        revisions it has, pins: {library, name, revision} of each locked
        library holding it and the revision it pinned, content_versions: how
        many content versions it has (its current revision reads the one
        its row's content_version names), upstream: whether the upstream of
        this clone owns it}. Its text, or an item's content, is read from its
        parts, whatever its size. NotFound without such a text."""
        found = self._rows(f"SELECT {FIELDS} FROM texts WHERE id = ?", (text,))
        if not found:
            raise NotFound(f"no text {text}")
        row = found[0]
        whole = self.content(text, row["content_version"])
        row = {**row, "text": whole if row["kind"] is None else None, "content": None if row["kind"] is None else whole}
        decided = {one["id"]: one for one in self._rows(
            "SELECT * FROM decisions WHERE id IN (SELECT value FROM json_each(?))", (_json(row["decisions"]),))}
        held = [{**one, "upstream": bool(one["upstream"])} for one in self._rows(
            f"SELECT *, {self.upstream_sql()} AS upstream FROM libraries "
            "WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id", (_json(row["libraries"]),))]
        notice = None if row["notice"] is None else self._one("SELECT text FROM notices WHERE id = ?", (row["notice"],))
        revisions = self._one("SELECT count(*) FROM texts_revisions WHERE id = ?", (text,))
        versions = self._one("SELECT count(*) FROM content_lists WHERE id = ?", (text,))
        pins = self._rows("SELECT p.library, l.name, p.revision FROM pins p JOIN libraries l ON l.id = p.library "
                          "WHERE p.text = ? ORDER BY p.library", (text,))
        decisions = [decided[one] for one in row["decisions"]]
        counting = {one["id"] for one in policy.counted(decisions)}
        return {"text": row, "status": policy.status(decisions), "notice": notice,
                "decisions": [{**one, "counts": one["id"] in counting} for one in decisions], "libraries": held,
                "revisions": revisions, "pins": pins, "content_versions": versions,
                "upstream": bool(self.upstream_owned([text]))}

    def search(self, where: str, offset: int = 0, limit: int = 50, order: str = "id", descending: bool = False,
               every_id: bool = False) -> dict:
        """The texts the SQL condition where matches, a page at a time:
        {total: how many it matches, statuses: how many of them are under
        review, admitted, refused (wpt_depot.policy), locked: how many of
        them a locked library holds, tags and licenses: how many of them
        have each tag and name each license, the most first, rows: the page
        of limit rows from offset, ordered by the column order (one of
        ORDERS; then by id), ids: with every_id, the id of every text it
        matches, in the order of their ids}. A row holds a text's credits,
        never its text or content: its id, kind, title, authors, tags,
        licenses, libraries, status, whether a locked library holds it, and
        its length in characters. BadQuery for a condition that is not one;
        Refused for an order that is none of ORDERS."""
        matched, by = self._ordered(where, order, descending)
        counts = self.db.person_rows(f"SELECT status, locked, count(*) AS n FROM ({matched}) GROUP BY status, locked")
        statuses = {status: sum(one["n"] for one in counts if one["status"] == status)
                    for status in (UNDER_REVIEW, ADMITTED, REFUSED)}
        found = [_decoded({**one, "locked": bool(one["locked"])}) for one in self.db.person_rows(
            f"SELECT id, kind, title, authors, tags, licenses, libraries, status, locked, length "
            f"FROM ({matched}) ORDER BY {by} LIMIT {int(limit)} OFFSET {int(offset)}")]
        named = {column: self.db.person_rows(
            f"SELECT j.value AS {name}, count(*) AS texts FROM (SELECT {column} FROM texts WHERE ({where})) t, "
            f"json_each(t.{column}) j GROUP BY j.value ORDER BY texts DESC, j.value")
                 for column, name in (("tags", "tag"), ("licenses", "license"))}
        answer = {"total": sum(statuses.values()), "statuses": statuses,
                  "locked": sum(one["n"] for one in counts if one["locked"]), **named, "rows": found}
        if every_id:
            answer["ids"] = self.matching(where)
        return answer

    def passages(self, where: str, offset: int = 0, limit: int = 50, characters: int = 200_000, order: str = "id",
                 descending: bool = False) -> dict:
        """The texts the SQL condition where matches with their text, a page
        at a time, in the order of search: {rows: from offset, at most limit
        texts holding at most characters characters together, each {id,
        kind, title, length: its length in characters, passage: its text, or
        an item's content, from its start, whole, or cut at characters for a
        first text longer than that}, next: the offset of the text after the
        page, None when the page ends the texts matched}. A later text that
        does not fit whole starts the next page. BadQuery for a condition
        that is not one; Refused for an order that is none of ORDERS or a
        bound below 1."""
        if limit < 1 or characters < 1:
            raise Refused("a page holds at least one text and one character")
        matched, by = self._ordered(where, order, descending)
        found = self.db.person_rows(f"SELECT id, kind, title, length FROM ({matched}) ORDER BY {by} "
                                    f"LIMIT {int(limit) + 1} OFFSET {int(offset)}")
        page, left = [], characters
        for one in found[:limit]:
            if page and one["length"] > left:
                break
            page.append({**one, "shown": min(one["length"], left)})
            left -= page[-1]["shown"]
        listed_parts = {one["id"]: one["parts"] for one in self._rows(
            f"SELECT c.id, c.parts FROM texts_revisions t {CONTENT_JOIN} WHERE t.latest "
            "AND t.id IN (SELECT value FROM json_each(?))", (_json([one["id"] for one in page]),))} if page else {}
        reader = self.reader()
        found_rows = [{"id": one["id"], "kind": one["kind"], "title": one["title"], "length": one["length"],
                       "passage": Parted(reader, one["id"], listed_parts[one["id"]])[:one["shown"]]} for one in page]
        return {"rows": found_rows, "next": int(offset) + len(page) if len(page) < len(found) else None}

    def _ordered(self, where: str, order: str, descending: bool) -> tuple[str, str]:
        """(a query of the texts where matches, with their status, whether a
        locked library holds them and their length; the ORDER BY list of
        order, then by id). Refused for an order that is none of ORDERS."""
        if order not in ORDERS:
            raise Refused(f"the texts are ordered by one of {', '.join(ORDERS)}, not {order!r}")
        direction = " DESC" if descending else ""
        by = ", ".join([f"{one}{direction}" for one in ORDERS[order]] + [ids.order_sql()])
        matched = (f"SELECT {FIELDS}, {policy.status_sql('decisions')} AS status, {IN_LOCKED} AS locked "
                   f"FROM texts WHERE ({where})")
        return matched, by

    def tags(self) -> list[dict]:
        """Every tag a text has, with how many texts have it, by name."""
        return self._rows("SELECT j.value AS tag, count(*) AS texts FROM texts_revisions t, json_each(t.tags) j "
                          "WHERE t.latest GROUP BY j.value ORDER BY j.value")

    def matching(self, where: str) -> list[str]:
        """The ids of the texts the SQL condition where matches, in the
        numeric order of their ids (wpt_depot.ids)."""
        return [one for (one,) in self.db.person_query(
            f"SELECT id FROM texts WHERE ({where}) ORDER BY {ids.order_sql()}")]

    def selected(self, texts: Selection) -> list[str]:
        """The ids of the texts a selection names, each once, in the numeric
        order of their ids: those an SQL condition matches now, or exactly
        the ids given. NotFound when a given id is no text of the depot."""
        if isinstance(texts, str):
            return self.matching(texts)
        given = sorted(set(texts))
        found = {one for (one,) in self.db.reader().execute(
            "SELECT id FROM texts_revisions WHERE latest AND id IN (SELECT value FROM json_each(?))", (_json(given),))}
        missing = [one for one in given if one not in found]
        if missing:
            raise NotFound(f"{len(missing)} of the texts named are no text of the depot (the first: {missing[0]})")
        return sorted(given, key=ids.order_key)

    def latest_decisions(self, limit: int = 10) -> list[dict]:
        """The latest decisions, the newest first, each with how many texts
        and erasure requests it decides and the library whose reported
        problem it records (None for any other)."""
        return self._rows(
            f"SELECT d.*, {_DECIDED_TEXTS} AS texts, "
            "(SELECT count(*) FROM erasures_revisions e WHERE e.latest "
            "AND EXISTS (SELECT 1 FROM json_each(e.decisions) WHERE value = d.id)) AS erasures, "
            "(SELECT min(l.id) FROM libraries_revisions l, json_each(coalesce(l.problems, '[]')) p "
            "WHERE l.latest AND p.value ->> 'decision' = d.id) AS library "
            f"FROM decisions d ORDER BY d.made_at DESC, d.id DESC LIMIT {int(limit)}")

    def decision(self, decision: str) -> dict:
        """{decision: its row, texts: how many texts it is on, erasures: the
        ids of the erasure requests it decides, problem: the reported
        problem it records, with its library ({library, texts, problem}),
        or None}. NotFound without such a decision."""
        found = self._rows(f"SELECT d.*, {_DECIDED_TEXTS} AS texts FROM decisions d WHERE d.id = ?", (decision,))
        if not found:
            raise NotFound(f"no decision {decision}")
        row = found[0]
        texts = row.pop("texts")
        erasures = [one for (one,) in self.db.reader().execute(
            "SELECT e.id FROM erasures e WHERE EXISTS (SELECT 1 FROM json_each(e.decisions) WHERE value = ?) "
            "ORDER BY e.id", (decision,))]
        problem = next(({"library": one["id"], "texts": problem["texts"], "problem": problem["problem"]}
                        for one in self._rows("SELECT id, problems FROM libraries ORDER BY id")
                        for problem in one["problems"] or [] if problem["decision"] == decision), None)
        return {"decision": row, "texts": texts, "erasures": erasures, "problem": problem}

    def bounds(self, depot: str, change: int) -> dict[str, int]:
        """What the depot called depot held at its change number change, as
        {the name of each depot whose rows it held: the highest change number
        of that depot's numbering it held}: its own rows up to change, and,
        when it follows an upstream, the upstream's rows up to the upstream
        change brought by its last pull at or before change (pulls, its
        clone included), and so on back through the upstream's own pulls."""
        pulls = self._rows("SELECT * FROM pulls")
        found = {}
        while depot not in found:
            found[depot] = change
            last = max((one for one in pulls if one["depot"] == depot and one["change"] <= change),
                       key=lambda one: one["change"], default=None)
            if last is None:
                break
            depot, change = last["upstream"], last["upstream_change"]
        return found

    def library_texts(self, library: str, change: int) -> list[dict]:
        """Every text the library held at the change number change of the
        depot that owns it (the one its id names), each at its revision
        then, with the content version that revision reads, as it is now (an
        erasure's cut reaches every version, and a text an erasure removed is
        in no past), as the list of its parts and its bytes, never its text
        (read from its parts, wpt_depot.parts): for a library locked
        at that change, the revisions it pinned; for an open one, the texts
        that had joined and not left it, each at its latest revision of that
        state (bounds: a text of another depot, such as one a pull brought
        to a clone, at its latest revision numbered within what the pulls
        up to that change brought). The columns of schema.LISTED_COLUMNS, in
        the numeric order of ids. NotFound without such a library, or when
        it did not exist yet at that change."""
        self._library(library)
        state = self._rows("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                           "ORDER BY revision DESC LIMIT 1", (library, change))
        if not state:
            raise NotFound(f"the library {library} did not exist yet at the change {change}")
        if state[0]["locked_at"] is not None:
            held = "SELECT p.text AS member, p.revision FROM pins p WHERE p.library = :library"
        else:
            held = ("SELECT m.text AS member, (SELECT max(r.revision) FROM texts_revisions r WHERE r.id = m.text "
                    f"AND r.change <= (SELECT b.value FROM json_each(:bounds) b WHERE b.key = {ids.owner_sql('r.id')})"
                    ") AS revision FROM memberships m WHERE m.library = :library "
                    "AND m.joined_change <= :change AND (m.left_change IS NULL OR m.left_change > :change)")
        query = (f"SELECT {LISTED_COLUMNS} FROM ({held}) h JOIN texts_revisions t ON t.id = h.member "
                 f"AND t.revision = h.revision {CONTENT_JOIN} ORDER BY {ids.order_sql('t.id')}")
        return self._rows(query, {"library": library, "change": change,
                                  "bounds": _json(self.bounds(ids.owner(library), change))})

    def pinned(self, library: str, offset: int = 0, limit: int = 50) -> dict:
        """The revisions a locked library pinned, a page at a time: {total:
        how many texts it pinned, newer: how many of them have a current
        revision newer than the one pinned, rows: limit rows from offset in
        the numeric order of ids, each {id, title: the pinned revision's,
        revision: the one pinned, current: the text's current revision}}.
        An open library pins none. NotFound without such a library."""
        self._library(library)
        query = ("SELECT p.text AS id, t.title, p.revision, c.revision AS current FROM pins p "
                 "JOIN texts_revisions t ON t.id = p.text AND t.revision = p.revision "
                 "JOIN texts_revisions c ON c.id = p.text AND c.latest WHERE p.library = ?")
        counts = self._rows(f"SELECT count(*) AS total, count(*) FILTER (WHERE current > revision) AS newer "
                            f"FROM ({query})", (library,))[0]
        found = self._rows(f"SELECT * FROM ({query}) ORDER BY {ids.order_sql()} LIMIT {int(limit)} "
                           f"OFFSET {int(offset)}", (library,))
        return {**counts, "rows": found}

    # Licenses

    def licenses(self) -> list[dict]:
        """Every license the depot holds, by id: its id and name, whether it
        is allowed (wpt_depot.licenses, a combined row only if every part
        is), and how many texts name it."""
        named = {one["id"]: one["texts"] for one in self._rows(
            "SELECT j.value AS id, count(*) AS texts FROM texts_revisions t, json_each(t.licenses) j "
            "WHERE t.latest GROUP BY j.value")}
        return [{"id": one["id"], "name": one["name"], "allowed": licenses.allowed(one["id"]),
                 "texts": named.get(one["id"], 0)} for one in self._rows("SELECT id, name FROM licenses ORDER BY id")]

    def license(self, license_id: str) -> dict:
        """One license: its id, name, official text, whether it is allowed
        (as licenses() says) and how many texts name it. NotFound without
        it."""
        found = self._rows("SELECT l.*, (SELECT count(*) FROM texts_revisions t, json_each(t.licenses) j "
                           "WHERE t.latest AND j.value = l.id) AS texts FROM licenses l WHERE l.id = ?", (license_id,))
        if not found:
            raise NotFound(f"no license {license_id}")
        return {**found[0], "allowed": licenses.allowed(license_id)}

    def add_license(self, license_id: str, name: str, text: str):
        """An official license text, stored once. The id is an SPDX name, a
        license with its exception ('Apache-2.0 WITH LLVM-exception') or
        licenses that apply together ('MIT AND CC-BY-4.0'). Refused when the
        id is held already."""
        _filled(license_id=license_id, name=name, text=text)
        with self._write("license") as (con, number):
            if self._rows("SELECT id FROM licenses WHERE id = ?", (license_id,)):
                raise Refused(f"the license {license_id} is held already")
            _insert(con, "licenses", [{"id": license_id, "name": name, "text": text}])

    # Texts and decisions

    def add(self, texts: list[NewText], declaration: Declaration) -> list[str]:
        """Add texts and items, under review, with one decision for them all
        made by declaration: their new ids, in order."""
        if not texts:
            raise Refused("nothing to add")
        with self._write("add") as (con, number):
            held_licenses = {one for (one,) in con.execute("SELECT id FROM licenses")}
            held_kinds = dict(con.execute("SELECT id, schema FROM kinds").fetchall())
            for one in texts:
                _check_text(one, held_licenses, held_kinds)
            decision = self._decision(con, number, UNDER_REVIEW, declaration)
            first = self._take(con, "texts", len(texts))
            notices, new_texts = {}, []
            for at, one in enumerate(texts):
                text_id = self._id(first + at)
                notice = None if one.notice is None else self._notice(con, number, one.notice, notices)
                self._store(con, number, [{"id": text_id, "version": 1, "kind": one.kind,
                                           "source": one.text if one.kind is None else one.content}])
                new_texts.append({"id": text_id, "revision": 1, "change": number, "latest": 1,
                                  "licenses": _json(list(one.licenses)), "notice": notice, "title": one.title,
                                  "authors": None if one.authors is None else _json(list(one.authors)),
                                  "tags": _json(list(one.tags)), "address": one.address,
                                  "decisions": _json([decision]), "content_version": 1, "scores": None})
            _insert(con, "texts_revisions", new_texts)
            return [one["id"] for one in new_texts]

    def blocked(self, texts: Selection, status: str = ADMITTED) -> list[dict]:
        """Of the texts a selection names, each one a decision of that status
        cannot cover, in the numeric order of their ids: {id, reasons: the
        reasons that hold, BELONGS_TO_UPSTREAM (a row the upstream owns,
        for any decision) and NO_ALLOWED_LICENSE (none of its licenses is
        allowed, wpt_depot.licenses, for an admission)}. The decision refuses
        any text listed here. NotFound when a given id is no text."""
        return self._blocked(self.selected(texts), status)

    def decide(self, status: str, texts: Selection, declaration: Declaration,
               reason: str | None = None) -> tuple[str, list[str]]:
        """One decision, admitted or refused, with an optional public reason,
        appended to every text selected as a new revision: (its id, the
        texts' ids). A refusal takes them out of every open library; a locked
        library keeps reading the revision it pinned. NotFound when nothing
        is selected or a given id is no text; Refused when admitting texts
        none of whose licenses is allowed (wpt_depot.licenses)."""
        if status not in (ADMITTED, REFUSED):
            raise Refused(f"a decision on texts admits or refuses them, not {status!r}")
        with self._write("decide") as (con, number):
            matched = self.selected(texts)
            if not matched:
                raise NotFound(f"no text matches {texts}" if isinstance(texts, str) else "no text is named")
            self._refuse_upstream("texts", matched)
            unlicensed = [one["id"] for one in self._blocked(matched, status) if NO_ALLOWED_LICENSE in one["reasons"]]
            if status == ADMITTED and unlicensed:
                raise Refused(f"{len(unlicensed)} of the texts have no allowed license, so none is admitted (the "
                              f"first: {unlicensed[0]}; docs/licenses.md, \"What the corpus accepts\")")
            decision = self._decision(con, number, status, declaration, reason)
            self._revise(con, "texts_revisions", matched, number, {"decisions": _APPEND}, {"decision": decision})
            if status == REFUSED:
                self._leave(con, matched, self._open_libraries(), number)
            return decision, matched

    @_answers
    def relicense(self, texts: Selection, new: Sequence[str],
                  declaration: Declaration) -> tuple[list[tuple[str, list[str]]], list[str]]:
        """Change the licenses of every text selected to the list new, as a
        new revision, their ids kept. Each distinct old list gets one
        decision 'under review' made by declaration, whose reason the depot
        writes (wpt_depot.policy.license_change), appended to the texts that
        had that list; the decisions before it no longer count, so each text
        is under review again and leaves every open library (a locked library
        keeps reading the revision it pinned). A text that names the licenses
        of new already is left out, so a change never only wipes its
        decisions. ([(each decision's id, the ids of its texts)], the ids
        left out), both in the order of the texts' ids; nothing is written
        when every text is left out. NotFound when nothing is selected or a
        given id is no text; Refused when new is empty or names a license the
        depot does not hold."""
        new = list(dict.fromkeys(new))
        if not new:
            raise Refused("a text names at least one license")
        with self._write("relicense") as (con, number):
            matched = self.selected(texts)
            if not matched:
                raise NotFound(f"no text matches {texts}" if isinstance(texts, str) else "no text is named")
            self._refuse_upstream("texts", matched)
            held_licenses = {one for (one,) in con.execute("SELECT id FROM licenses")}
            unknown = [one for one in new if one not in held_licenses]
            if unknown:
                raise Refused(f"the depot holds no license {', '.join(unknown)}")
            groups: dict[tuple[str, ...], list[str]] = {}
            for one in sorted(self._rows("SELECT id, licenses FROM texts_revisions WHERE latest "
                                         "AND id IN (SELECT value FROM json_each(?))", (_json(matched),)),
                              key=lambda one: ids.order_key(one["id"])):
                groups.setdefault(tuple(one["licenses"]), []).append(one["id"])
            skipped = sorted((text for old in groups if set(old) == set(new) for text in groups[old]),
                             key=matched.index)
            groups = {old: texts_of for old, texts_of in groups.items() if set(old) != set(new)}
            if not groups:
                raise _Nothing(([], skipped))
            changed = []
            for old, texts_of in groups.items():
                decision = self._decision(con, number, UNDER_REVIEW, declaration, policy.license_change(old, new),
                                          checked=False)
                self._revise(con, "texts_revisions", texts_of, number,
                             {"licenses": ":licenses", "decisions": _APPEND},
                             {"licenses": _json(new), "decision": decision})
                changed.append((decision, texts_of))
            self._leave(con, [text for _, texts_of in changed for text in texts_of], self._open_libraries(), number)
            return changed, skipped

    @_answers
    def change_content(self, text: str, plain: Source | None, content: str | None,
                       declaration: Declaration) -> tuple[str, int] | None:
        """Change the content of a text: a plain text takes its new text as
        plain (a string, or a FileText read as a stream) and no content; an item takes its new content as JSON text as
        content and no plain, which must still match its kind's schema (its
        kind never changes). It is stored as a new content version next to
        the old ones, and a new revision of the text, its id kept, reads it.
        The change is recorded as a decision 'under review' made by
        declaration, whose reason the depot writes
        (wpt_depot.policy.content_change); the decisions before it no longer
        count, so the text is under review again and leaves every open
        library at that change (a locked library keeps reading the content
        version of the revision it pinned). (the decision's id, the new
        content version), or None, with nothing written, when the new
        content is the one the text has (for an item, the same JSON value).
        NotFound without such a text; BadContent for an item's content that
        does not match its kind; Refused when the field given does not
        match the text (a plain text with content or without plain, an item
        with plain or without content) or is not text, and when the new
        content holds a passage an erasure removed from this text (matched
        by its public hash)."""
        with self._write("content") as (con, number):
            found = self._rows(f"SELECT c.kind, t.content_version, c.parts FROM texts_revisions t {CONTENT_JOIN} "
                               "WHERE t.latest AND t.id = ?", (text,))
            if not found:
                raise NotFound(f"no text {text}")
            self._refuse_upstream("texts", [text])
            row = found[0]
            if row["kind"] is None:
                if not isinstance(plain, (str, FileText)) or content is not None:
                    raise Refused("a plain text's new content is its text, with no item content")
                if listed(plain).parts == [(one["hash"], one["length"]) for one in row["parts"]]:
                    raise _Nothing(None)
                strings = [plain if isinstance(plain, str) else plain.blocks()]
            else:
                if not isinstance(content, str) or plain is not None:
                    raise Refused("an item's new content is its content as JSON text, with no plain text")
                schema = self._one("SELECT schema FROM kinds WHERE id = ?", (row["kind"],))
                kinds.check_content(row["kind"], schema, content)
                if _same_json(content, self.content(text, row["content_version"])):
                    raise _Nothing(None)
                strings = erasure.strings_of(json.loads(content))
            if erasure.holds_erased(strings, self._erased_from(text)):
                raise Refused(f"the new content of {text} holds a passage an erasure removed from it")
            version = self._one("SELECT max(version) FROM content_lists WHERE id = ?", (text,)) + 1
            decision = self._decision(con, number, UNDER_REVIEW, declaration,
                                      policy.content_change(row["content_version"], version), checked=False)
            self._store(con, number, [{"id": text, "version": version, "kind": row["kind"],
                                       "source": plain if row["kind"] is None else content}])
            self._revise(con, "texts_revisions", [text], number,
                         {"content_version": ":version", "decisions": _APPEND},
                         {"version": version, "decision": decision})
            self._leave(con, [text], self._open_libraries(), number)
            return decision, version

    def edit_credits(self, text: str, changes: dict, declaration: Declaration) -> tuple[str, str]:
        """Change credits of a text as a new revision, its id kept: changes
        maps each credit changed (CREDITS) to its new value, None to clear it
        (tags: a list, cleared as an empty list; notice: the notice's text,
        stored once). The change is recorded as a decision 'under review'
        made by declaration, whose reason the depot writes naming each
        credit's old and new value ('title changed from A to B; tags changed
        from x, y to x, z', a notice by its id); the decisions before it keep
        counting, so the text keeps its status and its libraries, and a
        locked library keeps reading the revision it pinned. (its id, the
        reason). NotFound without such a text; Refused for an unknown credit,
        an empty value, or when nothing would change."""
        unknown = sorted(set(changes) - set(CREDITS))
        if unknown:
            raise Refused(f"a text's credits are {', '.join(CREDITS)}, not {', '.join(unknown)}")
        for name, value in changes.items():
            if name in ("authors", "tags"):
                if value is not None and (not isinstance(value, (list, tuple)) or any(
                        not isinstance(one, str) or not one.strip() for one in value)):
                    raise Refused(f"{name} are a list of words that are not empty")
            elif value is not None:
                _filled(**{name: value})
        with self._write("credits") as (con, number):
            found = self._rows("SELECT * FROM texts_revisions WHERE latest AND id = ?", (text,))
            if not found:
                raise NotFound(f"no text {text}")
            self._refuse_upstream("texts", [text])
            row, values = found[0], {}
            for name in CREDITS:
                if name not in changes:
                    continue
                value = changes[name]
                if name == "tags":
                    value = list(dict.fromkeys(value or []))
                elif name == "authors":
                    value = None if value is None else list(value)
                elif name == "notice" and value is not None:
                    value = self._notice(con, number, value, {})
                if value != row[name]:
                    values[name] = value
            if not values:
                raise Refused(f"the credits of {text} are those already: nothing changes")
            reason = "; ".join(f"{name} changed from {_credit(row[name])} to {_credit(value)}"
                               for name, value in values.items())
            decision = self._decision(con, number, UNDER_REVIEW, declaration, reason, checked=False)
            self._revise(con, "texts_revisions", [text], number,
                         {**{name: f":{name}" for name in values}, "decisions": _APPEND},
                         {**{name: _json(value) if name in JSON_COLUMNS else value for name, value in values.items()},
                          "decision": decision})
            return decision, reason

    def edit_scores(self, changes: str | Path | list[dict], declaration: Declaration) -> tuple[str, str, int, int]:
        """Change the scores of many texts at once, each as a new revision,
        its id kept. changes is a JSON-lines file, or the lines as objects
        (a web request's): one {id, scores, remove} per text, scores a list
        of {name, value, content_version} (a name it has already is
        replaced, value a finite number, content_version one of the text's
        content versions, the one the score was made from) and remove a list
        of the names of scores to take away (a name it does not have changes
        nothing); both may be absent. The change is recorded as one decision
        'under review' made by declaration, on every text it changes, whose
        reason the depot writes naming the scores set and removed
        (policy.scores_change); the decisions before it keep counting, so
        each text keeps its status and its libraries, and a locked library
        keeps reading the revision it pinned. A text whose scores would stay
        as they are is left out. The lines go into a temporary table a few
        at a time and every check is a query over it (wpt_depot.scoring), so
        the memory it takes stays the same whatever the number of texts.
        (the decision's id, its reason, how many texts changed, how many
        were left out). NotFound for a text the depot does not hold;
        BelongsToUpstream for a text its upstream owns; Refused for changes
        that are not JSON lines of that shape, a text named twice, a name
        both set and removed or set twice, an empty name, a value that is
        not a finite number, a content version the text does not have, or
        when no text would change."""
        with self._write("scores") as (con, number):
            try:
                scoring.open_changes(con, changes)
                total, theirs = scoring.check(con, self.upstream_sql(), NAMED)
                if total:
                    self._refuse_upstream("texts", theirs, total=total)
                changed, left_out, set_names, removed = scoring.score(con)
                if not changed:
                    raise Refused("the scores of every text named are those already: nothing changes")
                reason = policy.scores_change(set_names, removed)
                decision = self._decision(con, number, UNDER_REVIEW, declaration, reason, checked=False)
                con.execute("CREATE TEMP TABLE scored_ids (id TEXT PRIMARY KEY)")
                con.execute("INSERT INTO temp.scored_ids SELECT id FROM temp.scored")
                self._revise(con, "texts_revisions", "SELECT id FROM temp.scored_ids", number,
                             {"scores": "(SELECT s.scores FROM temp.scored s WHERE s.id = t.id)",
                              "decisions": _APPEND}, {"decision": decision})
            finally:
                scoring.close(con)
            return decision, reason, changed, left_out

    # Libraries

    def new_library(self, name: str) -> str:
        _filled(name=name)
        with self._write("library") as (con, number):
            library = self._id(self._take(con, "libraries"))
            _insert(con, "libraries_revisions", [{"id": library, "revision": 1, "change": number, "latest": 1,
                                                  "name": name, "forked_from": None, "problems": "[]",
                                                  "locked_at": None}])
            return library

    def rename_library(self, library: str, name: str):
        """Give the library, open or locked, a new name, as a new revision:
        its name is not what it holds, so a locked library takes it too.
        NotFound without such a library; Refused for an empty name."""
        _filled(name=name)
        with self._write("rename") as (con, number):
            self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            self._revise(con, "libraries_revisions", [library], number, {"name": ":name"}, {"name": name})

    @_answers
    def put(self, library: str, texts: Selection) -> tuple[list[str], list[str]]:
        """Put into an open library every admitted text selected: (the ids
        that joined it, the ids selected and left out because they are not
        admitted). Locked when the library is locked; NotFound when a given
        id is no text."""
        with self._write("put") as (con, number):
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Locked(f"the library {library} is locked: no text joins it")
            matched = self.selected(texts)
            if not matched:
                raise _Nothing(([], []))
            found = {one["id"]: one for one in self._rows(
                f"SELECT t.id, {policy.admitted_sql('t.decisions')} AS admitted, EXISTS (SELECT 1 FROM memberships m "
                "WHERE m.library = :library AND m.text = t.id AND m.left_change IS NULL) AS held "
                "FROM texts_revisions t WHERE t.latest AND t.id IN (SELECT value FROM json_each(:ids))",
                {"library": library, "ids": _json(matched)})}
            joining = [text for text in matched if found[text]["admitted"] and not found[text]["held"]]
            left_out = [text for text in matched if not found[text]["admitted"]]
            if not joining:
                raise _Nothing(([], left_out))
            self._join(con, joining, library, number)
            return joining, left_out

    @_answers
    def take_out(self, library: str, texts: Selection) -> tuple[list[str], list[str]]:
        """Take out of an open library every text selected that it holds,
        each staying as it is, admitted included: (the ids that left it, the
        ids selected that it did not hold). Locked when the library is
        locked; NotFound when a given id is no text."""
        with self._write("take out") as (con, number):
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Locked(f"the library {library} is locked: no text leaves it")
            matched = self.selected(texts)
            if not matched:
                raise _Nothing(([], []))
            held = {one for (one,) in con.execute(
                "SELECT text FROM memberships WHERE library = ? AND left_change IS NULL "
                "AND text IN (SELECT value FROM json_each(?))", (library, _json(matched)))}
            leaving = [text for text in matched if text in held]
            not_held = [text for text in matched if text not in held]
            if not leaving:
                raise _Nothing(([], not_held))
            self._leave(con, leaving, [library], number)
            return leaving, not_held

    def lock(self, library: str):
        """Lock an open library for good, as a new revision: it pins the
        current revision of each text it holds. Refused when it is locked
        already."""
        with self._write("lock") as (con, number):
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Refused(f"the library {library} is locked already")
            con.execute("INSERT INTO pins (library, text, revision, change) SELECT m.library, t.id, t.revision, ? "
                        "FROM memberships m JOIN texts_revisions t ON t.id = m.text AND t.latest "
                        "WHERE m.library = ? AND m.left_change IS NULL", (number, library))
            self._revise(con, "libraries_revisions", [library], number, {"locked_at": ":locked_at"},
                         {"locked_at": _now()})

    def fork(self, library: str, name: str, without: Selection | None = None) -> str:
        """A new open library holding the texts library holds that are
        admitted now (only an admitted text joins a library), less those
        selected by without when it is given: its id. It starts with no
        problem. NotFound when a given id is no text."""
        _filled(name=name)
        with self._write("library") as (con, number):
            self._library(library)
            held = sorted((one for (one,) in con.execute(
                f"SELECT t.id FROM memberships m JOIN texts_revisions t ON t.id = m.text AND t.latest "
                f"WHERE m.library = ? AND m.left_change IS NULL AND {policy.admitted_sql('t.decisions')}",
                (library,))), key=ids.order_key)
            if without is not None:
                left = set(self.selected(without))
                held = [text for text in held if text not in left]
            fork = self._id(self._take(con, "libraries"))
            _insert(con, "libraries_revisions", [{"id": fork, "revision": 1, "change": number, "latest": 1,
                                                  "name": name, "forked_from": library, "problems": "[]",
                                                  "locked_at": None}])
            self._join(con, held, fork, number)
            return fork

    def report(self, library: str, problem: str, texts: Selection | None,
               declaration: Declaration) -> tuple[str, list[str]]:
        """A problem with the library, locked or not, as a new revision of
        it: the problem in words, and the texts of the library selected (none
        without a selection: the problem concerns the library as a whole).
        The reporter's declaration is recorded on a decision under review,
        which the problem names. (its id, the texts' ids). NotFound without
        such a library, when a condition matches no text the library holds,
        or a given id is no text; Refused when a given id is a text the
        library does not hold. Membership and texts stay as they are."""
        _filled(problem=problem)
        with self._write("report") as (con, number):
            self._library(library)
            self._refuse_upstream("libraries", [library])
            where, texts = texts, []
            if where is not None:
                held = {one for (one,) in con.execute(
                    "SELECT text FROM memberships WHERE library = ? AND left_change IS NULL", (library,))}
                chosen = self.selected(where)
                texts = [text for text in chosen if text in held]
                if not isinstance(where, str) and len(texts) < len(chosen):
                    outside = [text for text in chosen if text not in held]
                    raise Refused(f"{len(outside)} of the texts named are not in the library {library} (the first: "
                                  f"{outside[0]})")
                if not texts:
                    raise NotFound(f"no text of the library {library} matches {where}" if isinstance(where, str)
                                   else "no text is named")
            decision = self._decision(con, number, UNDER_REVIEW, declaration)
            self._revise(con, "libraries_revisions", [library], number,
                         {"problems": "json_insert(coalesce(t.problems, '[]'), '$[#]', json(:problem))"},
                         {"problem": _json({"texts": texts, "problem": problem, "decision": decision})})
            return decision, texts

    # Erasures

    def ask_erasure(self, reason: str, needles: list, declaration: Declaration) -> str:
        """A request to erase, under review. Each needle is a tuple (text
        id, start, end), (text id, start, end, field) or (text id, start,
        end, field, version), a Needle among them: a passage of one
        plain text in characters, a passage of a string field of one item
        (field: a JSON Pointer into its content, as wpt_depot.erasure says),
        or the whole text or item (no start, no end, no field), in the
        content version it names, its text's current one when it names
        none (version None), which the request records in each of its
        needles. Every cut version of an item must
        still match its kind's schema, or the request is refused. Its id.
        Refused for a content version the text does not have."""
        _filled(reason=reason)
        if any(not 3 <= len(one) <= 5 for one in needles):
            raise Refused("a needle is (text id, start, end), with a field and a content version after them or not")
        asked = [tuple(one) + (None,) * (5 - len(one)) for one in needles]
        if not asked:
            raise Refused("an erasure names at least one needle")
        with self._write("ask erasure") as (con, number):
            named = sorted({one[0] for one in asked})
            current = dict(con.execute("SELECT id, content_version FROM texts_revisions WHERE latest "
                                       "AND id IN (SELECT value FROM json_each(?))", (_json(named),)).fetchall())
            for one in asked:
                if one[0] not in current:
                    raise NotFound(f"no text {one[0]}")
            self._refuse_upstream("texts", named)
            kinds_of = self._kinds(named)
            needles = [Needle(text, start, end, field, current[text] if version is None else version)
                       for text, start, end, field, version in asked]
            documents = self._versions(named)
            for one in needles:
                if one.version not in documents.get(one.text, {}):
                    raise Refused(f"the text {one.text} has no content version {one.version}")
                document, item = documents[one.text][one.version], kinds_of[one.text] is not None
                erasure.check_needle({"kind": kinds_of[one.text], "text": None if item else document,
                                      "content": str(document) if item else None}, one)
            self._erased(_request(reason, needles, "dry run"))  # a cut that breaks an item is refused now
            decision = self._decision(con, number, UNDER_REVIEW, declaration)
            request = self._id(self._take(con, "erasures"))
            _insert(con, "erasures_revisions", [_erasure_row(_request(reason, needles, request, [decision]), number)])
            return request

    def decide_erasure(self, request: str, status: str, declaration: Declaration,
                       reason: str | None = None) -> bool:
        """A decision on an erasure request, with an optional public reason;
        once the request is admitted, it is carried out at once. Whether it
        was. Refused when it was carried out already."""
        return self.decide_erasures([request], status, declaration, reason)[request]

    def decide_erasures(self, requests: list[str], status: str, declaration: Declaration,
                        reason: str | None = None) -> dict[str, bool]:
        """The same decision on several erasure requests, in order, each
        recorded as a decision of its own with the same declaration and
        reason, and as a new revision of the request; each request admitted
        is carried out at once. Every request is checked before any is
        decided: NotFound for one that does not exist, Refused for one
        carried out already, and BadContent for one whose cut would break an
        item (each checked against the texts as they are now). The whole is
        one change. Whether each was carried out, by id."""
        if status not in (ADMITTED, REFUSED):
            raise Refused(f"a decision on an erasure admits or refuses it, not {status!r}")
        requests = list(dict.fromkeys(requests))
        if not requests:
            raise Refused("no erasure request is named")
        with self._write("decide erasure") as (con, number):
            found = {one["id"]: one for one in self._rows(
                "SELECT * FROM erasures WHERE id IN (SELECT value FROM json_each(?))", (_json(requests),))}
            carry_out = {}
            for request in requests:
                if request not in found:
                    raise NotFound(f"no erasure {request}")
            self._refuse_upstream("erasure requests", requests)
            self._refuse_upstream("texts", [needle["text"] for request in requests
                                            for needle in found[request]["needles"]])
            for request in requests:
                if found[request]["erased"] is not None:
                    raise Refused(f"the erasure {request} was carried out already")
                decided = {one["id"]: one for one in self._rows(
                    "SELECT id, status, reason FROM decisions WHERE id IN (SELECT value FROM json_each(?))",
                    (_json(found[request]["decisions"]),))}
                carry_out[request] = policy.admitted([*(decided[one] for one in found[request]["decisions"]),
                                                      {"status": status, "reason": reason}])
                if carry_out[request]:
                    self._erased(found[request])  # a cut that breaks an item is refused before any decision
            _check_declaration(declaration, reason)
            for request in requests:
                decision = self._decision(con, number, status, declaration, reason)
                erased = self._erased(found[request], (con, number)) if carry_out[request] else None
                self._revise(con, "erasures_revisions", [request], number,
                             {"decisions": _APPEND, **({} if erased is None else {"erased": ":erased"})},
                             {"decision": decision, "erased": _json(erased)})
        if any(carry_out.values()):
            self.db.checkpoint()
        return carry_out

    def erasures(self) -> list[dict]:
        """Every erasure request, in order, as it is now, with its status (wpt_depot.policy)."""
        return self._rows(f"SELECT e.*, {policy.status_sql('e.decisions')} AS status FROM erasures e ORDER BY e.id")

    def erasure_shown(self, request: str) -> dict:
        """What the erasure request asks to erase, without sending a whole
        text: {id, reason, status, carried_out, needles}, each needle with its
        text, field, start and end, its text's title and kind (None for a text
        the depot no longer holds), and what wpt_depot.erasure.shown gives
        for it. NotFound without such a request."""
        found = self._rows(f"SELECT e.*, {policy.status_sql('e.decisions')} AS status FROM erasures e "
                           "WHERE e.id = ?", (request,))
        if not found:
            raise NotFound(f"no erasure {request}")
        row = found[0]
        needles = _needles(row)
        named = sorted({one.text for one in needles})
        held = {one["id"]: one for one in self._rows(
            "SELECT id, title, kind FROM texts WHERE id IN (SELECT value FROM json_each(?))", (_json(named),))}
        erased = row["erased"]
        places = erasure.shown(needles, self._versions(named),
                               {text for text, one in held.items() if one["kind"] is not None}, erased)
        return {"id": row["id"], "reason": row["reason"], "status": row["status"], "carried_out": erased is not None,
                "needles": [{"text": one.text, "field": one.field, "start": one.start, "end": one.end,
                             "version": one.version, "title": held[one.text]["title"] if one.text in held else None,
                             "kind": held[one.text]["kind"] if one.text in held else None, **place}
                            for one, place in zip(needles, places)]}

    def _erased(self, request: dict, write: tuple | None = None) -> list[dict]:
        """What carrying out the erasure request erases: {length, hash_a,
        hash_b (hexadecimal), at, needle} of each passage. With write (the
        writer connection and the change number), it is carried out: every
        version it cuts lists its new parts, written as they are cut, every
        part of a text it cut that no version lists any longer goes, and
        the texts it names whole go. A plain text's version keeps its parts
        before the first one that loses a character, and the rest of it is
        cut again by the rule of wpt_depot.parts from there; its parts are
        read a few at a time. BadContent when a cut item no longer matches
        its kind's schema."""
        needles = _needles(request)
        found = self._kinds(sorted({one.text for one in needles}))
        kept = [at for at, one in enumerate(needles) if one.text in found]  # a text removed before has nothing left
        held = [needles[at] for at in kept]
        items = {text for text, kind in found.items() if kind is not None}
        schemas = dict(self.db.reader().execute("SELECT id, schema FROM kinds").fetchall())
        reader = self.reader()
        documents = self._versions(sorted(found))
        sizes = {(one["id"], one["version"]): one["bytes"] for one in self._rows(
            "SELECT id, version, bytes FROM content_lists WHERE id IN (SELECT value FROM json_each(?))",
            (_json(sorted(found)),))}
        cut, removed, passages = erasure.carried_out(documents, held, items)
        for text, versions in cut.items():
            known = self._held_parts(text) if write else set()
            for version, value in sorted(versions.items()):
                if text in items:
                    kinds.check_content(found[text], schemas[found[text]], value)
                    new = list(parted(value))
                    listed_parts = [(one.hash, one.length) for one in new]
                    length, size = sum(one.length for one in new), sum(one.bytes for one in new)
                else:
                    listed_parts, old, recut = _recut(reader, text, documents[text][version], value)
                    size, new = sizes[text, version], []
                    for one in recut:
                        listed_parts.append((one.hash, one.length))
                        size += one.bytes
                        if write:
                            new.append(one)
                    size -= old[0]
                    length = sum(part_length for _, part_length in listed_parts)
                if write:
                    con, number = write
                    for one in new:
                        if one.hash not in known:
                            known.add(one.hash)
                            con.execute("INSERT INTO parts (id, hash, change, value) VALUES (?, ?, ?, ?)",
                                        (text, one.hash, number, pack(one.value)))
                    con.execute("UPDATE content_lists SET parts = ?, length = ?, bytes = ? WHERE id = ? AND version = ?",
                                (_parts_json(listed_parts), length, size, text, version))
        # needle: the place among the request's needles of the first needle the passage covers; named before the
        # parts go, since a passage of a text removed whole is read from them
        named = [{**{name: value.hex() if isinstance(value, bytes) else value
                     for name, value in erasure.named(passage).items()},
                  "at": at, "needle": kept[merged.needles[0]]} for merged, passage, at in passages]
        if write:
            con, _ = write
            _drop_unlisted(con, sorted(cut))
            _remove_texts(con, removed)
        return named

    def _kinds(self, texts: list[str]) -> dict[str, str | None]:
        """{id: its kind, None for a plain text} of the texts named that the depot holds content of."""
        return dict(self.db.reader().execute(
            "SELECT DISTINCT id, kind FROM content_lists WHERE id IN (SELECT value FROM json_each(?))",
            (_json(texts),)).fetchall())

    def _held_parts(self, text: str) -> set[bytes]:
        """The hashes of the parts the depot holds of the text."""
        return {bytes(one) for (one,) in self.db.reader().execute("SELECT hash FROM parts WHERE id = ?", (text,))}

    def _versions(self, texts: list[str]) -> dict[str, dict[int, Parted | str]]:
        """Every content version of the texts named that the depot holds:
        {id: {version: its plain text, read lazily (parts.Parted), or an
        item's content as JSON text}}."""
        found, reader = {}, self.reader()
        for one in self._rows("SELECT id, version, kind, parts FROM content_lists "
                              "WHERE id IN (SELECT value FROM json_each(?)) ORDER BY version", (_json(texts),)):
            document = Parted(reader, one["id"], one["parts"])
            found.setdefault(one["id"], {})[one["version"]] = document if one["kind"] is None else str(document)
        return found

    def reader(self) -> Reader:
        """A reader of the depot's parts (wpt_depot.parts), through the connection of the thread that reads."""
        return Reader(self.db.reader)

    def content(self, text: str, version: int | None = None) -> str:
        """The whole content of one content version of a text (its current
        one by default), read from its parts: its plain text, or an item's
        JSON. NotFound without such a text or version."""
        return str(self.parted(text, version))

    def parted(self, text: str, version: int | None = None, reader: Reader | None = None) -> Parted:
        """One content version of a text (its current one for None), read
        lazily from its parts. NotFound without such a text or version."""
        if version is None:
            version = self._one("SELECT content_version FROM texts_revisions WHERE latest AND id = ?", (text,))
            if version is None:
                raise NotFound(f"no text {text}")
        found = self._rows("SELECT parts FROM content_lists WHERE id = ? AND version = ?", (text, version))
        if not found:
            raise NotFound(f"the text {text} has no content version {version}")
        return Parted(reader or self.reader(), text, found[0]["parts"])

    # Datasets

    def new_dataset(self, name: str, tokenizer: str, libraries: list[tuple[str | None, str]],
                    shares: list[tuple[str, float]] | None = None) -> str:
        """A dataset reading the libraries (depot address, None for this
        depot, and library id), each at its depot's change number now, with
        optional ordered shares (tag, share of the tokens): its id."""
        return self._new_dataset({"name": name, "tokenizer": tokenizer, "libraries": self._refs(libraries),
                                  "shares": None if shares is None else [{"tag": t, "share": s} for t, s in shares]})

    def import_dataset(self, remote: dict) -> str:
        """A dataset made from a remote (wpt_depot.remote): its id here."""
        from .remote import from_remote
        return self._new_dataset(from_remote(remote))

    def export_dataset(self, name: str) -> dict:
        """The dataset as a remote (wpt_depot.remote)."""
        from .remote import to_remote
        return to_remote(self.dataset(name))

    def dataset(self, name: str) -> dict:
        """The current row of the dataset named so. NotFound without one."""
        found = self._rows("SELECT * FROM datasets WHERE name = ?", (name,))
        if not found:
            raise NotFound(f"no dataset {name}")
        return found[0]

    def dataset_by_id(self, dataset: str) -> dict:
        """The current row of the dataset of that id. NotFound without one."""
        found = self._rows("SELECT * FROM datasets WHERE id = ?", (dataset,))
        if not found:
            raise NotFound(f"no dataset {dataset}")
        return found[0]

    def edit_dataset(self, dataset: str, name: str, libraries: list[tuple[str | None, str]],
                     shares: list[tuple[str, float]] | None = None):
        """Change the dataset of that id as a new revision, its id and
        tokenizer kept: its name, the libraries it reads (depot address, None
        for this depot, and library id) and its optional ordered shares (tag,
        share of the tokens), with the checks of new_dataset. A library it
        read already (the same library, its depot's address in any form, or
        either address of a library read here, reads_here) is kept exactly
        as stored, its address and change number included, its
        depot never opened, so a dataset made from a remote whose depots this
        machine cannot reach still changes; a library new to it is read at
        its depot's change number now. A corpus built before keeps its own
        copy of the dataset's row. NotFound without such a dataset."""
        old = self.dataset_by_id(dataset)
        self._refuse_upstream("datasets", [dataset])
        kept = {self._read_from(one["depot"], one["library"]): one for one in old["libraries"]}
        refs = []
        for address, library in libraries:
            stored = kept.get(self._read_from(address, library))
            refs.append(dict(stored) if stored is not None else self._refs([(address, library)])[0])
        values = {"name": name, "libraries": refs,
                  "shares": None if shares is None else [{"tag": t, "share": s} for t, s in shares]}
        _checked_dataset({**old, **values})
        with self._write("edit dataset") as (con, number):
            if self._rows("SELECT id FROM datasets WHERE name = ? AND id <> ?", (name, dataset)):
                raise Refused(f"the dataset name {name} is taken")
            self._revise(con, "datasets_revisions", [dataset], number,
                         {"name": ":name", "libraries": ":libraries", "shares": ":shares"},
                         {"name": name, "libraries": _json(refs), "shares": _json(values["shares"])})

    def _refs(self, libraries: list[tuple[str | None, str]]) -> list[dict]:
        """Each library (depot address, None for this depot, and library id)
        as a dataset names it, at a change number now of the depot that owns
        it. A library this depot holds and reads (reads_here: named by no
        address or this depot's, or a library of its upstream named by the
        upstream's web address once a pull brought it) is stored with no
        address, which means the depot that holds the row, so the row reads
        the same in a clone; its number is what this depot holds of its
        owner (bounds), so a library of the upstream is read by the
        upstream's numbers. A library at a web address (an upstream's library
        no pull brought yet, or another depot's) is asked of that depot's web
        app (wpt_depot.follow.library_change); one at another file, of that
        file. NotFound for a library its depot does not hold."""
        refs = []
        for address, library in libraries:
            place = self._place(address)
            if place is None or (self.reads_here(address, library) and self._holds_library(library)):
                self._library(library)
                refs.append({"depot": None, "library": library,
                             "change": self.bounds(self.name(), self.change())[ids.owner(library)]})
            elif isinstance(place, str) and place.startswith(("http://", "https://")):
                from .follow import library_change
                refs.append({"depot": place, "library": library, "change": library_change(place, library)})
            else:
                depot = Depot.view(address)
                depot._library(library)
                refs.append({"depot": depot.address, "library": library, "change": depot.change()})
        return refs

    def _holds_library(self, library: str) -> bool:
        return bool(self._rows("SELECT 1 FROM libraries WHERE id = ?", (library,)))

    def _read_from(self, address: str | None, library: str) -> tuple:
        """What a library a dataset names leads to, so two names of one
        library compare equal: the library's id alone when it is read here
        (reads_here), else where its address leads (_place) with its id."""
        return (None, library) if self.reads_here(address, library) else (self._place(address), library)

    def _place(self, address: str | None) -> Path | str | None:
        """Where a depot's address leads, None for this depot, so two forms
        of one address compare equal; a web address (an upstream's) or an
        address of no supported form stays as it is written, without a
        closing slash."""
        if address is None:
            return None
        try:
            found = location(address)
        except UnsupportedAddress:
            return _written(address)
        return None if found == self.db.path else found

    def is_here(self, address: str) -> bool:
        """Whether a depot's address leads to this depot, in any form
        (_place), opening nothing."""
        return self._place(address) is None

    def reads_here(self, address: str | None, library: str) -> bool:
        """Whether a library a dataset names (its depot's address, None for
        this depot, and its id) is one of this depot's rows: its address
        leads here (in any form, as _place compares them), or, in a depot
        that follows an upstream, it is the upstream's web address and the
        id is one the upstream owns, a library a pull brings here. A corpus
        reads such a library here."""
        place = self._place(address)
        if place is None:
            return True
        return (self.upstream() is not None and place == self._place(self.upstream())
                and bool(self.upstream_owned([library])))

    def _new_dataset(self, row: dict) -> str:
        _checked_dataset(row)
        with self._write("dataset") as (con, number):
            if self._rows("SELECT id FROM datasets WHERE name = ?", (row["name"],)):
                raise Refused(f"the dataset name {row['name']} is taken")
            dataset = self._id(self._take(con, "datasets"))
            _insert(con, "datasets_revisions", [{"id": dataset, "revision": 1, "change": number, "latest": 1,
                                                 "name": row["name"], "tokenizer": row["tokenizer"],
                                                 "libraries": _json(row["libraries"]),
                                                 "shares": _json(row["shares"])}])
            return dataset

    # Inside

    def _write(self, kind: str, past: int = 0):
        """One change of that kind (wpt_depot.database.Database.write)."""
        return self.db.write(kind, past)

    def _rows(self, sql: str, params=None) -> list[dict]:
        """The rows of the depot's own query, each list decoded (_decoded)."""
        return [_decoded(one) for one in self.db.query(sql, params)]

    def _one(self, sql: str, params=None):
        """The first value of the first row of a query, None without a row."""
        found = self.db.reader().execute(sql, params).fetchall()
        return found[0][0] if found else None

    def _decision(self, con, number: int, status: str, declaration: Declaration, reason: str | None = None,
                  checked: bool = True) -> str:
        """A person's decision, numbered next, written: its id. Refused for
        a reason in a form only the depot writes (wpt_depot.policy), unless
        the depot writes it (checked False)."""
        if checked:
            _check_declaration(declaration, reason)
        elif declaration.accepted is not True or not all(
                isinstance(one, str) and one.strip() for one in (declaration.name, declaration.email)):
            _check_declaration(declaration, None)
        decision = self._id(self._take(con, "decisions"))
        _insert(con, "decisions", [{"id": decision, "change": number, "status": status,
                                    "name": declaration.name.strip(), "email": declaration.email.strip(),
                                    "accepted": 1, "made_at": _now(),
                                    "reason": None if reason is None else reason.strip()}])
        return decision

    def _notice(self, con, number: int, text: str, made: dict) -> str:
        """The id of the notice whose text is text, stored once: a new one
        when the depot holds none (made: those this change made)."""
        if text not in made:
            found = con.execute("SELECT id FROM notices WHERE text = ?", (text,)).fetchall()
            if found:
                made[text] = found[0][0]
            else:
                made[text] = self._id(self._take(con, "notices"))
                _insert(con, "notices", [{"id": made[text], "change": number, "text": text}])
        return made[text]

    def _id(self, number: int) -> str:
        return ids.make(self.name(), number)

    @staticmethod
    def _take(con, name: str, count: int = 1) -> int:
        """The first of count numbers of the depot's own ids of that kind,
        taken: a number is never given twice (schema, numbers)."""
        last = (con.execute("SELECT last FROM numbers WHERE name = ?", (name,)).fetchall() or [(0,)])[0][0]
        con.execute("INSERT INTO numbers (name, last) VALUES (?, ?) ON CONFLICT (name) DO UPDATE SET last = excluded.last",
                    (name, last + count))
        return last + 1

    def _library(self, library: str) -> dict:
        found = self._rows("SELECT * FROM libraries WHERE id = ?", (library,))
        if not found:
            raise NotFound(f"no library {library}")
        return found[0]

    def _open_libraries(self) -> list[str]:
        return [one for (one,) in self.db.reader().execute("SELECT id FROM libraries WHERE locked_at IS NULL")]

    def _erased_from(self, text: str) -> list[dict]:
        """The erased entries of every carried-out request whose passage was in text."""
        found = self._rows("SELECT needles, erased FROM erasures WHERE erased IS NOT NULL")
        return [entry for row in found for entry in row["erased"] if row["needles"][entry["needle"]]["text"] == text]

    def _blocked(self, texts: list[str], status: str) -> list[dict]:
        """See blocked, over ids known to be texts."""
        found = sorted(self._rows("SELECT id, licenses FROM texts_revisions WHERE latest "
                                  "AND id IN (SELECT value FROM json_each(?))", (_json(texts),)),
                       key=lambda one: ids.order_key(one["id"]))
        theirs = set(self.upstream_owned(texts))
        reasons = [(one["id"], [reason for reason, holds in (
            (BELONGS_TO_UPSTREAM, one["id"] in theirs),
            (NO_ALLOWED_LICENSE, status == ADMITTED and not licenses.any_allowed(one["licenses"]))) if holds])
            for one in found]
        return [{"id": text, "reasons": held} for text, held in reasons if held]

    def _column_names(self, table: str) -> list[str]:
        if table not in self._names:
            self._names[table] = [one[1] for one in self.db.reader().execute(f"PRAGMA table_info({table})")]
        return self._names[table]

    def _revise(self, con, table: str, chosen: list[str] | str, number: int, sets: dict[str, str],
                params: dict | None = None):
        """A new revision, made at the change number, of each id chosen (a
        list, or a query of ids) of a revised table: its current row with
        each column of sets given by its SQL expression over the current row
        (t) and params, the revision before it no longer latest."""
        con.execute("CREATE TEMP TABLE IF NOT EXISTS revised (id TEXT PRIMARY KEY, revision INTEGER)")
        con.execute("DELETE FROM temp.revised")
        if isinstance(chosen, str):
            con.execute(f"INSERT INTO temp.revised SELECT r.id, r.revision FROM {table} r WHERE r.latest "
                        f"AND r.id IN ({chosen})")
        else:
            con.execute(f"INSERT INTO temp.revised SELECT r.id, r.revision FROM {table} r WHERE r.latest "
                        "AND r.id IN (SELECT value FROM json_each(?))", (_json(list(chosen)),))
        con.execute(f"UPDATE {table} SET latest = 0 WHERE latest AND id IN (SELECT id FROM temp.revised)")
        given = {"revision": "t.revision + 1", "change": ":change", "latest": "1", **sets}
        names = self._column_names(table)
        con.execute(f"INSERT INTO {table} ({', '.join(names)}) SELECT "
                    f"{', '.join(given.get(name, f't.{name}') for name in names)} "
                    f"FROM temp.revised c JOIN {table} t ON t.id = c.id AND t.revision = c.revision",
                    {"change": number, **(params or {})})
        con.execute("DELETE FROM temp.revised")

    @staticmethod
    def _join(con, texts: list[str], library: str, number: int):
        """The texts join the library at the change number; a text in it already stays as it is."""
        con.execute("INSERT INTO memberships (library, text, joined_change, left_change) "
                    "SELECT :library, j.value, :change, NULL FROM json_each(:texts) j WHERE NOT EXISTS "
                    "(SELECT 1 FROM memberships m WHERE m.library = :library AND m.text = j.value "
                    "AND m.left_change IS NULL)", {"library": library, "change": number, "texts": _json(texts)})

    @staticmethod
    def _leave(con, texts: list[str], libraries: list[str], number: int):
        """The texts leave the libraries at the change number."""
        if texts and libraries:
            con.execute("UPDATE memberships SET left_change = ? WHERE left_change IS NULL "
                        "AND library IN (SELECT value FROM json_each(?)) AND text IN (SELECT value FROM json_each(?))",
                        (number, _json(libraries), _json(texts)))

    def _store(self, con, number: int, contents: Sequence[dict]):
        """Write the content versions a change numbered number adds
        (contents: {id, version, kind, source: its plain text or an item's
        JSON, wpt_depot.parts.Source}): the parts of each that its text does
        not hold yet, read and written a few at a time, then its row of
        content_lists."""
        for one in contents:
            own = self._held_parts(one["id"]) if one["version"] > 1 else set()
            found = []
            for part in parted(one["source"]):
                found.append((part.hash, part.length, part.bytes))
                if part.hash not in own:
                    own.add(part.hash)
                    con.execute("INSERT INTO parts (id, hash, change, value) VALUES (?, ?, ?, ?)",
                                (one["id"], part.hash, number, pack(part.value)))
            _insert(con, "content_lists", [{"id": one["id"], "version": one["version"], "change": number,
                                            "kind": one["kind"],
                                            "parts": _parts_json([(hash_, length) for hash_, length, _ in found]),
                                            "length": sum(length for _, length, _ in found),
                                            "bytes": sum(size for _, _, size in found)}])


# how many texts the current revisions of which carry the decision d: a decision is appended to a text by the
# change that writes it, and a text's list of decisions only grows, so a text carries it now when it has a
# revision made at that change that carries it
_DECIDED_TEXTS = ("(SELECT count(*) FROM texts_revisions t WHERE t.change = d.change "
                  "AND EXISTS (SELECT 1 FROM json_each(t.decisions) WHERE value = d.id))")
# a new revision's decisions: the current ones and the change's decision
_APPEND = "json_insert(t.decisions, '$[#]', :decision)"


def location(address: str) -> Path:
    """The SQLite file of a depot's address: a path, or a file:// address.
    UnsupportedAddress for any other (a bucket, a web address)."""
    parsed = urlparse(address)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme and len(parsed.scheme) > 1:  # one letter is a Windows drive
        raise UnsupportedAddress(f"{address}: a depot is one SQLite file, a path or a file:// address")
    return Path(address).resolve()


def format_of(address: str) -> int:
    """The format of the depot at address: its about row's for an SQLite
    file, or that of a folder of Delta Lake tables (format 3 or older,
    wpt_depot.upgrade). NotADepot when it holds none."""
    path = location(address)
    if path.is_dir():
        from .legacy import Legacy
        return Legacy(path).format()
    if not path.is_file():
        raise NotADepot(f"{path} holds no depot")
    try:
        con = apsw.Connection(f"file:{quote(path.as_posix())}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    except apsw.Error as error:
        raise NotADepot(f"{path} is not a depot: {error}") from None
    try:
        return con.execute("SELECT format FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]
    except (apsw.Error, IndexError) as error:
        raise NotADepot(f"{path} is not a depot: {error or 'its about table holds no row'}") from None
    finally:
        con.close()


def other_format(address: str, found: int) -> str:
    """Why a depot of the format found is not read: what OtherFormat says."""
    if found < FORMAT:
        return (f"{address} is a depot of format {found}, older than the format {FORMAT} this wpt reads: "
                f"run `wpt depot upgrade` to bring it up to format {FORMAT}")
    return (f"{address} is a depot of format {found}, newer than the format {FORMAT} this wpt reads: "
            "update wpt to read it; a depot is never brought down to an older format")


def _written(address: str) -> str:
    """A web address as a dataset or an about row keeps it: without a closing slash."""
    return address.rstrip("/")


def _insert(con, table: str, found: list[dict]):
    """Add the rows (dicts of the table's columns) to the table."""
    if found:
        names = list(found[0])
        con.executemany(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join(':' + one for one in names)})",
                        found)


def _drop_unlisted(con, texts: list[str]):
    """Remove every part of the texts named that none of their content versions lists."""
    if texts:
        con.execute("DELETE FROM parts WHERE id IN (SELECT value FROM json_each(?)) AND NOT EXISTS "
                    "(SELECT 1 FROM content_lists c, json_each(c.parts) j WHERE c.id = parts.id "
                    "AND unhex(j.value ->> 'hash') = parts.hash)", (_json(texts),))


def _remove_texts(con, texts: list[str]):
    """Remove the texts whole: every revision, every content version and part, memberships and pins (an
    erasure's part)."""
    if texts:
        for table, column in (("texts_revisions", "id"), ("content_lists", "id"), ("parts", "id"),
                              ("memberships", "text"), ("pins", "text")):
            con.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT value FROM json_each(?))", (_json(texts),))


def _json(value) -> str | None:
    """value as the JSON text a list column holds; None stays None."""
    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parts_json(found: list[tuple[bytes, int]]) -> str:
    return _json([{"hash": hash_.hex(), "length": length} for hash_, length in found])


def _decoded(row: dict) -> dict:
    """A row with each list column (schema.JSON_COLUMNS) read from its JSON,
    a part's hash and an erased passage's hashes as bytes, and a decision's
    accepted as a boolean."""
    found = {}
    for name, value in row.items():
        if name in JSON_COLUMNS and isinstance(value, str):
            value = json.loads(value)
            if name == "parts":
                value = [{"hash": bytes.fromhex(one["hash"]), "length": one["length"]} for one in value]
            elif name == "erased":
                value = [{**one, "hash_a": bytes.fromhex(one["hash_a"]), "hash_b": bytes.fromhex(one["hash_b"])}
                         for one in value]
        elif name == "accepted" and isinstance(value, int):
            value = bool(value)
        found[name] = value
    return found


def _request(reason: str, needles: list, request_id: str, decisions: list | None = None) -> dict:
    return {"id": request_id, "revision": 1, "reason": reason, "decisions": decisions or [],
            "needles": [{"text": one.text, "field": one.field, "start": one.start, "end": one.end,
                         "version": one.version} for one in needles], "erased": None}


def _erasure_row(request: dict, number: int) -> dict:
    return {**request, "change": number, "latest": 1, "decisions": _json(request["decisions"]),
            "needles": _json(request["needles"]), "erased": None}


def _recut(reader: Reader, text: str, document: Parted, pieces: list[tuple[int, int]]):
    """(the parts [(hash, length)] a plain text's version keeps, before the
    first one that loses a character; [the bytes of the parts from that one
    on], filled as they are read; the parts of the rest of the cut text, a
    parts.Part each, as the rule of wpt_depot.parts cuts the whole cut text
    from there): document less its characters that pieces (the places
    [(start, end)] of those that stay, in order, as wpt_depot.erasure gives
    them) do not keep. The parts from that one on are read a few at a time
    as the last value is iterated."""
    kept = [[] for _ in document.parts]  # the pieces of each part, in its own places
    for a, b in pieces:
        first = bisect.bisect_right(document.starts, a) - 1
        for index in range(max(first, 0), len(document.parts)):
            start, end = document.starts[index], document.starts[index + 1]
            if start >= b:
                break
            if end > a and min(b, end) > max(a, start):
                kept[index].append((max(a, start) - start, min(b, end) - start))
    first = next((index for index, (_, length) in enumerate(document.parts) if kept[index] != [(0, length)]),
                 len(document.parts))
    old = [0]

    def rest():
        for index, value in enumerate(reader.stream(text, document.parts[first:]), first):
            old[0] += len(value.encode("utf-8"))
            yield "".join(value[a:b] for a, b in kept[index]).encode("utf-8")

    def recut():
        before, line = _before(reader, text, document, first)
        for data in split(rest(), before=before, line=line):
            yield made(data)
    return list(document.parts[:first]), old, recut()


def _before(reader: Reader, text: str, document: Parted, first: int) -> tuple[bytes, int]:
    """(the bytes just before the part first of document, at most a cut's
    window, and the place of the last line end before it, counted from it):
    what wpt_depot.parts.split cuts on with from that part."""
    before, line, walked = b"", None, 0
    for index in range(first - 1, -1, -1):
        data = reader.values(text, [document.parts[index][0]])[document.parts[index][0]].encode("utf-8")
        if line is None and (end := data.rfind(b"\n")) >= 0:
            line = -(walked + len(data) - end)
        walked += len(data)
        before = (data + before)[-(WINDOW - 1):]
        if line is not None and len(before) >= WINDOW - 1:
            break
    return before, -1 - walked if line is None else line


def _needles(request: dict) -> list[Needle]:
    """The needles of a row of erasures."""
    return [Needle(one["text"], one["start"], one["end"], one["field"], one["version"]) for one in request["needles"]]


def _same_json(one: str, other: str) -> bool:
    """Whether two JSON texts hold the same value (an object's keys in any order)."""
    return json.dumps(json.loads(one), sort_keys=True) == json.dumps(json.loads(other), sort_keys=True)


def _check_text(one: NewText, held_licenses: set, held_kinds: dict):
    if one.kind is None:
        if one.text is None or one.content is not None:
            raise Refused("a plain text has a text and no content")
    else:
        if one.text is not None or one.content is None:
            raise Refused("an item has a kind and its content, and no text")
        if one.kind not in held_kinds:
            raise Refused(f"the depot holds no kind {one.kind}")
        kinds.check_content(one.kind, held_kinds[one.kind], one.content)
    if not one.licenses:
        raise Refused("a text names at least one license")
    unknown = sorted(set(one.licenses) - held_licenses)
    if unknown:
        raise Refused(f"the depot holds no license {', '.join(unknown)}")


def _check_declaration(declaration: Declaration, reason: str | None):
    """Refused for a declaration without its confirmation or with an empty
    name or email, an empty reason, or a reason in a form only the depot
    writes (wpt_depot.policy.license_change, content_change)."""
    if isinstance(reason, str) and policy.starts_over(reason.strip()):
        start = next(one for one in policy.STARTS_OVER if reason.strip().startswith(one))
        what = "license change" if start == policy.LICENSE_CHANGE else "content change"
        raise Refused(f"a reason that starts with {start!r} is the depot's own record of a {what}; write the "
                      "reason another way")
    if declaration.accepted is not True:
        raise Refused("a decision needs the confirmation that the requirements were read, are accepted and met")
    _filled(name=declaration.name, email=declaration.email)
    if reason is not None:
        _filled(reason=reason)


def _checked_dataset(row: dict):
    _filled(name=row["name"], tokenizer=row["tokenizer"])
    if not row["libraries"]:
        raise Refused("a dataset reads at least one library")
    if row["shares"] is not None:
        tags = [one["tag"] for one in row["shares"]]
        if not tags or len(set(tags)) != len(tags) or not all(0 < one["share"] <= 1 for one in row["shares"]):
            raise Refused("shares are distinct tags, each with a share above 0 and at most 1")


def _credit(value) -> str:
    """A credit's value as a reason names it: a list's items joined by commas, 'none' for no value."""
    if isinstance(value, (list, tuple)):
        value = ", ".join(value)
    return value if value else "none"


def _filled(**values):
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise Refused(f"{name.replace('_', ' ')} is empty")


def _now() -> str:
    return datetime.now(UTC).isoformat()
