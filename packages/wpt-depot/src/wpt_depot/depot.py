"""A depot: texts and structured items, the decisions on them, libraries,
erasures and datasets, in the tables of one folder that one app writes
(wpt_depot.schema).

Depot is the only code that changes a depot's tables. Each operation
checks the whole change against the depot's rules first, so a refused
change writes nothing. Every change goes through the journal: written
first under the next change number (the content versions it adds right
after it, outside the journal), applied to each table, then marked done;
Depot.open applies again whatever a crash left undone, or cuts short a
change whose content versions were never written (wpt_depot.journal).
A new row (a text, a revision, a decision, a membership...) is appended,
reading none of the rows a table holds, so adding texts costs no more
for the text the depot holds already; only what changes in place (a membership
that ends, a content version an erasure cuts, a journal entry marked
done) is rewritten.

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
  1. An item's content matches its kind's schema, as the kinds table
  stores it. Every license a text names is a
  row of licenses (one of them is enough to allow it), and its notice is
  stored once.
- An operation on texts (a decision, a put, a fork's texts left out, a
  report) takes them as a selection: an SQL condition over the texts
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
  kept. The change is recorded as a decision 'under review' made by the
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
- Only admitted texts join a library. An open library reads the current
  revision of each text it holds, and can be read as it stood at any past
  change number of the depot that owns it (Depot.library_texts); a text
  of another depot it holds, in a clone, is read as the pulls up to that
  change brought it (Depot.bounds). Locking a library pins the revision
  of each text it holds: it reads those revisions for good, whatever
  revisions its texts take later, takes no new text and loses none but by
  an erasure. Its name is not what it holds, so it can be renamed, open or
  locked. A fork is a new open library holding the same texts that are
  admitted now, less those selected to leave out when it is given any.
- An erasure request is decided like a text, and several requests can be
  decided together, each checked before any is decided. Each needle names
  a place in one content version of its text, which the request records
  (its current one when the request is asked, unless it names one). Once
  admitted, the app carries it out at once (wpt_depot.erasure): every content version of the texts its needles
  name is cut in place in contents, the passage at the place named and
  every occurrence of it in each version, so the cut reaches every
  revision (an item is cut at a field of its JSON and every cut version is
  validated against its kind's schema again), or removed with every
  revision, content version, membership and pin; its
  `erased` gets each passage's length and hashes, the journal's rows of
  the texts removed go, and every older state of every table's storage is
  dropped. The hashes are public and keyless (wpt_depot.erasure),
  so any depot or clone can match an erased passage.
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
  Depot.upgrade, as a new revision of its about row; opening or reading a
  depot of another format than the code's is refused (OtherFormat).
- A depot that follows an upstream (a clone, wpt_depot.follow) only reads
  the rows its upstream owns: those whose id does not carry the depot's
  own name, that is, is not exactly the name, a hyphen and digits
  (wpt_depot.ids). Every operation that would change one is refused,
  writing nothing (BelongsToUpstream): a decision on, a license change of,
  a content change of or an edit of the credits of an upstream text, a request to erase one,
  a decision on an erasure request that is the upstream's or names an
  upstream text, renaming, locking, reporting
  on or putting texts into an upstream library, editing an upstream
  dataset. An operation on several texts is refused whole when one of
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

import contextlib
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import duckdb
import pyarrow as pa

from . import erasure, ids, kinds, licenses, policy
from .erasure import Needle
from .errors import (BadQuery, BelongsToUpstream, DepotExists, JournalUnfinished, Locked, NotADepot, NotFound,
                     OtherFormat, Refused, UnsupportedAddress)
from .filters import IN_LOCKED, literal, strings
from .journal import Journal
from .schema import (ADMITTED, CHANGES, CONTENT_JOIN, FORMAT, REFUSED, STEPS, TABLES, TEXT_COLUMNS, UNDER_REVIEW, latest,
                     rows, sql_name)
from .settings import bucket_settings
from .tables import Tables

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
    """A plain text (text) or a structured item (kind and its JSON content),
    with its credits. notice is the notice's text, stored once."""
    text: str | None = None
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
    """The depot in the folder `address` (a path or a file:// address),
    opened with open() or made with new()."""
    address: str
    settings: dict | None = None  # a bucket's settings (wpt_depot.settings), read from the environment when needed
    tables: Tables = field(init=False)
    journal: Journal = field(init=False)

    def __post_init__(self):
        where = location(self.address)
        if isinstance(where, str) and self.settings is None:
            self.settings = bucket_settings()
        self.tables = Tables(where, self.settings)
        self.address = str(self.tables.root)
        self.journal = Journal(self.tables)
        self._writing = threading.Lock()
        self._again = False  # applying a change again (Depot.open): rows it wrote already are not written twice
        self._unfinished = False  # a change of this object may be undone in the journal: see _writes
        self._name = None
        self._upstream = ...  # not read yet; None: it follows no upstream

    # Opening

    @classmethod
    def new(cls, address: str, about: About, settings: dict | None = None) -> "Depot":
        """An empty depot at address, with its about row (change 1) and the
        starting kinds (wpt_depot.kinds). DepotExists when the address holds
        tables; Refused when its name cannot begin an id (wpt_depot.ids)."""
        ids.check_name(about.name)
        depot = cls(address, settings)
        if depot.tables.exist():
            raise DepotExists(f"{depot.address} already holds a depot's tables")
        depot.tables.create()
        depot.tables.append("kinds", rows("kinds", kinds.starting_rows()))
        depot.tables.append("about", rows("about", [{"revision": 1, "change": 1, **vars(about), "format": FORMAT}]))
        return depot

    @classmethod
    def view(cls, address: str, settings: dict | None = None) -> "Depot":
        """The depot at address, to read: nothing is written, so a change
        its journal holds undone stays undone. For a depot another app
        owns, which only that app writes. NotADepot when a table is missing;
        OtherFormat when its format is not this code's (schema.FORMAT)."""
        depot = cls.find(address, settings)
        found = depot.format()
        if found != FORMAT:
            raise OtherFormat(depot._other_format(found))
        return depot

    @classmethod
    def open(cls, address: str, settings: dict | None = None) -> "Depot":
        """The depot at address, to write: every change the journal holds
        undone applied first, or cut short when its content versions were
        never written. NotADepot when a table is missing; OtherFormat when
        its format is not this code's (schema.FORMAT)."""
        depot = cls.view(address, settings)
        with depot._writing:
            depot._finish()
        return depot

    @classmethod
    def upgrade(cls, address: str, settings: dict | None = None) -> tuple["Depot", int]:
        """(the depot at address brought up to this code's format,
        schema.FORMAT; the format its about row gave): every change its
        journal holds undone finished (or cut short) first, with this code,
        then, from its format on, the step of schema.STEPS to the next
        format, each followed by the format it reached recorded as a new
        revision of its about row, a change of the journal like any other.
        An upgrade cut short resumes after the last format it recorded; a
        depot at FORMAT is left as opening it leaves it. NotADepot when a
        table is missing; OtherFormat for a newer format, which no code
        brings down."""
        depot = cls.find(address, settings)
        with depot._writing:
            start = depot.refuse_newer()
            depot._finish()  # an older format's undone changes, or the record of a format an upgrade cut short
            while (reached := depot.format()) < FORMAT:
                STEPS[reached](depot.tables)
                depot._change({"kind": "about", "values": {"format": reached + 1}})
        return depot, start

    @classmethod
    def find(cls, address: str, settings: dict | None = None) -> "Depot":
        """The depot at address, whatever its format: for what copies it as
        it is (wpt_depot.backup) or upgrades it, never to read or change its
        rows. NotADepot when a table is missing."""
        depot = cls(address, settings)
        missing = [name for name in TABLES if name not in depot.tables.exist()]
        if missing:
            raise NotADepot(f"{depot.address} is not a depot: no table {', '.join(missing)}")
        if not depot.tables.read("about", columns=["revision"]).num_rows:
            raise NotADepot(f"{depot.address} is not a depot: its about table holds no row")
        return depot

    def format(self) -> int:
        """The format of the depot's layout (wpt_depot.schema): its current
        about row's, or 0 when its about table has no format column."""
        if "format" not in self.tables.columns("about"):
            return 0
        return self.tables.query('SELECT "format" FROM about')["format"][0].as_py()

    def refuse_newer(self) -> int:
        """The depot's format. OtherFormat when it is newer than this code's
        (schema.FORMAT): no code brings a depot down to an older format."""
        found = self.format()
        if found > FORMAT:
            raise OtherFormat(self._other_format(found))
        return found

    def _other_format(self, found: int) -> str:
        """Why a depot of the format found is not read: what OtherFormat says."""
        if found < FORMAT:
            return (f"{self.address} is a depot of format {found}, older than the format {FORMAT} this wpt reads: "
                    f"run `wpt depot upgrade` to bring it up to format {FORMAT}")
        return (f"{self.address} is a depot of format {found}, newer than the format {FORMAT} this wpt reads: "
                "update wpt to read it; a depot is never brought down to an older format")

    def _finish(self):
        """Apply every change the journal holds undone, or cut it short when
        its content versions were never written."""
        self._again = True
        try:
            for entry, change in self.journal.pending():
                if _content_ids(change) and not self._contents_written(entry, change):
                    self._cut_short(entry, change)
                    continue
                self._apply(change, entry)
                self._done(entry, change)
        finally:
            self._again = False

    @contextlib.contextmanager
    def _writes(self):
        """Hold the depot for a write. When a change this object wrote is
        undone in the journal (it failed), it is finished first: the ids and
        change numbers the write takes next must not be the undone change's
        too. JournalUnfinished when it cannot be finished."""
        with self._writing:
            if self._unfinished:
                try:
                    self._finish()
                except Exception as error:
                    raise JournalUnfinished("a change the journal holds undone cannot be finished; opening the "
                                            "depot again finishes it") from error
                self._unfinished = False
            yield

    def change(self) -> int:
        """The depot's change number: the number of its latest change, past
        every change number its rows record."""
        parts = ["SELECT max(id) AS n FROM journal"] + [
            f"SELECT max({column}) FROM {sql_name(table)}" for table, columns in CHANGES.items() for column in columns]
        return self.tables.query(f"SELECT coalesce(max(n), 0) AS n FROM ({' UNION ALL '.join(parts)})")["n"][0].as_py()

    def about(self) -> dict:
        return self.tables.query("SELECT * FROM about").to_pylist()[0]

    def name(self) -> str:
        """The depot's name, which begins its ids."""
        if self._name is None:
            self._name = self.about()["name"]
        return self._name

    def upstream(self) -> str | None:
        """The address of the depot this one follows, None when it follows none."""
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
        """The same rule as upstream_owned, as a DuckDB condition over a column of ids."""
        if self.upstream() is None:
            return "false"
        return f"NOT {ids.carries_sql(self.name(), column)}"

    def _refuse_upstream(self, what: str, row_ids: Sequence[str], instead: str = ""):
        """BelongsToUpstream when one of the rows named is the upstream's:
        what names their kind in the plural ('texts'), instead what else
        can be done, after asking the upstream."""
        theirs = self.upstream_owned(row_ids)
        if theirs:
            listed = ", ".join(theirs[:NAMED]) + (f" and {len(theirs) - NAMED} more" if len(theirs) > NAMED else "")
            subject = (f"the {SINGULAR[what]} {listed} belongs" if len(theirs) == 1
                       else f"{len(theirs)} of the {what} belong ({listed})")
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
        with self._writes():
            self._change({"kind": "about", "values": {name: value.strip() for name, value in values.items()}})

    # Reading

    def sql(self, query: str) -> pa.Table:
        """The rows of one read query (a SELECT statement) over the depot's
        tables and views, by their names (wpt_depot.schema). BadQuery for
        anything else."""
        _one_select(query)
        try:
            return self.tables.query(query)
        except duckdb.Error as error:
            raise BadQuery(str(error)) from None

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
        this clone owns it}. NotFound without such a text."""
        found = self.tables.query("SELECT * FROM texts WHERE id = ?", [text]).to_pylist()
        if not found:
            raise NotFound(f"no text {text}")
        row = found[0]
        decided = {one["id"]: one for one in self.tables.query(
            "SELECT * FROM decisions WHERE list_contains(?, id)", [row["decisions"]]).to_pylist()}
        held = self.tables.query(f"SELECT *, {self.upstream_sql()} AS upstream FROM libraries "
                                 "WHERE list_contains(?, id) ORDER BY id", [row["libraries"]]).to_pylist()
        notice = None if row["notice"] is None else self.tables.query(
            "SELECT text FROM notices WHERE id = ?", [row["notice"]])["text"][0].as_py()
        revisions = self.tables.query("SELECT count(*) AS n FROM texts_revisions WHERE id = ?", [text])["n"][0].as_py()
        versions = self.tables.query("SELECT count(*) AS n FROM contents WHERE id = ?", [text])["n"][0].as_py()
        pins = self.tables.query("SELECT p.library, l.name, p.revision FROM pins p JOIN libraries l ON l.id = p.library "
                                 "WHERE p.text = ? ORDER BY p.library", [text]).to_pylist()
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
        if order not in ORDERS:
            raise Refused(f"the texts are ordered by one of {', '.join(ORDERS)}, not {order!r}")
        direction = " DESC" if descending else ""
        by = ", ".join([f"{one}{direction}" for one in ORDERS[order]] + [ids.order_sql()])
        matched = (f"SELECT *, {policy.status_sql('decisions')} AS status, {IN_LOCKED} AS locked, "
                   f"length(coalesce(text, content)) AS length FROM texts WHERE ({where})")
        counts = self.sql(f"SELECT status, locked, count(*) AS n FROM ({matched}) GROUP BY ALL").to_pylist()
        statuses = {status: sum(one["n"] for one in counts if one["status"] == status)
                    for status in (UNDER_REVIEW, ADMITTED, REFUSED)}
        found = self.sql(f"SELECT id, kind, title, authors, tags, licenses, libraries, status, locked, length "
                         f"FROM ({matched}) ORDER BY {by} LIMIT {int(limit)} OFFSET {int(offset)}").to_pylist()
        named = {column: self.sql(f"SELECT one AS {name}, count(*) AS texts FROM (SELECT unnest({column}) AS one "
                                  f"FROM texts WHERE ({where})) GROUP BY one ORDER BY texts DESC, one").to_pylist()
                 for column, name in (("tags", "tag"), ("licenses", "license"))}
        answer = {"total": sum(statuses.values()), "statuses": statuses,
                  "locked": sum(one["n"] for one in counts if one["locked"]), **named, "rows": found}
        if every_id:
            answer["ids"] = self.matching(where)
        return answer

    def tags(self) -> list[dict]:
        """Every tag a text has, with how many texts have it, by name."""
        return self.tables.query("SELECT tag, count(*) AS texts FROM (SELECT unnest(tags) AS tag FROM texts) "
                                 "GROUP BY tag ORDER BY tag").to_pylist()

    def matching(self, where: str) -> list[str]:
        """The ids of the texts the SQL condition where matches, in the
        numeric order of their ids (wpt_depot.ids)."""
        query = f"SELECT id FROM texts WHERE ({where}) ORDER BY {ids.order_sql()}"
        return self.sql(query)["id"].to_pylist()

    def selected(self, texts: Selection) -> list[str]:
        """The ids of the texts a selection names, each once, in the numeric
        order of their ids: those an SQL condition matches now, or exactly
        the ids given. NotFound when a given id is no text of the depot."""
        if isinstance(texts, str):
            return self.matching(texts)
        given = sorted(set(texts))
        found = set(self.tables.query("SELECT id FROM texts WHERE list_contains(?, id)", [given])["id"].to_pylist())
        missing = [one for one in given if one not in found]
        if missing:
            raise NotFound(f"{len(missing)} of the texts named are no text of the depot (the first: {missing[0]})")
        return self.matching(f"list_contains({strings(given)}, id)") if given else []

    def latest_decisions(self, limit: int = 10) -> list[dict]:
        """The latest decisions, the newest first, each with how many texts
        and erasure requests it decides and the library whose reported
        problem it records (None for any other)."""
        return self.tables.query(
            "SELECT d.*, (SELECT count(*) FROM texts t WHERE list_contains(t.decisions, d.id)) AS texts, "
            "(SELECT count(*) FROM erasures e WHERE list_contains(e.decisions, d.id)) AS erasures, "
            "(SELECT min(l.id) FROM libraries l WHERE list_contains(list_transform(coalesce(l.problems, []), "
            "lambda p: p.decision), d.id)) AS library "
            f"FROM decisions d ORDER BY d.made_at DESC, d.id DESC LIMIT {int(limit)}").to_pylist()

    def decision(self, decision: str) -> dict:
        """{decision: its row, texts: how many texts it is on, erasures: the
        ids of the erasure requests it decides, problem: the reported
        problem it records, with its library ({library, texts, problem}),
        or None}. NotFound without such a decision."""
        found = self.tables.query("SELECT * FROM decisions WHERE id = ?", [decision]).to_pylist()
        if not found:
            raise NotFound(f"no decision {decision}")
        texts = self.tables.query("SELECT count(*) AS n FROM texts WHERE list_contains(decisions, ?)",
                                  [decision])["n"][0].as_py()
        erasures = self.tables.query("SELECT id FROM erasures WHERE list_contains(decisions, ?) ORDER BY id",
                                     [decision])["id"].to_pylist()
        problem = next(({"library": one["id"], "texts": problem["texts"], "problem": problem["problem"]}
                        for one in self.tables.query("SELECT id, problems FROM libraries ORDER BY id").to_pylist()
                        for problem in one["problems"] or [] if problem["decision"] == decision), None)
        return {"decision": found[0], "texts": texts, "erasures": erasures, "problem": problem}

    def bounds(self, depot: str, change: int) -> dict[str, int]:
        """What the depot called depot held at its change number change, as
        {the name of each depot whose rows it held: the highest change number
        of that depot's numbering it held}: its own rows up to change, and,
        when it follows an upstream, the upstream's rows up to the upstream
        change brought by its last pull at or before change (pulls, its
        clone included), and so on back through the upstream's own pulls."""
        pulls = self.tables.query("SELECT * FROM pulls").to_pylist()
        found = {}
        while depot not in found:
            found[depot] = change
            last = max((one for one in pulls if one["depot"] == depot and one["change"] <= change),
                       key=lambda one: one["change"], default=None)
            if last is None:
                break
            depot, change = last["upstream"], last["upstream_change"]
        return found

    def library_texts(self, library: str, change: int) -> pa.Table:
        """Every text the library held at the change number change of the
        depot that owns it (the one its id names), each at its revision
        then, with the content version that revision reads, as it is now (an
        erasure's cut reaches every version, and a text an erasure removed is
        in no past): for a library locked
        at that change, the revisions it pinned; for an open one, the texts
        that had joined and not left it, each at its latest revision of that
        state (bounds: a text of another depot, such as one a pull brought
        to a clone, at its latest revision numbered within what the pulls
        up to that change brought). The columns of the texts view but
        libraries, in the numeric order of ids. NotFound without such a
        library, or when it did not exist yet at that change."""
        self._library(library)
        state = self.tables.query("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                                  "ORDER BY revision DESC LIMIT 1", [library, change]).to_pylist()
        if not state:
            raise NotFound(f"the library {library} did not exist yet at the change {change}")
        if state[0]["locked_at"] is not None:
            held = "SELECT text, revision FROM pins WHERE library = $library"
        else:
            held = (f"SELECT m.text, (SELECT max(r.revision) FROM texts_revisions r JOIN bounds b ON "
                    f"b.depot = {ids.owner_sql('r.id')} WHERE r.id = m.text AND r.change <= b.change) AS revision "
                    "FROM memberships m WHERE m.library = $library "
                    "AND m.joined_change <= $change AND (m.left_change IS NULL OR m.left_change > $change)")
        con = self.tables.connection()
        try:
            found = self.bounds(ids.owner(library), change)
            con.register("bounds", pa.table({"depot": pa.array(list(found), pa.string()),
                                             "change": pa.array(list(found.values()), pa.int64())}))
            return con.sql(f"SELECT {TEXT_COLUMNS} FROM ({held}) h JOIN texts_revisions t ON t.id = h.text "
                           f"AND t.revision = h.revision {CONTENT_JOIN} "
                           f"ORDER BY {ids.order_sql('t.id')}",
                           params={"library": library, **({"change": change} if "$change" in held else {})}
                           ).to_arrow_table()
        finally:
            con.close()

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
                 "JOIN texts c ON c.id = p.text WHERE p.library = ?")
        counts = self.tables.query(f"SELECT count(*) AS total, count(*) FILTER (WHERE current > revision) AS newer "
                                   f"FROM ({query})", [library]).to_pylist()[0]
        found = self.tables.query(f"SELECT * FROM ({query}) ORDER BY {ids.order_sql()} LIMIT {int(limit)} "
                                  f"OFFSET {int(offset)}", [library]).to_pylist()
        return {**counts, "rows": found}

    # Licenses

    def licenses(self) -> list[dict]:
        """Every license the depot holds, by id: its id and name, whether it
        is allowed (wpt_depot.licenses, a combined row only if every part
        is), and how many texts name it."""
        found = self.tables.query("SELECT l.id, l.name, (SELECT count(*) FROM texts t WHERE "
                                  "list_contains(t.licenses, l.id)) AS texts FROM licenses l ORDER BY l.id").to_pylist()
        return [{"id": one["id"], "name": one["name"], "allowed": licenses.allowed(one["id"]), "texts": one["texts"]}
                for one in found]

    def license(self, license_id: str) -> dict:
        """One license: its id, name, official text, whether it is allowed
        (as licenses() says) and how many texts name it. NotFound without
        it."""
        found = self.tables.query("SELECT l.*, (SELECT count(*) FROM texts t WHERE list_contains(t.licenses, l.id)) "
                                  "AS texts FROM licenses l WHERE l.id = ?", [license_id]).to_pylist()
        if not found:
            raise NotFound(f"no license {license_id}")
        return {**found[0], "allowed": licenses.allowed(license_id)}

    def add_license(self, license_id: str, name: str, text: str):
        """An official license text, stored once. The id is an SPDX name, a
        license with its exception ('Apache-2.0 WITH LLVM-exception') or
        licenses that apply together ('MIT AND CC-BY-4.0'). Refused when the
        id is held already."""
        _filled(license_id=license_id, name=name, text=text)
        with self._writes():
            if self.tables.query("SELECT id FROM licenses WHERE id = ?", [license_id]).num_rows:
                raise Refused(f"the license {license_id} is held already")
            self._change({"kind": "license", "license": {"id": license_id, "name": name, "text": text}})

    # Texts and decisions

    def add(self, texts: list[NewText], declaration: Declaration) -> list[str]:
        """Add texts and items, under review, with one decision for them all
        made by declaration: their new ids, in order."""
        if not texts:
            raise Refused("nothing to add")
        with self._writes():
            decision = self._decision(UNDER_REVIEW, declaration)
            held_licenses = set(self.tables.query("SELECT id FROM licenses")["id"].to_pylist())
            held_kinds = {one["id"]: one["schema"] for one in self.tables.query("SELECT id, schema FROM kinds").to_pylist()}
            notices = {one["text"]: one["id"] for one in self.tables.query("SELECT id, text FROM notices").to_pylist()}
            next_notice = self._next("notices")
            first = self._next("texts")
            new_notices, new_texts, contents = [], [], []
            for number, one in enumerate(texts):
                _check_text(one, held_licenses, held_kinds)
                if one.notice is not None and one.notice not in notices:
                    notices[one.notice] = self._id(next_notice)
                    new_notices.append({"id": notices[one.notice], "text": one.notice})
                    next_notice += 1
                text_id = self._id(first + number)
                contents.append({"id": text_id, "version": 1, "text": one.text, "kind": one.kind,
                                 "content": one.content})
                new_texts.append({
                    "id": text_id, "revision": 1, "licenses": list(one.licenses),
                    "notice": None if one.notice is None else notices[one.notice], "title": one.title,
                    "authors": None if one.authors is None else list(one.authors), "tags": list(one.tags),
                    "address": one.address, "decisions": [decision["id"]], "content_version": 1})
            self._change({"kind": "add", "decision": decision, "notices": new_notices, "texts": new_texts},
                         contents=contents)
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
        with self._writes():
            matched = self.selected(texts)
            if not matched:
                raise NotFound(f"no text matches {texts}" if isinstance(texts, str) else "no text is named")
            self._refuse_upstream("texts", matched)
            unlicensed = [one["id"] for one in self._blocked(matched, status) if NO_ALLOWED_LICENSE in one["reasons"]]
            if status == ADMITTED and unlicensed:
                raise Refused(f"{len(unlicensed)} of the texts have no allowed license, so none is admitted (the "
                              f"first: {unlicensed[0]}; docs/licenses.md, \"What the corpus accepts\")")
            decision = self._decision(status, declaration, reason)
            self._change({"kind": "decide", "decision": decision, "texts": matched,
                          "leave": self._open_libraries() if status == REFUSED else []})
            return decision["id"], matched

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
        with self._writes():
            matched = self.selected(texts)
            if not matched:
                raise NotFound(f"no text matches {texts}" if isinstance(texts, str) else "no text is named")
            self._refuse_upstream("texts", matched)
            held_licenses = set(self.tables.query("SELECT id FROM licenses")["id"].to_pylist())
            unknown = [one for one in new if one not in held_licenses]
            if unknown:
                raise Refused(f"the depot holds no license {', '.join(unknown)}")
            groups: dict[tuple[str, ...], list[str]] = {}
            for one in self.tables.query(f"SELECT id, licenses FROM texts WHERE list_contains(?, id) "
                                         f"ORDER BY {ids.order_sql()}", [matched]).to_pylist():
                groups.setdefault(tuple(one["licenses"]), []).append(one["id"])
            skipped = sorted((text for old in groups if set(old) == set(new) for text in groups[old]),
                             key=matched.index)
            groups = {old: texts_of for old, texts_of in groups.items() if set(old) != set(new)}
            if not groups:
                return [], skipped
            first = self._next("decisions")
            decisions = [self._recorded(UNDER_REVIEW, declaration, policy.license_change(old, new), first + number)
                         for number, old in enumerate(groups)]
            self._change({"kind": "relicense", "decisions": decisions, "licenses": new,
                          "leave": self._open_libraries(),
                          "groups": [{"decision": decision["id"], "texts": texts_of}
                                     for decision, texts_of in zip(decisions, groups.values())]})
            return [(decision["id"], texts_of) for decision, texts_of in zip(decisions, groups.values())], skipped

    def change_content(self, text: str, plain: str | None, content: str | None,
                       declaration: Declaration) -> tuple[str, int] | None:
        """Change the content of a text: a plain text takes its new text as
        plain and no content; an item takes its new content as JSON text as
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
        with self._writes():
            found = self.tables.query("SELECT kind, text, content, content_version FROM texts WHERE id = ?",
                                      [text]).to_pylist()
            if not found:
                raise NotFound(f"no text {text}")
            self._refuse_upstream("texts", [text])
            row = found[0]
            if row["kind"] is None:
                if not isinstance(plain, str) or content is not None:
                    raise Refused("a plain text's new content is its text, with no item content")
                if plain == row["text"]:
                    return None
            else:
                if not isinstance(content, str) or plain is not None:
                    raise Refused("an item's new content is its content as JSON text, with no plain text")
                schema = self.tables.query("SELECT schema FROM kinds WHERE id = ?", [row["kind"]])["schema"][0].as_py()
                kinds.check_content(row["kind"], schema, content)
                if _same_json(content, row["content"]):
                    return None
            if erasure.holds_erased([plain] if row["kind"] is None else erasure.strings_of(json.loads(content)),
                                    self._erased_from(text)):
                raise Refused(f"the new content of {text} holds a passage an erasure removed from it")
            version = self.tables.query("SELECT max(version) AS n FROM contents WHERE id = ?",
                                        [text])["n"][0].as_py() + 1
            decision = self._recorded(UNDER_REVIEW, declaration,
                                      policy.content_change(row["content_version"], version), self._next("decisions"))
            self._change({"kind": "content", "decision": decision, "text": text, "version": version,
                          "leave": self._open_libraries()},
                         contents=[{"id": text, "version": version, "kind": row["kind"], "text": plain,
                                    "content": content}])
            return decision["id"], version

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
        with self._writes():
            found = self.tables.query("SELECT * FROM texts WHERE id = ?", [text]).to_pylist()
            if not found:
                raise NotFound(f"no text {text}")
            self._refuse_upstream("texts", [text])
            row = found[0]
            notices = {one["text"]: one["id"] for one in self.tables.query("SELECT id, text FROM notices").to_pylist()}
            new_notices, values = [], {}
            for name in CREDITS:
                if name not in changes:
                    continue
                value = changes[name]
                if name == "tags":
                    value = list(dict.fromkeys(value or []))
                elif name == "authors":
                    value = None if value is None else list(value)
                elif name == "notice" and value is not None:
                    if value not in notices:
                        notices[value] = self._id(self._next("notices"))
                        new_notices.append({"id": notices[value], "text": value})
                    value = notices[value]
                if value != row[name]:
                    values[name] = value
            if not values:
                raise Refused(f"the credits of {text} are those already: nothing changes")
            reason = "; ".join(f"{name} changed from {_credit(row[name])} to {_credit(value)}"
                               for name, value in values.items())
            decision = self._recorded(UNDER_REVIEW, declaration, reason, self._next("decisions"))
            self._change({"kind": "credits", "decision": decision, "text": text, "values": values,
                          "notices": new_notices})
            return decision["id"], reason

    # Libraries

    def new_library(self, name: str) -> str:
        _filled(name=name)
        with self._writes():
            library = self._id(self._next("libraries"))
            self._change({"kind": "library", "library": {"id": library, "revision": 1, "name": name,
                                                         "forked_from": None, "problems": [], "locked_at": None},
                          "texts": []})
            return library

    def rename_library(self, library: str, name: str):
        """Give the library, open or locked, a new name, as a new revision:
        its name is not what it holds, so a locked library takes it too.
        NotFound without such a library; Refused for an empty name."""
        _filled(name=name)
        with self._writes():
            self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            self._change({"kind": "rename", "library": library, "name": name})

    def put(self, library: str, texts: Selection) -> tuple[list[str], list[str]]:
        """Put into an open library every admitted text selected: (the ids
        that joined it, the ids selected and left out because they are not
        admitted). Locked when the library is locked; NotFound when a given
        id is no text."""
        with self._writes():
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Locked(f"the library {library} is locked: no text joins it")
            matched = self.selected(texts)
            if not matched:
                return [], []
            found = self.tables.query(
                f"SELECT id, {policy.admitted_sql('decisions')} AS admitted, list_contains(libraries, ?) AS held "
                f"FROM texts WHERE list_contains(?, id) ORDER BY {ids.order_sql()}", [library, matched]).to_pylist()
            joining = [one["id"] for one in found if one["admitted"] and not one["held"]]
            left_out = [one["id"] for one in found if not one["admitted"]]
            if joining:
                self._change({"kind": "put", "library": library, "texts": joining})
            return joining, left_out

    def lock(self, library: str):
        """Lock an open library for good, as a new revision: it pins the
        current revision of each text it holds. Refused when it is locked
        already."""
        with self._writes():
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Refused(f"the library {library} is locked already")
            self._change({"kind": "lock", "library": library, "locked_at": _now().isoformat()})

    def fork(self, library: str, name: str, without: Selection | None = None) -> str:
        """A new open library holding the texts library holds that are
        admitted now (only an admitted text joins a library), less those
        selected by without when it is given: its id. It starts with no
        problem. NotFound when a given id is no text."""
        _filled(name=name)
        with self._writes():
            self._library(library)
            held = self.matching(f"list_contains(libraries, {literal(library)}) "
                                 f"AND {policy.admitted_sql('decisions')}")
            if without is not None:
                left = set(self.selected(without))
                held = [text for text in held if text not in left]
            row = {"id": self._id(self._next("libraries")), "revision": 1, "name": name, "forked_from": library,
                   "problems": [], "locked_at": None}
            self._change({"kind": "library", "library": row, "texts": held})
            return row["id"]

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
        with self._writes():
            self._library(library)
            self._refuse_upstream("libraries", [library])
            where, texts = texts, []
            if where is not None:
                held = set(self.tables.query("SELECT id FROM texts WHERE list_contains(libraries, ?)",
                                             [library])["id"].to_pylist())
                chosen = self.selected(where)
                texts = [text for text in chosen if text in held]
                if not isinstance(where, str) and len(texts) < len(chosen):
                    outside = [text for text in chosen if text not in held]
                    raise Refused(f"{len(outside)} of the texts named are not in the library {library} (the first: "
                                  f"{outside[0]})")
                if not texts:
                    raise NotFound(f"no text of the library {library} matches {where}" if isinstance(where, str)
                                   else "no text is named")
            decision = self._decision(UNDER_REVIEW, declaration)
            self._change({"kind": "report", "decision": decision, "library": library,
                          "problem": {"texts": texts, "problem": problem, "decision": decision["id"]}})
            return decision["id"], texts

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
        with self._writes():
            named = sorted({one[0] for one in asked})
            current = {one["id"]: one["content_version"] for one in self.tables.query(
                "SELECT id, content_version FROM texts WHERE list_contains(?, id)", [named]).to_pylist()}
            for one in asked:
                if one[0] not in current:
                    raise NotFound(f"no text {one[0]}")
            self._refuse_upstream("texts", named)
            found = {(one["id"], one["version"]): one for one in self.tables.query(
                "SELECT id, version, text, kind, content FROM contents WHERE list_contains(?, id)", [named]).to_pylist()}
            needles = [Needle(text, start, end, field, current[text] if version is None else version)
                       for text, start, end, field, version in asked]
            for one in needles:
                if (one.text, one.version) not in found:
                    raise Refused(f"the text {one.text} has no content version {one.version}")
                erasure.check_needle(found[one.text, one.version], one)
            self._erased(self._request(reason, needles, "dry run"))  # a cut that breaks an item is refused now
            decision = self._decision(UNDER_REVIEW, declaration)
            row = self._request(reason, needles, self._id(self._next("erasures")), [decision["id"]])
            self._change({"kind": "ask erasure", "decision": decision, "erasure": row})
            return row["id"]

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
        item (each checked against the texts as they are now). Whether each
        was carried out, by id."""
        if status not in (ADMITTED, REFUSED):
            raise Refused(f"a decision on an erasure admits or refuses it, not {status!r}")
        requests = list(dict.fromkeys(requests))
        if not requests:
            raise Refused("no erasure request is named")
        with self._writes():
            found = {one["id"]: one for one in self.tables.query(
                "SELECT * FROM erasures WHERE list_contains(?, id)", [requests]).to_pylist()}
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
                decided = {one["id"]: one for one in self.tables.query(
                    "SELECT id, status, reason FROM decisions WHERE list_contains(?, id)",
                    [found[request]["decisions"]]).to_pylist()}
                carry_out[request] = policy.admitted([*(decided[one] for one in found[request]["decisions"]),
                                                      {"status": status, "reason": reason}])
                if carry_out[request]:
                    self._erased(found[request])  # a cut that breaks an item is refused before any decision
            self._decision(status, declaration, reason)  # the declaration is checked before any decision
            for request in requests:
                decision = self._decision(status, declaration, reason)
                if carry_out[request]:
                    self._change({"kind": "erase", "decision": decision, **self._erased(found[request])})
                else:
                    self._change({"kind": "decide erasure", "decision": decision, "erasure": request})
            return carry_out

    def erasures(self) -> pa.Table:
        """Every erasure request, in order, as it is now, with its status (wpt_depot.policy)."""
        return self.tables.query(f"SELECT e.*, {policy.status_sql('e.decisions')} AS status "
                                 "FROM erasures e ORDER BY e.id")

    def erasure_shown(self, request: str) -> dict:
        """What the erasure request asks to erase, without sending a whole
        text: {id, reason, status, carried_out, needles}, each needle with its
        text, field, start and end, its text's title and kind (None for a text
        the depot no longer holds), and what wpt_depot.erasure.shown gives
        for it. NotFound without such a request."""
        found = self.tables.query(f"SELECT e.*, {policy.status_sql('e.decisions')} AS status FROM erasures e "
                                  "WHERE e.id = ?", [request]).to_pylist()
        if not found:
            raise NotFound(f"no erasure {request}")
        row = found[0]
        needles = _needles(row)
        named = sorted({one.text for one in needles})
        held = {one["id"]: one for one in self.tables.query(
            "SELECT id, title, kind FROM texts WHERE list_contains(?, id)", [named]).to_pylist()}
        erased = row["erased"]
        places = erasure.shown(needles, self._versions(named),
                               {text for text, one in held.items() if one["kind"] is not None}, erased)
        return {"id": row["id"], "reason": row["reason"], "status": row["status"], "carried_out": erased is not None,
                "needles": [{"text": one.text, "field": one.field, "start": one.start, "end": one.end,
                             "version": one.version, "title": held[one.text]["title"] if one.text in held else None,
                             "kind": held[one.text]["kind"] if one.text in held else None, **place}
                            for one, place in zip(needles, places)]}

    @staticmethod
    def _request(reason: str, needles: list, request_id: str, decisions: list | None = None) -> dict:
        return {"id": request_id, "revision": 1, "reason": reason, "decisions": decisions or [],
                "needles": [{"text": one.text, "field": one.field, "start": one.start, "end": one.end,
                             "version": one.version} for one in needles], "erased": None}

    def _erased(self, request: dict) -> dict:
        """What carrying out the erasure request leaves: the erasure's id and
        the rest of an 'erase' change (the decision that admits it is added
        by the caller, so one journal entry holds both). BadContent when a
        cut item no longer matches its kind's schema."""
        needles = _needles(request)
        found = {one["id"]: one["kind"] for one in self.tables.query(
            "SELECT DISTINCT id, kind FROM contents WHERE list_contains(?, id)",
            [sorted({one.text for one in needles})]).to_pylist()}
        kept = [at for at, one in enumerate(needles) if one.text in found]  # a text removed before has nothing left
        held = [needles[at] for at in kept]
        items = {text for text, kind in found.items() if kind is not None}
        cut, removed, passages = erasure.carried_out(self._versions(sorted(found)), held, items)
        schemas = {one["id"]: one["schema"] for one in self.tables.query("SELECT id, schema FROM kinds").to_pylist()}
        for text, versions in cut.items():
            if text in items:
                for content in versions.values():
                    kinds.check_content(found[text], schemas[found[text]], content)
        # needle: the place among the request's needles of the first needle the passage covers
        erased = [{**{name: value.hex() if isinstance(value, bytes) else value
                      for name, value in erasure.named(passage).items()},
                   "at": at, "needle": kept[merged.needles[0]]} for merged, passage, at in passages]
        return {"erasure": request["id"], "removed": removed, "erased": erased,
                "cut": [{"id": text, "version": version, "content" if text in items else "text": value}
                        for text, versions in cut.items() for version, value in sorted(versions.items())]}

    def _versions(self, texts: list[str]) -> dict[str, dict[int, str]]:
        """Every content version of the texts named that the depot holds:
        {id: {version: its plain text, or an item's content as JSON text}}."""
        found = {}
        for one in self.tables.query("SELECT id, version, text, kind, content FROM contents WHERE list_contains(?, id) "
                                     "ORDER BY version", [texts]).to_pylist():
            found.setdefault(one["id"], {})[one["version"]] = one["text"] if one["kind"] is None else one["content"]
        return found

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
        found = self.tables.query("SELECT * FROM datasets WHERE name = ?", [name]).to_pylist()
        if not found:
            raise NotFound(f"no dataset {name}")
        return found[0]

    def dataset_by_id(self, dataset: str) -> dict:
        """The current row of the dataset of that id. NotFound without one."""
        found = self.tables.query("SELECT * FROM datasets WHERE id = ?", [dataset]).to_pylist()
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
        self._checked_dataset({**old, **values})
        with self._writes():
            if self.tables.query("SELECT id FROM datasets WHERE name = ? AND id <> ?", [name, dataset]).num_rows:
                raise Refused(f"the dataset name {name} is taken")
            self._change({"kind": "edit dataset", "dataset": dataset, "values": values})

    def _refs(self, libraries: list[tuple[str | None, str]]) -> list[dict]:
        """Each library (depot address, None for this depot, and library id)
        as a dataset names it, at a change number now of the depot that owns
        it: for a library named by this depot's address, the number of what
        this depot holds of its owner (bounds), so a library of the
        upstream that a clone names so is read by the upstream's numbers;
        for a library of another depot, that depot's settled change number
        (wpt_depot.journal), which no change it is still writing reaches.
        NotFound for a library its depot does not hold."""
        refs = []
        for address, library in libraries:
            if self._place(address) is None:
                self._library(library)
                refs.append({"depot": self.address, "library": library,
                             "change": self.bounds(self.name(), self.change())[ids.owner(library)]})
            else:
                depot = Depot.view(address, self.settings)
                depot._library(library)
                refs.append({"depot": depot.address, "library": library, "change": depot.journal.settled()})
        return refs

    def _read_from(self, address: str | None, library: str) -> tuple:
        """What a library a dataset names leads to, so two names of one
        library compare equal: the library's id alone when it is read here
        (reads_here), else where its address leads (_place) with its id."""
        return (None, library) if self.reads_here(address, library) else (self._place(address), library)

    def _place(self, address: str | None) -> Path | str | None:
        """Where a depot's address leads, None for this depot, so two forms
        of one address compare equal; an address of no supported form stays
        as it is written."""
        if address is None:
            return None
        try:
            found = location(address)
        except UnsupportedAddress:
            return address
        return None if found == self.tables.root else found

    def reads_here(self, address: str | None, library: str) -> bool:
        """Whether a library a dataset names (its depot's address, None for
        this depot, and its id) is one of this depot's rows: its address
        leads here (in any form, as _place compares them), or, in a depot
        that follows an upstream, it leads to the upstream and the id is
        one the upstream owns, a library a pull brings here. A corpus still
        reads each library from the depot its address names."""
        place = self._place(address)
        if place is None:
            return True
        return (self.upstream() is not None and place == self._place(self.upstream())
                and bool(self.upstream_owned([library])))

    @staticmethod
    def _checked_dataset(row: dict):
        _filled(name=row["name"], tokenizer=row["tokenizer"])
        if not row["libraries"]:
            raise Refused("a dataset reads at least one library")
        if row["shares"] is not None:
            tags = [one["tag"] for one in row["shares"]]
            if not tags or len(set(tags)) != len(tags) or not all(0 < one["share"] <= 1 for one in row["shares"]):
                raise Refused("shares are distinct tags, each with a share above 0 and at most 1")

    def _new_dataset(self, row: dict) -> str:
        self._checked_dataset(row)
        with self._writes():
            if self.tables.query("SELECT id FROM datasets WHERE name = ?", [row["name"]]).num_rows:
                raise Refused(f"the dataset name {row['name']} is taken")
            row = {**row, "id": self._id(self._next("datasets")), "revision": 1}
            self._change({"kind": "dataset", "dataset": row})
            return row["id"]

    # Inside

    def _decision(self, status: str, declaration: Declaration, reason: str | None = None) -> dict:
        """A person's decision, numbered next. Refused for a reason in a form
        only the depot writes (wpt_depot.policy.license_change, content_change)."""
        if isinstance(reason, str) and policy.starts_over(reason.strip()):
            start = next(one for one in policy.STARTS_OVER if reason.strip().startswith(one))
            what = "license change" if start == policy.LICENSE_CHANGE else "content change"
            raise Refused(f"a reason that starts with {start!r} is the depot's own record of a {what}; write the "
                          "reason another way")
        return self._recorded(status, declaration, reason, self._next("decisions"))

    def _recorded(self, status: str, declaration: Declaration, reason: str | None, number: int) -> dict:
        if declaration.accepted is not True:
            raise Refused("a decision needs the confirmation that the requirements were read, are accepted and met")
        _filled(name=declaration.name, email=declaration.email)
        if reason is not None:
            _filled(reason=reason)
        return {"id": self._id(number), "status": status, "name": declaration.name.strip(),
                "email": declaration.email.strip(), "accepted": True, "made_at": _now().isoformat(),
                "reason": None if reason is None else reason.strip()}

    def _id(self, number: int) -> str:
        return ids.make(self.name(), number)

    def _next(self, table: str) -> int:
        """The next number of a table's ids, past the largest among the ids
        that carry this depot's name. A text's number is never used again:
        the ids that erasures name count too, since they stay in the depot."""
        prefix = ids.prefix(self.name())
        start = len(prefix) + 1
        numbered = (f'SELECT try_cast(substr(id, {start}) AS BIGINT) AS n FROM "{sql_name(table)}" '
                    "WHERE starts_with(id, ?)")
        params = [prefix]
        if table == "texts":
            numbered += (f" UNION ALL SELECT try_cast(substr(t, {start}) AS BIGINT) FROM (SELECT unnest(list_transform("
                         "needles, lambda n: n.text)) AS t FROM erasures_revisions) WHERE starts_with(t, ?)")
            params.append(prefix)
        return self.tables.query(f"SELECT coalesce(max(n), 0) + 1 AS n FROM ({numbered})", params)["n"][0].as_py()

    def _library(self, library: str) -> dict:
        found = self.tables.query("SELECT * FROM libraries WHERE id = ?", [library]).to_pylist()
        if not found:
            raise NotFound(f"no library {library}")
        return found[0]

    def _open_libraries(self) -> list[str]:
        return self.tables.query("SELECT id FROM libraries WHERE locked_at IS NULL")["id"].to_pylist()

    def _erased_from(self, text: str) -> list[dict]:
        """The erased entries of every carried-out request whose passage was in text."""
        rows = self.tables.query("SELECT needles, erased FROM erasures WHERE erased IS NOT NULL").to_pylist()
        return [entry for row in rows for entry in row["erased"] if row["needles"][entry["needle"]]["text"] == text]

    def _blocked(self, texts: list[str], status: str) -> list[dict]:
        """See blocked, over ids known to be texts."""
        found = self.tables.query(f"SELECT id, licenses FROM texts WHERE list_contains(?, id) "
                                  f"ORDER BY {ids.order_sql()}", [texts]).to_pylist()
        theirs = set(self.upstream_owned(texts))
        reasons = [(one["id"], [reason for reason, holds in (
            (BELONGS_TO_UPSTREAM, one["id"] in theirs),
            (NO_ALLOWED_LICENSE, status == ADMITTED and not licenses.any_allowed(one["licenses"]))) if holds])
            for one in found]
        return [{"id": text, "reasons": held} for text, held in reasons if held]

    def _change(self, change: dict, past: int = 0, contents: Sequence[dict] = ()):
        """Write change to the journal under the next change number (past
        past too), then the content versions it adds (contents: rows of
        contents but their change number, which the journal does not hold),
        apply it, mark it done. When the content versions cannot be written,
        the change is cut short and the error raised: nothing of it is
        visible."""
        entry = max(self.change(), past) + 1
        self._unfinished = True  # until it is done or cut short
        self.journal.write(change, entry)
        if contents:
            try:
                self.tables.append("contents", rows("contents", [{**one, "change": entry} for one in contents]))
            except Exception:
                with contextlib.suppress(Exception):  # when this fails too, the next write or opening cuts it short
                    self._cut_short(entry, change)
                    self._unfinished = False
                raise
        try:
            self._apply(change, entry)
            self._done(entry, change)
            self._unfinished = False
        except Exception as error:
            raise JournalUnfinished(f"the journal's change {entry} ({change['kind']}) is not applied to every "
                                    "table yet; the depot's next write or opening it again finishes it") from error

    def _done(self, entry: int, change: dict):
        if change["kind"] == "erase":  # its cut contents leave the journal, and the states that held them go
            self.journal.done(entry, {name: value for name, value in change.items() if name != "cut"})
            self.tables.drop_history(["journal"])
        else:
            self.journal.done(entry)

    def _contents_written(self, entry: int, change: dict) -> bool:
        """Whether the content versions the change adds are in contents, numbered by its entry."""
        found = _content_ids(change)
        return self.tables.query("SELECT count(*) AS n FROM contents WHERE change = ? AND list_contains(?, id)",
                                 [entry, found])["n"][0].as_py() == len(found)

    def _cut_short(self, entry: int, change: dict):
        """End a change whose content versions are not all written: those
        written go (they are written in one commit, so normally none is),
        then its entry is marked done as cut short."""
        listed = ", ".join(map(literal, _content_ids(change)))
        self.tables.delete("contents", f"change = {int(entry)} AND id IN ({listed})")
        self.journal.cut_short(entry, change)

    def _write(self, table: str, found: pa.Table):
        """Add new rows to a table: appended, or, when a change is applied
        again, only those whose key it does not hold yet."""
        (self.tables.insert if self._again else self.tables.append)(table, found)

    # Applying a change: each step leaves the tables as they are when it is applied again

    def _revise(self, table: str, chosen: list[str], number: int, values: Callable[[dict], dict]):
        """A new revision, made at the change number, of each id chosen of a
        revised table: its current row with values(row) put in. An id that
        has a revision made at that change already keeps it."""
        if not chosen:
            return
        found = self.tables.query(f"SELECT * FROM ({latest(table)}) t WHERE list_contains(?, t.id) AND NOT EXISTS "
                                  f"(SELECT 1 FROM {table}_revisions r WHERE r.id = t.id AND r.change = ?)",
                                  [chosen, number]).to_pylist()
        self._write(table, rows(table, [{**row, **values(row), "revision": row["revision"] + 1,
                                                "change": number} for row in found]))

    def _join(self, texts: list[str], library: str, number: int):
        """The texts join the library at the change number; a text in it already stays as it is."""
        if not texts:
            return
        held = set(self.tables.query("SELECT text FROM memberships WHERE library = ? AND left_change IS NULL",
                                     [library])["text"].to_pylist())
        self._write("memberships", rows("memberships", [
            {"library": library, "text": text, "joined_change": number, "left_change": None}
            for text in texts if text not in held]))

    def _leave(self, texts: list[str], libraries: list[str], number: int):
        """The texts leave the libraries at the change number."""
        if not (texts and libraries):
            return
        found = self.tables.query("SELECT library, text, joined_change FROM memberships WHERE left_change IS NULL "
                                  "AND list_contains(?, library) AND list_contains(?, text)", [libraries, texts])
        if found.num_rows:
            self.tables.merge_update("memberships", found.append_column(
                "left_change", pa.array([number] * found.num_rows, pa.int64())), {"left_change": "source.left_change"})

    def _apply(self, change: dict, number: int):
        """Apply a change of the journal, numbered number, to the tables;
        applying it again changes nothing more."""
        kind = change["kind"]
        if "decision" in change:
            self._write("decisions", rows("decisions", [_decision_row(change["decision"])]))
        if "decisions" in change:
            self._write("decisions", rows("decisions", [_decision_row(one) for one in change["decisions"]]))
        if kind == "about":  # in the columns the about table has, an older format's during an upgrade
            current = self.about()
            if current["change"] != number:
                self._write("about", pa.Table.from_pylist(
                    [{**current, **change["values"], "revision": current["revision"] + 1, "change": number}],
                    schema=self.tables.layout("about")))
        elif kind == "license":
            self._write("licenses", rows("licenses", [change["license"]]))
        elif kind == "add":
            self._write("notices", rows("notices", change["notices"]))
            self._write("texts", rows("texts", [{**one, "change": number} for one in change["texts"]]))
        elif kind == "decide":
            decision = change["decision"]["id"]
            self._revise("texts", change["texts"], number, lambda row: {"decisions": [*row["decisions"], decision]})
            self._leave(change["texts"], change["leave"], number)
        elif kind == "relicense":
            for group in change["groups"]:
                self._revise("texts", group["texts"], number, lambda row, group=group: {
                    "licenses": change["licenses"], "decisions": [*row["decisions"], group["decision"]]})
            self._leave([text for group in change["groups"] for text in group["texts"]], change["leave"], number)
        elif kind == "content":
            decision = change["decision"]["id"]
            self._revise("texts", [change["text"]], number, lambda row: {
                "content_version": change["version"], "decisions": [*row["decisions"], decision]})
            self._leave([change["text"]], change["leave"], number)
        elif kind == "credits":
            self._write("notices", rows("notices", change["notices"]))
            decision = change["decision"]["id"]
            self._revise("texts", [change["text"]], number,
                         lambda row: {**change["values"], "decisions": [*row["decisions"], decision]})
        elif kind == "library":  # a new library, or a fork with its texts
            self._write("libraries", rows("libraries", [{**change["library"], "change": number}]))
            self._join(change["texts"], change["library"]["id"], number)
        elif kind == "rename":
            self._revise("libraries", [change["library"]], number, lambda row: {"name": change["name"]})
        elif kind == "put":
            self._join(change["texts"], change["library"], number)
        elif kind == "lock":
            library = change["library"]
            self._write("pins", self.tables.query(
                "SELECT ? AS library, t.id AS text, t.revision FROM texts t WHERE list_contains(t.libraries, ?)",
                [library, library]).cast(rows("pins", []).schema))
            self._revise("libraries", [library], number,
                         lambda row: {"locked_at": datetime.fromisoformat(change["locked_at"])})
        elif kind == "report":
            self._revise("libraries", [change["library"]], number,
                         lambda row: {"problems": [*(row["problems"] or []), change["problem"]]})
        elif kind == "ask erasure":
            self._write("erasures", rows("erasures", [{**change["erasure"], "change": number}]))
        elif kind in ("decide erasure", "erase"):
            decision = change["decision"]["id"]
            erased = None if kind != "erase" else [
                {"length": one["length"], "hash_a": bytes.fromhex(one["hash_a"]), "hash_b": bytes.fromhex(one["hash_b"]),
                 "at": one["at"], "needle": one["needle"]} for one in change["erased"]]
            self._revise("erasures", [change["erasure"]], number, lambda row: {
                "decisions": [*row["decisions"], decision], **({} if erased is None else {"erased": erased})})
            if kind == "erase":  # the decision that admits an erasure and its carrying out are one change
                self._apply_erase(change)
        elif kind == "dataset":
            self._write("datasets", rows("datasets", [{**change["dataset"], "change": number}]))
        elif kind == "edit dataset":
            self._revise("datasets", [change["dataset"]], number, lambda row: change["values"])
        elif kind == "pull":
            from .follow import apply_pull
            apply_pull(self, change, number)
        else:
            raise ValueError(f"no change of kind {kind!r}")

    def _apply_erase(self, change: dict):
        for column in ("text", "content"):
            found = [one for one in change["cut"] if column in one]
            if found:
                source = pa.table({"id": pa.array([one["id"] for one in found], pa.string()),
                                   "version": pa.array([one["version"] for one in found], pa.int64()),
                                   column: pa.array([one[column] for one in found], pa.string())})
                self.tables.merge_update("contents", source, {column: f"source.{column}"})
        self._remove_texts(change["removed"])
        self.journal.rewrite(_without(self.journal.entries("add"), set(change["removed"])))
        self.tables.drop_history()

    def _remove_texts(self, texts: list[str]):
        """Remove the texts whole: every revision, every content version, memberships and pins (an erasure's part)."""
        if texts:
            listed = ", ".join(map(literal, texts))
            for table, column in (("texts", "id"), ("contents", "id"), ("memberships", "text"), ("pins", "text")):
                self.tables.delete(table, f"{column} IN ({listed})")


def location(address: str) -> Path | str:
    """The folder of a depot's address: a path, or a file:// address; for a
    bucket, the address s3://bucket/prefix as it is (without a closing
    slash). UnsupportedAddress for any other."""
    parsed = urlparse(address)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise UnsupportedAddress(f"{address}: a depot in a bucket is s3://bucket/prefix")
        return address.rstrip("/")
    if parsed.scheme and len(parsed.scheme) > 1:  # one letter is a Windows drive
        raise UnsupportedAddress(f"{address}: a depot is a folder (a path or a file:// address) or a bucket "
                                 "(s3://bucket/prefix)")
    return Path(address).resolve()


def _without(entries: list[tuple[int, dict]], removed: set[str]) -> dict[int, dict]:
    """The add entries that name a text an erasure removed, without its row."""
    changed = {}
    for entry, change in entries:
        kept = [row for row in change["texts"] if row["id"] not in removed]
        if kept != change["texts"]:
            changed[entry] = {**change, "texts": kept}
    return changed


def _content_ids(change: dict) -> list[str]:
    """The ids of the texts whose content versions a change of the journal adds outside it, in contents."""
    if change["kind"] == "add":
        return [row["id"] for row in change["texts"]]
    if change["kind"] == "content":
        return [change["text"]]
    return []


def _needles(request: dict) -> list[Needle]:
    """The needles of a row of erasures."""
    return [Needle(one["text"], one["start"], one["end"], one["field"], one["version"]) for one in request["needles"]]


def _same_json(one: str, other: str) -> bool:
    """Whether two JSON texts hold the same value (an object's keys in any order)."""
    return json.dumps(json.loads(one), sort_keys=True) == json.dumps(json.loads(other), sort_keys=True)


def _check_text(one: NewText, licenses: set, held_kinds: dict):
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
    unknown = sorted(set(one.licenses) - licenses)
    if unknown:
        raise Refused(f"the depot holds no license {', '.join(unknown)}")


def _credit(value) -> str:
    """A credit's value as a reason names it: a list's items joined by commas, 'none' for no value."""
    if isinstance(value, (list, tuple)):
        value = ", ".join(value)
    return value if value else "none"


def _decision_row(decision: dict) -> dict:
    return {**decision, "made_at": datetime.fromisoformat(decision["made_at"])}


def _filled(**values):
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise Refused(f"{name.replace('_', ' ')} is empty")


def _one_select(query: str):
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as error:
        raise BadQuery(str(error)) from None
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise BadQuery("a query is one SELECT statement")


def _now() -> datetime:
    return datetime.now(UTC)
