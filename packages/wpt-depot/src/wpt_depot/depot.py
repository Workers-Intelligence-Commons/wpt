"""A depot: texts and structured items, the decisions on them, libraries,
erasures, datasets and tokenizer recipes with their vocabularies, in the
tables of one SQLite file (wpt_depot.schema).

Depot is the only code that changes a depot's tables. Each operation is one
change: one transaction under the next change number, recorded for the
backups (wpt_depot.database). It checks the whole change against the
depot's rules inside it, so a refused or failed change writes nothing.

The depot's state is numbered by its change number (Depot.change): every
write raises it, and every revision row and membership records the change
number it was made at. What changes in place keeps its old versions as
revision rows of its own table, never overwritten or deleted but by an
erasure: a text's credits and licenses, a library's name, lock
and problems, a dataset, the depot's maintainer and takedown contact, an
erasure request's decisions and what it erased. A decision on a text is a
row of text_decisions with the change it was made at, never deleted but by
an erasure. Each write keeps the status stored on each current text it
changes and the counts (wpt_depot.counts) right, in its own transaction.

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
- An operation on texts (a decision, a license change, a put, a take-out,
  a fork's texts left out, a report) takes them as a selection: an SQL
  condition over the texts view (the current state), the exact ids of
  chosen texts, or a filter (wpt_depot.filters), each matched when the
  operation runs. Every id a selection names must be a text of the depot.
  A filter may carry the change number its page was read at and the ids
  unticked by hand: the operation leaves those ids out, and every text
  changed after that change number, and says how many it left out
  (CHANGED_SINCE). Every operation over many texts runs in SQL over a
  temporary table of the selected ids, and takes a progress function that
  it calls with the name of each stage it starts.
- A decision on texts is linked to every text selected, as a row of
  text_decisions, with an optional public reason; it makes no new revision
  of a text, and changes the status stored on its current revision. Admitting is refused when a
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
  the revision it pinned. The plain texts of many texts change at once the
  same way, each text as if changed alone, under one decision for each
  pair of old and new versions; the change reads its lines a few at a time
  into a temporary table and is checked by queries over it
  (wpt_depot.rewriting).
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
  upstream change its pulls brought), optional ordered shares by tag, an
  optional max_tokens (the size a build form starts at), the name of a
  tokenizer recipe, which need not exist yet, and, once set, the vocabulary
  it publishes (the sha256 of one of the depot's vocabularies), which
  training takes its corpus of. Its name, libraries, shares, max_tokens and
  vocabulary change as a new revision, its id and tokenizer recipe kept,
  with the checks of a new one;
  a library it read already keeps its change number. Exported, a dataset
  is a remote (wpt_depot.remote).
- A tokenizer recipe names the dataset whose sample trains its vocabulary,
  this depot's or another depot's by its web address, with sample_bytes,
  vocab_size and a note. It changes as a new revision; a change of what
  trains the vocabulary raises its version, a change of the note alone does
  not. A version has at most one vocabulary, written once with the list of
  the texts it trained on; a second one is refused.
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
import hashlib
import itertools
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import apsw

from . import counts, crediting, erasure, ids, kinds, licenses, policy, rewriting, scoring, titles
from .database import Database, person_columns
from .erasure import Needle
from .errors import (BelongsToUpstream, DepotExists, HalfUpgraded, Locked, NotADepot, NotFound, OtherFormat, Refused,
                     UnsupportedAddress)
from .filters import (IN_LOCKED, MEMBERSHIPS_KEY, PINS_KEY, TEXTS_KEY, Filter, changed_since, condition, is_id,
                      listed as listed_ids, literal, strings, words)
from .parts import WINDOW, FileText, Parted, Reader, Source, listed, made, pack, parted, split
from .schema import (ADMITTED, CONTENT_JOIN, FORMAT, JSON_COLUMNS, LISTED_COLUMNS, LONG, REFUSED, UNDER_REVIEW,
                     decision_list)

# the revision of the text h.text a library held at the change :change, the change numbers of each depot's rows it
# held then given by :bounds (Depot.bounds), and whether the membership h was open then
REVISION_THEN = (f"(SELECT max(r.revision) FROM texts_revisions r INDEXED BY {TEXTS_KEY} WHERE r.id = h.text "
                 f"AND r.change <= (SELECT b.value FROM json_each(:bounds) b WHERE b.key = {ids.owner_sql('r.id')}))")
MEMBER_THEN = "AND h.joined_change <= :change AND (h.left_change IS NULL OR h.left_change > :change) "
# texts an operation works on: an SQL condition over the texts view, matched when the operation runs, a filter
# (wpt_depot.filters) matched then, less the texts changed since its change number, or the exact ids of chosen texts
Selection = str | Filter | Sequence[str]
# why a decision cannot cover a text (Depot.blocked): none of its licenses is allowed (an admission only), or it is
# a row the upstream of a clone owns (any decision)
NO_ALLOWED_LICENSE = "no allowed license"
BELONGS_TO_UPSTREAM = "belongs to the upstream"
# why an operation on a filter with a change number leaves a text out: something changed it after that number
CHANGED_SINCE = "changed since"
# why putting texts back under review leaves a text out: it is under review already, and starting over would only
# wipe its decisions
ALREADY_UNDER_REVIEW = "already under review"
# why a decision asked to leave out the texts whose status it would not change (Depot.decide's leave_same) leaves a
# text out, by the decision's status
ALREADY = {ADMITTED: "already admitted", REFUSED: "already refused"}
# how many ids a refusal of upstream rows names before it gives the count of the rest
NAMED = 20
# what a clone can do instead of changing an upstream library: a fork is a library of its own
FORK = ", or fork it into a library of this depot"
SINGULAR = {"texts": "text", "libraries": "library", "erasure requests": "erasure request", "datasets": "dataset",
            "tokenizer recipes": "tokenizer recipe"}
# the credits of a text that change as a new revision (Depot.edit_credits), in the order its reason names them
CREDITS = crediting.CREDITS
# what the rows of a search can be ordered by: each the SQL expressions over the searched rows it orders by, in turn,
# before the numeric order of ids (_ordered); none of them is ever null, so a row's values of them compare as a whole
ORDERS = {"id": [], "title": ["title IS NOT NULL", "coalesce(title, '')"],
          "kind": ["kind IS NOT NULL", "coalesce(kind, '')"], "status": ["status"], "tags": ["tags"], "licenses": ["licenses"], "libraries": ["libraries"],
          "length": ["length"]}
# the orders an index of the depot serves (schema.INDEXES), so a page in them reads only its own rows; a page in any
# other order sorts every match, so it is given only when a search matches at most SORTED texts
INDEXED = ("id", "status")
SORTED = 100_000
# a tokenizer recipe's name, which a dataset names its vocabulary by
TOKENIZER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
# what a tokenizer recipe's revision changes that trains another vocabulary, and so raises its version
TRAINS_VOCABULARY = ("depot", "dataset", "sample_bytes", "vocab_size")
VOCABULARY_BASE = 513  # the smallest vocabulary: 256 bytes, the end of a text and 256 reserved ids (wpt_tokenizer)
# how many texts a step of Depot.count goes over at most
COUNT_STEP = 1_000_000
# how many candidates a page reads at most (Depot.search), and with a search in the texts themselves, which reads
# each candidate's text
WINDOW, TEXT_WINDOW = 10_000, 200
# how many revisions, the latest changes first, Depot.score_values goes over at most, and how many values it gives
SCORE_READ, SCORE_VALUES = 200_000, 12
# the columns of the texts view but those that hold a text: what a query reads that never reads a text's parts
FIELDS = ("id, revision, change, kind, content_version, length, licenses, notice, title, authors, tags, address, "
          "decisions, scores, status, libraries")
# every text with the columns a page reads and orders by (Depot.search), before the condition that picks them
MATCHED = f"SELECT {FIELDS}, {IN_LOCKED} AS locked FROM texts"


@dataclass(frozen=True)
class _Listing:
    """Rows a page, a count step or an operation reads its candidates from
    (Depot._plan): rows, an SQL query whose column key holds each text's
    place in the numeric order of ids (ids.order_sql); texts, how many rows
    it gives at most; ordered, whether an index gives them in that order
    (else they are sorted, so they are read only when at most SORTED);
    always, whether they are read whatever their number (ids given);
    parts, for the texts of several values, each value's rows, each in that
    order through its index, whose union is rows (_window merges them): an
    SQL query, or a part of the title index (titles.Part), which reads its
    own window; scans, whether they are the rows of the texts themselves
    (every text, or one status), which an operation over every match reads
    as they are stored rather than through the listing; total, when texts
    is only a lower bound (the title index, counted up to SORTED + 1 texts
    for a plan), what gives how many rows it holds, None past COUNT_STEP
    rows, whose counting would take as long as the steps."""
    rows: str
    texts: int
    ordered: bool
    always: bool = False
    parts: tuple["str | titles.Part", ...] = ()
    scans: bool = False
    total: Callable[[], int] | None = None


@dataclass(frozen=True)
class _Plan:
    """What reads the texts a selection matches: where, the SQL condition
    over the texts view that matches them; listing, the rows read as
    candidates; status, the one status a filter names (None for any
    other); in_text, whether its plain search reads the texts
    themselves."""
    where: str
    listing: _Listing
    status: str | None = None
    in_text: bool = False


def _listing(rows: str, values: Sequence[str], held: dict[str, int]) -> _Listing | None:
    """The listing of the texts that have one of the values (a filter's
    licenses, libraries or statuses), from rows, a query whose {names} is
    the test of its name column: each value read in key order from its
    index, several merged (_window); held gives how many texts have each
    (the depot's counts), their sum an upper bound of the texts of several.
    None without a value."""
    values = list(dict.fromkeys(values))
    if not values:
        return None
    total = sum(held.get(one, 0) for one in values)
    parts = tuple(rows.format(names=f"= {literal(one)}") for one in values)
    if len(parts) == 1:
        return _Listing(parts[0], total, True)
    return _Listing(" UNION ".join(f"SELECT key FROM ({one})" for one in parts), total, True, parts=parts)


def _credited(names) -> tuple[str, ...]:
    """The facets of counts, the listings and the index a change of the
    credits names keeps right (wpt_depot.counts): those of tags when the
    tags change, the title index when the title does, and always the pins
    older than their text's new revision."""
    return (((counts.TAG, counts.LIBRARY_TAG, counts.SPLIT_TAG) if "tags" in names else ())
            + ((counts.TITLE,) if "title" in names else ()) + (counts.NEWER,))


def _window(listing: _Listing, compare: str | None, direction: str, size: int) -> tuple[str, int]:
    """(the SQL of at most size keys of the listing, the first in the order
    asked (direction: '' or ' DESC') whose key compares with one parameter
    as compare says ('>', '<=', ...; None: from the first), how many
    parameters it takes, each that place): each part of the listing read in
    order through its index, at most size rows of each, then merged, so the
    read is bounded by size for each part whatever the depot's size."""
    after = "" if compare is None else f"WHERE key {compare} ?"
    reads, uses = [], 0
    for one in listing.parts or (listing.rows,):
        if isinstance(one, titles.Part):  # its bound is a row number of the index, which it reads from
            read, used = one.window(compare, direction, size)
            reads.append(f"SELECT key FROM ({read})")
            uses += used
        else:
            reads.append(f"SELECT key FROM ({one}) {after} ORDER BY key{direction} LIMIT {int(size)}")
            uses += 0 if compare is None else 1
    if len(reads) == 1:
        return reads[0], uses
    merged = " UNION ".join(f"SELECT key FROM ({one})" for one in reads)
    return f"SELECT key FROM ({merged}) ORDER BY key{direction} LIMIT {int(size)}", uses


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
        (schema.FORMAT), HalfUpgraded when an upgrade to it has not ended."""
        found = format_of(address)
        depot = cls(address, readonly=True)
        _check_format(depot, found)
        return depot

    @classmethod
    def open(cls, address: str) -> "Depot":
        """The depot at address, to write: the change files a crash left
        pending kept or dropped first (wpt_depot.database). NotADepot when
        the address holds none; OtherFormat when its format is not this
        code's (schema.FORMAT), HalfUpgraded when an upgrade to it has not
        ended."""
        found = format_of(address)
        depot = cls(address)
        _check_format(depot, found)
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
        change or putting back under review, wpt_depot.policy), and
        `put_back`: whether it put the text back under review,
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
        held = [{**one, "upstream": bool(one["upstream"])} for one in self._rows(
            f"SELECT *, {self.upstream_sql()} AS upstream FROM libraries "
            "WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id", (_json(row["libraries"]),))]
        notice = None if row["notice"] is None else self._one("SELECT text FROM notices WHERE id = ?", (row["notice"],))
        revisions = self._one("SELECT count(*) FROM texts_revisions WHERE id = ?", (text,))
        versions = self._one("SELECT count(*) FROM content_lists WHERE id = ?", (text,))
        pins = self._rows("SELECT p.library, l.name, p.revision FROM pins p JOIN libraries l ON l.id = p.library "
                          "WHERE p.text = ? ORDER BY p.library", (text,))
        decisions = self._rows(f"SELECT d.* FROM text_decisions l JOIN decisions d ON d.id = l.decision "
                               f"WHERE l.text = ? ORDER BY {ids.order_sql('d.id')}", (text,))
        counting = {one["id"] for one in policy.counted(decisions)}
        status = row.pop("status")
        return {"text": row, "status": status, "notice": notice,
                "decisions": [{**one, "counts": one["id"] in counting, "put_back": policy.put_back(one)}
                              for one in decisions], "libraries": held,
                "revisions": revisions, "pins": pins, "content_versions": versions,
                "upstream": bool(self.upstream_owned([text]))}

    def search(self, where: "Selection", after: Sequence | None = None, before: Sequence | None = None,
               limit: int = 50, order: str = "id", descending: bool = False, at: str | None = None) -> dict:
        """The texts a selection matches (an SQL condition over the texts
        view, a filter, or ids), a page at a time, with no count
        (Depot.count counts them a step at a time): {rows: the page of at
        most limit rows, ordered by the column order (one of ORDERS; then by
        id, in the numeric order of ids), descending turning the whole order
        around; next: the place where the next page starts, None when no
        row follows; previous: the place of its first row when rows come
        before it, None on the first page, and the place after itself for
        an empty page after a place, since rows can come before it; partial:
        whether the page stopped at its reading bound rather than at limit
        rows (next then names the last candidate read, and the next page
        reads on from there); sortable: whether the listing is small enough
        for every order of ORDERS, not only those of INDEXED; change: the
        depot's change number, read before its rows}.

        A page reads its candidates from the narrowest stored listing the
        selection names (_plan): the texts of one tag, one license, one
        library or one status, in the order of their ids, from their index;
        a decision's texts, several licenses or libraries, or the ids
        given, when they are at most SORTED, sorted; else every text in the
        order of its index. In the order id it reads at most WINDOW
        candidates (TEXT_WINDOW with a search in the texts themselves) after
        its place, and tests each on the rest of the selection, so a page
        is bounded whatever the depot's size and however few texts match:
        a page with fewer than limit rows and a next place is partial. In
        the order status, the same over the index of statuses, unless the
        listing is at most SORTED texts. In another order it sorts every
        candidate, so it is given only when the listing holds at most SORTED
        texts.

        A place is the list of a row's values of the order (_ordered), and a
        page starts after one (after) or ends before one (before), so a far
        page is read as fast as the first; a page before a place that has
        fewer than limit rows before it and no reading bound is the first
        page. A page at a text's id (at) starts at that text's place in the
        order, with the text itself when the selection matches it, so a page
        can open on a text the rows before it would push out; previous is
        then that page's first place, and an id the depot does not hold
        gives the first page. A row holds a text's credits, never its text
        or content: its id, kind, title, authors, tags, licenses, libraries,
        scores, status, whether a locked library holds it, and its length in
        characters. BadQuery for a condition that is not one; Refused for an
        order that is none of ORDERS, an order the listing cannot give
        within SORTED texts, more than one of after, before and at, or a
        place that is not one of the order."""
        if sum(one is not None for one in (after, before, at)) > 1:
            raise Refused("a page starts after one place, ends before one or starts at one text, only one of them")
        change = self.change()
        plan = self._plan(where)
        if at is not None:
            held = self.db.person_rows(f"SELECT json_array({', '.join(_ordered(order))}) AS place FROM ({MATCHED}) "
                                       "WHERE id = ?", (at,))
            if not held:
                return self.search(where, limit=limit, order=order, descending=descending)
            after = json.loads(held[0]["place"])
        columns = "id, kind, title, authors, tags, licenses, libraries, scores, status, locked, length"
        found, edge = self._page(plan, columns, order, descending, before if before is not None else after,
                                 before is not None, limit, at is not None)
        if before is not None and len(found) <= limit and edge is None:
            return self.search(where, limit=limit, order=order, descending=descending)
        page = found[:limit]
        places = [one.pop("place") for one in page]
        if before is not None:
            page.reverse()
            places.reverse()
            more_before = len(found) > limit or edge is not None
            answer_next, previous = before, (places[0] if page and more_before else edge if more_before else None)
            partial = len(found) <= limit
        else:
            answer_next = places[-1] if len(found) > limit else edge
            partial = len(found) <= limit and edge is not None
            previous = (places[0] if page and (after is not None or at is not None) else after)
        return {"rows": [_decoded({**one, "locked": bool(one["locked"])}) for one in page],
                "next": answer_next, "previous": previous, "partial": partial,
                "sortable": plan.listing.always or plan.listing.texts <= SORTED, "change": change}

    def passages(self, where: "Selection", after: Sequence | None = None, limit: int = 50,
                 characters: int = 200_000, order: str = "id", descending: bool = False) -> dict:
        """The texts a selection matches with their text, a page at a time,
        in the order of search and read as search reads them: {rows: after
        the place after (from the first text when None), at most limit texts
        holding at most characters characters together, each {id, kind,
        title, length: its length in characters, passage: its text, or an
        item's content, from its start, whole, or cut at characters for a
        first text longer than that}, next: the place where the next page
        starts, None when the page ends the texts matched; partial: as
        search says}. A later text that does not fit whole starts the next
        page. BadQuery for a condition that is not one; Refused as search
        is, and for a bound below 1."""
        if limit < 1 or characters < 1:
            raise Refused("a page holds at least one text and one character")
        found, edge = self._page(self._plan(where), "id, kind, title, length", order, descending, after, False,
                                 limit)
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
        cut = len(page) < len(found)
        return {"rows": found_rows, "next": page[-1]["place"] if cut else edge,
                "partial": not cut and edge is not None}

    def _plan(self, texts: "Selection") -> "_Plan":
        """How a page, a count step or an operation reads the texts a
        selection matches: the SQL condition over the texts view that
        matches them, and the listing whose rows it reads as candidates
        (search says which), with how many rows it holds."""
        if isinstance(texts, str):
            return _Plan(texts, self._every())
        if not isinstance(texts, Filter):
            given = sorted(set(texts))
            return _Plan(self._condition(given), _Listing(
                f"SELECT {ids.order_sql('value')} AS key FROM json_each({literal(_json(given))})", len(given), False))
        held = counts.read(self.db.reader())
        found = texts
        options = [self._every(held)]
        if found.ids is not None:
            given = sorted(set(found.ids))
            options.append(_Listing(f"SELECT {ids.order_sql('value')} AS key FROM json_each({literal(_json(given))})",
                                    len(given), False, always=True))
        for tag in found.tags:
            options.append(_Listing(f"SELECT key FROM text_tags WHERE tag = {literal(tag)}",
                                    held[counts.TAG].get(tag, 0), True))
        options.append(_listing("SELECT key FROM text_licenses WHERE license {names}", found.licenses,
                                held[counts.LICENSE]))
        options.append(_listing(f"SELECT {ids.order_sql('text')} AS key FROM memberships WHERE library {{names}} "
                                "AND left_change IS NULL", found.libraries, held[counts.LIBRARY]))
        statuses = _listing(f"SELECT {ids.order_sql('id')} AS key FROM texts_revisions WHERE latest AND status "
                            "{names}", found.statuses, held[counts.STATUS])
        options.append(statuses if statuses is None or statuses.parts else replace(statuses, scans=True))
        if found.decision is not None:  # in key order through the index text_decisions_order
            options.append(_Listing(f"SELECT {ids.order_sql('text')} AS key FROM text_decisions WHERE decision = "
                                    f"{literal(found.decision)}", held[counts.DECISION].get(found.decision, 0), True))
        options = [one for one in options if one is not None and (one.ordered or one.always or one.texts <= SORTED)]
        if found.search and not found.in_text:  # counted up to the smallest other listing it competes with
            titled = self._titled(words(found.search), min((one.texts for one in options[1:]), default=SORTED))
            options += [] if titled is None else [titled]
        listing = min(options, key=lambda one: one.texts)
        status = (found.statuses[0] if len(set(found.statuses)) == 1 else None)
        return _Plan(self.condition(found, self.change()), listing, status, found.in_text)

    def _titled(self, searched: list[str], rival: int) -> "_Listing | None":
        """The listing of the texts a plain search's words may match in their
        title or id (filters.condition): the texts the title index gives for
        every word of titles.SHORTEST characters or more, each owner's in
        the order of its index, and the text whose id is a word in the form
        of an id, by its key; None when no word is that long. How many: up
        to min(rival, SORTED) + 1 (rival: the texts of the smallest listing
        it competes with, every text's aside, which holds them all), so a
        page counts no further than the choice of a listing and whether
        every order may sort them need, counted from the index, exactly
        through total up to COUNT_STEP. Each count is a person's statement
        (Database.person_rows), stopped at the deadline of the request."""
        query = titles.match(searched)
        if query is None:
            return None
        con = self.db.reader()
        named = [one for one in dict.fromkeys(searched) if is_id(one)]
        by_id = tuple(f"SELECT {ids.order_sql('id')} AS key FROM texts_revisions WHERE latest AND id = {literal(one)}"
                      for one in named)
        parts = (*titles.parts(con, query), *by_id)
        if not parts:
            return _Listing("SELECT NULL AS key WHERE 0", 0, True)
        rows = " UNION ".join(f"SELECT key FROM ({getattr(one, 'rows', one)})" for one in parts)

        def counted(sql: str, params: tuple) -> int:
            return self.db.person_rows(sql, params)[0]["n"]

        def total() -> int | None:  # with the texts named by id that the index does not give already
            found = counted(*titles.count_sql(query, COUNT_STEP + 1))
            return None if found > COUNT_STEP else found + counted(*titles.outside_sql(query, named))
        return _Listing(rows, counted(*titles.count_sql(query, min(rival, SORTED) + 1)) + len(by_id), True,
                        parts=parts, total=total)

    def _every(self, held: dict | None = None) -> "_Listing":
        """The listing of every current text, in the order of its index."""
        held = counts.read(self.db.reader()) if held is None else held
        return _Listing(f"SELECT {ids.order_sql('id')} AS key FROM texts_revisions WHERE latest",
                        held[counts.TEXTS].get("", 0), True, scans=True)

    def _page(self, plan: "_Plan", columns: str, order: str, descending: bool, place: Sequence | None,
              backward: bool, limit: int, inclusive: bool = False) -> tuple[list[dict], list | None]:
        """(at most limit + 1 rows of the texts the plan matches, with
        columns and their place (_ordered), in the order asked, from the one
        after place (inclusive: from the one at it), or backward from the
        one before it, the nearest first; edge: the place of the last
        candidate read when the page stopped at its reading bound, None
        otherwise), read as search says."""
        keys = _ordered(order)
        down = descending != backward
        if place is not None and (not isinstance(place, (list, tuple)) or len(place) != len(keys) or not all(
                isinstance(one, (str, int, float)) and not isinstance(one, bool) for one in place)):
            raise Refused(f"a place in the order {order} is a list of {len(keys)} values, as a page gives it")
        listing, size = plan.listing, TEXT_WINDOW if plan.in_text else WINDOW
        compare = f"{'<' if down else '>'}{'=' if inclusive else ''}"
        direction = " DESC" if down else ""
        params: tuple = ()
        if order == "id":
            window, uses = _window(listing, None if place is None else compare, direction, size)
            params = (place[0],) * uses if place is not None else ()
            last = f"SELECT json_array(key) FROM ({window}) ORDER BY key{' ASC' if down else ' DESC'} LIMIT 1"
        elif order == "status" and listing.texts > SORTED and not listing.always:
            # one window per status, each a range of the index texts_status (status, then id order), joined in order
            statuses = sorted([plan.status] if plan.status is not None else [UNDER_REVIEW, ADMITTED, REFUSED],
                              reverse=down)
            if place is not None:
                statuses = [one for one in statuses if one == place[0] or (one < place[0]) == down]
            parts = []
            for one in statuses:
                after = f" AND key {compare} ?" if place is not None and one == place[0] else ""
                params += (place[1],) if after else ()
                parts.append(f"SELECT * FROM (SELECT t.status AS status, {ids.order_sql('t.id')} AS key "
                             f"FROM texts_revisions t INDEXED BY texts_status WHERE t.latest AND "
                             f"t.status = {literal(one)}{after.replace('key', ids.order_sql('t.id'))} "
                             f"ORDER BY key{direction} LIMIT {size})")
            window = (f"SELECT status, key FROM ({' UNION ALL '.join(parts) or 'SELECT NULL AS status, NULL AS key WHERE 0'})"
                      f" ORDER BY status{direction}, key{direction} LIMIT {size}")
            last = (f"SELECT json_array(status, key) FROM ({window}) ORDER BY status{' ASC' if down else ' DESC'}, "
                    f"key{' ASC' if down else ' DESC'} LIMIT 1")
        else:
            if listing.texts > SORTED and not listing.always:
                raise Refused(f"a page ordered by {order} sorts every text it reads, so it is given only when the "
                              f"filter names at most {SORTED:,} texts by a tag, a license, a library, a status or a "
                              f"decision, and this one names more: narrow the filter, or order by "
                              f"{' or '.join(INDEXED)}")
            window, last = f"SELECT key FROM ({listing.rows})", None
        # the window is read on its own (MATERIALIZED), in the order of its index, before any text is looked up
        matched = (f"SELECT {FIELDS}, {IN_LOCKED} AS locked FROM w CROSS JOIN texts "
                   f"ON texts.id = {ids.key_id_sql('w.key')} WHERE ({plan.where})")
        compared = ("" if place is None else
                    f"WHERE ({', '.join(keys)}) {compare} ({', '.join('?' * len(keys))})")
        by = ", ".join(f"{one}{direction}" for one in keys)
        found = self.db.person_rows(
            f"WITH w AS MATERIALIZED (SELECT key FROM ({window})) SELECT {columns}, json_array({', '.join(keys)}) AS place "
            f"FROM ({matched}) {compared} ORDER BY {by} LIMIT {int(limit) + 1}",
            params + (tuple(place) if place is not None else ()))
        edge = None
        if last is not None and len(found) <= limit:
            read = self.db.person_rows(f"SELECT count(*) AS n FROM ({window})", params)[0]["n"]
            if read >= size:
                edge = json.loads(self.db.person_rows(f"SELECT ({last}) AS place", params)[0]["place"])
        return [{**one, "place": json.loads(one["place"])} for one in found], edge

    def count(self, where: "Selection", start: str | None = None, size: int = COUNT_STEP,
              texts: Sequence[str] | None = None) -> dict:
        """How many texts a selection matches in one slice of the rows of
        the listing it is read from (_plan), so that every match is counted
        a step at a time, each step bounded whatever the depot's size: the
        slice is the size rows of the listing after the place start (from
        the first when None), in the order of its index or sorted, as search
        reads them; with texts (ids), it is exactly those of the texts named that
        the depot holds, in one step. {texts: how many rows the slice holds;
        total: how many rows every step goes over (the listing's, or the
        texts named that the depot holds), with the first step only (start
        None), None after it, and None too for the texts of a search's
        title index past COUNT_STEP; statuses: how many of the matches are under
        review, admitted, refused (the status the depot stores,
        wpt_depot.policy); locked: how many of them a locked library holds;
        tags and licenses: how many of them have each tag and name each
        license, the most first; blocked: {reason: how many of them an
        admission cannot cover for it} (Depot.blocked); next: the start of
        the next slice, the place of this slice's last row, None when this
        slice ends the listing; held: False, read from the texts rather
        than the depot's counts (held_count)}. BadQuery for a condition that
        is not one; Refused for a size below 1, or texts given with
        start."""
        if size < 1:
            raise Refused("a step counts at least one text")
        plan = self._plan(where)
        if texts is not None:
            if start is not None:
                raise Refused("the texts named are counted in one step, with no start")
            given = sorted(set(texts))
            listing = _Listing(f"SELECT {ids.order_sql('value')} AS key FROM json_each({literal(_json(given))})",
                               len(given), False)
            slice_ = f"SELECT key FROM ({listing.rows})"
            held = self._one("SELECT count(*) FROM texts_revisions WHERE latest AND id IN "
                             "(SELECT value FROM json_each(?))", (_json(given),))
            step, params = {"texts": held, "total": held, "next": None}, ()
        else:
            listing = plan.listing
            slice_, uses = _window(listing, None if start is None else ">", "", size)
            params = (start,) * uses
            read = self.db.person_rows(f"SELECT count(*) AS n, max(key) AS last FROM ({slice_})", params)[0]
            total = listing.total() if listing.total is not None else listing.texts
            step = {"texts": read["n"], "total": total if start is None else None,
                    "next": read["last"] if read["n"] >= size else None}
        found = self.db.person_rows(
            f"WITH w AS MATERIALIZED ({slice_}), "
            f"m AS MATERIALIZED (SELECT status, {IN_LOCKED} AS locked, {self.upstream_sql()} AS theirs, "
            f"{self._unlicensed_sql('licenses')} AS unlicensed, tags, licenses FROM w CROSS JOIN texts "
            f"ON texts.id = {ids.key_id_sql('w.key')} WHERE ({plan.where})) "
            "SELECT 'status' AS part, status AS name, locked, count(*) AS n FROM m GROUP BY status, locked "
            f"UNION ALL SELECT 'blocked', '{BELONGS_TO_UPSTREAM}', 0, count(*) FROM m WHERE theirs "
            f"UNION ALL SELECT 'blocked', '{NO_ALLOWED_LICENSE}', 0, count(*) FROM m WHERE unlicensed AND NOT theirs "
            "UNION ALL SELECT 'tag', j.value, 0, count(*) FROM m, json_each(m.tags) j GROUP BY j.value "
            "UNION ALL SELECT 'license', j.value, 0, count(*) FROM m, json_each(m.licenses) j GROUP BY j.value",
            params)
        statuses = [one for one in found if one["part"] == "status"]
        named_counts = {part: sorted(([one["name"], one["n"]] for one in found if one["part"] == part),
                                     key=lambda one: (-one[1], one[0])) for part in ("tag", "license")}
        return {**step, "held": False,
                "statuses": {status: sum(one["n"] for one in statuses if one["name"] == status)
                             for status in (UNDER_REVIEW, ADMITTED, REFUSED)},
                "locked": sum(one["n"] for one in statuses if one["locked"]),
                "blocked": {one["name"]: one["n"] for one in found if one["part"] == "blocked"},
                "tags": [{"tag": name, "texts": n} for name, n in named_counts["tag"]],
                "licenses": [{"license": name, "texts": n} for name, n in named_counts["license"]]}

    def held_count(self, found: Filter) -> dict | None:
        """The count of the texts a filter matches, in the shape of a count
        step over every text (count), read at once from the depot's counts
        (wpt_depot.counts) whatever the depot's size, or None when they
        cannot give it: they split every text, and the texts of each tag,
        license and library, by status and by whether a locked library
        holds them (counts.SPLIT), so they answer a filter of any statuses,
        of at most one tag, license or library, and of a locked part, and
        nothing else (no ids, search, SQL, outside, decision, unticked
        texts, two tags, licenses or libraries, nor a change number older
        than the depot's). tags, licenses and blocked are None where the
        counts do not give them: they do when the filter matches every text
        or none, and blocked also when no upstream is followed and every
        license a text names is allowed (every text names one). held:
        True."""
        named_parts = [(facet, values) for facet, values in ((counts.TAG, found.tags), (counts.LICENSE, found.licenses),
                                                             (counts.LIBRARY, found.libraries)) if values]
        if (found.ids is not None or found.unticked or found.outside or found.decision
                or any(one not in (UNDER_REVIEW, ADMITTED, REFUSED) for one in found.statuses)
                or (found.where or "").strip() or (found.search or "").strip() or len(named_parts) > 1
                or any(len(set(values)) > 1 for _, values in named_parts)
                or (found.change is not None and found.change < self.change())):
            return None
        con = self.db.reader()
        with con:  # one read transaction: every count is of one state
            held = counts.read(con)
            facet, value = (named_parts[0][0], named_parts[0][1][0]) if named_parts else (counts.TEXTS, "")
            parts = counts.split(con, facet, value)
        every = held[counts.TEXTS].get("", 0)
        wanted = set(found.statuses) or {UNDER_REVIEW, ADMITTED, REFUSED}
        kept = [(status, locked, n) for (status, locked), n in parts.items()
                if status in wanted and (found.locked is None or locked == found.locked)]
        statuses = {status: sum(n for one, _, n in kept if one == status)
                    for status in (UNDER_REVIEW, ADMITTED, REFUSED)}
        locked = sum(n for _, one, n in kept if one)
        matched = sum(statuses.values())
        whole = matched == every  # the filter matches every text, or the depot holds none
        named = {facet: sorted(([name, n] for name, n in held[facet].items()), key=lambda one: (-one[1], one[0]))
                 for facet in (counts.TAG, counts.LICENSE)}
        allowed = {one for (one,) in self.db.reader().execute("SELECT id FROM licenses") if licenses.allowed(one)}
        unlicensed_none = all(one in allowed for one in held[counts.LICENSE]) or (
            facet == counts.LICENSE and value in allowed)
        blocked = None
        if matched == 0:
            blocked = {BELONGS_TO_UPSTREAM: 0, NO_ALLOWED_LICENSE: 0}
        elif self.upstream() is None and unlicensed_none:
            blocked = {BELONGS_TO_UPSTREAM: 0, NO_ALLOWED_LICENSE: 0}
        return {"texts": every, "total": every, "next": None, "held": True, "statuses": statuses, "locked": locked,
                "blocked": blocked,
                "tags": [{"tag": name, "texts": n} for name, n in named[counts.TAG]] if whole and matched
                else [] if matched == 0 else None,
                "licenses": [{"license": name, "texts": n} for name, n in named[counts.LICENSE]] if whole and matched
                else [] if matched == 0 else None}

    def tags(self) -> list[dict]:
        """Every tag a text has, with how many texts have it, by name (wpt_depot.counts)."""
        return self._rows(f"SELECT name AS tag, texts FROM counts WHERE facet = '{counts.TAG}' ORDER BY name")

    def counts(self) -> dict[str, dict[str, int]]:
        """{facet: {name: count}} of the depot's counts of texts
        (wpt_depot.counts.COUNTED): its current texts, and how many of them
        have each status, tag and license, each library holds and each
        decision is on."""
        return counts.read(self.db.reader())

    def recount(self):
        """Make the depot's counts again from its rows (wpt_depot.counts),
        as one change that writes nothing else."""
        with self._write("recount") as (con, _):
            counts.recount(con, known=self.upstream() is None)

    def latest_decisions(self, limit: int = 10) -> list[dict]:
        """The latest decisions, the newest first, each with how many texts
        (wpt_depot.counts) and erasure requests it decides, the library
        whose reported problem it records (None for any other) and
        put_back: whether it put texts back under review
        (wpt_depot.policy.put_back)."""
        found = self._rows(
            f"SELECT d.*, {_DECIDED_TEXTS} AS texts, "
            "(SELECT count(*) FROM erasures_revisions e WHERE e.latest "
            "AND EXISTS (SELECT 1 FROM json_each(e.decisions) WHERE value = d.id)) AS erasures, "
            "(SELECT min(l.id) FROM libraries_revisions l, json_each(coalesce(l.problems, '[]')) p "
            "WHERE l.latest AND p.value ->> 'decision' = d.id) AS library "
            f"FROM decisions d ORDER BY d.made_at DESC, d.id DESC LIMIT {int(limit)}")
        return [{**one, "put_back": policy.put_back(one)} for one in found]

    def decision(self, decision: str) -> dict:
        """{decision: its row with put_back (wpt_depot.policy.put_back),
        texts: how many texts it is on, erasures: the ids of the erasure
        requests it decides, problem: the reported
        problem it records, with its library ({library, texts, problem}),
        or None}. NotFound without such a decision."""
        found = self._rows(f"SELECT d.*, {_DECIDED_TEXTS} AS texts FROM decisions d WHERE d.id = ?", (decision,))
        if not found:
            raise NotFound(f"no decision {decision}")
        row = found[0]
        row["put_back"] = policy.put_back(row)
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
        return _bounds(self.db.reader(), depot, change)

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
        held, params = self._held(library, change)
        bound = f"(SELECT b.value FROM json_each(:bounds) b WHERE b.key = {ids.owner_sql('t.id')})"
        query = (f"SELECT {LISTED_COLUMNS}, {decision_list('t.id', bound)} AS decisions, "
                 f"{policy.linked_status_sql('t.id', bound)} AS status FROM ({held}) h JOIN texts_revisions t "
                 f"ON t.id = h.member AND t.revision = h.revision {CONTENT_JOIN} ORDER BY {ids.order_sql('t.id')}")
        return self._rows(query, params)

    def library_range(self, library: str, change: int, low: int, high: int, limit: int | None = None
                      ) -> Iterator[tuple]:
        """The texts library_texts gives whose export key (wpt_depot.ids)
        is from low to high, both included, in the order of their keys, the
        first limit of them (None: all): (export key, id, revision, kind,
        tags as JSON text, bytes) of each. They are read from the library's
        side, as that range of the index memberships_key (of pins_key for a
        library locked then), which holds every column of the row it needs,
        each row's text revision and content version looked up after it
        (CROSS JOIN fixes that order), so neither the other libraries' rows
        nor the texts outside the range or past the limit are read."""
        if self._locked_then(library, change):
            held, revision = "pins h INDEXED BY pins_key", "h.revision"
            when = ""
            change = self._lock_change(library)
        else:
            held, revision, when = "memberships h INDEXED BY memberships_key", REVISION_THEN, MEMBER_THEN
        cursor = self.db.reader().cursor().execute(
            f"SELECT h.export_key, t.id, t.revision, c.kind, t.tags, c.bytes FROM {held} "
            f"CROSS JOIN texts_revisions t ON t.id = h.text AND t.revision = {revision} {CONTENT_JOIN} "
            f"WHERE h.library = :library AND h.export_key BETWEEN :low AND :high {when}ORDER BY h.export_key"
            + ("" if limit is None else f" LIMIT {int(limit)}"),
            {"library": library, "change": change, "low": low, "high": high,
             "bounds": _json(self.bounds(ids.owner(library), change))})
        yield from cursor

    def library_long(self, library: str, change: int) -> list[tuple]:
        """The plain texts library_texts gives whose content version is
        longer than schema.LONG characters, those a corpus build cuts into
        pieces (wpt_depot.pieces), in no set order: (export key, id,
        revision, tags as JSON text, length, bytes) of each. They are read
        from the side of those content versions, the partial index
        content_lists_long, each version's text looked up in the library by
        its key of memberships (of pins for a library locked then) and kept
        when the revision the library holds reads that version (CROSS JOIN
        fixes that order), so the read grows with the depot's long content
        versions, never with the library's texts."""
        if self._locked_then(library, change):
            held, revision, when = f"pins h INDEXED BY {PINS_KEY}", "h.revision", ""
            change = self._lock_change(library)
        else:
            held, revision, when = f"memberships h INDEXED BY {MEMBERSHIPS_KEY}", REVISION_THEN, MEMBER_THEN
        return self.db.reader().execute(
            "SELECT h.export_key, t.id, t.revision, t.tags, c.length, c.bytes FROM content_lists c INDEXED BY "
            f"content_lists_long CROSS JOIN {held} ON h.library = :library AND h.text = c.id {when}"
            f"CROSS JOIN texts_revisions t ON t.id = h.text AND t.revision = {revision} "
            f"WHERE c.kind IS NULL AND c.length > {LONG} AND t.content_version = c.version",
            {"library": library, "change": change, "bounds": _json(self.bounds(ids.owner(library), change))}
        ).fetchall()

    def library_spread(self, library: str, change: int, points: int) -> list[tuple]:
        """At most points of the texts library_texts gives, as (id,
        revision, kind, tags as JSON text, bytes), spread over the order of
        their export keys (wpt_depot.ids), which spread a library's texts
        evenly: every text of a library that holds at most points of them,
        else the first text at or after each of points evenly spaced keys
        from the library's lowest key to its highest, the same texts for the
        same depot state. Each point is one seek of the index memberships_key
        (pins_key for a library locked then) read as library_range reads it,
        so the read is bounded by points whatever the library's size."""
        table = "pins INDEXED BY pins_key" if self._locked_then(library, change) else (
            "memberships INDEXED BY memberships_key")
        # each end on its own: SQLite reads one end of an index range for a lone min() or max() only
        low, high = (self.db.reader().execute(f"SELECT {end}(export_key) FROM {table} WHERE library = ?",
                                              (library,)).fetchone()[0] for end in ("min", "max"))
        if low is None:
            return []
        first = [tuple(row[1:]) for row in self.library_range(library, change, low, high, limit=points + 1)]
        if len(first) <= points:
            return first
        found, seen = [], set()
        for at in range(points):
            start = low + (high - low) * at // max(points - 1, 1)
            for _, text, revision, kind, tags, size in self.library_range(library, change, start, high, limit=1):
                if text not in seen:
                    seen.add(text)
                    found.append((text, revision, kind, tags, size))
        return found

    def library_count(self, library: str, change: int) -> int:
        """How many texts library_texts gives, counted without reading them:
        read from the depot's counts at once when the library holds now what
        it held at that change (wpt_depot.counts.unmoved), else counted from
        the library's rows of the index memberships_key (of pins_key for a
        library locked then) alone, which reads every row of the library."""
        locked = self._locked_then(library, change)
        con = self.db.reader()
        with con:  # one read transaction: the move and the count are those of one state
            if counts.unmoved(con, library, change):
                found = con.execute("SELECT texts FROM counts WHERE facet = ? AND name = ?",
                                    (counts.LIBRARY, library)).fetchall()
                return found[0][0] if found else 0
            if locked:
                return con.execute("SELECT count(*) FROM pins INDEXED BY pins_key WHERE library = ?",
                                   (library,)).fetchall()[0][0]
            return con.execute("SELECT count(*) FROM memberships INDEXED BY memberships_key WHERE library = ? "
                               "AND joined_change <= ? AND (left_change IS NULL OR left_change > ?)",
                               (library, change, change)).fetchall()[0][0]

    def measure(self, libraries: Sequence[str], tags: Sequence[str] = ()) -> dict:
        """What the libraries hold now, read from the depot's counts
        (wpt_depot.counts, its rows of each library alone, so at once
        whatever their size), each text at its current revision: {texts,
        characters, bytes (UTF-8) of them all; plain, items (conversations
        and tasks, the items a corpus takes) and other (items of the other
        kinds): each {texts, characters, bytes}; shares: for each tag of
        tags, in order, the plain texts that have it, each {tag, texts,
        characters, bytes}; unshared: the plain texts left once the shares
        are taken away, when tags are given (None otherwise)}. Each library
        counts apart and the figures are their sum: a text that several of
        them hold counts once for each. A plain text with several tags of
        tags counts once for each of them, where a dataset's share takes it
        only for the first (wpt_depot.corpus): the shares and unshared are
        exact when each plain text has at most one tag of tags. Read in one
        read transaction, so they are those of one state of the depot.
        NotFound for a library the depot does not hold."""
        libraries, tags = list(dict.fromkeys(libraries)), list(tags)
        con = self.db.reader()
        with con:  # one read transaction: a savepoint opens it and holds its snapshot to the end
            held = {one for (one,) in con.execute(
                "SELECT id FROM libraries WHERE id IN (SELECT value FROM json_each(?))", (_json(libraries),))}
            missing = [one for one in libraries if one not in held]
            if missing:
                raise NotFound(f"no library {', '.join(missing)}")
            return counts.summed([counts.figures(con, library, tags) for library in libraries], tags)

    def change_times(self, changes: Sequence[int]) -> dict[int, str | None]:
        """{change number: the UTC time it was made at (None for a change
        made before format 4)} of each of the depot's changes named that it
        holds."""
        return dict(self.db.reader().execute(
            "SELECT id, made_at FROM changes WHERE id IN (SELECT value FROM json_each(?))",
            (_json([int(one) for one in changes]),)).fetchall())

    def listed_text(self, text: str, revision: int) -> dict:
        """One text at one revision, as library_texts gives a row but with
        neither its decisions nor its status. NotFound when the depot no
        longer holds it (an erasure removed it)."""
        found = self._rows(f"SELECT {LISTED_COLUMNS} FROM texts_revisions t {CONTENT_JOIN} "
                           "WHERE t.id = ? AND t.revision = ?", (text, revision))
        if not found:
            raise NotFound(f"the depot holds no revision {revision} of the text {text}")
        return found[0]

    def _held(self, library: str, change: int) -> tuple[str, dict]:
        """(the query of each text the library held at that change, as
        member and revision; its parameters), as library_texts says."""
        if self._locked_then(library, change):
            held = "SELECT p.text AS member, p.revision FROM pins p WHERE p.library = :library"
            change = self._lock_change(library)
        else:
            held = ("SELECT m.text AS member, (SELECT max(r.revision) FROM texts_revisions r INDEXED BY "
                    f"{TEXTS_KEY} WHERE r.id = m.text "
                    f"AND r.change <= (SELECT b.value FROM json_each(:bounds) b WHERE b.key = {ids.owner_sql('r.id')})"
                    ") AS revision FROM memberships m WHERE m.library = :library "
                    "AND m.joined_change <= :change AND (m.left_change IS NULL OR m.left_change > :change)")
        return held, {"library": library, "change": change, "bounds": _json(self.bounds(ids.owner(library), change))}

    def _locked_then(self, library: str, change: int) -> bool:
        """Whether the library was locked at that change. NotFound without
        such a library, or when it did not exist yet at that change."""
        self._library(library)
        locked = _locked_then(self.db.reader(), library, change)
        if locked is None:
            raise NotFound(f"the library {library} did not exist yet at the change {change}")
        return locked

    def _lock_change(self, library: str) -> int:
        """The change number at which the locked library was locked, in the numbering of the depot that owns it."""
        return _lock_change(self.db.reader(), library)

    def pinned(self, library: str, offset: int = 0, limit: int = 50, after: str | None = None) -> dict:
        """The revisions a locked library pinned, a page at a time: {total:
        how many texts it pinned, newer: how many of them have a current
        revision newer than the one pinned, both read from the depot's
        counts at once (wpt_depot.counts: what a locked library holds, and
        NEWER), rows: limit rows from offset in the numeric order of ids, or
        after the place after (a row's place) when given, each {id, title:
        the pinned revision's, revision: the one pinned, current: the
        text's current revision, status: its status when the library was
        locked, by the decisions made on it up to then (wpt_depot.policy)},
        next: the place of the page's last row when more follow, None
        otherwise}. The rows are read
        in that order from the index pins_order, skipping offset of its
        entries. An open library pins none. NotFound without such a
        library."""
        found_library = self._library(library)
        held = counts.read(self.db.reader())
        pinned = found_library["locked_at"] is not None
        totals = {"total": held[counts.LIBRARY].get(library, 0) if pinned else 0,
                  "newer": held[counts.NEWER].get(library, 0)}
        key = ids.order_sql("p.text")
        page = (f"SELECT p.text, p.revision, {key} AS place FROM pins p WHERE p.library = :library "
                + ("" if after is None else f"AND {key} > :after ")
                + f"ORDER BY {key} LIMIT {int(limit) + 1} OFFSET {int(offset)}")
        query = ("SELECT h.text AS id, t.title, h.revision, c.revision AS current, h.place FROM "
                 f"({page}) h CROSS JOIN texts_revisions t ON t.id = h.text AND t.revision = h.revision "
                 "CROSS JOIN texts_revisions c ON c.id = h.text AND c.latest")
        locked = self._lock_change(library)
        bounds = _json(self.bounds(ids.owner(library), locked)) if locked is not None else "{}"
        bound = f"(SELECT b.value FROM json_each(:bounds) b WHERE b.key = {ids.owner_sql('q.id')})"
        found = self._rows(f"SELECT q.*, {policy.linked_status_sql('q.id', bound)} AS status FROM ({query}) q "
                           "ORDER BY q.place", {"library": library, "bounds": bounds,
                                                **({} if after is None else {"after": after})})
        places = [one.pop("place") for one in found]
        return {**totals, "rows": found[:limit], "next": places[limit - 1] if len(found) > limit else None}

    # Licenses

    def licenses(self) -> list[dict]:
        """Every license the depot holds, by id: its id and name, whether it
        is allowed (wpt_depot.licenses, a combined row only if every part
        is), and how many texts name it (wpt_depot.counts)."""
        named = {one["name"]: one["texts"] for one in self._rows(
            f"SELECT name, texts FROM counts WHERE facet = '{counts.LICENSE}'")}
        return [{"id": one["id"], "name": one["name"], "allowed": licenses.allowed(one["id"]),
                 "texts": named.get(one["id"], 0)} for one in self._rows("SELECT id, name FROM licenses ORDER BY id")]

    def license(self, license_id: str) -> dict:
        """One license: its id, name, official text, whether it is allowed
        (as licenses() says) and how many texts name it. NotFound without
        it."""
        found = self._rows(f"SELECT l.*, coalesce((SELECT c.texts FROM counts c WHERE c.facet = '{counts.LICENSE}' "
                           "AND c.name = l.id), 0) AS texts FROM licenses l WHERE l.id = ?", (license_id,))
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
                                  "tags": _json(list(one.tags)), "address": one.address, "decisions": "[]",
                                  "content_version": 1, "scores": None, "status": UNDER_REVIEW})
            _insert(con, "texts_revisions", new_texts)
            added = [one["id"] for one in new_texts]
            self._choose(con, added)
            self._link(con, decision, number)  # counts the decision's texts
            counts.shift(con, counts.measure(con, "temp.chosen", (counts.TEXTS, counts.STATUS, counts.TAG,
                                                                  counts.LICENSE, counts.SPLIT_TEXTS,
                                                                  counts.SPLIT_TAG, counts.SPLIT_LICENSE)))
            counts.enlist(con, "temp.chosen", counts.LISTED)
            return added

    def blocked(self, texts: Selection, status: str = ADMITTED, after: Sequence | None = None,
                limit: int = 50) -> dict:
        """Of the texts a selection names, those a decision of that status
        cannot cover, a page at a time in the numeric order of their ids:
        {rows: at most limit rows, each {id, reasons: the reasons that
        hold, BELONGS_TO_UPSTREAM (a row the upstream owns, for any
        decision) and NO_ALLOWED_LICENSE (none of its licenses is allowed,
        wpt_depot.licenses, for an admission)}, next: the place where the
        next page starts when more may follow, None otherwise; partial: as
        search says}. The decision refuses, or leaves out, every text listed
        here. In a depot that follows no upstream, none is blocked when the
        decision admits no text or every license a text names is allowed
        (the depot's counts say which licenses texts name): that is answered
        at once. Otherwise its candidates are read as search reads them, from
        the narrowest of the selection's listing and the texts that name a
        license that is not allowed (text_licenses). The decision applies
        the same check. NotFound when a given id is no text; BadQuery for a
        condition that is not one."""
        upstream = self.upstream() is not None
        named = counts.read(self.db.reader())[counts.LICENSE]
        refused = sorted(one for one in named if not licenses.allowed(one))
        if not upstream and (status != ADMITTED or not refused):
            if not isinstance(texts, (str, Filter)):
                self._condition(texts)  # NotFound for a named id that is no text
            return {"rows": [], "next": None, "partial": False}
        theirs, unlicensed = self.upstream_sql(), self._unlicensed_sql("licenses", status)
        plan = self._plan(texts)
        if not upstream and status == ADMITTED:
            option = _listing("SELECT key FROM text_licenses WHERE license {names}", refused, named)
            if option.ordered or option.texts <= SORTED:
                plan = replace(plan, listing=min((plan.listing, option), key=lambda one: one.texts))
        found, edge = self._page(replace(plan, where=f"({plan.where}) AND ({theirs} OR {unlicensed})"),
                                 f"id, {theirs} AS theirs, {unlicensed} AS unlicensed", "id", False, after, False,
                                 limit)
        rows = [{"id": one["id"], "reasons": [reason for reason, holds in (
            (BELONGS_TO_UPSTREAM, one["theirs"]), (NO_ALLOWED_LICENSE, one["unlicensed"])) if holds]}
            for one in found[:limit]]
        more = len(found) > limit
        return {"rows": rows, "next": found[limit - 1]["place"] if more else edge,
                "partial": not more and edge is not None}

    def decide(self, status: str, texts: Selection, declaration: Declaration, reason: str | None = None,
               leave_blocked: bool = False, progress=None,
               leave_same: bool = False) -> tuple[str, int, dict[str, int]]:
        """One decision, admitted, refused or under review, with an optional
        public reason, on every text selected: a row of text_decisions for
        each, and its status by the policy (wpt_depot.policy) stored on its
        current revision, which stays the same revision. A decision under
        review puts the texts back under review, undoing their admission or
        refusal: its reason is the one the depot writes
        (wpt_depot.policy.review_again, the person's reason after it), the
        decisions before it no longer count, and a text under review already
        is left out (ALREADY_UNDER_REVIEW), so it never only wipes a text's
        decisions. A refusal and a putting back under review take the texts
        out of every open library, since an open library holds only admitted
        texts; a locked library keeps reading the revision it pinned, at the
        status its decisions up to the lock make. Its answer:
        (its id, how many texts it is on, {reason: how many texts were left
        out for it}: CHANGED_SINCE for a filter with a change number,
        ALREADY_UNDER_REVIEW, with leave_same ALREADY's reason, and with
        leave_blocked, Depot.blocked's reasons). An admission or a refusal
        with leave_same leaves out the texts whose status is the one it
        gives already (ALREADY), so it records no decision that changes no
        status; without it, every text selected gets the decision. A text a
        decision cannot cover (Depot.blocked) refuses the whole decision, or,
        with leave_blocked, is left out of it. progress, when given, is
        called with the name of each stage as it starts. NotFound when
        nothing is selected (or nothing is left once the blocked texts are
        left out) or a given id is no text; BelongsToUpstream for a text the
        upstream owns and Refused when admitting texts none of whose
        licenses is allowed (wpt_depot.licenses), both without
        leave_blocked."""
        if status not in (ADMITTED, REFUSED, UNDER_REVIEW):
            raise Refused(f"a decision on texts admits them, refuses them or puts them back under review, not "
                          f"{status!r}")
        again = status == UNDER_REVIEW
        if again:
            _check_declaration(declaration, reason)  # the person's own reason, before the depot's form is added
        stage = progress or _quiet
        with self._write("decide") as (con, number):
            stage("selecting the texts")
            changed = {}
            if not self._choose(con, texts, left=changed):
                raise NotFound(_none_left(texts, changed))
            stage("checking the texts")
            left = {**changed, **self._leave_blocked(con, status, leave_blocked)}
            if again:
                con.execute("DELETE FROM temp.chosen WHERE id IN (SELECT c.id FROM temp.chosen c CROSS JOIN "
                            f"texts_revisions t ON t.id = c.id AND t.latest WHERE t.status = '{UNDER_REVIEW}')")
                if con.changes():
                    left[ALREADY_UNDER_REVIEW] = con.changes()
            elif leave_same:
                con.execute("DELETE FROM temp.chosen WHERE id IN (SELECT c.id FROM temp.chosen c CROSS JOIN "
                            "texts_revisions t ON t.id = c.id AND t.latest WHERE t.status = ?)", (status,))
                if con.changes():
                    left[ALREADY[status]] = con.changes()
            decided = con.execute("SELECT count(*) FROM temp.chosen").fetchall()[0][0]
            if not decided:
                if left.get(ALREADY_UNDER_REVIEW) and len(left) == 1:
                    raise NotFound(f"every text the selection takes ({left[ALREADY_UNDER_REVIEW]}) is under review "
                                   "already, so none is put back under review")
                if status in ALREADY and left.get(ALREADY[status]) and len(left) == 1:
                    raise NotFound(f"every text the selection takes ({left[ALREADY[status]]}) is {status} already, so "
                                   "no decision is recorded")
                verb = "put back under review" if again else status
                raise NotFound(f"no text {'left to be ' + verb + ' once ' if left else ''}matches the selection")
            if again:
                decision = self._decision(con, number, status, declaration, policy.review_again(reason),
                                          checked=False)
            else:
                decision = self._decision(con, number, status, declaration, reason)
            stage("recording the decision on each text")
            self._link(con, decision, number)
            stage("storing the statuses")
            self._restatus(con, status, again)
            if status in (REFUSED, UNDER_REVIEW):
                stage("taking the texts out of the open libraries")
                self._leave(con, self._open_libraries(), number)
            stage("committing")
            return decision, decided, left

    @_answers
    def relicense(self, texts: Selection, new: Sequence[str], declaration: Declaration,
                  progress=None, left: dict | None = None) -> tuple[list[tuple[str, int]], int]:
        """Change the licenses of every text selected to the list new, as a
        new revision, their ids kept. Each distinct old list gets one
        decision 'under review' made by declaration, whose reason the depot
        writes (wpt_depot.policy.license_change), made on the texts that
        had that list; the decisions before it no longer count, so each text
        is under review again and leaves every open library (a locked library
        keeps reading the revision it pinned). A text that names the licenses
        of new already is left out, so a change never only wipes its
        decisions. ([(each decision's id, how many texts it is on)], in the
        numeric order of the first text of each old list, how many texts
        were left out); nothing is
        written when every text is left out. progress as for decide; left,
        when given, receives CHANGED_SINCE as _choose says. NotFound when
        nothing is selected or a given id is no text; Refused
        when new is empty or names a license the depot does not hold."""
        new = list(dict.fromkeys(new))
        if not new:
            raise Refused("a text names at least one license")
        stage = progress or _quiet
        with self._write("relicense") as (con, number):
            stage("selecting the texts")
            changed = {}
            chosen = self._choose(con, texts, left=changed)
            if left is not None:
                left.update(changed)
            if not chosen:
                raise NotFound(_none_left(texts, changed))
            stage("checking the texts")
            self._leave_blocked(con, UNDER_REVIEW, False)
            held_licenses = {one for (one,) in con.execute("SELECT id FROM licenses")}
            unknown = [one for one in new if one not in held_licenses]
            if unknown:
                raise Refused(f"the depot holds no license {', '.join(unknown)}")
            lists = [old for (old,) in con.execute(
                "SELECT t.licenses FROM temp.chosen c CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest "
                f"GROUP BY t.licenses ORDER BY min({ids.order_sql('t.id')})")]
            same = [old for old in lists if set(json.loads(old)) == set(new)]
            con.execute("CREATE TEMP TABLE IF NOT EXISTS skipped (id TEXT PRIMARY KEY)")
            con.execute("DELETE FROM temp.skipped")
            con.execute("INSERT INTO temp.skipped SELECT c.id FROM temp.chosen c CROSS JOIN texts_revisions t "
                        "ON t.id = c.id AND t.latest WHERE t.licenses IN (SELECT value FROM json_each(?))",
                        (_json(same),))
            skipped = con.execute("SELECT count(*) FROM temp.skipped").fetchall()[0][0]
            con.execute("DELETE FROM temp.chosen WHERE id IN (SELECT id FROM temp.skipped)")
            changed_lists = [old for old in lists if old not in same]
            if not changed_lists:
                raise _Nothing(([], skipped))
            stage("writing the new revisions")
            changed = []
            with counts.changing(con, "temp.chosen", (counts.STATUS, counts.LICENSE, counts.NEWER, *counts.SPLIT)):
                for old in changed_lists:
                    decision = self._decision(con, number, UNDER_REVIEW, declaration,
                                              policy.license_change(json.loads(old), new), checked=False)
                    con.execute("CREATE TEMP TABLE IF NOT EXISTS grouped (id TEXT PRIMARY KEY)")
                    con.execute("DELETE FROM temp.grouped")
                    con.execute("INSERT INTO temp.grouped SELECT c.id FROM temp.chosen c CROSS JOIN texts_revisions t "
                                "ON t.id = c.id AND t.latest WHERE t.licenses = ?", (old,))
                    self._revise(con, "texts_revisions", "SELECT id FROM temp.grouped", number,
                                 {"licenses": ":licenses", "status": f"'{UNDER_REVIEW}'"}, {"licenses": _json(new)})
                    self._link(con, decision, number, "grouped")
                    changed.append((decision, con.execute("SELECT count(*) FROM temp.grouped").fetchall()[0][0]))
            stage("taking the texts out of the open libraries")
            self._leave(con, self._open_libraries(), number)
            stage("committing")
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
            erased = self._erased_by_text().get(text, [])
            if row["kind"] is None:
                if not isinstance(plain, (str, FileText)) or content is not None:
                    raise Refused("a plain text's new content is its text, with no item content")
                if not _new_plain(text, plain, [(one["hash"], one["length"]) for one in row["parts"]], erased):
                    raise _Nothing(None)
            else:
                if not isinstance(content, str) or plain is not None:
                    raise Refused("an item's new content is its content as JSON text, with no plain text")
                schema = self._one("SELECT schema FROM kinds WHERE id = ?", (row["kind"],))
                kinds.check_content(row["kind"], schema, content)
                if _same_json(content, self.content(text, row["content_version"])):
                    raise _Nothing(None)
                _refuse_erased(text, erasure.strings_of(json.loads(content)), erased)
            version = self._one("SELECT max(version) FROM content_lists WHERE id = ?", (text,)) + 1
            decision = self._decision(con, number, UNDER_REVIEW, declaration,
                                      policy.content_change(row["content_version"], version), checked=False)
            self._store(con, number, [{"id": text, "version": version, "kind": row["kind"],
                                       "source": plain if row["kind"] is None else content}])
            self._choose(con, [text])
            with counts.changing(con, "temp.chosen", (counts.STATUS, *counts.SIZED, counts.NEWER, *counts.SPLIT)):
                self._revise(con, "texts_revisions", [text], number,
                             {"content_version": ":version", "status": f"'{UNDER_REVIEW}'"}, {"version": version})
            self._leave(con, self._open_libraries(), number)
            self._link(con, decision, number)
            return decision, version

    def change_content_many(self, changes: str | Path | list[dict], declaration: Declaration) -> rewriting.Rewritten:
        """Change the content of many plain texts at once, each as
        change_content changes one: a new content version and a new
        revision, its id kept; the decisions before it no longer count, so
        each text is under review again and leaves every open library (a
        locked library keeps reading the content version of the revision it
        pinned). changes is a JSON-lines file, or the lines as objects (a
        web request's): one {id, text, was} per text, text its new plain
        text, was (optional) the content version its writer read: the text
        changes only while that is still its current one. The change is
        recorded as one decision 'under review' made by declaration for
        each pair of old and new content versions, on the texts that pair
        moves, whose reason the depot writes as for one text
        (policy.content_change). A text whose content is that already, or
        whose was no longer holds, is left out. The lines go into a
        temporary table a few at a time, every check is a query over it,
        and the texts are read back and stored one at a time
        (wpt_depot.rewriting), so the memory it takes is that of one text
        whatever the number of texts. NotFound for a text the depot does
        not hold; BelongsToUpstream for a text its upstream owns; Refused
        for changes that are not JSON lines of that shape, more than
        rewriting.MOST lines, a text named twice, an item, a new content
        that holds a passage an erasure removed from its text, or when no
        text would change."""
        with self._write("content") as (con, number):
            try:
                rewriting.open_changes(con, changes)
                total, theirs = rewriting.check(con, self.upstream_sql(), NAMED)
                if total:
                    self._refuse_upstream("texts", theirs, total=total)
                erased = self._erased_by_text()
                con.execute("CREATE TEMP TABLE rewritten (id TEXT PRIMARY KEY, old INTEGER NOT NULL, "
                            "new INTEGER NOT NULL)")
                unchanged = stale = 0
                for text, plain, was, version, parts, last in rewriting.current(con):
                    if was is not None and was != version:
                        stale += 1
                    elif not _new_plain(text, plain, parts, erased.get(text, [])):
                        unchanged += 1
                    else:
                        self._store(con, number, [{"id": text, "version": last + 1, "kind": None, "source": plain}])
                        con.execute("INSERT INTO temp.rewritten VALUES (?, ?, ?)", (text, version, last + 1))
                stale_ids = [one for (one,) in con.execute(
                    "SELECT c.id FROM temp.content_changes c JOIN temp.content_now n ON n.id = c.id "
                    f"WHERE c.was IS NOT NULL AND c.was != n.version ORDER BY {ids.order_sql('c.id')} LIMIT {NAMED}")]
                pairs = con.execute(f"SELECT old, new FROM temp.rewritten GROUP BY old, new "
                                    f"ORDER BY min({ids.order_sql('id')})").fetchall()
                if not pairs:
                    raise Refused(f"no text named changes: {unchanged:,} have that content already, and {stale:,} "
                                  "a content version other than the one their line's was names")
                con.execute("CREATE TEMP TABLE rewritten_ids (id TEXT PRIMARY KEY)")
                con.execute("INSERT INTO temp.rewritten_ids SELECT id FROM temp.rewritten")
                con.execute("CREATE TEMP TABLE rewritten_group (id TEXT PRIMARY KEY)")
                decisions = []
                with counts.changing(con, "temp.rewritten_ids",
                                     (counts.STATUS, *counts.SIZED, counts.NEWER, *counts.SPLIT)):
                    for old, new in pairs:
                        reason = policy.content_change(old, new)
                        decision = self._decision(con, number, UNDER_REVIEW, declaration, reason, checked=False)
                        con.execute("DELETE FROM temp.rewritten_group")
                        con.execute("INSERT INTO temp.rewritten_group SELECT id FROM temp.rewritten "
                                    "WHERE old = ? AND new = ?", (old, new))
                        self._revise(con, "texts_revisions", "SELECT id FROM temp.rewritten_group", number,
                                     {"content_version": ":version", "status": f"'{UNDER_REVIEW}'"},
                                     {"version": new})
                        self._link(con, decision, number, "rewritten_group")
                        decisions.append((decision, reason, con.execute(
                            "SELECT count(*) FROM temp.rewritten_group").fetchall()[0][0]))
                self._leave(con, self._open_libraries(), number, "rewritten_ids")
                changed = sum(count for _, _, count in decisions)
            finally:
                rewriting.close(con)
            return rewriting.Rewritten(decisions, changed, unchanged, stale, stale_ids)

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
            self._choose(con, [text])
            with counts.changing(con, "temp.chosen", _credited(values)):
                self._revise(con, "texts_revisions", [text], number, {name: f":{name}" for name in values},
                             {name: _json(value) if name in JSON_COLUMNS else value for name, value in values.items()})
            self._link(con, decision, number)
            return decision, reason

    def edit_credits_many(self, changes: str | Path | list[dict],
                          declaration: Declaration) -> tuple[str, str, int, int]:
        """Change the credits of many texts at once, each as a new revision,
        its id kept. changes is a JSON-lines file, or the lines as objects
        (a web request's): one {id, title, authors, tags, address, notice}
        per text, each credit optional, as edit_credits takes them (a credit
        given replaces the text's, null clears it, one left out stays). The
        change is recorded as one decision 'under review' made by
        declaration, on every text it changes, whose reason the depot writes
        naming the credits that changed (policy.credits_change); the
        decisions before it keep counting, so each text keeps its status and
        its libraries, and a locked library keeps reading the revision it
        pinned. A line may name, under "was", the values its writer read of
        some credits: the text changes only while they still hold. A text
        whose credits would stay as they are, or whose was no longer holds,
        is left out. The lines go into a temporary table a few at a time and
        every check is a query over it (wpt_depot.crediting), so the memory
        it takes stays the same whatever the number of texts. (the
        decision's id, its
        reason, how many texts changed, how many were left out). NotFound
        for a text the depot does not hold; BelongsToUpstream for a text its
        upstream owns; Refused for changes that are not JSON lines of that
        shape, more than crediting.MOST lines, a text named twice, or when
        no text would change."""
        with self._write("credits") as (con, number):
            try:
                crediting.open_changes(con, changes)
                total, theirs = crediting.check(con, self.upstream_sql(), NAMED)
                if total:
                    self._refuse_upstream("texts", theirs, total=total)
                made = {}
                changed, left_out, names = crediting.credit(con, lambda text: self._notice(con, number, text, made))
                if not changed:
                    raise Refused("no text named changes: each one's credits are those already, or no longer the "
                                  "values its line's was names")
                reason = policy.credits_change(names)
                decision = self._decision(con, number, UNDER_REVIEW, declaration, reason, checked=False)
                con.execute("CREATE TEMP TABLE credited_ids (id TEXT PRIMARY KEY)")
                con.execute("INSERT INTO temp.credited_ids SELECT id FROM temp.credited")
                with counts.changing(con, "temp.credited_ids", _credited(names)):
                    self._revise(con, "texts_revisions", "SELECT id FROM temp.credited_ids", number,
                                 {name: f"s.{name}" for name in CREDITS},
                                 joined="JOIN temp.credited s ON s.id = t.id")
                self._link(con, decision, number, "credited_ids")
            finally:
                crediting.close(con)
            return decision, reason, changed, left_out

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
                with counts.changing(con, "temp.scored_ids", (counts.SCORE, counts.NEWER)):
                    self._revise(con, "texts_revisions", "SELECT id FROM temp.scored_ids", number,
                                 {"scores": "(SELECT s.scores FROM temp.scored s WHERE s.id = t.id)"})
                self._link(con, decision, number, "scored_ids")
            finally:
                scoring.close(con)
            return decision, reason, changed, left_out

    @_answers
    def score(self, texts: Selection, name: str, value: float | None, declaration: Declaration, progress=None,
              left: dict | None = None) -> tuple[str | None, int, int]:
        """Set the score name to value on every text selected, made from
        each text's current content version, or, with value None, remove
        the score name from each, as edit_scores records it: one decision
        under review made by declaration on every text it changes, whose
        reason the depot writes (policy.scores_change); the texts keep their
        status and libraries. A text whose scores would stay as they are is
        left out. (the decision's id, None when no text changes and nothing
        is written; how many texts changed; how many were left out).
        progress as for decide; left, when given, receives CHANGED_SINCE as
        _choose says. NotFound when nothing is selected or a given id is no
        text; BelongsToUpstream when a text is the upstream's; Refused for
        an empty name or a value that is not a finite number."""
        name = name.strip()
        if not name:
            raise Refused("a score has a name")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise Refused(f"a score's value is a finite number, not {value!r}")
        stage = progress or _quiet
        with self._write("scores") as (con, number):
            try:
                stage("selecting the texts")
                changed = {}
                chosen = self._choose(con, texts, left=changed)
                if left is not None:
                    left.update(changed)
                if not chosen:
                    raise NotFound(_none_left(texts, changed))
                stage("checking the texts")
                con.execute("CREATE TEMP TABLE changes (id TEXT NOT NULL, scores TEXT NOT NULL, remove TEXT NOT NULL)")
                if value is None:
                    con.execute("INSERT INTO temp.changes SELECT id, '[]', json_array(:name) FROM temp.chosen",
                                {"name": name})
                else:
                    con.execute("INSERT INTO temp.changes SELECT c.id, json_array(json_object('name', :name, 'value', "
                                ":value, 'content_version', t.content_version)), '[]' FROM temp.chosen c "
                                "CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest",
                                {"name": name, "value": float(value)})
                con.execute("CREATE INDEX temp.changes_id ON changes (id)")
                total, theirs = scoring.check(con, self.upstream_sql(), NAMED)
                if total:
                    self._refuse_upstream("texts", theirs, total=total)
                stage("writing the new revisions")
                count, left_out, set_names, removed = scoring.score(con)
                if not count:
                    raise _Nothing((None, 0, left_out))
                decision = self._decision(con, number, UNDER_REVIEW, declaration,
                                          policy.scores_change(set_names, removed), checked=False)
                con.execute("CREATE TEMP TABLE scored_ids (id TEXT PRIMARY KEY)")
                con.execute("INSERT INTO temp.scored_ids SELECT id FROM temp.scored")
                with counts.changing(con, "temp.scored_ids", (counts.SCORE, counts.NEWER)):
                    self._revise(con, "texts_revisions", "SELECT id FROM temp.scored_ids", number,
                                 {"scores": "(SELECT s.scores FROM temp.scored s WHERE s.id = t.id)"})
                self._link(con, decision, number, "scored_ids")
                stage("committing")
            finally:
                scoring.close(con)
            return decision, count, left_out

    def score_names(self) -> list[dict]:
        """Every name of a score a text has now, with how many texts have
        it, by name: [{name, texts}], read from the depot's counts
        (wpt_depot.counts) at once whatever the depot's size."""
        return self._rows(f"SELECT name, texts FROM counts WHERE facet = '{counts.SCORE}' ORDER BY name")

    def score_values(self, name: str) -> dict:
        """The values the score name has on the texts changed last, which
        a review offers as its choices: {name, read, texts, low, high,
        common}. It goes over the SCORE_READ revisions of the latest
        changes at most, newest first (a score set makes a revision, so
        the texts scored last come first), whatever the depot's size; read
        is how many it went over, texts how many of them are current and
        have the score, low and high its smallest and largest value there
        (null without one), common its SCORE_VALUES most frequent values,
        [{value, texts}], by value."""
        _filled(name=name)
        found = self.db.query(
            "WITH walked AS (SELECT latest, scores FROM texts_revisions ORDER BY change DESC LIMIT ?), "
            "seen AS (SELECT CAST(s.value ->> 'value' AS REAL) AS value FROM walked, json_each(walked.scores) s "
            "WHERE walked.latest AND walked.scores IS NOT NULL AND s.value ->> 'name' = ?) "
            "SELECT (SELECT count(*) FROM walked) AS read, (SELECT count(*) FROM seen) AS texts, "
            "(SELECT min(value) FROM seen) AS low, (SELECT max(value) FROM seen) AS high, "
            "(SELECT json_group_array(json_object('value', value, 'texts', n)) FROM (SELECT value, n FROM "
            "(SELECT value, count(*) AS n FROM seen GROUP BY value ORDER BY n DESC, value LIMIT ?) ORDER BY value)) "
            "AS common", (SCORE_READ, name.strip(), SCORE_VALUES))[0]
        return {"name": name.strip(), "read": found["read"], "texts": found["texts"], "low": found["low"],
                "high": found["high"], "common": json.loads(found["common"])}

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
    def put(self, library: str, texts: Selection, progress=None, left: dict | None = None) -> tuple[int, int]:
        """Put into an open library every admitted text selected: (how many
        joined it, how many were selected and left out because they are not
        admitted). progress as for decide; left, when given, receives
        CHANGED_SINCE as _choose says. Locked when the library is locked;
        NotFound when a given id is no text."""
        stage = progress or _quiet
        with self._write("put") as (con, number):
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Locked(f"the library {library} is locked: no text joins it")
            stage("selecting the texts")
            if not self._choose(con, texts, left=left):
                raise _Nothing((0, 0))
            stage("putting the admitted texts in the library")
            con.execute("INSERT INTO memberships (library, text, joined_change, left_change, export_key) "
                        "SELECT :library, c.id, :change, NULL, export_key(c.id) FROM temp.chosen c CROSS JOIN "
                        f"texts_revisions t ON t.id = c.id AND t.latest WHERE t.status = '{ADMITTED}' AND NOT EXISTS "
                        f"(SELECT 1 FROM memberships m INDEXED BY {MEMBERSHIPS_KEY} WHERE m.library = :library "
                        "AND m.text = c.id AND m.left_change IS NULL)", {"library": library, "change": number})
            joined = con.changes()
            left_out = con.execute("SELECT count(*) FROM temp.chosen c CROSS JOIN texts_revisions t ON t.id = c.id "
                                   f"AND t.latest WHERE t.status <> '{ADMITTED}'").fetchall()[0][0]
            if not joined:
                raise _Nothing((0, left_out))
            counts.shift(con, counts.held(con, "SELECT m.library, m.text FROM temp.chosen c CROSS JOIN memberships m "
                                               f"INDEXED BY {MEMBERSHIPS_KEY} ON m.library = ? AND m.text = c.id "
                                               "AND m.joined_change = ?", (library, number)))
            stage("committing")
            return joined, left_out

    @_answers
    def take_out(self, library: str, texts: Selection, progress=None, left: dict | None = None) -> tuple[int, int]:
        """Take out of an open library every text selected that it holds,
        each staying as it is, admitted included: (how many left it, how
        many were selected that it did not hold). progress and left as for
        put. Locked when the library is locked; NotFound when a given id is
        no text."""
        stage = progress or _quiet
        with self._write("take out") as (con, number):
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Locked(f"the library {library} is locked: no text leaves it")
            stage("selecting the texts")
            chosen = self._choose(con, texts, left=left)
            if not chosen:
                raise _Nothing((0, 0))
            stage("taking the texts out of the library")
            rows = (f"FROM temp.chosen c CROSS JOIN memberships m INDEXED BY {MEMBERSHIPS_KEY} ON m.library = ? "
                    "AND m.text = c.id AND m.left_change IS NULL")
            leaving = counts.held(con, f"SELECT m.library, m.text {rows}", (library,))
            con.execute(f"UPDATE memberships SET left_change = ? WHERE rowid IN (SELECT m.rowid {rows})",
                        (number, library))
            left = con.changes()
            if not left:
                raise _Nothing((0, chosen))
            counts.shift(con, {key: -by for key, by in leaving.items()})
            stage("committing")
            return left, chosen - left

    def lock(self, library: str):
        """Lock an open library for good, as a new revision: it pins the
        current revision of each text it holds. Refused when it is locked
        already."""
        with self._write("lock") as (con, number):
            found = self._library(library)
            self._refuse_upstream("libraries", [library], FORK)
            if found["locked_at"] is not None:
                raise Refused(f"the library {library} is locked already")
            # its texts come to be held by a locked library, which the counts of SPLIT tell apart
            con.execute("CREATE TEMP TABLE IF NOT EXISTS chosen (id TEXT PRIMARY KEY)")
            con.execute("DELETE FROM temp.chosen")
            con.execute("INSERT OR IGNORE INTO temp.chosen (id) SELECT m.text FROM memberships m INDEXED BY "
                        "memberships_order WHERE m.library = ? AND m.left_change IS NULL", (library,))
            with counts.changing(con, "temp.chosen", counts.SPLIT):
                con.execute("INSERT INTO pins (library, text, revision, change, export_key) "
                            "SELECT m.library, t.id, t.revision, ?, m.export_key "
                            "FROM memberships m JOIN texts_revisions t ON t.id = m.text AND t.latest "
                            "WHERE m.library = ? AND m.left_change IS NULL", (number, library))
                self._revise(con, "libraries_revisions", [library], number, {"locked_at": ":locked_at"},
                             {"locked_at": _now()})

    def fork(self, library: str, name: str, without: Selection | None = None, progress=None,
             left: dict | None = None) -> str:
        """A new open library holding the texts library holds that are
        admitted now (only an admitted text joins a library), less those
        selected by without when it is given: its id. It starts with no
        problem. progress and left as for put (a text left out of without
        stays in the fork). NotFound when a given id is no text."""
        _filled(name=name)
        stage = progress or _quiet
        with self._write("library") as (con, number):
            self._library(library)
            if without is not None:
                stage("selecting the texts left out")
                self._choose(con, without, left=left)
            stage("putting the texts in the fork")
            fork = self._id(self._take(con, "libraries"))
            _insert(con, "libraries_revisions", [{"id": fork, "revision": 1, "change": number, "latest": 1,
                                                  "name": name, "forked_from": library, "problems": "[]",
                                                  "locked_at": None}])
            con.execute("INSERT INTO memberships (library, text, joined_change, left_change, export_key) "
                        "SELECT :fork, m.text, :change, NULL, m.export_key FROM memberships m JOIN texts_revisions t "
                        "ON t.id = m.text AND t.latest WHERE m.library = :library AND m.left_change IS NULL "
                        f"AND t.status = '{ADMITTED}'"
                        + (" AND m.text NOT IN (SELECT id FROM temp.chosen)" if without is not None else ""),
                        {"fork": fork, "change": number, "library": library})
            counts.shift(con, counts.held(con, "SELECT library, text FROM memberships WHERE library = ?", (fork,)))
            stage("committing")
            return fork

    def report(self, library: str, problem: str, texts: Selection | None,
               declaration: Declaration, progress=None, left: dict | None = None) -> tuple[str, list[str]]:
        """A problem with the library, locked or not, as a new revision of
        it: the problem in words, and the texts of the library selected (none
        without a selection: the problem concerns the library as a whole).
        The reporter's declaration is recorded on a decision under review,
        which the problem names. (its id, the texts' ids, in the numeric
        order of ids). NotFound without such a library, when a condition
        matches no text the library holds, or a given id is no text;
        Refused when a given id is a text the library does not hold.
        Membership and texts stay as they are. progress and left as for
        put."""
        _filled(problem=problem)
        stage = progress or _quiet
        with self._write("report") as (con, number):
            self._library(library)
            self._refuse_upstream("libraries", [library])
            where, texts = texts, []
            if where is not None:
                stage("selecting the texts")
                chosen = self._choose(con, where, left=left)
                texts = [one for (one,) in con.execute(
                    f"SELECT c.id FROM temp.chosen c WHERE EXISTS (SELECT 1 FROM memberships m INDEXED BY "
                    f"{MEMBERSHIPS_KEY} WHERE m.library = ? AND m.text = c.id AND m.left_change IS NULL) "
                    f"ORDER BY {ids.order_sql('c.id')}", (library,))]
                if not isinstance(where, (str, Filter)) and len(texts) < chosen:
                    outside = [one for (one,) in con.execute(
                        "SELECT c.id FROM temp.chosen c WHERE NOT EXISTS (SELECT 1 FROM memberships m "
                        f"INDEXED BY {MEMBERSHIPS_KEY} WHERE m.library = ? AND m.text = c.id AND m.left_change IS NULL) "
                        f"ORDER BY {ids.order_sql('c.id')}", (library,))]
                    raise Refused(f"{len(outside)} of the texts named are not in the library {library} (the first: "
                                  f"{outside[0]})")
                if not texts:
                    raise NotFound(f"no text of the library {library} matches the selection"
                                   if isinstance(where, (str, Filter)) else "no text is named")
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
        if write:  # a cut version may be the one a text reads now, which changes what its libraries hold
            self._choose(write[0], sorted(cut), "cut")
            sized = counts.measure(write[0], "temp.cut", counts.SIZED)
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
            now = counts.measure(con, "temp.cut", counts.SIZED)
            now.subtract(sized)
            counts.shift(con, now)
            _drop_unlisted(con, sorted(cut))
            if removed:
                self._choose(con, removed)
                gone = counts.measure(con, "temp.chosen", counts.FACETS)
                counts.shift(con, {key: -count for key, count in gone.items()})
                counts.unlist(con, "temp.chosen", counts.LISTED)
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
                    shares: list[tuple[str, float | None]] | None = None, max_tokens: int | None = None) -> str:
        """A dataset reading the libraries (depot address, None for this
        depot, and library id), each at its depot's change number now, with
        optional ordered shares (tag, share of the tokens, None for an even
        part of what the others leave: spread_shares) and an optional
        max_tokens, the size a build form starts at, whose vocabulary is
        trained by the depot's tokenizer recipe named tokenizer: its id. It
        publishes no vocabulary until an edit sets one. The recipe need not exist
        yet, since a recipe trains on a dataset that exists."""
        return self._new_dataset({"name": name, "tokenizer": tokenizer, "libraries": self._refs(libraries),
                                  "shares": spread_shares(shares),
                                  "max_tokens": max_tokens})

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
                     shares: list[tuple[str, float | None]] | None = None, max_tokens: int | None = None,
                     vocabulary: str | None = None, revision: int | None = None):
        """Change the dataset of that id as a new revision, its id and
        tokenizer kept: its name, the libraries it reads (depot address, None
        for this depot, and library id), its optional ordered shares (tag,
        share of the tokens, None as in new_dataset), its optional max_tokens and the vocabulary it
        publishes (the sha256 of one of this depot's vocabularies, whatever
        depots its libraries are of; None for none), with the checks of
        new_dataset; Refused for a vocabulary the depot does not hold. A library it
        read already (the same library, its depot's address in any form, or
        either address of a library read here, reads_here) is kept exactly
        as stored, its address and change number included, its
        depot never opened, so a dataset made from a remote whose depots this
        machine cannot reach still changes; a library new to it is read at
        its depot's change number now. A corpus built before keeps its own
        copy of the dataset's row. With revision, the revision the change was
        made from: Refused when the dataset is at another one, so a change
        made from an older reading never undoes a later one. NotFound without
        such a dataset."""
        old = self.dataset_by_id(dataset)
        self._refuse_upstream("datasets", [dataset])
        kept = {self._read_from(one["depot"], one["library"]): one for one in old["libraries"]}
        refs = []
        for address, library in libraries:
            stored = kept.get(self._read_from(address, library))
            refs.append(dict(stored) if stored is not None else self._refs([(address, library)])[0])
        values = {"name": name, "libraries": refs, "shares": spread_shares(shares),
                  "max_tokens": max_tokens}
        _checked_dataset({**old, **values})
        with self._write("edit dataset") as (con, number):
            now = con.execute("SELECT revision FROM datasets WHERE id = ?", (dataset,)).fetchone()[0]
            if revision is not None and now != revision:
                raise Refused(f"the dataset {old['name']} changed after it was read: it is at revision {now}, and "
                              f"this change was made from revision {revision}; read it again and redo the change")
            if self._rows("SELECT id FROM datasets WHERE name = ? AND id <> ?", (name, dataset)):
                raise Refused(f"the dataset name {name} is taken")
            if vocabulary is not None and not con.execute("SELECT 1 FROM vocabularies WHERE sha256 = ?",
                                                          (vocabulary,)).fetchall():
                raise Refused(f"the depot holds no vocabulary of sha256 {vocabulary!r}: a dataset publishes one of "
                              "the depot's trained vocabularies")
            self._revise(con, "datasets_revisions", [dataset], number,
                         {"name": ":name", "libraries": ":libraries", "shares": ":shares",
                          "max_tokens": ":max_tokens", "vocabulary": ":vocabulary", "figures": ":figures"},
                         {"name": name, "libraries": _json(refs), "shares": _json(values["shares"]),
                          "max_tokens": max_tokens, "vocabulary": vocabulary,
                          "figures": self._figures(con, number, refs, values["shares"])})

    @_answers
    def read_again(self, dataset: str, revision: int | None = None) -> int:
        """A new revision of the dataset of that id, the same as its current
        one except that every library this depot holds and reads (_read_here)
        is read at the change number now, as _refs reads a library new to a
        dataset; a library locked then is read as its lock leaves it, as
        every read of a library at a change is (_locked_then). A library of
        another depot keeps its change number, its depot never opened. The
        revision stores the figures of what it reads (_figures). The current
        row is read inside the write, so a change committed before it is
        kept. Returns the number of libraries whose change number changed.
        It writes nothing and returns 0 when none did, or when the depot's
        latest change is this dataset's own read again, after which nothing
        moved. With revision, the revision the change was
        made from: Refused when the dataset is at another one. NotFound
        without such a dataset."""
        with self._write("read dataset again") as (con, number):
            old = self.dataset_by_id(dataset)
            self._refuse_upstream("datasets", [dataset])
            if revision is not None and old["revision"] != revision:
                raise Refused(f"the dataset {old['name']} changed after it was read: it is at revision "
                              f"{old['revision']}, and this change was made from revision {revision}; read it again "
                              "and redo the change")
            # the latest change is this dataset's own read again: nothing moved since, so a read now is the same
            if con.execute("SELECT 1 FROM changes c JOIN datasets_revisions r ON r.change = c.id WHERE c.id = ? "
                           "AND c.kind = 'read dataset again' AND r.id = ? AND r.latest",
                           (number - 1, dataset)).fetchall():
                raise _Nothing(0)
            refs = [self._read_here(con, number - 1, one["depot"], one["library"]) or dict(one)
                    for one in old["libraries"]]
            moved = sum(ref != one for ref, one in zip(refs, old["libraries"], strict=True))
            if not moved:
                raise _Nothing(0)
            self._revise(con, "datasets_revisions", [dataset], number,
                         {"libraries": ":libraries", "figures": ":figures"},
                         {"libraries": _json(refs), "figures": self._figures(con, number, refs, old["shares"])})
        return moved

    def read_again_shows(self, dataset: str) -> bool:
        """Whether reading the dataset of that id again (read_again) changes
        what its current revision says it reads, where the dataset's page
        shows it: a library this depot owns and reads whose stored figures
        are not known (dataset_figures) while a read now would know them
        (_figures), or that holds another number of texts now, or another
        library this depot holds and reads (in a clone, the upstream's)
        that a pull moved past the change it is read at. False for a dataset the upstream owns, which read_again refuses.
        NotFound without such a dataset."""
        row = self.dataset_by_id(dataset)
        if self.upstream_owned([dataset]):
            return False
        found = self.dataset_figures(dataset, row["revision"])
        con = self.db.reader()
        for ref, one in zip(row["libraries"], found["libraries"], strict=True):
            now = self._read_here(con, self.change(), ref["depot"], ref["library"])
            if now is None or now == ref:
                continue
            if ids.owner(ref["library"]) != self.name():
                return True
            if one["stored"] is None:
                if json.loads(self._figures(con, now["change"], [now], row["shares"]))["libraries"][0] is not None:
                    return True
            elif counts.summed([one["now"]], found["tags"])["texts"] != counts.summed([one["stored"]],
                                                                                    found["tags"])["texts"]:
                return True
        return False

    def _read_here(self, con, change: int, address: str | None, library: str) -> dict | None:
        """A library a dataset names (its depot's address, None for this
        depot, and its id) as a dataset stores it when read at this depot's
        change number change, read through con: when this depot holds it and
        reads it (reads_here), no address, which means the depot that holds
        the row, so the row reads the same in a clone, and the number of
        what this depot held of its owner then (bounds), so a library of the
        upstream is read by the upstream's numbers; None for a library read
        elsewhere."""
        if not (self.reads_here(address, library)
                and con.execute("SELECT 1 FROM libraries WHERE id = ?", (library,)).fetchall()):
            return None
        return {"depot": None, "library": library, "change": _bounds(con, self.name(), change)[ids.owner(library)]}

    def _refs(self, libraries: list[tuple[str | None, str]]) -> list[dict]:
        """Each library (depot address, None for this depot, and library id)
        as a dataset names it, at a change number now of the depot that owns
        it. A library this depot holds and reads (reads_here: named by no
        address or this depot's, or a library of its upstream named by the
        upstream's web address once a pull brought it) is read as _read_here
        stores it, at the change number now. A library at a web address (an upstream's library
        no pull brought yet, or another depot's) is asked of that depot's web
        app (wpt_depot.follow.library_change); one at another file, of that
        file. NotFound for a library its depot does not hold."""
        refs = []
        con = self.db.reader()
        for address, library in libraries:
            place = self._place(address)
            here = self._read_here(con, self.change(), address, library)
            if here is not None:
                refs.append(here)
            elif place is None:
                self._library(library)  # NotFound: this depot does not hold it
            elif isinstance(place, str) and place.startswith(("http://", "https://")):
                from .follow import library_change
                refs.append({"depot": place, "library": library, "change": library_change(place, library)})
            else:
                depot = Depot.view(address)
                depot._library(library)
                refs.append({"depot": depot.address, "library": library, "change": depot.change()})
        return refs

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
                                                 "shares": _json(row["shares"]),
                                                 "max_tokens": row.get("max_tokens"),
                                                 "figures": self._figures(con, number, row["libraries"],
                                                                          row["shares"])}])
            return dataset

    def _figures(self, con, number: int, refs: list[dict], shares: list[dict] | None) -> str:
        """The figures (wpt_depot.schema) of a revision of a dataset that
        reads the libraries refs with shares, written at the change number,
        as JSON text: what each library held at the change it is read at,
        copied from the counts of that library (wpt_depot.counts.figures)
        when they prove it exact (counts.unmoved at that change, or, for a
        library locked then, whose pins it reads, at the change that locked
        it); None for a library they do not prove, and for one this depot
        does not own (another depot's, or in a clone its upstream's), whose
        change is of another numbering."""
        tags = [one["tag"] for one in shares or []]
        found = []
        for ref in refs:
            library, change = ref["library"], ref["change"]
            if ref["depot"] is not None or ids.owner(library) != self.name():
                found.append(None)
                continue
            if _locked_then(con, library, change):
                change = _lock_change(con, library)
            found.append(counts.figures(con, library, tags) if counts.unmoved(con, library, change) else None)
        return _json({"at": number, "libraries": found})

    def dataset_figures(self, dataset: str, revision: int | None = None) -> dict:
        """What each library a revision of the dataset of that id reads (its
        current revision by default) held at the change it reads it at, as
        the revision stores it (figures, wpt_depot.schema), and what it
        holds now: {tags: those of the revision's shares, in order,
        libraries: for each library, in order, {stored: its figures (as
        wpt_depot.counts.figures gives them, for tags), None when not
        known; now: the same from the counts now, None for a library this
        depot does not hold}}. A stored figure is not known when the write
        of the revision could not prove it (_figures), or when an erasure
        carried out after the figures were read may have reached a text the
        library held: one whose text the library holds or held, one that
        removed a text whole (which leaves no row naming its libraries), and
        one of a depot whose rows the dataset's depot did not hold at the
        figures' change, whose change cannot be compared. An erasure a pull
        brought before the figures' change was carried out before them
        (bounds). NotFound without such a revision."""
        found = self._rows("SELECT libraries, shares, figures FROM datasets_revisions WHERE id = :id AND "
                           + ("latest" if revision is None else "revision = :revision"),
                           {"id": dataset, "revision": revision})
        if not found:
            raise NotFound(f"no {'current' if revision is None else f'revision {revision} of the'} dataset {dataset}")
        row = found[0]
        tags = [one["tag"] for one in row["shares"] or []]
        figures = row["figures"] or {"at": None, "libraries": [None] * len(row["libraries"])}
        stored = list(figures["libraries"])
        con = self.db.reader()
        with con:  # one read transaction: the stored figures, the erasures and the counts of one state
            reached = self._reached(con, ids.owner(dataset), figures["at"])
            held = {one for (one,) in con.execute("SELECT id FROM libraries")}
            libraries = []
            for ref, one in zip(row["libraries"], stored, strict=True):
                library = ref["library"]
                here = self.reads_here(ref["depot"], library) and library in held
                if one is not None and (reached is None or any(
                        _held_ever(con, library, text) for text in reached)):
                    one = None
                libraries.append({"stored": one, "now": counts.figures(con, library, tags) if here else None})
        return {"tags": tags, "libraries": libraries}

    @staticmethod
    def _reached(con, owner: str, at: int | None) -> list[str] | None:
        """The ids of the texts whose passages the erasures cut that the
        depot named owner did not hold carried out at its change at (bounds:
        one of another depot's numbering counts as held once a pull at or
        before at brought its carrying out), None when one of them removed
        a text whole, is of a depot whose rows owner held none of at at, or
        at is None."""
        if at is None:
            return None
        held = _bounds(con, owner, at)
        texts = set()
        for erasure, carried, needles in con.execute(
                "SELECT e.id, (SELECT min(r.change) FROM erasures_revisions r WHERE r.id = e.id "
                "AND r.erased IS NOT NULL), e.needles FROM erasures e WHERE e.erased IS NOT NULL").fetchall():
            bound = held.get(ids.owner(erasure))
            if bound is None:
                return None
            if carried <= bound:
                continue
            for needle in json.loads(needles):
                if needle["start"] is None and needle["field"] is None:
                    return None
                texts.add(needle["text"])
        return sorted(texts)

    # Tokenizer recipes and their vocabularies

    def new_tokenizer(self, name: str, depot: str | None, dataset: str, sample_bytes: int, vocab_size: int,
                      note: str | None = None) -> str:
        """A tokenizer recipe at version 1: its vocabulary trains on a
        sample of at most sample_bytes bytes of the plain texts of a dataset
        (one of this depot's, by its name or id, with depot None or this
        depot's address; or another depot's, by its name, with depot the web
        address of that depot's web app), up to vocab_size tokens: its id.
        Refused for a name that is taken or not one (TOKENIZER_NAME), a
        sample or a size out of range, an address that is not a web address;
        NotFound for a dataset of this depot it does not hold."""
        row = self._recipe_row(depot, dataset, sample_bytes, vocab_size, note)
        if not isinstance(name, str) or not TOKENIZER_NAME.fullmatch(name):
            raise Refused("a tokenizer recipe's name is lowercase a-z, 0-9, '.', '_' and '-', starting with a letter "
                          "or a digit")
        with self._write("tokenizer") as (con, number):
            if self._rows("SELECT id FROM tokenizers WHERE name = ?", (name,)):
                raise Refused(f"the tokenizer recipe name {name} is taken")
            recipe = self._id(self._take(con, "tokenizers"))
            _insert(con, "tokenizers_revisions", [{"id": recipe, "revision": 1, "change": number, "latest": 1,
                                                   "name": name, "version": 1, **row}])
            return recipe

    def edit_tokenizer(self, recipe: str, depot: str | None, dataset: str, sample_bytes: int, vocab_size: int,
                       note: str | None = None) -> int:
        """Change the tokenizer recipe of that id as a new revision, its id
        and name kept, with the checks of new_tokenizer: its version, which
        it answers, goes up by one when the revision changes what trains its
        vocabulary (the dataset, the sample's bytes, the vocabulary's size),
        and stays for a change of its note alone. A change that changes
        nothing writes nothing. NotFound without such a recipe;
        BelongsToUpstream for a recipe of the upstream."""
        old = self.tokenizer_by_id(recipe)
        self._refuse_upstream("tokenizer recipes", [recipe])
        row = self._recipe_row(depot, dataset, sample_bytes, vocab_size, note)
        if all(old[key] == value for key, value in row.items()):
            return old["version"]
        version = old["version"] + (any(old[key] != row[key] for key in TRAINS_VOCABULARY))
        with self._write("edit tokenizer") as (con, number):
            self._revise(con, "tokenizers_revisions", [recipe], number,
                         {**{key: f":{key}" for key in row}, "version": ":version"}, {**row, "version": version})
        return version

    def _recipe_row(self, depot: str | None, dataset: str, sample_bytes: int, vocab_size: int,
                    note: str | None) -> dict:
        """The columns of a tokenizer recipe's revision that a person gives, checked (new_tokenizer)."""
        whole = (lambda value: isinstance(value, int) and not isinstance(value, bool))
        if not whole(sample_bytes) or sample_bytes < 1:
            raise Refused("a tokenizer recipe's sample holds a whole number of bytes above 0")
        if not whole(vocab_size) or vocab_size < VOCABULARY_BASE:
            raise Refused(f"a vocabulary holds at least {VOCABULARY_BASE} tokens (256 bytes, the end of a text and "
                          "256 reserved ids)")
        _filled(dataset=dataset)
        if note is not None and not note.strip():
            note = None
        if depot is None or self.is_here(depot):
            return {"depot": None, "dataset": self._dataset_named(dataset)["id"], "sample_bytes": sample_bytes,
                    "vocab_size": vocab_size, "note": note}
        if not depot.startswith(("http://", "https://")):
            raise Refused(f"a tokenizer recipe trains on a dataset of this depot, or of another depot named by the "
                          f"web address of its web app (http:// or https://), not {depot}")
        return {"depot": _written(depot), "dataset": dataset.strip(), "sample_bytes": sample_bytes,
                "vocab_size": vocab_size, "note": note}

    def _dataset_named(self, dataset: str) -> dict:
        """The current row of the dataset of that id, or else of that name. NotFound without one."""
        found = self._rows("SELECT * FROM datasets WHERE id = ?", (dataset,))
        return found[0] if found else self.dataset(dataset)

    def tokenizer(self, name: str) -> dict:
        """The current revision of the tokenizer recipe named so. NotFound without one."""
        found = self._rows("SELECT * FROM tokenizers WHERE name = ?", (name,))
        if not found:
            raise NotFound(f"no tokenizer recipe {name}")
        return found[0]

    def tokenizer_by_id(self, recipe: str) -> dict:
        """The current revision of the tokenizer recipe of that id. NotFound without one."""
        found = self._rows("SELECT * FROM tokenizers WHERE id = ?", (recipe,))
        if not found:
            raise NotFound(f"no tokenizer recipe {recipe}")
        return found[0]

    def tokenizers(self) -> list[dict]:
        """Every tokenizer recipe's current revision, by name, with
        trained: whether its version has a vocabulary."""
        return self._rows("SELECT t.*, EXISTS (SELECT 1 FROM vocabularies v WHERE v.recipe = t.id "
                          "AND v.version = t.version) AS trained FROM tokenizers t ORDER BY t.name")

    def vocabulary(self, recipe: str, version: int) -> dict | None:
        """The vocabulary of a version of the tokenizer recipe of that id:
        {recipe, version, change, sha256, merges: the vocabulary file as
        text (its bytes in UTF-8), texts: how many texts it trained on, bytes: their UTF-8
        bytes, None when not known, made_at}; None when that version has none."""
        found = self._rows("SELECT * FROM vocabularies WHERE recipe = ? AND version = ?", (recipe, version))
        return found[0] if found else None

    def vocabulary_texts(self, recipe: str, version: int, offset: int = 0, limit: int = 50) -> list[dict]:
        """The texts a vocabulary trained on, a page at a time in the order
        they were read: each {position, depot: the web address of the depot
        that holds it, '' for the depot that holds the recipe, text: its id}."""
        return self._rows("SELECT position, depot, text FROM vocabulary_texts WHERE recipe = ? AND version = ? "
                          "ORDER BY position LIMIT ? OFFSET ?", (recipe, version, int(limit), int(offset)))

    def add_vocabulary(self, recipe: str, version: int, merges: bytes, texts: Iterable[tuple[str, str]],
                       size: int | None) -> str:
        """Keep the vocabulary of a version of the tokenizer recipe of that
        id, for good: merges, the vocabulary file's bytes (wpt_tokenizer),
        and texts, (the web address of the depot that holds it, '' for this
        depot; its id) of each text it trained on in the order they were
        read, read a few at a time, whose UTF-8 bytes add up to size: its
        sha256. Refused when that version has a vocabulary already, which
        never changes, or when the recipe never had that version; NotFound
        without such a recipe; BelongsToUpstream for a recipe of the
        upstream, whose vocabularies a pull brings."""
        self.tokenizer_by_id(recipe)
        self._refuse_upstream("tokenizer recipes", [recipe])
        sha = hashlib.sha256(merges).hexdigest()
        with self._write("vocabulary") as (con, number):
            if not con.execute("SELECT 1 FROM tokenizers_revisions WHERE id = ? AND version = ?",
                               (recipe, version)).fetchall():
                raise Refused(f"the tokenizer recipe {recipe} never had the version {version}")
            if con.execute("SELECT 1 FROM vocabularies WHERE recipe = ? AND version = ?", (recipe, version)).fetchall():
                raise Refused(f"the tokenizer recipe {recipe} has a vocabulary of its version {version} already: a "
                              "vocabulary never changes, and a new one takes a new version of the recipe")
            count = 0
            for batch in itertools.batched(texts, 10_000):
                con.executemany("INSERT INTO vocabulary_texts (recipe, version, position, depot, text, change) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                [(recipe, version, count + at, depot, text, number)
                                 for at, (depot, text) in enumerate(batch)])
                count += len(batch)
            _insert(con, "vocabularies", [{"recipe": recipe, "version": version, "change": number, "sha256": sha,
                                           "merges": merges.decode("utf-8"), "texts": count, "bytes": size, "made_at": _now()}])
        return sha

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

    def _erased_by_text(self) -> dict[str, list[dict]]:
        """The erased entries of every carried-out request, by the text whose passage each was in."""
        found = {}
        for row in self._rows("SELECT needles, erased FROM erasures WHERE erased IS NOT NULL"):
            for entry in row["erased"]:
                found.setdefault(row["needles"][entry["needle"]]["text"], []).append(entry)
        return found

    def condition(self, found: Filter, latest: int | None = None) -> str:
        """The SQL condition of a filter (filters.condition) for a page or a
        count step, which reads a decision's rows only when the decision is
        on at most SORTED texts (the table counts): a page of them then
        sorts at most as many texts as a page in an order outside INDEXED,
        and a decision on more texts is tested on each text the page or
        step reaches. Refused as filters.condition is."""
        few = found.decision is None or self._one(
            f"SELECT coalesce(max(texts), 0) FROM counts WHERE facet = '{counts.DECISION}' AND name = ?",
            (found.decision,)) <= SORTED
        return condition(found, latest, decision_rows=few)

    def _condition(self, texts: Selection) -> str:
        """The SQL condition over the texts view of a selection: the
        condition itself, or the ids given, written as one value
        (filters.listed), or a filter's for a page (Depot.condition). NotFound when a
        given id is no text."""
        if isinstance(texts, str):
            return texts
        if isinstance(texts, Filter):
            return self.condition(texts)
        given = _json(sorted(set(texts)))
        missing = self.db.reader().execute(
            "SELECT count(*), min(value) FROM json_each(?) WHERE NOT EXISTS (SELECT 1 FROM texts_revisions t "
            "WHERE t.latest AND t.id = value)", (given,)).fetchall()[0]
        if missing[0]:
            raise NotFound(f"{missing[0]} of the texts named are no text of the depot (the first: {missing[1]})")
        return f"id IN {listed_ids(json.loads(given))}"

    def _unlicensed_sql(self, column: str, status: str = ADMITTED) -> str:
        """Whether a decision of that status cannot cover a text because none
        of its licenses (the JSON list in column) is allowed: an admission
        only (wpt_depot.licenses), as an SQL condition."""
        if status != ADMITTED:
            return "false"
        allowed = [one for (one,) in self.db.reader().execute("SELECT id FROM licenses") if licenses.allowed(one)]
        return f"NOT EXISTS (SELECT 1 FROM json_each({column}) j WHERE j.value IN {listed_ids(allowed)})"

    def _choose(self, con, texts: Selection, table: str = "chosen", left: dict | None = None) -> int:
        """Fill the temporary table (one column, id, its key) with the ids
        of the texts a selection names, each once: those an SQL condition
        over the texts view matches now (a person's SQL, which only reads,
        Database.select_into), those a filter matches now less those
        changed since its change number (filters.changed_since; how many
        added to left under CHANGED_SINCE when left is given), or exactly
        the ids given. How many. NotFound when a given id, or an id a
        filter names, is no text;
        BadQuery for a condition that is not one."""
        con.execute(f"CREATE TEMP TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY)")
        con.execute(f"DELETE FROM temp.{table}")
        if isinstance(texts, Filter):
            if texts.ids:
                self._condition(list(texts.ids))  # NotFound for a named id that is no text
            plan = self._plan(replace(texts, change=None))
            if plan.listing.scans:  # every text, or one status: read as stored, never copied into a listing first
                self.db.select_into(con, table, f"SELECT id FROM texts WHERE {plan.where}")
            else:
                self.db.select_into(con, table, f"WITH w AS MATERIALIZED ({plan.listing.rows}) SELECT texts.id FROM w "
                                                f"CROSS JOIN texts ON texts.id = {ids.key_id_sql('w.key')} "
                                                f"WHERE {plan.where}")
            if texts.change is not None:
                con.execute(f"DELETE FROM temp.{table} WHERE EXISTS (SELECT 1 FROM texts_revisions texts "
                            f"WHERE texts.latest AND texts.id = temp.{table}.id AND {changed_since(texts.change)})")
                if left is not None and con.changes():
                    left[CHANGED_SINCE] = left.get(CHANGED_SINCE, 0) + con.changes()
        elif isinstance(texts, str):
            self.db.select_into(con, table, f"SELECT id FROM texts WHERE ({texts})")
        else:
            con.execute(f"INSERT OR IGNORE INTO temp.{table} (id) SELECT value FROM json_each(?)",
                        (_json(list(texts)),))
            missing = con.execute(f"SELECT count(*), min(c.id) FROM temp.{table} c WHERE NOT EXISTS "
                                  "(SELECT 1 FROM texts_revisions t WHERE t.latest AND t.id = c.id)").fetchall()[0]
            if missing[0]:
                raise NotFound(f"{missing[0]} of the texts named are no text of the depot (the first: {missing[1]})")
        return con.execute(f"SELECT count(*) FROM temp.{table}").fetchall()[0][0]

    def _leave_blocked(self, con, status: str, leave: bool) -> dict[str, int]:
        """Of the texts of temp.chosen, those a decision of that status
        cannot cover (blocked): taken out of temp.chosen with leave, and
        {reason: how many} returned; without it, BelongsToUpstream or
        Refused when there is one."""
        theirs = self.upstream_sql("c.id")
        upstream = con.execute(f"SELECT count(*) FROM temp.chosen c WHERE {theirs}").fetchall()[0][0]
        if upstream and not leave:
            named = [one for (one,) in con.execute(f"SELECT c.id FROM temp.chosen c WHERE {theirs} "
                                                   f"ORDER BY {ids.order_sql('c.id')} LIMIT {NAMED}")]
            self._refuse_upstream("texts", named, total=upstream)
        unlicensed = self._unlicensed_sql("t.licenses", status)
        found = con.execute(f"SELECT count(*), min(c.id) FROM temp.chosen c CROSS JOIN texts_revisions t ON t.id = c.id "
                            f"AND t.latest WHERE {unlicensed} AND NOT {theirs}").fetchall()[0]
        if found[0] and not leave:
            raise Refused(f"{found[0]} of the texts have no allowed license, so none is admitted (the first: "
                          f"{found[1]}; docs/licenses.md, \"What the corpus accepts\")")
        left = {reason: count for reason, count in ((BELONGS_TO_UPSTREAM, upstream), (NO_ALLOWED_LICENSE, found[0]))
                if count}
        if left:
            con.execute(f"DELETE FROM temp.chosen WHERE id IN (SELECT c.id FROM temp.chosen c CROSS JOIN texts_revisions t "
                        f"ON t.id = c.id AND t.latest WHERE {theirs} OR {unlicensed})")
        return left

    @staticmethod
    def _link(con, decision: str, number: int, table: str = "chosen"):
        """The decision made at the change number on every text of the temporary table: its rows of
        text_decisions, and its count (wpt_depot.counts)."""
        con.execute(f"INSERT INTO text_decisions (text, decision, change) SELECT id, ?, ? FROM temp.{table}",
                    (decision, number))
        counts.shift(con, {(counts.DECISION, decision): con.changes()})

    @staticmethod
    def _restatus(con, status: str, starts_over: bool):
        """Store on the current revision of every text of temp.chosen the
        status one more decision of that status makes of it
        (policy.combined_sql), and shift the counts of statuses and of
        wpt_depot.counts.SPLIT by it."""
        new = policy.combined_sql("t.status", status, starts_over)
        moved = con.execute(f"SELECT t.status, {new}, count(*) FROM temp.chosen c CROSS JOIN texts_revisions t "
                            "ON t.id = c.id AND t.latest GROUP BY t.status").fetchall()
        new = policy.combined_sql("status", status, starts_over)
        with counts.changing(con, "temp.chosen", counts.SPLIT):
            con.execute(f"UPDATE texts_revisions SET status = {new} WHERE rowid IN (SELECT t.rowid FROM temp.chosen c "
                        f"CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest) AND status IS NOT {new}")
        delta = {}
        for old, now, count in moved:
            delta[counts.STATUS, old] = delta.get((counts.STATUS, old), 0) - count
            delta[counts.STATUS, now] = delta.get((counts.STATUS, now), 0) + count
        counts.shift(con, delta)

    def _column_names(self, table: str) -> list[str]:
        if table not in self._names:
            self._names[table] = [one[1] for one in self.db.reader().execute(f"PRAGMA table_info({table})")]
        return self._names[table]

    def _revise(self, con, table: str, chosen: list[str] | str, number: int, sets: dict[str, str],
                params: dict | None = None, joined: str = ""):
        """A new revision, made at the change number, of each id chosen (a
        list, or a query of ids) of a revised table: its current row with
        each column of sets given by its SQL expression over the current row
        (t), params and the rows joined (a JOIN clause after the current
        row's, one row for each), the revision before it no longer
        latest."""
        con.execute("CREATE TEMP TABLE IF NOT EXISTS revised (id TEXT PRIMARY KEY, revision INTEGER)")
        con.execute("DELETE FROM temp.revised")
        # each id's current row by its key, from the ids' side (CROSS JOIN fixes that order)
        if isinstance(chosen, str):
            con.execute(f"INSERT INTO temp.revised SELECT r.id, r.revision FROM ({chosen}) x CROSS JOIN {table} r "
                        "ON r.id = x.id AND r.latest")
        else:
            con.execute(f"INSERT INTO temp.revised SELECT r.id, r.revision FROM json_each(?) x CROSS JOIN {table} r "
                        "ON r.id = x.value AND r.latest", (_json(list(chosen)),))
        con.execute(f"UPDATE {table} SET latest = 0 WHERE rowid IN (SELECT t.rowid FROM temp.revised c CROSS JOIN "
                    f"{table} t ON t.id = c.id AND t.revision = c.revision)")
        given = {"revision": "t.revision + 1", "change": ":change", "latest": "1",
                 **({"decisions": "'[]'"} if table == "texts_revisions" else {}), **sets}
        names = self._column_names(table)
        con.execute(f"INSERT INTO {table} ({', '.join(names)}) SELECT "
                    f"{', '.join(given.get(name, f't.{name}') for name in names)} "
                    f"FROM temp.revised c CROSS JOIN {table} t ON t.id = c.id AND t.revision = c.revision {joined}",
                    {"change": number, **(params or {})})
        con.execute("DELETE FROM temp.revised")

    @staticmethod
    def _leave(con, libraries: list[str], number: int, table: str = "chosen"):
        """The texts of the temporary table leave the libraries at the change
        number, and the counts of those libraries go down by what they held
        of them (wpt_depot.counts.held)."""
        if libraries:
            rows = (f"FROM temp.{table} c CROSS JOIN memberships m INDEXED BY memberships_held ON m.text = c.id "
                    "AND m.left_change IS NULL WHERE m.library IN (SELECT value FROM json_each(?))")
            leaving = counts.held(con, f"SELECT m.library, m.text {rows}", (_json(libraries),))
            con.execute(f"UPDATE memberships SET left_change = ? WHERE rowid IN (SELECT m.rowid {rows})",
                        (number, _json(libraries)))
            counts.shift(con, {key: -by for key, by in leaving.items()})

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


# how many texts the decision d is on (wpt_depot.counts)
_DECIDED_TEXTS = f"coalesce((SELECT c.texts FROM counts c WHERE c.facet = '{counts.DECISION}' AND c.name = d.id), 0)"
# a new revision of an erasure request's decisions: the current ones and the change's decision
_APPEND = "json_insert(t.decisions, '$[#]', :decision)"


def _quiet(stage: str):
    """A progress that says nothing."""


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


def _none_left(texts: Selection, left: dict) -> str:
    """What NotFound says when a selection names no text."""
    if left.get(CHANGED_SINCE):
        return f"every text the selection matches ({left[CHANGED_SINCE]}) changed since the page was read"
    return f"no text matches {texts}" if isinstance(texts, str) else (
        "no text matches the filter" if isinstance(texts, Filter) else "no text is named")


def half_upgraded(con) -> bool:
    """Whether the depot the connection reads is under an upgrade that has
    not ended: its file holds the table upgrade.UNDER_WAY."""
    return bool(con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'upgrade_under_way'")
                .fetchall())


def _check_format(depot: "Depot", found: int):
    if found != FORMAT:
        raise OtherFormat(other_format(depot.address, found))
    if half_upgraded(depot.db.reader()):
        raise HalfUpgraded(f"{depot.address} is under an upgrade to format {FORMAT} that has not ended: run "
                           "`wpt depot upgrade` again, which goes on where it stopped")


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


def _bounds(con, depot: str, change: int) -> dict[str, int]:
    """What the depot called depot held at its change number change, read through con, as Depot.bounds says."""
    pulls = con.execute("SELECT depot, change, upstream, upstream_change FROM pulls").fetchall()
    found = {}
    while depot not in found:
        found[depot] = change
        last = max((one for one in pulls if one[0] == depot and one[1] <= change), key=lambda one: one[1], default=None)
        if last is None:
            break
        depot, change = last[2], last[3]
    return found


def _locked_then(con, library: str, change: int) -> bool | None:
    """Whether the library was locked at that change number of the depot
    that owns it, read through con; None when it did not exist yet then.
    What a build reads (Depot._held) and a revision's figures
    (Depot._figures) both read it so."""
    state = con.execute("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                        "ORDER BY revision DESC LIMIT 1", (library, change)).fetchall()
    return None if not state else state[0][0] is not None


def _lock_change(con, library: str) -> int | None:
    """The change number at which the library was locked, in the numbering of the depot that owns it, read through
    con; None for one never locked."""
    found = con.execute("SELECT change FROM libraries_revisions WHERE id = ? AND locked_at IS NOT NULL "
                        "ORDER BY revision LIMIT 1", (library,)).fetchall()
    return found[0][0] if found else None


def _held_ever(con, library: str, text: str) -> bool:
    """Whether the library holds or held the text: a membership or a pin of it, read by their keys."""
    return bool(con.execute("SELECT 1 FROM memberships WHERE library = ? AND text = ? UNION ALL "
                            "SELECT 1 FROM pins WHERE library = ? AND text = ? LIMIT 1",
                            (library, text, library, text)).fetchall())


def _remove_texts(con, texts: list[str]):
    """Remove the texts whole: every revision, every content version and part, decision rows, memberships and
    pins (an erasure's part)."""
    if texts:
        for table, column in (("texts_revisions", "id"), ("content_lists", "id"), ("parts", "id"),
                              ("text_decisions", "text"), ("pins", "text")):
            con.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT value FROM json_each(?))", (_json(texts),))
        # a text's memberships, ended or not, by their key: each library named, never a library's rows
        con.execute(f"DELETE FROM memberships WHERE rowid IN (SELECT m.rowid FROM json_each(?) x CROSS JOIN "
                    f"libraries_revisions r CROSS JOIN memberships m INDEXED BY {MEMBERSHIPS_KEY} "
                    "ON m.library = r.id AND m.text = x.value WHERE r.latest)", (_json(texts),))


def _json(value) -> str | None:
    """value as the JSON text a list column holds; None stays None."""
    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parts_json(found: list[tuple[bytes, int]]) -> str:
    return _json([{"hash": hash_.hex(), "length": length} for hash_, length in found])


def _ordered(order: str) -> list[str]:
    """The SQL expressions a page in the order order (one of ORDERS) is ordered by, in turn: the order's own, then
    the numeric order of ids. Refused for an order that is none of ORDERS."""
    if order not in ORDERS:
        raise Refused(f"the texts are ordered by one of {', '.join(ORDERS)}, not {order!r}")
    return ORDERS[order] + [ids.order_sql()]


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


def _new_plain(text: str, plain: Source, parts: list[tuple[bytes, int]], erased: list[dict]) -> bool:
    """Whether plain is a new content for the plain text text, whose current
    content version is made of parts ((hash, length) in order): False when
    it is that content already. Refused when it holds a passage an erasure
    removed from the text (erased: those erased entries, matched by their
    public hash)."""
    if listed(plain).parts == parts:
        return False
    _refuse_erased(text, [plain if isinstance(plain, str) else plain.blocks()], erased)
    return True


def _refuse_erased(text: str, strings: list, erased: list[dict]):
    """Refused when one of the strings of a new content of text holds a passage an erasure removed from it."""
    if erasure.holds_erased(strings, erased):
        raise Refused(f"the new content of {text} holds a passage an erasure removed from it")


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
    writes (wpt_depot.policy.license_change, content_change, review_again)."""
    if isinstance(reason, str) and policy.starts_over(reason.strip()):
        start = next(one for one in policy.STARTS_OVER if reason.strip().startswith(one))
        what = {policy.LICENSE_CHANGE: "license change", policy.CONTENT_CHANGE: "content change",
                policy.REVIEW_AGAIN: "text put back under review"}[start]
        raise Refused(f"a reason that starts with {start!r} is the depot's own record of a {what}; write the "
                      "reason another way")
    if declaration.accepted is not True:
        raise Refused("a decision needs the confirmation that the requirements were read, are accepted and met")
    _filled(name=declaration.name, email=declaration.email)
    if reason is not None:
        _filled(reason=reason)


def spread_shares(shares: list[tuple[str, float | None]] | None) -> list[dict] | None:
    """A dataset's shares as it stores them, [{tag, share}, ...] in order
    (None without shares): a share given as None takes an even part of what
    the shares given as numbers leave of 1, so (a, 0.6), (b, None), (c, None)
    store 0.6, 0.2 and 0.2, and shares all None split 1 evenly. The parts are
    rounded to 12 decimals, so 0.4 split in 8 stores 0.05. Refused when a
    share is None while the numbers reach 1 already, since nothing is left
    for it. The stored shares are numbers only: what reads them (the build,
    its estimate, the measure, a remote) never sees a None."""
    if shares is None:
        return None
    blank = sum(share is None for _, share in shares)
    if blank:
        given = sum(share for _, share in shares if share is not None)
        if given >= 1:
            raise Refused(f"the shares given as numbers reach {given:g} already, so nothing is left for the "
                          f"{blank} share{'s' if blank > 1 else ''} without a number")
        part = round((1 - given) / blank, 12)
        shares = [(tag, part if share is None else share) for tag, share in shares]
    return [{"tag": tag, "share": share} for tag, share in shares]


def _checked_dataset(row: dict):
    _filled(name=row["name"], tokenizer=row["tokenizer"])
    limit = row.get("max_tokens")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise Refused("max_tokens is a whole number of tokens above 0, or none")
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
