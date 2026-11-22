"""A depot: texts and structured items, the decisions on them, erasures,
snapshots and tokenizer recipes with their vocabularies, in the tables of
one SQLite file (wpt_depot.schema). The tables of the libraries and
datasets an older format held stay in the file, and no write touches them
(wpt_depot.legacy reads them).

Depot is the only code that changes a depot's tables. Each operation is one
change: one transaction under the next change number, recorded for the
backups (wpt_depot.database). It checks the whole change against the
depot's rules inside it, so a refused or failed change writes nothing.

The depot's state is numbered by its change number (Depot.change): every
write raises it, and every revision row records the change number it was
made at. What changes in place keeps its old versions as
revision rows of its own table, never overwritten or deleted but by an
erasure: a text's credits and licenses, a tokenizer recipe, the depot's
maintainer and takedown contact, an
erasure request's decisions and what it erased. A decision on a text is a
row of text_decisions with the change it was made at, never deleted but by
an erasure. Each write keeps the status stored on each current text it
changes and the counts (wpt_depot.counts) right, in its own transaction.

The rules (the app's policy, wpt_depot.policy, says how decisions combine):

- Every id is the depot's name, a hyphen and a number (wpt_depot.ids).
- A new text comes with one decision 'under review'. It has a kind (wpt_depot.kinds), plain unless it names another,
  fixed for good: its content (a plain text's text, or the JSON content of
  a text of another kind) is its content version 1, stored as its parts
  (wpt_depot.parts): a plain text may come from a file read as a stream
  (FileText), so no text is ever too big to add. The content of a text of
  a kind other than plain matches its kind's schema, as the kinds table
  stores it, and each excerpt of a conversation quotes a plain text of the
  depot whose current content version its selector matches at exactly one
  place (wpt_depot.quotes), checked again on a change of its content.
  Every license a text names is a
  row of licenses (one of them is enough to allow it), and its notice is
  stored once.
- An operation on texts (a decision, a license change, a change of tags
  or scores) takes them as a selection: an SQL
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
  text it covers has no allowed license (wpt_depot.licenses). Depot.blocked says, for
  given texts, which ones an admission cannot cover and why; a decision
  applies the same check. A person's reason never takes the forms the depot
  keeps for license and content changes (wpt_depot.policy).
- A text's licenses change as a new revision, its id kept: the change is
  recorded as a decision 'under review' whose reason the depot writes,
  naming the old and new lists (one decision per distinct old list), and
  the decisions before it no longer count (wpt_depot.policy). Every new license is a row of licenses.
- A text's content (its plain text, or an item's content, which still
  matches its kind's schema) changes as a new content version stored next
  to the old ones, of the text's own kind: a change of kind is refused (a
  text of another kind is a new text), and a new revision of the text that reads it, its id
  kept. The new version stores only the parts its text holds that no
  version of the text held before. The change is recorded as a decision 'under review' made by the
  person's declaration, whose reason the depot writes naming the old and
  new versions; the decisions before it no longer count (wpt_depot.policy).
  A content that changes nothing writes nothing. A snapshot keeps reading
  the revision it was made at. The plain texts of many texts change at once the
  same way, each text as if changed alone, under one decision for each
  pair of old and new versions; the change reads its lines a few at a time
  into a temporary table and is checked by queries over it
  (wpt_depot.rewriting).
- A text's credits (title, authors, tags, address, notice) change as a new
  revision, its id kept, but its tags, which are rows of taggings
  (wpt_depot.taggings) and never part of a revision: a change of tags alone
  writes no revision. The change is recorded as a decision 'under
  review' made by the person's declaration, whose reason the depot writes,
  naming each credit's old and new value. The decisions before it keep
  counting, so the text keeps its status. A text's tags have no order of
  their own: they are read in the order of their names. Tags given to
  or taken from every text of a selection (retag) are one such decision on
  the texts it changes, a row of taggings and of the listing text_tags for
  each tag, about RETAG_BYTES of the file a text, and retag_estimate says
  what one would write before it does. What a text's tags were at any
  change stays readable, so a snapshot reads them as they were when it was
  made.
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
  version lists any longer goes. Every conversation that quotes a plain
  text it reaches, found through the table quotes, is cut too, those it
  names included, on top of their own cut (wpt_depot.erasure.cut_quotes):
  only its strings of text, never one that names or structures it, each
  excerpt losing every character over one the erasure removes from the
  quoted text. A text named whole is removed with every
  revision, content version and part; its
  `erased` gets each passage's length and hashes, and the file keeps no
  copy of what it removed (wpt_depot.database). The hashes are public and
  keyless (wpt_depot.erasure), so any depot or clone can match an erased
  passage.
- A snapshot (wpt_depot.snapshots) is a name on a change number, unique
  among the depot's own snapshots: the texts admitted at that change, each
  read at its revision as of it. It is written once, in a change of its
  own, and never changed nor removed, an erasure included.
- A tokenizer recipe names the sample that trains its vocabulary: the
  texts of some kinds of a snapshot, with sample_bytes, vocab_size and a
  note (a revision written before snapshots named a dataset instead, and
  needs a snapshot before it trains again). It changes as a new revision;
  a change of what trains the vocabulary raises its version, a change of
  the note alone does not. A version has at most one vocabulary, written
  once with the list of the texts it trained on; a second one is refused.
- The depot's maintainer and takedown contact change as a new revision;
  its name never does, since it begins every id. A tokenizer recipe and
  the depot's facts change with no declaration: they are no decision on a
  text.
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
  upstream text, an edit of an upstream tokenizer recipe. An operation on
  several texts is refused whole when one of them is the upstream's, and
  Depot.blocked names them. The depot's own rows stay fully changeable.
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

from . import (counts, crediting, erasure, ids, kinds, licenses, policy, quotes, rewriting, scoring, snapshots, taggings,
               titles)
from .database import Database, person_columns
from .erasure import Needle
from .errors import (BadContent, BelongsToUpstream, DepotExists, HalfUpgraded, NotADepot, NotFound, OtherFormat, Refused,
                     UnsupportedAddress)
from .filters import (Filter, changed_since, condition, is_id,
                      listed as listed_ids, literal, words)
from .kinds import PLAIN
from .parts import WINDOW, FileText, Parted, Reader, Source, listed, made, pack, parted, split
from .schema import (ADMITTED, CONTENT_JOIN, FORMAT, JSON_COLUMNS, LISTED_COLUMNS, REFUSED, SIZES, UNDER_REVIEW,
                     tags_now)

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
SINGULAR = {"texts": "text", "erasure requests": "erasure request", "tokenizer recipes": "tokenizer recipe"}
# the credits of a text that change as a new revision (Depot.edit_credits), in the order its reason names them
CREDITS = crediting.CREDITS
# what the rows of a search can be ordered by: each the SQL expressions over the searched rows it orders by, in turn,
# before the numeric order of ids (_ordered); none of them is ever null, so a row's values of them compare as a whole
ORDERS = {"id": [], "title": ["title IS NOT NULL", "coalesce(title, '')"],
          "kind": ["kind"], "status": ["status"], "tags": ["tags"], "licenses": ["licenses"],
          "length": ["length"]}
# the orders an index of the depot serves (schema.INDEXES), so a page in them reads only its own rows; a page in any
# other order sorts every match, so it is given only when a search matches at most SORTED texts
INDEXED = ("id", "status")
SORTED = 100_000
# a tokenizer recipe's name
TOKENIZER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
# what a tokenizer recipe's revision changes that trains another vocabulary, and so raises its version
TRAINS_VOCABULARY = ("depot", "dataset", "sample", "sample_bytes", "vocab_size")
# the kinds of texts a vocabulary's sample takes (Depot.new_tokenizer): a sample is read as text, which an item is not
SAMPLE_KINDS = (PLAIN,)
VOCABULARY_BASE = 513  # the smallest vocabulary: 256 bytes, the end of a text and 256 reserved ids (wpt_tokenizer)
# how many tags a retag (Depot.retag) removes from the text whose id is {text} (it has them: :remove, a JSON list),
# and how many it adds (it lacks them: :add), from its rows of taggings
_REMOVED = ("(SELECT count(*) FROM taggings g WHERE g.text = {text} AND g.left_change IS NULL "
            "AND g.tag IN (SELECT value FROM json_each(:remove)))")
_ADDED = ("(SELECT count(*) FROM json_each(:add) a WHERE NOT EXISTS (SELECT 1 FROM taggings g WHERE g.text = {text} "
          "AND g.tag = a.value AND g.left_change IS NULL))")
# what retagging one text costs, measured adding one tag to 1,000, 10,000 and 100,000 texts of the synthetic depot of
# 2,000,000 texts of packages/wpt-depot/bench (tags.py): the bytes the depot's file grows by, the bytes of the change
# file a backup sends (the rows of taggings, text_tags and text_decisions, and the copy of the text's tags its current
# revision is rewritten with), and the seconds it takes cold; each the highest per text of those sizes, which a
# smaller retag reaches (1,000 texts) and a larger one stays under
RETAG_BYTES, RETAG_CHANGE_BYTES, RETAG_SECONDS = 270, 330, 0.0006
# how many texts a step of Depot.count goes over at most
COUNT_STEP = 1_000_000
# how many candidates a page reads at most (Depot.search), and with a search in the texts themselves, which reads
# each candidate's text
WINDOW, TEXT_WINDOW = 10_000, 200
# how many revisions, the latest changes first, Depot.score_values goes over at most, and how many values it gives
SCORE_READ, SCORE_VALUES = 200_000, 12
# the columns of the texts view but those that hold a text: what a query reads that never reads a text's parts
FIELDS = ("id, revision, change, kind, content_version, length, licenses, notice, title, authors, tags, address, "
          "decisions, scores, status")
# every text with the columns a page reads and orders by (Depot.search), before the condition that picks them
MATCHED = f"SELECT {FIELDS} FROM texts"


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
    rows, whose counting would take as long as the steps; tag, for the
    texts of one tag (text_tags), that tag, which every row has."""
    rows: str
    texts: int
    ordered: bool
    always: bool = False
    parts: tuple["str | titles.Part", ...] = ()
    scans: bool = False
    total: Callable[[], int] | None = None
    tag: str | None = None


@dataclass(frozen=True)
class _Plan:
    """What reads the texts a selection matches: where, the SQL condition
    over the texts view that matches them; listing, the rows read as
    candidates; status, the one status a filter names (None for any
    other); in_text, whether its plain search reads the texts
    themselves; within, the condition a row of the listing must meet, where
    without what the listing holds already (the tag of a listing of one
    tag's texts), None when it is where itself."""
    where: str
    listing: _Listing
    status: str | None = None
    in_text: bool = False
    within: str | None = None

    @property
    def candidates(self) -> str:
        """The SQL condition a row of the listing is tested on."""
        return self.where if self.within is None else self.within


def _listing(rows: str, values: Sequence[str], held: dict[str, int]) -> _Listing | None:
    """The listing of the texts that have one of the values (a filter's
    licenses or statuses), from rows, a query whose {names} is
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
    tags change, and the title index when the title does."""
    return (((counts.TAG, counts.SPLIT_TAG, counts.PLAIN_TAGSET) if "tags" in names else ())
            + ((counts.TITLE,) if "title" in names else ()))


def _tag_change(add: Sequence[str], remove: Sequence[str]) -> tuple[list[str], list[str]]:
    """The tags a retag adds and removes, each once, in the order given.
    Refused for no tag at all, a tag that is not a word, or a tag both
    added and removed."""
    if isinstance(add, str) or isinstance(remove, str):
        raise Refused("the tags added and removed are lists of words")
    found = []
    for listed in (add, remove):
        if any(not isinstance(one, str) or not one.strip() for one in listed):
            raise Refused("a tag is a word that is not empty")
        found.append(list(dict.fromkeys(one.strip() for one in listed)))
    added, removed = found
    if not added and not removed:
        raise Refused("a change of tags adds or removes at least one tag")
    both = [one for one in added if one in removed]
    if both:
        raise Refused(f"a tag is added or removed, not both: {', '.join(both)}")
    return added, removed


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
    """A text with its credits: of kind plain (the default), its text (a
    string, or a FileText read as a stream); of any other kind (an item),
    its JSON content. notice is the notice's text, stored once."""
    text: Source | None = None
    kind: str = PLAIN
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
    def upgrade(cls, address: str, to: str | None = None, upstream: str | None = None,
                records: list[dict] | None = None, public: str | None = None, checked=None) -> tuple["Depot", int]:
        """(the depot at address brought up to this code's format, the
        format it had), as wpt_depot.upgrade.upgrade does."""
        from .upgrade import upgrade
        return upgrade(address, to, upstream, records=records, public=public, checked=checked)

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
        """Of the ids of rows (texts, erasure requests, tokenizer recipes),
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

    def is_here(self, address: str) -> bool:
        """Whether a depot's address leads to this depot, in any form
        (place), opening nothing."""
        return self.place(address) is None

    def place(self, address: str | None) -> Path | str | None:
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
        revisions: how many revisions it has, content_versions: how
        many content versions it has (its current revision reads the one
        its row's content_version names), upstream: whether the upstream of
        this clone owns it}. Its text, or an item's content, is read from its
        parts, whatever its size. NotFound without such a text."""
        found = self._rows(f"SELECT {FIELDS} FROM texts WHERE id = ?", (text,))
        if not found:
            raise NotFound(f"no text {text}")
        row = found[0]
        whole = self.content(text, row["content_version"])
        plain = row["kind"] == PLAIN
        row = {**row, "text": whole if plain else None, "content": None if plain else whole}
        notice = None if row["notice"] is None else self._one("SELECT text FROM notices WHERE id = ?", (row["notice"],))
        revisions = self._one("SELECT count(*) FROM texts_revisions WHERE id = ?", (text,))
        versions = self._one("SELECT count(*) FROM content_lists WHERE id = ?", (text,))
        decisions = self._rows(f"SELECT d.* FROM text_decisions l JOIN decisions d ON d.id = l.decision "
                               f"WHERE l.text = ? ORDER BY {ids.order_sql('d.id')}", (text,))
        counting = {one["id"] for one in policy.counted(decisions)}
        status = row.pop("status")
        return {"text": row, "status": status, "notice": notice,
                "decisions": [{**one, "counts": one["id"] in counting, "put_back": policy.put_back(one)}
                              for one in decisions],
                "revisions": revisions, "content_versions": versions,
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
        selection names (_plan): the texts of one tag, one license or one
        status, in the order of their ids, from their index;
        a decision's texts, several licenses, or the ids
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
        or content: its id, kind, title, authors, tags, licenses, scores,
        status, and its length in characters. BadQuery for a condition that is not one; Refused for an
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
        columns = "id, kind, title, authors, tags, licenses, scores, status, length"
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
        return {"rows": [_decoded(one) for one in page],
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
                                    held[counts.TAG].get(tag, 0), True, tag=tag))
        options.append(_listing("SELECT key FROM text_licenses WHERE license {names}", found.licenses,
                                held[counts.LICENSE]))
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
        latest = self.change()
        # a row of one tag's listing has that tag: its rows of taggings are not read again to test it
        within = None if listing.tag is None else self.condition(
            replace(found, tags=tuple(one for one in found.tags if one != listing.tag)), latest)
        return _Plan(self.condition(found, latest), listing, status, found.in_text, within)

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
                              f"filter names at most {SORTED:,} texts by a tag, a license, a status or a "
                              f"decision, and this one names more: narrow the filter, or order by "
                              f"{' or '.join(INDEXED)}")
            window, last = f"SELECT key FROM ({listing.rows})", None
        compared = ("" if place is None else
                    f"WHERE ({', '.join(keys)}) {compare} ({', '.join('?' * len(keys))})")
        if order == "id":
            # the window's keys, read through its index in the order of the page, are the outer loop and the page
            # is ordered by them (each the key of its text's id), so SQLite sorts nothing and looks up only the
            # texts of the window up to the page's last match, not every text of the window
            matched = (f"SELECT {FIELDS}, w.key AS window_key FROM ({window}) w CROSS JOIN texts "
                       f"ON texts.id = {ids.key_id_sql('w.key')} WHERE ({plan.candidates})")
            sql = (f"SELECT {columns}, json_array({', '.join(keys)}) AS place FROM ({matched}) {compared} "
                   f"ORDER BY window_key{direction} LIMIT {int(limit) + 1}")
        else:
            # the window is read on its own (MATERIALIZED), in the order of its index, before any text is looked up
            matched = (f"SELECT {FIELDS} FROM w CROSS JOIN texts "
                       f"ON texts.id = {ids.key_id_sql('w.key')} WHERE ({plan.candidates})")
            by = ", ".join(f"{one}{direction}" for one in keys)
            sql = (f"WITH w AS MATERIALIZED (SELECT key FROM ({window})) SELECT {columns}, "
                   f"json_array({', '.join(keys)}) AS place FROM ({matched}) {compared} ORDER BY {by} "
                   f"LIMIT {int(limit) + 1}")
        found = self.db.person_rows(sql, params + (tuple(place) if place is not None else ()))
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
        wpt_depot.policy); tags and licenses: how many of them have each tag and name each
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
            f"m AS MATERIALIZED (SELECT status, {self.upstream_sql()} AS theirs, "
            f"{self._unlicensed_sql('licenses')} AS unlicensed, id, licenses, tags FROM w CROSS JOIN texts "
            f"ON texts.id = {ids.key_id_sql('w.key')} WHERE ({plan.candidates})) "
            "SELECT 'status' AS part, status AS name, count(*) AS n FROM m GROUP BY status "
            f"UNION ALL SELECT 'blocked', '{BELONGS_TO_UPSTREAM}', count(*) FROM m WHERE theirs "
            f"UNION ALL SELECT 'blocked', '{NO_ALLOWED_LICENSE}', count(*) FROM m WHERE unlicensed AND NOT theirs "
            "UNION ALL SELECT 'tag', g.value, count(*) FROM m, json_each(m.tags) g GROUP BY g.value "
            "UNION ALL SELECT 'license', j.value, count(*) FROM m, json_each(m.licenses) j GROUP BY j.value",
            params)
        statuses = [one for one in found if one["part"] == "status"]
        named_counts = {part: sorted(([one["name"], one["n"]] for one in found if one["part"] == part),
                                     key=lambda one: (-one[1], one[0])) for part in ("tag", "license")}
        return {**step, "held": False,
                "statuses": {status: sum(one["n"] for one in statuses if one["name"] == status)
                             for status in (UNDER_REVIEW, ADMITTED, REFUSED)},
                "blocked": {one["name"]: one["n"] for one in found if one["part"] == "blocked"},
                "tags": [{"tag": name, "texts": n} for name, n in named_counts["tag"]],
                "licenses": [{"license": name, "texts": n} for name, n in named_counts["license"]]}

    def held_count(self, found: Filter) -> dict | None:
        """The count of the texts a filter matches, in the shape of a count
        step over every text (count), read at once from the depot's counts
        (wpt_depot.counts) whatever the depot's size, or None when they
        cannot give it: they split every text, and the texts of each tag and
        license, by status (counts.SPLIT), so they answer a filter of any
        statuses and of at most one tag or license; and they split the texts
        of each status by kind (counts.KIND), so they answer a filter of any
        statuses and kinds that names no tag nor license; and nothing else
        (no ids, search, SQL, decision, unticked texts, two tags or
        licenses, kinds with either, nor a change number older than the
        depot's). tags, licenses and blocked are None where the counts do
        not give them: they do when the filter matches every text or none,
        and blocked also when no upstream is followed and every license a
        text names is allowed (every text names one). held: True."""
        named_parts = [(facet, values) for facet, values in ((counts.TAG, found.tags), (counts.LICENSE, found.licenses))
                       if values]
        if (found.ids is not None or found.unticked or found.decision
                or any(one not in (UNDER_REVIEW, ADMITTED, REFUSED) for one in found.statuses)
                or (found.where or "").strip() or (found.search or "").strip() or len(named_parts) > 1
                or any(len(set(values)) > 1 for _, values in named_parts)
                or (found.change is not None and found.change < self.change())
                or (found.kinds and named_parts)):
            return None
        con = self.db.reader()
        wanted = set(found.statuses) or {UNDER_REVIEW, ADMITTED, REFUSED}
        with con:  # one read transaction: every count is of one state
            held = counts.read(con)
            facet, value = (named_parts[0][0], named_parts[0][1][0]) if named_parts else (counts.TEXTS, "")
            parts = (counts.of_kinds(con, wanted, found.kinds) if found.kinds else counts.split(con, facet, value))
        every = held[counts.TEXTS].get("", 0)
        statuses = {status: parts.get(status, 0) if status in wanted else 0
                    for status in (UNDER_REVIEW, ADMITTED, REFUSED)}
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
        return {"texts": every, "total": every, "next": None, "held": True, "statuses": statuses,
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
        have each status, tag and license and each decision is on."""
        return counts.read(self.db.reader())

    def recount(self):
        """Make the depot's counts again from its rows (wpt_depot.counts),
        after copying again onto each revision the kind, length and bytes
        of its content version (schema.sizes_sql) and onto each current one
        the copy of its text's tags (schema.tags_sql) where they differ, as
        one change that writes nothing else."""
        with self._write("recount") as (con, _):
            counts.recount(con)

    def latest_decisions(self, limit: int = 10) -> list[dict]:
        """The latest decisions, the newest first, each with how many texts
        (wpt_depot.counts) and erasure requests it decides and
        put_back: whether it put texts back under review
        (wpt_depot.policy.put_back)."""
        found = self._rows(
            f"SELECT d.*, {_DECIDED_TEXTS} AS texts, "
            "(SELECT count(*) FROM erasures_revisions e WHERE e.latest "
            "AND EXISTS (SELECT 1 FROM json_each(e.decisions) WHERE value = d.id)) AS erasures "
            f"FROM decisions d ORDER BY d.made_at DESC, d.id DESC LIMIT {int(limit)}")
        return [{**one, "put_back": policy.put_back(one)} for one in found]

    def decision(self, decision: str) -> dict:
        """{decision: its row with put_back (wpt_depot.policy.put_back),
        texts: how many texts it is on, erasures: the ids of the erasure
        requests it decides}. NotFound without such a decision."""
        found = self._rows(f"SELECT d.*, {_DECIDED_TEXTS} AS texts FROM decisions d WHERE d.id = ?", (decision,))
        if not found:
            raise NotFound(f"no decision {decision}")
        row = found[0]
        row["put_back"] = policy.put_back(row)
        texts = row.pop("texts")
        erasures = [one for (one,) in self.db.reader().execute(
            "SELECT e.id FROM erasures e WHERE EXISTS (SELECT 1 FROM json_each(e.decisions) WHERE value = ?) "
            "ORDER BY e.id", (decision,))]
        return {"decision": row, "texts": texts, "erasures": erasures}

    def bounds(self, depot: str, change: int) -> dict[str, int]:
        """What the depot called depot held at its change number change, as
        {the name of each depot whose rows it held: the highest change number
        of that depot's numbering it held}: its own rows up to change, and,
        when it follows an upstream, the upstream's rows up to the upstream
        change brought by its last pull at or before change (pulls, its
        clone included), and so on back through the upstream's own pulls."""
        return _bounds(self.db.reader(), depot, change)

    def change_times(self, changes: Sequence[int]) -> dict[int, str | None]:
        """{change number: the UTC time it was made at (None for a change
        made before format 4)} of each of the depot's changes named that it
        holds."""
        return dict(self.db.reader().execute(
            "SELECT id, made_at FROM changes WHERE id IN (SELECT value FROM json_each(?))",
            (_json([int(one) for one in changes]),)).fetchall())

    def listed_text(self, text: str, revision: int) -> dict:
        """One text at one revision: the columns of schema.LISTED_COLUMNS,
        the list of parts of the content version it reads and its bytes,
        never its text, and the tags the text has now (a revision holds no
        tags: a snapshot's reads take them at its bound, wpt_depot.order).
        NotFound when the depot no
        longer holds it (an erasure removed it)."""
        found = self._rows(f"SELECT {LISTED_COLUMNS} FROM texts_revisions t {CONTENT_JOIN} "
                           "WHERE t.id = ? AND t.revision = ?", (text, revision))
        if not found:
            raise NotFound(f"the depot holds no revision {revision} of the text {text}")
        return found[0]

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
                if one.kind == "conversation":
                    self._check_quotes(one.content)
            decision = self._decision(con, number, UNDER_REVIEW, declaration)
            first = self._take(con, "texts", len(texts))
            notices, new_texts = {}, []
            for at, one in enumerate(texts):
                text_id = self._id(first + at)
                notice = None if one.notice is None else self._notice(con, number, one.notice, notices)
                (sized,) = self._store(con, number, [{"id": text_id, "version": 1, "kind": one.kind,
                                                      "source": one.text if one.kind == PLAIN else one.content}])
                new_texts.append({**dict(zip(SIZES, sized)), "id": text_id, "revision": 1, "change": number, "latest": 1,
                                  "licenses": _json(list(one.licenses)), "notice": notice, "title": one.title,
                                  "authors": None if one.authors is None else _json(list(one.authors)),
                                  "tags": "[]", "address": one.address, "decisions": "[]",
                                  "content_version": 1, "scores": None, "status": UNDER_REVIEW})
            _insert(con, "texts_revisions", new_texts)
            con.executemany("INSERT INTO taggings (text, tag, joined_change, left_change, change) "
                            "VALUES (?, ?, ?, NULL, ?)", [(one["id"], tag, number, number) for one, given in
                                                          zip(new_texts, texts) for tag in dict.fromkeys(given.tags)])
            added = [one["id"] for one in new_texts]
            self._choose(con, added)
            taggings.copy(con, "SELECT id FROM temp.chosen")
            self._link(con, decision, number)  # counts the decision's texts
            counts.shift(con, counts.measure(con, "temp.chosen", (counts.TEXTS, counts.STATUS, counts.TAG,
                                                                  counts.LICENSE, counts.SPLIT_TEXTS,
                                                                  counts.SPLIT_TAG, counts.SPLIT_LICENSE,
                                                                  *counts.STATUSED)))
            counts.enlist(con, "temp.chosen", counts.LISTED)
            return added

    def _check_quotes(self, content: str):
        """BadContent unless each excerpt of the conversation whose content
        is given (JSON text that matches its kind) quotes a plain text of the
        depot whose current content version matches its selector at exactly
        one place (wpt_depot.quotes), read as its parts."""
        for block in quotes.excerpts(json.loads(content)):
            source, selector = block["source"], block["selector"]
            found = self._rows(f"SELECT c.kind, c.parts FROM texts_revisions t {CONTENT_JOIN} "
                               "WHERE t.latest AND t.id = ?", (source,))
            if not found:
                raise BadContent(f"an excerpt quotes {source}, which is no text of the depot")
            if found[0]["kind"] != PLAIN:
                raise BadContent(f"an excerpt quotes {source}, an item of kind {found[0]['kind']}: an excerpt "
                                 "quotes a plain text")
            if not selector["exact"]:
                raise BadContent(f"an excerpt of {source} quotes no character: its exact is the passage")
            places = quotes.find(Parted(self.reader(), source, found[0]["parts"]), selector)
            if len(places) != 1:
                raise BadContent(f"an excerpt of {source} matches {'no place' if not places else 'more than one place'}"
                                 " of its current content: its exact, with its prefix and suffix, picks one passage")

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
                listing = min((plan.listing, option), key=lambda one: one.texts)
                plan = replace(plan, listing=listing, within=plan.within if listing is plan.listing else None)
        found, edge = self._page(replace(plan, where=f"({plan.where}) AND ({theirs} OR {unlicensed})", within=None
                                         if plan.within is None else f"({plan.within}) AND ({theirs} OR {unlicensed})"),
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
        decisions. Its answer:
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
        is under review again. A text that names the licenses
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
            with counts.changing(con, "temp.chosen",
                                 (counts.STATUS, counts.LICENSE, *counts.BY_STATUS)):
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
            stage("committing")
            return changed, skipped

    @_answers
    def change_content(self, text: str, plain: Source | None, content: str | None,
                       declaration: Declaration, kind: str | None = None) -> tuple[str, int] | None:
        """Change the content of a text: a plain text takes its new text as
        plain (a string, or a FileText read as a stream) and no content; an item takes its new content as JSON text as
        content and no plain, which must still match its kind's schema. A
        text's kind is fixed when it is made: kind, when given, must be the
        text's own, and the new content is of that kind. It is stored as a new content version next to
        the old ones, and a new revision of the text, its id kept, reads it.
        The change is recorded as a decision 'under review' made by
        declaration, whose reason the depot writes
        (wpt_depot.policy.content_change); the decisions before it no longer
        count, so the text is under review again. (the decision's id, the new
        content version), or None, with nothing written, when the new
        content is the one the text has (for an item, the same JSON value).
        NotFound without such a text; BadContent for an item's content that
        does not match its kind; Refused when the field given does not
        match the text (a plain text with content or without plain, an item
        with plain or without content) or is not text, for a kind other
        than the text's (to change kind, add a new text), and when the new
        content holds a passage an erasure removed from this text (matched
        by its public hash)."""
        with self._write("content") as (con, number):
            found = self._rows(f"SELECT c.kind, t.content_version, c.parts FROM texts_revisions t {CONTENT_JOIN} "
                               "WHERE t.latest AND t.id = ?", (text,))
            if not found:
                raise NotFound(f"no text {text}")
            self._refuse_upstream("texts", [text])
            row = found[0]
            if kind is not None and kind != row["kind"]:
                raise Refused(f"{text} is a text of kind {row['kind']}, and a text's kind is fixed when it is made: "
                              f"to have it as a text of kind {kind}, add a new text")
            erased = self._erased_by_text().get(text, [])
            if row["kind"] == PLAIN:
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
                if row["kind"] == "conversation":
                    self._check_quotes(content)
                _refuse_erased(text, erasure.strings_of(json.loads(content)), erased)
            version = self._one("SELECT max(version) FROM content_lists WHERE id = ?", (text,)) + 1
            decision = self._decision(con, number, UNDER_REVIEW, declaration,
                                      policy.content_change(row["content_version"], version), checked=False)
            self._store(con, number, [{"id": text, "version": version, "kind": row["kind"],
                                       "source": plain if row["kind"] == PLAIN else content}])
            self._choose(con, [text])
            with counts.changing(con, "temp.chosen", (counts.STATUS, *counts.BY_STATUS)):
                self._revise(con, "texts_revisions", [text], number,
                             {**NEW_VERSION, "status": f"'{UNDER_REVIEW}'"}, {"version": version}, NEW_VERSION_JOIN)
            self._link(con, decision, number)
            return decision, version

    def change_content_many(self, changes: str | Path | list[dict], declaration: Declaration) -> rewriting.Rewritten:
        """Change the content of many plain texts at once, each as
        change_content changes one: a new content version and a new
        revision, its id kept; the decisions before it no longer count, so
        each text is under review again. changes is a JSON-lines file, or the lines as objects (a
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
                        self._store(con, number, [{"id": text, "version": last + 1, "kind": PLAIN,
                                                   "source": plain}])
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
                                     (counts.STATUS, *counts.BY_STATUS)):
                    for old, new in pairs:
                        reason = policy.content_change(old, new)
                        decision = self._decision(con, number, UNDER_REVIEW, declaration, reason, checked=False)
                        con.execute("DELETE FROM temp.rewritten_group")
                        con.execute("INSERT INTO temp.rewritten_group SELECT id FROM temp.rewritten "
                                    "WHERE old = ? AND new = ?", (old, new))
                        self._revise(con, "texts_revisions", "SELECT id FROM temp.rewritten_group", number,
                                     {**NEW_VERSION, "status": f"'{UNDER_REVIEW}'"}, {"version": new},
                                     NEW_VERSION_JOIN)
                        self._link(con, decision, number, "rewritten_group")
                        decisions.append((decision, reason, con.execute(
                            "SELECT count(*) FROM temp.rewritten_group").fetchall()[0][0]))
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
        counting, so the text keeps its status. (its id, the
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
            found = self._rows(f"SELECT t.*, {tags_now('t.id')} AS tags FROM texts_revisions t WHERE t.latest "
                               "AND t.id = ?", (text,))
            if not found:
                raise NotFound(f"no text {text}")
            self._refuse_upstream("texts", [text])
            row, values = found[0], {}
            for name in CREDITS:
                if name not in changes:
                    continue
                value = changes[name]
                if name == "tags":
                    value = sorted(dict.fromkeys(value or []))
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
            revised = {name: value for name, value in values.items() if name != "tags"}
            with counts.changing(con, "temp.chosen", _credited(values)):
                if revised:
                    self._revise(con, "texts_revisions", [text], number, {name: f":{name}" for name in revised},
                                 {name: _json(value) if name in JSON_COLUMNS else value
                                  for name, value in revised.items()})
                if "tags" in values:
                    taggings.put(con, number, f"SELECT {literal(text)} AS id, {literal(_json(values['tags']))} AS tags")
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
        decisions before it keep counting, so each text keeps its status. A
        line may name, under "was", the values its writer read of
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
                    revised = [name for name in CREDITS if name != "tags"]
                    if any(name in revised for name in names):
                        self._revise(con, "texts_revisions", "SELECT id FROM temp.credited WHERE revised", number,
                                     {name: f"s.{name}" for name in revised},
                                     joined="JOIN temp.credited s ON s.id = t.id")
                    if "tags" in names:
                        taggings.put(con, number, "SELECT id, tags FROM temp.credited")
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
        each text keeps its status. A text whose scores would stay
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
                with counts.changing(con, "temp.scored_ids", (counts.SCORE,)):
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
        status. A text whose scores would stay as they are is
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
                with counts.changing(con, "temp.scored_ids", (counts.SCORE,)):
                    self._revise(con, "texts_revisions", "SELECT id FROM temp.scored_ids", number,
                                 {"scores": "(SELECT s.scores FROM temp.scored s WHERE s.id = t.id)"})
                self._link(con, decision, number, "scored_ids")
                stage("committing")
            finally:
                scoring.close(con)
            return decision, count, left_out

    @_answers
    def retag(self, texts: Selection, add: Sequence[str], remove: Sequence[str], declaration: Declaration,
              progress=None, left: dict | None = None) -> tuple[str | None, int, int]:
        """Give the tags add to every text selected and take the tags remove
        from it, as rows of taggings (wpt_depot.taggings), with no new
        revision of any text. The change is recorded as one decision
        under review made by declaration, on every text it changes, whose
        reason the depot writes (policy.tags_change); the decisions before it
        keep counting, so each text keeps its status. A text that has
        every tag added and none removed already is left out, and gets
        nothing. (the decision's id, None when no text changes and nothing
        is written; how many texts changed; how many were left out). What it
        writes is what retag_estimate says. progress as for decide; left,
        when given, receives CHANGED_SINCE as _choose says. NotFound when
        nothing is selected or a given id is no text; BelongsToUpstream when
        a text selected is the upstream's; Refused for no tag at all, an
        empty tag, or a tag both added and removed."""
        add, remove = _tag_change(add, remove)
        stage = progress or _quiet
        with self._write("retag") as (con, number):
            stage("selecting the texts")
            changed = {}
            chosen = self._choose(con, texts, left=changed)
            if left is not None:
                left.update(changed)
            if not chosen:
                raise NotFound(_none_left(texts, changed))
            stage("checking the texts")
            self._leave_blocked(con, UNDER_REVIEW, False)  # a text the upstream owns refuses the whole change
            values = {"add": _json(add), "remove": _json(remove)}
            con.execute("CREATE TEMP TABLE IF NOT EXISTS retagged (id TEXT PRIMARY KEY)")
            con.execute("DELETE FROM temp.retagged")
            con.execute(f"INSERT INTO temp.retagged SELECT c.id FROM temp.chosen c WHERE {_REMOVED.format(text='c.id')} "
                        f"OR {_ADDED.format(text='c.id')}", values)
            count = con.execute("SELECT count(*) FROM temp.retagged").fetchall()[0][0]
            if not count:
                raise _Nothing((None, 0, chosen))
            stage("writing the tags")
            decision = self._decision(con, number, UNDER_REVIEW, declaration, policy.tags_change(add, remove),
                                      checked=False)
            # the counts kept right as counts.changing keeps them, but the listing text_tags changed only by the
            # tags added and removed, not every tag of each text taken out and put back
            facets = _credited(["tags"])
            before = counts.measure(con, "temp.retagged", facets)
            key = ids.order_sql("c.id")
            con.execute(f"DELETE FROM text_tags WHERE (tag, key) IN (SELECT r.value, {key} FROM temp.retagged c, "
                        "json_each(:remove) r)", values)
            pairs = taggings.pairs(con)
            con.execute(f"INSERT INTO {pairs} (text, tag) SELECT c.id, r.value FROM temp.retagged c, json_each(:remove) r",
                        values)
            taggings.take(con, number)
            taggings.pairs(con)
            con.execute(f"INSERT INTO {pairs} (text, tag) SELECT c.id, a.value FROM temp.retagged c, json_each(:add) a",
                        values)
            taggings.give(con, number)
            con.execute(f"DELETE FROM {pairs}")
            con.execute(f"INSERT OR IGNORE INTO text_tags (tag, key) SELECT a.value, {key} FROM temp.retagged c, "
                        "json_each(:add) a", values)
            after = counts.measure(con, "temp.retagged", facets)
            after.subtract(before)
            counts.shift(con, after)
            self._link(con, decision, number, "retagged")
            stage("committing")
            return decision, count, chosen - count

    def retag_estimate(self, texts: Selection, add: Sequence[str], remove: Sequence[str]) -> dict:
        """What retag would write with the same selection and tags, read now
        and writing nothing: {texts: how many texts the selection matches;
        changed: how many of them would change; upstream: how many of them
        the upstream owns (any refuses the change); tags_added and
        tags_removed: how many tags it would give and take, a text and a
        tag each; rows: {text_decisions, taggings, text_tags, decisions},
        the rows it would write: one of text_decisions for each text that
        changes, one of taggings and one of text_tags for each tag given or
        taken (a row added, or a row of taggings ended and one of text_tags
        removed), and one decision (none when nothing changes); written:
        those rows together (it also moves a few rows of counts); bytes:
        what the depot's file would grow by, and
        change_bytes: the change file a backup would send, both from the
        bytes per text retagging measured (RETAG_BYTES, RETAG_CHANGE_BYTES);
        seconds: how long it would take, at the time per text it measured
        (RETAG_SECONDS)}. A filter's change number leaves out the texts
        changed since, as retag does. NotFound when a given id is no text;
        BadQuery for a condition that is not one; Refused as retag is."""
        add, remove = _tag_change(add, remove)
        plan = self._plan(texts)
        if isinstance(texts, Filter) and not plan.listing.scans:
            rows = (f"WITH w AS MATERIALIZED ({plan.listing.rows}) SELECT texts.id FROM w CROSS JOIN texts "
                    f"ON texts.id = {ids.key_id_sql('w.key')} WHERE {plan.candidates}")
        else:
            if isinstance(texts, Filter) and texts.ids:
                self._condition(list(texts.ids))  # NotFound for a named id that is no text
            rows = f"SELECT id FROM texts WHERE ({plan.where})"
        found = self.db.person_rows(
            f"WITH m AS MATERIALIZED (SELECT id FROM ({rows})), "
            f"n AS MATERIALIZED (SELECT id, {_ADDED.format(text='m.id')} AS added, "
            f"{_REMOVED.format(text='m.id')} AS removed FROM m) "
            f"SELECT count(*) AS texts, coalesce(sum({self.upstream_sql('n.id')}), 0) AS upstream, "
            "coalesce(sum(added > 0 OR removed > 0), 0) AS changed, coalesce(sum(added), 0) AS added, "
            "coalesce(sum(removed), 0) AS removed FROM n",
            {"add": _json(add), "remove": _json(remove)})[0]
        changed = found["changed"]
        written = {"text_decisions": changed, "taggings": found["added"] + found["removed"],
                   "text_tags": found["added"] + found["removed"], "decisions": 1 if changed else 0}
        return {"texts": found["texts"], "changed": changed, "upstream": found["upstream"], "rows": written,
                "tags_added": found["added"], "tags_removed": found["removed"],
                "written": sum(written.values()), "bytes": round(changed * RETAG_BYTES),
                "change_bytes": round(changed * RETAG_CHANGE_BYTES), "seconds": changed * RETAG_SECONDS}

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
                document, item = documents[one.text][one.version], kinds_of[one.text] != PLAIN
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
                               {text for text, one in held.items() if one["kind"] != PLAIN}, erased)
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
        its kind's schema. The conversations that quote a plain text it
        reaches are cut too (quote_cuts), as items it names are."""
        needles = _needles(request)
        found = self._kinds(sorted({one.text for one in needles}))
        kept = [at for at, one in enumerate(needles) if one.text in found]  # a text removed before has nothing left
        held = [needles[at] for at in kept]
        items = {text for text, kind in found.items() if kind != PLAIN}
        schemas = dict(self.db.reader().execute("SELECT id, schema FROM kinds").fetchall())
        reader = self.reader()
        documents = self._versions(sorted(found))
        sizes = {(one["id"], one["version"]): one["bytes"] for one in self._rows(
            "SELECT id, version, bytes FROM content_lists WHERE id IN (SELECT value FROM json_each(?))",
            (_json(sorted(found)),))}
        cut, removed, passages = erasure.carried_out(documents, held, items)
        for text, versions in self.quote_cuts(documents, cut, [(one, value) for one, value, _ in passages], items,
                                              set(removed)).items():
            cut.setdefault(text, {}).update(versions)
            found[text] = "conversation"
            items.add(text)
        if write:  # a cut version may be the one a text reads now, which changes what its counts hold
            self._choose(write[0], sorted(cut), "cut")
            sized = counts.measure(write[0], "temp.cut", counts.TEXT_SIZED)
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
                    _resize(con, text, version, length, size)
                    if found[text] == "conversation":
                        _write_quotes(con, text, version, value)
        # needle: the place among the request's needles of the first needle the passage covers; named before the
        # parts go, since a passage of a text removed whole is read from them
        named = [{**{name: value.hex() if isinstance(value, bytes) else value
                     for name, value in erasure.named(passage).items()},
                  "at": at, "needle": kept[merged.needles[0]]} for merged, passage, at in passages]
        if write:
            con, _ = write
            now = counts.measure(con, "temp.cut", counts.TEXT_SIZED)
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

    def quote_cuts(self, documents: dict, cut: dict, passages: list[tuple], items: set[str], removed: set[str]
                   ) -> dict[str, dict[int, str]]:
        """What an erasure leaves of the conversations that quote a plain
        text it reaches (wpt_depot.erasure.cut_quotes): {id: {version: its
        cut content as JSON text}} for each version it changes. documents:
        every content version of the texts it names, before the cut, as
        carried_out read them; cut: what carried_out left of those it cut,
        a conversation it names being cut again from that; passages:
        (wpt_depot.erasure.Passage, its text) of each passage; items: the
        ids among its texts that are items, which no excerpt quotes;
        removed: the texts it removes whole, whose own conversations go."""
        reached = sorted({passage.text for passage, _ in passages if passage.text not in items})
        quoting = self.quoting(reached, removed)
        for text, versions in quoting.items():
            for version in versions:
                if version in cut.get(text, {}):
                    versions[version] = cut[text][version]
        kept = {text for text in reached if text not in removed}
        erased = {text: {version: erasure.erased_places(pieces, len(documents[text][version]))
                         for version, pieces in cut.get(text, {}).items()} for text in kept}
        removing = [value for passage, value in passages if passage.text in kept]
        return erasure.cut_quotes(quoting, {text: documents[text] for text in kept}, erased, removed & set(reached),
                                  removing)

    def put_cut_items(self, con, number: int, cut: dict[str, dict[int, str]]):
        """Write in the change number (con, its writer connection) the cut
        content versions of items ({id: {version: its content as JSON
        text}}), each checked against its kind's schema (BadContent): its
        parts the depot lacks, and the version listing its parts; then the
        parts of those items no version lists any longer go."""
        kinds_of = self._kinds(sorted(cut))
        schemas = dict(con.execute("SELECT id, schema FROM kinds").fetchall())
        for text, versions in cut.items():
            for version, value in sorted(versions.items()):
                kinds.check_content(kinds_of[text], schemas[kinds_of[text]], value)
                new = list(parted(value))
                con.executemany("INSERT OR IGNORE INTO parts (id, hash, change, value) VALUES (?, ?, ?, ?)",
                                [(text, one.hash, number, pack(one.value)) for one in new])
                length, size = sum(one.length for one in new), sum(one.bytes for one in new)
                con.execute("UPDATE content_lists SET parts = ?, length = ?, bytes = ? WHERE id = ? AND version = ?",
                            (_parts_json([(one.hash, one.length) for one in new]), length, size, text, version))
                _resize(con, text, version, length, size)
                if kinds_of[text] == "conversation":
                    _write_quotes(con, text, version, value)
        _drop_unlisted(con, sorted(cut))

    def quoting(self, texts: list[str], skip: set[str]) -> dict[str, dict[int, str]]:
        """{id: {version: its content as JSON text}}, every content version,
        of each conversation of the depot, but those of skip, one of whose
        versions quotes one of the texts (wpt_depot.quotes), found through
        the table quotes and its index quotes_text, a conversation at a
        time."""
        found, reader = {}, self.reader()
        if not texts:
            return found
        held = [one for (one,) in self.db.reader().execute(
            "SELECT DISTINCT conversation FROM quotes WHERE text IN (SELECT value FROM json_each(?)) "
            "ORDER BY conversation", (_json(sorted(set(texts))),)).fetchall() if one not in skip]
        for one in self._rows("SELECT id, version, parts FROM content_lists WHERE id IN (SELECT value FROM json_each(?)) "
                              "ORDER BY id, version", (_json(held),)):
            found.setdefault(one["id"], {})[one["version"]] = str(Parted(reader, one["id"], one["parts"]))
        return found

    def _kinds(self, texts: list[str]) -> dict[str, str]:
        """{id: its kind} of the texts named that the depot holds content of."""
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
            found.setdefault(one["id"], {})[one["version"]] = document if one["kind"] == PLAIN else str(document)
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

    # Tokenizer recipes and their vocabularies

    def new_tokenizer(self, name: str, snapshot: str, kinds: Sequence[str] | None, sample_bytes: int,
                      vocab_size: int, note: str | None = None) -> str:
        """A tokenizer recipe at version 1: its vocabulary trains on a
        sample of at most sample_bytes bytes of the texts of the kinds named
        (SAMPLE_KINDS, plain when None) of a snapshot (its id, or the name of
        one of this depot's own snapshots), up to vocab_size tokens: its id.
        Refused for a name that is taken or not one (TOKENIZER_NAME), a
        sample or a size out of range, a kind a sample cannot take; NotFound
        for a snapshot the depot does not hold."""
        row = self._recipe_row(snapshot, kinds, sample_bytes, vocab_size, note)
        if not isinstance(name, str) or not TOKENIZER_NAME.fullmatch(name):
            raise Refused("a tokenizer recipe's name is lowercase a-z, 0-9, '.', '_' and '-', starting with a letter "
                          "or a digit")
        with self._write("tokenizer") as (con, number):
            if self._rows("SELECT id FROM tokenizers WHERE name = ?", (name,)):
                raise Refused(f"the tokenizer recipe name {name} is taken")
            recipe = self._id(self._take(con, "tokenizers"))
            _insert(con, "tokenizers_revisions", [{"id": recipe, "revision": 1, "change": number, "latest": 1,
                                                   "name": name, "version": 1, **row,
                                                   "sample": _json(row["sample"])}])
            return recipe

    def edit_tokenizer(self, recipe: str, snapshot: str, kinds: Sequence[str] | None, sample_bytes: int,
                       vocab_size: int, note: str | None = None) -> int:
        """Change the tokenizer recipe of that id as a new revision, its id
        and name kept, with the checks of new_tokenizer: its version, which
        it answers, goes up by one when the revision changes what trains its
        vocabulary (the snapshot and kinds of its sample, or the dataset an
        older revision named, the sample's bytes, the vocabulary's size),
        and stays for a change of its note alone. A change that changes
        nothing writes nothing. NotFound without such a recipe;
        BelongsToUpstream for a recipe of the upstream."""
        old = self.tokenizer_by_id(recipe)
        self._refuse_upstream("tokenizer recipes", [recipe])
        row = self._recipe_row(snapshot, kinds, sample_bytes, vocab_size, note)
        if all(old[key] == value for key, value in row.items()):
            return old["version"]
        version = old["version"] + (any(old[key] != row[key] for key in TRAINS_VOCABULARY))
        with self._write("edit tokenizer") as (con, number):
            self._revise(con, "tokenizers_revisions", [recipe], number,
                         {**{key: f":{key}" for key in row}, "version": ":version"},
                         {**row, "sample": _json(row["sample"]), "version": version})
        return version

    def _recipe_row(self, snapshot: str, kinds: Sequence[str] | None, sample_bytes: int, vocab_size: int,
                    note: str | None) -> dict:
        """The columns of a tokenizer recipe's revision, from what a person
        gives, checked (new_tokenizer): sample {snapshot: its id, kinds},
        no depot and an empty dataset, the columns older revisions name
        what they train on by."""
        whole = (lambda value: isinstance(value, int) and not isinstance(value, bool))
        if not whole(sample_bytes) or sample_bytes < 1:
            raise Refused("a tokenizer recipe's sample holds a whole number of bytes above 0")
        if not whole(vocab_size) or vocab_size < VOCABULARY_BASE:
            raise Refused(f"a vocabulary holds at least {VOCABULARY_BASE} tokens (256 bytes, the end of a text and "
                          "256 reserved ids)")
        _filled(snapshot=snapshot)
        kinds = list(dict.fromkeys(kinds or [PLAIN]))
        other = [one for one in kinds if one not in SAMPLE_KINDS]
        if other:
            raise Refused(f"a vocabulary's sample takes texts of kind {', '.join(SAMPLE_KINDS)}, not "
                          f"{', '.join(other)}")
        if note is not None and not note.strip():
            note = None
        return {"depot": None, "dataset": "", "sample": {"snapshot": self._snapshot_id(snapshot.strip()),
                                                          "kinds": kinds},
                "sample_bytes": sample_bytes, "vocab_size": vocab_size, "note": note}

    def _snapshot_id(self, snapshot: str) -> str:
        """The id of the snapshot of that id, or else of this depot's own
        snapshot of that name. NotFound without one."""
        if self._rows("SELECT id FROM snapshots WHERE id = ?", (snapshot,)):
            return snapshot
        return self.snapshot(snapshot)["id"]

    def tokenizer(self, name: str) -> dict:
        """The current revision of the tokenizer recipe named so, with
        needs_snapshot: whether it names no sample (a revision written
        before snapshots, which named a dataset), so it trains nothing until
        an edit names a snapshot. NotFound without one."""
        found = self._rows("SELECT * FROM tokenizers WHERE name = ?", (name,))
        if not found:
            raise NotFound(f"no tokenizer recipe {name}")
        return _recipe_shown(found[0])

    def tokenizer_by_id(self, recipe: str) -> dict:
        """The current revision of the tokenizer recipe of that id, as tokenizer gives it. NotFound without one."""
        found = self._rows("SELECT * FROM tokenizers WHERE id = ?", (recipe,))
        if not found:
            raise NotFound(f"no tokenizer recipe {recipe}")
        return _recipe_shown(found[0])

    def tokenizers(self) -> list[dict]:
        """Every tokenizer recipe's current revision, by name, as tokenizer
        gives it, with trained: whether its version has a vocabulary."""
        return [_recipe_shown(one) for one in self._rows(
            "SELECT t.*, EXISTS (SELECT 1 FROM vocabularies v WHERE v.recipe = t.id "
            "AND v.version = t.version) AS trained FROM tokenizers t ORDER BY t.name")]

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

    # Snapshots (wpt_depot.snapshots)

    def create_snapshot(self, name: str, note: str | None = None, progress=None, read=None,
                        measured: dict | None = None) -> dict:
        """A new snapshot of the texts admitted now, called name: its row (snapshots.create)."""
        return snapshots.create(self, name, note, progress, read, measured=measured)

    def snapshots(self) -> list[dict]:
        """Every snapshot of the depot, without its summary (snapshots.snapshots)."""
        return snapshots.snapshots(self)

    def snapshot(self, name: str, owner: str | None = None) -> dict:
        """The snapshot called name of the depot owner, this one by default (snapshots.snapshot)."""
        return snapshots.snapshot(self, name, owner)

    def snapshot_texts(self, snapshot: str, kinds: Sequence[str] | None = None, after: str | None = None,
                       limit: int = 1000) -> dict:
        """A page of the texts of the snapshot whose id is snapshot (snapshots.texts)."""
        return snapshots.texts(self, snapshot, kinds, after, limit)

    def snapshot_count(self, snapshot: str, kinds: Sequence[str] | None = None) -> dict:
        """What the snapshot whose id is snapshot holds of the kinds named, from its summary (snapshots.count)."""
        return snapshots.count(self, snapshot, kinds)

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
                                                f"WHERE {plan.candidates}")
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
        wpt_depot.counts.BY_STATUS by it (counts.moving, one pass read
        before the statuses are written)."""
        delta = counts.moving(con, "temp.chosen", policy.combined_sql("t.status", status, starts_over))
        new = policy.combined_sql("status", status, starts_over)
        con.execute(f"UPDATE texts_revisions SET status = {new} WHERE rowid IN (SELECT t.rowid FROM temp.chosen c "
                    f"CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest) AND status IS NOT {new}")
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
        before = (f"SELECT t.rowid FROM temp.revised c CROSS JOIN {table} t ON t.id = c.id "
                  "AND t.revision = c.revision")
        con.execute(f"UPDATE {table} SET latest = 0 WHERE rowid IN ({before})")
        given = {"revision": "t.revision + 1", "change": ":change", "latest": "1",
                 **({"decisions": "'[]'", "tags": "'[]'"} if table == "texts_revisions" else {}), **sets}
        names = self._column_names(table)
        con.execute(f"INSERT INTO {table} ({', '.join(names)}) SELECT "
                    f"{', '.join(given.get(name, f't.{name}') for name in names)} "
                    f"FROM temp.revised c CROSS JOIN {table} t ON t.id = c.id AND t.revision = c.revision {joined}",
                    {"change": number, **(params or {})})
        if table == "texts_revisions":  # the new revision took the copy of the text's tags
            con.execute(f"UPDATE texts_revisions SET current_tags = NULL WHERE rowid IN ({before})")
        con.execute("DELETE FROM temp.revised")

    def _store(self, con, number: int, contents: Sequence[dict]) -> list[tuple[str, int, int]]:
        """Write the content versions a change numbered number adds
        (contents: {id, version, kind, source: its plain text or an item's
        JSON, wpt_depot.parts.Source}): the parts of each that its text does
        not hold yet, read and written a few at a time, then its row of
        content_lists. (kind, length, bytes) of each, in order, what a
        revision that reads it keeps (schema.SIZES)."""
        sizes = []
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
            sizes.append((one["kind"], sum(length for _, length, _ in found), sum(size for _, _, size in found)))
            if one["kind"] == "conversation":
                _write_quotes(con, one["id"], one["version"], one["source"])
        return sizes


# what a revision that reads a new content version (:version, its row of content_lists already written) sets, from
# that row (NEW_VERSION_JOIN, joined in Depot._revise): the version, and its kind, length and bytes (schema.SIZES)
NEW_VERSION = {"content_version": ":version", "content_kind": "v.kind", "content_length": "v.length",
               "content_bytes": "v.bytes"}
NEW_VERSION_JOIN = "CROSS JOIN content_lists v ON v.id = t.id AND v.version = :version"


def _resize(con, text: str, version: int, length: int, size: int):
    """The length and bytes an erasure's cut gives a content version of the
    text, copied into every revision of it that reads that version
    (schema.SIZES)."""
    con.execute("UPDATE texts_revisions SET content_length = ?, content_bytes = ? WHERE id = ? AND content_version = ?",
                (length, size, text, version))


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
        from .delta import DeltaFolder
        return DeltaFolder(path).format()
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
    """A web address as an about row keeps it: without a closing slash."""
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


def _recipe_shown(row: dict) -> dict:
    """A tokenizer recipe's row with needs_snapshot (Depot.tokenizer)."""
    return {**row, "needs_snapshot": row["sample"] is None}


def _remove_texts(con, texts: list[str]):
    """Remove the texts whole: every revision, every content version and part, decision rows, the quotes of its
    versions and its rows of taggings (an erasure's part)."""
    if texts:
        for table, column in (("texts_revisions", "id"), ("content_lists", "id"), ("parts", "id"),
                              ("text_decisions", "text"), ("quotes", "conversation"), ("taggings", "text")):
            con.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT value FROM json_each(?))", (_json(texts),))


def _write_quotes(con, conversation: str, version: int, content: str):
    """Write the rows of quotes of a content version of a conversation (its
    content as JSON text): the ids of the texts its excerpts quote, in place
    of those it had."""
    con.execute("DELETE FROM quotes WHERE conversation = ? AND version = ?", (conversation, version))
    con.executemany("INSERT INTO quotes (conversation, version, text) VALUES (?, ?, ?)",
                    [(conversation, version, text) for text in quotes.quoted(json.loads(content))])


def index_quotes(con, rows: list[dict]):
    """Write the rows of quotes of the content versions of conversations
    among rows (rows of content_lists: {id, version, kind, parts}), each
    read from the parts con's depot holds; one whose parts it lacks yet (a
    clone's first copy, which the pull that ends it puts right) is left out."""
    reader = Reader(lambda: con)
    for one in rows:
        if one["kind"] != "conversation":
            continue
        try:
            content = str(Parted(reader, one["id"], json.loads(one["parts"])))
        except LookupError:
            continue
        _write_quotes(con, one["id"], one["version"], content)


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
    if one.kind not in held_kinds:
        raise Refused(f"the depot holds no kind {one.kind}")
    if one.kind == PLAIN:
        if one.text is None or one.content is not None:
            raise Refused("a plain text has a text and no content")
    else:
        if one.text is not None or one.content is None:
            raise Refused(f"a text of kind {one.kind} has its JSON content, and no plain text")
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
