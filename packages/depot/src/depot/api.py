"""What code outside the depot asks of it: a snapshot, its listing, the
content and credits of its texts, and the erasures carried out
(docs/api/depot.md).

Every function takes the depot asked as its address, a str (the path or
file:// address of its SQLite file, as DEPOT names it), and otherwise
takes and gives only values JSON can carry, or a stream of bytes: never a
connection, a row object or a path into the depot's files. So each call
can become a route of the depot's web app without its callers changing.

The depot is opened read-only (depot.depot.Depot.view), once in each
process, and kept while VIEWS others are not asked for after it. The
listing of a snapshot is made once from its sets (depot.order) and
kept until an erasure reaches the depot (depot.snapshots.last_erasure)
or another snapshot is listed, so the walks of one build read the same
listing and make it once.

    about(depot)                                      the depot's about row: its name, upstream and format
    snapshot(depot, snapshot_id, name, owner)         a snapshot's row and summary, by id or by name
    listing(depot, snapshot, after, spread)           its texts and pieces in the depot's order, without content
    listing_size(depot, snapshot)                     how many entries its listing holds, and its long texts
    sampled(depot, snapshot, points)                  some of its texts spread over its numbers, in order
    texts(depot, snapshot, texts)                     some of its texts, as its listing reads them
    holds(depot, snapshot, texts)                     which texts its sets hold
    content(depot, text, revision, piece, span)       a text's content, a piece or a span, as UTF-8 bytes
    contents(depot, asked)                            the content of each entry asked, in the order asked
    matched(depot, text, revision, selectors, pieced) where quotes match a text, and the pieces they lie in
    quoted(depot, text, revision)                     the texts a conversation quotes
    credits(depot, texts)                             the credits of texts at given revisions
    licenses(depot, ids), notices(depot, ids)         the licenses and notices those credits name
    erasures(depot, since)                            the erasures carried out after a mark, and the mark now
    vocabularies(depot)                               the vocabularies its tokenizer recipes trained
    vocabulary_texts(depot, recipe, version, offset)  the texts one of them read, in order
"""

import json
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence

from . import ids, order, pieces, quotes, snapshots
from .depot import Depot, location
from .order import AT_BOUND
from .parts import Parted, scan

VIEWS = 8  # depots this process keeps open, read-only, the last asked
CREDITS = ("kind", "title", "authors", "address", "licenses", "tags", "notice", "scores")  # what credits gives

_views: OrderedDict = OrderedDict()  # (file, device, inode): Depot, the VIEWS asked last
_listed: list = []  # [(Depot, snapshot id, last erasure, Order)]: the listing made last
_lock = threading.Lock()


def _view(depot: str) -> Depot:
    """The depot at that address, opened read-only once in this process. A
    file replaced at the same path opens again."""
    path = location(depot)
    try:
        found = path.stat()
        key = (str(path), found.st_dev, found.st_ino)
    except OSError:
        return Depot.view(depot)  # no such file: NotADepot
    with _lock:
        held = _views.get(key)
        if held is not None:
            _views.move_to_end(key)
            return held
    held = Depot.view(depot)
    with _lock:
        _views[key] = held
        while len(_views) > VIEWS:
            _views.popitem(last=False)
    return held


def _order(view: Depot, snapshot: str) -> order.Order:
    """The order of the snapshot's texts (depot.order.of), made again
    only once an erasure reached the depot since it was made."""
    mark = snapshots.last_erasure(view)
    with _lock:
        if _listed and _listed[0][0] is view and _listed[0][1] == snapshot and _listed[0][2] == mark:
            return _listed[0][3]
        _listed.clear()  # its memory freed before the next is made
    made = order.of(view, snapshot)
    with _lock:
        _listed[:] = [(view, snapshot, mark, made)]
    return made


def _entry(one: order.Entry) -> dict:
    return {"key": str(one.key), "owner": one.owner, "number": one.number, "id": one.id, "piece": one.piece,
            "name": one.id if one.piece is None else pieces.name(one.id, one.piece), "revision": one.revision,
            "kind": one.kind, "tags": json.loads(one.tags), "bytes": one.bytes}


def about(depot: str) -> dict:
    """The depot's about row: {name (which begins its ids), maintainer,
    takedown_contact, upstream (the web address of the depot it follows,
    None for none), format, revision, change}."""
    return _view(depot).about()


def snapshot(depot: str, snapshot_id: str | None = None, *, name: str | None = None,
             owner: str | None = None) -> dict:
    """The snapshot whose id is snapshot_id, or the one called name of the
    depot owner (this depot by default): {id, owner, name, change,
    at_change (the change number it reads), made_at, origin, note, texts,
    characters, bytes, digest, summary (depot.snapshots)}. NotFound
    without one."""
    view = _view(depot)
    if snapshot_id is not None:
        return snapshots.snapshot_by_id(view, snapshot_id)
    return snapshots.snapshot(view, name, owner)


def listing(depot: str, snapshot: str, after: Sequence | None = None, spread: int | None = None
            ) -> Iterator[dict]:
    """The texts of the snapshot whose id is snapshot, in the depot's order
    (depot.order), a long plain text as its pieces, without content:
    for each, {key (its export key, a signed 64-bit integer written in
    decimal as a string, which a JSON reader of doubles would round),
    owner, number, id, piece (its number, None for a whole text), name (the
    text's id, or the piece's name), revision (as of its bound), kind, tags
    (as of its bound), bytes (a piece's: its share of its text's)}. after:
    the place [key, owner, number, piece] of an entry (piece -1 for a whole
    text) the listing goes on after, its key as listing gives it. spread:
    at most that many places spread evenly over the listing instead of
    every one, as an estimate reads it. A text an erasure removed whole is
    passed by. NotFound without such a snapshot."""
    made = _order(_view(depot), snapshot)
    if spread is not None:
        made = made.subset(made.spread(spread))
    if after is not None:
        after = [int(after[0]), *after[1:]]
    for one in order.walk(made, after):
        yield _entry(one)


def listing_size(depot: str, snapshot: str) -> dict:
    """What the listing of the snapshot holds: {entries (its places, a text
    an erasure removed whole since counting), long_texts (the plain texts
    that come as pieces), pieces (theirs, in all), piece_characters (the
    length past which a plain text comes as pieces, about each piece's)}."""
    made = _order(_view(depot), snapshot)
    return {"entries": len(made), "long_texts": len(made.long),
            "pieces": sum(one.count for one in made.long.values()), "piece_characters": pieces.LENGTH}


def sampled(depot: str, snapshot: str, points: int) -> Iterator[dict]:
    """At most points texts of the snapshot, each whole, spread over its
    numbers and given in the depot's order as listing gives them, the same
    for the same snapshot, without the export key of every text: what an
    estimate reads of a big snapshot (depot.order.sampled)."""
    for one in order.walk(order.sampled(_view(depot), snapshot, points)):
        yield _entry(one)


def texts(depot: str, snapshot: str, texts: Sequence[str]) -> dict[str, dict]:
    """{id: {revision, kind, tags, bytes}} of the texts named that the
    snapshot holds, each as its listing reads it whole: at its revision as
    of the bound of the lowest part of the snapshot that holds it. A text
    the snapshot does not hold, or that an erasure removed, is left out."""
    view = _view(depot)
    parts = snapshots.sets(view, snapshot)
    reader = view.db.reader()
    found = {}
    for text in texts:
        owner, number = ids.owner(text), ids.number(text)
        if number is None:
            continue
        for _, named, bound, bitmap in parts:
            if named == owner and number in bitmap:
                rows = reader.execute(AT_BOUND, (json.dumps([[text, bound]]),)).fetchall()
                if rows:
                    _, revision, kind, _, size, tags = rows[0]
                    found[text] = {"revision": revision, "kind": kind, "tags": json.loads(tags or "[]"),
                                   "bytes": size}
                break
    return found


def holds(depot: str, snapshot: str, texts: Sequence[str]) -> list[str]:
    """Those of the texts named that a set of the snapshot holds, any part
    of it, in the order given, whether or not an erasure removed them since."""
    held = {}
    for _, owner, _, bitmap in snapshots.sets(_view(depot), snapshot):
        held[owner] = held[owner] | bitmap if owner in held else bitmap
    found = []
    for text in texts:
        owner, number = ids.owner(text), ids.number(text)
        if number is not None and owner in held and number in held[owner]:
            found.append(text)
    return found


def content(depot: str, text: str, revision: int, piece: int | None = None,
            span: Sequence[int] | None = None) -> Iterator[bytes]:
    """The content of the text at that revision as UTF-8 bytes: the whole
    content version it reads, one part at a time, never whole in memory
    (a plain text's characters, an item's JSON); with piece, that piece of
    a long plain text (depot.pieces); with span, [start, end], its
    characters from start to end, excluded. NotFound when the depot no
    longer holds that revision."""
    view = _view(depot)
    parts = view.listed_text(text, revision)["parts"]
    if piece is not None:
        yield pieces.read(Parted(view.reader(), text, parts), piece).encode("utf-8")
        return
    if span is not None:
        yield Parted(view.reader(), text, parts)[span[0]:span[1]].encode("utf-8")
        return
    for _, value in scan(view.reader(), {text: parts}):
        if value is None:
            break
        yield value.encode("utf-8")


def contents(depot: str, asked: Sequence[Sequence]) -> Iterator[bytes]:
    """For each [id, revision, piece] of asked, in order (piece None for a
    whole text, as an entry of the listing names it): its content as UTF-8
    bytes, one value an entry, given as content gives it joined. An entry of
    the listing is never a whole long plain text, which comes as pieces, so
    no value holds more than a piece of one. Each is read once the one
    before it was taken. NotFound when the depot no longer holds one."""
    for text, revision, piece in asked:
        yield b"".join(content(depot, text, revision, piece))


def matched(depot: str, text: str, revision: int, selectors: list[dict], pieced: bool = False
            ) -> list[list[int] | None]:
    """For each quote selector, in order (depot.quotes): None when it
    matches no place or several of the content version the text's revision
    reads, else the pieces its passage lies in when pieced (the text is
    taken as pieces, depot.pieces.holding), [] when not. The text is
    read once for all the selectors, as its parts."""
    view = _view(depot)
    document = Parted(view.reader(), text, view.listed_text(text, revision)["parts"])
    return [None if len(places) != 1 else
            pieces.holding(document, places[0], places[0] + len(selector["exact"])) if pieced else []
            for selector, places in zip(selectors, quotes.find_each(document, selectors))]


def quoted(depot: str, text: str, revision: int) -> list[str]:
    """The ids of the texts the conversation text quotes at that revision,
    from the depot's table quotes."""
    return [one for (one,) in _view(depot).db.reader().execute(
        "SELECT q.text FROM texts_revisions t JOIN quotes q ON q.conversation = t.id AND q.version = t.content_version "
        "WHERE t.id = ? AND t.revision = ?", (text, revision)).fetchall()]


def credits(depot: str, texts: Sequence[Sequence]) -> list[dict]:
    """For each [id, revision] of texts, in order: its credits at that
    revision, {kind, title, authors, address, licenses, tags (the text's
    now: a revision holds none), notice (its id), scores}. NotFound when
    the depot no longer holds one."""
    view = _view(depot)
    found = []
    for text, revision in texts:
        row = view.listed_text(text, revision)
        found.append({key: row[key] for key in CREDITS})
    return found


def licenses(depot: str, named: Sequence[str]) -> list[dict]:
    """{id, name, text} of each license named that the depot holds, in the order given."""
    found = {row["id"]: row for row in _view(depot).rows(
        "SELECT id, name, text FROM licenses WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(list(named)),))}
    return [found[one] for one in named if one in found]


def notices(depot: str, named: Sequence[str]) -> list[dict]:
    """{id, text} of each notice named that the depot holds, in the order given."""
    found = {row["id"]: {"id": row["id"], "text": row["text"]} for row in _view(depot).rows(
        "SELECT id, text FROM notices WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(list(named)),))}
    return [found[one] for one in named if one in found]


def erasures(depot: str, since: int = 0) -> dict:
    """The erasures the depot has carried out after the mark since (0:
    every one), and the mark to ask from next: {mark, erasures: in the
    order of their ids, each {id, change (the change number of the revision
    that carried it out, in the numbering of the depot that owns it),
    texts (the ids of the texts it removed or cut, each once, in the order
    its request names them)}}. The mark rises with every erasure revision
    the depot gains, written here or brought by a pull
    (depot.snapshots.last_erasure); an erasure counts after since when
    the first of its revisions that holds what it erased came after it. A
    change number cannot be the mark: a pull brings revisions that keep
    their upstream's change numbers, which can be lower than the ones this
    depot has given already."""
    view = _view(depot)
    mark = snapshots.last_erasure(view)
    rows = view.rows(
        "SELECT e.id, c.change, e.needles FROM erasures e JOIN erasures_revisions c ON c.rowid = (SELECT min(r.rowid) "
        "FROM erasures_revisions r WHERE r.id = e.id AND r.erased IS NOT NULL) "
        "WHERE e.erased IS NOT NULL AND c.rowid > ? AND c.rowid <= ? ORDER BY e.id", (since, mark))
    return {"mark": mark, "erasures": [
        {"id": row["id"], "change": row["change"],
         "texts": list(dict.fromkeys(needle["text"] for needle in row["needles"]))} for row in rows]}


# the vocabularies the depot's tokenizer recipes trained, kept unwritten in its legacy tables
_VOCABULARIES = (
    "SELECT v.recipe, v.version, v.sha256, v.merges, v.texts, v.bytes, v.made_at, r.name, r.dataset, r.sample, "
    "r.sample_bytes, r.vocab_size FROM vocabularies v JOIN tokenizers_revisions r ON r.id = v.recipe "
    "AND r.version = v.version AND r.revision = (SELECT max(revision) FROM tokenizers_revisions "
    "WHERE id = v.recipe AND version = v.version) ORDER BY r.name, v.version")
_VOCABULARY_TEXTS = ("SELECT depot, text, change FROM vocabulary_texts WHERE recipe = ? AND version = ? "
                     "ORDER BY position LIMIT ? OFFSET ?")


def vocabularies(depot: str) -> list[dict]:
    """Every vocabulary the depot's tokenizer recipes trained, which its
    tables tokenizers_revisions, vocabularies and vocabulary_texts keep
    unwritten, in the order of their recipes' names and versions: {recipe
    (its id), name, version, sha256, merges (the vocabulary file, as text),
    texts, bytes, made_at, dataset, sample, sample_bytes, vocab_size}."""
    found = []
    for row in _view(depot).rows(_VOCABULARIES):
        sample = row["sample"]
        found.append({**row, "sample": json.loads(sample) if isinstance(sample, str) else sample})
    return found


def vocabulary_texts(depot: str, recipe: str, version: int, offset: int = 0, limit: int = 100_000) -> list[dict]:
    """The texts the vocabulary of that recipe (its id) and version read, in
    the order it read them, from the place offset on, at most limit: each
    {depot (the depot's name for one of its own texts), id, change}."""
    view = _view(depot)
    own = view.name()
    return [{"depot": row["depot"] or own, "id": row["text"], "change": row["change"]}
            for row in view.rows(_VOCABULARY_TEXTS, (recipe, version, limit, offset))]
