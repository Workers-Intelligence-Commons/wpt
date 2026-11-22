"""A depot that follows another, its upstream, as a Git clone follows its
remote: clone copies a depot, and pull fetches its upstream's updates,
admitted erasures included. The upstream is named by the web address of
its web app (`wpt depot web`), whose follow routes (FEED, served by the
functions of this module) only read: a clone reads nothing but those
answers, so only the upstream's current state travels and an erased
passage never leaves the upstream.

A clone holds every row of the upstream's tables that travel (TABLES):
every revision with the change number it was made at, every text's
content and tags (its rows of taggings, each picked by the change that last
wrote it, so a tag a text left since the last pull travels again with its
end), every snapshot with its sets, and every bitmap checked against
its snapshot's digest (wpt_depot.snapshots) before it is kept. The tables
of libraries and datasets an older format left (schema.LEGACY) do not
travel. Its about row is its own:
its own name, which begins the ids of the rows it adds (the upstream's
rows keep the upstream's ids, so the two never collide), its maintainer
and takedown contact, and the upstream's web address, at a change number
past every one it received; its own changes number on from there. Its
pulls table records the clone as its first pull (wpt_depot.schema). A
clone is written at the code's format, and clone and pull read only an
upstream of that format (OtherFormat): an upstream of an older format is
upgraded first, then its clones.

A clone is read-only on the rows its upstream owns, those whose id does
not carry the clone's name (wpt_depot.ids): every operation that would
change one is refused (wpt_depot.depot), so only the upstream changes them.
The clone's own rows stay fully changeable, and its own snapshots may
hold upstream texts. Snapshot names are unique among the snapshots of one
owner, so a clone's own "2026-11" stands beside its upstream's.

What travels is paged, so no answer holds more than PAGE_ROWS rows or about
PAGE_BYTES bytes and a copy of any size never sits whole in memory: the
rows of a table in the order of its key, from the key after which the last
page ended, and the parts a page of content versions lists that the
follower lacks, at most PARTS_CHARACTERS characters and PARTS_KEYS parts
of them per request.
Rows are chosen by change number: for each depot whose rows the upstream
holds (its own, and those its own pulls brought, Depot.bounds), the rows of
that depot's numbering made after what the follower holds of it and up to
what the upstream's head holds (a snapshot's set by the change of its
snapshot, whose owner it shares). The licenses, kinds and pulls tables
travel whole. A row of the same key (an id and a revision or a content
version) is replaced by the upstream's, but a snapshot or a set the
follower holds, which never changes, is kept. A column of bytes (BLOBS)
travels as base64 text. Which revision of an id is the current one is
worked out again for every id a page brings.

A clone is written as `<to>.cloning`, with what it has copied so far in
`<to>.cloning.json`, so a clone cut short goes on from its last page when
run again; once every table is in, it is renamed to `to` and pulls once,
which brings what the upstream changed meanwhile.

A pull is one change of the follower, numbered past every change number
the upstream's head holds, which holds the depot's write lock while it
reads the upstream. It records in pulls its own change number and the
upstream's change number it brought the depot up to. An erasure the
upstream carried out since the last pull is carried out here: the texts it
removed whole are removed with every revision, and the content versions
of the texts it
cut are read again from the upstream with the parts they list, each part
no version lists any longer dropped. So are the conversations of the
upstream that quote a text it reached, which the upstream cut
(wpt_depot.erasure.cut_quotes); the depot's own conversations that quote
one are cut here, every passage the erasure erased found by its length
and its public hash. An erasure the upstream carries out
while the pull reads it (a part asked for is gone, or the head has moved
past an erasure the pull did not bring) makes the pull start over, up to
TRIES times, writing nothing until it ends whole.
"""

import base64
import binascii
import json
import urllib.error
from collections.abc import Iterator
from pathlib import Path

from wpt_utils import web

from . import counts, ids, policy, snapshots
from .database import Database, now
from .depot import About, Depot, _drop_unlisted, _json, _remove_texts, index_quotes, location
from .errors import BadQuery, DepotExists, NoUpstream, OtherFormat, Refused, UpstreamUnreachable
from .erasure import cut_hashed
from .kinds import PLAIN
from .parts import Parted
from .schema import FORMAT, KEYS, REVISED, WITHOUT_ROWID, sizes_sql, tags_sql

PAGE_ROWS = 2000  # rows in one page at most
PAGE_BYTES = 16 << 20  # bytes of rows in one page, about
PARTS_CHARACTERS = 32 << 20  # characters of parts asked for in one request, about
PARTS_KEYS = 10_000  # parts named in one request at most
TIMEOUT = 100  # seconds a request may take, as the web server in front of the upstream allows
TRIES = 3  # how many times a pull that met an erasure under way starts over
WHOLE = ("licenses", "kinds", "pulls")  # the tables that travel whole
# the tables that travel by change number, in the order they are copied: the column that names the depot that
# owns a row, and the condition on its change numbers (t: the table), {s} and {u} the bounds of its owner's range
_CHANGED = "t.change > {s} AND t.change <= {u}"
# a snapshot's set has no change of its own: it is chosen by the change of its snapshot, whose owner it shares
_SNAPSHOT_CHANGED = "EXISTS (SELECT 1 FROM snapshots s WHERE s.id = t.snapshot AND s.change > {s} AND s.change <= {u})"
FOLLOWED = {
    "notices": ("id", _CHANGED), "decisions": ("id", _CHANGED), "content_lists": ("id", _CHANGED),
    "texts_revisions": ("id", _CHANGED), "taggings": ("text", _CHANGED), "text_decisions": ("decision", _CHANGED),
    "erasures_revisions": ("id", _CHANGED), "tokenizers_revisions": ("id", _CHANGED),
    "vocabularies": ("recipe", _CHANGED), "vocabulary_texts": ("recipe", _CHANGED),
    "snapshots": ("id", _CHANGED), "snapshot_sets": ("snapshot", _SNAPSHOT_CHANGED),
}
TABLES = (*WHOLE, *FOLLOWED)
# the followed tables whose rows a pull reads in the order of their change numbers, through the index of their
# column of change (wpt_depot.schema), rather than in the order of their key, then each row by its rowid, or by its
# key for a table without rowid
BY_CHANGE = {"texts_revisions": ("change",), "content_lists": ("change",), "text_decisions": ("change",),
             "taggings": ("change",)}
# the tables whose rows change what a text adds to the depot's counts (wpt_depot.counts), with the column that names
# the text
COUNTED = {"texts_revisions": "id", "content_lists": "id", "text_decisions": "text", "taggings": "text"}
# the columns that hold bytes, which travel as base64 text in a JSON line, by table
BLOBS = {"snapshot_sets": ("bitmap",)}
# the tables whose rows are written once and never changed (wpt_depot.snapshots): a row a follower holds already
# stays as it is
IMMUTABLE = frozenset({"snapshots", "snapshot_sets"})
CLONING, PROGRESS = ".cloning", ".cloning.json"


# The upstream's side: what its web app answers (FEED)


def head(depot: Depot) -> dict:
    """{name, format, change: the depot's change number, bounds: what it
    holds of each depot's numbering at that change (Depot.bounds)}."""
    change = depot.change()
    return {"name": depot.name(), "format": depot.format(), "change": change,
            "bounds": depot.bounds(depot.name(), change)}


def rows(depot: Depot, table: str, ranges: dict | None = None, after: list | None = None,
         texts: list[str] | None = None) -> Iterator[bytes]:
    """One page of the rows of a table, as JSON lines: each row an object,
    then {"next": the key of the last row, or null when the page ends the
    table}. ranges: {depot name: [since, upto]}, the change numbers chosen of
    each depot's rows (FOLLOWED); texts, for content_lists only, the ids of
    texts whose every version is wanted, ranges aside. A pull's page of a
    table of BY_CHANGE is read by change (_rows_by_change), and its next is
    then that reading's place, which the follower sends back as it came.
    BadQuery for a table that does not travel."""
    if table not in TABLES:
        raise BadQuery(f"the table {table} does not travel to a follower")
    if texts is None and table in BY_CHANGE and ranges and any(since > 0 for since, _ in ranges.values()):
        yield from _rows_by_change(depot, table, ranges, after)
        return
    keys = KEYS[table]
    conditions, params = [], {"limit": PAGE_ROWS}
    if after is not None:
        if len(after) != len(keys):
            raise BadQuery(f"a page of {table} starts after a key of {len(keys)} values")
        conditions.append(f"({', '.join(f't.{key}' for key in keys)}) > "
                          f"({', '.join(f':a{at}' for at in range(len(keys)))})")
        params.update({f"a{at}": value for at, value in enumerate(after)})
    if texts is not None:
        if table != "content_lists":
            raise BadQuery("only content_lists is read by the texts it lists")
        conditions.append("t.id IN (SELECT value FROM json_each(:texts))")
        params["texts"] = json.dumps(texts)
    elif table in FOLLOWED:
        owner, condition = FOLLOWED[table]
        params["ranges"] = json.dumps(ranges or {})
        chosen = condition.format(s="(r.value ->> 0)", u="(r.value ->> 1)")
        conditions.append(f"EXISTS (SELECT 1 FROM json_each(:ranges) r WHERE r.key = {ids.owner_sql('t.' + owner)} "
                          f"AND {chosen})")
    where = " AND ".join(conditions) or "true"
    cursor = depot.db.reader().cursor().execute(
        f"SELECT t.* FROM {table} t WHERE {where} ORDER BY {', '.join(f't.{key}' for key in keys)} LIMIT :limit",
        params)
    names = None
    count = size = 0
    last = None
    for one in cursor:
        names = names or [one[0] for one in cursor.description]
        row = dict(zip(names, one))
        line = _line(table, row)
        yield line
        count, size, last = count + 1, size + len(line), [row[key] for key in keys]
        if size >= PAGE_BYTES:
            break
    yield (json.dumps({"next": last if count == PAGE_ROWS or size >= PAGE_BYTES else None}) + "\n").encode()


def _rows_by_change(depot: Depot, table: str, ranges: dict, after: list | None) -> Iterator[bytes]:
    """One page of rows as rows gives it, for a pull (a range that does not
    start at 0) of a table of BY_CHANGE: the rows of each column of change
    in turn, and of each depot's range in turn (in the order of the depots'
    names), read through the index of that column from the place after,
    [column's place, depot's place, change, then the rowid or the key], so
    a page reads only the rows of the index in its range of changes
    whatever the table's size. The index holds every depot's rows by change
    alone, so a depot's range also reads the rows of the other depots
    numbered within it, which its owner test leaves out. BadQuery for a
    place that is not one."""
    columns, owner = BY_CHANGE[table], FOLLOWED[table][0]
    tie = list(KEYS[table]) if table in WITHOUT_ROWID else ["rowid"]
    if after is not None and (len(after) != 3 + len(tie) or not all(isinstance(one, int) for one in after[:3])):
        raise BadQuery(f"a page of {table} read by change starts after a place of {3 + len(tie)} values")
    owners = sorted(ranges)
    con = depot.db.reader()
    found, size = [], 0
    for at, column in enumerate(columns):
        for who, name in enumerate(owners):
            if after is not None and [at, who] < after[:2]:
                continue
            since, upto = ranges[name]
            # a row whose earlier column of change is in the range came in that column's pass
            earlier = "".join(f" AND NOT (t.{one} > :since AND t.{one} <= :upto)" for one in columns[:at])
            place = ""
            params = {"since": since, "upto": upto, "owner": name, "limit": PAGE_ROWS - len(found)}
            if after is not None and [at, who] == after[:2]:
                place = (f" AND (t.{column}, {', '.join(f't.{one}' for one in tie)}) > "
                         f"({', '.join(f':p{n}' for n in range(1 + len(tie)))})")
                params.update({f"p{n}": value for n, value in enumerate(after[2:])})
            cursor = con.cursor().execute(
                f"SELECT {', '.join(f't.{one} AS tie_{n}' for n, one in enumerate(tie))}, t.* FROM {table} t "
                f"WHERE t.{column} > :since AND t.{column} <= :upto{place} "
                f"AND {ids.owner_sql('t.' + owner)} = :owner{earlier} "
                f"ORDER BY t.{column}, {', '.join(f't.{one}' for one in tie)} LIMIT :limit", params)
            names = None
            for one in cursor:
                names = names or [one[0] for one in cursor.description]
                row = dict(zip(names, one))
                place_of = [at, who, row[column], *(row.pop(f"tie_{n}") for n in range(len(tie)))]
                line = _line(table, row)
                yield line
                found.append(place_of)
                size += len(line)
                if size >= PAGE_BYTES:
                    break
            if len(found) >= PAGE_ROWS or size >= PAGE_BYTES:
                yield (json.dumps({"next": found[-1]}) + "\n").encode()
                return
    yield (json.dumps({"next": None}) + "\n").encode()


def _line(table: str, row: dict) -> bytes:
    """The row as a JSON line, each column of BLOBS in base64."""
    for column in BLOBS.get(table, ()):
        if row[column] is not None:
            row[column] = base64.b64encode(row[column]).decode("ascii")
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def parts(depot: Depot, keys: list[list[str]]) -> Iterator[bytes]:
    """The parts named ([text id, hash in hexadecimal]), each a JSON line
    {id, hash, change, size} then its stored value (size bytes, compressed,
    wpt_depot.parts), or {id, hash, missing: true} for one the depot does
    not hold (an erasure cut it)."""
    con = depot.db.reader()
    for text, hash_ in keys:
        found = con.execute("SELECT change, value FROM parts WHERE id = ? AND hash = ?",
                            (text, bytes.fromhex(hash_))).fetchall()
        if not found:
            yield (json.dumps({"id": text, "hash": hash_, "missing": True}) + "\n").encode()
            continue
        change, value = found[0]
        yield (json.dumps({"id": text, "hash": hash_, "change": change, "size": len(value)}) + "\n").encode()
        yield bytes(value)


# The follower's side


class Upstream:
    """The follow routes of the web app at the web address url."""

    def __init__(self, url: str):
        if not url.startswith(("http://", "https://")):
            raise Refused(f"an upstream is the web address of its web app (https://...), not {url!r}")
        self.url = url.rstrip("/")

    def _post(self, route: str, body: dict):
        try:
            return web.urlopen(f"{self.url}/api/follow/{route}", json.dumps(body).encode(),
                              {"Content-Type": "application/json"}, method="POST", timeout=TIMEOUT)
        except urllib.error.HTTPError as error:
            try:
                problems = "; ".join(json.loads(error.read())["problems"])
            except (ValueError, KeyError, TypeError):
                problems = error.reason
            raise UpstreamUnreachable(f"the upstream {self.url} answered {error.code}: {problems}") from None
        except (urllib.error.URLError, OSError) as error:
            raise UpstreamUnreachable(f"the upstream {self.url} cannot be reached: {error}") from None

    def head(self) -> dict:
        with self._post("head", {}) as answer:
            return json.loads(answer.read())

    def page(self, table: str, ranges: dict | None, after: list | None,
             texts: list[str] | None = None) -> tuple[list[dict], list | None]:
        """(the rows of one page, the key the next page starts after, None after the last)."""
        body = {"table": table, "ranges": ranges, "after": after}
        if texts is not None:
            body["texts"] = texts
        found = []
        try:
            with self._post("rows", body) as answer:
                for line in answer:
                    found.append(json.loads(line))
        except (OSError, ValueError) as error:
            raise UpstreamUnreachable(f"the upstream {self.url} broke off a page of {table}: {error}") from None
        if not found or set(found[-1]) != {"next"}:
            raise UpstreamUnreachable(f"the upstream {self.url} sent a page of {table} cut short")
        return found[:-1], found[-1]["next"]

    def parts(self, keys: list[tuple[str, bytes]]) -> Iterator[tuple[str, bytes, int, bytes | None]]:
        """(text id, hash, change, stored value or None when the upstream no longer holds it) of each part named."""
        try:
            with self._post("parts", {"keys": [[text, hash_.hex()] for text, hash_ in keys]}) as answer:
                for _ in keys:
                    header = json.loads(answer.readline())
                    if header.get("missing"):
                        yield header["id"], bytes.fromhex(header["hash"]), 0, None
                        continue
                    value = answer.read(header["size"])
                    if len(value) != header["size"]:
                        raise UpstreamUnreachable(f"the upstream {self.url} sent a part cut short")
                    yield header["id"], bytes.fromhex(header["hash"]), header["change"], value
        except (OSError, ValueError) as error:
            raise UpstreamUnreachable(f"the upstream {self.url} broke off sending parts: {error}") from None


class _Again(Exception):
    """An erasure the upstream carried out while it was read: the pull starts over."""


def clone(address: str, to: str, about: About) -> Depot:
    """A new depot at `to` holding what the upstream at the web address
    address holds, following it, with about as its own name, maintainer and
    takedown contact. DepotExists when `to` holds a file; Refused when the
    name is the upstream's or cannot begin an id; OtherFormat for an
    upstream of another format; UpstreamUnreachable when it cannot be read.
    Run again after it was cut short, it goes on where it stopped."""
    ids.check_name(about.name)
    upstream = Upstream(address)
    target = location(to)
    if target.exists():
        raise DepotExists(f"{target} already exists: a clone is a new file")
    working = target.with_name(target.name + CLONING)
    progress_file = target.with_name(target.name + PROGRESS)
    progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else None
    if progress is None or progress["upstream"] != upstream.url or not working.exists():
        top = upstream.head()
        _check_head(top, about.name)
        working.unlink(missing_ok=True)
        _remove_waiting(working)
        progress = {"upstream": upstream.url, "head": top, "after": {}}
        Database(working).create()
    top = progress["head"]
    db = Database(working)
    ranges = {owner: [0, upto] for owner, upto in top["bounds"].items()}
    for table in TABLES:
        after = progress["after"].get(table)
        while after != "done":
            found, following = upstream.page(table, None if table in WHOLE else ranges,
                                             None if after is None else after)
            with db.bulk() as con:
                _put(con, table, found)
                if table == "content_lists":
                    _bring_parts(con, upstream, found, tolerate=True)
                    index_quotes(con, found)
            after = "done" if following is None else following
            progress["after"][table] = after
            _write_progress(progress_file, progress)
    try:
        with db.bulk() as con:
            _check_digests(con)
    except UpstreamUnreachable:  # a run again copies everything anew
        db.close()
        working.unlink(missing_ok=True)
        _remove_waiting(working)
        progress_file.unlink(missing_ok=True)
        raise
    with db.bulk() as con:
        for table in REVISED:
            con.execute(f"UPDATE {table} SET latest = (revision = (SELECT max(r.revision) FROM {table} r "
                        f"WHERE r.id = {table}.id))")
        con.execute(f"UPDATE texts_revisions SET status = {policy.linked_status_sql('texts_revisions.id')} "
                    "WHERE latest")
        counts.recount(con)
        number = _highest(con) + 1
        con.execute("INSERT INTO changes (id, kind, made_at) VALUES (?, 'clone', ?)", (number, now()))
        con.execute("INSERT OR REPLACE INTO pulls (depot, change, upstream, upstream_change) VALUES (?, ?, ?, ?)",
                    (about.name, number, top["name"], top["change"]))
        con.execute("INSERT INTO about_revisions (revision, change, name, maintainer, takedown_contact, upstream, "
                    "format) VALUES (1, ?, ?, ?, ?, ?, ?)", (number, about.name, about.maintainer,
                                                             about.takedown_contact, upstream.url, FORMAT))
    db.close()
    working.rename(target)
    waiting = working.with_name(working.name + ".waiting")
    if waiting.exists():
        waiting.rename(target.with_name(target.name + ".waiting"))
    progress_file.unlink(missing_ok=True)
    Database(target).mark_full_copy()
    depot = Depot.open(str(target))
    pull(depot)
    return depot


def pull(depot: Depot) -> list[str]:
    """Fetch the upstream's updates into depot: the ids of the erasures the
    upstream carried out that the depot now has carried out too.
    NoUpstream when it follows none; OtherFormat for an upstream of another
    format; UpstreamUnreachable when it cannot be read."""
    address = depot.upstream()
    if address is None:
        raise NoUpstream(f"{depot.address} follows no upstream")
    upstream = Upstream(address)
    for _ in range(TRIES):
        try:
            brought = _pull(depot, upstream)
        except _Again:
            continue
        if brought:
            depot.db.checkpoint()
        return brought
    raise UpstreamUnreachable(f"the upstream {upstream.url} carried out erasures while it was read, {TRIES} times "
                              "in a row: pull again later")


def _pull(depot: Depot, upstream: Upstream) -> list[str]:
    top = upstream.head()
    _check_head(top, depot.name())
    name = depot.name()
    last = depot.db.query("SELECT upstream, upstream_change FROM pulls WHERE depot = ? ORDER BY change DESC LIMIT 1",
                          (name,))
    if last and last[0]["upstream"] != top["name"]:
        raise Refused(f"the upstream at {upstream.url} is the depot {top['name']}, not {last[0]['upstream']}, "
                      "the one this depot follows")
    held = depot.bounds(top["name"], last[0]["upstream_change"]) if last else {}
    ranges = {owner: [held.get(owner, 0), upto] for owner, upto in top["bounds"].items()}
    before = _carried_out(depot)
    with depot.db.write("pull", max(top["bounds"].values(), default=0)) as (con, number):
        tracked = _Tracked(con)
        received = []
        for table in TABLES:
            after = None
            while True:
                found, after = upstream.page(table, None if table in WHOLE else ranges, after)
                if table in COUNTED:
                    tracked.add({one[COUNTED[table]] for one in found})
                if table == "snapshots":
                    received += [one["id"] for one in found]
                _put(con, table, found)
                if table == "content_lists":
                    _bring_parts(con, upstream, found)
                    index_quotes(con, found)
                if after is None:
                    break
        _check_digests(con, received)
        brought = sorted(_carried_out(depot) - before)
        if brought:  # before the erasures are carried out here, while the depot holds the texts they reached uncut
            _cut_own_quotes(con, depot, brought, tracked, number)
        for erasure in brought:
            _carry_out(con, depot, upstream, erasure, tracked)
        tracked.settle(depot, number)
        con.execute("INSERT INTO pulls (depot, change, upstream, upstream_change) VALUES (?, ?, ?, ?)",
                    (name, number, top["name"], top["change"]))
        moved = upstream.head()
        if moved["change"] != top["change"]:
            later = {owner: [upto, moved["bounds"].get(owner, upto)] for owner, upto in top["bounds"].items()}
            found, _ = upstream.page("erasures_revisions", later, None)
            if any(one["erased"] is not None for one in found):
                raise _Again()
    return brought


def _check_head(top: dict, name: str):
    if not isinstance(top.get("format"), int) or top["format"] < FORMAT:
        raise OtherFormat(f"the upstream is a depot of format {top.get('format')}, older than the format {FORMAT} "
                          "this wpt follows: upgrade the upstream first (`wpt depot upgrade` where it is kept), then "
                          "clone or pull again")
    if top["format"] > FORMAT:
        raise OtherFormat(f"the upstream is a depot of format {top['format']}, newer than the format {FORMAT} this "
                          "wpt follows: update wpt first, then upgrade this depot (`wpt depot upgrade`)")
    if top["name"] == name:
        raise Refused(f"a clone has a name of its own: {name!r} is the upstream's, and ids carry the name")


def _put(con, table: str, found: list[dict]):
    """The rows of a page into the table, each replacing the row of its key;
    a revision's latest worked out again for every id it names, and a
    past revision of a text losing its copy of the text's tags
    (current_tags, which only the current one holds). Every row
    names the same columns, each one of the table's own: a name the
    upstream sent never reaches the SQL otherwise (UpstreamUnreachable).
    A column of BLOBS comes as base64 and is stored as its bytes; a row of
    a table of IMMUTABLE the depot holds already is kept as it is."""
    if not found:
        return
    if table in REVISED:
        found = [{**one, "latest": 0} for one in found]
    if table in BLOBS:
        found = [{**one, **{column: _bytes(table, one.get(column)) for column in BLOBS[table]}} for one in found]
    names = list(found[0])
    columns = {one[1] for one in con.execute(f"PRAGMA table_info({table})")}
    strange = sorted({name for one in found for name in one} - columns)
    if strange or any(one.keys() != found[0].keys() for one in found):
        raise UpstreamUnreachable(f"the upstream sent rows of {table} whose columns are not the table's "
                                  f"({', '.join(map(repr, strange)) or 'columns that differ from row to row'}): "
                                  "nothing is written")
    con.executemany(f"INSERT OR {'IGNORE' if table in IMMUTABLE else 'REPLACE'} INTO {table} ({', '.join(names)}) "
                    f"VALUES ({', '.join(':' + one for one in names)})", found)
    if table in REVISED:
        touched = _json(sorted({one["id"] for one in found}))
        con.execute(f"UPDATE {table} SET latest = 0 WHERE latest AND id IN (SELECT value FROM json_each(?))",
                    (touched,))
        con.execute(f"UPDATE {table} SET latest = 1 WHERE id IN (SELECT value FROM json_each(?)) AND revision = "
                    f"(SELECT max(r.revision) FROM {table} r WHERE r.id = {table}.id)", (touched,))
        if table == "texts_revisions":  # a revision a newer one took over drops its copy of the text's tags
            con.execute("UPDATE texts_revisions SET current_tags = NULL WHERE id IN (SELECT value FROM json_each(?)) "
                        "AND NOT latest AND current_tags IS NOT NULL", (touched,))


def _bytes(table: str, value) -> bytes | None:
    """The bytes a column of BLOBS sent as base64 holds. UpstreamUnreachable when it is not base64."""
    if value is None:
        return None
    try:
        if not isinstance(value, str):
            raise ValueError(value)
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise UpstreamUnreachable(f"the upstream sent rows of {table} whose bytes are not base64: nothing is "
                                  "written") from None


def _check_digests(con, chosen: list[str] | None = None):
    """Check that the sets of each snapshot the depot holds, or of those
    chosen, match its digest (wpt_depot.snapshots), as an outsider would
    recompute it from the numbers. UpstreamUnreachable for one that does
    not: the upstream sent sets damaged or cut short."""
    where, params = ("", ()) if chosen is None else (" WHERE id IN (SELECT value FROM json_each(?))",
                                                       (_json(sorted(chosen)),))
    for snapshot, digest, at_change in con.execute(f"SELECT id, digest, at_change FROM snapshots{where}",
                                                    params).fetchall():
        sets = [(part, owner, bound, snapshots.FrozenBitMap.deserialize(bitmap)) for part, owner, bound, bitmap in
                con.execute("SELECT part, owner, bound, bitmap FROM snapshot_sets WHERE snapshot = ?", (snapshot,))]
        if snapshots.digest_of(sets, at_change) != digest:
            raise UpstreamUnreachable(f"the upstream sent the snapshot {snapshot} with sets that do not match its "
                                      "digest: nothing of it is kept")


def _bring_parts(con, upstream: Upstream, lists: list[dict], tolerate: bool = False):
    """Fetch the parts the content versions of lists name that the depot
    lacks. _Again for a part the upstream no longer holds (an erasure cut
    it while it was read), unless tolerate (a clone's first copy, which the
    pull that ends it puts right)."""
    wanted, size = [], 0
    seen = set()
    for one in lists:
        for hash_, length in Parted.listed(json.loads(one["parts"])):
            key = (one["id"], hash_)
            if key in seen or con.execute("SELECT 1 FROM parts WHERE id = ? AND hash = ?", key).fetchall():
                continue
            seen.add(key)
            wanted.append(key)
            size += length
            if size >= PARTS_CHARACTERS or len(wanted) == PARTS_KEYS:
                _fetch(con, upstream, wanted, tolerate)
                wanted, size = [], 0
    if wanted:
        _fetch(con, upstream, wanted, tolerate)


def _fetch(con, upstream: Upstream, keys: list[tuple[str, bytes]], tolerate: bool):
    for text, hash_, change, value in upstream.parts(keys):
        if value is None:
            if tolerate:
                continue
            raise _Again()
        con.execute("INSERT OR REPLACE INTO parts (id, hash, change, value) VALUES (?, ?, ?, ?)",
                    (text, hash_, change, value))


def _carried_out(depot: Depot) -> set[str]:
    return {one for (one,) in depot.db.reader().execute("SELECT id FROM erasures WHERE erased IS NOT NULL")}


def _cut_own_quotes(con, depot: Depot, brought: list[str], tracked: "_Tracked", number: int):
    """Cut, in the pull's change number, the depot's own conversations
    that quote a text one of the erasures brought reached, read before the
    erasures are carried out here, so the depot still holds those texts as
    they were before them (wpt_depot.erasure.cut_hashed):
    every passage the erasures erased from a plain text, found by its
    length and its hash_a, leaves their strings of text and each excerpt
    loses every character over one in the quoted text, and every excerpt of
    a text they removed whole keeps no character of it."""
    rows = [(json.loads(needles), json.loads(erased)) for needles, erased in con.execute(
        "SELECT needles, erased FROM erasures WHERE id IN (SELECT value FROM json_each(?))", (_json(brought),))]
    reached = sorted({one["text"] for needles, _ in rows for one in needles})
    gone = {one["text"] for needles, _ in rows for one in needles if one["start"] is None and one["field"] is None}
    erased = [entry for needles, entries in rows for entry in entries
              if needles[entry["needle"]]["start"] is not None and needles[entry["needle"]]["field"] is None]
    quoting = depot.quoting(reached, gone)
    theirs = set(depot.upstream_owned(sorted(quoting)))
    plain = sorted(text for text, kind in depot._kinds(sorted(set(reached) - gone)).items() if kind == PLAIN)
    cut = cut_hashed({text: versions for text, versions in quoting.items() if text not in theirs},
                     depot._versions(plain), erased, gone)
    tracked.add(sorted(cut))
    depot.put_cut_items(con, number, cut)


def _carry_out(con, depot: Depot, upstream: Upstream, erasure: str, tracked: "_Tracked"):
    """Carry out here an erasure the upstream carried out: the texts it
    removed whole go, and the texts it cut are read again, with the
    upstream's conversations that quote one of the texts it reached
    (Depot.quoting, through the table quotes), whose rows of quotes are
    written again from what is read."""
    needles = json.loads(con.execute("SELECT needles FROM erasures WHERE id = ?", (erasure,)).fetchall()[0][0])
    reached = sorted({one["text"] for one in needles})
    removed = sorted({one["text"] for one in needles if one["start"] is None and one["field"] is None})
    cut = sorted({*reached, *depot.upstream_owned(sorted(depot.quoting(reached, set(removed))))} - set(removed))
    tracked.add([*removed, *cut])  # a cut changes the characters and bytes of a version a text may read
    _remove_texts(con, removed)
    if cut:
        for table, column in (("content_lists", "id"), ("quotes", "conversation")):
            con.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT value FROM json_each(?))", (_json(cut),))
        after = None
        while True:
            found, after = upstream.page("content_lists", None, after, texts=cut)
            _put(con, "content_lists", found)
            _bring_parts(con, upstream, found)
            index_quotes(con, found)
            if after is None:
                break
        _drop_unlisted(con, cut)


class _Tracked:
    """The texts a pull changes, in the temporary table followed: what each
    adds to the depot's counts is measured before the pull first changes it
    (add), and once every row is in, settle stores the status of each from
    its decisions and shifts the counts by the difference
    (wpt_depot.counts)."""

    def __init__(self, con):
        self.con = con
        self.before = counts.Counter()
        for table in ("followed", "fresh"):
            con.execute(f"CREATE TEMP TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY)")
            con.execute(f"DELETE FROM temp.{table}")

    def add(self, texts):
        """Measure the texts named that no earlier call named, before they change."""
        self._add("SELECT value AS id FROM json_each(?)", (_json(sorted(texts)),))

    def _add(self, query: str, params: tuple):
        """Measure the texts the query gives (its column id) that no earlier call named, before they change."""
        con = self.con
        con.execute("DELETE FROM temp.fresh")
        con.execute(f"INSERT OR IGNORE INTO temp.fresh (id) SELECT x.id FROM ({query}) x "
                    "WHERE x.id NOT IN (SELECT id FROM temp.followed)", params)
        self.before.update(counts.measure(con, "temp.fresh", counts.FACETS))
        counts.unlist(con, "temp.fresh", counts.LISTED)
        con.execute("INSERT INTO temp.followed SELECT id FROM temp.fresh")

    def settle(self, depot: Depot, number: int):
        con = self.con
        # a carried-out erasure's cut rewrites rows of content_lists the revisions here read (schema.SIZES)
        con.execute(sizes_sql("temp.followed"))
        # a revision the upstream changed in place carries no new change: each text's copy of its tags is set again
        con.execute(tags_sql("SELECT id FROM temp.followed"))
        con.execute(f"UPDATE texts_revisions SET status = {policy.linked_status_sql('texts_revisions.id')} "
                    "WHERE rowid IN (SELECT t.rowid FROM temp.followed f CROSS JOIN texts_revisions t ON t.id = f.id "
                    "AND t.latest)")
        after = counts.measure(con, "temp.followed", counts.FACETS)
        after.subtract(self.before)
        counts.shift(con, after)
        counts.enlist(con, "temp.followed", counts.LISTED)


def _highest(con) -> int:
    """The highest change number any row of the file records."""
    return con.execute("""SELECT max(n) FROM (
        SELECT max(id) AS n FROM changes UNION ALL SELECT max(change) FROM about_revisions
        UNION ALL SELECT max(change) FROM texts_revisions UNION ALL SELECT max(change) FROM content_lists
        UNION ALL SELECT max(change) FROM erasures_revisions UNION ALL SELECT max(change) FROM pulls
        UNION ALL SELECT max(change) FROM decisions UNION ALL SELECT max(change) FROM text_decisions
        UNION ALL SELECT max(change) FROM notices UNION ALL SELECT max(change) FROM tokenizers_revisions
        UNION ALL SELECT max(change) FROM vocabularies UNION ALL SELECT max(change) FROM vocabulary_texts
        UNION ALL SELECT max(change) FROM snapshots UNION ALL SELECT max(change) FROM taggings)""").fetchall()[0][0] or 0


def _write_progress(path: Path, progress: dict):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(progress), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _remove_waiting(working: Path):
    waiting = working.with_name(working.name + ".waiting")
    if waiting.exists():
        for one in waiting.iterdir():
            one.unlink()
        waiting.rmdir()
    for suffix in ("-wal", "-shm"):
        working.with_name(working.name + suffix).unlink(missing_ok=True)

