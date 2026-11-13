"""A depot that follows another, its upstream, as a Git clone follows its
remote: clone copies a depot, and pull fetches its upstream's updates,
admitted erasures included. The upstream is named by the web address of
its web app (`wpt depot web`), whose follow routes (FEED, served by the
functions of this module) only read: a clone reads nothing but those
answers, so only the upstream's current state travels and an erased
passage never leaves the upstream.

A clone holds every row of the upstream's tables but its changes, numbers
and about rows: every revision with the change number it was made at,
every text's content, every membership and pin. Its about row is its own:
its own name, which begins the ids of the rows it adds (the upstream's
rows keep the upstream's ids, so the two never collide), its maintainer
and takedown contact, and the upstream's web address, at a change number
past every one it received; its own changes number on from there. Its
pulls table records the clone as its first pull (wpt_depot.schema). A
clone is written at the code's format, and clone and pull read only an
upstream of that format (OtherFormat).

A clone is read-only on the rows its upstream owns, those whose id does
not carry the clone's name (wpt_depot.ids): every operation that would
change one is refused (wpt_depot.depot), so only the upstream changes them.
The clone's own rows stay fully changeable, and its own libraries may hold
upstream texts.

What travels is paged, so no answer holds more than PAGE_ROWS rows or about
PAGE_BYTES bytes and a copy of any size never sits whole in memory: the
rows of a table in the order of its key, from the key after which the last
page ended, and the parts a page of content versions lists that the
follower lacks, at most PARTS_CHARACTERS characters and PARTS_KEYS parts
of them per request.
Rows are chosen by change number: for each depot whose rows the upstream
holds (its own, and those its own pulls brought, Depot.bounds), the rows of
that depot's numbering made after what the follower holds of it and up to
what the upstream's head holds (a membership by its joining or its leaving
change). The licenses, kinds and pulls tables travel whole. A row of the
same key (an id and a revision or a content version, a membership's
library, text and joining change) is replaced by the upstream's, which
brings a membership that ended. Which revision of an id is the current one
is worked out again for every id a page brings.

A clone is written as `<to>.cloning`, with what it has copied so far in
`<to>.cloning.json`, so a clone cut short goes on from its last page when
run again; once every table is in, it is renamed to `to` and pulls once,
which brings what the upstream changed meanwhile.

A pull is one change of the follower, numbered past every change number
the upstream's head holds, which holds the depot's write lock while it
reads the upstream. It records in pulls its own change number and the
upstream's change number it brought the depot up to. An erasure the
upstream carried out since the last pull is carried out here: the texts it
removed whole are removed with every revision, membership and pin, the
clone's own libraries' included, and the content versions of the texts it
cut are read again from the upstream with the parts they list, each part
no version lists any longer dropped. An erasure the upstream carries out
while the pull reads it (a part asked for is gone, or the head has moved
past an erasure the pull did not bring) makes the pull start over, up to
TRIES times, writing nothing until it ends whole.

A text the upstream refused, or started over (its licenses or its content
changed, wpt_depot.policy), leaves every open library of the depot's own
that holds it, as such a decision made here would: the pull ends those
memberships at its own change number, so its own libraries read at an
earlier change still hold the text. Its locked libraries keep their pins.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from . import ids, policy
from .database import Database, now
from .depot import About, Depot, _drop_unlisted, _json, _remove_texts, location
from .errors import BadQuery, DepotExists, NoUpstream, NotFound, OtherFormat, Refused, UpstreamUnreachable
from .parts import Parted
from .schema import FORMAT, KEYS, REVISED

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
FOLLOWED = {
    "notices": ("id", _CHANGED), "decisions": ("id", _CHANGED), "content_lists": ("id", _CHANGED),
    "texts_revisions": ("id", _CHANGED), "libraries_revisions": ("id", _CHANGED),
    "memberships": ("library", "((t.joined_change > {s} AND t.joined_change <= {u}) "
                               "OR (t.left_change > {s} AND t.left_change <= {u}))"),
    "pins": ("library", _CHANGED), "erasures_revisions": ("id", _CHANGED), "datasets_revisions": ("id", _CHANGED),
}
TABLES = (*WHOLE, *FOLLOWED)
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
    texts whose every version is wanted, ranges aside. BadQuery for a
    table that does not travel."""
    if table not in TABLES:
        raise BadQuery(f"the table {table} does not travel to a follower")
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
        line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        yield line
        count, size, last = count + 1, size + len(line), [row[key] for key in keys]
        if size >= PAGE_BYTES:
            break
    yield (json.dumps({"next": last if count == PAGE_ROWS or size >= PAGE_BYTES else None}) + "\n").encode()


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


def library_change(url: str, library: str) -> int:
    """The change number now of the depot whose web app is at url, after
    checking through its follow routes that it holds the library; a
    dataset naming that library is read at that number. NotFound when it
    does not hold it."""
    upstream = Upstream(url)
    top = upstream.head()
    owner = ids.owner(library)
    if owner not in top["bounds"]:
        raise NotFound(f"no library {library} at {url}")
    ranges, after = {owner: [0, top["bounds"][owner]]}, None
    while True:
        found, after = upstream.page("libraries_revisions", ranges, after)
        if any(row["id"] == library for row in found):
            return top["change"]
        if after is None:
            raise NotFound(f"no library {library} at {url}")



class Upstream:
    """The follow routes of the web app at the web address url."""

    def __init__(self, url: str):
        if not url.startswith(("http://", "https://")):
            raise Refused(f"an upstream is the web address of its web app (https://...), not {url!r}")
        self.url = url.rstrip("/")

    def _post(self, route: str, body: dict):
        request = urllib.request.Request(f"{self.url}/api/follow/{route}", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"}, method="POST")
        try:
            return urllib.request.urlopen(request, timeout=TIMEOUT)
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
            after = "done" if following is None else following
            progress["after"][table] = after
            _write_progress(progress_file, progress)
    with db.bulk() as con:
        for table in REVISED:
            con.execute(f"UPDATE {table} SET latest = (revision = (SELECT max(r.revision) FROM {table} r "
                        f"WHERE r.id = {table}.id))")
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
        touched = set()
        for table in TABLES:
            after = None
            while True:
                found, after = upstream.page(table, None if table in WHOLE else ranges, after)
                _put(con, table, found)
                if table == "content_lists":
                    _bring_parts(con, upstream, found)
                if table == "texts_revisions":
                    touched.update(one["id"] for one in found)
                if after is None:
                    break
        brought = sorted(_carried_out(depot) - before)
        for erasure in brought:
            _carry_out(con, depot, upstream, erasure)
        _leave(con, depot, sorted(touched), number)
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
    if top["format"] != FORMAT:
        raise OtherFormat(f"the upstream is a depot of format {top['format']}, and this wpt follows only one of "
                          f"format {FORMAT}: both are upgraded to the same format first")
    if top["name"] == name:
        raise Refused(f"a clone has a name of its own: {name!r} is the upstream's, and ids carry the name")


def _put(con, table: str, found: list[dict]):
    """The rows of a page into the table, each replacing the row of its key;
    a revision's latest worked out again for every id it names. Every row
    names the same columns, each one of the table's own: a name the
    upstream sent never reaches the SQL otherwise (UpstreamUnreachable)."""
    if not found:
        return
    if table in REVISED:
        found = [{**one, "latest": 0} for one in found]
    names = list(found[0])
    columns = {one[1] for one in con.execute(f"PRAGMA table_info({table})")}
    strange = sorted({name for one in found for name in one} - columns)
    if strange or any(one.keys() != found[0].keys() for one in found):
        raise UpstreamUnreachable(f"the upstream sent rows of {table} whose columns are not the table's "
                                  f"({', '.join(map(repr, strange)) or 'columns that differ from row to row'}): "
                                  "nothing is written")
    con.executemany(f"INSERT OR REPLACE INTO {table} ({', '.join(names)}) "
                    f"VALUES ({', '.join(':' + one for one in names)})", found)
    if table in REVISED:
        touched = _json(sorted({one["id"] for one in found}))
        con.execute(f"UPDATE {table} SET latest = 0 WHERE latest AND id IN (SELECT value FROM json_each(?))",
                    (touched,))
        con.execute(f"UPDATE {table} SET latest = 1 WHERE id IN (SELECT value FROM json_each(?)) AND revision = "
                    f"(SELECT max(r.revision) FROM {table} r WHERE r.id = {table}.id)", (touched,))


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


def _carry_out(con, depot: Depot, upstream: Upstream, erasure: str):
    """Carry out here an erasure the upstream carried out: the texts it
    removed whole go, and the texts it cut are read again."""
    needles = json.loads(con.execute("SELECT needles FROM erasures WHERE id = ?", (erasure,)).fetchall()[0][0])
    removed = sorted({one["text"] for one in needles if one["start"] is None and one["field"] is None})
    cut = sorted({one["text"] for one in needles} - set(removed))
    _remove_texts(con, removed)
    if cut:
        con.execute("DELETE FROM content_lists WHERE id IN (SELECT value FROM json_each(?))", (_json(cut),))
        after = None
        while True:
            found, after = upstream.page("content_lists", None, after, texts=cut)
            _put(con, "content_lists", found)
            _bring_parts(con, upstream, found)
            if after is None:
                break
        _drop_unlisted(con, cut)


def _leave(con, depot: Depot, texts: list[str], number: int):
    """The texts a pull brought that are not admitted now leave the depot's own open libraries."""
    if texts:
        con.execute(f"""UPDATE memberships SET left_change = ? WHERE left_change IS NULL
            AND library IN (SELECT id FROM libraries WHERE locked_at IS NULL AND {ids.carries_sql(depot.name())})
            AND text IN (SELECT t.id FROM texts_revisions t WHERE t.latest
                         AND t.id IN (SELECT value FROM json_each(?)) AND NOT {policy.admitted_sql('t.decisions')})""",
                    (number, _json(texts)))


def _highest(con) -> int:
    """The highest change number any row of the file records."""
    return con.execute("""SELECT max(n) FROM (
        SELECT max(id) AS n FROM changes UNION ALL SELECT max(change) FROM about_revisions
        UNION ALL SELECT max(change) FROM texts_revisions UNION ALL SELECT max(change) FROM content_lists
        UNION ALL SELECT max(change) FROM libraries_revisions UNION ALL SELECT max(change) FROM erasures_revisions
        UNION ALL SELECT max(change) FROM datasets_revisions UNION ALL SELECT max(joined_change) FROM memberships
        UNION ALL SELECT max(left_change) FROM memberships UNION ALL SELECT max(change) FROM pulls
        UNION ALL SELECT max(change) FROM decisions UNION ALL SELECT max(change) FROM notices
        UNION ALL SELECT max(change) FROM pins)""").fetchall()[0][0] or 0


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

