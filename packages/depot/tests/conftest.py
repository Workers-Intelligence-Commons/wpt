import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pytest
from deltalake import write_deltalake

from depot import About, Declaration, Depot, follow, format3, kinds, titles
from depot.delta import STEPS, DeltaFolder
from depot.ids import export_key
from depot.legacy import Legacy

ME = Declaration("Ada Lovelace", "ada@example.org", True)
# the revisions of texts whose kind, length and bytes (schema.SIZES) are not those of the content version they read,
# as its row of content_lists holds them, or are missing: none when every write kept them exact
SIZES_APART = ("SELECT t.id, t.revision, t.content_kind, t.content_length, t.content_bytes, c.kind, c.length, c.bytes "
               "FROM texts_revisions t LEFT JOIN content_lists c ON c.id = t.id AND c.version = t.content_version "
               "WHERE (t.content_kind, t.content_length, t.content_bytes) IS NOT (c.kind, c.length, c.bytes) "
               "OR t.content_kind IS NULL ORDER BY t.id, t.revision")

# the current revisions whose copy of their text's tags (schema.tags_sql) is not the tags its rows of taggings give,
# and the past revisions that still hold a copy: none when every write kept it exact
TAGS_APART = ("SELECT t.id, t.revision, t.current_tags, (SELECT json_group_array(g.tag ORDER BY g.tag) FROM taggings g "
              "WHERE g.text = t.id AND g.left_change IS NULL) AS now FROM texts_revisions t "
              "WHERE CASE WHEN t.latest THEN t.current_tags IS NOT now ELSE t.current_tags IS NOT NULL END "
              "ORDER BY t.id, t.revision")


def tags_apart(depot: Depot) -> list[tuple]:
    """The rows of TAGS_APART in the depot."""
    return [tuple(row.values()) for row in depot.rows(TAGS_APART)]


def sizes_apart(depot: Depot) -> list[tuple]:
    """The rows of SIZES_APART in the depot."""
    return [tuple(row.values()) for row in depot.rows(SIZES_APART)]


@pytest.fixture
def me():
    return ME


@pytest.fixture
def make_depot(tmp_path):
    """make_depot(name): a new depot in the test's folder, with the license
    CC0-1.0."""
    def make(name="depot"):
        depot = Depot.new(str(tmp_path / f"{name}.sqlite"), About(name, "Ada Lovelace", "ada@example.org"))
        depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
        return depot
    return make


# each revision of a text given, in its column tags, the tags the text had while it was its current one (until the
# change before the next revision), as a format older than 13 kept them on its revisions
TAGS_ON_REVISIONS = ("UPDATE texts_revisions SET tags = (SELECT json_group_array(g.tag ORDER BY g.tag) FROM taggings g "
                     "WHERE g.text = texts_revisions.id AND g.joined_change <= u.until AND (g.left_change IS NULL OR "
                     "g.left_change > u.until)) FROM (SELECT r.id, r.revision, coalesce((SELECT min(n.change) - 1 FROM "
                     "texts_revisions n WHERE n.id = r.id AND n.revision > r.revision), 1 << 62) AS until "
                     "FROM texts_revisions r) u WHERE u.id = texts_revisions.id AND u.revision = texts_revisions.revision")


class LegacyRows:
    """Writes the rows of the legacy tables (schema.LEGACY) that no code
    writes any more into a depot of the code's format, as an older format
    left them, for the frozen reader of those tables (depot.legacy),
    which reads the tags of a text from its revisions, as those formats
    kept them: each write gives every revision the tags its text had while
    it was the current one (TAGS_ON_REVISIONS)."""

    def library(self, depot: Depot, name: str, texts: list[str], locked: bool = False) -> str:
        """A library of the depot holding texts from now on, locked at once
        when locked (its pins at each text's current revision): its id."""
        with depot.db.write("legacy library") as (con, number):
            con.execute(TAGS_ON_REVISIONS)
            library = depot._id(depot._take(con, "libraries"))
            con.execute("INSERT INTO libraries_revisions (id, revision, change, latest, name, locked_at) "
                        "VALUES (?, 1, ?, 1, ?, ?)", (library, number, name, "2026-01-01T00:00:00Z" if locked else None))
            for text in texts:
                key = export_key(text)
                con.execute("INSERT INTO memberships (library, text, joined_change, export_key) VALUES (?, ?, ?, ?)",
                            (library, text, number, key))
                if locked:
                    con.execute("INSERT INTO pins (library, text, revision, change, export_key) SELECT ?, id, "
                                "revision, ?, ? FROM texts_revisions WHERE id = ? AND latest", (library, number, key, text))
            if texts:
                self._moved(con, library, number)
        return library

    def _moved(self, con, library: str, number: int):
        """The library's memberships moved at the change number, as library_moves records it."""
        con.execute("INSERT INTO library_moves (library, change) VALUES (?, ?) ON CONFLICT (library) DO UPDATE SET "
                    "change = excluded.change", (library, number))

    def put(self, depot: Depot, library: str, texts: list[str]):
        """The texts in the open library from now on."""
        with depot.db.write("legacy put") as (con, number):
            con.executemany("INSERT INTO memberships (library, text, joined_change, export_key) VALUES (?, ?, ?, ?)",
                            [(library, text, number, export_key(text)) for text in texts])
            self._moved(con, library, number)

    def lock(self, depot: Depot, library: str):
        """The open library locked now: a revision of it with locked_at, its
        texts pinned at their current revisions."""
        with depot.db.write("legacy lock") as (con, number):
            [(revision, name)] = con.execute("SELECT revision, name FROM libraries_revisions WHERE id = ? AND latest",
                                             (library,)).fetchall()
            con.execute("UPDATE libraries_revisions SET latest = 0 WHERE id = ?", (library,))
            con.execute("INSERT INTO libraries_revisions (id, revision, change, latest, name, locked_at) "
                        "VALUES (?, ?, ?, 1, ?, '2026-01-01T00:00:00Z')", (library, revision + 1, number, name))
            con.execute("INSERT INTO pins (library, text, revision, change, export_key) SELECT m.library, m.text, "
                        "t.revision, ?, m.export_key FROM memberships m JOIN texts_revisions t ON t.id = m.text AND "
                        "t.latest WHERE m.library = ? AND m.left_change IS NULL", (number, library))

    def dataset(self, depot: Depot, name: str, libraries: list[str], shares: list[tuple[str, float]] | None = None,
                max_tokens: int | None = None) -> dict:
        """A dataset of the depot reading its libraries at the current
        change, with no figures stored: its current row, as Legacy reads it."""
        with depot.db.write("legacy dataset") as (con, number):
            con.execute(TAGS_ON_REVISIONS)
            dataset = depot._id(depot._take(con, "datasets"))
            refs = [{"depot": None, "library": one, "change": number - 1} for one in libraries]
            con.execute("INSERT INTO datasets_revisions (id, revision, change, latest, name, tokenizer, libraries, "
                        "shares, max_tokens) VALUES (?, 1, ?, 1, ?, 'tok', ?, ?, ?)",
                        (dataset, number, name, json.dumps(refs),
                         None if shares is None else json.dumps([{"tag": tag, "share": share} for tag, share in shares]),
                         max_tokens))
        return Legacy(depot.address).dataset_by_id(dataset)


@pytest.fixture
def legacy_rows():
    """The writer of legacy rows (LegacyRows)."""
    return LegacyRows()


@pytest.fixture
def library_figures():
    """library_figures(con): (what the counts of the depot con reads say
    each library holds of each kind and by tag, as a depot of format 9 to 12
    counted it, the same computed from its rows directly: each open
    membership's text at its current revision and content version, a
    content version of no kind, as format 11 and older leave it, plain),
    each {(facet, name): (texts, characters, bytes)}."""
    def figures(con):
        stored = {(facet, name): (texts, characters, size) for facet, name, texts, characters, size in con.execute(
            "SELECT facet, name, texts, characters, bytes FROM counts WHERE facet IN ('library kind', 'library tag')")}
        direct = {}
        for library, kind, length, size, tags in con.execute(
                "SELECT m.library, coalesce(c.kind, 'plain'), c.length, c.bytes, t.tags FROM memberships m "
                "JOIN texts_revisions t ON t.id = m.text AND t.latest "
                "JOIN content_lists c ON c.id = t.id AND c.version = t.content_version WHERE m.left_change IS NULL"):
            keys = [("library kind", f"{library} {kind}")]
            if kind == "plain":
                keys += [("library tag", f"{library} {tag}") for tag in json.loads(tags)]
            for key in keys:
                texts, characters, held = direct.get(key, (0, 0, 0))
                direct[key] = (texts + 1, characters + length, held + size)
        return stored, direct
    return figures


def read_title_index(depot: Depot) -> dict:
    """What the depot's title index holds, as far as it can be read: {terms: each trigram it holds with how many
    texts it gives for it, unfound: the current texts with a title the index does not give for every trigram of
    their title}; a recount makes the terms again from the texts, so a write kept the index exact when they stay
    equal and unfound stays empty."""
    con = depot.db.reader()
    con.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.title_terms USING fts5vocab(main, {titles.TABLE}, row)")
    terms = con.execute("SELECT term, doc FROM temp.title_terms ORDER BY term").fetchall()
    unfound = [text for text, title in con.execute("SELECT id, title FROM texts_revisions WHERE latest AND title "
                                                   "IS NOT NULL AND length(title) >= 3 ORDER BY id")
               if not con.execute(f"SELECT 1 FROM (SELECT ? AS id) x, {titles.TABLE} WHERE {titles.TABLE} MATCH ? "
                                  f"AND rowid = {titles.row_sql('x.id')}", (text, titles.match([title]))).fetchall()]
    return {"terms": terms, "unfound": unfound}


@pytest.fixture
def title_index():
    """title_index(depot): read_title_index."""
    return read_title_index


def legacy_figures(depot: Depot, dataset: str) -> list[dict]:
    """For each library of the dataset's current revision, in order, what it held at the change the revision reads
    it at, computed directly from the legacy rows (a library locked then: its pins; another: the texts that had
    joined it and not left it then, each at its latest revision then), as a revision's figures name them: {kinds:
    {kind: {texts, characters, bytes}} of every kind of the depot, plain first, then by id, tags: the same for the
    plain texts that have each tag of its shares}; None for a library named by another depot's address, which no
    revision's figures know."""
    con = depot.db.reader()
    [(refs, shares)] = con.execute("SELECT libraries, shares FROM datasets_revisions WHERE id = ? AND latest",
                                   (dataset,)).fetchall()
    tags = [one["tag"] for one in json.loads(shares or "[]")]
    kinds = ["plain"] + [kind for (kind,) in con.execute("SELECT id FROM kinds WHERE id <> 'plain' ORDER BY id")]
    found = []
    for ref in json.loads(refs):
        if ref["depot"] is not None:
            found.append(None)
            continue
        library, change = ref["library"], ref["change"]
        [(locked,)] = con.execute("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? ORDER BY "
                                  "revision DESC LIMIT 1", (library, change)).fetchall()
        held = ("SELECT text, revision FROM pins WHERE library = :library" if locked is not None else
                "SELECT m.text, (SELECT max(r.revision) FROM texts_revisions r WHERE r.id = m.text AND r.change <= "
                ":change) AS revision FROM memberships m WHERE m.library = :library AND m.joined_change <= :change "
                "AND (m.left_change IS NULL OR m.left_change > :change)")
        one = {"kinds": {kind: {"texts": 0, "characters": 0, "bytes": 0} for kind in kinds},
               "tags": [{"texts": 0, "characters": 0, "bytes": 0} for _ in tags]}
        for kind, length, size, text_tags in con.execute(
                f"SELECT c.kind, c.length, c.bytes, t.tags FROM ({held}) h JOIN texts_revisions t ON t.id = h.text "
                "AND t.revision = h.revision JOIN content_lists c ON c.id = t.id AND c.version = t.content_version",
                {"library": library, "change": change}):
            into = [one["kinds"][kind]] + ([one["tags"][at] for at, tag in enumerate(tags) if tag in json.loads(text_tags)]
                                           if kind == "plain" else [])
            for sums in into:
                sums["texts"] += 1
                sums["characters"] += length
                sums["bytes"] += size
        found.append(one)
    return found


@pytest.fixture
def figures_at():
    """figures_at(depot, dataset): legacy_figures."""
    return legacy_figures


@pytest.fixture
def serve():
    """serve(depot): the web address of a server answering the follow routes
    over the depot, as the web app's do (depot.follow), stopped after
    the test."""
    servers = []

    def start(depot: Depot) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                route = self.path.rsplit("/", 1)[-1]
                if route == "head":
                    data = [json.dumps(follow.head(depot)).encode()]
                elif route == "rows":
                    data = follow.rows(depot, body["table"], body.get("ranges"), body.get("after"), body.get("texts"))
                else:
                    data = follow.parts(depot, body["keys"])
                self.send_response(200)
                self.end_headers()
                for chunk in data:
                    self.wfile.write(chunk)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


TIME = pa.timestamp("us", tz="UTC")
NAMES = pa.list_(pa.string())
_NEEDLE = pa.struct([("text", pa.string()), ("field", pa.string()), ("start", pa.int64()), ("end", pa.int64()),
                     pa.field("version", pa.int64(), nullable=False)])
_ERASED = pa.struct([("length", pa.int64()), ("hash_a", pa.binary()), ("hash_b", pa.binary()), ("at", pa.int64()),
                     ("needle", pa.int64())])
_PROBLEM = pa.struct([("texts", NAMES), ("problem", pa.string()), ("decision", pa.string())])
_REF = pa.struct([("depot", pa.string()), ("library", pa.string()), ("change", pa.int64())])
_SHARE = pa.struct([("tag", pa.string()), ("share", pa.float64())])
_SCORES = pa.list_(pa.struct([("name", pa.string()), ("value", pa.float64()), ("content_version", pa.int64())]))
_REVISION = (("revision", pa.int64()), ("change", pa.int64()))


def _schema(*fields) -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=bool(rest and rest[0])) for name, kind, *rest in fields])


# the tables of a depot of format 2, a folder of Delta Lake tables, as that format laid them out
FORMAT_2 = {
    "about": _schema(*_REVISION, ("name", pa.string()), ("maintainer", pa.string()),
                     ("takedown_contact", pa.string()), ("upstream", pa.string(), True), ("format", pa.int64())),
    "journal": _schema(("id", pa.int64()), ("change", pa.string()), ("done", pa.bool_())),
    "licenses": _schema(("id", pa.string()), ("name", pa.string()), ("text", pa.string())),
    "notices": _schema(("id", pa.string()), ("text", pa.string())),
    "kinds": _schema(("id", pa.string()), ("meaning", pa.string()), ("schema", pa.string())),
    "decisions": _schema(("id", pa.string()), ("status", pa.string()), ("name", pa.string()), ("email", pa.string()),
                         ("accepted", pa.bool_()), ("made_at", TIME), ("reason", pa.string(), True)),
    "texts": _schema(("id", pa.string()), *_REVISION, ("licenses", NAMES), ("notice", pa.string(), True),
                     ("title", pa.string(), True), ("authors", NAMES, True), ("tags", NAMES),
                     ("address", pa.string(), True), ("decisions", NAMES), ("content_version", pa.int64()),
                     ("scores", _SCORES, True)),
    "contents": _schema(("id", pa.string()), ("version", pa.int64()), ("change", pa.int64()),
                        ("text", pa.string(), True), ("kind", pa.string(), True), ("content", pa.string(), True)),
    "libraries": _schema(("id", pa.string()), *_REVISION, ("name", pa.string()), ("forked_from", pa.string(), True),
                         ("problems", pa.list_(_PROBLEM), True), ("locked_at", TIME, True)),
    "memberships": _schema(("library", pa.string()), ("text", pa.string()), ("joined_change", pa.int64()),
                           ("left_change", pa.int64(), True)),
    "pins": _schema(("library", pa.string()), ("text", pa.string()), ("revision", pa.int64())),
    "erasures": _schema(("id", pa.string()), *_REVISION, ("reason", pa.string()), ("decisions", NAMES),
                        ("needles", pa.list_(_NEEDLE)), ("erased", pa.list_(_ERASED), True)),
    "datasets": _schema(("id", pa.string()), *_REVISION, ("name", pa.string()), ("tokenizer", pa.string()),
                        ("libraries", pa.list_(_REF)), ("shares", pa.list_(_SHARE), True)),
    "pulls": _schema(("depot", pa.string()), ("change", pa.int64()), ("upstream", pa.string()),
                     ("upstream_change", pa.int64())),
}


def format_2_rows(folder: str, upstream: str | None) -> dict[str, list[dict]]:
    """The rows of a small depot called old as format 2 held them: two texts, one admitted and held by an open
    library and pinned by a locked one, an erasure request under review and a dataset, each change a done entry
    of the journal."""
    at = datetime(2024, 1, 1, tzinfo=UTC)
    decision = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True, "made_at": at, "reason": None}
    text = {"licenses": ["CC0-1.0"], "title": None, "authors": None, "address": None, "content_version": 1,
            "scores": None}
    return {
        "about": [{"revision": 1, "change": 1, "name": "old", "maintainer": "Ada Lovelace",
                   "takedown_contact": "ada@example.org", "upstream": upstream, "format": 2}],
        "journal": [{"id": number, "change": json.dumps({"kind": kind, **({"decision": {"made_at": at.isoformat()}}
                                                                          if kind == "decide" else {})}),
                     "done": True} for number, kind in enumerate(
            ("new", "add", "decide", "library", "library", "lock", "ask erasure", "dataset"), 1)],
        "licenses": [{"id": "CC0-1.0", "name": "Creative Commons Zero v1.0 Universal", "text": "The CC0 text."}],
        "notices": [{"id": "old-1", "text": "a notice"}],
        "kinds": kinds.starting_rows(),
        "decisions": [{**decision, "id": "old-1", "status": "under review"},
                      {**decision, "id": "old-2", "status": "admitted"},
                      {**decision, "id": "old-3", "status": "under review"}],
        "texts": [{**text, "id": "old-1", "revision": 1, "change": 2, "notice": "old-1", "tags": ["book"],
                   "decisions": ["old-1"]},
                  {**text, "id": "old-1", "revision": 2, "change": 3, "notice": "old-1", "tags": ["book"],
                   "decisions": ["old-1", "old-2"]},
                  {**text, "id": "old-2", "revision": 1, "change": 2, "notice": None, "tags": [],
                   "decisions": ["old-1"]}],
        "contents": [{"id": "old-1", "version": 1, "change": 2, "text": "hello world", "kind": None, "content": None},
                     {"id": "old-2", "version": 1, "change": 2, "text": "a second text", "kind": None,
                      "content": None}],
        "libraries": [{"id": "old-1", "revision": 1, "change": 4, "name": "open", "forked_from": None,
                       "problems": None, "locked_at": None},
                      {"id": "old-2", "revision": 1, "change": 5, "name": "locked", "forked_from": None,
                       "problems": None, "locked_at": None},
                      {"id": "old-2", "revision": 2, "change": 6, "name": "locked", "forked_from": None,
                       "problems": None, "locked_at": at}],
        "memberships": [{"library": "old-1", "text": "old-1", "joined_change": 4, "left_change": None},
                        {"library": "old-2", "text": "old-1", "joined_change": 5, "left_change": None}],
        "pins": [{"library": "old-2", "text": "old-1", "revision": 2}],
        "erasures": [{"id": "old-1", "revision": 1, "change": 7, "reason": "a word", "decisions": ["old-3"],
                      "needles": [{"text": "old-2", "field": None, "start": 0, "end": 2, "version": 1}],
                      "erased": None}],
        "datasets": [{"id": "old-1", "revision": 1, "change": 8, "name": "mine", "tokenizer": "tok",
                      "libraries": [{"depot": folder, "library": "old-1", "change": 4}], "shares": None}],
        "pulls": [],
    }


@pytest.fixture
def make_legacy(tmp_path):
    """make_legacy(format, upstream=None): the folder of a depot of that format (0 to 3), its Delta Lake tables
    written as format 2 laid them out and brought to the format asked through the steps of depot.delta
    (from 2 to 3) or as the older formats laid them out (below 2)."""
    def make(reached: int = 3, upstream: str | None = None) -> Path:
        folder = tmp_path / f"old-{reached}"
        for name, schema in FORMAT_2.items():
            write_deltalake(str(folder / name), pa.Table.from_pylist(
                format_2_rows(str(folder), upstream)[name], schema=schema), mode="error")
        old = DeltaFolder(folder)
        old.table("contents").alter.add_constraint({format3.CONSTRAINT_2: (
            "(kind IS NULL AND text IS NOT NULL AND content IS NULL) "
            "OR (kind IS NOT NULL AND text IS NULL AND content IS NOT NULL)")})
        if reached == 3:
            STEPS[2](old)
            old.record_format(3)
        if reached < 2:
            old.rewrite("texts", old.read("texts").drop_columns(["scores"]))
            old.rewrite("about", old.read("about").set_column(
                old.read("about").schema.get_field_index("format"), "format", pa.array([1], pa.int64())))
        if reached < 1:
            old.rewrite("about", old.read("about").drop_columns(["format"]))
        return folder
    return make
