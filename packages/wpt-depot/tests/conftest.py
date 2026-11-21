import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pytest
from deltalake import write_deltalake

from wpt_depot import About, Declaration, Depot, follow, format3, kinds, titles
from wpt_depot.legacy import STEPS, Legacy

ME = Declaration("Ada Lovelace", "ada@example.org", True)


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


@pytest.fixture
def library_figures():
    """library_figures(depot): (what the depot's counts say each library
    holds of each kind and by tag, the same computed from its rows directly: each
    open membership's text at its current revision and content version),
    each {(facet, name): (texts, characters, bytes)}."""
    def figures(depot: Depot):
        con = depot.db.reader()
        stored = {(facet, name): (texts, characters, size) for facet, name, texts, characters, size in con.execute(
            "SELECT facet, name, texts, characters, bytes FROM counts WHERE facet IN ('library kind', 'library tag')")}
        direct = {}
        for library, kind, length, size, tags in con.execute(
                "SELECT m.library, c.kind, c.length, c.bytes, t.tags FROM memberships m "
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


@pytest.fixture
def figures_at():
    """figures_at(depot, dataset): for each library of the dataset's current revision, in order, what it held at the
    change the revision reads it at, computed directly from the texts it held then (Depot.library_texts, each at its
    revision then), as a revision's figures name them: {kinds: {kind: {texts, characters, bytes}} of every kind of
    the depot, plain first, then by id, tags: the same for the plain texts that have each tag of its shares}; None
    for a library named by another depot's
    address, which no revision's figures know."""
    def figures(depot: Depot, dataset: str) -> list[dict]:
        row = depot.dataset_by_id(dataset)
        tags = [one["tag"] for one in row["shares"] or []]
        found = []
        for ref in row["libraries"]:
            if ref["depot"] is not None:
                found.append(None)
                continue
            kinds = ["plain"] + sorted(one["id"] for one in depot.sql("SELECT id FROM kinds") if one["id"] != "plain")
            one = {"kinds": {kind: {"texts": 0, "characters": 0, "bytes": 0} for kind in kinds},
                   "tags": [{"texts": 0, "characters": 0, "bytes": 0} for _ in tags]}
            for text in depot.library_texts(ref["library"], ref["change"]):
                kind = text["kind"]
                into = [one["kinds"][kind]] + ([one["tags"][at] for at, tag in enumerate(tags) if tag in text["tags"]]
                                               if kind == "plain" else [])
                for sums in into:
                    sums["texts"] += 1
                    sums["characters"] += text["length"]
                    sums["bytes"] += text["bytes"]
            found.append(one)
        return found
    return figures


@pytest.fixture
def serve():
    """serve(depot): the web address of a server answering the follow routes
    over the depot, as the web app's do (wpt_depot.follow), stopped after
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
    written as format 2 laid them out and brought to the format asked through the steps of wpt_depot.legacy
    (from 2 to 3) or as the older formats laid them out (below 2)."""
    def make(reached: int = 3, upstream: str | None = None) -> Path:
        folder = tmp_path / f"old-{reached}"
        for name, schema in FORMAT_2.items():
            write_deltalake(str(folder / name), pa.Table.from_pylist(
                format_2_rows(str(folder), upstream)[name], schema=schema), mode="error")
        old = Legacy(folder)
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
