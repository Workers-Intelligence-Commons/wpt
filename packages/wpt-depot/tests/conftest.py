import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pytest
from deltalake import write_deltalake

from wpt_depot import About, Declaration, Depot, follow, format3, kinds
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
