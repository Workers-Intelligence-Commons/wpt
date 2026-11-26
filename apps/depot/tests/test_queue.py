import base64
import hashlib
import http.client
import io
import json
import pickle
import socket
import sqlite3
import threading
import time
from compression import zstd
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pyroaring import BitMap

from depot import About, Declaration, Depot, NewText, snapshots
from depot.legacy import Legacy
from depot_app import jobs as jobs_module
from depot_app.downloads import Downloads
from depot_app.jobs import KIND_GONE, WRITES_BEFORE, Jobs, connected
from depot_app.web.routes import STATUSES, routes
from depot_app.web.server import make_server

ME = Declaration("Ada Lovelace", "ada@example.org", True)
TEXTS = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs",
         "how vexingly quick daft zebras jump", "sphinx of black quartz, judge my vow"]
PUBLIC = "https://depot.example"  # the served depot's public address, DEPOT_URL


def legacy_dataset(depot: Depot, name: str) -> str:
    """The rows of the legacy tables, which nothing writes any more, of an open library holding every admitted
    text of depot and of a dataset called name reading it with the tokenizer recipe "tiny", as a depot of format
    12 left them for the frozen dataset routes (depot_app.web.legacy): the dataset's id."""
    with depot.db.write("legacy dataset") as (con, number):
        library, dataset = (depot._id(depot._take(con, kind)) for kind in ("libraries", "datasets"))
        con.execute("INSERT INTO libraries_revisions (id, revision, change, latest, name) VALUES (?, 1, ?, 1, 'all')",
                    (library, number))
        con.execute("INSERT INTO memberships (library, text, joined_change, export_key) SELECT ?, id, ?, "
                    "export_key(id) FROM texts WHERE status = 'admitted'", (library, number))
        con.execute("INSERT INTO datasets_revisions (id, revision, change, latest, name, tokenizer, libraries) "
                    "VALUES (?, 1, ?, 1, ?, 'tiny', ?)",
                    (dataset, number, name, json.dumps([{"depot": None, "library": library, "change": number}])))
    return dataset


def depot_with_texts(address: str, name: str, tags=((),) * len(TEXTS)) -> Depot:
    """A new depot whose four plain texts, admitted, each with its tags, are read by the legacy dataset "first"
    (legacy_dataset), with a snapshot of them."""
    depot = Depot.new(address, About(name, "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add([NewText(text=text, licenses=("CC0-1.0",), tags=tuple(tagged)) for text, tagged in zip(TEXTS, tags)],
              ME)
    depot.decide("admitted", "true", ME)
    legacy_dataset(depot, "first")
    depot.create_snapshot()
    return depot


@pytest.fixture
def served(tmp_path, monkeypatch):
    """serve(depot) -> call(method, target, body=None, headers=None) -> (status, headers, payload): the depot
    served with its queue running, in a folder with no .env, DOCK naming a dock of the test's folder and
    DEPOT_URL the depot's public address PUBLIC. Each server and queue stops when the test ends."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEPOT_URL", PUBLIC + "/")
    monkeypatch.setenv("DOCK", (tmp_path / "store").as_uri())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>interface</p>")
    started = []

    def serve(depot):
        jobs = Jobs(depot.address, STATUSES)
        jobs.start()
        server = make_server(routes(depot, jobs), static)
        started.append((server, jobs))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        def call(method, target, body=None, headers=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
            data = None if body is None else json.dumps(body).encode()
            sent = {**({} if body is None else {"Content-Type": "application/json"}), **(headers or {})}
            connection.request(method, target, data, sent)
            response = connection.getresponse()
            raw = response.read()
            payload = json.loads(raw) if response.getheader("Content-Type") == "application/json" else raw
            return response.status, dict(response.getheaders()), payload
        call.url = f"http://127.0.0.1:{port}"
        return call

    yield serve
    for server, jobs in started:
        server.shutdown()
        server.server_close()
        jobs.stop()


def ended(call, job: str) -> dict:
    """The job's state once it ended."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        state = call("GET", f"/api/jobs/{job}")[2]
        if state["state"] not in ("queued", "running"):
            return state
        time.sleep(0.1)
    raise AssertionError(f"the job {job} did not end")


def test_the_export_gives_every_text_in_parts_the_same_bytes_each_time_with_no_tokenizer_and_no_job(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    base = f"/api/datasets/{Legacy(depot.address).dataset('first')['id']}/export"

    status, _, shown = call("GET", base)

    assert status == 200 and shown["texts"] == len(TEXTS)
    [part] = shown["parts"]
    assert part["forms"]["jsonl.zst"] == {"name": "part-00000.jsonl.zst", "download": "first_part-00000.jsonl.zst",
                                          "bytes": None, "sha256": None, "left_out": None}
    status, headers, first = call("GET", f"{base}/part-00000.jsonl.zst")
    assert headers["Content-Disposition"] == 'attachment; filename="first_part-00000.jsonl.zst"'  # saved by its dataset's name
    assert call("GET", f"{base}/part-00000.jsonl.zst")[2] == first
    rows = [json.loads(line) for line in zstd.decompress(first).decode().splitlines()]
    assert sorted(row["text"] for row in rows) == sorted(TEXTS)
    assert all(row["depot"] == PUBLIC and row["revision"] == 1 for row in rows)  # never the depot's file
    table = call("GET", f"{base}/part-00000.parquet")[2]
    assert pq.read_table(io.BytesIO(table)).to_pylist() == rows
    known = call("GET", base)[2]["parts"][0]["forms"]["jsonl.zst"]
    assert known == {"name": "part-00000.jsonl.zst", "download": "first_part-00000.jsonl.zst", "bytes": len(first), "sha256": hashlib.sha256(first).hexdigest(),
                     "left_out": {"quote unmatched": 0, "quoted text left out": 0}}  # its conversations left out, by why
    status, headers, piece = call("GET", f"{base}/part-00000.jsonl.zst", headers={"Range": "bytes=3-10"})
    assert (status, piece, headers["Content-Range"]) == (206, first[3:11], f"bytes 3-10/{len(first)}")
    assert call("GET", f"{base}/part-00001.parquet")[0] == 404
    assert call("GET", "/api/jobs")[2] == []  # nothing was queued


def test_text_exports_are_bounded_in_all_and_per_visitor(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    dataset = Legacy(depot.address).dataset("first")["id"]
    text = f"/api/datasets/{dataset}/export/part-00000.jsonl.zst"
    downloads = Downloads(Jobs(depot.address).path)  # the web app's own record of the streams being sent

    with downloads.streaming(None, "203.0.113.1", "text"):  # one export to the first visitor
        assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 429
        assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.2"})[0] == 200
        with downloads.streaming(None, "203.0.113.2", "text"):  # a second one, to another
            assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.3"})[0] == 429
    assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 200  # once they ended


def slow_once(depot, marker: str, *, progress, job):
    """A job that runs for a minute the first time and answers at once the next: how many times it started."""
    path = Path(marker)
    starts = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(starts))
    if starts == 1:
        time.sleep(60)
    return {"starts": starts}


def test_a_job_a_restart_of_the_web_app_cut_short_runs_again_when_the_web_app_starts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))  # where the job's process finds slow_once by name
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    marker = tmp_path / "starts"
    jobs = Jobs(depot.address, STATUSES)
    job = jobs.add("snapshot", "a slow job", slow_once, str(marker))["id"]
    jobs.start()
    deadline = time.monotonic() + 60
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    jobs.stop()
    assert jobs.state(job)["state"] == "running"

    again = Jobs(depot.address, STATUSES)
    again.start()
    try:
        found = again.wait(job, 60)
    finally:
        again.stop()
    assert (found["state"], found["answer"], found["restarts"]) == ("done", {"starts": 2}, 1)


def from_a_worker(depot, *, progress, job):
    """A job that starts a worker process of its own: the worker's exit code."""
    import multiprocessing
    worker = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(0,))
    worker.start()
    worker.join()
    return {"exit": worker.exitcode}


def test_a_job_may_start_worker_processes_of_its_own(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))  # where the job's process finds from_a_worker by name
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    jobs = Jobs(depot.address, STATUSES)
    job = jobs.add("snapshot", "a job with a worker", from_a_worker)["id"]
    jobs.start()
    try:
        found = jobs.wait(job, 60)
    finally:
        jobs.stop()
    assert (found["state"], found["answer"]) == ("done", {"exit": 0})


def answered(depot, *, progress):
    """A write that ends at once."""
    return {"done": True}


def test_a_job_queued_while_the_runner_waits_starts_at_once_not_at_its_next_look_at_the_queue(tmp_path, monkeypatch):
    # a write over many texts is a job: the runner takes it when it is queued, not up to POLL seconds later
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))  # where the job's process finds answered by name
    monkeypatch.setattr(jobs_module, "POLL", 600)
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    jobs = Jobs(depot.address, STATUSES)
    jobs.start()
    try:
        time.sleep(0.2)  # the runner found the queue empty and waits
        found = jobs.wait(jobs.add("write", "a write", answered)["id"], 60)
    finally:
        jobs.stop()
    assert (found["state"], found["answer"]) == ("done", {"done": True})


def test_a_snapshot_is_made_as_a_job_once_confirmed_with_the_estimate_it_was_shown_and_named_by_its_time(
        served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    first = depot.snapshots()[0]["name"]
    call = served(depot)
    status, _, offered = call("GET", "/api/snapshots/new")
    assert status == 200 and offered["job"] is None and offered["estimate"]["texts"] == len(TEXTS)
    assert "name" not in offered  # the depot names it by the time it is made
    shown = offered["estimate"]

    assert call("POST", "/api/snapshots", {"confirmed": True})[0] == 400  # no estimate given back
    assert call("POST", "/api/snapshots", {"confirmed": False, "estimate": shown})[0] == 422
    assert call("POST", "/api/snapshots", {"name": "mine", "confirmed": True, "estimate": shown})[0] == 400
    status, _, refused = call("POST", "/api/snapshots", {"confirmed": True,  # it now costs more
                                                         "estimate": {**shown, "seconds": -100}})
    assert status == 409 and refused["estimate"]["texts"] == len(TEXTS)
    assert call("GET", "/api/jobs")[2] == []  # nothing was queued

    before = datetime.now(UTC).replace(microsecond=0)
    status, _, job = call("POST", "/api/snapshots", {"note": "monthly", "confirmed": True, "estimate": shown})
    assert (status, job["kind"], job["lane"]) == (200, "snapshot", "heavy")
    state = ended(call, job["id"])
    assert state["state"] == "done", state
    assert [one["stage"] for one in state["stages"]] == ["reading texts", "catching up", "writing"]
    name = state["answer"]["name"]
    assert before <= datetime.strptime(name, snapshots.NAMED).replace(tzinfo=UTC) <= datetime.now(UTC)
    row = depot.snapshot(name)
    assert state["answer"] == {"id": row["id"], "name": name, "at_change": row["at_change"], "texts": len(TEXTS),
                               "digest": row["digest"]}
    assert (row["note"], call("GET", "/api/snapshots/new")[2]["job"]) == ("monthly", None)
    assert [one["name"] for one in call("GET", "/api/snapshots")[2]] == [first, name]


def test_one_snapshot_job_at_a_time(tmp_path):
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    jobs = Jobs(depot.address, STATUSES)  # no runner: the job stays queued
    queued = jobs.add("snapshot", "make a snapshot", slow_once, str(tmp_path / "starts"))
    assert jobs.open_job("snapshot")["id"] == queued["id"]
    with pytest.raises(jobs_module.Busy, match="being made"):
        jobs.add("snapshot", "make another", slow_once, str(tmp_path / "starts"))


def test_a_snapshot_s_page_gives_its_summary_what_erasures_did_since_and_sets_anyone_can_check(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    row = depot.snapshots()[0]

    status, _, page = call("GET", f"/api/snapshots/{row['id']}")
    assert status == 200 and page["summary"]["kinds"]["plain"][0] == len(TEXTS) and "recipes" not in page
    assert page["erased_since"] == {"texts": 0, "erasures": []}
    status, _, found = call("GET", f"/api/snapshots/{row['id']}/sets")
    sets = [(one["part"], one["owner"], one["bound"], BitMap.deserialize(base64.b64decode(one["bitmap"])))
            for one in found["sets"]]
    assert snapshots.digest_of(sets, found["at_change"]) == found["digest"] == row["digest"]
    status, _, texts = call("POST", f"/api/snapshots/{row['id']}/texts", {"limit": 3})
    assert status == 200 and len(texts["texts"]) == 3 and texts["after"] == texts["texts"][-1]["id"]
    rest = call("POST", f"/api/snapshots/{row['id']}/texts", {"after": texts["after"]})[2]
    assert len(rest["texts"]) == 1 and rest["after"] is None
    assert call("POST", f"/api/snapshots/{row['id']}/texts", {"kinds": ["conversation"]})[2]["texts"] == []
    assert call("GET", "/api/snapshots/test-depot-99")[0] == 404

    text = rest["texts"][0]["id"]
    erasure = depot.ask_erasure("asked", [(text, None, None, None)], ME)
    depot.decide_erasures([erasure], "admitted", ME, None)
    assert call("GET", f"/api/snapshots/{row['id']}")[2]["erased_since"] == {"texts": 1, "erasures": [erasure]}
    assert Jobs(depot.address).erased_since(row["id"], snapshots.last_erasure(depot)) == {"texts": 1,
                                                                                           "erasures": [erasure]}


def test_a_sample_whose_reader_went_away_leaves_the_sample_route_to_the_next_request(served, tmp_path):
    # the frozen sample route of a dataset (depot_app.web.legacy): the sample is far bigger than the socket's
    # buffers, so its producer meets a full queue once the reader is gone, and must still free its slot
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    words = [hashlib.sha256(str(n).encode()).hexdigest() for n in range(4096)]
    depot.add([NewText(text=" ".join(words[n:] + words[:n]), licenses=("CC0-1.0",)) for n in range(400)], ME)
    depot.decide("admitted", "true", ME)
    legacy_dataset(depot, "first")
    call = served(depot)
    route = "/api/datasets/first/sample?bytes=1000000000"
    with socket.create_connection(("127.0.0.1", int(call.url.rsplit(":", 1)[1])), timeout=60) as gone:
        gone.sendall(f"GET {route} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
        assert gone.recv(1000).startswith(b"HTTP/1.0 200")  # the reader goes away after its first bytes

    answered = []
    # another visitor: the slot of the first stays taken until the server finds its reader gone
    reader = threading.Thread(target=lambda: answered.append(call("GET", route, headers={
        "X-Forwarded-For": "203.0.113.9"})), daemon=True)
    reader.start()
    reader.join(120)
    assert answered and answered[0][0] == 200
    body = answered[0][2]
    assert json.loads(body[body.rindex(b'{"end"'):])["texts"] == 400  # the whole sample, to its last line
    # the same visitor as the one that went away, one sample at a time (VISITOR_EXPORTS): refused (429) until its
    # slot is given back, then sent; a slot never given back would refuse it for good
    deadline = time.monotonic() + 60
    while (status := call("GET", "/api/datasets/first/sample?bytes=1000")[0]) == 429:
        assert time.monotonic() < deadline, "the slot of the sample whose reader went away was never given back"
        time.sleep(0.1)
    assert status == 200


def written_once(depot, text: str, *, progress) -> dict:
    """A write job: the text admitted."""
    decision, texts, _ = depot.decide("admitted", f"id = '{text}'", ME)
    return {"texts": texts}


def test_writes_inline_and_as_jobs_run_while_a_heavy_job_runs(served, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))  # where the job's process finds the operations
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    first, second = depot.add([NewText(text="the fox", licenses=("CC0-1.0",)),
                               NewText(text="the dog", licenses=("CC0-1.0",))], ME)
    call = served(depot)
    marker = tmp_path / "starts"
    jobs = Jobs(depot.address, STATUSES)
    heavy = jobs.add("snapshot", "a slow snapshot", slow_once, str(marker))["id"]
    deadline = time.monotonic() + 60
    while not marker.exists():  # the heavy job runs
        assert time.monotonic() < deadline
        time.sleep(0.05)

    status, _, state = call("POST", "/api/decisions", {"status": "admitted", "selection": {"filter": {"ids": [first]}},
                                                        "declaration": {"name": "Ada Lovelace",
                                                                        "email": "ada@example.org", "accepted": True}})
    assert (status, state["state"]) == (200, "done")  # answered at once, not queued behind the snapshot
    queued = jobs.add("write", "admit the dog", written_once, second)
    found = jobs.wait(queued["id"], 60)
    assert (found["state"], found["lane"], found["answer"]) == ("done", "write", {"texts": 1})
    assert jobs.state(heavy)["state"] == "running"  # the write lane did not wait for the heavy one


def test_tags_change_over_a_selection_only_once_confirmed_with_the_estimate_shown(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot", tags=[["a"], [], ["a"], []])
    call = served(depot)
    declaration = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True}
    selection = {"filter": {"kinds": ["plain"], "statuses": ["admitted"]}}

    status, _, shown = call("POST", "/api/texts/retag/estimate", {"selection": selection, "add": ["a"]})

    assert status == 200 and (shown["texts"], shown["changed"]) == (4, 2) and shown["rows"]["taggings"] == 2
    body = {"selection": selection, "add": ["a"], "declaration": declaration}
    assert call("POST", "/api/texts/retag", {**body, "confirmed": True})[0] == 400
    assert call("POST", "/api/texts/retag", {**body, "confirmed": False, "estimate": shown})[0] == 422
    assert call("POST", "/api/texts/retag", {**body, "confirmed": True,
                                             "estimate": {**shown, "changed": -100}})[0] == 409
    assert depot.tags() == [{"tag": "a", "texts": 2}]
    revisions = depot.sql("SELECT id, revision FROM texts ORDER BY id")

    status, _, state = call("POST", "/api/texts/retag", {**body, "confirmed": True, "estimate": shown})

    assert (status, state["state"], state["lane"]) == (200, "done", "write")
    assert state["answer"]["changed"] == 2 and state["answer"]["left_out"] == 2
    assert depot.tags() == [{"tag": "a", "texts": 4}]
    # rows of taggings, no new revision; one decision under review, whose declaration is public
    assert depot.sql("SELECT id, revision FROM texts ORDER BY id") == revisions
    decided = depot.sql(f"SELECT status, name, reason FROM decisions WHERE id = '{state['answer']['decision']}'")
    assert decided == [{"status": "under review", "name": "Ada Lovelace", "reason": "tags changed: added a"}]
    assert call("POST", "/api/texts/search", {"filter": {"kinds": ["conversation"]}})[2]["rows"] == []
    assert len(call("POST", "/api/texts/search", {"filter": {"kinds": ["plain"]}})[2]["rows"]) == 4


def test_a_queue_file_of_the_time_of_datasets_is_brought_up_and_its_dataset_jobs_fail(tmp_path):
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    path = Path(str(tmp_path / "depot.sqlite") + ".jobs")
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY, number INTEGER NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
            dataset TEXT, recipe TEXT, target TEXT, operation TEXT NOT NULL, arguments BLOB NOT NULL,
            state TEXT NOT NULL, stages TEXT NOT NULL, progress TEXT, answer TEXT, problem TEXT, status INTEGER,
            created REAL NOT NULL, started REAL, ended REAL, restarts INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE checkpoints (dataset TEXT PRIMARY KEY, state TEXT NOT NULL);""")
    arguments = pickle.dumps(((), {}))
    rows = [("a", 1, "corpus", "build first", "wic-1", None, "queued", "wpt.depot.web.builds:build_job"),
            ("b", 2, "write", "fork", None, None, "running", "wpt.depot.web.routes:fork_library"),
            ("c", 3, "vocabulary", "train", None, "tiny", "queued", "wpt.depot.web.builds:train_job"),
            ("d", 4, "write", "admit", None, None, "done", "wpt.depot.web.routes:decide_texts"),
            ("e", 5, "write", "admit library X", None, None, "queued", "wpt.depot.web.routes:decide_texts")]
    old.executemany("INSERT INTO jobs (id, number, kind, title, dataset, recipe, state, operation, arguments, stages, "
                    "created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', 0)", [(*row, arguments) for row in rows])
    old.execute("INSERT INTO checkpoints VALUES ('wic-1', '{}')")
    old.commit()
    old.close()

    jobs = Jobs(depot.address, STATUSES)

    found = {one["id"]: one for one in jobs.listed()}
    assert (found["a"]["state"], found["a"]["problem"], found["a"]["key"]) == ("failed", KIND_GONE, "wic-1")
    assert found["b"]["state"] == "failed" and "no longer exists" in found["b"]["problem"]
    assert (found["c"]["state"], found["c"]["problem"]) == ("failed", KIND_GONE)
    assert (found["d"]["state"], found["d"]["lane"]) == ("done", "write")
    # its selection may have named a library, which unpickles as nothing: run, it would reach every text
    assert (found["e"]["state"], found["e"]["problem"]) == ("failed", WRITES_BEFORE)
    with connected(path) as con:  # what only builds kept is gone
        assert not con.execute("SELECT name FROM sqlite_master WHERE name IN ('checkpoints', 'downloads')").fetchall()
    Jobs(depot.address, STATUSES)  # brought up once: opening it again changes nothing
    assert {one["id"]: one["state"] for one in jobs.listed()} == {key: one["state"] for key, one in found.items()}


def test_a_vocabulary_or_corpus_job_queued_in_a_queue_of_today_s_layout_fails_when_it_is_opened(tmp_path):
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    jobs = Jobs(depot.address, STATUSES)
    with connected(jobs.path, write=True) as con:
        con.executemany("INSERT INTO jobs (id, number, kind, lane, title, operation, arguments, state, stages, "
                        "created) VALUES (?, ?, ?, 'heavy', 'old', 'gone:job', ?, ?, '[]', 0)",
                        [("v", 1, "vocabulary", pickle.dumps(((), {})), "queued"),
                         ("c", 2, "corpus", pickle.dumps(((), {})), "running"),
                         ("d", 3, "corpus", pickle.dumps(((), {})), "done")])

    found = {one["id"]: one for one in Jobs(depot.address, STATUSES).listed()}

    assert {key: (one["state"], one["problem"]) for key, one in found.items()} == {
        "v": ("failed", KIND_GONE), "c": ("failed", KIND_GONE), "d": ("done", None)}


def test_the_routes_of_tokenizer_recipes_recipes_and_builds_are_gone_and_the_frozen_reads_stay(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    for method, route in (("GET", "/api/tokenizers"), ("POST", "/api/tokenizers"), ("GET", "/api/vocabularies"),
                          ("POST", "/api/recipes/check"), ("POST", "/api/recipes/build"),
                          ("POST", "/api/recipes/compose"), ("GET", f"/api/builds/{'0' * 64}")):
        assert call(method, route, {} if method == "POST" else None)[0] == 404, route
    assert call("GET", "/api/tokenizers/none")[2]["problems"] == ["NotFound: no tokenizer recipe none"]
