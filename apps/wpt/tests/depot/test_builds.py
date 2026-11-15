import hashlib
import http.client
import io
import json
import socket
import threading
import time
from compression import zstd
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from wpt_depot import About, Declaration, Depot, NewText
from wpt.depot import shared
from wpt.depot.downloads import Downloads
from wpt.depot.jobs import Jobs
from wpt.depot.web.routes import STATUSES, routes
from wpt.depot.web.server import make_server

ME = Declaration("Ada Lovelace", "ada@example.org", True)
TEXTS = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs",
         "how vexingly quick daft zebras jump", "sphinx of black quartz, judge my vow"]
PUBLIC = "https://depot.example"  # the served depot's public address, WPT_DEPOT_URL


def depot_with_texts(address: str, name: str) -> Depot:
    """A new depot whose four plain texts, admitted, fill the library "all", which the dataset "first" reads
    with the tokenizer recipe "tiny" that it does not hold yet."""
    depot = Depot.new(address, About(name, "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add([NewText(text=text, licenses=("CC0-1.0",)) for text in TEXTS], ME)
    depot.decide("admitted", "true", ME)
    library = depot.new_library("all")
    depot.put(library, "true")
    depot.new_dataset("first", "tiny", [(None, library)])
    return depot


@pytest.fixture
def served(tmp_path, monkeypatch):
    """serve(depot) -> call(method, target, body=None, headers=None) -> (status, headers, payload): the depot
    served with its queue running, in a folder with no .env, WPT_STORE naming a store of the test's folder and
    WPT_DEPOT_URL the depot's public address PUBLIC. Each server and queue stops when the test ends."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WPT_DEPOT_URL", PUBLIC + "/")
    monkeypatch.setenv("WPT_STORE", (tmp_path / "store").as_uri())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    monkeypatch.delenv(shared.CAP, raising=False)
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


def test_a_build_trains_the_missing_vocabulary_first_and_the_dataset_s_link_gives_its_files_and_texts_the_same_each_time(
        served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    assert call("GET", "/api/about")[2]["address"] == PUBLIC  # what names it in its corpora
    dataset = depot.dataset("first")["id"]
    assert call("POST", "/api/tokenizers", {"name": "tiny", "dataset": "first", "sample_bytes": 1000,
                                           "vocab_size": 560})[0] == 200
    assert call("GET", f"/api/datasets/{dataset}/download")[2]["state"] == "not built"

    status, _, job = call("POST", f"/api/datasets/{dataset}/build", {})
    assert status == 200 and job["kind"] == "corpus" and job["after"] is not None
    assert call("POST", f"/api/datasets/{dataset}/build", {})[0] == 409  # one corpus job per dataset
    assert ended(call, job["id"])["state"] == "done"

    assert [(one["kind"], one["state"]) for one in call("GET", "/api/jobs")[2]] == [
        ("corpus", "done"), ("vocabulary", "done")]  # the latest that ended first
    recipe = depot.tokenizer("tiny")
    assert depot.vocabulary(recipe["id"], 1) is not None
    status, _, shown = call("GET", f"/api/datasets/{dataset}/download")
    assert status == 200 and shown["state"] == "built" and shown["corpus"]["texts"] == len(TEXTS)
    base = f"/api/datasets/{dataset}/download"

    # a token file: whole, the same twice and as listed; one range of it; a range past its end
    listed = max(shown["files"], key=lambda one: one["bytes"] or 0)
    status, headers, whole = call("GET", f"{base}/files/{listed['name']}")
    assert status == 200 and hashlib.sha256(whole).hexdigest() == listed["sha256"]
    assert headers["Accept-Ranges"] == "bytes" and int(headers["Content-Length"]) == len(whole)
    assert call("GET", f"{base}/files/{listed['name']}")[2] == whole
    status, headers, piece = call("GET", f"{base}/files/{listed['name']}", headers={"Range": "bytes=2-9"})
    assert (status, piece, headers["Content-Range"]) == (206, whole[2:10], f"bytes 2-9/{len(whole)}")
    assert call("GET", f"{base}/files/{listed['name']}", headers={"Range": f"bytes={len(whole)}-"})[0] == 416

    # the texts, in both forms: the same bytes each time, the corpus's texts in its order
    [part] = shown["parts"]
    assert part["forms"]["jsonl.zst"] == {"name": "part-00000.jsonl.zst", "bytes": None, "sha256": None}
    first = call("GET", f"{base}/text/part-00000.jsonl.zst")[2]
    assert call("GET", f"{base}/text/part-00000.jsonl.zst")[2] == first
    rows = [json.loads(line) for line in zstd.decompress(first).decode().splitlines()]
    assert sorted(row["text"] for row in rows) == sorted(TEXTS)
    assert all(row["depot"] == PUBLIC and row["revision"] == 1 for row in rows)  # never the depot's file
    table = call("GET", f"{base}/text/part-00000.parquet")[2]
    assert call("GET", f"{base}/text/part-00000.parquet")[2] == table
    assert pq.read_table(io.BytesIO(table)).to_pylist() == rows
    known = call("GET", base)[2]["parts"][0]["forms"]
    assert known["jsonl.zst"] == {"name": "part-00000.jsonl.zst", "bytes": len(first),
                                  "sha256": hashlib.sha256(first).hexdigest()}
    assert call("GET", f"{base}/text/part-00001.parquet")[0] == 404


def test_text_exports_are_bounded_in_all_and_per_visitor_while_token_files_are_not(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    dataset = depot.dataset("first")["id"]
    call("POST", "/api/tokenizers", {"name": "tiny", "dataset": "first", "sample_bytes": 1000, "vocab_size": 560})
    ended(call, call("POST", f"/api/datasets/{dataset}/build", {})[2]["id"])
    shown = call("GET", f"/api/datasets/{dataset}/download")[2]
    text = f"/api/datasets/{dataset}/download/text/part-00000.jsonl.zst"
    token_file = f"/api/datasets/{dataset}/download/files/{shown['files'][0]['name']}"
    downloads = Downloads(Jobs(depot.address).path)  # the web app's own record of the streams being sent

    with downloads.streaming(shown["corpus"]["id"], "203.0.113.1", "text"):  # one export to the first visitor
        assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 429
        assert call("GET", token_file, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 200
        assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.2"})[0] == 200
        with downloads.streaming(shown["corpus"]["id"], "203.0.113.2", "text"):  # a second one, to another
            assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.3"})[0] == 429
    assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 200  # once they ended


def test_an_admitted_erasure_makes_the_link_stale_until_a_rebuild_gives_a_corpus_without_the_passage(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    dataset = depot.dataset("first")["id"]
    call("POST", "/api/tokenizers", {"name": "tiny", "dataset": "first", "sample_bytes": 1000, "vocab_size": 560})
    ended(call, call("POST", f"/api/datasets/{dataset}/build", {})[2]["id"])
    shown = call("GET", f"/api/datasets/{dataset}/download")[2]
    built = shown["corpus"]["id"]
    token_file = f"/api/datasets/{dataset}/download/files/{shown['files'][0]['name']}"
    # while an erasure's corpora are being marked, nothing is sent: here one the web app never carried out
    Downloads(Jobs(depot.address).path).marking(["test-depot-9"])
    assert call("GET", f"/api/datasets/{dataset}/download")[2]["state"] == "checking an erasure"
    assert call("GET", token_file)[0] == 422
    declaration = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True}
    call("POST", "/api/erasures", {"reason": "a fox", "needles": [{"text": "test-depot-1", "start": 16, "end": 19}],
                                   "declaration": declaration})
    assert call("POST", "/api/erasures/decisions", {"erasures": ["test-depot-1"], "status": "admitted",
                                                    "declaration": declaration})[0] == 200

    deadline = time.monotonic() + 60  # the corpora are marked in the background
    while (shown := call("GET", f"/api/datasets/{dataset}/download")[2])["state"] == "checking an erasure":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert (shown["state"], shown["corpus"]["stale"]) == ("stale", {"erasures": ["test-depot-1"]})
    assert call("GET", f"/api/datasets/{dataset}/download/text/part-00000.jsonl.zst")[0] == 404
    time.sleep(1.5)  # nothing builds it again by itself
    assert call("GET", f"/api/datasets/{dataset}/download")[2]["state"] == "stale"

    assert ended(call, call("POST", f"/api/datasets/{dataset}/build", {})[2]["id"])["state"] == "done"
    shown = call("GET", f"/api/datasets/{dataset}/download")[2]
    assert shown["state"] == "built" and shown["corpus"]["id"] != built
    rows = [json.loads(line) for line in zstd.decompress(
        call("GET", f"/api/datasets/{dataset}/download/text/part-00000.jsonl.zst")[2]).decode().splitlines()]
    assert "the quick brown  jumps over the lazy dog" in {row["text"] for row in rows}


def test_a_vocabulary_trains_on_another_depot_s_dataset_read_through_its_sample_route(served, tmp_path):
    upstream = depot_with_texts(str(tmp_path / "upstream.sqlite"), "upstream")
    call = served(upstream)
    here = Depot.new(str(tmp_path / "here.sqlite"), About("here", "Ada Lovelace", "ada@example.org"))
    here.new_tokenizer("theirs", call.url, "first", 1000, 560)

    shared.train_vocabulary(here, "theirs", lambda *_: None, work=tmp_path)

    recipe = here.tokenizer("theirs")
    texts = here.vocabulary_texts(recipe["id"], 1)
    assert {one["depot"] for one in texts} == {call.url}
    assert sorted(one["text"] for one in texts) == [f"upstream-{n}" for n in range(1, 5)]


def slow_once(depot, marker: str, *, progress, job):
    """A job that runs for a minute the first time and answers at once the next: how many times it started."""
    path = Path(marker)
    starts = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(starts))
    if starts == 1:
        time.sleep(60)
    return {"starts": starts}


def test_a_job_a_restart_of_the_web_app_cut_short_runs_again_when_the_web_app_starts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[4]))  # where the job's process finds slow_once by name
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    marker = tmp_path / "starts"
    jobs = Jobs(depot.address, STATUSES)
    job = jobs.add("vocabulary", "a slow job", slow_once, str(marker), recipe="r")["id"]
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
    """A job that starts a worker process of its own, as a corpus build's tokenizing does: the worker's exit code."""
    import multiprocessing
    worker = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(0,))
    worker.start()
    worker.join()
    return {"exit": worker.exitcode}


def test_a_job_may_start_worker_processes_of_its_own(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[4]))  # where the job's process finds from_a_worker by name
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    jobs = Jobs(depot.address, STATUSES)
    job = jobs.add("corpus", "a job with a worker", from_a_worker, dataset="d")["id"]
    jobs.start()
    try:
        found = jobs.wait(job, 60)
    finally:
        jobs.stop()
    assert (found["state"], found["answer"]) == ("done", {"exit": 0})


def test_a_small_write_runs_in_the_request_while_a_job_that_holds_no_write_lock_runs(served, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[4]))  # where the job's process finds slow_once by name
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    [text] = depot.add([NewText(text="the fox", licenses=("CC0-1.0",))], ME)
    call = served(depot)
    marker = tmp_path / "starts"
    Jobs(depot.address).add("vocabulary", "a slow training", slow_once, str(marker), recipe="r")
    deadline = time.monotonic() + 60
    while not marker.exists():  # the training runs
        assert time.monotonic() < deadline
        time.sleep(0.05)

    status, _, state = call("POST", "/api/decisions", {"status": "admitted", "selection": {"filter": {"ids": [text]}},
                                                        "declaration": {"name": "Ada Lovelace",
                                                                        "email": "ada@example.org", "accepted": True}})
    assert (status, state["state"]) == (200, "done")  # answered at once, not queued behind the training


def test_a_sample_whose_reader_went_away_leaves_the_sample_route_to_the_next_request(served, tmp_path):
    # the sample is far bigger than the socket's buffers, so its producer meets a full queue once the reader is gone
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    words = [hashlib.sha256(str(n).encode()).hexdigest() for n in range(4096)]
    depot.add([NewText(text=" ".join(words[n:] + words[:n]), licenses=("CC0-1.0",)) for n in range(400)], ME)
    depot.decide("admitted", "true", ME)
    library = depot.new_library("all")
    depot.put(library, "true")
    depot.new_dataset("first", "tiny", [(None, library)])
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
