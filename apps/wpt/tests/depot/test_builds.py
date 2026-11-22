import base64
import hashlib
import http.client
import io
import json
import os
import pickle
import socket
import sqlite3
import subprocess
import tarfile
import threading
import time
from compression import zstd
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pyroaring import BitMap

from wpt_depot import About, Declaration, Depot, NewText, corpora, pieces, snapshots
from wpt_depot.corpus import build
from wpt_depot.legacy import Legacy
from wpt_recipes import Recipes
from wpt_store import Store, store_from_settings
from wpt.depot import shared
from wpt.depot import jobs as jobs_module
from wpt.depot.downloads import Downloads
from wpt.depot.jobs import DATASETS_GONE, WRITES_BEFORE, Jobs, connected
from wpt.depot.web.recipes import build_job
from wpt.depot.web.routes import STATUSES, routes
from wpt.depot.web.server import make_server

ME = Declaration("Ada Lovelace", "ada@example.org", True)
TEXTS = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs",
         "how vexingly quick daft zebras jump", "sphinx of black quartz, judge my vow"]
PUBLIC = "https://depot.example"  # the served depot's public address, WPT_DEPOT_URL


def legacy_dataset(depot: Depot, name: str) -> str:
    """The rows of the legacy tables, which nothing writes any more, of an open library holding every admitted
    text of depot and of a dataset called name reading it with the tokenizer recipe "tiny", as a depot of format
    12 left them for the frozen dataset routes (wpt.depot.web.legacy): the dataset's id."""
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
    (legacy_dataset), with its snapshot "s1" of them, whose sample a vocabulary trains on."""
    depot = Depot.new(address, About(name, "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add([NewText(text=text, licenses=("CC0-1.0",), tags=tuple(tagged)) for text, tagged in zip(TEXTS, tags)],
              ME)
    depot.decide("admitted", "true", ME)
    legacy_dataset(depot, "first")
    depot.create_snapshot("s1")
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


def trained(call, depot) -> str:
    """Make the tokenizer recipe "tiny" of the snapshot "s1" and train its vocabulary through the routes,
    confirmed: its sha256."""
    assert call("POST", "/api/tokenizers", {"name": "tiny", "snapshot": "s1", "sample_bytes": 1000,
                                           "vocab_size": 560})[0] == 200
    assert ended(call, call("POST", "/api/tokenizers/tiny/train", {"confirmed": True})[2]["id"])["state"] == "done"
    return call("GET", "/api/tokenizers/tiny")[2]["vocabularies"][0]["sha256"]


def test_a_vocabulary_training_starts_only_once_confirmed_after_its_estimate(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    call("POST", "/api/tokenizers", {"name": "tiny", "snapshot": "s1", "sample_bytes": 1000, "vocab_size": 560})

    status, _, estimate = call("GET", "/api/tokenizers/tiny/estimate")
    assert status == 200 and estimate["sample_bytes"] == 1000 and estimate["seconds"] == 0  # 1000 bytes: no time
    assert call("POST", "/api/tokenizers/tiny/train", None)[0] == 400
    assert call("POST", "/api/tokenizers/tiny/train", {"confirmed": False})[0] == 422
    assert call("GET", "/api/jobs")[2] == []
    assert call("POST", "/api/tokenizers/tiny/train", {"confirmed": True})[2]["kind"] == "vocabulary"


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
    assert call("GET", "/api/jobs")[2] == []  # nothing was queued, and no tokenizer recipe exists


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
    job = jobs.add("corpus", "a job with a worker", from_a_worker, key="d")["id"]
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
    monkeypatch.syspath_prepend(str(Path(__file__).parents[4]))  # where the job's process finds answered by name
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


def test_a_vocabulary_lists_a_long_text_once_whatever_pieces_of_it_its_sample_took(tmp_path):
    # the sample takes a long text as pieces, each at its own place; the list of the texts the vocabulary trained
    # on, and its count, name the text once
    long_text = "".join(f"=== src/file{n}.txt\n" + "words of a file\n" * 2000 for n in range(110))
    assert len(long_text) > 3 * pieces.LENGTH
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("here", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add([NewText(text=text, licenses=("CC0-1.0",)) for text in (*TEXTS, long_text)], ME)
    depot.decide("admitted", "true", ME)
    depot.create_snapshot("s1")
    depot.new_tokenizer("tiny", "s1", None, 10 * pieces.LENGTH, 560)

    shared.train_vocabulary(depot, "tiny", lambda *_: None, work=tmp_path)

    recipe = depot.tokenizer("tiny")
    listed = [one["text"] for one in depot.vocabulary_texts(recipe["id"], 1)]
    assert sorted(listed) == sorted(set(listed)) and len(listed) == len(TEXTS) + 1  # each text once, the long one too
    assert depot.vocabulary(recipe["id"], 1)["texts"] == len(TEXTS) + 1


# Snapshots, recipes and builds

REPOSITORY = Path(__file__).parents[4]  # this checkout, whose recipes give a model part that passes its schema


def checkout(tmp_path, recipes: dict[str, dict], series: dict[str, dict] = {}) -> Path:
    """A checkout of a repository holding the model recipes {name: data section}, each with the model part of a
    recipe of this checkout, and the series {name: data section}, each with the rest of a series of this checkout,
    committed: its folder."""
    model = Recipes(REPOSITORY).recipe("model", Recipes(REPOSITORY).names("model")[0])
    folder = tmp_path / "repository"
    (folder / "data" / "recipes" / "models").mkdir(parents=True)
    for name, data in recipes.items():
        (folder / "data" / "recipes" / "models" / f"{name}.json").write_text(json.dumps({**model, "data": data}),
                                                                            encoding="utf-8")
    if series:
        whole = Recipes(REPOSITORY).recipe("series", Recipes(REPOSITORY).names("series")[0])
        (folder / "data" / "recipes" / "series").mkdir(parents=True)
        for name, data in series.items():
            (folder / "data" / "recipes" / "series" / f"{name}.json").write_text(
                json.dumps({**whole, "data": data}), encoding="utf-8")
    git = ["git", "-C", str(folder), "-c", "user.name=Test", "-c", "user.email=test@example.org"]
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    subprocess.run([*git, "init", "-q"], check=True, env=env)
    subprocess.run([*git, "add", "."], check=True, env=env)
    subprocess.run([*git, "commit", "-q", "-m", "recipes"], check=True, env=env)
    return folder


def data_of(held, vocabulary: str, size: int | None, total_tokens=1000, **changes) -> dict:
    """The data section of a recipe of the depot's snapshot s1 at total_tokens, with the vocabulary of that sha256
    and size."""
    row = held.snapshot("s1")
    return {"depot": PUBLIC, "snapshot": "s1", "change": row["at_change"], "digest": row["digest"], "kinds": [],
            "shares": [], "total_tokens": total_tokens, "vocabulary": {"sha256": vocabulary, "size": size}, **changes}


def test_a_snapshot_is_made_as_a_job_once_confirmed_with_the_estimate_it_was_shown(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    status, _, offered = call("GET", "/api/snapshots/new")
    month = datetime.now(UTC).strftime("%Y-%m")
    assert status == 200 and offered["name"] == month and offered["estimate"]["texts"] == len(TEXTS)
    shown = offered["estimate"]

    assert call("POST", "/api/snapshots", {"name": month, "confirmed": True})[0] == 400  # no estimate given back
    assert call("POST", "/api/snapshots", {"name": month, "confirmed": False, "estimate": shown})[0] == 422
    status, _, refused = call("POST", "/api/snapshots", {"name": month, "confirmed": True,  # it now costs more
                                                         "estimate": {**shown, "seconds": -100}})
    assert status == 409 and refused["estimate"]["texts"] == len(TEXTS)
    assert call("POST", "/api/snapshots", {"name": "s1", "confirmed": True, "estimate": shown})[0] == 422  # taken
    assert call("GET", "/api/jobs")[2] == []  # nothing was queued

    status, _, job = call("POST", "/api/snapshots", {"name": month, "note": "monthly", "confirmed": True,
                                                     "estimate": shown})
    assert (status, job["kind"], job["lane"], job["key"]) == (200, "snapshot", "heavy", month)
    state = ended(call, job["id"])
    assert state["state"] == "done", state
    assert [one["stage"] for one in state["stages"]] == ["reading texts", "catching up", "writing"]
    row = depot.snapshot(month)
    assert state["answer"] == {"id": row["id"], "name": month, "at_change": row["at_change"], "texts": len(TEXTS),
                               "digest": row["digest"]}
    assert call("GET", "/api/snapshots/new")[2]["name"] == f"{month}b"
    assert [one["name"] for one in call("GET", "/api/snapshots")[2]] == ["s1", month]


def test_a_snapshot_s_page_gives_its_summary_what_erasures_did_since_and_sets_anyone_can_check(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    row = depot.snapshot("s1")

    status, _, page = call("GET", f"/api/snapshots/{row['id']}")
    assert status == 200 and page["summary"]["kinds"]["plain"][0] == len(TEXTS) and page["recipes"] == []
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


def test_recipes_are_listed_from_the_checkout_wpt_repository_names_with_its_commit(served, tmp_path, monkeypatch):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    monkeypatch.delenv(shared.REPOSITORY, raising=False)
    status, _, listed = call("GET", "/api/recipes")
    assert status == 200 and listed["repository"] is None and shared.REPOSITORY in listed["problem"]
    assert call("GET", "/api/recipes/one")[0] == 422  # no repository: said, not a crash

    vocabulary = trained(call, depot)
    size = shared.vocabulary_by_sha256(depot, vocabulary)[2].vocab_size
    folder = checkout(tmp_path, {
        "ready": data_of(depot, vocabulary, size),
        "elsewhere": data_of(depot, vocabulary, size, depot="https://other.example"),
        "moved": data_of(depot, vocabulary, size, change=0),
        "untrained": data_of(depot, "0" * 64, size),
        "old": {"depot": PUBLIC, "dataset": "first", "total_tokens": 1000}})
    monkeypatch.setenv(shared.REPOSITORY, str(folder))
    commit = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True,
                            check=True).stdout.decode().strip()

    status, _, listed = call("GET", "/api/recipes")

    assert status == 200 and listed["repository"] == {"commit": commit, "dirty": False}
    states = {one["name"]: one["state"] for one in listed["recipes"]}
    assert states == {"ready": "ready", "elsewhere": "snapshot not here", "moved": "snapshot differs",
                      "untrained": "no vocabulary", "old": "dataset"}
    ready = next(one for one in listed["recipes"] if one["name"] == "ready")
    snapshot = depot.snapshot("s1")
    assert ready["build_key"] == corpora_key(snapshot, ready["data"], vocabulary) and ready["corpus"] is None
    assert call("GET", f"/api/snapshots/{snapshot['id']}")[2]["recipes"] == [
        {"name": "moved", "matches": False}, {"name": "ready", "matches": True},
        {"name": "untrained", "matches": True}]
    assert call("GET", "/api/recipes/nothing")[0] == 404


def test_a_series_is_listed_as_series_colon_with_the_build_key_every_size_of_it_trains_from(served, tmp_path,
                                                                                            monkeypatch):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    vocabulary = trained(call, depot)
    size = shared.vocabulary_by_sha256(depot, vocabulary)[2].vocab_size
    section = {key: value for key, value in data_of(depot, vocabulary, size).items() if key != "total_tokens"}
    folder = checkout(tmp_path, {}, {"tiny": section})
    monkeypatch.setenv(shared.REPOSITORY, str(folder))

    [listed] = call("GET", "/api/recipes")[2]["recipes"]

    assert (listed["name"], listed["file"], listed["state"]) == ("tiny:", "data/recipes/series/tiny.json", "ready")
    whole = Recipes(folder).data_part("tiny:")  # the series' data part at its largest size
    assert whole["total_tokens"] > 0
    size_name = f"tiny:{next(iter(Recipes(folder).recipe('series', 'tiny')['sizes']))['name']}"
    assert Recipes(folder).data_part(size_name) == whole  # what wpt train SERIES:SIZE keys its corpus by
    assert listed["build_key"] == corpora_key(depot.snapshot("s1"), whole, vocabulary)
    status, _, page = call("GET", "/api/recipes/tiny:")
    assert status == 200 and page["state"] == "ready" and page["problems"] == [] and page["estimate"] is not None
    assert call("GET", f"/api/snapshots/{depot.snapshot('s1')['id']}")[2]["recipes"] == [
        {"name": "tiny:", "matches": True}]


def test_a_corpus_record_names_the_depot_the_recipe_names_not_the_one_that_builds_it(served, tmp_path):
    # a clone builds a recipe of its upstream: the record, and the provenance of a model trained on it, name the
    # upstream, as the recipe does
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    vocabulary = trained(call, depot)
    size = shared.vocabulary_by_sha256(depot, vocabulary)[2].vocab_size
    data = data_of(depot, vocabulary, size, depot="https://upstream.example")

    record = shared.build_corpus(depot, depot.snapshot("s1")["id"], data, lambda line: None)

    assert record["depot"] == "https://upstream.example"


def corpora_key(snapshot, data, vocabulary):
    from wpt_depot.corpus import build_key
    return build_key(snapshot, data, vocabulary)


def test_a_recipe_builds_only_once_confirmed_with_its_estimate_and_its_corpus_downloads_by_build_key(
        served, tmp_path, monkeypatch):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    vocabulary = trained(call, depot)
    size = shared.vocabulary_by_sha256(depot, vocabulary)[2].vocab_size
    monkeypatch.setenv(shared.REPOSITORY, str(checkout(tmp_path, {"ready": data_of(depot, vocabulary, size)})))

    status, _, page = call("GET", "/api/recipes/ready")

    assert status == 200 and page["state"] == "ready" and page["job"] is None
    assert page["measure"]["plain"]["texts"] == len(TEXTS) and page["measure"]["shares"][0]["tag"] is None
    shown = page["estimate"]
    assert shown["build_key"] == page["build_key"] and shown["tokens"] > 0 and shown["bytes"] > 0
    assert (shown["cap"], shown["evicted"], shown["fits"], shown["built"]) == (None, [], True, None)
    assert call("POST", "/api/recipes/ready/build", {"confirmed": True})[0] == 400
    assert call("POST", "/api/recipes/ready/build", {"confirmed": False, "estimate": shown})[0] == 422
    assert call("POST", "/api/recipes/ready/build", {"confirmed": True,
                                                     "estimate": {**shown, "build_key": "0" * 64}})[0] == 409
    status, _, refused = call("POST", "/api/recipes/ready/build", {"confirmed": True,
                                                                   "estimate": {**shown, "bytes": -10**9}})
    assert status == 409 and refused["estimate"]["bytes"] == shown["bytes"]  # the new estimate, to confirm
    assert call("GET", "/api/jobs")[2][0]["kind"] == "vocabulary"  # only the training: nothing was queued

    status, _, job = call("POST", "/api/recipes/ready/build", {"confirmed": True, "estimate": shown})
    assert (status, job["kind"], job["lane"], job["key"], job["recipe"]) == (200, "corpus", "heavy",
                                                                              page["build_key"], "ready")
    state = ended(call, job["id"])
    assert state["state"] == "done", state
    record = corpora.of_key(store_from_settings(), page["build_key"])
    assert record["id"] == state["answer"]["corpus"] and record["recipe"] == "ready" and record["depot"] == PUBLIC
    assert record["code_commit"] == call("GET", "/api/recipes")[2]["repository"]["commit"]
    assert call("POST", "/api/recipes/ready/build", {"confirmed": True, "estimate": call(
        "GET", "/api/recipes/ready")[2]["estimate"]})[0] == 422  # built already

    status, _, built = call("GET", f"/api/builds/{page['build_key']}")
    assert status == 200 and built["state"] == "built" and built["corpus"]["id"] == record["id"]
    first = built["files"][0]
    status, headers, data = call("GET", f"/api/builds/{page['build_key']}/files/{first['name']}")
    assert status == 200 and hashlib.sha256(data).hexdigest() == first["sha256"]
    assert headers["Content-Disposition"].startswith(f'attachment; filename="ready_{page["build_key"][:12]}_')
    status, _, whole = call("GET", f"/api/builds/{page['build_key']}/everything.tar")
    assert status == 200 and len(whole) == built["archive"]["bytes"]
    with tarfile.open(fileobj=io.BytesIO(whole)) as held:
        assert sorted(held.getnames()) == sorted(one["download"] for one in built["files"])
    assert call("GET", f"/api/builds/{'0' * 64}")[2]["state"] == "not built"
    assert call("GET", f"/api/builds/{'0' * 64}/everything.tar")[0] == 404


def built_corpus(call, depot) -> tuple[str, dict]:
    """(the build key, the record) of a corpus of the snapshot s1 in several token files, built in the test."""
    vocabulary = trained(call, depot)
    recipe, version, tokenizer = shared.vocabulary_by_sha256(depot, vocabulary)
    record = build(depot, depot.snapshot("s1")["id"], data_of(depot, vocabulary, tokenizer.vocab_size),
                   store_from_settings(), tokenizer, address=PUBLIC, tokenizer_recipe=recipe["name"],
                   tokenizer_version=version, shard_tokens=8, log=lambda *args: None)
    return record["build_key"], record


def test_a_corpus_file_and_the_archive_of_its_build_key_send_any_range_of_their_bytes(served, tmp_path,
                                                                                      monkeypatch):
    # a download resumes: curl -C - asks the rest with a Range, and If-Range with the ETag it saw
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    key, record = built_corpus(call, depot)
    sized = Counter()
    size = Store.size
    monkeypatch.setattr(Store, "size", lambda self, what: sized.update([what.key]) or size(self, what))
    link = f"/api/builds/{key}"
    shown = call("GET", link)[2]
    assert shown["state"] == "built" and len([one for one in shown["files"] if one["role"] == "plain tokens"]) > 1
    call("GET", f"{link}/everything.tar", headers={"Range": "bytes=0-0"})
    assert not sized  # the record names every file's size: the store is asked none

    status, headers, whole = call("GET", f"{link}/everything.tar")
    assert status == 200 and int(headers["Content-Length"]) == len(whole) == shown["archive"]["bytes"]
    assert headers["Accept-Ranges"] == "bytes" and call("GET", f"{link}/everything.tar")[2] == whole
    with tarfile.open(fileobj=io.BytesIO(whole)) as tar:
        members = tar.getmembers()
        found = {one.name: hashlib.sha256(tar.extractfile(one).read()).hexdigest() for one in members}
    assert found == {one["download"]: one["sha256"] for one in shown["files"]}
    # one range in a header, in a file, in the padding after a file and across them: the archive's bytes there
    padded = next(one for one in members if one.size % 512)
    ends = padded.offset_data + padded.size
    for start, end in ((padded.offset + 100, padded.offset + 300), (padded.offset_data + 1, ends - 1),
                       (ends + 1, ends + 3), (padded.offset + 10, ends + 5)):
        status, headers, piece = call("GET", f"{link}/everything.tar", headers={"Range": f"bytes={start}-{end}"})
        assert (status, piece, headers["Content-Range"]) == (206, whole[start:end + 1],
                                                            f"bytes {start}-{end}/{len(whole)}")
    assert call("GET", f"{link}/everything.tar", headers={"Range": "bytes=-1500"})[:3:2] == (206, whole[-1500:])
    assert call("GET", f"{link}/everything.tar", headers={"Range": f"bytes={len(whole)}-"})[0] == 416
    etag = call("GET", f"{link}/everything.tar", headers={"Range": "bytes=0-0"})[1]["ETag"]
    assert call("GET", f"{link}/everything.tar", headers={"Range": "bytes=10-20", "If-Range": etag})[:3:2] == (
        206, whole[10:21])
    assert call("GET", f"{link}/everything.tar", headers={"Range": "bytes=10-20", "If-Range": '"other"'})[:3:2] == (
        200, whole)

    # one file the same way, its ETag its sha256
    first = shown["files"][0]
    status, headers, data = call("GET", f"{link}/files/{first['name']}")
    assert (status, headers["ETag"], hashlib.sha256(data).hexdigest()) == (200, f'"{first["sha256"]}"', first["sha256"])
    last = len(data) - 1
    status, headers, piece = call("GET", f"{link}/files/{first['name']}", headers={"Range": f"bytes=1-{last}"})
    assert (status, piece, headers["Content-Range"]) == (206, data[1:], f"bytes 1-{last}/{len(data)}")
    asked = {"Range": "bytes=2-", "If-Range": headers["ETag"]}
    assert call("GET", f"{link}/files/{first['name']}", headers=asked)[:3:2] == (206, data[2:])
    asked = {"Range": "bytes=2-", "If-Range": '"other"'}
    assert call("GET", f"{link}/files/{first['name']}", headers=asked)[:3:2] == (200, data)
    assert call("GET", f"{link}/files/{first['name']}", headers={"Range": f"bytes={len(data)}-"})[0] == 416
    assert call("GET", f"{link}/files/nothing")[0] == 404


def test_the_corpus_of_a_build_key_is_held_back_while_an_erasure_is_checked_and_stale_once_it_reached_it(
        served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    key, record = built_corpus(call, depot)
    link = f"/api/builds/{key}"
    token_file = f"{link}/files/{call('GET', link)[2]['files'][0]['name']}"
    # while an erasure's corpora are being marked, nothing is sent: here one the web app never carried out
    Downloads(Jobs(depot.address).path).marking(["test-depot-9"])
    assert call("GET", link)[2]["state"] == "checking an erasure"
    assert call("GET", token_file)[0] == 422
    assert call("GET", f"{link}/everything.tar")[0] == 422
    declaration = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True}
    asked = call("POST", "/api/erasures", {"reason": "a fox", "needles": [{"text": "test-depot-1", "start": 16,
                                                                             "end": 19}], "declaration": declaration})
    erasure = asked[2]["id"]
    assert call("POST", "/api/erasures/decisions", {"erasures": [erasure], "status": "admitted",
                                                    "declaration": declaration})[0] == 200

    deadline = time.monotonic() + 60  # the corpora are marked in the background
    while (shown := call("GET", link)[2])["state"] == "checking an erasure":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert (shown["state"], shown["corpus"]["stale"]) == ("stale", {"erasures": [erasure]})
    assert call("GET", token_file)[0] == 404
    assert call("GET", f"{link}/everything.tar")[0] == 404


def test_a_sample_whose_reader_went_away_leaves_the_sample_route_to_the_next_request(served, tmp_path):
    # the frozen sample route of a dataset (wpt.depot.web.legacy): the sample is far bigger than the socket's
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


def test_a_recipe_is_composed_from_clicks_with_its_measure_and_estimate_and_nothing_written(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot", tags=[["a"], ["b"], ["a"], []])
    call = served(depot)
    vocabulary = trained(call, depot)
    snapshot = depot.snapshot("s1")
    change = depot.change()

    status, _, composed = call("POST", "/api/recipes/compose", {
        "snapshot": snapshot["id"], "kinds": ["plain"], "shares": [{"tag": "a", "share": 0.75},
                                                                   {"tag": "b", "share": 0.25}],
        "total_tokens": 500, "vocabulary": vocabulary})

    assert status == 200
    data = composed["recipe"]["data"]
    assert (data["depot"], data["snapshot"], data["change"], data["digest"]) == (PUBLIC, "s1", snapshot["at_change"],
                                                                                 snapshot["digest"])
    assert data["vocabulary"]["sha256"] == vocabulary and data["total_tokens"] == 500
    assert [(one["tag"], one["texts"]) for one in composed["measure"]["shares"]] == [("a", 2), ("b", 1)]
    assert composed["measure"]["no_share"] == 1 and composed["estimate"]["build_key"] == composed["build_key"]
    assert depot.change() == change and len(call("GET", "/api/jobs")[2]) == 1  # the training alone
    assert call("POST", "/api/recipes/compose", {"snapshot": snapshot["id"], "total_tokens": 500,
                                                 "vocabulary": "0" * 64})[0] == 422


def test_a_corpus_job_a_restart_cut_short_goes_on_when_the_web_app_starts_again(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    vocabulary = trained(call, depot)
    size = shared.vocabulary_by_sha256(depot, vocabulary)[2].vocab_size
    data = data_of(depot, vocabulary, size)
    snapshot = depot.snapshot("s1")
    key = corpora_key(snapshot, data, vocabulary)
    stopped = Jobs(depot.address, STATUSES)  # no runner: the job stays queued until marked as a crash left it
    job = stopped.add("corpus", "build s1", build_job, snapshot["id"], data, vocabulary, key=key)["id"]
    with connected(stopped.path, write=True) as con:  # the web app stopped while the job ran
        con.execute("UPDATE jobs SET state = 'running', started = ? WHERE id = ?", (time.time(), job))
    with connected(stopped.path, write=True) as con:
        con.execute("INSERT INTO checkpoints (key, state) VALUES (?, ?)", (key, json.dumps({})))

    again = Jobs(depot.address, STATUSES)
    again.start()
    try:
        found = again.wait(job, 300)
    finally:
        again.stop()

    assert (found["state"], found["restarts"]) == ("done", 1), found
    assert corpora.of_key(store_from_settings(), key)["id"] == found["answer"]["corpus"]
    with connected(stopped.path) as con:  # the checkpoint of its key forgotten once it ended
        assert con.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0


def written_once(depot, text: str, *, progress) -> dict:
    """A write job: the text admitted."""
    decision, texts, _ = depot.decide("admitted", f"id = '{text}'", ME)
    return {"texts": texts}


def test_writes_inline_and_as_jobs_run_while_a_heavy_job_runs(served, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[4]))  # where the job's process finds the operations
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    first, second = depot.add([NewText(text="the fox", licenses=("CC0-1.0",)),
                               NewText(text="the dog", licenses=("CC0-1.0",))], ME)
    call = served(depot)
    marker = tmp_path / "starts"
    jobs = Jobs(depot.address, STATUSES)
    heavy = jobs.add("corpus", "a slow build", slow_once, str(marker), key="k")["id"]
    deadline = time.monotonic() + 60
    while not marker.exists():  # the heavy job runs
        assert time.monotonic() < deadline
        time.sleep(0.05)

    status, _, state = call("POST", "/api/decisions", {"status": "admitted", "selection": {"filter": {"ids": [first]}},
                                                        "declaration": {"name": "Ada Lovelace",
                                                                        "email": "ada@example.org", "accepted": True}})
    assert (status, state["state"]) == (200, "done")  # answered at once, not queued behind the build
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
    assert (found["a"]["state"], found["a"]["problem"], found["a"]["key"]) == ("failed", DATASETS_GONE, "wic-1")
    assert found["b"]["state"] == "failed" and "no longer exists" in found["b"]["problem"]
    assert (found["c"]["state"], found["c"]["lane"]) == ("queued", "heavy")
    assert (found["d"]["state"], found["d"]["lane"]) == ("done", "write")
    # its selection may have named a library, which unpickles as nothing: run, it would reach every text
    assert (found["e"]["state"], found["e"]["problem"]) == ("failed", WRITES_BEFORE)
    with connected(path) as con:
        assert con.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0
    Jobs(depot.address, STATUSES)  # brought up once: opening it again changes nothing
    assert {one["id"]: one["state"] for one in jobs.listed()} == {key: one["state"] for key, one in found.items()}
