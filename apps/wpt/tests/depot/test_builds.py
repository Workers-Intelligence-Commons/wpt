import hashlib
import http.client
import io
import json
import re
import socket
import subprocess
import tarfile
import threading
import time
from collections import Counter
from compression import zstd
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from wpt_depot import About, Declaration, Depot, NewText, pieces
from wpt_depot.corpus import build
from wpt_store import Store, store_from_settings
from wpt_utils.programs import program
from wpt.depot import shared
from wpt.depot.downloads import Downloads
from wpt.depot.jobs import Jobs
from wpt.depot.web.routes import STATUSES, routes
from wpt.depot.web.server import make_server

ME = Declaration("Ada Lovelace", "ada@example.org", True)
TEXTS = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs",
         "how vexingly quick daft zebras jump", "sphinx of black quartz, judge my vow"]
PUBLIC = "https://depot.example"  # the served depot's public address, WPT_DEPOT_URL


def depot_with_texts(address: str, name: str, tags=((),) * len(TEXTS)) -> Depot:
    """A new depot whose four plain texts, admitted, each with its tags, fill the library "all", which the
    dataset "first" reads with the tokenizer recipe "tiny" that it does not hold yet."""
    depot = Depot.new(address, About(name, "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add([NewText(text=text, licenses=("CC0-1.0",), tags=tuple(tagged)) for text, tagged in zip(TEXTS, tags)],
              ME)
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


def trained(call, depot) -> str:
    """Make the tokenizer recipe "tiny" the dataset "first" names and train its vocabulary through the routes,
    confirmed: its sha256."""
    assert call("POST", "/api/tokenizers", {"name": "tiny", "dataset": "first", "sample_bytes": 1000,
                                           "vocab_size": 560})[0] == 200
    assert ended(call, call("POST", "/api/tokenizers/tiny/train", {"confirmed": True})[2]["id"])["state"] == "done"
    return call("GET", "/api/tokenizers/tiny")[2]["vocabularies"][0]["sha256"]


def test_a_vocabulary_training_starts_only_once_confirmed_after_its_estimate(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    call("POST", "/api/tokenizers", {"name": "tiny", "dataset": "first", "sample_bytes": 1000, "vocab_size": 560})

    status, _, estimate = call("GET", "/api/tokenizers/tiny/estimate")
    assert status == 200 and estimate["sample_bytes"] == 1000 and estimate["seconds"] == 0  # 1000 bytes: no time
    assert call("POST", "/api/tokenizers/tiny/train", None)[0] == 400
    assert call("POST", "/api/tokenizers/tiny/train", {"confirmed": False})[0] == 422
    assert call("GET", "/api/jobs")[2] == []
    assert call("POST", "/api/tokenizers/tiny/train", {"confirmed": True})[2]["kind"] == "vocabulary"


def test_a_build_needs_a_trained_vocabulary_a_size_and_a_confirmation_and_each_size_has_its_own_link(
        served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    assert call("GET", "/api/about")[2]["address"] == PUBLIC  # what names it in its corpora
    dataset = depot.dataset("first")["id"]
    base = f"/api/datasets/{dataset}"
    untrained = "0" * 64
    assert call("POST", f"{base}/build", {"vocabulary": untrained, "max_tokens": "all", "confirmed": True})[0] == 422
    vocabulary = trained(call, depot)
    link = f"{base}/download/{vocabulary}/all"
    status, _, shown = call("GET", link)
    assert status == 200 and shown["state"] == "not built" and shown["job"] is None
    for refused in ({"max_tokens": "all", "confirmed": True}, {"vocabulary": vocabulary, "confirmed": True},
                    {"vocabulary": vocabulary, "max_tokens": "all"}, {}):
        assert call("POST", f"{base}/build", refused)[0] == 400
    assert call("POST", f"{base}/build", {"vocabulary": vocabulary, "max_tokens": "all", "confirmed": False})[0] == 422
    assert [one["kind"] for one in call("GET", "/api/jobs")[2]] == ["vocabulary"]  # neither link nor refusal queued

    status, _, job = call("POST", f"{base}/build", {"vocabulary": vocabulary, "max_tokens": "all", "confirmed": True})
    assert status == 200 and job["kind"] == "corpus" and job["target"] == {"vocabulary": vocabulary, "max_tokens": None}
    assert call("POST", f"{base}/build", {"vocabulary": vocabulary, "max_tokens": 5, "confirmed": True})[0] == 409
    assert ended(call, job["id"])["state"] == "done"
    status, _, shown = call("GET", link)
    assert status == 200 and shown["state"] == "built" and shown["corpus"]["texts"] == len(TEXTS)
    assert (shown["corpus"]["conversations"], shown["corpus"]["tasks"]) == (0, 0)  # what the texts count holds
    assert ended(call, job["id"])["answer"]["conversations"] == 0

    # a token file: whole, the same twice and as listed; one range of it; a range past its end
    listed = max(shown["files"], key=lambda one: one["bytes"] or 0)
    status, headers, whole = call("GET", f"{link}/files/{listed['name']}")
    assert status == 200 and hashlib.sha256(whole).hexdigest() == listed["sha256"]
    assert headers["Accept-Ranges"] == "bytes" and int(headers["Content-Length"]) == len(whole)
    # saved under a name that says which dataset, vocabulary and size it comes from, each file's name its own
    assert headers["Content-Disposition"] == f'attachment; filename="{listed["download"]}"'
    assert listed["download"].startswith("first_") and "_all_" in listed["download"]
    assert len({one["download"] for one in shown["files"]}) == len(shown["files"])
    # the vocabulary, stored with no extension, is saved and sent as the JSON it is
    words = next(one for one in shown["files"] if one["role"] == "vocabulary")
    status, headers, data = call("GET", f"{link}/files/{words['name']}")
    assert words["download"].endswith("_vocabulary.json") and headers["Content-Type"] == "application/json"
    assert status == 200 and data["format"] == "wpt-bpe" and data["merges"]  # read as JSON by the helper
    assert call("GET", f"{link}/files/{listed['name']}")[2] == whole
    status, headers, piece = call("GET", f"{link}/files/{listed['name']}", headers={"Range": "bytes=2-9"})
    assert (status, piece, headers["Content-Range"]) == (206, whole[2:10], f"bytes 2-9/{len(whole)}")
    assert call("GET", f"{link}/files/{listed['name']}", headers={"Range": f"bytes={len(whole)}-"})[0] == 416

    # another size of the same dataset and vocabulary: another corpus, at its own link, beside the first
    small = f"{base}/download/{vocabulary}/5"
    assert call("GET", small)[2]["state"] == "not built"
    assert ended(call, call("POST", f"{base}/build", {"vocabulary": vocabulary, "max_tokens": 5,
                                                      "confirmed": True})[2]["id"])["state"] == "done"
    assert call("GET", small)[2]["state"] == "built" and call("GET", link)[2]["state"] == "built"
    assert call("GET", small)[2]["corpus"]["id"] != call("GET", link)[2]["corpus"]["id"]
    listing = call("GET", f"{base}/download")[2]
    assert sorted((one["size"], one["state"], one["link"]) for one in listing["corpora"]) == [
        ("5", "built", small), ("all", "built", link)]


def test_a_built_corpus_page_gives_each_share_s_tokens_and_each_file_its_role_tokens_and_numbered_name(
        served, tmp_path):
    # the corpus page compares the shares asked with those built and warns when no plain text was taken from them
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot", tags=[["a"], ["b"], ["a"], ["b"]])
    call = served(depot)
    row = depot.dataset("first")
    depot.edit_dataset(row["id"], "first", [(None, one["library"]) for one in row["libraries"]],
                       [("a", 0.5), ("b", 0.5)])
    vocabulary = trained(call, depot)
    recipe, version, tokenizer = shared.vocabulary_by_sha256(depot, vocabulary)
    # built here rather than by the queue, so its token files can be small: one for every few tokens
    record = build(depot, "first", store_from_settings(), tokenizer, version, url=shared.public_url(),
                   max_tokens=None, recipe=recipe["name"], shard_tokens=8, log=lambda *args: None)
    shown = call("GET", f"/api/datasets/{row['id']}/download/{vocabulary}/all")[2]

    corpus = shown["corpus"]
    assert shown["state"] == "built" and corpus["id"] == record["id"]
    assert [one["tag"] for one in corpus["shares_tokens"]] == ["a", "b"]
    assert sum(one["tokens"] for one in corpus["shares_tokens"]) == corpus["total_tokens"] > 0
    assert corpus["items_tokens"] == 0  # no conversation or task in the dataset
    # the dataset's page counts the same corpus with the same figures, so both pages show one size for it
    listed = call("GET", f"/api/datasets/{row['id']}")[2]["corpora"]
    assert [(one["id"], one["texts"], one["total_tokens"], one["items_tokens"]) for one in listed] == [
        (corpus["id"], corpus["texts"], corpus["total_tokens"], corpus["items_tokens"])]
    texts = [one for one in shown["files"] if one["role"] == "text tokens"]
    assert len(texts) > 1 and sum(one["tokens"] for one in texts) == corpus["total_tokens"]
    numbers = [re.fullmatch(r"first_.+_all_text-tokens-(\d+)\..+", one["download"]) for one in texts]
    assert [int(found[1]) for found in numbers] == list(range(1, len(texts) + 1))
    assert len({one["download"] for one in shown["files"]}) == len(shown["files"])


def test_the_archive_of_a_corpus_holds_every_file_under_its_download_name_and_any_range_of_it(served, tmp_path,
                                                                                            monkeypatch):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    row = depot.dataset("first")
    vocabulary = trained(call, depot)
    recipe, version, tokenizer = shared.vocabulary_by_sha256(depot, vocabulary)
    build(depot, "first", store_from_settings(), tokenizer, version, url=shared.public_url(), max_tokens=None,
          recipe=recipe["name"], shard_tokens=8, log=lambda *args: None)  # several token files
    sized = Counter()
    size = Store.size
    monkeypatch.setattr(Store, "size", lambda self, what: sized.update([what.key]) or size(self, what))
    link = f"/api/datasets/{row['id']}/download/{vocabulary}/all"
    shown = call("GET", link)[2]
    assert shown["archive"]["name"] == "everything.tar"
    call("GET", link)
    call("GET", f"{link}/everything.tar", headers={"Range": "bytes=0-0"})
    assert not sized  # the record names every file's size: the store is asked none, whatever the corpus's size

    status, headers, whole = call("GET", f"{link}/everything.tar")
    assert status == 200 and int(headers["Content-Length"]) == len(whole) == shown["archive"]["bytes"]
    assert headers["Content-Disposition"] == f'attachment; filename="{shown["archive"]["download"]}"'
    assert shown["archive"]["download"] == shown["files"][0]["download"].split("_text-tokens")[0] + ".tar"
    assert call("GET", f"{link}/everything.tar")[2] == whole  # the same bytes each time
    # every file of the corpus, the token files first as the page lists them, under its download name
    with tarfile.open(fileobj=io.BytesIO(whole)) as tar:
        members = tar.getmembers()
        found = {one.name: hashlib.sha256(tar.extractfile(one).read()).hexdigest() for one in members}
    assert [one.name for one in members] == [one["download"] for one in shown["files"]]
    assert found == {one["download"]: one["sha256"] for one in shown["files"]}
    assert all((one.mode, one.uid, one.gid, one.uname, one.gname) == (0o644, 0, 0, "", "") for one in members)

    # one range in a header, in a file and in the padding after a file: the bytes of the whole archive there
    padded = next(one for one in members if one.size % 512)
    ends = padded.offset_data + padded.size
    for start, end in ((padded.offset + 100, padded.offset + 300), (padded.offset_data + 1, ends - 1),
                       (ends + 1, ends + 3), (padded.offset + 10, ends + 5)):
        status, headers, piece = call("GET", f"{link}/everything.tar", headers={"Range": f"bytes={start}-{end}"})
        assert (status, piece, headers["Content-Range"]) == (206, whole[start:end + 1],
                                                            f"bytes {start}-{end}/{len(whole)}")
    status, _, tail = call("GET", f"{link}/everything.tar", headers={"Range": "bytes=-1500"})
    assert (status, tail) == (206, whole[-1500:])
    assert call("GET", f"{link}/everything.tar", headers={"Range": f"bytes={len(whole)}-"})[0] == 416
    # a resume whose If-Range names the archive's ETag gets the range, another ETag the whole archive
    etag = call("GET", f"{link}/everything.tar", headers={"Range": "bytes=0-0"})[1]["ETag"]
    asked = {"Range": "bytes=10-20", "If-Range": etag}
    assert call("GET", f"{link}/everything.tar", headers=asked)[:3:2] == (206, whole[10:21])
    assert call("GET", f"{link}/everything.tar", headers={**asked, "If-Range": '"another"'})[:3:2] == (200, whole)

    # the system's tar lists and extracts it to the same files
    (tmp_path / "archive.tar").write_bytes(whole)
    tar = program("tar")
    listing = subprocess.run([tar, "-tvf", str(tmp_path / "archive.tar")], capture_output=True, text=True, check=True)
    assert len(listing.stdout.splitlines()) == len(members)
    (tmp_path / "out").mkdir()
    subprocess.run([tar, "-xf", str(tmp_path / "archive.tar"), "-C", str(tmp_path / "out")], check=True)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (tmp_path / "out").iterdir()} == found


def test_an_estimate_names_tokens_bytes_time_and_room_and_writes_and_queues_nothing(served, tmp_path, monkeypatch):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    dataset = depot.dataset("first")["id"]
    vocabulary = trained(call, depot)
    monkeypatch.setenv(shared.CAP, "1TB")
    jobs = call("GET", "/api/jobs")[2]
    files = sorted(path.name for path in tmp_path.rglob("*") if path.is_file())

    status, _, estimate = call("GET", f"/api/datasets/{dataset}/estimate?vocabulary={vocabulary}&max_tokens=all")

    assert status == 200 and estimate["tokens"] > 0 and estimate["bytes"] > 0 and estimate["seconds"] is not None
    assert (estimate["cap"], estimate["others"], estimate["fits"], estimate["evicts"]) == (10**12, 0, True, 0)
    assert estimate["vocabulary"] == {"sha256": vocabulary, "recipe": "tiny", "version": 1}
    assert call("GET", "/api/jobs")[2] == jobs
    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == files
    assert call("GET", f"/api/datasets/{dataset}/estimate?vocabulary={'0' * 64}")[0] == 422  # not trained
    assert call("GET", f"/api/datasets/test-depot-99/estimate?vocabulary={vocabulary}")[0] == 404
    assert call("GET", f"/api/datasets/{dataset}/estimate?vocabulary={vocabulary}&max_tokens=0")[0] == 400


def test_the_export_gives_every_text_in_parts_the_same_bytes_each_time_with_no_tokenizer_and_no_job(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    base = f"/api/datasets/{depot.dataset('first')['id']}/export"

    status, _, shown = call("GET", base)

    assert status == 200 and shown["texts"] == len(TEXTS)
    [part] = shown["parts"]
    assert part["forms"]["jsonl.zst"] == {"name": "part-00000.jsonl.zst", "download": "first_part-00000.jsonl.zst",
                                          "bytes": None, "sha256": None}
    status, headers, first = call("GET", f"{base}/part-00000.jsonl.zst")
    assert headers["Content-Disposition"] == 'attachment; filename="first_part-00000.jsonl.zst"'  # saved by its dataset's name
    assert call("GET", f"{base}/part-00000.jsonl.zst")[2] == first
    rows = [json.loads(line) for line in zstd.decompress(first).decode().splitlines()]
    assert sorted(row["text"] for row in rows) == sorted(TEXTS)
    assert all(row["depot"] == PUBLIC and row["revision"] == 1 for row in rows)  # never the depot's file
    table = call("GET", f"{base}/part-00000.parquet")[2]
    assert pq.read_table(io.BytesIO(table)).to_pylist() == rows
    known = call("GET", base)[2]["parts"][0]["forms"]["jsonl.zst"]
    assert known == {"name": "part-00000.jsonl.zst", "download": "first_part-00000.jsonl.zst", "bytes": len(first), "sha256": hashlib.sha256(first).hexdigest()}
    status, headers, piece = call("GET", f"{base}/part-00000.jsonl.zst", headers={"Range": "bytes=3-10"})
    assert (status, piece, headers["Content-Range"]) == (206, first[3:11], f"bytes 3-10/{len(first)}")
    assert call("GET", f"{base}/part-00001.parquet")[0] == 404
    assert call("GET", "/api/jobs")[2] == []  # nothing was queued, and no tokenizer recipe exists


def test_text_exports_are_bounded_in_all_and_per_visitor_while_token_files_are_not(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    dataset = depot.dataset("first")["id"]
    vocabulary = trained(call, depot)
    ended(call, call("POST", f"/api/datasets/{dataset}/build", {"vocabulary": vocabulary, "max_tokens": "all",
                                                                "confirmed": True})[2]["id"])
    shown = call("GET", f"/api/datasets/{dataset}/download/{vocabulary}/all")[2]
    text = f"/api/datasets/{dataset}/export/part-00000.jsonl.zst"
    token_file = f"/api/datasets/{dataset}/download/{vocabulary}/all/files/{shown['files'][0]['name']}"
    downloads = Downloads(Jobs(depot.address).path)  # the web app's own record of the streams being sent

    with downloads.streaming(None, "203.0.113.1", "text"):  # one export to the first visitor
        assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 429
        assert call("GET", token_file, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 200
        assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.2"})[0] == 200
        with downloads.streaming(None, "203.0.113.2", "text"):  # a second one, to another
            assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.3"})[0] == 429
    assert call("GET", text, headers={"X-Forwarded-For": "203.0.113.1"})[0] == 200  # once they ended


def test_an_admitted_erasure_makes_the_link_stale_until_a_rebuild_gives_a_corpus_without_the_passage(served, tmp_path):
    depot = depot_with_texts(str(tmp_path / "depot.sqlite"), "test-depot")
    call = served(depot)
    dataset = depot.dataset("first")["id"]
    vocabulary = trained(call, depot)
    build = {"vocabulary": vocabulary, "max_tokens": "all", "confirmed": True}
    link = f"/api/datasets/{dataset}/download/{vocabulary}/all"
    ended(call, call("POST", f"/api/datasets/{dataset}/build", build)[2]["id"])
    shown = call("GET", link)[2]
    built = shown["corpus"]["id"]
    token_file = f"{link}/files/{shown['files'][0]['name']}"
    # while an erasure's corpora are being marked, nothing is sent: here one the web app never carried out
    Downloads(Jobs(depot.address).path).marking(["test-depot-9"])
    assert call("GET", link)[2]["state"] == "checking an erasure"
    assert call("GET", token_file)[0] == 422
    declaration = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True}
    call("POST", "/api/erasures", {"reason": "a fox", "needles": [{"text": "test-depot-1", "start": 16, "end": 19}],
                                   "declaration": declaration})
    assert call("POST", "/api/erasures/decisions", {"erasures": ["test-depot-1"], "status": "admitted",
                                                    "declaration": declaration})[0] == 200

    deadline = time.monotonic() + 60  # the corpora are marked in the background
    while (shown := call("GET", link)[2])["state"] == "checking an erasure":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert (shown["state"], shown["corpus"]["stale"]) == ("stale", {"erasures": ["test-depot-1"]})
    assert call("GET", token_file)[0] == 404
    time.sleep(1.5)  # nothing builds it again by itself
    assert call("GET", link)[2]["state"] == "stale"

    assert ended(call, call("POST", f"/api/datasets/{dataset}/build", build)[2]["id"])["state"] == "done"
    shown = call("GET", link)[2]
    assert shown["state"] == "built" and shown["corpus"]["id"] != built
    rows = [json.loads(line) for line in zstd.decompress(
        call("GET", f"/api/datasets/{dataset}/export/part-00000.jsonl.zst")[2]).decode().splitlines()]
    assert "the quick brown  jumps over the lazy dog" in {row["text"] for row in rows}


def test_an_erasure_of_a_text_the_corpus_left_out_a_new_default_size_and_a_published_vocabulary_leave_the_link_built(
        served, tmp_path):
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add([NewText(text=text, licenses=("CC0-1.0",)) for text in TEXTS]
              + [NewText(text="the five boxing wizards jump quickly", licenses=("CC0-1.0",), tags=("en",))], ME)
    depot.decide("admitted", "true", ME)
    library = depot.new_library("all")
    depot.put(library, "true")
    depot.new_dataset("first", "tiny", [(None, library)])
    row = depot.dataset("first")
    call = served(depot)
    vocabulary = trained(call, depot)
    depot.edit_dataset(row["id"], "first", [(None, library)], [("en", 1.0)])  # the untagged texts are left out
    link = f"/api/datasets/{row['id']}/download/{vocabulary}/all"
    ended(call, call("POST", f"/api/datasets/{row['id']}/build",
                     {"vocabulary": vocabulary, "max_tokens": "all", "confirmed": True})[2]["id"])
    shown = call("GET", link)[2]
    assert (shown["state"], shown["corpus"]["texts"]) == ("built", 1)
    built, token_file = shown["corpus"]["id"], f"{link}/files/{shown['files'][0]['name']}"

    depot.edit_dataset(row["id"], "first", [(None, library)], [("en", 1.0)], max_tokens=1000)  # a build never reads it
    assert call("GET", link)[2]["state"] == "built"
    edit = {"name": "first", "libraries": [{"library": library}], "shares": [{"tag": "en", "share": 1.0}],
            "max_tokens": 1000}
    assert call("POST", f"/api/datasets/{row['id']}", {**edit, "vocabulary": "0" * 64})[0] == 422  # none trained so
    assert call("POST", f"/api/datasets/{row['id']}", {**edit, "vocabulary": vocabulary})[0] == 200
    assert depot.dataset("first")["vocabulary"] == vocabulary
    assert call("GET", link)[2]["state"] == "built"  # a build never reads it either
    declaration = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True}
    call("POST", "/api/erasures", {"reason": "a fox", "needles": [{"text": "test-depot-1", "start": 16, "end": 19}],
                                   "declaration": declaration})
    assert call("POST", "/api/erasures/decisions", {"erasures": ["test-depot-1"], "status": "admitted",
                                                    "declaration": declaration})[0] == 200

    deadline = time.monotonic() + 60  # the corpora are marked in the background
    while (shown := call("GET", link)[2])["state"] == "checking an erasure":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert (shown["state"], shown["corpus"]["id"]) == ("built", built)
    assert call("GET", token_file)[0] == 200
    assert [one["kind"] for one in call("GET", "/api/jobs")[2]].count("corpus") == 1  # nothing queued another build


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


def test_a_vocabulary_lists_a_long_text_once_whatever_pieces_of_it_its_sample_took(tmp_path):
    # the sample takes a long text as pieces, each at its own place; the list of the texts the vocabulary trained
    # on, and its count, name the text once
    long_text = "".join(f"=== src/file{n}.txt\n" + "words of a file\n" * 2000 for n in range(110))
    assert len(long_text) > 3 * pieces.LENGTH
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("here", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add([NewText(text=text, licenses=("CC0-1.0",)) for text in (*TEXTS, long_text)], ME)
    depot.decide("admitted", "true", ME)
    library = depot.new_library("all")
    depot.put(library, "true")
    depot.new_dataset("first", "tiny", [(None, library)])
    depot.new_tokenizer("tiny", None, "first", 10 * pieces.LENGTH, 560)

    shared.train_vocabulary(depot, "tiny", lambda *_: None, work=tmp_path)

    recipe = depot.tokenizer("tiny")
    listed = [one["text"] for one in depot.vocabulary_texts(recipe["id"], 1)]
    assert sorted(listed) == sorted(set(listed)) and len(listed) == len(TEXTS) + 1  # each text once, the long one too
    assert depot.vocabulary(recipe["id"], 1)["texts"] == len(TEXTS) + 1
