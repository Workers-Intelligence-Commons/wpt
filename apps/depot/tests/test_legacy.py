"""The frozen routes (depot_app.web.legacy) answer what the web app of commit 6b470328c7 answered, as the golden
answers of legacy-golden/ hold them, on the format-12 fixture depot and its dock, and on that depot brought to the
code's format by its upgrade; the routes of the tokenizer pages, which only a person read, are gone; and an erasure
the depot carries out leaves the corpora on the dock, and what these routes answer of them, as they were."""

import hashlib
import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import depot_app
from depot import Declaration, Depot
from depot_app import corpora
from depot_app.downloads import Downloads
from depot_app.jobs import Jobs
from depot_app.web.legacy import legacy
from depot_app.web.routes import STATUSES, refusals, routes
from dock import dock_from_settings
from depot_app.web.server import make_server

GOLDEN = Path(__file__).parent / "legacy-golden"
FIXTURES = Path(__file__).parents[3] / "packages" / "depot" / "tests" / "fixtures"
KEPT = ("Content-Type", "Content-Length", "Content-Disposition", "Content-Range", "Accept-Ranges", "ETag")
PORTS = range(8560, 8600)  # the ports a test server may take on 127.0.0.1
# the reads of a tokenizer recipe old code makes, whose answers once held more fields: their golden answers must be
# held in what they answer now, each field old code reads with its value
ADDED_TO = ("/api/tokenizers/",)
# the routes of the tokenizer pages, which no code read: gone, answered 404
GONE = ("/api/tokenizers", "/api/vocabularies", "/api/tokenizers/tok/vocabularies/1",
        "/api/tokenizers/tok/vocabularies/1/texts")


def served(routes, static: Path):
    for port in PORTS:
        try:
            return make_server(routes, static, port)
        except OSError:
            continue
    raise OSError(f"no free port in {PORTS.start}-{PORTS.stop - 1}")


def ask(base: str, route: str, headers: dict) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(base + route, headers={"User-Agent": "wpt-golden", **headers})
    try:
        with urllib.request.urlopen(request, timeout=60) as answer:
            return answer.status, dict(answer.headers), answer.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def held_in(golden, found) -> bool:
    """Whether golden is held in found: the same value, or for objects each of golden's keys in found with a value
    golden's holds, and for lists as many items, each held so."""
    if isinstance(golden, dict):
        return isinstance(found, dict) and all(key in found and held_in(value, found[key])
                                               for key, value in golden.items())
    if isinstance(golden, list):
        return isinstance(found, list) and len(golden) == len(found) and all(map(held_in, golden, found))
    return golden == found


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(params=["format 12", "brought up"])
def fixture_depot(request, tmp_path, monkeypatch):
    """The format-12 fixture depot and its dock, copied into tmp_path: as it is, or after `depot upgrade`
    brought it to the code's format; the settings the golden answers were recorded with (no public address)."""
    address = tmp_path / "depot.sqlite"
    shutil.copy(FIXTURES / "format-12.sqlite", address)
    shutil.copytree(FIXTURES / "format-12-store", tmp_path / "store")
    if request.param == "brought up":
        Depot.upgrade(str(address))
    monkeypatch.chdir(tmp_path)  # no .env but this folder's
    monkeypatch.setenv("DOCK", (tmp_path / "store").as_uri())
    monkeypatch.delenv("DEPOT_URL", raising=False)
    monkeypatch.delenv("DEPOT", raising=False)
    return address


def test_the_legacy_routes_answer_as_the_web_app_of_6b470328c7_did(fixture_depot, tmp_path):
    before = sha256(fixture_depot)
    jobs = Jobs(str(fixture_depot), STATUSES)
    routes = legacy(str(fixture_depot), Downloads(jobs.path), refusals)
    (tmp_path / "static").mkdir()
    server = served(routes, tmp_path / "static")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    differ, compared = [], 0
    try:
        for listed in json.loads((GOLDEN / "answers.json").read_text(encoding="utf-8")):
            compared += 1
            golden = json.loads((GOLDEN / "answers" / listed["file"]).read_text(encoding="utf-8"))
            status, headers, body = ask(base, golden["route"], golden["asked_headers"])
            if golden["route"] in GONE:
                if status != 404:
                    differ.append(f"{listed['file']} {golden['route']}: status {status}, gone (404)")
                continue
            added_to = golden["route"].startswith(ADDED_TO) and "body_file" not in golden
            kept = {name: headers[name] for name in KEPT if name in headers and not (
                added_to and name == "Content-Length")}
            expected = {name: value for name, value in golden["headers"].items() if not (
                added_to and name == "Content-Length")}
            if "body_file" in golden:
                same_body = body == (GOLDEN / "answers" / golden["body_file"]).read_bytes()
            else:
                found = json.loads(body)
                same_body = held_in(golden["body"], found) if added_to else found == golden["body"]
            if (status, kept, same_body) != (golden["status"], expected, True):
                differ.append(f"{listed['file']} {golden['route']}: status {status} (golden {golden['status']}), "
                              f"headers {kept} (golden {expected}), body {'same' if same_body else 'differs'}")
    finally:
        server.shutdown()
        server.server_close()
    assert compared == 48 and not differ, "\n".join(differ)
    assert sha256(fixture_depot) == before  # nothing wrote the depot's file


def every_file(folder: Path) -> dict[str, str]:
    """{path under folder: its sha256} of every file under folder."""
    return {str(path.relative_to(folder)): sha256(path) for path in sorted(folder.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("fixture_depot", ["brought up"], indirect=True)
def test_an_erasure_the_depot_carries_out_leaves_the_corpora_on_the_dock_and_their_download_as_they_were(
        fixture_depot, tmp_path, monkeypatch):
    # the depot marks nothing on the dock (docs/erasure.md, "What it covers"): a corpus holding the erased text
    # stays fresh and the frozen routes still send it until the freight's next check, whether the erasure is
    # decided by `depot decide admit --erasure` or through POST /api/erasures/decisions
    monkeypatch.setenv("DEPOT", str(fixture_depot))
    store = dock_from_settings(tmp_path / ".env")
    [record, *_] = [one for one in corpora.records(store) if corpora.fresh(one)]
    texts = record["tables"]["texts"]
    held = pq.read_table(store.path(texts[0] if isinstance(texts, list) else texts), columns=["id"])["id"]
    first, second = held.to_pylist()[:2]
    depot = Depot.open(str(fixture_depot))
    declaration = Declaration("Ada Lovelace", "ada@example.org", True)
    asked = [depot.ask_erasure("a passage", [(text, 0, 1)], declaration) for text in (first, second)]
    jobs = Jobs(str(fixture_depot), STATUSES)
    (tmp_path / "static").mkdir()
    server = served([*routes(depot, jobs), *legacy(str(fixture_depot), Downloads(jobs.path), refusals)],
                    tmp_path / "static")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    listing = f"/api/datasets/{record['dataset']['id']}/download"
    try:
        dock_before, listed_before = every_file(tmp_path / "store"), ask(base, listing, {})

        assert depot_app.main(["decide", "admit", "--erasure", asked[0], "--name", "Ada Lovelace", "--email",
                               "ada@example.org", "--accept"]) == 0
        decided = urllib.request.Request(
            base + "/api/erasures/decisions", method="POST", headers={"Content-Type": "application/json"},
            data=json.dumps({"erasures": [asked[1]], "status": "admitted",
                             "declaration": {"name": "Ada Lovelace", "email": "ada@example.org",
                                             "accepted": True}}).encode())
        with urllib.request.urlopen(decided, timeout=60) as answer:
            assert json.loads(answer.read()) == {"carried_out": {asked[1]: True}}

        assert [row["erased"] is not None for row in depot.erasures() if row["id"] in asked] == [True, True]
        assert every_file(tmp_path / "store") == dock_before
        listed = ask(base, listing, {})
        assert listed[0] == 200 and listed[2] == listed_before[2]
    finally:
        server.shutdown()
        server.server_close()
