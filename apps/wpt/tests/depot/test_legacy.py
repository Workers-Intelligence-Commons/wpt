"""The frozen dataset routes (wpt.depot.web.legacy) answer what the web app of commit 6b470328c7 answered, as the
golden answers of legacy-golden/ hold them, on the format-12 fixture depot and its store, and on that depot brought to
the code's format by its upgrade."""

import hashlib
import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from wpt_depot import Depot
from wpt.depot.downloads import Downloads
from wpt.depot.jobs import Jobs
from wpt.depot.web.builds import builds
from wpt.depot.web.legacy import legacy
from wpt.depot.web.routes import STATUSES, refusals
from wpt.depot.web.server import make_server

GOLDEN = Path(__file__).parent / "legacy-golden"
FIXTURES = Path(__file__).parents[4] / "packages" / "wpt-depot" / "tests" / "fixtures"
KEPT = ("Content-Type", "Content-Length", "Content-Disposition", "Content-Range", "Accept-Ranges", "ETag")
PORTS = range(8560, 8600)  # the ports a test server may take on 127.0.0.1
# the routes of tokenizer recipes stay in the app (wpt.depot.web.builds), which may add fields to them: their golden
# answers must be held in what they answer now, each field old code reads with its value
ADDED_TO = "/api/tokenizers", "/api/vocabularies"


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
    """The format-12 fixture depot and its store, copied into tmp_path: as it is, or after `wpt depot upgrade`
    brought it to the code's format; the settings the golden answers were recorded with (no public address)."""
    address = tmp_path / "depot.sqlite"
    shutil.copy(FIXTURES / "format-12.sqlite", address)
    shutil.copytree(FIXTURES / "format-12-store", tmp_path / "store")
    if request.param == "brought up":
        Depot.upgrade(str(address))
    monkeypatch.chdir(tmp_path)  # no .env but this folder's
    monkeypatch.setenv("WPT_STORE", (tmp_path / "store").as_uri())
    monkeypatch.delenv("WPT_DEPOT_URL", raising=False)
    monkeypatch.delenv("WPT_DEPOT", raising=False)
    return address


def test_the_legacy_routes_answer_as_the_web_app_of_6b470328c7_did(fixture_depot, tmp_path):
    before = sha256(fixture_depot)
    jobs = Jobs(str(fixture_depot), STATUSES)
    # the tokenizer routes read through the depot's own reads, opened read-only and of any format
    routes = [*legacy(str(fixture_depot), Downloads(jobs.path), refusals),
              *builds(Depot(str(fixture_depot), readonly=True), jobs, refusals)]
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


def test_a_tokenizer_recipe_of_a_dataset_is_offered_the_snapshot_the_upgrade_made_of_it(fixture_depot, tmp_path):
    jobs = Jobs(str(fixture_depot), STATUSES)
    routes = builds(Depot(str(fixture_depot), readonly=True), jobs, refusals)
    (tmp_path / "static").mkdir()
    server = served(routes, tmp_path / "static")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, _, body = ask(f"http://127.0.0.1:{server.server_address[1]}", "/api/tokenizers", {})
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    offers = {one["name"]: one["snapshot_offer"] for one in json.loads(body)}
    depot = Depot(str(fixture_depot), readonly=True)
    if not depot.rows("SELECT name FROM sqlite_master WHERE name = 'legacy_snapshot_of'"):
        assert offers and set(offers.values()) == {None}  # format 12: no snapshot to offer
    else:  # the upgrade made a snapshot of each recipe's dataset: offered by its id and name
        snapshots = {one["id"]: one["name"] for one in depot.snapshots()}
        assert offers and all(offer is not None and snapshots[offer["id"]] == offer["name"] for offer in offers.values())
