import http.client
import json
import threading

import pytest

from wpt_depot import About, Depot, clone
from wpt.depot import main
from wpt.depot.shared import CONFIRMATION, REQUIREMENTS
from wpt.depot.web.routes import routes
from wpt.depot.web.server import Route, make_server

ME = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True}
CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


@pytest.fixture
def serve(tmp_path):
    """serve(depot): the depot served on a free port, stopped when the test
    ends: request(method, target, body) -> (status, payload)."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>interface</p>")
    servers = []

    def start(depot):
        server = make_server(routes(depot), static)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        def request(method, target, body=None):
            connection = http.client.HTTPConnection("127.0.0.1", port)
            data = None if body is None else json.dumps(body).encode()
            connection.request(method, target, data, {} if body is None else {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
            return response.status, (json.loads(raw) if response.getheader("Content-Type") == "application/json"
                                     else raw)
        return request

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def web(tmp_path, serve):
    """A new depot holding the license CC0-1.0, served on a free port:
    (request(method, target, body) -> (status, payload), the depot)."""
    depot = Depot.new(str(tmp_path / "depot"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    return serve(depot), depot


def test_the_pages_routes_take_texts_through_decisions_libraries_datasets_and_erasures(web):
    request, depot = web
    status, about = request("GET", "/api/about")
    assert status == 200 and about["about"]["name"] == "test-depot" and about["confirmation"] == CONFIRMATION
    assert about["requirements"] == REQUIREMENTS
    assert request("GET", "/api/licenses") == (200, [{"id": "CC0-1.0", "name": "Creative Commons Zero v1.0 Universal",
                                                       "allowed": True, "texts": 0}])
    assert {kind["id"] for kind in request("GET", "/api/kinds")[1]} == {"conversation", "task", "rating", "probe"}

    assert request("POST", "/api/texts", {"text": "The fox.", "licenses": ["CC0-1.0"], "tags": ["en"], "title": "Fox",
                                          "notice": "(c) Ann", "declaration": ME}) == (200, {"id": "test-depot-1"})
    assert request("POST", "/api/texts", {"kind": "conversation", "content": CONVERSATION, "licenses": ["CC0-1.0"],
                                          "tags": ["en"], "declaration": ME}) == (200, {"id": "test-depot-2"})
    both = ["test-depot-1", "test-depot-2"]
    assert request("POST", "/api/decisions", {"status": "admitted", "texts": both, "reason": "free and clear",
                                              "declaration": ME}) == (200, {"decision": "test-depot-3", "texts": both})
    assert request("POST", "/api/libraries", {"name": "first model"}) == (200, {"id": "test-depot-1"})
    assert request("POST", "/api/libraries/test-depot-1/texts", {"texts": both}) == (200, {"joined": both, "left_out": []})
    assert request("POST", "/api/libraries/test-depot-1/forks", {"name": "next"}) == (200, {"id": "test-depot-2"})
    assert request("POST", "/api/libraries/test-depot-1/lock", {}) == (200, {})
    status, held = request("GET", "/api/libraries")
    assert [(one["name"], one["texts"], one["locked_at"] is not None) for one in held] == [
        ("first model", 2, True), ("next", 2, False)]

    status, shown = request("GET", "/api/texts/test-depot-2")
    assert status == 200 and json.loads(shown["text"]["content"]) == CONVERSATION
    assert [(one["status"], one["reason"]) for one in shown["decisions"]] == [("under review", None),
                                                                          ("admitted", "free and clear")]
    status, found = request("POST", "/api/sql", {"query": "SELECT id, title FROM texts ORDER BY id"})
    assert (status, found) == (200, {"columns": ["id", "title"], "rows": [{"id": "test-depot-1", "title": "Fox"},
                                                                         {"id": "test-depot-2", "title": None}]})

    assert "classics" in request("GET", "/api/tokenizers")[1]  # the repository's own recipes
    change = depot.change()
    assert request("POST", "/api/datasets", {"name": "first", "tokenizer": "classics", "libraries": [{"library": "test-depot-1"}],
                                             "shares": [{"tag": "en", "share": 1}]}) == (200, {"id": "test-depot-1"})
    status, remote = request("GET", "/api/datasets/first/remote")
    assert status == 200 and remote["libraries"] == [{"depot": depot.address, "library": "test-depot-1", "change": change}]
    assert request("POST", "/api/remotes", {"remote": {**remote, "name": "again"}}) == (200, {"id": "test-depot-2"})
    assert [one["name"] for one in request("GET", "/api/datasets")[1]] == ["first", "again"]

    assert request("POST", "/api/erasures", {"reason": "a name", "needles": [{"text": "test-depot-1", "start": 4, "end": 7}],
                                             "declaration": ME}) == (200, {"id": "test-depot-1"})
    [asked] = request("GET", "/api/erasures")[1]
    assert asked["needles"] == [{"text": "test-depot-1", "field": None, "start": 4, "end": 7, "version": 1}] and asked["erased"] is None


def test_the_pages_routes_add_a_license_decide_an_erasure_report_a_problem_and_fork_without_texts(web):
    request, depot = web
    assert request("POST", "/api/licenses", {"id": "MIT", "name": "MIT License", "text": "The MIT text."}) == (200, {})
    assert request("POST", "/api/licenses", {"id": "MIT", "name": "MIT License", "text": "again"})[0] == 422

    for text, tag in (("call 555-0100 now", "a"), ("another", "b")):
        request("POST", "/api/texts", {"text": text, "licenses": ["MIT"], "tags": [tag], "declaration": ME})
    assert [(one["id"], one["texts"]) for one in request("GET", "/api/licenses")[1]] == [("CC0-1.0", 0), ("MIT", 2)]
    request("POST", "/api/decisions", {"status": "admitted", "texts": ["test-depot-1", "test-depot-2"], "declaration": ME})
    request("POST", "/api/libraries", {"name": "first"})
    request("POST", "/api/libraries/test-depot-1/texts", {"texts": ["test-depot-1", "test-depot-2"]})
    request("POST", "/api/libraries/test-depot-1/lock", {})
    assert request("POST", "/api/libraries/test-depot-1/problems", {"problem": "a bad text", "texts": ["test-depot-2"],
                                                          "declaration": ME}) == (200, {"decision": "test-depot-4", "texts": ["test-depot-2"]})
    assert request("POST", "/api/libraries/test-depot-1/problems", {"problem": "a bad text", "texts": ["test-depot-9"],
                                                          "declaration": ME})[0] == 404
    assert request("POST", "/api/libraries/test-depot-1/forks", {"name": "cleaner", "without": ["test-depot-2"]}) \
        == (200, {"id": "test-depot-2"})
    held = {one["name"]: (one["texts"], one["problems"]) for one in request("GET", "/api/libraries")[1]}
    assert held == {"first": (2, [{"texts": ["test-depot-2"], "problem": "a bad text", "decision": "test-depot-4"}]), "cleaner": (1, [])}

    request("POST", "/api/erasures", {"reason": "a number", "needles": [{"text": "test-depot-1", "start": 5, "end": 13}],
                                      "declaration": ME})
    request("POST", "/api/erasures", {"reason": "another", "needles": [{"text": "test-depot-2", "start": 0, "end": 3}],
                                      "declaration": ME})
    # every request is checked before any is decided: one that does not exist leaves the other undecided
    assert request("POST", "/api/erasures/decisions", {"erasures": ["test-depot-1", "test-depot-7"], "status": "admitted",
                                                       "declaration": ME})[0] == 404
    assert [one["status"] for one in request("GET", "/api/erasures")[1]] == ["under review", "under review"]
    assert request("POST", "/api/erasures/decisions", {"erasures": ["test-depot-1", "test-depot-2"], "status": "admitted",
                                                       "declaration": ME}) \
        == (200, {"carried_out": {"test-depot-1": True, "test-depot-2": True}})
    assert depot.show("test-depot-1")["text"]["text"] == "call  now"
    assert depot.show("test-depot-2")["text"]["text"] == "ther"
    status, shown = request("GET", "/api/erasures/test-depot-1")
    assert status == 200 and shown["carried_out"] is True
    assert [{key: one[key] for key in ("text", "state", "before", "passage", "after", "erased")} for one in shown["needles"]] \
        == [{"text": "test-depot-1", "state": "erased", "before": "call ", "passage": None, "after": " now", "erased": 8}]
    assert request("GET", "/api/erasures/test-depot-9")[0] == 404
    assert request("POST", "/api/erasures/decisions", {"erasures": ["test-depot-1"], "status": "refused",
                                                       "declaration": ME})[0] == 422  # done


def test_each_refusal_answers_its_status_and_writes_nothing(web):
    # the pages tell a refused change from a missing thing or a malformed request by the status alone
    request, depot = web
    unconfirmed = {"text": "fox", "licenses": ["CC0-1.0"], "declaration": {**ME, "accepted": False}}
    status, refused = request("POST", "/api/texts", unconfirmed)
    assert status == 422 and refused["problems"][0].startswith("Refused: a decision needs the confirmation")
    assert request("POST", "/api/texts", {**unconfirmed, "declaration": ME, "licenses": ["MIT"]})[0] == 422
    assert depot.sql("SELECT count(*) AS n FROM texts")["n"][0].as_py() == 0
    assert request("GET", "/api/texts/test-depot-9")[0] == 404
    assert request("POST", "/api/sql", {"query": "DELETE FROM texts"})[0] == 400
    assert request("POST", "/api/decisions", {"status": "admitted", "texts": ["test-depot-1"]})[0] == 400  # no declaration
    assert request("POST", "/api/decisions", {"status": "admitted", "where": "true", "declaration": ME})[0] == 400  # no ids


def test_the_web_command_serves_the_depot_its_setting_names(tmp_path, monkeypatch):
    address = str(tmp_path / "depot")
    Depot.new(address, About("served", "Ada Lovelace", "ada@example.org"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WPT_DEPOT", address)
    monkeypatch.setattr("wpt.depot.web.ui.built", lambda: tmp_path)
    served = []

    class Stopped:
        server_address = ("127.0.0.1", 0)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def made(found, static, port):
        served.append(({route.pattern for route in found}, port))
        return Stopped()

    monkeypatch.setattr("wpt.depot.web.make_server", made)
    assert main(["web", "--port", "8411"]) == 0
    [(patterns, port)] = served
    assert "/api/texts/{id}" in patterns and port == 8411
    monkeypatch.delenv("WPT_DEPOT")
    assert main(["web"]) == 1  # no depot to serve


def test_a_review_gets_every_id_a_filter_matches_and_the_decision_takes_exactly_those_ids(web):
    # the pages review the texts a filter matches, then send their ids: a text that comes to match the filter
    # between the review and the click is not decided
    request, depot = web
    for text, tags in (("one", ["en", "a"]), ("two", ["en"]), ("three", ["en", "a"]), ("four", ["fr"])):
        request("POST", "/api/texts", {"text": text, "licenses": ["CC0-1.0"], "tags": tags, "declaration": ME})
    request("POST", "/api/decisions", {"status": "refused", "texts": ["test-depot-4"], "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "texts": ["test-depot-1"], "declaration": ME})
    request("POST", "/api/libraries", {"name": "first"})
    request("POST", "/api/libraries/test-depot-1/texts", {"texts": ["test-depot-1"]})
    request("POST", "/api/libraries/test-depot-1/lock", {})
    assert request("GET", "/api/tags") == (200, [{"tag": "a", "texts": 2}, {"tag": "en", "texts": 3},
                                                 {"tag": "fr", "texts": 1}])

    status, every = request("POST", "/api/texts/search", {"filter": {}, "limit": 2})
    assert status == 200 and every["total"] == 4 and every["locked"] == 1 and "ids" not in every
    assert every["statuses"] == {"under review": 2, "admitted": 1, "refused": 1}
    assert every["tags"] == [{"tag": "en", "texts": 3}, {"tag": "a", "texts": 2}, {"tag": "fr", "texts": 1}]
    assert every["licenses"] == [{"license": "CC0-1.0", "texts": 4}]
    assert [(one["id"], one["status"], one["locked"], one["length"]) for one in every["rows"]] == [
        ("test-depot-1", "admitted", True, 3), ("test-depot-2", "under review", False, 3)]
    status, second = request("POST", "/api/texts/search", {"filter": {}, "offset": 2, "limit": 2})
    assert [one["id"] for one in second["rows"]] == ["test-depot-3", "test-depot-4"]
    status, longest = request("POST", "/api/texts/search", {"filter": {}, "order": "length", "descending": True})
    assert [one["id"] for one in longest["rows"]] == ["test-depot-3", "test-depot-4", "test-depot-1", "test-depot-2"]

    status, found = request("POST", "/api/texts/search", {"filter": {"tags": ["en"], "statuses": ["under review", "refused"],
                                                                     "locked": False}, "every_id": True, "limit": 1})
    assert (found["total"], found["ids"]) == (2, ["test-depot-2", "test-depot-3"])
    request("POST", "/api/texts", {"text": "five", "licenses": ["CC0-1.0"], "tags": ["en"], "declaration": ME})
    assert request("POST", "/api/decisions", {"status": "admitted", "texts": found["ids"], "declaration": ME}) \
        == (200, {"decision": "test-depot-8", "texts": ["test-depot-2", "test-depot-3"]})
    assert request("GET", "/api/texts/test-depot-5")[1]["status"] == "under review"
    assert request("POST", "/api/decisions", {"status": "admitted", "texts": ["test-depot-5", "test-depot-9"],
                                              "declaration": ME})[0] == 404  # one is no text: nothing is decided
    assert request("GET", "/api/texts/test-depot-5")[1]["status"] == "under review"
    status, chosen = request("POST", "/api/texts/search", {"filter": {"libraries": ["test-depot-1"]}})
    assert [one["id"] for one in chosen["rows"]] == ["test-depot-1"]
    assert request("POST", "/api/texts/search", {"filter": {"statuses": ["pending"]}})[0] == 400
    assert request("POST", "/api/texts/search", {"filter": {"where": "no_such_column"}})[0] == 400

    status, latest = request("GET", "/api/decisions/latest")
    assert [(one["id"], one["status"], one["texts"]) for one in latest[:2]] == [
        ("test-depot-8", "admitted", 2), ("test-depot-7", "under review", 1)]


def test_the_review_of_an_admission_gets_the_texts_it_cannot_cover_with_their_reasons(web):
    # the page lists these texts apart before an admission: the route answers the depot's own check
    request, depot = web
    request("POST", "/api/licenses", {"id": "CC-BY-NC-4.0", "name": "Non commercial", "text": "text"})
    for text, licenses in (("free", ["CC0-1.0"]), ("held", ["CC0-1.0"]), ("unlicensed", ["CC-BY-NC-4.0"])):
        request("POST", "/api/texts", {"text": text, "licenses": licenses, "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "texts": ["test-depot-2"], "declaration": ME})
    request("POST", "/api/libraries", {"name": "first"})
    request("POST", "/api/libraries/test-depot-1/texts", {"texts": ["test-depot-2"]})
    request("POST", "/api/libraries/test-depot-1/lock", {})

    assert request("POST", "/api/texts/blocked", {"texts": ["test-depot-3", "test-depot-2", "test-depot-1"]}) == (200, {
        "texts": [{"id": "test-depot-3", "reasons": ["no allowed license"]}]})  # a locked library blocks nothing
    assert request("POST", "/api/texts/blocked", {"texts": ["test-depot-1", "test-depot-9"]})[0] == 404
    assert request("POST", "/api/texts/blocked", {"texts": []})[0] == 400



def test_a_license_change_from_the_pages_starts_the_texts_over_and_a_locked_library_keeps_its_revision(web):
    # the review's blocked list and a text's page change licenses through this route, then admit the texts
    request, depot = web
    request("POST", "/api/licenses", {"id": "CC-BY-NC-4.0", "name": "Non commercial", "text": "text"})
    for text in ("one", "two"):
        request("POST", "/api/texts", {"text": text, "licenses": ["CC-BY-NC-4.0"], "declaration": ME})
    request("POST", "/api/texts", {"text": "held", "licenses": ["CC0-1.0"], "declaration": ME})
    request("POST", "/api/decisions", {"status": "refused", "texts": ["test-depot-1"], "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "texts": ["test-depot-3"], "declaration": ME})
    request("POST", "/api/libraries", {"name": "first"})
    request("POST", "/api/libraries/test-depot-1/texts", {"texts": ["test-depot-3"]})
    request("POST", "/api/libraries/test-depot-1/lock", {})

    status, changed = request("POST", "/api/texts/licenses", {"texts": ["test-depot-2", "test-depot-1"],
                                                             "licenses": ["CC0-1.0"], "declaration": ME})

    assert (status, changed) == (200, {"decisions": [{"decision": "test-depot-6",
                                                      "texts": ["test-depot-1", "test-depot-2"]}],
                                        "skipped": []})
    shown = request("GET", "/api/texts/test-depot-1")[1]
    assert shown["status"] == "under review" and shown["text"]["licenses"] == ["CC0-1.0"]
    assert shown["decisions"][-1]["reason"] == "licenses changed from CC-BY-NC-4.0 to CC0-1.0"
    assert request("POST", "/api/texts/blocked", {"texts": ["test-depot-1", "test-depot-2"]}) == (200, {"texts": []})
    locked = {"texts": ["test-depot-3"], "licenses": ["CC-BY-NC-4.0"], "declaration": ME}
    assert request("POST", "/api/texts/licenses", {**locked, "texts": ["test-depot-9"]})[0] == 404
    assert request("POST", "/api/texts/licenses", {**locked, "licenses": []})[0] == 400
    assert request("POST", "/api/texts/licenses", locked)[0] == 200
    shown = request("GET", "/api/texts/test-depot-3")[1]
    assert (shown["text"]["licenses"], shown["revisions"]) == (["CC-BY-NC-4.0"], 3)
    assert shown["pins"] == [{"library": "test-depot-1", "name": "first", "revision": 2}]
    assert request("POST", "/api/libraries/test-depot-1/pins", {}) == (200, {"total": 1, "newer": 1, "rows": [
        {"id": "test-depot-3", "title": None, "revision": 2, "current": 3}]})

def test_a_content_change_from_the_page_of_a_text_takes_a_plain_text_or_an_item_s_content(web):
    request, depot = web
    request("POST", "/api/texts", {"text": "old words", "licenses": ["CC0-1.0"], "declaration": ME})
    request("POST", "/api/texts", {"kind": "conversation", "content": CONVERSATION, "licenses": ["CC0-1.0"],
                                   "declaration": ME})

    assert request("POST", "/api/texts/test-depot-1/content", {"declaration": ME})[0] == 400
    assert request("POST", "/api/texts/test-depot-1/content", {"text": "a", "content": "b", "declaration": ME})[0] == 400
    assert request("POST", "/api/texts/test-depot-1/content", {"content": {"x": 1}, "declaration": ME})[0] == 422
    assert request("POST", "/api/texts/test-depot-2/content", {"text": "words", "declaration": ME})[0] == 422
    assert [request("GET", f"/api/texts/test-depot-{n}")[1]["content_versions"] for n in (1, 2)] == [1, 1]
    assert request("POST", "/api/texts/test-depot-1/content", {"text": "new words", "declaration": ME}) == (
        200, {"decision": "test-depot-3", "version": 2})
    assert request("POST", "/api/texts/test-depot-1/content", {"text": "new words", "declaration": ME}) == (
        200, {"decision": None, "version": None})
    assert request("POST", "/api/texts/test-depot-2/content", {"content": {"messages": "none"},
                                                               "declaration": ME})[0] == 422
    changed = {"messages": [CONVERSATION["messages"][0]]}
    assert request("POST", "/api/texts/test-depot-2/content", {"content": changed, "declaration": ME})[0] == 200
    shown = request("GET", "/api/texts/test-depot-2")[1]
    assert (json.loads(shown["text"]["content"]), shown["content_versions"]) == (changed, 2)


def test_a_failure_no_route_names_is_answered_with_its_problem(tmp_path):
    # without an answer the browser gets a closed connection, and the page could show nothing
    def broken(values, body):
        raise RuntimeError("the disk went away")

    server = make_server([Route("GET", "/api/broken", broken)], tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        connection.request("GET", "/api/broken")
        response = connection.getresponse()
        assert response.status == 500
        assert json.loads(response.read()) == {"problems": ["RuntimeError: the disk went away"]}
    finally:
        server.shutdown()
        server.server_close()


def test_each_thing_a_page_lists_has_a_view_and_the_details_change_through_the_routes(web, tmp_path, monkeypatch):
    request, depot = web
    monkeypatch.chdir(tmp_path)  # no .env: the settings are the test's alone
    monkeypatch.delenv("WPT_STORE", raising=False)
    request("POST", "/api/texts", {"text": "The fox.", "licenses": ["CC0-1.0"], "tags": ["en"], "title": "Fox",
                                   "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "texts": ["test-depot-1"], "reason": "clear",
                                       "declaration": ME})
    request("POST", "/api/libraries", {"name": "first"})
    request("POST", "/api/libraries/test-depot-1/texts", {"texts": ["test-depot-1"]})
    request("POST", "/api/libraries/test-depot-1/problems", {"problem": "too short", "texts": ["test-depot-1"],
                                                            "declaration": ME})
    request("POST", "/api/datasets", {"name": "mix", "tokenizer": "classics", "libraries": [{"library": "test-depot-1"}]})

    status, license = request("GET", "/api/licenses/CC0-1.0")
    assert (status, license["text"], license["allowed"], license["texts"]) == (200, "The CC0 text.", True, 1)
    assert request("GET", "/api/licenses/MIT")[0] == 404
    status, decision = request("GET", "/api/decisions/test-depot-2")
    assert (status, decision["decision"]["reason"], decision["texts"], decision["problem"]) == (200, "clear", 1, None)
    status, problem = request("GET", "/api/decisions/test-depot-3")
    assert problem["problem"] == {"library": "test-depot-1", "texts": ["test-depot-1"], "problem": "too short"}
    assert problem["decision"]["name"] == ME["name"]
    assert request("GET", "/api/decisions/latest")[0] == 200  # still the latest decisions
    status, dataset = request("GET", "/api/datasets/test-depot-1")
    assert status == 200 and dataset["dataset"]["name"] == "mix" and dataset["remote"]["name"] == "mix"
    assert [(one["name"], one["local"]) for one in dataset["libraries"]] == [("first", True)]
    assert dataset["corpora"] is None and dataset["store"]  # no store is set: only the definition
    monkeypatch.setenv("WPT_STORE", (tmp_path / "store").as_uri())
    assert request("GET", "/api/datasets/test-depot-1")[1]["corpora"] == []

    assert request("POST", "/api/texts/test-depot-1/credits", {"title": "The fox", "tags": ["en", "animals"],
                                                              "declaration": ME}) == (
        200, {"decision": "test-depot-4", "reason": "title changed from Fox to The fox; tags changed from en to en, "
                                                    "animals"})
    shown = request("GET", "/api/texts/test-depot-1")[1]
    assert (shown["status"], shown["text"]["title"], shown["text"]["libraries"]) == ("admitted", "The fox",
                                                                                     ["test-depot-1"])
    request("POST", "/api/libraries/test-depot-1/lock", {})
    assert request("POST", "/api/texts/test-depot-1/credits", {"title": "Fox", "declaration": ME})[0] == 200
    assert request("POST", "/api/libraries/test-depot-1/name", {"name": "first model"}) == (200, {})
    request("POST", "/api/libraries", {"name": "second"})
    assert request("POST", "/api/datasets/test-depot-1", {"name": "mix two", "libraries": [
        {"library": "test-depot-1"}, {"library": "test-depot-2"}], "shares": None}) == (200, {"id": "test-depot-1"})
    assert [one["name"] for one in request("GET", "/api/datasets/test-depot-1")[1]["libraries"]] == ["first model",
                                                                                                    "second"]
    assert request("POST", "/api/about", {"maintainer": "Grace Hopper"})[1]["about"]["maintainer"] == "Grace Hopper"
    assert request("POST", "/api/about", {})[0] == 422


def test_a_clone_s_pages_say_what_belongs_to_the_upstream_and_refuse_changing_it(web, serve, tmp_path):
    request, upstream = web
    assert request("POST", "/api/texts", {"text": "Theirs.", "licenses": ["CC0-1.0"], "declaration": ME})[0] == 200
    assert request("POST", "/api/libraries", {"name": "theirs"})[0] == 200
    follower = clone(upstream.address, str(tmp_path / "clone"), About("test-clone", "Grace Hopper", "grace@example.org"))
    mine = serve(follower)
    assert mine("POST", "/api/texts", {"text": "Mine.", "licenses": ["CC0-1.0"], "declaration": ME}) == (
        200, {"id": "test-clone-1"})
    assert mine("POST", "/api/libraries", {"name": "mine"}) == (200, {"id": "test-clone-1"})
    change = follower.change()

    status, refused = mine("POST", "/api/decisions", {"status": "refused", "texts": ["test-depot-1", "test-clone-1"],
                                                       "declaration": ME})
    assert status == 422 and "test-depot-1" in refused["problems"][0] and "to the upstream" in refused["problems"][0]
    assert mine("POST", "/api/libraries/test-depot-1/name", {"name": "renamed"})[0] == 422
    assert follower.change() == change

    assert mine("GET", "/api/texts/test-depot-1")[1]["upstream"] is True
    assert mine("GET", "/api/texts/test-clone-1")[1]["upstream"] is False
    assert [(one["id"], one["upstream"]) for one in mine("GET", "/api/libraries")[1]] == [
        ("test-clone-1", False), ("test-depot-1", True)]
    assert mine("POST", "/api/texts/blocked", {"texts": ["test-depot-1", "test-clone-1"], "status": "refused"}) == (
        200, {"texts": [{"id": "test-depot-1", "reasons": ["belongs to the upstream"]}]})
    assert mine("POST", "/api/decisions", {"status": "refused", "texts": ["test-clone-1"], "declaration": ME})[0] == 200
    assert request("GET", "/api/texts/test-depot-1")[1]["upstream"] is False  # in the upstream, it is its own


def test_a_clone_s_upstream_library_lists_the_datasets_naming_it_by_either_depot_s_address(web, serve, tmp_path):
    request, upstream = web
    assert request("POST", "/api/libraries", {"name": "theirs"}) == (200, {"id": "test-depot-1"})
    assert request("POST", "/api/datasets", {"name": "upstream-set", "tokenizer": "t",
                                             "libraries": [{"library": "test-depot-1"}]})[0] == 200
    follower = clone(upstream.address, str(tmp_path / "clone"), About("test-clone", "Grace Hopper", "grace@example.org"))
    mine = serve(follower)
    assert mine("POST", "/api/libraries", {"name": "mine"}) == (200, {"id": "test-clone-1"})
    for name, depot in (("by-clone", None), ("by-upstream-url", (tmp_path / "depot").as_uri())):
        assert mine("POST", "/api/datasets", {"name": name, "tokenizer": "t",
                                              "libraries": [{"depot": depot, "library": "test-depot-1"}]})[0] == 200

    status, found = mine("GET", "/api/datasets")

    assert status == 200
    assert {one["name"]: one["here"] for one in found} == {
        "upstream-set": ["test-depot-1"], "by-clone": ["test-depot-1"], "by-upstream-url": ["test-depot-1"]}
    assert request("GET", "/api/datasets")[1][0]["here"] == ["test-depot-1"]  # in the upstream, by its own address
    # the dataset's own page gives its row as the list does
    assert [mine("GET", f"/api/datasets/{one['id']}")[1]["dataset"] for one in found] == found


def test_a_clone_s_page_of_an_upstream_dataset_marks_the_upstream_library_as_read_here(web, serve, tmp_path):
    request, upstream = web
    assert request("POST", "/api/libraries", {"name": "theirs"}) == (200, {"id": "test-depot-1"})
    assert request("POST", "/api/datasets", {"name": "upstream-set", "tokenizer": "t",
                                             "libraries": [{"library": "test-depot-1"}]}) == (200, {"id": "test-depot-1"})
    follower = clone(upstream.address, str(tmp_path / "clone"), About("test-clone", "Grace Hopper", "grace@example.org"))
    mine = serve(follower)

    status, shown = mine("GET", "/api/datasets/test-depot-1")

    assert status == 200
    assert [(one["library"], one["local"], one["name"], one["locked_at"]) for one in shown["libraries"]] == [
        ("test-depot-1", True, "theirs", None)]


def test_an_upstream_library_the_clone_does_not_hold_yet_is_read_here_only_once_a_pull_brings_it(
        web, serve, tmp_path):
    request, upstream = web
    follower = clone(upstream.address, str(tmp_path / "clone"), About("test-clone", "Grace Hopper", "grace@example.org"))
    mine = serve(follower)
    assert request("POST", "/api/libraries", {"name": "theirs"}) == (200, {"id": "test-depot-1"})  # after the clone
    assert mine("POST", "/api/datasets", {"name": "set", "tokenizer": "t", "libraries": [
        {"depot": (tmp_path / "depot").as_uri(), "library": "test-depot-1"}]}) == (200, {"id": "test-clone-1"})

    def shown():
        listed = mine("GET", "/api/datasets")[1][0]["here"]
        page = mine("GET", "/api/datasets/test-clone-1")[1]
        return listed, page["dataset"]["here"], [one["local"] for one in page["libraries"]]

    assert shown() == ([], [], [False])
    assert mine("POST", "/api/pull", {})[0] == 200
    assert shown() == (["test-depot-1"], ["test-depot-1"], [True])
