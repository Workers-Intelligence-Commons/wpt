import contextlib
import hashlib
import http.client
import io
import json
import sys
import threading
import time

import pytest

from wpt_depot import About, Declaration, Depot, NewText, clone
from wpt_depot.schema import FORMAT
from wpt_store import store_from_settings
from wpt_utils.lines import bare
from wpt.depot import main
from wpt.depot.shared import CONFIRMATION, REQUIREMENTS
from wpt.depot.web import readers, translations
from wpt.depot.jobs import Jobs, work_folder
from wpt.depot.web.routes import STATUSES, routes
from wpt.depot.web.server import Route, make_server
from wpt.depot.web.translations import ATTACHMENTS, RECORDS, pairs_of

ME = {"name": "Ada Lovelace", "email": "ada@example.org", "accepted": True}


def named(ids) -> dict:
    """The selection of exactly the texts ids, as a text's page sends it."""
    return {"filter": {"ids": list(ids)}}


def wrote(request, target, body):
    """(status, answer) of a write over a selection, its job followed to its end: the operation's answer, or the
    refusal's status and problems."""
    status, state = request("POST", target, body)
    while status == 202 or (status == 200 and state["state"] in ("queued", "running")):
        time.sleep(0.05)
        status, state = request("GET", f"/api/jobs/{state['id']}")
    if status != 200:
        return status, state
    if state["state"] != "done":
        return state["status"], {"problems": [state["problem"]]}
    return 200, state["answer"]
CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


@pytest.fixture
def serve(tmp_path):
    """serve(depot): the depot served on a free port, stopped when the test
    ends: request(method, target, body) -> (status, payload), whose url is
    the server's web address."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>interface</p>")
    servers, runners = [], []

    def start(depot):
        jobs = Jobs(depot.address, STATUSES)
        jobs.start()
        runners.append(jobs)
        server = make_server(routes(depot, jobs), static)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        def request(method, target, body=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)  # far above any answer's time
            data = None if body is None else json.dumps(body).encode()
            connection.request(method, target, data, {} if body is None else {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
            return response.status, (json.loads(raw) if response.getheader("Content-Type") == "application/json"
                                     else raw)
        request.url = f"http://127.0.0.1:{port}"  # the address a clone follows
        return request

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()
    for jobs in runners:
        jobs.stop()


@pytest.fixture
def web(tmp_path, serve):
    """A new depot holding the license CC0-1.0, served on a free port:
    (request(method, target, body) -> (status, payload), the depot)."""
    depot = Depot.new(str(tmp_path / "depot.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    return serve(depot), depot


def test_the_pages_routes_take_texts_through_decisions_tokenizer_recipes_and_erasures(web):
    request, depot = web
    status, about = request("GET", "/api/about")
    assert status == 200 and about["about"]["name"] == "test-depot" and about["confirmation"] == CONFIRMATION
    assert about["requirements"] == REQUIREMENTS
    assert request("GET", "/api/licenses") == (200, [{"id": "CC0-1.0", "name": "Creative Commons Zero v1.0 Universal",
                                                       "allowed": True, "texts": 0}])
    assert {kind["id"] for kind in request("GET", "/api/kinds")[1]} == {"plain", "conversation", "task", "rating", "probe"}

    assert request("POST", "/api/texts", {"text": "The fox.", "licenses": ["CC0-1.0"], "tags": ["en"], "title": "Fox",
                                          "notice": "(c) Ann", "declaration": ME}) == (200, {"id": "test-depot-1"})
    assert request("POST", "/api/texts", {"kind": "conversation", "content": CONVERSATION, "licenses": ["CC0-1.0"],
                                          "tags": ["en"], "declaration": ME}) == (200, {"id": "test-depot-2"})
    both = ["test-depot-1", "test-depot-2"]
    assert wrote(request, "/api/decisions", {"status": "admitted", "selection": named(both), "reason": "free and clear",
                                             "declaration": ME}) == (200, {"decision": "test-depot-3", "texts": 2, "left": {}})
    assert request("GET", "/api/libraries")[0] == 404  # libraries are gone
    assert request("POST", "/api/libraries", {"name": "first model"})[0] == 404

    status, shown = request("GET", "/api/texts/test-depot-2")
    assert status == 200 and json.loads(shown["text"]["content"]) == CONVERSATION and not shown["cut"]
    assert request("GET", "/api/texts/test-depot-1/passage?from=4&characters=3") == (
        200, {"id": "test-depot-1", "version": 1, "length": 8, "from": 4, "text": "fox"})
    assert request("GET", "/api/texts/test-depot-1/passage?from=x")[0] == 422
    assert request("GET", "/api/texts/test-depot-2/passage")[0] == 422  # an item is read whole
    assert request("GET", "/api/texts/test-depot-9/passage")[0] == 404
    assert [(one["status"], one["reason"]) for one in shown["decisions"]] == [("under review", None),
                                                                          ("admitted", "free and clear")]
    status, found = request("POST", "/api/sql", {"query": "SELECT id, title FROM texts ORDER BY id"})
    assert (status, found) == (200, {"columns": ["id", "title"], "rows": [{"id": "test-depot-1", "title": "Fox"},
                                                                         {"id": "test-depot-2", "title": None}]})

    assert request("GET", "/api/tokenizers") == (200, [])  # the depot's own recipes: none yet
    snapshot = depot.create_snapshot("s1")["id"]
    recipe = {"name": "tiny", "snapshot": "s1", "sample_bytes": 1000, "vocab_size": 560}
    assert request("POST", "/api/tokenizers", {**recipe, "kinds": ["conversation"]})[0] == 422  # plain only, for now
    assert request("POST", "/api/tokenizers", {**recipe, "snapshot": "s9"})[0] == 404
    assert request("POST", "/api/tokenizers", recipe)[0] == 200
    [shown] = request("GET", "/api/tokenizers")[1]
    assert (shown["sample"], shown["snapshot_name"], shown["needs_snapshot"], shown["dataset_name"]) == (
        {"snapshot": snapshot, "kinds": ["plain"]}, "s1", False, None)
    assert request("PUT", "/api/tokenizers/tiny", {"snapshot": snapshot, "sample_bytes": 2000,
                                                   "vocab_size": 560}) == (200, {"version": 2})
    assert request("POST", "/api/datasets", {"name": "first", "tokenizer": "tiny", "libraries": [
        {"library": "test-depot-1"}]})[0] == 404  # datasets are gone

    assert request("POST", "/api/erasures", {"reason": "a name", "needles": [{"text": "test-depot-1", "start": 4, "end": 7}],
                                             "declaration": ME}) == (200, {"id": "test-depot-1"})
    [asked] = request("GET", "/api/erasures")[1]
    assert asked["needles"] == [{"text": "test-depot-1", "field": None, "start": 4, "end": 7, "version": 1}] and asked["erased"] is None


def test_the_pages_routes_add_a_license_and_decide_erasures(web):
    request, depot = web
    assert request("POST", "/api/licenses", {"id": "MIT", "name": "MIT License", "text": "The MIT text."}) == (200, {})
    assert request("POST", "/api/licenses", {"id": "MIT", "name": "MIT License", "text": "again"})[0] == 422

    for text, tag in (("call 555-0100 now", "a"), ("another", "b")):
        request("POST", "/api/texts", {"text": text, "licenses": ["MIT"], "tags": [tag], "declaration": ME})
    assert [(one["id"], one["texts"]) for one in request("GET", "/api/licenses")[1]] == [("CC0-1.0", 0), ("MIT", 2)]
    request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-1", "test-depot-2"]), "declaration": ME})
    depot.create_snapshot("first")

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
    assert depot.sql("SELECT count(*) AS n FROM texts") == [{"n": 0}]
    assert request("GET", "/api/texts/test-depot-9")[0] == 404
    assert request("POST", "/api/sql", {"query": "DELETE FROM texts"})[0] == 400
    assert request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-1"])})[0] == 400  # no declaration
    assert request("POST", "/api/decisions", {"status": "admitted", "where": "true", "declaration": ME})[0] == 400  # no ids


def test_a_pull_and_a_change_of_the_depot_s_facts_are_no_routes(web):
    # the server is open to anyone: these reach past the depot (another address, the store) or change the depot's
    # own facts, so only the commands do them
    request, depot = web
    change, about = depot.change(), depot.about()

    for target, body in (("/api/pull", {}), ("/api/jobs/x/cancel", {}), ("/api/about", {"maintainer": "Grace Hopper"})):
        assert request("POST", target, body) == (404, {"problems": [f"no route POST {target}"]})

    assert (depot.change(), depot.about()) == (change, about)
    paths = request("GET", "/api/openapi.json")[1]["paths"]
    assert "/api/pull" not in paths and "/api/remotes" not in paths and "/api/datasets/{name}/corpus" not in paths
    assert list(paths["/api/about"]) == ["get"]
    assert request("GET", "/api/about")[0] == 200


FOREVER = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT count(*) FROM c"  # never ends


def counted(most):
    """A query of the numbers 1 to most, a column n."""
    return f"WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < {most}) SELECT n FROM c"


@pytest.mark.parametrize("read", ["load_extension('{outside}/secret')", "readfile('{outside}/secret.txt')"])
def test_a_query_or_a_filter_s_condition_reads_nothing_outside_the_depot(web, tmp_path, read):
    request, depot = web
    request("POST", "/api/texts", {"text": "fox", "licenses": ["CC0-1.0"], "declaration": ME})
    secret = "the server's secret"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(secret, encoding="utf-8")
    read = read.format(outside=outside)

    for target, body in (("/api/sql", {"query": f"SELECT {read} AS value"}),
                         ("/api/texts/search", {"filter": {"where": f"{read} IS NOT NULL"}}),
                         ("/api/texts/passages", {"filter": {"where": f"EXISTS (SELECT {read})"}}),
                         ("/api/texts/counts", {"filter": {"where": f"{read} IS NOT NULL"}})):
        status, answer = request("POST", target, body)
        assert status == 400 and answer["problems"][0].startswith("BadQuery") and secret not in json.dumps(answer)


def test_a_query_can_neither_write_attach_load_nor_change_a_setting(web, tmp_path):
    request, depot = web
    change = depot.change()
    other = tmp_path / "other.sqlite"
    for statement in (f"ATTACH '{other}' AS other", f"VACUUM INTO '{tmp_path / 'copy.sqlite'}'",
                      "DELETE FROM texts", "INSERT INTO changes VALUES (99, 'x', 'y')", "CREATE TABLE mine (a)",
                      "DROP VIEW texts", "PRAGMA writable_schema = 1", "PRAGMA journal_mode = DELETE",
                      "SELECT 1; DELETE FROM texts", f"SELECT load_extension('{tmp_path / 'x'}')"):
        status, answer = request("POST", "/api/sql", {"query": statement})
        assert status == 400 and answer["problems"][0].startswith("BadQuery"), statement
    assert not other.exists() and not (tmp_path / "copy.sqlite").exists() and depot.change() == change


def test_a_query_or_a_condition_past_the_time_limit_is_stopped_and_the_server_answers_the_next(web, monkeypatch):
    # SQLite stops the statement at the reader's deadline; past the server's own, it kills the reader process
    request, depot = web
    request("POST", "/api/texts", {"text": "fox", "licenses": ["CC0-1.0"], "declaration": ME})
    monkeypatch.setattr(readers, "SECONDS", 3)

    for target, body in (("/api/sql", {"query": FOREVER}),
                         ("/api/texts/search", {"filter": {"where": f"({FOREVER}) > 0"}}),
                         ("/api/texts/passages", {"filter": {"where": f"({FOREVER}) > 0"}}),
                         ("/api/texts/counts", {"filter": {"where": f"({FOREVER}) > 0"}})):
        started = time.monotonic()
        status, answer = request("POST", target, body)
        assert time.monotonic() - started < 3 + 2  # the limit, and the time to kill the process
        assert status == 422 and answer["problems"][0].startswith("TimeLimitReached: "), answer
    assert request("POST", "/api/sql", {"query": "SELECT count(*) AS n FROM texts"}) == (
        200, {"columns": ["n"], "rows": [{"n": 1}]})


@pytest.mark.skipif(sys.platform != "linux", reason="the memory of a reader process is capped on Linux")
def test_a_query_past_the_memory_cap_is_stopped_and_the_server_answers_the_next(web, monkeypatch):
    request, depot = web
    monkeypatch.setattr(readers, "MEMORY", 64 << 20)
    # one string of about 300 MB, far past the cap
    query = f"SELECT length(group_concat(printf('%0100d', n), '')) AS n FROM ({counted(3_000_000)})"
    status, answer = request("POST", "/api/sql", {"query": query})
    assert status in (400, 422) and "emory" in answer["problems"][0], answer  # SQLite's own refusal, or the process lost
    assert request("POST", "/api/sql", {"query": "SELECT 1 AS n"}) == (200, {"columns": ["n"], "rows": [{"n": 1}]})


def test_a_query_s_answer_past_its_bounds_is_refused(web, monkeypatch):
    request, depot = web
    monkeypatch.setattr(readers, "ANSWER_ROWS", 1000)
    monkeypatch.setattr(readers, "ANSWER_BYTES", 50_000)
    assert request("POST", "/api/sql", {"query": counted(1000)})[0] == 200
    status, answer = request("POST", "/api/sql", {"query": counted(1001)})
    assert (status, answer["problems"][0].split(":")[0]) == (422, "AnswerTooLarge")
    status, answer = request("POST", "/api/sql", {"query": "SELECT hex(zeroblob(50000)) AS n"})
    assert (status, answer["problems"][0].split(":")[0]) == (422, "AnswerTooLarge")


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


def test_a_review_takes_the_texts_a_filter_matched_at_its_change_and_never_one_changed_since(web):
    # the pages review the texts a filter matches, then send the filter with the change it was read at: a text
    # that comes to match the filter between the review and the click is not decided
    request, depot = web
    for text, tags in (("one", ["en", "a"]), ("two", ["en"]), ("three", ["en", "a"]), ("four", ["fr"])):
        request("POST", "/api/texts", {"text": text, "licenses": ["CC0-1.0"], "tags": tags, "declaration": ME})
    request("POST", "/api/decisions", {"status": "refused", "selection": named(["test-depot-4"]), "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-1"]), "declaration": ME})
    assert request("GET", "/api/tags") == (200, [{"tag": "a", "texts": 2}, {"tag": "en", "texts": 3},
                                                 {"tag": "fr", "texts": 1}])

    status, every = request("POST", "/api/texts/search", {"filter": {}, "limit": 2})
    assert status == 200 and "ids" not in every and "total" not in every and every["previous"] is None
    assert [(one["id"], one["status"], one["length"]) for one in every["rows"]] == [
        ("test-depot-1", "admitted", 3), ("test-depot-2", "under review", 3)]
    assert "locked" not in every["rows"][0] and "libraries" not in every["rows"][0]
    status, second = request("POST", "/api/texts/search", {"filter": {}, "after": every["next"], "limit": 2})
    assert [one["id"] for one in second["rows"]] == ["test-depot-3", "test-depot-4"] and second["next"] is None
    status, back = request("POST", "/api/texts/search", {"filter": {}, "before": second["previous"], "limit": 2})
    assert back["rows"] == every["rows"] and back["previous"] is None and back["next"] == every["next"]
    status, longest = request("POST", "/api/texts/search", {"filter": {}, "order": "length", "descending": True})
    assert [one["id"] for one in longest["rows"]] == ["test-depot-3", "test-depot-4", "test-depot-2", "test-depot-1"]
    status, shortest = request("POST", "/api/texts/search", {"filter": {}, "order": "length", "limit": 3})
    status, rest = request("POST", "/api/texts/search", {"filter": {}, "order": "length", "after": shortest["next"]})
    assert [one["id"] for one in shortest["rows"] + rest["rows"]] == [
        "test-depot-1", "test-depot-2", "test-depot-4", "test-depot-3"]
    assert request("POST", "/api/texts/search", {"filter": {}, "after": ["x", 1]})[0] == 422  # not a place of the order
    status, opened = request("POST", "/api/texts/search", {"filter": {}, "at": "test-depot-3", "limit": 1})
    assert [one["id"] for one in opened["rows"]] == ["test-depot-3"] and opened["previous"] is not None
    assert request("POST", "/api/texts/search", {"filter": {}, "at": "test-depot-3", "after": every["next"]})[0] == 400

    # the counts come in steps, each one slice of the texts, which a page adds up
    status, whole = request("POST", "/api/texts/counts", {"filter": {}})
    assert status == 200 and (whole["texts"], whole["total"], whole["next"]) == (4, 4, None)
    # every text is read at once from the depot's counts, and gives what a step over every text counts
    assert whole["held"] is True
    status, stepped = request("POST", "/api/texts/counts", {"filter": {}, "held": False})
    assert stepped["held"] is False and {**stepped, "held": True} == whole
    assert whole["statuses"] == {"under review": 2, "admitted": 1, "refused": 1} and "locked" not in whole
    assert whole["tags"] == [{"tag": "en", "texts": 3}, {"tag": "a", "texts": 2}, {"tag": "fr", "texts": 1}]
    assert whole["licenses"] == [{"license": "CC0-1.0", "texts": 4}]
    # the depot's counts read at once give each kind's texts, every status together
    status, depot_counts = request("GET", "/api/counts")
    assert status == 200 and depot_counts["texts"] == 4 and depot_counts["kinds"] == {"plain": 4}
    steps, start = [], None
    while True:
        status, step = request("POST", "/api/texts/counts", {"filter": {"tags": ["en"]}, "size": 3, "held": False,
                                                             **({"start": start} if start else {})})
        steps.append(step)
        start = step["next"]
        if start is None:
            break
    # the steps go over the listing of the tag, its 3 texts; the first step is full, so one more step reads none
    assert [(one["texts"], one["total"]) for one in steps] == [(3, 3), (0, None)]
    assert sum(one["statuses"]["under review"] + one["statuses"]["admitted"] for one in steps) == 3
    # one tag is read at once from the depot's counts too, its statuses as the steps count them
    status, tagged = request("POST", "/api/texts/counts", {"filter": {"tags": ["en"]}})
    assert tagged["held"] is True and tagged["statuses"] == {
        status: sum(one["statuses"][status] for one in steps) for status in tagged["statuses"]}
    # the counts alone answer on a route of their own, the tags and licenses of a filter of statuses included, and
    # say so when they cannot answer a filter
    assert request("POST", "/api/texts/held", {"filter": {"tags": ["en"]}}) == (200, tagged)
    status, reviewed = request("POST", "/api/texts/held", {"filter": {"statuses": ["under review"]}})
    status, walked = request("POST", "/api/texts/counts", {"filter": {"statuses": ["under review"]}, "held": False})
    assert reviewed["held"] is True and (reviewed["tags"], reviewed["licenses"]) == (walked["tags"], walked["licenses"])
    assert request("POST", "/api/texts/held", {"filter": {"tags": ["en", "fr"]}}) == (200, {"held": False})
    status, given = request("POST", "/api/texts/counts", {"filter": {"ids": ["test-depot-2", "test-depot-9"]}})
    assert (given["texts"], given["total"], given["next"], given["statuses"]["under review"]) == (1, 1, None, 1)

    # a selection is the filter, the change its page was read at and the ids unticked: a text that came to match
    # the filter after that change is left out, and counted; a filter that names no ids runs as a job
    shown = {"tags": ["en"], "statuses": ["under review", "refused"]}
    status, found = request("POST", "/api/texts/search", {"filter": shown, "limit": 1})
    assert [one["id"] for one in found["rows"]] == ["test-depot-2"] and found["change"] == depot.change()
    request("POST", "/api/texts", {"text": "five", "licenses": ["CC0-1.0"], "tags": ["en"], "declaration": ME})
    request("POST", "/api/texts", {"text": "six", "licenses": ["CC0-1.0"], "tags": ["en"], "declaration": ME})
    selection = {"filter": shown, "change": found["change"], "unticked": ["test-depot-6"]}
    status, counted = request("POST", "/api/texts/counts", {"filter": shown, "change": found["change"],
                                                            "unticked": ["test-depot-6"]})
    assert counted["statuses"]["under review"] == 2
    status, started = request("POST", "/api/decisions", {"status": "admitted", "selection": selection,
                                                         "declaration": ME})
    assert status in (200, 202) and (started["kind"], started["title"]) == (  # the Jobs page names the filter
        "write", "admit texts under review or refused, tagged en, less 1 unticked")
    job = started
    while job["state"] in ("queued", "running"):
        time.sleep(0.05)
        job = request("GET", f"/api/jobs/{started['id']}")[1]
    assert (job["state"], job["answer"]) == ("done", {"decision": "test-depot-9", "texts": 2,
                                                       "left": {"changed since": 1}})
    assert "selecting the texts" in [one["stage"] for one in job["stages"]]
    assert wrote(request, "/api/decisions", {"status": "admitted", "selection": selection, "declaration": ME})[0] \
        == 404  # what still matches came after the page was read
    assert request("GET", "/api/jobs/nothing")[0] == 404
    assert request("GET", "/api/texts/test-depot-5")[1]["status"] == "under review"
    assert request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-5", "test-depot-9"]),
                                              "declaration": ME})[0] == 404  # one is no text: nothing is decided
    assert request("GET", "/api/texts/test-depot-5")[1]["status"] == "under review"
    assert request("POST", "/api/texts/search", {"filter": {"statuses": ["pending"]}})[0] == 400
    assert request("POST", "/api/texts/search", {"filter": {"where": "no_such_column"}})[0] == 400

    status, latest = request("GET", "/api/decisions/latest")
    assert [(one["id"], one["status"], one["texts"]) for one in latest[:2]] == [
        ("test-depot-9", "admitted", 2), ("test-depot-8", "under review", 1)]


def test_passages_give_the_texts_a_filter_matches_with_their_text_a_page_bounded_by_count_and_characters(web):
    # the blob of the pages reads a collection page by page this way, never the whole collection at once
    request, depot = web
    for text, tags in (("a" * 10, ["x"]), ("b" * 4, ["x"]), ("c" * 4, ["y"]), ("d" * 3, ["x"]), ("e" * 2, ["x"])):
        request("POST", "/api/texts", {"text": text, "licenses": ["CC0-1.0"], "tags": tags, "title": text[0],
                                       "declaration": ME})
    tagged = {"tags": ["x"]}

    status, first = request("POST", "/api/texts/passages", {"filter": tagged, "characters": 6})
    assert status == 200 and first["rows"] == [{"id": "test-depot-1", "kind": "plain", "title": "a", "length": 10,
                                                "passage": "aaaaaa"}] and first["next"] is not None
    status, second = request("POST", "/api/texts/passages", {"filter": tagged, "after": first["next"], "characters": 6})
    assert [(one["id"], one["passage"]) for one in second["rows"]] == [("test-depot-2", "bbbb")]
    status, third = request("POST", "/api/texts/passages", {"filter": tagged, "after": second["next"], "limit": 1})
    assert [one["id"] for one in third["rows"]] == ["test-depot-4"]
    status, last = request("POST", "/api/texts/passages", {"filter": tagged, "after": third["next"]})
    assert [(one["id"], one["passage"]) for one in last["rows"]] == [("test-depot-5", "ee")] and last["next"] is None
    status, longest = request("POST", "/api/texts/passages", {"filter": {}, "order": "length", "descending": True,
                                                              "characters": 100})
    assert [one["id"] for one in longest["rows"]] == ["test-depot-1", "test-depot-3", "test-depot-2", "test-depot-4",
                                                      "test-depot-5"] and longest["next"] is None
    status, starts = request("POST", "/api/texts/passages", {"filter": {}, "order": "length", "descending": True,
                                                             "characters": 7, "each": 3})
    assert [one["passage"] for one in starts["rows"]] == ["aaa", "ccc"] and starts["next"] is not None
    assert request("POST", "/api/texts/passages", {"filter": tagged, "characters": 0})[0] == 400


def test_a_snapshot_read_whole_queues_its_order_by_length_once_then_pages_its_texts_the_longest_first(web):
    # a snapshot of the public depot holds 22 million texts: its strip never sorts them at a request, it reads the order
    # a job of the read lane made once, and the first request says at once that the job was taken
    request, depot = web
    lengths = [4, 9, 1, 6]
    texts = depot.add([NewText(text=c * n, licenses=("CC0-1.0",), tags=("t",) if n > 3 else ())
                       for c, n in zip("abcd", lengths)], Declaration(**ME))
    depot.decide("admitted", texts, Declaration(**ME))
    made = depot.create_snapshot("first")
    depot.decide("refused", [texts[1]], Declaration(**ME))

    status, first = request("POST", f"/api/snapshots/{made['id']}/strip", {})
    assert status == 200 and first["ready"] is False and first["texts"] == 4
    assert first["job"]["kind"] == "sizes" and first["job"]["lane"] == "read"
    status, again = request("POST", f"/api/snapshots/{made['id']}/strip", {})  # the same job, never a second one
    assert again["ready"] or again["job"]["id"] == first["job"]["id"]
    deadline = time.monotonic() + 120
    while not (found := request("POST", f"/api/snapshots/{made['id']}/strip", {"limit": 2, "characters": 100})[1])[
            "ready"]:
        assert found["job"]["state"] in ("queued", "running") and time.monotonic() < deadline
        time.sleep(0.05)
    assert [(one["id"], one["passage"], one["status"]) for one in found["rows"]] == [
        (texts[1], "b" * 9, "refused"), (texts[3], "d" * 6, "admitted")] and found["total"] == 4
    status, rest = request("POST", f"/api/snapshots/{made['id']}/strip", {"after": found["next"]})
    assert [one["id"] for one in rest["rows"]] == [texts[0], texts[2]] and rest["next"] is None
    status, starts = request("POST", f"/api/snapshots/{made['id']}/strip", {"each": 2, "limit": 2})
    assert [one["passage"] for one in starts["rows"]] == ["bb", "dd"]
    status, bare_rows = request("POST", f"/api/snapshots/{made['id']}/strip", {"descending": False, "passages": False})
    assert [one["id"] for one in bare_rows["rows"]] == [texts[2], texts[0], texts[3], texts[1]]
    assert "passage" not in bare_rows["rows"][0]
    assert request("GET", "/api/jobs/" + first["job"]["id"])[1]["answer"] == {"texts": 4}

    status, held = request("GET", "/api/counts/admitted")
    assert status == 200 and held["tags"] == {"t": [2, 10, 10]} and held["kinds"] == {"plain": [3, 11, 11]}
    assert request("GET", "/api/counts/nothing")[0] == 404


def test_a_snapshot_order_that_failed_is_answered_until_asked_again_never_queued_at_every_reading(web):
    # the page reads the order's state every half second: a job that failed must reach it, not a new job each time
    request, depot = web
    texts = depot.add([NewText(text=c * n, licenses=("CC0-1.0",)) for c, n in zip("ab", [3, 5])], Declaration(**ME))
    depot.decide("admitted", texts, Declaration(**ME))
    made = depot.create_snapshot("first")
    blocker = work_folder(depot.address) / "sizes"  # a file where the order's folder goes: keeping it fails
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("in the way")
    target = f"/api/snapshots/{made['id']}/strip"

    status, first = request("POST", target, {})
    deadline = time.monotonic() + 120
    while (found := request("POST", target, {})[1])["job"]["state"] in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert found["ready"] is False and found["job"]["id"] == first["job"]["id"]
    assert found["job"]["state"] in ("failed", "refused") and found["job"]["problem"]
    assert request("POST", target, {})[1]["job"]["id"] == first["job"]["id"]  # still that one, nothing queued
    assert [one["id"] for one in request("GET", "/api/jobs")[1] if one["kind"] == "sizes"] == [first["job"]["id"]]

    blocker.unlink()
    status, again = request("POST", target, {"again": True})
    assert again["ready"] or again["job"]["id"] != first["job"]["id"]
    while not (found := request("POST", target, {})[1])["ready"]:
        assert found["job"]["state"] in ("queued", "running") and time.monotonic() < deadline
        time.sleep(0.05)
    assert [one["id"] for one in found["rows"]] == [texts[1], texts[0]]


def test_the_review_of_an_admission_gets_the_texts_it_cannot_cover_with_their_reasons(web):
    # the page lists these texts apart before an admission: the route answers the depot's own check
    request, depot = web
    request("POST", "/api/licenses", {"id": "CC-BY-NC-4.0", "name": "Non commercial", "text": "text"})
    for text, licenses in (("free", ["CC0-1.0"]), ("held", ["CC0-1.0"]), ("unlicensed", ["CC-BY-NC-4.0"])):
        request("POST", "/api/texts", {"text": text, "licenses": licenses, "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-2"]), "declaration": ME})

    assert request("POST", "/api/texts/blocked", {"selection": named(["test-depot-3", "test-depot-2", "test-depot-1"])}) == (200, {
        "rows": [{"id": "test-depot-3", "reasons": ["no allowed license"]}], "next": None, "partial": False})
    assert request("POST", "/api/texts/blocked", {"selection": {"filter": {}}, "limit": 1})[1]["rows"] == [
        {"id": "test-depot-3", "reasons": ["no allowed license"]}]
    assert request("POST", "/api/texts/blocked", {"selection": named([])}) == (200, {"rows": [], "next": None, "partial": False})
    assert request("POST", "/api/texts/blocked", {"selection": {}})[0] == 400



def test_a_license_change_from_the_pages_starts_the_texts_over_and_a_snapshot_keeps_its_revision(web):
    # the review's blocked list and a text's page change licenses through this route, then admit the texts
    request, depot = web
    request("POST", "/api/licenses", {"id": "CC-BY-NC-4.0", "name": "Non commercial", "text": "text"})
    for text in ("one", "two"):
        request("POST", "/api/texts", {"text": text, "licenses": ["CC-BY-NC-4.0"], "declaration": ME})
    request("POST", "/api/texts", {"text": "held", "licenses": ["CC0-1.0"], "declaration": ME})
    request("POST", "/api/decisions", {"status": "refused", "selection": named(["test-depot-1"]), "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-3"]), "declaration": ME})
    snapshot = depot.create_snapshot("first")["id"]

    status, changed = wrote(request, "/api/texts/licenses", {"selection": named(["test-depot-2", "test-depot-1"]),
                                                            "licenses": ["CC0-1.0"], "declaration": ME})

    assert (status, changed) == (200, {"decisions": [{"decision": "test-depot-6", "texts": 2}], "skipped": 0,
                                       "left": {}})
    shown = request("GET", "/api/texts/test-depot-1")[1]
    assert shown["status"] == "under review" and shown["text"]["licenses"] == ["CC0-1.0"]
    assert shown["decisions"][-1]["reason"] == "licenses changed from CC-BY-NC-4.0 to CC0-1.0"
    assert request("POST", "/api/texts/blocked", {"selection": named(["test-depot-1", "test-depot-2"])}) == (
        200, {"rows": [], "next": None, "partial": False})
    held = {"selection": named(["test-depot-3"]), "licenses": ["CC-BY-NC-4.0"], "declaration": ME}
    assert request("POST", "/api/texts/licenses", {**held, "selection": named(["test-depot-9"])})[0] == 404
    assert request("POST", "/api/texts/licenses", {**held, "licenses": []})[0] == 400
    assert request("POST", "/api/texts/licenses", held)[0] == 200
    shown = request("GET", "/api/texts/test-depot-3")[1]
    assert (shown["text"]["licenses"], shown["revisions"]) == (["CC-BY-NC-4.0"], 2)
    assert [(one["id"], one["revision"], one["licenses"]) for one in depot.snapshot_texts(snapshot)["texts"]] == [
        ("test-depot-3", 1, ["CC0-1.0"])]


def test_the_content_of_many_texts_changes_from_the_route_as_from_the_command(web, tmp_path, monkeypatch):
    request, depot = web
    other = Depot.new(str(tmp_path / "other.sqlite"), About("test-depot", "Ada Lovelace", "ada@example.org"))
    other.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    me = Declaration(ME["name"], ME["email"], True)
    for one in (depot, other):
        one.add([NewText(text=f"text {n}", licenses=("CC0-1.0",), title=f"Title {n}", tags=("en",)) for n in range(4)]
                + [NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",))], me)
        one.decide("admitted", "true", me)
    lines = [{"id": "test-depot-1", "text": "text 0, as its source publishes it"},
             {"id": "test-depot-2", "text": "text 1"},
             {"id": "test-depot-3", "text": "text 2, as its source publishes it", "was": 3}]

    assert request("POST", "/api/texts/content", {"texts": lines + [{"id": "test-depot-5", "text": "x"}],
                                                  "declaration": ME})[0] == 422  # an item
    assert request("POST", "/api/texts/content", {"texts": [{"id": "test-depot-9", "text": "x"}],
                                                  "declaration": ME})[0] == 404
    assert request("POST", "/api/texts/content", {"texts": [{"id": "test-depot-1", "text": "x", "was": 0}],
                                                  "declaration": ME})[0] == 400
    assert request("POST", "/api/texts/content", {"texts": [{"id": "test-depot-1", "text": "x"}]})[0] == 400
    assert request("POST", "/api/texts/content", {"texts": lines, "declaration": ME}) == (200, {
        "decisions": [{"decision": "test-depot-3", "reason": "content changed from version 1 to version 2",
                       "texts": 1}],
        "changed": 1, "unchanged": 1, "stale": 1, "stale_ids": ["test-depot-3"]})
    (tmp_path / "content.jsonl").write_text("".join(json.dumps(one) + "\n" for one in lines), encoding="utf-8")
    monkeypatch.setenv("WPT_DEPOT", other.address)
    assert main(["content-file", str(tmp_path / "content.jsonl"), "--name", ME["name"], "--email", ME["email"],
                 "--accept"]) == 0

    def state(one):
        return (one.sql("SELECT id, revision, content_version, status, title, tags, licenses, text "
                        "FROM texts ORDER BY id"),
                one.sql("SELECT id, status, reason FROM decisions ORDER BY id"),
                one.sql("SELECT text, decision FROM text_decisions ORDER BY text, decision"), one.counts())
    assert state(other) == state(depot)


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


def test_scores_from_the_pages_change_many_texts_under_one_decision_and_the_text_s_page_shows_them(web):
    request, depot = web
    for words in ("one", "two"):
        request("POST", "/api/texts", {"text": words, "licenses": ["CC0-1.0"], "declaration": ME})
    rank = {"name": "rank", "value": 2.5, "content_version": 1}

    assert request("POST", "/api/texts/scores", {"texts": [{"id": "test-depot-1", "scores": [{"name": "rank"}]}],
                                                 "declaration": ME})[0] == 400
    assert request("POST", "/api/texts/scores", {"texts": [{"id": "test-depot-1", "scores": [
        {**rank, "content_version": 2}]}], "declaration": ME})[0] == 422
    assert request("POST", "/api/texts/scores", {"texts": [{"id": "test-depot-9", "scores": [rank]}],
                                                 "declaration": ME})[0] == 404
    assert request("POST", "/api/texts/scores", {"texts": [{"id": "test-depot-1", "scores": [rank]},
                                                           {"id": "test-depot-2", "scores": [rank]}],
                                                 "declaration": ME}) == (
        200, {"decision": "test-depot-3", "reason": "scores changed: set rank", "changed": 2, "left_out": 0})
    shown = request("GET", "/api/texts/test-depot-2")[1]
    assert (shown["text"]["scores"], shown["revisions"]) == ([rank], 2)
    assert request("POST", "/api/texts/scores", {"texts": [{"id": "test-depot-1", "remove": ["rank"]},
                                                           {"id": "test-depot-2", "scores": [rank]}],
                                                 "declaration": ME})[1]["left_out"] == 1
    assert request("GET", "/api/texts/test-depot-1")[1]["text"]["scores"] is None


def test_one_score_set_on_a_selection_from_the_pages_is_one_decision_and_its_name_is_listed(web):
    request, depot = web
    for words in ("one", "two", "three"):
        request("POST", "/api/texts", {"text": words, "licenses": ["CC0-1.0"], "tags": ["en"], "declaration": ME})
    selection = {"filter": {"tags": ["en"]}}

    status, done = wrote(request, "/api/texts/score", {"selection": selection, "name": "rank", "value": 3,
                                                       "declaration": ME})

    assert (status, done) == (200, {"decision": "test-depot-4", "changed": 3, "left_out": 0, "left": {}})
    assert request("GET", "/api/texts/test-depot-2")[1]["text"]["scores"] == [
        {"name": "rank", "value": 3.0, "content_version": 1}]
    assert request("GET", "/api/scores") == (200, [{"name": "rank", "texts": 3}])
    assert request("GET", "/api/scores/rank/values") == (200, {
        "name": "rank", "texts": 3, "low": 3.0, "high": 3.0, "common": [{"value": 3.0, "texts": 3}]})
    assert wrote(request, "/api/texts/score", {"selection": named(["test-depot-1"]), "name": "rank", "remove": True,
                                               "declaration": ME})[1]["changed"] == 1
    assert request("GET", "/api/scores") == (200, [{"name": "rank", "texts": 2}])
    # a value and a removal at once, or neither, is no request
    for wrong in ({"value": 1, "remove": True}, {}, {"remove": False}):
        assert request("POST", "/api/texts/score", {"selection": selection, "name": "rank", **wrong,
                                                    "declaration": ME})[0] == 400


def test_a_refusal_made_from_the_pages_takes_its_reason_afterwards_from_its_maker_only(web):
    # the Texts page refuses at one key, then offers reasons: the reason chosen must be that refusal's public reason
    request, depot = web
    request("POST", "/api/texts", {"text": "one", "licenses": ["CC0-1.0"], "declaration": ME})
    _, made = wrote(request, "/api/decisions", {"status": "refused", "selection": named(["test-depot-1"]),
                                                "declaration": ME})

    status, row = request("POST", f"/api/decisions/{made['decision']}/reason", {"reason": "bad quality",
                                                                                "declaration": ME})

    assert status == 200 and (row["id"], row["status"], row["reason"], row["name"]) == (
        made["decision"], "refused", "bad quality", ME["name"])
    assert [(one["status"], one["reason"]) for one in request("GET", "/api/texts/test-depot-1")[1]["decisions"]][-1] \
        == ("refused", "bad quality")
    other = {**ME, "name": "Grace Hopper"}
    assert request("POST", f"/api/decisions/{made['decision']}/reason", {"reason": "mine", "declaration": other})[0] \
        == 422
    assert request("POST", f"/api/decisions/{made['decision']}/reason", {"reason": "", "declaration": ME})[0] == 400


def test_credits_of_many_texts_change_under_one_decision_from_the_route(web):
    request, depot = web
    for words in ("one", "two"):
        request("POST", "/api/texts", {"text": words, "licenses": ["CC0-1.0"], "declaration": ME})

    assert request("POST", "/api/texts/credits", {"texts": [{"id": "test-depot-1", "licenses": ["MIT"]}],
                                                  "declaration": ME})[0] == 400
    assert request("POST", "/api/texts/credits", {"texts": [{"id": "test-depot-9", "title": "x"}],
                                                  "declaration": ME})[0] == 404
    assert request("POST", "/api/texts/credits", {"texts": [{"id": "test-depot-1", "title": "owner/repo"},
                                                            {"id": "test-depot-2", "authors": ["Ann Lee"]}],
                                                  "declaration": ME}) == (
        200, {"decision": "test-depot-3", "reason": "credits changed: title, authors", "changed": 2, "left_out": 0})
    shown = request("GET", "/api/texts/test-depot-1")[1]
    assert (shown["text"]["title"], shown["revisions"]) == ("owner/repo", 2)
    assert request("POST", "/api/texts/credits", {"texts": [{"id": "test-depot-1", "title": "owner/repo"}],
                                                  "declaration": ME})[0] == 422
    assert request("POST", "/api/texts/credits", {"texts": [{"id": "test-depot-2", "title": "x",
                                                             "was": {"licenses": None}}], "declaration": ME})[0] == 400
    assert request("POST", "/api/texts/credits", {"texts": [{"id": "test-depot-1", "title": "x",
                                                             "was": {"title": None}},
                                                            {"id": "test-depot-2", "title": "Two",
                                                             "was": {"title": None}}], "declaration": ME})[1] == {
        "decision": "test-depot-4", "reason": "credits changed: title", "changed": 1, "left_out": 1}


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


def test_a_write_over_several_named_texts_names_its_first_texts_on_the_jobs_page(web):
    request, depot = web
    for title in ("Fox", None, "Dog", "Owl"):
        request("POST", "/api/texts", {"text": f"The {title}.", "licenses": ["CC0-1.0"], "title": title, "declaration": ME})
    request("POST", "/api/texts", {"kind": "conversation", "content": CONVERSATION, "licenses": ["CC0-1.0"],
                                   "declaration": ME})
    ids = [f"test-depot-{n}" for n in range(1, 5)]
    status, admitted = request("POST", "/api/decisions", {"status": "admitted", "selection": named(ids),
                                                          "declaration": ME})
    # the first three by the name the pages give them, never a bare id, so a person tells which texts a job decided
    assert status in (200, 202) and admitted["title"] == "admit 4 named texts (Fox, an untitled text, Dog, ...)"
    status, item = request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-5"]),
                                                      "declaration": ME})
    assert item["title"] == "admit a conversation"


def test_a_decision_that_leaves_out_the_same_status_records_none_on_the_texts_it_would_not_change(web):
    request, depot = web
    for title in ("Fox", "Dog", "Owl"):
        request("POST", "/api/texts", {"text": f"The {title}.", "licenses": ["CC0-1.0"], "title": title, "declaration": ME})
    request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-1"]), "declaration": ME})
    every = named(["test-depot-1", "test-depot-2", "test-depot-3"])
    status, admitted = wrote(request, "/api/decisions", {"status": "admitted", "selection": every, "leave_same": True,
                                                         "declaration": ME})
    assert status == 200 and (admitted["texts"], admitted["left"]) == (2, {"already admitted": 1})
    assert len(request("GET", "/api/texts/test-depot-1")[1]["decisions"]) == 2  # its add and admission, no second one
    status, refused = wrote(request, "/api/decisions", {"status": "admitted", "selection": every, "leave_same": True,
                                                        "declaration": ME})
    assert status == 404 and "admitted already" in refused["problems"][0]
    status, again = wrote(request, "/api/decisions", {"status": "admitted", "selection": named(["test-depot-1"]),
                                                      "declaration": ME})
    assert status == 200 and len(request("GET", "/api/texts/test-depot-1")[1]["decisions"]) == 3  # asked for: recorded


def test_each_thing_a_page_lists_has_a_view_and_the_details_change_through_the_routes(web, tmp_path, monkeypatch):
    request, depot = web
    monkeypatch.chdir(tmp_path)  # no .env: the settings are the test's alone
    monkeypatch.delenv("WPT_STORE", raising=False)
    request("POST", "/api/texts", {"text": "The fox.", "licenses": ["CC0-1.0"], "tags": ["en"], "title": "Fox",
                                   "declaration": ME})
    status, admitted = request("POST", "/api/decisions", {"status": "admitted", "selection": named(["test-depot-1"]),
                                                          "reason": "clear", "declaration": ME})
    assert admitted["title"] == "admit Fox"  # a write over one text names it on the Jobs page

    status, license = request("GET", "/api/licenses/CC0-1.0")
    assert (status, license["text"], license["allowed"], license["texts"]) == (200, "The CC0 text.", True, 1)
    assert request("GET", "/api/licenses/MIT")[0] == 404
    status, decision = request("GET", "/api/decisions/test-depot-2")
    assert (status, decision["decision"]["reason"], decision["texts"]) == (200, "clear", 1)
    assert set(decision) == {"decision", "texts", "erasures"}  # no reported problem
    assert request("GET", "/api/decisions/latest")[0] == 200  # still the latest decisions

    assert request("POST", "/api/texts/test-depot-1/credits", {"title": "The fox", "tags": ["en", "animals"],
                                                              "declaration": ME}) == (
        200, {"decision": "test-depot-3", "reason": "title changed from Fox to The fox; tags changed from en to animals, "
                                                    "en"})
    shown = request("GET", "/api/texts/test-depot-1")[1]
    assert (shown["status"], shown["text"]["title"]) == ("admitted", "The fox")
    assert "libraries" not in shown["text"] and "pins" not in shown
    assert request("POST", "/api/texts/test-depot-1/credits", {"title": "Fox", "declaration": ME})[0] == 200


def test_a_clone_s_pages_say_what_belongs_to_the_upstream_and_refuse_changing_it(web, serve, tmp_path):
    request, upstream = web
    assert request("POST", "/api/texts", {"text": "Theirs.", "licenses": ["CC0-1.0"], "declaration": ME})[0] == 200
    upstream.decide("admitted", ["test-depot-1"], Declaration(ME["name"], ME["email"], True))
    upstream.create_snapshot("theirs")
    assert request("POST", "/api/tokenizers", {"name": "theirs", "snapshot": "theirs", "sample_bytes": 1000,
                                               "vocab_size": 560})[0] == 200
    follower = clone(request.url, str(tmp_path / "clone.sqlite"), About("test-clone", "Grace Hopper", "grace@example.org"))
    mine = serve(follower)
    assert mine("POST", "/api/texts", {"text": "Mine.", "licenses": ["CC0-1.0"], "declaration": ME}) == (
        200, {"id": "test-clone-1"})
    change = follower.change()

    status, refused = mine("POST", "/api/decisions", {"status": "refused", "selection": named(["test-depot-1", "test-clone-1"]),
                                                       "declaration": ME})
    assert status == 422 and "test-depot-1" in refused["problems"][0] and "to the upstream" in refused["problems"][0]
    assert mine("PUT", "/api/tokenizers/theirs", {"snapshot": "test-depot-1", "sample_bytes": 2000,
                                                  "vocab_size": 560})[0] == 422
    assert follower.change() == change

    assert mine("GET", "/api/texts/test-depot-1")[1]["upstream"] is True
    assert mine("GET", "/api/texts/test-clone-1")[1]["upstream"] is False
    assert mine("GET", "/api/tokenizers/theirs")[1]["upstream"] is True
    assert mine("POST", "/api/texts/blocked", {"selection": named(["test-depot-1", "test-clone-1"]), "status": "refused"}) == (
        200, {"rows": [{"id": "test-depot-1", "reasons": ["belongs to the upstream"]}], "next": None, "partial": False})
    assert mine("POST", "/api/decisions", {"status": "refused", "selection": named(["test-clone-1"]), "declaration": ME})[0] == 200
    assert request("GET", "/api/texts/test-depot-1")[1]["upstream"] is False  # in the upstream, it is its own


def test_the_web_command_refuses_a_depot_of_another_format_before_serving(tmp_path, monkeypatch, capsys):
    address = str(tmp_path / "depot")
    depot = Depot.new(address, About("served", "Ada Lovelace", "ada@example.org"))
    with depot.db.write("test") as (con, _):
        con.execute("UPDATE about_revisions SET format = ?", (FORMAT + 1,))
    depot.db.close()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WPT_DEPOT", address)
    monkeypatch.setattr("wpt.depot.web.ui.built", lambda: tmp_path)
    monkeypatch.setattr("wpt.depot.web.make_server", lambda *args: pytest.fail("a depot of another format is served"))
    capsys.readouterr()

    assert main(["web"]) == 1
    assert bare(capsys.readouterr().err).strip() == (
        f"wpt depot web: OtherFormat: {address} is a depot of format {FORMAT + 1}, newer than the format {FORMAT} "
        "this wpt reads: update wpt to read it; a depot is never brought down to an older format")


def test_texts_put_back_under_review_from_the_pages_and_the_job_says_so(web):
    request, depot = web
    for text in ("one", "two"):
        request("POST", "/api/texts", {"text": text, "licenses": ["CC0-1.0"], "tags": ["en"], "declaration": ME})
    both = ["test-depot-1", "test-depot-2"]
    request("POST", "/api/decisions", {"status": "admitted", "selection": named(both), "declaration": ME})
    request("POST", "/api/decisions", {"status": "refused", "selection": named(["test-depot-2"]), "declaration": ME})

    status, started = request("POST", "/api/decisions", {"status": "under review", "selection": {"filter": {"tags": ["en"]}},
                                                         "reason": "too quick", "declaration": ME})

    assert status in (200, 202) and started["title"] == "put back under review texts tagged en"
    job = started
    while job["state"] in ("queued", "running"):
        time.sleep(0.05)
        job = request("GET", f"/api/jobs/{started['id']}")[1]
    assert (job["state"], job["answer"]) == ("done", {"decision": "test-depot-5", "texts": 2, "left": {}})
    shown = request("GET", "/api/texts/test-depot-1")[1]
    assert shown["status"] == "under review"
    assert (shown["decisions"][-1]["reason"], shown["decisions"][-1]["put_back"]) == (
        "put back under review: too quick", True)
    assert request("GET", "/api/texts/test-depot-2")[1]["status"] == "under review"
    status, refused = request("POST", "/api/decisions", {"status": "under review", "selection": named(both),
                                                         "declaration": ME})
    assert status == 404 and "under review already" in refused["problems"][0]


def test_change_times_gives_the_time_of_each_change_the_depot_holds(web):
    request, depot = web
    request("POST", "/api/texts", {"text": "one", "licenses": ["CC0-1.0"], "declaration": ME})
    now = depot.change()

    status, times = request("POST", "/api/changes/times", {"changes": [now, now, now + 50]})

    assert status == 200 and [one["change"] for one in times] == [now] and times[0]["made_at"]


def record(source, target, version, part, data=b"", filtered=""):
    """A Remote Settings record of one model file holding data."""
    return {"fromLang": source, "toLang": target, "version": version, "fileType": part,
            "filter_expression": filtered,
            "attachment": {"hash": hashlib.sha256(data).hexdigest(), "size": len(data),
                           "location": f"{source}{target}-{version}-{part}"}}


def model(source, target, version, filtered="", vocabularies=("vocab",)):
    return [record(source, target, version, part, f"{source}{target}{version}{part}".encode(), filtered)
            for part in ("model", "lex", *vocabularies)]


def test_each_pair_gets_its_newest_complete_model_that_every_firefox_gets():
    records = [*model("de", "en", "1.0"), *model("de", "en", "2.0"),
               *[one for one in model("de", "en", "2.1") if one["fileType"] != "lex"],  # incomplete
               *model("en", "fr", "1.0a1"), *model("en", "fr", "2.0", filtered="env.channel == 'nightly'"),
               *model("ja", "en", "1.0", vocabularies=("srcvocab", "trgvocab"))]
    found = {(pair["from"], pair["to"]): (pair["version"], sorted(pair["files"])) for pair in pairs_of(records)}
    assert found == {("de", "en"): ("2.0", ["lex", "model", "vocab"]),
                     ("en", "fr"): ("1.0a1", ["lex", "model", "vocab"]),
                     ("ja", "en"): ("1.0", ["lex", "model", "srcvocab", "trgvocab"])}


def test_a_model_file_is_fetched_once_checked_kept_in_the_store_and_served_from_it(web, tmp_path, monkeypatch):
    request, _ = web
    monkeypatch.chdir(tmp_path)  # no .env: the settings are the test's alone
    monkeypatch.setenv("WPT_STORE", (tmp_path / "store").as_uri())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    good = record("de", "en", "1.0", "model", b"the model's bytes")
    lying = record("de", "en", "1.0", "lex", b"the shortlist's bytes")
    served = {good["attachment"]["location"]: b"the model's bytes",
              lying["attachment"]["location"]: b"other bytes"}
    fetched = []

    def mozilla(url, timeout=None):
        if url == RECORDS:
            return contextlib.closing(io.BytesIO(json.dumps({"data": [good, lying, record(
                "de", "en", "1.0", "vocab", b"v")]}).encode()))
        fetched.append(url)
        return contextlib.closing(io.BytesIO(served[url.removeprefix(ATTACHMENTS)]))
    monkeypatch.setattr(translations, "urlopen", mozilla)

    status, listed = request("GET", "/api/translations")
    assert status == 200 and [(p["from"], p["to"], p["version"]) for p in listed["pairs"]] == [("de", "en", "1.0")]
    assert listed["pairs"][0]["files"]["model"] == {"sha256": good["attachment"]["hash"], "size": 17}

    model_url = f"/api/translations/files/{good['attachment']['hash']}"
    assert request("GET", model_url) == (200, b"the model's bytes")
    assert request("GET", model_url) == (200, b"the model's bytes")
    assert len(fetched) == 1  # the second answer came from the store
    assert store_from_settings().get_bytes(good["attachment"]["hash"]) == b"the model's bytes"

    status, refused = request("GET", f"/api/translations/files/{lying['attachment']['hash']}")
    assert status == 503 and "other bytes" in refused["problems"][0]
    assert not store_from_settings().has(lying["attachment"]["hash"])
    assert not store_from_settings().has(hashlib.sha256(b"other bytes").hexdigest())

    assert request("GET", f"/api/translations/files/{'0' * 64}")[0] == 404  # only the files Mozilla lists

    monkeypatch.delenv("WPT_STORE")
    status, refused = request("GET", model_url)
    assert status == 503 and "WPT_STORE" in refused["problems"][0]
