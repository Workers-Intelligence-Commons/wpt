import hashlib
import json
import re
import shutil
import threading
from pathlib import Path

import apsw
import pytest

import depot_app
from depot import Declaration, Depot, NewText
from depot.schema import FORMAT
from wic_utils.lines import bare
from depot_app.jobs import Jobs
from depot_app.web.routes import routes
from depot_app.web.server import make_server

ME = ["--name", "Ada Lovelace", "--email", "ada@example.org", "--accept"]
PROBE = {"question": "What is in the box?", "patterns": ["(?i)secret"]}


@pytest.fixture
def depot(tmp_path, monkeypatch):
    """A new depot made by `depot new`, named by DEPOT, with the
    license CC0-1.0 added by `depot license add`, in a folder that holds
    no .env, so the settings are the test's alone; DOCK names a dock
    of the test's folder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEPOT", raising=False)
    address = str(tmp_path / "depot.sqlite")
    assert depot_command("new", address, "--name", "test-depot", "--maintainer", "Ada Lovelace",
                         "--takedown-contact", "ada@example.org") == 0
    monkeypatch.setenv("DEPOT", address)
    monkeypatch.setenv("DOCK", (tmp_path / "dock").as_uri())
    (tmp_path / "cc0.txt").write_text("The CC0 text.", encoding="utf-8")
    assert depot_command("license", "add", "CC0-1.0", "Creative Commons Zero v1.0 Universal",
                         str(tmp_path / "cc0.txt")) == 0
    return address


REPOSITORY = Path(__file__).parents[3]  # this checkout, whose orders give a model part that passes its schema


def depot_command(*argv) -> int:
    return depot_app.main(list(argv))


def printed_rows(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_texts_go_from_a_file_through_a_decision_into_a_snapshot(depot, tmp_path, capsys):
    # the order of work the commands exist for: add, decide, make a snapshot of what is admitted
    (tmp_path / "fox.txt").write_text("The quick brown fox.\n", encoding="utf-8")
    (tmp_path / "notice.txt").write_text("(c) 2026 Ann", encoding="utf-8")
    (tmp_path / "probe.json").write_text(json.dumps(PROBE), encoding="utf-8")
    assert depot_command("add", str(tmp_path / "fox.txt"), "--license", "CC0-1.0", "--notice",
                         str(tmp_path / "notice.txt"), "--tag", "en", "--title", "Fox", "--author", "Ann",
                         "--address", "https://example.org/fox", *ME) == 0
    assert depot_command("add", str(tmp_path / "probe.json"), "--kind", "probe", "--license", "CC0-1.0", *ME) == 0
    assert depot_command("decide", "admit", "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'en')", *ME) == 0
    assert depot_command("snapshot", "new") == 0
    made = bare(capsys.readouterr().out)

    assert depot_command("show", "test-depot-1") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["text"] | {"decisions": None} == {
        "id": "test-depot-1", "revision": 1, "change": shown["text"]["change"], "text": "The quick brown fox.\n", "kind": "plain", "content": None, "content_version": 1, "length": 21, "licenses": ["CC0-1.0"],
        "notice": "test-depot-1", "title": "Fox", "authors": ["Ann"], "tags": ["en"], "address": "https://example.org/fox", "scores": None,
        "decisions": None}
    assert [(one["status"], one["name"], one["email"]) for one in shown["decisions"]] == [
        ("under review", "Ada Lovelace", "ada@example.org"), ("admitted", "Ada Lovelace", "ada@example.org")]
    assert "libraries" not in shown and "pins" not in shown
    assert depot_command("show", "test-depot-1", "--from", "4", "--characters", "5") == 0
    assert json.loads(capsys.readouterr().out) == {"id": "test-depot-1", "version": 1, "length": 21, "from": 4,
                                                    "text": "quick"}

    assert depot_command("sql", "SELECT id, kind, status FROM texts ORDER BY id") == 0
    assert printed_rows(capsys) == [{"id": "test-depot-1", "kind": "plain", "status": "admitted"},
                                    {"id": "test-depot-2", "kind": "probe", "status": "under review"}]

    [snapshot] = Depot.open(depot).snapshots()
    assert f"snapshot {snapshot['id']}: {snapshot['name']}," in made
    assert [one["id"] for one in Depot.open(depot).snapshot_texts(snapshot["id"])["texts"]] == ["test-depot-1"]
    with pytest.raises(SystemExit):  # no such command any more
        depot_command("library", "new", "next-model")


def test_a_license_is_added_once_with_its_official_text_read_from_a_file(depot, tmp_path, capsys):
    (tmp_path / "mit.txt").write_text("Permission is hereby granted…\n", encoding="utf-8")

    assert depot_command("license", "add", "MIT", "MIT License", str(tmp_path / "mit.txt")) == 0

    assert Depot.open(depot).sql("SELECT * FROM licenses WHERE id = 'MIT'") == [
        {"id": "MIT", "name": "MIT License", "text": "Permission is hereby granted…\n"}]
    capsys.readouterr()
    assert depot_command("license", "add", "MIT", "MIT License", str(tmp_path / "mit.txt")) == 2  # held already
    assert bare(capsys.readouterr().err).startswith("Refused:")
    assert depot_command("license", "add", "ISC", "ISC License", str(tmp_path / "absent.txt")) == 2
    assert Depot.open(depot).sql("SELECT count(*) AS n FROM licenses")[0]["n"] == 2


def test_a_decision_without_the_confirmation_is_refused_and_writes_nothing(depot, tmp_path, capsys):
    (tmp_path / "fox.txt").write_text("fox", encoding="utf-8")
    assert depot_command("add", str(tmp_path / "fox.txt"), "--license", "CC0-1.0", "--name", "Ada Lovelace",
                         "--email", "ada@example.org") == 2
    assert bare(capsys.readouterr().err).startswith("Refused: a decision needs the confirmation")
    assert Depot.open(depot).sql("SELECT count(*) AS n FROM texts")[0]["n"] == 0



def test_texts_change_licenses_by_their_ids_and_start_over_under_review(depot, tmp_path, capsys):
    (tmp_path / "mit.txt").write_text("Permission is hereby granted…\n", encoding="utf-8")
    assert depot_command("license", "add", "MIT", "MIT License", str(tmp_path / "mit.txt")) == 0
    first, second = Depot.open(depot).add([NewText(text="one", licenses=("CC0-1.0",)),
                                           NewText(text="two", licenses=("CC0-1.0",))], _declaration())
    assert depot_command("decide", "refuse", "true", *ME) == 0
    capsys.readouterr()

    assert depot_command("relicense", first, second, "--license", "MIT", *ME) == 0

    assert "licenses changed to MIT, on 2 texts" in capsys.readouterr().out
    shown = Depot.open(depot).show(second)
    assert shown["status"] == "under review" and shown["text"]["licenses"] == ["MIT"]
    assert shown["decisions"][-1]["reason"] == "licenses changed from CC0-1.0 to MIT"
    assert depot_command("relicense", first, "--license", "GPL-2.0-only", *ME) == 2  # a license it does not hold
    assert bare(capsys.readouterr().err).startswith("Refused: the depot holds no license GPL-2.0-only")
    assert depot_command("decide", "refuse", "true", "--reason", "licenses changed from MIT to CC0-1.0", *ME) == 2
    assert Depot.open(depot).show(first)["status"] == "under review"


def test_back_to_review_undoes_an_admission_or_a_refusal_and_never_an_erasure_decision(depot, capsys):
    admitted, refused = Depot.open(depot).add([NewText(text="one", licenses=("CC0-1.0",)),
                                               NewText(text="two", licenses=("CC0-1.0",))], _declaration())
    assert depot_command("decide", "admit", f"id = '{admitted}'", *ME) == 0
    assert depot_command("decide", "refuse", f"id = '{refused}'", *ME) == 0
    capsys.readouterr()

    assert depot_command("decide", "back-to-review", "true", "--reason", "decided too fast", *ME) == 0

    assert "put back under review, on 2 texts" in capsys.readouterr().out
    for text in (admitted, refused):
        shown = Depot.open(depot).show(text)
        assert shown["status"] == "under review"
        assert shown["decisions"][-1]["reason"] == "put back under review: decided too fast"
    assert depot_command("decide", "back-to-review", "true", *ME) == 2  # every text is under review already
    with pytest.raises(SystemExit):
        depot_command("decide", "back-to-review", "--erasure", "test-depot-1", *ME)


def test_a_refusal_takes_its_reason_afterwards_from_the_reason_command(depot, capsys):
    [text] = Depot.open(depot).add([NewText(text="one", licenses=("CC0-1.0",))], _declaration())
    assert depot_command("decide", "refuse", f"id = '{text}'", *ME) == 0
    decision = Depot.open(depot).show(text)["decisions"][-1]["id"]
    capsys.readouterr()

    assert depot_command("reason", decision, "bad license", *ME) == 0

    assert bare(capsys.readouterr().out.splitlines()[-1]) == f"decision {decision}: its reason is now 'bad license'"
    assert Depot.open(depot).show(text)["decisions"][-1]["reason"] == "bad license"
    assert depot_command("reason", decision, "mine", "--name", "Grace Hopper", "--email", "ada@example.org",
                         "--accept") == 2  # only its maker


def test_a_texts_content_changes_from_a_file_and_it_starts_over_under_review(depot, tmp_path, capsys):
    [text] = Depot.open(depot).add([NewText(text="old words", licenses=("CC0-1.0",))], _declaration())
    (tmp_path / "new.txt").write_text("new words", encoding="utf-8")

    assert depot_command("content", text, str(tmp_path / "new.txt"), *ME) == 0
    assert depot_command("content", text, str(tmp_path / "new.txt"), *ME) == 0

    out = capsys.readouterr().out
    assert "under review, content version 2" in out and "its content is this already, nothing written" in out
    shown = Depot.open(depot).show(text)
    assert (shown["text"]["text"], shown["content_versions"]) == ("new words", 2)
    assert shown["decisions"][-1]["reason"] == "content changed from version 1 to version 2"


def test_an_erasure_is_asked_by_place_and_listed_with_its_request(depot, tmp_path, capsys):
    [text] = Depot.open(depot).add([NewText(text="call 555-0100 now", licenses=("CC0-1.0",))],
                                   _declaration())
    assert depot_command("erase", str(text), "5", "13", "--reason", "a phone number", *ME) == 0
    assert depot_command("erase", str(text), "--reason", "the whole text", *ME) == 0
    capsys.readouterr()
    assert depot_command("erasures") == 0
    assert [(one["reason"], one["needles"], one["erased"]) for one in printed_rows(capsys)] == [
        ("a phone number", [{"text": text, "field": None, "start": 5, "end": 13, "version": 1}], None),
        ("the whole text", [{"text": text, "field": None, "start": None, "end": None, "version": 1}], None)]
    with pytest.raises(SystemExit):  # a passage is a start and an end
        depot_command("erase", str(text), "5", "--reason", "half a place", *ME)


def test_an_erasure_request_is_decided_by_its_id_and_an_admitted_one_is_carried_out(depot, tmp_path, capsys):
    [kept, text] = Depot.open(depot).add([NewText(text="a", licenses=("CC0-1.0",)),
                                          NewText(text="call 555-0100 now", licenses=("CC0-1.0",))], _declaration())
    assert depot_command("erase", str(text), "5", "13", "--reason", "a phone number", *ME) == 0
    assert depot_command("erase", str(text), "--reason", "not wanted", *ME) == 0
    capsys.readouterr()

    assert depot_command("decide", "refuse", "--erasure", "test-depot-2", *ME) == 0
    assert "refused" in bare(capsys.readouterr().out)
    assert depot_command("decide", "admit", "--erasure", "test-depot-1", *ME) == 0
    assert "carried out" in bare(capsys.readouterr().out)

    assert Depot.open(depot).show(text)["text"]["text"] == "call  now"
    assert depot_command("erasures") == 0
    erased, refused = printed_rows(capsys)
    assert erased["erased"][0]["length"] == 8 and len(erased["erased"][0]["hash_a"]) == 64  # hexadecimal text
    assert refused["erased"] is None
    for argv in (["decide", "admit", *ME], ["decide", "admit", "true", "--erasure", "test-depot-1", *ME]):
        with pytest.raises(SystemExit):  # texts or an erasure request, one of the two
            depot_command(*argv)
    assert depot_command("decide", "admit", "--erasure", "test-depot-9", *ME) == 2  # no such request


def test_a_clone_follows_its_upstream_over_its_web_app_and_pulls_what_it_added_and_refuses_changing_it(
        depot, tmp_path, monkeypatch, capsys):
    opened = Depot.open(depot)
    server = make_server(routes(opened, Jobs(opened.address)), tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    clone = str(tmp_path / "clone.sqlite")
    try:
        assert depot_command("clone", address, clone, "--name", "test-clone", "--maintainer", "Grace Hopper",
                             "--takedown-contact", "grace@example.org") == 0
        Depot.open(depot).add([NewText(text="added upstream later", licenses=("CC0-1.0",))], _declaration())
        monkeypatch.setenv("DEPOT", clone)
        assert depot_command("pull") == 0
    finally:
        server.shutdown()
        server.server_close()
    assert Depot.open(clone).sql("SELECT text FROM texts") == [{"text": "added upstream later"}]
    assert Depot.open(clone).upstream() == address
    change = Depot.open(clone).change()
    capsys.readouterr()
    assert depot_command("decide", "admit", "true", *ME) == 2  # the upstream's text: only the upstream decides on it
    assert bare(capsys.readouterr().err).startswith("BelongsToUpstream: the text test-depot-1 belongs to the upstream")
    assert Depot.open(clone).change() == change


def test_a_backup_and_its_restore_are_made_by_commands_and_read_as_the_depot(depot, tmp_path, capsys):
    Depot.open(depot).add([NewText(text="kept", licenses=("CC0-1.0",))], _declaration())
    assert depot_command("backup") == 0
    Depot.open(depot).add([NewText(text="later", licenses=("CC0-1.0",))], _declaration())
    assert depot_command("backup") == 0
    capsys.readouterr()

    assert depot_command("restore", str(tmp_path / "restored.sqlite")) == 0
    assert depot_command("restore", str(tmp_path / "earlier.sqlite"), "--at", "3") == 0

    assert "restored from the dock's backups of test-depot, at its change 4" in bare(capsys.readouterr().out)
    assert Depot.open(str(tmp_path / "restored.sqlite")).sql("SELECT text FROM texts ORDER BY id") == [
        {"text": "kept"}, {"text": "later"}]
    assert Depot.open(str(tmp_path / "earlier.sqlite")).sql("SELECT text FROM texts") == [{"text": "kept"}]
    assert depot_command("restore", str(tmp_path / "restored.sqlite")) == 2  # it never overwrites


def test_a_snapshot_is_named_by_the_time_it_is_made_and_shown_by_that_name(depot, capsys):
    opened = Depot.open(depot)
    opened.add([NewText(text="the quick brown fox", licenses=("CC0-1.0",), title="Fox")], _declaration())
    opened.decide("admitted", "true", _declaration())
    capsys.readouterr()
    assert depot_command("snapshot", "new", "--note", "the first") == 0
    [made] = Depot.open(depot).snapshots()
    assert f"snapshot test-depot-1: {made['name']}, 1 texts at change" in bare(capsys.readouterr().out)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d-\d\d-\d\dZ", made["name"]) and made["note"] == "the first"
    with pytest.raises(SystemExit):  # it takes no name
        depot_command("snapshot", "new", "s1")
    assert depot_command("snapshot", "list") == 0
    assert [(row["name"], row["texts"]) for row in printed_rows(capsys)] == [(made["name"], 1)]
    assert depot_command("snapshot", "show", made["name"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["summary"]["kinds"]["plain"][0] == 1 and shown["erased_since"] == {"texts": 0, "erasures": []}
    assert depot_command("snapshot", "show", "s9") == 2
    for gone in (["build", "recipe.json"], ["tokenizer", "new", "tiny"], ["tokenizer", "train", "tiny"]):
        with pytest.raises(SystemExit):  # no such command any more
            depot_command(*gone)


def _declaration():
    return Declaration("Ada Lovelace", "ada@example.org", True)


def test_a_texts_credits_and_the_depots_facts_change_from_the_commands(depot, tmp_path, capsys):
    opened = Depot.open(depot)
    [text] = opened.add([NewText(text="The fox.", licenses=("CC0-1.0",), title="Fox", tags=("en",))], _declaration())
    opened.decide("admitted", "true", _declaration())
    snapshot = opened.create_snapshot()["id"]
    (tmp_path / "notice.txt").write_text("(c) Ann", encoding="utf-8")

    assert depot_command("credits", text, "--title", "The fox", *ME) == 0  # a new revision; the snapshot keeps its own
    assert depot_command("about", "--takedown-contact", "takedown@example.org") == 0
    with pytest.raises(SystemExit):  # a new title or none, not both
        depot_command("credits", text, "--title", "x", "--clear", "title", *ME)

    shown = Depot.open(depot)
    assert shown.about()["takedown_contact"] == "takedown@example.org"
    assert shown.show(text)["text"]["title"] == "The fox"
    assert [one["title"] for one in shown.snapshot_texts(snapshot)["texts"]] == ["Fox"]

    [other] = shown.add([NewText(text="A hen.", licenses=("CC0-1.0",), tags=("en",))], _declaration())
    assert depot_command("credits", other, "--title", "Hen", "--tag", "en", "--tag", "birds", "--notice",
                         str(tmp_path / "notice.txt"), *ME) == 0
    after = Depot.open(depot).show(other)
    assert (after["text"]["title"], after["text"]["tags"], after["notice"]) == ("Hen", ["birds", "en"], "(c) Ann")
    assert after["decisions"][-1]["reason"].startswith("title changed from none to Hen; tags changed from en to birds, en")
    # tags alone are rows of taggings: no new revision of the text
    assert depot_command("credits", other, "--tag", "birds", *ME) == 0
    tagged = Depot.open(depot).show(other)
    assert (tagged["text"]["tags"], tagged["text"]["revision"]) == (["birds"], after["text"]["revision"])


def test_the_score_command_sets_one_score_on_the_texts_a_condition_matches_and_score_names_lists_it(depot, capsys):
    opened = Depot.open(depot)
    first, second = opened.add([NewText(text="one", licenses=("CC0-1.0",)), NewText(text="two", licenses=("CC0-1.0",))],
                               _declaration())
    capsys.readouterr()

    assert depot_command("score", "rank", "--value", "4", "true", *ME) == 0

    assert bare(capsys.readouterr().out.splitlines()[-1]) == (
        "decision test-depot-2: under review, set rank to 4, on 2 texts")
    assert depot_command("score", "rank", "--remove", f"id = '{first}'", *ME) == 0
    assert bare(capsys.readouterr().out.splitlines()[-1]) == (
        "decision test-depot-3: under review, removed rank, on 1 texts")
    assert depot_command("score-names") == 0
    assert printed_rows(capsys) == [{"name": "rank", "texts": 1}]
    assert depot_command("score-values", "rank") == 0
    assert printed_rows(capsys) == [{"name": "rank", "texts": 1, "low": 4.0, "high": 4.0,
                                     "common": [{"value": 4.0, "texts": 1}]}]


def test_the_retag_command_says_its_cost_and_changes_the_tags_only_with_yes(depot, capsys):
    opened = Depot.open(depot)
    first, second = opened.add([NewText(text="one", licenses=("CC0-1.0",), tags=("en", "old")),
                                NewText(text="two", licenses=("CC0-1.0",), tags=("fr",))], _declaration())
    capsys.readouterr()

    assert depot_command("retag", "true", "--add", "book", "--remove", "old", *ME) == 0
    assert bare(capsys.readouterr().out.splitlines()[-1]).startswith("Nothing was written")
    assert [Depot.open(depot).show(one)["text"]["tags"] for one in (first, second)] == [["en", "old"], ["fr"]]

    assert depot_command("retag", "true", "--add", "book", "--remove", "old", "--yes", *ME) == 0
    lines = [bare(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0].startswith("2 of the 2 texts the condition matches would change")
    assert lines[-1].startswith("decision ") and lines[-1].endswith("tags changed on 2 texts")
    assert [Depot.open(depot).show(one)["text"]["tags"] for one in (first, second)] == [["book", "en"],
                                                                                     ["book", "fr"]]


def test_the_measure_command_writes_scores_files_the_scores_command_records(depot, tmp_path, capsys):
    opened = Depot.open(depot)
    texts = opened.add([NewText(text="The river runs north.", licenses=("CC0-1.0",)),
                        NewText(text="| a | b |", licenses=("CC0-1.0",)),
                        NewText(text="# Title", licenses=("CC0-1.0",))], _declaration())
    out = tmp_path / "measured"
    capsys.readouterr()

    assert depot_command("measure", "true", "--out", str(out), "--lines", "2") == 0

    assert bare(capsys.readouterr().out.splitlines()[-1]) == f"3 texts measured into 2 files in {out}"
    files = sorted(out.iterdir())
    assert [path.name for path in files] == ["scores-00001.jsonl", "scores-00002.jsonl"]
    lines = [json.loads(line) for path in files for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["id"] for line in lines] == sorted(texts)
    for path in files:
        assert depot_command("scores", str(path), *ME) == 0
    assert Depot.open(depot).score_names() == [{"name": "running-text", "texts": 3}, {"name": "symbols", "texts": 3}]
    capsys.readouterr()
    assert depot_command("measure", "true", "--out", str(out)) != 0
    assert "exists already" in capsys.readouterr().err


def test_the_credits_file_command_reads_one_text_per_line_and_records_one_decision_for_them_all(depot, tmp_path,
                                                                                                    capsys):
    opened = Depot.open(depot)
    first, second = opened.add([NewText(text="one", licenses=("CC0-1.0",), title="One"),
                                NewText(text="two", licenses=("CC0-1.0",))], _declaration())
    lines = [{"id": first, "title": "One"}, {"id": second, "title": "owner/repo", "authors": ["Ann Lee"]}]
    (tmp_path / "credits.jsonl").write_text("".join(json.dumps(one) + "\n" for one in lines), encoding="utf-8")
    capsys.readouterr()

    assert depot_command("credits-file", str(tmp_path / "credits.jsonl"), *ME) == 0

    assert bare(capsys.readouterr().out).strip() == (
        "decision test-depot-2: under review, credits changed: title, authors, on 1 texts; 1 texts left out, their "
        "credits unchanged")
    shown = Depot.open(depot).show(second)["text"]
    assert (shown["title"], shown["authors"]) == ("owner/repo", ["Ann Lee"])
    assert depot_command("credits-file", str(tmp_path / "credits.jsonl"), *ME) == 2  # nothing changes
    assert depot_command("credits-file", str(tmp_path / "missing.jsonl"), *ME) == 2
    # a line of tags alone writes rows of taggings, no new revision of the text
    (tmp_path / "tags.jsonl").write_text(json.dumps({"id": first, "tags": ["b", "a"]}) + "\n", encoding="utf-8")
    revision = Depot.open(depot).show(first)["text"]["revision"]
    assert depot_command("credits-file", str(tmp_path / "tags.jsonl"), *ME) == 0
    tagged = Depot.open(depot).show(first)
    assert (tagged["text"]["tags"], tagged["text"]["revision"]) == (["a", "b"], revision)
    assert tagged["decisions"][-1]["reason"] == "credits changed: tags"


def test_the_content_file_command_changes_only_the_content_of_the_texts_it_names_and_starts_each_over(
        depot, tmp_path, capsys):
    opened = Depot.open(depot)
    changed, same, stale, guarded, untouched = opened.add(
        [NewText(text=f"text {n}", licenses=("CC0-1.0",), title=f"Title {n}", authors=("Ann Lee",), tags=("en",),
                 notice="(c) Ann") for n in range(5)], _declaration())
    opened.decide("admitted", "true", _declaration())
    before = {one: opened.show(one) for one in (changed, same, stale, guarded, untouched)}
    lines = [{"id": changed, "text": "text 0, as its source publishes it"},
             {"id": same, "text": "text 1"},
             {"id": stale, "text": "text 2, from an older reading", "was": 2},
             {"id": guarded, "text": "text 3, as its source publishes it", "was": 1}]
    (tmp_path / "content.jsonl").write_text("".join(json.dumps(one) + "\n" for one in lines), encoding="utf-8")
    capsys.readouterr()

    assert depot_command("content-file", str(tmp_path / "content.jsonl"), *ME) == 0

    decision = "test-depot-3"
    assert [bare(line) for line in capsys.readouterr().out.splitlines()] == [
        f"decision {decision}: under review, content changed from version 1 to version 2, on 2 texts",
        f"2 texts changed, under review again; 1 left out, their content this "
        f"already; 1 left out, their content version no longer the one their line's was names (the first: {stale})"]
    now = Depot.open(depot)
    for text, words in ((changed, "text 0, as its source publishes it"), (guarded, "text 3, as its source publishes it")):
        shown = now.show(text)
        assert (shown["text"]["text"], shown["content_versions"], shown["status"]) == (words, 2, "under review")
        assert (shown["decisions"][-1]["id"], shown["decisions"][-1]["reason"]) == (
            decision, "content changed from version 1 to version 2")
        kept = ("title", "authors", "tags", "licenses", "address", "scores")
        assert {name: shown["text"][name] for name in kept} == {name: before[text]["text"][name] for name in kept}
        assert shown["notice"] == before[text]["notice"]
    for text in (same, stale, untouched):
        assert now.show(text) == before[text]
    counted = now.counts()
    now.recount()
    assert now.counts() == counted
    assert counted["status"] == {"admitted": 3, "under review": 2}
    assert depot_command("content-file", str(tmp_path / "content.jsonl"), *ME) == 2  # nothing changes now


def test_the_content_file_command_refuses_a_change_too_large_an_item_and_an_erased_passage_brought_back(
        depot, tmp_path, monkeypatch):
    opened = Depot.open(depot)
    secret = "TOKEN = 'abc123'"
    first, second = opened.add([NewText(text=f"a = 1\n{secret}\n", licenses=("CC0-1.0",)),
                                NewText(text="b = 2\n", licenses=("CC0-1.0",))], _declaration())
    (tmp_path / "probe.json").write_text(json.dumps(PROBE), encoding="utf-8")
    assert depot_command("add", str(tmp_path / "probe.json"), "--kind", "probe", "--license", "CC0-1.0", *ME) == 0
    item = "test-depot-4"
    request = opened.ask_erasure("a secret", [(first, 6, 6 + len(secret))], _declaration())
    opened.decide_erasure(request, "admitted", _declaration())
    change = Depot.open(depot).change()
    path = tmp_path / "content.jsonl"

    def refused(lines) -> bool:
        path.write_text("".join(json.dumps(one) + "\n" for one in lines), encoding="utf-8")
        return depot_command("content-file", str(path), *ME) == 2

    monkeypatch.setattr("depot.rewriting.MOST", 2)
    assert refused([{"id": first, "text": "x"}, {"id": second, "text": "y"}, {"id": item, "text": "z"}])
    assert refused([{"id": second, "text": "y"}, {"id": item, "text": "z"}])
    assert refused([{"id": second, "text": "y"}, {"id": first, "text": f"a = 2\n{secret}\n"}])
    assert refused([{"id": second, "text": "y"}, {"id": second, "text": "z"}])
    assert refused([{"id": second, "text": "y", "licenses": ["MIT"]}])
    assert refused([{"id": second, "text": "y"}, {"id": "test-depot-99", "text": "z"}])
    assert depot_command("content-file", str(tmp_path / "missing.jsonl"), *ME) == 2
    with pytest.raises(SystemExit):  # the decider's declaration is required
        depot_command("content-file", str(path))
    assert Depot.open(depot).change() == change
    assert refused([]) and not refused([{"id": second, "text": "y"}, {"id": first, "text": "a = 2\n"}])


def test_the_scores_command_reads_one_text_per_line_and_records_one_decision_for_them_all(depot, tmp_path, capsys):
    opened = Depot.open(depot)
    first, second = opened.add([NewText(text="one", licenses=("CC0-1.0",)), NewText(text="two", licenses=("CC0-1.0",))],
                               _declaration())
    lines = [{"id": first, "scores": [{"name": "rank", "value": 4, "content_version": 1}]},
             {"id": second, "scores": [{"name": "rank", "value": 0.5, "content_version": 1},
                                       {"name": "machine-made", "value": 0.03, "content_version": 1}]}]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(one) + "\n" for one in lines), encoding="utf-8")
    capsys.readouterr()

    assert depot_command("scores", str(tmp_path / "scores.jsonl"), *ME) == 0

    assert bare(capsys.readouterr().out).strip() == (
        "decision test-depot-2: under review, scores changed: set machine-made, rank, on 2 texts")
    assert depot_command("show", second) == 0
    assert json.loads(capsys.readouterr().out)["text"]["scores"] == [
        {"name": "machine-made", "value": 0.03, "content_version": 1},
        {"name": "rank", "value": 0.5, "content_version": 1}]
    assert depot_command("sql", "SELECT id, scores ->> '$[0].value' AS first FROM texts ORDER BY id") == 0
    assert printed_rows(capsys) == [{"id": first, "first": 4.0}, {"id": second, "first": 0.03}]

    (tmp_path / "again.jsonl").write_text(json.dumps({"id": first, "remove": ["rank"]}) + "\n"
                                          + json.dumps(lines[1]) + "\n", encoding="utf-8")
    assert depot_command("scores", str(tmp_path / "again.jsonl"), *ME) == 0
    assert bare(capsys.readouterr().out).strip() == (
        "decision test-depot-3: under review, scores changed: removed rank, on 1 texts; 1 texts left out, their scores "
        "unchanged")
    (tmp_path / "broken.jsonl").write_text('{"id": "test-depot-1", "remove": ["x"]}\nnot json\n', encoding="utf-8")
    assert depot_command("scores", str(tmp_path / "broken.jsonl"), *ME) == 2
    assert "not JSON lines" in bare(capsys.readouterr().err)
    assert depot_command("scores", str(tmp_path / "missing.jsonl"), *ME) == 2
    (tmp_path / "broken.jsonl").write_text('{"id": "test-depot-2", "scores": [{"name": "rank", "value": NaN, '
                                           '"content_version": 1}]}\n', encoding="utf-8")
    assert depot_command("scores", str(tmp_path / "broken.jsonl"), *ME) == 2
    assert "NaN is no JSON number" in bare(capsys.readouterr().err)
    assert Depot.open(depot).show(first)["revisions"] == 3


FIXTURES = REPOSITORY / "packages" / "depot" / "tests" / "fixtures"


@pytest.fixture
def older_depot(tmp_path, monkeypatch):
    """A copy of the fixture depot of format 12, named by DEPOT, and of its dock, named by DOCK, in a
    folder that holds no .env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEPOT_URL", raising=False)
    path = tmp_path / "depot.sqlite"
    shutil.copy(FIXTURES / "format-12.sqlite", path)
    shutil.copytree(FIXTURES / "format-12-dock", tmp_path / "dock")
    monkeypatch.setenv("DEPOT", str(path))
    monkeypatch.setenv("DOCK", (tmp_path / "dock").as_uri())
    return path


def test_the_upgrade_says_what_it_would_do_and_writes_nothing_until_yes(older_depot, capsys):
    before = hashlib.sha256(older_depot.read_bytes()).hexdigest()

    assert depot_command("upgrade", "--check") == 0
    checked = bare(capsys.readouterr().out)
    assert "What converts (step 12 to 13):" in checked and "-> snapshot ds-two-r2" in checked  # the dock's revision
    assert "every revision equal" in checked and "Cost of the steps left together:" in checked
    assert checked.strip().splitlines()[-2:] == [
        "Run: depot upgrade --yes",
        "Nothing was written: run `depot upgrade --yes` to upgrade, which runs this check again first."]
    saved = older_depot.with_name(older_depot.name + ".upgrade-check.json")
    assert saved.is_file()  # beside the depot's file, which stays as it was
    assert depot_command("upgrade") == 0  # the check is what it does by default, taking what the first one saved
    again = bare(capsys.readouterr().out)
    assert "Saved check: the check saved in " in again and "reading what each snapshot" not in again

    def timeless(said):  # what it measured as it read, the lines of the saved check aside
        return [line for line in said.splitlines() if "seconds" not in line and "saved" not in line.lower()
                and "reading what each snapshot" not in line]
    assert timeless(again) == timeless(checked)
    assert hashlib.sha256(older_depot.read_bytes()).hexdigest() == before

    assert depot_command("upgrade", "--yes") == 0

    said = bare(capsys.readouterr().out)
    assert "What converts" in said and f"brought up to format {FORMAT}" in said
    assert "those the check right before read" in said and not saved.exists()
    depot = Depot.open(str(older_depot))
    assert sorted(one["name"] for one in depot.snapshots()) == ["ds-mixed", "ds-two", "ds-two-r2", "lib-a", "lib-b"]


def test_the_upgrade_writes_nothing_when_its_check_fails(older_depot, capsys):
    con = apsw.Connection(str(older_depot))  # a pin of a revision newer than its lock, which no snapshot can read
    con.execute("UPDATE pins SET revision = (SELECT max(r.revision) FROM texts_revisions r WHERE r.id = pins.text) "
                "WHERE library = (SELECT id FROM libraries_revisions WHERE name = 'lib-a' AND latest)")
    con.close()
    before = hashlib.sha256(older_depot.read_bytes()).hexdigest()

    assert depot_command("upgrade", "--yes") == 2

    printed = capsys.readouterr()
    assert "The check FAILED" in bare(printed.out) and "Nothing was written." in bare(printed.err)
    assert hashlib.sha256(older_depot.read_bytes()).hexdigest() == before


def test_a_depot_of_a_newer_format_is_refused_and_the_upgrade_says_when_there_is_nothing_to_do(depot, capsys):
    assert depot_command("upgrade") == 0
    assert bare(capsys.readouterr().out).strip() == f"{depot}: at format {FORMAT} already, nothing to upgrade"
    found = Depot.open(depot)
    with found.db.write("test") as (con, _):
        con.execute("UPDATE about_revisions SET format = ?", (FORMAT + 1,))
    found.db.close()

    assert depot_command("sql", "SELECT name FROM about") == 2
    assert bare(capsys.readouterr().err).strip() == (
        f"OtherFormat: {depot} is a depot of format {FORMAT + 1}, newer than the format {FORMAT} this wpt reads: "
        "update wpt to read it; a depot is never brought down to an older format")
