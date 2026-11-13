import json
import threading

import pytest

import wpt
from wpt_depot import Declaration, Depot, NewText
from wpt_depot.corpus import read_table
from wpt_depot.schema import FORMAT
from wpt_recipes import Recipes
from wpt_store import open_store
from wpt_tokenizer import Tokenizer
from wpt_utils.lines import bare
from wpt.depot.web.routes import routes
from wpt.depot.web.server import make_server

ME = ["--name", "Ada Lovelace", "--email", "ada@example.org", "--accept"]
PROBE = {"question": "What is in the box?", "patterns": ["(?i)secret"]}


@pytest.fixture
def depot(tmp_path, monkeypatch):
    """A new depot made by `wpt depot new`, named by WPT_DEPOT, with the
    license CC0-1.0 added by `wpt depot license add`, in a folder that holds
    no .env, so the settings are the test's alone; WPT_STORE names a store
    of the test's folder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WPT_DEPOT", raising=False)
    address = str(tmp_path / "depot.sqlite")
    assert depot_command("new", address, "--name", "test-depot", "--maintainer", "Ada Lovelace",
                         "--takedown-contact", "ada@example.org") == 0
    monkeypatch.setenv("WPT_DEPOT", address)
    monkeypatch.setenv("WPT_STORE", (tmp_path / "store").as_uri())
    (tmp_path / "cc0.txt").write_text("The CC0 text.", encoding="utf-8")
    assert depot_command("license", "add", "CC0-1.0", "Creative Commons Zero v1.0 Universal",
                         str(tmp_path / "cc0.txt")) == 0
    return address


def depot_command(*argv) -> int:
    return wpt.main(["depot", *argv])


def printed_rows(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_texts_go_from_a_file_through_a_decision_into_libraries_and_a_dataset_that_travels_as_a_remote(
        depot, tmp_path, capsys):
    # the order of work the commands exist for: add, decide, fill a library, lock it, make a dataset, export it
    (tmp_path / "fox.txt").write_text("The quick brown fox.\n", encoding="utf-8")
    (tmp_path / "notice.txt").write_text("(c) 2026 Ann", encoding="utf-8")
    (tmp_path / "probe.json").write_text(json.dumps(PROBE), encoding="utf-8")
    assert depot_command("add", str(tmp_path / "fox.txt"), "--license", "CC0-1.0", "--notice",
                         str(tmp_path / "notice.txt"), "--tag", "en", "--title", "Fox", "--author", "Ann",
                         "--address", "https://example.org/fox", *ME) == 0
    assert depot_command("add", str(tmp_path / "probe.json"), "--kind", "probe", "--license", "CC0-1.0", *ME) == 0
    assert depot_command("decide", "admit", "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'en')", *ME) == 0
    assert depot_command("library", "new", "first model") == 0
    assert depot_command("library", "put", "test-depot-1", "true") == 0
    assert depot_command("library", "fork", "test-depot-1", "next model") == 0
    assert depot_command("library", "lock", "test-depot-1") == 0
    assert depot_command("dataset", "new", "first", "--tokenizer", "classics", "--library", "test-depot-1",
                         "--share", "en=1") == 0
    capsys.readouterr()

    assert depot_command("show", "test-depot-1") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["text"] | {"decisions": None} == {
        "id": "test-depot-1", "revision": 2, "change": shown["text"]["change"], "text": "The quick brown fox.\n", "kind": None, "content": None, "content_version": 1, "length": 21, "licenses": ["CC0-1.0"],
        "notice": "test-depot-1", "title": "Fox", "authors": ["Ann"], "tags": ["en"], "address": "https://example.org/fox", "scores": None,
        "decisions": None, "libraries": ["test-depot-1", "test-depot-2"]}
    assert [(one["status"], one["name"], one["email"]) for one in shown["decisions"]] == [
        ("under review", "Ada Lovelace", "ada@example.org"), ("admitted", "Ada Lovelace", "ada@example.org")]
    assert [(one["name"], one["locked_at"] is not None) for one in shown["libraries"]] == [
        ("first model", True), ("next model", False)]

    assert depot_command("sql", "SELECT id, kind, libraries FROM texts ORDER BY id") == 0
    assert printed_rows(capsys) == [{"id": "test-depot-1", "kind": None, "libraries": '["test-depot-1","test-depot-2"]'},
                                    {"id": "test-depot-2", "kind": "probe", "libraries": "[]"}]  # the probe is still under review

    assert depot_command("dataset", "export", "first") == 0
    remote = json.loads(capsys.readouterr().out)
    assert remote == {"name": "first", "tokenizer": "classics",
                      "libraries": [{"depot": None, "library": "test-depot-1", "change": Depot.open(depot).change() - 1}],
                      "shares": [{"tag": "en", "share": 1.0}]}
    (tmp_path / "remote.json").write_text(json.dumps({**remote, "name": "again"}), encoding="utf-8")
    assert depot_command("dataset", "import", str(tmp_path / "remote.json")) == 0
    assert Depot.open(depot).dataset("again")["libraries"] == remote["libraries"]


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


def test_a_problem_is_reported_on_a_locked_library_and_a_fork_can_leave_texts_out(depot, tmp_path, capsys):
    opened = Depot.open(depot)
    opened.add([NewText(text="good", licenses=("CC0-1.0",), tags=("a",)),
                NewText(text="bad", licenses=("CC0-1.0",), tags=("b",))], _declaration())
    opened.decide("admitted", "true", _declaration())
    assert depot_command("library", "new", "first") == 0
    assert depot_command("library", "put", "test-depot-1", "true") == 0
    assert depot_command("library", "lock", "test-depot-1") == 0

    assert depot_command("library", "report", "test-depot-1", "not in English", "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'b')", *ME) == 0
    assert depot_command("library", "report", "test-depot-1", "too small", *ME) == 0
    assert depot_command("library", "fork", "test-depot-1", "cleaner", "--without", "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'b')") == 0

    capsys.readouterr()
    assert depot_command("sql", "SELECT id, problems FROM libraries ORDER BY id") == 0
    first, fork = printed_rows(capsys)
    assert json.loads(first["problems"]) == [{"texts": ["test-depot-2"], "problem": "not in English", "decision": "test-depot-3"},
                                 {"texts": [], "problem": "too small", "decision": "test-depot-4"}]
    assert fork["problems"] is None or json.loads(fork["problems"]) == []
    assert opened.sql("SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = "
                      "'test-depot-2')") == [{"id": "test-depot-1"}]
    assert depot_command("library", "report", "test-depot-1", "x", "id = 99", *ME) == 2  # no such text in the library

    assert depot_command("library", "take-out", "test-depot-1", "true") == 2  # locked: the way out is a fork
    assert depot_command("library", "take-out", "test-depot-2", "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'a')") == 0
    assert opened.sql("SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = "
                      "'test-depot-2')") == []
    assert opened.show("test-depot-1")["status"] == "admitted"


def test_a_tokenizer_recipe_trains_its_vocabulary_on_its_datasets_texts_and_locks_it(depot, tmp_path, monkeypatch,
                                                                                    capsys):
    store_folder = tmp_path / "store"
    monkeypatch.setenv("WPT_STORE", store_folder.as_uri())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    opened = Depot.open(depot)
    opened.add([NewText(text="the quick brown fox jumps over the lazy dog", licenses=("CC0-1.0",)),
                NewText(text="pack my box with five dozen liquor jugs", licenses=("CC0-1.0",)),
                NewText(kind="probe", content=json.dumps(PROBE), licenses=("CC0-1.0",)),
                NewText(text="x" * 5000, licenses=("CC0-1.0",))], _declaration())
    opened.decide("admitted", "true", _declaration())
    library = opened.new_library("all")
    opened.put(library, "true")
    opened.new_dataset("first", "tiny", [(None, library)])
    tokenizers = tmp_path / "data" / "recipes" / "tokenizers"
    tokenizers.mkdir(parents=True)
    recipe = {"version": 1, "depot": "", "dataset": "", "sample_bytes": 200, "vocab_size": 560}
    (tokenizers / "tiny.json").write_text(json.dumps(recipe), encoding="utf-8")

    assert depot_command("tokenizer", "tiny") == 2  # a person has not named the depot and the dataset yet
    assert "InvalidRecipe" in capsys.readouterr().err
    recipe = {**recipe, "depot": depot, "dataset": "first"}
    (tokenizers / "tiny.json").write_text(json.dumps(recipe), encoding="utf-8")
    assert depot_command("tokenizer", "tiny") == 0

    lock = Recipes(tmp_path).generated("lock", "tiny")
    # the two short plain texts fit the 200 bytes, the long one is passed over, the item is no sample
    assert lock["recipe"] == recipe
    assert sorted(lock["texts"], key=lambda one: one["id"]) == [{"depot": depot, "id": "test-depot-1"}, {"depot": depot, "id": "test-depot-2"}]
    store = open_store(store_folder.as_uri())
    assert 513 < Tokenizer.from_bytes(store.get_bytes(lock["vocabulary_sha256"])).vocab_size <= 560  # merges learned
    capsys.readouterr()
    assert depot_command("tokenizer", "tiny") == 2  # locked at its version: never trained again
    assert bare(capsys.readouterr().err).startswith("TokenizerLocked:")
    assert depot_command("build", "first") == 0  # the build reads the vocabulary the lock names


def test_a_clone_follows_its_upstream_over_its_web_app_and_pulls_what_it_added_and_refuses_changing_it(
        depot, tmp_path, monkeypatch, capsys):
    server = make_server(routes(Depot.open(depot)), tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    clone = str(tmp_path / "clone.sqlite")
    try:
        assert depot_command("clone", address, clone, "--name", "test-clone", "--maintainer", "Grace Hopper",
                             "--takedown-contact", "grace@example.org") == 0
        Depot.open(depot).add([NewText(text="added upstream later", licenses=("CC0-1.0",))], _declaration())
        monkeypatch.setenv("WPT_DEPOT", clone)
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

    assert "restored from the store's backups of test-depot, at its change 4" in bare(capsys.readouterr().out)
    assert Depot.open(str(tmp_path / "restored.sqlite")).sql("SELECT text FROM texts ORDER BY id") == [
        {"text": "kept"}, {"text": "later"}]
    assert Depot.open(str(tmp_path / "earlier.sqlite")).sql("SELECT text FROM texts") == [{"text": "kept"}]
    assert depot_command("restore", str(tmp_path / "restored.sqlite")) == 2  # it never overwrites


def test_a_dataset_builds_into_a_corpus_with_its_tokenizer_recipes_locked_vocabulary(depot, tmp_path, monkeypatch,
                                                                                      capsys):
    # the build reads the vocabulary from the store, by the sha256 the recipe's lock names
    store_folder = tmp_path / "store"
    monkeypatch.setenv("WPT_STORE", store_folder.as_uri())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    opened = Depot.open(depot)
    opened.add([NewText(text="the quick brown fox", licenses=("CC0-1.0",), title="Fox")], _declaration())
    opened.decide("admitted", "true", _declaration())
    library = opened.new_library("all")
    opened.put(library, "true")
    opened.new_dataset("first", "tiny", [(None, library)])

    assert depot_command("build", "first") == 2  # the repository has no tokenizer recipe tiny yet
    assert "NotFound" in capsys.readouterr().err
    tokenizers = tmp_path / "data" / "recipes" / "tokenizers"
    tokenizers.mkdir(parents=True)
    recipe = {"version": 1, "depot": depot, "dataset": "first", "sample_bytes": 1000, "vocab_size": 560}
    (tokenizers / "tiny.json").write_text(json.dumps(recipe), encoding="utf-8")
    assert depot_command("build", "first") == 2
    assert bare(capsys.readouterr().err) == "NoVocabulary: the tokenizer recipe tiny has no lock of its version 1\n"

    tokenizer = Tokenizer.train([b"the quick brown fox"] * 4, vocab_size=560)
    store = open_store(store_folder.as_uri())
    vocabulary = store.put_bytes(tokenizer.to_bytes())
    Recipes(tmp_path).write_generated("lock", "tiny", {"recipe": recipe, "texts": [{"depot": depot, "id": "test-depot-1"}],
                                                       "vocabulary_sha256": vocabulary})
    assert depot_command("build", "first") == 0
    corpus = bare(capsys.readouterr().out).splitlines()[-1].removeprefix("corpus ")
    [record] = [store.get_json(f"corpora/{folder}/corpus.json") for folder in _corpora(store_folder)]
    assert record["id"] == corpus
    assert [(one["id"], one["title"]) for one in read_table(store, record, "texts").to_pylist()] == [("test-depot-1", "Fox")]


def _corpora(store_folder):
    return [path.name for path in (store_folder / "corpora").iterdir()]


def _declaration():
    return Declaration("Ada Lovelace", "ada@example.org", True)


def test_a_texts_credits_a_librarys_name_a_dataset_and_the_depots_facts_change_from_the_commands(depot, tmp_path,
                                                                                                 capsys):
    opened = Depot.open(depot)
    [text] = opened.add([NewText(text="The fox.", licenses=("CC0-1.0",), title="Fox", tags=("en",))], _declaration())
    opened.decide("admitted", "true", _declaration())
    assert depot_command("library", "new", "first") == 0
    assert depot_command("library", "put", "test-depot-1", "true") == 0
    assert depot_command("library", "lock", "test-depot-1") == 0
    assert depot_command("library", "new", "second") == 0
    assert depot_command("dataset", "new", "mix", "--tokenizer", "tok", "--library", "test-depot-1") == 0
    (tmp_path / "notice.txt").write_text("(c) Ann", encoding="utf-8")

    assert depot_command("library", "rename", "test-depot-1", "first model") == 0
    assert depot_command("credits", text, "--title", "The fox", *ME) == 0  # a new revision; the locked library keeps its own
    assert depot_command("dataset", "edit", "mix", "--new-name", "mix two", "--library", "test-depot-1",
                         "--library", "test-depot-2", "--share", "en=1") == 0
    assert depot_command("about", "--takedown-contact", "takedown@example.org") == 0
    with pytest.raises(SystemExit):  # a new title or none, not both
        depot_command("credits", text, "--title", "x", "--clear", "title", *ME)

    shown = Depot.open(depot)
    assert shown.sql("SELECT name FROM libraries ORDER BY id") == [{"name": "first model"}, {"name": "second"}]
    row = shown.dataset("mix two")
    assert [one["library"] for one in row["libraries"]] == ["test-depot-1", "test-depot-2"]
    assert row["shares"] == [{"tag": "en", "share": 1.0}]
    assert shown.about()["takedown_contact"] == "takedown@example.org"
    assert shown.show(text)["text"]["title"] == "The fox"
    assert [one["title"] for one in shown.library_texts("test-depot-1", shown.change())] == ["Fox"]

    [other] = shown.add([NewText(text="A hen.", licenses=("CC0-1.0",), tags=("en",))], _declaration())
    assert depot_command("credits", other, "--title", "Hen", "--tag", "en", "--tag", "birds", "--notice",
                         str(tmp_path / "notice.txt"), *ME) == 0
    after = Depot.open(depot).show(other)
    assert (after["text"]["title"], after["text"]["tags"], after["notice"]) == ("Hen", ["en", "birds"], "(c) Ann")
    assert after["decisions"][-1]["reason"].startswith("title changed from none to Hen; tags changed from en to en, birds")


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
