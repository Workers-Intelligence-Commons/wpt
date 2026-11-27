"""An order's vocabulary and corpus, asked of the freight by `wpt
build-resources` and trained on by `wpt train` (wpt.resources): the
vocabulary named by its sha256, an adopted delivery of the dock, or
learned for the order; the keys worked out from the order alone; what the
model folder keeps of both deliveries; a corpus delivery given by its id
or key, as `wpt manager` gives it; a run of `wpt build-resources` stopped
and run again; a vocabulary a depot's tokenizer recipe trained, found by
its sha256 as the dock holds it; an erasure, written into the
vocabulary's note, which stays, while the corpus is made again; and the corpora the depot built
before the freight, which only the freight's check marks stale."""

import hashlib
import json

import pytest

from freight import api
from wpt import resources
from wpt.resources import main as build_resources
from wpt.train import main as train
from depot import Declaration
from wpt_model import load_tokenizer
from wpt_corpus import corpora as readers
from wic_utils.lines import bare

ME = Declaration("Ada Writer", "ada@example.org", True)


def order(tmp_path, name="d") -> dict:
    return json.loads((tmp_path / "data" / "orders" / f"{name}.json").read_text())


def learn(tmp_path, sample_bytes=10**6, name="d") -> dict:
    """Rewrite the family so it learns its vocabulary, of the size it names, on sample_bytes bytes: its data."""
    path = tmp_path / "data" / "orders" / f"{name}.json"
    document = json.loads(path.read_text())
    document["data"]["vocabulary"] = {"size": document["data"]["vocabulary"]["size"], "sample_bytes": sample_bytes}
    path.write_text(json.dumps(document))
    return document["data"]


def printed_notes(out: str) -> dict:
    return json.loads(out[out.index("\n{\n") + 1:])  # printed last, after the lines of the jobs


@pytest.mark.parametrize("trained", ["named", "learned"], indirect=True)
def test_a_model_trains_from_an_order_whose_vocabulary_is_named_by_its_sha256_or_learned(trained, tmp_path):
    data = order(tmp_path)["data"]
    vocabulary, corpus = trained.notes["vocabulary"], trained.notes["corpus"]
    # the keys come from the order alone (and, for a vocabulary named by its sha256, the dock's notes)
    assert vocabulary["key"] == resources.vocabulary_key(data)
    assert corpus["key"] == resources.corpus_key(data, vocabulary["key"])
    assert corpus["preparation"]["inputs"] == {"vocabulary": vocabulary["key"]}
    rows = api.items(vocabulary["key"])
    if "sha256" in data["vocabulary"]:  # the depot's vocabulary, adopted, with the texts it read in order
        assert vocabulary["key"] == f"adopted-{trained.vocabulary}" and vocabulary["on_erasure"] == "note"
        assert (vocabulary["adopted"]["recipe"], vocabulary["adopted"]["version"]) == ("t", 1)
        listed = [row["id"] for row in trained.depot.sql("SELECT id FROM texts ORDER BY id")]
        assert [row["id"] for row in rows] == listed and {row["depot"] for row in rows} == {trained.depot.name()}
    else:  # learned from every story, as the order's sample of bytes allows
        assert vocabulary["preparation"]["name"] == "wpt-vocabulary" and len(rows) == 16
        assert vocabulary["request"]["selection"]["size"] == {"unit": "bytes", "amount": data["vocabulary"]["sample_bytes"]}
    # the model reads the vocabulary of the delivery, and its folder names both deliveries
    assert hashlib.sha256(load_tokenizer(trained.folder)).hexdigest() == vocabulary["result"]["sha256"]
    provenance = json.loads((trained.folder / "provenance.json").read_text())
    assert provenance["data"] == data
    assert {role: (one["key"], one["id"]) for role, one in provenance["deliveries"].items()} == {
        "vocabulary": (vocabulary["key"], vocabulary["id"]), "corpus": (corpus["key"], corpus["id"])}
    assert (provenance["corpus"]["id"], provenance["corpus"]["delivery"]) == (corpus["id"], corpus["key"])
    # every story but the two of the held-out portion (conftest.HELD_OUT), which no training reads
    assert len((trained.folder / "CREDITS").read_text().split("\n\n", 1)[0].splitlines()) == 14
    assert f"delivery `{corpus['key']}`" in (trained.folder / "README.md").read_text()


def test_an_erasure_is_written_into_the_vocabulary_s_note_and_the_corpus_is_made_again(trained, tmp_path, capsys):
    # the merges learned from an erased text stay (docs/tokenizer.md): the vocabulary stays fresh and every
    # order naming it keeps working, while the corpus that holds the text is made again without the passage
    vocabulary, corpus = trained.notes["vocabulary"], trained.notes["corpus"]
    text = api.items(corpus["key"])[0]["id"]
    erased = trained.depot.ask_erasure("a passage", [(text, 0, 5)], ME)
    trained.depot.decide_erasure(erased, "admitted", ME)
    checked = api.check()
    assert checked["erasures"] == [erased] and checked["stale"] == [corpus["key"]]
    assert checked["noted"] == [vocabulary["key"]]
    assert api.delivery(vocabulary["key"])["erasures_since"] == [erased]
    capsys.readouterr()

    with pytest.raises(SystemExit) as refused:
        train(["d:s", "--out", "again"])
    said = bare(capsys.readouterr().err)
    assert refused.value.code == 2 and f"corpus delivery {corpus['key']} of the order is stale" in said
    assert "wpt build-resources" in said

    assert build_resources(["d:s"]) == 0
    notes = printed_notes(capsys.readouterr().out)
    assert notes["vocabulary"]["id"] == vocabulary["id"]  # the same vocabulary, unchanged
    assert notes["corpus"]["key"] == corpus["key"] and notes["corpus"]["id"] != corpus["id"]
    assert "stale" not in notes["corpus"]
    train(["d:s", "--out", "again"])
    assert json.loads((tmp_path / "again" / "provenance.json").read_text())["corpus"]["id"] == notes["corpus"]["id"]


def test_only_the_freight_s_check_marks_stale_a_corpus_the_depot_built_before_it(built, tmp_path):
    # an erasure the depot carried out leaves the corpus under corpora/ fresh until the freight checks, which marks
    # it once (docs/erasure.md, "What it covers")
    text = readers.rows(built.dock, readers.find(built.dock, built.corpus_id), "texts")[0]["id"]
    erased = built.depot.ask_erasure("a passage", [(text, 0, 5)], ME)
    built.depot.decide_erasure(erased, "admitted", ME)
    assert "stale" not in readers.find(built.dock, built.corpus_id)

    home = str(tmp_path / "freight")
    checked = api.check(home=home)
    assert checked["erasures"] == [erased] and built.corpus_id in checked["corpora"]
    assert readers.find(built.dock, built.corpus_id)["stale"] == {"erasures": [erased]}
    assert api.check(home=home)["corpora"] == []  # each erasure is checked once


def test_an_order_naming_a_vocabulary_the_dock_does_not_hold_makes_no_request(built, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FREIGHT_HOME", str(tmp_path / "freight"))
    assert build_resources(["d:s"]) == 1
    said = bare(capsys.readouterr().err)
    assert f"the dock holds no vocabulary {built.vocabulary}" in said and "let the order learn" in said
    assert api.jobs() == []
    with pytest.raises(SystemExit) as refused:  # the corpus the depot built of every token is not of its size
        train(["d:s", "--out", "model"])
    said = bare(capsys.readouterr().err)
    assert refused.value.code == 2 and f"the dock holds no vocabulary {built.vocabulary}" in said
    assert "run `wpt build-resources FAMILY:SIZE`" in said


@pytest.mark.parametrize("trained", ["learned"], indirect=True)
def test_wpt_train_given_a_corpus_delivery_by_its_id_or_key_trains_on_it_when_it_is_the_order_s(trained, tmp_path,
                                                                                                  capsys):
    # wpt manager hands the corpus's id to wpt train on the machine it rents
    corpus = trained.notes["corpus"]
    for given, out in ((corpus["id"][:16], "by-id"), (corpus["key"][:16], "by-key")):
        train(["d:s", given, "--out", out])
        assert json.loads((tmp_path / out / "provenance.json").read_text())["corpus"]["id"] == corpus["id"]
    capsys.readouterr()
    learn(tmp_path, sample_bytes=10**5)  # another vocabulary: the corpus is no longer the order's
    with pytest.raises(SystemExit) as refused:
        train(["d:s", corpus["id"][:16], "--out", "other"])
    said = bare(capsys.readouterr().err)
    assert refused.value.code == 2 and "with portions, shares or vocabulary other than the order's" in said


def test_wpt_build_resources_run_again_waits_for_the_job_a_stopped_run_left(built, tmp_path, monkeypatch, capsys):
    # a run stopped while its job was queued or running leaves that job; the next run waits for it, and the job
    # goes on from its checkpoint, instead of being refused for a key that has a job already
    monkeypatch.setenv("FREIGHT_HOME", str(tmp_path / "freight"))
    data = learn(tmp_path)
    left = api.request(resources.vocabulary_request(data))
    assert left["state"] == "queued" and left["key"] == resources.vocabulary_key(data)
    assert build_resources(["d:s"]) == 0
    notes = printed_notes(capsys.readouterr().out)
    assert api.job(left["id"])["state"] == "done" and api.job(left["id"])["answer"]["id"] == notes["vocabulary"]["id"]
    assert [job["id"] for job in api.jobs() if job["key"] == notes["vocabulary"]["key"]] == [left["id"]]


def test_a_vocabulary_a_depot_s_tokenizer_recipe_trained_is_found_by_its_sha256(built, tmp_path):
    # the delivery adopted-<sha256> as the docks hold it for each vocabulary a depot's tokenizer recipes trained:
    # its note's `adopted` what the depot recorded of it, its items the texts it read, an erasure noted
    listed = [row["id"] for row in built.depot.sql("SELECT id FROM texts ORDER BY id")]
    change = built.depot.snapshot(built.names["d"])["at_change"]
    sample = {"snapshot": built.depot.snapshot(built.names["d"])["id"], "kinds": ["plain"]}
    about = {"by": "wpt adopt-vocabularies", "depot": built.depot.name(), "recipe": "t", "version": 1,
             "dataset": "", "sample": sample, "sample_bytes": 10**6, "texts": len(listed), "bytes": None,
             "made_at": "2026-01-01T00:00:00+00:00"}
    result = {"sha256": built.vocabulary, "size": built.tokenizer.vocab_size,
              "tokenizer_version": json.loads(built.tokenizer.to_bytes())["tokenizer_version"]}
    key = f"{api.ADOPTED}{built.vocabulary}"
    api.adopt(key, {"vocabulary.json": built.tokenizer.to_bytes()}, result, about, on_erasure="note",
              items=[{"depot": built.depot.name(), "id": one, "change": change} for one in listed])
    data = order(tmp_path)["data"]
    assert data["vocabulary"] == {"sha256": built.vocabulary, "size": built.tokenizer.vocab_size}
    assert resources.vocabulary_key(data) == key
    note = api.delivery(key)
    assert resources.label(note) == f"vocabulary {built.tokenizer.vocab_size:,} · adopted · tokenizer recipe t version 1"
    assert resources.corpus_request(data, key)["preparation"]["inputs"] == {"vocabulary": key}
