"""The corpus of an order cut into the portions of its roles
(wpt_corpus.roles), asked of the freight and prepared by wpt-tokens: the
held-out portion first, on HELD_OUT_BYTES bytes with the order's shares;
wpt-tokens writes each portion's files under the folder of its name, by
kind, and lists them by portion; training's view of the delivery names no
file of the held-out portion; two corpora of one snapshot and one set of
shares hold the same held-out texts whatever their vocabulary and sizes,
and wpt eval's bits_per_byte measures two vocabularies on the same bytes;
a job cut short goes on to the same delivery; the order schemas and
checks of wpt-orders, which cannot import wpt_corpus, name the roles and
refuse a held-out portion in too small a corpus as wpt_corpus.roles does;
and wpt build-resources refuses such an order before it asks for
anything."""

import base64
import importlib.util
import io
import json
import math
from importlib.resources import files
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from depot import Declaration, NewText
from freight import api
from freight.job import LEFT_OUT, Job
from wpt import resources
from wpt.resources import main as build_resources
from wpt.eval import BUILT_IN
from wpt_corpus import corpora, programs, roles
from wpt_orders import orders
from wpt_tokenizer import Tokenizer
from wic_utils.lines import bare
import dock_files

ME = Declaration("Ada Writer", "ada@example.org", True)
HELD_OUT = 4000  # the held-out portion's bytes here: a few of the test texts
SHORT = 300  # tokens a token file holds before it closes: each portion in several files
TASK = {"task": {"instructions": "Write hello into out.txt.", "image": "base",
                 "limits": {"max_turns": 4, "max_tokens": 512, "command_seconds": 10}, "needs": []},
        "files": {"notes.txt": {"mode": "100644", "content": base64.b64encode(b"say hello\n").decode("ascii")}},
        "check": {"mode": "100755", "content": base64.b64encode(b"#!/bin/sh\necho 1\n").decode("ascii")}}


def talk(messages: list) -> NewText:
    return NewText(kind="conversation", content=json.dumps({"messages": messages}), licenses=("MIT",), tags=("en",))


@pytest.fixture
def cut(built, monkeypatch):
    """The test depot with a snapshot "talk": its 16 stories, 8 fables (en), 6 stories in German (de), two
    conversations (one quotes a fable) and a task; the held-out portion HELD_OUT bytes. Gives what built gives,
    with vocabulary_key (the test vocabulary's delivery) and data(**changes), an order's data section of "talk"
    with the roles held_out, pretrain and finetune and the shares en 0.8, de 0.2."""
    monkeypatch.setattr(roles, "HELD_OUT_BYTES", HELD_OUT)
    plain = [NewText(text=f"fable {n}: a crow found cheese number {n}. " + "it sang and the fox took it. " * 8,
                     licenses=("MIT",), tags=("en", "fable")) for n in range(8)]
    plain += [NewText(text=f"Geschichte {n}: der schnelle Fuchs springt. " * 10, licenses=("MIT",), tags=("de",))
              for n in range(6)]
    added = built.depot.add(plain, ME)
    built.depot.decide("admitted", f"id IN ({', '.join(repr(one) for one in added)})", ME)
    quoted = {"type": "excerpt", "source": added[0],
              "selector": {"type": "TextQuoteSelector", "exact": "found cheese number 0"}}
    built.snapshot("talk", [
        talk([{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
              {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]),
        talk([{"role": "user", "content": [{"type": "text", "text": "What is it?"}, quoted]},
              {"role": "assistant", "content": [{"type": "text", "text": "A crow."}]}]),
        NewText(kind="task", content=json.dumps(TASK), licenses=("MIT",), tags=("en",))])
    key = f"{api.ADOPTED}{built.vocabulary}"
    api.adopt(key, {"vocabulary.json": built.tokenizer.to_bytes()},
              {"sha256": built.vocabulary, "size": built.tokenizer.vocab_size}, {"by": "a test"}, on_erasure="note",
              dock=built.dock)

    def data(**changes) -> dict:
        return {**built.data("talk", 1500), "roles": ["held_out", "pretrain", "finetune"],
                "shares": [{"tag": "en", "share": 0.8}, {"tag": "de", "share": 0.2}], **changes}

    built.vocabulary_key, built.cut_data = key, data
    return built


def asked(built, tmp_path, request: dict) -> dict:
    """The delivery note of the request, run by the freight, its token files closing every SHORT tokens."""
    request = {**request, "preparation": {**request["preparation"], "settings": {"shard_tokens": SHORT}}}
    where = {"home": str(tmp_path / "freight"), "depot": built.depot.address, "dock": built.dock}
    done = api.wait(api.request(request, **where)["id"], home=where["home"])
    assert done["state"] == "done", done["problem"]
    return done["answer"]


def documents(dock: str, note: dict, path: str) -> list[str]:
    (file,) = [one for one in note["files"] if one["path"] == path]
    return pq.read_table(io.BytesIO(dock_files.fetch(dock, file)), columns=["document"])["document"].to_pylist()


def test_an_order_s_corpus_is_cut_into_its_roles_portions_each_in_its_own_files(cut, tmp_path):
    request = resources.corpus_request(cut.cut_data(), cut.vocabulary_key)
    portions = request["selection"]["portions"]
    assert [one["name"] for one in portions] == ["held_out", "pretrain", "finetune"]
    assert portions[0] == {"name": "held_out", "kinds": ["plain"], "shares": cut.cut_data()["shares"],
                           "size": {"unit": "bytes", "amount": HELD_OUT}}
    assert portions[1]["shares"] == portions[0]["shares"] and portions[1]["size"] == {"unit": "tokens", "amount": 1500}
    assert (portions[2]["kinds"], portions[2]["size"]) == (["conversation", "task"], None)
    note = asked(cut, tmp_path, request)
    assert note["preparation"]["version"] == programs.PORTIONED_VERSION  # not the files VERSION gives
    made, rows = note["result"]["portions"], api.items(note["key"], dock=cut.dock)
    assert list(made) == ["held_out", "pretrain", "finetune"]
    for name, portion in made.items():
        taken = [row for row in rows if row["portion"] == name]
        assert {row["kind"] for row in taken} <= set(roles.KINDS[name])
        for kind, files in portion["shards"].items():
            # the token files of a portion and a kind hold its items of that kind, in order, from the token 0
            assert all(one["path"].startswith(f"{name}/{kind}-") for one in files)
            held = [document for one in files for document in documents(cut.dock, note, one["path"])]
            mine = [row for row in taken if row["kind"] == kind]
            assert held == [row["id"] if row["piece"] is None else f"{row['id']}#{row['piece']}" for row in mine]
            assert [row["tokens"] for row in mine[:1]] in ([], [0])
    assert len(made["held_out"]["shards"]["plain"]) >= 1 and len(made["pretrain"]["shards"]["plain"]) > 1
    assert made["finetune"]["tasks"] == {"path": "finetune/tasks.jsonl", "count": 1}
    assert made["held_out"]["tasks"] is None and made["finetune"]["shards"]["plain"] == []
    # the conversation that quotes a fable takes it in its own portion only: the fable is plain, so it is left out
    finetune = note["portions"][2]
    assert finetune["items"]["kinds"] == {"conversation": 1, "task": 1} and finetune["left_out"][LEFT_OUT] == 1
    assert note["portions"][0]["bytes"] <= HELD_OUT

    vocabulary = api.delivery(cut.vocabulary_key, dock=cut.dock)
    record = resources.corpus_record(note, vocabulary)
    # training is told of the held-out portion's token files alone, which its validation reads, never of its rows
    held_out = {one["key"] for one in note["files"] if one["path"].startswith("held_out/")}
    plain = {one["key"] for one in note["files"] if one["path"].startswith("held_out/plain-")}
    assert plain and {one["key"] for one in record["held_out"]["shards"]} == plain
    assert record["held_out"]["total_tokens"] == made["held_out"]["total_tokens"]
    without = json.dumps({key: value for key, value in record.items() if key != "held_out"})
    assert not any(key in without for key in held_out)
    assert record["trains_on"] == ["pretrain", "finetune"]
    assert [one["token_sha256"] for one in record["shards"]["plain"]] == [
        one["token_sha256"] for one in made["pretrain"]["shards"]["plain"]]
    assert [one["token_sha256"] for one in record["shards"]["conversation"]] == [
        one["token_sha256"] for one in made["finetune"]["shards"]["conversation"]]
    assert record["tasks"]["count"] == 1 and record["tasks"]["file"]["path"] == "finetune/tasks.jsonl"
    assert record["total_tokens"] == made["pretrain"]["total_tokens"] and record["max_tokens"] == 1500
    assert record["fingerprints"] == {"plain": made["pretrain"]["fingerprints"]["plain"],
                                      "conversation": made["finetune"]["fingerprints"]["conversation"],
                                      "task": made["finetune"]["fingerprints"]["task"]}
    texts = corpora.rows(cut.dock, record, "texts")
    assert {row["portion"] for row in texts} == {"pretrain", "finetune"}
    assert record["texts"] == len({(row["id"]) for row in texts})
    found = resources.held_out(note)
    assert [one["key"] for one in found["shards"]] == [
        one["key"] for one in note["files"] if one["path"].startswith("held_out/plain-")]
    assert found["texts"] == sum(1 for row in rows if row["portion"] == "held_out")
    assert "held_out" in resources.label(note) and "finetune" in resources.label(note)
    # a delivery cut into portions by a wpt-tokens that wrote them into the same files is refused, not misread
    flat = {**note, "preparation": {**note["preparation"], "version": programs.VERSION},
            "result": {key: value for key, value in note["result"].items() if key != "portions"}}
    with pytest.raises(ValueError, match=f"wpt-tokens of version {programs.VERSION}, which wrote every portion"):
        resources.corpus_record(flat, vocabulary)


def test_two_corpora_of_one_snapshot_and_shares_hold_the_same_held_out_texts_whatever_their_sizes_and_vocabulary(
        cut, tmp_path):
    small = asked(cut, tmp_path, resources.corpus_request(cut.cut_data(roles=["held_out", "pretrain"]),
                                                          cut.vocabulary_key))
    large = asked(cut, tmp_path, resources.corpus_request(cut.cut_data(total_tokens=4000, finetune_tokens=100),
                                                          cut.vocabulary_key))
    other = Tokenizer.train([f"fable {n}: der Fuchs. ".encode() * 40 for n in range(30)], vocab_size=600)
    key = f"{api.ADOPTED}{'6' * 64}"
    api.adopt(key, {"vocabulary.json": other.to_bytes()}, {"sha256": "6" * 64, "size": other.vocab_size},
              {"by": "a test"}, on_erasure="note", dock=cut.dock)
    elsewhere = asked(cut, tmp_path, resources.corpus_request(cut.cut_data(), key))
    held = [[(row["id"], row["piece"]) for row in api.items(note["key"], dock=cut.dock) if row["portion"] == "held_out"]
            for note in (small, large, elsewhere)]
    assert held[0] and held[0] == held[1] == held[2]
    hashes = [[one["token_sha256"] for one in note["result"]["portions"]["held_out"]["shards"]["plain"]]
              for note in (small, large, elsewhere)]
    assert hashes[0] == hashes[1] != hashes[2]  # the same tokens with the same vocabulary
    assert len({small["key"], large["key"], elsewhere["key"]}) == 3


def test_bits_per_byte_measures_two_vocabularies_on_the_same_bytes(cut, tmp_path):
    other = Tokenizer.train([f"Geschichte {n}: der schnelle Fuchs. ".encode() * 40 for n in range(30)], vocab_size=700)
    key = f"{api.ADOPTED}{'7' * 64}"
    api.adopt(key, {"vocabulary.json": other.to_bytes()}, {"sha256": "7" * 64, "size": other.vocab_size},
              {"by": "a test"}, on_erasure="note", dock=cut.dock)
    spec = importlib.util.spec_from_file_location("bits_per_byte_under_test", BUILT_IN / "bits_per_byte.py")
    bits_per_byte = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bits_per_byte)

    def one_nat_each(windows):
        return [np.ones(len(window) - 1, dtype=np.float32) for window in windows]

    found = []
    for tokenizer, vocabulary in ((cut.tokenizer, cut.vocabulary_key), (other, key)):
        note = asked(cut, tmp_path, resources.corpus_request(cut.cut_data(total_tokens=2000), vocabulary))
        context = SimpleNamespace(folder=tmp_path / "model", dock=cut.dock, tokenizer=tokenizer,
                                  provenance={"deliveries": {"corpus": {"key": note["key"]}}},
                                  model=SimpleNamespace(config=SimpleNamespace(block_size=16), losses=one_nat_each))
        found.append(bits_per_byte.run(context).data)
    # every byte of the same held-out texts, read in tokens of two vocabularies
    assert found[0]["bytes"] == found[1]["bytes"] == resources.held_out(note)["bytes"] > 0
    assert found[0]["texts"] == found[1]["texts"] > 1 and found[0]["tokens"] != found[1]["tokens"]
    assert {pool: one["bytes"] for pool, one in found[0]["by_share"].items()} == {
        pool: one["bytes"] for pool, one in found[1]["by_share"].items()}
    for data in found:  # one nat a token
        assert data["bits_per_byte"] == pytest.approx(data["tokens"] / math.log(2) / data["bytes"])


@pytest.mark.parametrize("stop", [1, 3, 5])
def test_a_job_of_portions_cut_short_goes_on_to_the_same_files(cut, tmp_path, stop):
    request = resources.corpus_request(cut.cut_data(), cut.vocabulary_key)
    request = {**request, "preparation": {**request["preparation"], "settings": {"shard_tokens": SHORT}}}
    docks = {name: (tmp_path / name).as_uri() for name in ("whole", "cut")}
    for dock in docks.values():  # the vocabulary, on each dock
        api.adopt(cut.vocabulary_key, {"vocabulary.json": cut.tokenizer.to_bytes()},
                  {"sha256": cut.vocabulary, "size": cut.tokenizer.vocab_size}, {"by": "a test"}, on_erasure="note",
                  dock=dock)
    whole = Job(request, cut.depot.address, docks["whole"], tmp_path / "first").run()
    assert len(whole["files"]) > stop
    saved = []

    class Stop(Exception):
        pass

    def save(state):
        saved.append(state)
        if len(saved) == stop:
            raise Stop()
    dock = docks["cut"]
    with pytest.raises(Stop):
        Job(request, cut.depot.address, dock, tmp_path / "work", save=save).run()
    note = Job(request, cut.depot.address, dock, tmp_path / "work", checkpoint=saved[-1]).run()
    assert (note["id"], note["result"]) == (whole["id"], whole["result"])
    assert [(one["path"], one["sha256"]) for one in note["files"]] == [
        (one["path"], one["sha256"]) for one in whole["files"]]


def _roles_field(schema):
    """The field `roles` of an order schema, wherever it lies."""
    if isinstance(schema, dict):
        if isinstance(schema.get("properties"), dict) and "roles" in schema["properties"]:
            return schema["properties"]["roles"]
        schema = list(schema.values())
    for one in schema if isinstance(schema, list) else []:
        found = _roles_field(one)
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("schema", ["model.schema.json", "series.schema.json"])
def test_an_order_schema_names_the_roles_their_order_pretrain_and_the_default_as_wpt_corpus_does(schema):
    field = _roles_field(json.loads((files("wpt_orders") / "schemas" / schema).read_text(encoding="utf-8")))
    assert field["items"]["enum"] == list(roles.ROLES) and field["contains"] == {"const": roles.PRETRAIN}
    assert f"left out: {' and '.join(roles.DEFAULT)}" in field["description"]


@pytest.mark.parametrize("named", [None, ["pretrain"], ["held_out", "pretrain"], ["pretrain", "finetune"],
                                   ["held_out", "pretrain", "finetune"]])
def test_the_order_check_refuses_a_size_of_the_finetune_portion_where_wpt_corpus_does(named):
    data = {"finetune_tokens": 10, **({"roles": named} if named else {})}
    try:
        orders._finetune("an order", data, "data.")
        checked = True
    except orders.InvalidOrder:
        checked = False
    try:
        roles.names(data)
        cut = True
    except roles.Refused:
        cut = False
    assert checked == cut == ("finetune" in (named or roles.DEFAULT))


@pytest.mark.parametrize("named", [None, ["pretrain"], ["held_out", "pretrain"]])
@pytest.mark.parametrize("total", [roles.HELD_OUT_MIN_TOKENS - 1, roles.HELD_OUT_MIN_TOKENS])
def test_the_order_check_refuses_a_held_out_portion_in_too_small_a_corpus_where_wpt_corpus_does(named, total):
    # the real sizes, which the tests that build corpora set smaller
    assert orders.HELD_OUT_MIN_TOKENS == roles.HELD_OUT_MIN_TOKENS == roles.HELD_OUT_BYTES == 64 * 1024 * 1024
    data = {"snapshot": "s", "total_tokens": total, **({"roles": named} if named else {})}
    try:
        orders._held_out("an order", data, "data.")
        checked = True
    except orders.InvalidOrder:
        checked = False
    try:
        roles.names(data)
        cut = True
    except roles.Refused:
        cut = False
    assert checked == cut == ("held_out" not in (named or roles.DEFAULT) or total >= roles.HELD_OUT_MIN_TOKENS)


def test_wpt_build_resources_refuses_a_held_out_portion_its_corpus_is_too_small_for_before_any_request(
        built, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FREIGHT_HOME", str(tmp_path / "freight"))
    monkeypatch.setattr(roles, "HELD_OUT_MIN_TOKENS", 10**6)  # the order "d" asks for a few thousand tokens
    assert build_resources(["d"]) == 1
    said = bare(capsys.readouterr().err)
    assert "too small for a held-out portion" in said and 'name the roles without held_out, such as ["pretrain"]' in said
    assert api.jobs() == []
