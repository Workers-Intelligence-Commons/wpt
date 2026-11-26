"""The corpus of an order cut into the portions of its roles
(wpt_corpus.roles), asked of the freight and prepared by wpt-tokens: the
held-out portion first, on HELD_OUT_BYTES bytes with the order's shares;
wpt-tokens writes each portion's files under the folder of its name, by
kind, and lists them by portion; training's view of the delivery names no
file of the held-out portion; two corpora of one snapshot and one set of
shares hold the same held-out texts whatever their vocabulary and sizes;
and a job cut short goes on to the same delivery."""

import base64
import io
import json

import pyarrow.parquet as pq
import pytest

from depot import Declaration, NewText
from freight import api
from freight.job import LEFT_OUT, Job
from wpt import resources
from wpt_corpus import corpora, roles
from wpt_tokenizer import Tokenizer
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
    held_out = {one["key"] for one in note["files"] if one["path"].startswith("held_out/")}
    assert held_out and not any(key in json.dumps(record) for key in held_out)  # training is never told of it
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
