"""A corpus asked of the freight, its program wpt-tokens preparing it,
against the corpus `wpt depot build` builds (wpt_depot.corpus.build) from
the same snapshot, data section and vocabulary: the same token files, by
their token hashes, and the same rows of the texts table."""

import base64
import hashlib
import json

import pyarrow.parquet as pq
import pytest

from freight import api
from wpt_depot import Declaration, NewText
from wpt_depot.corpora import read_table
from wpt_depot.corpus import build
from wpt_store.names import File

ME = Declaration("Ada Writer", "ada@example.org", True)
TASK = {"task": {"instructions": "Write hello into out.txt.", "image": "base",
                 "limits": {"max_turns": 4, "max_tokens": 512, "command_seconds": 10}, "needs": []},
        "files": {"notes.txt": {"mode": "100644", "content": base64.b64encode(b"say hello\n").decode("ascii")}},
        "check": {"mode": "100755", "content": base64.b64encode(b"#!/bin/sh\necho 1\n").decode("ascii")}}
SHORT = 600  # tokens a token file holds before it closes, on both sides: a corpus of several files


def item(kind, content, *tags):
    return NewText(kind=kind, content=json.dumps(content), licenses=("MIT",), tags=tags or ("en",))


def quoting(source, exact):
    """A conversation whose question quotes exact of the text source (docs/chat-format.md, "Excerpts")."""
    return item("conversation", {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                     {"type": "excerpt", "source": source,
                                      "selector": {"type": "TextQuoteSelector", "exact": exact}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A fox."}]}]})


def rich(built) -> str:
    """The snapshot "rich" of the test depot: its 16 stories (tagged en), plain texts of other tags (en with a
    second tag, de, a tag no share names), a text longer than a piece, conversations that quote a text, a text no
    share takes or a piece, a conversation that quotes nothing, a task and a probe."""
    line = "the quick brown fox jumps over the lazy dog\n"
    long_text = "".join(f"=== src/file{n}.c\n" + line * 3000 for n in range(10))  # past pieces.LENGTH
    plain = [NewText(text=f"fable {n}: a crow found cheese number {n}. " + "it sang and the fox took it. " * 6,
                     licenses=("MIT",), tags=("en", "fable")) for n in range(6)]
    plain += [NewText(text=f"news {n}: the river rose by {n} metres. " * 5, licenses=("MIT",), tags=("en", "news"))
              for n in range(5)]
    plain += [NewText(text=f"die Geschichte {n}: der Fuchs springt. " * 6, licenses=("MIT",), tags=("de",))
              for n in range(7)]
    plain += [NewText(text="a text no share takes", licenses=("MIT",), tags=("xx",)),
              NewText(text=long_text, licenses=("MIT",), title="A repository", tags=("en", "code"))]
    added = built.depot.add(plain, ME)
    built.depot.decide("admitted", f"id IN ({', '.join(repr(one) for one in added)})", ME)
    fable, nobody, repository = added[0], added[-2], added[-1]
    items = [quoting(fable, "found cheese number 0"), quoting(nobody, "no share"),
             quoting(repository, "=== src/file7.c"), item("conversation", {"messages": [
                 {"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}),
             item("task", TASK), item("probe", {"question": "q?", "patterns": ["x"]})]
    built.snapshot("rich", items)
    return "rich"


def section(built, name, kinds, shares, size) -> dict:
    snapshot = built.depot.snapshot(name)
    return {"depot": built.depot.name(), "snapshot": name, "change": snapshot["at_change"],
            "digest": snapshot["digest"], "kinds": kinds, "shares": [{"tag": tag, "share": share}
                                                                     for tag, share in shares],
            "total_tokens": size, "vocabulary": {"sha256": built.vocabulary, "size": built.tokenizer.vocab_size}}


def vocabulary(built, tmp_path) -> str:
    """The test vocabulary put on the dock as an adopted delivery: its key."""
    path = tmp_path / "vocabulary.json"
    path.write_bytes(built.tokenizer.to_bytes())
    key = f"adopted-{built.vocabulary}"
    api.adopt(key, {"vocabulary.json": str(path)}, {"sha256": built.vocabulary, "size": built.tokenizer.vocab_size},
              {"by": "a test"}, dock=built.store.root.as_uri())
    return key


def asked(built, tmp_path, name, kinds, shares, size, vocabulary_key) -> dict:
    """The delivery note of the corpus request of that snapshot, kinds, shares and size, run by the freight."""
    snapshot = built.depot.snapshot(name)
    request = {"depot": built.depot.name(),
               "snapshot": {"name": name, "change": snapshot["at_change"], "digest": snapshot["digest"]},
               "selection": {"kinds": kinds, "shares": [{"tag": tag, "share": share} for tag, share in shares],
                             "size": None if size is None else {"unit": "tokens", "amount": size}},
               "preparation": {"name": "wpt-tokens", "settings": {"shard_tokens": SHORT},
                               "inputs": {"vocabulary": vocabulary_key}}}
    where = {"home": str(tmp_path / "freight"), "depot": built.depot.address, "dock": built.store.root.as_uri()}
    state = api.request(request, log=lambda *_: None, **where)
    done = api.wait(state["id"], home=where["home"])
    assert done["state"] == "done", done["problem"]
    assert done["answer"]["key"] == api.key(request, done["answer"]["preparation"]["version"])
    return done["answer"]


def same(built, record: dict, note: dict):
    """The corpus record and the delivery note hold the same corpus."""
    store, result = built.store, note["result"]
    for kind in ("plain", "conversation"):
        assert [one["token_sha256"] for one in result["shards"][kind]] == [
            one["token_sha256"] for one in record["shards"][kind]], kind
    built_rows = read_table(store, record, "texts").to_pylist()
    rows = api.items(note["key"], dock=store.root.as_uri())
    assert [{key: value for key, value in row.items() if key != "weight"} for row in rows] == built_rows
    assert result["total_tokens"] == record["total_tokens"] and result["items_tokens"] == record["items_tokens"]
    assert result["fingerprints"] == record["fingerprints"]
    tasks = next((one for one in note["files"] if one["path"] == "tasks.jsonl"), None)
    assert (tasks is None) == ("tasks" not in record)
    if tasks is not None:
        assert tasks["sha256"] == record["tasks"]["file"]["sha256"] and result["tasks"]["count"] == record["tasks"]["count"]
    for table in ("licenses", "notices"):
        assert pq.read_table(store.path(File.of(note["tables"][table]))).to_pylist() == \
            read_table(store, record, table).to_pylist()
    assert (note["items"]["texts"], note["items"]["pieces"], note["items"]["kinds"].get("conversation", 0)) == (
        record["texts"], record["pieces"], record["conversations"])
    assert note["left_out"] == record["left_out"]
    assert note["shares"] == (None if record["shares_tokens"] is None else [
        {"tag": one["tag"], "weight": one["tokens"],
         "groups": [{"tag": group["tag"], "texts": group["texts"], "weight": group["tokens"]}
                    for group in one["groups"]]} for one in record["shares_tokens"]])
    assert sum(row["weight"] for row in rows) == result["total_tokens"] + result["items_tokens"]


@pytest.mark.parametrize("kinds, shares, size", [
    ([], [("en", 0.7), ("de", 0.3)], None),        # the scarcest share whole, the other trimmed, its groups by bytes
    ([], [("en", 0.7), ("de", 0.3)], 3000),        # the items' share of a size, then each share's by weight
    ([], [], None),                                  # every text
    (["plain", "conversation"], [("en", 1.0)], 1500),
])
def test_a_corpus_request_gives_the_token_files_and_rows_wpt_depot_build_gives(built, tmp_path, kinds, shares, size):
    name = rich(built)
    key = vocabulary(built, tmp_path)
    record = build(built.depot, built.depot.snapshot(name)["id"], section(built, name, kinds, shares, size),
                   built.store, built.tokenizer, address=built.depot.name(), shard_tokens=SHORT,
                   log=lambda *_: None)
    note = asked(built, tmp_path, name, kinds, shares, size, key)
    assert len(record["shards"]["plain"]) > 1
    if not shares:  # every text: the pieces of the long one, every conversation and the task
        assert (record["pieces"], record["conversations"], record["tasks"]["count"]) == (2, 4, 1)
    same(built, record, note)
    assert hashlib.sha256(json.dumps(note["request"], sort_keys=True).encode()).hexdigest()  # the request kept


def test_the_same_request_again_is_answered_at_once_with_its_delivery(built, tmp_path):
    key = vocabulary(built, tmp_path)
    first = asked(built, tmp_path, "d", [], [("en", 1.0)], 700, key)
    record = build(built.depot, built.depot.snapshot("d")["id"], section(built, "d", [], [("en", 1.0)], 700),
                   built.store, built.tokenizer, address=built.depot.name(), shard_tokens=SHORT, log=lambda *_: None)
    same(built, record, first)
    snapshot = built.depot.snapshot("d")
    request = {"depot": built.depot.name(),
               "snapshot": {"name": "d", "change": snapshot["at_change"], "digest": snapshot["digest"]},
               "selection": {"shares": [{"tag": "en", "share": 1.0}], "size": {"unit": "tokens", "amount": 700}},
               "preparation": {"name": "wpt-tokens", "settings": {"shard_tokens": SHORT},
                               "inputs": {"vocabulary": key}}}
    state = api.request(request, home=str(tmp_path / "freight"), depot=built.depot.address,
                        dock=built.store.root.as_uri(), log=lambda *_: None)
    assert state["state"] == "done" and state["answer"]["id"] == first["id"]


class Stop(Exception):
    """A job cut short."""


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_a_job_cut_short_goes_on_from_its_checkpoint_to_the_same_delivery(built, tmp_path, cut):
    # the job stops right after its cut-th checkpoint (in the walk, or among the conversations once it ended);
    # started again with that state and the same work folder, its program told which files it wrote, it gives
    # the corpus a job that never stopped gives
    from freight.job import Job

    name = rich(built)
    key = vocabulary(built, tmp_path)
    snapshot = built.depot.snapshot(name)
    request = {"depot": built.depot.name(),
               "snapshot": {"name": name, "change": snapshot["at_change"], "digest": snapshot["digest"]},
               "selection": {"kinds": [], "shares": [{"tag": "en", "share": 0.7}, {"tag": "de", "share": 0.3}],
                             "size": {"unit": "tokens", "amount": 3000}},
               "preparation": {"name": "wpt-tokens", "settings": {"shard_tokens": SHORT},
                               "inputs": {"vocabulary": key}}}
    saved = []

    def save(state):
        saved.append(state)
        if len(saved) == cut:
            raise Stop()
    work = tmp_path / "work"
    with pytest.raises(Stop):
        Job(request, built.depot.address, built.store, work, save=save, log=lambda *_: None).run()
    assert saved[-1]["files"]
    note = Job(request, built.depot.address, built.store, work, checkpoint=saved[-1],
               log=lambda *_: None).run()
    record = build(built.depot, snapshot["id"], section(built, name, [], [("en", 0.7), ("de", 0.3)], 3000),
                   built.store, built.tokenizer, address=built.depot.name(), shard_tokens=SHORT, log=lambda *_: None)
    same(built, record, note)


def test_the_command_queues_a_request_waits_for_it_and_finds_its_delivery(built, tmp_path, monkeypatch, capsys):
    from freight_app import main

    key = vocabulary(built, tmp_path)
    snapshot = built.depot.snapshot("d")
    request = {"depot": built.depot.name(),
               "snapshot": {"name": "d", "change": snapshot["at_change"], "digest": snapshot["digest"]},
               "selection": {"shares": [{"tag": "en", "share": 1.0}], "size": None},
               "preparation": {"name": "wpt-tokens", "inputs": {"vocabulary": key}}}
    (tmp_path / "request.json").write_text(json.dumps(request))
    for name, value in (("FREIGHT_HOME", str(tmp_path / "freight")), ("WPT_DEPOT", built.depot.address),
                        ("WPT_STORE", built.store.root.as_uri())):
        monkeypatch.setenv(name, value)
    assert main(["request", str(tmp_path / "request.json"), "--wait"]) == 0
    note = json.loads(capsys.readouterr().out)
    assert note["result"]["total_tokens"] > 0 and note["preparation"]["name"] == "wpt-tokens"
    assert main(["delivery", note["key"]]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == note["id"]
    assert main(["jobs"]) == 0
    assert [one["state"] for one in json.loads(capsys.readouterr().out)] == ["done"]
    assert main(["delivery", "0" * 64]) == 1
