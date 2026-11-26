"""Training on a corpus cut into the portions of its order's roles
(wpt_corpus.roles): each step reads its role's portion, the steps that read
text windows the pretraining one and those that read conversations the
fine-tuning one; no file of the held-out portion is ever fetched, and the
texts a model learned from never name one of its texts; a corpus delivered
without portions still trains, all of it its pretraining portion."""

import json

import pytest

from depot import NewText
from dock import api as dock_api
from freight import api as freight
from wpt.resources import main as build_resources
from wpt.train import main as train
from wpt_corpus import roles

TALK = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}]}
HELD_OUT = 3500  # two of the 16 stories


def portion_files(note: dict, name: str, kind: str = "") -> set[str]:
    """The keys of the files of a delivery's portion (of a kind, when one is given)."""
    return {one["key"] for one in note["files"] if one["path"].startswith(f"{name}/{kind}")}


def credited(folder) -> list[str]:
    """The lines of the texts a model folder's CREDITS names."""
    return (folder / "CREDITS").read_text().split("\n\n", 1)[0].splitlines()


@pytest.fixture
def delivered(built, tmp_path, monkeypatch, capsys):
    """resources(name, roles, steps): writes the order name of the snapshot "chat" (the 16 stories and two
    conversations tagged answer-as-assistant) with those roles and steps after its read-text step, asks the freight
    for its corpus through `wpt build-resources` and gives its delivery note. The vocabulary is the test one, which
    the dock holds as an adopted delivery; the held-out portion is HELD_OUT bytes."""
    monkeypatch.setenv("FREIGHT_HOME", str(tmp_path / "freight"))
    monkeypatch.setattr(roles, "HELD_OUT_BYTES", HELD_OUT)
    built.snapshot("chat", [NewText(kind="conversation", content=json.dumps(TALK), tags=("answer-as-assistant",))] * 2)
    freight.adopt(f"{freight.ADOPTED}{built.vocabulary}", {"vocabulary.json": built.tokenizer.to_bytes()},
                  {"sha256": built.vocabulary, "size": built.tokenizer.vocab_size}, {"by": "the test"},
                  on_erasure="note")

    def resources(name, order_roles, steps=()):
        built.order(name, training={"stages": [{"block_size": 64, "steps": 1, "batch_size": 4}]}, steps=steps,
                    snapshot="chat")
        path = tmp_path / "data" / "orders" / "models" / f"{name}.json"
        order = json.loads(path.read_text())
        order["data"]["roles"] = order_roles
        path.write_text(json.dumps(order))
        capsys.readouterr()
        assert build_resources([name]) == 0
        out = capsys.readouterr().out
        return json.loads(out[out.index("\n{\n") + 1:])["corpus"]

    return resources


def test_the_held_out_portion_is_never_read_nor_credited(trained):
    note = trained.notes["corpus"]
    assert [one["name"] for one in note["portions"]] == ["held_out", "pretrain"]
    held_out = portion_files(note, "held_out")
    assert held_out and not held_out & set(trained.fetched)
    assert portion_files(note, "pretrain", "plain-") <= set(trained.fetched)  # what the read-text step read
    rows = freight.items(note["key"])
    lines = credited(trained.folder)
    for row in rows:
        named = any(line.startswith(f"{row['depot']} {row['id']}:") for line in lines)
        assert named == (row["portion"] == "pretrain"), row  # the step read the whole pretraining portion
    provenance = json.loads((trained.folder / "provenance.json").read_text())
    assert provenance["texts"] == sum(1 for row in rows if row["portion"] == "pretrain")


def test_the_conversation_step_reads_the_fine_tuning_portion(built, delivered, tmp_path, monkeypatch, capsys):
    chat_step = [{"step": "answer-as-assistant"}]
    note = delivered("talks", ["held_out", "pretrain", "finetune"], chat_step)
    assert [one["name"] for one in note["portions"]] == ["held_out", "pretrain", "finetune"]
    fetched = []
    for name in ("fetch", "fetch_into", "fetch_range", "stream"):
        def reading(dock, key, *args, _read=getattr(dock_api, name), **kwargs):
            fetched.append(key)
            return _read(dock, key, *args, **kwargs)
        monkeypatch.setattr(dock_api, name, reading)
    train(["talks", "--out", "model"])
    assert portion_files(note, "finetune", "conversation-") <= set(fetched)
    assert not portion_files(note, "held_out") & set(fetched)
    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["steps"][1]["learned"]["conversation"] == 2
    talks = [row for row in freight.items(note["key"]) if row["kind"] == "conversation"]
    assert {row["portion"] for row in talks} == {"finetune"}
    assert all(any(line.startswith(f"{row['depot']} {row['id']}:") for line in credited(tmp_path / "model"))
               for row in talks)
    # the same step, its order's corpus without a fine-tuning portion: nothing for it to read
    delivered("plain", ["held_out", "pretrain"], chat_step)
    with pytest.raises(SystemExit) as stopped:
        train(["plain", "--out", "other"])
    assert stopped.value.code == 2
    assert "step 2 answer-as-assistant: no conversations" in capsys.readouterr().err


def test_a_corpus_delivered_without_portions_trains_as_its_pretraining_portion(built, delivered, tmp_path):
    note = delivered("whole", ["held_out", "pretrain"])
    snapshot = note["request"]["snapshot"]
    request = {"depot": note["request"]["depot"], "snapshot": snapshot,
               "selection": {"kinds": [], "shares": [{"tag": "en", "share": 1.0}], "size": None},
               "preparation": note["request"]["preparation"]}
    done = freight.wait(freight.request(request)["id"])
    assert done["state"] == "done", done["problem"]
    old = done["answer"]
    assert "portions" not in old["result"]
    train(["whole", old["id"], "--out", "model"])
    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["corpus"]["id"] == old["id"]
    assert provenance["texts"] == old["items"]["kinds"]["plain"]  # every story: none held out
