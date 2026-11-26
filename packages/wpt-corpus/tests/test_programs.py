"""wpt-tokens and wpt-count spoken to as the freight speaks to them
(docs/api/preparation.md): the version each answers, and a delivery cut
into portions resumed in a fresh folder."""

import io
import json

import pytest

from wpt_corpus import channel, programs
from wpt_tokenizer import Tokenizer

TEXTS = [f"text {n}: the quick brown fox jumps over the lazy dog. ".encode() * 3 for n in range(6)]


@pytest.fixture
def vocabulary(tmp_path):
    """A folder holding vocabulary.json, a small vocabulary learned from TEXTS."""
    folder = tmp_path / "vocabulary"
    folder.mkdir()
    (folder / "vocabulary.json").write_bytes(Tokenizer.train(TEXTS, vocab_size=540).to_bytes())
    return folder


def talk(files: bool, start: dict, items=()) -> list[dict]:
    """The messages the program answers to start, then a take for each of items ({ref, kind, portion,
    content}), then end."""
    sent = io.BytesIO()
    channel.write(sent, {"type": "start", "threads": 1, **start})
    for item in items:
        channel.write(sent, {"type": "take", "ref": item["ref"], "depot": "d", "id": f"i{item['ref']}",
                             "name": f"i{item['ref']}", "piece": None, "kind": item["kind"], "pool": "",
                             "bytes": len(item["content"]), **({"portion": item["portion"]} if "portion" in item
                                                                else {})}, item["content"])
    if items:
        channel.write(sent, {"type": "end"})
    sent.seek(0)
    answered = io.BytesIO()
    programs._Program("wpt-tokens" if files else "wpt-count", files=files).run(sent, answered)
    return [json.loads(line) for line in answered.getvalue().splitlines()]


@pytest.mark.parametrize("files, portions, version", [
    (True, None, programs.VERSION), (True, ["a", "b"], programs.PORTIONED_VERSION),
    (False, None, programs.VERSION), (False, ["a", "b"], programs.VERSION)])
def test_wpt_tokens_answers_a_version_of_its_own_for_a_delivery_cut_into_portions(
        vocabulary, tmp_path, files, portions, version):
    # the files and result of a delivery cut into portions are not those VERSION gave: a request without portions
    # keeps VERSION, so its delivery key stays the same, and wpt-count gives the same result either way
    (ready,) = talk(files, {"inputs": {"vocabulary": str(vocabulary)}, "out": str(tmp_path / "out"),
                            **({"portions": portions} if portions else {})})
    assert (ready["type"], ready["version"]) == ("ready", version)


def test_a_delivery_cut_into_portions_resumed_in_a_fresh_folder_keeps_the_hash_of_an_earlier_portion_s_tasks(
        vocabulary, tmp_path):
    # the portion a holds a task, and its tasks file was said before b's first token file: the resumed run's result
    # gives a's tasks the hash of what a wrote, though the fresh folder holds no tasks file of a
    task = json.dumps({"task": {"instructions": "say hello"}}).encode()
    items = [{"ref": 0, "kind": "task", "portion": "a", "content": task},
             *({"ref": n, "kind": "plain", "portion": "b", "content": TEXTS[n]} for n in range(1, 6))]
    start = {"inputs": {"vocabulary": str(vocabulary)}, "portions": ["a", "b"], "settings": {"shard_tokens": 10}}
    whole = talk(True, {**start, "out": str(tmp_path / "whole")}, items)
    files = [one for one in whole if one["type"] == "file"]
    assert [one["portion"] for one in files[:2]] == ["a", "b"] and files[0]["path"] == "a/tasks.jsonl"
    stop = whole.index(files[2])  # after b's second token file
    after = [one["ref"] for one in whole[:stop] if one["type"] == "taken"][-1]
    resumed = talk(True, {**start, "out": str(tmp_path / "fresh"), "resume": {"files": files[:3], "after": after}},
                   [one for one in items if one["ref"] > after])
    assert resumed[-1]["type"] == "done" and whole[-1]["type"] == "done"
    assert resumed[-1]["result"] == whole[-1]["result"]
    assert whole[-1]["result"]["portions"]["a"]["tasks"] == {"path": "a/tasks.jsonl", "count": 1}
