import base64
import json

import pytest

from wpt_store import open_store
from wpt_store.shards import read_verified
from wpt_tokenizer import END, Tokenizer, chat
from wpt_depot import NewText
from wpt_depot.corpus import build, mark_stale, read_table, records, sample
from wpt_depot.errors import Refused
from wpt_depot.notice import model_notice

TOKENIZER = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, vocab_size=560)
CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


TASK = {"task": {"instructions": "Write hello into out.txt.", "image": "base",
                 "limits": {"max_turns": 4, "max_tokens": 512, "command_seconds": 10}, "needs": []},
        "files": {"notes.txt": {"mode": "100644", "content": encoded(b"say hello\n")}},
        "check": {"mode": "100755", "content": encoded(b"#!/bin/sh\necho 1\n")}}


def item(kind, content, *tags):
    return NewText(kind=kind, content=json.dumps(content), licenses=("CC0-1.0",), tags=tags)


def test_a_corpus_holds_its_libraries_texts_tokenized_with_their_credits_and_a_notice_line_each(make_depot, me,
                                                                                                tmp_path):
    depot, other = make_depot("depot"), make_depot("other")
    same = NewText(text="the same words", licenses=("CC0-1.0",), title="Twice", authors=("Ann",), tags=("en",),
                   notice="(c) Ann")
    first, twin = depot.add([same, same], me)
    [french] = depot.add([NewText(text="le renard", licenses=("CC0-1.0",), tags=("fr",))], me)
    talk, task, probe = depot.add([item("conversation", CONVERSATION, "en"), item("task", TASK, "en"),
                                   item("probe", {"question": "q?", "patterns": ["x"]}, "en")], me)
    [theirs] = other.add([NewText(text="from another depot", licenses=("CC0-1.0",), tags=("en",),
                                  notice="(c) Other")], me)
    for one in (depot, other):
        one.decide("admitted", "true", me)
    library, their_library = depot.new_library("mine"), other.new_library("theirs")
    depot.put(library, "true")
    other.put(their_library, "true")
    depot.new_dataset("first model", "classics", [(None, library), (other.address, their_library)])
    store = open_store((tmp_path / "store").as_uri())

    record = build(depot, "first model", store, TOKENIZER, log=lambda *_: None)

    rows = [row for shard in record["shards"] for row in read_verified(store, shard)]
    documents = {(source, document): (list(tokens), pool) for tokens, source, document, pool, _ in rows}
    assert set(documents) == {(depot.address, first), (depot.address, twin), (depot.address, french),
                              (other.address, theirs)}  # duplicates kept
    assert documents[(depot.address, french)] == ([*TOKENIZER.encode("le renard".encode()), END], "")
    [conversation] = [row for shard in record["conversation_shards"] for row in read_verified(store, shard)]
    assert list(conversation[0]) == chat.encode(TOKENIZER, CONVERSATION)[0]
    [kept_task] = store.get_bytes(record["tasks"]["file"]).decode("utf-8").splitlines()
    assert json.loads(kept_task) == {"depot": depot.address, "id": task, "content": TASK}
    assert record["left_out"] == {"no share": 0, "trimmed": 0, "kinds": {"probe": 1}}

    texts = {(one["depot"], one["id"]): one for one in read_table(store, record, "texts").to_pylist()}
    assert probe not in {id_ for _, id_ in texts}
    assert texts[(depot.address, first)]["title"] == "Twice" and texts[(depot.address, first)]["authors"] == ["Ann"]
    assert texts[(depot.address, first)]["tags"] == ["en"] and texts[(depot.address, talk)]["tags"] == ["en"]
    notices = {one["id"]: one["text"] for one in read_table(store, record, "notices").to_pylist()}
    assert notices == {"depot-1": "(c) Ann", "other-1": "(c) Other"}  # ids carry the depot: copied as they are
    assert notices[texts[(other.address, theirs)]["notice"]] == "(c) Other"
    assert read_table(store, record, "licenses").to_pylist() == [
        {"id": "CC0-1.0", "name": "Creative Commons Zero v1.0 Universal", "text": "The CC0 text."}]
    [dataset] = read_table(store, record, "dataset").to_pylist()
    assert dataset["name"] == "first model" and dataset["shares"] is None
    start = 0
    for tokens, source, document, _, _ in rows:  # each text's first token, over the token files in order
        assert texts[(source, document)]["tokens"] == start
        start += len(tokens)
    assert texts[(depot.address, talk)]["tokens"] == 0  # the one conversation starts the conversations' files

    lines = model_notice(store, record).split("\n\n")[0].splitlines()
    assert len(lines) == len(texts)
    assert f"{depot.address} {first}: Twice by Ann; CC0-1.0; notice {texts[(depot.address, first)]['notice']}" in lines
    # a model that learned less is credited with what it learned: the plain texts that start in the tokens it
    # read from the start, and the conversations or tasks when it learned from them
    _, source, document, _, _ = rows[0]
    assert model_notice(store, record, {"text": 1}).split("\n\n")[0].splitlines() == [
        line for line in lines if line.startswith(f"{source} {document}:")]
    assert model_notice(store, record, {"conversations": 1}).split("\n\n")[0].splitlines() == [
        line for line in lines if line.startswith(f"{depot.address} {talk}:")]

    assert build(depot, "first model", store, TOKENIZER, log=lambda *_: None)["id"] == record["id"]


def shared_depot(make_depot, me, english, french):
    """A depot of `english` texts tagged en and `french` texts tagged fr, each text 2 tokens long (plus END),
    an untagged text and a conversation, all admitted in one library."""
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(english)]
              + [NewText(text="the fox", licenses=("CC0-1.0",), tags=("fr",)) for _ in range(french)]
              + [NewText(text="the dog", licenses=("CC0-1.0",), tags=("de",)), item("conversation", CONVERSATION, "en")],
              me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    return depot, library


def test_shares_trim_to_their_proportions_keeping_the_scarcest_whole_and_a_seeded_subset_of_the_others(
        make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=12, french=3)
    depot.new_dataset("mix", "classics", [(None, library)], [("en", 0.5), ("fr", 0.5)])
    store = open_store((tmp_path / "store").as_uri())

    record = build(depot, "mix", store, TOKENIZER, log=lambda *_: None)

    rows = [row for shard in record["shards"] for row in read_verified(store, shard)]
    by_pool = {}
    for tokens, _, document, pool, _ in rows:
        by_pool.setdefault(pool, []).append(document)
    assert sorted(map(len, by_pool.values())) == [3, 3]  # french is the scarcest: all 3, as many english
    assert len(set(by_pool["en"])) == 3  # no text repeated
    assert record["shares_tokens"] == [{"tag": "en", "tokens": record["shares_tokens"][1]["tokens"]},
                                       {"tag": "fr", "tokens": record["shares_tokens"][1]["tokens"]}]
    assert record["left_out"]["no share"] == 1 and record["left_out"]["trimmed"] == 9
    assert record["conversations"] == 1  # an item is no plain text: shares leave it whole
    again = build(depot, "mix", open_store((tmp_path / "other").as_uri()), TOKENIZER, log=lambda *_: None)
    assert again["id"] == record["id"]  # the seed makes the same subset


def test_shares_in_other_proportions_and_the_sample_count_the_same_way(make_depot, me):
    depot, library = shared_depot(make_depot, me, english=12, french=3)
    depot.new_dataset("mix", "classics", [(None, library)], [("en", 0.75), ("fr", 0.25)])

    found = sample(depot, "mix", 10**6)

    pools = [depot.show(one["id"])["text"]["tags"][0] for one in found]
    assert (pools.count("en"), pools.count("fr")) == (9, 3)


def test_shares_naming_a_tag_no_text_has_refuse_the_build(make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=2, french=0)
    depot.new_dataset("mix", "classics", [(None, library)], [("en", 0.5), ("fr", 0.5)])

    with pytest.raises(Refused):
        build(depot, "mix", open_store((tmp_path / "store").as_uri()), TOKENIZER, log=lambda *_: None)

    assert records(open_store((tmp_path / "store").as_uri())) == []


def test_an_erasure_marks_stale_every_corpus_holding_its_text_and_a_new_build_is_fresh(make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=2, french=2)
    depot.new_dataset("all", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    record = build(depot, "all", store, TOKENIZER, log=lambda *_: None)
    [text] = depot.sql("SELECT id FROM texts WHERE list_contains(tags, 'de')")["id"].to_pylist()
    assert "stale" not in record

    assert mark_stale(store, ["depot-99"], ["depot-1"]) == []
    assert mark_stale(store, [text], ["depot-1"]) == [record["id"]]

    assert records(store)[0]["stale"] == {"erasures": ["depot-1"]}


def test_a_dataset_builds_the_same_corpus_at_its_change_numbers_whatever_its_libraries_did_since(make_depot, me,
                                                                                                 tmp_path):
    depot = make_depot()
    first, second = depot.add([NewText(text="the fox", licenses=("CC0-1.0",), title="Fox"),
                               NewText(text="the dog", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, [first])
    depot.new_dataset("pinned", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    before = build(depot, "pinned", store, TOKENIZER, log=lambda *_: None)
    depot.put(library, [second])
    depot.edit_credits(first, {"title": "The fox"}, me)
    depot.decide("refused", [first], me)

    again = build(depot, "pinned", store, TOKENIZER, log=lambda *_: None)

    assert again["id"] == before["id"]
    assert [(one["id"], one["title"]) for one in read_table(store, again, "texts").to_pylist()] == [(first, "Fox")]


def test_a_dataset_keeps_its_corpora_through_a_rename_and_a_new_dataset_of_its_old_name_has_none(make_depot, me,
                                                                                                 tmp_path):
    from wpt_depot.corpus import latest
    depot = make_depot()
    depot.add([NewText(text="the fox", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, "true")
    dataset = depot.new_dataset("mix", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    record = build(depot, "mix", store, TOKENIZER, log=lambda *_: None)
    assert (record["dataset"]["id"], record["dataset"]["revision"]) == (dataset, 1)

    exported = depot.export_dataset("mix")
    depot.edit_dataset(dataset, "mix renamed", [(None, library)])
    newcomer = depot.import_dataset(exported)  # the old dataset's row as it was, under a new id

    assert latest(store, depot.address, dataset)["id"] == record["id"]
    assert latest(store, depot.address, newcomer) is None

    # the newcomer's row is the old one's then, yet another dataset gets a corpus of its own
    own = build(depot, "mix", store, TOKENIZER, log=lambda *_: None)
    assert own["id"] != record["id"] and own["dataset"]["id"] == newcomer
    assert latest(store, depot.address, newcomer)["id"] == own["id"]
    assert latest(store, depot.address, dataset)["id"] == record["id"]
