import base64
import hashlib
import json
import sqlite3
from compression import zstd

from pathlib import Path

import pytest

from wpt_store import open_store
from wpt_store.shards import read_verified
from wpt_tokenizer import END, Tokenizer, chat
from wpt_depot import About, Depot, NewText, clone
from wpt_depot import corpora, order, pieces
from wpt_depot.corpora import Room, TooBig, latest, mark_stale, read_table, records
from wpt_depot.corpus import build, sample, task_text
from wpt_depot.errors import Refused
from wpt_depot.exports import Export
from wpt_depot.ids import export_key
from wpt_depot.notice import model_notice
from wpt_depot.parts import Parted

TOKENIZER = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, vocab_size=560)
CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


HERE = "https://depot.example"  # the public address of the depot that holds the dataset


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

    record = build(depot, "first model", store, TOKENIZER, max_tokens=None, url=HERE, log=lambda *_: None)

    assert record["depot"] == HERE  # its public address names the depot that holds the dataset, never its file
    rows = [row for shard in record["shards"] for row in read_verified(store, shard)]
    documents = {(source, document): (list(tokens), pool) for tokens, source, document, pool, _ in rows}
    assert set(documents) == {(HERE, first), (HERE, twin), (HERE, french),
                              ("other", theirs)}  # duplicates kept; another depot by its own name
    assert documents[(HERE, french)] == ([*TOKENIZER.encode("le renard".encode()), END], "")
    [conversation] = [row for shard in record["conversation_shards"] for row in read_verified(store, shard)]
    assert list(conversation[0]) == chat.encode(TOKENIZER, CONVERSATION)[0]
    [kept_task] = store.get_bytes(record["tasks"]["file"]).decode("utf-8").splitlines()
    assert json.loads(kept_task) == {"depot": HERE, "id": task, "content": TASK}
    assert record["left_out"] == {"no share": 0, "trimmed": 0, "trimmed items": 0, "other kinds": 1}

    texts = {(one["depot"], one["id"]): one for one in read_table(store, record, "texts").to_pylist()}
    assert probe not in {id_ for _, id_ in texts}
    assert texts[(HERE, first)]["title"] == "Twice" and texts[(HERE, first)]["authors"] == ["Ann"]
    assert texts[(HERE, first)]["tags"] == ["en"] and texts[(HERE, talk)]["tags"] == ["en"]
    notices = {one["id"]: one["text"] for one in read_table(store, record, "notices").to_pylist()}
    assert notices == {"depot-1": "(c) Ann", "other-1": "(c) Other"}  # ids carry the depot: copied as they are
    assert notices[texts[("other", theirs)]["notice"]] == "(c) Other"
    assert read_table(store, record, "licenses").to_pylist() == [
        {"id": "CC0-1.0", "name": "Creative Commons Zero v1.0 Universal", "text": "The CC0 text."}]
    [dataset] = read_table(store, record, "dataset").to_pylist()
    assert dataset["name"] == "first model" and dataset["shares"] is None
    start = 0
    for tokens, source, document, _, _ in rows:  # each text's first token, over the token files in order
        assert texts[(source, document)]["tokens"] == start
        start += len(tokens)
    assert texts[(HERE, talk)]["tokens"] == 0  # the one conversation starts the conversations' files

    lines = model_notice(store, record).split("\n\n")[0].splitlines()
    assert len(lines) == len(texts)
    assert f"{HERE} {first}: Twice by Ann; CC0-1.0; notice {texts[(HERE, first)]['notice']}" in lines
    # a model that learned less is credited with what it learned: the plain texts that start in the tokens it
    # read from the start, and the conversations or tasks when it learned from them
    _, source, document, _, _ = rows[0]
    assert model_notice(store, record, {"text": 1}).split("\n\n")[0].splitlines() == [
        line for line in lines if line.startswith(f"{source} {document}:")]
    assert model_notice(store, record, {"conversations": 1}).split("\n\n")[0].splitlines() == [
        line for line in lines if line.startswith(f"{HERE} {talk}:")]

    assert build(depot, "first model", store, TOKENIZER, max_tokens=None, url=HERE,
                 log=lambda *_: None)["id"] == record["id"]


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

    record = build(depot, "mix", store, TOKENIZER, max_tokens=None, log=lambda *_: None)

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
    again = build(depot, "mix", open_store((tmp_path / "other").as_uri()), TOKENIZER, max_tokens=None,
                  log=lambda *_: None)
    assert again["id"] == record["id"]  # the seed makes the same subset


def test_the_sample_gives_each_share_its_part_of_the_bytes_in_the_corpus_s_order(make_depot, me):
    depot, library = shared_depot(make_depot, me, english=12, french=3)
    depot.new_dataset("mix", "classics", [(None, library)], [("en", 0.75), ("fr", 0.25)])

    # each text is 7 bytes: 28 bytes give the english share 21 (3 texts) and the french one 7 (1 text)
    found = list(sample(depot, "mix", 28))

    assert {where for where, _, _ in found} == {""}  # '' names the depot that holds the dataset
    pools = [depot.show(text)["text"]["tags"][0] for _, text, _ in found]
    assert (pools.count("en"), pools.count("fr")) == (3, 1)
    assert {data for _, _, data in found} == {b"the dog", b"the fox"}
    assert list(sample(depot, "mix", 28)) == found  # the order of the export keys, whatever the run


def test_shares_naming_a_tag_no_text_has_refuse_the_build(make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=2, french=0)
    depot.new_dataset("mix", "classics", [(None, library)], [("en", 0.5), ("fr", 0.5)])

    with pytest.raises(Refused):
        build(depot, "mix", open_store((tmp_path / "store").as_uri()), TOKENIZER, max_tokens=None, log=lambda *_: None)

    assert records(open_store((tmp_path / "store").as_uri())) == []


def test_an_erasure_marks_stale_every_corpus_holding_its_text_and_a_new_build_is_fresh(make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=2, french=2)
    depot.new_dataset("all", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    record = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    [text] = [row["id"] for row in depot.sql("SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'de')")]
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
    before = build(depot, "pinned", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    depot.put(library, [second])
    depot.edit_credits(first, {"title": "The fox"}, me)
    depot.decide("refused", [first], me)

    again = build(depot, "pinned", open_store((tmp_path / "other").as_uri()), TOKENIZER, max_tokens=None,
                  log=lambda *_: None)

    assert again["id"] == before["id"]
    assert [(one["id"], one["title"]) for one in read_table(store, again, "texts").to_pylist()] == [(first, "Fox")]


def test_a_dataset_keeps_its_corpora_through_a_rename_and_a_new_dataset_of_its_old_name_has_none(make_depot, me,
                                                                                                 tmp_path):
    depot = make_depot()
    depot.add([NewText(text="the fox", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, "true")
    dataset = depot.new_dataset("mix", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    record = build(depot, "mix", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    assert (record["dataset"]["id"], record["dataset"]["revision"]) == (dataset, 1)

    exported = depot.export_dataset("mix")
    depot.edit_dataset(dataset, "mix renamed", [(None, library)])
    newcomer = depot.import_dataset(exported)  # the old dataset's row as it was, under a new id

    assert latest(store, depot.name(), dataset)["id"] == record["id"]
    assert latest(store, depot.name(), newcomer) is None

    # the newcomer's row is the old one's then, yet another dataset gets a corpus of its own
    own = build(depot, "mix", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    assert own["id"] != record["id"] and own["dataset"]["id"] == newcomer
    assert latest(store, depot.name(), newcomer)["id"] == own["id"]
    assert latest(store, depot.name(), dataset)["id"] == record["id"]


def test_a_corpus_copies_each_text_s_scores_with_their_content_versions_and_chooses_nothing_by_them(
        make_depot, me, tmp_path):
    depot = make_depot()
    scored, unscored = depot.add([NewText(text="the fox", licenses=("CC0-1.0",)),
                                  NewText(text="the dog", licenses=("CC0-1.0",))], me)
    depot.change_content(scored, "the quick fox", None, me)
    depot.edit_scores(list([{"id": scored, "scores": [{"name": "rank", "value": -1.5, "content_version": 1},
                                                       {"name": "machine-made", "value": 0.03,
                                                        "content_version": 2}]}]), me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    depot.new_dataset("scored", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())

    record = build(depot, "scored", store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    texts = {one["id"]: one["scores"] for one in read_table(store, record, "texts").to_pylist()}
    assert texts == {scored: [{"name": "machine-made", "value": 0.03, "content_version": 2},
                              {"name": "rank", "value": -1.5, "content_version": 1}], unscored: None}


def test_an_erasure_carried_out_while_a_corpus_is_built_makes_it_stale_once_written_and_keeps_the_older_one(
        make_depot, me, tmp_path):
    # the erasure's own marking reads only the records that exist, and the build's record does not exist yet
    depot, library = shared_depot(make_depot, me, english=2, french=0)
    dataset = depot.new_dataset("all", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    older = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    depot.edit_dataset(dataset, "everything", [(None, library)])  # a new revision, the same vocabulary and size
    [text] = [row["id"] for row in depot.sql("SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(tags) "
                                             "WHERE value = 'de')")]
    asked = []

    def erase_while_finishing(state):
        if state["stage"] == "finishing":
            asked.append(depot.ask_erasure("a dog", [(text, 4, 7)], me))
            assert depot.decide_erasure(asked[0], "admitted", me) is True

    record = build(depot, "everything", store, TOKENIZER, max_tokens=None, progress=erase_while_finishing,
                   log=lambda *_: None)

    assert record["stale"] == {"erasures": asked}
    kept = {one["id"]: one for one in records(store)}
    assert kept[record["id"]]["stale"] == {"erasures": asked}
    assert corpora.fresh(kept[older["id"]])  # a stale corpus supersedes nothing


def test_a_corpus_takes_its_texts_in_the_order_of_the_export_keys_each_once_and_a_smaller_size_takes_that_order(
        make_depot, me, tmp_path, monkeypatch):
    # the build reads each library a page at a time by key and merges them: with pages of 3 rows, the 20 texts
    # of two libraries that share 5 of them come in the export's order, each once, and a size takes that order,
    # leaving out only what does not fit
    monkeypatch.setattr(order, "PAGE", 3)
    depot = make_depot()
    ids = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",), tags=("en",)) for n in range(20)], me)
    depot.decide("admitted", "true", me)
    first, second = depot.new_library("first"), depot.new_library("second")
    depot.put(first, ids[:12])
    depot.put(second, ids[7:])
    dataset = depot.new_dataset("all", "classics", [(None, first), (None, second)])
    store = open_store((tmp_path / "store").as_uri())

    record = build(depot, "all", store, TOKENIZER, max_tokens=None, url=HERE, log=lambda *_: None)

    taken = [document for shard in record["shards"] for _, _, document, _, _ in read_verified(store, shard)]
    export = Export(depot, depot.dataset_by_id(dataset), HERE)
    assert taken == [text for _, text, _, _ in export.members(0, 1)] == sorted(ids, key=export_key)
    smaller = build(depot, "all", store, TOKENIZER, max_tokens=record["total_tokens"] // 2, url=HERE,
                    log=lambda *_: None)
    cut = [document for shard in smaller["shards"] for _, _, document, _, _ in read_verified(store, shard)]
    assert 0 < len(cut) < len(taken) and cut == [text for text in taken if text in set(cut)]


def test_a_build_counted_from_the_texts_takes_what_a_build_from_stored_figures_takes(make_depot, me, tmp_path):
    # the figures count a text two libraries hold once for each, and a text with both share tags once in each
    # share; a dataset reading another depot's libraries has no figures, so its build counts their texts, the
    # same way: it takes the same texts as the same dataset built in that depot from its stored figures
    other, depot = make_depot("other"), make_depot("depot")
    ids = other.add([NewText(text=f"the quick brown fox {n}", licenses=("CC0-1.0",),
                             tags=("en", "fr") if n % 3 == 0 else ("en",) if n % 3 == 1 else ("fr",))
                     for n in range(30)]
                    + [NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",))], me)
    other.decide("admitted", "true", me)
    first, second = other.new_library("first"), other.new_library("second")
    other.put(first, ids[:20])
    other.put(second, ids[10:])
    shares = [("en", 0.5), ("fr", 0.5)]
    other.new_dataset("mix", "classics", [(None, first), (None, second)], shares)
    depot.new_dataset("mix", "classics", [(other.address, first), (other.address, second)], shares)

    def taken(record, store):
        return ([document for shard in record["shards"] for _, _, document, _, _ in read_verified(store, shard)],
                record["shares_tokens"], record["items_tokens"], record["left_out"])

    for size in (None, 60):
        said = {}
        for name, held in (("known", other), ("counted", depot)):
            store = open_store((tmp_path / f"{name}-{size}").as_uri())
            lines = said[name] = []
            said[name, "record"] = taken(build(held, "mix", store, TOKENIZER, max_tokens=size, log=lines.append),
                                         store)
        assert said["known", "record"] == said["counted", "record"]
        assert not any("not known" in line for line in said["known"])
        assert any("not known" in line for line in said["counted"])


def test_a_text_that_would_pass_the_size_is_left_out_and_the_texts_after_it_fill_the_corpus(make_depot, me,
                                                                                            tmp_path):
    depot = make_depot()
    [long] = depot.add([NewText(text="the quick brown fox jumps over the lazy dog " * 8, licenses=("CC0-1.0",),
                                tags=("en",))], me)
    short = depot.add([NewText(text=f"the dog {n}", licenses=("CC0-1.0",), tags=("en",)) for n in range(10)], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    depot.new_dataset("all", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    whole = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    rows = [(document, len(tokens)) for shard in whole["shards"] for tokens, _, document, _, _ in
            read_verified(store, shard)]
    order = [document for document, _ in rows]
    assert order.index(long) < len(order) - 1  # some short texts come after the long one
    limit = whole["total_tokens"] - dict(rows)[long]  # every short text fits, the long one does not

    record = build(depot, "all", store, TOKENIZER, max_tokens=limit, log=lambda *_: None)

    taken = [document for shard in record["shards"] for _, _, document, _, _ in read_verified(store, shard)]
    assert taken == [one for one in order if one != long] and sorted(taken) == sorted(short)
    assert record["total_tokens"] == limit and record["left_out"]["trimmed"] == 1


def items_depot(make_depot, me, plain: int):
    """A depot of `plain` texts tagged en (each 3 tokens with END), two conversations and one task, all
    admitted in one library read by the dataset "all"; and the tokens a conversation and the task count."""
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(plain)]
              + [item("conversation", CONVERSATION, "en"), item("conversation", CONVERSATION, "en"),
                 item("task", TASK, "en")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    depot.new_dataset("all", "classics", [(None, library)])
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    task = len(TOKENIZER.encode(task_text(TASK).encode("utf-8")))
    return depot, library, conversation, task


def corpus_tokens(store, record) -> int:
    """Every token the corpus holds: its plain texts', its conversations' and its tasks' (task_text)."""
    tasks = store.get_bytes(record["tasks"]["file"]).decode("utf-8").splitlines() if "tasks" in record else []
    return (record["total_tokens"] + sum(one["tokens"] for one in record["conversation_shards"])
            + sum(len(TOKENIZER.encode(task_text(json.loads(line)["content"]).encode("utf-8"))) for line in tasks))


def items_part(depot, limit: int) -> int:
    """The items' share of a size: their part of the UTF-8 bytes of every text of the depot, of limit."""
    rows = depot.rows("SELECT c.kind, c.bytes AS size FROM texts t JOIN content_lists c "
                      "ON c.id = t.id AND c.version = t.content_version")
    items = sum(row["size"] for row in rows if row["kind"] is not None)
    return limit * items // sum(row["size"] for row in rows)


def test_a_size_gives_the_items_their_natural_share_and_the_plain_texts_the_rest(make_depot, me, tmp_path):
    depot, library, conversation, task = items_depot(make_depot, me, plain=300)
    store = open_store((tmp_path / "store").as_uri())
    every = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    limit = every["items_tokens"]  # the items' share of it leaves some of them out, the plain texts' part fills
    part = items_part(depot, limit)
    assert part < limit

    record = build(depot, "all", store, TOKENIZER, max_tokens=limit, log=lambda *_: None)

    # every token: every item and every plain text, the items' tokens counted
    assert (every["conversations"], every["tasks"]["count"], every["total_tokens"]) == (2, 1, 300 * 3)
    assert every["items_tokens"] == 2 * conversation + task == corpus_tokens(store, every) - every["total_tokens"]
    # the items take their share of the size, within one item, and the plain texts what is left, within one text
    assert part - max(conversation, task) < record["items_tokens"] <= part
    assert record["items_tokens"] == corpus_tokens(store, record) - record["total_tokens"]
    assert limit - record["items_tokens"] - 3 < record["total_tokens"] <= limit - record["items_tokens"]
    # the same depot, dataset, vocabulary and size build the same corpus
    again = build(depot, "all", open_store((tmp_path / "again").as_uri()), TOKENIZER, max_tokens=limit,
                  log=lambda *_: None)
    assert (again["id"], again["folder"]) == (record["id"], record["folder"])


def test_with_shares_the_items_have_their_share_of_the_corpus_of_every_token(make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(300)]
              + [NewText(text="the fox", licenses=("CC0-1.0",), tags=("fr",)) for _ in range(10)]
              + [item("conversation", CONVERSATION, "en") for _ in range(6)], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    depot.new_dataset("mix", "classics", [(None, library)], [("en", 0.5), ("fr", 0.5)])
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    items = 6 * len(json.dumps(CONVERSATION).encode("utf-8"))
    store = open_store((tmp_path / "store").as_uri())
    every = build(depot, "mix", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    limit = every["items_tokens"]
    part = limit * items // (items + 2 * 10 * 7)  # without a size, en keeps as many bytes as fr: 10 texts of 7

    record = build(depot, "mix", store, TOKENIZER, max_tokens=limit, log=lambda *_: None)

    assert [one["tokens"] for one in every["shares_tokens"]] == [30, 30]
    assert part - conversation < record["items_tokens"] <= part
    assert record["items_tokens"] > limit * items // (items + 310 * 7)  # more than their share of every plain text


def test_the_item_share_stops_at_its_first_item_that_would_pass_its_part(make_depot, me, tmp_path):
    depot, library, conversation, task = items_depot(make_depot, me, plain=3)
    limit = 3 * 3 + conversation + task  # the items hold nearly all the bytes: their part is most of the size
    part = items_part(depot, limit)
    assert part < 2 * conversation + task  # so not every item fits
    store = open_store((tmp_path / "store").as_uri())

    record = build(depot, "all", store, TOKENIZER, max_tokens=limit, log=lambda *_: None)

    taken = record["conversations"] + record.get("tasks", {}).get("count", 0)
    assert 0 < taken < 3 and record["left_out"]["trimmed items"] == 3 - taken
    assert record["items_tokens"] <= part and corpus_tokens(store, record) <= limit


def test_conversations_and_tasks_are_one_share_where_an_item_that_does_not_fit_leaves_room_for_the_next(
        make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(20)]
              + [item("conversation", CONVERSATION, "en") for _ in range(5)] + [item("task", TASK, "en"), item("task", TASK, "en")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    depot.new_dataset("all", "classics", [(None, library)])
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    task = len(TOKENIZER.encode(task_text(TASK).encode("utf-8")))
    store = open_store((tmp_path / "store").as_uri())
    every = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    kinds = [one["kind"] for one in read_table(store, every, "texts").to_pylist() if one["kind"] is not None]
    before = kinds.index("task")  # the conversations the corpus's order puts before the first task
    assert task > conversation and "conversation" in kinds[before + 1:]  # a smaller conversation follows it
    # the smallest size whose item part fits every conversation before the first task and one more, not the task
    limit = next(size for size in range(1, 100_000) if items_part(depot, size) >= (before + 1) * conversation)
    part = items_part(depot, limit)
    assert part < before * conversation + task

    record = build(depot, "all", store, TOKENIZER, max_tokens=limit, log=lambda *_: None)

    # the first task does not fit and is left out; the conversations after it fill what is left of the part
    assert record.get("tasks", {}).get("count", 0) == 0
    assert record["conversations"] == min(kinds.count("conversation"), part // conversation) > before
    assert record["items_tokens"] == record["conversations"] * conversation
    assert record["left_out"]["trimmed items"] == len(kinds) - record["conversations"]


def test_a_dataset_of_items_alone_fills_its_size_with_them(make_depot, me, tmp_path):
    depot, library, conversation, task = items_depot(make_depot, me, plain=0)
    limit = 2 * conversation + task - 1  # every item but the last in order fits
    store = open_store((tmp_path / "store").as_uri())

    record = build(depot, "all", store, TOKENIZER, max_tokens=limit, log=lambda *_: None)

    assert record["total_tokens"] == 0
    assert limit - max(conversation, task) < record["items_tokens"] <= limit
    assert record["items_tokens"] == corpus_tokens(store, record)
    assert record["left_out"]["trimmed items"] == 1


def test_a_depot_moved_to_another_file_builds_the_same_corpus(make_depot, me, tmp_path):
    depot, library, _, _ = items_depot(make_depot, me, plain=4)
    first = build(depot, "all", open_store((tmp_path / "first").as_uri()), TOKENIZER, max_tokens=None,
                  log=lambda *_: None)
    moved = tmp_path / "elsewhere" / "moved.sqlite"
    moved.parent.mkdir()
    with sqlite3.connect(depot.address) as source, sqlite3.connect(moved) as target:
        source.backup(target)
    second_store = open_store((tmp_path / "second").as_uri())

    second = build(Depot.view(str(moved)), "all", second_store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    assert second["id"] == first["id"] and second["depot"] == first["depot"] == depot.name()
    for kind in ("shards", "conversation_shards"):
        assert [one["sha256"] for one in second[kind]] == [one["sha256"] for one in first[kind]]
    assert second["tasks"]["file"]["sha256"] == first["tasks"]["file"]["sha256"]
    assert [one["sha256"] for one in second["tables"]["texts"]] == [one["sha256"] for one in first["tables"]["texts"]]


def test_a_clone_builds_its_upstream_s_dataset_into_the_upstream_s_corpus(make_depot, me, tmp_path, serve):
    upstream, _, _, _ = items_depot(make_depot, me, plain=4)
    address = serve(upstream)
    built = build(upstream, "all", open_store((tmp_path / "upstream").as_uri()), TOKENIZER, max_tokens=None,
                  url=address,
                  log=lambda *_: None)
    follower = clone(address, str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))

    again = build(follower, "all", open_store((tmp_path / "follower").as_uri()), TOKENIZER, max_tokens=None,
                  log=lambda *_: None)

    assert again["depot"] == address and again["id"] == built["id"]


def test_a_build_cut_short_goes_on_from_its_last_token_file_and_writes_the_same_corpus(make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=40, french=0)
    depot.new_dataset("all", "classics", [(None, library)])
    whole = build(depot, "all", open_store((tmp_path / "whole").as_uri()), TOKENIZER, max_tokens=None, shard_tokens=20,
                  log=lambda *_: None)
    saved = []

    def stop_after_two(state):
        saved.append(state)
        if len(saved) == 2:
            raise KeyboardInterrupt

    store = open_store((tmp_path / "store").as_uri())
    with pytest.raises(KeyboardInterrupt):
        build(depot, "all", store, TOKENIZER, max_tokens=None, work=tmp_path / "work", save=stop_after_two,
              shard_tokens=20,
              log=lambda *_: None)
    assert records(store) == []  # no record until the build is whole

    resumed = build(depot, "all", store, TOKENIZER, max_tokens=None, work=tmp_path / "work", checkpoint=saved[-1],
                    shard_tokens=20,
                    log=lambda *_: None)

    assert resumed["id"] == whole["id"]
    assert [one["sha256"] for one in resumed["shards"]] == [one["sha256"] for one in whole["shards"]]


def test_a_new_corpus_deletes_only_the_older_one_of_its_dataset_vocabulary_and_size_which_builds_again_the_same(
        make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=3, french=0)
    dataset = depot.new_dataset("all", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    vocabulary = hashlib.sha256(TOKENIZER.to_bytes()).hexdigest()
    first = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    small = build(depot, "all", store, TOKENIZER, max_tokens=5, log=lambda *_: None)
    depot.edit_dataset(dataset, "everything", [(None, library)])  # a new revision, which the corpora copy

    second = build(depot, "everything", store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    kept = {one["id"]: one for one in records(store)}
    assert kept[first["id"]]["deleted"]["whole"] and kept[first["id"]]["deleted"]["why"] == corpora.SUPERSEDED
    assert not any(store.has(corpora.File.of(one)) for one in corpora.files(first))
    assert corpora.fresh(kept[small["id"]]) and small["folder"] != first["folder"]  # another size: its own corpus
    assert (small["max_tokens"], second["max_tokens"]) == (5, None)
    assert corpora.latest_fresh(store, depot.name(), dataset, vocabulary, None)["id"] == second["id"]
    assert corpora.latest_fresh(store, depot.name(), dataset, vocabulary, 5)["id"] == small["id"]

    small_again = build(depot, "everything", store, TOKENIZER, max_tokens=5, log=lambda *_: None)

    kept = {one["id"]: one for one in records(store)}
    assert "deleted" in kept[small["id"]] and corpora.fresh(kept[second["id"]])
    assert corpora.latest_fresh(store, depot.name(), dataset, vocabulary, 5)["id"] == small_again["id"]
    # built again from its own record, the deleted corpus is the same corpus
    again = build(depot, "all", store, TOKENIZER, max_tokens=None, dataset=kept[first["id"]]["dataset"],
                  log=lambda *_: None)
    assert again["id"] == first["id"] and corpora.fresh(again)
    # of an older revision, it leaves the corpus of the current revision as it is, and the latest one still
    assert corpora.fresh({one["id"]: one for one in records(store)}[second["id"]])
    assert corpora.latest_fresh(store, depot.name(), dataset, vocabulary, None)["id"] == second["id"]


def test_publishing_a_vocabulary_builds_the_same_corpus_into_the_same_folder_and_deletes_nothing(
        make_depot, me, tmp_path):
    # a build never reads the vocabulary its dataset publishes: building again after publishing one is the same
    # corpus, so nothing is built twice and supersede deletes nothing
    depot, library = shared_depot(make_depot, me, english=3, french=0)
    dataset = depot.new_dataset("all", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    first = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    recipe = depot.new_tokenizer("classics", None, "all", 1000, 600)
    sha256 = depot.add_vocabulary(recipe, 1, TOKENIZER.to_bytes(), [], 0)
    depot.edit_dataset(dataset, "all", [(None, library)], vocabulary=sha256)

    again = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    assert (again["id"], again["folder"]) == (first["id"], first["folder"])
    assert all(corpora.fresh(one) for one in records(store))


def test_a_cap_deletes_the_corpus_downloaded_least_recently_and_refuses_a_corpus_bigger_than_itself(
        make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=3, french=3)
    store = open_store((tmp_path / "store").as_uri())
    depot.new_dataset("en", "classics", [(None, library)], [("en", 1.0)])
    depot.new_dataset("fr", "classics", [(None, library)], [("fr", 1.0)])
    english = build(depot, "en", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    french = build(depot, "fr", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    size = sum(one["bytes"] for one in corpora.files(english))
    downloaded = {english["id"]: 2.0, french["id"]: 1.0}  # french was downloaded least recently
    depot.new_dataset("all", "classics", [(None, library)])

    record = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None,
                   room=lambda folder: Room(store, 2 * size + size // 2, folder, downloaded.get, set))

    kept = {one["id"]: one for one in records(store)}
    assert "deleted" in kept[french["id"]] and "deleted" not in kept[english["id"]]
    assert corpora.fresh(kept[record["id"]])
    with pytest.raises(TooBig):
        build(depot, "fr", store, TOKENIZER, max_tokens=4,
              log=lambda *_: None, room=lambda folder: Room(store, 10, folder))


def test_a_cap_weighs_only_the_items_max_tokens_keeps(make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(20)]
              + [item("conversation", CONVERSATION, "en") for _ in range(200)], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    limit = 2 * conversation + conversation // 2  # the conversations' part holds two of them
    depot.new_dataset("all", "classics", [(None, library)])
    alone = build(depot, "all", open_store((tmp_path / "alone").as_uri()), TOKENIZER, max_tokens=limit,
                  shard_tokens=6,
                  log=lambda *_: None)
    size = sum(one["bytes"] for one in corpora.files(alone))
    store = open_store((tmp_path / "store").as_uri())

    # the 198 conversations the limit leaves out take no room in the estimate a cap of three times the size checks
    record = build(depot, "all", store, TOKENIZER, max_tokens=limit, shard_tokens=6, log=lambda *_: None,
                   room=lambda folder: Room(store, 3 * size, folder))

    assert record["id"] == alone["id"] and record["conversations"] == 2


def test_a_corpus_bigger_than_the_cap_is_refused_before_any_other_corpus_is_deleted_for_it(make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=40, french=2)
    depot.new_dataset("fr", "classics", [(None, library)], [("fr", 1.0)])
    depot.new_dataset("all", "classics", [(None, library)])
    whole = build(depot, "all", open_store((tmp_path / "whole").as_uri()), TOKENIZER, max_tokens=None, shard_tokens=20,
                  log=lambda *_: None)
    size = sum(one["bytes"] for one in corpora.files(whole))
    store = open_store((tmp_path / "store").as_uri())
    french = build(depot, "fr", store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    with pytest.raises(TooBig, match="no corpus was deleted"):
        build(depot, "all", store, TOKENIZER, max_tokens=None, shard_tokens=20, log=lambda *_: None,
              room=lambda folder: Room(store, size // 2, folder))

    assert corpora.fresh(records(store)[0]) and records(store)[0]["id"] == french["id"]


def test_a_cap_deletes_the_files_a_build_left_without_a_record_and_that_build_started_again_starts_over(
        make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=40, french=0)
    dataset = depot.new_dataset("all", "classics", [(None, library)])
    row = depot.dataset("all")
    whole = build(depot, "all", open_store((tmp_path / "whole").as_uri()), TOKENIZER, max_tokens=None, shard_tokens=20,
                  log=lambda *_: None)
    size = sum(one["bytes"] for one in corpora.files(whole))
    store = open_store((tmp_path / "store").as_uri())
    saved = []

    def stop_after_five(state):
        saved.append(state)
        if len(saved) == 5:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        build(depot, "all", store, TOKENIZER, max_tokens=None, work=tmp_path / "work", save=stop_after_five,
              shard_tokens=20,
              log=lambda *_: None)
    [left] = [key for key in store.kind.list("corpora/", deep=False)]
    left_bytes = sum(store.kind.list(left, deep=True).values())

    record = build(depot, "all", store, TOKENIZER, max_tokens=10**6, shard_tokens=20, log=lambda *_: None,
                    # a new folder
                   room=lambda folder: Room(store, size + left_bytes - 1, folder))

    assert corpora.fresh(record)
    assert sum(store.kind.list(left, deep=True).values()) < left_bytes  # a file at a time, as the build needed room
    resumed = build(depot, "all", store, TOKENIZER, max_tokens=None, dataset=row, work=tmp_path / "work",
                    checkpoint=saved[-1],
                    shard_tokens=20, log=lambda *_: None)
    assert resumed["id"] == whole["id"] and not corpora.missing(store, resumed["folder"], corpora.files(resumed))


def test_an_erasure_of_a_text_the_dataset_never_read_leaves_its_corpus_s_folder_and_id_as_they_were(
        make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=3, french=0)
    [outside] = depot.add([NewText(text="the cat", licenses=("CC0-1.0",))], me)
    depot.new_dataset("all", "classics", [(None, library)])
    store = open_store((tmp_path / "store").as_uri())
    first = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    corpora.delete(store, first, "a test deleted it")
    request = depot.ask_erasure("a cat", [(outside, 4, 7)], me)
    assert depot.decide_erasure(request, "admitted", me) is True

    again = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    assert (again["id"], again["folder"]) == (first["id"], first["folder"]) and corpora.fresh(again)
    assert corpora.find(store, first["id"])["id"] == first["id"]  # one record of that id


def two_depots(make_depot, me):
    """(a depot, another depot named other, its library, its text, the id of the dataset all of the first reading
    one library of each)."""
    depot, library = shared_depot(make_depot, me, english=3, french=0)
    other = make_depot("other")
    [theirs] = other.add([NewText(text="the lazy dog", licenses=("CC0-1.0",), tags=("en",))], me)
    other.decide("admitted", "true", me)
    their_library = other.new_library("theirs")
    other.put(their_library, "true")
    dataset = depot.new_dataset("all", "classics", [(None, library), (other.address, their_library)])
    return depot, other, their_library, theirs, dataset


def moved(depot, other, their_library, dataset, to):
    """Copy the other depot's file to to, delete the first file and point the dataset at the copy."""
    to.parent.mkdir()
    with sqlite3.connect(other.address) as source, sqlite3.connect(to) as target:
        source.backup(target)
    Path(other.address).unlink()
    library = depot.dataset_by_id(dataset)["libraries"][0]["library"]
    depot.edit_dataset(dataset, "all", [(None, library), (str(to), their_library)])


def test_another_depot_moved_to_another_file_builds_the_same_corpus_which_records_no_path(make_depot, me, tmp_path):
    depot, other, their_library, theirs, dataset = two_depots(make_depot, me)
    first = build(depot, "all", open_store((tmp_path / "first").as_uri()), TOKENIZER, max_tokens=None,
                  log=lambda *_: None)
    moved(depot, other, their_library, dataset, tmp_path / "elsewhere" / "moved.sqlite")
    second_store = open_store((tmp_path / "second").as_uri())

    second = build(depot, "all", second_store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    assert second["id"] == first["id"] and second["build"] == first["build"]
    assert second["dataset"]["libraries"][1]["depot"] == "other" and "other" in second["erasures"]
    # every file but the dataset table, which holds the build's time and the dataset's revision
    files = [*second["shards"], *second["conversation_shards"], *second["tables"]["texts"],
             second["tables"]["licenses"], second["tables"]["notices"]]
    assert [one["sha256"] for one in files] == [one["sha256"] for one in [
        *first["shards"], *first["conversation_shards"], *first["tables"]["texts"], first["tables"]["licenses"],
        first["tables"]["notices"]]]
    sources = {source for shard in second["shards"] for _, source, _, _, _ in read_verified(second_store, shard)}
    assert sources == {depot.name(), "other"}
    assert {one["depot"] for one in read_table(second_store, second, "texts").to_pylist()} == {depot.name(), "other"}
    [copied] = read_table(second_store, second, "dataset").to_pylist()
    assert str(tmp_path) not in json.dumps(second) + json.dumps(copied, default=str)


def test_a_build_where_two_depots_would_carry_one_name_is_refused(make_depot, me, tmp_path):
    depot, library = shared_depot(make_depot, me, english=1, french=0)
    store = open_store((tmp_path / "store").as_uri())
    namesakes = []
    for folder, name in (("twin", depot.name()), ("first", "other"), ("second", "other")):
        (tmp_path / folder).mkdir()
        one = Depot.new(str(tmp_path / folder / "depot.sqlite"), About(name, "Ada Lovelace", "ada@example.org"))
        namesakes.append((one.address, one.new_library("theirs")))
    depot.new_dataset("same as the holder", "classics", [(None, library), namesakes[0]])
    depot.new_dataset("two others", "classics", [(None, library), *namesakes[1:]])

    for dataset in ("same as the holder", "two others"):
        with pytest.raises(Refused, match="carry the name"):
            build(depot, dataset, store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    assert records(store) == []


def test_an_erasure_in_another_depot_makes_its_corpus_stale_and_its_rebuild_another_corpus(make_depot, me, tmp_path):
    depot, other, _, theirs, _ = two_depots(make_depot, me)
    store = open_store((tmp_path / "store").as_uri())
    record = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    request = other.ask_erasure("a dog", [(theirs, 4, 8)], me)
    assert other.decide_erasure(request, "admitted", me) is True

    assert mark_stale(store, [theirs], [request]) == [record["id"]]  # what carrying an erasure out does to the store
    again = build(depot, "all", store, TOKENIZER, max_tokens=None, dataset=record["dataset"], log=lambda *_: None)

    assert again["id"] != record["id"] and corpora.fresh(again) and again["erasures"]["other"] == [request]


def test_a_corpus_built_before_another_depot_moved_is_the_one_its_build_finds_after_the_move(
        make_depot, me, tmp_path):
    depot, other, their_library, theirs, dataset = two_depots(make_depot, me)
    store = open_store((tmp_path / "store").as_uri())
    record = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    moved(depot, other, their_library, dataset, tmp_path / "elsewhere" / "moved.sqlite")

    again = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)

    assert again["id"] == record["id"]


def test_a_stale_corpus_built_before_another_depot_moved_is_rebuilt_from_the_moved_file(make_depot, me, tmp_path):
    depot, other, their_library, theirs, dataset = two_depots(make_depot, me)
    store = open_store((tmp_path / "store").as_uri())
    record = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    moved(depot, other, their_library, dataset, tmp_path / "elsewhere" / "moved.sqlite")
    elsewhere = Depot.open(str(tmp_path / "elsewhere" / "moved.sqlite"))
    request = elsewhere.ask_erasure("a dog", [(theirs, 4, 8)], me)
    assert elsewhere.decide_erasure(request, "admitted", me) is True
    mark_stale(store, [theirs], [request])

    again = build(depot, "all", store, TOKENIZER, max_tokens=None, dataset=record["dataset"], log=lambda *_: None)

    assert again["id"] != record["id"] and corpora.fresh(again) and again["erasures"]["other"] == [request]


def repository(files: int, file_characters: int) -> str:
    """A repository as the depot holds one, as one text: each file a line `=== <path>` and its lines."""
    line = "the quick brown fox jumps over the lazy dog\n"
    return "".join(f"=== src/file{n}.c\n" + line * (file_characters // len(line)) for n in range(files))


def long_depot(make_depot, me):
    """A depot of 200 short texts and a repository about 3.4 times pieces.LENGTH long, cut into 4 pieces, all
    admitted in one library read by the dataset "all": (the depot, the short texts' ids, the repository's)."""
    depot = make_depot()
    [repo] = depot.add([NewText(text=repository(12, 300_000), licenses=("CC0-1.0",), title="A repository",
                                tags=("code",))], me)
    short = depot.add([NewText(text=f"the dog {n}", licenses=("CC0-1.0",), tags=("en",)) for n in range(200)], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, "true")
    depot.new_dataset("all", "classics", [(None, library)])
    return depot, short, repo


def test_a_long_text_comes_as_pieces_cut_between_files_each_at_its_own_place_of_the_order(make_depot, me, tmp_path):
    depot, short, repo = long_depot(make_depot, me)
    text = depot.content(repo)
    assert pieces.count(len(text)) == 4
    store = open_store((tmp_path / "store").as_uri())

    record = build(depot, "all", store, TOKENIZER, max_tokens=None, url=HERE, log=lambda *_: None)

    rows = [(document, list(tokens)) for shard in record["shards"] for tokens, _, document, _, _ in
            read_verified(store, shard)]
    names = [pieces.name(repo, n) for n in range(4)]
    order = [document for document, _ in rows]
    # each piece at the export key of its name, among the other texts, never one after the other
    assert order == sorted([*short, *names], key=export_key)
    places = sorted(order.index(name) for name in names)
    assert all(later - earlier > 1 for earlier, later in zip(places, places[1:]))
    # the pieces are the whole text cut at the start of a file's line, each encoded as a text of its own
    held = Parted(depot.reader(), repo, depot.listed_text(repo, 1)["parts"])
    cut = [pieces.read(held, n) for n in range(4)]
    assert "".join(cut) == text and all(one.startswith("=== src/") for one in cut)
    assert all(abs(len(one) - pieces.LENGTH) < 300_100 for one in cut[:3])
    assert all(dict(rows)[name] == [*TOKENIZER.encode(one.encode()), END] for name, one in zip(names, cut))
    # each piece has its row, with its number and its text's credits; the text counts once
    table = [one for one in read_table(store, record, "texts").to_pylist() if one["id"] == repo]
    assert sorted(one["piece"] for one in table) == [0, 1, 2, 3]
    assert {one["title"] for one in table} == {"A repository"}
    assert all(one["piece"] is None for one in read_table(store, record, "texts").to_pylist() if one["id"] != repo)
    assert (record["texts"], record["pieces"]) == (201, 4)
    assert [line for line in model_notice(store, record).split("\n\n")[0].splitlines()
            if line.startswith(f"{HERE} {repo}:")] == [
        f"{HERE} {repo}: A repository; CC0-1.0"]
    # the same depot builds the same bytes
    again = build(depot, "all", open_store((tmp_path / "again").as_uri()), TOKENIZER, max_tokens=None, url=HERE,
                  log=lambda *_: None)
    assert again["id"] == record["id"] and again["shards"] == record["shards"]


def test_a_long_text_no_longer_ends_a_corpus_smaller_than_it_which_takes_the_pieces_that_fit(make_depot, me,
                                                                                               tmp_path):
    # the whole repository would pass the size: the corpus still takes every short text and the pieces that fit
    depot, short, repo = long_depot(make_depot, me)
    store = open_store((tmp_path / "store").as_uri())
    whole = build(depot, "all", store, TOKENIZER, max_tokens=None, log=lambda *_: None)
    tokens = {document: len(row) for shard in whole["shards"] for row, _, document, _, _ in
              read_verified(store, shard)}
    short_tokens = sum(tokens[one] for one in short)
    smallest = min(range(4), key=lambda n: tokens[pieces.name(repo, n)])
    others = [tokens[pieces.name(repo, n)] for n in range(4) if n != smallest]
    limit = short_tokens + tokens[pieces.name(repo, smallest)] + 10_000  # a thousandth of it is under 10,000
    assert min(others) > limit  # room for the smallest piece alone

    record = build(depot, "all", store, TOKENIZER, max_tokens=limit, log=lambda *_: None)

    taken = [document for shard in record["shards"] for _, _, document, _, _ in read_verified(store, shard)]
    assert sorted(taken) == sorted([*short, pieces.name(repo, smallest)])
    assert record["total_tokens"] <= limit and record["left_out"]["trimmed"] == 0
    # the sample of a vocabulary cuts and fills the same way, a piece under its text's id
    sample_bytes = sum(len(f"the dog {n}") for n in range(200)) + 2 * pieces.LENGTH
    found = list(sample(depot, "all", sample_bytes))
    assert {text for _, text, _ in found} >= set(short)
    assert [len(data) for _, text, data in found if text == repo] and all(
        len(data) < 1.5 * pieces.LENGTH for _, text, data in found if text == repo)


def test_a_piece_is_cut_at_a_line_start_without_a_file_in_reach_and_at_its_place_without_a_line(make_depot, me):
    depot = make_depot()
    one_file = "=== big.txt\n" + "a line of a very long file\n" * 100_000  # longer than a piece: no file in reach
    no_line = "x" * (3 * pieces.LENGTH)
    texts = depot.add([NewText(text=one, licenses=("CC0-1.0",)) for one in (one_file, no_line)], me)

    for text, content in zip(texts, (one_file, no_line)):
        held = Parted(depot.reader(), text, depot.listed_text(text, 1)["parts"])
        cut = [pieces.read(held, n) for n in range(pieces.count(len(content)))]
        assert "".join(cut) == content
        starts = [sum(map(len, cut[:n])) for n in range(1, len(cut))]
        if content is one_file:  # the first line start from each multiple on
            assert all(content[at - 1] == "\n" and 0 <= at - n * pieces.LENGTH < 28 for n, at in enumerate(starts, 1))
        else:
            assert starts == [pieces.LENGTH, 2 * pieces.LENGTH]


def test_a_text_two_libraries_hold_comes_as_pieces_only_when_its_revision_in_the_first_is_long(make_depot, me,
                                                                                                tmp_path):
    # a locked library keeps a long revision of a text that an open library holds short now: the dataset takes
    # the revision of the first library that holds it, whole or as pieces, never both
    depot = make_depot()
    [text] = depot.add([NewText(text=repository(4, 300_000), licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    locked, open_ = depot.new_library("locked"), depot.new_library("open")
    depot.put(locked, [text])
    depot.lock(locked)
    depot.change_content(text, "a short text now", None, me)
    depot.decide("admitted", [text], me)
    depot.put(open_, [text])
    depot.new_dataset("open first", "classics", [(None, open_), (None, locked)])
    depot.new_dataset("locked first", "classics", [(None, locked), (None, open_)])
    store = open_store((tmp_path / "store").as_uri())

    found = {}
    for name in ("open first", "locked first"):
        record = build(depot, name, store, TOKENIZER, max_tokens=None, log=lambda *_: None)
        found[name] = [document for shard in record["shards"] for _, _, document, _, _ in read_verified(store, shard)]

    assert found["open first"] == [text]
    assert sorted(found["locked first"]) == sorted(pieces.name(text, n) for n in range(2))


def test_a_library_s_long_texts_are_read_from_the_long_contents_never_from_the_library_s_rows(make_depot, me):
    # the real depot holds 166 million texts: the build finds the long ones before it walks, through the index of
    # long content versions, each looked up by its key in the library and in the revisions
    depot, _, repo = long_depot(make_depot, me)
    library = depot.rows("SELECT id FROM libraries")[0]["id"]
    locked = depot.new_library("locked")
    depot.put(locked, [repo])
    depot.lock(locked)
    reader = depot.db.reader()
    traced = []
    reader.exec_trace = lambda cursor, sql, bindings: traced.append((sql, bindings)) or True

    found = {name: depot.library_long(name, depot.change()) for name in (library, locked)}

    reader.exec_trace = None
    assert [row[1] for row in found[library]] == [repo] and [row[1] for row in found[locked]] == [repo]
    reads = [one for one in traced if "content_lists_long" in one[0]]
    assert len(reads) == 2
    for sql, bindings in reads:
        plan = [row[3] for row in reader.execute(f"EXPLAIN QUERY PLAN {sql}", bindings)]
        assert plan[0].startswith("SCAN c USING INDEX content_lists_long"), plan
        held = "sqlite_autoindex_pins_1" if bindings["library"] == locked else "sqlite_autoindex_memberships_1"
        assert any(one.startswith(f"SEARCH h USING INDEX {held} (library=? AND text=?") for one in plan), plan
        # past the long contents, only the bounds' few values (json_each) are scanned
        assert not any(one.startswith("SCAN") and "VIRTUAL TABLE" not in one for one in plan[1:]), plan


def test_a_build_cut_short_right_after_a_piece_goes_on_and_writes_the_same_corpus(make_depot, me, tmp_path):
    # a token file closes after each piece here, so the state that resumes names a piece's place, and the size
    # leaves one piece out, so what each part took and left out is part of that state
    depot, short, repo = long_depot(make_depot, me)
    whole = build(depot, "all", open_store((tmp_path / "whole").as_uri()), TOKENIZER, max_tokens=None,
                  log=lambda *_: None)
    store = open_store((tmp_path / "whole").as_uri())
    tokens = {document: len(row) for shard in whole["shards"] for row, _, document, _, _ in
              read_verified(store, shard)}
    limit = whole["total_tokens"] - max(tokens[pieces.name(repo, n)] for n in range(4))
    straight = build(depot, "all", open_store((tmp_path / "straight").as_uri()), TOKENIZER, max_tokens=limit,
                     shard_tokens=5_000, log=lambda *_: None)
    saved = []

    def stop_after_a_piece(state):
        saved.append(state)
        if state["after"][3] >= 0:
            raise KeyboardInterrupt

    store = open_store((tmp_path / "store").as_uri())
    with pytest.raises(KeyboardInterrupt):
        build(depot, "all", store, TOKENIZER, max_tokens=limit, work=tmp_path / "work", save=stop_after_a_piece,
              shard_tokens=5_000, log=lambda *_: None)

    resumed = build(depot, "all", store, TOKENIZER, max_tokens=limit, work=tmp_path / "work", checkpoint=saved[-1],
                    shard_tokens=5_000, log=lambda *_: None)

    assert resumed["id"] == straight["id"] and resumed["pieces"] == 3
    assert [one["sha256"] for one in resumed["shards"]] == [one["sha256"] for one in straight["shards"]]
