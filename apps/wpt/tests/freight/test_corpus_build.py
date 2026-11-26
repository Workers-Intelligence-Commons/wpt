import base64
import hashlib
import json
import sqlite3

import pytest

from dock import open_dock
from wpt_corpus.shards import read_verified
from wpt_tokenizer import END, Tokenizer, chat
from depot import About, Depot, NewText, clone, ids, pull, snapshots
from depot import order, pieces
from freight import corpora
from freight.corpora import mark_stale, records
from wpt_corpus.corpora import find, fresh
from wpt_corpus.corpora import rows as corpus_rows
from corpus_build import build, build_key, census, task_text
from corpus_room import Room, TooBig
import corpus_room
from depot.errors import Refused
from depot.ids import export_key
from wpt.train.notice import model_notice
from depot.parts import Parted

TOKENIZER = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, vocab_size=560)
OTHER = Tokenizer.train([b"the quick brown fox jumps over the lazy cat"] * 4, vocab_size=561)
CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


HERE = "https://depot.example"  # the public address of the depot the order names


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


TASK = {"task": {"instructions": "Write hello into out.txt.", "image": "base",
                 "limits": {"max_turns": 4, "max_tokens": 512, "command_seconds": 10}, "needs": []},
        "files": {"notes.txt": {"mode": "100644", "content": encoded(b"say hello\n")}},
        "check": {"mode": "100755", "content": encoded(b"#!/bin/sh\necho 1\n")}}


def item(kind, content, *tags):
    return NewText(kind=kind, content=json.dumps(content), licenses=("CC0-1.0",), tags=tags)


def vocabulary(tokenizer=TOKENIZER) -> dict:
    return {"sha256": hashlib.sha256(tokenizer.to_bytes()).hexdigest(), "size": tokenizer.vocab_size}


def snapshot_of(depot) -> dict:
    """The depot's last snapshot, with its summary; one made now when it has none."""
    made = [one for one in depot.snapshots() if one["owner"] == depot.name()]
    return depot.snapshot(made[-1]["name"]) if made else depot.create_snapshot()


def section(snapshot, tokenizer=TOKENIZER, total_tokens=None, shares=(), kinds=(), **more) -> dict:
    """An order's data section over the snapshot, as an order names it."""
    return {"depot": HERE, "snapshot": snapshot["name"], "change": snapshot["at_change"],
            "digest": snapshot["digest"], "kinds": list(kinds),
            "shares": [{"tag": tag, "share": share} for tag, share in shares], "total_tokens": total_tokens,
            "vocabulary": vocabulary(tokenizer), **more}


def make(depot, store, snapshot=None, tokenizer=TOKENIZER, total_tokens=None, shares=(), kinds=(), **options):
    """The corpus of the snapshot (default: the depot's last, snapshot_of) with that size, shares and kinds."""
    snapshot = snapshot or snapshot_of(depot)
    options.setdefault("log", lambda *_: None)
    return build(depot, snapshot["id"], section(snapshot, tokenizer, total_tokens, shares, kinds), store, tokenizer,
                 address=HERE, **options)


def documents(store, record, kind="plain") -> list[str]:
    return [document for shard in record["shards"][kind] for _, _, document, _, _ in read_verified(store, shard)]


def test_a_corpus_holds_its_snapshot_s_texts_tokenized_with_their_credits_and_a_notice_line_each(make_depot, me,
                                                                                                  tmp_path):
    depot = make_depot("depot")
    same = NewText(text="the same words", licenses=("CC0-1.0",), title="Twice", authors=("Ann",), tags=("en",),
                   notice="(c) Ann")
    first, twin = depot.add([same, same], me)
    [french] = depot.add([NewText(text="le renard", licenses=("CC0-1.0",), tags=("fr",))], me)
    talk, task, probe = depot.add([item("conversation", CONVERSATION, "en"), item("task", TASK, "en"),
                                   item("probe", {"question": "q?", "patterns": ["x"]}, "en")], me)
    [waiting] = depot.add([NewText(text="not admitted", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", f"id <> '{waiting}'", me)
    snapshot = depot.create_snapshot()
    depot.retag([first, talk], ["later"], ["en"], me)  # after the snapshot: its corpus keeps the tags it read
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, code_commit="abc123", recipe="first")

    assert record["depot"] == HERE  # the order's address, recorded only
    assert record["snapshot"] == {"id": snapshot["id"], "name": snapshot["name"], "owner": "depot",
                                  "change": snapshot["at_change"], "digest": snapshot["digest"]}
    assert (record["recipe"], record["code_commit"], record["rules"]) == ("first", "abc123", 8)
    assert record["vocabulary"] == vocabulary() and record["max_tokens"] is None
    assert record["build_key"] == build_key(snapshot, section(snapshot), vocabulary()["sha256"])
    assert record["folder"] == f"corpora/{record['build_key']}"
    rows = [row for shard in record["shards"]["plain"] for row in read_verified(store, shard)]
    found = {(source, document): (list(tokens), pool) for tokens, source, document, pool, _ in rows}
    # every row names the depot by the snapshot's owner, never by an address or a file
    assert set(found) == {("depot", first), ("depot", twin), ("depot", french)}  # duplicates kept
    assert found[("depot", french)] == ([*TOKENIZER.encode("le renard".encode()), END], "")
    [conversation] = [row for shard in record["shards"]["conversation"] for row in read_verified(store, shard)]
    assert list(conversation[0]) == chat.encode(TOKENIZER, CONVERSATION)[0]
    [kept_task] = store.get_bytes(record["tasks"]["file"]).decode("utf-8").splitlines()
    assert json.loads(kept_task) == {"depot": "depot", "id": task, "content": TASK}
    assert record["left_out"] == {"no share": 0, "trimmed": 0, "trimmed items": 0,
                                  "kinds not taken": {"probe": 1}, "quoted text left out": 0, "quote unmatched": 0}

    texts = {(one["depot"], one["id"]): one for one in corpus_rows(store, record, "texts")}
    assert probe not in {id_ for _, id_ in texts} and waiting not in {id_ for _, id_ in texts}
    assert texts[("depot", first)]["title"] == "Twice" and texts[("depot", first)]["authors"] == ["Ann"]
    assert texts[("depot", first)]["tags"] == ["en"] and texts[("depot", talk)]["tags"] == ["en"]
    notices = {one["id"]: one["text"] for one in corpus_rows(store, record, "notices")}
    assert notices == {"depot-1": "(c) Ann"}  # ids carry the depot: copied as they are
    assert corpus_rows(store, record, "licenses") == [
        {"id": "CC0-1.0", "name": "Creative Commons Zero v1.0 Universal", "text": "The CC0 text."}]
    start = 0
    for tokens, source, document, _, _ in rows:  # each text's first token, over the token files in order
        assert texts[(source, document)]["tokens"] == start
        start += len(tokens)
    assert texts[("depot", talk)]["tokens"] == 0  # the one conversation starts the conversations' files

    lines = model_notice(store, record).split("\n\n")[0].splitlines()
    assert len(lines) == len(texts)
    assert f"depot {first}: Twice by Ann; CC0-1.0; notice {texts[('depot', first)]['notice']}" in lines
    # a model that learned less is credited with what it learned: the plain texts that start in the tokens it
    # read from the start, and the conversations or tasks when it learned from them
    _, source, document, _, _ = rows[0]
    assert model_notice(store, record, {"plain": 1}).split("\n\n")[0].splitlines() == [
        line for line in lines if line.startswith(f"{source} {document}:")]
    assert model_notice(store, record, {"conversation": 1}).split("\n\n")[0].splitlines() == [
        line for line in lines if line.startswith(f"depot {talk}:")]

    assert make(depot, store)["id"] == record["id"]


def test_a_order_s_kinds_keep_only_those_kinds_and_the_others_are_counted(make_depot, me, tmp_path):
    depot = make_depot()
    [plain] = depot.add([NewText(text="the fox", licenses=("CC0-1.0",), tags=("en",))], me)
    talk, task = depot.add([item("conversation", CONVERSATION, "en"), item("task", TASK, "en")], me)
    depot.decide("admitted", "true", me)
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, kinds=["plain", "task"])

    assert documents(store, record) == [plain] and record["conversations"] == 0 and record["tasks"]["count"] == 1
    assert record["left_out"]["kinds not taken"] == {"conversation": 1}
    other = make(depot, store, kinds=["task", "plain"])  # the same kinds in another order: the same build
    assert other["build_key"] == record["build_key"] and other["id"] == record["id"]


def quoting(source, exact, prefix=None):
    """A conversation item whose question quotes exact of the text source (docs/chat-format.md, "Excerpts")."""
    selector = {"type": "TextQuoteSelector", "exact": exact, **({} if prefix is None else {"prefix": prefix})}
    return item("conversation", {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                     {"type": "excerpt", "source": source, "selector": selector}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A fox."}]}]}, "en")


def test_a_conversation_comes_with_the_passages_it_quotes_and_only_when_the_corpus_holds_them(make_depot, me,
                                                                                               tmp_path):
    # a conversation brings out what the corpus taught, so one whose quoted text the corpus leaves out, or whose
    # quote no longer finds one place in the version it took, is left out with it, and counted
    depot = make_depot()
    fox, outside, twice = depot.add([NewText(text=words, licenses=("CC0-1.0",), title=title, tags=("en",))
                                     for words, title in (("the quick brown fox jumps", "Fox"),
                                                          ("the lazy dog sleeps", "Dog"), ("a fox and a dog", "Two"))],
                                    me)
    kept, left_out, unmatched = depot.add([quoting(fox, "brown fox", "quick "), quoting(outside, "lazy dog"),
                                           quoting(twice, "dog")], me)
    depot.change_content(twice, "a dog and a dog", None, me)  # its quote now finds two places
    depot.decide("admitted", f"id <> '{outside}'", me)  # the quoted text left out of the snapshot
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store)

    [row] = [row for shard in record["shards"]["conversation"] for row in read_verified(store, shard)]
    typed = json.loads(depot.content(kept))
    typed["messages"][0]["content"][1] = {"type": "text", "text": "brown fox"}
    assert (row[2], list(row[0])) == (kept, chat.encode(TOKENIZER, typed)[0])  # the passage in the block's place
    assert record["conversations"] == 1
    assert record["left_out"] == {"no share": 0, "trimmed": 0, "trimmed items": 0, "kinds not taken": {},
                                  "quoted text left out": 1, "quote unmatched": 1}
    texts = {one["id"]: one for one in corpus_rows(store, record, "texts")}
    assert left_out not in texts and unmatched not in texts
    assert texts[kept]["quotes"] == [fox] and texts[fox]["quotes"] is None
    # a model that learned the conversations alone is credited with the text it quotes too
    assert model_notice(store, record, {"conversation": 1}).split("\n\n")[0].splitlines() == [
        line for line in model_notice(store, record).split("\n\n")[0].splitlines()
        if line.startswith((f"depot {kept}:", f"depot {fox}:"))]
    # an erasure of the passage reaches the corpus through the quoted text it holds, and the depot's conversation;
    # the same snapshot builds again into the same folder, read cut
    erased = depot.ask_erasure("a word", [(fox, 10, 15)], me)
    depot.decide_erasure(erased, "admitted", me)
    assert mark_stale(store, [fox], [erased]) == [record["id"]]
    assert "brown" not in depot.content(kept)
    again = make(depot, store)
    assert again["folder"] == record["folder"] and again["id"] != record["id"] and fresh(again)
    assert again["erasures"] == [erased]
    [row] = [row for shard in again["shards"]["conversation"] for row in read_verified(store, shard)]
    assert "brown" not in chat.show(TOKENIZER, list(row[0])) and " fox" in chat.show(TOKENIZER, list(row[0]))
    # the files of the record it replaced that it does not name are gone
    named = {one["key"] for one in corpora.files(again)}
    assert all(store.has(corpora.File.of(one)) == (one["key"] in named) for one in corpora.files(record))


def test_a_build_of_quoting_conversations_cut_short_anywhere_goes_on_to_the_same_corpus(make_depot, me, tmp_path):
    # the conversations are taken once the walk ends, from what the work folder kept of it: a stop in the walk or
    # among the conversations goes on to the same bytes
    depot = make_depot()
    texts = depot.add([NewText(text=f"the dog number {n} runs", licenses=("CC0-1.0",), tags=("en",))
                       for n in range(6)], me)
    depot.add([quoting(text, f"number {n}") for n, text in enumerate(texts)], me)
    depot.decide("admitted", f"id <> '{texts[0]}'", me)
    whole = make(depot, open_dock((tmp_path / "whole").as_uri()), shard_tokens=20)
    saved = []
    make(depot, open_dock((tmp_path / "count").as_uri()), shard_tokens=20, save=saved.append)
    assert (whole["conversations"], whole["left_out"]["quoted text left out"]) == (5, 1) and len(saved) > 4

    for stop in range(1, len(saved)):
        kept = []

        def stopping(state):
            kept.append(state)
            if len(kept) == stop:
                raise KeyboardInterrupt
        store, work = open_dock((tmp_path / f"store{stop}").as_uri()), tmp_path / f"work{stop}"
        with pytest.raises(KeyboardInterrupt):
            make(depot, store, work=work, save=stopping, shard_tokens=20)

        resumed = make(depot, store, work=work, checkpoint=kept[-1], shard_tokens=20)

        assert resumed["id"] == whole["id"] and resumed["left_out"] == whole["left_out"], stop


def test_quoting_conversations_read_on_workers_page_by_page_make_the_corpus_one_process_makes(make_depot, me,
                                                                                               tmp_path, monkeypatch):
    # a big build reads the conversations on its workers and each quoted text once for a page of them: several
    # quoting one text, some of whose quotes find no place or two, across pages, must give the same bytes
    depot = make_depot()
    dog, cat = depot.add([NewText(text=words, licenses=("CC0-1.0",), tags=("en",))
                          for words in ("the dog runs and the dog sleeps", "the cat sits")], me)
    depot.add([quoting(dog, "runs"), quoting(cat, "cat"), quoting(dog, "sleeps", "dog "), quoting(cat, "sits"),
               quoting(dog, "runs")], me)
    depot.change_content(dog, "the dog sleeps and the dog sleeps", None, me)  # "runs" gone, "dog sleeps" twice
    depot.decide("admitted", "true", me)
    alone = make(depot, open_dock((tmp_path / "alone").as_uri()))
    monkeypatch.setattr("corpus_build.IN_PROCESS", 0)
    monkeypatch.setattr("corpus_build.HELD_PAGE", 2)
    import corpus_build as corpus
    ran, run = [], corpus.in_order

    def watched(work, *rest, **options):
        ran.append(work.__name__)
        return run(work, *rest, **options)
    monkeypatch.setattr(corpus, "in_order", watched)

    spread = make(depot, open_dock((tmp_path / "spread").as_uri()))

    assert {"_conversations", "_matched"} <= set(ran)
    assert (alone["conversations"], alone["left_out"]["quote unmatched"]) == (2, 3)
    assert spread["id"] == alone["id"] and spread["left_out"] == alone["left_out"]
    assert [one["sha256"] for one in spread["shards"]["conversation"]] == [
        one["sha256"] for one in alone["shards"]["conversation"]]


def repository(files: int, file_characters: int) -> str:
    """A repository as the depot holds one, as one text: each file a line `=== <path>` and its lines."""
    line = "the quick brown fox jumps over the lazy dog\n"
    return "".join(f"=== src/file{n}.c\n" + line * (file_characters // len(line)) for n in range(files))


def long_depot(make_depot, me, *more):
    """A depot of 200 short texts and a repository about 3.4 times pieces.LENGTH long, cut into 4 pieces, and the
    texts more, all admitted: (the depot, the short texts' ids, the repository's, the ids of more)."""
    depot = make_depot()
    [repo] = depot.add([NewText(text=repository(12, 300_000), licenses=("CC0-1.0",), title="A repository",
                                tags=("code",))], me)
    short = depot.add([NewText(text=f"the dog {n}", licenses=("CC0-1.0",), tags=("en",)) for n in range(200)], me)
    added = depot.add(list(more), me) if more else []
    depot.decide("admitted", "true", me)
    return depot, short, repo, added


def test_a_quote_of_a_long_text_needs_the_pieces_its_passage_lies_in(make_depot, me, tmp_path):
    depot, _, repo, _ = long_depot(make_depot, me)
    text = depot.content(repo)
    held = Parted(depot.reader(), repo, depot.listed_text(repo, 1)["parts"])
    starts = [0]
    for n in range(pieces.count(len(text))):
        starts.append(starts[-1] + len(pieces.read(held, n)))
    for start, end in ((0, 1), (starts[1] - 1, starts[1] + 1), (starts[2], starts[3] + 5), (len(text) - 1, len(text))):
        assert pieces.holding(held, start, end) == [n for n in range(len(starts) - 1)
                                                    if starts[n] < end and starts[n + 1] > start]
    [talk] = depot.add([quoting(repo, "=== src/file7.c")], me)
    depot.decide("admitted", [talk], me)

    record = make(depot, open_dock((tmp_path / "store").as_uri()))

    assert record["conversations"] == 1 and record["pieces"] == 4


def shared_depot(make_depot, me, english, french):
    """A depot of `english` texts tagged en and `french` texts tagged fr, each text 2 tokens long (plus END),
    a text tagged de and a conversation, all admitted."""
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(english)]
              + [NewText(text="the fox", licenses=("CC0-1.0",), tags=("fr",)) for _ in range(french)]
              + [NewText(text="the dog", licenses=("CC0-1.0",), tags=("de",)), item("conversation", CONVERSATION, "en")],
              me)
    depot.decide("admitted", "true", me)
    return depot


def test_shares_trim_to_their_proportions_keeping_the_scarcest_whole_and_a_seeded_subset_of_the_others(
        make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=12, french=3)
    store = open_dock((tmp_path / "store").as_uri())
    shares = [("en", 0.5), ("fr", 0.5)]

    record = make(depot, store, shares=shares)

    rows = [row for shard in record["shards"]["plain"] for row in read_verified(store, shard)]
    by_pool = {}
    for tokens, _, document, pool, _ in rows:
        by_pool.setdefault(pool, []).append(document)
    assert sorted(map(len, by_pool.values())) == [3, 3]  # french is the scarcest: all 3, as many english
    assert len(set(by_pool["en"])) == 3  # no text repeated
    tokens = record["shares_tokens"][1]["tokens"]
    alone = [{"tag": None, "texts": 3, "tokens": tokens}]  # each share's texts carry no other tag: one group
    assert record["shares_tokens"] == [{"tag": "en", "tokens": tokens, "groups": alone},
                                       {"tag": "fr", "tokens": tokens, "groups": alone}]
    assert record["left_out"]["no share"] == 1 and record["left_out"]["trimmed"] == 9
    assert record["conversations"] == 1  # an item is no plain text: shares leave it whole
    again = make(depot, open_dock((tmp_path / "other").as_uri()), shares=shares)
    assert again["id"] == record["id"]  # the order makes the same subset


def test_shares_naming_a_tag_no_text_has_refuse_the_build(make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=2, french=0)

    with pytest.raises(Refused, match="tag fr"):
        make(depot, open_dock((tmp_path / "store").as_uri()), shares=[("en", 0.5), ("fr", 0.5)])

    assert records(open_dock((tmp_path / "store").as_uri())) == []


def test_an_erasure_marks_stale_every_corpus_holding_its_text_and_a_new_build_is_fresh(make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=2, french=2)
    store = open_dock((tmp_path / "store").as_uri())
    record = make(depot, store)
    [text] = [row["id"] for row in depot.sql("SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(tags) "
                                             "WHERE value = 'de')")]
    assert "stale" not in record

    assert mark_stale(store, ["depot-99"], ["depot-1"]) == []
    assert mark_stale(store, [text], ["depot-1"]) == [record["id"]]

    assert records(store)[0]["stale"] == {"erasures": ["depot-1"]}


def test_a_snapshot_builds_the_same_corpus_whatever_its_texts_did_since(make_depot, me, tmp_path):
    depot = make_depot()
    first, second = depot.add([NewText(text="the fox", licenses=("CC0-1.0",), title="Fox"),
                               NewText(text="the dog", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", [first], me)
    snapshot = depot.create_snapshot()
    store = open_dock((tmp_path / "store").as_uri())
    before = make(depot, store, snapshot)
    depot.decide("admitted", [second], me)
    depot.edit_credits(first, {"title": "The fox"}, me)
    depot.decide("refused", [first], me)
    depot.add([NewText(text="a new text", licenses=("CC0-1.0",))], me)

    again = make(depot, open_dock((tmp_path / "other").as_uri()), snapshot)

    assert again["id"] == before["id"] and again["build_key"] == before["build_key"]
    assert [(one["id"], one["title"]) for one in corpus_rows(store, again, "texts")] == [(first, "Fox")]


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
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store)

    texts = {one["id"]: one["scores"] for one in corpus_rows(store, record, "texts")}
    assert texts == {scored: [{"name": "machine-made", "value": 0.03, "content_version": 2},
                              {"name": "rank", "value": -1.5, "content_version": 1}], unscored: None}


def test_an_erasure_carried_out_while_a_corpus_is_built_makes_it_stale_once_written_and_leaves_the_others(
        make_depot, me, tmp_path):
    # the erasure's own marking reads only the records that exist, and the build's record does not exist yet
    depot = shared_depot(make_depot, me, english=2, french=0)
    store = open_dock((tmp_path / "store").as_uri())
    older = make(depot, store)
    [text] = [row["id"] for row in depot.sql("SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(tags) "
                                             "WHERE value = 'de')")]
    asked = []

    def erase_while_finishing(state):
        if state["stage"] == "finishing":
            asked.append(depot.ask_erasure("a dog", [(text, 4, 7)], me))
            assert depot.decide_erasure(asked[0], "admitted", me) is True

    record = make(depot, store, total_tokens=1000, progress=erase_while_finishing)

    assert record["stale"] == {"erasures": asked}
    kept = {one["id"]: one for one in records(store)}
    assert kept[record["id"]]["stale"] == {"erasures": asked}
    assert fresh(kept[older["id"]])  # nothing deletes another key's corpus


def test_a_corpus_takes_its_texts_in_the_order_of_the_export_keys_each_once_and_a_smaller_size_takes_that_order(
        make_depot, me, tmp_path, monkeypatch):
    # the walk reads the order a page at a time: with pages of 3 texts, the 20 texts come in the order of their
    # export keys, each once, and a smaller size takes the start of what a larger one takes, leaving out only what
    # does not fit
    monkeypatch.setattr(order, "PAGE", 3)
    depot = make_depot()
    added = depot.add([NewText(text=f"text {'n' * (n % 5)} {n}", licenses=("CC0-1.0",), tags=("en",))
                       for n in range(20)], me)
    depot.decide("admitted", "true", me)
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store)

    taken = documents(store, record)
    assert taken == sorted(added, key=export_key)
    for size in (record["total_tokens"] // 2, record["total_tokens"] // 3):
        smaller = make(depot, store, total_tokens=size)
        cut = documents(store, smaller)
        assert 0 < len(cut) < len(taken) and cut == [text for text in taken if text in set(cut)]
        first_left_out = next(at for at, text in enumerate(taken) if text not in set(cut))
        assert cut[:first_left_out] == taken[:first_left_out]  # the same texts up to the first it leaves out
        assert smaller["build_key"] != record["build_key"]


def test_the_census_comes_from_the_summary_equals_a_count_of_the_texts_and_stays_after_an_erasure(make_depot, me):
    depot = make_depot()
    added = depot.add([NewText(text=f"the quick brown fox {n}", licenses=("CC0-1.0",),
                               tags=("en", "fr") if n % 3 == 0 else ("en",) if n % 3 == 1 else ("fr", "de"))
                       for n in range(30)]
                      + [NewText(text="no tag", licenses=("CC0-1.0",))]
                      + [item("conversation", CONVERSATION), item("task", TASK), item("probe", {"question": "q?", "patterns": ["x"]})], me)
    depot.decide("admitted", "true", me)
    snapshot = snapshot_of(depot)
    shares = [{"tag": "fr", "share": 0.4}, {"tag": "en", "share": 0.6}]

    def recount(kinds=None) -> dict:
        texts = depot.snapshot_texts(snapshot["id"], limit=10_000)["texts"]
        pool_bytes, pool_texts, no_share = {0: 0, 1: 0}, {0: 0, 1: 0}, 0
        for one in texts:
            if one["kind"] != "plain" or (kinds and "plain" not in kinds):
                continue
            pool = next((at for at, share in enumerate(shares) if share["tag"] in one["tags"]), None)
            if pool is None:
                no_share += 1
            else:
                pool_bytes[pool] += one["bytes"]
                pool_texts[pool] += 1
        items = [one for one in texts if one["kind"] in ("conversation", "task") and (not kinds or one["kind"] in kinds)]
        return {"pool_bytes": pool_bytes, "pool_texts": pool_texts, "no_share": no_share,
                "items": len(items), "item_bytes": sum(one["bytes"] for one in items), "texts": len(texts)}

    def wanted(found: dict) -> dict:
        return {key: found[key] for key in ("pool_bytes", "pool_texts", "no_share", "items", "item_bytes", "texts")}

    found = census(snapshot["summary"], shares)
    assert wanted(found) == recount() and found["other"] == 1 and found["plain"] == 30
    assert wanted(census(snapshot["summary"], shares, ["plain"])) == recount(["plain"])
    assert census(snapshot["summary"], shares, ["plain"])["other"] == 3
    # an erasure removes a text whole: the snapshot's census, from its summary, is the same as before
    erased = depot.ask_erasure("whole", [(added[1], None, None)], me)
    depot.decide_erasure(erased, "admitted", me)
    assert depot.snapshot_texts(snapshot["id"], limit=10_000)["erased"] == 1
    assert census(depot.snapshot(snapshot["name"])["summary"], shares) == found


def test_a_text_that_would_pass_the_size_is_left_out_and_the_texts_after_it_fill_the_corpus(make_depot, me,
                                                                                            tmp_path):
    depot = make_depot()
    [long] = depot.add([NewText(text="the quick brown fox jumps over the lazy dog " * 8, licenses=("CC0-1.0",),
                                tags=("en",))], me)
    short = depot.add([NewText(text=f"the dog {n}", licenses=("CC0-1.0",), tags=("en",)) for n in range(10)], me)
    depot.decide("admitted", "true", me)
    store = open_dock((tmp_path / "store").as_uri())
    whole = make(depot, store)
    rows = [(document, len(tokens)) for shard in whole["shards"]["plain"] for tokens, _, document, _, _ in
            read_verified(store, shard)]
    taken_order = [document for document, _ in rows]
    assert taken_order.index(long) < len(taken_order) - 1  # some short texts come after the long one
    limit = whole["total_tokens"] - dict(rows)[long]  # every short text fits, the long one does not

    record = make(depot, store, total_tokens=limit)

    taken = documents(store, record)
    assert taken == [one for one in taken_order if one != long] and sorted(taken) == sorted(short)
    assert record["total_tokens"] == limit and record["left_out"]["trimmed"] == 1


def items_depot(make_depot, me, plain: int):
    """A depot of `plain` texts tagged en (each 3 tokens with END), two conversations and one task, all
    admitted; and the tokens a conversation and the task count."""
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(plain)]
              + [item("conversation", CONVERSATION, "en"), item("conversation", CONVERSATION, "en"),
                 item("task", TASK, "en")], me)
    depot.decide("admitted", "true", me)
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    task = len(TOKENIZER.encode(task_text(TASK).encode("utf-8")))
    return depot, conversation, task


def corpus_tokens(store, record) -> int:
    """Every token the corpus holds: its plain texts', its conversations' and its tasks' (task_text)."""
    tasks = store.get_bytes(record["tasks"]["file"]).decode("utf-8").splitlines() if "tasks" in record else []
    return (record["total_tokens"] + sum(one["tokens"] for one in record["shards"]["conversation"])
            + sum(len(TOKENIZER.encode(task_text(json.loads(line)["content"]).encode("utf-8"))) for line in tasks))


def items_part(depot, limit: int) -> int:
    """The items' share of a size: their part of the UTF-8 bytes of every text of the depot, of limit."""
    rows = depot.rows("SELECT c.kind, c.bytes AS size FROM texts t JOIN content_lists c "
                      "ON c.id = t.id AND c.version = t.content_version")
    items = sum(row["size"] for row in rows if row["kind"] != "plain")
    return limit * items // sum(row["size"] for row in rows)


def test_a_size_gives_the_items_their_natural_share_and_the_plain_texts_the_rest(make_depot, me, tmp_path):
    depot, conversation, task = items_depot(make_depot, me, plain=300)
    store = open_dock((tmp_path / "store").as_uri())
    every = make(depot, store)
    limit = every["items_tokens"]  # the items' share of it leaves some of them out, the plain texts' part fills
    part = items_part(depot, limit)
    assert part < limit

    record = make(depot, store, total_tokens=limit)

    # every token: every item and every plain text, the items' tokens counted
    assert (every["conversations"], every["tasks"]["count"], every["total_tokens"]) == (2, 1, 300 * 3)
    assert every["items_tokens"] == 2 * conversation + task == corpus_tokens(store, every) - every["total_tokens"]
    # the items take their share of the size, within one item, and the plain texts what is left, within one text
    assert part - max(conversation, task) < record["items_tokens"] <= part
    assert record["items_tokens"] == corpus_tokens(store, record) - record["total_tokens"]
    assert limit - record["items_tokens"] - 3 < record["total_tokens"] <= limit - record["items_tokens"]
    # the same snapshot, data section and vocabulary build the same corpus into the same folder
    again = make(depot, open_dock((tmp_path / "again").as_uri()), total_tokens=limit)
    assert (again["id"], again["folder"]) == (record["id"], record["folder"])


def test_with_shares_the_items_have_their_share_of_the_corpus_of_every_token(make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(300)]
              + [NewText(text="the fox", licenses=("CC0-1.0",), tags=("fr",)) for _ in range(10)]
              + [item("conversation", CONVERSATION, "en") for _ in range(6)], me)
    depot.decide("admitted", "true", me)
    shares = [("en", 0.5), ("fr", 0.5)]
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    items = 6 * len(json.dumps(CONVERSATION).encode("utf-8"))
    store = open_dock((tmp_path / "store").as_uri())
    every = make(depot, store, shares=shares)
    limit = every["items_tokens"]
    part = limit * items // (items + 2 * 10 * 7)  # without a size, en keeps as many bytes as fr: 10 texts of 7

    record = make(depot, store, total_tokens=limit, shares=shares)

    assert [one["tokens"] for one in every["shares_tokens"]] == [30, 30]
    assert part - conversation < record["items_tokens"] <= part
    assert record["items_tokens"] > limit * items // (items + 310 * 7)  # more than their share of every plain text


def test_the_item_share_stops_at_its_first_item_that_would_pass_its_part(make_depot, me, tmp_path):
    depot, conversation, task = items_depot(make_depot, me, plain=3)
    limit = 3 * 3 + conversation + task  # the items hold nearly all the bytes: their part is most of the size
    part = items_part(depot, limit)
    assert part < 2 * conversation + task  # so not every item fits
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, total_tokens=limit)

    taken = record["conversations"] + record.get("tasks", {}).get("count", 0)
    assert 0 < taken < 3 and record["left_out"]["trimmed items"] == 3 - taken
    assert record["items_tokens"] <= part and corpus_tokens(store, record) <= limit


def test_conversations_and_tasks_are_one_share_where_an_item_that_does_not_fit_leaves_room_for_the_next(
        make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(20)]
              + [item("conversation", CONVERSATION, "en") for _ in range(5)] + [item("task", TASK, "en"),
                                                                                item("task", TASK, "en")], me)
    depot.decide("admitted", "true", me)
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    task = len(TOKENIZER.encode(task_text(TASK).encode("utf-8")))
    store = open_dock((tmp_path / "store").as_uri())
    every = make(depot, store)
    kinds = [one["kind"] for one in corpus_rows(store, every, "texts") if one["kind"] != "plain"]
    assert task > conversation
    if "conversation" not in kinds[kinds.index("task") + 1:]:
        pytest.skip("this order puts no conversation after the first task")
    before = kinds.index("task")  # the conversations the corpus's order puts before the first task
    # the smallest size whose item part fits every conversation before the first task and one more, not the task
    limit = next(size for size in range(1, 100_000) if items_part(depot, size) >= (before + 1) * conversation)
    part = items_part(depot, limit)
    assert part < before * conversation + task

    record = make(depot, store, total_tokens=limit)

    # the first task does not fit and is left out; the conversations after it fill what is left of the part
    assert record.get("tasks", {}).get("count", 0) == 0
    assert record["conversations"] == min(kinds.count("conversation"), part // conversation) > before
    assert record["items_tokens"] == record["conversations"] * conversation
    assert record["left_out"]["trimmed items"] == len(kinds) - record["conversations"]


def test_a_snapshot_of_items_alone_fills_its_size_with_them(make_depot, me, tmp_path):
    depot, conversation, task = items_depot(make_depot, me, plain=0)
    limit = 2 * conversation + task - 1  # every item but the last in order fits
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, total_tokens=limit)

    assert record["total_tokens"] == 0
    assert limit - max(conversation, task) < record["items_tokens"] <= limit
    assert record["items_tokens"] == corpus_tokens(store, record)
    assert record["left_out"]["trimmed items"] == 1


def same_files(one: dict, other: dict) -> bool:
    return (all([s["sha256"] for s in one["shards"][kind]] == [s["sha256"] for s in other["shards"][kind]]
                for kind in ("plain", "conversation"))
            and one.get("tasks", {}).get("file", {}).get("sha256") == other.get("tasks", {}).get("file", {}).get("sha256")
            and [s["sha256"] for s in one["tables"]["texts"]] == [s["sha256"] for s in other["tables"]["texts"]])


def test_a_depot_moved_to_another_file_builds_the_same_corpus(make_depot, me, tmp_path):
    depot, _, _ = items_depot(make_depot, me, plain=4)
    first = make(depot, open_dock((tmp_path / "first").as_uri()))
    moved = tmp_path / "elsewhere" / "moved.sqlite"
    moved.parent.mkdir()
    with sqlite3.connect(depot.address) as source, sqlite3.connect(moved) as target:
        source.backup(target)

    second = make(Depot.view(str(moved)), open_dock((tmp_path / "second").as_uri()), snapshot_of(depot))

    assert second["id"] == first["id"] and second["folder"] == first["folder"] and same_files(first, second)


def test_a_clone_holding_its_upstream_s_snapshot_builds_the_same_corpus_under_the_same_key(make_depot, me, tmp_path,
                                                                                           serve):
    upstream, _, _ = items_depot(make_depot, me, plain=4)
    address = serve(upstream)
    follower = clone(address, str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    made = upstream.create_snapshot()
    pull(follower)
    built = make(upstream, open_dock((tmp_path / "upstream").as_uri()), made)

    again = build(follower, made["id"], section(made), open_dock((tmp_path / "follower").as_uri()), TOKENIZER,
                  address="https://a-clone.example", log=lambda *_: None)

    assert again["build_key"] == built["build_key"] and again["id"] == built["id"] and same_files(built, again)
    assert again["depot"] == "https://a-clone.example"  # the address it was asked with, recorded only


def test_a_build_cut_short_goes_on_from_its_last_token_file_and_writes_the_same_corpus(make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=40, french=0)
    whole = make(depot, open_dock((tmp_path / "whole").as_uri()), shard_tokens=20)
    saved = []

    def stop_after_two(state):
        saved.append(state)
        if len(saved) == 2:
            raise KeyboardInterrupt

    store = open_dock((tmp_path / "store").as_uri())
    with pytest.raises(KeyboardInterrupt):
        make(depot, store, work=tmp_path / "work", save=stop_after_two, shard_tokens=20)
    assert records(store) == []  # no record until the build is whole
    assert saved[-1]["key"] == whole["build_key"]  # the state is keyed by the build's key

    resumed = make(depot, store, work=tmp_path / "work", checkpoint=saved[-1], shard_tokens=20)

    assert resumed["id"] == whole["id"] and same_files(resumed, whole)
    # a state of another build starts nothing of it: this one starts over
    other = make(depot, open_dock((tmp_path / "other").as_uri()), total_tokens=30, work=tmp_path / "work2",
                 checkpoint=saved[-1], shard_tokens=20)
    assert other["build_key"] != whole["build_key"] and other["total_tokens"] <= 30


def test_a_build_stopped_before_an_erasure_that_reached_its_snapshot_starts_over(make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=40, french=0)
    saved = []

    def stop_after_two(state):
        saved.append(state)
        if len(saved) == 2:
            raise KeyboardInterrupt

    store = open_dock((tmp_path / "store").as_uri())
    with pytest.raises(KeyboardInterrupt):
        make(depot, store, work=tmp_path / "work", save=stop_after_two, shard_tokens=20)
    [text] = [row["id"] for row in depot.sql("SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(tags) "
                                             "WHERE value = 'de')")]
    erased = depot.ask_erasure("a dog", [(text, 4, 7)], me)
    depot.decide_erasure(erased, "admitted", me)
    lines = []

    resumed = make(depot, store, work=tmp_path / "work", checkpoint=saved[-1], shard_tokens=20, log=lines.append)

    assert any("starts over" in line for line in lines)
    assert resumed["id"] == make(depot, open_dock((tmp_path / "fresh").as_uri()), shard_tokens=20)["id"]
    assert resumed["erasures"] == [erased]


def test_a_snapshot_change_digest_or_vocabulary_other_than_the_order_s_is_refused(make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=2, french=0)
    snapshot = snapshot_of(depot)
    store = open_dock((tmp_path / "store").as_uri())

    def refused(data, tokenizer=TOKENIZER, match=None):
        with pytest.raises(Refused, match=match):
            build(depot, snapshot["id"], data, store, tokenizer, log=lambda *_: None)

    refused(section(snapshot, change=snapshot["at_change"] + 1),
            match=f"snapshot {snapshot['name']} of depot depot reads change {snapshot['at_change']} with digest {snapshot['digest']}; "
                  f"the order says {snapshot['at_change'] + 1}, {snapshot['digest']}")
    refused(section(snapshot, digest="0" * 64), match="the order says .*, 0{64}")
    refused(section(snapshot, vocabulary={"sha256": vocabulary()["sha256"], "size": 999}),
            match=f"the vocabulary given has sha256 {vocabulary()['sha256']} and size {TOKENIZER.vocab_size}; the order says .*, 999")
    refused(section(snapshot), OTHER, match="the vocabulary given has sha256")
    refused(section(snapshot, total_tokens=0), match="whole number of tokens above 0")
    refused(section(snapshot, shares=[("en", 1), ("en", 1)]), match="name a tag twice")
    assert records(store) == []
    assert fresh(build(depot, snapshot["id"], {"vocabulary": vocabulary()}, store, TOKENIZER,
                               log=lambda *_: None))  # a section that names no change nor digest checks none


def test_a_cap_deletes_the_corpus_downloaded_least_recently_and_refuses_a_corpus_bigger_than_itself(
        make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=3, french=3)
    store = open_dock((tmp_path / "store").as_uri())
    english = make(depot, store, shares=[("en", 1.0)])
    french = make(depot, store, shares=[("fr", 1.0)])
    size = sum(one["bytes"] for one in corpora.files(english))
    downloaded = {english["id"]: 2.0, french["id"]: 1.0}  # french was downloaded least recently
    cap = 2 * size + size // 2
    whole = make(depot, open_dock((tmp_path / "alone").as_uri()))
    whole_size = sum(one["bytes"] for one in corpora.files(whole))
    # what the cap would delete, told before the build: the corpus it then deletes, nothing deleted yet
    foretold = Room(store, cap, "corpora/next", downloaded.get, set).evicted(whole_size)
    assert [one["corpus"] for one in foretold] == [french["id"]] and fresh(find(store, french["id"]))
    assert Room(store, cap, "corpora/next", downloaded.get, lambda: {french["id"]}).evicted(whole_size)[0]["corpus"] \
        == english["id"]  # one being downloaded is never deleted
    assert Room(store, None, "corpora/next").evicted(10**12) == [] and Room(store, 10, "corpora/next").evicted(11) is None

    record = make(depot, store, room=lambda folder: Room(store, cap, folder, downloaded.get, set))

    kept = {one["id"]: one for one in records(store)}
    assert "deleted" in kept[french["id"]] and "deleted" not in kept[english["id"]]
    assert fresh(kept[record["id"]])
    with pytest.raises(TooBig):
        make(depot, store, total_tokens=4, shares=[("fr", 1.0)], room=lambda folder: Room(store, 10, folder))


def test_a_cap_weighs_only_the_items_the_size_keeps(make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)) for _ in range(20)]
              + [item("conversation", CONVERSATION, "en") for _ in range(200)], me)
    depot.decide("admitted", "true", me)
    conversation = len(chat.encode(TOKENIZER, CONVERSATION)[0])
    limit = 2 * conversation + conversation // 2  # the conversations' part holds two of them
    alone = make(depot, open_dock((tmp_path / "alone").as_uri()), total_tokens=limit, shard_tokens=6)
    size = sum(one["bytes"] for one in corpora.files(alone))
    store = open_dock((tmp_path / "store").as_uri())

    # the 198 conversations the limit leaves out take no room in the estimate a cap of ten times the size checks
    record = make(depot, store, total_tokens=limit, shard_tokens=6, room=lambda folder: Room(store, 10 * size, folder))

    assert record["id"] == alone["id"] and record["conversations"] == 2


def test_a_corpus_bigger_than_the_cap_is_refused_before_any_other_corpus_is_deleted_for_it(make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=40, french=2)
    whole = make(depot, open_dock((tmp_path / "whole").as_uri()), shard_tokens=20)
    size = sum(one["bytes"] for one in corpora.files(whole))
    store = open_dock((tmp_path / "store").as_uri())
    french = make(depot, store, shares=[("fr", 1.0)])

    with pytest.raises(TooBig, match="no corpus was deleted"):
        make(depot, store, shard_tokens=20, room=lambda folder: Room(store, size // 2, folder))

    assert fresh(records(store)[0]) and records(store)[0]["id"] == french["id"]


def test_a_cap_deletes_the_files_a_build_left_without_a_record_and_that_build_started_again_starts_over(
        make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=40, french=0)
    whole = make(depot, open_dock((tmp_path / "whole").as_uri()), shard_tokens=20)
    store = open_dock((tmp_path / "store").as_uri())
    saved = []

    def stop_after_five(state):
        saved.append(state)
        if len(saved) == 5:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        make(depot, store, work=tmp_path / "work", save=stop_after_five, shard_tokens=20)
    [left] = [key for key in store.kind.list("corpora/", deep=False)]
    left_bytes = sum(store.kind.list(left, deep=True).values())

    # another key: a new folder, in one token file, whose first upload is all of it (its estimate is its size)
    other = make(depot, open_dock((tmp_path / "other").as_uri()), total_tokens=10**6, shard_tokens=1_000)
    other_size = sum(one["bytes"] for one in corpora.files(other))
    record = make(depot, store, total_tokens=10**6, shard_tokens=1_000,
                  room=lambda folder: Room(store, other_size + left_bytes - 1, folder))

    assert fresh(record)
    assert sum(store.kind.list(left, deep=True).values()) < left_bytes  # a file at a time, as the build needed room
    resumed = make(depot, store, work=tmp_path / "work", checkpoint=saved[-1], shard_tokens=20)
    assert resumed["id"] == whole["id"] and not corpus_room.missing(store, resumed["folder"], corpora.files(resumed))


def test_an_erasure_of_a_text_the_snapshot_does_not_hold_leaves_its_corpus_s_folder_and_id_as_they_were(
        make_depot, me, tmp_path):
    depot = shared_depot(make_depot, me, english=3, french=0)
    [outside] = depot.add([NewText(text="the cat", licenses=("CC0-1.0",))], me)  # under review: not in the snapshot
    store = open_dock((tmp_path / "store").as_uri())
    first = make(depot, store)
    corpus_room.delete(store, first, "a test deleted it")
    request = depot.ask_erasure("a cat", [(outside, 4, 7)], me)
    assert depot.decide_erasure(request, "admitted", me) is True

    again = make(depot, store)

    assert (again["id"], again["folder"]) == (first["id"], first["folder"]) and fresh(again)
    assert again["erasures"] == []
    assert find(store, first["id"])["id"] == first["id"]  # one record of that id


def two_parts(depot, parts):
    """Write the snapshot "two" of depot with those sets (part, owner, bound, bitmap), as an upgrade writes the
    snapshot of a dataset reading several libraries: a part per library."""
    late = max(bound for _, _, bound, _ in parts)
    summary = depot.create_snapshot()["summary"]
    with depot.db.write("snapshot") as (con, number):
        made = depot._id(depot._take(con, "snapshots"))
        con.execute(f"INSERT INTO snapshots ({snapshots.COLUMNS}) VALUES (?, ?, 'two', ?, 'now', ?, NULL, "
                    "2, 0, 0, ?, ?)", (made, number, late, json.dumps({"kind": "made"}),
                                       snapshots.digest_of(parts, late), json.dumps(summary)))
        con.executemany("INSERT INTO snapshot_sets (snapshot, part, owner, bound, bitmap, texts) "
                        "VALUES (?, ?, ?, ?, ?, 1)", [(made, p, o, b, m.serialize()) for p, o, b, m in parts])


def test_a_number_two_parts_hold_is_built_once_at_its_first_part_s_bound(make_depot, me, tmp_path):
    # a snapshot an upgrade makes from a dataset has a part per library, each read at its own change
    depot = make_depot()
    first, second = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",)) for n in range(2)], me)
    depot.decide("admitted", [first, second], me)
    early = depot.change()
    depot.change_content(first, "text zero, later", None, me)
    depot.decide("admitted", [first], me)
    late = depot.change()
    two_parts(depot, [(0, "depot", early, snapshots.BitMap([ids.number(first)])),
                      (1, "depot", late, snapshots.BitMap([ids.number(first), ids.number(second)]))])
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, depot.snapshot("two"))

    texts = {one["id"]: one["revision"] for one in corpus_rows(store, record, "texts")}
    assert texts == {first: 1, second: 1}  # first read at the bound of part 0, before its change
    [row] = [row for shard in record["shards"]["plain"] for row in read_verified(store, shard) if row[2] == first]
    assert list(row[0]) == [*TOKENIZER.encode(b"text 0"), END]


def test_an_erasure_of_a_text_only_an_earlier_part_holds_is_one_the_corpus_lists(make_depot, me, tmp_path):
    # the parts of one owner hold different texts: an erasure counts when any of them holds its text
    depot = make_depot()
    first, second = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",)) for n in range(2)], me)
    depot.decide("admitted", [first, second], me)
    at = depot.change()
    two_parts(depot, [(0, "depot", at, snapshots.BitMap([ids.number(first)])),
                      (1, "depot", at, snapshots.BitMap([ids.number(second)]))])
    request = depot.ask_erasure("a word", [(first, 0, 4)], me)
    assert depot.decide_erasure(request, "admitted", me) is True

    record = make(depot, open_dock((tmp_path / "store").as_uri()), depot.snapshot("two"))

    assert record["erasures"] == [request]


def test_a_long_text_comes_as_pieces_cut_between_files_each_at_its_own_place_of_the_order(make_depot, me, tmp_path):
    depot, short, repo, _ = long_depot(make_depot, me)
    text = depot.content(repo)
    assert pieces.count(len(text)) == 4
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store)

    rows = [(document, list(tokens)) for shard in record["shards"]["plain"] for tokens, _, document, _, _ in
            read_verified(store, shard)]
    names = [pieces.name(repo, n) for n in range(4)]
    taken = [document for document, _ in rows]
    # each piece at the export key of its name, among the other texts, never one after the other
    assert taken == sorted([*short, *names], key=export_key)
    places = sorted(taken.index(name) for name in names)
    assert all(later - earlier > 1 for earlier, later in zip(places, places[1:]))
    # the pieces are the whole text cut at the start of a file's line, each encoded as a text of its own
    held = Parted(depot.reader(), repo, depot.listed_text(repo, 1)["parts"])
    cut = [pieces.read(held, n) for n in range(4)]
    assert "".join(cut) == text and all(one.startswith("=== src/") for one in cut)
    assert all(abs(len(one) - pieces.LENGTH) < 300_100 for one in cut[:3])
    assert all(dict(rows)[name] == [*TOKENIZER.encode(one.encode()), END] for name, one in zip(names, cut))
    # each piece has its row, with its number and its text's credits; the text counts once
    table = [one for one in corpus_rows(store, record, "texts") if one["id"] == repo]
    assert sorted(one["piece"] for one in table) == [0, 1, 2, 3]
    assert {one["title"] for one in table} == {"A repository"}
    assert all(one["piece"] is None for one in corpus_rows(store, record, "texts") if one["id"] != repo)
    assert (record["texts"], record["pieces"]) == (201, 4)
    assert [line for line in model_notice(store, record).split("\n\n")[0].splitlines()
            if line.startswith(f"depot {repo}:")] == [f"depot {repo}: A repository; CC0-1.0"]
    # the same snapshot builds the same bytes
    again = make(depot, open_dock((tmp_path / "again").as_uri()))
    assert again["id"] == record["id"] and again["shards"]["plain"] == record["shards"]["plain"]


def test_a_long_text_no_longer_ends_a_corpus_smaller_than_it_which_takes_the_pieces_that_fit(make_depot, me,
                                                                                               tmp_path):
    # the whole repository would pass the size: the corpus still takes every short text and the pieces that fit
    depot, short, repo, _ = long_depot(make_depot, me)
    store = open_dock((tmp_path / "store").as_uri())
    whole = make(depot, store)
    tokens = {document: len(row) for shard in whole["shards"]["plain"] for row, _, document, _, _ in
              read_verified(store, shard)}
    short_tokens = sum(tokens[one] for one in short)
    smallest = min(range(4), key=lambda n: tokens[pieces.name(repo, n)])
    others = [tokens[pieces.name(repo, n)] for n in range(4) if n != smallest]
    limit = short_tokens + tokens[pieces.name(repo, smallest)] + 10_000  # a thousandth of it is under 10,000
    assert min(others) > limit  # room for the smallest piece alone

    record = make(depot, store, total_tokens=limit)

    taken = documents(store, record)
    assert sorted(taken) == sorted([*short, pieces.name(repo, smallest)])
    assert record["total_tokens"] <= limit and record["left_out"]["trimmed"] == 0


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


def test_a_text_of_more_pieces_than_a_16_bit_number_holds_comes_as_every_one_of_its_pieces(make_depot, me, tmp_path,
                                                                                           monkeypatch):
    # pieces of 16 characters cut a text a little over schema.LONG long into more than 32,767 pieces, as pieces
    # of pieces.LENGTH cut a text of more than 32 billion characters
    depot = make_depot()
    content = "a line\n" * 160_000
    [text] = depot.add([NewText(text=content, licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    monkeypatch.setattr(pieces, "LENGTH", 16)
    count = pieces.count(len(content))
    assert count > 1 << 16
    snapshot = snapshot_of(depot)

    made = order.of(depot, snapshot["id"])

    assert sorted(made.piece.tolist()) == list(range(count))
    assert [made.place(at) for at in range(len(made))] == sorted(made.place(at) for at in range(len(made)))
    # a walk goes on right after the place of a piece numbered past any 16-bit number
    at = made.piece.tolist().index(count - 1)
    assert made.start(list(made.place(at)[:1]) + ["depot", *made.place(at)[2:]]) == at + 1
    store = open_dock((tmp_path / "store").as_uri())
    record = make(depot, store, total_tokens=20_000)  # the first pieces of the order, most numbered past 32,767
    taken = documents(store, record)
    numbers = [int(name.rpartition("#")[2]) for name in taken]
    assert record["pieces"] == len(taken) and max(numbers) > 1 << 15
    assert taken == sorted(taken, key=export_key)
    table = {one["piece"] for one in corpus_rows(store, record, "texts")}
    assert table == set(numbers)
    held = Parted(depot.reader(), text, depot.listed_text(text, 1)["parts"])
    rows = {document: list(tokens) for shard in record["shards"]["plain"]
            for tokens, _, document, _, _ in read_verified(store, shard)}
    for name, number in zip(taken[:50], numbers):
        assert rows[name] == [*TOKENIZER.encode(pieces.read(held, number).encode()), END]


def test_a_text_long_at_the_snapshot_s_bound_comes_as_pieces_even_once_it_is_short(make_depot, me, tmp_path):
    # the order reads each text as of its bound: a text edited short after the snapshot still comes as its pieces
    depot = make_depot()
    [text] = depot.add([NewText(text=repository(4, 300_000), licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    snapshot = depot.create_snapshot()
    depot.change_content(text, "a short text now", None, me)
    depot.decide("admitted", [text], me)
    store = open_dock((tmp_path / "store").as_uri())

    then = make(depot, store, snapshot)
    now = make(depot, store, depot.create_snapshot())

    assert sorted(documents(store, then)) == sorted(pieces.name(text, n) for n in range(2))
    assert documents(store, now) == [text]


def test_the_long_texts_are_found_from_the_long_contents_never_from_every_text(make_depot, me):
    # the real depot holds tens of millions of texts: the order finds the long ones before it walks, through the
    # index of long content versions, each read at its bound
    depot, _, repo, _ = long_depot(make_depot, me)
    snapshot = snapshot_of(depot)
    reader = depot.db.reader()
    traced = []
    reader.exec_trace = lambda cursor, sql, bindings: traced.append((sql, bindings)) or True

    made = order.of(depot, snapshot["id"])

    reader.exec_trace = None
    assert [ids.make("depot", number) for _, number in made.long] == [repo]
    [(sql, bindings)] = [one for one in traced if "content_lists_long" in one[0]]
    plan = [row[3] for row in reader.execute(f"EXPLAIN QUERY PLAN {sql}", bindings or ())]
    assert any(one.startswith("SCAN content_lists USING") and "content_lists_long" in one for one in plan), plan


def test_a_build_cut_short_right_after_a_piece_goes_on_and_writes_the_same_corpus(make_depot, me, tmp_path):
    # a token file closes after each piece here, so the state that resumes names a piece's place, and the size
    # leaves one piece out, so what each part took and left out is part of that state
    depot, short, repo, _ = long_depot(make_depot, me)
    store = open_dock((tmp_path / "whole").as_uri())
    whole = make(depot, store)
    tokens = {document: len(row) for shard in whole["shards"]["plain"] for row, _, document, _, _ in
              read_verified(store, shard)}
    limit = whole["total_tokens"] - max(tokens[pieces.name(repo, n)] for n in range(4))
    straight = make(depot, open_dock((tmp_path / "straight").as_uri()), total_tokens=limit, shard_tokens=5_000)
    saved = []

    def stop_after_a_piece(state):
        saved.append(state)
        if state["after"][3] >= 0:
            raise KeyboardInterrupt

    store = open_dock((tmp_path / "store").as_uri())
    with pytest.raises(KeyboardInterrupt):
        make(depot, store, total_tokens=limit, work=tmp_path / "work", save=stop_after_a_piece, shard_tokens=5_000)

    resumed = make(depot, store, total_tokens=limit, work=tmp_path / "work", checkpoint=saved[-1],
                   shard_tokens=5_000)

    assert resumed["id"] == straight["id"] and resumed["pieces"] == 3
    assert [one["sha256"] for one in resumed["shards"]["plain"]] == [one["sha256"] for one in straight["shards"]["plain"]]


SMALL = "the dog"  # 2 tokens, 3 with END, 7 bytes


def tagged_depot(make_depot, me, *kinds):
    """A depot holding, for each (tags, how many, text) of kinds, that many admitted plain texts with those tags
    and that text: (the depot, {tags: the ids of those texts})."""
    depot = make_depot()
    added = {}
    for tags, count, text in kinds:
        added[tags] = depot.add([NewText(text=text, licenses=("CC0-1.0",), tags=tags) for _ in range(count)], me)
    depot.decide("admitted", "true", me)
    return depot, added


def groups_taken(record) -> dict:
    """{each group's tag of the first share: (texts, tokens)} of a corpus's record."""
    return {one["tag"]: (one["texts"], one["tokens"]) for one in record["shares_tokens"][0]["groups"]}


def test_a_share_splits_its_part_over_its_texts_tags_by_the_square_root_of_what_each_holds(make_depot, me, tmp_path):
    # a holds four times the bytes of b, so it aims at twice its tokens: 36 tokens are 12 texts, 8 of a and 4 of b
    depot, _ = tagged_depot(make_depot, me, (("en", "a"), 80, SMALL), (("en", "b"), 20, SMALL))
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, total_tokens=36, shares=[("en", 1.0)])

    assert groups_taken(record) == {"a": (8, 24), "b": (4, 12)}


def test_a_small_group_gives_all_it_has_and_its_leftover_goes_to_the_others(make_depot, me, tmp_path):
    # 30 tokens are 10 texts of 3 tokens: the group c holds only 2, so it gives all it has, and a and b, which hold
    # alike, split the rest evenly, whatever the order of their texts
    depot, _ = tagged_depot(make_depot, me, (("en", "a"), 20, SMALL), (("en", "b"), 20, SMALL),
                            (("en", "c"), 2, SMALL))
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, total_tokens=30, shares=[("en", 1.0)])

    assert groups_taken(record) == {"a": (4, 12), "b": (4, 12), "c": (2, 6)}
    assert record["shares_tokens"][0]["tokens"] == record["total_tokens"] == 30


def test_a_request_for_tokens_names_texts_of_at_most_its_bytes_and_chooses_the_same_texts(make_depot, me, tmp_path,
                                                                                         monkeypatch):
    # many groups ask for tokens ahead; with a byte allowance of 3 texts, every request names the text it is for
    # and at most 3 texts' bytes beyond it, and the corpus is the one a build with no allowance makes
    from depot import groups
    depot, _ = tagged_depot(make_depot, me, *((("en", tag), 30, SMALL) for tag in "abcdefgh"))
    monkeypatch.setattr("corpus_build.IN_PROCESS", 0)  # count in worker processes, as a big build does
    wide = make(depot, open_dock((tmp_path / "wide").as_uri()), total_tokens=90, shares=[("en", 1.0)])
    asked = []
    choose = groups._choose

    def watched(entries, budget, address, count, progress):
        def counted(items):
            asked.append(len(items))
            return count(items)
        return choose(entries, budget, address, counted, progress)

    monkeypatch.setattr(groups, "ASKED_BYTES", 3 * len(SMALL))
    monkeypatch.setattr(groups, "_choose", watched)
    narrow = make(depot, open_dock((tmp_path / "narrow").as_uri()), total_tokens=90, shares=[("en", 1.0)])

    assert asked and max(asked) <= 1 + 3
    assert [one["sha256"] for one in narrow["shards"]["plain"]] == [one["sha256"] for one in wide["shards"]["plain"]]
    assert groups_taken(narrow) == groups_taken(wide)


def test_a_text_of_two_groups_counts_for_both_and_is_taken_once(make_depot, me, tmp_path):
    # 24 tokens are 8 texts: each text tagged a and b gives both groups its tokens, so 4 of them balance 4 texts of c
    depot, _ = tagged_depot(make_depot, me, (("en", "a", "b"), 10, SMALL), (("en", "c"), 10, SMALL))
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, total_tokens=24, shares=[("en", 1.0)])

    assert groups_taken(record) == {"a": (4, 12), "b": (4, 12), "c": (4, 12)}
    taken = documents(store, record)
    assert len(taken) == len(set(taken)) == 8 and record["total_tokens"] == 24


def test_a_text_that_does_not_fit_what_its_share_has_left_is_left_out_and_its_group_goes_on(make_depot, me,
                                                                                             tmp_path):
    big = "the quick brown fox jumps over the lazy dog " * 4
    depot, added = tagged_depot(make_depot, me, (("en", "a"), 1, big), (("en", "a"), 16, SMALL),
                                (("en", "b"), 4, SMALL))
    store = open_dock((tmp_path / "store").as_uri())

    record = make(depot, store, total_tokens=18, shares=[("en", 1.0)])

    assert added[("en", "a")][0] not in documents(store, record)  # far more than 18 tokens
    assert groups_taken(record) == {"a": (4, 12), "b": (2, 6)}  # a holds about ten times the bytes of b


def test_a_share_of_one_group_takes_its_texts_in_the_order_as_a_order_without_shares_does(make_depot, me,
                                                                                           tmp_path):
    depot, _ = tagged_depot(make_depot, me, (("en", "a"), 12, SMALL))
    store = open_dock((tmp_path / "store").as_uri())

    shared = make(depot, store, total_tokens=15, shares=[("en", 1.0)])
    plain = make(depot, store, total_tokens=15)

    assert documents(store, shared) == documents(store, plain) == documents(store, make(depot, store))[:5]
    assert groups_taken(shared) == {"a": (5, 15)}


def test_a_build_of_shares_split_by_group_cut_short_goes_on_to_the_same_bytes_even_without_its_chosen_texts_file(
        make_depot, me, tmp_path):
    depot, _ = tagged_depot(make_depot, me, (("en", "a"), 40, SMALL), (("en", "b"), 40, SMALL),
                            (("en", "c"), 5, SMALL))
    shares = [("en", 1.0)]
    whole = make(depot, open_dock((tmp_path / "whole").as_uri()), total_tokens=150, shares=shares, shard_tokens=20)
    again = make(depot, open_dock((tmp_path / "again").as_uri()), total_tokens=150, shares=shares, shard_tokens=20)
    assert again["id"] == whole["id"] and groups_taken(whole)["c"] == (5, 15)
    store, work, saved = open_dock((tmp_path / "store").as_uri()), tmp_path / "work", []

    def stop_after(count):
        def save(state):
            saved.append(state)
            if len(saved) == count:
                raise KeyboardInterrupt
        return save

    with pytest.raises(KeyboardInterrupt):
        make(depot, store, total_tokens=150, shares=shares, work=work, save=stop_after(1), shard_tokens=20)
    (work / "chosen.bin").unlink()  # chosen again, the same
    with pytest.raises(KeyboardInterrupt):
        make(depot, store, total_tokens=150, shares=shares, work=work, checkpoint=saved[-1], save=stop_after(3),
             shard_tokens=20)
    resumed = make(depot, store, total_tokens=150, shares=shares, work=work, checkpoint=saved[-1], shard_tokens=20)

    assert resumed["id"] == whole["id"]
    assert [one["sha256"] for one in resumed["shards"]["plain"]] == [one["sha256"] for one in whole["shards"]["plain"]]
    assert resumed["shares_tokens"] == whole["shares_tokens"]


def test_the_order_sorts_by_key_owner_number_and_piece_whatever_the_key_bits_its_sort_shares(make_depot, me,
                                                                                            monkeypatch):
    # the sort keeps a key's high bits beside an entry's position, and puts in order the entries whose high bits
    # are the same; with two high bits left, nearly every entry ties, and the order is still the one of the keys
    depot, short, repo, _ = long_depot(make_depot, me)
    snapshot = snapshot_of(depot)

    def listed(made):
        return [(made.text(at), int(made.piece[at])) for at in range(len(made))]
    usual = order.of(depot, snapshot["id"])
    monkeypatch.setattr(order, "LOW_BITS", 62)
    tied = order.of(depot, snapshot["id"])

    names = [(text, order.WHOLE) for text in short] + [(repo, n) for n in range(4)]
    wanted = sorted(names, key=lambda one: (export_key(one[0] if one[1] < 0 else pieces.name(*one)),
                                            one[0].encode(), ids.number(one[0]), one[1]))
    assert listed(usual) == listed(tied) == wanted
    # each place found again by its key alone: a walk goes on right after it
    assert all(usual.start(list(usual.place(at)[:1]) + ["depot", *usual.place(at)[2:]]) == at + 1
               for at in range(len(usual)))
