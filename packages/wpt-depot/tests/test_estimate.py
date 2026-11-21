import json
from pathlib import Path

from wpt_store import open_store
from wpt_tokenizer import Tokenizer

from wpt_depot import NewText
from wpt_depot.corpus import build
from wpt_depot import estimate as estimates
from wpt_depot.estimate import estimate

TOKENIZER = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, vocab_size=560)
CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


def test_an_estimate_of_a_dataset_its_sample_covers_gives_the_tokens_its_build_holds_and_writes_nothing(
        make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text=f"the quick brown fox {n} jumps over the lazy dog", licenses=("CC0-1.0",), tags=("en",))
               for n in range(30)]
              + [NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("all")
    depot.put(library, "true")
    dataset = depot.dataset_by_id(depot.new_dataset("all", "classics", [(None, library)]))
    before = sorted((path.name, path.stat().st_mtime_ns) for path in Path(tmp_path).iterdir())

    found = estimate(depot, dataset, TOKENIZER, None)

    assert sorted((path.name, path.stat().st_mtime_ns) for path in Path(tmp_path).iterdir()) == before
    record = build(depot, "all", open_store((tmp_path / "store").as_uri()), TOKENIZER, max_tokens=None, log=lambda *_: None)
    conversation_tokens = sum(one["tokens"] for one in record["shards"]["conversation"])
    # the sample holds every text, so the estimate of the tokens is what the build takes
    assert (found["plain_tokens"], found["item_tokens"]) == (record["total_tokens"], conversation_tokens)
    assert found["texts"] == record["texts"] and found["sample"]["tokenized"] == 31
    assert found["bytes"] > 0 and found["seconds"] is not None and found["workers"] == 1
    # a size cuts the estimate as it cuts the build: the items their share of the bytes, the plain texts the rest,
    # each taking texts in the build's order until the first that does not fit, so the forecast is what it builds
    size = 4 * conversation_tokens
    rows = depot.rows("SELECT c.kind, c.bytes FROM texts t JOIN content_lists c ON c.id = t.id "
                      "AND c.version = t.content_version")
    share = sum(r["bytes"] for r in rows if r["kind"] != "plain") / sum(r["bytes"] for r in rows)
    cut = estimate(depot, dataset, TOKENIZER, size)
    built = build(depot, "all", open_store((tmp_path / "store").as_uri()), TOKENIZER, max_tokens=size,
                  log=lambda *_: None)
    assert cut["item_share"] == share and cut["forecast"] is True
    # each text counted at the sample's tokens per byte: off by a few tokens, never by a whole text
    one_text = record["total_tokens"] / 30
    assert abs(cut["plain_tokens"] - built["total_tokens"]) < one_text and cut["item_tokens"] == built["items_tokens"]
    assert cut["tokens"] <= size
    # and it counts the texts as the build does: the conversations it takes of those the dataset holds, then the
    # plain texts, the one at the edge of a part in or out as the rate per byte falls
    assert (cut["item_texts"], cut["items_held"]) == (built["conversations"], record["conversations"])
    assert abs(cut["plain_texts"] - (built["texts"] - built["conversations"])) <= 1
    assert cut["texts"] == cut["plain_texts"] + cut["item_texts"]


def test_an_estimate_counts_no_conversation_whose_quoted_text_the_dataset_does_not_hold(make_depot, me, tmp_path):
    # the build leaves such a conversation out, so the estimate must not count it either
    depot = make_depot()
    inside, outside = depot.add([NewText(text=f"the quick brown fox {n} jumps", licenses=("CC0-1.0",))
                                 for n in range(2)], me)

    def quoting(source, exact):
        return NewText(kind="conversation", licenses=("CC0-1.0",), content=json.dumps({"messages": [
            {"role": "user", "content": [{"type": "excerpt", "source": source,
                                          "selector": {"type": "TextQuoteSelector", "exact": exact}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A fox."}]}]}))
    kept, dropped = depot.add([quoting(inside, "fox 0"), quoting(outside, "fox 1")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("all")
    depot.put(library, f"id <> '{outside}'")
    dataset = depot.dataset_by_id(depot.new_dataset("all", "classics", [(None, library)]))

    found = estimate(depot, dataset, TOKENIZER, None)
    record = build(depot, "all", open_store((tmp_path / "store").as_uri()), TOKENIZER, max_tokens=None,
                   log=lambda *_: None)

    assert record["conversations"] == 1 and record["left_out"]["quoted text left out"] == 1
    assert found["item_texts"] == 1
    assert found["item_tokens"] == sum(one["tokens"] for one in record["shards"]["conversation"])
    sized = estimate(depot, dataset, TOKENIZER, 1000)
    assert any("quote texts the dataset holds" in line for line in sized["warnings"])


def test_at_a_small_size_the_estimate_forecasts_the_share_a_long_text_leaves_short_and_the_size_where_shares_even_out(
        make_depot, me, tmp_path):
    # a share whose texts are each longer than its part at a small size gets nothing from the build, which takes
    # a share's texts in order until the first that does not fit; the estimate says so, and names a size where
    # every share comes within 10 points of its proportion
    depot = make_depot()
    sentence = "the quick brown fox jumps over the lazy dog. "
    depot.add([NewText(text=f"{n} " + sentence * 40, licenses=("CC0-1.0",), tags=("long",)) for n in range(12)]
              + [NewText(text=f"{n} " + sentence * 2, licenses=("CC0-1.0",), tags=("short",)) for n in range(60)], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("all")
    depot.put(library, "true")
    dataset = depot.dataset_by_id(depot.new_dataset("mix", "classics", [(None, library)], [("long", 0.5), ("short", 0.5)]))
    store = open_store((tmp_path / "store").as_uri())
    long_text = estimate(depot, dataset, TOKENIZER, None)["share_tokens"][0]["tokens"] / 12
    size = int(long_text)  # half of it, the long share's part, holds no long text

    small = estimate(depot, dataset, TOKENIZER, size)
    built = build(depot, "mix", store, TOKENIZER, max_tokens=size, log=lambda *_: None)

    got = {one["tag"]: one["tokens"] for one in built["shares_tokens"]}
    forecast = {one["tag"]: one["tokens"] for one in small["share_tokens"]}
    assert got["long"] == forecast["long"] == 0 and got["short"] > 0 and small["forecast"] is True
    assert abs(forecast["short"] - got["short"]) < long_text / 40 * 3  # a few short texts at most
    assert small["even_size"] is not None and small["even_size"] > size
    even = build(depot, "mix", store, TOKENIZER, max_tokens=small["even_size"], log=lambda *_: None)
    shares = {one["tag"]: one["tokens"] / even["total_tokens"] for one in even["shares_tokens"]}
    assert abs(shares["long"] - 0.5) <= 0.1 + 0.02  # the forecast's tokens per byte may be off by a little


def test_an_estimate_of_a_revision_whose_figures_are_not_known_reads_every_text_and_says_so(make_depot, me):
    # figures are known only for this depot's own libraries: a dataset reading another depot's has none, so the
    # estimate counts the items' part by reading every text, as its build will, and warns that it does
    depot, other = make_depot("depot"), make_depot("other")
    libraries = []
    for one in (depot, other):
        one.add([NewText(text=f"the quick brown fox {n}", licenses=("CC0-1.0",), tags=("en",)) for n in range(10)]
                + [NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",))], me)
        one.decide("admitted", "true", me)
        libraries.append(one.new_library("all"))
        one.put(libraries[-1], "true")
    mine = depot.dataset_by_id(depot.new_dataset("mine", "classics", [(None, libraries[0])]))
    theirs = depot.dataset_by_id(depot.new_dataset("theirs", "classics", [(other.address, libraries[1])]))

    known, unknown = (estimate(depot, row, TOKENIZER, 200) for row in (mine, theirs))

    assert (known["figures"], known["warnings"]) == (True, [])
    assert unknown["figures"] is False and "reading the dataset again" in unknown["warnings"][0]
    # the two libraries hold the same texts: what the walk counts is what the figures say
    assert (unknown["item_share"], unknown["items_held"]) == (known["item_share"], known["items_held"])
    assert known["items_held"] == 1 and 0 < known["item_share"] < 1


def test_a_dataset_without_conversations_is_forecast_through_the_build_s_order_not_said_rough(
        make_depot, me, tmp_path, monkeypatch):
    # the walk follows fewer texts than the dataset holds, and closes the plain text's part at its first text that
    # does not fit: with no conversation to wait for, that is the whole forecast
    monkeypatch.setattr(estimates, "FOLLOWED", 20)
    depot = make_depot()
    depot.add([NewText(text=f"the quick brown fox {n} jumps over the lazy dog", licenses=("CC0-1.0",))
               for n in range(40)], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("all")
    depot.put(library, "true")
    dataset = depot.dataset_by_id(depot.new_dataset("all", "classics", [(None, library)]))
    whole = estimate(depot, dataset, TOKENIZER, None)["tokens"]

    small = estimate(depot, dataset, TOKENIZER, whole // 4)

    built = build(depot, "all", open_store((tmp_path / "store").as_uri()), TOKENIZER, max_tokens=whole // 4,
                  log=lambda *_: None)
    assert small["forecast"] is True and small["item_tokens"] == 0
    assert abs(small["plain_texts"] - built["texts"]) <= 1
