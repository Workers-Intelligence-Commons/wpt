import hashlib
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
VOCABULARY = {"sha256": hashlib.sha256(TOKENIZER.to_bytes()).hexdigest(), "size": TOKENIZER.vocab_size}


def data(total_tokens=None, shares=()) -> dict:
    return {"kinds": [], "shares": [{"tag": tag, "share": share} for tag, share in shares],
            "total_tokens": total_tokens, "vocabulary": VOCABULARY}


def built(depot, snapshot, store, total_tokens=None, shares=()):
    return build(depot, snapshot["id"], data(total_tokens, shares), store, TOKENIZER, log=lambda *_: None)


def test_an_estimate_of_a_snapshot_its_sample_covers_gives_the_tokens_its_build_holds_and_writes_nothing(
        make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([NewText(text=f"the quick brown fox {n} jumps over the lazy dog", licenses=("CC0-1.0",), tags=("en",))
               for n in range(30)]
              + [NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    snapshot = depot.create_snapshot("s1")
    before = sorted((path.name, path.stat().st_mtime_ns) for path in Path(tmp_path).iterdir())

    found = estimate(depot, snapshot["id"], data(), TOKENIZER)

    assert sorted((path.name, path.stat().st_mtime_ns) for path in Path(tmp_path).iterdir()) == before
    record = built(depot, snapshot, open_store((tmp_path / "store").as_uri()))
    assert found["build_key"] == record["build_key"]  # the key the build files its corpus under
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
    cut = estimate(depot, snapshot["id"], data(size), TOKENIZER)
    small = built(depot, snapshot, open_store((tmp_path / "store").as_uri()), size)
    assert cut["item_share"] == share and cut["forecast"] is True and cut["build_key"] == small["build_key"]
    # each text counted at the sample's tokens per byte: off by a few tokens, never by a whole text
    one_text = record["total_tokens"] / 30
    assert abs(cut["plain_tokens"] - small["total_tokens"]) < one_text and cut["item_tokens"] == small["items_tokens"]
    assert cut["tokens"] <= size
    # and it counts the texts as the build does: the conversations it takes of those the snapshot holds, then the
    # plain texts, the one at the edge of a part in or out as the rate per byte falls
    assert (cut["item_texts"], cut["items_held"]) == (small["conversations"], record["conversations"])
    assert abs(cut["plain_texts"] - (small["texts"] - small["conversations"])) <= 1
    assert cut["texts"] == cut["plain_texts"] + cut["item_texts"]


def test_an_estimate_counts_no_conversation_whose_quoted_text_the_snapshot_does_not_hold(make_depot, me, tmp_path):
    # the build leaves such a conversation out, so the estimate must not count it either
    depot = make_depot()
    inside, outside = depot.add([NewText(text=f"the quick brown fox {n} jumps", licenses=("CC0-1.0",))
                                 for n in range(2)], me)

    def quoting(source, exact):
        return NewText(kind="conversation", licenses=("CC0-1.0",), content=json.dumps({"messages": [
            {"role": "user", "content": [{"type": "excerpt", "source": source,
                                          "selector": {"type": "TextQuoteSelector", "exact": exact}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A fox."}]}]}))
    depot.add([quoting(inside, "fox 0"), quoting(outside, "fox 1")], me)
    depot.decide("admitted", f"id <> '{outside}'", me)
    snapshot = depot.create_snapshot("s1")

    found = estimate(depot, snapshot["id"], data(), TOKENIZER)
    record = built(depot, snapshot, open_store((tmp_path / "store").as_uri()))

    assert record["conversations"] == 1 and record["left_out"]["quoted text left out"] == 1
    assert found["item_texts"] == 1
    assert found["item_tokens"] == sum(one["tokens"] for one in record["shards"]["conversation"])
    sized = estimate(depot, snapshot["id"], data(1000), TOKENIZER)
    assert any("quote texts the snapshot holds" in line for line in sized["warnings"])


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
    snapshot = depot.create_snapshot("s1")
    shares = [("long", 0.5), ("short", 0.5)]
    store = open_store((tmp_path / "store").as_uri())
    long_text = estimate(depot, snapshot["id"], data(shares=shares), TOKENIZER)["share_tokens"][0]["tokens"] / 12
    size = int(long_text)  # half of it, the long share's part, holds no long text

    small = estimate(depot, snapshot["id"], data(size, shares), TOKENIZER)
    record = built(depot, snapshot, store, size, shares)

    got = {one["tag"]: one["tokens"] for one in record["shares_tokens"]}
    forecast = {one["tag"]: one["tokens"] for one in small["share_tokens"]}
    assert got["long"] == forecast["long"] == 0 and got["short"] > 0 and small["forecast"] is True
    assert abs(forecast["short"] - got["short"]) < long_text / 40 * 3  # a few short texts at most
    assert small["even_size"] is not None and small["even_size"] > size
    even = built(depot, snapshot, store, small["even_size"], shares)
    proportions = {one["tag"]: one["tokens"] / even["total_tokens"] for one in even["shares_tokens"]}
    assert abs(proportions["long"] - 0.5) <= 0.1 + 0.02  # the forecast's tokens per byte may be off by a little


def test_a_snapshot_bigger_than_the_order_the_estimate_follows_is_sampled_over_its_numbers_and_said_rough(
        make_depot, me, monkeypatch):
    # the estimate never works out the export key of every text of a big snapshot: it samples evenly spaced
    # numbers, and its parts are proportions of the size, not a forecast
    monkeypatch.setattr(estimates, "FOLLOWED", 20)
    depot = make_depot()
    depot.add([NewText(text=f"the quick brown fox {n} jumps over the lazy dog", licenses=("CC0-1.0",))
               for n in range(40)], me)
    depot.decide("admitted", "true", me)
    snapshot = depot.create_snapshot("s1")
    whole = estimate(depot, snapshot["id"], data(), TOKENIZER)

    small = estimate(depot, snapshot["id"], data(whole["tokens"] // 4), TOKENIZER)

    assert whole["sample"]["tokenized"] == 40 and small["forecast"] is False
    assert small["plain_tokens"] <= whole["tokens"] // 4 and small["item_tokens"] == 0
