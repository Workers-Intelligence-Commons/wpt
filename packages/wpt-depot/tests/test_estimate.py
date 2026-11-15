import json
from pathlib import Path

from wpt_store import open_store
from wpt_tokenizer import Tokenizer

from wpt_depot import NewText
from wpt_depot.corpus import build
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
    conversation_tokens = sum(one["tokens"] for one in record["conversation_shards"])
    # the sample holds every text, so the estimate of the tokens is what the build takes
    assert (found["plain_tokens"], found["item_tokens"]) == (record["total_tokens"], conversation_tokens)
    assert found["texts"] == record["texts"] and found["sample"]["tokenized"] == 31
    assert found["bytes"] > 0 and found["seconds"] is not None and found["workers"] == 1
    # a size cuts the estimate as it cuts the build: the items first, the plain texts what they leave
    assert estimate(depot, dataset, TOKENIZER, conversation_tokens + 10)["tokens"] == conversation_tokens + 10
