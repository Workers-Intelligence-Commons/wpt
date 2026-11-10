"""`cache/` holds only what can be rebuilt: emptying it leaves every corpus and every checkpoint usable."""

import json
import shutil

import torch

from wpt.train import StoreCheckpoint
from wpt.train.remote import named_files
from wpt_depot import NewText
from wpt_depot import corpus as corpora
from wpt_store import open_store
from wpt_store.shards import documents, read_verified
from wpt_telemetry import Telemetry
from wpt_tokenizer import Tokenizer

TALK = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}]}


def test_emptying_the_cache_leaves_a_corpus_readable_and_a_checkpoint_to_resume_from(built):
    corpus_id = built.dataset("talks", [NewText(kind="conversation", content=json.dumps(TALK))])
    store = built.store
    record = corpora.find(store, corpus_id)
    assert record["conversations"] == 1

    # every file the record refers to lies in the corpus's folder, none in cache/
    files = named_files(record).values()
    assert files and all(file.key.startswith(f"{corpora.FOLDER}/") for file in files)

    checkpoint = StoreCheckpoint(store, "fox-1", Telemetry.off(), set())
    checkpoint.save({"step": 3, "weights": torch.arange(4)})
    state = checkpoint.state_file()
    assert state["key"].startswith("models/fox-1/")

    shutil.rmtree(store.root / "cache", ignore_errors=True)
    shutil.rmtree(store.cache, ignore_errors=True)  # and the copies this machine made of the store's files
    fresh = open_store(store.root.as_uri())

    assert Tokenizer.from_bytes(fresh.get_bytes(record["tokenizer"]["vocabulary"])).vocab_size > 0
    assert list(documents(fresh, record["shards"]))
    [talk] = [row for shard in record["conversation_shards"] for row in read_verified(fresh, shard)]
    assert len(talk[0]) > 0

    resumed = StoreCheckpoint(fresh, "fox-1", Telemetry.off(), set())
    assert resumed.exists() and resumed.load()["step"] == 3
