"""`cache/` holds only what can be rebuilt: emptying it leaves every corpus and every checkpoint usable."""

import json
import shutil

import torch

from wpt.train import DockCheckpoint
from wpt.train.remote import named_files
from depot import NewText
from wpt_corpus import corpora
from wpt_corpus.files import DockFiles
from wpt_corpus.shards import documents, read_verified
from wpt_telemetry import Telemetry
from wpt_tokenizer import Tokenizer
import dock_files

TALK = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}]}


def test_emptying_the_cache_leaves_a_corpus_readable_and_a_checkpoint_to_resume_from(built):
    corpus_id = built.snapshot("talks", [NewText(kind="conversation", content=json.dumps(TALK))])
    store = built.store
    record = corpora.find(store, corpus_id)
    assert record["conversations"] == 1

    # every file the record refers to lies in the corpus's folder, none in cache/
    files = named_files(record).values()
    assert files and all(file["key"].startswith(f"{corpora.FOLDER}/") for file in files)

    checkpoint = DockCheckpoint(store, "fox-1", Telemetry.off(), set())
    checkpoint.save({"step": 3, "weights": torch.arange(4)})
    state = checkpoint.state_file()
    assert state["key"].startswith("models/fox-1/")

    shutil.rmtree(dock_files.folder(store) / "cache", ignore_errors=True)
    fresh = store
    files = DockFiles(fresh, dock_files.folder(store).parent / "reader")

    assert Tokenizer.from_bytes(dock_files.fetch(fresh, record["tokenizer"]["vocabulary"])).vocab_size > 0
    assert list(documents(files, record["shards"]["plain"]))
    [talk] = [row for shard in record["shards"]["conversation"] for row in read_verified(files, shard)]
    assert len(talk[0]) > 0

    resumed = DockCheckpoint(fresh, "fox-1", Telemetry.off(), set())
    assert resumed.exists() and resumed.load()["step"] == 3
