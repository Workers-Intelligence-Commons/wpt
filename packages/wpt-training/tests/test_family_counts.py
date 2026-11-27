"""The sizes of the model families against the models they train
(docs/model.md, "Model families"): wpt_orders.counts counts the
parameters each model has, and a size reads exactly the start of the
corpus it shares."""

from pathlib import Path

import numpy as np
import pytest
import torch

from wpt_model import PARTS, ModelConfig, WPT
from wpt_model.engram import Engram
from wpt_model.experts import Experts
from wpt_orders import Orders, counts, family
from dock import api
from wpt_corpus.files import DockFiles
from wpt_corpus.shards import write_shard
from wpt_telemetry import Telemetry
from wpt_training import Data, Strategy, load_step

ROOT = Path(__file__).resolve().parents[3]


def models():
    """The model section of every size of every family of the repository, by its FAMILY:SIZE."""
    repo = Orders(ROOT)
    return {f"{name}:{size}": repo.model_order(f"{name}:{size}")["model"]
            for name in repo.names("family") for size in family.sizes(repo.family(name))}


def test_the_counts_are_the_parameters_of_the_model_each_size_builds():
    # the count without the vocabulary tables decides a size's tokens per parameter: it must be the model's own,
    # so it is checked against the model wpt_model builds at every size
    for size, model in models().items():
        model = {key: value for key, value in model.items() if key != "tied_tables"}
        config = ModelConfig(vocab_size=32768, block_size=32768, **model)
        with torch.device("meta"):
            built = WPT(config)
        parts = built.parts()
        tables = sum(p.numel() for part in ("embedding", "output-head") for p in parts[part])
        everything = sum(p.numel() for p in built.parameters())
        assert counts.parameters(model, 32768) == (everything - tables, everything), size
        assert tuple(counts.layer_kinds(model)) == config.kinds and counts.layer_count(model) == config.n_layer
        with torch.device("meta"):
            tied = WPT(ModelConfig(vocab_size=32768, block_size=32768, **model, tied_tables=True))
        one_table = sum(p.numel() for p in tied.parameters())
        assert counts.parameters({**model, "tied_tables": True}, 32768) == (everything - tables, one_table), size
        assert one_table == everything - tables // 2, size
        # a loop over the first half: the loop's output and the coda's first sublayer take the sums of one
        # boundary, and nothing else changes
        looping = {**model, "loops": 1, "loop_layers": config.first_half}
        with torch.device("meta"):
            coda = WPT(ModelConfig(vocab_size=32768, block_size=32768, **looping))
        assert sum(p.numel() for p in coda.parameters()) == everything, size
        assert counts.parameters(looping, 32768) == (everything - tables, everything), size


def test_the_counts_of_the_parameters_one_token_uses_are_those_of_the_model_each_size_builds():
    # the model's own count without the vocabulary tables, less the routed experts a token does not reach
    # (n_experts - top_k of each layer's stacked experts) and less Engram's lookup tables
    for size, model in models().items():
        with torch.device("meta"):
            built = WPT(ModelConfig(vocab_size=32768, block_size=32768, **model))
        unused = tables = 0
        for module in built.modules():
            if isinstance(module, Experts):
                routed = sum(p.numel() for p in (module.w_gate, module.w_up, module.w_down))
                unused += routed - routed // module.n * module.k
            if isinstance(module, Engram):
                tables += sum(p.numel() for p in module.tables.parameters())
        assert unused > 0
        assert counts.active_parameters(model) == counts.parameters(model, 0)[0] - unused - tables, size


@pytest.fixture
def dataset(tmp_path):
    """A dock holding 4 token files of 100 tokens each, token i at place i
    of the corpus's plain text, each file its own text; and the corpus's
    record."""
    dock = (tmp_path / "dock").as_uri()
    shards = []
    for i in range(4):
        path = tmp_path / f"{i}.parquet"
        digest = write_shard(path, [(list(range(100 * i, 100 * i + 100)), f"s{i}", f"s{i}/d", "p", 0)])
        stored = api.put_from(dock, f"corpora/d/{digest}.parquet", path)
        shards.append({"key": stored["key"], "sha256": stored["sha256"], "token_sha256": digest, "rows": 1,
                       "tokens": 100})
    record = {"shards": {"plain": shards}, "total_tokens": 400}
    return DockFiles(dock, tmp_path / "reader"), record


def windows(dock, record, prefix, stream_shards=None):
    """Every window the read-text step draws reading its prefix 20 times in updates of 8 windows of 16 tokens."""
    step = load_step({"step": "read-text", "settings": {
        "stages": [{"block_size": 16, "share": 1}], "tokens_per_update": 128, "prefix_tokens": prefix,
        "readings": 20}}, PARTS)
    step.check_corpus(record, dock)
    data = Data(step.name, dock, record, None, Strategy(torch.device("cpu")), Telemetry.off(), lambda _: None, 1337,
                1, stream_shards=stream_shards)
    environment = step.start_environment(data)
    try:
        [stage] = step.stages()
        found = [x.numpy() for update in range(1, stage["steps"] + 1) for x, _ in environment.batch(update, None, None)]
        return np.concatenate(found), environment.learned
    finally:
        environment.close()


@pytest.mark.parametrize("stream_shards", [None, 2])
def test_a_smaller_size_reads_exactly_the_start_of_the_corpus_built_for_a_larger_one(dataset, stream_shards):
    # a prefix of 250 tokens ends inside the third token file: every window is a run of the first
    # 250 tokens, the windows reach its end, and the step learned from those 250 tokens alone; the
    # same windows on every run
    dock, record = dataset
    found, learned = windows(dock, record, 250, stream_shards)
    assert (found[:, 1:] == found[:, :-1] + 1).all()
    assert found.min() >= 0 and found.max() <= 249 and found.max() >= 240
    assert learned == {"plain": 250}
    again, _ = windows(dock, record, 250, stream_shards)
    assert (found == again).all()
    # the largest size's prefix: the whole corpus
    everything, learned = windows(dock, record, 400, stream_shards)
    assert everything.max() >= 250 and learned == {"plain": 400}


def test_a_prefix_longer_than_the_corpus_is_refused_before_any_step_runs(dataset):
    dock, record = dataset
    with pytest.raises(ValueError, match="reads the first 401 tokens; the corpus has 400"):
        windows(dock, record, 401)
