"""A size of a model series trained through the commands (docs/model.md,
"Model series"): SERIES:SIZE goes wherever a model order's name goes."""

import json
from pathlib import Path
from types import SimpleNamespace

from wpt_corpus import corpora
from wpt_orders import Orders
from wpt_training import WINDOWS
from wpt_worker.training import drafter_shards
from wpt.train import main as train

# a series of two sizes of the application tests' tiny model; laws chosen so that C = M and D = M / 1600:
# the small size (60,880 parameters, M 371,616) reads 232 tokens as 4 updates of 4 windows of 16 tokens, the
# large one (219,744, M 1,302,720) 814 tokens as 13 updates
TINY = {
    "data": None,  # the test's: the snapshot d (series_data)
    "reference": {"window_layers": 1, "sparse_groups": 1, "sparse_layers": 1, "d_model": 16, "attention_heads": 2,
                  "attention_head_dim": 64, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
                  "shared_width": 16, "engram_layers": [2], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
                  "engram_table_size": 31, "residual_groups": 8, "loops": 0},
    "block_size": 16,
    "laws": {"learning_rate": {"coefficient": 0.002, "exponent": 0}, "batch_tokens": {"coefficient": 64, "exponent": 0},
             "model_flops": {"coefficient": 1, "exponent": 1}, "tokens": {"coefficient": 0.000625, "exponent": 1}},
    "warmup_share": 0.04,
    "seed": 1337,
    "sizes": [{"name": "small", "d_model": 16}, {"name": "large", "d_model": 32}],
    "drafter": {"epochs": 1, "prompts": 4, "prompt_tokens": 8, "answer_tokens": 8, "batch": 2, "anchors": 4,
                "held_out": 0.25},
}


def series_data(built) -> dict:
    """The series' data part: the snapshot d of the test's depot, as a model order names it, without a size."""
    return {key: value for key, value in built.data("d", None).items() if key != "total_tokens"}


def test_the_drafter_reads_the_shards_the_run_s_steps_read_and_none_past_their_prefix():
    shards = [{"object": str(number), "tokens": 100} for number in range(4)]
    spec = {"record": {"shards": {"plain": shards}}}

    def text(prefix):
        return SimpleNamespace(reads=WINDOWS, settings={"prefix_tokens": prefix})

    turns = SimpleNamespace(reads="conversations", settings={})
    assert drafter_shards(spec, [text(150)]) == shards[:2]
    assert drafter_shards(spec, [text(150), text(250), turns]) == shards[:3]  # the longest prefix
    assert drafter_shards(spec, [text(150), text(0)]) == shards  # one step reads every training token
    assert drafter_shards(spec, [turns]) == shards  # no text step


def test_every_size_of_a_series_reads_one_corpus_and_a_size_trains_on_its_start(built, tmp_path):
    folder = tmp_path / "data" / "orders" / "series"
    folder.mkdir(parents=True)
    (folder / "tiny.json").write_text(json.dumps({**TINY, "data": series_data(built)}))
    orders = Orders(tmp_path)
    vocabulary = corpora.find(built.dock, built.corpus_id)["tokenizer"]["vocab_size"]  # the size training plans for
    data = orders.model_order("tiny:small", vocabulary)["data"]
    assert data == orders.model_order("tiny:large", vocabulary)["data"]
    assert data["total_tokens"] == 13 * 4 * 17  # the large size's prefix
    corpus = built.sized("d", data["total_tokens"])["id"]  # the corpus of the series' largest prefix
    train(["tiny:small", "--out", "model"])  # found by the build key of the snapshot, data part and largest prefix
    provenance = json.loads((Path("model") / "provenance.json").read_text())
    assert provenance["order_name"] == "tiny:small" and provenance["corpus"]["id"] == corpus
    assert provenance["order"] == orders.model_order("tiny:small", vocabulary)
    [step] = provenance["order"]["steps"]
    assert step["settings"]["stages"] == [{"block_size": 16, "steps": 4, "batch_size": 4}]
    assert step["settings"]["prefix_tokens"] == 4 * 4 * 17 == provenance["learned"]["plain"]  # the corpus's start
    assert step["optimizer"]["settings"] == {"lr": 0.002, "warmup": 1}
    assert provenance["model"]["window_layers"] == 1 and provenance["model"]["d_model"] == 16


def test_train_series_prints_every_size_and_trains_nothing(built, tmp_path, capsys):
    folder = tmp_path / "data" / "orders" / "series"
    folder.mkdir(parents=True)
    (folder / "tiny.json").write_text(json.dumps({**TINY, "data": series_data(built)}))

    train(["series", "tiny"])

    printed = capsys.readouterr().out
    assert f"series tiny: one corpus of {13 * 4 * 17:,} training tokens" in printed
    assert "small: width 16" in printed and "large: width 32" in printed and "prefix 272 tokens" in printed
    assert not (tmp_path / "model").exists()
