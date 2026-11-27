"""A size of a model family trained through the commands (docs/model.md,
"Model families"): FAMILY:SIZE goes wherever an order's name goes, and
wpt train family NAME prints every size's numbers."""

import json
from pathlib import Path
from types import SimpleNamespace

from wpt_orders import Orders
from wpt_training import WINDOWS
from wpt_worker.training import drafter_shards
from wpt.train import main as train


def tiny(built) -> dict:
    """Two widths of the application tests' tiny model (the family d's) reading the start of one corpus of the
    snapshot d in updates of 4 windows of 16 tokens: the small one 256 tokens, 4 updates, the large one 832, 13
    updates."""
    base = json.loads(Path("data/orders/d.json").read_text())
    data = {key: value for key, value in base["data"].items() if key != "total_tokens"}
    model = {key: value for key, value in base["model"].items() if key not in ("d_model", "shared_width")}
    return {"data": data, "model": model, "seed": 1337, "drafter": base["drafter"],
            "steps": [{"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.002, "warmup": 0.25}},
                       "settings": {"stages": [{"block_size": 16, "share": 1}], "tokens_per_update": 64}}],
            "sizes": {"columns": ["size", "model.d_model", "model.shared_width", "data.total_tokens"],
                      "rows": [["small", 16, 16, 256], ["large", 32, 32, 832]]}}


def test_the_drafter_reads_the_shards_the_run_s_steps_read_and_none_past_their_prefix():
    shards = [{"object": str(number), "tokens": 100} for number in range(4)]
    spec = {"record": {"shards": {"plain": shards}}}

    def text(prefix):
        return SimpleNamespace(reads=WINDOWS, settings={"prefix_tokens": prefix})

    turns = SimpleNamespace(reads="conversations", settings={})
    assert drafter_shards(spec, [text(150)]) == shards[:2]
    assert drafter_shards(spec, [text(150), text(250), turns]) == shards[:3]  # the longest prefix
    assert drafter_shards(spec, [turns]) == shards  # no text step


def test_every_size_of_a_family_reads_one_corpus_and_a_size_trains_on_its_start(built, tmp_path):
    built.write("tiny", tiny(built))
    orders = Orders(tmp_path)
    data = orders.model_order("tiny:small")["data"]
    assert data == orders.model_order("tiny:large")["data"]
    assert data["total_tokens"] == 832  # the large size's tokens
    corpus = built.sized("d", data["total_tokens"])["id"]  # the corpus of the most tokens a size reads
    train(["tiny:small", "--out", "model"])  # found by the build key of the snapshot, data part and size
    provenance = json.loads((Path("model") / "provenance.json").read_text())
    assert (provenance["order_name"], provenance["family"], provenance["size"]) == ("tiny:small", "tiny", "small")
    assert provenance["corpus"]["id"] == corpus
    assert provenance["order"] == orders.model_order("tiny:small")  # the size's whole order
    [step] = provenance["order"]["steps"]
    assert step["settings"]["prefix_tokens"] == 256 == provenance["learned"]["plain"]  # the corpus's start
    # the record says the updates the stages make: 256 tokens in updates of 64
    assert [(s["steps"], s["batch_size"], s["block_size"]) for s in provenance["steps"][0]["stages"]] == [(4, 4, 16)]
    assert provenance["model"]["window_layers"] == 1 and provenance["model"]["d_model"] == 16


def test_train_family_prints_every_size_and_trains_nothing(built, tmp_path, capsys):
    built.write("tiny", tiny(built))

    train(["family", "tiny"])

    printed = capsys.readouterr().out
    assert "tiny:small: width 16, 5 layers" in printed and "tiny:large: width 32, 5 layers" in printed
    assert "reads the first 256 tokens of a corpus of 832" in printed
    assert "step 1: 13 updates of 4 windows of 16 (13 updates)" in printed
    assert "832 tokens trained" in printed and "FLOPs per token" in printed
    assert not (tmp_path / "model").exists()


def test_the_run_s_record_says_how_the_stages_were_rounded_to_whole_groups():
    # pocket 64: 811.3 updates computed, nine stages of 90.1; 51 groups of 16 updates scheduled
    from wpt.train import readings
    from wpt_model import PARTS
    from wpt_training import load_step
    entry = Orders(Path(__file__).resolve().parents[4]).model_order("pocket:64")["steps"][0]
    [(line, strays)] = readings([load_step(entry, PARTS)])
    assert not strays and "816 updates" in line and "1.01 times" in line
    assert ("its stages are rounded to whole groups of 16 updates, so none skips the end of a group: "
            + ", ".join(["90.1"] * 9) + " computed, " + ", ".join(["96"] * 6 + ["80"] * 3) + " scheduled") in line
