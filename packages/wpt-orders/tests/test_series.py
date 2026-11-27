"""Model series (docs/model.md, "Model series"): the repository's series
file derives each size's shape and training settings, and every size
shares one corpus."""

import copy
from pathlib import Path

import pytest

from wpt_orders import InvalidOrder, NotFound, Orders, series, validate

ROOT = Path(__file__).resolve().parents[3]
SERIES = "eldest-titmouse"
WINDOW = 32768
VOCABULARY = 32768  # the size of the vocabulary the series' data part names


@pytest.fixture(scope="module")
def repo():
    return Orders(ROOT)


@pytest.fixture(scope="module")
def ladder(repo):
    return repo.order("series", SERIES)


def test_the_sizes_scale_every_width_of_the_reference_and_keep_its_layers(repo, ladder):
    # widths in multiples of 128, every width inside the model in proportion to d (64 of width per attention head),
    # every count kept: the reference's layout of 21 layers, its Engram layer and its full form of Attention
    # Residuals; Engram's table size in proportion to the cube of the width (50021 x (d / 512)^3, to the nearest prime)
    table_size = {256: 6257, 384: 21101, 512: 50021, 640: 97687, 768: 168803}
    assert series.sizes(ladder) == ["256x5", "384x8", "512x10", "640x13", "768x15"]
    for size in series.sizes(ladder):
        model = series.shape(ladder, size)
        d = model["d_model"]
        assert size.startswith(f"{d}x")
        assert model == {
            "window_layers": 4, "sparse_groups": 2, "sparse_layers": 5, "d_model": d,
            "attention_heads": d // 64, "attention_head_dim": 128,
            "n_experts": 32, "top_k": 4, "expert_width": d // 2, "latent_width": d // 2, "shared_width": d,
            "engram_layers": [2], "engram_order": 3, "engram_heads": 8, "engram_dim": 16,
            "engram_table_size": table_size[d], "residual_groups": 42, "loops": 0,
        }
        assert series.layer_count(model) == 21
    # the reference size is interstitial-pecker's model, and every size uses the 32,768-token vocabulary
    assert series.shape(ladder, "512x10") == repo.order("model", "interstitial-pecker")["model"]


def test_a_size_s_training_follows_deepseek_llm_s_laws_at_the_compute_its_size_pairs_with(repo, ladder):
    # DeepSeek LLM (arXiv 2401.02954), eq. 1 and 4, written here from the paper: M_opt = 0.1715 C^0.5243
    # gives the C of a size's M = 6 x the parameters a token uses + 12 x layers x width x window (plus In-Place TTT's
    # work: 6 x 2 matrices of width x 2 shared widths in each of its 4 layers, at one pass of the loop), then
    # D_opt = 5.8316 C^0.4757, eta_opt = 0.3118 C^-0.1250 and B_opt = 0.2920 C^0.3271 (tokens)
    plan = series.plan(ladder, "512x10", VOCABULARY)
    assert (plan.parameters, plan.total_parameters) == (221_873_112, 255_427_544)  # the model's own count
    # a token uses 93,447,320 parameters (4 of the 32 routed experts of each layer, no Engram table; checked against
    # the model in wpt-training's tests), through 21 layers of width 512, over a window of 32,768 tokens
    M = 6 * 93_447_320 + 12 * 21 * 512 * WINDOW + 6 * 4 * 2 * 512 * 1024
    C = (M / 0.1715) ** (1 / 0.5243)
    assert plan.flops_per_token == M == 4_813_708_176
    assert plan.compute == pytest.approx(C, rel=1e-12) and C == pytest.approx(8.471e19, rel=1e-3)
    assert plan.tokens == pytest.approx(5.8316 * C ** 0.4757, rel=1e-12)
    assert plan.learning_rate == pytest.approx(0.3118 * C ** -0.125, rel=5e-4)
    assert plan.learning_rate == 0.001007
    assert plan.batch_tokens == pytest.approx(0.2920 * C ** 0.3271, rel=1e-12)
    assert plan.batch_size == 29  # 963,407 tokens: 29.40 windows of 32K
    assert plan.updates == 18_521  # 1.7600e10 tokens / (29 x 32,768) = 18,520.7
    assert plan.warmup == 741  # 4% of 18,521 = 740.8
    assert plan.prefix_tokens == 18_521 * 29 * (WINDOW + 1)


def test_the_batch_is_whole_windows_at_least_one_and_the_warm_up_a_share_of_the_updates(ladder):
    def planned(batch_windows, tokens=1000 * WINDOW, warmup_share=0.04):
        # laws of exponent 0 give the batch and the tokens straight away
        changed = copy.deepcopy(ladder)
        changed["laws"]["batch_tokens"] = {"coefficient": batch_windows * WINDOW, "exponent": 0}
        changed["laws"]["tokens"] = {"coefficient": tokens, "exponent": 0}
        changed["warmup_share"] = warmup_share
        return series.plan(changed, "256x5", 32768)

    assert planned(2.5).batch_size == 3  # halves up
    assert planned(2.49).batch_size == 2
    assert planned(0.3).batch_size == 1  # at least one window
    assert planned(2, 2000 * WINDOW).updates == 1000
    assert planned(2, 2000 * WINDOW).warmup == 40
    assert planned(2, 2000 * WINDOW, 0.0125).warmup == 13  # 12.5: halves up
    assert planned(2, 20 * WINDOW).warmup == 1  # 0.4 of 10 updates: at least one


def test_each_size_is_a_model_order_and_every_size_shares_the_data_of_the_largest(repo, ladder):
    # eldest-titmouse names a snapshot: every size takes its data with the largest size's total tokens
    plans = repo.series_plans(SERIES, VOCABULARY)
    largest = max(plans, key=lambda plan: plan.parameters)
    assert largest.size == "768x15"
    data = {**ladder["data"], "total_tokens": largest.prefix_tokens}
    assert set(data) == {"depot", "snapshot", "change", "digest", "roles", "total_tokens", "vocabulary"}
    assert repo.data_part(f"{SERIES}:{largest.size}") == data
    for plan in plans:
        order = repo.model_order(f"{SERIES}:{plan.size}", VOCABULARY)
        validate(order, "model", f"{SERIES}:{plan.size}")
        assert order["data"] == data  # one corpus: every size reads the same one
        assert order["model"] == plan.model and order["seed"] == ladder["seed"]
        assert order["drafter"] == ladder["drafter"]
        [step] = order["steps"]
        assert step["step"] == "read-text" and step["locked"] == []
        assert step["optimizer"] == {"name": "by-part", "settings": {"lr": plan.learning_rate, "warmup": plan.warmup}}
        assert step["settings"] == {"stages": [{"block_size": WINDOW, "steps": plan.updates,
                                                "batch_size": plan.batch_size}],
                                    "prefix_tokens": plan.prefix_tokens, "mtp_switch": 0.8}
        assert plan.prefix_tokens <= data["total_tokens"]
    # a plain name is still a model order's
    assert repo.model_order("interstitial-pecker") == repo.order("model", "interstitial-pecker")


def test_a_size_or_a_series_that_does_not_exist_is_refused(repo, ladder):
    with pytest.raises(NotFound, match="no size '1024x20' in the series eldest-titmouse"):
        repo.model_order(f"{SERIES}:1024x20", VOCABULARY)
    with pytest.raises(NotFound, match="no series 'nothing'"):
        repo.model_order("nothing:256x5", VOCABULARY)
    odd = copy.deepcopy(ladder)
    odd["sizes"].append({"name": "300x6", "d_model": 300})  # 300 / 512 of 8 attention heads is not whole
    with pytest.raises(ValueError, match="size 300x6: attention_heads"):
        series.check(odd)


SNAPSHOT_DATA = {"depot": "https://depot.example", "snapshot": "2026-11", "change": 1234, "digest": "d" * 64,
                 "roles": ["held_out", "pretrain"], "shares": [{"tag": "fiction", "share": 0.5}],
                 "vocabulary": {"sha256": "5" * 64, "size": VOCABULARY}}


def test_a_series_of_the_snapshot_shape_gives_every_size_its_data_and_the_largest_size_s_tokens(ladder):
    snapshot = {**copy.deepcopy(ladder), "data": copy.deepcopy(SNAPSHOT_DATA)}
    validate(snapshot, "series", "snapshot")
    largest = series.data_part(ladder)["total_tokens"]
    assert series.data_part(snapshot) == {**SNAPSHOT_DATA, "total_tokens": largest}
    for size in series.sizes(snapshot):
        order = series.model_order(snapshot, series.plan(snapshot, size, VOCABULARY))
        validate(order, "model", f"snapshot:{size}")
        assert order["data"] == {**SNAPSHOT_DATA, "total_tokens": largest}
    # a series' sizes give the size: its data part names none
    with pytest.raises(InvalidOrder, match="at data: Additional properties are not allowed .'total_tokens' was"):
        validate({**snapshot, "data": {**SNAPSHOT_DATA, "total_tokens": 100}}, "series", "snapshot")


def test_a_series_that_reads_its_text_several_times_gives_every_size_its_readings_over_the_same_prefix(ladder):
    # docs/training.md, "Readings": each size's read-text step reads its prefix that many times; the updates and
    # the warm-up stay those of one reading, which the step stretches, and the corpus every size reads is the same
    four = {**copy.deepcopy(ladder), "readings": 4}
    validate(four, "series", "four readings")
    assert series.data_part(four) == series.data_part(ladder)
    for size in series.sizes(four):
        once, again = (series.model_order(chosen, series.plan(chosen, size, VOCABULARY)) for chosen in (ladder, four))
        [step], [read] = once["steps"], again["steps"]
        assert read["settings"] == {**step["settings"], "readings": 4}
        assert read["optimizer"] == step["optimizer"]
        for run in series.runs(four, size):  # a decay branch's trunk and decay read as many times
            assert all(entry["settings"]["readings"] == 4
                       for entry in series.run_order(four, series.plan(four, size, VOCABULARY), run)["steps"])
    for readings in (0, 1.5):
        with pytest.raises(InvalidOrder, match="at readings"):
            validate({**four, "readings": readings}, "series", "readings")
