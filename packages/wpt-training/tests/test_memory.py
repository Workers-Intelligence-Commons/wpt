"""How a stage runs its passes when the run leaves it to the runner: every
activation kept or the layers recomputed, the windows of a pass, whole
windows or segments (wpt_training.memory; docs/training.md, "Recomputing
layers"), on faked memory numbers."""

import pytest
import torch

from wpt_model import WPT, ModelConfig
from wpt_training import memory
from wpt_training.memory import HEADROOM, MARGIN, choose, measure

GIB = 2 ** 30
TINY = dict(vocab_size=50, block_size=64, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
            attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
            shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
            residual_groups=3)


def test_a_pass_is_kept_whole_only_when_its_foreseen_peak_fits_what_is_left_less_the_headroom():
    # 4 GiB held, 1 GiB of optimizer state to come, 0.5 GiB fixed, 16 KiB per token times 32,768 tokens
    # (0.5 GiB) plus MARGIN; usable is free plus reserved less HEADROOM of the total; recomputing, 0.25 GiB
    # fixed and 4 KiB per token
    kept, recomputed, tokens = (GIB / 2, 16 * 2 ** 10), (GIB / 4, 4 * 2 ** 10), 32768
    foreseen = 4 * GIB + GIB + GIB / 2 + 16 * 2 ** 10 * tokens * (1 + MARGIN)
    total = 8 * GIB
    for free, recompute in ((foreseen + HEADROOM * total - GIB, False), (foreseen + HEADROOM * total - GIB - 1, True)):
        choice = choose(4 * GIB, GIB, kept, recomputed, 1, tokens, free, GIB, total)
        assert choice.recompute is recompute and choice.foreseen == pytest.approx(foreseen)
        assert choice.fits and (choice.windows, choice.segment, choice.pass_tokens) == (1, 0, tokens)
    text = choice.text()
    assert text.startswith("recompute the layers: a pass of 32,768 tokens foreseen at 6.0 GiB without recomputing")
    assert "16 KiB per token measured" in text and "against 6.0 GiB usable" in text
    assert "and at 5.4 GiB recomputing (0.25 GiB plus 4 KiB per token measured)" in text


MIB = 2 ** 20
KEPT, RECOMPUTED = (0.0, 2 * MIB), (0.0, MIB)  # per token, with nothing held and no state to come


def left(tokens, recompute=True):
    """The free memory that holds a pass of that many tokens and no more."""
    return (RECOMPUTED[1] if recompute else KEPT[1]) * tokens * (1 + MARGIN)


def chosen(free, **options):
    """The choice for a share of 8 windows of 1,024 tokens, segments in multiples of 128, as (recompute,
    windows per pass, the rule's segment, fits)."""
    choice = choose(0, 0, KEPT, RECOMPUTED, 8, 1024, free, 0, 0, **{"part": 128, **options})
    return choice.recompute, choice.windows, choice.segment, choice.fits


def test_the_first_way_that_fits_is_chosen_whole_then_recomputed_then_fewer_windows_then_segments():
    # docs/training.md, "Recomputing layers": the whole pass of 8,192 tokens with every activation kept;
    # recomputing; the largest number of windows that divides the share of 8; one window in the longest
    # segments of a multiple of 128 tokens shorter than the window; and nothing when even 128 tokens do not fit
    assert chosen(left(8192, recompute=False)) == (False, 8, 0, True)
    assert chosen(left(8192, recompute=False) - 1) == (True, 8, 0, True)
    assert chosen(left(8192)) == (True, 8, 0, True)
    assert chosen(left(8192) - 1) == (True, 4, 0, True)
    assert chosen(left(3 * 1024)) == (True, 2, 0, True)  # 3 does not divide 8
    assert chosen(left(1024)) == (True, 1, 0, True)
    assert chosen(left(1024) - 1) == (True, 1, 896, True)
    assert chosen(left(300)) == (True, 1, 256, True)
    assert chosen(left(128)) == (True, 1, 128, True)
    assert chosen(left(128) - 1) == (True, 1, 128, False)
    assert chosen(left(1024) - 1, part=0) == (True, 1, 0, False)  # windows that cannot run in segments


def test_the_options_windows_per_pass_and_segment_are_kept_and_the_rule_chooses_what_is_left():
    # --micro-batch 4: a pass of 4 windows, never fewer, in segments when they do not fit whole; --segment
    # 512: each window brings 512 tokens to a pass, and fewer windows fit; both: nothing is left to choose
    assert chosen(left(4096), windows=4) == (True, 4, 0, True)
    assert chosen(left(2048), windows=4) == (True, 4, 512, True)
    assert chosen(left(4 * 128) - 1, windows=4) == (True, 4, 128, False)
    assert chosen(left(4096), segment=512) == (True, 8, 0, True)
    assert chosen(left(1024), segment=512) == (True, 2, 0, True)
    assert chosen(left(512) - 1, segment=512) == (True, 1, 0, False)
    assert chosen(left(2048), windows=4, segment=512) == (True, 4, 0, True)
    assert chosen(left(2048) - 1, windows=4, segment=512) == (True, 4, 0, False)
    choice = choose(0, 0, KEPT, RECOMPUTED, 8, 1024, left(1024), 0, 0, segment=512)
    assert (choice.tokens, choice.pass_tokens) == (4096, 1024)


def test_the_log_s_words_name_the_chosen_pass_and_a_refusal_names_the_smallest():
    fewer = choose(0, 0, KEPT, RECOMPUTED, 8, 1024, left(2048), 0, 0, part=128).text()
    assert fewer.startswith("recompute the layers, 2 windows per pass: a pass of 8,192 tokens foreseen at 16.8 GiB "
                            "without recomputing")
    assert "and at 8.4 GiB recomputing" in fewer and "against 2.1 GiB usable" in fewer
    assert fewer.endswith("; a pass of 2,048 tokens foreseen at 2.1 GiB recomputing")
    segments = choose(0, 0, KEPT, RECOMPUTED, 8, 1024, left(256), 0, 0, part=128).text()
    assert segments.startswith("recompute the layers, 1 window per pass in segments of 256 tokens: a pass of 8,192")
    assert segments.endswith("; a pass of 256 tokens foreseen at 0.3 GiB recomputing")
    none = choose(0, 0, KEPT, RECOMPUTED, 8, 1024, left(64), 0, 0, part=128).text()
    assert none.startswith("no pass fits the device's memory, down to 1 window per pass in segments of 128 tokens: ")
    assert "against 0.1 GiB usable; a pass of 128 tokens foreseen at 0.1 GiB recomputing" in none
    asked = choose(0, 0, KEPT, RECOMPUTED, 8, 1024, left(64), 0, 0, 8, 1024).text()
    assert asked.startswith("no pass fits the device's memory: a pass of 8,192 tokens") and "; a pass" not in asked


def test_the_probes_give_the_fixed_part_and_the_bytes_per_token_and_the_optimizer_s_state_counts_until_it_exists(
        monkeypatch):
    # probes of 2,048 and 1,024 tokens at 1 GiB + 100 KiB per token, and half of each recomputing; a fresh
    # AdamW has 8 bytes per weight to come, none once it stepped; a window, or the options' segment, shorter
    # than the probe probes itself and half of it
    model = WPT(ModelConfig(**TINY))
    optimizer = torch.optim.AdamW(model.parameters())
    probed = []

    def probe(model, tokens, found, generator, recompute):
        probed.append((tokens, recompute))
        return (GIB + 100 * 2 ** 10 * tokens) // (2 if recompute else 1)

    monkeypatch.setattr(memory, "probe", probe)

    class Left:
        def allocated(self):
            return 2 * GIB

        def left(self):
            return 10 * GIB, 3 * GIB, 16 * GIB

    choice = measure(model, optimizer, 2, 32768, Left())
    weights = sum(p.numel() for p in model.parameters())
    assert sorted(probed) == [(1024, False), (1024, True), (2048, False), (2048, True)]
    assert choice.fixed == pytest.approx(GIB) and choice.per_token == pytest.approx(100 * 2 ** 10)
    assert choice.recomputed_fixed == pytest.approx(GIB / 2)
    assert choice.recomputed_per_token == pytest.approx(50 * 2 ** 10)
    assert choice.state == 8 * weights and choice.tokens == 65536
    assert choice.usable == pytest.approx(13 * GIB - HEADROOM * 16 * GIB)
    ids = torch.zeros(1, 8, dtype=torch.long)
    sum(model(ids, next_tokens=ids, targets=(ids, ids))).backward()
    optimizer.step()
    probed.clear()
    assert measure(model, optimizer, 2, 32, Left()).state == 0
    assert sorted({tokens for tokens, _ in probed}) == [16, 32]
    probed.clear()
    assert measure(model, optimizer, 2, 32, Left(), segment=16).tokens == 32 and min(probed) == (8, False)


@pytest.mark.parametrize("recompute", [False, True], ids=["kept", "recomputed"])
def test_a_probe_leaves_the_weights_their_gradients_the_statistics_and_the_random_streams_as_they_were(recompute):
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "dropout": 0.1} if "dropout" in ModelConfig.__dataclass_fields__ else TINY))
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stream = torch.get_rng_state()

    class Nothing:
        def allocated(self):
            return 0

        def start(self):
            pass

        def peak(self):
            return 0

    assert memory.probe(model, 32, Nothing(), torch.Generator().manual_seed(0), recompute) == 0
    assert model.recompute is False
    assert torch.equal(torch.get_rng_state(), stream)
    assert all(p.grad is None for p in model.parameters())
    assert all(getattr(module, "histogram", None) is None for module in model.modules())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
