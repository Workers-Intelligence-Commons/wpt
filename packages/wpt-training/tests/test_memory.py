"""The choice between keeping every activation and recomputing the layers
(wpt_training.memory; docs/training.md, "Recomputing layers"), on faked
memory numbers."""

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
    # (0.5 GiB) plus MARGIN: foreseen 6.15 GiB; usable is free plus reserved less HEADROOM of the total
    per_token, tokens = 16 * 2 ** 10, 32768
    foreseen = 4 * GIB + GIB + GIB / 2 + per_token * tokens * (1 + MARGIN)
    total = 8 * GIB
    for free, recompute in ((foreseen + HEADROOM * total - GIB, False), (foreseen + HEADROOM * total - GIB - 1, True)):
        choice = choose(4 * GIB, GIB, GIB / 2, per_token, tokens, free, GIB, total)
        assert choice.recompute is recompute and choice.foreseen == pytest.approx(foreseen)
    text = choose(4 * GIB, GIB, GIB / 2, per_token, tokens, 0, 0, total).text()
    assert text.startswith("recompute the layers: a pass of 32,768 tokens foreseen at 6.2 GiB without recomputing")
    assert "16 KiB per token measured" in text and "against -0.8 GiB usable" in text


def test_the_probes_give_the_fixed_part_and_the_bytes_per_token_and_the_optimizer_s_state_counts_until_it_exists(
        monkeypatch):
    # probes of 2,048 and 1,024 tokens at 1 GiB + 100 KiB per token; a fresh AdamW has 8 bytes per weight to
    # come, none once it stepped; a window shorter than the probe probes the window and half of it
    model = WPT(ModelConfig(**TINY))
    optimizer = torch.optim.AdamW(model.parameters())
    probed = []

    def probe(model, tokens, found, generator):
        probed.append(tokens)
        return GIB + 100 * 2 ** 10 * tokens

    monkeypatch.setattr(memory, "probe", probe)

    class Left:
        def allocated(self):
            return 2 * GIB

        def left(self):
            return 10 * GIB, 3 * GIB, 16 * GIB

    choice = measure(model, optimizer, 65536, 32768, Left())
    weights = sum(p.numel() for p in model.parameters())
    assert sorted(probed) == [1024, 2048] and choice.fixed == pytest.approx(GIB)
    assert choice.per_token == pytest.approx(100 * 2 ** 10) and choice.state == 8 * weights
    assert choice.usable == pytest.approx(13 * GIB - HEADROOM * 16 * GIB)
    ids = torch.zeros(1, 8, dtype=torch.long)
    sum(model(ids, next_tokens=ids, targets=(ids, ids))).backward()
    optimizer.step()
    probed.clear()
    assert measure(model, optimizer, 64, 32, Left()).state == 0 and sorted(probed) == [16, 32]


def test_a_probe_leaves_the_weights_their_gradients_the_statistics_and_the_random_streams_as_they_were():
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

    assert memory.probe(model, 32, Nothing(), torch.Generator().manual_seed(0)) == 0
    assert torch.equal(torch.get_rng_state(), stream)
    assert all(p.grad is None for p in model.parameters())
    assert all(getattr(module, "histogram", None) is None for module in model.modules())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
