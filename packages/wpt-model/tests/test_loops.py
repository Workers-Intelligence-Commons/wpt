"""Loops (docs/model.md, "Loops"): the text passes `loops` more times
through every layer with the same weights; at 0 nothing of the loop
remains. A prompt read through the cache in one call gives what the
forward pass gives; a later token reads, in every pass, the last pass's
values of the earlier tokens."""

import torch

from wpt_model import WPT, ModelConfig, Routes
from wpt_model import attention, ttt
from wpt_model.carry import Entries
from wpt_model.sequence import Carried

CONFIG = dict(vocab_size=300, block_size=64, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
              attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
              shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
              residual_groups=10)


def model(loops):
    torch.manual_seed(0)
    found = WPT(ModelConfig(**CONFIG, loops=loops)).eval()
    with torch.no_grad():  # the parameters that start at zero (sinks, convolutions, queries) made to count
        for p in found.parameters():
            if p.abs().max() == 0:
                p.normal_(std=0.3)
    return found


def text(T=150):
    ids = torch.randint(256, (2, T), generator=torch.Generator().manual_seed(1))
    ids[:, 70] = ttt.END_OF_DOCUMENT
    return ids


def test_loops_add_no_weight_and_each_pass_starts_from_the_one_before_s_output():
    plain, looped = model(0), model(2)
    assert {k: v.shape for k, v in plain.state_dict().items()} == {k: v.shape for k, v in looped.state_dict().items()}
    looped.load_state_dict(plain.state_dict())
    ids = text()
    with torch.no_grad():
        embedding = plain.embedding(ids)
        hidden = embedding
        for loop in range(3):  # one pass of the plain model, from the pass before's output
            hidden, _ = plain._pass(hidden, ids, plain.canonical[ids], embedding, None, None, Carried(0, {}), loop, ())
        expected = plain.head(plain.final_norm(hidden))
        assert torch.allclose(looped(ids), expected, atol=1e-5)
        assert not torch.allclose(plain(ids), expected, atol=1e-3)


def test_each_pass_routes_on_its_own_and_replays_its_own_choices():
    looped, routes = model(1), Routes()
    ids = text()[:1]
    with torch.no_grad():
        recorded = looped(ids, record=routes)
        replayed = looped(ids, replay=routes)
    routed = [layer.experts for layer in looped.layers if not layer.experts.hashed]
    assert set(routes.chosen) == {(experts, loop) for experts in routed for loop in (0, 1)}
    assert torch.equal(recorded, replayed)


def test_a_prompt_read_at_once_gives_the_forward_pass_and_later_tokens_read_the_last_pass(monkeypatch):
    monkeypatch.setattr(attention, "TOP_K", 16)
    monkeypatch.setattr(ttt, "CHUNK", 32)
    looped = model(2)
    ids = text()
    with torch.no_grad():
        cache = looped.new_cache(2)
        scores, _ = looped.extend(ids[:, :100], cache)
        assert torch.allclose(scores, looped(ids[:, :100]), atol=1e-4)
        # the next tokens: every pass reads the earlier tokens' values the cache holds, the last pass's
        carried = {}
        for module in looped.modules():
            for name, item in (module.passes_on().items() if hasattr(module, "passes_on") else ()):
                value = cache.text.values.get((id(module), name))
                if value is None:
                    continue
                if isinstance(item, Entries):
                    value = value[:, :100]
                carried.update({(id(module), name, loop): value for loop in range(3)})
        later, _ = looped.extend(ids[:, 100:], cache)
        assert torch.allclose(later, looped(ids[:, 100:], sequence=Carried(100, carried)), atol=1e-4)
        # which differs from reading the whole text: the earlier passes' own values are gone
        assert not torch.allclose(later, looped(ids)[:, 100:], atol=1e-3)
