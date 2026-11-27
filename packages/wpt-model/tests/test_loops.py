"""Loops (docs/model.md, "Loops"): the text passes `loops` more times
through every layer with the same weights; at 0 nothing of the loop
remains. A text read through the cache, in any number of calls, gives
what the forward pass gives: every pass reads its own values of the
earlier tokens. The rules of how the passes follow each other (decay
between passes) are off unless the config sets them, and on they give
their formula."""

import json

import pytest
import torch

from wpt_model import WPT, ModelConfig, Routes
from wpt_model import attention, ttt
from wpt_model.sequence import Carried

CONFIG = dict(vocab_size=300, block_size=64, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
              attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
              shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
              residual_groups=10)


def model(loops, **rules):
    torch.manual_seed(0)
    found = WPT(ModelConfig(**CONFIG, loops=loops, **rules)).eval()
    with torch.no_grad():  # the parameters that start at zero (sinks, convolutions, queries) made to count
        for p in found.parameters():
            if p.abs().max() == 0:
                p.normal_(std=0.3)
    return found


def text(T=150):
    ids = torch.randint(256, (2, T), generator=torch.Generator().manual_seed(1))
    ids[:, 70] = ttt.END_OF_DOCUMENT
    return ids


def one_pass(plain, first, ids, loop):
    """One pass of the plain model's layers from first, its first source."""
    return plain._pass(first, ids, plain.canonical[ids], plain.embedding(ids), None, None, Carried(0, {}), loop, ())[0]


def test_loops_add_no_weight_and_each_pass_starts_from_the_one_before_s_output():
    plain, looped = model(0), model(2)
    assert {k: v.shape for k, v in plain.state_dict().items()} == {k: v.shape for k, v in looped.state_dict().items()}
    looped.load_state_dict(plain.state_dict())
    ids = text()
    with torch.no_grad():
        hidden = plain.embedding(ids)
        for loop in range(3):  # one pass of the plain model, from the pass before's output
            hidden = one_pass(plain, hidden, ids, loop)
        expected = plain.head(plain.final_norm(hidden))
        assert torch.allclose(looped(ids), expected, atol=1e-5)
        assert not torch.allclose(plain(ids), expected, atol=1e-3)


@pytest.mark.parametrize("decay_last_pass", [False, True])
def test_with_a_decay_each_pass_starts_from_the_input_plus_the_decayed_change(decay_last_pass):
    # Loopback's Eq. 13: the next pass starts from h0 + lambda (out - h0), h0 the loop's input (here the embedding)
    plain, decayed = model(0), model(2, loop_decay=0.25, decay_last_pass=decay_last_pass)
    decayed.load_state_dict(plain.state_dict())
    ids = text()
    with torch.no_grad():
        start = plain.embedding(ids)
        hidden = start
        for loop in range(3):
            out = one_pass(plain, hidden, ids, loop)
            hidden = start + 0.25 * (out - start)
        head_reads = hidden if decay_last_pass else out
        assert torch.allclose(decayed(ids), plain.head(plain.final_norm(head_reads)), atol=1e-5)
        assert not torch.allclose(decayed(ids), model(2)(ids), atol=1e-3)


def test_a_config_without_the_rules_of_the_loop_is_saved_as_before_they_existed(tmp_path):
    # config.json, the steps' fingerprints and the scheduler's setup read ModelConfig.numbers()
    ModelConfig(**CONFIG, loops=2).save(tmp_path / "config.json")
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "loop_decay" not in saved and "decay_last_pass" not in saved
    assert ModelConfig.load(tmp_path / "config.json") == ModelConfig(**CONFIG, loops=2)
    ModelConfig(**CONFIG, loops=2, loop_decay=0.25).save(tmp_path / "decayed.json")
    assert ModelConfig.load(tmp_path / "decayed.json").loop_decay == 0.25


@pytest.mark.parametrize("rules, problem", [
    (dict(loops=0, loop_decay=0.25), "loop_decay shapes the passes of a loop, and loops is 0"),
    (dict(loops=1, loop_decay=1.0), "loop_decay must be above 0 and below 1"),
    (dict(loops=1, decay_last_pass=True), "decay_last_pass decays by loop_decay, which is left out"),
])
def test_rules_of_the_loop_that_cannot_hold_are_refused(rules, problem):
    with pytest.raises(ValueError, match=problem):
        ModelConfig(**CONFIG, **rules)


def test_each_pass_routes_on_its_own_and_replays_its_own_choices():
    looped, routes = model(1), Routes()
    ids = text()[:1]
    with torch.no_grad():
        recorded = looped(ids, record=routes)
        replayed = looped(ids, replay=routes)
    routed = [layer.experts for layer in looped.layers if not layer.experts.hashed]
    assert set(routes.chosen) == {(experts, loop) for experts in routed for loop in (0, 1)}
    assert torch.equal(recorded, replayed)


@pytest.mark.parametrize("rules", [{}, dict(loop_decay=0.25), dict(loop_decay=0.25, decay_last_pass=True)])
def test_a_text_read_through_the_cache_in_pieces_gives_the_forward_pass(monkeypatch, rules):
    monkeypatch.setattr(attention, "TOP_K", 16)
    monkeypatch.setattr(ttt, "CHUNK", 32)
    looped = model(2, **rules)
    ids = text()
    with torch.no_grad():
        whole = looped(ids)
        cache = looped.new_cache(2)
        pieces = [looped.extend(ids[:, a:b], cache)[0] for a, b in ((0, 100), (100, 101), (101, 102), (102, 150))]
        assert torch.allclose(torch.cat(pieces, 1), whole, atol=1e-4)
        # each pass keeps its own values of the earlier tokens
        kept = {key[:2] for key in cache.text.values}
        assert set(cache.text.values) == {(*key, loop) for key in kept for loop in range(3)}


def trained(rules, updates=12):
    """The loss of a tiny looping model on one batch of windows before and
    after a few updates of AdamW on it."""
    torch.manual_seed(0)
    found = WPT(ModelConfig(**CONFIG, loops=2, **rules)).train()
    optimizer = torch.optim.AdamW(found.parameters(), lr=3e-3)
    ids = torch.randint(256, (4, 66), generator=torch.Generator().manual_seed(2))
    x, y, after = ids[:, :-2], ids[:, 1:-1], ids[:, 2:]
    losses = []
    for _ in range(updates):
        next_sum, ahead_sum = found(x, next_tokens=y, targets=(y, after))
        loss = next_sum / y.numel() + 0.3 * ahead_sum / after.numel()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("rules", [dict(loop_decay=0.25), dict(loop_decay=0.25, decay_last_pass=True)])
def test_a_looping_model_with_its_rules_learns(rules):
    losses = trained(rules)
    assert losses[-1] < losses[0] - 0.1
