"""Loops (docs/model.md, "Loops"): the text passes `loops` more times
through every layer with the same weights; at 0 nothing of the loop
remains. A prompt read through the cache in one call gives what the
forward pass gives; a later token reads, in every pass, the last pass's
values of the earlier tokens. With a loop gate, each token keeps a learned
share of each extra pass's result; without it, or with no loop, the model
is the one without the field."""

import torch

from wpt_model import WPT, ModelConfig, Routes
from wpt_model import attention, ttt
from wpt_model.carry import Entries
from wpt_model.sequence import Carried

CONFIG = dict(vocab_size=300, block_size=64, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
              attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
              shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
              residual_groups=10)


def model(loops, **more):
    torch.manual_seed(0)
    found = WPT(ModelConfig(**CONFIG, loops=loops, **more)).eval()
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


def gated(loops=2):
    """A looping model with a gate whose shares differ from token to token."""
    found = model(loops, loop_gate=True)
    with torch.no_grad():
        found.loop_gate.bias.copy_(torch.linspace(-0.5, 0.5, loops))
    return found


def test_without_a_gate_or_without_a_loop_the_model_is_the_one_without_the_field():
    ids = text()
    for loops, loop_gate in ((2, False), (0, True)):
        torch.manual_seed(0)
        plain = WPT(ModelConfig(**CONFIG, loops=loops)).eval()
        torch.manual_seed(0)
        same = WPT(ModelConfig(**CONFIG, loops=loops, loop_gate=loop_gate)).eval()
        assert same.loop_gate is None and same.loop_gate_shares() == []
        assert list(same.state_dict()) == list(plain.state_dict())
        assert all(torch.equal(a, b) for a, b in zip(same.state_dict().values(), plain.state_dict().values()))
        with torch.no_grad():
            assert torch.equal(same(ids), plain(ids))


def test_the_gate_adds_one_weight_row_and_one_bias_per_extra_pass_and_starts_near_the_plain_loop():
    plain, looped = model(2), model(2, loop_gate=True)
    added = {name: tuple(value.shape) for name, value in looped.state_dict().items()
             if name not in plain.state_dict()}
    assert added == {"loop_gate.weight": (2, 16), "loop_gate.bias": (2,)}
    assert looped.parts()["loop-gate"] == [looped.loop_gate.weight, looped.loop_gate.bias]
    assert {role for name, _, role, _ in looped.optimizer_roles() if name.startswith("loop_gate.")} == {"vector"}
    looped = WPT(looped.config).eval()  # as it starts: the weight at zero, so one share for every token
    assert not looped.loop_gate.weight.any()
    ids = text()
    start = torch.sigmoid(torch.tensor(looped.loop_gate.START))
    with torch.no_grad():
        embedding = looped.embedding(ids)
        before, _ = looped._pass(embedding, ids, looped.canonical[ids], embedding, None, None, Carried(0, {}), 0, ())
        looped.loop_gate = None  # the pass's result, before the gate mixes it
        result, _ = looped._pass(before, ids, looped.canonical[ids], embedding, None, None, Carried(0, {}), 1, ())
        looped.loop_gate = WPT(looped.config).loop_gate.eval()
        after, _ = looped._pass(before, ids, looped.canonical[ids], embedding, None, None, Carried(0, {}), 1, ())
    assert torch.allclose(after, start * result + (1 - start) * before, atol=1e-6)
    assert 0.98 < start < 1 and torch.allclose(looped.loop_gate.shares[0], start)


def test_each_token_keeps_its_own_share_of_each_extra_pass():
    looped = gated()
    ids = text()
    with torch.no_grad():
        embedding = looped.embedding(ids)
        hidden, _ = looped._pass(embedding, ids, looped.canonical[ids], embedding, None, None, Carried(0, {}), 0, ())
        gate, looped.loop_gate = looped.loop_gate, None
        shares = []
        for loop in (1, 2):  # the plain pass, then the mix by hand
            result, _ = looped._pass(hidden, ids, looped.canonical[ids], embedding, None, None, Carried(0, {}), loop,
                                     ())
            g = torch.sigmoid(gate.norm(hidden) @ gate.weight[loop - 1] + gate.bias[loop - 1])[..., None]
            assert 0 < g.min() and g.max() < 1 and g.std() > 0.01  # between 0 and 1, and not one share for all
            hidden = g * result + (1 - g) * hidden
            shares.append(g.mean())
        looped.loop_gate = gate
        scores, features = looped(ids, features=(looped.config.n_layer,))
        assert torch.allclose(scores, looped.head(looped.final_norm(hidden)), atol=1e-5)
        assert torch.allclose(features, hidden, atol=1e-5)  # the drafter reads the output head's input, the mix
    assert torch.allclose(torch.tensor(looped.loop_gate_shares()), torch.stack(shares), atol=1e-5)


def test_the_gate_gets_gradients_and_recomputing_gives_the_same_ones():
    looped = gated().train()
    ids = text()

    def gradients(recompute):
        torch.manual_seed(2)  # the same draws in both runs
        looped.recompute = recompute
        looped.zero_grad()
        next_sum, ahead_sum = looped(ids[:, :-2], next_tokens=ids[:, 1:-1], targets=(ids[:, 1:-1], ids[:, 2:]))
        (next_sum + ahead_sum).backward()
        return {name: p.grad.clone() for name, p in looped.named_parameters() if p.grad is not None}

    plain, again = gradients(False), gradients(True)
    assert plain["loop_gate.weight"].abs().sum(1).min() > 0 and plain["loop_gate.bias"].abs().min() > 0
    assert plain.keys() == again.keys()
    assert all(torch.allclose(plain[name], again[name], atol=1e-5, rtol=1e-4) for name in plain)


def test_a_gated_window_in_segments_gives_the_whole_window_and_replays_its_routes():
    looped, routes = gated(), Routes()
    ids = text(256)[:1]
    with torch.no_grad():
        whole = looped(ids, record=routes)
        carried, parts = {}, []
        for a in (0, 128):
            place = Carried(a, carried)
            parts.append(looped(ids[:, a:a + 128], sequence=place))
            carried = {**place.passed, **{key: torch.cat([carried[key], new], 1) if key in carried else new
                                          for key, new in place.added.items()}}
        assert torch.allclose(torch.cat(parts, 1), whole, atol=1e-4)
        assert torch.equal(looped(ids, replay=routes), whole)


def test_a_gated_prompt_read_through_the_cache_gives_the_forward_pass(monkeypatch):
    monkeypatch.setattr(attention, "TOP_K", 16)
    monkeypatch.setattr(ttt, "CHUNK", 32)
    looped = gated()
    ids = text()
    with torch.no_grad():
        cache = looped.new_cache(2)
        scores, hidden, features = looped.extend(ids[:, :100], cache, features=(looped.config.n_layer,))
        expected, found = looped(ids[:, :100], features=(looped.config.n_layer,))
        assert torch.allclose(scores, expected, atol=1e-4)
        assert torch.allclose(features, found, atol=1e-4) and torch.allclose(hidden.float(), found, atol=1e-4)
