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
import torch.nn.functional as F

from wpt_model import WPT, ModelConfig, Routes
from wpt_model import attention, ttt
from wpt_model.sequence import WHOLE, Carried

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


@pytest.mark.parametrize("decay", [{}, dict(loop_decay=0.25, decay_last_pass=True)])
def test_with_loop_losses_the_next_token_loss_is_the_weighted_sum_of_each_pass_s(decay):
    # Mixture-of-Loops' Eq. 8: sum_r w_r / sum(w) L_r, each pass read through the one head; the multi-token
    # prediction block and the scores read the last pass alone
    plain, last, weighed = model(0), model(2, **decay), model(2, loop_losses=[1, 0, 3], **decay)
    last.load_state_dict(plain.state_dict())
    weighed.load_state_dict(plain.state_dict())
    ids = text()[:, :90]
    x, y, after = ids[:, :-2], ids[:, 1:-1], ids[:, 2:]
    with torch.no_grad():
        start = hidden = plain.embedding(x)
        each = []
        for loop in range(3):
            out = one_pass(plain, hidden, x, loop)
            hidden = start + 0.25 * (out - start) if decay else out
            logits = plain.head(plain.final_norm(hidden if decay else out)).float()
            each.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum"))
        next_sum, ahead_sum, by_pass = weighed(x, next_tokens=y, targets=(y, after), by_pass=True)
        assert torch.allclose(by_pass, torch.stack([each[0], each[2]]), rtol=1e-5)
        assert torch.allclose(next_sum, (each[0] + 3 * each[2]) / 4, rtol=1e-5)
        assert torch.equal(ahead_sum, last(x, next_tokens=y, targets=(y, after))[1])
        assert torch.equal(weighed(x), last(x))
        # the last pass alone, said or left out, is the loss of a loop without the rule
        alone = model(2, loop_losses=[0, 0, 5], **decay)
        alone.load_state_dict(plain.state_dict())
        assert all(torch.equal(a, b) for a, b in zip(alone(x, next_tokens=y, targets=(y, after)),
                                                     last(x, next_tokens=y, targets=(y, after))))


FIRST_HALF = 3  # CONFIG's layers: window, exact, window (the first half), then exact and sparse


def through(found, numbers, residuals, first, ids, loop, bridge=None):
    """Layers `numbers` of the model found, in order, from first, through
    one set of Attention Residuals' sums, each reading what docs/model.md
    says it reads: (their output, the first half's output as the second
    half's exact layers read it)."""
    sources, handoff = residuals.start(first), None
    canonical, embedding = found.canonical[ids], found.embedding(ids)
    for i, number in enumerate(numbers):
        layer = found.layers[number - 1]
        h = sources.input(2 * i)
        if number == FIRST_HALF + 1:
            bridge = h
        out, given = layer.attend(h, canonical, WHOLE, bridge, handoff)
        handoff = given if given is not None else handoff
        sources.output(2 * i, out)
        sources.output(2 * i + 1, layer.feed(sources.input(2 * i + 1), ids, embedding, WHOLE, loop=loop))
    return sources.end(), bridge


@pytest.mark.parametrize("looped", [1, 2, 3])
@pytest.mark.parametrize("rules", [{}, dict(loop_decay=0.25, decay_last_pass=True, loop_losses=[1, 0, 2])])
def test_with_loop_layers_the_loop_runs_the_first_layers_and_the_rest_run_once_after_it(looped, rules):
    # the first `looped` layers on every pass, each pass from the one before's output (or its decayed change);
    # then the other layers once, from what follows the last pass, through sums of their own, the second half's
    # exact layers reading the last pass's first-half output. A weighed pass's loss reads the layers after the loop
    # run from what follows that pass
    found = model(2, loop_layers=looped, **rules)
    ids = text()[:, :90]
    x, y = ids[:, :-1], ids[:, 1:]
    loop_part, coda = range(1, looped + 1), range(looped + 1, 6)
    with torch.no_grad():
        start = hidden = found.embedding(x)
        each = []
        for loop in range(3):
            out, bridge = through(found, loop_part, found.residuals, hidden, x, loop)
            hidden = start + 0.25 * (out - start) if rules else out
            if loop == 2 or rules and loop == 0:
                final, _ = through(found, coda, found.coda_residuals, hidden, x, loop, bridge)
                scores = found.head(found.final_norm(final)).float()
                each.append(F.cross_entropy(scores.flatten(0, 1), y.flatten(), reduction="sum"))
        got, features = found(x, features=(looped, 5))
        assert torch.allclose(got, scores, atol=1e-5)
        # a drafter's features keep the layers' numbers: layer `looped`'s is what follows the loop
        assert torch.allclose(features[..., :16], hidden.float(), atol=1e-5)
        assert torch.allclose(features[..., 16:], final.float(), atol=1e-5)
        if rules:
            _, by_pass = found(x, targets=(y,), by_pass=True)
            assert torch.allclose(by_pass, torch.stack(each), rtol=1e-5)


def test_a_loop_of_the_first_layers_has_the_parameters_of_a_loop_of_every_layer():
    every, first = model(1), model(1, loop_layers=2)
    assert sum(p.numel() for p in every.parameters()) == sum(p.numel() for p in first.parameters())
    assert {part: len(found) for part, found in every.parts().items()} == \
        {part: len(found) for part, found in first.parts().items()}


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
    (dict(loops=0, loop_losses=[1]), "loop_losses shapes the passes of a loop, and loops is 0"),
    (dict(loops=2, loop_losses=[1, 1]), "loop_losses gives one weight per pass: 3, not 2"),
    (dict(loops=1, loop_losses=[1, 0]), "the last pass's above 0"),
    (dict(loops=1, loop_losses=[-1, 2]), "loop_losses are at least 0"),
    (dict(loops=0, loop_layers=2), "loop_layers shapes the passes of a loop, and loops is 0"),
    (dict(loops=1, loop_layers=5), "loop_layers must be between 1 and 4"),
    (dict(loops=1, loop_layers=0), "loop_layers must be between 1 and 4"),
    (dict(loops=1, loop_layers=4), "would run sparse layer 5 once after the loop, while exact layer 4"),
])
def test_rules_of_the_loop_that_cannot_hold_are_refused(rules, problem):
    with pytest.raises(ValueError, match=problem):
        ModelConfig(**CONFIG, **rules)


@pytest.mark.parametrize("loop_layers", [None, 3])
def test_each_pass_routes_on_its_own_and_replays_its_own_choices(loop_layers):
    # the layers after a loop of the first layers (here the routed layers 4 and 5) route once, in the last pass
    looped, routes = model(1, **({} if loop_layers is None else dict(loop_layers=loop_layers))), Routes()
    ids = text()[:1]
    with torch.no_grad():
        recorded = looped(ids, record=routes)
        replayed = looped(ids, replay=routes)
    last = looped.config.looped_layers
    routed = [(number, layer.experts) for number, layer in enumerate(looped.layers, start=1)
              if not layer.experts.hashed]
    assert set(routes.chosen) == {(experts, loop) for number, experts in routed
                                  for loop in ((0, 1) if number <= last else (1,))}
    assert torch.equal(recorded, replayed)


@pytest.mark.parametrize("rules", [{}, dict(loop_decay=0.25), dict(loop_decay=0.25, decay_last_pass=True),
                                   dict(loop_layers=1), dict(loop_layers=3, loop_decay=0.25, decay_last_pass=True)])
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
        # each pass keeps its own values of the earlier tokens; the layers after the loop, the last pass's only
        modules = {id(module): number for number, layer in enumerate(looped.layers, start=1)
                   for module in layer.modules()}
        loop_part = rules.get("loop_layers", 5)
        for module, name, loop in cache.text.values:
            assert modules[module] <= loop_part or loop == 2
        kept = {key[:2] for key in cache.text.values if modules[key[0]] <= loop_part}
        assert kept and {key for key in cache.text.values if key[:2] in kept} == \
            {(*key, loop) for key in kept for loop in range(3)}


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


@pytest.mark.parametrize("rules", [dict(loop_decay=0.25), dict(loop_decay=0.25, decay_last_pass=True),
                                   dict(loop_losses=[1, 1, 1]), dict(loop_decay=0.25, loop_losses=[1, 0, 2]),
                                   dict(loop_layers=3, loop_decay=0.25, decay_last_pass=True, loop_losses=[1, 1, 1])])
def test_a_looping_model_with_its_rules_learns(rules):
    losses = trained(rules)
    assert losses[-1] < losses[0] - 0.1
