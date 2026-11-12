import json
import math
from pathlib import Path

import pytest
import torch

from wpt_model import PARTS, WPT, ModelConfig
from wpt_telemetry import Telemetry
from wpt_training import (Data, FolderCheckpoint, Interrupted, Limit, PluginError, Run, Strategy, load_state, load_step,
                          train, weights)

TINY = dict(vocab_size=50, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
            attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
            shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
            residual_groups=3)
CPU = Strategy(torch.device("cpu"))


def read_text(**settings):
    """A read-text step under by-part with these settings (docs/training.md, "Steps")."""
    return load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": settings},
                      "settings": {"stages": [{"block_size": 128, "steps": 1, "batch_size": 4}]}}, PARTS)


def by_part(model, steps=100, **settings):
    return read_text(**settings).start_optimizer(model, Run(tokens=1000, parameters=1, steps=steps,
                                                            log=lambda *_: None))


def test_the_learning_rate_warms_up_then_drops_twice():
    # docs/model.md: linear warm-up, then 31.6% of the maximum after 80% of
    # the steps and 10% after 90%; Engram tables at 5 times the rate
    method = by_part(WPT(ModelConfig(**TINY)), steps=100, lr=0.01, warmup=10)
    seen = {}
    for step in (5, 50, 85, 95):
        method.before_update(step)
        seen[step] = sorted({round(g["lr"], 6) for g in method.optimizer.param_groups})
    assert seen == {5: [0.005, 0.025], 50: [0.01, 0.05], 85: [0.00316, 0.0158], 95: [0.001, 0.005]}


def test_a_constant_schedule_keeps_the_learning_rate_at_every_update():
    # docs/training.md, "Optimizers": no warm-up and no drops; Engram tables
    # still at 5 times the rate
    method = by_part(WPT(ModelConfig(**TINY)), steps=100, lr=0.01, schedule="constant")
    for step in (1, 50, 95):
        method.before_update(step)
        assert sorted({round(g["lr"], 6) for g in method.optimizer.param_groups}) == [0.01, 0.05]


def test_early_stop_is_off_unless_patience_turns_it_on():
    # docs/training.md, "Optimizers": with the average, no evaluation is the
    # best; without it or patience every new lowest loss is the best and
    # nothing stops; with patience, a stage stops once the smoothed loss has
    # not improved by min_improve for patience evaluations, and an
    # improvement is the best
    losses = [3.0, 2.0, 2.0, 2.0, 2.0, 1.9]
    averaging = by_part(WPT(ModelConfig(**TINY)))
    assert [(v.best, v.stop) for v in (averaging.evaluated(i, loss) for i, loss in enumerate(losses))] == [
        (False, False)] * 6
    plain = by_part(WPT(ModelConfig(**TINY)), average_states=0)
    assert [(v.best, v.stop) for v in (plain.evaluated(i, loss) for i, loss in enumerate(losses))] == [
        (True, False), (True, False), (False, False), (False, False), (False, False), (True, False)]
    stopping = by_part(WPT(ModelConfig(**TINY)), patience=2, smoothing=1, average_states=0)  # 1000 tokens/param: 1%
    verdicts = [stopping.evaluated(i, loss) for i, loss in enumerate(losses[:4])]
    assert [(v.best, v.stop) for v in verdicts] == [(True, False), (True, False), (False, False), (False, True)]
    # a stage after the first starts counting again
    stopping.stage(1, 32)
    assert stopping.evaluated(5, 2.5).best
    with pytest.raises(PluginError, match=r"schedule 'cosine': one of steps, constant"):
        read_text(schedule="cosine")
    with pytest.raises(PluginError, match=r"average_states and patience"):
        read_text(patience=2)


def test_the_step_keeps_an_exponential_average_of_its_last_states():
    # IMU-1's post-hoc EMA: the states at the last update and every average_every updates before it, at most
    # average_states of them, from the oldest, each later one taken in with weight 1 - average_beta
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    model.optimizer_roles = lambda: [("weight", model.weight, "matrix", (1, 3, 4)),
                                     ("bias", model.bias, "vector", None)]
    method = by_part(model, steps=10, average_states=3, average_every=3, average_beta=0.8)
    states = {}
    for update in range(1, 11):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))  # what an update would do
        method.after_update(update, None)
        assert method.kept() is None or update == 10  # not whole before the last update
        states[update] = {name: t.clone() for name, t in model.state_dict().items()}
    kept = method.kept()
    for name in kept:
        expected = 0.8 * (0.8 * states[4][name] + 0.2 * states[7][name]) + 0.2 * states[10][name]
        assert torch.allclose(kept[name], expected, atol=1e-6)
    # a step shorter than the states counts only the updates it has
    short = by_part(model, steps=5, average_states=3, average_every=3)
    for update in range(1, 6):
        short.after_update(update, None)
    assert all(torch.equal(t, model.state_dict()[name]) for name, t in short.kept().items())


def test_a_step_stopped_and_resumed_keeps_the_average_of_its_states(dataset, tmp_path):
    # through the runner: a step of 6 updates averaging 3 states 2 updates apart, stopped after updates 2, 4 and
    # 6 and resumed each time from its checkpoint (the average with it); its result, the state its checkpoint
    # keeps, holds the average of the weights at updates 2, 4 and 6, and so does the model
    store, record = dataset
    step = load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": {
        "lr": 0.01, "warmup": 2, "average_states": 3, "average_every": 2}}, "settings": {
        "stages": [{"block_size": 16, "steps": 6, "batch_size": 2}], "prefix_tokens": 500}}, PARTS)
    torch.manual_seed(1337)
    model = WPT(ModelConfig(**TINY))
    checkpoint, states = FolderCheckpoint(tmp_path / "checkpoint"), []
    for resume in (False, True, True, True):
        environment = step.start_environment(Data(step.name, store, record, None, CPU, Telemetry.off(),
                                                  lambda _: None, 1337, 1))
        try:
            train(model, CPU, step, step.start_method(), environment, 1337, 1, checkpoint=checkpoint, resume=resume,
                  identity=1, log=lambda *_: None, save_last=True, limit=Limit(updates=2))
        except Interrupted:
            pass
        finally:
            environment.close()
        states.append(load_state(checkpoint.path))
    assert [state["step"] for state in states] == [2, 4, 6, 6]
    s2, s4, s6 = (state["model"] for state in states[:3])
    for name, value in weights(states[3]).items():
        if value.is_floating_point():
            expected = 0.8 * (0.8 * s2[name] + 0.2 * s4[name]) + 0.2 * s6[name]
            assert torch.allclose(value, expected, atol=1e-6), name
            assert torch.allclose(model.state_dict()[name], expected, atol=1e-6), name


@pytest.mark.parametrize("weight", [
    lambda m: m.layers[0].attention.q.weight,     # a window layer's queries, 2 heads
    lambda m: m.layers[0].attention.gate.weight,  # its output gate, 2 heads
    lambda m: m.layers[1].attention.q.weight,     # an exact layer's queries
    lambda m: m.layers[4].attention.q.weight,     # a sparse layer's queries
    lambda m: m.layers[4].attention.gate.weight,  # and its output gate
])
def test_each_head_moves_as_much_whatever_its_gradient_size(weight):
    # per-head NorMuon (K3, DeepSeek-V4.1): a part with tiny gradients is not swamped by a part
    # with large ones; each part's step has an RMS of 0.18
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY))
    method = by_part(model, lr=0.01, warmup=1, weight_decay=0.0)
    weight = weight(model)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    rows = weight.shape[0] // 2
    weight.grad = torch.randn_like(weight)
    weight.grad[rows:] *= 1e-4
    before = weight.detach().clone()
    method.before_update(1)
    method.optimizer.step()
    step = (weight.detach() - before) / 0.01
    for part in (step[:rows], step[rows:]):
        assert abs(part.pow(2).mean().sqrt().item() - 0.18) < 1e-4


@pytest.mark.parametrize("shape", [(64, 256), (256, 64), (96, 96)])
def test_a_matrix_s_update_has_an_rms_of_0_18(shape):
    # DeepSeek-V4's RMS of 0.18, which NorMuon's scaling sets exactly, so
    # AdamW's learning rates carry over: before weight decay, a matrix moves
    # by lr times an update whose root mean square is 0.18, whatever its
    # shape and gradient size
    torch.manual_seed(0)
    model = torch.nn.Linear(shape[1], shape[0], bias=False)
    model.optimizer_roles = lambda: [("weight", model.weight, "matrix", (1, *shape))]
    method = by_part(model, lr=0.01, warmup=1, weight_decay=0.0)
    model.weight.grad = torch.randn(shape) * 1e3
    before = model.weight.detach().clone()
    method.before_update(1)
    method.optimizer.step()
    rms = ((model.weight.detach() - before) / 0.01).pow(2).mean().sqrt()
    assert abs(rms.item() - 0.18) < 0.002


def test_every_parameter_takes_a_step():
    # the attention's parameters (sinks, gates, QK-norm weights, the MTP
    # block's window attention) each belong to one optimizer group, once
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY))
    method = by_part(model, lr=0.01, warmup=1)
    grouped = [id(p) for group in method.optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    before = [p.detach().clone() for p in model.parameters()]
    method.before_update(1)
    method.optimizer.step()
    still = [name for (name, p), old in zip(model.named_parameters(), before) if torch.equal(p, old)]
    assert still == []


def test_the_classics_test_recipe_trains_with_a_falling_loss():
    # the test recipe's architecture (data/recipes/models/classics-test.json),
    # on a repeating text it can learn in a few steps: the loss stays finite
    # and falls, through every part
    path = Path(__file__).parents[3] / "data/recipes/models/classics-test.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))
    torch.manual_seed(0)
    model = WPT(ModelConfig(vocab_size=64, **{**recipe["model"], "block_size": 128})).train()
    method = by_part(model, steps=20, **{**recipe["steps"][0]["optimizer"]["settings"], "warmup": 2})
    loss_of = read_text().start_method()  # next-token, with its multi-token prediction schedule
    text = torch.arange(129).remainder(13)[None].expand(4, -1)  # a period of 13 tokens
    x, y = text[:, :-1], text[:, 1:]
    losses = []
    for step in range(1, 13):
        method.optimizer.zero_grad()
        loss = loss_of.loss(model, x, y)
        loss.backward()
        method.before_update(step)
        method.optimizer.step()
        model.after_update(lambda tensor, op: tensor)
        method.after_update(step, lambda tensor, op: tensor)
        loss_of.after_update(step)
        losses.append(loss.item())
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < 0.5 * losses[0]

