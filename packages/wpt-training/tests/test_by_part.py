import contextlib
import math
from pathlib import Path

import pytest
import torch

from wpt_model import PARTS, WPT, ModelConfig
from wpt_telemetry import Telemetry
from wpt_training import (Data, FolderCheckpoint, Interrupted, Limit, Options, PluginError, Run, Strategy, load_state,
                          load_step, train, weights)

TINY = dict(vocab_size=50, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
            attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
            shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
            residual_groups=3)
CPU = Strategy(torch.device("cpu"))


def read_text(**settings):
    """A read-text step under by-part with these settings (docs/training.md, "Steps")."""
    return load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": settings},
                      "settings": {"stages": [{"block_size": 128, "share": 1}], "tokens_per_update": 512,
                                   "prefix_tokens": 1000}}, PARTS)


def by_part(model, steps=100, **settings):
    return read_text(**settings).start_optimizer(model, Run(tokens=1000, parameters=1, steps=steps,
                                                            log=lambda *_: None))


def test_the_learning_rate_warms_up_then_drops_twice():
    # docs/model.md: linear warm-up over its share of the updates, then 31.6% of the maximum after 80% of
    # the steps and 10% after 90%; Engram tables at 5 times the rate
    method = by_part(WPT(ModelConfig(**TINY)), steps=100, lr=0.01, warmup=0.1)
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
    # IMU-1's post-hoc EMA: the states at the last update and every average_every of the updates before it (a
    # share, 3 of 10), at most average_states of them, from the oldest, each later one taken in with weight
    # 1 - average_beta
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    model.optimizer_roles = lambda: [("weight", model.weight, "matrix", (1, 3, 4)),
                                     ("bias", model.bias, "vector", None)]
    method = by_part(model, steps=10, average_states=3, average_every=0.3, average_beta=0.8)
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
    short = by_part(model, steps=5, average_states=3, average_every=0.6)
    for update in range(1, 6):
        short.after_update(update, None)
    assert all(torch.equal(t, model.state_dict()[name]) for name, t in short.kept().items())


def test_a_step_stopped_and_resumed_keeps_the_average_of_its_states(dataset, tmp_path):
    # through the runner: a step of 6 updates averaging 3 states 2 updates apart, stopped after updates 2, 4 and
    # 6 and resumed each time from its checkpoint (the average with it); its result, the state its checkpoint
    # keeps, holds the average of the weights at updates 2, 4 and 6, and so does the model
    dock, record = dataset
    step = load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": {
        "lr": 0.01, "warmup": 0.34, "average_states": 3, "average_every": 0.34}}, "settings": {
        "stages": [{"block_size": 16, "share": 1}], "tokens_per_update": 32, "prefix_tokens": 192}}, PARTS)
    torch.manual_seed(1337)
    model = WPT(ModelConfig(**TINY))
    checkpoint, states = FolderCheckpoint(tmp_path / "checkpoint"), []
    for resume in (False, True, True, True):
        environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(),
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


# 2 updates of 4 windows of 16 tokens, then 2 of 2 windows of 32
TWO_STAGES = {"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.01, "warmup": 0.25}},
              "settings": {"stages": [{"block_size": 16, "share": 1}, {"block_size": 32, "share": 1}],
                           "tokens_per_update": 64, "prefix_tokens": 256}}


def test_recompute_from_keeps_every_activation_in_the_shorter_stages_and_trains_the_same(dataset):
    # docs/training.md, "Recomputing layers": through the runner, with --recompute and --recompute-from 32,
    # the stage of 16 tokens keeps every activation and the stage of 32 runs its layers again; the weights
    # are those of a run that recomputes every stage and of one that recomputes none, bit for bit, and the
    # model recomputes again once the step is over
    dock, record = dataset
    step = load_step(TWO_STAGES, PARTS)
    results = []
    for recompute, recompute_from in ((False, 0), (True, 0), (True, 32)):
        torch.manual_seed(1337)
        model = WPT(ModelConfig(**{**TINY, "block_size": 32}))
        model.recompute = recompute
        passes, run_pass = [], model._pass

        def counted(first, idx, *args, model=model, passes=passes, run_pass=run_pass):
            if torch.is_grad_enabled():
                passes.append((idx.shape[1], model.recompute))
            return run_pass(first, idx, *args)

        model._pass = counted
        environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(),
                                                  lambda _: None, 1337, 1))
        try:
            train(model, CPU, step, step.start_method(), environment, 1337, 1,
                  options=Options(recompute_from=recompute_from), log=lambda *_: None)
        finally:
            environment.close()
        results.append(({k: v.clone() for k, v in model.state_dict().items()}, sorted(set(passes)), model.recompute))
    (none, _, _), (every, every_passes, _), (longer, longer_passes, after) = results
    assert every_passes == [(16, True), (32, True)] and longer_passes == [(16, False), (32, True)] and after
    for name, value in none.items():
        assert torch.equal(value, every[name]) and torch.equal(value, longer[name]), name


GIB = 2 ** 30


class FakeMemory:
    """A device's memory as wpt_training.memory reads it, its numbers given: nothing held, (free, reserved,
    total) left, and every probe pass at the peak given for its way, (every activation kept, recomputed),
    whatever its tokens: a probe's longer window is a stage's window when it is short, so each window of a
    pass is foreseen at that peak, and 5 % more."""

    def __init__(self, left, model=None, peaks=(0, 0)):
        self.numbers, self.model, self.peaks = left, model, peaks

    def allocated(self):
        return 0

    def start(self):
        pass

    def peak(self):
        return self.peaks[self.model.recompute] if self.model is not None else 0

    def left(self):
        return self.numbers


def stages_run(dataset, monkeypatch, stages, usable, automatic=True, micro_batch=0, block_size=32):
    """Train a read-text step of these stages, each (window length, windows per update) and 2 updates, on a
    device that foresees 4 GiB for each window of a pass, 2 GiB recomputing, with `usable` GiB left: (the
    weights, the passes with gradients as (windows, tokens, recomputed), the log's lines)."""
    from wpt_training import memory

    dock, record = dataset
    [per] = {length * batch for length, batch in stages}  # every stage's update holds as many tokens
    step = load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.01, "warmup": 0.25}},
                      "settings": {"stages": [{"block_size": length, "share": 1} for length, _ in stages],
                                   "tokens_per_update": per, "prefix_tokens": 2 * per * len(stages)}}, PARTS)
    assert [(stage["steps"], stage["batch_size"]) for stage in step.stages()] == [(2, batch) for _, batch in stages]
    torch.manual_seed(1337)
    model = WPT(ModelConfig(**{**TINY, "block_size": block_size}))
    monkeypatch.setattr(memory, "reader", lambda device: FakeMemory((usable * GIB, 0, 0), model, (4 * GIB, 2 * GIB)))
    passes, run_pass, said = [], model._pass, []

    def counted(first, idx, *args):
        if torch.is_grad_enabled():
            passes.append((*idx.shape, model.recompute))
        return run_pass(first, idx, *args)

    model._pass = counted
    environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                              1337, 1, micro_batch=micro_batch))
    try:
        train(model, CPU, step, step.start_method(), environment, 1337, 1,
              options=Options(automatic=automatic, micro_batch=micro_batch), log=lambda text, *_: said.append(text))
    finally:
        environment.close()
    return {k: v.clone() for k, v in model.state_dict().items()}, passes, said


@pytest.mark.parametrize("fits", [True, False], ids=["fits", "does-not-fit"])
def test_a_stage_left_to_the_runner_chooses_from_memory_and_trains_as_chosen_by_hand(dataset, monkeypatch, fits):
    # docs/training.md, "Recomputing layers": with nothing asked, each stage of text windows probes its
    # memory (here real probe passes on the CPU, read through faked numbers that make every pass fit, or
    # only the first stage's pass that recomputes) and keeps every activation or recomputes; the weights are
    # bit for bit those of a run that recomputes none, and of one that recomputes every stage: the probes
    # leave no trace
    from wpt_training import memory

    dock, record = dataset
    step = load_step(TWO_STAGES, PARTS)
    results = []
    for recompute, automatic in ((False, False), (True, False), (False, True)):
        torch.manual_seed(1337)
        model = WPT(ModelConfig(**{**TINY, "block_size": 32}))
        model.recompute = recompute
        # the first stage's pass of 4 windows is foreseen at 16.8 GiB, and at 8.4 GiB recomputing; the second
        # stage's pass of 2 windows at 8.4 GiB, and at 4.2 GiB recomputing
        monkeypatch.setattr(memory, "reader", lambda device, model=model: FakeMemory(
            ((20 if fits else 9) * GIB, 0, 0), model, (4 * GIB, 2 * GIB)))
        passes, run_pass, said = [], model._pass, []

        def counted(first, idx, *args, model=model, passes=passes, run_pass=run_pass):
            if torch.is_grad_enabled():
                passes.append((idx.shape[1], model.recompute))
            return run_pass(first, idx, *args)

        model._pass = counted
        environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(),
                                                  lambda _: None, 1337, 1))
        try:
            train(model, CPU, step, step.start_method(), environment, 1337, 1,
                  options=Options(automatic=automatic), log=lambda text, *_: said.append(text))
        finally:
            environment.close()
        results.append(({k: v.clone() for k, v in model.state_dict().items()}, set(passes), said))
    (none, _, _), (every, _, _), (chosen, chosen_passes, said) = results
    # half a window and a window, of each stage, each way
    probes = {(length, recompute) for length in (8, 16, 32) for recompute in (False, True)}
    assert chosen_passes == probes | ({(16, True)} if not fits else set())
    choices = [line for line in said if "foreseen" in line]
    assert len(choices) == 2 and ("every activation kept: " if fits else "recompute the layers: ") in choices[0]
    assert "every activation kept: " in choices[1]
    assert "stage 1: " in choices[0] and "a pass of 64 tokens" in choices[0]  # 4 windows of 16
    for name, value in none.items():
        assert torch.equal(value, every[name]) and torch.equal(value, chosen[name]), name


@pytest.mark.parametrize("usable, windows", [(5, (2, 2)), (3, (1, 1))], ids=["two-windows", "one-window"])
def test_a_stage_whose_pass_does_not_fit_takes_fewer_windows_per_pass_and_trains_the_same(dataset, monkeypatch,
                                                                                         usable, windows):
    # docs/training.md, "Recomputing layers": a stage of 4 windows per update then one of 2, each window
    # foreseen at 2.1 GiB recomputing: with 5 GiB the first stage takes 2 windows per pass and the second
    # its whole pass, with 3 GiB each takes 1; the passes add up to the update of the whole batch, within
    # rounding
    stages = [(16, 4), (32, 2)]
    whole, _, _ = stages_run(dataset, monkeypatch, stages, usable, automatic=False)
    fewer, passes, said = stages_run(dataset, monkeypatch, stages, usable)
    # each update's passes, and where a pass holds 1 window, the probes of as many tokens that recompute
    first, second = windows
    assert passes.count((first, 16, True)) == 2 * 4 // first + (2 if first == 1 else 0)
    assert passes.count((second, 32, True)) == 2 * 2 // second + (1 if second == 1 else 0)
    assert {one[0] for one in passes} == {1, *windows}
    choices = [line for line in said if "foreseen" in line]
    assert f"stage 1: recompute the layers, {windows[0]} window{'s' if windows[0] > 1 else ''} per pass: " in choices[0]
    assert "a pass of 64 tokens" in choices[0] and f"a pass of {windows[0] * 16} tokens foreseen at" in choices[0]
    assert all(torch.allclose(value, fewer[name], atol=1e-4) for name, value in whole.items())


def test_a_stage_whose_window_does_not_fit_runs_in_segments_and_one_no_segment_fits_is_refused(dataset, monkeypatch):
    # one window of 256 tokens per update, foreseen at 2.1 GiB recomputing: with 1.5 GiB it runs in segments
    # of 128 tokens, and with 1 GiB no pass fits, which stops the step before the stage starts
    _, passes, said = stages_run(dataset, monkeypatch, [(256, 1)], 1.5, block_size=256)
    assert {one for one in passes if one[1] not in (128, 256)} == set()
    assert (1, 128, True) in passes and (1, 256, True) in passes  # the stage's segments, and a probe
    assert passes.count((1, 128, True)) == 1 + 2 * 2  # a probe, then 2 segments in each of 2 updates
    assert any("stage 1: recompute the layers, 1 window per pass in segments of 128 tokens: " in line for line in said)
    with pytest.raises(ValueError, match="stage 1: no pass fits the device's memory, down to 1 window per pass in "
                                         "segments of 128 tokens: a pass of 256 tokens .* nothing was started"):
        stages_run(dataset, monkeypatch, [(256, 1)], 1, block_size=256)


def test_the_windows_per_pass_the_options_ask_for_are_kept(dataset, monkeypatch):
    # --micro-batch 2 of a batch of 4: with 5 GiB the pass of 2 windows recomputes (4.2 GiB); with 3 GiB it
    # does not fit, and the stage is refused instead of taking 1 window per pass
    _, passes, said = stages_run(dataset, monkeypatch, [(16, 4)], 5, micro_batch=2)
    assert {one for one in passes if one[0] != 1} == {(2, 16, True)}
    assert any("stage 1: recompute the layers: a pass of 32 tokens" in line for line in said)
    with pytest.raises(ValueError, match="stage 1: no pass fits the device's memory: a pass of 32 tokens"):
        stages_run(dataset, monkeypatch, [(16, 4)], 3, micro_batch=2)


def test_every_process_runs_a_stage_as_the_one_with_the_least_memory_left(monkeypatch):
    # 2 processes share a batch of 8 windows of 256 tokens, 4 each; this one has 9 GiB and foresees its pass
    # of 4 windows at 8.4 GiB recomputing; the other's choice, as the processes exchange it, is taken where
    # it is more careful: fewer windows, segments, or a refusal
    from types import SimpleNamespace

    from wpt_training import Stage, memory, runner

    class Two(Strategy):
        def __init__(self, other):
            super().__init__(torch.device("cpu"), 0, 2)
            self.other = other

        def reduce(self, tensor, op="sum"):
            assert op == "max"
            return torch.maximum(tensor, torch.tensor(self.other))

    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 256})).train()
    monkeypatch.setattr(memory, "reader", lambda device: FakeMemory((9 * GIB, 0, 0), model, (4 * GIB, 2 * GIB)))

    def chosen(*other):
        return runner._automatic(model, read_text().start_method(), SimpleNamespace(per_pass=None),
                                 SimpleNamespace(optimizer=torch.optim.AdamW(model.parameters())), Stage(1, 8, 256),
                                 0, Options(automatic=True), Two([float(value) for value in other]),
                                 runner._Unbroken(Telemetry.off()))

    recompute, windows, segment, said = chosen(0, -4, -256, 0)  # the other keeps every activation of its 4 windows
    assert (recompute, windows, segment) == (True, 0, 0) and "another process" not in said
    recompute, windows, segment, said = chosen(1, -2, -256, 0)
    assert (recompute, windows, segment) == (True, 2, 0) and "recompute the layers, 2 windows per pass: " in said
    assert said.endswith("another process has less memory left, and every process runs the same way")
    assert chosen(1, -1, -128, 0)[:3] == (True, 1, 128)
    with pytest.raises(ValueError, match="stage 1: no pass fits the device's memory, down to 1 window per pass in "
                                         "segments of 128 tokens"):
        chosen(1, -1, -128, 1)


def test_fewer_windows_per_pass_add_up_to_the_gradient_of_the_whole_pass(dataset):
    # docs/training.md, "The batch is the same on any hardware": a stage's batch of 4 windows in passes of
    # 4, 2 or 1 windows (TextWindows.per_pass, the runner's choice) holds the same windows in the same
    # order, and the passes' gradients add up to the whole pass's within rounding
    dock, record = dataset
    step = load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.01}},
                      "settings": {"stages": [{"block_size": 16, "share": 1}], "tokens_per_update": 64,
                                   "prefix_tokens": 500}}, PARTS)
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY)).train()
    method = step.start_method()
    environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                              1337, 1))
    try:
        found = []
        for windows in (4, 2, 1):
            environment.per_pass(0, windows)
            passes = environment.batch(1, model, CPU)
            assert [len(x) for x, _ in passes] == [windows] * (4 // windows)
            model.zero_grad()
            for examples in passes:
                method.learn(model, model, examples, len(passes), CPU, 0, lambda final: contextlib.nullcontext())
            found.append((torch.cat([x for x, _ in passes]),
                          {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}))
        with pytest.raises(ValueError, match="stage 1's batch of 4 windows does not split into micro-batches of 3"):
            environment.per_pass(0, 3)
    finally:
        environment.close()
    (batch, whole), *others = found
    for windows, grads in others:
        assert torch.equal(windows, batch) and grads.keys() == whole.keys()
        assert all(torch.allclose(grads[name], value, atol=1e-6) for name, value in whole.items())


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
    method = by_part(model, lr=0.01, warmup=0.01, weight_decay=0.0)
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
    method = by_part(model, lr=0.01, warmup=0.01, weight_decay=0.0)
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
    method = by_part(model, lr=0.01, warmup=0.01)
    grouped = [id(p) for group in method.optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    before = [p.detach().clone() for p in model.parameters()]
    method.before_update(1)
    method.optimizer.step()
    still = [name for (name, p), old in zip(model.named_parameters(), before) if torch.equal(p, old)]
    assert still == []


def test_the_test_family_s_model_trains_with_a_falling_loss():
    # the architecture of the test family's size (data/orders/test.json),
    # on a repeating text it can learn in a few steps: the loss stays finite
    # and falls, through every part
    from wpt_orders import Orders
    order = Orders(Path(__file__).parents[3]).model_order("test:128")
    torch.manual_seed(0)
    model = WPT(ModelConfig(vocab_size=64, **{**order["model"], "block_size": 128})).train()
    method = by_part(model, steps=20, **{**order["steps"][0]["optimizer"]["settings"], "warmup": 0.1})
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

