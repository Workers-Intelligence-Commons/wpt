"""How a read-text step reads its tokens (docs/training.md, "Readings"):
N readings train on the updates of N times its tokens; every reading
is a pass that predicts each token once, in another order and cut at
another place each time; a resumed step draws as the uninterrupted one,
and the evaluation's windows stay those of every other run."""

import numpy as np
import pytest
import torch

from dock import api
from wpt_corpus.files import DockFiles
from wpt_corpus.shards import write_shard
from wpt_model import PARTS, WPT, ModelConfig
from wpt_telemetry import Telemetry
from wpt_training import (Data, FolderCheckpoint, Interrupted, Limit, PluginError, Run, Strategy, check_continuations,
                          load_state, load_step, train, window_tokens_left)

TINY = dict(vocab_size=400, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
            attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
            shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
            residual_groups=3)
CPU = Strategy(torch.device("cpu"))
# 400 tokens read in updates of 5 windows of 15 tokens predicted, 75 tokens: 5 updates for one reading
# (5.33 rounded), 11 for two (10.67) and 16 for three (exactly 1,200 tokens)
BLOCK, BATCH = 15, 5
UPDATES = {1: 5, 2: 11, 3: 16}


@pytest.fixture
def positions(tmp_path):
    """A dock holding 4 token files of 100 tokens, token i at place i of the
    corpus's 400 tokens of plain text, so a window's first token is where it
    starts; and the corpus's record."""
    dock = (tmp_path / "dock").as_uri()
    shards = []
    for i in range(4):
        path = tmp_path / f"{i}.parquet"
        digest = write_shard(path, [(list(range(100 * i, 100 * i + 100)), f"s{i}", f"s{i}/d", "p", 0)])
        stored = api.put_from(dock, f"corpora/d/{digest}.parquet", path)
        shards.append({"key": stored["key"], "sha256": stored["sha256"], "token_sha256": digest, "rows": 1,
                       "tokens": 100})
    return DockFiles(dock, tmp_path / "reader"), {"shards": {"plain": shards}, "total_tokens": 400}


def read_text(readings=1, last_update=0, **optimizer):
    settings = {"stages": [{"block_size": BLOCK, "share": 1}], "tokens_per_update": BLOCK * BATCH,
                "prefix_tokens": 400}
    if readings != 1:
        settings["readings"] = readings
    if last_update:
        settings["last_update"] = last_update
    return load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.01, "warmup": 0.2,
                                                                                          **optimizer}},
                      "settings": settings}, PARTS)


def drawn(dataset, step, seed=1337, stream_shards=None, number=1):
    """The windows of every update of the step, one array per update, and those of its evaluation at update 1."""
    dock, record = dataset
    environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                              seed, number, stream_shards=stream_shards))
    try:
        found = []
        for index, stage in enumerate(environment.stage_list):
            environment.stage(index)
            first = len(found) + 1
            found += [torch.cat([x for x, _ in environment.batch(update, None, CPU)]).numpy()
                      for update in range(first, first + stage.steps)]
        environment.stage(0)
        return found, torch.cat([x for x, _ in environment.evaluation_passes(1)]).numpy()
    finally:
        environment.close()


def test_three_readings_train_on_the_updates_of_three_times_the_tokens(positions, tmp_path):
    one, three = read_text(), read_text(3)
    assert window_tokens_left([one]) == (UPDATES[1] * BLOCK * BATCH, [])
    assert window_tokens_left([three]) == (3 * 400, [])
    dock, record = positions
    environment = three.start_environment(Data(three.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                               1337, 1))
    try:
        torch.manual_seed(1337)
        trained = train(WPT(ModelConfig(**TINY)), CPU, three, three.start_method(), environment, 1337, 1,
                        log=lambda *_: None)
    finally:
        environment.close()
    assert trained.steps == UPDATES[3]
    assert environment.learned == {"plain": 400}  # the same texts, read three times


def test_the_schedule_of_three_readings_is_the_schedule_of_one_stretched_three_times():
    model = WPT(ModelConfig(**TINY))

    def rates(readings, updates):
        optimizer = read_text(readings).start_optimizer(model, Run(tokens=400, parameters=1, steps=updates,
                                                                   log=lambda *_: None, readings=readings))
        return [optimizer.factor(update) for update in range(1, updates + 1)]

    one, three = rates(1, 10), rates(3, 30)
    assert three[2::3] == one  # update 3u of three readings runs at the rate of update u of one
    assert three[:6] == [u / 6 for u in range(1, 7)]  # a warm-up of a fifth: 2 updates of 10, 6 of 30


def test_the_weight_decay_of_n_readings_is_the_order_s_over_n_for_every_part_that_decays():
    # the learning rate times the weight decay times the updates stays that of one reading
    model, said = WPT(ModelConfig(**TINY)), []

    def decays(readings):
        optimizer = read_text(readings, weight_decay=0.3).start_optimizer(
            model, Run(tokens=400, parameters=1, steps=10 * readings, log=said.append, readings=readings))
        return {("matrix" if group["muon"] else "table" if group["lr_factor"] != 1.0 else "vector"):
                group["weight_decay"] for group in optimizer.optimizer.param_groups}

    assert decays(1) == {"matrix": 0.3, "vector": 0.3, "table": 0.0}
    assert any(line.endswith("; weight decay 0.3") for line in said)
    assert decays(3) == {"matrix": 0.3 / 3, "vector": 0.3 / 3, "table": 0.0}  # Engram's tables never decay
    assert any(line.endswith("; weight decay 0.1 (weight_decay 0.3 over 3 readings)") for line in said)


def passes(updates):
    """The windows of a step's updates, in order, cut into its passes: a pass
    ends before the first window that predicts a token the pass predicted."""
    found, seen = [[]], set()
    for window in (window for update in updates for window in update):
        predicted = set(window[1:].tolist())
        if predicted & seen:
            found.append([])
            seen = set()
        found[-1].append(window)
        seen |= predicted
    return found


def test_every_reading_predicts_each_token_once_in_an_order_of_its_own(positions):
    thrice, _ = drawn(positions, read_text(3))
    assert len(thrice) == UPDATES[3]
    windows = np.concatenate(thrice)
    assert ((windows[:, 1:] == windows[:, :-1] + 1)).all()  # each a run of the corpus's tokens
    found = passes(thrice)
    # 80 windows of 15 tokens, 1,200 tokens predicted: 3 passes over 400 tokens, each of 25 or 26 windows after its
    # offset, and the start of a fourth
    assert len(found) == 4
    orders = []
    for one in found[:-1]:
        starts = [int(window[0]) for window in one]
        # cut from one offset into windows that start every 15 tokens, each window once, in a drawn order: every
        # token but those before the offset and after the last whole window is predicted once
        offset = min(starts) % BLOCK
        assert sorted(starts) == list(range(offset, offset + BLOCK * len(starts), BLOCK))
        assert 400 - 1 - 2 * BLOCK < BLOCK * len(starts) <= 400 - 1 and starts != sorted(starts)
        orders.append(starts)
    assert orders[0] != orders[1]  # another order, or another cut, each pass
    again, _ = drawn(positions, read_text(3))
    assert all((a == b).all() for a, b in zip(thrice, again))  # one seed, the same windows
    other, _ = drawn(positions, read_text(3), seed=7)
    assert any((a != b).any() for a, b in zip(thrice, other))


def test_a_stage_that_starts_inside_a_pass_predicts_no_token_the_pass_predicted(positions, monkeypatch):
    # the longest window gives the slots; a stage of shorter windows reads windows inside them, and the next
    # stage starts at the next group of slots, so a pass never predicts a token twice
    # 15 tokens an update over 400 tokens, 26.67 updates: 3 of 3 windows of 5, then 24 of 1 window of 15
    step = load_step({"step": "read-text", "settings": {"prefix_tokens": 400, "tokens_per_update": 15, "stages": [
        {"block_size": 5, "share": 3}, {"block_size": 15, "share": 24}]}}, PARTS)
    assert [(stage["steps"], stage["batch_size"]) for stage in step.stages()] == [(3, 3), (24, 1)]
    monkeypatch.setattr(step.environment.module, "GROUP", 2)  # groups of 2 slots of 15 tokens
    # a schedule not rounded to whole groups, as a stage that crosses into another pass or token file is
    monkeypatch.setattr(step.environment.module, "rounded", lambda settings: False)
    windows, _ = drawn(positions, step)
    first = passes(windows)[0]
    # 9 windows of 5 tokens fill a group and half of the next; the second stage starts at the group after
    assert [len(window) for window in first[:9]] == [5] * 9 and {len(window) for window in first[9:]} == {15}
    read = [token for window in first for token in window.tolist()]
    assert len(read) == len(set(read))  # inputs of windows that each start on the last token of another
    every = len(read) + 15  # the 15 tokens the second group's skipped part held
    assert 400 - 2 * 15 < every <= 400


def test_a_corpus_with_a_held_out_portion_is_evaluated_on_it_alone_on_the_same_windows_for_every_run(positions,
                                                                                                     tmp_path):
    dock, record = positions
    path = tmp_path / "held.parquet"  # 300 tokens no training file holds: 1000 to 1299
    digest = write_shard(path, [(list(range(1000, 1300)), "h", "h/d", "p", 0)])
    stored = api.put_from((tmp_path / "dock").as_uri(), f"corpora/d/{digest}.parquet", path)
    held = {"shards": [{"key": stored["key"], "sha256": stored["sha256"], "token_sha256": digest, "rows": 1,
                        "tokens": 300}], "total_tokens": 300}
    with_held = (dock, {**record, "held_out": held})
    trained, evaluated = drawn(with_held, read_text(3))
    assert evaluated.min() >= 1000 and evaluated.max() < 1300  # held-out tokens only
    assert all((a == b).all() for a, b in zip(trained, drawn(positions, read_text(3))[0]))  # training unchanged
    for seed in (1337, 7):  # the same windows for every run, seed and step
        for number in (1, 2):
            assert (drawn(with_held, read_text(), seed=seed, number=number)[1] == evaluated).all()
    step = read_text()
    environment = step.start_environment(Data(step.name, dock, with_held[1], None, CPU, Telemetry.off(),
                                              lambda _: None, 1337, 1))
    environment.close()
    assert environment.validation == "held_out"
    environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                              1337, 1))
    environment.close()
    assert environment.validation == "training"


def test_the_evaluation_reads_the_same_windows_whatever_the_readings(positions):
    _, once = drawn(positions, read_text())
    _, thrice = drawn(positions, read_text(3))
    assert (once == thrice).all()


def test_a_stream_reads_each_token_file_in_passes_without_crossing_from_one_to_the_next(positions):
    twice, _ = drawn(positions, read_text(2), stream_shards=2)
    assert len(twice) == UPDATES[2]
    windows = np.concatenate(twice)
    assert ((windows[:, 1:] == windows[:, :-1] + 1)).all() and (windows[:, 0] // 100 == windows[:, -1] // 100).all()
    found = passes(twice)
    files = [int(window[0]) // 100 for window in found[0]]
    assert files == sorted(files)  # the token files in their order, each one's windows in an order of its own
    # each of the 4 files of 100 tokens holds 6 windows of 15 tokens predicted after its offset, or 5
    assert 4 * 5 <= len(found[0]) <= 4 * 6


def test_a_step_stopped_in_its_second_reading_and_resumed_trains_as_the_uninterrupted_one(positions, tmp_path):
    dock, record = positions
    step = read_text(2)

    def run(folder, stops):
        torch.manual_seed(1337)
        model = WPT(ModelConfig(**TINY))
        checkpoint = FolderCheckpoint(folder)
        for number, stop in enumerate(stops):
            environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(),
                                                      lambda _: None, 1337, 1))
            try:
                train(model, CPU, step, step.start_method(), environment, 1337, 1, checkpoint=checkpoint,
                      resume=number > 0, identity=1, log=lambda *_: None, save_last=True,
                      limit=Limit(updates=stop) if stop else None)
            except Interrupted:
                pass
            finally:
                environment.close()
        return load_state(checkpoint.path)

    whole = run(tmp_path / "whole", [0])
    # stopped after update 7, inside the second pass, which starts in update 6 (a pass predicts at most 390 tokens)
    resumed = run(tmp_path / "resumed", [7, 0])
    assert whole["step"] == resumed["step"] == UPDATES[2]
    for name, value in whole["model"].items():
        assert torch.equal(resumed["model"][name], value), name


def test_each_stage_makes_its_share_of_the_updates_of_every_reading():
    # 2 readings of 1,000 tokens in updates of 64 tokens: 31.25 updates, shared 1:1:2 as 7.8, 7.8 and 15.6, each
    # rounded; each stage's batch is 64 tokens in windows of its length
    settings = {"stages": [{"block_size": 4, "share": 1}, {"block_size": 8, "share": 1}, {"block_size": 16, "share": 2}],
                "tokens_per_update": 64, "prefix_tokens": 1000, "readings": 2}
    step = load_step({"step": "read-text", "settings": settings}, PARTS)
    assert [(stage["steps"], stage["batch_size"]) for stage in step.stages()] == [(8, 16), (8, 8), (16, 4)]
    # a part of that schedule, cut after its update 12: the second stage's first 4 updates, and no third stage
    cut = load_step({"step": "read-text", "settings": {**settings, "last_update": 12}}, PARTS)
    assert [(stage["steps"], stage["batch_size"]) for stage in cut.stages()] == [(8, 16), (4, 8)]
    with pytest.raises(PluginError, match="last_update 33: the schedule makes 32 updates"):
        load_step({"step": "read-text", "settings": {**settings, "last_update": 33}}, PARTS)


def test_the_test_family_reads_its_corpus_once_in_updates_of_64_windows_of_256():
    # 4,290,110 tokens in updates of 16,384: 261.85 updates, 262
    from pathlib import Path

    from wpt_orders import Orders
    [entry] = Orders(Path(__file__).resolve().parents[3]).model_order("test:128")["steps"]
    assert load_step(entry, PARTS).stages() == [{"block_size": 256, "steps": 262, "batch_size": 64}]


def test_a_step_whose_updates_fill_a_group_per_stage_rounds_its_stages_to_whole_groups():
    # updates of 64 tokens, a longest window of 16: a group of 64 slots is 1,024 tokens, 16 updates
    stages = [{"block_size": 4, "share": 1}, {"block_size": 8, "share": 1}, {"block_size": 16, "share": 2}]
    settings = {"stages": stages, "tokens_per_update": 64, "prefix_tokens": 6400}
    step = load_step({"step": "read-text", "settings": settings}, PARTS)
    module = step.environment.module
    assert module.group_updates(step.environment_settings) == 16 and module.rounded(step.environment_settings)
    # 100 updates computed, 25, 25 and 50: 6.25 groups, 6, shared 1.5, 1.5 and 3, the first remainder first
    assert [float(one) for one in module.computed(step.environment_settings)] == [25, 25, 50]
    assert [stage["steps"] for stage in step.stages()] == [32, 16, 48]
    # a stage too small for a group of its own still gets one
    small = load_step({"step": "read-text", "settings": {**settings, "stages": [
        {"block_size": 4, "share": 1}, {"block_size": 16, "share": 30}]}}, PARTS)
    assert [stage["steps"] for stage in small.stages()] == [16, 80]
    # 31.25 updates do not fill a group for each of three stages: the computed updates, rounded half up
    few = load_step({"step": "read-text", "settings": {**settings, "prefix_tokens": 2000}}, PARTS)
    assert not module.rounded(few.environment_settings)
    assert [stage["steps"] for stage in few.stages()] == [8, 8, 16]


@pytest.mark.parametrize("order", ["pocket:64", "pocket:96", "pocket:128", "pocket:192", "pocket:256", "coherent:512"])
def test_each_size_of_nine_stages_reads_its_text_once_skipping_no_group(order):
    # one pass over the size's own tokens (one token file, as without --stream): no stage boundary skips the
    # end of a group, and the stages predict the tokens of the pass 1.00 times within 1%
    from pathlib import Path

    from wpt_orders import Orders
    [entry] = Orders(Path(__file__).resolve().parents[3]).model_order(order)["steps"]
    step = load_step(entry, PARTS)
    module, settings = step.environment.module, step.environment_settings
    assert module.rounded(settings)
    longest = max(stage["block_size"] for stage in settings["stages"])
    walk = module._Walk([(0, settings["prefix_tokens"])], longest, (1337, 1))
    place = 0
    for stage in step.stages():
        assert walk.aligned(place) == place
        place += stage["steps"] * stage["batch_size"] * stage["block_size"]
    walk._pass(place)
    assert abs(place / walk.starts[1] - 1) < 0.01


@pytest.mark.parametrize("readings", [0, -1, 1.5, "two"])
def test_readings_other_than_a_whole_number_of_1_or_more_are_refused(readings):
    with pytest.raises(PluginError, match="readings"):
        load_step({"step": "read-text", "settings": {"stages": [{"block_size": 16, "share": 1}],
                                                     "tokens_per_update": 16, "prefix_tokens": 400,
                                                     "readings": readings}}, PARTS)


def test_a_window_length_that_does_not_divide_the_longest_is_refused():
    with pytest.raises(PluginError, match="stage 1: its window length 12 does not divide the longest, 32"):
        load_step({"step": "read-text", "settings": {"stages": [{"block_size": 12, "share": 1},
                                                                {"block_size": 32, "share": 1}],
                                                     "tokens_per_update": 96, "prefix_tokens": 400}}, PARTS)


def test_a_step_continues_only_one_that_reads_its_tokens_as_many_times():
    check_continuations([read_text(2, 4, first_drop=1.0), read_text(2, 5)])
    continued = load_step({"step": "read-text", "continues": True, "optimizer": {"name": "by-part"},
                           "settings": {"stages": [{"block_size": BLOCK, "share": 1}],
                                        "tokens_per_update": BLOCK * BATCH, "prefix_tokens": 400}}, PARTS)
    with pytest.raises(ValueError, match="which reads its tokens 2 times: it must read them as many times"):
        check_continuations([read_text(2, 4), continued])
