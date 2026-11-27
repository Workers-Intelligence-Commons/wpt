"""How a read-text step reads its tokens (docs/training.md, "Readings"):
N readings train on N times the updates and tokens of one; every reading
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
BLOCK, BATCH, UPDATES = 15, 5, 5  # one reading of 400 tokens: 5 updates of 5 windows of 16 tokens


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


def read_text(readings=1, updates=UPDATES, **optimizer):
    settings = {"stages": [{"block_size": BLOCK, "steps": updates, "batch_size": BATCH}], "prefix_tokens": 400}
    if readings != 1:
        settings["readings"] = readings
    return load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.01, "warmup": 2,
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


def test_three_readings_train_on_three_times_the_updates_and_tokens_of_one(positions, tmp_path):
    one, three = read_text(), read_text(3)
    assert window_tokens_left([three]) == (3 * window_tokens_left([one])[0], [])
    dock, record = positions
    environment = three.start_environment(Data(three.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                               1337, 1))
    try:
        torch.manual_seed(1337)
        trained = train(WPT(ModelConfig(**TINY)), CPU, three, three.start_method(), environment, 1337, 1,
                        log=lambda *_: None)
    finally:
        environment.close()
    assert trained.steps == 3 * UPDATES
    assert environment.learned == {"plain": 400}  # the same texts, read three times


def test_the_schedule_of_three_readings_is_the_schedule_of_one_stretched_three_times():
    model = WPT(ModelConfig(**TINY))

    def rates(readings, updates):
        optimizer = read_text(readings).start_optimizer(model, Run(tokens=400, parameters=1, steps=updates,
                                                                   log=lambda *_: None, readings=readings))
        return [optimizer.factor(update) for update in range(1, updates + 1)]

    one, three = rates(1, 10), rates(3, 30)
    assert three[2::3] == one  # update 3u of three readings runs at the rate of update u of one
    assert three[:6] == [u / 6 for u in range(1, 7)]  # a warm-up of 2 updates of one reading: 6 of three


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
    assert len(thrice) == 3 * UPDATES
    windows = np.concatenate(thrice)
    assert ((windows[:, 1:] == windows[:, :-1] + 1)).all()  # each a run of the corpus's tokens
    found = passes(thrice)
    assert len(found) == 3  # 75 windows of 15 tokens predicted: a little under 3 passes over 400 tokens
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
    step = load_step({"step": "read-text", "settings": {"prefix_tokens": 400, "stages": [
        {"block_size": 5, "steps": 4, "batch_size": 2}, {"block_size": 15, "steps": 30, "batch_size": 1}]}}, PARTS)
    monkeypatch.setattr(step.environment.module, "GROUP", 2)  # groups of 2 slots of 15 tokens
    windows, _ = drawn(positions, step)
    first = passes(windows)[0]
    # 8 windows of 5 tokens fill a group and a third of the next; the second stage starts at the group after
    assert [len(window) for window in first[:8]] == [5] * 8 and {len(window) for window in first[8:]} == {15}
    read = [token for window in first for token in window.tolist()]
    assert len(read) == len(set(read))  # inputs of windows that each start on the last token of another
    every = len(read) + 20  # the 20 tokens the second group's skipped part held
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
    assert len(twice) == 2 * UPDATES
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
    resumed = run(tmp_path / "resumed", [7, 0])  # stopped after update 7, inside the second reading (6 to 10)
    assert whole["step"] == resumed["step"] == 2 * UPDATES
    for name, value in whole["model"].items():
        assert torch.equal(resumed["model"][name], value), name


@pytest.mark.parametrize("readings", [0, -1, 1.5, "two"])
def test_readings_other_than_a_whole_number_of_1_or_more_are_refused(readings):
    with pytest.raises(PluginError, match="readings"):
        load_step({"step": "read-text", "settings": {"stages": [{"block_size": 16, "steps": 1, "batch_size": 1}],
                                                     "readings": readings}}, PARTS)


def test_a_window_length_that_does_not_divide_the_longest_is_refused():
    with pytest.raises(PluginError, match="stage 1: its window length 12 does not divide the longest, 32"):
        load_step({"step": "read-text", "settings": {"stages": [{"block_size": 12, "steps": 1, "batch_size": 1},
                                                                {"block_size": 32, "steps": 1, "batch_size": 1}]}},
                  PARTS)


def test_a_step_continues_only_one_that_reads_its_tokens_as_many_times():
    check_continuations([read_text(2, 4, first_drop=1.0), read_text(2, 5)])
    continued = load_step({"step": "read-text", "continues": True, "optimizer": {"name": "by-part"},
                           "settings": {"stages": [{"block_size": BLOCK, "steps": 5, "batch_size": BATCH}]}}, PARTS)
    with pytest.raises(ValueError, match="which reads its tokens 2 times: it must read them as many times"):
        check_continuations([read_text(2, 4), continued])
