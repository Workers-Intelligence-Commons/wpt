"""A read-text step that reads its tokens several times (docs/training.md,
"Readings"): it trains on that many times the updates and tokens of one
reading, on the windows of one reading first, then on passes that take
every window of the tokens once, in another order and cut at another place
each time; a resumed step draws as the uninterrupted one, and the
evaluation's windows stay those of every other run."""

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


def drawn(dataset, step, seed=1337, stream_shards=None):
    """The windows of every update of the step, one array per update, and those of its evaluation at update 1."""
    dock, record = dataset
    environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                              seed, 1, stream_shards=stream_shards))
    try:
        updates = sum(stage.steps for stage in environment.stage_list)
        found = [torch.cat([x for x, _ in environment.batch(update, None, CPU)]).numpy()
                 for update in range(1, updates + 1)]
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


def test_a_step_of_several_readings_reads_one_reading_s_windows_then_every_window_of_each_pass_once(positions):
    once, _ = drawn(positions, read_text())
    thrice, _ = drawn(positions, read_text(3))
    assert len(thrice) == 3 * UPDATES
    # the first reading: the windows a step of one reading draws, at random places
    assert all((a == b).all() for a, b in zip(once, thrice[:UPDATES]))
    passes = []
    for number in (1, 2):
        windows = np.concatenate(thrice[number * UPDATES:(number + 1) * UPDATES])
        assert ((windows[:, 1:] == windows[:, :-1] + 1)).all()  # each a run of the corpus's tokens
        starts = windows[:, 0].tolist()
        offset = min(starts)
        fit = list(range(offset, 400 - BLOCK, BLOCK + 1))
        # cut from one place: every window that fits after it, each once, in a drawn order; where one or two
        # fewer fit than the 25 places of the pass, its last places take its first windows again
        assert offset <= BLOCK and set(starts) == set(fit) and starts[:len(fit)] != sorted(starts[:len(fit)])
        assert starts[len(fit):] == starts[:len(starts) - len(fit)]
        passes.append(starts)
    assert passes[0] != passes[1]  # another order, and another place, each pass
    again, _ = drawn(positions, read_text(3))
    assert all((a == b).all() for a, b in zip(thrice, again))  # one seed, the same windows
    other, _ = drawn(positions, read_text(3), seed=7)
    assert any((a != b).any() for a, b in zip(thrice[UPDATES:], other[UPDATES:]))


def test_the_evaluation_reads_the_same_windows_whatever_the_readings(positions):
    _, once = drawn(positions, read_text())
    _, thrice = drawn(positions, read_text(3))
    assert (once == thrice).all()


def test_a_stream_reads_its_tokens_as_many_times_drawing_as_a_step_of_one_reading(positions):
    once, _ = drawn(positions, read_text(), stream_shards=2)
    twice, _ = drawn(positions, read_text(2), stream_shards=2)
    assert len(twice) == 2 * UPDATES and all((a == b).all() for a, b in zip(once, twice))
    windows = np.concatenate(twice)
    assert ((windows[:, 1:] == windows[:, :-1] + 1) | ((windows[:, :-1] == 399) & (windows[:, 1:] == 0))).all()


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


def test_a_step_continues_only_one_that_reads_its_tokens_as_many_times():
    check_continuations([read_text(2, 4, first_drop=1.0), read_text(2, 5)])
    continued = load_step({"step": "read-text", "continues": True, "optimizer": {"name": "by-part"},
                           "settings": {"stages": [{"block_size": BLOCK, "steps": 5, "batch_size": BATCH}]}}, PARTS)
    with pytest.raises(ValueError, match="which reads its tokens 2 times: it must read them as many times"):
        check_continuations([read_text(2, 4), continued])
