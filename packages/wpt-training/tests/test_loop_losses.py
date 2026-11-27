"""A looping model whose loss weighs several passes (docs/training.md, "In-
Place TTT and loops in training"): the run's record and log give each
weighed pass's training and validation loss, and a model that weighs the
last pass alone records what it always did."""

import contextlib
import math

import torch

from wpt_model import PARTS, WPT, ModelConfig
from wpt_telemetry import Telemetry
from wpt_training import Data, Strategy, draw_numbers, load_step, train

TINY = dict(vocab_size=50, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
            attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
            shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
            residual_groups=3, loops=1)
CPU = Strategy(torch.device("cpu"))


class Kept:
    """A run record that keeps its events."""

    def __init__(self):
        self.events = []

    def event(self, name, /, values=None, **attributes):
        self.events.append((name, values or {}))

    def log(self, *args, **kwargs):
        pass

    @contextlib.contextmanager
    def span(self, name, /, **attributes):
        yield None


def run(dataset, **rules):
    """6 updates of 2 windows of 16 tokens, evaluated at the third and the
    last: the events of the run's record, and its log lines."""
    dock, record = dataset
    step = load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.01, "warmup": 2}},
                      "settings": {"stages": [{"block_size": 16, "steps": 6, "batch_size": 2}], "prefix_tokens": 500,
                                   "eval_every": 3}}, PARTS)
    [draws] = draw_numbers([step])
    torch.manual_seed(1337)
    model = WPT(ModelConfig(**TINY, **rules))
    environment = step.start_environment(Data(step.name, dock, record, None, CPU, Telemetry.off(), lambda _: None,
                                              1337, draws))
    kept, lines = Kept(), []
    try:
        train(model, CPU, step, step.start_method(), environment, 1337, draws, log=lines.append, telemetry=kept)
    finally:
        environment.close()
    return kept.events, lines


def test_each_weighed_pass_s_losses_are_in_the_record_and_the_log(dataset):
    events, lines = run(dataset, loop_losses=[1, 1])
    steps = [values for name, values in events if name == "training.step"]
    evaluations = [values for name, values in events if name == "training.evaluation"]
    assert len(steps) == 6 and len(evaluations) == 3
    for values in steps:
        assert {"train_loss_pass_1", "train_loss_pass_2"} <= values.keys()
        # the update's loss: the two passes' next-token losses at equal weight, plus the multi-token prediction's
        assert values["train_loss"] > (values["train_loss_pass_1"] + values["train_loss_pass_2"]) / 2
    for values in evaluations:
        # what evaluations compare and keep the best weights by: the last pass's
        assert values["val_loss_pass_2"] == values["val_loss"]
        assert math.isfinite(values["val_loss_pass_1"]) and values["val_loss_pass_1"] != values["val_loss"]
    assert all("pass 1 " in line and "pass 2 " in line for line in lines if "val loss" in line)


def test_a_model_that_weighs_the_last_pass_alone_records_no_pass_s_loss(dataset):
    events, lines = run(dataset)
    assert not any("_pass_" in key for _, values in events for key in values)
    assert not any("pass 1" in line for line in lines)
