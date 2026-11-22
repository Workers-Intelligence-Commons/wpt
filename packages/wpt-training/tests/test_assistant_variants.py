"""The answer-as-assistant-dft and answer-as-assistant-neftune steps and their
methods (docs/training.md, "The answer-as-assistant-dft step", "The
answer-as-assistant-neftune step")."""

import contextlib
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wpt_model import PARTS
from wpt_training import PluginError, load_step

SINGLE = SimpleNamespace(world_size=1)


def quiet(final):
    return contextlib.nullcontext()


class Tiny(nn.Module):
    """An embedding and a head; keeps the embeddings its last pass read."""

    def __init__(self, vocab=7, width=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, idx):
        self.seen = self.embedding(idx)
        return self.head(self.seen)


def method_of(name, settings=None):
    step = load_step({"step": name, "settings": settings or {}}, PARTS)
    return step.method.module.start(step.method_settings)


@pytest.mark.parametrize("name, method", [("answer-as-assistant-dft", "assistant-tokens-dft"),
                                          ("answer-as-assistant-neftune", "assistant-tokens-neftune")])
def test_both_steps_load_through_the_step_registry(name, method):
    step = load_step({"step": name}, PARTS)
    assert (step.method.name, step.environment.name, step.optimizer.name) == (method, "conversations", "by-part")
    assert step.kinds == ("conversation",) and step.locked == ("balancing",)
    assert step.settings["tag"] == "answer-as-assistant"
    assert step.record()["step"]["file"] == f"wpt_training/steps/{name}.py"


def test_neftune_s_alpha_is_a_method_setting_of_5_by_default_and_never_below_0():
    assert load_step({"step": "answer-as-assistant-neftune"}, PARTS).method_settings == {"alpha": 5.0}
    assert load_step({"step": "answer-as-assistant-neftune", "settings": {"alpha": 2}}, PARTS).method_settings == {
        "alpha": 2.0}
    with pytest.raises(PluginError, match="alpha: 0 or more"):
        load_step({"step": "answer-as-assistant-neftune", "settings": {"alpha": -1}}, PARTS)


def test_dft_s_loss_is_minus_p_log_p_of_the_learned_tokens_and_no_gradient_goes_through_p():
    torch.manual_seed(0)
    model = Tiny()
    inputs = torch.tensor([[1, 2, 3], [4, 5, 0]])
    targets = torch.tensor([[2, -100, 4], [5, 6, -100]])
    learned = 4
    share = method_of("answer-as-assistant-dft").learn(model, model, (inputs, targets, learned), 1, SINGLE, None,
                                                       quiet)
    found = {name: p.grad.clone() for name, p in model.named_parameters()}

    # the formula, token by token
    with torch.no_grad():
        probabilities = model(inputs).softmax(-1)
    expected = 0.0
    for (row, column), target in zip([(0, 0), (0, 2), (1, 0), (1, 1)], [2, 4, 5, 6]):
        p = probabilities[row, column, target].item()
        expected += -p * math.log(p)
    assert share.item() == pytest.approx(expected / learned, rel=1e-5)

    # the gradient: the next-token loss's scaled by each token's p, p a number
    model.zero_grad()
    log_p = model(inputs).log_softmax(-1)
    picked = torch.stack([log_p[0, 0, 2], log_p[0, 2, 4], log_p[1, 0, 5], log_p[1, 1, 6]])
    (-(picked.detach().exp() * picked).sum() / learned).backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(found[name], p.grad)

    # with the gradient through p, it would differ
    model.zero_grad()
    log_p = model(inputs).log_softmax(-1)
    picked = torch.stack([log_p[0, 0, 2], log_p[0, 2, 4], log_p[1, 0, 5], log_p[1, 1, 6]])
    (-(picked.exp() * picked).sum() / learned).backward()
    assert not torch.allclose(found["head.weight"], model.head.weight.grad)


def test_neftune_s_noise_appears_only_in_training_passes_at_alpha_over_the_root_of_length_times_width():
    torch.manual_seed(0)
    vocab, width, length = 50, 64, 40
    model = Tiny(vocab, width)
    inputs = torch.randint(vocab, (3, length))
    targets = torch.randint(vocab, (3, length))
    method = method_of("answer-as-assistant-neftune", {"alpha": 5})
    bound = 5 / math.sqrt(length * width)

    model.train()
    method.learn(model, model, (inputs, targets, targets.numel()), 1, SINGLE, None, quiet)
    noise = model.seen.detach() - model.embedding(inputs).detach()
    assert noise.abs().max().item() <= bound * (1 + 1e-5)
    assert noise.abs().max().item() > 0.95 * bound  # 7,680 uniform draws reach near the bound
    assert noise.mean().abs().item() < 0.05 * bound  # centred on 0
    assert noise.std().item() == pytest.approx(bound / math.sqrt(3), rel=0.05)  # uniform in [-bound, bound]

    # outside its passes (evaluation, generation), in training mode or not, no noise
    model(inputs)
    torch.testing.assert_close(model.seen, model.embedding(inputs))
    model.eval()
    with torch.no_grad():
        model(inputs)
        torch.testing.assert_close(model.seen, model.embedding(inputs))

    # a pass with the model in evaluation mode sees none either
    method.learn(model, model, (inputs, targets, targets.numel()), 1, SINGLE, None, quiet)
    torch.testing.assert_close(model.seen.detach(), model.embedding(inputs).detach())
