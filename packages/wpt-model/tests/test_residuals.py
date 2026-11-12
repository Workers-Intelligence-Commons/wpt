"""Attention Residuals (docs/model.md, "Attention Residuals"): the sources
of one pass, read through one tensor with a written-out backward pass,
give what the paper's formula gives over stacked sources, in the inputs
of the sublayers and in every gradient, in the full form and in groups."""

import pytest
import torch

from wpt_model.layers import EPS
from wpt_model.residuals import Residuals


def formula(residuals, embedding, sublayers):
    """The paper's form, written from it: each sublayer's input is the
    softmax over its sources of w . RMSNorm(v), times v, summed; the sources
    the embedding, each complete group's sum and the current group's sum
    so far. Returns every sublayer's input and the output head's."""
    complete, partial, inputs = [embedding], None, []

    def mixed(j):
        if j == 0:
            return embedding
        mix = residuals.mixes[j - 1]
        stacked = torch.stack(complete + ([partial] if partial is not None else []))
        keys = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + EPS) * mix.norm_weight
        return (torch.softmax(keys @ mix.query, 0)[..., None] * stacked).sum(0)

    for j, sublayer in enumerate(sublayers):
        inputs.append(mixed(j))
        out = sublayer(inputs[-1])
        partial = out if partial is None else partial + out
        if residuals.ends[j]:
            complete.append(partial)
            partial = None
    return inputs + [mixed(len(sublayers))]


def through_sources(residuals, embedding, sublayers):
    sources, inputs = residuals.start(embedding), []
    for j, sublayer in enumerate(sublayers):
        inputs.append(sources.input(j))
        sources.output(j, sublayer(inputs[-1]))
    return inputs + [sources.end()]


@pytest.mark.parametrize("sublayers, groups", [(6, 6), (6, 2), (7, 3), (4, 1)])
def test_the_sums_and_their_gradients_are_the_paper_s(sublayers, groups):
    # the full form (a group per sublayer), even groups, uneven groups, and
    # one group (a plain residual sum beside the embedding); trained queries
    # and norm weights, and sublayers that read their inputs, so every
    # gradient goes through the sums of later sublayers
    torch.manual_seed(0)
    residuals = Residuals(8, sublayers, groups)
    with torch.no_grad():
        for mix in residuals.mixes:
            mix.query.normal_()
            mix.norm_weight.uniform_(0.5, 1.5)
    embedding = torch.randn(2, 5, 8, requires_grad=True)
    weights = [torch.randn(8, 8, requires_grad=True) for _ in range(sublayers)]
    layers = [lambda h, w=w: torch.tanh(h @ w) for w in weights]
    parameters = [embedding, *weights, *residuals.parameters()]
    got, expected = through_sources(residuals, embedding, layers), formula(residuals, embedding, layers)
    directions = [torch.randn_like(h) for h in expected]
    got_grads = torch.autograd.grad(sum((h * d).sum() for h, d in zip(got, directions)), parameters)
    expected_grads = torch.autograd.grad(sum((h * d).sum() for h, d in zip(expected, directions)), parameters)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    for a, b in zip(got_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)  # FP32, summed in another order


def test_with_its_queries_at_zero_every_sum_is_the_plain_average_of_its_sources():
    # the paper's start: every query zero, so training begins from equal weights
    torch.manual_seed(0)
    residuals = Residuals(8, 4, 2)  # sublayers 0, 1 in the first group, 2, 3 in the second
    embedding = torch.randn(1, 3, 8)
    outputs = [torch.randn(1, 3, 8) for _ in range(4)]
    sources = residuals.start(embedding)
    inputs = []
    for j, out in enumerate(outputs):
        inputs.append(sources.input(j))
        sources.output(j, out)
    first = outputs[0] + outputs[1]
    expected = [embedding, (embedding + outputs[0]) / 2, (embedding + first) / 2,
                (embedding + first + outputs[2]) / 3, (embedding + first + outputs[2] + outputs[3]) / 3]
    for a, b in zip(inputs + [sources.end()], expected):
        torch.testing.assert_close(a, b)
