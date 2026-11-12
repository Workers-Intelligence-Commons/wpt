"""The update rule (docs/model.md, "Training"): NorMuon for matrices, AdamW
for vectors, both with cautious weight decay, Adam for tables. Each test
writes the papers' formulas one tensor at a time, and the optimizer, which
batches them, must give the same weights."""

import math

import pytest
import torch

from wpt_model import NorMuonAdamW, orthogonalize
from wpt_utils.device import select_device


def agreeing(update, weight):
    """IMU-1's Eq. 7: the weight decays only where sign(u) = sign(w)."""
    return (torch.sign(update) == torch.sign(weight)).float()


def test_normuon_moves_each_matrix_of_a_batch_as_its_algorithm_moves_it_alone():
    # NorMuon's Algorithm 1 for each matrix of each view on its own, with the Nesterov momentum and the RMS of
    # 0.18 WPT keeps, and cautious decay: grouping matrices of the same rows and columns changes nothing but speed
    torch.manual_seed(0)
    views = [(1, 6, 4), (1, 6, 4), (3, 6, 4), (2, 4, 6), (1, 8, 8), (1, 3, 8), (2, 3, 8)]
    parameters = [torch.nn.Parameter(torch.randn(view[0] * view[1], view[2])) for view in views]
    lr, wd, mu = 0.1, 0.1, 0.95
    rule = NorMuonAdamW([(str(i), p, "matrix", view) for i, (p, view) in enumerate(zip(parameters, views))], lr, wd, mu)
    weights = [p.detach().clone().view(view) for p, view in zip(parameters, views)]
    momenta = [torch.zeros_like(w) for w in weights]
    neurons = [torch.zeros(view[:2]) for view in views]
    for _ in range(3):
        for i, (p, view) in enumerate(zip(parameters, views)):
            p.grad = torch.randn_like(p)
            grad = p.grad.view(view)
            momenta[i] = mu * momenta[i] + grad
            for j in range(view[0]):
                O = orthogonalize((grad[j] + mu * momenta[i][j])[None])[0]
                neurons[i][j] = 0.95 * neurons[i][j] + 0.05 * (O * O).mean(dim=1)
                O = O / (neurons[i][j].sqrt() + 1e-8)[:, None]
                u = 0.18 * math.sqrt(O.numel()) * O / O.norm()
                weights[i][j] = weights[i][j] - lr * (u + wd * agreeing(u, weights[i][j]) * weights[i][j])
        rule.step()
    for p, w in zip(parameters, weights):
        assert torch.allclose(p.detach(), w.view(p.shape), atol=1e-5)


def test_adamw_decays_only_where_its_update_agrees_and_tables_never_and_a_weight_without_gradient_waits():
    # AdamW with bias correction and cautious decay; Adam for a table at table_lr_factor times the rate, without
    # decay; a weight the loss did not reach in an update keeps its value and its count
    torch.manual_seed(0)
    vector, table, idle = (torch.nn.Parameter(torch.randn(16)) for _ in range(3))
    lr, wd, eps, (beta1, beta2) = 0.01, 0.5, 1e-8, (0.9, 0.95)
    rule = NorMuonAdamW([("vector", vector, "vector", None), ("idle", idle, "vector", None),
                         ("table", table, "table", None)], lr, wd, table_lr_factor=5.0)
    expected = {"vector": vector.detach().clone(), "table": table.detach().clone()}
    moments = {name: [torch.zeros(16), torch.zeros(16)] for name in expected}
    idle_before = idle.detach().clone()
    for step in (1, 2, 3):
        for name, p in (("vector", vector), ("table", table)):
            p.grad = torch.randn(16)
            mean, square = moments[name]
            mean.mul_(beta1).add_((1 - beta1) * p.grad)
            square.mul_(beta2).add_((1 - beta2) * p.grad ** 2)
            u = (mean / (1 - beta1 ** step)) / ((square / (1 - beta2 ** step)).sqrt() + eps)
            w = expected[name]
            if name == "vector":
                expected[name] = w - lr * (u + wd * agreeing(u, w) * w)
            else:
                expected[name] = w - 5.0 * lr * u
        idle.grad = torch.randn(16) if step == 3 else None
        rule.step()
    assert torch.allclose(vector.detach(), expected["vector"], atol=1e-6)
    assert torch.allclose(table.detach(), expected["table"], atol=1e-6)
    assert rule.state[vector]["step"] == 3 and rule.state[idle]["step"] == 1
    assert not torch.equal(idle.detach(), idle_before)  # it moved once, at its one gradient
    assert type(rule.state[vector]["step"]) is int  # a number, so a checkpoint reads on any device


@pytest.mark.skipif(select_device().type == "cpu", reason="needs a GPU")
def test_a_state_moves_between_the_cpu_and_a_gpu_and_the_update_stays_the_same():
    # docs/training.md, "Optimizers": the same update on every device within rounding, and a run resumes on another
    device = select_device()

    def made(on):
        torch.manual_seed(0)
        matrix = torch.nn.Parameter(torch.randn(12, 8, device=on))
        vector = torch.nn.Parameter(torch.randn(8, device=on))
        return [matrix, vector], NorMuonAdamW([("m", matrix, "matrix", (2, 6, 8)), ("v", vector, "vector", None)],
                                               0.01, 0.1)

    grads = [[torch.randn(12, 8), torch.randn(8)] for _ in range(2)]
    cpu, cpu_rule = made("cpu")
    gpu, gpu_rule = made(device)
    for p, g in zip(cpu, grads[0]):
        p.grad = g.clone()
    cpu_rule.step()
    gpu_rule.load_state_dict(cpu_rule.state_dict())  # the first update's state, continued on the GPU
    with torch.no_grad():
        for p, q in zip(gpu, cpu):
            p.copy_(q)
    for parameters, step in ((cpu, cpu_rule.step), (gpu, gpu_rule.step)):
        for p, g in zip(parameters, grads[1]):
            p.grad = g.to(p.device)
        step()
    for p, q in zip(cpu, gpu):
        assert torch.allclose(p.detach(), q.detach().cpu(), atol=1e-4)
