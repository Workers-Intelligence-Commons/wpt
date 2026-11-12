"""The update rule of WPT's weights (docs/model.md, "Training"): NorMuon
for matrix weights, AdamW for the rest, Adam for Engram's tables, with
cautious weight decay. The model and its drafter each name their
parameters' roles (optimizer_roles, by the one rule of roles()); training
wraps NorMuonAdamW with a schedule, its stopping and the average of its
last states (wpt_training's `by-part` optimizer, wpt_inference.drafting).

NorMuon (arXiv 2510.05491, Algorithm 1) as IMU-1 (arXiv 2602.02522) uses
it: the Nesterov momentum of each matrix is orthogonalized by DeepSeek-V4's
hybrid Newton-Schulz steps (8 that bring the singular values near 1, then 2
that settle them at 1) into O; each row of O (an output neuron) is divided
by the root of a running mean of its squares, v = b v + (1 - b) mean(O^2)
over the row's columns, b = NEURON_BETA; the result is scaled to an RMS of
UPDATE_RMS, DeepSeek-V4's 0.18, so AdamW's learning rates carry over. Query
and gate projections are orthogonalized and normalized head by head and
routed experts one by one. No QK-Clip: every attention layer normalizes its
queries and keys, which keeps their scores bounded.

Weight decay is cautious (arXiv 2510.12402; IMU-1's Eq. 7): with the update
u (W -= lr u), an entry decays only where u and W have the same sign,
W -= lr (u + wd [u W > 0] W). It applies to every role with a weight decay:
the matrices' NorMuon updates and the vectors' AdamW updates.

Open to review: NorMuon's paper gives no Nesterov momentum, no epsilon and an
RMS of 0.2; this keeps the momentum and the RMS of DeepSeek-V4's Muon, which
the learning rates were fitted with, and NEURON_EPS is WPT's own. IMU-1
orthogonalizes with 7 Polar Express steps; this keeps DeepSeek-V4's 10.

The whole update runs as PyTorch's foreach operations (one call over every
tensor of a group) and one batch per shape of matrix (the same rows and
columns) for the orthogonalization, on every device: no value comes back
to the host. A batch is a temporary FP32 copy of its matrices' updates and
weights. The state is the same on every device, so a run resumes on another.
"""

import math

import torch
import torch.nn as nn

from .attention import ExactAttention, SparseAttention, WindowAttention
from .engram import Engram

NEWTON_SCHULZ = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2  # DeepSeek-V4's hybrid steps
UPDATE_RMS = 0.18  # DeepSeek-V4's gamma: the RMS of a matrix's update
NEURON_BETA = 0.95  # NorMuon's beta2: the running mean of each row's squares
NEURON_EPS = 1e-8  # added to that mean's root; open to review (NorMuon gives none)


def orthogonalize(G):
    """(matrices, A, B) -> each matrix's nearest semi-orthogonal matrix, approximately."""
    X = G.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in NEWTON_SCHULZ:
        A = X @ X.transpose(-1, -2)
        X = a * X + (b * A + c * A @ A) @ X
    return X.transpose(-1, -2) if tall else X


def roles(module: nn.Module, vectors=()):
    """(name, parameter, role, view) for every parameter of module. Role
    "matrix" (NorMuon): view is the (matrices, rows, columns) shape it is
    orthogonalized as, one matrix per head for the attention's query and
    gate projections, one per routed expert, one elsewhere (the one
    key-value head's projections among them). Role "vector" (AdamW): the
    parameters given in vectors, those of fewer than 2 dimensions (norms,
    biases, sinks, gating factors, the queries of Attention Residuals) and
    the convolutions. Role "table": Engram's tables. view is None for the
    last two."""
    heads, chosen, tables = {}, {id(p) for p in vectors}, set()
    for part in module.modules():
        if isinstance(part, (WindowAttention, ExactAttention, SparseAttention)):
            heads[id(part.q.weight)] = part.heads
            if hasattr(part, "gate"):
                heads[id(part.gate.weight)] = part.heads
        elif isinstance(part, nn.Conv1d):
            chosen.add(id(part.weight))
        elif isinstance(part, Engram):
            tables.update(id(table.weight) for table in part.tables)
    for name, p in module.named_parameters():
        if id(p) in tables:
            yield name, p, "table", None
        elif id(p) in chosen or p.ndim < 2:
            yield name, p, "vector", None
        elif p.ndim == 3:
            yield name, p, "matrix", tuple(p.shape)
        else:
            n = heads.get(id(p), 1)
            yield name, p, "matrix", (n, p.shape[0] // n, p.shape[1])


class NorMuonAdamW(torch.optim.Optimizer):
    """NorMuon over the "matrix" parameters of roles, each in its view,
    AdamW over the "vector" ones, both with cautious weight decay, and Adam
    at table_lr_factor times lr, with no weight decay, over the "table"
    ones; a parameter that does not require a gradient (a locked part, one
    the loss never reaches) is left out, and so is a role with no
    parameter. Each group's "lr" is lr times its "lr_factor": a schedule
    sets it before each step."""

    def __init__(self, roles, lr, weight_decay, momentum=0.95, betas=(0.9, 0.95), eps=1e-8, table_lr_factor=1.0):
        found = {"matrix": ([], []), "vector": ([], []), "table": ([], [])}
        for _, parameter, role, view in roles:
            if parameter.requires_grad:
                found[role][0].append(parameter)
                found[role][1].append(view)
        groups = [
            {"params": found["matrix"][0], "views": found["matrix"][1], "muon": True, "lr_factor": 1.0,
             "weight_decay": weight_decay},
            {"params": found["vector"][0], "muon": False, "lr_factor": 1.0, "weight_decay": weight_decay},
            {"params": found["table"][0], "muon": False, "lr_factor": table_lr_factor, "weight_decay": 0.0},
        ]
        self.counts = {role: len(params) for role, (params, _) in found.items()}
        super().__init__([{**g, "lr": lr * g["lr_factor"]} for g in groups if g["params"]],
                         {"lr": lr, "momentum": momentum, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["muon"]:
                self._normuon(group)
            else:
                self._adamw(group)

    def _normuon(self, group):
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        trained = [(p, view) for p, view in zip(group["params"], group["views"]) if p.grad is not None]
        if not trained:
            return
        for p, view in trained:
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                state["neurons"] = torch.zeros(view[:-1], dtype=torch.float32, device=p.device)  # one per row
        params = [p for p, _ in trained]
        momenta = [self.state[p]["momentum"] for p in params]
        grads = [p.grad.float() for p in params]
        torch._foreach_mul_(momenta, mu)
        torch._foreach_add_(momenta, grads)
        shapes = {}  # matrices of the same rows and columns go through in one batch, one shape at a time
        for number, (_, view) in enumerate(trained):
            shapes.setdefault(view[-2:], []).append(number)
        changes = [None] * len(params)
        for (rows, columns), numbers in shapes.items():
            views = [trained[i][1] for i in numbers]
            counts = [view[0] for view in views]
            nesterov = torch._foreach_add([grads[i] for i in numbers], [momenta[i] for i in numbers], alpha=mu)
            O = orthogonalize(torch.cat([m.view(view) for m, view in zip(nesterov, views)]))
            del nesterov
            states = [self.state[params[i]]["neurons"] for i in numbers]
            neurons = torch.cat(states).lerp_(O.square().mean(-1), 1 - NEURON_BETA)
            torch._foreach_copy_(states, list(neurons.split(counts)))
            O.div_(neurons.sqrt_().add_(NEURON_EPS)[..., None])
            # each matrix's RMS to UPDATE_RMS; a matrix whose gradient is 0 stays 0
            O.mul_(UPDATE_RMS * math.sqrt(rows * columns) / O.norm(dim=(-2, -1), keepdim=True).clamp_min_(1e-30))
            if wd:
                cautious([O], [torch.cat([params[i].detach().float().view(view) for i, view in zip(numbers, views)])],
                         wd)
            for i, change in zip(numbers, O.split(counts)):
                changes[i] = change.view(params[i].shape)
        _add(params, changes, -lr)

    def _adamw(self, group):
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        for p in params:
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0  # a number, never a tensor: no value comes back from the device
                state["mean"] = torch.zeros_like(p, dtype=torch.float32)
                state["square"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] += 1
        steps = [self.state[p]["step"] for p in params]
        grads = [p.grad.float() for p in params]
        means, squares = [self.state[p]["mean"] for p in params], [self.state[p]["square"] for p in params]
        torch._foreach_lerp_(means, grads, 1 - beta1)
        torch._foreach_mul_(squares, beta2)
        torch._foreach_addcmul_(squares, grads, grads, value=1 - beta2)
        roots = torch._foreach_div(squares, [1 - beta2 ** s for s in steps])
        torch._foreach_sqrt_(roots)
        torch._foreach_add_(roots, eps)
        changes = torch._foreach_div(means, roots)
        torch._foreach_div_(changes, [1 - beta1 ** s for s in steps])
        if wd:
            cautious(changes, [p.detach().float() for p in params], wd)
        _add(params, changes, -lr)


def cautious(updates, weights, wd):
    """updates += wd * weights where an update and its weight have the same
    sign (u W > 0), in place: the cautious weight decay of an update that
    is subtracted from the weights."""
    signs = torch._foreach_sign(updates)
    decay = torch._foreach_mul(signs, weights)
    torch._foreach_clamp_min_(decay, 0.0)  # |W| where the signs agree, else 0
    torch._foreach_mul_(decay, signs)
    torch._foreach_add_(updates, decay, alpha=wd)


def _add(params, changes, alpha):
    """params += alpha * changes (FP32), each in its parameter's dtype."""
    if any(p.dtype != torch.float32 for p in params):
        changes = [c.to(p.dtype) for p, c in zip(params, changes)]
    torch._foreach_add_(params, changes, alpha=alpha)
