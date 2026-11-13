"""The output head and its summed cross-entropy in one pass, gradients
included (docs/model.md, "Training"): the forward pass computes the scores
of a chunk of tokens, then in one Triton kernel their losses and, in place
of the scores, their gradient, then the gradients of the hidden states and
of the head weight, which the backward pass only scales by the incoming
gradient. Nothing waits on the host, so a CUDA graph can record it
(model.py).

The kernel takes one token's row of scores per program, in blocks of
BLOCK: a first pass keeps a running maximum and sum of exponentials in
FP32, which give the row's log-sum-exp, and a second pass writes each
score's gradient, softmax minus the target's one, in the scores' dtype.
The second pass reads the row again from the GPU's cache, where the first
pass has just left it, so the scores cross the GPU's memory about once
each way: the cost of a kernel bound by its memory traffic. A target of
-100 counts for nothing, as in F.cross_entropy: a loss and a gradient of
0. BLOCK and WARPS are a first choice measured on an RTX 5080 laptop GPU
(sm_120) at 4,096 tokens and 32,768 scores each, where blocks of 8,192 to
32,768 and 8 to 32 warps tie; they are open to review on an H100 (sm_90).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

BLOCK = 8192  # scores per block of a row
WARPS = 8  # warps per row
LOG2E = tl.constexpr(1.4426950408889634)  # the kernel works in base 2, for exp2
LN2 = tl.constexpr(0.6931471805599453)
IGNORED = tl.constexpr(-100)  # F.cross_entropy's ignore_index


@triton.jit
def _loss_kernel(scores_ptr, targets_ptr, losses_ptr, V, BLOCK: tl.constexpr):
    """One row's loss into losses_ptr, and its scores (V,) replaced by
    their gradient."""
    row = tl.program_id(0).to(tl.int64)
    scores = scores_ptr + row * V
    target = tl.load(targets_ptr + row)
    kept = target != IGNORED
    # read before the first loop: its reductions order this load before the stores below
    x_target = tl.load(scores + target, mask=kept, other=0.0).to(tl.float32) * LOG2E
    columns = tl.arange(0, BLOCK)
    top = tl.full([BLOCK], float("-inf"), tl.float32)  # each column's running maximum, and the sum below it
    total = tl.zeros([BLOCK], tl.float32)
    for v0 in range(0, V, BLOCK):
        x = tl.load(scores + v0 + columns, mask=v0 + columns < V, other=float("-inf")).to(tl.float32) * LOG2E
        higher = tl.maximum(top, x)
        safe = tl.where(higher == float("-inf"), 0.0, higher)  # a column past V: no score yet, and none to add
        total = total * tl.exp2(top - safe) + tl.exp2(x - safe)
        top = higher
    best = tl.max(top, 0)
    lse = best + tl.log2(tl.sum(total * tl.exp2(top - best), 0))
    for v0 in range(0, V, BLOCK):
        v = v0 + columns
        x = tl.load(scores + v, mask=v < V, other=float("-inf")).to(tl.float32) * LOG2E
        gradient = tl.exp2(x - lse) - (v == target).to(tl.float32)
        tl.store(scores + v, tl.where(kept, gradient, 0.0).to(scores_ptr.dtype.element_ty), mask=v < V)
    tl.store(losses_ptr + row, tl.where(kept, (lse - x_target) * LN2, 0.0))


class _HeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, targets, weight):
        hidden, targets = hidden.contiguous(), targets.contiguous()
        scores = F.linear(hidden, weight)
        rows, V = scores.shape
        losses = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        # the losses, and in place of the scores their gradient
        _loss_kernel[(rows,)](scores, targets, losses, V, BLOCK=min(BLOCK, triton.next_power_of_2(V)),
                              num_warps=WARPS)
        ctx.save_for_backward(scores @ weight, scores.t() @ hidden)
        return losses.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        d_hidden, d_weight = ctx.saved_tensors
        return d_hidden.mul_(grad), None, d_weight.mul_(grad)


def head_loss(hidden, targets, weight):
    """The summed cross-entropy of the scores hidden (m, d_model) @ weight^T
    against targets (m,), in hidden's dtype, which weight must share."""
    return _HeadLoss.apply(hidden, targets, weight)
