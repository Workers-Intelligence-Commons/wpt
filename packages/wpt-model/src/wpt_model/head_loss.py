"""The output head and its summed cross-entropy in one pass, gradients
included, through flash-linear-attention's kernels (docs/model.md,
"Training"): the forward pass computes the scores of a chunk of tokens,
their log-sum-exp, and in place of the scores their gradient, then the
gradients of the hidden states and of the head weight, which the backward
pass only scales by the incoming gradient.

It is flash-linear-attention's fused_linear_cross_entropy_loss, with one
chunk, the sum as reduction and the weight's gradient in the weight's
dtype, called through WPT's own wrapper: flash-linear-attention's reads
the count of targets back to the host in its forward pass and compares
the incoming gradient with 1 on the host in its backward pass, which makes
the CPU wait for the GPU and cannot be recorded in a CUDA graph
(model.py). This wrapper waits on nothing: the count serves only a mean,
and the backward pass always multiplies.
"""

import torch
import torch.nn.functional as F
from fla.modules.fused_linear_cross_entropy import MAX_FUSED_SIZE, STATIC_WARPS, cross_entropy_kernel, logsumexp_fwd
from torch.autograd.function import once_differentiable


class _HeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, targets, weight):
        hidden, targets = hidden.contiguous(), targets.contiguous()
        scores = F.linear(hidden, weight)
        rows, V = scores.shape
        lse = logsumexp_fwd(scores, dtype=torch.float)
        losses = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        # the losses, and in place of the scores their gradient
        cross_entropy_kernel[(rows,)](logits=scores, lse=lse, target=targets, loss=losses, total=1,
                                      ignore_index=-100, label_smoothing=0.0, logit_scale=1.0,
                                      logit_softcapping=None, reduction="sum", V=V,
                                      BV=min(MAX_FUSED_SIZE, 1 << (V - 1).bit_length()), num_warps=STATIC_WARPS)
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
