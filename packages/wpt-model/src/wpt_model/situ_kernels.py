"""SiTU-GLU (layers.situ_glu) as one Triton kernel each way, for a GPU
outside torch.compile, which fuses the same math by itself (docs/model.md,
"Experts"). Written in PyTorch, its forward pass is seven elementwise
passes over the hidden width, each rounded to the inputs' dtype (BF16 on
a GPU), and autograd keeps several of their results for the backward
pass. Here the forward pass reads the two branches' pre-activations once
and writes the output once, computing in FP32 and rounding once; the
backward pass keeps only those two inputs and computes both gradients
from them in one pass, in FP32 too. So it agrees with the PyTorch form
within that form's BF16 rounding, and keeps less memory.

tanh(x) is computed as 2 sigmoid(2x) - 1, which Triton has on every GPU.
BLOCK, the values per program, is a first choice, not measured.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

BLOCK = 1024


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _forward_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    there = at < n
    g = tl.load(gate_ptr + at, mask=there, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + at, mask=there, other=0.0).to(tl.float32)
    out = 4 * _tanh(g / 4) * tl.sigmoid(g) * (25 * _tanh(u / 25))
    tl.store(out_ptr + at, out.to(out_ptr.dtype.element_ty), mask=there)


@triton.jit
def _backward_kernel(gate_ptr, up_ptr, grad_ptr, d_gate_ptr, d_up_ptr, n, BLOCK: tl.constexpr):
    """d/dg [4 tanh(g / 4) sigmoid(g)] = (1 - tanh(g / 4)^2) sigmoid(g) + 4 tanh(g / 4) sigmoid(g) (1 - sigmoid(g)),
    d/du [25 tanh(u / 25)] = 1 - tanh(u / 25)^2."""
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    there = at < n
    g = tl.load(gate_ptr + at, mask=there, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + at, mask=there, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + at, mask=there, other=0.0).to(tl.float32)
    t, s, v = _tanh(g / 4), tl.sigmoid(g), _tanh(u / 25)
    d_gate = grad * 25 * v * ((1 - t * t) * s + 4 * t * s * (1 - s))
    d_up = grad * 4 * t * s * (1 - v * v)
    tl.store(d_gate_ptr + at, d_gate.to(d_gate_ptr.dtype.element_ty), mask=there)
    tl.store(d_up_ptr + at, d_up.to(d_up_ptr.dtype.element_ty), mask=there)


class _SituGlu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate, up = gate.contiguous(), up.contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        _forward_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        d_gate, d_up = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        _backward_kernel[(triton.cdiv(n, BLOCK),)](gate, up, grad.contiguous(), d_gate, d_up, n, BLOCK=BLOCK)
        return d_gate, d_up


def situ_glu(gate, up):
    """layers.situ_glu of two tensors of one shape and dtype on a GPU."""
    return _SituGlu.apply(gate, up)
