"""The sums of Attention Residuals (residuals.py) as Triton kernels, for a
compiled training pass on a GPU: the forward and backward passes of one
sum over n complete sources, with no partial sum, and each source's
gradient over the sums that read it. residuals.py imports this module
only there.

The sums move data: each reads its n sources (n, B, T, D) in BF16, and
there are as many sums as sublayers, so the sources are read about
sublayers^2 / 2 times per pass. Each kernel takes a tile of tokens and
goes through its sources, or its sums, in turn, from memory once:

    forward   each source's scores, then an online softmax (a running
              maximum and denominator) that adds the source into the sum
              at once; it writes the sum, and per token the log of the
              softmax's denominator (Kept)
    backward  one loop over the sources computes each one's score and
              weight again and adds up mean = sum_i alpha_i (g . v_i),
              which every source's gradient needs, and the query's
              gradient over the tile's tokens, one FP32 row per tile,
              summed afterwards; it keeps the sum's output gradient g and
              mean (Kept)
    gradient  once every sum that reads a source is done, one loop over
              those sums computes the source's score and weight in each
              again, reads each one's g and adds up the source's gradient
              in FP32, written once

So a source's gradient is one FP32 sum over the sums that read it: the
backward pass reads each source once per sum that reads it, and each
sum's output gradient once per source it reaches, instead of also
reading and writing every source's gradient, in BF16, once per sum. The
math is _mix's and _unmix's in FP32, rounded
once when written; torch.compile's form of the same math rounds its
scores, its weights and some factors of the backward pass to BF16 on the
way, so the two agree within BF16 rounding.

A tile holds TILE values of a source (tokens x D padded to a power of 2),
the same on every GPU; the threads per tile are Triton's autotuner's
choice per D, from configs that fit an H100 (sm_90) and an RTX 5080
laptop GPU (sm_120). TILE was chosen among 1024 to 8192 on the RTX
5080, where it changed the time by less than the runs' spread; it is
open to review on an H100.
"""

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
TILE = 2048  # values of one source per tile: tokens x D padded to a power of 2
_CONFIGS = [triton.Config({}, num_warps=warps) for warps in (2, 4, 8)]


def _shape(values):
    """(n, tokens M, D, DP the padded width, TOKENS per tile)."""
    n, D = values.shape[0], values.shape[-1]
    DP = triton.next_power_of_2(D)
    return n, values[0].numel() // D, D, DP, max(1, TILE // DP)


class Kept:
    """What one pass's sums keep for their sources' gradients, a slot per
    sum s (from 0, in the order of the sums), which reads the first
    counts[s] sources: its u us[s] (D,) and the log of its softmax's
    denominator lses[s] (B, T), which its forward pass writes, and, once
    its backward pass has run (flags[s] = 1), its output's gradient
    grads[s] (B, T, D) and its mean (B, T) means[s]. A sum's scores and
    weights are computed again from these and the sources, so nothing of
    size sources x tokens is kept. The gradients and the means are made at
    the first backward pass. The kernels write the flags themselves, so a
    recorded CUDA graph's replay writes them again."""

    def __init__(self, counts, M, D, dtype, device):
        S = len(counts)
        self.counts, self.M, self.D = counts, M, D
        self.us = torch.empty(S, D, dtype=dtype, device=device)
        self.lses = torch.empty(S, M, dtype=torch.float32, device=device)
        self.ran = [False] * S  # the sums whose backward pass has run, as the host knows them
        self.grads = self.means = self.flags = None

    def made(self):
        if self.grads is None:
            S, device = len(self.counts), self.us.device
            self.grads = torch.empty(S, self.M, self.D, dtype=self.us.dtype, device=device)
            self.means = torch.empty(S, self.M, dtype=torch.float32, device=device)
            self.flags = torch.zeros(S, dtype=torch.int32, device=device)


def mix(u, values, scales, kept, slot):
    """_mix's h without a partial sum, keeping u and the softmax's
    denominator in kept's slot: u (D,) and values (n, B, T, D) in the
    compute dtype, scales (n, B, T) FP32, every one contiguous."""
    n, M, D, DP, TOKENS = _shape(values)
    if n != kept.counts[slot]:
        raise RuntimeError(f"attention residuals: sum {slot} reads {n} sources, its slot holds {kept.counts[slot]}")
    h = torch.empty(values.shape[1:], dtype=values.dtype, device=values.device)
    _mix_kernel[(triton.cdiv(M, TOKENS),)](values, scales, u, h, kept.us[slot], kept.lses[slot], n, M, D=D, DP=DP,
                                           TOKENS=TOKENS)
    return h


def unmix(grad, values, scales, kept, slot):
    """_unmix's d_u without a partial sum, keeping in kept's slot what its
    sources' gradients need from it (gradient): grad (B, T, D) in the
    compute dtype, the rest as mix takes them."""
    n, M, D, DP, TOKENS = _shape(values)
    kept.made()
    tiles = triton.cdiv(M, TOKENS)
    parts = torch.empty(tiles, D, dtype=torch.float32, device=values.device)
    _unmix_kernel[(tiles,)](grad, values, scales, kept.us[slot], kept.lses[slot], kept.grads[slot], kept.means[slot],
                            kept.flags[slot:], parts, n, M, D=D, DP=DP, TOKENS=TOKENS)
    kept.ran[slot] = True
    return parts.sum(0)


def gradient(index, values, scales, kept):
    """Source index's gradient (B, T, D) in the compute dtype over the sums
    whose backward pass has run, or None when none of them reads it:
    values (n, B, T, D) and scales (n, B, T) as mix takes them."""
    first = next((s for s, n in enumerate(kept.counts) if n > index), len(kept.counts))
    if not any(kept.ran[first:]):
        return None
    _, M, D, DP, TOKENS = _shape(values)
    out = torch.empty(values.shape[1:], dtype=values.dtype, device=values.device)
    _gradient_kernel[(triton.cdiv(M, TOKENS),)](values[index], scales[index], kept.us, kept.lses, kept.grads,
                                                 kept.means, kept.flags, out, first, len(kept.counts), M, D=D, DP=DP,
                                                 TOKENS=TOKENS)
    return out


@triton.jit
def _tile(pid, M, D: tl.constexpr, DP: tl.constexpr, TOKENS: tl.constexpr):
    """A tile's tokens (int64), its width's columns, whether each token is
    real, and the (TOKENS, DP) mask of real values."""
    m = pid.to(tl.int64) * TOKENS + tl.arange(0, TOKENS)
    d = tl.arange(0, DP)
    real = m < M
    return m, d, real, real[:, None] & (d[None, :] < D)


@triton.jit
def _score(v, u, r):
    """The scores (u . v) r of a tile's (TOKENS, DP) values, FP32."""
    return tl.sum(v * u[None, :], 1) * r


@triton.autotune(configs=_CONFIGS, key=["D"])
@triton.jit
def _mix_kernel(values_ptr, scales_ptr, u_ptr, h_ptr, kept_u_ptr, lse_ptr, n, M, D: tl.constexpr, DP: tl.constexpr,
                TOKENS: tl.constexpr):
    """One tile of tokens' sum over the n sources, one read of each, and
    the log of its softmax's denominator; the first tile also keeps u."""
    pid = tl.program_id(0)
    m, d, real, there = _tile(pid, M, D, DP, TOKENS)
    u_raw = tl.load(u_ptr + d, mask=d < D, other=0.0)
    if pid == 0:
        tl.store(kept_u_ptr + d, u_raw, mask=d < D)
    u = u_raw.to(tl.float32)
    best = tl.full([TOKENS], float("-inf"), tl.float32)
    total = tl.zeros([TOKENS], tl.float32)
    acc = tl.zeros([TOKENS, DP], tl.float32)
    for i in range(n):
        at = i * M + m
        v = tl.load(values_ptr + at[:, None] * D + d[None, :], mask=there, other=0.0).to(tl.float32)
        s = _score(v, u, tl.load(scales_ptr + at, mask=real, other=0.0))
        new = tl.maximum(best, s)
        alpha = tl.exp2((best - new) * LOG2E)
        p = tl.exp2((s - new) * LOG2E)
        acc = acc * alpha[:, None] + p[:, None] * v
        total = total * alpha + p
        best = new
    tl.store(h_ptr + m[:, None] * D + d[None, :], (acc / total[:, None]).to(h_ptr.dtype.element_ty), mask=there)
    tl.store(lse_ptr + m, best + tl.log2(total) / LOG2E, mask=real)


@triton.autotune(configs=_CONFIGS, key=["D"])
@triton.jit
def _unmix_kernel(grad_ptr, values_ptr, scales_ptr, u_ptr, lse_ptr, kept_grad_ptr, mean_ptr, flag_ptr, parts_ptr, n,
                  M, D: tl.constexpr, DP: tl.constexpr, TOKENS: tl.constexpr):
    """One tile of tokens' share of a sum's backward pass, one read of each
    source, its weights alpha_i = exp(score_i - lse) computed again: mean
    = sum_i alpha_i (g . v_i), and the tile's row of the query's gradient
    sum_i alpha_i (g . v_i - mean) r_i v_i, from the sums over i of
    alpha_i r_i (g . v_i) v_i and of alpha_i r_i v_i; keeps g, mean and
    the flag for _gradient_kernel."""
    pid = tl.program_id(0)
    m, d, real, there = _tile(pid, M, D, DP, TOKENS)
    g_raw = tl.load(grad_ptr + m[:, None] * D + d[None, :], mask=there, other=0.0)
    tl.store(kept_grad_ptr + m[:, None] * D + d[None, :], g_raw, mask=there)
    if pid == 0:
        tl.store(flag_ptr, 1)
    g = g_raw.to(tl.float32)
    u = tl.load(u_ptr + d, mask=d < D, other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + m, mask=real, other=0.0)
    mean = tl.zeros([TOKENS], tl.float32)
    along = tl.zeros([TOKENS, DP], tl.float32)  # sum_i alpha_i r_i (g . v_i) v_i
    plain = tl.zeros([TOKENS, DP], tl.float32)  # sum_i alpha_i r_i v_i
    for i in range(n):
        at = i * M + m
        v = tl.load(values_ptr + at[:, None] * D + d[None, :], mask=there, other=0.0).to(tl.float32)
        r = tl.load(scales_ptr + at, mask=real, other=0.0)
        alpha = tl.exp2((_score(v, u, r) - lse) * LOG2E)
        gv = tl.sum(g * v, 1)
        mean += alpha * gv
        along += (alpha * r * gv)[:, None] * v
        plain += (alpha * r)[:, None] * v
    tl.store(mean_ptr + m, mean, mask=real)
    tl.store(parts_ptr + pid.to(tl.int64) * D + d, tl.sum(along - mean[:, None] * plain, 0), mask=d < D)


@triton.autotune(configs=_CONFIGS, key=["D"])
@triton.jit
def _gradient_kernel(values_ptr, scales_ptr, us_ptr, lses_ptr, grads_ptr, means_ptr, flags_ptr, out_ptr, first, sums,
                     M, D: tl.constexpr, DP: tl.constexpr, TOKENS: tl.constexpr):
    """One tile of tokens of a source's gradient, summed in FP32 over the
    sums first .. sums - 1 whose flag is set, and written once: alpha g +
    along_u u + along_v v for each sum, with its score and weight alpha
    computed again, along_u = alpha (g . v - mean) r and along_v =
    -along_u score r / D (d score / d v = r u - score r^2 v / D)."""
    m, d, real, there = _tile(tl.program_id(0), M, D, DP, TOKENS)
    offsets = m[:, None] * D + d[None, :]
    v = tl.load(values_ptr + offsets, mask=there, other=0.0).to(tl.float32)
    r = tl.load(scales_ptr + m, mask=real, other=0.0)
    acc = tl.zeros([TOKENS, DP], tl.float32)
    along_v = tl.zeros([TOKENS], tl.float32)
    for s in range(first, sums):
        if tl.load(flags_ptr + s) != 0:
            row = s.to(tl.int64) * M + m
            u = tl.load(us_ptr + s * D + d, mask=d < D, other=0.0).to(tl.float32)
            score = _score(v, u, r)
            alpha = tl.exp2((score - tl.load(lses_ptr + row, mask=real, other=0.0)) * LOG2E)
            g = tl.load(grads_ptr + s.to(tl.int64) * M * D + offsets, mask=there, other=0.0).to(tl.float32)
            along_u = alpha * (tl.sum(g * v, 1) - tl.load(means_ptr + row, mask=real, other=0.0)) * r
            along_v -= along_u * score * r / D
            acc += alpha[:, None] * g + along_u[:, None] * u[None, :]
    tl.store(out_ptr + offsets, (acc + along_v[:, None] * v).to(out_ptr.dtype.element_ty), mask=there)
