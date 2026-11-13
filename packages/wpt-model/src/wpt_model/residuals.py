"""Attention Residuals (docs/model.md, "Attention Residuals"), after Kimi's
Attention Residuals (arXiv 2603.15031): the links between sublayers.

Every sublayer (each layer's attention and its experts, which the paper
calls layers) reads a weighted sum of sources instead of one running
residual sum:

    h_l = sum_i alpha_(i->l) v_i,   alpha_(i->l) = softmax_i(w_l . RMSNorm_l(v_i))

w_l is the sublayer's own learned vector, its query, which starts at zero,
so that training begins from a plain average of the sources. The first
sublayer has one source, the embedding, which it reads as it is: a sum
over one source has nothing to learn. The
sublayers are cut into `groups` consecutive groups (the paper's Block
AttnRes); the sources of a sublayer are the embedding, the sum of the
outputs of each earlier group, and the sum of the outputs of its own group
so far, when that group already holds one. With one group per sublayer,
every output is a source of its own (the paper's full form). The output
head reads one more such sum, with its own query, over every group.

Sublayer j belongs to group j * groups // sublayers, so the groups differ
in size by at most one sublayer. The paper leaves this split open when the
groups do not divide the sublayers evenly; this rule is open to review.

How it runs: the sources of one pass sit side by side in one tensor of the
compute dtype (BF16 on a GPU), each written there once when it is complete
(Sources.add), and each source's RMS is computed then, once. A sublayer's
sum reads them through two batched products over that tensor, one for the
scores and one for the sum, never through a copy stacked per sublayer:
since RMSNorm_l(v_i) = g_l * v_i * r_i, with g_l the norm's weight and r_i
the inverse RMS of v_i, the score is ((w_l * g_l) . v_i) r_i. Scores and
softmax are FP32. The backward pass is written out (_Mixing) so that it
keeps nothing per sublayer but the FP32 scores and weights of the sources:
the sources themselves stay only in that one tensor. The sources'
gradients sit side by side the same way, in one tensor of the compute
dtype (_Gradients): each sum's backward pass adds its share to every
source's place there at once, and each source takes its place's sum when
every sum that reads it is done, so autograd adds up no gradient source
by source. A pass's graph is gone through backward once (each sum
refuses a second time). In a compiled training pass
(WPT.compile_training) on a GPU, both passes of a sum with no partial
sum, every sum of the full form, run as Triton kernels
(residual_kernels.py) that read each source from memory once; such a
sum keeps no scores or weights, only per token the log of its softmax's
denominator, from which the kernels compute them again; its backward
pass adds nothing into the gradients' tensor, but keeps its output's
gradient, and each source's gradient is then one FP32 sum
over the sums that read it, made when the source takes it. Elsewhere in
a compiled pass, the math of both passes runs through torch.compile,
which fuses it into a few kernels.
"""

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from .layers import EPS, compute_dtype


def group_ends(sublayers, groups):
    """For each of `sublayers` sublayers (from 0), whether it is the last
    of its group: sublayer j is in group j * groups // sublayers."""
    if not 1 <= groups <= sublayers:
        raise ValueError(f"attention residuals: groups must be between 1 and the {sublayers} sublayers, not {groups}")
    return tuple((j + 1) * groups // sublayers != j * groups // sublayers for j in range(sublayers))


def _inverse_rms(value):
    """1 / RMS(value) over its last dimension, FP32, without an FP32 copy of value."""
    return torch.rsqrt(value.square().mean(-1, dtype=torch.float32) + EPS)


class Mix(nn.Module):
    """One sublayer's sum: its query w_l and its RMSNorm's weight g_l."""

    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.norm_weight = nn.Parameter(torch.ones(width))


class Residuals(nn.Module):
    """The sums of `sublayers` sublayers cut into `groups` groups: mixes[j - 1]
    is sublayer j's (from 1; the first reads the embedding), and
    mixes[sublayers - 1] the output head's."""

    def __init__(self, width, sublayers, groups):
        super().__init__()
        self.ends = group_ends(sublayers, groups)
        self.groups = groups
        self.mixes = nn.ModuleList(Mix(width) for _ in range(sublayers))

    def start(self, embedding, compiled=False):
        """The sources of one pass, the embedding (B, T, D) the first.
        compiled: the sums' math runs through torch.compile."""
        return Sources(self, embedding, compiled)


class Sources:
    """The sources of one pass through Residuals, in order: the input of
    sublayer j (input), then its output (output), for each sublayer in
    turn, then the output head's input (end)."""

    def __init__(self, residuals, embedding, compiled=False):
        self.compiled = compiled
        self.dtype = compute_dtype(embedding)
        self.residuals = residuals
        self.values = embedding.new_empty((residuals.groups + 1, *embedding.shape), dtype=self.dtype)
        self.scales = embedding.new_empty((residuals.groups + 1, *embedding.shape[:-1]), dtype=torch.float32)
        self.added = []    # a stand-in for each complete source, which carries its gradient (_Add)
        counts = [1 + sum(residuals.ends[:j]) for j in range(1, len(residuals.ends) + 1)]  # each sum's sources
        self.gradients = _Gradients(self.values, self.scales, counts)
        self.partial = None  # the sum of the current group's outputs so far
        self.embedding = embedding.to(self.dtype)
        self.add(self.embedding)

    def add(self, value):
        """value (B, T, D) becomes the next complete source."""
        self.added.append(_Add.apply(value, self, len(self.added)))

    def input(self, j):
        """The input of sublayer j (B, T, D), in the compute dtype."""
        if j == 0:
            return self.embedding
        mix = self.residuals.mixes[j - 1]
        return _Mixing.apply(mix.query * mix.norm_weight, self.values, self.scales, self.partial, self.compiled,
                             self.gradients, j - 1, *self.added)

    def output(self, j, out):
        """Sublayer j's output joins its group's sum, which becomes a source
        when j ends the group."""
        out = out.to(self.dtype)
        self.partial = out if self.partial is None else self.partial + out
        if self.residuals.ends[j]:
            self.add(self.partial)
            self.partial = None

    def end(self):
        """The output head's input, over every group."""
        return self.input(len(self.residuals.ends))


class _Gradients:
    """The sources' gradients of one pass, a place per source as in
    Sources.values: the backward passes of the sums through torch.compile
    or uncompiled add to them (_Mixing), made at the first one, while the
    sums through the Triton kernels keep what their sources' gradients
    need in kept (residual_kernels.Kept), made at the first one, slot s
    for the sum of sublayer s + 1 (counts[s] sources). Each source takes
    its own (take)."""

    def __init__(self, values, scales, counts):
        self.values, self.scales, self.counts = values, scales, counts
        self.sums = self.kept = None

    def places(self, n):
        """The first n sources' gradients so far, to add to in place."""
        if self.sums is None:
            self.sums = torch.zeros_like(self.values)
        return self.sums[:n]

    def kernels_kept(self):
        """kept, made at its first use."""
        if self.kept is None:
            M, D = self.values[0].numel() // self.values.shape[-1], self.values.shape[-1]
            self.kept = _residual_kernels().Kept(self.counts, M, D, self.values.dtype, self.values.device)
        return self.kept

    def take(self, index):
        """Source index's gradient in the compute dtype, or None when no
        sum has handed it one: its place, plus the kernel sums' shares,
        added up in FP32 at once (residual_kernels.gradient)."""
        gradient = None if self.sums is None else self.sums[index]
        if self.kept is not None:
            share = _residual_kernels().gradient(index, self.values, self.scales, self.kept)
            if share is not None:
                gradient = share if gradient is None else gradient + share
        return gradient


class _Add(torch.autograd.Function):
    """Writes a source into its place of sources.values (and its inverse
    RMS into sources.scales), and returns a stand-in of its shape that holds
    no memory (one zero, expanded): autograd runs its backward pass once
    every sum that reads the stand-in has run its own, and it hands the
    source its gradient from sources.gradients (_Gradients.take), while
    only the place keeps its values."""

    @staticmethod
    def forward(ctx, value, sources, index):
        place = sources.values[index]
        place.copy_(value)
        sources.scales[index] = _inverse_rms(place)
        ctx.gradients, ctx.index, ctx.dtype = sources.gradients, index, value.dtype
        ctx.set_materialize_grads(False)  # the sums hand the stand-in nothing (_Mixing)
        return place.new_zeros(()).expand(place.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, _nothing):
        gradient = ctx.gradients.take(ctx.index)
        return (None if gradient is None else gradient.to(ctx.dtype)), None, None


def _mix(u, values, scales, partial):
    """_Mixing's forward pass: (h, scores, weights, partial's inverse RMS or None)."""
    n = values.shape[0]
    dtype = values.dtype
    partial_scale = None
    w = u.to(dtype)
    scores = torch.einsum("nbtd,d->nbt", values, w).float() * scales
    if partial is not None:
        partial_scale = _inverse_rms(partial)
        scores = torch.cat([scores, ((partial @ w).float() * partial_scale)[None]])
    weights = scores.softmax(0)
    h = torch.einsum("nbt,nbtd->btd", weights[:n].to(dtype), values)
    if partial is not None:
        h = torch.addcmul(h, weights[n, ..., None].to(dtype), partial)
    return h, scores, weights, partial_scale


def _unmix(grad, u, values, scales, scores, weights, partial, partial_scale, d_values_sum):
    """_Mixing's backward pass: (d_u, d_partial or None), the sources'
    gradients d_values (n, B, T, D) added into d_values_sum in place."""
    n, dtype, width = len(values), values.dtype, values.shape[-1]
    if partial is not None:
        scales = torch.cat([scales, partial_scale[None]])
    g, w = grad.to(dtype), u.to(dtype)
    d_weights = torch.einsum("btd,nbtd->nbt", g, values).float()
    if partial is not None:
        d_weights = torch.cat([d_weights, (g * partial).sum(-1, dtype=torch.float32)[None]])
    d_scores = weights * (d_weights - (weights * d_weights).sum(0, keepdim=True))
    along_u = d_scores * scales                    # d score_i / d v_i = r_i u - score_i r_i^2 v_i / D
    along_v = (along_u * scores * scales / width).neg()
    d_values = values * along_v[:n, ..., None].to(dtype)
    d_values = torch.addcmul(d_values, weights[:n, ..., None].to(dtype), g)
    d_values = torch.addcmul(d_values, along_u[:n, ..., None].to(dtype), w)
    d_values_sum.add_(d_values)
    d_u = torch.einsum("nbt,nbtd->d", along_u[:n].to(dtype), values).float()
    d_partial = None
    if partial is not None:
        d_partial = (partial * along_v[n, ..., None].to(dtype)
                     + weights[n, ..., None].to(dtype) * g + along_u[n, ..., None].to(dtype) * w)
        d_u = d_u + (along_u[n, ..., None].to(dtype) * partial).sum((0, 1), dtype=torch.float32)
    return d_u, d_partial


_compiled = {}  # _mix and _unmix through torch.compile, made at first use


def _through_compile(function):
    """function through torch.compile, its shapes dynamic after a first
    recompile, so that the sublayers' numbers of sources share a few forms."""
    if function not in _compiled:
        _compiled[function] = torch.compile(function)
    return _compiled[function]


def _kernels(compiled, values, partial):
    """Whether a sum runs through the Triton kernels (residual_kernels.py):
    in a compiled pass on a GPU, with no partial sum. The kernels cover
    the full form, where every sum has none, which WPT's recipes use; a
    sum with a partial sum runs through torch.compile, open to review."""
    return compiled and partial is None and values.is_cuda


def _residual_kernels():
    """residual_kernels.py, imported at first use: it needs Triton."""
    from . import residual_kernels
    return residual_kernels


class _Mixing(torch.autograd.Function):
    """h = sum_i alpha_i v_i over the n complete sources (the first n places
    of values) and the partial sum when there is one, alpha the softmax of
    the scores (u . v_i) r_i; u = w * g. `added` are the complete sources as
    autograd sees them: its backward pass adds their gradients into
    gradients (_Gradients), or keeps what they need in its slot of
    gradients.kept, and hands them nothing. compiled: both passes
    run through the Triton kernels of residual_kernels.py on a GPU when
    there is no partial sum (_kernels), else their math (_mix, _unmix)
    through torch.compile, which fuses it into a few kernels."""

    @staticmethod
    def forward(ctx, u, values, scales, partial, compiled, gradients, slot, *added):
        n = len(added)
        values, scales = values[:n], scales[:n]
        scores = weights = partial_scale = None
        with torch.autocast(values.device.type, enabled=False):
            if _kernels(compiled, values, partial):  # what its backward pass needs in gradients.kept's slot
                h = _residual_kernels().mix(u.to(values.dtype), values, scales, gradients.kernels_kept(), slot)
            else:
                mix = _through_compile(_mix) if compiled else _mix
                h, scores, weights, partial_scale = mix(u, values, scales, partial)
        # the places are no saved tensors: later places are written after this, while these never change
        ctx.values, ctx.scales, ctx.compiled, ctx.gradients, ctx.done = values, scales, compiled, gradients, False
        ctx.slot = slot
        ctx.save_for_backward(u, scores, weights, partial, partial_scale)
        return h

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if ctx.done:  # the gradients' sums hold the first time's shares
            raise RuntimeError("attention residuals: a pass's graph is gone through backward once")
        ctx.done = True
        u, scores, weights, partial, partial_scale = ctx.saved_tensors
        n, values = len(ctx.values), ctx.values
        with torch.autocast(values.device.type, enabled=False):
            if _kernels(ctx.compiled, values, partial):
                d_u = _residual_kernels().unmix(grad.to(values.dtype).contiguous(), values, ctx.scales,
                                                ctx.gradients.kept, ctx.slot)
                d_partial = None
            else:
                unmix = _through_compile(_unmix) if ctx.compiled else _unmix
                d_u, d_partial = unmix(grad, u, values, ctx.scales, scores, weights, partial, partial_scale,
                                       ctx.gradients.places(n))
        return (d_u.to(u.dtype), None, None, d_partial, None, None, None, *([None] * n))
