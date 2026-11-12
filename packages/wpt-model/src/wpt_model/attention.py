"""HySparse2's attention (arXiv 2609.26368; docs/model.md, "Attention"):
three kinds of layer. Every one has a single key-value head shared by all
its query heads (MQA), normalizes each query head and each key by RMSNorm
before comparing them (QK-norm, IMU-1, arXiv 2602.02522), and scales the
scores by 1 / sqrt(head width).

    WindowAttention  each query reads its last WINDOW tokens, itself
                     included; rotary position encoding (base ROPE_BASE) on
                     the last ROPE_DIMS dimensions of each query and key, the
                     only position encoding in the model
    ExactAttention   each query reads every token up to itself, without
                     position encoding, and picks its TOP_K tokens among
                     those at least WINDOW tokens back (kernels.py); it
                     hands its keys and values and those choices to the
                     sparse layers after it. An exact layer no sparse layer
                     follows (the first half's) picks none. Its keys and
                     values come from
                     the input it is given: its own, or in the model's
                     second half the first half's output (KV bridging)
    SparseAttention  no keys or values of its own: each query reads the
                     exact layer's at its TOP_K chosen tokens plus its last
                     WINDOW tokens, through the layer's own query
                     projection; no position encoding

The window and sparse layers multiply their output by a sigmoid gate of
their input, per dimension of each head (Qiu et al., 2025), and add a
learned sink logit per head to the softmax's denominator, as HySparse2
says of its sparse and window layers. The exact layer has neither: the
paper names them for those two kinds only, and this reading of it is open
to review.

A window layer passes on (carry.py) the keys and values of its last
WINDOW - 1 tokens, an exact layer the key and value of every token; a pass
may follow earlier tokens (sequence.py: a segment or a slice of a training
window, or new tokens read through a generation cache), and the same
forward computes every case. Positions count from the text's first token.

The window layers' attention runs here in PyTorch (window_attention), in
blocks of WINDOW queries that each read two blocks of keys, so its cost
grows with the text times WINDOW; the exact and sparse layers' runs in
kernels.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .carry import Entries, Tail
from .kernels import exact_attention, sparse_attention
from .layers import EPS
from .sequence import WHOLE

WINDOW = 128      # tokens a window layer reads, itself included; a sparse layer reads them too
TOP_K = 1024      # tokens an exact layer picks for each query, outside its last WINDOW
ROPE_DIMS = 64    # rotary dimensions at the end of each query and key of a window layer
ROPE_BASE = 10000


def rotate(x, positions):
    """Rotary position encoding of the last ROPE_DIMS dimensions of x (B, T,
    d) or (B, T, H, d), in adjacent pairs, at positions (T,), or (B, T) for
    rows at different places."""
    pairs = torch.arange(ROPE_DIMS // 2, dtype=torch.float32, device=x.device)
    angles = positions.float()[..., None] / ROPE_BASE ** (2 * pairs / ROPE_DIMS)
    cos, sin = angles.cos(), angles.sin()
    if x.ndim == 4:
        cos, sin = cos[..., None, :], sin[..., None, :]
    rest, turning = x[..., :-ROPE_DIMS], x[..., -ROPE_DIMS:].float()
    even, odd = turning[..., 0::2], turning[..., 1::2]
    turned = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
    return torch.cat([rest, turned.to(x.dtype)], -1)


def dense_attention(q, k, v, sink, visible, scale):
    """Attention of q (..., Q, H, c) over k and v (..., K, c), each query
    reading the keys visible (..., Q, K) shows it, with sink (H,) added to
    each head's softmax denominator: (..., Q, H, c). Scores and softmax in
    FP32."""
    z = torch.einsum("...qhc,...kc->...hqk", q, k).float() * scale
    z = z.masked_fill(~visible[..., None, :, :], float("-inf"))
    lse = torch.logaddexp(torch.logsumexp(z, -1), sink.float()[:, None])
    return torch.einsum("...hqk,...kc->...qhc", (z - lse[..., None]).exp().to(v.dtype), v)


def window_attention(q, k, v, sink, positions, scale):
    """q (B, T, H, c) at positions (T,) or (B, T); k and v (B, WINDOW - 1 +
    T, c): the WINDOW - 1 tokens before the pass's first, then its own.
    Each block of WINDOW queries reads the block of keys before it and its
    own, and each query the last WINDOW of them, itself included, none
    before the text: (B, T, H, c)."""
    B, T, H, c = q.shape
    W = WINDOW
    n = -(-T // W)
    pad = n * W - T
    q = F.pad(q, (0, 0, 0, 0, 0, pad)).view(B, n, W, H, c)

    def blocks(table):  # (B, W - 1 + T, c) -> (B, n, 2W, c): for block i, the table's blocks i and i + 1
        table = F.pad(table, (0, 0, 1, pad)).view(B, n + 1, W, -1)
        return torch.cat([table[:, :-1], table[:, 1:]], 2)

    later = positions[..., -1:] + 1 + torch.arange(pad, device=positions.device)
    places = torch.cat([positions, later.expand(*positions.shape[:-1], pad)], -1)
    places = places.view(*positions.shape[:-1], n, W)  # (n, W) or (B, n, W)
    steps = torch.arange(W, device=q.device)
    distance = W + steps[:, None] - torch.arange(2 * W, device=q.device)  # (W, 2W): how far back each key is
    visible = (distance >= 0) & (distance < W) & (places[..., None] >= distance)
    return dense_attention(q, blocks(k), blocks(v), sink, visible, scale).reshape(B, n * W, H, c)[:, :T]


class WindowAttention(nn.Module):
    """A window layer's attention (see the module's docstring)."""

    def __init__(self, config):
        super().__init__()
        D, H, c = config.d_model, config.attention_heads, config.attention_head_dim
        self.heads, self.head_dim = H, c
        self.q = nn.Linear(D, H * c, bias=False)
        self.q_norm = nn.RMSNorm(c, eps=EPS)
        self.k = nn.Linear(D, c, bias=False)
        self.k_norm = nn.RMSNorm(c, eps=EPS)
        self.v = nn.Linear(D, c, bias=False)
        self.gate = nn.Linear(D, H * c, bias=False)
        self.sink = nn.Parameter(torch.zeros(H))  # one sink logit per head, starting at 0
        self.out = nn.Linear(H * c, D, bias=False)

    def passes_on(self):
        """The keys and values of its last WINDOW - 1 tokens, side by side."""
        return {"window": Tail(WINDOW - 1)}

    def queries(self, x, positions):
        B, T, _ = x.shape
        return rotate(self.q_norm(self.q(x).view(B, T, self.heads, self.head_dim)), positions)

    def entries(self, x, positions):
        """Each token's key and value, side by side (B, T, 2c)."""
        return torch.cat([rotate(self.k_norm(self.k(x)), positions), self.v(x)], -1)

    def output(self, o, x):
        """The gated heads o (B, T, H, c), projected back to the width."""
        return self.out((o * torch.sigmoid(self.gate(x)).view(o.shape)).flatten(2))

    def forward(self, x, sequence=WHOLE):
        """(B, T, D) -> (B, T, D). sequence: where these tokens sit in the
        window (sequence.py)."""
        positions = sequence.positions(x.shape[1], x.device)
        entries = self.entries(x, positions)
        table = torch.cat([sequence.before(self, "window", entries).to(entries.dtype), entries], 1)
        c = self.head_dim
        o = window_attention(self.queries(x, positions), table[..., :c], table[..., c:], self.sink, positions,
                             c ** -0.5)
        return self.output(o, x)


class ExactAttention(nn.Module):
    """An exact layer's attention (see the module's docstring)."""

    def __init__(self, config, chooses=True):
        """chooses: whether sparse layers follow it and read its choice of tokens."""
        super().__init__()
        D, H, c = config.d_model, config.attention_heads, config.attention_head_dim
        self.heads, self.head_dim = H, c
        self.chooses = chooses
        self.q = nn.Linear(D, H * c, bias=False)
        self.q_norm = nn.RMSNorm(c, eps=EPS)
        self.k = nn.Linear(D, c, bias=False)
        self.k_norm = nn.RMSNorm(c, eps=EPS)
        self.v = nn.Linear(D, c, bias=False)
        self.out = nn.Linear(H * c, D, bias=False)

    def passes_on(self):
        """The key and value of every token, side by side."""
        return {"entries": Entries()}

    def forward(self, x, source, sequence=WHOLE):
        """x (B, T, D), the queries' input, and source (B, T, D), the keys'
        and values' -> (output (B, T, D), (keys, values, chosen)): every
        token's key and value so far (B, N, c), token p's at row p, and each
        query's chosen rows (B, T, K), what the sparse layers after it
        read."""
        B, T, _ = x.shape
        H, c = self.heads, self.head_dim
        positions = sequence.positions(T, x.device)
        q = self.q_norm(self.q(x).view(B, T, H, c))
        new = torch.cat([self.k_norm(self.k(source)), self.v(source)], -1)
        table = sequence.everything(self, "entries", new)
        keys, values = table[..., :c], table[..., c:]
        o, chosen = exact_attention(q, keys, values, positions, c ** -0.5, TOP_K if self.chooses else 0, WINDOW)
        return self.out(o.flatten(2)), (keys, values, chosen)


class SparseAttention(nn.Module):
    """A sparse layer's attention (see the module's docstring)."""

    def __init__(self, config):
        super().__init__()
        D, H, c = config.d_model, config.attention_heads, config.attention_head_dim
        self.heads, self.head_dim = H, c
        self.q = nn.Linear(D, H * c, bias=False)
        self.q_norm = nn.RMSNorm(c, eps=EPS)
        self.gate = nn.Linear(D, H * c, bias=False)
        self.sink = nn.Parameter(torch.zeros(H))  # one sink logit per head, starting at 0
        self.out = nn.Linear(H * c, D, bias=False)

    def passes_on(self):
        return {}

    def forward(self, x, handoff, sequence=WHOLE):
        """x (B, T, D), handoff: what the exact layer before it gave
        (ExactAttention.forward) -> (B, T, D)."""
        B, T, _ = x.shape
        H, c = self.heads, self.head_dim
        keys, values, chosen = handoff
        positions = sequence.positions(T, x.device)
        q = self.q_norm(self.q(x).view(B, T, H, c))
        o = sparse_attention(q, keys, values, self.sink, chosen, positions, c ** -0.5, WINDOW)
        return self.out((o * torch.sigmoid(self.gate(x)).view(B, T, H, c)).flatten(2))
