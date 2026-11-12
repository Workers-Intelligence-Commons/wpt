"""The attention kernels for Triton, on an NVIDIA or AMD GPU (Linux, and
Windows through triton-windows): the two operations of kernels.py, with the
contract and the results of pytorch_kernels.py. tests/test_kernels.py
compares them with it.

Exact attention, forward and backward, is flash attention for one
key-value head (MQA): no score matrix is ever held, only tiles of it.
Because every head of a query reads the same keys, a program's tile of
queries stacks its queries' heads as rows (ROWS = queries x heads, heads
padded to a power of 2), so the matrix units get full tiles even at 4 or
8 heads. Scores are in FP32 from BF16 products, and the weights are cast
to the inputs' dtype before they multiply the values, as in flash
attention. The backward pass is two kernels with no atomic additions: one
program per block of keys sums its keys' and values' gradients over every
query that sees them, and one per block of queries its queries' gradients.

The choice of each query's top_k rows is one kernel after the forward
pass, since a row's score (its largest weight over the heads) needs each
head's final softmax denominator; it writes no scores and waits on nothing
on the host. Each score becomes a key, its FP32 bits as an integer in the
same order. A program takes a tile of queries and streams their candidate
rows several times, recomputing the scores each time:

    counting  each pass counts the keys in BINS bins of 2^shift keys,
              8 bits a bin packed into int64s, until each query's top_k-th
              key lies in a bin of at most CANDIDATES rows or of a single
              key. The first pass's bins span about SPAN bits below
              log2(heads / top_k), above which no top_k-th score lies, or
              below the query's best score when that is lower; a pass that
              finds the top_k-th below its bins counts below them with
              bins 4 times wider, one that finds it in a bin splits that
              bin into BINS
    writing   one pass writes the rows above the bin, and puts the bin's
              rows into a buffer as (key, ~row)
    ranking   the rows still wanted are the bin's best by their rank
              among its rows: a higher score first, then an earlier row

So the choice is exact: it differs from torch.topk on the same scores only
among equal scores. A bin of one key holds equal scores only, and when it
has more rows than CANDIDATES its rows are taken in the order the pass
meets them. The pass count is decided per tile, so a tile does as many as
its slowest query needs. BINS, SPAN and CANDIDATES are a first choice from
a few measured at 32K tokens on an RTX 5080 laptop GPU on random inputs,
open to review on a trained model's scores.

Sparse attention is one program per query, which gathers its chosen rows
and its last `window` rows in blocks of BLOCK and keeps an online softmax
in FP32, the sink added to the denominator at the end. Its tiles are
(rows, heads), the heads padded to at least 8: the matrix units' smallest
tile is 8 wide in a product's last dimension and 16 in its first, so a
query's few heads waste less of each product there. The program for a
query's gradient loads its next block of rows while it computes on the
current one. Its backward pass
takes the window rows in tiles, as exact attention does with a band mask,
since neighbouring queries share them: one program per block of keys
writes their key and value gradients and one per block of queries its
queries' gradients, with no atomic additions. The chosen rows differ from
query to query. One program per query sums its gradient over its chosen
rows.

The chosen rows' gradients take one of two paths, by the number of heads,
since their cost is moving data. With more than GATHER_HEADS heads, the
program of each query also computes its share of its chosen rows' key and
value gradients, (rows, heads) x (heads, CP) products with the heads
padded to 16 (the products' middle dimension), and adds them by
atomic additions of 4 FP32 values onto the window's share; the order of
those additions, and with it the rounding, may differ from run to run.
That moves 2 x CP FP32 values per (query, row) pair, whatever the heads.
With GATHER_HEADS heads or fewer, the backward pass first inverts the
choice: for each row, the list of queries that chose it (a count per row,
its running sum, then each query written into its rows' lists), made
once for every sparse layer that reads the same choice
(pytorch_kernels.inverse). One
program per row then sums its key and value gradients over that list,
adds the window's share and writes them once, with no atomic additions
on the gradients; the queries' weights are computed again there, from
their q, d_out, lse and delta, on the vector units, since one row against
a few queries fills no matrix tile. Each pass gathers from memory: a
query's chosen rows of k and v, a row's choosers' q and d_out. That
moves 2 x heads x CP BF16 values per pair, which costs less than the atomic
additions at 2 heads and more at 4 heads, measured at 32K tokens on an RTX
5080 laptop GPU; 3 heads were not measured, and that limit is open to
review.

The lists are filled by atomic additions on their counters, so the order
of the queries in a list, and with it the rounding of a row's sum, may
differ from run to run. A list in the order of the queries (a stable sort)
cost about 6 ms more per sparse layer at 32K tokens on an RTX 5080 laptop
GPU, against about 2 ms for the atomic lists; that trade is open to review.

Tile sizes are chosen by Triton's autotuner, per head count and head width
(and, for the exact passes, per power of 2 of the queries, for the sparse passes per number of chosen rows), from configs
that fit both an H100 (sm_90) and an RTX 5080 laptop GPU (sm_120, about
100 KB of shared memory per block); a config that does not fit is skipped.
"""

import math

import torch
import triton
import triton.language as tl

from .pytorch_kernels import chosen_columns, inverse

NAME = "Triton"
LOG2E = tl.constexpr(1.4426950408889634)  # scores are kept in base 2 inside the kernels, for exp2
LN2 = tl.constexpr(0.6931471805599453)
NONE = tl.constexpr(-(1 << 62))  # the key of a row a query may not choose, below every score's (_key)
ALL = tl.constexpr(1 << 62)  # above every score's key
BINS = 64  # bins per counting pass of the choice of rows, a power of 2 of at least 8
SPAN = 8.0  # the first counting pass's bins span about this many bits (base-2 log-weights) below its start
CANDIDATES = 512  # the most rows a query's last bin may hold to be ordered exactly between them
LEVELS = 16  # the most counting passes
GATHER_HEADS = 2  # the most heads for which each chosen row gathers its queries; above, the queries scatter


def _configs(shapes):
    return [triton.Config({"ROWS": rows, "KEYS": keys}, num_warps=warps, num_stages=stages)
            for rows, keys, warps, stages in shapes]


def _fits(configs, _arguments, **constants):
    """The configs whose tile of rows holds at least one query's heads."""
    kept = [config for config in configs if config.kwargs["ROWS"] >= constants["HP"]]
    return kept or configs[:1]


_PRUNE = {"early_config_prune": _fits}
_FORWARD = _configs([(128, 64, 8, 2), (128, 32, 4, 2), (64, 64, 4, 2), (64, 32, 4, 3), (256, 32, 8, 2)])
_CHOICE = _configs([(64, 64, 4, 2), (64, 32, 4, 2), (128, 64, 8, 2), (64, 128, 4, 2), (32, 64, 4, 2)])
_BACKWARD_KEYS = _configs([(16, 64, 4, 2), (32, 64, 4, 2), (32, 128, 4, 2), (64, 64, 4, 2), (64, 128, 8, 2)])
_SPARSE = [triton.Config({"BLOCK": block}, num_warps=warps, num_stages=stages)
           for block, warps, stages in [(32, 2, 2), (64, 4, 2), (64, 2, 3), (128, 4, 2), (128, 8, 2)]]
_SPARSE_BACKWARD = [triton.Config({"BLOCK": block}, num_warps=warps, num_stages=stages)
                    for block, warps, stages in [(16, 4, 2), (32, 2, 2), (32, 4, 1), (32, 4, 2), (32, 8, 2),
                                                 (64, 4, 1), (64, 4, 2), (64, 8, 2), (128, 8, 1)]]
_SPARSE_KEYS_BACKWARD = [triton.Config({"E": e}, num_warps=warps, num_stages=stages)
                         for e, warps, stages in [(8, 2, 1), (8, 4, 2), (16, 2, 2), (16, 2, 3), (16, 4, 2),
                                                  (32, 4, 2), (32, 4, 3), (32, 8, 2), (64, 8, 2)]]
_BACKWARD_QUERIES = _configs([(128, 64, 8, 2), (128, 32, 4, 2), (64, 64, 4, 2), (64, 32, 4, 3)])


def _size_class(T):
    """The autotuner's key for the number of queries: its power of 2, up to 8,192."""
    return min(triton.next_power_of_2(max(T, 1)), 8192)


def _heads(H, c, least=1):
    """The kernels' shape constants: heads padded to a power of 2 of at
    least `least`, the head width to one of at least 16."""
    return dict(H=H, HP=max(least, triton.next_power_of_2(H)), C=c, CP=max(16, triton.next_power_of_2(c)))


def _table(x):
    """A key or value table (B, N, c) whose last dimension is contiguous,
    such as the halves of a layer's entries: no copy when it already is."""
    return x if x.stride(-1) == 1 else x.contiguous()


# -- exact attention --

def exact_attention(q, k, v, positions, scale, top_k, window):
    """pytorch_kernels.exact_attention's result."""
    return _Exact.apply(q, k, v, positions, scale, top_k, window)


class _Exact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, positions, scale, top_k, window):
        q, k, v, positions = q.contiguous(), _table(k), _table(v), positions.contiguous()
        out, lse, top = _exact_forward(q, k, v, positions, scale)
        chosen = _choose(q, k, positions, lse, top, scale, top_k, window)
        ctx.save_for_backward(q, k, v, positions, out, lse)
        ctx.scale = scale
        ctx.mark_non_differentiable(chosen)
        return out, chosen

    @staticmethod
    def backward(ctx, d_out, _d_chosen):
        q, k, v, positions, out, lse = ctx.saved_tensors
        dq, dk, dv = _exact_backward(q, k, v, positions, out, lse, d_out.contiguous(), ctx.scale)
        return dq, dk, dv, None, None, None, None


def _exact_forward(q, k, v, positions, scale):
    B, T, H, c = q.shape
    N = k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    top = torch.empty_like(lse)
    shape = _heads(H, c)
    grid = lambda meta: (triton.cdiv(T, meta["ROWS"] // shape["HP"]), B)
    _exact_forward_kernel[grid](q, k, v, positions, out, lse, top, scale * LOG2E.value, T, N, k.stride(0),
                                k.stride(1), v.stride(0), v.stride(1), _size_class(T), **shape)
    return out, lse, top


def _choose(q, k, positions, lse, top, scale, top_k, window):
    """Each query's top_k rows among those at least window tokens back, by
    their largest weight over the heads: (B, T, K), -1 for none, K =
    chosen_columns(top_k, window, N). top (B, T, H): the forward pass's
    running maximum of each head's scores, in natural logs."""
    B, T, H, c = q.shape
    N = k.shape[1]
    K = chosen_columns(top_k, window, N)
    chosen = torch.empty(B, T, K, dtype=torch.long, device=q.device)
    if K == 0:
        return chosen
    shape = _heads(H, c)
    bin_rows = torch.empty(B, T, CANDIDATES, dtype=torch.long, device=q.device)
    grid = lambda meta: (triton.cdiv(T, meta["ROWS"] // shape["HP"]), B)
    _exact_choice_kernel[grid](q, k, positions, lse, top, chosen, bin_rows, scale * LOG2E.value, T, N, K, window,
                               k.stride(0), k.stride(1), _size_class(T), BINS=BINS, LOG_BINS=BINS.bit_length() - 1,
                               CAP=CANDIDATES, LOG_WIDTH=round(math.log2(SPAN / BINS)), LEVELS=LEVELS, **shape)
    return chosen


def _exact_backward(q, k, v, positions, out, lse, d_out, scale):
    B, T, H, c = q.shape
    N = k.shape[1]
    delta = (d_out.float() * out.float()).sum(-1)  # (B, T, H): sum_j p_j dp_j
    reach = positions.cummax(1).values.contiguous()  # (B, T): the farthest row any query up to t sees
    dq = torch.empty_like(q)
    dk = torch.empty(B, N, c, dtype=k.dtype, device=k.device)
    dv = torch.empty(B, N, c, dtype=v.dtype, device=v.device)
    shape, qk_scale = _heads(H, c), scale * LOG2E.value
    strides = (k.stride(0), k.stride(1), v.stride(0), v.stride(1))
    _exact_keys_backward_kernel[lambda meta: (triton.cdiv(N, meta["KEYS"]), B)](
        q, k, v, d_out, lse, delta, positions, reach, dk, dv, qk_scale, scale, T, N, *strides, _size_class(T), **shape)
    _exact_queries_backward_kernel[lambda meta: (triton.cdiv(T, meta["ROWS"] // shape["HP"]), B)](
        q, k, v, d_out, lse, delta, positions, dq, qk_scale, scale, T, N, *strides, _size_class(T), **shape)
    return dq, dk, dv


@triton.jit
def _query_rows(pos_ptr, b, t0, T, end, H: tl.constexpr, HP: tl.constexpr, ROWS: tl.constexpr):
    """A tile's rows: query t0 + r // HP of T, head r % HP. (each row's
    query, head, whether it is a real query before end and a real head, its
    query's position)."""
    r = tl.arange(0, ROWS)
    t = t0 + r // HP
    h = r % HP
    p = tl.load(pos_ptr + b * T + t, mask=t < end, other=0)
    return t, h, (t < end) & (h < H), p


@triton.jit
def _load_rows(ptr, b, t, h, valid, T, H: tl.constexpr, C: tl.constexpr, CP: tl.constexpr):
    """The (ROWS, CP) values of each row's query and head, 0 where none."""
    d = tl.arange(0, CP)
    rows = (b * T + t) * H + h
    return tl.load(ptr + rows[:, None] * C + d[None, :], mask=valid[:, None] & (d[None, :] < C), other=0.0)


@triton.jit
def _load_keys(ptr, b, n, N, stride_b, stride_n, C: tl.constexpr, CP: tl.constexpr):
    """Rows n of a table (KEYS, CP), 0 past its N rows."""
    d = tl.arange(0, CP)
    return tl.load(ptr + b * stride_b + n[:, None] * stride_n + d[None, :], mask=(n[:, None] < N) & (d[None, :] < C),
                   other=0.0)


@triton.jit
def _forward_tile(q, acc, m, l, k_ptr, v_ptr, b, n0, N, p, qk_scale, stride_kb, stride_kn, stride_vb, stride_vn,
                  C: tl.constexpr, CP: tl.constexpr, KEYS: tl.constexpr, MASKED: tl.constexpr):
    n = n0 + tl.arange(0, KEYS)
    k = _load_keys(k_ptr, b, n, N, stride_kb, stride_kn, C, CP)
    z = tl.dot(q, tl.trans(k)) * qk_scale
    if MASKED:
        z = tl.where((n[None, :] <= p[:, None]) & (n[None, :] < N), z, float("-inf"))
    m_new = tl.maximum(m, tl.max(z, 1))  # every row reads row 0 in its first tile, so m_new is finite
    alpha = tl.exp2(m - m_new)
    w = tl.exp2(z - m_new[:, None])
    v = _load_keys(v_ptr, b, n, N, stride_vb, stride_vn, C, CP)
    return acc * alpha[:, None] + tl.dot(w.to(v.dtype), v), m_new, l * alpha + tl.sum(w, 1)


@triton.autotune(configs=_FORWARD, key=["H", "C", "size"], prune_configs_by=_PRUNE)
@triton.jit
def _exact_forward_kernel(q_ptr, k_ptr, v_ptr, pos_ptr, out_ptr, lse_ptr, top_ptr, qk_scale, T, N, stride_kb, stride_kn,
                          stride_vb, stride_vn, size, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr,
                          CP: tl.constexpr, ROWS: tl.constexpr, KEYS: tl.constexpr):
    QT: tl.constexpr = ROWS // HP
    b = tl.program_id(1).to(tl.int64)
    t, h, valid, p = _query_rows(pos_ptr, b, tl.program_id(0) * QT, T, T, H, HP, ROWS)
    q = _load_rows(q_ptr, b, t, h, valid, T, H, C, CP)
    end = tl.minimum(tl.max(tl.where(valid, p, 0)) + 1, N)
    full = tl.minimum(tl.min(tl.where(valid, p, N)) + 1, N) // KEYS * KEYS  # tiles every row sees whole
    m = tl.full([ROWS], float("-inf"), tl.float32)
    l = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, CP], tl.float32)
    for n0 in range(0, full, KEYS):
        acc, m, l = _forward_tile(q, acc, m, l, k_ptr, v_ptr, b, n0, N, p, qk_scale, stride_kb, stride_kn, stride_vb,
                                  stride_vn, C, CP, KEYS, False)
    for n0 in range(full, end, KEYS):
        acc, m, l = _forward_tile(q, acc, m, l, k_ptr, v_ptr, b, n0, N, p, qk_scale, stride_kb, stride_kn, stride_vb,
                                  stride_vn, C, CP, KEYS, True)
    d = tl.arange(0, CP)
    rows = (b * T + t) * H + h
    tl.store(out_ptr + rows[:, None] * C + d[None, :], (acc / l[:, None]).to(out_ptr.dtype.element_ty),
             mask=valid[:, None] & (d[None, :] < C))
    tl.store(lse_ptr + rows, (m + tl.log2(l)) * LN2, mask=valid)
    tl.store(top_ptr + rows, m * LN2, mask=valid)


@triton.jit
def _head_rows(pos_ptr, b, t0, T, H: tl.constexpr, HP: tl.constexpr, ROWS: tl.constexpr):
    """A tile's rows head by head: query t0 + r % QT, head r // QT, QT =
    ROWS // HP. (each row's query, head, whether it is a real query and
    head, its query's position)."""
    QT: tl.constexpr = ROWS // HP
    r = tl.arange(0, ROWS)
    t = t0 + r % QT
    h = r // QT
    p = tl.load(pos_ptr + b * T + t, mask=t < T, other=0)
    return t, h, (t < T) & (h < H), p


@triton.jit
def _score_keys(q, lse, k_ptr, b, n0, N, last, qk_scale, stride_kb, stride_kn, C: tl.constexpr, CP: tl.constexpr,
                QT: tl.constexpr, HP: tl.constexpr, KEYS: tl.constexpr):
    """The scores of rows n0 .. n0 + KEYS for a tile's queries, its rows
    head by head (_head_rows), as keys (QT, KEYS) int64 (_key), NONE where
    a query may not choose the row; and the rows."""
    n = n0 + tl.arange(0, KEYS)
    k = _load_keys(k_ptr, b, n, N, stride_kb, stride_kn, C, CP)
    z = tl.dot(q, tl.trans(k)) * qk_scale - lse[:, None]  # lse is +inf on rows that are no real query and head
    s = tl.max(tl.reshape(z, (HP, QT, KEYS)), 0)
    return tl.where(n[None, :] <= last[:, None], _key(s), NONE), n


@triton.jit
def _key(s):
    """The bits of FP32 scores as int64 in the same order: a higher score,
    a larger key; equal keys, equal scores."""
    bits = s.to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64)


@triton.jit
def _column(x, j, J: tl.constexpr):
    """Column j (QT,) of x (QT, J), j (QT,) one column per row; 0 where j is outside 0 .. J - 1."""
    return tl.sum(tl.where(tl.arange(0, J)[None, :] == j[:, None], x, 0), 1)


@triton.jit
def _bin_counts(q, lse, k_ptr, b, end, N, last, hi, top, shift, open_, qk_scale, stride_kb, stride_kn,
                C: tl.constexpr, CP: tl.constexpr, QT: tl.constexpr, HP: tl.constexpr, KEYS: tl.constexpr,
                BINS: tl.constexpr):
    """One counting pass: (QT, BINS), column j the rows of the open
    queries with a key at most hi in bins 0 .. j, bin j holding the keys
    top - key in [j, j + 1) 2^shift, bin 0 also those above top. The bins
    are counted 8 bits a bin, 8 bins to an int64, one sum per 8 bins and
    tile."""
    j = tl.arange(0, BINS)
    counts = tl.zeros([QT, BINS], tl.int32)
    for n0 in range(0, end, KEYS):
        key, n = _score_keys(q, lse, k_ptr, b, n0, N, last, qk_scale, stride_kb, stride_kn, C, CP, QT, HP, KEYS)
        counted = (key <= hi[:, None]) & open_[:, None]
        bin = tl.where(counted, tl.minimum(tl.maximum((top[:, None] - key) >> shift[:, None], 0), BINS), BINS)
        bin = bin.to(tl.int32)
        one = tl.full([QT, KEYS], 1, tl.int64) << ((bin % 8) * 8).to(tl.int64)
        for a in tl.static_range(BINS // 8):
            packed = tl.sum(tl.where(bin // 8 == a, one, 0), 1)  # (QT,): bins 8a .. 8a + 7, 8 bits each
            counts += tl.where(j[None, :] // 8 == a, ((packed[:, None] >> ((j[None, :] % 8) * 8)) & 255).to(tl.int32),
                               0)
    return tl.cumsum(counts, 1)


@triton.autotune(configs=_CHOICE, key=["H", "C", "size"], prune_configs_by=_PRUNE)
@triton.jit
def _exact_choice_kernel(q_ptr, k_ptr, pos_ptr, lse_ptr, top_ptr, chosen_ptr, bin_ptr, qk_scale, T, N, K, window,
                         stride_kb, stride_kn, size, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr,
                         CP: tl.constexpr, ROWS: tl.constexpr, KEYS: tl.constexpr, BINS: tl.constexpr,
                         LOG_BINS: tl.constexpr, CAP: tl.constexpr, LOG_WIDTH: tl.constexpr, LEVELS: tl.constexpr):
    """The chosen rows (B, T, K) of a tile of queries, from streaming passes
    over their candidate rows (the module's docstring): passes that count
    their scores' keys in BINS bins until each query's top_k-th key lies in
    a bin [lo, hi] of at most CAP rows, or of one key, one that writes the
    rows above hi and puts the bin's rows into bin_ptr (B, T, CAP), then
    the bin's best rows by their exact order."""
    QT: tl.constexpr = ROWS // HP
    b = tl.program_id(1).to(tl.int64)
    t0 = tl.program_id(0) * QT
    t, h, valid, p = _head_rows(pos_ptr, b, t0, T, H, HP, ROWS)
    q = _load_rows(q_ptr, b, t, h, valid, T, H, C, CP)
    rows = (b * T + t) * H + h
    lse = tl.load(lse_ptr + rows, mask=valid, other=float("inf")) * LOG2E
    best = tl.load(top_ptr + rows, mask=valid, other=0.0) * LOG2E - lse
    best = tl.max(tl.reshape(best, (HP, QT)), 0)  # (QT,): no score is above it
    queries = t0 + tl.arange(0, QT)
    real = queries < T
    last = tl.minimum(tl.load(pos_ptr + b * T + queries, mask=real, other=0) - window, N - 1)
    last = tl.where(real, last, -1)  # (QT,): the last row each query may choose
    count = tl.maximum(last + 1, 0)
    wanted = tl.minimum(count, K)  # rows to choose; every candidate when there are no more than K
    end = tl.max(last) + 1
    # the bin: rows with a key above hi are chosen, the top_k-th has a key in [lo, hi]
    open_ = wanted < count
    hi = tl.where(open_, ALL, NONE)
    lo = hi
    above = tl.where(open_, 0, wanted)  # rows with a key above hi
    in_bin = tl.zeros([QT], tl.int32)
    # The weights of a query's best `wanted` rows add up to at most H, so its wanted-th score is at most
    # log2(H / wanted): the first pass's bins start from there, or from its best score when lower, and are
    # about SPAN / BINS = 2^LOG_WIDTH bits wide there: 2^shift keys, that width over the FP32 spacing there
    start = tl.minimum(best, tl.log2(H / tl.maximum(wanted, 1).to(tl.float32)))
    top = _key(start)
    exponent = tl.maximum(((start.to(tl.int32, bitcast=True) >> 23) & 255) - 127, 0)
    shift = tl.minimum(tl.maximum(23 - exponent + LOG_WIDTH, 0), 40).to(tl.int64)
    level = 0
    while (tl.max(open_.to(tl.int32)) > 0) & (level < LEVELS):
        counts = _bin_counts(q, lse, k_ptr, b, end, N, last, hi, top, shift, open_, qk_scale, stride_kb, stride_kn,
                             C, CP, QT, HP, KEYS, BINS)
        need = wanted - above
        first = tl.sum((counts < need[:, None]).to(tl.int32), 1)  # the first bin whose count reaches need
        found = first < BINS
        before = tl.where(first > 0, _column(counts, first - 1, BINS), 0)
        reached = _column(counts, first, BINS)
        width = tl.full([QT], 1, tl.int64) << shift
        bin_top = tl.where(first > 0, top - first.to(tl.int64) * width, top)
        settled = found & ((reached - before <= CAP) | (shift == 0))  # one key: equal scores
        settled = settled | (level == LEVELS - 1)
        # found: into the bin, its keys split into BINS bins; else every row counted is chosen, and the next
        # pass counts below them, its bins 4 times wider
        lowest = top - BINS * width
        hi = tl.where(open_, tl.where(found, tl.where(first > 0, bin_top, hi), lowest), hi)
        lo = tl.where(open_ & found, top - (first + 1).to(tl.int64) * width + 1,
                      tl.where(open_ & settled, NONE + 1, lo))
        above = tl.where(open_, above + tl.where(found, before, _column(counts, tl.full([QT], BINS - 1, tl.int32),
                                                                        BINS)), above)
        in_bin = tl.where(open_ & found, reached - before, tl.where(open_ & settled, CAP + 1, in_bin))
        top = tl.where(open_, tl.where(found, bin_top, lowest), top)
        shift = tl.where(open_, tl.where(found, tl.maximum(shift - LOG_BINS, 0), shift + 2), shift)
        open_ = open_ & ~settled
        level += 1
    # the writing pass: rows with a key above hi into the first `above` columns, the bin's rows after them
    # when it holds more than CAP (equal scores, taken in the order met), else into bin_ptr
    direct = in_bin > CAP
    chosen_base = chosen_ptr + (b * T + queries) * K
    bin_base = bin_ptr + (b * T + queries) * CAP
    sure_slot = tl.zeros([QT], tl.int32)
    bin_slot = tl.zeros([QT], tl.int32)
    any_direct = tl.max(direct.to(tl.int32)) > 0
    # running counts along a tile's rows as one product with a triangle of ones, on the matrix units: the rows
    # above hi count 1 and the bin's 256, exact in FP32
    triangle = (tl.arange(0, KEYS)[:, None] <= tl.arange(0, KEYS)[None, :]).to(tl.float16)
    for n0 in range(0, end, KEYS):
        key, n = _score_keys(q, lse, k_ptr, b, n0, N, last, qk_scale, stride_kb, stride_kn, C, CP, QT, HP, KEYS)
        sure = (key > hi[:, None]) & (key != NONE)
        in_it = (key >= lo[:, None]) & (key <= hi[:, None]) & (key != NONE)
        marks = sure.to(tl.int32) + in_it.to(tl.int32) * 256
        running = tl.dot(marks.to(tl.float16), triangle).to(tl.int32)
        slot = sure_slot[:, None] + running % 256 - 1
        tl.store(chosen_base[:, None] + slot, n[None, :].to(tl.int64), mask=sure)
        slot = bin_slot[:, None] + running // 256 - 1
        if any_direct:
            tl.store(chosen_base[:, None] + above[:, None] + slot, n[None, :].to(tl.int64),
                     mask=in_it & direct[:, None] & (above[:, None] + slot < wanted[:, None]))
        tl.store(bin_base[:, None] + slot, (key << 32) | (n[None, :].to(tl.int64) ^ 0xFFFFFFFF),
                 mask=in_it & ~direct[:, None])
        added = tl.sum(marks, 1)
        sure_slot += added % 256
        bin_slot += added // 256
    # other threads of the program wrote the rows that the ranking reads
    tl.debug_barrier()
    # the bin's best rows, `need` of them, by their rank among its rows: (key, ~row) larger first, a higher
    # score, then an earlier row
    need = tl.where(direct, 0, wanted - above)
    listed = tl.max(tl.where(need > 0, in_bin, 0))
    for c0 in range(0, listed, KEYS):
        c = c0 + tl.arange(0, KEYS)
        present = (c[None, :] < in_bin[:, None]) & (need[:, None] > 0)
        ranked = tl.load(bin_base[:, None] + c[None, :], mask=present, other=0)
        rank = tl.zeros([QT, KEYS], tl.int32)
        for i in range(0, listed):
            other = tl.load(bin_base + i, mask=(i < in_bin) & (need > 0), other=0)
            rank += ((other[:, None] > ranked) & (i < in_bin)[:, None]).to(tl.int32)
        tl.store(chosen_base[:, None] + above[:, None] + rank, ranked & 0xFFFFFFFF ^ 0xFFFFFFFF,
                 mask=present & (rank < need[:, None]))
    # -1 past each query's rows
    for c0 in range(tl.min(tl.where(real, wanted, K)) // KEYS * KEYS, K, KEYS):
        col = c0 + tl.arange(0, KEYS)
        tl.store(chosen_base[:, None] + col[None, :], tl.full([QT, KEYS], -1, tl.int64),
                 mask=real[:, None] & (col[None, :] >= wanted[:, None]) & (col[None, :] < K))


@triton.jit
def _first_at_least(ptr, T, value):
    """The first t of a nondecreasing (T,) row whose value is at least value, or T."""
    lo = T * 0
    hi = T + 0
    while lo < hi:
        mid = (lo + hi) // 2
        reached = tl.load(ptr + mid) >= value
        hi = tl.where(reached, mid, hi)
        lo = tl.where(reached, lo, mid + 1)
    return lo


@triton.jit
def _keys_tile(dk, dv, k, v, n, q_ptr, do_ptr, lse_ptr, delta_ptr, pos_ptr, b, t0, T, N, qk_scale, window,
               H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, ROWS: tl.constexpr,
               WINDOWED: tl.constexpr):
    """One tile of queries' share of a block's dk and dv. Every tile is
    masked: tiles split into masked and whole ones, as the queries' kernel
    does, ran 3 to 5 times slower on the one GPU this was measured on (an
    AMD Radeon 860M), and this choice is open to review on NVIDIA GPUs.
    WINDOWED: each query sees only its last `window` rows."""
    t, h, valid, p = _query_rows(pos_ptr, b, t0, T, T, H, HP, ROWS)
    q = _load_rows(q_ptr, b, t, h, valid, T, H, C, CP)
    do = _load_rows(do_ptr, b, t, h, valid, T, H, C, CP)
    rows = (b * T + t) * H + h
    lse = tl.load(lse_ptr + rows, mask=valid, other=0.0) * LOG2E
    delta = tl.load(delta_ptr + rows, mask=valid, other=0.0)
    seen = (n[:, None] <= p[None, :]) & (n[:, None] < N) & valid[None, :]
    if WINDOWED:
        seen = seen & (p[None, :] - n[:, None] < window)
    w = tl.where(seen, tl.exp2(tl.dot(k, tl.trans(q)) * qk_scale - lse[None, :]), 0.0)  # (KEYS, ROWS)
    dv += tl.dot(w.to(do.dtype), do)
    ds = w * (tl.dot(v, tl.trans(do)) - delta[None, :])
    return dk + tl.dot(ds.to(q.dtype), q), dv


@triton.autotune(configs=_BACKWARD_KEYS, key=["H", "C", "size"], prune_configs_by=_PRUNE)
@triton.jit
def _exact_keys_backward_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, pos_ptr, reach_ptr, dk_ptr, dv_ptr,
                                qk_scale, scale, T, N, stride_kb, stride_kn, stride_vb, stride_vn, size,
                                H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr,
                                ROWS: tl.constexpr, KEYS: tl.constexpr):
    """dk and dv of one block of KEYS rows, over every query that sees one
    of them: those from the first whose reach (the largest position up to
    it) gets to the block."""
    QT: tl.constexpr = ROWS // HP
    b = tl.program_id(1).to(tl.int64)
    n = tl.program_id(0) * KEYS + tl.arange(0, KEYS)
    k = _load_keys(k_ptr, b, n, N, stride_kb, stride_kn, C, CP)
    v = _load_keys(v_ptr, b, n, N, stride_vb, stride_vn, C, CP)
    dk = tl.zeros([KEYS, CP], tl.float32)
    dv = tl.zeros([KEYS, CP], tl.float32)
    first = _first_at_least(reach_ptr + b * T, T, tl.program_id(0) * KEYS)  # the first query that sees the block
    for t0 in range(first // QT * QT, T, QT):
        dk, dv = _keys_tile(dk, dv, k, v, n, q_ptr, do_ptr, lse_ptr, delta_ptr, pos_ptr, b, t0, T, N, qk_scale, 0, H,
                            HP, C, CP, ROWS, False)
    d = tl.arange(0, CP)
    there = (n[:, None] < N) & (d[None, :] < C)
    offsets = (b * N + n)[:, None] * C + d[None, :]
    tl.store(dk_ptr + offsets, (dk * scale).to(dk_ptr.dtype.element_ty), mask=there)
    tl.store(dv_ptr + offsets, dv.to(dv_ptr.dtype.element_ty), mask=there)


@triton.jit
def _queries_tile(q, do, dq, lse, delta, k_ptr, v_ptr, b, n0, N, p, qk_scale, window, stride_kb, stride_kn, stride_vb,
                  stride_vn, C: tl.constexpr, CP: tl.constexpr, KEYS: tl.constexpr, MASKED: tl.constexpr,
                  WINDOWED: tl.constexpr):
    """One tile of keys' share of a tile of queries' dq, without the scale.
    WINDOWED: each query sees only its last `window` rows."""
    n = n0 + tl.arange(0, KEYS)
    k = _load_keys(k_ptr, b, n, N, stride_kb, stride_kn, C, CP)
    v = _load_keys(v_ptr, b, n, N, stride_vb, stride_vn, C, CP)
    w = tl.exp2(tl.dot(q, tl.trans(k)) * qk_scale - lse[:, None])
    if MASKED:
        w = tl.where((n[None, :] <= p[:, None]) & (n[None, :] < N), w, 0.0)
    if WINDOWED:
        w = tl.where(p[:, None] - n[None, :] < window, w, 0.0)
    ds = w * (tl.dot(do, tl.trans(v)) - delta[:, None])
    return dq + tl.dot(ds.to(k.dtype), k)


@triton.autotune(configs=_BACKWARD_QUERIES, key=["H", "C", "size"], prune_configs_by=_PRUNE)
@triton.jit
def _exact_queries_backward_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, pos_ptr, dq_ptr, qk_scale, scale,
                                   T, N, stride_kb, stride_kn, stride_vb, stride_vn, size, H: tl.constexpr,
                                   HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, ROWS: tl.constexpr,
                                   KEYS: tl.constexpr):
    QT: tl.constexpr = ROWS // HP
    b = tl.program_id(1).to(tl.int64)
    t, h, valid, p = _query_rows(pos_ptr, b, tl.program_id(0) * QT, T, T, H, HP, ROWS)
    q = _load_rows(q_ptr, b, t, h, valid, T, H, C, CP)
    do = _load_rows(do_ptr, b, t, h, valid, T, H, C, CP)
    rows = (b * T + t) * H + h
    lse = tl.load(lse_ptr + rows, mask=valid, other=0.0) * LOG2E
    delta = tl.load(delta_ptr + rows, mask=valid, other=0.0)
    end = tl.minimum(tl.max(tl.where(valid, p, 0)) + 1, N)
    full = tl.minimum(tl.min(tl.where(valid, p, N)) + 1, N) // KEYS * KEYS
    dq = tl.zeros([ROWS, CP], tl.float32)
    for n0 in range(0, full, KEYS):
        dq = _queries_tile(q, do, dq, lse, delta, k_ptr, v_ptr, b, n0, N, p, qk_scale, 0, stride_kb, stride_kn,
                           stride_vb, stride_vn, C, CP, KEYS, False, False)
    for n0 in range(full, end, KEYS):
        dq = _queries_tile(q, do, dq, lse, delta, k_ptr, v_ptr, b, n0, N, p, qk_scale, 0, stride_kb, stride_kn,
                           stride_vb, stride_vn, C, CP, KEYS, True, False)
    d = tl.arange(0, CP)
    tl.store(dq_ptr + rows[:, None] * C + d[None, :], (dq * scale).to(dq_ptr.dtype.element_ty),
             mask=valid[:, None] & (d[None, :] < C))


# -- sparse attention --

def sparse_attention(q, k, v, sink, idx, positions, scale, window):
    """pytorch_kernels.sparse_attention's result."""
    return _Sparse.apply(q, k, v, sink, idx, positions, scale, window)


class _Sparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink, idx, positions, scale, window):
        q, k, v, idx, positions = q.contiguous(), _table(k), _table(v), idx.contiguous(), positions.contiguous()
        B, T, H, c = q.shape
        out = torch.empty_like(q)
        lse = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
        _sparse_forward_kernel[(B * T,)](q, k, v, sink.contiguous(), idx, positions, out, lse, scale * LOG2E.value, T,
                                         k.shape[1], idx.shape[-1], window, k.stride(0), k.stride(1), v.stride(0),
                                         v.stride(1), **_heads(H, c, 8))
        ctx.save_for_backward(q, k, v, sink, idx, positions, out, lse)
        ctx.scale, ctx.window = scale, window
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, sink, idx, positions, out, lse = ctx.saved_tensors
        B, T, H, c = q.shape
        N = k.shape[1]
        d_out = d_out.contiguous()
        delta = (d_out.float() * out.float()).sum(-1)  # (B, T, H): sum_j p_j dp_j
        reach = positions.cummax(1).values.contiguous()  # (B, T): the farthest row any query up to t sees
        floor = positions.flip(1).cummin(1).values.flip(1).contiguous()  # (B, T): the nearest any from t on sees
        near_dq = torch.empty(B, T, H, c, dtype=torch.float32, device=q.device)
        near_dk = torch.empty(B, N, c, dtype=torch.float32, device=q.device)
        near_dv = torch.empty(B, N, c, dtype=torch.float32, device=q.device)
        qk_scale, shape = ctx.scale * LOG2E.value, _heads(H, c)
        strides = (k.stride(0), k.stride(1), v.stride(0), v.stride(1))
        _window_keys_backward_kernel[lambda meta: (triton.cdiv(N, meta["KEYS"]), B)](
            q, k, v, d_out, lse, delta, positions, reach, floor, near_dk, near_dv, qk_scale, ctx.scale, T, N,
            ctx.window, *strides, **shape)
        _window_queries_backward_kernel[lambda meta: (triton.cdiv(T, meta["ROWS"] // shape["HP"]), B)](
            q, k, v, d_out, lse, delta, positions, near_dq, qk_scale, T, N, ctx.window, *strides, **shape)
        dq = torch.empty_like(q)
        scatter = H > GATHER_HEADS  # near_dk and near_dv then receive the chosen rows' shares too
        _sparse_queries_backward_kernel[(B * T,)](q, k, v, d_out, idx, lse, delta, near_dq, dq, near_dk, near_dv,
                                                  qk_scale, ctx.scale, T, N, idx.shape[-1], *strides,
                                                  SCATTER=scatter, **_heads(H, c, 16 if scatter else 8))
        if scatter:
            dk, dv = near_dk.to(k.dtype), near_dv.to(v.dtype)
        else:
            starts, choosers = inverse(idx, N, lambda: _choosers(idx, N))
            dk = torch.empty(B, N, c, dtype=k.dtype, device=q.device)
            dv = torch.empty(B, N, c, dtype=v.dtype, device=q.device)
            _sparse_keys_backward_kernel[(B * N,)](q, k, v, d_out, lse, delta, starts, choosers, near_dk, near_dv,
                                                   dk, dv, qk_scale, ctx.scale, N, *strides, **shape)
        d_sink = -((sink.float() - lse).exp() * delta).sum((0, 1))  # the sink carries no value
        return dq, dk, dv, d_sink.to(sink.dtype), None, None, None, None


def _choosers(idx, N):
    """The inverse of the chosen rows idx (B, T, K) of a table of N rows:
    (starts (B N + 1,) int64, choosers int32), where row n of batch row b
    was chosen by the queries choosers[starts[b N + n]:starts[b N + n + 1]],
    each b T + t, in no particular order."""
    B, T, K = idx.shape
    starts = torch.zeros(B * N + 1, dtype=torch.int64, device=idx.device)
    choosers = torch.empty(max(1, B * T * K), dtype=torch.int32, device=idx.device)
    if K:
        counts = torch.zeros(B * N, dtype=torch.int32, device=idx.device)
        _choosers_kernel[(B * T,)](idx, counts, choosers, T, N, K, KP=triton.next_power_of_2(K), PLACE=False)
        starts[1:] = counts.cumsum(0)
        _choosers_kernel[(B * T,)](idx, starts[:-1].clone(), choosers, T, N, K, KP=triton.next_power_of_2(K),
                                   PLACE=True)
    return starts, choosers


@triton.jit
def _choosers_kernel(idx_ptr, slots_ptr, choosers_ptr, T, N, K, KP: tl.constexpr, PLACE: tl.constexpr):
    """One query's chosen rows: counts it in each row's slot, or, PLACE,
    writes it at each row's slot, the next free place in that row's list."""
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, KP)
    n = tl.load(idx_ptr + row * K + j, mask=j < K, other=-1)
    there = n >= 0
    at = tl.atomic_add(slots_ptr + row // T * N + n, 1, mask=there, sem="relaxed")
    if PLACE:
        tl.store(choosers_ptr + at, row.to(tl.int32), mask=there)


@triton.jit
def _sparse_rows(idx_ptr, row, p, j0, N, K, window, BLOCK: tl.constexpr):
    """Rows j0 .. j0 + BLOCK of a query's list, its K chosen rows then its
    last window: (each row, or 0 where none; whether it is there)."""
    j = j0 + tl.arange(0, BLOCK)
    chosen = tl.load(idx_ptr + row * K + j, mask=j < K, other=-1)
    n = tl.where(j < K, chosen, p - window + 1 + (j - K))
    there = (j < K + window) & (n >= 0) & (n < N)
    return tl.where(there, n, 0), there


@triton.jit
def _query_columns(ptr, row, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr):
    """A query's (CP, HP) values, its heads as columns, 0 for padded heads
    and dimensions."""
    heads, d = tl.arange(0, HP), tl.arange(0, CP)
    return tl.load(ptr + (row * H + heads)[None, :] * C + d[:, None], mask=(heads[None, :] < H) & (d[:, None] < C),
                   other=0.0)


@triton.autotune(configs=_SPARSE, key=["H", "C", "K"])
@triton.jit
def _sparse_forward_kernel(q_ptr, k_ptr, v_ptr, sink_ptr, idx_ptr, pos_ptr, out_ptr, lse_ptr, qk_scale, T, N, K,
                           window, stride_kb, stride_kn, stride_vb, stride_vn, H: tl.constexpr, HP: tl.constexpr,
                           C: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    """One query: its tiles are (rows, heads) and its sums (CP, HP), the
    heads as the matrix products' last dimension, where the matrix units'
    smallest tile is narrowest."""
    row = tl.program_id(0).to(tl.int64)
    b = row // T
    p = tl.load(pos_ptr + row)
    q = _query_columns(q_ptr, row, H, HP, C, CP)
    m = tl.full([HP], float("-inf"), tl.float32)
    l = tl.zeros([HP], tl.float32)
    acc = tl.zeros([CP, HP], tl.float32)
    for j0 in range(0, K + window, BLOCK):
        n, there = _sparse_rows(idx_ptr, row, p, j0, N, K, window, BLOCK)
        keys = _load_keys(k_ptr, b, tl.where(there, n, N), N, stride_kb, stride_kn, C, CP)
        z = tl.where(there[:, None], tl.dot(keys, q) * qk_scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(z, 0))
        safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - safe)
        w = tl.exp2(z - safe[None, :])
        values = _load_keys(v_ptr, b, tl.where(there, n, N), N, stride_vb, stride_vn, C, CP)
        acc = acc * alpha[None, :] + tl.dot(tl.trans(values), w.to(values.dtype))
        l = l * alpha + tl.sum(w, 0)
        m = m_new
    heads, d = tl.arange(0, HP), tl.arange(0, CP)
    sink = tl.load(sink_ptr + heads, mask=heads < H, other=0.0) * LOG2E
    top = tl.maximum(m, sink)
    kept = tl.exp2(m - top)
    total = l * kept + tl.exp2(sink - top)  # the sink joins the denominator
    out = acc * (kept / total)[None, :]
    tl.store(out_ptr + (row * H + heads)[None, :] * C + d[:, None], out.to(out_ptr.dtype.element_ty),
             mask=(heads[None, :] < H) & (d[:, None] < C))
    tl.store(lse_ptr + row * H + heads, (top + tl.log2(total)) * LN2, mask=heads < H)


@triton.autotune(configs=_SPARSE_BACKWARD, key=["H", "C", "K"], restore_value=["dk_ptr", "dv_ptr"])
@triton.jit
def _sparse_queries_backward_kernel(q_ptr, k_ptr, v_ptr, do_ptr, idx_ptr, lse_ptr, delta_ptr, near_dq_ptr, dq_ptr,
                                    dk_ptr, dv_ptr, qk_scale, scale, T, N, K, stride_kb, stride_kn, stride_vb,
                                    stride_vn, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr,
                                    BLOCK: tl.constexpr, SCATTER: tl.constexpr):
    """One query's dq through its chosen rows, the window rows' share
    near_dq added; tiles (rows, heads) and sums (CP, HP), as the forward
    pass. SCATTER: also adds its share of each chosen row's dk and dv
    (B, N, C) in FP32, both rows of a (rows, heads) x (heads, CP) product,
    by atomic additions; HP is then at least 16, a product's middle
    dimension."""
    row = tl.program_id(0).to(tl.int64)
    b = row // T
    q = _query_columns(q_ptr, row, H, HP, C, CP)
    do = _query_columns(do_ptr, row, H, HP, C, CP)
    heads, d = tl.arange(0, HP), tl.arange(0, CP)
    lse = tl.load(lse_ptr + row * H + heads, mask=heads < H, other=0.0) * LOG2E
    delta = tl.load(delta_ptr + row * H + heads, mask=heads < H, other=0.0)
    rows = (row * H + heads)[None, :] * C + d[:, None]
    there_q = (heads[None, :] < H) & (d[:, None] < C)
    dq = tl.load(near_dq_ptr + rows, mask=there_q, other=0.0)
    n, there = _sparse_rows(idx_ptr, row, 0, 0, N, K, 0, BLOCK)
    keys = _load_keys(k_ptr, b, tl.where(there, n, N), N, stride_kb, stride_kn, C, CP)
    values = _load_keys(v_ptr, b, tl.where(there, n, N), N, stride_vb, stride_vn, C, CP)
    for j0 in range(0, K, BLOCK):
        n2, there2 = _sparse_rows(idx_ptr, row, 0, j0 + BLOCK, N, K, 0, BLOCK)
        keys2 = _load_keys(k_ptr, b, tl.where(there2, n2, N), N, stride_kb, stride_kn, C, CP)
        values2 = _load_keys(v_ptr, b, tl.where(there2, n2, N), N, stride_vb, stride_vn, C, CP)
        w = tl.where(there[:, None], tl.exp2(tl.dot(keys, q) * qk_scale - lse[None, :]), 0.0)
        ds = w * (tl.dot(values, do) - delta[None, :])
        dq += tl.dot(tl.trans(keys), ds.to(keys.dtype))
        if SCATTER:
            out = (b * N + n)[:, None] * C + d[None, :]
            there_out = there[:, None] & (d[None, :] < C)
            tl.atomic_add(dk_ptr + out, tl.dot((ds * scale).to(q.dtype), tl.trans(q)), mask=there_out, sem="relaxed")
            tl.atomic_add(dv_ptr + out, tl.dot(w.to(do.dtype), tl.trans(do)), mask=there_out, sem="relaxed")
        keys, values, there, n = keys2, values2, there2, n2
    tl.store(dq_ptr + rows, (dq * scale).to(dq_ptr.dtype.element_ty), mask=there_q)


@triton.autotune(configs=_SPARSE_KEYS_BACKWARD, key=["H", "C"])
@triton.jit
def _sparse_keys_backward_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, starts_ptr, choosers_ptr,
                                 near_dk_ptr, near_dv_ptr, dk_ptr, dv_ptr, qk_scale, scale, N, stride_kb, stride_kn,
                                 stride_vb, stride_vn, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr,
                                 CP: tl.constexpr, E: tl.constexpr):
    """One table row's dk and dv: the window rows' share near_dk and
    near_dv plus every query that chose it, E queries' heads at a time on
    the vector units (one row against a few queries fills no matrix
    tile), written once in the table's dtype."""
    key = tl.program_id(0).to(tl.int64)
    b = key // N
    n = key % N
    d = tl.arange(0, CP)
    k = tl.load(k_ptr + b * stride_kb + n * stride_kn + d, mask=d < C, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + b * stride_vb + n * stride_vn + d, mask=d < C, other=0.0).to(tl.float32)
    start = tl.load(starts_ptr + key)
    end = tl.load(starts_ptr + key + 1)
    r = tl.arange(0, E * HP)
    e, h = r // HP, r % HP
    dk = tl.zeros([CP], tl.float32)
    dv = tl.zeros([CP], tl.float32)
    for i0 in range(start, end, E):
        i = i0 + e
        there = (i < end) & (h < H)
        rows = tl.load(choosers_ptr + i, mask=i < end, other=0).to(tl.int64) * H + h  # (query, head) rows
        mask = there[:, None] & (d[None, :] < C)
        q = tl.load(q_ptr + rows[:, None] * C + d[None, :], mask=mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + rows[:, None] * C + d[None, :], mask=mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + rows, mask=there, other=0.0) * LOG2E
        delta = tl.load(delta_ptr + rows, mask=there, other=0.0)
        w = tl.where(there, tl.exp2(tl.sum(q * k[None, :], 1) * qk_scale - lse), 0.0)
        ds = w * (tl.sum(do * v[None, :], 1) - delta)
        dk += tl.sum(ds[:, None] * q, 0)
        dv += tl.sum(w[:, None] * do, 0)
    out = key * C + d
    dk = dk * scale + tl.load(near_dk_ptr + out, mask=d < C, other=0.0)
    dv += tl.load(near_dv_ptr + out, mask=d < C, other=0.0)
    tl.store(dk_ptr + out, dk.to(dk_ptr.dtype.element_ty), mask=d < C)
    tl.store(dv_ptr + out, dv.to(dv_ptr.dtype.element_ty), mask=d < C)


@triton.autotune(configs=_BACKWARD_KEYS, key=["H", "C"], prune_configs_by=_PRUNE)
@triton.jit
def _window_keys_backward_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, pos_ptr, reach_ptr, floor_ptr, dk_ptr,
                                 dv_ptr, qk_scale, scale, T, N, window, stride_kb, stride_kn, stride_vb, stride_vn,
                                 H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr,
                                 ROWS: tl.constexpr, KEYS: tl.constexpr):
    """The window rows' share of a sparse layer's dk and dv, for one block
    of KEYS rows, written in FP32 without atomic additions: the queries from
    the first whose reach (the largest position up to it) gets to the block
    to the first whose floor (the smallest position from it on) is past the
    block's last row plus the window."""
    QT: tl.constexpr = ROWS // HP
    b = tl.program_id(1).to(tl.int64)
    n = tl.program_id(0) * KEYS + tl.arange(0, KEYS)
    k = _load_keys(k_ptr, b, n, N, stride_kb, stride_kn, C, CP)
    v = _load_keys(v_ptr, b, n, N, stride_vb, stride_vn, C, CP)
    dk = tl.zeros([KEYS, CP], tl.float32)
    dv = tl.zeros([KEYS, CP], tl.float32)
    first = _first_at_least(reach_ptr + b * T, T, tl.program_id(0) * KEYS)
    stop = _first_at_least(floor_ptr + b * T, T, tl.program_id(0) * KEYS + KEYS - 1 + window)
    for t0 in range(first // QT * QT, stop, QT):
        dk, dv = _keys_tile(dk, dv, k, v, n, q_ptr, do_ptr, lse_ptr, delta_ptr, pos_ptr, b, t0, T, N, qk_scale, window,
                            H, HP, C, CP, ROWS, True)
    d = tl.arange(0, CP)
    there = (n[:, None] < N) & (d[None, :] < C)
    offsets = (b * N + n)[:, None] * C + d[None, :]
    tl.store(dk_ptr + offsets, dk * scale, mask=there)
    tl.store(dv_ptr + offsets, dv, mask=there)


@triton.autotune(configs=_BACKWARD_QUERIES, key=["H", "C"], prune_configs_by=_PRUNE)
@triton.jit
def _window_queries_backward_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, pos_ptr, dq_ptr, qk_scale, T, N,
                                    window, stride_kb, stride_kn, stride_vb, stride_vn, H: tl.constexpr,
                                    HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, ROWS: tl.constexpr,
                                    KEYS: tl.constexpr):
    """The window rows' share of a sparse layer's dq, in FP32 and without
    the scale, for one tile of queries: the rows from the earliest query's
    window to the latest query."""
    QT: tl.constexpr = ROWS // HP
    b = tl.program_id(1).to(tl.int64)
    t, h, valid, p = _query_rows(pos_ptr, b, tl.program_id(0) * QT, T, T, H, HP, ROWS)
    q = _load_rows(q_ptr, b, t, h, valid, T, H, C, CP)
    do = _load_rows(do_ptr, b, t, h, valid, T, H, C, CP)
    rows = (b * T + t) * H + h
    lse = tl.load(lse_ptr + rows, mask=valid, other=0.0) * LOG2E
    delta = tl.load(delta_ptr + rows, mask=valid, other=0.0)
    start = tl.maximum(tl.min(tl.where(valid, p, N)) - window + 1, 0) // KEYS * KEYS
    end = tl.minimum(tl.max(tl.where(valid, p, 0)) + 1, N)
    dq = tl.zeros([ROWS, CP], tl.float32)
    for n0 in range(start, end, KEYS):
        dq = _queries_tile(q, do, dq, lse, delta, k_ptr, v_ptr, b, n0, N, p, qk_scale, window, stride_kb, stride_kn,
                           stride_vb, stride_vn, C, CP, KEYS, True, True)
    d = tl.arange(0, CP)
    tl.store(dq_ptr + rows[:, None] * C + d[None, :], dq, mask=valid[:, None] & (d[None, :] < C))
