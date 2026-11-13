"""The attention kernels for TileLang, on an NVIDIA GPU under Linux: the two
operations of kernels.py, with the contract and the results of
pytorch_kernels.py. tests/test_kernels.py compares them with it. kernels.py
imports this module only once it has decided that TileLang is used.

The attention kernels compute in BF16 on the matrix units, with FP32 sums
and softmax, and never hold more than one tile of scores:

    exact_forward_kernel   the exact layer's attention: flash attention over
                           every row a query sees. The query heads share one
                           key-value head, so a tile's rows are (token, head)
                           pairs, M of them, and a few heads still fill the
                           matrix units. A query sees the rows up to its
                           own, so the blocks of the last rows, the
                           costliest, run first, which keeps the GPU from
                           waiting on them at the end
    exact_select_kernel    each query's top rows, in streaming passes over
                           the same scores (below), the last tokens first
                           for the same reason
    rank_kernel            the last of them, among a few rows per query
    exact_backward_kernel  the exact layer's gradients: one program per tile
                           of keys, its key and value gradients held until
                           the end, the query gradients added up by atomic
                           additions in FP32
    sparse_forward_kernel  the sparse layer's attention: one program per
                           query, its chosen rows and its window gathered
                           into shared memory in blocks by asynchronous
                           copies; its tiles are (rows, heads), heads
                           padded to 8 columns, the matrix units' narrowest
                           tile, and one warp per 16 rows of a block
    sparse_queries_backward_kernel
                           its query gradients: one program per query, as
                           the forward pass reads its rows, written once;
                           its tiles are (rows, heads), heads padded to 8
                           columns, the matrix units' narrowest tile, and
                           its blocks of rows are gathered by asynchronous
                           copies; with more than GATHER_HEADS heads, it
                           also adds its share of its rows' key and value
                           gradients by atomic additions in FP32
    choosers_kernel        the inverse of the rows each query reads (its
                           chosen rows and its window): for each table row,
                           the list of queries that read it, from a count per
                           row, its running sum, then each query written
                           into its rows' lists; made once for every sparse
                           layer that reads the same choice
                           (pytorch_kernels.inverse)
    sparse_keys_backward_kernel
                           its key and value gradients with GATHER_HEADS
                           heads or fewer: one program per table row, over
                           its list of queries, written once

The rows' gradients in the sparse backward pass take one of two paths, by
the number of heads, as in the Triton form (triton_kernels.py), whose
docstring gives the measured costs. With more heads than GATHER_HEADS, the
queries' kernel adds each query's share of its rows' gradients in FP32
by atomic additions, and their order, with it the rounding, may differ
from run to run. With GATHER_HEADS heads or fewer, the choosers' lists are
filled by atomic additions on their counters, so the order of the queries
in a list, and with it the rounding of a row's gradient, may differ from
run to run. The keys' kernel gathers each listed query's q and d_out
and computes its weights again, all on the vector units, since one row
against a few queries fills no matrix tile: each (query, head) row's
products add up in its own row of a sum the size of a step, reduced to
one row at the end, so a step waits on no reduction but its two dot
products, and the next step's rows are loaded while the current one
computes. Its rows per step (KEYS_ROWS) and threads were chosen among a
few measured on an RTX 5080 laptop GPU at 4K and 32K tokens; an H100 runs
the same, open to review there.

Top rows. A row's score is its largest log-weight over the heads,
s = max_h (z_h - lse_h), and its key the bits of s as an int32 in the same
order. The selection is the Triton form's (triton_kernels.py): passes that
count each query's keys in BINS bins until its top_k-th key lies in a bin
of at most CANDIDATES rows or of one key, the first pass's bins about SPAN
bits wide below log2(heads / top_k) or the best score u = max_h (m_h -
lse_h) from the forward pass's running maximum m; then a pass that writes
the rows above the bin and lists the bin's rows, and rank_kernel, which
keeps the best of those by their rank, a higher score first, then an
earlier row. Each pass recomputes the scores: memory holds one tile of
them and a histogram per query, which the pass fills by atomic additions
in shared memory. So the rows chosen differ from an exact top_k of the
same scores only among equal scores; when more than CANDIDATES rows of one
key are left, the ones kept depend on the order in which the GPU's threads
reach them. The chosen rows come back sorted, so the sparse layer reads
them in the order of the table.

The tiles follow the GPU's shared memory: TILES_LARGE where a block may take
at least 200 KiB (H100, B200), TILES_SMALL elsewhere (RTX 50 series, A100).
The tile sizes are a first choice that no measurement settled yet, open to
review.

Inputs: q (B, T, H, c), and k and v (B, N, c), which the kernels read as one
table (B, N, 2c) with the keys first, as the exact layer hands them on
(attention.py); separate k and v are put side by side first. H is at most
64, and c a multiple of 16 of at most 256.
"""

import functools
import math
from dataclasses import dataclass

import torch
import tilelang
import tilelang.language as T
import tvm

from .pytorch_kernels import chosen_columns, inverse

NAME = "TileLang"
BF16, FP32, INT32, INT64 = "bfloat16", "float32", "int32", "int64"
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
BINS = 64  # bins per counting pass of the selection
LOG_BINS = 6
SPAN = 8.0  # the first counting pass's bins span about this many bits (base-2 log-weights) below its start
CANDIDATES = 512  # the most rows a query's last bin may hold to be ordered exactly between them
LEVELS = 16  # the most counting passes
NONE = 1 - (1 << 31)  # the key of a row a query may not choose, below every score's (_key)
ALL = 1 << 62  # above every score's key
GATHER_HEADS = 2  # the most heads for which each table row gathers its queries; above, the queries scatter
KEYS_ROWS = 8  # (query, head) rows per step of the sparse keys' backward kernel
KEYS_THREADS = 64
# Kernels whose warps are not specialized and that copy without the Tensor Memory Accelerator: the sparse kernels,
# which read their rows by index, the selection, which TileLang 0.1.8 fails to build with either one, and the
# backward pass, whose registers spill on an H100 once its warps are specialized
PLAIN = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True}


@dataclass(frozen=True)
class Tiles:
    """One GPU class's tile sizes: rows of (token, head) pairs and keys per
    tile of each exact kernel, the most tokens per block of the selection,
    rows per block of the sparse kernels, pipeline stages, threads, the
    threads of the sparse queries' backward kernel: one warp per 16 rows
    of its block, since the rows are its matrix products' first dimension
    and the heads, 8 columns or more, cannot be split between warps; and
    the rows per block of the sparse forward kernel, which takes one warp
    per 16 of them for the same reason.

    The sparse forward kernel, bound by its gathers, ran fastest on an
    RTX 5080 laptop GPU with blocks of 16 rows and one warp, the most
    programs at once on each multiprocessor. TILES_LARGE takes blocks of
    32 and two warps, for an H100's more registers per multiprocessor and
    its higher limit of warps; that choice was not timed (open to review).

    The selection's shared memory grows with its tokens per block (a
    histogram and a tile of keys per token), and a block of few heads
    holds many tokens: select_tokens keeps it within the GPU's shared
    memory, 99 KiB per block on the RTX 50 series."""
    forward_rows: int
    forward_keys: int
    select_rows: int
    select_keys: int
    select_tokens: int
    backward_rows: int
    backward_keys: int
    sparse_block: int
    stages: int
    threads: int
    sparse_threads: int
    sparse_forward_block: int


TILES_LARGE = Tiles(128, 128, 128, 64, 64, 64, 64, 64, 2, 256, 128, 32)
TILES_SMALL = Tiles(64, 64, 64, 64, 32, 64, 32, 32, 2, 128, 64, 16)


def tiles(device):
    """The Tiles for device's GPU."""
    return _tiles(torch.cuda.current_device() if device.index is None else device.index)


@functools.cache
def _tiles(index):
    properties = torch.cuda.get_device_properties(index)
    return TILES_LARGE if properties.shared_memory_per_block_optin >= 200 * 1024 else TILES_SMALL


def _padded_heads(H, size):
    """Query heads padded to a multiple of size: 16 as a matrix product's
    rows, 8 as its columns (the matrix units' narrowest tile)."""
    return max(size, -(-H // size) * size)


def _tokens(chosen_tiles, H):
    """Tokens per block of the selection, a power of 2 of at most
    select_tokens: the rows of its select_rows past them times H are
    unused."""
    return min(1 << (chosen_tiles.select_rows // H).bit_length() - 1, chosen_tiles.select_tokens)


def _head_max(buffer, i, j, H):
    """max over the H heads of token i's rows of buffer, at column j."""
    value = buffer[i * H, j]
    for h in range(1, H):
        value = T.max(value, buffer[i * H + h, j])
    return value


@tilelang.jit
def exact_forward_kernel(H: int, c: int, scale: float, M: int, BN: int, stages: int, threads: int):
    """o (B, R, c), lse and the running maximum mx (B, R), in natural logs,
    of q (B, R = T H, c) over the rows of kv each query sees: rows up to its
    position, below N. ends (B, blocks): the rows any query of a block
    sees."""
    b, r, t, n, blocks = T.symbolic("b"), T.symbolic("r"), T.symbolic("t"), T.symbolic("n"), T.symbolic("blocks")
    scale2 = scale * LOG2E

    @T.prim_func
    def kernel(q: T.Tensor((b, r, c), BF16), kv: T.Tensor((b, n, 2 * c), BF16), pos: T.Tensor((b, t), INT32),
               ends: T.Tensor((b, blocks), INT32), o: T.Tensor((b, r, c), BF16), lse: T.Tensor((b, r), FP32),
               mx: T.Tensor((b, r), FP32)):
        with T.Kernel(T.ceildiv(r, M), b, threads=threads) as (block, by):
            # the last rows, which see the most keys, first; counted from ends' blocks, since the same order from
            # T.ceildiv(r, M) hung on the GPU when one compiled kernel ran again at another size
            bx = blocks - 1 - block
            q_s = T.alloc_shared((M, c), BF16)
            k_s = T.alloc_shared((BN, c), BF16)
            v_s = T.alloc_shared((BN, c), BF16)
            s = T.alloc_fragment((M, BN), FP32)
            p = T.alloc_fragment((M, BN), BF16)
            acc = T.alloc_fragment((M, c), FP32)
            m = T.alloc_fragment((M,), FP32)
            m_prev = T.alloc_fragment((M,), FP32)
            alpha = T.alloc_fragment((M,), FP32)
            total = T.alloc_fragment((M,), FP32)
            l = T.alloc_fragment((M,), FP32)
            row_pos = T.alloc_fragment((M,), INT32)

            T.copy(q[by, bx * M:(bx + 1) * M, :], q_s)
            for i in T.Parallel(M):
                row_pos[i] = pos[by, T.min((bx * M + i) // H, t - 1)]
            T.fill(acc, 0)
            T.fill(l, 0)
            T.fill(m, -T.infinity(FP32))
            for k in T.Pipelined(T.ceildiv(ends[by, bx], BN), num_stages=stages):
                T.copy(kv[by, k * BN:(k + 1) * BN, 0:c], k_s)
                for i, j in T.Parallel(M, BN):  # row 0 is seen by every query: m is finite after the first tile
                    s[i, j] = T.if_then_else((k * BN + j <= row_pos[i]) & (k * BN + j < n), 0, -T.infinity(FP32))
                T.gemm(q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m, m_prev)
                T.reduce_max(s, m, dim=1, clear=False)
                for i in T.Parallel(M):
                    alpha[i] = T.exp2((m_prev[i] - m[i]) * scale2)
                for i, j in T.Parallel(M, BN):
                    s[i, j] = T.exp2(s[i, j] * scale2 - m[i] * scale2)
                T.reduce_sum(s, total, dim=1)
                for i in T.Parallel(M):
                    l[i] = l[i] * alpha[i] + total[i]
                T.copy(s, p)
                for i, j in T.Parallel(M, c):
                    acc[i, j] *= alpha[i]
                T.copy(kv[by, k * BN:(k + 1) * BN, c:2 * c], v_s)
                T.gemm(p, v_s, acc, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(M, c):
                acc[i, j] /= l[i]
            T.copy(acc, o[by, bx * M:(bx + 1) * M, :])
            for i in T.Parallel(M):
                if bx * M + i < r:
                    lse[by, bx * M + i] = m[i] * scale + T.log2(l[i]) * LN2
                    mx[by, bx * M + i] = m[i] * scale

    return kernel


def _long(value):
    """value as an int64 constant of a kernel."""
    return tvm.tir.const(value, INT64)


def _key(score):
    """An FP32 score's bits as an int32 in the same order: a higher score, a
    larger key; equal keys, equal scores. Every score's key is above NONE."""
    bits = T.reinterpret(INT32, score)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


@tilelang.jit(pass_configs={**PLAIN, tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,  # slots by atomics
                             tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})  # fails on an atomic's index
def exact_select_kernel(H: int, c: int, scale: float, window: int, M: int, BT: int, BN: int, stages: int,
                        threads: int, cap: int, log_width: int):
    """Each query's top rows into idx (B, T, K), prefilled with -1 (the
    module's docstring, "Top rows"): the rows above its bin and, when the bin
    holds more than cap rows, as many of the bin's as still wanted; else the
    bin's rows go into bins (B, T, cap) as (key, ~row) for rank_kernel, and
    state (B, T, 3) gets the rows above the bin, the bin's rows and the rows
    wanted. lse and mx (B, R) are the forward pass's; ends (B, blocks): the
    candidate rows of a block's queries. BT tokens per block, BT H <= M."""
    b, r, t, n = T.symbolic("b"), T.symbolic("r"), T.symbolic("t"), T.symbolic("n")
    kc, blocks = T.symbolic("kc"), T.symbolic("blocks")
    scale2 = scale * LOG2E

    @T.macro
    def keys(kv, q_s, k_s, s, z_s, key_s, row_lse, last, by, k):
        """key_s: the keys of the rows k BN .. (k + 1) BN, NONE where a query
        may not choose the row."""
        T.copy(kv[by, k * BN:(k + 1) * BN, 0:c], k_s)
        T.clear(s)
        T.gemm(q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
        for i, j in T.Parallel(M, BN):
            z_s[i, j] = s[i, j] * scale2 - row_lse[i]
        T.sync_threads()  # other threads read z_s: TileLang 0.1.8 puts no barrier here in the counting pass
        for i, j in T.Parallel(BT, BN):
            key_s[i, j] = T.if_then_else(k * BN + j <= last[i], _key(_head_max(z_s, i, j, H)),
                                         tvm.tir.const(NONE, INT32))

    @T.prim_func
    def kernel(q: T.Tensor((b, r, c), BF16), kv: T.Tensor((b, n, 2 * c), BF16), pos: T.Tensor((b, t), INT32),
               lse: T.Tensor((b, r), FP32), mx: T.Tensor((b, r), FP32), ends: T.Tensor((b, blocks), INT32),
               idx: T.Tensor((b, t, kc), INT64), bins: T.Tensor((b, t, cap), INT64), state: T.Tensor((b, t, 3), INT32)):
        with T.Kernel(T.ceildiv(t, BT), b, threads=threads) as (block, by):
            bx = blocks - 1 - block  # the last tokens, which have the most candidates, first (see exact_forward_kernel)
            q_s = T.alloc_shared((M, c), BF16)
            k_s = T.alloc_shared((BN, c), BF16)
            z_s = T.alloc_shared((M, BN), FP32)  # each (token, head) row's log-weights, in base 2
            key_s = T.alloc_shared((BT, BN), INT32)  # each token's keys
            hist = T.alloc_shared((BT, BINS), INT32)
            wanted = T.alloc_shared((BT,), INT32)  # rows to choose
            last = T.alloc_shared((BT,), INT32)  # the last row that is a candidate
            open_ = T.alloc_shared((BT,), INT32)  # still counting
            top = T.alloc_shared((BT,), INT64)  # bin j: top - key in [j, j + 1) 2^shift
            shift = T.alloc_shared((BT,), INT64)
            hi = T.alloc_shared((BT,), INT64)  # rows with a key above hi are chosen
            lo = T.alloc_shared((BT,), INT64)  # the bin: keys in [lo, hi]
            above = T.alloc_shared((BT,), INT32)  # rows with a key above hi
            in_bin = T.alloc_shared((BT,), INT32)
            sure_n = T.alloc_shared((BT,), INT32)  # slots taken, in the writing pass
            bin_n = T.alloc_shared((BT,), INT32)
            s = T.alloc_fragment((M, BN), FP32)
            row_lse = T.alloc_fragment((M,), FP32)
            run = T.alloc_local((1,), INT32)
            first = T.alloc_local((1,), INT32)
            before = T.alloc_local((1,), INT32)
            reached = T.alloc_local((1,), INT32)
            start = T.alloc_local((1,), FP32)
            flag = T.alloc_local((1,), INT32)  # queries still counting

            for i, j in T.Parallel(M, c):
                q_s[i, j] = T.if_then_else((i < BT * H) & (bx * BT * H + i < r), q[by, bx * BT * H + i, j], 0)
            for i in T.Parallel(M):
                row_lse[i] = T.if_then_else((i < BT * H) & (bx * BT * H + i < r), lse[by, bx * BT * H + i] * LOG2E,
                                            T.infinity(FP32))
            tx = T.get_thread_binding()
            if tx < BT:
                token = bx * BT + tx
                if token < t:
                    last[tx] = T.min(pos[by, token] - window, n - 1)
                else:
                    last[tx] = -1
                wanted[tx] = T.min(T.max(last[tx] + 1, 0), kc)
                open_[tx] = T.if_then_else(wanted[tx] < last[tx] + 1, 1, 0)
                # the weights of the best `wanted` rows add up to at most H: the wanted-th score is at most
                # log2(H / wanted), and the first pass's bins start there, or at the best score when lower
                start[0] = T.min(T.log2(H * 1.0 / T.Cast(FP32, T.max(wanted[tx], 1))), _best(lse, mx, by, token, t, H))
                top[tx] = _key(start[0])
                shift[tx] = T.Cast(INT64, T.min(T.max(23 - T.max(((T.reinterpret(INT32, start[0]) >> 23) & 255) - 127,
                                                                  0) + log_width, 0), 40))
                hi[tx] = T.if_then_else(open_[tx] > 0, _long(ALL), _long(NONE))
                lo[tx] = hi[tx]
                above[tx] = T.if_then_else(open_[tx] > 0, 0, wanted[tx])
                in_bin[tx] = 0
            for level in T.serial(LEVELS):
                T.sync_threads()  # every thread reads the same open_, so all take the same branch
                flag[0] = 0
                for i in T.serial(BT):
                    flag[0] = flag[0] + open_[i]
                if flag[0] > 0:
                    T.clear(hist)
                    for k in T.Pipelined(T.ceildiv(ends[by, bx], BN), num_stages=stages):
                        keys(kv, q_s, k_s, s, z_s, key_s, row_lse, last, by, k)
                        for i, j in T.Parallel(BT, BN):
                            if (open_[i] > 0) & (T.Cast(INT64, key_s[i, j]) <= hi[i]):
                                bin_ = T.Cast(INT32, T.min(T.max(T.shift_right(top[i] - T.Cast(INT64, key_s[i, j]),
                                                                               shift[i]), 0), BINS))
                                if bin_ < BINS:
                                    T.atomic_add(hist[i, bin_], 1)
                    if tx < BT:
                        if open_[tx] > 0:
                            # the first bin whose rows, with those before it, reach the rows still wanted
                            run[0] = 0
                            first[0] = BINS
                            before[0] = 0
                            reached[0] = 0
                            for j in T.serial(BINS):
                                if first[0] == BINS:
                                    if run[0] + hist[tx, j] >= wanted[tx] - above[tx]:
                                        first[0] = j
                                        before[0] = run[0]
                                        reached[0] = run[0] + hist[tx, j]
                                run[0] = run[0] + hist[tx, j]
                            width = T.shift_left(T.Cast(INT64, 1), shift[tx])
                            if first[0] < BINS:
                                # into the bin, its keys split into BINS bins; settled at most cap rows, or
                                # one key (equal scores), or the last pass
                                if first[0] > 0:
                                    hi[tx] = top[tx] - T.Cast(INT64, first[0]) * width
                                lo[tx] = top[tx] - T.Cast(INT64, first[0] + 1) * width + 1
                                above[tx] = above[tx] + before[0]
                                in_bin[tx] = reached[0] - before[0]
                                if (in_bin[tx] <= cap) | (shift[tx] == 0) | (level == LEVELS - 1):
                                    open_[tx] = 0
                                top[tx] = top[tx] - T.Cast(INT64, first[0]) * width
                                shift[tx] = T.max(shift[tx] - LOG_BINS, 0)
                            else:
                                # every row counted is chosen: the next pass counts below them, 4 times wider
                                above[tx] = above[tx] + run[0]
                                hi[tx] = top[tx] - T.Cast(INT64, BINS) * width
                                top[tx] = hi[tx]
                                shift[tx] = shift[tx] + 2
                                if level == LEVELS - 1:
                                    lo[tx] = _long(NONE + 1)
                                    in_bin[tx] = cap + 1
                                    open_[tx] = 0
            if tx < BT:
                sure_n[tx] = 0
                bin_n[tx] = 0
            for k in T.Pipelined(T.ceildiv(ends[by, bx], BN), num_stages=stages):  # the rows chosen
                keys(kv, q_s, k_s, s, z_s, key_s, row_lse, last, by, k)
                for i, j in T.Parallel(BT, BN):
                    if T.Cast(INT64, key_s[i, j]) > hi[i]:
                        sure = T.atomic_add(sure_n[i], 1, return_prev=True)
                        idx[by, bx * BT + i, sure] = k * BN + j
                    elif (T.Cast(INT64, key_s[i, j]) >= lo[i]) & (key_s[i, j] != tvm.tir.const(NONE, INT32)):
                        place = T.atomic_add(bin_n[i], 1, return_prev=True)
                        if in_bin[i] > cap:  # equal scores, taken in the order met
                            if above[i] + place < wanted[i]:
                                idx[by, bx * BT + i, above[i] + place] = k * BN + j
                        else:
                            listed = T.shift_left(T.Cast(INT64, key_s[i, j]), T.Cast(INT64, 32))
                            bins[by, bx * BT + i, place] = T.bitwise_or(listed, T.Cast(INT64, (k * BN + j) ^ 0x7FFFFFFF))
            if tx < BT:
                if bx * BT + tx < t:
                    state[by, bx * BT + tx, 0] = above[tx]
                    state[by, bx * BT + tx, 1] = in_bin[tx]
                    state[by, bx * BT + tx, 2] = wanted[tx]

    return kernel


def _best(lse, mx, by, token, t, H):
    """The best score of query (by, token), in base 2: the forward pass's
    running maximum over its heads, -inf past the queries."""
    best = T.if_then_else(token < t, (mx[by, T.min(token, t - 1) * H] - lse[by, T.min(token, t - 1) * H]) * LOG2E,
                          -T.infinity(FP32))
    for h in range(1, H):
        best = T.max(best, (mx[by, T.min(token, t - 1) * H + h] - lse[by, T.min(token, t - 1) * H + h]) * LOG2E)
    return best


@tilelang.jit(pass_configs=PLAIN)
def rank_kernel(cap: int, threads: int):
    """The bin's best rows of each query whose bin holds at most cap rows,
    after exact_select_kernel: the rows still wanted, by their rank among
    the bin's (key, ~row), written after the rows above the bin."""
    b, t, kc = T.symbolic("b"), T.symbolic("t"), T.symbolic("kc")

    @T.prim_func
    def kernel(bins: T.Tensor((b, t, cap), INT64), state: T.Tensor((b, t, 3), INT32), idx: T.Tensor((b, t, kc), INT64)):
        with T.Kernel(t, b, threads=threads) as (bx, by):
            listed = T.alloc_shared((cap,), INT64)
            rank = T.alloc_fragment((cap,), INT32)
            above, count, wanted = state[by, bx, 0], state[by, bx, 1], state[by, bx, 2]
            if (count <= cap) & (wanted > above):
                for j in T.Parallel(cap):
                    listed[j] = T.if_then_else(j < count, bins[by, bx, j], 0)
                T.clear(rank)
                for i in T.serial(count):
                    for j in T.Parallel(cap):
                        rank[j] += T.if_then_else(listed[i] > listed[j], 1, 0)
                for j in T.Parallel(cap):
                    if (j < count) & (rank[j] < wanted - above):
                        idx[by, bx, above + rank[j]] = T.Cast(INT64, T.Cast(INT32, listed[j] & 0x7FFFFFFF) ^ 0x7FFFFFFF)

    return kernel


@tilelang.jit(pass_configs=PLAIN)
def exact_backward_kernel(H: int, c: int, scale: float, M: int, BN: int, stages: int, threads: int):
    """dkv (B, N, 2c), the keys' and values' gradients side by side, and
    dq (B, R, c) in FP32, zeroed before, added to; from q, kv, d_out (B, R,
    c), lse and delta (B, R), delta the sum over c of d_out times the
    output. start (B, key tiles): the first block of M rows that sees each
    tile."""
    b, r, t, n, tiles_ = T.symbolic("b"), T.symbolic("r"), T.symbolic("t"), T.symbolic("n"), T.symbolic("tiles")
    scale2 = scale * LOG2E

    @T.prim_func
    def kernel(q: T.Tensor((b, r, c), BF16), kv: T.Tensor((b, n, 2 * c), BF16), d_out: T.Tensor((b, r, c), BF16),
               lse: T.Tensor((b, r), FP32), delta: T.Tensor((b, r), FP32), pos: T.Tensor((b, t), INT32),
               start: T.Tensor((b, tiles_), INT32), dq: T.Tensor((b, r, c), FP32), dkv: T.Tensor((b, n, 2 * c), BF16)):
        with T.Kernel(T.ceildiv(n, BN), b, threads=threads) as (bx, by):
            k_s = T.alloc_shared((BN, c), BF16)
            v_s = T.alloc_shared((BN, c), BF16)
            q_s = T.alloc_shared((M, c), BF16)
            do_s = T.alloc_shared((M, c), BF16)
            p_s = T.alloc_shared((BN, M), BF16)
            ds_s = T.alloc_shared((BN, M), BF16)
            lse_s = T.alloc_shared((M,), FP32)
            delta_s = T.alloc_shared((M,), FP32)
            pos_s = T.alloc_shared((M,), INT32)
            p_t = T.alloc_fragment((BN, M), FP32)  # transposed: rows are keys
            dp_t = T.alloc_fragment((BN, M), FP32)
            dk = T.alloc_fragment((BN, c), FP32)
            dv = T.alloc_fragment((BN, c), FP32)
            dq_f = T.alloc_fragment((M, c), FP32)

            T.copy(kv[by, bx * BN:(bx + 1) * BN, 0:c], k_s)
            T.copy(kv[by, bx * BN:(bx + 1) * BN, c:2 * c], v_s)
            T.clear(dk)
            T.clear(dv)
            first = start[by, bx]
            for step in T.Pipelined(T.ceildiv(r, M) - first, num_stages=stages):
                rows = (first + step) * M
                T.copy(q[by, rows:rows + M, :], q_s)
                T.copy(d_out[by, rows:rows + M, :], do_s)
                for i in T.Parallel(M):
                    lse_s[i] = T.if_then_else(rows + i < r, lse[by, rows + i] * LOG2E, 0)
                    delta_s[i] = T.if_then_else(rows + i < r, delta[by, rows + i], 0)
                    pos_s[i] = T.if_then_else(rows + i < r, pos[by, T.min((rows + i) // H, t - 1)], -1)
                T.clear(p_t)
                T.gemm(k_s, q_s, p_t, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BN, M):
                    p_t[i, j] = T.if_then_else((bx * BN + i <= pos_s[j]) & (bx * BN + i < n),
                                               T.exp2(p_t[i, j] * scale2 - lse_s[j]), 0)
                T.clear(dp_t)
                T.gemm(v_s, do_s, dp_t, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(p_t, p_s)
                T.gemm(p_s, do_s, dv, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BN, M):
                    dp_t[i, j] = p_t[i, j] * (dp_t[i, j] - delta_s[j]) * scale
                T.copy(dp_t, ds_s)
                T.gemm(ds_s, q_s, dk, policy=T.GemmWarpPolicy.FullRow)
                T.clear(dq_f)
                T.gemm(ds_s, k_s, dq_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(M, c):
                    if rows + i < r:
                        T.atomic_add(dq[by, rows + i, j], dq_f[i, j])
            T.copy(dk, dkv[by, bx * BN:(bx + 1) * BN, 0:c])
            T.copy(dv, dkv[by, bx * BN:(bx + 1) * BN, c:2 * c])

    return kernel


def _rows_of(idx, pos, kc, n, window, BLK, blk, nb1, by, bx, j):
    """The table row that block blk's j-th entry of query (by, bx) reads, -1
    for none: its chosen rows' blocks first, then its window's."""
    chosen = T.if_then_else(blk * BLK + j < kc, T.Cast(INT32, idx[by, bx, T.min(blk * BLK + j, kc - 1)]), -1)
    near = (blk - nb1) * BLK + j
    recent = T.if_then_else(near < window, pos[by, bx] - window + 1 + near, -1)
    row = T.if_then_else(blk < nb1, chosen, recent)
    return T.if_then_else((row >= 0) & (row < n), row, -1)


@tilelang.jit(pass_configs=PLAIN)
def sparse_forward_kernel(H: int, c: int, scale: float, window: int, BLK: int):
    """o (B, T, H, c) and lse (B, T, H), natural, the sink included, of
    each query over its rows idx (B, T, K) and its last window rows; tiles
    (rows, heads) and sums (c, heads), as the queries' backward pass, and
    one warp per 16 rows of a block of BLK."""
    HP = _padded_heads(H, 8)
    b, t, n, kc = T.symbolic("b"), T.symbolic("t"), T.symbolic("n"), T.symbolic("kc")
    scale2 = scale * LOG2E

    @T.prim_func
    def kernel(q: T.Tensor((b, t, H, c), BF16), kv: T.Tensor((b, n, 2 * c), BF16), sink: T.Tensor((H,), FP32),
               idx: T.Tensor((b, t, kc), INT64), pos: T.Tensor((b, t), INT32), o: T.Tensor((b, t, H, c), BF16),
               lse: T.Tensor((b, t, H), FP32)):
        with T.Kernel(t, b, threads=2 * BLK) as (bx, by):
            q_s = T.alloc_shared((HP, c), BF16)
            k_s = T.alloc_shared((BLK, c), BF16)
            v_s = T.alloc_shared((BLK, c), BF16)
            rows = T.alloc_shared((BLK,), INT32)
            s = T.alloc_fragment((BLK, HP), FP32)
            p_s = T.alloc_shared((BLK, HP), BF16)
            acc = T.alloc_fragment((c, HP), FP32)
            m = T.alloc_fragment((HP,), FP32)
            m_prev = T.alloc_fragment((HP,), FP32)
            alpha = T.alloc_fragment((HP,), FP32)
            total = T.alloc_fragment((HP,), FP32)
            l = T.alloc_fragment((HP,), FP32)
            safe = T.alloc_fragment((HP,), FP32)  # m, or 0 while no row has counted: no -inf minus -inf
            nb1 = T.ceildiv(kc, BLK)

            for i, j in T.Parallel(HP, c):
                q_s[i, j] = T.if_then_else(i < H, q[by, bx, T.min(i, H - 1), j], 0)
            T.fill(acc, 0)
            T.fill(l, 0)
            T.fill(m, -T.infinity(FP32))
            # one stage still makes the gathers asynchronous copies; more measured slower
            for blk in T.Pipelined(nb1 + T.ceildiv(window, BLK), num_stages=1):
                for j in T.Parallel(BLK):
                    rows[j] = _rows_of(idx, pos, kc, n, window, BLK, blk, nb1, by, bx, j)
                for j, d in T.Parallel(BLK, c):
                    k_s[j, d] = T.if_then_else(rows[j] >= 0, kv[by, T.max(rows[j], 0), d], 0)
                    v_s[j, d] = T.if_then_else(rows[j] >= 0, kv[by, T.max(rows[j], 0), c + d], 0)
                for j, i in T.Parallel(BLK, HP):
                    s[j, i] = T.if_then_else(rows[j] >= 0, 0, -T.infinity(FP32))
                T.gemm(k_s, q_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m, m_prev)
                T.reduce_max(s, m, dim=0, clear=False)
                for i in T.Parallel(HP):
                    safe[i] = T.if_then_else(m[i] == -T.infinity(FP32), 0, m[i])
                    alpha[i] = T.exp2((m_prev[i] - safe[i]) * scale2)
                for j, i in T.Parallel(BLK, HP):
                    s[j, i] = T.exp2(s[j, i] * scale2 - safe[i] * scale2)
                T.reduce_sum(s, total, dim=0)
                for i in T.Parallel(HP):
                    l[i] = l[i] * alpha[i] + total[i]
                T.copy(s, p_s)
                for d, i in T.Parallel(c, HP):
                    acc[d, i] *= alpha[i]
                T.gemm(v_s, p_s, acc, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
            for i in T.Parallel(HP):  # the sink joins the denominator: l + e^(sink - m)
                safe[i] = T.if_then_else(m[i] == -T.infinity(FP32), 0, m[i] * scale)
                l[i] = l[i] + T.exp2((sink[T.min(i, H - 1)] - safe[i]) * LOG2E)
            for d, i in T.Parallel(c, HP):
                if i < H:
                    o[by, bx, i, d] = T.Cast(BF16, acc[d, i] / l[i])
            for i in T.Parallel(HP):
                if i < H:
                    lse[by, bx, i] = safe[i] + T.log2(l[i]) * LN2

    return kernel


@tilelang.jit(pass_configs=PLAIN)
def sparse_queries_backward_kernel(H: int, c: int, scale: float, window: int, BLK: int, threads: int, scatter: bool):
    """dq (B, T, H, c) from q, kv, idx, pos, d_out (B, T, H, c), lse and
    delta (B, T, H): each query's gradient over its chosen rows and its
    window, written once; tiles (rows, heads) and sums (c, heads).
    scatter: also adds each query's share of its rows' key and value
    gradients into dkv (B, N, 2c) in FP32, zeroed before, by atomic
    additions of 4 values; the heads are then padded to 16, the middle
    dimension of the products (rows, heads) x (heads, c) that make it."""
    HP = _padded_heads(H, 16 if scatter else 8)
    # the scatter's 17 statements of the loop over blocks, run in their order: its own analysis fails on them
    order, stages = (list(range(17)), [0] * 17) if scatter else (None, None)
    b, t, n, kc = T.symbolic("b"), T.symbolic("t"), T.symbolic("n"), T.symbolic("kc")
    b2, n2 = T.symbolic("b2"), T.symbolic("n2")  # dkv's, of any size when not scatter
    scale2 = scale * LOG2E

    @T.prim_func
    def kernel(q: T.Tensor((b, t, H, c), BF16), kv: T.Tensor((b, n, 2 * c), BF16), idx: T.Tensor((b, t, kc), INT64),
               pos: T.Tensor((b, t), INT32), d_out: T.Tensor((b, t, H, c), BF16), lse: T.Tensor((b, t, H), FP32),
               delta: T.Tensor((b, t, H), FP32), dq: T.Tensor((b, t, H, c), BF16),
               dkv: T.Tensor((b2, n2, 2 * c), FP32)):
        with T.Kernel(t, b, threads=threads) as (bx, by):
            q_s = T.alloc_shared((HP, c), BF16)
            do_s = T.alloc_shared((HP, c), BF16)
            k_s = T.alloc_shared((BLK, c), BF16)
            v_s = T.alloc_shared((BLK, c), BF16)
            ds_s = T.alloc_shared((BLK, HP), BF16)
            rows = T.alloc_shared((BLK,), INT32)
            s = T.alloc_fragment((BLK, HP), FP32)
            dp = T.alloc_fragment((BLK, HP), FP32)
            dq_f = T.alloc_fragment((c, HP), FP32)
            lse_f = T.alloc_fragment((HP,), FP32)
            delta_f = T.alloc_fragment((HP,), FP32)
            if scatter:
                p_s = T.alloc_shared((BLK, HP), BF16)
                dk_f = T.alloc_fragment((BLK, c), FP32)
                dv_f = T.alloc_fragment((BLK, c), FP32)
                dk_s = T.alloc_shared((BLK, c), FP32)
                dv_s = T.alloc_shared((BLK, c), FP32)
            nb1 = T.ceildiv(kc, BLK)

            for i, j in T.Parallel(HP, c):
                q_s[i, j] = T.if_then_else(i < H, q[by, bx, T.min(i, H - 1), j], 0)
                do_s[i, j] = T.if_then_else(i < H, d_out[by, bx, T.min(i, H - 1), j], 0)
            for i in T.Parallel(HP):  # a padded head weighs nothing
                lse_f[i] = T.if_then_else(i < H, lse[by, bx, T.min(i, H - 1)] * LOG2E, T.infinity(FP32))
                delta_f[i] = T.if_then_else(i < H, delta[by, bx, T.min(i, H - 1)], 0)
            T.clear(dq_f)
            # one stage still makes the gathers asynchronous copies, which T.serial does not
            for blk in T.Pipelined(nb1 + T.ceildiv(window, BLK), num_stages=1, order=order, stage=stages):
                for j in T.Parallel(BLK):
                    rows[j] = _rows_of(idx, pos, kc, n, window, BLK, blk, nb1, by, bx, j)
                for j, d in T.Parallel(BLK, c):
                    k_s[j, d] = T.if_then_else(rows[j] >= 0, kv[by, T.max(rows[j], 0), d], 0)
                    v_s[j, d] = T.if_then_else(rows[j] >= 0, kv[by, T.max(rows[j], 0), c + d], 0)
                T.clear(s)
                T.gemm(k_s, q_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.clear(dp)
                T.gemm(v_s, do_s, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for j, i in T.Parallel(BLK, HP):
                    s[j, i] = T.if_then_else(rows[j] >= 0, T.exp2(s[j, i] * scale2 - lse_f[i]), 0)  # the weights
                    dp[j, i] = s[j, i] * (dp[j, i] - delta_f[i]) * scale
                T.copy(dp, ds_s)
                T.gemm(k_s, ds_s, dq_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                if scatter:
                    T.copy(s, p_s)
                    T.clear(dk_f)
                    T.gemm(ds_s, q_s, dk_f, policy=T.GemmWarpPolicy.FullRow)
                    T.clear(dv_f)
                    T.gemm(p_s, do_s, dv_f, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dk_f, dk_s)
                    T.copy(dv_f, dv_s)
                    for j, d4 in T.Parallel(BLK, c // 4):
                        if rows[j] >= 0:
                            T.atomic_addx4(dkv[by, rows[j], d4 * 4], dk_s[j, d4 * 4])
                            T.atomic_addx4(dkv[by, rows[j], c + d4 * 4], dv_s[j, d4 * 4])
            for d, i in T.Parallel(c, HP):
                if i < H:
                    dq[by, bx, i, d] = T.Cast(BF16, dq_f[d, i])

    return kernel


@tilelang.jit(pass_configs={**PLAIN, tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,  # slots by atomics
                             tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})  # fails on an atomic's index
def choosers_kernel(window: int, BLK: int, threads: int, place: bool):
    """Each query's rows, its chosen rows idx (B, T, K) and its window: adds
    one to each row's slot (B N,), or, place, writes the query, b T + t,
    into choosers at each row's slot, the next free place in that row's
    list, and moves the slot on."""
    b, t, n, kc, m = T.symbolic("b"), T.symbolic("t"), T.symbolic("n"), T.symbolic("kc"), T.symbolic("m")

    @T.prim_func
    def kernel(idx: T.Tensor((b, t, kc), INT64), pos: T.Tensor((b, t), INT32), slots: T.Tensor((b, n), INT32),
               choosers: T.Tensor((m,), INT32)):
        with T.Kernel(t, b, threads=threads) as (bx, by):
            rows = T.alloc_shared((BLK,), INT32)
            nb1 = T.ceildiv(kc, BLK)
            for blk in T.serial(nb1 + T.ceildiv(window, BLK)):
                for j in T.Parallel(BLK):
                    rows[j] = _rows_of(idx, pos, kc, n, window, BLK, blk, nb1, by, bx, j)
                for j in T.Parallel(BLK):
                    if rows[j] >= 0:
                        if place:
                            choosers[T.atomic_add(slots[by, rows[j]], 1, return_prev=True)] = by * t + bx
                        else:
                            T.atomic_add(slots[by, rows[j]], 1)

    return kernel


@tilelang.jit(pass_configs=PLAIN)
def sparse_keys_backward_kernel(H: int, c: int, scale: float, rows_per_step: int, threads: int):
    """dkv (B, N, 2c), the keys' and values' gradients side by side: each
    table row's over the queries that read it, choosers[starts[b N +
    n]:starts[b N + n + 1]], rows_per_step (query, head) rows at a time
    (the module's docstring), heads padded to a power of 2, written once. q and d_out (B T H, c), lse
    and delta (B T H,), flat."""
    HQ = 1 << (H - 1).bit_length()
    E = max(1, rows_per_step // HQ)  # queries per step
    EH = E * HQ
    b, r, n, m, s1 = T.symbolic("b"), T.symbolic("r"), T.symbolic("n"), T.symbolic("m"), T.symbolic("s1")
    scale2 = scale * LOG2E

    @T.prim_func
    def kernel(q: T.Tensor((r, c), BF16), kv: T.Tensor((b, n, 2 * c), BF16), d_out: T.Tensor((r, c), BF16),
               lse: T.Tensor((r,), FP32), delta: T.Tensor((r,), FP32), starts: T.Tensor((s1,), INT32),
               choosers: T.Tensor((m,), INT32), dkv: T.Tensor((b, n, 2 * c), BF16)):
        with T.Kernel(n, b, threads=threads) as (bx, by):
            k_s = T.alloc_shared((c,), FP32)
            v_s = T.alloc_shared((c,), FP32)
            rows = T.alloc_fragment((EH,), INT32)  # (query, head) rows, -1 for none
            q_f = T.alloc_fragment((EH, c), BF16)
            do_f = T.alloc_fragment((EH, c), BF16)
            q_n = T.alloc_fragment((EH, c), BF16)  # the next step's, loaded while this one computes
            do_n = T.alloc_fragment((EH, c), BF16)
            lse_f = T.alloc_fragment((EH,), FP32)
            delta_f = T.alloc_fragment((EH,), FP32)
            lse_n = T.alloc_fragment((EH,), FP32)
            delta_n = T.alloc_fragment((EH,), FP32)
            prod = T.alloc_fragment((EH, c), FP32)
            z = T.alloc_fragment((EH,), FP32)
            dp = T.alloc_fragment((EH,), FP32)
            dk = T.alloc_fragment((EH, c), FP32)  # each (query, head) row's own sum, reduced over rows at the end
            dv = T.alloc_fragment((EH, c), FP32)
            dk_sum = T.alloc_fragment((c,), FP32)
            dv_sum = T.alloc_fragment((c,), FP32)
            first = starts[by * n + bx]
            count = starts[by * n + bx + 1] - first

            for d in T.Parallel(c):
                k_s[d] = T.Cast(FP32, kv[by, bx, d])
                v_s[d] = T.Cast(FP32, kv[by, bx, c + d])
            T.clear(dk)
            T.clear(dv)
            for i in T.Parallel(EH):
                e = i // HQ
                rows[i] = T.if_then_else((e < count) & (i % HQ < H), choosers[first + T.min(e, count - 1)] * H + i % HQ,
                                         -1)
            for i, d in T.Parallel(EH, c):
                q_n[i, d] = T.if_then_else(rows[i] >= 0, q[T.max(rows[i], 0), d], 0)
                do_n[i, d] = T.if_then_else(rows[i] >= 0, d_out[T.max(rows[i], 0), d], 0)
            for i in T.Parallel(EH):
                lse_n[i] = T.if_then_else(rows[i] >= 0, lse[T.max(rows[i], 0)] * LOG2E, T.infinity(FP32))
                delta_n[i] = T.if_then_else(rows[i] >= 0, delta[T.max(rows[i], 0)], 0)
            for step in T.serial(T.ceildiv(count, E)):
                T.copy(q_n, q_f)
                T.copy(do_n, do_f)
                T.copy(lse_n, lse_f)
                T.copy(delta_n, delta_f)
                for i in T.Parallel(EH):
                    e = (step + 1) * E + i // HQ
                    rows[i] = T.if_then_else((e < count) & (i % HQ < H),
                                             choosers[first + T.min(e, count - 1)] * H + i % HQ, -1)
                for i, d in T.Parallel(EH, c):
                    q_n[i, d] = T.if_then_else(rows[i] >= 0, q[T.max(rows[i], 0), d], 0)
                    do_n[i, d] = T.if_then_else(rows[i] >= 0, d_out[T.max(rows[i], 0), d], 0)
                for i in T.Parallel(EH):
                    lse_n[i] = T.if_then_else(rows[i] >= 0, lse[T.max(rows[i], 0)] * LOG2E, T.infinity(FP32))
                    delta_n[i] = T.if_then_else(rows[i] >= 0, delta[T.max(rows[i], 0)], 0)
                for i, d in T.Parallel(EH, c):
                    prod[i, d] = T.Cast(FP32, q_f[i, d]) * k_s[d]
                T.reduce_sum(prod, z, dim=1)  # q . k
                for i, d in T.Parallel(EH, c):
                    prod[i, d] = T.Cast(FP32, do_f[i, d]) * v_s[d]
                T.reduce_sum(prod, dp, dim=1)  # d_out . v
                for i in T.Parallel(EH):  # a missing row has lse = inf: weight 0
                    z[i] = T.exp2(z[i] * scale2 - lse_f[i])
                    dp[i] = z[i] * (dp[i] - delta_f[i])
                for i, d in T.Parallel(EH, c):
                    dk[i, d] += dp[i] * T.Cast(FP32, q_f[i, d])
                    dv[i, d] += z[i] * T.Cast(FP32, do_f[i, d])
            T.reduce_sum(dk, dk_sum, dim=0)
            T.reduce_sum(dv, dv_sum, dim=0)
            for d in T.Parallel(c):
                dkv[by, bx, d] = T.Cast(BF16, dk_sum[d] * scale)
                dkv[by, bx, c + d] = T.Cast(BF16, dv_sum[d])

    return kernel


def _check(q):
    B, T_, H, c = q.shape
    if not (H <= 64 and c % 16 == 0 and c <= 256):
        raise ValueError(f"the TileLang kernels take at most 64 heads and a head width that is a multiple of 16 of "
                         f"at most 256, not {H} heads of {c}")


def _table(k, v):
    """k and v (B, N, c) as one BF16 table (B, N, 2c), keys first: a view
    when they are the two halves of such a table already."""
    B, N, c = k.shape
    halves = (k.dtype == v.dtype == torch.bfloat16 and k.stride() == v.stride() and k.stride()[1:] == (2 * c, 1)
              and (B == 1 or k.stride(0) == N * 2 * c) and v.data_ptr() == k.data_ptr() + c * k.element_size())
    if halves:
        return k.as_strided((B, N, 2 * c), (N * 2 * c, 2 * c, 1))
    return torch.cat([k, v], -1).to(torch.bfloat16).contiguous()


def _block_max(values, size, fill):
    """values (B, L) -> (B, ceil(L / size)): each block's largest, int32."""
    B, L = values.shape
    padded = torch.nn.functional.pad(values, (0, -L % size), value=fill)
    return padded.view(B, -1, size).amax(-1).to(torch.int32).contiguous()


class _Exact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, positions, scale, top_k, window):
        _check(q)
        B, T_, H, c = q.shape
        N = k.shape[1]
        chosen_tiles = tiles(q.device)
        qb = q.to(torch.bfloat16).contiguous().view(B, T_ * H, c)
        table = _table(k, v)
        pos = positions.to(torch.int32).contiguous()
        seen = (positions + 1).clamp(max=N)  # (B, T): the rows each query sees
        o = torch.empty_like(qb)
        lse = q.new_empty(B, T_ * H, dtype=torch.float32)
        mx = torch.empty_like(lse)
        M = chosen_tiles.forward_rows
        exact_forward_kernel(H, c, scale, M, chosen_tiles.forward_keys, chosen_tiles.stages, chosen_tiles.threads)(
            qb, table, pos, _block_max(seen.repeat_interleave(H, 1), M, 0), o, lse, mx)
        K = chosen_columns(top_k, window, N)
        chosen = torch.full((B, T_, K), -1, dtype=torch.long, device=q.device)
        if K:
            BT = _tokens(chosen_tiles, H)
            far = (positions - window + 1).clamp(min=0, max=N)  # (B, T): each query's candidate rows
            bins = torch.empty(B, T_, CANDIDATES, dtype=torch.long, device=q.device)
            state = torch.empty(B, T_, 3, dtype=torch.int32, device=q.device)
            exact_select_kernel(H, c, scale, window, chosen_tiles.select_rows, BT, chosen_tiles.select_keys,
                                chosen_tiles.stages, chosen_tiles.threads, CANDIDATES,
                                round(math.log2(SPAN / BINS)))(qb, table, pos, lse, mx, _block_max(far, BT, 0), chosen,
                                                               bins, state)
            rank_kernel(CANDIDATES, chosen_tiles.threads)(bins, state, chosen)
            chosen = chosen.sort(-1).values
        ctx.save_for_backward(qb, table, pos, o, lse)
        ctx.shape, ctx.scale, ctx.dtypes = (B, T_, H, c, N), scale, (q.dtype, k.dtype, v.dtype)
        ctx.mark_non_differentiable(chosen)
        return o.view(B, T_, H, c).to(q.dtype), chosen

    @staticmethod
    def backward(ctx, d_out, _d_chosen):
        qb, table, pos, o, lse = ctx.saved_tensors
        B, T_, H, c, N = ctx.shape
        chosen_tiles = tiles(qb.device)
        d_out = d_out.to(torch.bfloat16).contiguous().view(B, T_ * H, c)
        delta = (d_out.float() * o.float()).sum(-1)
        dq = torch.zeros(B, T_ * H, c, dtype=torch.float32, device=qb.device)
        dkv = torch.empty(B, N, 2 * c, dtype=torch.bfloat16, device=qb.device)
        M, BN = chosen_tiles.backward_rows, chosen_tiles.backward_keys
        # each key tile's first block of rows: the first whose rows so far reach the tile
        reach = _block_max(pos.long().repeat_interleave(H, 1), M, -1).long().cummax(-1).values  # (B, row blocks)
        tile_start = torch.arange(0, N, BN, device=qb.device).expand(B, -1).contiguous()
        start = torch.searchsorted(reach, tile_start).to(torch.int32).contiguous()
        exact_backward_kernel(H, c, ctx.scale, M, BN, chosen_tiles.stages, chosen_tiles.threads)(
            qb, table, d_out, lse, delta, pos, start, dq, dkv)
        q_dtype, k_dtype, v_dtype = ctx.dtypes
        return (dq.view(B, T_, H, c).to(q_dtype), dkv[..., :c].to(k_dtype), dkv[..., c:].to(v_dtype),
                None, None, None, None)


def exact_attention(q, k, v, positions, scale, top_k, window):
    """pytorch_kernels.exact_attention's result; the chosen rows come back
    sorted, -1 first."""
    return _Exact.apply(q, k, v, positions, scale, top_k, window)


class _Sparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink, idx, positions, scale, window):
        _check(q)
        B, T_, H, c = q.shape
        chosen_tiles = tiles(q.device)
        qb = q.to(torch.bfloat16).contiguous()
        table = _table(k, v)
        pos = positions.to(torch.int32).contiguous()
        idx = idx.contiguous() if idx.shape[-1] else idx.new_full((B, T_, 1), -1)  # no rows chosen yet
        o = torch.empty_like(qb)
        lse = q.new_empty(B, T_, H, dtype=torch.float32)
        sparse_forward_kernel(H, c, scale, window, chosen_tiles.sparse_forward_block)(
            qb, table, sink.float().contiguous(), idx, pos, o, lse)
        ctx.save_for_backward(qb, table, sink, idx, pos, o, lse)
        ctx.scale, ctx.window, ctx.dtypes = scale, window, (q.dtype, k.dtype, v.dtype)
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, d_out):
        qb, table, sink, idx, pos, o, lse = ctx.saved_tensors
        B, T_, H, c = qb.shape
        chosen_tiles = tiles(qb.device)
        d_out = d_out.to(torch.bfloat16).contiguous()
        delta = (d_out.float() * o.float()).sum(-1)  # (B, T, H)
        dq = torch.empty_like(qb)
        scatter = H > GATHER_HEADS
        dkv = torch.zeros(table.shape, dtype=torch.float32, device=qb.device) if scatter else torch.empty(
            table.shape, dtype=torch.bfloat16, device=qb.device)
        sparse_queries_backward_kernel(H, c, ctx.scale, ctx.window, chosen_tiles.sparse_block,
                                       chosen_tiles.sparse_threads, scatter)(
            qb, table, idx, pos, d_out, lse, delta, dq, dkv if scatter else dkv.new_empty(1, 1, 2 * c).float())
        if not scatter:
            N = table.shape[1]
            starts, choosers = inverse(idx, (N, ctx.window), lambda: _choosers(idx, pos, N, ctx.window, chosen_tiles))
            sparse_keys_backward_kernel(H, c, ctx.scale, KEYS_ROWS, KEYS_THREADS)(
                qb.view(-1, c), table, d_out.view(-1, c), lse.view(-1), delta.view(-1), starts, choosers, dkv)
        d_sink = -((sink.float() - lse).exp() * delta).sum((0, 1))  # the sink carries no value
        q_dtype, k_dtype, v_dtype = ctx.dtypes
        return (dq.to(q_dtype), dkv[..., :c].to(k_dtype), dkv[..., c:].to(v_dtype), d_sink.to(sink.dtype),
                None, None, None, None)


def _choosers(idx, pos, N, window, chosen_tiles):
    """The inverse of each query's rows, its chosen rows idx (B, T, K) and
    its last window rows: (starts (B N + 1,), choosers), both int32, where
    row n of batch row b was read by the queries choosers[starts[b N +
    n]:starts[b N + n + 1]], each b T + t, in no particular order."""
    B, T_, K = idx.shape
    if B * T_ * (K + window) >= 1 << 31:
        raise ValueError(f"the TileLang sparse backward pass lists at most 2^31 rows read, not {B * T_ * (K + window)}")
    counts = torch.zeros(B, N, dtype=torch.int32, device=idx.device)
    choosers = torch.empty(max(1, B * T_ * (K + window)), dtype=torch.int32, device=idx.device)
    block = chosen_tiles.sparse_block
    choosers_kernel(window, block, chosen_tiles.threads, False)(idx, pos, counts, choosers)
    starts = torch.zeros(B * N + 1, dtype=torch.int32, device=idx.device)
    starts[1:] = counts.view(-1).cumsum(0)
    choosers_kernel(window, block, chosen_tiles.threads, True)(idx, pos, starts[:-1].view(B, N).clone(), choosers)
    return starts, choosers


def sparse_attention(q, k, v, sink, idx, positions, scale, window):
    """pytorch_kernels.sparse_attention's result."""
    return _Sparse.apply(q, k, v, sink, idx, positions, scale, window)
