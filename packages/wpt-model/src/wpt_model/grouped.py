"""Grouped matrix products in Triton, for the routed experts (experts.py)
on a GPU where PyTorch's own grouped product (F.grouped_mm) has no kernel
of its own: there PyTorch runs one matrix product per group, after
reading the groups' sizes back to the host, which makes the CPU wait for
the GPU at every call and launches a few kernels per group. These kernels
read the groups' ends on the GPU, so a call waits on nothing and a CUDA
graph can record it.

    grouped_mm(a, b, ends)  a (M, K), its rows grouped in order, group g's
                            ending at row ends[g] (int32); b (G, K, N):
                            row r of the output (M, N) is a[r] @ b[g] for
                            the group g of r

Its backward pass is the same product for a's gradient, d_out @ b[g]^T,
and one kernel for b's: group g's gradient sums a[r]^T d_out[r] over its
rows. Each kernel multiplies BF16 tiles into an FP32 sum, as F.grouped_mm
does, and writes no atomic additions, so a result is the same from run to
run.

The rows kernel gives each program one tile of BLOCK_ROWS rows of one
group, so a group's last tile is part empty: there are at most M /
BLOCK_ROWS + G tiles. The weights kernel gives each program one tile of
one group's gradient, and loops over the group's rows. The tile sizes are
a first choice measured on an RTX 5080 laptop GPU (sm_120) at the series'
widths, open to review.

PyTorch's grouped product runs as one kernel of its own on the GPUs NATIVE
names (compute capabilities 9 and 10: H100, B200), where these kernels are
not used; on an RTX 5080 laptop GPU (compute capability 12) it runs one
product per group. The list is open to review: these kernels have not been
measured against PyTorch's on an H100. On AMD GPUs PyTorch's runs.
"""

import torch
import triton
import triton.language as tl

NATIVE = (9, 10)  # CUDA compute capability majors where F.grouped_mm runs as one kernel
BLOCK_ROWS, BLOCK_N, BLOCK_K = 64, 64, 32  # the rows kernel's tile
WEIGHT_BLOCK, WEIGHT_ROWS = 64, 64  # the weights kernel's tile (K and N), and its rows per step


def needed(device):
    """Whether grouped_mm, rather than F.grouped_mm, runs on device: a CUDA
    GPU (not ROCm) whose compute capability NATIVE does not name."""
    return (device.type == "cuda" and torch.version.hip is None
            and torch.cuda.get_device_capability(device)[0] not in NATIVE)


@triton.jit
def _rows_kernel(a, b, out, ends, K, N, s_am, s_ak, s_bg, s_bk, s_bn, s_om, s_on,
                 G: tl.constexpr, GROUPS: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    tile, pid_n = tl.program_id(0), tl.program_id(1)
    groups = tl.arange(0, GROUPS)
    real = groups < G
    end = tl.load(ends + groups, mask=real, other=0)
    start = tl.load(ends + groups - 1, mask=real & (groups > 0), other=0)
    tiles = tl.where(real, (end - start + BLOCK_ROWS - 1) // BLOCK_ROWS, 0)
    through = tl.cumsum(tiles, 0)  # the tiles of the groups up to each, included
    g = tl.sum((through <= tile).to(tl.int32), 0)  # this tile's group: the first whose tiles reach past it
    mine = groups == g
    first = tl.sum(tl.where(mine, start, 0), 0) + (tile - tl.sum(tl.where(mine, through - tiles, 0), 0)) * BLOCK_ROWS
    last = tl.sum(tl.where(mine, end, 0), 0)  # 0 past the last group, so no row is written
    rows = first + tl.arange(0, BLOCK_ROWS)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    keep = rows < last
    weights = b + g.to(tl.int64) * s_bg
    total = tl.zeros((BLOCK_ROWS, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ks = k + inner
        x = tl.load(a + rows[:, None].to(tl.int64) * s_am + ks[None, :] * s_ak,
                    mask=keep[:, None] & (ks[None, :] < K), other=0.0)
        w = tl.load(weights + ks[:, None] * s_bk + cols[None, :] * s_bn,
                    mask=(ks[:, None] < K) & (cols[None, :] < N) & (g < G), other=0.0)
        total = tl.dot(x, w, total)
    tl.store(out + rows[:, None].to(tl.int64) * s_om + cols[None, :] * s_on, total.to(out.dtype.element_ty),
             mask=keep[:, None] & (cols[None, :] < N))


@triton.jit
def _weights_kernel(a, d, out, ends, K, N, s_am, s_ak, s_dm, s_dn, s_og, s_ok, s_on,
                    BLOCK: tl.constexpr, ROWS: tl.constexpr):
    g, pid_k, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    end = tl.load(ends + g)
    start = tl.load(ends + tl.maximum(g - 1, 0)) * (g > 0)
    ks = pid_k * BLOCK + tl.arange(0, BLOCK)
    ns = pid_n * BLOCK + tl.arange(0, BLOCK)
    steps = tl.arange(0, ROWS)
    total = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for r in range(start, end, ROWS):
        rows = r + steps
        keep = rows < end
        x = tl.load(a + rows[None, :].to(tl.int64) * s_am + ks[:, None] * s_ak,
                    mask=keep[None, :] & (ks[:, None] < K), other=0.0)  # (BLOCK, ROWS): a's rows, transposed
        y = tl.load(d + rows[:, None].to(tl.int64) * s_dm + ns[None, :] * s_dn,
                    mask=keep[:, None] & (ns[None, :] < N), other=0.0)
        total = tl.dot(x, y, total)
    tl.store(out + g.to(tl.int64) * s_og + ks[:, None] * s_ok + ns[None, :] * s_on, total.to(out.dtype.element_ty),
             mask=(ks[:, None] < K) & (ns[None, :] < N))


def _rows(a, b, ends):
    """a (M, K) times b (G, K, N), group by group: (M, N) in a's dtype."""
    M, K = a.shape
    G, _, N = b.shape
    out = a.new_empty(M, N)
    grid = (triton.cdiv(M, BLOCK_ROWS) + G, triton.cdiv(N, BLOCK_N))
    _rows_kernel[grid](a, b, out, ends, K, N, *a.stride(), *b.stride(), *out.stride(), G=G,
                       GROUPS=triton.next_power_of_2(G), BLOCK_ROWS=BLOCK_ROWS, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out


def _weights(a, d, ends, G):
    """Each group's a^T d over its rows: (G, K, N) in a's dtype, zeros for an empty group."""
    K, N = a.shape[1], d.shape[1]
    out = a.new_empty(G, K, N)
    grid = (G, triton.cdiv(K, WEIGHT_BLOCK), triton.cdiv(N, WEIGHT_BLOCK))
    _weights_kernel[grid](a, d, out, ends, K, N, *a.stride(), *d.stride(), *out.stride(), BLOCK=WEIGHT_BLOCK,
                          ROWS=WEIGHT_ROWS)
    return out


class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, ends):
        ctx.save_for_backward(a, b, ends)
        return _rows(a, b, ends)

    @staticmethod
    def backward(ctx, grad):
        a, b, ends = ctx.saved_tensors
        grad = grad.to(a.dtype)
        d_a = _rows(grad, b.transpose(1, 2), ends) if ctx.needs_input_grad[0] else None
        d_b = _weights(a, grad, ends, b.shape[0]).to(b.dtype) if ctx.needs_input_grad[1] else None
        return d_a, d_b, None


def grouped_mm(a, b, ends):
    """a (M, K) times b (G, K, N) group by group, as F.grouped_mm(a, b,
    offs=ends) computes it (see the module's docstring)."""
    return _GroupedMM.apply(a, b, ends)
