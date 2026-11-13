"""The TileLang kernels' own cases (tilelang_kernels.py); test_kernels.py
compares them with the PyTorch form on the recipes' shapes.

- Every kernel builds for both GPUs it targets, an H100 (sm_90) and an RTX
  50 series laptop (sm_120), each with its tiles, wherever TileLang is
  installed: a GPU runs only its own, so the other is built here.
- Where TileLang runs, the exact and sparse kernels match the PyTorch form
  on what the recipes' shapes leave out: a pass after earlier tokens, a
  generation's padding query past the table, a head count that does not
  fill the matrix units' rows, 2 heads, whose selection blocks hold the
  most tokens and so the most shared memory, and fewer columns of chosen
  rows than top_k.
- The choice of rows among many equal scores."""

import importlib.util
import sys

import pytest
import torch

from wpt_model import kernels as dispatch
from wpt_model import pytorch_kernels
from wpt_utils.device import select_device

installed = importlib.util.find_spec("tilelang") is not None and sys.platform == "linux"
needs_tilelang = pytest.mark.skipif(not installed, reason="TileLang is not installed (the cuda group, on Linux)")


def runs_here():
    return installed and torch.version.cuda is not None and select_device().type == "cuda" and dispatch.nvcc()


@needs_tilelang
@pytest.mark.parametrize("arch, size", [("sm_90", "TILES_LARGE"), ("sm_120", "TILES_SMALL")])
def test_every_kernel_builds_for_both_gpus(arch, size):
    # the smallest recipe's 4 heads of 128; with the CUDA compiler there,
    # the source compiles to the GPU's code, as TileLang does when it runs
    import tilelang
    import tvm
    from tilelang.contrib import nvcc
    from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

    from wpt_model import tilelang_kernels as tl

    t = getattr(tl, size)
    H, c, scale, window = 4, 128, 128 ** -0.5, 128
    programs = [
        (tl.exact_forward_kernel, (H, c, scale, t.forward_rows, t.forward_keys, t.stages, t.threads)),
        (tl.exact_select_kernel, (H, c, scale, window, t.select_rows, tl._tokens(t, H), t.select_keys, t.stages,
                                  t.threads, tl.CANDIDATES, -3)),
        (tl.rank_kernel, (tl.CANDIDATES, t.threads)),
        (tl.exact_backward_kernel, (H, c, scale, t.backward_rows, t.backward_keys, t.stages, t.threads)),
        (tl.sparse_forward_kernel, (H, c, scale, window, t.sparse_forward_block)),
        (tl.sparse_queries_backward_kernel, (H, c, scale, window, t.sparse_block, t.sparse_threads, False)),
        (tl.sparse_queries_backward_kernel, (H, c, scale, window, t.sparse_block, t.sparse_threads, True)),
        (tl.choosers_kernel, (window, t.sparse_block, t.threads, False)),
        (tl.choosers_kernel, (window, t.sparse_block, t.threads, True)),
        (tl.sparse_keys_backward_kernel, (H, c, scale, tl.KEYS_ROWS, tl.KEYS_THREADS)),
    ]
    target = f"cuda -arch={arch}"
    for factory, arguments in programs:
        with tvm.transform.PassContext(config=factory.pass_configs or {}), tvm.target.Target(target):
            lowered = tilelang.lower(factory.get_tir(*arguments), target=target)
        if dispatch.nvcc():
            assert nvcc.compile_cuda(lowered.kernel_source, "cubin", [f"-arch={arch}a"],
                                     options=["-std=c++17", f"-I{TILELANG_TEMPLATE_PATH}", f"-I{CUTLASS_INCLUDE_DIR}"])


def relative_error(got, expected):
    return float((got.float() - expected.float()).abs().max() / expected.float().abs().max())


@pytest.mark.skipif(not runs_here(), reason="TileLang runs on an NVIDIA GPU under Linux with nvcc")
@pytest.mark.parametrize("heads, top_k, padding", [(3, 64, 0), (2, 64, 0), (4, 1024, 2)])
def test_the_kernels_give_the_pytorch_result_after_earlier_tokens(heads, top_k, padding):
    # 200 queries after 500 earlier tokens, the last `padding` of them a
    # generation cache's padding past the table's last row; top_k 1,024
    # leaves fewer candidates than columns, so the column count is theirs
    from wpt_model import tilelang_kernels as tl

    device = select_device()
    generator = torch.Generator().manual_seed(0)
    B, T, N, c, window = 2, 200, 700, 128, 128

    def leaf(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator).to(device, dtype).requires_grad_()

    q, k, v, sink = leaf(B, T, heads, c), leaf(B, N, c), leaf(B, N, c), leaf(heads, dtype=torch.float32)
    positions = (N - T + padding + torch.arange(T)).expand(B, T).to(device)
    scale = c ** -0.5
    weights = torch.randn(B, T, heads, c, generator=generator).to(device)

    out, rows = tl.exact_attention(q, k, v, positions, scale, top_k, window)
    expected, expected_rows = pytorch_kernels.exact_attention(q, k, v, positions, scale, top_k, window)
    got = torch.autograd.grad((out.float() * weights).sum(), (q, k, v))
    want = torch.autograd.grad((expected.float() * weights).sum(), (q, k, v))
    assert rows.shape == expected_rows.shape and ((rows >= 0).sum(-1) == (expected_rows >= 0).sum(-1)).all()
    assert relative_error(out, expected) < 1e-2 and all(relative_error(a, b) < 1e-2 for a, b in zip(got, want))

    sparse = tl.sparse_attention(q, k, v, sink, expected_rows, positions, scale, window)
    expected = pytorch_kernels.sparse_attention(q, k, v, sink, expected_rows, positions, scale, window)
    got = torch.autograd.grad((sparse.float() * weights).sum(), (q, k, v, sink))
    want = torch.autograd.grad((expected.float() * weights).sum(), (q, k, v, sink))
    assert relative_error(sparse, expected) < 1e-2 and all(relative_error(a, b) < 1e-2 for a, b in zip(got, want))


@pytest.mark.skipif(not runs_here(), reason="TileLang runs on an NVIDIA GPU under Linux with nvcc")
@pytest.mark.parametrize("candidates", [512, 4])
def test_rows_of_equal_scores_are_chosen_as_many_as_wanted(monkeypatch, candidates):
    # every key one of two vectors, so most scores are equal: each query
    # still chooses as many distinct rows as the PyTorch form, whether the
    # equal rows fit the last bin (512) or not (4), and through several
    # counting passes when 4 rows is the most a bin may hold
    from wpt_model import tilelang_kernels as tl

    device = select_device()
    monkeypatch.setattr(tl, "CANDIDATES", candidates)
    generator = torch.Generator().manual_seed(3)
    B, T, H, c, top_k, window = 1, 400, 2, 128, 64, 8
    q = torch.randn(B, T, H, c, generator=generator).to(device, torch.bfloat16)
    pair = torch.randn(2, c, generator=generator)
    k = pair[(torch.arange(T) % 7 == 0).long()][None].to(device, torch.bfloat16)
    v = torch.randn(B, T, c, generator=generator).to(device, torch.bfloat16)
    positions = torch.arange(T, device=device).expand(B, T)
    _, rows = tl.exact_attention(q, k, v, positions, c ** -0.5, top_k, window)
    _, expected = pytorch_kernels.exact_attention(q, k, v, positions, c ** -0.5, top_k, window)
    assert rows.shape == expected.shape and ((rows >= 0).sum(-1) == (expected >= 0).sum(-1)).all()
    z = torch.einsum("bthc,bnc->bthn", q.float(), k.float()) * c ** -0.5
    z = z.masked_fill(torch.arange(T, device=device) > positions[..., None, None], float("-inf"))
    score = (z - z.logsumexp(-1, keepdim=True)).amax(2)
    kth = score.gather(-1, expected.clamp(min=0)).masked_fill(expected < 0, float("inf")).amin(-1)
    picked = score.gather(-1, rows.clamp(min=0)).masked_fill(rows < 0, float("inf"))
    assert (picked >= kth[..., None] - 1e-2).all()
    for row in rows[0]:
        kept = row[row >= 0]
        assert len(set(kept.tolist())) == len(kept)
