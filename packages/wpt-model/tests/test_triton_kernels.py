"""What the Triton kernels do beyond the shapes test_kernels.py compares:
a generation cache's pass, whose rows sit at different places and may sit
past the table's last row, with keys and values that are halves of a
layer's entries; and the choice of rows through several counting passes
and among equal scores. Both against the PyTorch form, on a GPU where
Triton runs."""

import importlib

import pytest
import torch

from wpt_model import pytorch_kernels
from wpt_utils.device import select_device


def triton_kernels():
    """The Triton kernel module, or None where it cannot run (no GPU, or
    Triton not installed)."""
    if select_device().type != "cuda":  # select_device first: AMD GPUs need it
        return None
    try:
        return importlib.import_module("wpt_model.triton_kernels")
    except ImportError:
        return None


KERNELS = triton_kernels()
needs_triton = pytest.mark.skipif(KERNELS is None, reason="needs a GPU where Triton runs")


def relative_error(got, expected):
    return float((got.float() - expected.float()).abs().max() / expected.float().abs().max())


def assert_chosen_alike(q, k, positions, chosen, expected):
    """chosen has expected's shape and its -1s in the same places, and
    every row it chose is one the reference weighs within 1e-2 (in log) of
    its own last choice, since BF16 scores may order near ties another
    way."""
    N = k.shape[1]
    z = torch.einsum("bthc,bnc->bthn", q.float(), k.float()) * q.shape[-1] ** -0.5
    z = z.masked_fill(torch.arange(N, device=q.device) > positions[..., None, None], float("-inf"))
    score = (z - z.logsumexp(-1, keepdim=True)).amax(2)  # (B, T, N): each row's largest log-weight
    kth = score.gather(-1, expected.clamp(min=0)).masked_fill(expected < 0, float("inf")).amin(-1)
    picked = score.gather(-1, chosen.clamp(min=0)).masked_fill(chosen < 0, float("inf"))
    assert chosen.shape == expected.shape and ((chosen >= 0) == (expected >= 0)).all()
    assert (picked >= kth[..., None] - 1e-2).all()


@needs_triton
def test_a_generation_pass_gives_the_pytorch_result():
    # three rows of a cache at different places, the last one's later
    # queries past the table (padding), reading the halves of an entries
    # table as keys and values: outputs, chosen rows and gradients match
    device = select_device()
    generator = torch.Generator().manual_seed(0)
    B, T, H, c, N, top_k, window = 3, 4, 4, 128, 400, 64, 16

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device, torch.bfloat16).requires_grad_()

    q, entries = rand(B, T, H, c), rand(B, N, 2 * c)
    k, v = entries[..., :c], entries[..., c:]
    sink = torch.randn(H, generator=generator).to(device).requires_grad_()
    positions = torch.tensor([[396, 397, 398, 399], [100, 101, 102, 103], [398, 399, 400, 401]], device=device)
    results = []
    for form in (KERNELS, pytorch_kernels):
        out, chosen = form.exact_attention(q, k, v, positions, c ** -0.5, top_k, window)
        sparse = form.sparse_attention(q, k, v, sink, chosen, positions, c ** -0.5, window)
        weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(1)).to(device)
        grads = torch.autograd.grad(((out.float() + sparse.float()) * weights).sum(), (q, entries, sink))
        results.append((out, chosen, sparse, grads))
    (out, chosen, sparse, grads), (out_, chosen_, sparse_, grads_) = results
    assert relative_error(out, out_) < 1e-2 and relative_error(sparse, sparse_) < 1e-2
    assert_chosen_alike(q, k, positions, chosen, chosen_)
    assert all(relative_error(a, b) < 1e-2 for a, b in zip(grads, grads_))


@needs_triton
@pytest.mark.parametrize("span, candidates", [(0.5, 4), (64.0, 2)])
def test_the_rows_chosen_through_several_counting_passes_are_the_pytorch_ones(monkeypatch, span, candidates):
    # bins that hold at most a few rows, so that each query's top_k-th
    # score is found through several counting passes, its first one
    # spanning too little (0.5 bits: the passes go on below it) or much
    # (64 bits: they go on into its bins)
    device = select_device()
    monkeypatch.setattr(KERNELS, "SPAN", span)
    monkeypatch.setattr(KERNELS, "CANDIDATES", candidates)
    generator = torch.Generator().manual_seed(2)
    B, T, H, c = 2, 300, 4, 128
    q, k, v = (torch.randn(*shape, generator=generator).to(device, torch.bfloat16)
               for shape in ((B, T, H, c), (B, T, c), (B, T, c)))
    positions = torch.arange(T, device=device).expand(B, T)
    _, chosen = KERNELS.exact_attention(q, k, v, positions, c ** -0.5, 32, 8)
    _, expected = pytorch_kernels.exact_attention(q, k, v, positions, c ** -0.5, 32, 8)
    assert_chosen_alike(q, k, positions, chosen, expected)


@needs_triton
@pytest.mark.parametrize("candidates", [512, 4])
def test_rows_of_equal_scores_are_chosen_as_many_as_wanted(monkeypatch, candidates):
    # every key one of two vectors, so most scores are equal: each query
    # still chooses top_k distinct rows, every one with a score at least
    # its top_k-th, whether the equal rows fit the last bin (512) or not (4)
    device = select_device()
    monkeypatch.setattr(KERNELS, "CANDIDATES", candidates)
    generator = torch.Generator().manual_seed(3)
    B, T, H, c, top_k, window = 1, 400, 2, 128, 64, 8
    q = torch.randn(B, T, H, c, generator=generator).to(device, torch.bfloat16)
    pair = torch.randn(2, c, generator=generator)
    k = pair[(torch.arange(T) % 7 == 0).long()][None].to(device, torch.bfloat16)
    v = torch.randn(B, T, c, generator=generator).to(device, torch.bfloat16)
    positions = torch.arange(T, device=device).expand(B, T)
    _, chosen = KERNELS.exact_attention(q, k, v, positions, c ** -0.5, top_k, window)
    _, expected = pytorch_kernels.exact_attention(q, k, v, positions, c ** -0.5, top_k, window)
    assert_chosen_alike(q, k, positions, chosen, expected)
    for row in chosen[0]:
        kept = row[row >= 0]
        assert len(set(kept.tolist())) == len(kept)
