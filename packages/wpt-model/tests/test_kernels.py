"""The two attention operations of kernels.py (docs/model.md, "Attention"):
the PyTorch form against the formulas, every kernel module that runs here
against the PyTorch form, and which form a run gets. The kernel tests are
the one place where any kernel module, Triton's or TileLang's, is compared
with the reference; a kernel not written yet skips its tests."""

import importlib
import sys
import types

import pytest
import torch

import wpt_model
from wpt_model import kernels as dispatch
from wpt_model import pytorch_kernels
from wpt_model.kernels import KernelUnavailable, kernels
from wpt_utils.device import select_device


# -- the formulas, one query at a time --

def exact_by_formula(q, k, v, positions, scale, top_k, window):
    """(output, the set of chosen rows of each query) of exact attention:
    softmax over rows 0 .. p, and the top_k rows among 0 .. p - window by
    their largest weight over the heads."""
    B, T, H, c = q.shape
    out = torch.zeros(B, T, H, c)
    chosen = [[set() for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for t in range(T):
            p = int(positions[b, t])
            weights = torch.softmax((q[b, t] @ k[b, :p + 1].T) * scale, -1)  # (H, p + 1)
            out[b, t] = weights @ v[b, :p + 1]
            far = p + 1 - window
            if far > 0:
                score = weights[:, :far].amax(0)
                chosen[b][t] = set(score.topk(min(top_k, far)).indices.tolist())
    return out, chosen


def sparse_by_formula(q, k, v, sink, idx, positions, scale, window):
    """Each query over its rows idx and its last window rows, with each
    head's sink in the denominator."""
    B, T, H, c = q.shape
    out = torch.zeros(B, T, H, c)
    for b in range(B):
        for t in range(T):
            p = int(positions[b, t])
            rows = sorted({int(i) for i in idx[b, t] if i >= 0} | set(range(max(0, p - window + 1), p + 1)))
            z = (q[b, t] @ k[b, rows].T) * scale  # (H, rows)
            denominator = z.exp().sum(-1) + sink.exp()
            out[b, t] = (z.exp() / denominator[:, None]) @ v[b, rows]
    return out


def inputs(B=2, T=40, H=3, c=16, N=None, seed=0, dtype=torch.float32, device="cpu"):
    """q (B, T, H, c), k and v (B, N, c), sink (H,) and the positions of a
    pass whose T queries are the table's last ones."""
    generator = torch.Generator().manual_seed(seed)
    N = N or T

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device, dtype).requires_grad_()

    positions = (N - T + torch.arange(T)).expand(B, T).to(device)
    return rand(B, T, H, c), rand(B, N, c), rand(B, N, c), rand(H), positions


def test_exact_attention_and_its_gradients_follow_the_formula(monkeypatch):
    # in blocks of a few queries, so that the blocks' joins are exercised;
    # the pass's queries are the table's last 30 of 50
    monkeypatch.setattr(pytorch_kernels, "BLOCK_VALUES", 4 * 3 * 50)
    q, k, v, _, positions = inputs(T=30, N=50)
    out, chosen = pytorch_kernels.exact_attention(q, k, v, positions, 0.3, 5, 8)
    expected, expected_chosen = exact_by_formula(q, k, v, positions, 0.3, 5, 8)
    assert torch.allclose(out, expected, atol=1e-5)
    assert chosen.shape == (2, 30, 5)
    assert [[set(row[row >= 0].tolist()) for row in query] for query in chosen] == expected_chosen
    weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(1))
    got = torch.autograd.grad((out * weights).sum(), (q, k, v))
    want = torch.autograd.grad((expected * weights).sum(), (q, k, v))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(got, want))


def test_a_query_with_fewer_far_tokens_than_top_k_chooses_them_all():
    # a query at position p has p + 1 - window rows outside its window;
    # the column count is what the table's last query has, the rest -1
    q, k, v, _, positions = inputs(T=12)
    _, chosen = pytorch_kernels.exact_attention(q, k, v, positions, 0.25, 100, 4)
    assert chosen.shape == (2, 12, 8)
    for t in range(12):
        assert set(chosen[0, t][chosen[0, t] >= 0].tolist()) == set(range(max(0, t - 3)))


def test_sparse_attention_and_its_gradients_follow_the_formula(monkeypatch):
    monkeypatch.setattr(pytorch_kernels, "BLOCK_VALUES", 4 * 12 * 16)
    q, k, v, sink, positions = inputs(T=30, N=50)
    generator = torch.Generator().manual_seed(2)
    # rows outside each query's window, -1 here and there
    idx = torch.stack([torch.stack([torch.randperm(int(p) - 4, generator=generator)[:6]
                                    if int(p) >= 10 else torch.full((6,), -1) for p in row]) for row in positions])
    idx[:, ::3, -2:] = -1
    out = pytorch_kernels.sparse_attention(q, k, v, sink, idx, positions, 0.3, 5)
    expected = sparse_by_formula(q, k, v, sink, idx, positions, 0.3, 5)
    assert torch.allclose(out, expected, atol=1e-5)
    weights = torch.randn(out.shape, generator=generator)
    got = torch.autograd.grad((out * weights).sum(), (q, k, v, sink))
    want = torch.autograd.grad((expected * weights).sum(), (q, k, v, sink))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(got, want))


# -- every kernel module that runs here, against the PyTorch form --

def runnable_kernels():
    """The kernel modules that can run on this machine's GPU: TileLang on an
    NVIDIA GPU under Linux with TileLang and nvcc, Triton on any GPU it
    supports. Each is tested, whichever kernels() would choose."""
    device = select_device()  # select_device: AMD GPUs need it
    found = []
    if device.type != "cuda":
        return found
    for name, needs in (("tilelang_kernels", torch.version.cuda and dispatch.nvcc()), ("triton_kernels", True)):
        try:
            module = importlib.import_module(f"wpt_model.{name}") if needs else None
        except ImportError:
            module = None
        if module is not None:
            found.append(module)
    return found


KERNELS = runnable_kernels()
each_kernel = pytest.mark.parametrize("chosen", KERNELS, ids=lambda module: module.NAME)
# (heads, head width, tokens, top k): the recipes' heads and head width over windows where the choice matters,
# with the window and the top k of the model (128 and 1,024) in the last
SHAPES = [(4, 128, 300, 16, 32), (8, 128, 700, 64, 128), (8, 128, 2000, 1024, 128)]


def relative_error(got, expected):
    """The largest difference, as a share of the expected values' largest
    magnitude."""
    return float((got.float() - expected.float()).abs().max() / expected.float().abs().max())


def written(operation, *args):
    """operation(*args), or a skip when the kernel is not written yet."""
    try:
        return operation(*args)
    except NotImplementedError as missing:
        pytest.skip(str(missing))


@each_kernel
@pytest.mark.parametrize("heads, width, T, top_k, window", SHAPES)
def test_the_exact_attention_kernel_gives_the_pytorch_result(chosen, heads, width, T, top_k, window):
    # from BF16 inputs: the output and gradients within 1e-2 of their
    # largest value; each query chose as many rows as the reference, in any
    # order (pytorch_kernels.py), and every row the kernel chose is one the
    # reference weighs within 1e-2 (in log) of its own top_k-th, since BF16
    # scores may order near ties another way
    device = select_device()
    q, k, v, _, positions = inputs(B=2, T=T, H=heads, c=width, dtype=torch.bfloat16, device=device)
    out, rows = written(chosen.exact_attention, q, k, v, positions, width ** -0.5, top_k, window)
    weights = torch.randn(out.shape, device=device, generator=torch.Generator(device).manual_seed(1))
    got = torch.autograd.grad((out.float() * weights).sum(), (q, k, v))
    expected, expected_rows = pytorch_kernels.exact_attention(q, k, v, positions, width ** -0.5, top_k, window)
    want = torch.autograd.grad((expected.float() * weights).sum(), (q, k, v))
    assert out.dtype == q.dtype and rows.shape == expected_rows.shape
    assert relative_error(out, expected) < 1e-2
    assert all(relative_error(a, b) < 1e-2 for a, b in zip(got, want))
    z = torch.einsum("bthc,bnc->bthn", q.float(), k.float()) * width ** -0.5
    z = z.masked_fill(torch.arange(T, device=device) > positions[..., None, None], float("-inf"))
    score = (z - z.logsumexp(-1, keepdim=True)).amax(2)  # (B, T, N): each row's largest log-weight
    kth = score.gather(-1, expected_rows.clamp(min=0)).masked_fill(expected_rows < 0, float("inf")).amin(-1)
    picked = score.gather(-1, rows.clamp(min=0)).masked_fill(rows < 0, float("inf"))
    assert ((rows >= 0).sum(-1) == (expected_rows >= 0).sum(-1)).all() and (picked >= kth[..., None] - 1e-2).all()


@each_kernel
@pytest.mark.parametrize("heads, width, T, top_k, window", SHAPES)
def test_the_sparse_attention_kernel_gives_the_pytorch_result(chosen, heads, width, T, top_k, window):
    # over the rows the reference chose, -1 among them, from BF16 inputs:
    # the output and gradients within 1e-2 of their largest value
    device = select_device()
    q, k, v, sink, positions = inputs(B=2, T=T, H=heads, c=width, dtype=torch.bfloat16, device=device)
    sink = sink.detach().float().requires_grad_()
    with torch.no_grad():
        _, idx = pytorch_kernels.exact_attention(q, k, v, positions, width ** -0.5, top_k, window)
    out = written(chosen.sparse_attention, q, k, v, sink, idx, positions, width ** -0.5, window)
    weights = torch.randn(out.shape, device=device, generator=torch.Generator(device).manual_seed(1))
    got = torch.autograd.grad((out.float() * weights).sum(), (q, k, v, sink))
    expected = pytorch_kernels.sparse_attention(q, k, v, sink, idx, positions, width ** -0.5, window)
    want = torch.autograd.grad((expected.float() * weights).sum(), (q, k, v, sink))
    assert out.dtype == q.dtype and relative_error(out, expected) < 1e-2
    assert all(relative_error(a, b) < 1e-2 for a, b in zip(got, want))


# -- which form runs --

def fake_kernels(fails):
    """A stand-in for a kernel module, whose kernels fail with fails."""
    def kernel(*_):
        raise fails
    return types.SimpleNamespace(NAME="Triton", exact_attention=kernel, sparse_attention=kernel)


def test_installed_kernels_that_cannot_work_stop_instead_of_falling_back(monkeypatch):
    # docs/model.md: which form runs is decided once; TileLang installed
    # without nvcc, or kernels that do not run on this GPU, are an error
    # with its reason, and neither TileLang nor Triton installed means
    # PyTorch. Stand-ins play the installed packages and an NVIDIA GPU
    # under Linux, whatever runs the test: neither the real import nor a GPU
    # is exercised
    gpu = torch.device("cuda")
    monkeypatch.setattr(dispatch, "_kernels", dispatch._UNDECIDED)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(dispatch, "_installed", lambda name: True)
    monkeypatch.setattr(dispatch, "nvcc", lambda: None)
    with pytest.raises(KernelUnavailable, match="nvcc"):
        kernels(gpu)

    monkeypatch.setattr(dispatch, "_installed", lambda name: name == "triton")
    monkeypatch.setitem(sys.modules, "wpt_model.triton_kernels", fake_kernels(RuntimeError("no image")))
    monkeypatch.delattr(wpt_model, "triton_kernels", raising=False)
    monkeypatch.setattr(dispatch, "_try", lambda chosen, device: chosen.exact_attention())
    with pytest.raises(KernelUnavailable, match="Triton kernels do not run on this GPU: RuntimeError: no image"):
        kernels(gpu)

    monkeypatch.setattr(dispatch, "_installed", lambda name: False)
    assert kernels(gpu) is None and kernels(torch.device("cpu")) is None


def test_named_kernels_that_cannot_work_stop_instead_of_another_form(monkeypatch):
    # docs/model.md: a form a run names is the one that runs or the run
    # stops with the reason: kernels not installed, TileLang off an NVIDIA
    # GPU under Linux, kernels without a GPU; pytorch is PyTorch though
    # kernels are installed, and another form named once it is decided is
    # an error too. Stand-ins play the installed packages and the
    # machine, as above
    gpu = torch.device("cuda")
    monkeypatch.setattr(dispatch, "_kernels", dispatch._UNDECIDED)
    monkeypatch.setattr(dispatch, "_installed", lambda name: name == "tilelang")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(KernelUnavailable, match="Triton kernels were asked for, and triton is not installed"):
        kernels(gpu, "triton")
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(KernelUnavailable, match="TileLang kernels run on an NVIDIA GPU under Linux only"):
        kernels(gpu, "tilelang")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.version, "cuda", None)  # an AMD GPU
    with pytest.raises(KernelUnavailable, match="TileLang kernels run on an NVIDIA GPU under Linux only"):
        kernels(gpu, "tilelang")
    with pytest.raises(KernelUnavailable, match="Triton kernels run on a GPU, and this process runs on cpu"):
        kernels(torch.device("cpu"), "triton")

    monkeypatch.setattr(dispatch, "_installed", lambda name: True)
    assert kernels(gpu, "pytorch") is None and kernels(torch.device("cpu"), "pytorch") is None
    with pytest.raises(KernelUnavailable, match="the attention's form is pytorch in this process"):
        kernels(gpu, "triton")
