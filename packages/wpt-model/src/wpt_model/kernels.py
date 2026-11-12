"""What computes the two attention operations whose cost grows with the
text (docs/model.md, "Attention"):

    exact_attention   an exact layer's attention over every earlier token,
                      which also gives each query's top rows
    sparse_attention  a sparse layer's attention over each query's chosen
                      rows and its last tokens

pytorch_kernels.py states their contract and is their reference. Three
forms compute them, each a module with that contract:

    PyTorch   pytorch_kernels.py, on every device
    Triton    triton_kernels.py, on an NVIDIA or AMD GPU where Triton is
              installed
    TileLang  tilelang_kernels.py, on an NVIDIA GPU under Linux with
              TileLang installed and the CUDA compiler (nvcc)

Which form runs is decided once per process, at the first GPU pass (or
before, by wpt train, wpt eval and wpt chat, which may name it: FORMS).
Unnamed, the form is the first of TileLang and Triton installed for this
GPU, else PyTorch; a later call that names another form is
KernelUnavailable. A kernel form, named or installed, that cannot work (not
installed when named, TileLang off an NVIDIA GPU under Linux or without its
CUDA compiler, no GPU, an import that fails, a kernel that does not compile
or run on this GPU) is KernelUnavailable, which stops the run. A failure
inside a later call is an error like any other, never a switch to another
form. Off a GPU, PyTorch runs.
"""

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import torch

from . import pytorch_kernels
from .layers import compute_dtype

FORMS = ("tilelang", "triton", "pytorch")  # the forms a run may name
_NAMES = {"tilelang": "TileLang", "triton": "Triton"}
_UNDECIDED = object()
_kernels = _UNDECIDED
_form = None  # the form decided, one of FORMS, once _kernels is


def named_form(text):
    """A form a command's --kernels or WPT_KERNELS names (FORMS), as an
    argparse type."""
    if text not in FORMS:
        raise argparse.ArgumentTypeError(f"not one of {', '.join(FORMS)}: {text!r}")
    return text


class KernelUnavailable(RuntimeError):
    """The kernels named, or installed for this GPU, cannot work here: the
    reason says why. Nothing has been computed."""


def kernels(device, form=None):
    """The kernel module that computes the attention on device (NAME,
    exact_attention, sparse_attention), or None for the PyTorch form;
    decided at the first call on a GPU. form, one of FORMS, names the form
    at that call; None chooses it. Raises KernelUnavailable when the kernels
    named or installed cannot work, or when a later call names another form
    than the one decided."""
    global _kernels, _form
    if form not in (None, *FORMS):
        raise ValueError(f"no attention form {form!r}: one of {', '.join(FORMS)}")
    if device.type != "cuda":
        if form in _NAMES:
            raise KernelUnavailable(f"{_NAMES[form]} kernels run on a GPU, and this process runs on {device.type}")
        return None
    if _kernels is _UNDECIDED:
        form = form or ("tilelang" if torch.version.cuda and sys.platform == "linux" and _installed("tilelang")
                        else "triton" if _installed("triton") else "pytorch")
        _kernels, _form = _decide(device, form), form
    elif form not in (None, _form):
        raise KernelUnavailable(f"the attention's form is {_form} in this process, decided at its first GPU "
                                f"pass, and {form} was asked for after it")
    return _kernels


def _decide(device, form):
    if form in _NAMES and not _installed(form):
        raise KernelUnavailable(f"{_NAMES[form]} kernels were asked for, and {form} is not installed")
    if form == "pytorch":
        return None
    if form == "tilelang":
        if not (torch.version.cuda and sys.platform == "linux"):
            raise KernelUnavailable("TileLang kernels run on an NVIDIA GPU under Linux only")
        if nvcc() is None:
            raise KernelUnavailable("TileLang compiles its attention kernels with the CUDA compiler (nvcc), which "
                                    "is not here: install the CUDA Toolkit and put nvcc on PATH or set CUDA_HOME, "
                                    "or name the Triton kernels instead")
        try:
            from . import tilelang_kernels as chosen
        except Exception as error:  # an ImportError, or an error of TileLang's own while it starts
            raise KernelUnavailable(f"TileLang is installed but does not import: {error}") from error
    else:
        try:
            from . import triton_kernels as chosen
        except Exception as error:
            raise KernelUnavailable(f"Triton is installed but does not import: {error}") from error
    try:  # every kernel compiled and run once, forward and backward, so that a GPU they cannot run on stops now
        _try(chosen, device)
    except Exception as error:
        raise KernelUnavailable(f"{chosen.NAME} kernels do not run on this GPU: {type(error).__name__}: {error}"
                                ) from error
    return chosen


def _installed(name):
    """Whether a module is installed, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:  # a module already imported without a spec
        return True


def _try(chosen, device):
    """Both operations once, forward and backward, on a small input of the
    recipes' shapes, with gradients on even when the first GPU pass runs
    under torch.no_grad (generation, evaluation)."""
    with torch.enable_grad():
        _try_both(chosen, device)


def _try_both(chosen, device):
    generator = torch.Generator().manual_seed(0)
    B, T, H, c, window, top_k = 1, 8, 4, 128, 2, 4

    def leaf(*shape):
        return torch.randn(*shape, generator=generator).to(device, torch.bfloat16).requires_grad_()

    q, k, v = leaf(B, T, H, c), leaf(B, T, c), leaf(B, T, c)
    sink = torch.zeros(H, device=device, requires_grad=True)
    positions = torch.arange(T, device=device).expand(B, T)
    out, chosen_rows = chosen.exact_attention(q, k, v, positions, c ** -0.5, top_k, window)
    sparse = chosen.sparse_attention(q, k, v, sink, chosen_rows, positions, c ** -0.5, window)
    (out.float().sum() + sparse.float().sum()).backward()
    torch.cuda.synchronize(device)


def nvcc():
    """The CUDA compiler's path, looked for where TileLang looks (CUDA_HOME,
    CUDA_PATH, PATH, /usr/local/cuda), or None."""
    homes = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    found = [Path(home) / "bin" / "nvcc" for home in homes if home] + [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    return next((str(path) for path in found if path and Path(path).is_file()), None)


def _positions(positions, B):
    """positions (T,) or (B, T) -> (B, T) int64, as the contract reads them."""
    return positions.to(torch.long).expand(B, -1).contiguous()


@torch.compiler.disable
def exact_attention(q, k, v, positions, scale, top_k, window):
    """q (B, T, H, c); k, v (B, N, c), token p at row p; positions (T,) or
    (B, T). Returns (output (B, T, H, c) in the compute dtype (compute_dtype),
    each query's chosen rows (B, T, K), K = chosen_columns(top_k, window, N),
    -1 for none): the top_k rows it weighs most among those at least window
    tokens before it (pytorch_kernels.py)."""
    form = kernels(q.device) or pytorch_kernels
    dtype = compute_dtype(q)
    with torch.autocast(q.device.type, enabled=False):
        return form.exact_attention(q.to(dtype), k.to(dtype), v.to(dtype), _positions(positions, q.shape[0]), scale,
                                    top_k, window)


@torch.compiler.disable
def sparse_attention(q, k, v, sink, idx, positions, scale, window):
    """The output (B, T, H, c), in the compute dtype (compute_dtype), of attention
    over each query's rows idx (B, T, K) and its last window tokens, with
    sink (H,) added to each head's denominator; k, v (B, N, c) as
    exact_attention's."""
    form = kernels(q.device) or pytorch_kernels
    dtype = compute_dtype(q)
    with torch.autocast(q.device.type, enabled=False):
        return form.sparse_attention(q.to(dtype), k.to(dtype), v.to(dtype), sink.float(), idx,
                                     _positions(positions, q.shape[0]), scale, window)
