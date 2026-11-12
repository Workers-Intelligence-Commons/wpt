"""Small parts several sublayers share."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .carry import Tail

EPS = 1e-6
CONV_KERNEL = 4


def compute_dtype(tensor):
    """The dtype a pass computes in: autocast's (BF16 on a GPU) when it is on
    for the tensor's device, else the tensor's own."""
    device = tensor.device.type
    return torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else tensor.dtype


class ShortConv(nn.Module):
    """A depthwise causal convolution over time: each channel sees its own
    last `kernel` values, `dilation` tokens apart. (B, T, C) -> (B, T, C).
    earlier: the `reach` inputs before x (B, reach, C), zeros when None
    (before the text). It passes on its last `reach` inputs (carry.py),
    "inputs"."""

    def __init__(self, channels, kernel=CONV_KERNEL, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, dilation=dilation, bias=False)

    @property
    def reach(self):
        """How many inputs before a token its output reads."""
        return (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]

    def passes_on(self):
        return {"inputs": Tail(self.reach)}

    def forward(self, x, earlier=None):
        if earlier is None:
            earlier = x.new_zeros(x.shape[0], self.reach, x.shape[2])
        return self.conv(torch.cat([earlier.to(x.dtype), x], 1).transpose(1, 2)).transpose(1, 2)


def situ_glu(gate, up):
    """SiTU-GLU (K3) of the two branches' pre-activations: each branch is
    capped by a scaled tanh, so one unit's output never exceeds 100."""
    return 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))


class FeedForward(nn.Module):
    """One SiTU-GLU feed-forward block: width -> hidden -> width."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(situ_glu(self.gate(x), self.up(x)))


def expert_forward(x, w_gate, w_up, w_down):
    """One routed expert, whose weights are slices of the stacked tensors."""
    return F.linear(situ_glu(F.linear(x, w_gate), F.linear(x, w_up)), w_down)


def grouped_experts(x, ends, w_gate, w_up, w_down):
    """All routed experts at once: x holds the tokens grouped by expert in
    expert order, and expert j's tokens end at row ends[j] (int32; an expert
    with no tokens ends where the one before it does). The weights are the
    stacked tensors, cast to x's dtype, which must be BF16 on a CUDA device.
    Each width must be a multiple of 8, so that every row is a multiple of
    16 bytes. The products are PyTorch's grouped ones, or grouped.py's on a
    GPU PyTorch's have no kernel for, which neither waits on the host."""
    w_gate, w_up, w_down = (w.to(x.dtype).transpose(1, 2) for w in (w_gate, w_up, w_down))
    product = _grouped_product(x.device)
    hidden = situ_glu(product(x, w_gate, ends), product(x, w_up, ends))
    return product(hidden, w_down, ends)


def _grouped_mm(a, b, ends):
    return F.grouped_mm(a, b, offs=ends)


_products = {}  # device -> the grouped product that runs there


def _grouped_product(device):
    """The grouped matrix product for device, decided once per device:
    PyTorch's where Triton does not import, as before it."""
    if device not in _products:
        try:
            from . import grouped
        except ImportError:
            _products[device] = _grouped_mm
        else:
            _products[device] = grouped.grouped_mm if grouped.needed(device) else _grouped_mm
    return _products[device]
