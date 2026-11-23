"""Whether a stage of text windows keeps every activation or recomputes
its layers, when the run leaves it to the runner (docs/training.md,
"Recomputing layers").

The choice never tries the pass and catches an out-of-memory error: on
Windows the NVIDIA driver does not raise one, it spills into the
machine's memory, and the machine stops answering. It is made from a
measure instead. Two probe passes, forward and backward without
recomputation, on one window of PROBE_TOKENS tokens (no longer than the
stage's windows) and one of half as many, give the memory a pass takes per token and the part that does not
grow with the tokens (the head's gradient, the loss's buffers). The pass
of the stage is extrapolated from them, its activations given MARGIN more
(the memory of a pass follows its tokens, whether they come as more
windows or longer ones, and for the recipes' shapes the extrapolation
alone stood 4 % to 27 % above the peak of a pass of 32,768 tokens), the
optimizer's state added while it does not exist yet (two FP32 numbers per
trained weight, as AdamW keeps), and the whole compared with the memory
the device has left, free or held by PyTorch, less HEADROOM of the
device's whole memory (the CUDA context, fragments of the allocator).

A probe leaves the run as it was: it reads no weight's gradient into
.grad, changes no weight and no optimizer state, collects no statistic
(expert balancing), runs uncompiled (a segment's place, never the
recorded graphs), and puts back every random stream it may draw from.

choose is the rule on numbers; measure gives it a device's. Off a CUDA
device, nothing measures memory, and a stage keeps every activation.
"""

from dataclasses import dataclass

import torch

PROBE_TOKENS = 2048  # the longer probe window; the other holds half as many
MARGIN = 0.05  # more memory per token than the probes measured
HEADROOM = 0.1  # of the device's whole memory, never planned for
GIB = 2 ** 30


@dataclass(frozen=True)
class Choice:
    """A stage's choice and the numbers it was made from, in bytes: the
    memory PyTorch holds for tensors now, the optimizer's state to come,
    the probes' part that does not grow with the tokens and their bytes per
    token, the tokens of one pass, the peak foreseen for that pass, and the
    memory it may reach."""
    recompute: bool
    allocated: int
    state: int
    fixed: float
    per_token: float
    tokens: int
    foreseen: float
    usable: float

    def text(self) -> str:
        """The log line's words: what was chosen and why."""
        return (f"{'recompute the layers' if self.recompute else 'every activation kept'}: a pass of "
                f"{self.tokens:,} tokens foreseen at {self.foreseen / GIB:.1f} GiB without recomputing ("
                f"{self.allocated / GIB:.1f} GiB held now, {self.state / GIB:.1f} GiB of optimizer state to come, "
                f"{self.fixed / GIB:.2f} GiB plus {self.per_token / 2 ** 10:.0f} KiB per token measured, "
                f"{MARGIN:.0%} more per token), against {self.usable / GIB:.1f} GiB usable")


def choose(allocated, state, fixed, per_token, tokens, free, reserved, total) -> Choice:
    """The choice for a pass of `tokens` tokens: every activation kept when
    its foreseen peak fits what the device has left (free: not held by any
    process; reserved: held by this process's allocator, of which
    `allocated` holds tensors) less HEADROOM of its total."""
    foreseen = allocated + state + max(0.0, fixed) + per_token * tokens * (1 + MARGIN)
    usable = free + reserved - HEADROOM * total
    return Choice(foreseen > usable, allocated, state, max(0.0, fixed), per_token, tokens, foreseen, usable)


def state_to_come(optimizer) -> int:
    """The bytes of the state a torch optimizer has yet to make: two FP32
    numbers per trained weight that has none."""
    return sum(8 * p.numel() for group in optimizer.param_groups for p in group["params"]
               if p.requires_grad and not optimizer.state.get(p))


def reader(device):
    """What measures device's memory: a DeviceMemory for a CUDA device, else None."""
    return DeviceMemory(device) if device.type == "cuda" else None


class DeviceMemory:
    """A CUDA device's memory, as PyTorch's allocator and the driver count it."""

    def __init__(self, device):
        self.device = device

    def allocated(self) -> int:
        return torch.cuda.memory_allocated(self.device)

    def start(self):
        """Peaks count from here."""
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self) -> int:
        torch.cuda.synchronize(self.device)
        return torch.cuda.max_memory_allocated(self.device)

    def left(self) -> tuple[int, int, int]:
        """(free, reserved, total): the memory no process holds, this
        process's allocator holds, and the device has."""
        free, total = torch.cuda.mem_get_info(self.device)
        return free, torch.cuda.memory_reserved(self.device), total


def probe(model, tokens, memory, generator) -> int:
    """The peak bytes above what is held now of one training pass, forward
    and backward without recomputation, over one window of `tokens`
    synthetic tokens; leaves no trace (the module's docstring)."""
    device = next(model.parameters()).device
    trained = [p for p in model.parameters() if p.requires_grad]
    ids = torch.randint(model.config.vocab_size, (1, tokens + 2), generator=generator).to(device)
    x, y, after = ids[:, :-2], ids[:, 1:-1], ids[:, 2:]
    cpu, gpus = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    recompute, replayed = model.recompute, model.replayed
    try:
        model.recompute = False
        base = memory.allocated()
        memory.start()
        place = model.carried(0, {}, collect=False)  # uncompiled, and no statistic collected
        next_sum, ahead_sum = model(x, next_tokens=y, targets=(y, after), sequence=place)
        grads = torch.autograd.grad(next_sum + ahead_sum, trained, allow_unused=True)
        found = memory.peak() - base
        del next_sum, ahead_sum, grads
        return found
    finally:
        model.recompute, model.replayed = recompute, replayed
        torch.set_rng_state(cpu)
        if gpus is not None:
            torch.cuda.set_rng_state_all(gpus)


def measure(model, optimizer, tokens, window, memory) -> Choice:
    """The choice for a pass of `tokens` tokens in windows of `window` of
    model on memory's device, from two probe passes no longer than a window
    (the module's docstring)."""
    longer = min(PROBE_TOKENS, window)
    shorter = max(1, longer // 2)
    generator = torch.Generator().manual_seed(0)
    peaks = {length: probe(model, length, memory, generator) for length in {shorter, longer}}
    per_token = (peaks[longer] - peaks[shorter]) / (longer - shorter) if shorter != longer else 0.0
    fixed = peaks[shorter] - per_token * shorter
    if per_token <= 0:  # one probe, or two the allocator's rounding put out of order: all of it grows
        fixed, per_token = 0.0, peaks[longer] / longer
    free, reserved, total = memory.left()
    return choose(memory.allocated(), state_to_come(optimizer), fixed, per_token, tokens, free, reserved, total)
