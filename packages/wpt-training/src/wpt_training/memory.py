"""How a stage of text windows runs its passes, when the run leaves it to
the runner (docs/training.md, "Recomputing layers"): every activation kept
or the layers recomputed, how many windows in one pass, and whole windows
or segments.

The choice never tries the pass and catches an out-of-memory error: on
Windows the NVIDIA driver does not raise one, it spills into the
machine's memory, and the machine stops answering. It is made from a
measure instead. Two probe passes, forward and backward without
recomputation, on one window of PROBE_TOKENS tokens (no longer than the
stage's windows) and one of half as many, give the memory a pass takes per
token and the part that does not grow with the tokens (the head's
gradient, the loss's buffers); two more, with recomputation, give the same
two numbers for a pass that recomputes its layers. A pass is extrapolated
from them, its activations given MARGIN more (the memory of a pass follows
its tokens, whether they come as more windows or longer ones, and for the
recipes' shapes the extrapolation without recomputation alone stood 4 % to
27 % above the peak of a pass of 32,768 tokens), the optimizer's state
added while it does not exist yet (two FP32 numbers per trained weight, as
AdamW keeps), and the whole compared with the memory the device has left,
free or held by PyTorch, less HEADROOM of the device's whole memory (the
CUDA context, fragments of the allocator).

The first of these that is foreseen to fit is chosen: the whole pass with
every activation kept; the whole pass with the layers recomputed; fewer
windows per pass, the largest number that divides the process's share of
the batch, with the layers recomputed; one window per pass in segments
(parts.py), the longest that fits, with the layers recomputed. The number
of windows per pass and the segment's length are chosen only where the
options name none. When nothing fits, the choice says so (Choice.fits) and
the stage does not start. A segment is foreseen as a pass of its tokens:
what the segments hand on, which grows with the window, is not counted.

A probe leaves the run as it was: it reads no weight's gradient into
.grad, changes no weight and no optimizer state, collects no statistic
(expert balancing), runs uncompiled (a segment's place, never the
recorded graphs), and puts back every random stream it may draw from.

choose is the rule on numbers; measure gives it a device's. Off a CUDA
device, nothing measures memory, and a stage keeps every activation.
"""

from dataclasses import dataclass, replace

import torch

PROBE_TOKENS = 2048  # the longer probe window; the other holds half as many
MARGIN = 0.05  # more memory per token than the probes measured
HEADROOM = 0.1  # of the device's whole memory, never planned for
GIB = 2 ** 30


@dataclass(frozen=True)
class Choice:
    """A stage's choice and the numbers it was made from, in bytes: whether
    it recomputes its layers, the windows of one pass in one process, the
    length of the segments the rule chose (0: none, the windows run as the
    options say), whether that pass is foreseen to fit; the memory PyTorch
    holds for tensors now, the optimizer's state to come, the probes' part
    that does not grow with the tokens and their bytes per token, without
    recomputation then with it, the tokens of the pass the options ask for
    (the process's share of the batch, or the option's windows, each whole
    or a segment of the option's length), the tokens each of its windows
    brings, and the memory a pass may reach."""
    recompute: bool
    windows: int
    segment: int
    fits: bool
    allocated: int
    state: int
    fixed: float
    per_token: float
    recomputed_fixed: float
    recomputed_per_token: float
    tokens: int
    length: int
    usable: float

    def peak(self, recompute: bool, tokens: int) -> float:
        """The peak foreseen for a pass of `tokens` tokens."""
        fixed, per_token = ((self.recomputed_fixed, self.recomputed_per_token) if recompute
                            else (self.fixed, self.per_token))
        return self.allocated + self.state + fixed + per_token * tokens * (1 + MARGIN)

    @property
    def foreseen(self) -> float:
        """The peak foreseen for the pass the options ask for, without recomputation."""
        return self.peak(False, self.tokens)

    @property
    def pass_tokens(self) -> int:
        """The tokens of the chosen pass."""
        return self.windows * (self.segment or self.length)

    def text(self) -> str:
        """The log line's words: what was chosen and why."""
        fewer = self.pass_tokens < self.tokens
        way = ("no pass fits the device's memory" if not self.fits
               else "recompute the layers" if self.recompute else "every activation kept")
        if fewer:
            way += ((", down to " if not self.fits else ", ")
                    + f"{self.windows} window{'' if self.windows == 1 else 's'} per pass"
                    + (f" in segments of {self.segment:,} tokens" if self.segment else ""))
        text = (f"{way}: a pass of {self.tokens:,} tokens foreseen at {self.foreseen / GIB:.1f} GiB without "
                f"recomputing ({self.allocated / GIB:.1f} GiB held now, {self.state / GIB:.1f} GiB of optimizer "
                f"state to come, {self.fixed / GIB:.2f} GiB plus {self.per_token / 2 ** 10:.0f} KiB per token "
                f"measured, {MARGIN:.0%} more per token)")
        if self.recompute:
            text += (f" and at {self.peak(True, self.tokens) / GIB:.1f} GiB recomputing "
                     f"({self.recomputed_fixed / GIB:.2f} GiB plus {self.recomputed_per_token / 2 ** 10:.0f} KiB per "
                     "token measured)")
        text += f", against {self.usable / GIB:.1f} GiB usable"
        if fewer:
            text += (f"; a pass of {self.pass_tokens:,} tokens foreseen at "
                     f"{self.peak(True, self.pass_tokens) / GIB:.1f} GiB recomputing")
        return text


def choose(allocated, state, kept, recomputed, share, length, free, reserved, total, windows=0, segment=0,
           part=0) -> Choice:
    """The choice for a stage whose batch gives this process `share`
    windows of `length` tokens each, from the first that is foreseen to fit
    what the device has left (free: not held by any process; reserved: held
    by this process's allocator, of which `allocated` holds tensors) less
    HEADROOM of its total: the whole pass with every activation kept, the
    whole pass recomputing, fewer windows per pass recomputing (the largest
    number that divides the share), one window per pass in segments
    recomputing (the longest multiple of `part` tokens shorter than a
    window). kept and recomputed: (the part that does not grow with the
    tokens, the bytes per token) of a pass without recomputation and with
    it. windows and segment: the options', which the rule keeps; 0 leaves
    each to the rule. part: 0 when the stage cannot run in segments. When
    nothing fits, fits is False and the choice holds the smallest pass."""
    asked = min(windows or share, share)
    whole = min(length, segment) if segment else length
    usable = free + reserved - HEADROOM * total
    choice = Choice(False, asked, 0, True, allocated, state, max(0.0, kept[0]), kept[1], max(0.0, recomputed[0]),
                    recomputed[1], asked * whole, whole, usable)
    if choice.foreseen <= usable:
        return choice
    choice = replace(choice, recompute=True)
    if choice.peak(True, choice.tokens) <= usable:
        return choice
    for fewer in () if windows else [count for count in range(asked - 1, 0, -1) if asked % count == 0]:
        if choice.peak(True, fewer * whole) <= usable:
            return replace(choice, windows=fewer)
    count = asked if windows else 1
    lengths = range((whole - 1) // part * part, 0, -part) if part and not segment else ()
    for tokens in lengths:
        if choice.peak(True, count * tokens) <= usable:
            return replace(choice, windows=count, segment=tokens)
    return replace(choice, windows=count, segment=min(lengths, default=0), fits=False)


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


def probe(model, tokens, memory, generator, recompute=False) -> int:
    """The peak bytes above what is held now of one training pass, forward
    and backward, with its layers recomputed or not, over one window of
    `tokens` synthetic tokens; leaves no trace (the module's docstring)."""
    device = next(model.parameters()).device
    trained = [p for p in model.parameters() if p.requires_grad]
    ids = torch.randint(model.config.vocab_size, (1, tokens + 2), generator=generator).to(device)
    x, y, after = ids[:, :-2], ids[:, 1:-1], ids[:, 2:]
    cpu, gpus = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    recomputes, replayed = model.recompute, model.replayed
    try:
        model.recompute = recompute
        base = memory.allocated()
        memory.start()
        place = model.carried(0, {}, collect=False)  # uncompiled, and no statistic collected
        next_sum, ahead_sum = model(x, next_tokens=y, targets=(y, after), sequence=place)
        grads = torch.autograd.grad(next_sum + ahead_sum, trained, allow_unused=True)
        found = memory.peak() - base
        del next_sum, ahead_sum, grads
        return found
    finally:
        model.recompute, model.replayed = recomputes, replayed
        torch.set_rng_state(cpu)
        if gpus is not None:
            torch.cuda.set_rng_state_all(gpus)


def _per_token(peaks) -> tuple[float, float]:
    """(the part that does not grow with the tokens, the bytes per token)
    of probe passes' peaks by their tokens."""
    shorter, longer = min(peaks), max(peaks)
    per_token = (peaks[longer] - peaks[shorter]) / (longer - shorter) if shorter != longer else 0.0
    if per_token <= 0:  # one probe, or two the allocator's rounding put out of order: all of it grows
        return 0.0, peaks[longer] / longer
    return peaks[shorter] - per_token * shorter, per_token


def measure(model, optimizer, share, length, memory, windows=0, segment=0, part=0) -> Choice:
    """The choice for a stage that gives this process `share` windows of
    `length` tokens, of model on memory's device, from probe passes no
    longer than a window or the options' segment; windows, segment and
    part as choose reads them (the module's docstring)."""
    longer = min(PROBE_TOKENS, length, segment or length)
    lengths = sorted({max(1, longer // 2), longer})
    generator = torch.Generator().manual_seed(0)
    kept, recomputed = (_per_token({tokens: probe(model, tokens, memory, generator, recompute) for tokens in lengths})
                        for recompute in (False, True))
    free, reserved, total = memory.left()
    return choose(memory.allocated(), state_to_come(optimizer), kept, recomputed, share, length, free, reserved,
                  total, windows, segment, part)
