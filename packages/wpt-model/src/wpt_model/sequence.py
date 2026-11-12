"""Where a pass's tokens sit (docs/model.md, "What a layer passes on"). A
module that reads earlier tokens says once what it passes on to later
tokens (carry.py: Tail, Entries, Fold), and asks the place of its pass
for what came before its first token:

    before(module, name, x)          a Tail's steps, or a Fold's value,
                                     before x's first token; x is the
                                     pass's own steps or changes
    everything(module, name, new)    an Entries': every entry so far, the
                                     pass's own after the earlier ones, so
                                     that token p's entry sits at row p

A place also gives the pass's positions (positions) and whether training
statistics are recorded (collect). A model that loops (docs/model.md,
"Loops") reads the same tokens once per pass, each through at(loop), the
place of that pass: training's places keep what each pass passes on apart, by
(module, name, loop), so a window in parts gives what the whole window
gives; a generation cache keeps one value per module and name, the last
pass's (cache.py).

Four places answer, from that one description:

    Whole    the pass is the whole window: nothing came before
    Carried  one segment of a window trained on one GPU (wpt_training's
             parts.py): what the segment before passed on, as tensors whose
             gradients the runner carries back across the boundary
    Split    one process's slice of a window, or of one part of it,
             split across processes (DeepSeek-V4's contextual
             parallelism, section 3.4.3): each rank gets what the ranks
             before it hand on (the last steps of their slices, their
             changes of a Fold) and every entry by an all-gather; rank 0
             gets what the part before passed on, as Carried does, and the
             part passes on what its last slice ends with
    Reading  generation (cache.py): B rows, each at its own place, reading
             what the cache keeps and keeping it up to date, with undo

Training's places are exact: a window trained in segments or split across
processes gives the loss and gradients of the whole window at once. A
segment or slice holds a multiple of WPT.PART tokens, the window of the
window layers; a window's last segment may be shorter. A tail may be longer
than a slice: a rank then reads the last steps of several ranks before it.
"""

import copy

import torch
import torch.distributed as dist

from .carry import described


class Place:
    """Where a pass's tokens sit. start: the first token's position, an int
    for every row alike, or (B,) for rows at different places."""

    start = 0
    collect = True  # training statistics (expert balancing) are recorded
    loop = 0  # the pass of a looping model these tokens are read in

    def at(self, loop):
        """The place of pass `loop` (from 0) of a looping model: the same
        place, whose values it keeps under that pass."""
        place = copy.copy(self)  # what it keeps (carried, passed, added, a cache's rows) is shared
        place.loop = loop
        return place

    def key(self, module, name):
        return id(module), name, self.loop

    def positions(self, T, device):
        """The pass's T positions: (T,), or (B, T) for rows at different places."""
        steps = torch.arange(T, device=device)
        return self.start + steps if isinstance(self.start, int) else self.start[:, None] + steps


class Whole(Place):
    """The pass is the whole window."""

    def at(self, loop):
        """Nothing to keep apart: nothing came before any pass."""
        return self

    def before(self, module, name, x):
        return described(module, name).start(x, x.shape[0])

    def everything(self, module, name, new):
        return new


WHOLE = Whole()


class Carried(Place):
    """One segment of a window: carried holds what the segment before passed
    on (empty for the first), by key (module, name, loop). After the pass, passed
    holds what this one passes on (each module's tails), and added the
    entries it added to everything (the entries of all segments so far are
    what came before plus each one's added).
    collect: record training statistics in this pass."""

    def __init__(self, offset, carried, collect=True):
        self.start = offset
        self.carried, self.collect = carried, collect
        self.passed, self.added = {}, {}

    def before(self, module, name, x):
        key, item = self.key(module, name), described(module, name)
        earlier = self.carried.get(key)
        if earlier is None:
            earlier = item.start(x, x.shape[0])
        self.passed[key] = item.follow(earlier, x)
        return earlier

    def everything(self, module, name, new):
        key = self.key(module, name)
        described(module, name)
        self.added[key] = new
        earlier = self.carried.get(key)
        return new if earlier is None else torch.cat([earlier.to(new.dtype), new], 1)


class Split(Place):
    """Process rank's slice of a window split across world processes, each
    slice of `length` tokens, in order from token start of the window: the
    whole window, or one part of it when the window runs as consecutive
    parts each split across the processes (wpt_training's parts.py). carried
    holds what the part before passed on (empty for the first part or a
    whole window), by key (module, name, loop), as for Carried; after the pass,
    passed and added hold what this part passes on, the same on every
    process. Every exchange is an all-gather whose backward pass sums each
    slice's gradient over the processes, so the processes' gradients added
    up are the whole window's. Each process builds the same graph
    (rank-dependent values, never rank-dependent operations), so their
    backward passes meet in the same order. collect: record training
    statistics in this pass."""

    def __init__(self, rank, world, length, start=0, carried=None, collect=True, group=None):
        self.rank, self.world, self.group = rank, world, group
        self.start = start + rank * length
        self.collect = collect
        self.carried, self.passed, self.added = carried or {}, {}, {}

    def before(self, module, name, x):
        key, item = self.key(module, name), described(module, name)
        shares = gather(item.share(x), self.group)  # stage 1: what each rank hands the ranks after it
        earlier = self.carried.get(key)
        if earlier is None:  # the window's start: nothing came before
            earlier = item.start(x, x.shape[0])
        earlier, self.passed[key] = item.across(earlier, shares, self.rank)
        return earlier

    def everything(self, module, name, new):
        key = self.key(module, name)
        described(module, name)
        entries = gather(new, self.group).transpose(0, 1).flatten(1, 2)  # stage 2: every rank's entries, in order
        self.added[key] = entries
        earlier = self.carried.get(key)
        return entries if earlier is None else torch.cat([earlier.to(entries.dtype), entries], 1)


class _Gather(torch.autograd.Function):
    """all_gather with a backward pass that sums each slice's gradient over
    the processes (an all-reduce, which every backend has)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.stack(parts)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return grad[dist.get_rank(ctx.group)], None


def gather(x, group=None):
    """(world, *x.shape): every process's x, in order of rank."""
    if not x.is_floating_point():
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.stack(parts)
    return _Gather.apply(x, group)
