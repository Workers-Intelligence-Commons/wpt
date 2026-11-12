"""What a layer passes on to later tokens (docs/model.md, "What a layer
passes on"), said once per module. Every module that reads earlier tokens
has passes_on(): {name: what}, each one of

    Tail(steps, fill)  the last `steps` time steps of a value each token
                       has: a window layer's keys and values, Engram's
                       convolution inputs and canonical ids, the tokens of
                       In-Place TTT's unfinished chunk; before the text,
                       fill's value
    Entries()          one entry per token, every one kept: an exact
                       layer's keys and values
    Fold()             a running value that changes at some tokens only:
                       In-Place TTT's sum of chunk updates since the
                       document began

Training's places (sequence.py: a window in segments, or split across
processes) carry these from one part of a window to the next, and the
generation cache (cache.py) keeps them from one call to the next; both read
what comes before the text, what a pass leaves (follow), what it leaves
when each process holds one slice (across), and how to take tokens back
(undo) from here.
"""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Tail:
    """The last `steps` time steps of a value each token has (B, T, ...).
    fill(zeros) gives the steps before the text from zeros of their shape;
    zeros when None."""
    steps: int
    fill: Callable | None = None

    def start(self, x, rows):
        """What comes before the text, for rows rows of values like x."""
        empty = x.new_zeros(rows, self.steps, *x.shape[2:])
        return empty if self.fill is None else self.fill(empty)

    def follow(self, earlier, x):
        """The tail once x's steps follow the tail earlier."""
        return torch.cat([earlier.to(x.dtype), x], 1)[:, -self.steps:]

    def share(self, x):
        """What one process's slice x hands the others: its last steps (all
        of it when it is shorter than the tail)."""
        return x[:, max(0, x.shape[1] - self.steps):]

    def across(self, earlier, shares, rank):
        """(what comes before slice rank, what the last slice leaves), given
        earlier, the tail before the first slice, and shares (world, B, m,
        ...), every slice's share in order. The same operations on every
        process: only the place read depends on rank."""
        found = torch.cat([earlier.to(shares.dtype), *shares], 1)
        return found.narrow(1, rank * shares.shape[2], self.steps), found[:, -self.steps:]

    def undo(self, before, new, kept):
        """The tail once the first kept[b] of row b's new steps follow the
        tail before them."""
        return last(torch.cat([before, new], 1), kept, self.steps)


@dataclass(frozen=True)
class Entries:
    """One entry per token, token p's at row p; every one is kept. Taking
    tokens back needs nothing: an entry past a row's tokens is never read,
    and is written again when the row reaches it."""


@dataclass(frozen=True)
class Fold:
    """A running value (B, K) that a pass changes at some of its tokens,
    each change an element of x (B, n, 2 + K): the offset of the token,
    within the pass, after which it applies; r, 1 when the value starts
    over from zero first, else 0; and d, what is then added. Before the
    text, zeros (FP32)."""

    def start(self, x, rows):
        return torch.zeros(rows, x.shape[-1] - 2, dtype=torch.float32, device=x.device)

    def follow(self, earlier, x):
        """The value after every change of x, in order."""
        value = earlier.float()
        for element in x.unbind(1):
            value = self.step(value, element[:, 1:2], element[:, 2:])
        return value

    def step(self, value, restart, change):
        """The value (B, K) after one change: restart (B, 1) and change (B,
        K) are the change's r and d, given apart."""
        return torch.addcmul(change.float(), value, 1 - restart.float())

    def share(self, x):
        return x

    def across(self, earlier, shares, rank):
        values = [earlier.float()]
        for x in shares:
            values.append(self.follow(values[-1], x))
        return torch.stack(values[:-1])[rank], values[-1]

    def undo(self, before, new, kept):
        """The value once only the changes at row b's first kept[b] tokens
        follow before."""
        applied = (new[..., 0] < kept[:, None]).to(new.dtype)[..., None]
        return self.follow(before, torch.cat([new[..., :1], new[..., 1:] * applied], -1))


def described(module, name):
    """What module's passes_on() says it passes on under name."""
    found = module.passes_on().get(name)
    if found is None:
        raise KeyError(f"{type(module).__name__} passes on no {name!r}")
    return found


def last(rows, counts, size):
    """rows (B, size + n, ...) -> (B, size, ...): for each row b, the `size`
    rows that end where its first counts[b] rows after the first `size` end."""
    index = counts[:, None] + torch.arange(size, device=rows.device)
    return rows[torch.arange(len(rows), device=rows.device)[:, None], index]
