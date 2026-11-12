"""The generation cache (docs/model.md, "Generation"): what a model keeps of
the tokens it has read, so that the next ones cost only themselves, for B
rows (attempts) at once. WPT.new_cache makes one, WPT.extend reads new
tokens through it, WPT.undo takes back the last tokens of an undoable
extend (rejected draft tokens), and Cache.select copies rows (a prompt read
once, shared by several attempts).

The cache keeps what each layer says it passes on to later tokens
(carry.py), the same description training's segments and splits carry
(sequence.py):

    window layer  the last 127 keys and values (the window holds 128, the
                  new token's included)
    exact layer   the key and value of every token read
    sparse layer  nothing: it reads the exact layer's
    Engram        the last order - 1 canonical ids and the last inputs of
                  its convolution; the tables stay where they are
    In-Place TTT  the tokens of its unfinished chunk (at most 1,023) and
                  its sum of chunk updates: each row its own copy of the
                  changed matrix
    MTP block     its window, like a window layer's, for the pairs it read

Nothing grows with the text but the exact layers' keys and values, one of
each per token and exact layer, so generation goes on past block_size. The
experts keep nothing, but a cache made with routes records each row's
routed experts (R3) as its tokens are read.

New tokens are read by the model's own forward pass, through a Reading:
the place (sequence.py) of B rows each at its own length, which gives each
module what the cache keeps and keeps what the pass leaves. An extend may
give each row its own number of new tokens (counts), the rest of its row
being padding: a tail keeps the last steps of a row's own tokens, and only
a row's own tokens write entries. Undoing goes back to what an undoable
reading started from and keeps the first kept[b] tokens of row b, as each
kind of carried value says (carry.py): tails cut, entries left (one past a
row's tokens is never read, and is written again when the row reaches
it).

A model that loops (docs/model.md, "Loops") reads a call's tokens once per
pass, and the cache keeps one value per module and name, the last pass's.
Every pass starts from what the cache held when the call began, and reads
its own entries and tails for the call's tokens: a prompt read in one call
gives what the model's forward pass gives, and a token read later reads,
in every pass, the last pass's values of the earlier tokens.
"""

import torch

from .carry import Tail, described
from .experts import Routes
from .sequence import Place


class Rows:
    """One stream of B rows' carried values: the text the model reads, or
    the (hidden state, next token) pairs its multi-token prediction block
    reads. lengths: the tokens each row has read; values: what each module
    passes on, by (module, name), made at the first reading; pending: what
    undo needs after an undoable reading, None otherwise."""

    def __init__(self, rows, device):
        self.lengths = torch.zeros(rows, dtype=torch.long, device=device)
        self.values = {}
        self.pending = None

    def reading(self, counts, undoable):
        return Reading(self, counts, undoable)

    def undo(self, kept):
        """Back to the state before the last undoable reading, then its first
        kept[b] tokens of row b again."""
        if self.pending is None:
            raise ValueError("no undoable extend to undo")
        lengths, saved = self.pending
        for key, (item, before, new) in saved.items():
            self.values[key] = item.undo(before, new, kept)
        self.lengths, self.pending = lengths + kept, None


class Reading(Place):
    """The place of new tokens read through rows (Rows): row b's first
    counts[b] tokens follow its lengths[b] tokens, the rest are padding.
    The model's forward pass reads through it; finish then records the
    tokens read. With undoable, it keeps what Rows.undo needs."""

    collect = True  # a cache made with routes records them (balancing counts only in training mode)

    def __init__(self, rows, counts, undoable):
        self.rows, self.counts = rows, counts
        self.start = rows.lengths
        self.saved = {} if undoable else None
        self.began = {}  # what each tail or fold held when the reading began, which every pass starts from

    def key(self, module, name):
        return id(module), name

    def before(self, module, name, x):
        key, item = self.key(module, name), described(module, name)
        if key not in self.began:
            held = self.rows.values.get(key)
            self.began[key] = item.start(x, x.shape[0]) if held is None else held
        earlier = self.began[key]
        if isinstance(item, Tail):  # a tail in the pass's dtype; a fold stays FP32
            earlier = earlier.to(x.dtype)
        self.rows.values[key] = item.undo(earlier, x, self.counts)  # what each row's own tokens leave
        if self.saved is not None:
            self.saved[key] = (item, earlier, x)
        return earlier

    def everything(self, module, name, new):
        """Writes the entries of each row's own tokens after its earlier
        ones; returns every row's entries so far (B, N, ...), N the most any
        row has, a row's entries past its own number stale."""
        key = self.key(module, name)
        described(module, name)
        first, made = self.start, self.counts
        need = int((first + made).max()) if len(first) else 0
        held = self.rows.values.get(key)
        if held is None:
            held = new.new_zeros(len(first), 0, *new.shape[2:])
        if need > held.shape[1]:
            grown = max(need, 2 * held.shape[1])
            held = torch.cat([held, held.new_zeros(len(first), grown - held.shape[1], *held.shape[2:])], 1)
        row, j = (torch.arange(new.shape[1], device=new.device)[None] < made[:, None]).nonzero(as_tuple=True)
        held = held.to(new.dtype)
        held[row, first[row] + j] = new[row, j]
        self.rows.values[key] = held
        return held[:, :need]

    def finish(self):
        """The rows have read their tokens."""
        rows = self.rows
        rows.pending = (rows.lengths, self.saved) if self.saved is not None else None
        rows.lengths = rows.lengths + self.counts


class Cache:
    """B rows' cache of one model (WPT.new_cache): text, what its layers
    keep of the tokens each row has read; mtp, what its multi-token
    prediction block keeps of the pairs it has read; routes: each row's
    Routes, when the cache records them."""

    def __init__(self, model, rows, routes=False):
        device = model.embedding.weight.device
        self.text, self.mtp = Rows(rows, device), Rows(rows, device)
        self.routes = [Routes() for _ in range(rows)] if routes else None

    @property
    def lengths(self):
        """The tokens each row has read."""
        return self.text.lengths

    @property
    def rows(self):
        return len(self.text.lengths)

    def select(self, index):
        """A new cache of the rows index names (a row may come several times),
        copied: the copies change apart from each other and from this one."""
        if self.text.pending is not None or self.mtp.pending is not None:
            raise ValueError("an undoable extend is pending: undo it before copying rows")
        index = torch.as_tensor(index, dtype=torch.long, device=self.lengths.device)
        copy = object.__new__(Cache)
        copy.text, copy.mtp = (_selected(rows, index) for rows in (self.text, self.mtp))
        copy.routes = None if self.routes is None else [self.routes[i].copy() for i in index.tolist()]
        return copy


def _selected(rows, index):
    copy = Rows(len(index), rows.lengths.device)
    copy.lengths = rows.lengths[index]
    copy.values = {key: value[index] for key, value in rows.values.items()}
    return copy


class Recorder:
    """Records the routed experts of each row's real tokens (not its
    padding) into its own Routes; Experts.route calls record."""

    def __init__(self, routes, counts):
        self.routes, self.counts = routes, counts

    def record(self, experts, chosen):
        chosen = chosen.view(len(self.routes), -1, chosen.shape[-1])
        for routes, row, count in zip(self.routes, chosen, self.counts):
            routes.append(experts, row[:count])

