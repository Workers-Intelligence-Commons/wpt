"""A compiled training pass over whole windows as two recorded CUDA graphs
(docs/training.md, "Compiling the passes"): the forward pass up to the two
summed losses, and its backward pass. Recorded once per shape of the
windows, they replay the pass's GPU work, its many small kernels launched
by the GPU itself, so the CPU's time per window no longer grows with them.
torch.cuda.make_graphed_callables records them, after a few passes that
it runs as they are; the pass is the model's own (WPT.losses).

Everything a replay must redo is GPU work with no wait on the host: the
attention's choice of tokens, the routed experts' grouped products
(grouped.py), the fused head and loss (head_loss.py). What lives on past
a replay is kept apart from the graph's memory, which the next replay
writes over:

    losses     the two sums come back as copies
    gradients  a parameter takes a copy of its first gradient of an update
               (model.py's _kept), and adds the later windows' to it
    counts     each routed layer's balancing histogram is one tensor kept
               here, which every replay adds to; before a replay it takes
               over the histogram the layer holds, or starts at zero when
               balancing took it (Experts.balance)

Recording runs the pass several times on the window it is given; the
counts they add are taken back, so balancing counts that window once.
Recording again, for another shape or another set of parameters that
train, drops the graphs of the shape before and their memory.

Open to review: the graphs' memory is held between windows, as large as
one pass's peak; a pass whose shape changes every window (none in
training's whole windows) would record every time.
"""

import torch
import torch.nn as nn

from .experts import BINS


class _Pass(nn.Module):
    """The model's pass to its two summed losses, as a module whose
    parameters are the model's (what make_graphed_callables records)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx, next_tokens, targets, targets_after):
        return self.model.losses(idx, next_tokens, (targets, targets_after))


class WindowGraphs:
    """The recorded passes of one model, the last shape's."""

    def __init__(self, model):
        self.model = model
        self.key = None
        self.graphed = None
        self.counts = {}  # each routed layer's Experts -> the histogram the replays add to

    def losses(self, idx, next_tokens, targets):
        """The model's two summed losses over windows idx (as WPT.losses),
        through the graphs of their shape, recorded first if needed."""
        inputs = (idx, next_tokens, *targets)
        key = (tuple(t.shape for t in inputs), idx.device,
               tuple(p.requires_grad for p in self.model.parameters()))
        if key != self.key:
            self._record(inputs)
            self.key = key
        self._hand_over_counts()
        first, after = self.graphed(*inputs)
        return first.clone(), after.clone()

    def _record(self, inputs):
        self.key, self.graphed, self.counts = None, None, {}
        torch.cuda.empty_cache()  # the graphs before and the memory the passes before kept go back to the GPU
        held = {}
        for layer in self.model.layers:
            experts = layer.experts
            if not experts.hashed:
                held[experts] = experts.histogram
                self.counts[experts] = torch.zeros(experts.n, BINS, dtype=torch.long, device=inputs[0].device)
                experts.histogram = self.counts[experts]
        # the parameters' gradient nodes last ran on the default stream, and the recording runs on its own
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            # copies of the inputs: the graphs read their inputs from these, and each replay copies new ones in
            self.graphed = torch.cuda.make_graphed_callables(_Pass(self.model), tuple(t.clone() for t in inputs),
                                                             allow_unused_input=True)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)
        for experts, histogram in held.items():  # the recording's own counts taken back
            experts.histogram = histogram

    def _hand_over_counts(self):
        for experts, kept in self.counts.items():
            if experts.histogram is None:
                kept.zero_()
            elif experts.histogram is not kept:
                kept.copy_(experts.histogram)
            experts.histogram = kept
