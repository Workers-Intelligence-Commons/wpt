"""In-Place Test-Time Training (arXiv 2604.06169; docs/model.md, "In-Place
TTT"): in every EVERY-th layer, the down projection of the shared experts
changes while the model reads a text.

The two shared experts' down projections side by side are one matrix W
(d_model x F, F = 2 x shared_width), and their hidden states side by side
are z (F). The text is cut into chunks of CHUNK tokens, counted from its
first token. Each chunk reads the matrix as the chunks before it left it,
then changes it:

    output of token t     z_t W^T, with W = W_0 + RATE S, S the sum of
                          the updates since the document began
    update of a chunk     d = sum over its tokens of v_t z_t^T,
                          v = W_target conv(e): a depthwise convolution of
                          KERNEL taps over the token embeddings e, zero
                          beyond the chunk, then a learned projection

The loss is the paper's LM-aligned one, minus the inner product of z W^T
and v, whose one gradient step is that update. The convolution starts at
zero and W_target as a diagonal of N(0, TARGET_INIT^2) values, the paper's
initialization, so the model starts as one without TTT; both are trained
with the model.

Documents: an end-of-document token (END_OF_DOCUMENT) closes its
document, and the matrix returns to W_0 for the next one. A token reads
S only when no document ended in its chunk before it; a chunk's update
counts only its tokens after the last end in it, the convolution reading
none of the earlier ones, and S starts over at a chunk that holds an end.

What it passes on to later tokens (carry.py): the z, embeddings and
end-of-document marks of the last CHUNK - 1 tokens ("tokens", a Tail),
from which a chunk's update is computed once its last token is read, and S
("state", a Fold of the chunks' updates), so segments, slices split across
processes and the generation cache give what the whole text gives, and
each cached row holds its own matrix.

How it runs: a pass computes the updates of the chunks it completes in one
batch, S at each chunk start by a running sum (the paper's prefix sum), and
the change RATE z S^T of each run of tokens that reads the same S in one
batched product, so its cost is about two down projections' more.

Open to review: CHUNK (the issue's "about 1,000 tokens"; the paper reports
512 to 1024 best), KERNEL and the centred convolution, RATE and the layers
(1, 7, 13, ...: every sixth from the first) are the published code's; the
paper says "causal padding" and "every sixth layer". The chunks are counted
from the text's first token, not from each document's. The matrix is the
shared experts' down projection, since WPT's feed-forward is a mixture of
experts. TARGET_INIT is the published code's initializer range.
"""

import torch
import torch.nn as nn

from .carry import Fold, Tail
from .sequence import WHOLE

EVERY = 6           # layers 1, 1 + EVERY, 1 + 2 EVERY, ... (from 1) change their matrix
CHUNK = 1024        # tokens per update
KERNEL = 5          # taps of the target's convolution, centred: 2 before, the token, 2 after
RATE = 0.3          # the update's learning rate (eta)
TARGET_INIT = 0.02  # the standard deviation of W_target's diagonal at the start
END_OF_DOCUMENT = 256  # the tokenizer's end-of-document id (wpt_tokenizer.END)


def changes(number):
    """Whether layer `number` (from 1) has In-Place TTT."""
    return (number - 1) % EVERY == 0


class InPlaceTTT(nn.Module):
    """The parts that build the target, and the update (see the module's docstring)."""

    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv1d(width, width, KERNEL, padding=KERNEL // 2, groups=width, bias=False)
        self.target = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.conv.weight)
            self.target.weight.copy_(torch.diag(torch.randn(width) * TARGET_INIT))

    def passes_on(self):
        return {"tokens": Tail(CHUNK - 1), "state": Fold()}

    def forward(self, z, embedding, ids, sequence):
        """z (B, T, F) the shared experts' hidden states, embedding (B, T, D)
        the tokens' embeddings, ids (B, T) the tokens -> RATE z S^T (B, T,
        D), what the change of the matrix adds to the shared experts' output.
        sequence: where these tokens sit (sequence.py)."""
        B, T, F_ = z.shape
        D = embedding.shape[-1]
        ends = ids == END_OF_DOCUMENT
        own = torch.cat([z, embedding.to(z.dtype), ends[..., None].to(z.dtype)], -1)
        tokens = torch.cat([sequence.before(self, "tokens", own).to(own.dtype), own], 1)  # CHUNK - 1 before, then own
        first = sequence.start  # the first token's position: an int, or (B,) for rows at different places
        # the offsets (within the pass) of the chunks' last tokens: ends[b, j] = done[b] + j * CHUNK
        done = (CHUNK - 1 - first % CHUNK) % CHUNK
        counts = (T - 1 - done) // CHUNK + 1 if isinstance(first, int) else ((T - 1 - done) // CHUNK + 1).clamp(min=0)
        n = max(counts, 0) if isinstance(first, int) else int(counts.max())
        offsets, restarts, changes = self._updates(tokens, done, counts, n, F_, D)
        updates = torch.cat([offsets, restarts, changes], -1)  # as Fold changes
        state = sequence.before(self, "state", self._padded(updates, T, B, D * F_, z.device))  # S before this pass
        # S after each chunk end: Fold's step, on the updates' parts before they are joined (read back out
        # of the joined Fold changes, they make torch.compile's Triton code index a column -1 it cannot print)
        fold = self.passes_on()["state"]
        states = [state.float()]
        for j in range(n):
            states.append(fold.step(states[-1], restarts[:, j], changes[:, j]))
        states = torch.stack(states, 1).view(B, n + 1, D, F_)
        # a token reads S unless a document ended in its chunk before it
        positions = sequence.positions(T, z.device).expand(B, T)
        marks = tokens[..., -1].float()
        before = marks.cumsum(1) - marks  # ends among the earlier tokens of `tokens`
        chunk_start = (CHUNK - 1) + torch.arange(T, device=z.device) - positions % CHUNK
        reads = (before[:, CHUNK - 1:] - before.gather(1, chunk_start)) == 0
        z = z * reads[..., None].to(z.dtype)
        change = RATE * self._applied(z, states, done, counts, n, first, sequence is WHOLE)
        if n == 0:  # no chunk ends here: the parts that build the target stay in the graph, with a zero gradient,
            # so that every process of DDP has a gradient for them
            change = change + (self.conv.weight.sum() + self.target.weight.sum()).to(change.dtype) * 0
        return change

    def _updates(self, tokens, done, counts, n, F_, D):
        """The updates of the pass's n completed chunks, the parts of their
        Fold changes, FP32: the offset of each chunk's last token (B, n, 1),
        whether it holds an end of document (B, n, 1), and its update (B, n,
        D F); a row with fewer chunks gets none for the rest."""
        B = tokens.shape[0]
        if n == 0:
            empty = tokens.new_zeros(B, 0, 1, dtype=torch.float32)
            return empty, empty, tokens.new_zeros(B, 0, D * F_, dtype=torch.float32)
        steps = torch.arange(n * CHUNK, device=tokens.device)
        if isinstance(done, int):  # every row alike: the chunks are one slice of `tokens`
            chunks = tokens[:, done:done + n * CHUNK].view(B, n, CHUNK, -1)
            offsets = (done + CHUNK * torch.arange(n, device=tokens.device)).expand(B, n)
            real = torch.ones(B, n, dtype=torch.bool, device=tokens.device)
        else:
            index = (done[:, None] + steps).clamp(max=tokens.shape[1] - 1)
            chunks = tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1])).view(B, n, CHUNK, -1)
            offsets = done[:, None] + CHUNK * torch.arange(n, device=tokens.device)
            real = torch.arange(n, device=tokens.device) < counts[:, None]
        z, e, marks = chunks[..., :F_], chunks[..., F_:F_ + D], chunks[..., -1]
        after = marks.flip(-1).cumsum(-1).flip(-1) == 0  # no end of document at or after the token in its chunk
        e = e * after[..., None].to(e.dtype)
        v = self.conv(e.reshape(B * n, CHUNK, D).transpose(1, 2)).transpose(1, 2).reshape(B, n, CHUNK, D)
        v = self.target(v) * after[..., None].to(e.dtype)
        update = torch.einsum("bjcd,bjcf->bjdf", v, z).float().flatten(2)
        restart = (marks > 0).any(-1)
        keep = real[..., None].float()
        return offsets[..., None].float(), (restart & real)[..., None].float(), update * keep

    @staticmethod
    def _padded(updates, T, B, K, device):
        """updates with no-change elements after them up to T // CHUNK + 1,
        the most any pass of T tokens completes, so that every slice of a
        window split across processes hands on the same shape."""
        missing = T // CHUNK + 1 - updates.shape[1]
        pad = torch.zeros(B, missing, 2 + K, device=device)
        pad[..., 0] = T  # after the pass's last token: undo never applies them, nor does a fold change
        return torch.cat([updates, pad], 1)

    @staticmethod
    def _applied(z, states, done, counts, n, first, whole):
        """z S^T (B, T, D), each token through the S of its chunk: states
        (B, n + 1, D, F), S before the pass's first chunk end, then after
        each. The tokens that read one S run as one product per row. whole:
        the pass is the whole text, so its first chunk reads S = 0, with no
        product."""
        B, T, _ = z.shape
        D = states.shape[2]
        if isinstance(first, int):
            cuts = [0] + [done + j * CHUNK + 1 for j in range(n)] + [T]
            return torch.cat([z.new_zeros(B, b - a, D) if whole and j == 0 else
                              torch.bmm(z[:, a:b], states[:, j].transpose(1, 2).to(z.dtype))
                              for j, (a, b) in enumerate(zip(cuts, cuts[1:])) if b > a], 1)
        # rows at different places: cut wherever any row's chunk ends, each row reading its own S
        ends = (done[:, None] + CHUNK * torch.arange(n, device=z.device) + 1)
        cuts = sorted({0, T, *[c for c in ends[torch.arange(n, device=z.device) < counts[:, None]].tolist() if c < T]})
        out, rows = [], torch.arange(B, device=z.device)
        for a, b in zip(cuts, cuts[1:]):
            j = ((ends <= a) & (torch.arange(n, device=z.device) < counts[:, None])).sum(1)
            out.append(torch.bmm(z[:, a:b], states[rows, j].transpose(1, 2).to(z.dtype)))
        return torch.cat(out, 1)
