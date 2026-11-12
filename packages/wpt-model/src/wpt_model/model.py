"""The WPT model (docs/model.md): token ids -> next-token scores, and the
scores of the token after next from the multi-token prediction block.

The model only maps token ids to scores. Loss belongs to training, sampling
belongs to inference. On a GPU it computes in BF16, keeping FP32 where
docs/model.md says (the attention's scores and softmax, the scores of
Attention Residuals, routing scores).

Each layer is two sublayers, its attention and its experts, linked by
Attention Residuals (residuals.py): each sublayer reads a weighted sum of
the embedding and the earlier groups of sublayer outputs, and the output
head reads one more. The sums run between the sublayers, outside them.

The layers follow HySparse2's layout (config.layer_kinds; attention.py):
the first half is window layers around one exact layer, the second half
groups of one exact layer and the sparse layers that read its keys and
values and its choice of tokens. The second half's exact layers make their
keys and values from the first half's output, the input Attention
Residuals gives the second half's first sublayer. Which input that output
is, among those Attention Residuals can give, is open to review.

In-Place TTT (ttt.py) changes the shared experts' down projection of every
sixth layer, from the first, while the text is read.

Loops (Ouro, arXiv 2510.25741; docs/model.md, "Loops"): with config.loops
= n, the text passes n more times through every layer with the same
weights. Each pass after the first takes the pass before's output (the
output head's input) as its first source, where the first pass takes the
embedding, and its sums of Attention Residuals cover its own sources only.
The output head, the multi-token prediction block and the features read
the last pass. In-Place TTT's targets are the token embeddings in every
pass. At 0 the model is one pass: no weight and no work of the loop.
Open to review: the loss reads the last pass only, and no exit gate learns
when to stop (Ouro's).

With recompute set, training keeps only each sublayer's input and runs the
sublayer again during the backward pass, instead of keeping everything it
computed: the same result with much less memory, for about a third more
time. The second run records no statistics (expert balancing), so a step
counts every token once.

compile_training runs each sublayer (Layer.attend, Layer.feed) and the
multi-token prediction block through torch.compile in training passes over
whole windows, one window shape at a time, and the math of the sums of
Attention Residuals between them too (residuals.py). Each module keeps its name, so the weights and
checkpoints are the same in either mode. The exact and sparse attention
(kernels.py), the routing and the routed experts stay out of the compiled
graphs (torch.compiler.disable on kernels.py's entry points and on
experts.Experts._routed). On a CUDA device, such a pass to the summed
losses, forward and backward, then replays CUDA graphs recorded once per
window shape (graphs.py; not with recompute, nor with routes recorded or
replayed). A pass over part of a window, without gradients
or in evaluation runs uncompiled: a part's place changes from pass to
pass, and every new place would compile again.

record and replay (experts.Routes) keep the experts the routers chose for
one sequence and route a later pass the same way (R3).

Every layer says once what it passes on to later tokens (carry.py), and a
pass may follow earlier tokens (sequence.py): a segment on one GPU,
carrying what earlier segments passed on (carried), or one process's slice
of a window split across processes, or of one part of a window run as
parts split across processes (split). Each gives exactly the whole
window's result.

Generation reads tokens through a cache (cache.py) that keeps the same
description: new_cache, then extend with each call's new tokens, undo to
take back the last tokens of an undoable extend, and extend_mtp for the
multi-token prediction block. extend is forward's own pass through the
cache's place, so the same scores within rounding, and it has no length
limit.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import WINDOW, ExactAttention, SparseAttention, WindowAttention
from .cache import Cache, Recorder
from .config import EXACT, SPARSE, ModelConfig
from .engram import Engram
from .experts import SHARED_EXPERTS, Experts, Routes
from .graphs import WindowGraphs
from .layers import EPS, FeedForward, compute_dtype
from .optimizer import roles
from .residuals import Residuals
from .sequence import WHOLE, Carried, Split
from .ttt import changes

HASH_LAYERS = 3   # the first 3 layers' experts are hash-routed
CHUNK = 4096  # tokens whose scores a summed loss holds at once: the 32,768 scores of each take most of a pass's memory
COMPILED_FORMS = 256  # torch.compile's forms of one piece of code: kinds of layer, times window lengths, times steps

# the model's parts, by the names a training step locks them by (docs/model.md, "Parts"): every parameter
# belongs to exactly one, and BALANCING is the routers' balancing biases, which change after each update
EMBEDDING, ATTENTION, ROUTERS, BALANCING = "embedding", "attention", "routers", "balancing"
EXPERTS, ENGRAM, RESIDUALS, FINAL_NORM = "experts", "engram", "residuals", "final-norm"
OUTPUT_HEAD, MULTI_TOKEN_PREDICTION = "output-head", "multi-token-prediction"
PARTS = (EMBEDDING, ATTENTION, ROUTERS, BALANCING, EXPERTS, ENGRAM, RESIDUALS, FINAL_NORM, OUTPUT_HEAD,
         MULTI_TOKEN_PREDICTION)
# the first module names of a parameter's name, in order of precedence, and the part they make it
_PART_OF = ((("mtp",), MULTI_TOKEN_PREDICTION), (("embedding",), EMBEDDING), (("head",), OUTPUT_HEAD),
            (("final_norm",), FINAL_NORM),
            (("attention",), ATTENTION), (("attention_norm",), ATTENTION), (("experts", "router"), ROUTERS),
            (("experts",), EXPERTS), (("experts_norm",), EXPERTS), (("engram",), ENGRAM),
            (("residuals",), RESIDUALS))


def part_of(name: str) -> str:
    """The part (PARTS) a parameter belongs to, by its name in the model
    (WPT.named_parameters): a layer's are read after `layers.<n>.`."""
    words = name.split(".")
    if words[0] == "layers":
        words = words[2:]
    for prefix, part in _PART_OF:
        if tuple(words[:len(prefix)]) == prefix:
            return part
    raise ValueError(f"{name}: a parameter of no part")


class Layer(nn.Module):
    """Two sublayers: attention of the layer's kind (with Engram before it
    in the layers the recipe names), then experts (with In-Place TTT in
    every sixth layer, ttt.py). Attention Residuals gives each its input
    (residuals.py)."""

    def __init__(self, config, number):
        super().__init__()
        D = config.d_model
        self.kind = config.kinds[number - 1]
        if self.kind == EXACT:  # its choice of tokens is made only for the sparse layers that follow it
            self.attention = ExactAttention(config, chooses=config.kinds[number:number + 1] == (SPARSE,))
        else:
            self.attention = (SparseAttention if self.kind == SPARSE else WindowAttention)(config)
        self.experts = Experts(config, hashed=number <= HASH_LAYERS, ttt=changes(number))
        self.engram = Engram(config) if number in config.engram_layers else None
        self.attention_norm, self.experts_norm = nn.RMSNorm(D, eps=EPS), nn.RMSNorm(D, eps=EPS)

    def attend(self, h, canonical, sequence=WHOLE, bridge=None, handoff=None):
        """The attention sublayer: its input h -> (its output, what an exact
        layer hands the sparse layers after it, None from another kind).
        sequence: where these tokens sit in the window. bridge: what a
        second-half exact layer makes its keys and values from (its own
        input when None). handoff: what a sparse layer reads, the exact
        layer's. Engram's Y is added to h before the attention reads it,
        and is part of the output, as a residual sum would hold it."""
        added = None
        if self.engram is not None:
            added = self.engram(h, canonical, sequence)
            h = h + added
        x = self.attention_norm(h)
        given = None
        if self.kind == EXACT:
            out, given = self.attention(x, x if bridge is None else self.attention_norm(bridge), sequence)
        elif self.kind == SPARSE:
            out = self.attention(x, handoff, sequence)
        else:
            out = self.attention(x, sequence)
        return (out if added is None else out + added), given

    def feed(self, h, ids, embedding, sequence=WHOLE, collect=True, record=None, replay=None, loop=0):
        """The experts sublayer: its input h -> its output. embedding: the
        tokens' embeddings, In-Place TTT's targets; sequence: where these
        tokens sit. collect: record training statistics (False when the
        sublayer runs again for the backward pass). record, replay: the
        routed experts' choices (R3), kept apart for each pass, loop, of a
        looping model."""
        return self.experts(self.experts_norm(h), ids, collect, record, replay, loop, embedding, sequence)


def _counts(ids, counts):
    """counts (B,) as a tensor on ids' device: every one of ids' n tokens when None."""
    B, n = ids.shape
    if counts is None:
        return torch.full((B,), n, dtype=torch.long, device=ids.device)
    return torch.as_tensor(counts, dtype=torch.long, device=ids.device)


_fla = None
_fla_error = None  # why _fla is False


def _fla_fused_loss():
    """The fused head and loss through flash-linear-attention's kernels
    (head_loss.py), or False when flash-linear-attention does not import
    here; imported once."""
    global _fla, _fla_error
    if _fla is None:
        try:
            import fla.modules.fused_linear_cross_entropy  # noqa: F401 (imported again here: head_loss may be loaded)
            from .head_loss import head_loss
            _fla = head_loss
        except Exception as error:  # an ImportError, or an error of Triton's own while it starts
            _fla, _fla_error = False, error
    return _fla


def fla_failure():
    """None when flash-linear-attention's fused head and loss imports here;
    else why it does not."""
    if _fla_fused_loss():
        return None
    return f"{type(_fla_error).__name__}: {_fla_error}"


def _chunk_loss(head, hidden, targets):
    return F.cross_entropy(head(hidden).float(), targets, reduction="sum")


def _first_run_collects(feed, sequence, record, replay, loop):
    """The experts sublayer as a function whose first call records
    statistics (when the place collects them) and whose later call, the
    run again for the backward pass, does not."""
    runs = []

    def run(h, ids, embedding):
        runs.append(None)
        return feed(h, ids, embedding, sequence, collect=len(runs) == 1 and sequence.collect, record=record,
                    replay=replay, loop=loop)
    return run


def _kept(parameter, grad):
    """A parameter's gradient as the backward pass hands it over: a copy
    when the parameter holds none yet, which it would otherwise keep as it
    is. In a compiled pass the gradient is an output of a recorded CUDA
    graph, whose memory the graph's next replay writes over, while the
    parameter adds the later windows' gradients of the update to it."""
    return grad.clone() if parameter.grad is None else grad


class MultiTokenPrediction(nn.Module):
    """DeepSeek-V3's MTP module: joins the final hidden state and the next
    token's embedding, then one layer. WPT's layer: a window layer's
    attention over the last 128 tokens (as MiMo-V2-Flash's MTP blocks) and
    a dense SiTU-GLU as wide as the shared experts together, on one
    stream."""

    def __init__(self, config):
        super().__init__()
        D = config.d_model
        self.hidden_norm, self.embedding_norm = nn.RMSNorm(D, eps=EPS), nn.RMSNorm(D, eps=EPS)
        self.join = nn.Linear(2 * D, D, bias=False)
        self.attention_norm, self.attention = nn.RMSNorm(D, eps=EPS), WindowAttention(config)
        self.feed_forward_norm = nn.RMSNorm(D, eps=EPS)
        self.feed_forward = FeedForward(D, SHARED_EXPERTS * config.shared_width)
        self.final_norm = nn.RMSNorm(D, eps=EPS)

    def forward(self, hidden, next_embedding, sequence=WHOLE):
        h = self.join(torch.cat([self.hidden_norm(hidden), self.embedding_norm(next_embedding)], -1))
        h = h + self.attention(self.attention_norm(h), sequence)
        h = h + self.feed_forward(self.feed_forward_norm(h))
        return self.final_norm(h)


class WPT(nn.Module):
    PART = WINDOW  # a segment or a slice of a window holds a multiple of this many tokens

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Layer(config, number) for number in range(1, config.n_layer + 1))
        self.residuals = Residuals(config.d_model, 2 * config.n_layer, config.residual_groups)
        self.final_norm = nn.RMSNorm(config.d_model, eps=EPS)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp = MultiTokenPrediction(config)
        # Engram's canonical id of each token id, from the vocabulary (engram.canonical_ids); saved with the weights
        self.register_buffer("canonical", torch.arange(config.vocab_size))
        self.recompute = False  # how training uses memory, not part of the model: see the module's docstring
        self.compiled = {}  # each module compile_training compiled: a plain dict, so no name of the weights changes
        self._graph_hooks = []  # compile_training's hooks on the parameters' gradients (_kept)
        self.graphs = None  # compile_training's recorded passes over whole windows (graphs.py)

    def compile_training(self, on=True):
        """Whether training passes over whole windows run through
        torch.compile from here on (see the module's docstring)."""
        self.compiled, self.graphs = {}, None
        for hook in self._graph_hooks:
            hook.remove()
        self._graph_hooks = []
        if on:
            torch._dynamo.config.recompile_limit = COMPILED_FORMS
            self._graph_hooks = [p.register_hook(partial(_kept, p)) for p in self.parameters() if p.requires_grad]
            self.compiled = {self.mtp: torch.compile(self.mtp, dynamic=False)}
            for layer in self.layers:
                self.compiled[layer] = (torch.compile(layer.attend, dynamic=False),
                                        torch.compile(layer.feed, dynamic=False))
            self.graphs = WindowGraphs(self)

    def _whole(self, sequence):
        """Whether this pass is a training pass over whole windows, which runs compiled."""
        return sequence is WHOLE and self.training and torch.is_grad_enabled()

    def _running(self, module, sequence):
        """module as this pass runs it: compiled for a training pass over whole windows."""
        return self.compiled.get(module, module) if self._whole(sequence) else module

    def _sublayers(self, layer, sequence):
        """(attend, feed) of layer as this pass runs them."""
        return self.compiled.get(layer, (layer.attend, layer.feed)) if self._whole(sequence) else (layer.attend,
                                                                                                   layer.feed)

    def set_canonical_ids(self, ids: torch.Tensor):
        self.canonical.copy_(ids)

    def forward(self, idx, next_tokens=None, *, record: Routes | None = None, replay: Routes | None = None,
                sequence=None, features=(), targets=None):
        """(B, T) token ids -> (B, T, vocab_size) next-token scores. With
        next_tokens, the (B, T) tokens that follow idx, also returns the
        multi-token prediction block's scores for the token after each next
        token. Returns the scores alone, or the tuple (scores, [mtp scores],
        [features]) of what was asked. With targets, a pair (tokens, tokens
        after next) of (B, t) token ids, which needs next_tokens, no scores
        come back: the tuple starts instead with two cross-entropies summed
        over the tokens (the next-token scores against the first, the
        multi-token prediction scores of the first t positions against the
        second), computed CHUNK tokens at a time so that the scores of all
        the tokens are never held at once (a fused kernel, or the backward
        pass computing each chunk's scores again; see _summed). With record
        or replay (B = 1), the routed layers record their choices into it,
        or route as it holds. sequence: where idx sits in a longer window
        (carried, split), the whole window when None. features: layer
        numbers (from 1) whose hidden states come back side by side, (B, T,
        len(features) * d_model), what a drafter reads (drafter.py): a
        layer's hidden state is the input Attention Residuals gives the next
        layer's attention, and the last layer's is the output head's
        input."""
        sequence = WHOLE if sequence is None else sequence
        if targets is not None and next_tokens is None:
            raise ValueError("targets need next_tokens")
        if (targets is not None and self.graphs is not None and idx.is_cuda and self._whole(sequence)
                and record is None and replay is None and not features and not self.recompute):
            return self.graphs.losses(idx, next_tokens, targets)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=idx.is_cuda):
            hidden, found = self._hidden(idx, record, replay, sequence, features)
            if targets is None:
                outputs = [self.head(self.final_norm(hidden)).float()]
                if next_tokens is not None:
                    mtp = self._running(self.mtp, sequence)
                    outputs.append(self.head(mtp(hidden, self.embedding(next_tokens), sequence)).float())
            else:
                outputs = list(self._sums(hidden, next_tokens, targets, sequence))
            if features:
                outputs.append(found)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def losses(self, idx, next_tokens, targets, record=None, replay=None, sequence=WHOLE):
        """forward's two summed cross-entropies, computed as they are (no
        recorded graph): what a recorded pass records (graphs.py)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=idx.is_cuda):
            hidden, _ = self._hidden(idx, record, replay, sequence, ())
            return self._sums(hidden, next_tokens, targets, sequence)

    def _sums(self, hidden, next_tokens, targets, sequence):
        """The two summed cross-entropies from the final hidden states, inside forward's autocast."""
        ahead = self._running(self.mtp, sequence)(hidden, self.embedding(next_tokens), sequence)
        return (self._summed(self.final_norm(hidden), targets[0]),
                self._summed(ahead[:, :targets[1].shape[1]], targets[1]))

    def _summed(self, hidden, targets):
        """The cross-entropy of the head's scores for hidden (B, t, d_model)
        against targets (B, t), summed, CHUNK tokens at a time. With
        gradients on a GPU where flash-linear-attention imports, its fused
        head and loss does it, one call per chunk: the gradients come out of
        its forward pass, and the head weight is cast to the autocast dtype
        (the hidden states' without autocast) for each call, so that each
        chunk's gradient is one product in that dtype and the chunks' add up
        in FP32 into the FP32 parameter's. Otherwise, with gradients, each chunk is a checkpoint: it
        keeps only its hidden states, and the backward pass computes its
        scores again, in the same autocast, through the one head weight,
        whose gradient adds up over the chunks. A chunk holds only the head
        and the loss: no routing, statistic or random draw runs twice."""
        hidden, targets = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        grads = torch.is_grad_enabled()
        total = 0
        if grads and hidden.is_cuda and (fused := _fla_fused_loss()):
            # flash-linear-attention's kernel, which computes the gradients inside its forward pass: only with
            # gradients, and outside autocast so that its matrix products keep the dtypes given here. A chunk's
            # weight gradient is one product in the autocast dtype, as autocast gives any weight's; one call over
            # every chunk would add them up through FP32 products, several times slower
            dtype = compute_dtype(hidden)
            with torch.autocast("cuda", enabled=False):
                for a in range(0, len(targets), CHUNK):
                    total = total + fused(hidden[a:a + CHUNK].to(dtype), targets[a:a + CHUNK],
                                          self.head.weight.to(dtype))
            return total
        for a in range(0, len(targets), CHUNK):
            args = (self.head, hidden[a:a + CHUNK], targets[a:a + CHUNK])
            total = total + (checkpoint(_chunk_loss, *args, use_reentrant=False) if grads else _chunk_loss(*args))
        return total

    def _hidden(self, idx, record, replay, sequence, features):
        """(the final hidden states, the features' hidden states side by
        side or None) of idx at its place, inside forward's autocast: every
        pass of the loop in turn, each from the one before's output."""
        canonical = self.canonical[idx]
        embedding = self.embedding(idx)
        hidden = embedding
        for loop in range(self.config.loops + 1):
            hidden, found = self._pass(hidden, idx, canonical, embedding, record, replay, sequence.at(loop), loop,
                                       features if loop == self.config.loops else ())
        return hidden, found

    def _pass(self, first, idx, canonical, embedding, record, replay, sequence, loop, features):
        """One pass through every layer, from its first source: (its output, the features or None)."""
        sources = self.residuals.start(first, compiled=bool(self.compiled) and self._whole(sequence))
        recompute = self.recompute and self.training and torch.is_grad_enabled()
        found = {}
        bridge = handoff = None
        for number, layer in enumerate(self.layers, start=1):
            attend, feed = self._sublayers(layer, sequence)
            j = 2 * (number - 1)
            h = sources.input(j)
            if number - 1 in features:
                found[number - 1] = h.float()
            if number == self.config.first_half + 1:
                bridge = h  # the first half's output, as Attention Residuals gives it to the second half
            args = (h, canonical, sequence, bridge, handoff)
            out, given = checkpoint(attend, *args, use_reentrant=False) if recompute else attend(*args)
            if given is not None:
                handoff = given
            sources.output(j, out)
            h = sources.input(j + 1)
            if recompute:
                out = checkpoint(_first_run_collects(feed, sequence, record, replay, loop), h, idx, embedding,
                                 use_reentrant=False)
            else:
                out = feed(h, idx, embedding, sequence, collect=sequence.collect, record=record, replay=replay,
                           loop=loop)
            sources.output(j + 1, out)
        hidden = sources.end()
        if len(self.layers) in features:
            found[len(self.layers)] = hidden.float()
        return hidden, torch.cat([found[number] for number in features], -1) if features else None

    @staticmethod
    def carried(offset, carried, collect=True):
        """The place of a segment starting at token offset of its window, given
        what the segment before passed on (sequence.Carried)."""
        return Carried(offset, carried, collect)

    @staticmethod
    def split(rank, world, length, start=0, carried=None, collect=True):
        """The place of process rank's slice, of length tokens, of a window
        split across world processes, or of the part of it that starts at
        token start, given what the part before passed on (sequence.Split)."""
        return Split(rank, world, length, start, carried, collect)

    def new_cache(self, rows, routes=False) -> Cache:
        """An empty cache of `rows` rows (cache.py); with routes, it records
        each row's routed experts (R3) as its tokens are read."""
        return Cache(self, rows, routes)

    @torch.no_grad()
    def extend(self, ids, cache, counts=None, undoable=False, features=()):
        """Reads ids (B, n), the next tokens of the cache's B rows: the
        first counts[b] of row b (all n by default), the rest padding.
        Returns (B, n, vocab_size) next-token scores, the same as forward's
        over each row's whole text within rounding (padding's are
        meaningless), and the (B, n, D) final hidden states, which
        extend_mtp reads; with features (layer numbers), also those layers'
        hidden states as forward gives them. With undoable, undo can then
        take back the last tokens of this call."""
        counts = _counts(ids, counts)
        reading = cache.text.reading(counts, undoable)
        record = Recorder(cache.routes, counts.tolist()) if cache.routes is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            hidden, found = self._hidden(ids, record, None, reading, features)
            scores = self.head(self.final_norm(hidden)).float()
        reading.finish()
        return (scores, hidden, found) if features else (scores, hidden)

    @torch.no_grad()
    def undo(self, cache, kept):
        """After an undoable extend: keeps only its first kept[b] tokens of
        row b, as if the others had never been read (the routes recorded
        for them too)."""
        cache.text.undo(torch.as_tensor(kept, dtype=torch.long, device=cache.lengths.device))
        if cache.routes is not None:
            for routes, length in zip(cache.routes, cache.lengths.tolist()):
                routes.keep(length)

    @torch.no_grad()
    def extend_mtp(self, hidden, next_ids, cache, counts=None, undoable=False):
        """The multi-token prediction block's scores (B, n, vocab_size) for
        the token after next, through its window in the cache: hidden, the
        final hidden states extend returned for n positions, and next_ids,
        the token that follows each. The block reads these pairs in order,
        apart from extend; undo_mtp takes back those of an undoable call."""
        reading = cache.mtp.reading(_counts(next_ids, counts), undoable)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=next_ids.is_cuda):
            scores = self.head(self.mtp(hidden, self.embedding(next_ids), reading)).float()
        reading.finish()
        return scores

    @torch.no_grad()
    def undo_mtp(self, cache, kept):
        """undo, for the pairs of the last undoable extend_mtp."""
        cache.mtp.undo(torch.as_tensor(kept, dtype=torch.long, device=cache.lengths.device))

    def after_update(self, reduce, balance=True):
        """After each optimizer step: Quantile Balancing sets the routers'
        biases from the step's scores, summed over processes by reduce.
        Without balance (a step that locks BALANCING), the biases stay as
        they are and the step's scores are dropped."""
        for layer in self.layers:
            if balance:
                layer.experts.balance(reduce)
            else:
                layer.experts.histogram = None

    def parts(self) -> dict[str, list[torch.nn.Parameter]]:
        """{part: its parameters} for every name of PARTS, in the model's
        order (BALANCING has none: its biases are not trained by gradient);
        every parameter is in exactly one."""
        found = {part: [] for part in PARTS}
        for name, parameter in self.named_parameters():
            found[part_of(name)].append(parameter)
        return found

    def optimizer_roles(self):
        """(name, parameter, role, view) for every parameter, by the rule of
        optimizer.roles, with the embedding and the output head among the
        "vector" ones (AdamW)."""
        return roles(self, (self.embedding.weight, self.head.weight))
