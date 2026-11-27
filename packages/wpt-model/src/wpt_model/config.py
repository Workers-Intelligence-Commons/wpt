"""The numbers that define a WPT model (docs/model.md). Saved as config.json
next to the weights, so any saved model can be rebuilt without knowing how
it was trained. What docs/model.md fixes for every size (the window of 128
tokens, the 1,024 tokens each sparse layer picks, the rotary dimensions,
hash routing in the first 3 layers, 2 shared experts, In-Place TTT's
layers and chunk) is in the code, not here."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .attention import ROPE_DIMS

WINDOW, EXACT, SPARSE = "window", "exact", "sparse"  # the kinds of layer
# the rules of how a loop's passes follow each other, each with its value when it is off: today's loop
LOOP_RULES = {"loop_decay": None, "decay_last_pass": False}


def layer_kinds(window_layers, sparse_groups, sparse_layers):
    """The kind of each layer, in order: the first half, window_layers
    window layers on each side of one exact layer, then the second half,
    sparse_groups groups of one exact layer followed by sparse_layers
    sparse layers."""
    first = (WINDOW,) * window_layers + (EXACT,) + (WINDOW,) * window_layers
    return first + ((EXACT,) + (SPARSE,) * sparse_layers) * sparse_groups


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int         # tokens per training window (the model itself has no length limit)
    window_layers: int      # window layers on each side of the first half's exact layer
    sparse_groups: int      # groups of the second half: one exact layer, then sparse_layers sparse layers
    sparse_layers: int      # sparse layers of each group
    d_model: int            # width of the hidden states
    attention_heads: int    # query heads of every attention layer and the MTP block
    attention_head_dim: int  # each query head and the one key-value head; a window layer rotates the last 64
    n_experts: int          # routed experts per layer
    top_k: int              # routed experts per token
    expert_width: int       # hidden width of a routed expert, in the latent space
    latent_width: int       # width of the space routed experts work in
    shared_width: int       # hidden width of each shared expert, at full width
    engram_layers: tuple    # layers (numbered from 1) that get Engram memory
    engram_order: int       # largest N-gram, N; orders 2 .. N are read
    engram_heads: int       # hash heads per order
    engram_dim: int         # width of one table row
    engram_table_size: int  # each table has a distinct prime size, the smallest at or above this
    residual_groups: int    # groups of sublayers whose outputs Attention Residuals sums into one source;
                            # 2 * n_layer (every attention and every experts sublayer its own) is the full form
    loops: int = 0          # extra passes of the text through every layer, with the same weights; 0: one pass
    tied_tables: bool = False  # one vocabulary table: the embedding reads the output head's rows (TiedEmbedding)
    # how the passes of the loop follow each other (docs/model.md, "Loops"); each off at its value in LOOP_RULES
    loop_decay: float | None = None  # lambda: a pass after the first starts from h0 + lambda (out - h0), h0 the
                                     # first pass's first source, out the pass before's output; None: from out
    decay_last_pass: bool = False    # the last pass's output decayed the same way before the output head reads it

    def __post_init__(self):
        object.__setattr__(self, "engram_layers", tuple(self.engram_layers))
        problems = [f"{f.name} must be at least 1" for f in fields(self)
                    if f.name not in ("engram_layers", "loops", "tied_tables", *LOOP_RULES)
                    and getattr(self, f.name) < 1]
        if self.loops < 0:
            problems.append("loops must be at least 0")
        problems += self._loop_problems()
        if self.block_size < 2:
            problems.append("block_size must be at least 2: multi-token prediction reads the token after next")
        if not self.top_k < self.n_experts:
            problems.append("top_k must be smaller than n_experts: Quantile Balancing reads the (k+1)-th score")
        if not (self.attention_head_dim >= ROPE_DIMS and self.attention_head_dim % 2 == 0):
            problems.append(f"attention_head_dim must be even and at least the {ROPE_DIMS} rotary dimensions")
        if self.engram_order < 2:
            problems.append("engram_order must be at least 2")
        if not self.engram_table_size < 2**31:
            problems.append("engram_table_size must be below 2^31, so hashing stays inside 64-bit integers")
        if not self.residual_groups <= 2 * self.n_layer:
            problems.append(f"residual_groups must be at most the {2 * self.n_layer} sublayers (2 per layer)")
        problems += [f"engram layer {n} is not between 1 and n_layer" for n in self.engram_layers
                     if not 1 <= n <= self.n_layer]
        if problems:
            raise ValueError("model config: " + "; ".join(problems))

    def _loop_problems(self):
        """Why the rules of the loop (LOOP_RULES) cannot go together with
        the rest of the config, one sentence each."""
        problems = []
        if self.loops == 0:
            problems += [f"{name} shapes the passes of a loop, and loops is 0: one pass" for name, off in
                         LOOP_RULES.items() if getattr(self, name) != off]
        if self.loop_decay is not None and not 0 < self.loop_decay < 1:
            problems.append("loop_decay must be above 0 and below 1 (leave it out for no decay)")
        if self.decay_last_pass and self.loop_decay is None:
            problems.append("decay_last_pass decays by loop_decay, which is left out")
        return problems

    def numbers(self) -> dict:
        """The config as config.json holds it: every number, the rules of the
        loop (LOOP_RULES) left out where they are off, so a model that uses
        none of them is described as before they existed."""
        return {key: value for key, value in asdict(self).items()
                if key not in LOOP_RULES or value != LOOP_RULES[key]}

    @property
    def kinds(self):
        """The kind of each layer, in order (layer_kinds)."""
        return layer_kinds(self.window_layers, self.sparse_groups, self.sparse_layers)

    @property
    def n_layer(self):
        return len(self.kinds)

    @property
    def first_half(self):
        """The layers of the first half, whose output the second half's
        exact layers make their keys and values from."""
        return 2 * self.window_layers + 1

    def save(self, path):
        Path(path).write_text(json.dumps(self.numbers(), indent=2) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
