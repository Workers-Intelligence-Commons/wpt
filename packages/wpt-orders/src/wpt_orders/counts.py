"""What a model order's `model` numbers make: its layers, its parameters
and the training compute of one token (docs/model.md, "Model families").

Pure functions of the numbers, counted as wpt_model builds the model
(wpt-training's tests check the counts against the model itself):

    layer_kinds   the kind of each layer, in order
    parameters    without and with the vocabulary tables (the embedding
                  and the output head, or the one table of `tied_tables`)
    active_parameters
                  the parameters one token uses: without the vocabulary
                  tables, with only the top_k routed experts of each layer,
                  and without Engram's lookup tables
    compute_per_token
                  the FLOPs of training on one token: 6 x the active
                  parameters + 12 x layers x d_model x window (DeepSeek
                  LLM's attention term), plus In-Place TTT's work, for each
                  pass of the loop, plus 6 x vocabulary x d_model for each
                  time the output head reads a token (every pass the loss
                  weighs, and the multi-token prediction block), plus the
                  layers after a loop that run again for each earlier pass
                  the loss weighs
"""

import math

# what wpt_model fixes for every model, as docs/model.md says
SHARED_EXPERTS = 2
HASH_LAYERS = 3                  # the first 3 layers' experts are hash-routed, without a router
CONV_KERNEL = 4                  # Engram's short convolution
TTT_EVERY, TTT_KERNEL = 6, 5     # In-Place TTT in layers 1, 7, 13, ...; its target's convolution
FLOPS_PER_PARAMETER = 6          # a training token's forward and backward pass through each weight
ATTENTION_FLOPS = 12             # DeepSeek LLM's attention term: 12 x layers x width x window
MTP_HEADS = 1                    # the multi-token prediction block's scores, read through the output head


def layer_kinds(model: dict) -> list[str]:
    """The kind of each layer of a model order's `model`, in order
    (wpt_model.config.layer_kinds): window_layers window layers on each
    side of an exact layer, then sparse_groups groups of an exact layer
    and sparse_layers sparse layers."""
    window, groups, sparse = model["window_layers"], model["sparse_groups"], model["sparse_layers"]
    return ["window"] * window + ["exact"] + ["window"] * window + (["exact"] + ["sparse"] * sparse) * groups


def layer_count(model: dict) -> int:
    """The layers of a model order's `model`."""
    return len(layer_kinds(model))


def _primes_from(start: int, count: int) -> list[int]:
    """The count smallest primes at or above start (wpt_model.engram's table sizes)."""
    found, n = [], max(start, 2)
    while len(found) < count:
        if all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
        n += 1
    return found


def parameters(model: dict, vocab_size: int) -> tuple[int, int]:
    """(parameters without the vocabulary tables, every parameter) of the
    model wpt_model builds from these numbers (a model order's `model`):
    two tables, the embedding and the output head, or one with
    `tied_tables`."""
    count = _count(model, active=False)
    return count, count + (1 if model.get("tied_tables") else 2) * vocab_size * model["d_model"]


def active_parameters(model: dict) -> int:
    """The parameters one token uses, of the model wpt_model builds from
    these numbers: `parameters` without the vocabulary tables, with the
    top_k routed experts of each layer instead of all n_experts, and
    without Engram's lookup tables (the rows read by hash; its projections
    count)."""
    return _count(model, active=True)


def _count(model: dict, active: bool) -> int:
    layers, rest = _counts(model, active)
    return sum(layers) + rest


def _counts(model: dict, active: bool) -> tuple[list[int], int]:
    """(the parameters of each layer, in order, those of the rest: the
    final norm and the multi-token prediction block), as _count counts them."""
    D = model["d_model"]
    H, c = model["attention_heads"], model["attention_head_dim"]
    norm, mix = D, 2 * D  # a sum of Attention Residuals: its query and its norm's weight
    queries, keys = D * H * c + c, D * c + c  # each with its norm, per head or for the one key-value head
    gated = D * H * c + H  # a window or sparse layer's gate and sinks
    attention = {"window": queries + keys + D * c + gated + H * c * D,
                 "exact": queries + keys + D * c + H * c * D,
                 "sparse": queries + gated + H * c * D}

    def feed_forward(width, hidden):
        return 3 * width * hidden

    def experts(hashed):
        L, E, n = model["latent_width"], model["expert_width"], model["n_experts"]
        routed = model["top_k"] if active else n
        count = SHARED_EXPERTS * feed_forward(D, model["shared_width"]) + 2 * D * L + L + 3 * routed * E * L
        return count if hashed else count + D * n

    def engram():
        tables = (model["engram_order"] - 1) * model["engram_heads"]
        width = tables * model["engram_dim"]
        rows = 0 if active else sum(_primes_from(model["engram_table_size"], tables))
        return rows * model["engram_dim"] + 2 * width * D + 3 * D + D * CONV_KERNEL

    ttt = D * TTT_KERNEL + D * D  # In-Place TTT's convolution and W_target
    layers = []
    for number, kind in enumerate(layer_kinds(model), start=1):
        count = attention[kind] + experts(number <= HASH_LAYERS) + 2 * norm + 2 * mix
        count += engram() if number in model["engram_layers"] else 0
        count += ttt if _changes(number) else 0
        layers.append(count)
    multi_token = (2 * norm + 2 * D * D + norm + attention["window"] + norm
                   + feed_forward(D, SHARED_EXPERTS * model["shared_width"]) + norm)
    # the first sublayer reads the embedding without a sum, and the output head reads one: 2 sums per layer (with
    # loop_layers, the loop's output and the coda's first sublayer stand for them)
    return layers, norm + multi_token  # the final norm, the multi-token prediction block


def _changes(number: int) -> bool:
    """Whether layer `number` (from 1) has In-Place TTT (wpt_model.ttt.changes)."""
    return (number - 1) % TTT_EVERY == 0


def weighed_passes(model: dict) -> int:
    """The passes of the loop the next-token loss reads through the output
    head: those loop_losses weighs above 0, else the last alone."""
    weights = model.get("loop_losses")
    return 1 if weights is None else sum(1 for weight in weights if weight > 0)


def compute_per_token(model: dict, vocab_size: int, block_size: int) -> int:
    """The FLOPs of training the model on one token in windows of
    block_size tokens (the module's docstring). The layers: 6 x their
    active parameters + 12 x d_model x block_size each, plus In-Place
    TTT's update and change, two more d_model x 2 shared_width matrices in
    each layer that has it; every layer once per pass of the loop (loops +
    1), or with loop_layers the looped layers once per pass and the others
    once, and those again for each other pass the loss weighs. The output
    head: 6 x vocab_size x d_model each time it reads a token, once for
    each pass the loss weighs and once for the multi-token prediction
    block."""
    layers, rest = _counts(model, active=True)

    def work(numbers, parameters):
        ttt = sum(_changes(number) for number in numbers)
        return (FLOPS_PER_PARAMETER * parameters
                + ATTENTION_FLOPS * len(numbers) * model["d_model"] * block_size
                + FLOPS_PER_PARAMETER * ttt * 2 * model["d_model"] * SHARED_EXPERTS * model["shared_width"])

    every = range(1, len(layers) + 1)
    looped = model.get("loop_layers")
    if looped is None:
        body = (model["loops"] + 1) * work(every, sum(layers))
    else:
        coda = range(looped + 1, len(layers) + 1)
        body = (work(every, sum(layers)) + model["loops"] * work(range(1, looped + 1), sum(layers[:looped]))
                + (weighed_passes(model) - 1) * work(coda, sum(layers[looped:])))
    heads = (weighed_passes(model) + MTP_HEADS) * FLOPS_PER_PARAMETER * vocab_size * model["d_model"]
    return body + FLOPS_PER_PARAMETER * rest + heads
