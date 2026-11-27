# Model architecture

This page describes the architecture WPT models use and why; the code is
`packages/wpt-model`, update rule included, and the `by-part` optimizer of
`packages/wpt-training`, which runs it with a schedule.
Every component comes from published research (cited at the end) and is
trained from scratch: no weights or outputs of those models are used.

The design is one family for every size: a mixture-of-experts model that
reads long texts, aiming at a context of 1M tokens, which an order reaches
by stages of growing windows (training.md, "Stages"). The numbers of a given size (layers, width, experts, table
sizes) are not fixed here: each size is a size of a model family
("Model families", below), since a small model may not want the ratios
of a large one.

## The shape

```
token ids
  -> embedding
  -> layer 1 .. L, each:
       attention sublayer: a window, exact or sparse layer, in the layout
         the order names (first half: window layers on each side of an
         exact layer; second half: groups of an exact layer and sparse
         layers); Engram memory added to its input in the layers the
         order names
       feed-forward sublayer: experts (hash-routed in the first 3 expert
         layers, router-chosen after), 2 shared experts, whose down
         projection changes while the text is read in every sixth layer
         (In-Place TTT)
       every sublayer's input a weighted sum of the embedding and the
         earlier sublayers' outputs (Attention Residuals)
  -> `loops` more times through layers 1 .. L (or the first `loop_layers`,
     the others then once after the last pass), each pass starting from
     the one before's output, or from the decayed change it made (Loops)
  -> one more such sum, RMSNorm, output head        (next-token scores)
  -> one multi-token prediction block, window attention (the token after next)
```

## Attention

The attention is [HySparse2][hysparse2]'s, as published, with a window of
128 tokens, aiming at a context of 1M tokens. Its layers are of three
kinds, and the order sets their layout with three numbers
(`window_layers`, `sparse_groups`, `sparse_layers`):

- **the first half**: `window_layers` window layers, one exact layer, then
  `window_layers` window layers again;
- **the second half**: `sparse_groups` groups, each one exact layer
  followed by `sparse_layers` sparse layers.

The full layout is `(4, 2, 5)`: 21 layers, 4 window, 1 exact, 4 window,
then twice 1 exact and 5 sparse. A model for a single small GPU uses `(2,
1, 5)`: 11 layers.

Every attention layer has one key-value head shared by all its query heads
(MQA), normalizes each query head and each key by an RMSNorm with a learned
weight before comparing them (QK-norm, as [IMU-1][imu1]), and scales the
scores by `1/sqrt(c)`, `c` the head width. The scores and the softmax run
in FP32.

- **Window layers** read, for each query, its last 128 tokens, itself
  included. Rotary position encoding (base 10,000) turns the last 64
  dimensions of each query and key, the only position encoding in the
  model. Positions count from the text's first token, also in a segment
  or slice of a long window ("Long windows", below).
- **Exact layers** read every token up to the query, without position
  encoding. Each query also picks its 1,024 tokens with the highest score
  among those at least 128 tokens back (HySparse2's oracle selection): a
  token's weight in one head is its softmax probability, and its score is
  its largest weight over the heads ([HySparse][hysparse]'s maximum over a
  group of heads, where MQA makes all heads one group). A query with fewer
  such tokens picks all of them. Equal scores go to whichever token the
  top-`k` operation returns first. The choice has no gradient. The first
  half's exact layer, which no sparse layer follows, picks none.
- **Sparse layers** have no keys or values of their own. Each query reads
  the keys and values of the exact layer before it at that query's 1,024
  chosen tokens and its last 128 tokens, through the sparse layer's own
  query projection, without position encoding.

The window and sparse layers multiply each head's output by a sigmoid gate
of their input, one value per dimension of each head ([gated
attention][gated]), and add a learned sink logit per head, starting at 0,
to the softmax's denominator, so a head can give almost no weight to
anything. HySparse2 names the gate and the sink for those two kinds only,
and the exact layers have neither.

**The second half reads the first half's output** (HySparse2's KV
bridging). The second half's exact layers make their keys and values from
the first half's output, while their queries come from their own input.
That output is the input Attention Residuals gives the first sublayer of
the second half ("Attention Residuals", below), normalized by the exact
layer's own attention norm. The first half's exact layer makes its keys
and values from its own input.

Two operations grow with the text: the exact layers' attention, which also
gives each query's chosen tokens, and the sparse layers' attention over
given tokens and the window (`kernels.py`). The window layers' attention
is a sparse layer's with no chosen tokens. Where kernels compute the
attention on a GPU, outside `torch.compile` (which fuses the PyTorch form
by itself), a training or evaluation pass runs it through the same
sparse kernels, which keep no scores for the backward pass. Elsewhere,
and in a generation cache's passes, it runs in PyTorch, in blocks of 128
queries that each read two blocks of keys. Either way its cost grows with
the text times 128.

`pytorch_kernels.py` states the two operations' contract and computes them
in plain PyTorch: the reference, which runs on every device. It computes in
FP32 one block of queries at a time, in the forward and the backward pass,
so its memory grows with the text and never with its square, and its
backward pass is written out. Two GPU forms keep the same contract:
**TileLang** (`tilelang_kernels.py`), on an NVIDIA GPU under Linux with
TileLang and the CUDA compiler `nvcc`, and **Triton** (`triton_kernels.py`),
on an NVIDIA or AMD GPU where Triton is installed. TileLang builds its
kernels at head widths 64, 128 and 256 only, so it computes at the head
width padded with zero columns to the next of those, as Triton pads it to
a power of 2: the results are those of the head width itself. Both GPU forms run the
exact layer as flash attention whose tiles stack a block of queries' heads
as rows, so a few heads still fill the matrix units, followed by a kernel
that picks each query's top 1,024 from the scores and the forward pass's
softmax denominators, with no scores written and nothing waiting on the
GPU from the host. It streams the keys again, recomputing the scores:
passes that count each query's scores in bins until the 1,024th lies in a
bin of a few hundred rows or of one score, then one that writes the rows
above that bin and ranks the bin's rows exactly. The rows chosen are an
exact top-`k` of the kernel's scores, the order among equal scores aside.
The first half's exact layer runs no choice at all. The sparse layer runs one program per
query that gathers its chosen rows and its window in blocks. Its backward
pass sums each query's gradient over its rows, one program per query. For
the rows' key and value gradients it takes the path that moves less data.
With more than 2 heads, each query's program also adds its share to its
rows' gradients by FP32 atomic additions. With 2 heads or fewer, it first
lists, for each table row, the queries that read it (once for all the
sparse layers that read one exact layer's choice), then sums each row's
gradients over its list, one program per row. Both the atomic additions
and the order in which they fill the lists vary, and both forms' exact
layers add their query gradients with FP32 atomic additions, so the
gradients can differ in the last bits from run to run. The Triton
kernels' tile sizes are Triton's autotuner's choice, which it keeps in
Triton's cache on disk: a later process on the same machine reads it
instead of timing every config again at its start, and gets the same
tiles.

Which form runs is decided once, before the first use (`wpt train` decides
it before training and logs it). `wpt train`, `wpt eval` and `wpt chat`
each take `--kernels` (or `WPT_KERNELS`) to name it: `triton`, `tilelang`
or `pytorch`. Unnamed, Triton runs where it is installed, since it is the
faster of the two (1.45 to 1.64 times on an RTX 4090 at width 512),
else TileLang where it is installed on an NVIDIA GPU under Linux, else the
PyTorch form. The
training and the evaluation `wpt manager` runs on rented machines name
`triton` unless its settings name the form. A form named after the decision
must be the one decided. A kernel form that cannot work stops the run with
its reason, never a switch to another form: one named but not installed,
TileLang named off an NVIDIA GPU under Linux, kernels named on a machine
with no GPU, TileLang without `nvcc` (on `PATH`, or under `CUDA_HOME` or
`CUDA_PATH`), a TileLang or Triton that fails to import, or a kernel that
does not compile or run on this GPU (each is compiled and run once, forward
and backward, on a small input). A failure inside a later call is an error,
never a switch to the PyTorch form.

Tests check the PyTorch form against the formulas, and each kind of layer
against HySparse2's formulas one token at a time. The tests of a kernel
form against the PyTorch form sit in one place
(`packages/wpt-model/tests/test_kernels.py`), at the orders' head widths
and 1,024 chosen tokens, and run for every form that runs on the machine.

## Long windows

Training targets 1M tokens and reaches it by stages in the order, with
segments on one GPU, windows split across several GPUs, and sparse layers
from the start (above). The stages are in training.md, "Stages".
A window too long for one GPU runs in parts, and both ways give exactly the
loss and gradients of the whole window at once, so they are hardware
options (training.md, "Long windows"), never part of the order:

- **Segments**, on one GPU (`wpt train --segment N`): the window runs as
  consecutive segments of `N` tokens, each passing to the next what its
  layers need of the earlier tokens, and the gradients flow back across
  every boundary.
- **Slices**, across processes (`wpt train --strategy split`):
  DeepSeek-V4's two-stage contextual parallelism ([V4][v4], section
  3.4.3). Each process holds a contiguous slice of `s` tokens; stage 1,
  each process sends the later ones what they need of its last tokens
  (the window layers' last 127 keys and values, Engram's tails, In-Place
  TTT's unfinished chunk and its updates); stage 2, an
  all-gather of every process's exact-layer keys and values gives each
  process the window's full set, in order, and each query's position
  names the tokens it sees. V4 pads that set at the tail because its
  slices hold packed documents of varying lengths; WPT's slices are equal,
  so none is needed.
- **Slices in segments**, both at once (`--strategy split --segment N`):
  when a slice is longer than `N`, the window runs as consecutive parts of
  `N` tokens per process, each split across the processes; process `i`
  trains the `i`-th segment of every part, the first process takes what
  the part before handed on, and each part hands on what its last slice
  ends with, so the gradients flow back across both kinds of boundary.

What crosses a boundary is what each layer passes on to later tokens
("What a layer passes on", below), the same description the generation
cache keeps.

A segment or slice holds a multiple of 128 tokens (the window layers'
window); a window's last segment may be shorter. A tail longer than a
slice (In-Place TTT's 1,023 tokens) is read from as many processes before
as it reaches. The losses are the parts' shares of the whole window's
(training.md, "Long windows").

### What a layer passes on

What a layer passes on to later tokens is said once, per layer, and both
training's segments and slices (`sequence.py`) and generation's cache
(`cache.py`) carry exactly that. Each module that reads earlier tokens
describes what it passes on (`passes_on()`, `carry.py`), as one of three
kinds:

- **a tail**: the last `n` time steps of a value each token has, with what
  stands before the text (zeros, or Engram's padding id);
- **entries**: one per token, every one kept;
- **a fold**: a running value that some tokens change, zero before the
  text; each pass gives its changes, each at the token after which it
  applies.

| Layer | What it passes on |
| --- | --- |
| window layer, the MTP block's window | a tail of the last 127 keys and values |
| exact layer | the key and value of every token |
| sparse layer | nothing: it reads the exact layer's |
| Engram | a tail of the last `N - 1` canonical ids; a tail of the last `3N` inputs of its convolution |
| In-Place TTT | a tail of the last 1,023 tokens' hidden states, embeddings and end-of-document marks; a fold of the chunks' updates since the document began |
| experts, Attention Residuals | nothing: they work token by token |

A query of a window layer sees itself and the 127 tokens before it, so 127
is all a later token needs. A segment, a slice and a cache take the same
values from the same description, so they cannot drift apart: a test
checks that a segment and a cache carry exactly the described values and
no other, and that both give the whole window's results. Taking tokens
back (generation, below) needs one rule per kind: a tail is cut back to
the tokens kept, entries past a row's tokens are never read, so they
need nothing, and a fold keeps the changes at the tokens kept.

A model that loops ("Loops", below) reads a window once per pass, and
training's segments and slices carry each pass's values apart, so they
still give the whole window's results.

## Experts

**Latent experts with SiTU-GLU** ([Kimi K3][k3]'s Stable LatentMoE). Routed
experts work in a narrower space:

```
z = W_down x
u = sum over chosen experts i of p_i E_i(z)
y = sum of the 2 shared experts E_shared(x) + W_up RMSNorm(u)
```

Shared experts are always active and work at full width ([DeepSeekMoE][moe]'s
idea: they hold common knowledge so routed experts can specialise). K3
uses half the width for the latent space.

Every expert is a SiTU-GLU feed-forward block, whose branches are capped by
scaled tanh functions so one unit's output never exceeds 100:

```
SiTU-GLU(x) = [4 tanh(W_g x / 4) * sigmoid(W_g x)] * [25 tanh(W_u x / 25)]
```

followed by the expert's down projection. On a GPU where Triton imports,
outside `torch.compile` (which fuses it by itself), the activation is one
Triton kernel each way (`wpt_model/situ_kernels.py`): it computes in FP32
and rounds once, where the operations written in PyTorch round each
result to BF16, and it keeps only the two branches' pre-activations for
the backward pass.

**Routing.** Router scores are `s = sigmoid(W_r x)`. The top `k` experts
are chosen on `s + b`; their weights are `p_i = s_i / sum of the chosen s`,
without `b`.

**Quantile Balancing** ([Kimi K3][k3]) sets the biases `b` so every expert gets
about its share of tokens, with no auxiliary loss. After each step's
forward pass, for `m` tokens, `n` experts and top `k`:

- each token's cutoff `alpha_i` is its `(k+1)`-th largest biased score;
- each expert's new bias is `-quantile_{1 - k/n}(s_j - alpha)` over the
  step's tokens, then all biases are shifted to have mean 0;
- the new biases route the next step, never the batch they came from, and
  are frozen for inference.

WPT reads the quantile from a histogram of each expert's `s_j - alpha`,
which every process adds up: 4096 bins within 0.5 of the expert's previous
quantile (`-b_j`), with values outside counted in the edge bins. A bin is
about 0.00024 wide, and one step moves a bias by at most 0.5.

**Hash routing in the first 3 expert layers** ([DeepSeek-V4][v4]). Early layers
have seen little context, so their experts are chosen from the token id
alone, by a fixed function, with equal weights `1/k`. DeepSeek-V4 does not
publish its function; WPT uses, for token id `t` and `j = 0 .. k-1`:
`expert_j(t) = (h(t) + j * n / k) mod n`, with
`h(t) = (t * 2654435761 mod 2^32) mod n`. These layers have no router and
no bias.

## In-Place TTT

In every sixth layer, from the first (layers 1, 7, 13 and 19 of 21), one
weight matrix changes while the model reads a text, as in [In-Place
TTT][ttt]. The paper changes the last projection of a gated feed-forward
block. WPT's feed-forward is a mixture of experts, so the matrix is the
down projection of the two shared experts, which are always active: their
two down projections side by side, `W` (`d_model x 2 shared_width`), and
their hidden states side by side, `z`.

The text is cut into chunks of 1,024 tokens, counted from its first
token. Each chunk reads the matrix that the chunks before it left, then
changes it by one step:

```
output of token t   z_t (W_0 + 0.3 S)^T, S the sum of the updates so far
update of a chunk   d = sum over its tokens of v_t z_t^T
target              v = W_target Conv1D(e)
```

`e` is the token embeddings, and `Conv1D` a depthwise convolution of 5
taps centred on the token, zero beyond the chunk. The update is one
gradient step of the paper's loss aligned with next-token prediction, the
negative inner product of `z W^T` and `v`; its rate is 0.3. The
convolution starts at zero and `W_target` as a diagonal of `N(0, 0.02^2)`
values, so the model starts as one without the change. Both are trained
with the model from the first update, with the experts' part.

Each document starts from the trained matrix `W_0`. The end-of-document
token closes its document. A token reads `S` only when no document ended
in its chunk before it. A chunk's update counts only its tokens after its
last end of document, the convolution reading none of the tokens before
that end. `S` starts over at a chunk that holds an end.

A pass computes the updates of the chunks it completes in one batch, `S`
at each chunk start by a running sum (the paper's prefix sum), and the
change for each run of tokens that reads the same `S` in one batched
product. That costs about two more down projections in each of these
layers. `S` is FP32. While no chunk ends within a pass, the convolution
and `W_target` get a zero gradient, so every process of DDP has one.

The paper's published code gives the chunk size, the centred convolution,
the rate, the layers and the initialization; the paper itself says "causal
padding" and "every sixth layer", and reports chunks of 512 to 1,024 best.
Counting chunks from the text's first token rather than from each
document's, and the shared experts' down projection as the matrix, are
WPT's own. All of these are open to review. Trained from scratch at 4B
parameters, a model with only window layers went from 9.91 to 26.80 on
RULER at 8,000 tokens, and one with full attention from 38.09 to 43.82.
The smallest model the paper measured has 500M parameters, and it was
never tested with a window of 128 tokens or with HySparse2.

## Loops

An order number, `loops`, sets how many more times the text passes
through every layer with the same weights, as in [Ouro][ouro]. At 0 the
model is one pass: nothing of the loop is in its weights or its cost.

- Each pass after the first takes the pass before's output, the input the
  output head would read, as its first source, where the first pass takes
  the embedding. Its sums of Attention Residuals cover its own sources
  only.
- The output head, the multi-token prediction block and the drafter's
  features read the last pass. In-Place TTT's targets are the token
  embeddings in every pass.
- The routed experts route each pass on its own, and R3 records and
  replays each pass's choices apart.
- The loss reads the last pass, unless `loop_losses` weighs others too
  (below). Ouro also trains an exit gate that learns when to stop
  looping; WPT has none, which is open to review.

The evidence is mixed, which is why this is a number and not a fixed
choice. In Ouro's test at equal compute, from 53M to 1.36B parameters,
the plain model had the lower loss in every case, by 0.015 to 0.039;
accuracy by number of passes peaks at 3 to 4. No paper combines loops
with HySparse2 or with Attention Residuals, so the rule for the sources
above and the cache's rule (Generation, below) are open to review. A
size's compute count holds `loops + 1` times the work per token of one
pass, or with `loop_layers` one pass plus `loops` times the looped
layers' work ("Model families").

**Decay between passes.** An order number, `loop_decay` (`lambda`, above 0
and below 1), changes what a pass after the first starts from. With `h0`
the first pass's first source (the embedding) and `out` the pass before's
output, it starts from `h0 + lambda (out - h0)` instead of `out`: each
pass's change to the input is shrunk at the end of the pass, so the
changes of earlier passes weigh geometrically less. This is the
decay-only rule of [Loopback][loopback] (its Eq. 13), computed in FP32;
it adds no weight. `decay_last_pass` says whether the last pass's output
is decayed the same way before the output head reads it, or read as it
is. Left out, there is no decay and the last pass is read as it is.

The paper always has layers that run once before and after the loop
(a prefix and a coda), loops only the middle ones, and decays after
every pass, the last one included, before the coda. A WPT model whose
loop runs every layer has no coda, a shape the paper never tested: with
`decay_last_pass` the output head reads `h0 + lambda (out - h0)`, the
embedding itself weighing `1 - lambda` in it; without it the head reads
`out`, and only the passes in between are decayed. Which of the two
suits a model that loops every layer is not known. With `loop_layers`
(below), layers run once after the loop, as in the paper. In the paper's
ablation (its Table 4a), at 191M parameters with 2 layers before, 4
layers looped 4 times and 2 after, the decay of 1/4 lowered the
validation loss from 2.6802 to 2.6220 against 2.6234 for a model without
a loop; at 374M (24 layers) from 2.4415 to 2.3999 against 2.4198. Those
runs had no Attention Residuals between sublayers and no Mixture of
Experts.

**A loss on every pass.** An order number, `loop_losses`, gives each pass
a weight in the next-token loss, one weight per pass (`loops + 1`), each
at least 0 and the last above 0. The training loss is then
`sum_r (w_r / sum w) x L_r`, `L_r` the next-token cross-entropy of the
output head reading what follows pass `r` (with `decay_last_pass`, its
decayed output): the loop forcing of [Mixture-of-Loops][mol] (its Eq. 8),
through the one output head and final norm. Left out, the last pass
alone has weight 1, the loss of a model without it. The scores the
model gives, in generation and in evaluation, are the last pass's
whatever the weights; the multi-token prediction block reads the last
pass alone. In that paper every 113M model, its plain 2-pass loop
included, trains with a loss on both passes, so it measures no loop
without one. Its Table 3 compares the weights on its own looped model
(U-MoLT, which also reads keys and values across passes): at 1:1, 1:2
and 1:4 the second pass's cross-entropy is 3.279, 3.276 and 3.272, and
the first's 3.306, 3.315 and 3.322.

**Looping the first layers.** An order number, `loop_layers`, makes the
loop run only the first `loop_layers` layers on every pass; the other
layers run once, after the last pass, as ordinary layers. Left out, the
loop runs every layer.

- The looped layers have their own sums of Attention Residuals, which
  end with one more sum, the loop's output, where the head's sum ends
  them without it. The layers after the loop have theirs too: their
  first source is what follows the last pass (its decayed output with
  `decay_last_pass`), and the head's sum ends them. The groups of
  `residual_groups` are cut where the loop ends, so a group that spans
  the boundary becomes two. The model has the same parameters as one
  that loops every layer.
- A second-half exact layer after the loop makes its keys and values
  from the first half's output of the last pass, when the first half is
  in the loop.
- A loop may not end just before a sparse layer: the sparse layer reads
  its exact layer's keys, values and choice of tokens, made on every
  pass. The order is refused, naming both layers.
- Engram and In-Place TTT keep their layers wherever the loop ends.
- With `loop_losses`, the layers after the loop run once more from what
  follows each other weighed pass, and the head reads that.
- A drafter's features keep the layers' numbers: the feature of layer
  `loop_layers` is what follows the loop.
- R3 records the routed experts of the layers after the loop at the last
  pass only, and the cache keeps their values at the last pass only.

With `decay_last_pass`, this is Loopback's shape without a prefix: the
loop, the decay after every pass, then layers that run once. It is not
Mixture-of-Loops' U-MoLT, whose second half reads earlier tokens through
keys and values made from the looped half's output: the layers after a
WPT loop are ordinary layers.

## One vocabulary table

A model has two vocabulary tables, the embedding and the output head, each
of the vocabulary's size times the width. An order number, `tied_tables`,
makes them one: the output head keeps its table, and a token's embedding is
its row of that table times a fixed number, the square root of 3 times the
width. The head starts as it does with two tables, and that number gives
the layers an input of the scale a separate embedding has.

- Every reader of the embedding reads the same rows: the first pass, the
  multi-token prediction block, In-Place TTT's targets and the drafter.
- The table is in the part `output-head` ("Parts", below); the part
  `embedding` then holds nothing.
- The model has one table's parameters fewer, and its compute per token
  does not change.

Why, and how sure: [knowledge/vocabulary-tables.md](knowledge/vocabulary-tables.md).
No order sets it.

## Engram memory

[Engram][engram] (DeepSeek) adds a lookup that costs almost no compute: embeddings of
the last few tokens' N-grams, read from hashed tables, added to the
hidden state through a learned gate.

- Token ids first map to a canonical id: tokens whose decoded text is the
  same after NFKC normalization and lowercasing share one id (a map built
  from the vocabulary; tokens that are not whole UTF-8 text map to
  themselves). This ties the tables to the vocabulary.
- For each order `n = 2 .. N` (N = 3 in Engram-27B) and each of `K` hash
  heads (8 in Engram-27B), the suffix N-gram's canonical ids are hashed into
  a table of prime size and one row is read. In WPT, each (order, head)
  pair has its own table, of a distinct prime size: the smallest primes at
  or above the order's table size. The rows are concatenated into
  `e_t`. DeepSeek describes its hash only as multiplicative-XOR; WPT's
  constants are in the code.
- Gate: `k_t = W_K e_t`, `v_t = W_V e_t`,
  `a_t = sigmoid(RMSNorm(h_t) . RMSNorm(k_t) / sqrt(d))`, value `a_t v_t`,
  where `h_t` is the input Attention Residuals gives the layer's attention.
- Then `Y = SiLU(Conv1D(RMSNorm(V))) + V`, a depthwise causal convolution
  of kernel 4 and dilation N, starting at zero so the module starts as
  nothing. The attention reads `h_t + Y`, and the sublayer's output is
  `Y` plus the attention's output, the share a residual sum would add
  (the papers place Engram on a residual stream, never between
  Attention Residuals' sums).
- The order names the layers (Engram-27B: layers 2 and 15; the 3B
  reference: 2 and 6).

DeepSeek found a U-shaped law: at fixed parameters, moving 20 to 25% of the
sparse parameter budget (everything not active per token) from experts to
Engram gave the lowest loss. Lookups depend only on input tokens, so large
tables can stay in CPU memory and be fetched ahead.

## Attention Residuals

[Attention Residuals][attnres] (Kimi) replaces the running residual sum
with softmax attention over depth. The paper counts each attention and
each feed-forward block as a layer; this section calls them sublayers.
Each sublayer `l` reads

```
h_l = sum_i alpha_(i->l) v_i
alpha_(i->l) = softmax_i(w_l . RMSNorm_l(v_i))
```

over its sources `v_i`. `w_l` is the sublayer's own learned vector, its
query. Every query starts at zero, so training begins from a plain average
of the sources, as the paper requires. The RMSNorm keeps a source of large
magnitude from taking every weight. The first sublayer has one source,
the embedding, and reads it as it is.

The sublayers are cut into `residual_groups` consecutive groups, an order
number (the paper's Block AttnRes). A sublayer's sources are the
embedding, the sum of the outputs of each earlier group, and the sum of
the outputs of its own group so far, when the group already holds one.
With twice as many groups as layers, one per sublayer, every output is a source of
its own: the paper's full form, which WPT's orders use. Sublayer `j` (from
0) belongs to group `j x groups // sublayers`, so groups differ by at most
one sublayer; the paper leaves the split of uneven groups open. The output
head reads one more sum, with its own query, over every group. Each sum
adds `2 x d_model` parameters: its query and its RMSNorm's weight.

At 194M to 528M active parameters, the paper's full form scores slightly
above mHC and the grouped form matches it; it also reports that the design
prefers more, narrower layers.

The sources of one pass are kept side by side in one tensor of the compute
dtype (BF16 on a GPU), each written once, when complete, with its inverse
RMS `r_i`. Since `w_l . RMSNorm_l(v_i) = ((w_l * g_l) . v_i) r_i`, with
`g_l` the RMSNorm's weight, a sum is two batched products over that
tensor, one for the scores and one for the weighted sum. On a GPU where
Triton imports, each sum's forward and backward passes are Triton kernels
instead, in every pass, compiled or not, which read each source from
memory once
(`wpt_model/residual_kernels.py`). The scores and the softmax are FP32. The backward pass is written out, so that it keeps
nothing per sublayer but the scores and weights: the sources stay only in
that tensor. The sources' gradients sit side by side in one tensor too:
each sum's backward pass adds its share to all of them at once, and each
source takes its own when every sum that reads it is done. The Triton
kernels keep no scores or weights, only per token the log of the
softmax's denominator, and compute them again from the sources; they add
nothing into the gradients' tensor: each sum's backward pass keeps its output's
gradient, and a source's gradient is one FP32 sum over the sums that read
it, made when every one of them is done. A test checks
the sums and every gradient against the formula over stacked sources.

## Multi-token prediction

One extra block ([DeepSeek-V3][v3]'s MTP module) predicts the token after next:
it joins the final hidden state and the next token's embedding (each
RMS-normalized, concatenated, projected back to the model width), runs one
layer, and uses the shared output head. WPT's layer is attention over the
last 128 tokens alone and a dense SiTU-GLU as wide as the two shared
experts together, on one stream. Its attention is a window layer's
(above); [MiMo-V2-Flash][mimo] chose a sliding window for its MTP
block too, so that the block stays light when it drafts tokens. It never
sees further back than 128 tokens. Its loss is added with weight 0.3,
then 0.1 (DeepSeek-V3 switched after 10T of 14.8T tokens; the order sets
the switch). The model works without it at generation time; speculative
decoding uses a separate drafter instead ("The drafter", below), and the
block's window is kept in the generation cache for a drafter that reads
it.

## Generation

Generation reads every token once, through a cache of what each layer
passes on ("What a layer passes on", above; `packages/wpt-model`,
`cache.py`), instead of reading the whole text again for each new token.
The cache is forward's own pass, with the tokens at each row's place: the
same code as training, which reads the earlier tokens' values from the
cache instead of from a segment before.

- **Window layers and the MTP block's window** keep the last 127 keys and
  values.
- **Exact layers** keep the key and value of every token; the sparse
  layers read them.
- **Engram** keeps each attempt's last `N - 1` canonical ids and the last
  inputs of its convolution; its tables stay on the model's device.
- **In-Place TTT** keeps each attempt's unfinished chunk and its sum of
  updates: every attempt holds its own copy of the changed matrix.
- The experts keep nothing, but a cache can record each attempt's routed
  experts as its tokens are read, for R3 (training.md).

A model that loops keeps each pass's values apart: each call runs every
pass over its new tokens, and every pass reads its own values of the
earlier tokens, as in training, so a text read in any number of calls
gives what reading it whole gives. Ouro shares one pass's keys and values
with the others and measured 78.85 against 78.92 on GSM8K; WPT shares
nothing: a model trained here writes no coherent text through the last
pass's values alone. Every pass reads the prompt through every
layer: stopping the last pass after the first half for the tokens whose
scores are not needed is not built.

Only the exact layers' keys and values grow, one key and one value per
token, exact layer and pass, so generation goes on past `block_size` with
no limit. It is the same computation as reading the whole text, so the same
scores within rounding: on the CPU, tests check each kind of layer and
the whole model read in pieces against reading it whole, and the tokens
that sampling draws with one seed. A rounding can still fall the other way
(a near tie in an exact layer's choice or the routers'), which changes a
score by more than the rounding itself; the tests allow for that. A prompt
is read through every layer, the second half included.

A cache holds several rows, the attempts of one task, read together in one
batch: each row has its own length, and each call gives each row its own
number of new tokens, the rest of the row being padding that changes
nothing. A prompt shared by the attempts is read once and copied into
every row.

**Speculative decoding.** A drafter proposes a few tokens for each row
(`DRAFT`, 5, the most DSpark's production drafter proposes); the model
reads those worth verifying in one pass (the scheduler, "The drafter",
below); each is kept with probability `min(1, p/q)` (`p` the model's
probability, `q` the drafter's), a rejected one is replaced by a draw from
`max(0, p - q)`, normalized, and when all are kept the model adds one more
token. This leaves the model's own distribution unchanged.
The cache then undoes the rejected tokens, by kind ("What a layer passes
on"): each tail is cut back to the kept tokens, keys and values past a
row's tokens are left unread and written again when the row reaches them,
and the recorded routes are cut. WPT's drafter is DSpark's
("The drafter", below); the machinery takes any drafter (`wpt_inference`
says what one gives), and generation without one samples a token at a
time through the cache.

**Kernels.** The exact and sparse layers' attention runs through the same
form as training ("Attention"); the window layers' runs in PyTorch.

`wpt_inference.Writer` writes through the cache (`sample_tokens` for one
text); `wpt chat`, the sample `wpt train` prints, the drafter's training
and the attempts of the `finish-tasks` and `finish-tasks-with-less-writing`
steps use it. `wpt chat` and `wpt train`'s sample draft with the model's
drafter. The attempts of those steps are written by the model alone,
without a drafter: the weights change at every update, a drafter is trained
only at the end of a run, and the attempts then depend on the seed alone,
never on this machine's times.

## The drafter

The drafter is a separate model like DSpark, built with the published
values of DeepSeek's production drafter, and learned only from answers
WPT's own model writes. [DSpark][dspark] is DeepSeek's drafter for V4, on
the parallel backbone of [DFlash][dflash]. The code is `wpt_model.drafter`,
and its training `wpt_inference.drafting`; its weights live in the model
folder's `drafter/` (`config.json`, `model.safetensors`,
`provenance.json`), under the model's license and header
([licenses.md](licenses.md)).

**One unit**. The model and its drafter are one unit and a hard dependency,
with separate weights:

- `wpt train` trains a drafter at the end of every run, after the
  order's last step, against the final weights, the model frozen
  (training.md, "The drafter"). The drafter is not a training step.
  `--no-drafter` saves the model without one, the weights the same.
- The drafter's `provenance.json` names the sha256 of every weight file it
  was trained against (`target.weights`). `wpt_model.load_model` refuses a
  folder whose drafter is missing, was not saved whole, or names other
  weights (`DrafterMismatch`). The one folder without a drafter it loads
  says it has none: its `provenance.json` holds `"drafter": false` among
  its `options`, which `wpt train --no-drafter` writes. Such a model writes
  alone, without speculative decoding: the same text, slower.
- A drafter that cannot be trained (too few prompts, a loss that is not
  finite, no draft verified) fails the run, which writes nothing.
- The prompts are the beginnings of the training documents the run read
  (each the document start and its first `prompt_tokens` tokens; the rest
  of the document is never read), and the model alone writes the answers,
  each up to the end of a document and, when a step trained on
  conversations or tasks, also up to `<|im_end|>`, the end of a turn. A
  test checks that no text past a prompt reaches the drafter's training.

**Settings.** The model order's `drafter` section ("Model
families") gives the drafter's shape (`features`, `layers`, `block`,
`markov_rank`) and its training (`epochs`, `learning_rate`, `weight_decay`,
`warm_up`, `clip`, `anchors`, `batch`, `prompts`, `prompt_tokens`,
`answer_tokens`, `temperature`, `top_k`, `held_out`, `ece_bins`, `seed`); a
setting left out keeps its default below. The families of the
repository state the training values explicitly ("Model families"),
equal to the defaults, so a change of a default does not change their
drafters. `features`
numbers the model's layers from 1 to its last;
`DrafterConfig.for_model` checks them against the model's layers when a run
starts, so `wpt train` refuses a layer the model does not have before
any step runs.

A round drafts a block after the anchor, the last token the model wrote
and has not read yet:

- **Context** (DFlash). The hidden states of 5 target layers, spread evenly
  from the 2nd to the third-to-last (fewer in a short model: the 2nd and 3rd
  in 5 layers), at each of the last 128 tokens the model read, side by
  side, projected to the model's width and RMS-normalized.
- **Backbone** (DSpark's production values). 3 layers, each a window
  attention of 128 tokens and experts. The block is the
  anchor's embedding and 4 mask vectors (DSpark's anchor-first block: 5
  places give 5 drafts), read in one pass. Every layer reads the context as
  keys and values through its own key-value projection (DFlash's KV
  injection).
- **Scores.** The target's output head, frozen, gives each place's base
  scores `U_k`; the target's embedding, frozen, reads the anchor.
- **Markov head** (DSpark's default, and its production head): `p_k(v) =
  softmax(U_k + W1[x_{k-1}] W2)`, sampled from left to right, `x_0` the
  anchor. DSpark's rank is 256 at V4's width of 4096; WPT's default rank
  follows the width, `256 * d / 4096`: 8 at width 128, 20 at 320 and 32 at
  512. With a vocabulary of 8,192,
  the head holds 131K of the drafter's 1.5M parameters at width 128, 328K
  of 11.7M at 320 and 524K of 28.9M at 512; at rank 256 it held 4.2M of
  5.6M, 4.2M of 15.6M and 4.2M of 32.6M.
- **Confidence head**: `c_k = sigmoid(w . [h_k ; W1[x_{k-1}]] / T_k)`, the
  chance that draft `k` is kept when every draft before it was; `T_k` comes
  from DSpark's Sequential Temperature Scaling after training.
- **Scheduler** (DSpark's Hardware-Aware Prefix Scheduler, its Algorithm
  1). A draft is kept with the product of the confidences up to it; drafts
  are verified while the expected tokens per second of the model's pass
  grow, from a table of the pass's times, and the first that does not pay
  stops the admission, so a draft's admission never depends on the draft
  itself. The times are measured once per run of `wpt train`
  (`wpt_inference.measure_costs`) and written into the
  run's record as `scheduler-costs.json`, with the setup they were measured
  on: the model's config, the drafter's block (the drafts per round), the
  writer's rows (1) and the device, by its kind and name (such as "cuda
  (NVIDIA GeForce RTX 5080 Laptop GPU)"), never its number among GPUs of
  one name. On the CPU, the name is the processor's, such as "cpu (AMD
  Ryzen AI 7 450 w/ Radeon 860M)", so times measured on one processor are
  never replayed on another. `wpt_utils.device.processor_name` reads it
  from what each system reports: the first `model name` line of
  `/proc/cpuinfo` on Linux, the registry's `ProcessorNameString` of the
  first processor on Windows, `sysctl -n machdep.cpu.brand_string` on
  macOS, with runs of spaces made one (`platform.processor()` is empty on
  many Linux systems and names only the architecture elsewhere). A Linux
  `/proc/cpuinfo` without a `model name` line, as on most Arm processors,
  gets a name built from its `CPU implementer` and `CPU part` codes, as
  `lscpu` does: the implementer, then each kind of core once, such as "cpu
  (Arm Limited Cortex-X925 + Cortex-A725)". WPT writes its own table of
  these names from the facts Arm publishes (the implementers in Arm's
  register reference, DDI 0601, `MIDR_EL1`; each part in its core's
  Technical Reference Manual), never from `lscpu`'s code, and keeps it to
  the cores of common Arm machines that run PyTorch under Linux; a part it
  does not hold is named by its implementer and its code in hexadecimal,
  such as "Arm Limited part 0xd0d", and an implementer Arm does not publish
  by its code, such as "implementer 0x61 part 0x22". Only when those codes
  are missing too is it the machine's architecture, such as `aarch64`, the
  last resort. On macOS, a `sysctl` that is not on the search path never
  stops a run: the name is then the architecture. A file recorded on
  the CPU before, whose device is `cpu` alone, is refused like any other
  setup. `wpt train
  --scheduler-costs FOLDER` replays the times an earlier run recorded in its record folder `FOLDER`
  instead of measuring them, and `wpt chat --scheduler-costs FOLDER` reads
  them instead of measuring them when it loads the model; each refuses
  times recorded on another setup, naming each difference, `wpt train`
  before any step runs. A writer given no
  times verifies every draft.

  What a replay guarantees, exactly: the scheduler uses the recorded
  seconds, so for the same drafter confidences it admits the same drafts
  as the run that measured them. With the same weights (model and
  drafter), the same prompt, the same sampling settings and the same
  seed, on a device whose passes give the same numbers, the writer then
  verifies the same drafts and draws the same tokens as a writer given
  that same table. The setup check compares only what the file records:
  the model's config, the block, the rows and the device's kind and name.
  It does not compare the weights, the drivers, the library versions or
  the machine's load, and it does not check that the times are true for
  this machine: times that are wrong here make writing slower, never
  change the model's distribution.

Training (DSpark, section 3.3): the model writes an answer to each prompt,
then reads each answer whole; random anchors in the answer start blocks
of 5; the drafter learns
`0.1 L_ce + 0.9 L_tv + 1.0 L_conf`, each place `k` (from 1) weighed
`exp(-(k - 1) / 5)`: cross-entropy on the answer's tokens, the total
variation distance to the model's distribution, and binary cross-entropy
of the confidence against `c*_k = 1 - ||p_d - p_t||_1 / 2`. The target is
frozen; only the drafter learns.

Details DSpark leaves open, and WPT's choice for each:

- Width: the model's, which sharing its embedding and head requires.
- The backbone's layers: a window layer's attention, as the multi-token
  prediction block's, and the model's
  experts (latent SiTU-GLU experts, 2 shared, the model's numbers), every
  layer routed by its router and balanced by Quantile Balancing (no hash
  routing); DSpark says only "MoE layers with mHC".
- The links between the backbone's sublayers: Attention Residuals in its
  full form, as the model's.
- Each target layer's hidden state: the input Attention Residuals gives
  the next layer's attention; the last layer's is the output head's input.
- What each place sees: the whole block both ways (DFlash), and the context
  tokens within 128 positions of it, so the window is each place's own.
- The mask vector: learned, starting from a normal draw at the embedding's
  scale. The final RMSNorm before the head: the drafter's own.
- Starting values: `W2` at zero, so the Markov head starts as nothing; the
  confidence projection has a bias.
- The drafts' distribution: the writer's temperature and top `k` applied to
  the drafter's scores, so it follows what the model samples.
- The answers: written side by side, up to 32 prompts per pass of the
  model and at most 2,048 prompt tokens in a batch's first pass, which
  bounds its scores to 512 MiB at a vocabulary of 65,536. Each prompt has
  its own random stream, seeded from the drafter's `seed` and the prompt's
  place, so each answer is what the model writes for that prompt alone
  with that stream: token for token on the CPU (`test_drafting.py`),
  within rounding on a GPU. A row leaves the batch at its stop token or
  its last allowed token. The calibration still drafts for one held-out
  prompt at a time: batching it needs a stream per row in the drafter's
  draws and in the rule that keeps or rejects them.
- Calibration: on real outcomes. On the held-out prompts, the drafter
  drafts for the model, every draft verified, its temperatures at 1, and
  each round records each draft's confidence and whether the model kept it
  (a draft after a rejected one counts as not kept). Each position's
  temperature is chosen among 200 values evenly spaced in log from 0.05 to
  20 to lower the expected calibration error over 15 equal-width bins on
  those outcomes, position by position (Sequential Temperature Scaling).
  The same rounds measure how many drafts the model keeps, recorded in the
  drafter's provenance.
- The scheduler: DSpark's production server schedules from the confidences
  of two steps before and drops the early stop, for its zero-overhead
  scheduling, which WPT does not have; WPT runs the lossless Algorithm 1 as
  published. WPT's batch reads every row as wide as its widest, so a pass's
  cost follows its width alone: widths are admitted from the narrowest,
  each with every row's draft at that place, instead of drafts one by one
  (a draft inside a width already paid for costs nothing). The time table
  holds, for each width, the fastest of 3 passes through a cache that has
  read 256 tokens, measured once per run for one row.
- A row whose pending tokens are more than its anchor (a tool result) first
  has them read in a pass of their own, so the drafter's context reaches
  its anchor.
- The update rule: the model's ("Training"). The fusing projection, the
  attention matrices and the expert matrices take NorMuon, head by head and
  expert by expert as in the model; the mask vector, the norms, the
  queries of Attention Residuals, the Markov head's `W1` and `W2` and the
  confidence projection take AdamW, as the model's embedding and output
  head do. AdamW's part keeps the model's betas, 0.9 and 0.95.
- Training values DSpark does not give come from DFlash: a learning rate
  of 6e-4, gradients clipped at 1, a cosine schedule after a 4% warm-up,
  512 anchors per answer redrawn each epoch; DSpark's 10 epochs; weight
  decay 0, 8 answers per update, 256 prompts of at most 64 tokens each
  drawn from the training documents by a seeded reservoir, answers of at
  most 256 tokens written at temperature 1 without top `k`, and 10% of the
  prompts held out (WPT's).

Conflicts with DeepSeek-V4's published values: none; the drafter replaces
V4's MTP-1 for drafting, as DSpark replaced it in V4's production. A cost
of the scheduler: it follows this machine's measured times, so a seeded
generator may draw other tokens from one run to the next (never another
distribution), unless the run replays the times an earlier one recorded.
The attempts of the task steps are written without a drafter, so they
never depend on the times.

## Parts

A training step locks parts of the model by name ([training.md](training.md),
"Locked parts"). The model gives its parts these names (`wpt_model.PARTS`),
and every parameter belongs to exactly one of them (`WPT.parts`):

- `embedding`: the token embedding (nothing with `tied_tables`, "One
  vocabulary table");
- `attention`: every layer's attention (window, exact or sparse, with its
  norm);
- `routers`: the routers of the experts layers;
- `balancing`: the routers' balancing biases, which no gradient trains:
  Quantile Balancing sets them after each update, unless a step locks
  them;
- `experts`: the experts, shared and routed, with their norm and In-Place
  TTT's convolution and `W_target`;
- `engram`: the Engram memories;
- `residuals`: the queries and norm weights of Attention Residuals, the
  sums of the layers after a loop of the first layers included;
- `final-norm`: the last norm;
- `output-head`: the output head;
- `multi-token-prediction`: the MTP block.

## Training

**Optimizer.** NorMuon for matrix weights ([NorMuon][normuon], as
[IMU-1][imu1] uses it), built on Muon as [DeepSeek-V4][v4] runs it (its
Algorithm 1). The momentum (with Nesterov's trick) is divided by its
Frobenius norm and orthogonalized by hybrid Newton-Schulz steps, 8 with
coefficients `(3.4445, -4.7750, 2.0315)` that bring the singular values
near 1, then 2 with `(2, -1.5, 0.5)` that settle them at 1, into `O`.
Each row of `O`, one output neuron, is divided by the root of a running
mean of its squares, `v = 0.95 v + 0.05 mean(O^2)` over the row, plus
1e-8. The result is scaled to an RMS of 0.18, DeepSeek-V4's, so AdamW's
learning rates carry over. Moonlight reports Muon reaching AdamW's quality
with about 52% of the compute ([Moonlight][moonlight]); IMU-1 measured
NorMuon with cautious weight decay lowering the loss of a 70M model from
3.250 to 3.125 for about 3% more time.

Weight decay is cautious ([cautious weight decay][cwd], IMU-1's rule): with
`u` the update of a matrix or vector, `W = W - lr * (u + wd * [u W > 0] W)`,
an entry decaying only where its update and its weight have the same sign.
NorMuon's paper gives no Nesterov momentum, no epsilon and an RMS of 0.2,
and IMU-1 orthogonalizes with 7 Polar Express steps: the momentum, the
steps and the RMS here are DeepSeek-V4's, which the learning rates were
fitted with, and the epsilon is WPT's own, all open to review.

- Per-head NorMuon ([K3][k3], [DeepSeek-V4.1][v41]): the query projections
  and the output gates of every attention layer are orthogonalized head by
  head, so heads with large gradients do not dominate the others. Each
  layer has one key-value head, so its key and value projections are one
  matrix each.
- No QK-Clip: every attention layer RMS-normalizes its queries and keys
  (QK-norm), which keeps the scores from exploding.
- AdamW for the embedding, the output head, RMSNorm weights (DeepSeek-V4's
  split), the queries of Attention Residuals, the convolutions (Engram's
  and In-Place TTT's), and for the attention
  sinks (biases and scaling factors take AdamW in
  [DeepSeek-V4.1][v41]); Adam with 5 times the base learning
  rate and no weight decay for Engram tables (DeepSeek's Engram setting).

The kind of weight chooses the rule, never the training step: every
built-in step trains under this rule, through the `by-part` optimizer
([training.md](training.md), "Optimizers"), and so does the drafter ("The
drafter").

**Average.** `read-text` keeps an exponential moving average of the
model's last 10 states, 2% of its updates apart, with a factor of 0.8, as IMU-1
averages its last checkpoints ([training.md](training.md), "Optimizers");
IMU-1 measured its 430M model's mean benchmark score rising from 0.560 to
0.574 with it. The steps after it keep no average.

**Schedule.** Multi-step for `read-text` ([DeepSeek LLM][llm], also used in the [mHC paper][mhc]): linear
warm-up to the maximum over the order's `warmup`, a share of the
updates (4% by default), then 31.6% of
it after 80% of the training tokens and 10% after 90%. DeepSeek reports final results matching
cosine decay, and a run can be extended by continuing from a checkpoint
taken before the first drop; a size's decay branches go on from
such a state ("Model families"). The steps after it run at a constant learning
rate. Their early stop on the validation loss is on, but their
environments make no evaluations, so it never stops them and they keep
their last weights.

**Numbers.** BF16 for weights' computations on GPUs, FP32 for the weights
themselves, the optimizer state, the attention scores and softmax, routing
scores, the scores of Attention Residuals and In-Place TTT's sum of
updates; the sources of Attention
Residuals are BF16 on GPUs. CPUs compute in FP32. Training on a CUDA
device computes FP32 matrix products in TF32. No layer uses dropout, as in the
papers.

**Memory.** Recomputing, which `wpt train` chooses for each stage from
the GPU's memory unless `--recompute` or `--no-recompute` says, keeps only
the input of each window layer's attention and of each experts sublayer
during a pass and runs the sublayer again during the backward pass: the
same result with much less memory, and slower (training.md). The exact and sparse layers' attention
runs once and keeps what its backward pass reads: their kernels keep only
their inputs, output and softmax denominators, so running them again
would cost a second forward pass of the costliest kernels for little
memory. The sources of
Attention Residuals stay in memory either way: each is a sublayer's output.
The sublayers' second run records no balancing counts.

**Repeatability.** Runs stay the same for one seed on any hardware
(training.md): routing ties, balancing quantiles and expert gradient sums
are computed in a fixed order, over the whole global batch.

## Model families

A model family, `data/orders/<name>.json`, is one file for the sizes of
one model, named by what the family is for. It writes once what its
sizes share, then a table of its sizes. `FAMILY:SIZE` names one size
wherever an order's name goes (`wpt build-resources`, `wpt train`,
`WPT_ORDER`). A release model is rebuilt from the family file, the name
of its size and the repository.

**What a size resolves to.** Each size resolves to a model order
(`wpt_orders.family.resolve`, checked against `model.schema.json`),
which holds:

- the data: a depot (`depot`, its
  public address, [data-selection.md](data-selection.md)), the name of one of
  its snapshots (`snapshot`) with the change number it reads (`change`) and
  its digest (`digest`), the roles of the portions its corpus is cut into
  (`roles`: `held_out`, `pretrain`, `finetune`; `pretrain` always;
  `held_out` only with a `total_tokens` large enough for it; left out:
  `held_out` and `pretrain`; [data-selection.md](data-selection.md), "The
  portions of a corpus"), the share of the plain text each tag takes
  (`shares`, in order, a tag once, summing to at most 1; left out: the
  snapshot's own proportions), the size of the pretraining portion in
  tokens (`total_tokens`), the size of the fine-tuning one
  (`finetune_tokens`; left out: every conversation and task left; only with
  `finetune`) and the vocabulary (`vocabulary`: its
  file's `sha256` and its `size`, or the `size` and `sample_bytes` of one
  the freight learns, [tokenizer.md](tokenizer.md), "A vocabulary in an
  order"). A data part that names a dataset of a depot instead, as the
  orders of models trained before snapshots do, is refused
  ([training.md](training.md), "Models trained before snapshots");
- the architecture's numbers: the layout of the layers (`window_layers`,
  `sparse_groups`, `sparse_layers`, "Attention"), width, attention heads
  and head width, expert counts and widths, top `k`, latent width,
  Engram orders, heads, table sizes and layers, the groups of Attention
  Residuals (`residual_groups`), the extra passes of the loop (`loops`,
  "Loops"), optionally the rules of how the passes
  follow each other, are trained and which layers they run (`loop_decay`,
  `decay_last_pass`, `loop_losses`, `loop_layers`, "Loops"), and
  optionally one vocabulary table in place of two
  (`tied_tables`, "One vocabulary table");
- the `seed`;
- the `steps`, in order, each a step with, when they differ from the step
  file's, its optimizer and that optimizer's settings, its settings and
  the parts it locks ([training.md](training.md), "Steps"): the
  `read-text` step's stages (each its window length and its share of the
  tokens), tokens per update, readings, evaluations, MTP loss weights and
  `by-part` settings among them; the model's context is the last window
  length of the last step that reads text windows;
- the drafter (`drafter`): its shape and how it is trained ("The
  drafter"; an empty section keeps every default);
- a `note`, which says what the family is for.

Settings that change the result come only from the family. Command-line
options remain only for what does not: device strategy (splitting windows
across processes included), number of processes, streaming, output folder,
run name, resuming, micro-batch, recomputing, segments, checkpoints, weight
file size and restarts. A vocabulary is named by its sha256 or learned
by its own request, never held in the family, since several models share
it. A model has at least 5 layers: one window
layer on each side of an exact layer, then one exact and one sparse layer.
A head is at least 64 wide and even, for the 64 rotary dimensions.

A step whose optimizer is `muon` or `adamw` is named `by-part`; the
entry keeps its own settings and gains none.

**The table of sizes.** `sizes` holds `columns` and `rows`. The first
column is `size`, each row's name. Each other column is the path of a
setting that varies between sizes, its names joined by `.` from the top
of a model order: `model.d_model`, `data.total_tokens`,
`data.vocabulary`, `steps.1.optimizer.settings.lr`, the steps numbered
from 1. A row is a size's name, then one value per column. A setting
without a column is common to every size. No value is computed from
another: every size's values are written.

```json
"sizes": {
  "columns": ["size", "model.d_model", "model.attention_heads", "data.total_tokens"],
  "rows": [["64", 64, 1, 106334292], ["128", 128, 2, 338014608]]
}
```

`wpt_orders.family.check` refuses, with a sentence that says what is
wrong:

- a first column other than `size`, a column named twice, or a column
  inside another;
- a path that names no setting of a model order, or a step the family
  does not write;
- a setting written both as common and as a column;
- a step's `prefix_tokens`, written anywhere;
- a size named twice, or by a name other than lowercase `a-z`, `0-9`,
  `.`, `_` and `-`;
- a row with another number of values than the columns, or with an empty
  value (`null`): every size fills every column;
- with branches, a step listing stages whose `mtp_switch` is not 0.8 or
  whose `first_drop` is not 0.8 (below).

`wpt_orders.Orders.check` then applies the rules of a model order to
every resolved size (shares, empty data fields, the fine-tuning portion,
the held-out portion), naming the file and the size.

**Tokens and corpora.** A size's `data.total_tokens` is the tokens it
reads. Sizes whose data parts are the same but for `total_tokens` share
one corpus, of the most tokens any of them reads: each resolved order's
data part asks for that corpus. Each step whose settings list `stages`
gets the size's own `total_tokens` as its `prefix_tokens`, so it reads
the first part of the shared corpus ([training.md](training.md),
"Prefix"), and every size sees the same texts in the same order. Sizes
of other data parts, another vocabulary among them, read corpora of
their own. `wpt build-resources FAMILY:SIZE` asks the freight for the
corpus of that size, and its vocabulary when the family learns one.

The model's provenance, its run's record and its model card name the
family and the size, and hold the size's whole resolved order.

**Other runs of a size.** A size of a family whose order is one step
listing stages has other runs, each named `FAMILY:SIZE:RUN` wherever an
order's name goes:

- `branch-SHARE`, a decay branch, one per share of the family's
  `branches` (above 0, at most 1): SHARE times the size's updates,
  rounded half up. Every branch of a size goes on from one trunk: the
  size's step at its peak rate after its warm-up, with no drop and the
  early weight of multi-token prediction. A branch starts its decay
  after 80% of its updates, rounded down, where by-part's first drop is
  ("Schedule"). Its order is the trunk cut into consecutive steps that
  end at the starts of the size's branches up to its own start, each a
  part of the size's schedule ended by `last_update`, then its decay,
  the size's step ended at the branch's updates; each continues the one
  before it ([training.md](training.md), "Continued steps"). Two
  branches of a size hold the same trunk steps up to the shorter one's
  start, so `wpt train` reuses their kept results and the trunk trains
  once. Every part keeps the size's whole schedule for its warm-up and
  its windows, so a branch reads the size's tokens in the size's order.
  A family with branches writes `mtp_switch` 0.8 in each step that lists
  stages and leaves `first_drop` at its default of 0.8: the size's own
  run then switches the multi-token prediction's weight and drops its
  rate where every branch does, and `branch-1` trains the same weights
  as the size's own run. Two branches of the same updates are refused,
  and so is a branch too short to decay after one update.
- `lr-FACTOR`, at the smallest size only (the fewest parameters without
  the vocabulary tables), one per factor of the family's
  `learning_rate_check`: the size's own run at FACTOR times the learning
  rate the family writes (`steps.1.optimizer.settings.lr`), rounded to 4
  significant digits.

Each run's model folder and record are those of any `wpt train` run.

**The numbers of every size.** `wpt train family NAME` prints, for every
size of the family and training nothing: its width, layers, heads and
experts, its parameters without and with the vocabulary tables and those
one token uses, its vocabulary, the tokens it reads of its corpus, the
updates and batch of each stage of each step that reads text windows,
the tokens its windows train on and their number per parameter (every
parameter counted), the compute per token and in all, and each other
run of the size.

The parameters are counted as `wpt_model` builds the model
(`wpt_orders.counts`; wpt-training's tests check the counts against the
model itself). The parameters one token uses leave out the vocabulary
tables, the `n_experts - top_k` routed experts of each layer that do not
run for the token and Engram's lookup tables (a row is read, not
multiplied), and count the shared experts, Engram's projections and
every other weight. The compute of training on one token in windows of
`l` tokens (`counts.compute_per_token`) is, for each pass of the loop,
`6 N + 12 L d_model l` ([DeepSeek LLM][llm]'s form, with `N` the
parameters one token uses in the layers and `L` the layers) plus
In-Place TTT's update and change, `6 x 2 x d_model x 2 shared_width`
in each layer that has it. With `loop_layers`, the passes after the
first count the looped layers alone, and the layers after the loop run
again for each earlier pass the loss weighs (`loop_losses`). The output
head adds `6 x vocabulary x d_model` each time it reads a token: once
for each pass the loss weighs, and once for the multi-token prediction
block. The attention term is the paper's: the window and sparse layers
attend less than full attention over the window, so it overstates their
attention compute. A size's compute in all adds each stage's tokens at
the compute per token of its window length.

**The families of the repository.** Every family of the repository names
a snapshot of the public depot, uses attention heads 128 wide, of which
the window layers rotate the last 64, and states the same drafter
training values: 256 prompts of at most 64 tokens, answers of at most
256 tokens, 10 epochs, learning rate 6e-4, 10% held out, 512 anchors, 8
answers per update, seed 0; the settings a section leaves out (the
drafter's shape, `weight_decay`, `warm_up`, `clip`, `temperature`,
`top_k`, `ece_bins`) keep their defaults. `pocket` and `coherent` read
their text once through nine stages of windows of 128, 256, 512, 1,024,
2,048, 4,096, 8,192, 16,384 and 32,768 tokens, each an equal share of
the tokens, in updates of 131,072 tokens, with `mtp_switch` 0.8 and a
warm-up of 4%.

- `pocket`: sizes to compare on bits per byte of one held-out portion,
  11 layers laid out `(2, 1, 5)`, heads of 128, 32 experts with top 4,
  no Engram layer, one vocabulary table of 16,384 tokens learned on
  10^10 bytes of its snapshot, a learning rate of 0.004. Its sizes are
  its widths; every size reads the start of one corpus of 1,179,648,000
  tokens, about 36.6 tokens per parameter, every parameter counted. The
  updates are whole groups of 16 (training.md, "Stages"):

  | size | heads | parameters | with the table | one token uses | tokens | updates |
  | --- | --- | --- | --- | --- | --- | --- |
  | `64` | 1 | 1.86M | 2.90M | 0.91M | 106.3M | 816 |
  | `96` | 2 | 4.28M | 5.85M | 2.15M | 214.1M | 1,632 |
  | `128` | 2 | 7.14M | 9.24M | 3.36M | 338.0M | 2,576 |
  | `192` | 3 | 15.86M | 19.00M | 7.34M | 695.7M | 5,312 |
  | `256` | 4 | 28.01M | 32.20M | 12.87M | 1,179.6M | 9,008 |

- `coherent`: one size, `512`, a model for one GPU meant to read coherent
  text: 11 layers laid out `(2, 1, 5)`, 8 heads of 128, 32 experts with
  top 4, Engram memory in layer 2, one vocabulary table of 65,536 tokens,
  one loop, a learning rate of 0.0034; 196.0M parameters with the table,
  162.4M without, 3.72B tokens in 28,400 updates.
- `test`: one size, `128`, a test to see every step work on one machine:
  6.6M parameters without its two vocabulary tables of 32,768 tokens,
  15.0M with them, 11 layers laid out `(2, 1, 5)` with 4 heads of 128,
  8 experts with top 2, its 4.29M tokens read once in 262 updates of 64
  windows of 256 tokens. In windows this short, every sparse layer reads
  every earlier token: the exact layer picks 1,024 of them outside the
  last 128.

## Limits of the evidence

Every result above was measured on models of 3B parameters or more, most at
27B to 2.8T, by the labs that built them; none was measured at the sizes a
single 16 GB GPU trains. A first order sized for such a GPU carries the
papers' ratios untested, and its first runs are the first check of whether
they hold small. HySparse2 in particular was built for long contexts at
larger sizes; WPT trains it at 32K tokens and fewer, at sizes no paper has
measured. No WPT model has yet trained beyond 32K tokens: the stages to
1M, the segments and the slices are measured only in tests and in single
steps on random tokens (training.md, "Long windows"). The drafter's
production values were set for V4-Flash and V4-Pro; WPT scales its Markov
head with the width ("The drafter"), and no drafter has yet been measured
on a real WPT run. The memory of the 32K-token sizes is not measured. The
details
the papers leave out (the hash routing function, Engram's hash constants
and tables, how uneven groups of Attention Residuals split, the histogram behind Quantile
Balancing, the MTP layer, the starting values of the attention sinks)
are WPT's own choices,
marked as such above.

## Sources

- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [DeepSeekMoE: expert specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V4: a million-token context](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4.1-Flash: KV cache compression](https://arxiv.org/abs/2609.19969)
- [HySparse2](https://arxiv.org/abs/2609.26368)
- [HySparse: hybrid sparse attention with oracle token selection](https://arxiv.org/abs/2602.03560)
- [IMU-1: QK-norm, NorMuon, cautious weight decay and a checkpoint average](https://arxiv.org/abs/2602.02522)
- [Gated attention for large language models](https://arxiv.org/abs/2505.06708)
- [MiMo-V2-Flash: a sliding-window MTP block](https://arxiv.org/abs/2601.02780)
- [Engram: conditional memory via lookup](https://arxiv.org/abs/2601.07372)
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [In-Place Test-Time Training](https://arxiv.org/abs/2604.06169)
- [Ouro: scaling latent reasoning via looped language models](https://arxiv.org/abs/2510.25741)
- [Loopback Transformer: towards a unified view of residual looped Transformers](https://arxiv.org/abs/2611.07226)
- [Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923)
- [Moonlight: Muon at scale](https://arxiv.org/abs/2502.16982)
- [NorMuon: neuron-wise normalized Muon](https://arxiv.org/abs/2510.05491)
- [Cautious weight decay](https://arxiv.org/abs/2510.12402)
- [DeepSeek LLM: scaling laws](https://arxiv.org/abs/2401.02954)
- [DSpark: confidence-scheduled speculative decoding](https://arxiv.org/abs/2607.05147)
- [DFlash: block diffusion for flash speculative decoding](https://arxiv.org/abs/2602.06036)

[k3]: https://arxiv.org/abs/2607.24653
[moe]: https://arxiv.org/abs/2401.06066
[v3]: https://arxiv.org/abs/2412.19437
[v4]: https://arxiv.org/abs/2606.19348
[v41]: https://arxiv.org/abs/2609.19969
[hysparse2]: https://arxiv.org/abs/2609.26368
[hysparse]: https://arxiv.org/abs/2602.03560
[imu1]: https://arxiv.org/abs/2602.02522
[gated]: https://arxiv.org/abs/2505.06708
[mimo]: https://arxiv.org/abs/2601.02780
[engram]: https://arxiv.org/abs/2601.07372
[mhc]: https://arxiv.org/abs/2512.24880
[attnres]: https://arxiv.org/abs/2603.15031
[ttt]: https://arxiv.org/abs/2604.06169
[ouro]: https://arxiv.org/abs/2510.25741
[loopback]: https://arxiv.org/abs/2611.07226
[mol]: https://arxiv.org/abs/2611.34923
[moonlight]: https://arxiv.org/abs/2502.16982
[normuon]: https://arxiv.org/abs/2510.05491
[cwd]: https://arxiv.org/abs/2510.12402
[llm]: https://arxiv.org/abs/2401.02954
[dspark]: https://arxiv.org/abs/2607.05147
[dflash]: https://arxiv.org/abs/2602.06036
