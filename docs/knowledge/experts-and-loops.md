# At small sizes, are parameters better spent on experts or compute on loops?

This file asks one question: for a small model, how far the number of
routed experts can go down, and whether passing the text through the same
layers several times gives back what the experts gave. Its marks are
defined in [README.md](README.md).

## Answer

The question is open at small sizes: published sources measure each half
of it, and none crosses the number of experts with the number of loops by
model size. What they say goes half with the idea of fewer experts and more
loops at small sizes, and half against it.

- At an equal number of parameters, a model that loops beats one that does
  not, and the gain is larger the smaller the model. **published, agreed**
- At equal training compute, a model that loops with ordinary residual
  addition, each pass adding its change to what the passes before it
  left, loses to a deeper one, and loses more at small budgets. A loop is
  worth about half a layer of its own. **published, agreed**
- With another residual rule, one source finds the opposite. When the
  changes of all passes so far are multiplied by 1/4 at the end of each
  pass, toward the loop's input, a looped model has a validation loss at or
  under a standard model of the same executed depth on the same tokens, so
  at equal compute, from 191M to 3B; a small learned reweighting of the
  changes adds about 0.003 to the decay. The caveats are the source's own: at
  191M the gain is within the spread between seeds (-0.0047, +0.0011 and
  -0.0064 over three seeds); at 374M it is 0.023 against the standard
  model at its best depth, over two seeds; at 778M it is 0.002 to 0.004
  against that best depth, and its larger margins at 374M and 778M compare
  with standard models deeper than their best depth. Every looped model has
  unshared layers before and after the loop, and its own learning rate, up
  to 3 times the standard model's. **published, one source**
- A loop can keep, while writing, the keys and values of one pass instead
  of one per pass. Each pass starts again from the embedding and reads the
  earlier tokens only through keys and values made from the previous
  pass's output (loop KV). Writing one pass per token then gives at token
  t exactly what an unrolled loop of at least t + 2 passes gives: **proved**
  by its source, for that architecture. The proof needs no assumption that
  the passes' keys and values are alike. It does not show equality with
  the 2 passes the model was trained with: that gap is measured, 0.0002 to
  0.0014 nats at 113M. **published, one source**
- At 113M with 2 passes, loop KV ties a plain loop, 3.291 to 3.308 against
  3.299, when each layer also reads its own key and value of the current
  token; without them it does no better than the model without loop,
  3.310 and 3.322 against 3.318. **published, one source**, one run each
- Run with more passes than it was trained with, a looped model does not
  lower its loss. With ordinary addition the loss rises by 0.25 to 0.45
  from 191M to 1.4B; with the decay above by 0.03 to 0.07.
  **published, agreed**
- Loops add reasoning and not knowledge: what a model can hold stays tied to
  its parameters. **published, agreed**
- A small mixture of experts beats a dense model of equal compute and at
  best ties one of equal parameters, and it needs more training before it
  pays. **published, agreed**
- At a width of 256 with one shared vocabulary table, adding 2 loops and a
  lookup table of 13.0M parameters lowers the validation loss by 0.14,
  from 2.79 to 2.65, on the same 1.18B tokens, for 3 times the compute per
  token. The model copies from its context less well, 84% of the tokens of
  a repeated sequence against 95%. The measure does not separate the loops
  from the table. The two runs share their seed, so their validation
  windows; the gap in copying, 11 points, is within the 13 points two
  seeds of one order show
  ([run-to-run-spread.md](run-to-run-spread.md)). **measured here**, one run each
- A learned share between 0 and 1, by token, of what each extra pass
  changes does not lower the loss: 2.67 against 2.65 without it, on the
  same windows. It keeps 98% of the second pass and 42% of the third.
  **measured here**, one run each
- A model that loops, trained with every pass reading its own keys and
  values of the earlier tokens and with the loss on the last pass alone,
  cannot read a text through the last pass's keys and values alone: read
  token by token that way from its first token, a text costs 8.32 nats per
  token against 2.50 at a width of 512 with 1 loop, and 5.34 against 2.71
  at a width of 256 with 2 loops, and what the model writes is no longer
  words. **measured here**, one text of 39 tokens on each of two models
- In a model of 4 passes trained with a loss on every pass, the tokens it
  writes can keep the last pass's keys and values alone, every pass
  reading them, at almost no loss: 78.85 against 78.92 on GSM8K, 80.40
  against 82.40 on MATH-500. The prompt cannot: read that way it loses
  more than 10 points on GSM8K, so every pass keeps its own keys and values
  of the prompt. **published, one source**
- So a model that loops by passing each pass's output to the next keeps
  each pass's keys and values of its prompt, which multiplies what it keeps
  of the prompt by the number of passes; loop KV, above, does not.
  For the tokens it writes, the last pass's may be enough; the measure
  above read the prompt through them too, so it does not test this.
  **published, one source**, and the measure above agrees on the prompt
- More experts and more loops add up more than they replace each other.
  **published, one source**
- So the idea holds where the limit is the size of the model, its memory or
  its file. Where the limit is training time it fails with ordinary
  addition between passes, and holds only if the decay between passes
  gains under 191M, which nothing measures. **hypothesis**
- The two numbers do not trade the same thing. The number of experts
  changes the parameters and not the work per token, since 4 run whatever
  their number. Loops change the work per token, `loops + 1` times one
  pass, and not the parameters. Fewer experts and more loops is a smaller
  model that computes more. **proved**, by counting the parameters and the
  passes of WPT's layers (docs/model.md, "Model series").
- The routed experts are the largest part of a small model's body, so the
  number of experts is the strongest lever on its size after the
  vocabulary. **proved**, by counting, for the shapes of the table below

## Evidence

### Counted from the model

Parameters of the body, without the vocabulary's two tables, for 11 layers
and 4 experts per token, counted from the model's code:

| Width | 32 experts | 16 experts | 8 experts |
| --- | --- | --- | --- |
| 512 | 124.0M | 89.4M | 72.0M |
| 256 | 41.0M | 32.3M | 27.9M |

The 32 routed experts are 56% of the body at a width of 512 and 43% at 256.
**proved**, by counting

### Published

All figures in this section are **unchecked**: they were read through an
automatic reader of the papers' pages. Each source exists on arXiv under
its id, with the subject given here. Three sets of figures were compared
with the papers' own text: those of *Ouro* on what the passes keep while
writing, under "Measured with WPT", and all those of *Mixture-of-Loops
Transformer* and *Loopback Transformer*.

#### Mixtures of experts in small models

- arXiv 2402.13089 (2024). A base of 124M parameters, about 6B tokens. 4
  experts with 2 per token, 295M in all: perplexity 9.91 against 10.07 for
  a dense model of 295M. With 2, 4 and 6 experts: 10.76, 9.91, 9.47.
  Leaving out the balancing loss costs nothing.
- *CoSMoEs* (2025, arXiv 2503.00245). 8 experts, 2 per token. 377M in all,
  188M active: 43.22 against 39.90 for a dense model of 189M and 42.91 for
  one of 380M. At the size of a phone it loses to the dense model of equal
  parameters.
- arXiv 2604.14419 (2026). 76M parameters, 1.64B tokens. The mixture
  reaches 33.93 against 34.49 for the dense model of equal compute and
  29.36 for the one of equal parameters; it passes the first only after
  about 0.33B tokens.
- *Unified Scaling Laws for Routed Language Models* (2022, arXiv
  2202.01169). 15M to 1.3B, 2 to 512 experts, 130B tokens. Each added
  expert gives less than the one before.
- arXiv 2402.07871 (2024). 129M to 3.7B, 16B to 130B tokens. A mixture
  needs longer training to compete, and finer experts fail at a width of
  256.
- *Mixture of Parrots* (2024, arXiv 2410.19034). 8 to 64 experts, about 65B
  tokens. More experts raise what the model memorises; reasoning follows
  the active parameters.
- arXiv 2506.12119 (2025). 2B to 7B only. The best active share is about
  20% at every size, and a mixture needs about 4.6 times more distinct data
  at equal compute.

#### Loops and layers that share weights

- arXiv 2604.21106 (2026). 35M to 1.14B, 1 to 8 loops, 116 runs. One loop
  is worth 0.46 of a layer of its own. At equal training compute the looped
  model loses 0.03 to 0.06 nats with 2 loops, 0.05 to 0.08 with 4, 0.09 to
  0.12 with 8, and more at lower budgets.
- *Parcae* (2026, arXiv 2604.12946). 8 loops against a model of equal
  parameters, perplexity 19.06 against 21.48 at 140M, 14.49 against 15.79
  at 370M, 12.49 against 13.08 at 770M, 11.42 against 11.95 at 1.3B.
- arXiv 2502.17416 (2025). About 1B, 250B tokens. 12 layers: 8.16; the same
  looped twice: 7.90; 24 layers of their own: 7.40. The looped model wins
  on arithmetic.
- *Ouro* (2025, arXiv 2510.25741). 170M to 1.3B. The gain is steep up to 3
  loops and flat at 4; more loops than in training lower the quality. At
  equal compute the plain model wins at every size, by 0.039 falling to
  0.025 as size grows. About 2 bits of knowledge per parameter with or
  without loops. While writing, its passes can share the last pass's keys
  and values of the written tokens, not of the prompt (under "Measured
  with WPT", below).
- *MobileLLM* (2024, arXiv 2402.14905). Each block run twice: +0.7 point at
  125M and +0.8 at 350M, for about 2.6% more time on a phone. 3 or 4 times
  show no clear trend.
- *ALBERT* (2019, arXiv 1909.11942). Every layer shared: 79.8 at 31M
  against 82.3 at 108M without sharing.
- *Mixture-of-Recursions* (2025, arXiv 2507.10524). A shared block looped,
  a router giving each token 1 to 3 passes; 135M to 1.7B, 20B tokens. With
  2 passes and 167M parameters outside the tables: loss 2.7511, against
  2.7824 for a plain model of 315M and 2.8079 for a plain loop of 167M.
  With 3 passes and the other way of routing: 2.9163. The figures at 135M
  were not read.
- arXiv 2611.06206 (2026). 97M, 12 layers looped 4 times, 100M words, 3
  seeds. A grammar test at 76.16 against 75.21 for the same model without
  loop, a gap the source does not find significant; held-out loss 5.83
  against 5.84.
- arXiv 2602.11451 (2026). 1B to 3B, 25B tokens. A looped model at a
  perplexity of 10.28 against 9.49 for a dense one.
- arXiv 2610.00673 (2026). 1.4B, one seed. A looped model trains with the
  normalisation before each layer and a last one at the end; with it after
  the layer the loss stays near 7.4. A depth of 8 passes gains nothing
  over 4. A small matrix of its own for each pass moves the result by 2
  points at most.
- *Mixture-of-Loops Transformer* (2026, arXiv 2611.34923), its Tables 2,
  3, 6 and 11 and its Appendix B. Its small models have about 113M
  parameters: width 768, 12 layers, the GPT-2 tokenizer, windows of 1,024
  tokens, 3B tokens of FineWeb, the Muon optimizer, and one seed shared by
  every run. All of them have 2 passes, and every pass has a loss of equal
  weight, the plain loop included. Validation loss of the last pass:
  - No loop 3.318; plain loop 3.299: 0.019 better at an equal number of
    parameters, for twice the compute.
  - Loop KV, with each layer also reading at the current token its own key
    and value: 3.296 to 3.308. Adding those of the earlier layers of the
    current pass: 3.291 to 3.302. Adding those of the previous pass at the
    current token: 3.286 to 3.297, but then a token waits for its previous
    pass and cannot be written in one pass. Without the current token's
    keys and values: 3.310 and 3.322. Loop KV adds 1.8M to 3.5M parameters
    for the projections of the previous pass's output.
  - Looping only the first 6 layers, the other 6 running once on each
    pass's output and reading the earlier tokens only through keys and
    values made from it: 3.279 to 3.300 in the forms that can be written
    one pass per token, 3.279 to 3.282 when each of the 6 layers has its
    own projections of that output, 3.273 at best.
  - The loss weights of the two passes, on the form above (Table 3): 1 to
    1 gives 3.306 at the first pass and 3.279 at the last; 1 to 2 gives
    3.315 and 3.276; 1 to 4 gives 3.322 and 3.272. A weight of 0 on the
    first pass is not tried; the single-pass writing below assumes every
    pass has a loss.
  - Written one pass per token (Table 6), two other models of this size,
    trained without gradient between the passes, on 8 windows of a prompt
    of 128 tokens then 896 scored tokens: from an empty cache +0.0002
    against the 2 trained passes; with the prompt read with 1 pass, +0.0014
    and -0.0002 for the two models; with 2 passes, +0.0002 and -0.0005.
  - What Appendix B proves, for loop KV: written one pass per token from an
    empty cache, token t gets what an unrolled loop of at least t + 2
    passes gives; with the prompt read in parallel, the written tokens get
    what an unrolled loop gives that holds the prompt fixed. The keys and
    values kept while writing are one pass's, as many as a model without
    loop of the same layers keeps, against one set per pass for a loop
    that passes its output on.
  - At 700M (38 layers against 19 looped layers and 19 that read their
    output, 400B tokens, AdamW, one run each), the looped model reaches the
    standard model's final validation loss after 232.8B tokens, 1.7 times
    fewer, for 2.01 times the training compute per token: 1.17 times the
    standard model's total compute. At that total compute, 198.9B tokens,
    it has not reached it yet; the paper says an exact comparison at equal
    compute would need its own run.
  - Not measured: anything under 113M, more than 2 passes, a loss on the
    last pass alone, a second seed, and a deeper model of equal compute.
- *Loopback Transformer* (2026, arXiv 2611.07226), its Tables 2, 4, 5, 7,
  10 and 12. 191M to 3B, on ClimbMix, 20 tokens per parameter of the
  standard model, windows of 4,096 tokens, AdamW with no weight decay.
  Every looped model has 1 to 6 unshared layers before the loop and 1 to 6
  after it; at 191M, 2 layers, 4 layers looped 4 times, then 2. The decay:
  at the end of each pass, the loop's next input is its input plus a
  quarter of what the passes have changed so far, including after the last
  pass, before the unshared last layers. Only the factors 1/4 and 1 (no
  decay) are tried. A learned reader, one vector per layer application,
  then reweights the stored changes without changing their sum (11,520
  parameters at 191M).
  - Validation loss (Table 4a). At 191M: plain loop 2.6802, reader alone
    2.6621, decay alone 2.6220, both 2.6187; the standard model 2.6234. At
    374M with 24 layers: 2.4415, 2.4308, 2.3999, 2.3971; the standard
    model 2.4198. The decay gives almost all of the gain, and the reader
    adds about 0.003.
  - Against the standard model of equal executed depth (Tables 2 and 9),
    one run each: 2.6187 against 2.6234 at 191M with 20 layers, 2.4008
    against 2.4395 at 374M with 32 layers, 2.2404 against 2.2644 at 778M
    with 48 layers, 2.1073 against 2.1283 at 1.4B, 1.9849 against 2.0005
    at 3B. The looped models store 40% to 75% of the standard model's
    layer parameters.
  - Seeds (Table 10 and its text): at 191M the gap to the standard model
    is -0.0047, +0.0011 and -0.0064 over three seeds; at 374M with 24
    layers, -0.0227 and -0.0261 over two.
  - Depth at a fixed number of parameters (Table 12), 16 to 64 layers,
    always 2 passes: the standard model is best at 24 layers at 374M and
    at 778M. Against it there, the gap is -0.0227 at 374M; at 778M,
    -0.0016 at 24 layers and -0.0037 for the looped model's own best depth,
    48 layers. At 778M the standard model trained at a peak rate of 1e-3
    and the looped one at 1.5e-3; the standard model at 1.5e-3 does 0.0020
    better at 24 layers, as much as the gap there.
  - The plain loop of the same tables loses to the standard model of equal
    depth at 191M by 0.057, at 374M with 24 layers by 0.022, and at 778M at
    every depth; it wins at 374M only at 32 to 64 layers, where the
    standard model is past its best depth. At an equal number of stored
    parameters, a standard model of 8 layers reaches 2.7376 against 2.6802
    for the plain loop.
  - More passes than trained, without retraining: at 191M with 8 passes
    instead of 4 the loss rises by 0.060 with the decay and 0.454 for the
    plain loop; with 4 instead of 2, by 0.074 and 0.270 at 374M, 0.028 and
    0.248 at 1.4B. The paper says extra passes never lowered the loss.
  - Learning rates: the looped models use 3e-3 at 191M against 1e-3 for
    the standard model, the best of four rates for it; the same 2e-3 at
    374M; 1.5e-3 against 1e-3 at 778M and 1.4B; the same 1e-3 at 3B.
  - Not measured: anything under 191M, a loop over all layers with no
    unshared layers, another factor of decay, and text other than
    ClimbMix. The configurations disagree between tables: Table 5 gives
    374M and 778M as 24 layers, [2, 10 looped twice, 2] and [2, 8 looped
    twice, 6], while Table 2 reports 32 and 48 layers at those sizes, and
    Table 12's 24-layer model at 778M is [2, 10 looped twice, 2], not the
    [2, 8 looped twice, 6] of Tables 5 and 15.
- Models of a few million parameters that loop over their own state solve
  puzzles (arXiv 2506.21734, arXiv 2510.04871) and are not models of
  language. The one model of language found built that way (arXiv
  2605.28919, 83M, 10B tokens) scores under dense models of 90M and 120M
  from the same source.

#### Both together

- *MoEUT* (2024, arXiv 2405.16039). Shared layers and 155 to 565 small
  experts, 44M to 1,040M. Against a dense model of equal parameters: 18.30
  against 18.97 at 44M, 13.24 against 13.40 at 244M. Training is about 1.7
  times slower.
- arXiv 2609.18176 (2026). 114M to 1.15B, 6.55B tokens. Here the model of
  the source above loses to a dense one of equal compute at the small size,
  16.94 against 16.30, and a plain mixture wins, 15.98.
- arXiv 2605.09165 (2026). 16M to 305M active, 10B tokens, 8 experts with 2
  per token. A looped dense model scales worse than a plain one; a looped
  mixture scales like it. Going from 8 to 16 experts helps more with 4
  loops, 3.5723 to 3.5537, than with 2, 3.5698 to 3.5664.
- arXiv 2610.01153 (2026). A looped mixture with no dense model beside it:
  at 100M and 5B tokens, perplexity 26.26 without loop and 19.26 with 12;
  at 350M and 10B tokens, 20.07 and 14.80 with 9.
- arXiv 2606.16825 (2026). Sharing the experts over 4 layers costs 0.05 to
  0.07 of loss at about 0.5B and nothing at 7B.

#### What the sources dispute

- Where loops stop paying: 2 passes, 4, or 9 to 12, by source.
- Whether a looped model loses to a deeper one at equal compute. It does
  with ordinary addition between passes, in arXiv 2604.21106, *Ouro*, and
  the plain loop of *Loopback Transformer*. With the passes' changes
  decayed toward the loop's input, *Loopback Transformer* finds it does
  not from 191M to 3B, with unshared layers around the loop, by margins
  that are within the spread between seeds at 191M.
- Whether shared layers with many small experts beat a dense model.
- Whether the best share of active experts depends on the model's size.

### Measured with WPT

Two models of width 256, 11 layers, 32 routed experts, a vocabulary of
65,536 and one shared vocabulary table, trained once each on the same
1.18B tokens with the same stages (windows of 2,048, 8,192 then 32,768
tokens, 8,990 updates of 131,072 tokens), the same seed and the same
optimizer, at commit b7dca7b2f7. One has no loop and no Engram memory
(44.8M parameters, RTX 5080 Laptop GPU). The other runs its layers 3 times
for each token (2 loops) and has an Engram memory of 13.0M parameters at
layer 2 (57.7M parameters, RTX 5090).

| | No loop, no table | 2 loops and the table |
| --- | --- | --- |
| Validation loss at update 2,400 | 3.51 | 3.41 |
| Validation loss at update 6,000 | 3.25 | 3.12 |
| Validation loss at update 8,000 | 2.92 | 2.78 |
| Validation loss at update 8,990 | 2.79 | 2.65 |
| Repeated sequence, tokens written | 95% | 84% |
| Repeated sequence, written whole | 89% | 72% |
| Stated word, likeliest of four, English and French | 100%, 100% | 98%, 97% |
| Stated word, written, English and French | 14%, 30% | 12%, 24% |
| Stated number, likeliest of four, English and French | 100%, 100% | 100%, 100% |
| Stated number, written, English and French | 84%, 90% | 32%, 36% |

The two models differ by the loops and by the table at once, and by 12.9M
parameters. With two vocabulary tables and no loop, the same lookup table
lowered the loss by 0.05 ([lookup-memory.md](lookup-memory.md)); if it
does the same here, the loops give about 0.09. The copying figures are
those of the `copying` test of `wpt eval`, on 200 prompts each; the stated
values were measured on the processor, at commit 321bb02efa.

A third model is the second with a gate on each extra pass: for each
token, a share between 0 and 1 computed from the token's state before the
pass (one weight vector of the model's width and one bias for each pass,
514 parameters), by which what the pass changes is multiplied. The share
starts at 0.98. Trained once the same way (commit a067f3ba0d, RTX 5090).

| | 2 loops and the table | The same with the gate |
| --- | --- | --- |
| Validation loss at update 8,000 | 2.78 | 2.79 |
| Validation loss at update 8,990 | 2.654 | 2.671 |
| Mean share kept of the second pass, at the end | | 0.98 |
| Mean share kept of the third pass, at the end | | 0.42 |
| Repeated sequence, tokens written | 84% | 95% |
| Repeated sequence, written whole | 72% | 90% |
| Stated word, likeliest of four, English and French | 98%, 97% | 78%, 72%, by tokens |
| Stated word, written, English and French | 12%, 24% | 12%, 6%, by tokens |
| Stated number, likeliest of four, English and French | 100%, 100% | 91%, 85%, by tokens |
| Stated number, written, English and French | 32%, 36% | 0%, 1%, by tokens |

The share kept of the third pass fell to 0.02 in the first 200 updates,
rose again, and moved between 0.3 and 0.5 from update 5,000 to the end.
Both runs measure their loss on the same windows. The copying figures of
the repeated sequence differ by less than two runs of one recipe do
([run-to-run-spread.md](run-to-run-spread.md)). The model with the gate
cannot be loaded by `wpt eval`, whose models have no gate, so its stated
values are counted by tokens: each of the four scored alone, without the
sentence's period that the tokenizer may join to its last letters or
digits, and written only when the tokens equal those of the value alone,
which refuses most values written right. The stated values of the model
without the gate were measured on the processor, at commit 321bb02efa.

A model that loops writes token by token through what it keeps of the
earlier tokens. Two trained models, the second of the first table (width
256, 2 loops) and one of width 512 with 1 loop (196.0M parameters, 3.72B
tokens), read one English text of 39 tokens three ways (processor, commit
bce0c7ab64 and the one before it). The loss is the mean over the text's
tokens.

| | Width 256, 2 loops | Width 512, 1 loop |
| --- | --- | --- |
| The whole text at once | 2.71 | 2.50 |
| Token by token, each pass reading its own values of the earlier tokens | 2.71 | 2.50 |
| Token by token, every pass reading the last pass's values | 5.34 | 8.32 |
| In two halves, the second reading the last pass's values of the first | 3.12 | 3.23 |

Written through the last pass's values, both models give a few words then
single letters and digits.

*Ouro* (2025, arXiv 2510.25741), section 5.4.2 and its Table 14, read in
the paper's own text, tests what the passes of its model of 4 passes keep
of the earlier tokens:

| What each pass reads of the earlier tokens, while writing | GSM8K | MATH-500 | Kept |
| --- | --- | --- | --- |
| Its own pass's keys and values | 78.92 | 82.40 | 4 times one pass |
| The last pass's | 78.85 | 80.40 | one pass |
| The average of the four passes' | 78.73 | 78.52 | one pass |
| The first pass's | 18.73 | 8.43 | one pass |

- The sharing is tested only on the tokens the model writes. For the
  prompt, the paper says every pass needs its own keys and values, and
  that reusing them while reading the prompt loses more than 10 points on
  GSM8K, with no table.
- The section names "our 4-step model" and no size. Its first row equals
  the scores of Ouro-1.4B with 4 passes in the paper's table of base
  models, so the model is most likely Ouro-1.4B, trained on 7.7T tokens.
- The paper presents the sharing as a choice at inference: it does not say
  the model was trained with it. Its definition of the model, the stack of
  layers applied to the whole text once per pass, has every pass read its
  own pass's states of the earlier tokens in training.
- Each pass of Ouro's training has its own loss: the output head reads
  every pass, and the loss weighs each pass's loss by a learned
  probability of stopping there, kept spread by an entropy term. WPT's loss
  reads the last pass only (docs/model.md, "Loops").

So the paper's setting has what the measure above did not: a prompt read
with each pass's own values, a loss on every pass, 4 passes, 1.4B
parameters and 7.7T tokens. The paper does not say which of these makes
the sharing work. That the loss on every pass does, by keeping each pass's
states close to what the output head reads, is a **hypothesis**. The row
of the measure above that reads the first half through the last pass's
values, 3.12 and 3.23 against 2.71 and 2.50, is the closest to the
paper's reading of the prompt that way, and both lose.

Loop KV (*Mixture-of-Loops Transformer*, above) keeps one pass's keys and
values by another route. In Ouro and in the models measured above, each
pass takes the previous pass's output at the same token, and every pass
has its own keys and values of the earlier tokens; reading the last
pass's instead is a substitution made at writing time, which works only
as far as the passes' keys and values happen to be alike. In loop KV,
each pass starts again from the embedding, and its layers read the
earlier tokens only through keys and values made from the previous pass's
output, in training as in writing. A token then never waits for its own
previous pass, and writing one pass per token is the architecture's own
computation, not a substitution: the proof that it equals an unrolled loop
of at least t + 2 passes needs no assumption about the passes' keys and
values. What remains measured is how far that unrolled loop is from the 2
passes trained, 0.0002 to 0.0014 nats at 113M.

## Open

- **Sharing keys and values between passes on the tokens written.** A
  measure reads the prompt with each pass's own keys and values and the
  written tokens through the last pass's alone, on a model trained with the
  loss on the last pass only and on the same model trained with a loss on
  every pass, and compares each with every pass keeping its own. It would
  say whether the published result holds under 1.4B and with fewer than 4
  passes, and whether the loss on every pass is what makes it work.
- **One loop against two.** The gate keeps less than half of the third
  pass. A model with 1 loop, at two thirds of the compute of 2, measured
  against both.
- **Loops alone and the table alone on the model with one table**: the
  measure above gives their sum.
- **The same loss by a deeper model of equal compute.** A model that runs
  its layers 3 times costs the compute of one 3 times as deep. With
  ordinary addition between passes the published sources find the deeper
  one better; with the decay below, one source finds the loop at or ahead
  of it from 191M. The measure trains the deeper model beside the loops of
  the decay measure, on the same tokens, each at its own best learning
  rate.
- **Decay between passes, with no parameter, on a model that loops over
  all its layers.** *Loopback Transformer* measures the decay only with
  unshared layers before and after the loop; it never tests a model whose
  every layer loops and whose output head reads the last pass. A measure
  trains at 45M (width 256, 11 layers, 2 loops), on the same tokens: a
  plain loop; the same with each pass's input set to the embedding plus a
  quarter of what the passes have changed so far, with the last pass
  decayed; the same with the last pass not decayed; and the model without
  loop. Whether the last pass is decayed is a choice to measure, not a
  detail: decayed, the output head reads a quarter of the last pass's
  change. Two seeds of each. It closes when the gap to the plain loop is
  several times the spread between the two seeds of each model; the
  published gain, 0.04 to 0.06 at 191M and 374M, would be. If it gains,
  the same at about 120M.
- **A loss on every pass against the last pass alone.** At 45M with 2
  loops, on the same tokens: the loss on the last pass alone, the loss on
  every pass with equal weights, and with the last pass weighted 4 times
  the first. It compares the last pass's validation loss on the same
  windows, and gives the first passes' loss beside it. *Mixture-of-Loops
  Transformer* finds 1 to 4 better than 1 to 1 by 0.007 at 113M, one run
  each, and never tries the last pass alone. It closes when the gaps are
  read against two seeds of each.
- **Loop KV, written one pass per token.** At 45M, on the same tokens:
  a loop KV model with 2 loops and a loss on every pass, each layer
  reading its own key and value of the current token, against a plain
  loop with the same passes and the model without loop. It measures the
  validation loss with the trained passes, then the loss of texts written
  one pass per token, from an empty cache and after a prompt read with
  every pass, against the trained passes. It closes the question for that
  size if loop KV is within the spread between seeds of the plain loop and
  the writing one pass per token within 0.01 of the trained passes: a
  looped model would then keep one pass's keys and values while writing.
  The published measure stops at 113M and 2 passes.
- **Looping the first half of the layers only.** At 45M, on the same
  tokens: the first half looped, the second half run once on each pass's
  output, against all layers looped with the same passes and the model
  without loop, each with its compute per token beside its loss (1.5
  against 2 times one pass, with 2 passes). *Mixture-of-Loops Transformer*
  finds the half loop better by 0.01 to 0.03 at 113M, one run each, with a
  second half that reads the earlier tokens only through the looped half's
  output. It closes when the gap is read against two seeds of each.
- **A learning rate for looped models.** *Loopback Transformer* gives its
  looped models their own peak rate, 3 times the standard model's at
  191M, 1.5 times at 778M and 1.4B, and the same at 374M and 3B. A measure
  trains at 45M a model with 2 loops at 1, 1.5, 2 and 3 times the best
  rate of the model without loop, on the same tokens. It closes when the
  best rate is found and its loss is read against the loss at the
  no-loop model's rate and the spread between seeds; every comparison of a
  loop with another model then gives each its own best rate.
- **The validation loss at more passes than trained.** No training: the
  models of the measures above, trained with 2 loops, evaluated with 3
  and 4 loops on the same windows. The published loss rises with every
  extra pass, by 0.25 to 0.45 for a plain loop and 0.03 to 0.07 with the
  decay, from 191M to 1.4B. It closes when measured at 45M: whether the
  loss rises, by how much, and whether the decay or a loss on every pass
  holds it.

- **The number of experts a small model needs.** A measure trains one shape
  at a width of 256 or 512 with 32, 16 and 8 experts, 4 per token, on the
  same tokens, and compares the validation loss and the copying test of
  `wpt eval` (docs/evaluation.md). The time per update should not move.
- **Whether loops give back what the experts gave.** A measure compares, at
  an equal number of parameters, a model with 32 experts and no loop with a
  wider model of 8 experts and 1 to 3 loops, and gives each one's time per
  token beside its loss: a loop is paid in time at training and at
  generation.
- Whether the answer changes with size: experts may pay only once a model
  has enough tokens per expert to specialise them.
- Whether fewer than 4 experts per token is enough at small sizes, which
  lowers the work per token as well.
- **The crossing itself.** No source trains a grid of numbers of experts by
  numbers of loops at several sizes. A grid of 3 numbers of experts by 3
  numbers of loops at 2 widths would be the first.
- The tokens an expert needs before it pays, under 5B training tokens: the
  one source there finds 0.33B at 76M parameters.
- A looped mixture against a dense model at equal parameters under 100M.
- **A number of passes chosen for each token.** The one source that routes
  tokens through a loop finds the routing, not the loop, gives the gain. A
  soft share of each pass, measured above, gives none and saves no
  compute. A measure trains, at an equal number of parameters and equal
  compute, a plain model, a plain loop and a loop where a token that
  leaves does not run the later passes.
- The figures of the published sources above, to be compared with their
  texts, except those of *Ouro* on what the passes keep while writing and
  those of *Mixture-of-Loops Transformer* and *Loopback Transformer*.
