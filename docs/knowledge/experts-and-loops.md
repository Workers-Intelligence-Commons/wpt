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
- At equal training compute, a model that loops loses to a deeper one, and
  loses more at small budgets. A loop is worth about half a layer of its
  own. **published, agreed**
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
- So a model that loops keeps each pass's keys and values of its prompt,
  which multiplies what it keeps of the prompt by the number of passes.
  For the tokens it writes, the last pass's may be enough; the measure
  above read the prompt through them too, so it does not test this.
  **published, one source**, and the measure above agrees on the prompt
- More experts and more loops add up more than they replace each other.
  **published, one source**
- So the idea holds where the limit is the size of the model, its memory or
  its file, and fails where the limit is training time. **hypothesis**
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
its id, with the subject given here. The figures of *Ouro* on what the
passes keep while writing, under "Measured with WPT", were compared with
the paper's own text.

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
  its layers 3 times costs the compute of one 3 times as deep; the
  published sources find the deeper one better.

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
  texts, except those of *Ouro* on what the passes keep while writing.
