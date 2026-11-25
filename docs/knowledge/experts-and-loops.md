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
  seeds of one recipe show
  ([run-to-run-spread.md](run-to-run-spread.md)). **measured here**, one run each
- A learned share between 0 and 1, by token, of what each extra pass
  changes does not lower the loss: 2.67 against 2.65 without it, on the
  same windows. It keeps 98% of the second pass and 42% of the third.
  **measured here**, one run each
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
its id, with the subject given here.

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
  without loops.
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
| Stated word, likeliest of four, English and French | 77%, 80% | 76%, 71% |
| Stated number, likeliest of four, English and French | 92%, 86% | 86%, 76% |

The two models differ by the loops and by the table at once, and by 12.9M
parameters. With two vocabulary tables and no loop, the same lookup table
lowered the loss by 0.05 ([lookup-memory.md](lookup-memory.md)); if it
does the same here, the loops give about 0.09. The copying figures are
those of the `copying` test of `wpt eval`, on 200 prompts each.

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
| Stated word, likeliest of four, English and French | 76%, 71% | 78%, 72% |
| Stated number, likeliest of four, English and French | 86%, 76% | 91%, 85% |

The share kept of the third pass fell to 0.02 in the first 200 updates,
rose again, and moved between 0.3 and 0.5 from update 5,000 to the end.
Both runs measure their loss on the same windows. The copying figures
differ by less than two runs of one recipe do
([run-to-run-spread.md](run-to-run-spread.md)).

## Open

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
  texts.
