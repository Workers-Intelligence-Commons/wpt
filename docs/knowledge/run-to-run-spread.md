# How far do two runs of one recipe differ?

This file asks one question: how much the result of training changes when
the same model is trained again with another random seed, what causes it,
and which gap between two models one run of each can show. Its marks are
defined in [README.md](README.md).

## Answer

- Two runs of one recipe never give the same model: training amplifies
  any difference, down to one bit of one weight, and no single source of
  randomness is the cause. **published, agreed**, **unchecked**
- The validation loss of a language model of 80M to 300M parameters
  differs between seeds by about 0.01 nats (one standard deviation).
  **published, agreed**, two sources, **unchecked**
- Changing only the initial weights, or only the order of the text, gives
  nearly the whole spread each. **published, one source**, **unchecked**
- Tests that count right answers differ far more between seeds than the
  loss does, by 2 points and more, and a continuous measure such as the
  loss or bits per byte separates two models better. **published,
  agreed**, **unchecked**
- Before the end of training the spread is far larger: two runs of one
  recipe of 45M parameters differ by 0.06 nats of validation loss at the
  same update, one ahead at two thirds of the run and behind near its end.
  **measured here**, two runs
- So a gap between two models is read at the end of training only, and one
  run of each shows it when it is several times the spread there: about
  0.02 nats if the published spread holds at this size, which is not
  measured. A gap of 0.05 from one run of each is undecided until it is.
  **hypothesis**
- The rank of two models cannot be read early in training: predicting the
  final result needs most of the run. **published, one source**,
  **unchecked**; and **measured here**, one run each
  ([choices-by-size.md](choices-by-size.md))
- Giving two different models the same seed removes little of the spread,
  and training made exactly repeatable costs 3 to 8 times the time.
  **published, one source** each, **unchecked**
- Nothing published measures initial weights without randomness, or an
  average of the last weights, as a way to shrink the spread between seeds
  of a language model.

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

### The size of the spread

- *Pre-training under infinite compute* (2025, arXiv 2509.14786). 300M
  parameters, 200M tokens, 5 runs for each setting. Standard deviation of
  the loss: 0.0082 nats with the initial weights and the order of the text
  both changed, 0.0076 with the weights alone, 0.0072 with the order
  alone.
- arXiv 2609.11917 (2026). 80M, 5 seeds, each token read once: 0.00 to
  0.02 nats, 0.01 on average. With the text read 32 times: 0.02 to 0.10.
- arXiv 2410.11840 (2024). The seed can move the loss by up to 4%.
- *PolyPythias* (2025, arXiv 2503.09543). 14M to 410M, about 10 seeds
  each. At 410M, 2 seeds of 10 stand apart with jumps of the loss.
  Predicting the final result from part of a run needed about 120,000 of
  143,000 updates.

### Its causes

- arXiv 2103.04514 (2021) and arXiv 2304.01910 (2023). A change of one bit
  in the initial weights gives as different a model as another seed; every
  source of randomness acts alike.
- arXiv 2001.11396 (2020). On image models, the randomness of the GPU's
  own operations gave 74% to 87% of the spread.
- arXiv 2106.11872 (2021). Training made exactly repeatable on a GPU costs
  196% to 746% more time.

### What separates two models with few runs

- *Signal and Noise* (2025, arXiv 2508.13144). The movement of a measure
  from one saved state to the next within a run follows the spread between
  seeds (R² 0.82 for the initial weights, 0.86 for the order of the text).
  Deciding between two models on bits per byte is right 83.7% of the time
  against 77.0% on a count of right answers; on the average of the last
  states, 71.3% against 68.9%.
- arXiv 2406.10229 (2024). At 7B, 10 seeds: counts of right answers
  differ by 0.2 to 2 points.
- arXiv 2504.11393 (2025). The spread between runs reaches about 2 points
  at 1B; the rank of two recipes at 150M predicts the better one at 1B in
  about 80% of the comparisons.
- arXiv 2103.03098 (2021). The spread is estimated by changing every
  source of randomness together, and an estimate from few runs is poor;
  pairing two models on one seed helps little when the models differ.

### Measured with WPT

One recipe trained twice with two seeds: width 256, 11 layers, 32 routed
experts, a vocabulary of 65,536 in one shared table, 44.8M parameters,
1.18B tokens in 8,990 updates, the learning rate at its peak until update
7,192 (commit b7dca7b2f7, RTX 5080 Laptop GPU). Validation loss on the
same windows:

| Update | First seed | Second seed |
| --- | --- | --- |
| 5,400 | 3.32 | 3.27 |
| 6,000 | 3.25 | 3.20 |
| 8,000 | 2.92 | 2.98 |

## Open

- **The spread at 40M to 200M parameters on 1B to 4B tokens.** No source
  measures it there. A measure trains one recipe again with another seed;
  every such repeat, of any recipe, adds to one estimate.
- **The spread of a test of copying from the context.** The same repeats
  give it, on the `copying` test of `wpt eval`.
- **Whether an average of the last weights shrinks the spread.** The same
  runs evaluated with and without the average.
- **Whether a model with loops, or with a lookup table, has the same
  spread** as a plain one.
