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
- Two runs of one recipe of 45M parameters copy a repeated sequence at
  95% and at 82% of its tokens: a test of copying moves by 13 points
  between seeds, so it ranks two models only when they differ by much
  more. **measured here**, two runs
- A loss measured on a few million tokens moves by about 0.05 nats with
  the windows drawn, as much as most gaps between models. Two models are
  compared on the same windows, or the gap says nothing: two runs of one
  recipe measured on different windows end 0.06 apart, which the windows
  alone can give. **measured here**, two runs
- So one run of each model shows a gap in validation loss when it is
  several times the spread between seeds, on the same windows: about 0.02
  nats if the published spread holds at this size, which is not measured.
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
1.18B tokens in 8,990 updates (commit b7dca7b2f7, RTX 5080 Laptop GPU).
Each run measures its validation loss on 20 batches of windows drawn anew
at each measure from its own seed, so the two runs never read the same
windows, and within one run the loss moves by 0.05 to 0.08 from one
measure to the next while the training loss falls smoothly.

| | First seed | Second seed |
| --- | --- | --- |
| Validation loss at update 6,000, own windows | 3.25 | 3.20 |
| Validation loss at update 8,000, own windows | 2.92 | 2.98 |
| Validation loss at update 8,990, own windows | 2.79 | 2.85 |
| Repeated sequence, tokens written | 95% | 82% |
| Repeated sequence, written whole | 89% | 74% |
| Stated word, likeliest of four, English and French | 77%, 80% | 80%, 74% |
| Stated number, likeliest of four, English and French | 92%, 86% | 90%, 80% |

The copying figures are those of the `copying` test of `wpt eval`, the
same 200 prompts for both runs.

## Open

- **The spread at 40M to 200M parameters on 1B to 4B tokens.** No source
  measures it there. The two runs above give it once their loss is
  measured on one set of windows, large enough for its own error to be
  under 0.01 nats.
- **The spread of a test of copying from the context.** The same repeats
  give it, on the `copying` test of `wpt eval`.
- **Whether an average of the last weights shrinks the spread.** The same
  runs evaluated with and without the average.
- **Whether a model with loops, or with a lookup table, has the same
  spread** as a plain one.
