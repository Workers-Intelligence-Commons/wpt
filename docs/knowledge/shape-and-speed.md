# What does a model's shape cost in training speed?

This file answers one question: how many tokens per second a training run
reads on one GPU, according to the model's width, its depth and the length
of its windows. Its marks are defined in [README.md](README.md).

## Answer

- On one laptop GPU, a model with 9 times the parameters of a narrow one
  trains at a half to a third of its speed, not a ninth: a narrow model does
  not use the GPU. Width is cheap, and the time of a narrow model goes to
  costs that do not shrink with it. **measured here**
- Shorter windows do not make a run faster per token. **measured here**
- More windows per pass make a run faster when they fit in memory.
  **measured here**
- The multi-token prediction block costs at most 8% of a pass at width 128.
  **measured here**
- `--stop-after-minutes` gives the speed of a recipe on a machine before a
  whole run is paid for (docs/training.md, "A limit and a measure").

## Evidence

### Measured with WPT

One machine: RTX 5080 Laptop GPU with 16 GiB, Windows, Triton kernels,
commit d0d454b22b, a vocabulary of 65,536 tokens, `--recompute` in every
run. The narrow model has a width of 128 and 21 layers, 13.7M parameters
without the vocabulary's two tables and 30.5M with them. The wide model has
a width of 512 and 11 layers, 124.0M parameters without the tables and
191.1M with them.

| Model | Windows per update | `--micro-batch` | Updates measured | Tokens per second |
| --- | --- | --- | --- | --- |
| narrow | 12 of 32,768 | 1 | 3,000 | 32,400 |
| narrow | 192 of 2,048 | 4 | 9 | 19,200 |
| wide | 64 of 2,048 | 4 | 600 | 16,200 |
| wide | 64 of 2,048 | 16 | 19 | 16,600 |
| wide | 64 of 2,048 | 2 | 23 | 13,800 |
| wide | 4 of 32,768 | 1 | 26 | 12,300 |

The runs of 600 and 3,000 updates include their evaluations; the others are
`--stop-after-minutes` measures, which leave the first update out.

The cost of the multi-token prediction block: one pass, forward and
backward, over 4 windows of 4,096 tokens with the narrow model, without
`--recompute` and without the optimizer's step, timed with the block and
with the block left out. 33,500 tokens per second with it, 36,300 without.
The optimizer's step is not in the measure, so the share in a whole update
is lower.

### Published

- *PanGu-π Pro* (2024, arXiv 2402.02791): generation speed by shape at about
  1B parameters on one GPU, 12.8 tokens per second at 40 layers and 57.5 at
  9. **published, one source**, **unchecked**

## Open

- **Where the narrow model's time goes.** The output over 65,536 tokens, the
  routing of 32 experts, the number of layers and the optimizer's step are
  the candidates; `--profile-update` gives the share of each
  (docs/training.md, "A limit and a measure").
- The speed of the same shapes without `--recompute`, which a GPU with more
  memory allows.
- The speed of a model of width 512 and 21 layers, which the comparison of
  depths at equal parameters needs
  ([depth-and-width.md](depth-and-width.md)).
- Training speed by shape on other GPUs: no published source measures it.
