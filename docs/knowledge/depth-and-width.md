# How should parameters be split between depth and width?

This file answers one question: at a fixed number of parameters, how many
layers and what width (`d_model`) a language model should have. Its marks
are defined in [README.md](README.md).

## Answer

- Each of the two has a floor. Above both floors the split matters little:
  the loss moves by a few percent over a wide range of shapes. **published,
  agreed**
- The floor of depth is about 8 to 10 layers; models of 1 to 8 layers lose
  clearly wherever they are measured. The floor of width is the subject of
  [width-and-vocabulary.md](width-and-vocabulary.md). **published, agreed**
- The best depth rises with the size of the model: about 5 layers at 40M
  parameters, 12 at 130M, 24 at 370M. **published, one source**
- Past 20 to 30 layers a plain transformer gains little or loses. A model
  whose layers read earlier layers through more than one residual stream, as
  WPT's Attention Residuals do, keeps gaining with depth. **published, one
  source**
- Training time grows about linearly with depth at a fixed size.
  **published, one source**
- A learning rate found at one depth does not carry to another depth, so a
  series of sizes that shares its settings keeps one depth. **published,
  one source**

## Evidence

All figures from published sources in this section are **unchecked**.

### The optimum is flat, and very shallow loses

- *Scaling Laws for Neural Language Models* (2020, arXiv 2001.08361). At a
  fixed count of parameters without the embeddings, the loss varies by a few
  percent while the ratio of width to depth changes 40 times: 6 layers of
  width 4,288 are within 3% of 48 layers of width 1,600. Models under 2
  layers fall off. Counting the embeddings makes depth look more important
  than it is.
- *The Impact of Depth on Compositional Generalization* (2023, arXiv
  2310.19956). Three sizes, 41M, 134M and 374M parameters, 1 to 32 layers,
  131B tokens. Best perplexity at 5, 12 and 24 layers. One layer is 1.6 to
  2 times worse. Gains on generalization saturate by 2 to 6 layers. The
  study trades depth against the width of the feed-forward part only, and
  perplexity worsens once that width falls below the model's.
- *The Model Shape Behind Scaling Laws* (2026, arXiv 2605.27989). Byte-level
  models of 0.3M to 50M parameters, 254 runs. At 50M: 40 layers 0.9659, 64
  layers 0.9679, 1 layer 1.2411. Its fit, depth ≈ 5.3 × (parameters /
  1M)^0.46, holds for a vocabulary of 256, which favours depth.

### Sources that favour depth

- *MobileLLM* (2024, arXiv 2402.14905). 19 shapes at about 125M and 350M
  parameters, scored on zero-shot tasks. At 125M: 4 layers 43.3, 12 layers
  43.9, 30 layers 44.8, 62 layers 44.7. Single runs, gaps often under one
  point, a token budget far from data-limited.
- *The Depth-to-Width Interplay in Self-Attention* (2020, arXiv 2006.12467).
  Depth pays only up to a threshold that grows with the logarithm of the
  width. The fit of its measures, ln(width) = 5.039 + 0.0555 × layers, gives
  about 22 layers at width 512 and 29 at 768. Measured with a vocabulary of
  2,000 and windows of 128 tokens.
- *PanGu-π Pro* (2024, arXiv 2402.02791). At about 1B parameters: 40 layers
  42.02, 20 layers 41.19, 15 layers 37.57, 9 layers 35.46, on three tasks.

### Sources that favour width

- *DepthBench* (2026, arXiv 2609.32534). About 400M parameters, 7 shapes
  from 16 to 70 layers, 8B tokens, learning rate tuned per shape. A plain
  pre-norm transformer gets worse with depth (16 layers 2.759, 32 layers
  2.782). Designs with richer residual connections reverse it: full
  Attention Residuals go from 2.751 to 2.718 as depth grows.

### Why the sources disagree

- **The measure.** Sources that favour depth score tasks; sources that find
  the split unimportant score the loss.
- **The token budget.** The source that favours width trains on 20 tokens
  per parameter, the main one that favours depth on far more.
- **The count.** A narrow model spends less on its vocabulary tables, so a
  comparison at equal total parameters gives it a larger body.
- **The residual design**, as *DepthBench* shows.

### Depth costs training time

- *The Impact of Depth*: training time grows about linearly with depth at a
  fixed size.

### Settings do not carry across depths

- *Let's Scale Step by Step* (arXiv 2608.20061). Its rules scale the initial
  values and the learning rate with the width, and it holds the depth fixed
  because its rule did not carry from one depth to another.

## Open

- **A shape for each size of a series.** Whether 11 layers at 124M
  parameters lose to 21 layers at the same count with WPT's layers, which
  have Attention Residuals: one run of each on the same tokens, compared by
  loss and by what the models do.
- A sweep of shapes at equal parameters without the tables, scored by loss
  and by tasks, over several token budgets, with several seeds. No source
  has all four.
- The best shape under 50M parameters with a subword vocabulary: the
  evidence there is byte-level or uses a very small vocabulary.
- How the best shape moves with the length of the context.
