# How large a model for how many training tokens?

This file answers one question: given a number of training tokens, how many
parameters a model should have, and how many tokens a model needs before it
shows a behaviour. Its marks are defined in [README.md](README.md).

## Answer

- When every token is read once and compute is the limit, the best size is
  one parameter for 15 to 20 training tokens, all parameters counted, the
  vocabulary tables included. **published, agreed**
- Under 1B tokens that rule is an extrapolation, with one measure in its
  favour at 100M tokens. **published, one source**
- When the data is the limit and compute is not, a larger model that reads
  the data several times wins: up to 4 readings are worth about as much as
  new data, and the gain fades near 16. **published, one source**
- A training method can reach a given loss with far fewer tokens than the
  rule suggests: a model of 124M parameters reaches the loss of GPT-2 small
  with about 330M tokens. This is a property of the method, not another best
  ratio. **published, one source**
- Copying from the context appears after a number of updates more than after
  a number of tokens, hardly depends on the model's size, and needs at least
  2 layers. Small batches bring it sooner in tokens. **published, agreed**
- Recalling a fact depends on how many times the corpus states it, about
  100 to 1,000 times, and on the model's capacity, about 2 bits of knowledge
  per parameter. A test of facts measures the corpus as much as the model.
  **published, agreed**

## Evidence

All figures from published sources in this section are **unchecked**.

### Parameters for a token budget

- *Training Compute-Optimal Large Language Models* (2022, arXiv 2203.15556).
  More than 400 runs, 70M to 16B parameters, 5B to more than 400B tokens.
  The best size grows as compute to the power 0.46 to 0.50, about 20 tokens
  per parameter: 400M parameters for 8B tokens. Its smallest run reads 5B
  tokens.
- *Chinchilla Scaling: A replication attempt* (2024, arXiv 2404.10102). A
  refit of the same runs: exponent 0.513, about 20 tokens per parameter,
  with 4 to 40 inside the error bars.
- *Resolving Discrepancies in Compute-Optimal Scaling* (2024, arXiv
  2406.19146). 55M to 901M parameters. The best ratio is 11 to 22 tokens per
  parameter once warm-up, batch, learning rate and the optimizer's second
  moment are tuned for each size; small models need their own settings.
- *Reconciling Kaplan and Chinchilla Scaling Laws* (2024, arXiv 2406.12907).
  The older, steeper law came from leaving the embeddings out of the count
  at small sizes.
- *Scaling Data-Constrained Language Models* (2023, arXiv 2305.16264). More
  than 400 runs, 7M to 9B parameters, 100M, 400M and 1.5B distinct tokens,
  up to 1,500 readings. With 100M tokens read once, the best size is about
  7M parameters; the best loss comes from a model 20 to 60 times larger that
  reads the data many times. About 44 readings diverged.
- modded-nanogpt (github.com/KellerJordan/modded-nanogpt). 124M parameters,
  11 layers of width 768, reach a validation loss of 3.28 on FineWeb with
  under 330M tokens; the first recipe for that loss read 10B.

### When copying appears

- *In-context Learning and Induction Heads* (2022, arXiv 2209.11895).
  Induction heads form abruptly at about 2.5B to 5B tokens across 34 models,
  and never in models of one layer.
- *LLM Circuit Analyses Are Consistent Across Training and Scale* (2024,
  arXiv 2407.10827). Models of 70M to 2.8B parameters, 2.1M tokens per
  update: induction heads just after 2B tokens, about the same at every
  size, which is about 1,000 updates.
- *Language Models Grow Less Humanlike beyond Phase Transition* (2025, arXiv
  2502.18802). The transition comes near 2B tokens for three sizes with
  large batches, and near 64M tokens for a model of 2 layers with small
  batches.
- *Predicting the Emergence of Induction Heads* (2025, arXiv 2511.16893).
  The updates until the heads form grow as the square root of the batch
  times the context.

### When a fact is recalled

- *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws*
  (2024, arXiv 2404.05405). About 2 bits of knowledge per parameter when
  each fact is seen 1,000 times, at any depth of 2 or more; about 1 bit at
  100 times. A fact that sits among unrelated text is learned up to 20 times
  worse.
- *How much do language models memorize?* (2025, arXiv 2505.24832). A raw
  capacity of about 3.6 bits per parameter.

### Measured with WPT

- 3,000 updates and 1.18B tokens do not bring copying to a model of width
  128 with a vocabulary of 65,536
  ([width-and-vocabulary.md](width-and-vocabulary.md)): the number of
  updates is not enough when the model is under the floor of width.
  **measured here**

## Open

- The best size under 1B tokens with a current training method: the one
  study there uses an older architecture and one dataset.
- Whether the limit of about 16 readings holds for models under 100M
  parameters and for a corpus that is not web text.
- The tokens, or updates, until a model recalls a common fact, by model size,
  under 1B parameters.
- Whether the best width or depth changes with the token budget: every study
  of shape fixes one budget.
