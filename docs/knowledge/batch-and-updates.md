# With the same tokens, does a small model gain from more, smaller updates?

This file asks one question: for a fixed number of training tokens, how
many tokens one update should read, and whether fewer tokens per update,
so more updates, bring a lower loss or earlier abilities. Its marks are
defined in [README.md](README.md).

## Answer

The question is open: the sources agree on a critical size of update and
dispute whether the smallest update or a middle one is the best.

- Under a critical size of update nothing is lost per token, and above it
  tokens are wasted. **published, agreed**
- That size follows the number of training tokens and hardly the model's
  size. **published, agreed**
- One fit gives the best update as 0.667 times the number of training
  tokens to the power 0.566: about 90,000 tokens for 1.18B training tokens,
  by this file's arithmetic. **published, one source**
- Copying from the context comes sooner in tokens with smaller updates: 8
  times fewer tokens per update bring it about 3.7 times sooner.
  **published, one source**
- Whether the smallest update is the best, or a middle one, is disputed.
- The working assumption: with 1.18B training tokens, 131,072 tokens per
  update is near the best for the loss, and a model that does not copy yet
  is the case where 4 times fewer could pay. **hypothesis**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

- arXiv 2507.07101 (2025). 30M on 600M tokens and 124M on 2.5B. Every
  optimizer does best at the smallest update, down to one window of 512 to
  1,024 tokens, when the optimizer's second-moment decay is raised with it
  (0.9999 at one window).
- arXiv 2410.21676 (2024). 85M to 1.2B, windows of 512. The largest update
  that costs 20% more tokens: about 742 windows at 85M, about 380,000
  tokens, and about 2,477 at 1.2B. At fixed data it grows as the model's
  size to the power 0.087, and as the data to the power 0.47.
- *Power Lines* (2025, arXiv 2505.13738). The best update grows as the data
  to the power 0.4, the critical one to the power 0.5, both whatever the
  model's size.
- arXiv 2607.01487 (2026). 50M to 1B. The best update is 0.667 times the
  data to the power 0.566, in tokens; only updates of 32,000 tokens and
  more were tried.
- *Predicting the Emergence of Induction Heads* (2025, arXiv 2511.16893).
  50M to 350M, 1B tokens. The tokens until the copying heads form grow as
  the update's windows to the power 0.63 times the window's length to the
  power 0.38, over 500 to 2M tokens per update.
- arXiv 2610.08975 (2026). A loss of 3.27 at 256,000 tokens per update
  against 3.38 at 2M.

## Open

- **A sweep of the update's size on WPT's corpus.** A measure trains one
  small shape on the same tokens with 32,768, 131,072 and 524,288 tokens
  per update, the learning rate chosen for each, and compares the loss, the
  copying test of `wpt eval` at equal tokens, and the time.
- Whether the optimizer's settings (`by-part`, docs/training.md) must move
  with the update's size, as the first source above finds for Adam.
- Whether long windows count as large updates for this question: the last
  source's law has the window's length in it.
