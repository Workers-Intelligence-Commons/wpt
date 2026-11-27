# With the same model and text, what in the training changes the result?

This file asks one question: which choices of loss, optimizer, schedule
and number of passes over the text change what a small model reaches, the
model and the text staying the same. The size of an update is in
[batch-and-updates.md](batch-and-updates.md), and learning from another
model in [teachers-and-specialists.md](teachers-and-specialists.md). Its
marks are defined in [README.md](README.md).

## Answer

- A learning rate that falls in a straight line to zero gives a lower loss
  than one that stops at a tenth of its peak, by 1.6% at 111M parameters
  and 200 tokens per parameter, and a higher one at 2 tokens per
  parameter. **published, one source**, **unchecked**
- Optimizers that work on whole matrices reach a loss with fewer tokens
  than a well tuned AdamW: 1.4 times fewer at 130M, 1.1 times at 1.2B.
  Against a badly tuned AdamW the gap looks twice as large.
  **published, agreed**, **unchecked**
- When text is scarce, a weight decay tuned with the number of passes
  gains: 0.8 and 16 passes at 150M parameters on 200M tokens. How many
  passes, and the weight decay with them, are in
  [reading-scarce-text.md](reading-scarce-text.md), whose figures are
  checked. **published, one source**
- A model trained to fill masked tokens keeps learning from the same text
  for about 100 passes, where a model trained on the next token stops
  after about 4; with one pass the next-token model wins.
  **published, one source**, **unchecked**
- A loss that mixes the next token with masked tokens won a contest of
  small models on 10M and 100M words, on tests of grammar; its gain on
  the loss of the next token is not given. **published, one source**,
  **unchecked**
- Dropout, an order of the text from easy to hard, and losses built on
  linguistic rules do not help in the sources read. **published, agreed**,
  **unchecked**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

### Schedule and optimizer

- arXiv 2502.15938 (2025). At 111M and 200 tokens per parameter, 1.6% less
  loss with a decay to zero than to 10%; 0.77% at 610M and 20 tokens per
  parameter; worse at 2 tokens per parameter.
- arXiv 2509.02046 (2025). The best gain over a tuned AdamW: 1.4 times at
  130M, about 1.3 up to 520M, 1.1 at 1.2B. Tuning only the learning rate
  of a weak AdamW gives up to 2 times.
- arXiv 2505.02222 (2025). Muon needs 10% to 15% fewer tokens, from 100M.
- arXiv 2505.24788 (2025). No dropout gives a lower loss than dropout at
  160M and 1.4B.

### Scarce text

- arXiv 2509.14786 (2025). 200M tokens, 150M to 1.4B parameters. The best
  weight decay is 0.8 at 150M; the best number of passes 16
  ([reading-scarce-text.md](reading-scarce-text.md)). A model taught by
  a trained copy of itself beats the best plain run.
- arXiv 2507.15857 (2025). 7M to 2.5B parameters, 25M to 100M tokens. On
  100M tokens trained long: loss 3.55 for the masked model against 3.71.
  With one pass: 10.65 against 7.07. Giving the next-token model 16
  orders of the tokens nearly closes the gap.

### Other losses

- *GPT or BERT: why not both?* (2024, arXiv 2410.24159), with the contest's
  report (arXiv 2412.05149). 30M on 10M words: 81.2 on a grammar test
  against 69.8 for the contest's best baseline; 119M on 100M words: 86.1
  against 69.2. The baselines differ by more than the loss.
- arXiv 2511.05963 (2025). Predicting the model's own next state beside
  the next token, 57M: the same loss, 12% to 50% more time.
- The contest's reports (arXiv 2504.08165, arXiv 2412.05149). An order of
  the text from easy to hard mostly failed; methods built on linguistic
  rules lost; the score follows the compute spent.

## Open

- **The decay to zero at 20 tokens per parameter under 100M.** The source
  measures it at 200 and at 2, with opposite signs.
- **Masked tokens in the loss, at 1B tokens and more.** The sources train
  on 100M words at most. A measure trains one order with and without a
  share of masked tokens and compares the loss of the next token and the
  copying test of `wpt eval`.
- **Many passes over less text against one pass over more**, at an equal
  number of updates, under 100M parameters: the measure is written in
  [reading-scarce-text.md](reading-scarce-text.md).
- **How many stages of growing windows, and their shares of the tokens.**
  No source read measures the number of stages or their shares: DeepSeek-V4
  names its window lengths (4K, 16K, 64K, 1M) and states no tokens per
  stage. A measure trains the same models, under 30M parameters, on the
  same tokens in two ways: nine stages of windows doubling from 128 to
  32,768 tokens, each an equal share of the tokens, against three stages
  (2,048, 8,192 and 32,768 tokens). It compares the bits per byte on one
  held-out text read in windows of 32,768 tokens, and the copying test.
  A gap larger than the spread between two runs at every size, of one
  sign, closes the question for these sizes.
