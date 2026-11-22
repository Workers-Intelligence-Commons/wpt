# At an equal number of bytes, few precise weights or many coarse ones?

This file asks one question: whether a model trained with weights of 1 to 4
bits, and more of them, beats a model of the same size in bytes with
weights of 16 bits, and whether passing the text through the same layers
several times gives back the precision the coarse weights lose. Its marks
are defined in [README.md](README.md).

## Answer

The question is open: WPT trains weights of full precision only.

- At an equal number of parameters, a model trained with three-valued
  weights always loses, and loses more the smaller it is: a lot under 100M,
  2 to 4 points from 100M to 500M, almost nothing at 3B. **published,
  agreed**
- At an equal number of bytes it wins only above a few hundred million
  parameters. **published, one source**
- Training with coarse weights needs a much higher learning rate and is
  less stable. **published, agreed**
- Nothing measures coarse weights together with loops. The one source on
  loops and coarse weights quantizes after training and finds that the
  error made where the loop is entered comes back at every pass.
  **published, one source**
- The vocabulary's tables stay precise in every published model, and they
  are a third to a half of a small WPT model
  ([vocabulary-tables.md](vocabulary-tables.md)): the bytes saved are far
  under what the bits per weight suggest. **hypothesis**
- The working assumption: under 500M parameters, full precision. A loop
  reads the same coarse weights again, so it repeats their error and does
  not average it out; what a loop adds is depth, with coarse weights as
  with precise ones. Whether more depth makes up for coarser weights is a
  measure to make, not a known result. **hypothesis**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

- *Spectra* (2024, arXiv 2407.12327). 99M to 3.9B, 300B tokens. Mean of 6
  tests, three-valued against full precision at equal parameters: 43.5
  against 44.3 at 99M, 42.4 against 46.7 at 190M, 46.4 against 48.5 at
  390M, 48.4 against 50.5 at 560M. At equal bits, the three-valued model of
  190M is under the precise one of 99M, and the three-valued one of 560M is
  over the precise one of 190M. About 6 times the learning rate, spikes in
  the loss.
- *BitNet b1.58* (2024, arXiv 2402.17764). Perplexity against 16 bits:
  12.87 against 12.33 at 700M, 9.91 against 10.04 at 3B.
- *BitNet b1.58 Reloaded* (2024, arXiv 2407.09527). 6M to 48M, 1.35B
  tokens. Perplexity 26.8 against 16.6 at 48M, 36.3 against 21.4 at 24M.
  Doubling the width makes up the loss at 6M to 12M and not at 48M.
- arXiv 2502.16440 (2025). The share of a precise parameter one coarse
  weight is worth: 0.923 at 4 bits, 0.702 at 2, 0.466 at 1. With the
  activations coarse as well, 0.067 at 1 bit.
- *ParetoQ* (2025, arXiv 2502.02631). A model of 125M, tuned and not
  trained from scratch: 48.5 at full precision, 48.7 at 4 bits, 47.1 at 2,
  46.5 at three values.
- arXiv 2406.02528 (2024). At 370M, 40.3 against 41.1; three-valued
  attention does not converge.
- arXiv 2502.11895 (2025). At 1B, a loss of 3.15 for three values from
  scratch, 3.09 when training starts at 16 bits and switches, 2.95 at 16
  bits.
- *TernaryLM* (2026, arXiv 2602.07374). 132M: perplexity 58.4 against 28.25.
- arXiv 2607.02893 (2026). 131M at 1.82 bits on average against 55M at 16
  bits: perplexity 4.2 against 4.4 on simple stories, for 3.8 times less
  storage.
- arXiv 2609.30820 (2026). A looped model quantized to 4 bits after
  training: the error at the loop's entry is fed back at later passes.

## Open

- **Coarse weights and loops together.** No source trains them together. A
  measure trains, at an equal number of bytes, a precise model with no
  loop, a three-valued model 4 to 8 times larger with no loop, and the same
  three-valued model with 1 to 3 loops, and gives the loss, the copying
  test of `wpt eval` and the time per token of each.
- Where the crossing at equal bytes falls when the tables are shared or
  coarse too.
- Which parts of WPT bear coarse weights: the routed experts are most of
  the body and are read 4 at a time; attention is where published work
  fails.
- The cost in training time of coarse weights on one GPU: no source gives
  it.
