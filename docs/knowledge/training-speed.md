# What makes a training run faster without changing the model it trains?

This file answers one question: which parts of the training code and which
options of `wpt train` decide how many tokens per second a run reads, when
the trained model has to stay the same. Its marks are defined in
[README.md](README.md). What a model's shape costs is in
[shape-and-speed.md](shape-and-speed.md).

## Answer

- The time of a pass goes to memory traffic on the GPU, not to Python and
  not to loading tokens: one fused kernel in place of a chain of separate
  operations is where speed is found. **measured here**
- Three chains carry most of it: the sums of the Attention Residuals
  (+14%), the attention of the window layers (+13%) and the experts'
  activation (+9%). **measured here**
- Recomputation costs 9% to 12% of the speed once the exact and sparse
  attention are left out of it, and it divides memory by more than two.
  A stage therefore keeps every activation whenever its pass fits the GPU.
  **measured here**
- Compilation adds 5% to 16% and costs 20 to 35 seconds at the first update
  of each window length. A run therefore compiles whenever the machine can.
  **measured here**
- Longer windows are slower per token at an equal number of tokens per
  pass: about a quarter slower at 32,768 tokens than at 2,048.
  **measured here**
- A change is kept only with a proof that the model stays the same, at one
  of two levels: work removed gives weights identical bit for bit on a CPU;
  the same mathematics in another order gives gradients equal within
  rounding on one pass. On a GPU, two runs of the same code already differ,
  so a GPU comparison is read against that spread.
- The speeds below come from synthetic tokens and a plain optimizer. They
  give ratios between two states of the code, not the speed of a recipe.
  **hypothesis**: the ratios hold on a real run.

## Evidence

### Measured with WPT

One machine: RTX 5090 with 32 GiB, Linux, Triton kernels, PyTorch 2.14.1
for CUDA 13.0. The script `scripts/compare_training.py` trains on random
tokens with AdamW, 32,768 tokens per pass (16 windows of 2,048, 4 of 8,192
or 1 of 32,768), one pass per update. Both models have 11 layers, 32 routed
experts and a vocabulary of 65,536 tokens. "Before" is commit fdb17c5e61,
"after" is commit 55eb0e950f.

Tokens per second, width 512:

| Code | Recomputation | Compiled | 2,048 | 8,192 | 32,768 | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| before | yes | no | 56,558 | 51,817 | 44,418 | 8.0 GiB |
| after | yes | no | 87,081 | 78,525 | 64,287 | 10.1 GiB |
| after | no | no | 96,864 | 86,246 | 69,572 | 25.0 GiB |
| before | yes | yes | 87,418 | 77,371 | 62,314 | 6.9 GiB |
| after | yes | yes | 101,410 | 91,214 | 75,251 | 9.1 GiB |
| before | no | yes | 101,994 | 91,715 | 75,700 | 24.4 GiB at the first update |
| after | no | yes | 110,914 | 98,823 | 80,509 | 21.4 GiB at the first update |

Before, without recomputation and uncompiled, the pass did not fit the
32 GiB.

Tokens per second, width 256:

| Code | Recomputation | Compiled | 2,048 | 8,192 | 32,768 | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| before | yes | no | 108,583 | 97,795 | 84,477 | 4.3 GiB |
| after | yes | no | 159,736 | 141,543 | 119,123 | 5.6 GiB |
| before | no | no | 125,098 | 113,704 | 100,709 | 17.8 GiB |
| after | no | no | 172,854 | 151,662 | 126,114 | 13.3 GiB |
| after | yes | yes | 181,221 | 158,771 | 132,721 | 5.1 GiB |
| after | no | yes | 192,663 | 168,550 | 139,311 | 12.9 GiB at the first update |

From "before, with recomputation, uncompiled" to "after, without
recomputation, compiled", the speed is multiplied by 1.81 to 1.96 at width
512 and by 1.65 to 1.77 at width 256. With the same options on both sides,
it is multiplied by 1.45 to 1.54 at width 512 and by 1.41 to 1.47 at width
256.

Commit by commit, width 512, windows of 8,192 tokens, with recomputation,
uncompiled:

| Change | Tokens per second | Gain |
| --- | --- | --- |
| before | 51,817 | |
| the exact and sparse attention run once under recomputation | 54,555 | +5% |
| the Attention Residuals sums through their kernels in every pass | 62,443 | +14% |
| the experts' copies gathered in one step | 62,881 | +1% |
| the head's chunk gradients in one sum | 63,765 | +1% |
| the experts' activation in one kernel each way | 69,387 | +9% |
| the window layers through the sparse kernels | 78,684 | +13% |

Keeping the Triton autotuner's choices on disk takes the first update from
7.8 seconds to 0.8. Skipping the multi-token prediction block in
validation, reusing an unchanged token file without hashing it and copying
a first gradient only after a recorded pass move no speed this measure can
see: it runs no validation and reads no token file.

Sameness. On a CPU, a model of 11 layers trained for 20 updates has weights
identical bit for bit at every commit, with recomputation off, on, and only
on the longer windows. On the GPU, the same comparison gives differences of
0.011 to 0.035 between a commit and the one before it, where two runs of the
"before" code differ by 0.013 to 0.032. The spread comes from two runs.

### What the measure does not cover

- The optimizer of the recipes (`by-part`), the evaluations, the
  checkpoints and the drafter phase are not in these speeds.
- The peak memory of a compiled pass without recomputation reads 2.45 GiB
  at width 256 and 5.77 GiB at width 512 after the first update. The memory
  the compiled graphs hold is probably not counted in it.
- The TileLang kernels were not timed: every speed above is Triton's.

## Open

- **The gain on a real run.** A run of a recipe on its corpus, before and
  after, on the same machine, with the validation loss at the same update.
  It would turn the ratios above from a hypothesis into a measure.
- **The rule that chooses recomputation.** It foresees the memory of a pass
  from two short probe passes, with a margin of 30% per token and 10% of
  the GPU left free. Both numbers come from one GPU. A measure of foreseen
  against peak memory on a second GPU, and on Windows, where the driver
  spills into system memory rather than refusing, would settle them.
- **The drafter phase.** It writes its answers one token at a time. On a
  run of 12 hours at width 256 (RTX 5080 Laptop GPU, commit d0d454b22b) it
  takes 55 minutes, 43 of them to write the 256 answers: 7% of the run, and
  a fixed time that weighs more on a short run. Writing the answers in
  batches would change the drafter's random draws, so it is not a change
  that leaves the drafter the same.
- **A larger GPU.** Whether 80 GiB buys speed beyond what 32 GiB gives
  depends on the passes that do not fit 32 GiB without recomputation; at
  width 512 and 32,768 tokens per pass, every pass fits.
- **Fusing further.** The head still writes its scores to memory, and the
  routed experts stay outside the compiled passes.
