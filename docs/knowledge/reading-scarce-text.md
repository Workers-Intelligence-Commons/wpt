# How many times can a scarce text be read?

This file asks one question: when the text is fixed and compute is not,
how many times a model should read it, at which size, and what in the
training must change with the number of readings. How large a model
should be for text read once is in [size-and-tokens.md](size-and-tokens.md).
Its marks are defined in [README.md](README.md).

## Answer

- Up to about 4 readings of a text lower the loss nearly as much as as
  many tokens of new text. Each reading after that is worth less: the
  value of repeated tokens fades with about 15 repetitions as its scale,
  and runs of about 44 readings got worse midway. **published, one
  source**. A second source extrapolates the same 4 readings for models
  that predict the next token. **unchecked**
- How many readings pay depends on the model's size against the text: a
  larger model on less distinct text is hurt by repetition sooner, and
  the harm grows faster than the number of readings. With tuned settings
  the best number of passes over 200M tokens is 16 at 150M and 300M
  parameters and 8 at 600M and 1.4B. **published, agreed**
- Stronger weight decay lets a model gain from more readings, and costs
  loss when the text is plentiful. The best weight decay rises with the
  model's size and falls with the amount of text. **published, agreed**.
  No source measures the best weight decay against the number of readings
  alone: the one that tunes both tunes them together with the learning
  rate, at each size and amount of text, and the other compares two
  values on one grid. No rule that sets weight decay from the number of
  readings follows from them. **hypothesis**, a reading of the sources
  below
- A source fitted on large models finds the product of the learning rate
  and the weight decay at its best falls as 1 over the number of updates,
  and holds the repetition of each source of text fixed while fitting. At
  a fixed learning rate, that law asks for less weight decay in a run of
  more updates, where the two sources above ask for more when the extra
  updates are readings. **hypothesis**: the two do not measure the same
  thing, and nothing here separates them.
- Each reading in another order is what the sources that gained from
  readings did; one that kept the same order every pass says a new order
  might have done better, and giving a model 16 orders of the same tokens
  helped. **published, one source** each, the last **unchecked**
- These sources measure loss, and some benchmarks, never whether a model
  recalls a fact its text states. What repetition does to facts is
  measured on synthetic biographies: about 1,000 exposures of a fact for
  2 bits of knowledge per parameter, 100 for at least 1, and a fact read
  many times in the same wording is stored without being retrievable.
  **published, one source** each, **unchecked**
- Against repetition: a model writes out more of its training text the
  more times the text was repeated, and the more parameters it has;
  repeating a small part of the text many times hurts a model far more
  than its share of the tokens. **published, agreed**, **unchecked**
- At small sizes the evidence is thin: one source trains from 15M
  parameters on 50M to 6B distinct tokens up to 16 readings, with windows
  of 512 tokens and AdamW. Nothing measures readings of about 1B tokens
  with a mixture of experts, NorMuon or long windows. **published, one
  source**

## Evidence

Each figure below was compared with the source's own text, except where
it says **unchecked**.

### For

- *Scaling Data-Constrained Language Models* (2023, arXiv 2305.16264).
  More than 400 models from 10M (7M in its fits) to 9B parameters, GPT-2's
  architecture and vocabulary, C4, distinct text of 100M, 400M or 1.5B
  tokens and larger budgets at fixed compute, up to 1,500 readings, the
  data shuffled after each reading. "training with up to 4 epochs of
  repeated data yields negligible changes to loss compared to having
  unique data"; the 8.7B model read 4 times ends "with only 0.5% higher
  validation loss than the single-epoch model". The value of repeated
  data is `D' = U + U R* (1 - exp(-R / R*))`, `R` the repetitions (the
  readings less one), fitted at `R* = 15.39` for the data and 5.31 for
  excess parameters, so it advises spending more on readings than on
  parameters. Runs of 44 readings got worse midway, and the fit
  underestimates their final loss. Dropout 0.1 and weight decay 0.1
  throughout, never tuned; the authors expect better regularization to
  keep excess parameters from hurting. It measures test loss and 19
  benchmarks, no recall of facts.
- *Pre-training under infinite compute* (2025, arXiv 2509.14786). AdamW,
  a Llama shape, windows of 4,096 tokens, 200M tokens of DCLM (209M
  exactly) and 400M, 800M and 1.7B, models of 150M, 300M, 600M and 1.4B.
  The learning rate, the weight decay and the number of passes are tuned
  together at each size and amount of text. At 200M tokens:

  | | 150M | 300M | 600M | 1.4B |
  | --- | --- | --- | --- | --- |
  | Learning rate | 3e-3 | 3e-3 | 1e-3 | 1e-3 |
  | Passes | 16 | 16 | 8 | 8 |
  | Weight decay | 0.8 | 1.6 | 3.2 | 3.2 |

  "The optimal weight decay increases for larger models. Similarly, the
  optimal weight decay decreases for larger token counts. When fixing the
  parameter-to-token ratio, the weight decay stays around 0.8." The
  passes follow the opposite way and stay around 16 at a fixed ratio. The
  tuned recipe, extrapolated to an infinite model, is worth 2.29 times
  the text of the usual recipe at 200M tokens: that figure is the limit of
  every size tuned, not the gain of the 150M model. It reused the same
  order of windows at every pass and writes that a new order each pass
  might have done better. It measures validation loss and three
  benchmarks, no recall of facts.
- *Prescriptive Scaling Laws for Data Constrained Training* (2026, arXiv
  2605.01640). Llama 2's shape and vocabulary, FineWeb, windows of 512
  tokens, AdamW at a learning rate of 2e-4, nine sizes from 15M to 1B
  parameters, 50M to 6B distinct tokens, up to 16 readings, weight decay
  0.1, 0.5, 1.0 and 3.0 on the same grid. It adds to the Chinchilla law a
  penalty that grows with the repetitions and with the parameters over the
  distinct tokens; the penalty grows faster than the repetitions. A weight
  decay of 1.0 "reduces this coefficient by approximately 70%", and
  "Stronger regularization is strictly worse when data is abundant". At
  250M distinct tokens, weight decay 1.0 overtakes 0.1 from about 3.2e18
  FLOPs. It measures validation loss and bits per byte on 19 tasks.
- *Diffusion Beats Autoregressive in Data-Constrained Settings* (2025,
  arXiv 2507.15857). 7M to 2.5B parameters, 25M to 100M distinct tokens,
  up to 800 readings. A model of the next token is best at about 50
  readings of 100M tokens and worsens after; repeated text is worth nearly
  as much as new up to about 4 readings; 16 fixed orders of the tokens
  brought the next-token model near the masked one. **unchecked**
- *Frontier Pre-training at Microsoft AI* (Microsoft AI, 2026, a report,
  microsoft.ai/pdf/Frontier-Pre-training-at-Microsoft-AI.pdf). It names no
  author. A ladder of mixtures of experts named by their layers, L12 to
  L42, the width 256/3 times the layers, at 30 to 677 tokens per active
  parameter; the learning rate and the weight decay swept together on a
  grid of the learning rate and their product, a quadratic fitted to each
  slice. Its fits: `LR* x WD* = 17.04 / steps` and
  `log LR* = -1.505 - 0.572 log steps + 0.174 log TPP` (Figure 4; the
  text rounds them to -1.51, -0.573 and 0.175), TPP the tokens per active
  parameter. It reads the first as a decoupled weight decay time scale
  `1 / (LR x WD)` that stays a fixed share of training. "Source scaling":
  a small run restricts its distinct text so each source repeats as many
  times as in the full run; without it, it writes, small runs "can favor
  under-regularized" choices. Its large run trains with Muon and AdamW;
  it states neither the optimizer of the sweeps, nor their batch, nor the
  base of its logarithms. It shows one run per point, ablates nothing of
  source scaling, and gives no rung in parameters; at a width of 1,024
  and more, none is likely under a few hundred million active
  parameters (**hypothesis**). Its constants may not carry to another
  recipe.

### Against

- *To Repeat or Not To Repeat: Insights from Scaling LLM under
  Token-Crisis* (2023, arXiv 2305.13230). T5 models of masked tokens from
  247M to 2.8B, C4 down to about 134M tokens read 256 times: the larger
  model reading less distinct text did worse than a smaller one with 4
  times the text; dropout recovered part of the loss, and weight decay
  with Adafactor diverged. **unchecked**
- *Scaling Laws and Interpretability of Learning from Repeated Data*
  (2022, arXiv 2205.10487). 1.5M to 800M parameters, 100B tokens, a part
  of the text repeated many times: 0.1% of the data repeated 100 times
  brings an 800M model down to a model half its size; the harm peaks
  where the training loss on the repeated part nears zero, and copying
  from the context is hurt far more than the loss. No regularization was
  tried. **unchecked**
- *Quantifying Memorization Across Neural Language Models* (2022, arXiv
  2202.07646). GPT-Neo from 125M to 6B: the share of training sequences
  a model writes out grows log-linearly with the times a sequence was
  duplicated, and with the model's size. Duplicates within one reading,
  not whole readings. **unchecked**

### Facts, not loss

- *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws*
  (2024, arXiv 2404.05405). GPT-2 shapes from 1M to 0.5B on synthetic
  biographies: about 2 bits per parameter after 1,000 exposures of each
  fact, at least 1 after 100. **unchecked**
- *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction*
  (2023, arXiv 2309.14316). A fact read in one wording, even 1,350 times,
  is stored and cannot be retrieved by questions; 5 wordings of each
  biography raise the answers from 9.7% to 96.6%. **unchecked**
- *How Do Large Language Models Acquire Factual Knowledge During
  Pretraining?* (2024, arXiv 2406.11813). OLMo 1B and 7B: a fact
  duplicated is learned faster and forgotten faster than one paraphrased,
  ending about the same. **unchecked**

## Open

- **Readings of one text at WPT's sizes.** The same 1.18B tokens read 1,
  2, 4 and 8 times by two sizes, one near 10M parameters without the
  tables and one near 60M, everything else equal, compared on the bits
  per byte of the held-out portion at the end of training, never on the
  validation loss, whose windows are drawn from the training text and so
  are read as many times. It closes the question for these sizes when the
  gain of each doubling of the readings is known against the spread of
  two runs of one order ([run-to-run-spread.md](run-to-run-spread.md)):
  the readings past which a doubling no longer passes that spread. A run
  on twice the distinct text, where the snapshot holds it, says whether 2
  readings are worth new text here as in the sources.
- **Weight decay under readings.** At one size and 4 readings, weight
  decay 0.1, 0.4, 0.8 and 1.6 with NorMuon's cautious decay, against the
  same values at 1 reading, on the held-out bits per byte. It says whether
  the best weight decay moves with the readings at a fixed size and text,
  which no source separates, and in which direction.
- **Facts against loss.** No test of `wpt eval` asks a model for a fact
  its corpus states. A test of facts stated once and stated in several
  wordings, read 1 and 4 times, would say whether readings that lower the
  loss also make a fact retrievable.
- **Copying and repetition.** The copying test of `wpt eval` on the runs
  above, since a source finds repetition hurts copying more than loss;
  read against its spread, which needs more than one run of each.
