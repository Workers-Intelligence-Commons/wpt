# Does a parameter do more in a lookup table than in a layer?

This file asks one question: for a small model, whether parameters placed
in tables read by the last tokens, as WPT's Engram memory does
(docs/model.md, "Engram memory"), are worth more than the same parameters
in layers. Its marks are defined in [README.md](README.md).

## Answer

The question is open at small sizes.

- A table lowers the loss for almost no compute, gains most on facts, and
  works best early in the model. **published, agreed**
- The gain grows with the logarithm of the table's size: each 4 times more
  rows adds about the same small step. **published, agreed**
- Per parameter a table is a poor trade: it pays per unit of compute and
  not per byte. **published, agreed**
- The gain of a table shrinks as the model grows, in the one source that
  measures two small sizes. **published, one source**
- A lower loss from a lookup does not always give better generated text.
  **published, one source**
- At a width of 256 with a vocabulary of 65,536, a table of 13.0M
  parameters, 17% of the model, lowers the validation loss by 0.05, from
  2.85 to 2.79, with the same training tokens. **measured here**, one run
  with the table and one without
- The table does not help the model copy from its context: without it the
  model copies a repeated sequence as well or better (48% of the tokens
  against 43%). A table reads the last tokens, not the context, so it can
  bring what the training text repeats and not what the prompt states.
  **measured here**, one run each; the direction of the gap is within what
  one run can tell
- That gap of 0.05 comes from one run of each model with the same seed,
  so on the same validation windows; the spread between two seeds at this
  size is not measured ([run-to-run-spread.md](run-to-run-spread.md)): it
  is a gap to confirm, not a settled gain.
- Whether those 13.0M parameters do more in the table than in layers is not
  measured: the model without the table is smaller, not wider.
- A table 16 times smaller, 0.9M parameters, keeps none of that gain: its
  loss is that of the model without a table (2.84 against 2.85). The
  published law, one equal step for each 4 times more rows, would leave
  part of the gain; it does not describe tables this small. **measured
  here**, one run
- The three models copy a repeated sequence at 43%, 48% and 70% of its
  tokens, the best being the one with the small table. Nothing here says
  how far two runs of one order differ on that test, so these three
  figures do not rank the models. **measured here**, one run each
- A trained model does not copy through its table. With the table's
  output set to zero after training, the model with the small table copies
  as well (74% of the tokens against 69%), and the one with the large
  table a little less (37% against 44%). What differs between the models
  is what their layers learned during training, by the table's presence
  or by chance. **measured here**, one run each
- The working assumption: a small model that must stay small in bytes
  keeps small tables, and one that must stay small in compute takes large
  ones. Loops bring depth and no knowledge, tables bring knowledge and no
  depth ([experts-and-loops.md](experts-and-loops.md)), so the two are the
  natural pair to try together. **hypothesis**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

- *Engram* (2026, arXiv 2601.07372). At an equal total of 26.7B parameters
  and equal compute: loss 1.622 against 1.634. The best share is 20% to 25%
  of the parameters not active per token; the best place is layer 2. With
  the table switched off, tests of facts keep 29% to 44% and reading tests
  81% to 93%.
- arXiv 2606.08347 (2026). 9 layers of width 512 and 18 of width 1,024, one
  seed. Bits per byte 1.251 to 1.209 with 26M of table at the small size,
  1.086 to 1.070 with 60M at the large one.
- *Over-Tokenized Transformer* (2025, arXiv 2501.16975). A model of 400M
  with a hashed table of 12.8M rows of 2 and 3 tokens reaches the training
  loss of one of 1B. Each 4 times more rows: 0.015 less loss. 4.6% slower.
- *SCONE* (2025, arXiv 2502.01637). 419M: perplexity 26.1 to 23.4 with a
  side model of 170M for its table, better than a plain 589M at 24.7.
- *Byte Latent Transformer* (2024, arXiv 2412.09871). 1B: bits per byte
  0.850 to 0.822 with hashed tables; 4 more layers in their place give
  0.843.
- *Memory Layers at Scale* (2024, arXiv 2412.09764). 134M with memory, 937M
  in all: 0.91 to 3.16 and 7.7 to 18.77 on two tests of facts. At 8B a code
  test goes down.
- arXiv 2510.02375 (2025). 160M with 18M fetched: 34.1 to 40.3 on specific
  knowledge; the same 18M always present: 35.7.
- *Mixture of Lookup Experts* (2025, arXiv 2503.15798). Experts that read
  the token's vector alone, so their output is a table with one row per
  token. 160M active, 100B tokens: a mean score of 38.8 for the dense
  model, 40.8 with 4 such experts and 41.9 with 16, where a mixture of 34
  ordinary experts gives 41.8. The table of the 4 experts holds 1.8B
  numbers.
- arXiv 2602.00398 (2026). Feed-forward layers that read the token's
  vector alone, 50B tokens. 245M active: perplexity 22.08 against 23.19
  for a dense 265M. At an equal total of parameters the dense model wins.
- *PEER* (2024, arXiv 2407.04153) and *UltraMem* (2024, arXiv 2411.12364).
  A million experts of one neuron, or a sparse memory 12 times the dense
  part. At 151M dense and 500B tokens and more: loss 2.96 dense, 2.67 with
  the memory, 2.63 with a mixture of experts.
- *Infini-gram* (2024, arXiv 2401.17377). The counts of a text of 5,000B
  tokens mixed with a model of 117M: perplexity 22.82 to 13.71. The gain
  falls to 11% with a model of 70B, and the model had not read that text.
- arXiv 2601.21204 (2026). From 280M active. A table should stay under
  about half the parameters, read 3 to 5 tokens, and have its vectors
  scaled up before the layers read them.
- Against: *kNN-LM Does Not Improve Open-ended Text Generation* (2023,
  arXiv 2305.14625). Perplexity 14.8 to 12.6, and readers do not prefer the
  text.

### Counted from the model

Tables whose number of rows and row size are set apart from the width hold
the same parameters at every width, so they are a larger share of a
narrower model. An Engram memory of 16 tables (2 orders, 8 hash heads) of
about 50,021 rows of 16 numbers is 13.0M parameters at a width of 256 and
13.1M at 512, 12.8M of them the tables: 32% of the body at 256 and 11% at
512, for 11 layers and 32 experts. **proved**, by counting

### Measured with WPT

Three models of width 256, 11 layers, 32 routed experts and a vocabulary of
65,536, trained once each on the same 1.18B tokens with the same stages
(windows of 2,048, 8,192 then 32,768 tokens, 8,990 updates of 131,072
tokens), the same seed and the same optimizer. One has the Engram memory
counted above at layer 2 (74.5M parameters, 15.8 tokens per parameter,
commit d0d454b22b, RTX 5080 Laptop GPU), one has the same memory with
tables of about 3,126 rows, 16 times fewer (62.5M, 18.9 tokens per
parameter, commit 8777a037b8, RTX 5090), and one has none (61.6M, 19.1
tokens per parameter, commit b7dca7b2f7, RTX 5090).

| | Table of 13.0M | Table of 0.9M | No table |
| --- | --- | --- | --- |
| Validation loss at update 8,000 | 2.92 | 2.97 | 2.97 |
| Validation loss at update 8,990 | 2.79 | 2.84 | 2.85 |
| Repeated sequence, tokens written | 43% | 70% | 48% |
| Repeated sequence, written whole | 24% | 50% | 29% |
| Stated word, likeliest of four, English and French | 68%, 68%, by tokens | 96%, 98% | 95%, 96% |
| Stated word, written, English and French | 2%, 4%, by tokens | 18%, 38% | 4%, 15% |
| Stated number, likeliest of four, English and French | 84%, 74%, by tokens | 100%, 100% | 100%, 100% |
| Stated number, written, English and French | 0%, 1%, by tokens | 43%, 62% | 18%, 38% |

The copying figures are those of the `copying` test of `wpt eval`, on 200
prompts each; the stated values of the tables of 0.9M and of none were
measured on the processor, at commit 321bb02efa. The model with the table
of 13.0M has no drafter, which `wpt eval` needs, so its stated values are
counted by tokens: each of the four scored alone, without the sentence's
period that the tokenizer may join to its last letters or digits, and
written only when the tokens equal those of the value alone, which
refuses most values written right. With the weights that turn the table's rows into what it
adds (W_V) set to zero in the trained model, the same test gives 74% of
the tokens and 54% of the sequences whole for the small table (69% and 50%
with it, on the same machine), and 37% and 19% for the large one (44% and
24% with it). The models differ by their table and, with it, by their
number of parameters and of tokens per parameter: the comparison says what
the table adds to this model, not what its parameters would do elsewhere.

## Open

- **Tables against layers at an equal number of parameters, under 500M.**
  A measure trains one shape with no table, with the order's tables and
  with tables 4 times larger, each at the same total of parameters by
  changing the width, and compares the loss, `general_knowledge` and
  `copying` of `wpt eval`.
- **Whether a table pays depends on the total of parameters.** The gain of
  a table shrinks as the model grows, and so does its share of the model.
  Which of the two shrinks faster decides at which sizes a table is worth
  its parameters. A measure repeats the comparison above at two totals at
  least, with and without the table at an equal total.
- **The spread between two runs of one order.** The same order trained
  again with another seed says how far the loss and the copying test move
  by chance. Without it, a gap of 0.05 in loss rests on its being the same
  at two updates, and the copying figures cannot be compared at all.
- **Between 0.9M and 13.0M.** Where the gain of the table appears: a table
  4 times smaller than the larger one would say whether the gain grows
  step by step or needs a size.
- Whether the tables' size should follow the model's width.
- Tables with loops: whether a looped model reads the table at every pass
  to advantage.
- Whether the gain in loss shows in generated text.
