# Does a parameter do more in a lookup table than in a layer?

This file asks one question: for a small model, whether parameters placed
in tables read by the last tokens, as WPT's Engram memory does
(docs/model.md, "Engram memory"), are worth more than the same parameters
in layers. Its marks are defined in [README.md](README.md).

## Answer

The question is open at WPT's sizes: every recipe has the same Engram
tables, and none has been trained without them or with others.

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
- Against: *kNN-LM Does Not Improve Open-ended Text Generation* (2023,
  arXiv 2305.14625). Perplexity 14.8 to 12.6, and readers do not prefer the
  text.

### Measured with WPT

Every recipe's Engram memory is 16 tables (2 orders, 8 hash heads) of about
50,021 rows of 16 numbers at layer 2. Counted from the model's code, the
memory is 13.0M parameters at a width of 256 and 13.1M at 512, 12.8M of
them the tables, which do not follow the width. That is 32% of the body at
256 and 11% at 512. No run measures what they bring. **measured here**

## Open

- **Tables against layers at an equal number of parameters, under 500M.**
  A measure trains one shape with no table, with the recipe's tables and
  with tables 4 times larger, each at the same total of parameters by
  changing the width, and compares the loss, `general_knowledge` and
  `copying` of `wpt eval`.
- Whether the tables' size should follow the model's width: today it does
  not.
- Tables with loops: whether a looped model reads the table at every pass
  to advantage.
- Whether the gain in loss shows in generated text.
