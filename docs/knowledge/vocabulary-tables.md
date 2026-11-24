# Can the vocabulary's two tables shrink without losing precision on the vocabulary?

This file asks one question: how a small model can spend fewer parameters
on its input embedding and its output head, which are two separate tables
in WPT, without lowering what its output can express. Its marks are defined
in [README.md](README.md).

## Answer

The question is open under 100M parameters.

- Two tables hold twice the vocabulary times the width, and are the largest
  part of a small model: 33.5M of 74.5M parameters at a width of 256 with a
  vocabulary of 65,536, 67.1M of 191.1M at 512. **proved**, by counting
- Sharing one table between the input and the output costs nothing or gains
  a little from 120M to 600M parameters, with 13% to 50% fewer parameters.
  **published, agreed**
- Under 100M parameters no source compares the two at an equal number of
  parameters, and one source finds sharing harmful at a width of 64.
  **published, one source**
- A narrower table written as a product of two smaller ones loses quality
  for every parameter it saves. **published, agreed**
- A head of higher rank than the width, such as a mixture of softmaxes,
  gains at most about 2% of perplexity and sometimes loses. **published,
  agreed**
- The working assumption: share the table and give the freed parameters to
  the width. Sharing leaves the output's table whole, so the output keeps
  its rank, and a wider model raises that rank
  ([width-and-vocabulary.md](width-and-vocabulary.md)). **hypothesis**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

### One table or two

- *MobileLLM* (2024, arXiv 2402.14905). 30 layers, width 512. Two tables,
  135M: 44.8 of mean accuracy; one table, 119M: 44.6; one table and 2 more
  layers, 125M: 45.0.
- arXiv 2611.10556 (2026). One seed. One table, 142.5M: 2.8069 nats; two,
  181.1M: 2.8132.
- *Leviathan* (2026, arXiv 2601.22040). Width 1,024, vocabulary 200,000,
  8.4B tokens, one seed. One table, 407M: 2.784 nats; two, 611M: 2.801.
- *Adaptive Input Representations* (2018, arXiv 1809.10853). One table is
  better on WikiText-103, 23.45 against 24.25, and worse on Billion Word,
  27.51 against 27.13.
- Against: *SLlama* (2025, aclanthology.org/2025.emnlp-main.1198). Width
  64, 10M tokens. One table, 2.4M parameters: 56.0% on a grammar test; two,
  4.4M: 91.9%. The numbers of parameters differ.
- Against: arXiv 2603.26663 (2026). A shared table ends up shaped for the
  output; measured at 1B parameters and more.
- In use: SmolLM-135M and SmolLM2-135M share one table (width 576,
  vocabulary 49,152), as does Gemma 3 270M.

### Narrower and factorized tables

- *Tensorized Embedding Layers* (2019, arXiv 1901.10787). Width 512,
  267,735 words. Whole tables, 192M: perplexity 24.37; rank 128, 89M:
  26.92; rank 48, 72M: 28.51.
- *Leviathan*, above. A narrow input with a whole head: 2.807, worse than
  one shared table. An input composed by a small network, with a whole
  head: 2.724 against 2.784 at 400M.
- arXiv 2610.22428 (2026). Fixed codes in place of a learned input table:
  3.358 nats against 3.155.
- *Differentiable Product Quantization* (2019, arXiv 1908.09756). The input
  table 163 times smaller at 83.2 against 83.4 of perplexity, on a small
  recurrent model, the output left whole.

### Heads of higher rank

- arXiv 2611.10556, above, at 142M: mixture of 2 softmaxes 2.8239 against
  2.8069 for the plain head; sigsoftmax 2.7992 with no parameter added.
- *Softmax Bottleneck Makes Language Models Unable to Represent Multi-mode
  Word Distributions* (2022, aclanthology.org/2022.acl-long.554). GPT-2
  small: 18.72 for the plain head, 18.63 for sigsoftmax and for a mixture
  of 3, 18.29 for a head with several facets that takes 138 ms against 84.

### A smaller vocabulary

- *Scaling Laws with Vocabulary* (2024, arXiv 2407.13623), read again by
  arXiv 2611.04215 (2026). The measure that compares vocabularies decides
  the result: bits per character favour the largest vocabulary, and bits
  per byte have a minimum near 24,000 even for an untrained model.
- *Over-Tokenized Transformer* (2025, arXiv 2501.16975). A larger output
  vocabulary hurts a model of 2.4M and helps one of 85M.

### Counted from the model

Parameters of a model of 11 layers with a vocabulary of 65,536 and two
tables. **proved**, by counting

| Width | Body | The two tables | Share of the model |
| --- | --- | --- | --- |
| 256 | 41.0M | 33.5M | 45% |
| 512 | 124.0M | 67.1M | 35% |

## Open

- **One table against two at an equal number of parameters, under 100M.** A
  measure trains a width of 256 with two tables, and a wider model with one
  table and the same total, on the same tokens, and compares the loss and
  the copying test of `wpt eval`.
- A sweep of vocabularies from 4,096 to 65,536 at an equal total of
  parameters and equal training bytes, with a measure per byte.
- A head of higher rank at a width of 128 to 384 on real text.
