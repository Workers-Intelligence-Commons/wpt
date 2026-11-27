# Can the vocabulary's two tables shrink without losing precision on the vocabulary?

This file asks one question: how a small model can spend fewer parameters
on its input embedding and its output head, which are two separate tables
in WPT, without lowering what its output can express. Its marks are defined
in [README.md](README.md).

## Answer

Under 100M parameters one measure made here favours one shared table; it
is one run of each model.

- Two tables hold twice the vocabulary times the width, and are the largest
  part of a small model: 33.5M of 74.5M parameters at a width of 256 with a
  vocabulary of 65,536, 67.1M of 191.1M at 512. **proved**, by counting
- Sharing one table between the input and the output costs nothing or gains
  a little from 120M to 600M parameters, with 13% to 50% fewer parameters.
  **published, agreed**
- Under 100M parameters no source compares the two at an equal number of
  parameters, and one source finds sharing harmful at a width of 64.
  **published, one source**
- At a width of 256 with a vocabulary of 65,536, the model with one shared
  table, 44.8M parameters, beats the same model with two, 61.6M, trained
  on the same 1.18B tokens: validation loss 2.79 against 2.85, and 95% of
  the tokens of a repeated sequence copied against 48%. The gap in loss is
  there at every update compared. **measured here**, one run each
- A second run of the model with one table, with another seed, copies 82%
  of the tokens and ends at a validation loss of 2.85 on windows of its
  own ([run-to-run-spread.md](run-to-run-spread.md)). What the three runs
  show: the model with one table copies better than the one with two at
  both seeds, 95% and 82% against 48%, with 27% fewer parameters. Its gain
  in loss is not shown.
- Sharing changes two things at once, the number of parameters and the
  tie between reading a token and predicting it; the measure does not
  say which of the two gives the gain.
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
- Against the source above: arXiv 2605.09751 (2026). Fixed binary codes in
  place of the input table, vocabulary 65,536, width 1,024, 17B tokens, 3
  seeds: a perplexity of 2.36 against 2.44 at 32 layers and 3.76 against
  3.77 at 9, within the spread of the seeds.
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
- arXiv 2608.11361 (2026). Width 512, 12 layers, 47M to 307M parameters
  with the tables, 50M tokens. Bits per byte: 1.368 at a vocabulary of
  8,192, 1.344 at 16,384, 1.346 at 65,536, 1.355 at 131,072. The whole
  spread is under 2%, on very little text.

### Counted from the model

Parameters of a model of 11 layers with a vocabulary of 65,536 and two
tables. **proved**, by counting

| Width | Body | The two tables | Share of the model |
| --- | --- | --- | --- |
| 256 | 41.0M | 33.5M | 45% |
| 512 | 124.0M | 67.1M | 35% |

### Measured with WPT

Two models of width 256, 11 layers, 32 routed experts, a vocabulary of
65,536 and no Engram memory, trained once each on the same 1.18B tokens
with the same stages (windows of 2,048, 8,192 then 32,768 tokens, 8,990
updates of 131,072 tokens), the same seed and the same optimizer, at
commit b7dca7b2f7. One has two tables (61.6M parameters, 19.1 tokens per
parameter, RTX 5090), the other one shared table (44.8M, 26.3 tokens per
parameter, RTX 5080 Laptop GPU).

| | Two tables | One table |
| --- | --- | --- |
| Validation loss at update 5,400 | 3.40 | 3.32 |
| Validation loss at update 8,000 | 2.97 | 2.92 |
| Validation loss at update 8,990 | 2.85 | 2.79 |
| Repeated sequence, tokens written | 48% | 95% |
| Repeated sequence, written whole | 29% | 89% |
| Stated word, likeliest of four, English and French | 95%, 96% | 100%, 100% |
| Stated word, written, English and French | 4%, 15% | 14%, 30% |
| Stated number, likeliest of four, English and French | 100%, 100% | 100%, 100% |
| Stated number, written, English and French | 18%, 38% | 84%, 90% |

The copying figures are those of the `copying` test of `wpt eval`, on 200
prompts each; the stated values were measured on the processor, at commit
321bb02efa. Other models of this shape copy 43% to 70% of the tokens
from one run to another ([lookup-memory.md](lookup-memory.md)), so a gap
of 47 points is outside what those runs show, and the spread between two
runs of one order is still not measured.

## Open

- **The same measure with another seed**, for each of the two models: the
  gap in loss, 0.05, is the size of a gap that one run cannot settle.
- **What gives the gain.** A model with one table made as wide as the
  model with two is large says whether the freed parameters do more in
  the body; a model with two tables started equal says whether the tie
  itself helps.

- A sweep of vocabularies from 4,096 to 65,536 at an equal total of
  parameters and equal training bytes, with a measure per byte.
- A head of higher rank at a width of 128 to 384 on real text.
