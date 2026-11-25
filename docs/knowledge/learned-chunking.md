# Can a model learn where to cut its text instead of reading a fixed vocabulary?

This file asks one question: whether a small model does better when it
reads bytes and learns, together with the rest of its weights, how to
group them, than when it reads the tokens of a vocabulary fixed before
training. Its marks are defined in [README.md](README.md).

## Answer

The question is open under 200M parameters: no source measures it there on
text.

- A fixed vocabulary costs its two tables, twice the vocabulary times the
  width: 45% of a model of 11 layers at a width of 256 with a vocabulary of
  65,536 ([vocabulary-tables.md](vocabulary-tables.md)). A model that reads
  bytes has 256 entries: its two tables hold 512 times the width, 0.13M
  parameters at a width of 256 against 33.5M. **proved**, by counting
- A model that reads bytes sees as many positions as the text has bytes,
  several times more than a model that reads tokens. Grouping the bytes
  before the main layers brings the number of positions those layers see
  back near the number of tokens; the layers that read the bytes themselves
  still run on every byte. **proved**, by counting
- With a vocabulary of 256 the output's rank is bounded by 256 and no
  longer by a vocabulary far larger than the width, so the floor on the
  width that a large vocabulary sets
  ([width-and-vocabulary.md](width-and-vocabulary.md)) no longer applies.
  **proved**
- A model that learns its cuts, with one level of grouping, equals a model
  of tokens at an equal amount of compute per byte, and with two levels it
  is better by about 2% of bits per byte, from 680M to 1.6B parameters.
  **published, one source**, **unchecked**
- That model starts behind the model of tokens and passes it only after 30B
  to 100B bytes of training. **published, one source**, **unchecked**
- Its layers that read the bytes are state-space layers; with attention
  layers in their place the result is the worst of the source's variants.
  **published, one source**, **unchecked**
- The gain in training data is measured on DNA, 3.6 times, and not on
  English text. **published, one source**, **unchecked**
- At an equal number of parameters a model of bytes reaches fewer bits per
  byte than a model of tokens, and at equal compute the model of tokens
  wins, from 200M to 3B parameters. **published, one source**, **unchecked**
- Cuts placed at spaces by a rule equal a model of tokens at equal compute;
  groups of a fixed number of bytes are worse by about 10% of bits per
  byte. **published, one source**, **unchecked**
- The working assumption: at small sizes the share of the tables is what a
  learned cutting could give back, and the amount of training it needs to
  catch up is what could take it away. Nothing says which wins. **hypothesis**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

### Learned cuts

- *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (H-Net,
  2025, arXiv 2507.07955). An encoder reads the bytes; a cut is placed
  where the vector of a position differs enough from the one before it (a
  cosine similarity of two projections); only the vectors at the cuts go
  through the main layers; their outputs are smoothed, spread back over the
  bytes, and a decoder predicts the next byte. The loss is the
  cross-entropy on the next byte plus 0.03 times a term that pulls the rate
  of cuts to a target. Bits per byte, on 100B tokens of FineWeb-Edu:

  | Size | Tokens | One level | Two levels |
  | --- | --- | --- | --- |
  | about 700M | 0.756 | 0.755 | 0.743 |
  | about 1.3B | 0.730 | 0.728 | 0.715 |

  The groups hold 4.7 to 4.8 bytes with one level, where a token of GPT-2's
  vocabulary holds 4.6. The cuts fall mostly on spaces. Without the
  smoothing, the size of the groups swings and the result drops. The
  comparison holds the compute per byte equal, not the number of
  parameters. Nothing is measured under 200M parameters on text.
- *SOMBRERO* (2026, arXiv 2601.22805). Taking the method again, a target
  of 5 bytes per group gave 3.95 with the weight of 0.03, which the authors
  raised to 1.0.

### Fixed cuts and cuts by a small model

- *Byte Latent Transformer* (2024, arXiv 2412.09871). The cuts come from a
  small model of bytes trained beforehand, which cuts where the next byte
  is hard to predict; the cutting is not learned with the model.
- *MegaByte* (2023, arXiv 2305.07185). Groups of a fixed number of bytes.
- *SpaceByte* (2024, arXiv 2404.14408). Cuts at spaces and like bytes, by a
  rule. At equal compute: 1.009 bits per byte against 1.013 for a model of
  tokens on books, 0.748 against 0.796 on research papers; with groups of a
  fixed size, 1.112 on books.
- *Hierarchical Autoregressive Transformers* (2025, arXiv 2501.10322) and
  *Autoregressive U-Nets* (2025, arXiv 2506.14761). Bytes grouped into
  words at spaces, at 1B parameters and more: equal to a model of tokens
  for the first, 73.7 against 70.2 on a test of sentence endings for the
  second. Nothing under 200M.

### Bytes against tokens at an equal size

- arXiv 2610.05978 (2026). 200M to 3B, trained from nothing. Each byte's
  vector is averaged with vectors looked up from hashes of the 5 to 8 bytes
  before it. At equal parameters the bytes reach fewer bits per byte; at
  equal compute the tokens do. The model of bytes wants 3.2 times more
  bytes of text per parameter. On a test of spelling at 200M: 99.19 against
  69.89. Nothing under 200M, and no figure for the hashed tables alone.
- *T-FREE* (2024, arXiv 2406.19223). Words enter and leave as sets of
  hashed groups of three characters, the tables 85% smaller. At 1B and 3B
  it equals a vocabulary of 64,000 on 18 tests; no loss is given.

### From a model of tokens

- *Distilling Token-Trained Models into Byte-Level Models* (2026, arXiv
  2602.01007) and *Bolmo* (2025, arXiv 2512.15586). A model of tokens
  already trained is turned into a model of bytes with about 125B bytes,
  where training one from nothing takes thousands of billions. Both need
  the model of tokens first.

## Open

- **A model of learned cuts under 100M parameters.** A measure trains one
  level of grouping at about 60M parameters against a model of tokens of
  the same total on the same bytes, and compares bits per byte at several
  amounts of training. It closes the question at that size if the gap is
  closed, or says how fast it closes.
- **What reads the bytes.** The same measure with attention over a short
  window in place of state-space layers says whether the published failure
  of attention holds at small sizes.
- **Cuts by a rule, under 200M.** The sources above find cuts at spaces as
  good as a vocabulary and groups of a fixed size worse, at 150M
  parameters and more. A measure trains a model of bytes cut at spaces
  against a model of tokens of the same total: whether the gain needs the
  cuts to be learned, or only the tables to be gone.
- **Bytes at an equal number of parameters, under 200M.** The one source
  that compares them starts at 200M, where the tables are a smaller share
  of the model than at 60M.
- **A vocabulary whose merges are chosen in training.** A model that reads
  the tokens of a vocabulary and learns when to merge two neighbouring
  entries in one position: no source found measures it.
- **The share that a smaller or shared vocabulary already gives back**
  ([vocabulary-tables.md](vocabulary-tables.md)): a learned cutting is
  worth its cost only beyond that.
