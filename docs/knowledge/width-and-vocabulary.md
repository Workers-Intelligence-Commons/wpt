# How wide must a model be for its vocabulary?

This file answers one question: below which width (`d_model`) does a
language model stop learning, and what sets that floor. Its marks are
defined in [README.md](README.md).

## Answer

- A model has a floor of width, and the floor rises with the size of the
  vocabulary. **published, agreed**
- No published formula gives the floor. The working rule is at most 50
  vocabulary entries per dimension of width: a vocabulary of 65,536 asks for
  a width of about 1,300, one of 16,384 for about 330. **hypothesis**
- Full quality of the output asks for a width of about 1,000 or more with a
  vocabulary of 30,000 to 50,000. **published, one source**
- A model that must be narrow takes a small vocabulary: every narrow model
  that works has one. **published, agreed**
- Whether a model copies a token from its context is the test of the floor,
  not the loss: a model under the floor still writes grammatical text.
  **measured here**

## Evidence

All figures from published sources in this section are **unchecked**.

### The output layer limits what a narrow model can express and learn

- *Breaking the Softmax Bottleneck* (2017, arXiv 1711.03953). The scores of
  every token are a product of the hidden state and the output table, so the
  matrix of log-probabilities has a rank of at most the width. The paper
  argues that language needs a higher rank, and shows gains from a head of
  higher rank on word vocabularies (10,000 and 100,000) and none on
  characters.
- *Why do small language models underperform?* (2024, arXiv 2404.07647). On
  models of 14M to 410M parameters with a vocabulary of 50,000, quality
  drops when the rank of the output head falls below about 1,000, whatever
  the size of the model.
- *Lost in Backpropagation: The LM Head is a Gradient Bottleneck* (2026,
  arXiv 2603.10145). One update can move the scores in at most twice the
  width directions, of up to the vocabulary's size minus one. Between 95%
  and 99% of the gradient's norm is lost through the head. A model of width
  576 trained on sequences that repeat one token learns them with a
  vocabulary of 16,384, in part with 32,768, and not with 65,536 or 131,072
  (41M tokens of training). The paper gives no threshold in entries per
  dimension, and tested one width.

### A small vocabulary lowers the floor

- *Which transformer architecture fits my data? A vocabulary bottleneck in
  self-attention* (2021, arXiv 2105.03928). Width above the rank of the
  embedding, itself at most the vocabulary's size, adds little. It is a
  ceiling on useful width, not a floor, and the only published rule that
  names the vocabulary.
- *Scaling Laws with Vocabulary* (2024, arXiv 2407.13623). The best number of
  vocabulary parameters grows as the other parameters to the power 0.83, and
  the best vocabulary shrinks when training data is limited (from 16,000 to
  10,000 in their case). The smallest width tested is 512.
- Narrow models that work: *TinyStories* (2023, arXiv 2305.07759), widths 64
  to 128 with 10,000 tokens of child-level text; induction at width 128 with
  a vocabulary of 65 characters (*Birth of a Transformer*, 2023, arXiv
  2306.00802); *GPT or BERT: why not both?* (2024, arXiv 2410.24159), width
  384 with a vocabulary of 8,192 on 10M words and width 768 with 16,384 on
  100M words.

### What speaks against a simple ratio

- Models in use sit above 50 entries per dimension and work: a width of 768
  with a vocabulary of 50,257 is 65 per dimension.
- The repetition experiment that fails at 114 entries per dimension trains on
  41M tokens. Whether more training closes the gap is not measured.
- The rank limit concerns the output. A narrow input embedding costs little
  when the hidden width stays large (*ALBERT*, 2019, arXiv 1909.11942).

### Measured with WPT

- A model of width 128 and 21 layers (13.7M parameters without the
  vocabulary's two tables, 30.5M with them), vocabulary 65,536, 512 entries
  per dimension, trained on 1.18B tokens in 3,000 updates of 12 windows of
  32,768 tokens, validation loss 3.49: it writes grammatical English and
  copies nothing from its context. Given a sentence that states a fact and
  the start of the same sentence, it does not repeat the fact; it does not
  continue a repeated sequence of four letters. (RTX 5080 Laptop GPU, commit
  d0d454b22b.) **measured here**

## Open

- **The floor as a function of the vocabulary.** No source crosses width
  with vocabulary size below a width of 512 for a model that predicts the
  next token, with a test of copying. A measure needs one corpus built with
  several vocabularies (8,192, 16,384, 65,536), several widths under 512 at
  one depth, and the share of prompts where the model copies a token from
  its context.
- Whether more training tokens, or more updates, lift a model that is under
  the floor.
- Whether a head of higher rank than the width (a mixture of heads, or a
  factorized output) moves the floor.
- How the floor moves with depth: depth helps copying, and does not lift the
  limit on the output's rank.
