# Which changes to a layer lower the loss at an equal number of parameters?

This file asks one question: which changes to the attention, to the
feed-forward part or to the paths between layers make a small model better
without adding parameters, and which are shown only on large models. Its
marks are defined in [README.md](README.md).

## Answer

Few changes are measured under 200M parameters, and one is measured there
by two sources.

- Adding the values of the first layer's attention to the values of every
  later layer lowers the loss by about 0.03 at 82M parameters, or keeps it
  with 16% fewer parameters. **published, agreed**, two sources, **unchecked**
- Paths from every earlier layer to each layer, weighted for each
  position, lower the perplexity by about 8% at 405M for 0.2% more
  parameters. Nothing under 400M. **published, one source**, **unchecked**
- Layers that keep a state from token to token, in place of attention, do
  not beat attention at an equal number of parameters in any source read
  under 200M. Mixed with attention they gain from 300M.
  **published, agreed**, **unchecked**
- A model with no feed-forward part loses 0.02 nats at an equal number of
  parameters from 6M to 87M, for twice the compute. **published, one
  source**, **unchecked**
- Matrices of low rank, or matrices that share their numbers, lose to
  whole matrices at an equal number of parameters. **published, agreed**,
  **unchecked**
- Predicting several tokens from one position hurts under 1B parameters.
  **published, one source**, **unchecked**
- The working assumption: the changes that pay at small sizes are those
  that give a layer something already computed, such as the first layer's
  values, the earlier layers' outputs or the last few tokens, and not
  those that change what a layer computes. **hypothesis**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

### Measured under 200M

- *Value Residual Learning* (2024, arXiv 2410.17897). 82M to 468M, about
  20B tokens. At 82M with 8 layers: loss 2.739 to 2.705. The same loss
  with 16.1% fewer parameters, or with 20.3% fewer tokens at 468M. Only
  the first layer's values help; values from later layers add nothing or
  hurt. Sharing one set of values between all layers loses.
- arXiv 2602.02522 (2026). At 70M: 0.013 less loss with the first layer's
  values, 0.002 with a gate after the attention.
- arXiv 2607.18363 (2026). 6M to 87M. No feed-forward part: 0.02 nats
  worse at equal parameters, about twice the compute per token, and worse
  recall of facts.
- *Hymba* (2024, arXiv 2411.13676). Attention and a state-keeping layer
  side by side. At 300M and 100B tokens: perplexity 28.5 against 30.0. A
  model of 125M exists with no comparison at that size.
- *RWKV-7* (2025, arXiv 2503.14456). 0.1B: a mean score of 50.5 against
  51.0 for a model of attention of 135M trained on other text.

### Measured on larger models only

- *MUDDFormer* (2025, arXiv 2502.12170). 405M, 7B tokens: perplexity 11.68
  to 10.77; fixed weights alone 11.44. 0.23% more parameters, 4% to 16%
  slower, 20% to 30% more memory. Sparse paths lose.
- arXiv 2512.17351 (2025). A convolution over 4 tokens before the
  attention and before the feed-forward part: 2 to 4 times deeper
  reasoning on built tasks for under 0.5% more parameters. Its runs on text
  move by 2% to 4% between seeds.
- arXiv 2603.09078 (2026). Removing from the attention's output the part
  along the token's own value: +0.26 point at 0.7B, +1.03 at 1.3B, +1.36 at
  2.7B. The gain shrinks with the model.
- *Samba* (2024, arXiv 2406.07522). 421M, 20B tokens: perplexity 10.06
  for state-keeping layers with attention over a short window, 11.14 for
  attention alone, 10.70 for state-keeping layers alone.
- arXiv 2507.06457 (2025). 340M and 1.3B. The best mix is 3 to 6
  state-keeping layers for one of attention.
- *Gated Delta Networks* (2024, arXiv 2412.06464). From 400M: perplexity
  16.42 against 18.53 for attention.
- *nGPT* (2024, arXiv 2410.01131). 0.5B and 1B: 4 to 20 times fewer
  updates, each 60% to 80% slower.
- *Differential Transformer* (2024, arXiv 2410.05258). 1.4B: loss 3.087 to
  3.062. Nothing under 830M.

### Matrices of fewer numbers

- arXiv 2410.02117 (2024). 120K to 76M parameters. The matrices that scale
  best share no number and have full rank; a mixture of experts inside
  each matrix is worth 5.3 times the compute of a dense one.
- *SLTrain* (2024, arXiv 2406.02214). A matrix written as a sparse one plus
  one of low rank. 60M on 1.1B tokens: perplexity 34.15 with 44M
  parameters against 34.06 with 58M. 130M on 2.2B: 26.04 with 97M against
  24.36 with 134M.
- arXiv 2406.06248 (2024). Matrices built as products of smaller ones lose
  to dense ones.

### More than one token at a time

- *Better and Faster Large Language Models via Multi-token Prediction*
  (2024, arXiv 2404.19737). Worse than next-token training at 0.3B, better
  from 1.3B.
- *ParScale* (2025, arXiv 2505.10475). The same weights run on P changed
  copies of the input act like the logarithm of P times the parameters,
  for P times the compute: +1.6 point with 8 copies at 0.5B. Nothing under
  0.5B.
- *Think before you speak* (2023, arXiv 2310.02226). Learned pause tokens
  placed in the training text: better on 6 tests of 9 at 130M, 8 of 9 at
  1B.

## Open

- **The first layer's values, with what WPT already carries between
  layers.** A measure trains one recipe with and without them, twice each
  with different seeds, and compares the loss.
- **Paths from every earlier layer, under 100M.** The same measure with
  weighted paths, beside the time per update.
- **State-keeping layers in place of the layers that read a short
  window**, at an equal number of parameters, with the copying test of
  `wpt eval` beside the loss: the sources find such layers weaker at
  recalling what the context states.
- The share of a small model's parameters that should go to attention and
  to the feed-forward part: no controlled measure found near 100M.
