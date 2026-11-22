# At small sizes, are parameters better spent on experts or compute on loops?

This file asks one question: for a small model, how far the number of
routed experts can go down, and whether passing the text through the same
layers several times gives back what the experts gave. Its marks are
defined in [README.md](README.md).

## Answer

The question is open: nothing measures it for WPT, and no source has been
collected yet.

- The working assumption is the one every recipe uses: 32 routed experts,
  4 of them per token, and no loop (docs/model.md, "Experts" and "Loops").
  **hypothesis**
- The two numbers do not trade the same thing. The number of experts
  changes the parameters and not the work per token, since 4 run whatever
  their number. Loops change the work per token, `loops + 1` times one
  pass, and not the parameters. Fewer experts and more loops is a smaller
  model that computes more. **measured here** for the parameters, by
  counting them; the work per token is the model's rule (docs/model.md,
  "Model series").
- The routed experts are the largest part of a small model's body, so the
  number of experts is the strongest lever on its size after the
  vocabulary. **measured here**

## Evidence

### Measured with WPT

Parameters of the body, without the vocabulary's two tables, for 11 layers
and 4 experts per token, counted from the model's code:

| Width | 32 experts | 16 experts | 8 experts |
| --- | --- | --- | --- |
| 512 | 124.0M | 89.4M | 72.0M |
| 256 | 41.0M | 32.3M | 27.9M |

The 32 routed experts are 56% of the body at a width of 512 and 43% at 256.
No model with fewer than 32 experts, and no model that loops, has been
trained.

### Published

- *Ouro* (2025, arXiv 2510.25741), as docs/model.md reads it: at equal
  compute, from 53M to 1.36B parameters, the plain model has the lower loss
  in every case, by 0.015 to 0.039, and accuracy by number of passes peaks
  at 3 to 4. This compares at equal compute, not at equal parameters, which
  is the comparison this question needs. **published, one source**,
  **unchecked**

No source on the number of experts in models under 200M parameters has been
collected.

## Open

- **The number of experts a small model needs.** A measure trains one shape
  at a width of 256 or 512 with 32, 16 and 8 experts, 4 per token, on the
  same tokens, and compares the validation loss and the copying test of
  `wpt eval` (docs/evaluation.md). The time per update should not move.
- **Whether loops give back what the experts gave.** A measure compares, at
  an equal number of parameters, a model with 32 experts and no loop with a
  wider model of 8 experts and 1 to 3 loops, and gives each one's time per
  token beside its loss: a loop is paid in time at training and at
  generation.
- Whether the answer changes with size: experts may pay only once a model
  has enough tokens per expert to specialise them.
- Whether fewer than 4 experts per token is enough at small sizes, which
  lowers the work per token as well.
- What published sources say on mixtures of experts in small models and on
  weight sharing across passes: none has been read for this file.
