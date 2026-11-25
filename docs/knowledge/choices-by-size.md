# How should a model's choices change with its size?

This file asks one question: which choices of a model and of its training
stay the same from a few million parameters to a few billion, which must
change, and in which direction. It gathers what the other files of this
folder say about size, and each line names the file that holds its
evidence. Its marks are defined in [README.md](README.md).

## Answer

No choice here is measured over the whole range. What is known gives a
direction for most choices and a value for few.

### What follows the size by a rule

- **The training tokens.** 15 to 20 tokens per parameter, every parameter
  counted, when each token is read once
  ([size-and-tokens.md](size-and-tokens.md)). **published, agreed**
- **The size of an update.** It follows the number of training tokens and
  hardly the model's size ([batch-and-updates.md](batch-and-updates.md)).
  **published, agreed**
- **What the model can hold.** About 2 bits of knowledge per parameter for
  facts seen often: about 15 MB at 60M parameters, 250 MB at 1B. A small
  model cannot be asked for the facts of its corpus, at any training
  ([size-and-tokens.md](size-and-tokens.md)). **published, agreed**
- **The share of the vocabulary's tables.** Twice the vocabulary times the
  width, while the body grows with the square of the width: the share
  falls as the model widens, 45% at a width of 256 and 35% at 512 with
  65,536 entries and 11 layers ([vocabulary-tables.md](vocabulary-tables.md)).
  **proved**

### What a small model must do differently

- **A smaller vocabulary, or none.** A narrow model has a floor of width
  set by its vocabulary: with 65,536 entries a width of 256 copies from
  its context and a width of 128 does not
  ([width-and-vocabulary.md](width-and-vocabulary.md)). **measured here**,
  one run at each width. Under that width the vocabulary shrinks, or the
  model reads bytes ([learned-chunking.md](learned-chunking.md)).
  **hypothesis**
- **One table for the input and the output.** Sharing costs nothing from
  120M to 600M. At 62M it lowers the loss by 0.05 with 27% fewer
  parameters and the model copies from its context far better
  ([vocabulary-tables.md](vocabulary-tables.md)). **published, agreed**
  above 120M; **measured here** at 62M, one run each
- **Full precision.** Three-valued weights lose more the smaller the
  model, and win per byte only above a few hundred million parameters
  ([bits-and-parameters.md](bits-and-parameters.md)). **published, agreed**
- **Fewer experts.** A mixture of experts needs tokens before each expert
  pays, and a small model reads few
  ([experts-and-loops.md](experts-and-loops.md)). **hypothesis**
- **Loops, if the limit is the number of parameters.** At an equal number
  of parameters a loop gains more the smaller the model; at equal compute
  it loses, and loses more at small budgets
  ([experts-and-loops.md](experts-and-loops.md)). **published, agreed**.
  At 45M parameters, 2 loops with a lookup table lower the loss by 0.14
  for 3 times the compute. **measured here**, one run each
- **Lookup tables that grow slower than the model.** The gain of a table
  shrinks as the model grows; a table of 13.0M parameters lowers the loss
  of a model of 62M by 0.05, and one 16 times smaller does nothing
  ([lookup-memory.md](lookup-memory.md)). **measured here**, one run each.
  Whether those parameters do more in layers at an equal total is open.

### What changes as the model grows

- **Depth.** The best depth rises with the size, about 5 layers at 40M, 12
  at 130M and 24 at 370M, above a floor of 8 to 10; a learning rate found
  at one depth does not carry to another
  ([depth-and-width.md](depth-and-width.md)). **published, one source**.
  The two pull apart: a series that shares its settings keeps one depth,
  and the best series changes it.
- **Bytes lose their advantage.** At an equal number of parameters bytes
  beat tokens from 200M to 3B, and the tables they remove are a smaller
  share of a larger model; at equal compute tokens win
  ([learned-chunking.md](learned-chunking.md)). **published, one source**,
  **unchecked**. So the gain of bytes should be largest at the smallest
  sizes, where nothing measures it. **hypothesis**
- **Learned cuts become possible.** A model that learns where to cut its
  text passes a model of tokens after 30B to 100B bytes of training, which
  a model under a few hundred million parameters does not read at 20
  tokens per parameter ([learned-chunking.md](learned-chunking.md)).
  **published, one source**, **unchecked**
- **Some changes to the layers pay only on large models.** The first
  layer's values carried to every layer gain from 82M to 468M; removing
  from the attention the part along the token's own value gains more the
  larger the model, and most other changes are measured from 400M
  ([layer-changes.md](layer-changes.md)). **published**, **unchecked**
- **The optimizer matters less.** Optimizers on whole matrices save 1.4
  times the tokens at 130M and 1.1 times at 1.2B
  ([training-recipe.md](training-recipe.md)). **published, agreed**,
  **unchecked**
- **Coarse weights and many experts become worth their cost**
  ([bits-and-parameters.md](bits-and-parameters.md),
  [experts-and-loops.md](experts-and-loops.md)). **published, agreed**
- **A teacher is already there.** A series trains a larger model beside
  each small one; imitation pays only when the teacher exists and the
  student reads few tokens
  ([teachers-and-specialists.md](teachers-and-specialists.md)).
  **published, one source**

### How often a parameter works

One reading holds the statements above together. **hypothesis**

- A model has two budgets, its parameters and its compute per token, and
  what a choice costs depends on which is short.
- A small model is short of parameters. A parameter that is read for few
  tokens, or that repeats another, is the costly one: a second vocabulary
  table, an expert that runs for one token in eight, a lookup row. Compute
  is what it can add: the same layers run again raise the loss less per
  parameter than anything else measured here, and the published gain of a
  loop at an equal number of parameters is largest at the smallest sizes.
- A large model is short of compute. It adds parameters that cost no
  compute, many experts of which few run, lookup tables, and avoids
  running a layer twice: at equal compute a deeper model beats a looped
  one at every size measured.
- Inside the layers the rule is the same at every size: a matrix whose
  numbers are shared or of low rank loses to a whole one
  ([layer-changes.md](layer-changes.md)). Sharing pays between whole
  parts that do the same job, two tables or two passes, not inside a
  part.
- So the small model is not the one with the most distinct parameters,
  and the large one is not the one that can afford more compute: the
  evidence points the other way on both. What a small model wants is
  every parameter at work on every token, more than once if needed.

### The shapes these give

Each line is a reading of the statements above, not a measured best.
**hypothesis**

| Parameters | Text enters as | Tables | Body | Trained with |
| --- | --- | --- | --- | --- |
| under 30M | bytes cut at spaces, or a vocabulary of a few thousand | one, or none | 8 to 10 layers, few experts, loops | small updates, several passes if text is scarce |
| 30M to 150M | a vocabulary sized to the width, one table | a lookup table only if it beats layers at an equal total | 10 to 12 layers, the first layer's values carried on, a loop with passes by token | 15 to 20 tokens per parameter, an optimizer on whole matrices |
| 150M to 1B | a vocabulary of tens of thousands, one table | lookup tables of a falling share | depth rising with size, layers that keep a state mixed with attention | the same rule of tokens |
| above 1B | a vocabulary, or learned cuts over bytes | lookup tables for compute, not for size | many experts, coarse weights where bytes are the limit | the same rule of tokens |

## Evidence

The evidence for each statement is in the file its line names. Two things
hold for any comparison of sizes or of choices, and come from measures
made with WPT.

- **A comparison stopped early does not rank two models.** Two models of
  62M and 74.5M parameters trained on the same 1.18B tokens had the same
  validation loss at update 5,400 of 8,990, 3.330 and 3.335, and ended
  0.05 apart ([lookup-memory.md](lookup-memory.md)). **measured here**,
  one run each
- **A test of copying moves between runs by more than the choices
  compared.** Three models that differ by one table copy 43%, 48% and 70%
  of a repeated sequence, in an order no choice explains
  ([lookup-memory.md](lookup-memory.md)). **measured here**, one run each

## Open

Each measure names the sizes it needs. A series answers a question when
the same change is measured at three sizes at least, with the same tokens
per parameter at each.

### Before any comparison

- **The spread between two runs of one recipe.** The same recipe with
  another seed, at the smallest size and at one larger. Every gap below is
  read against it.

### On the model as it is, one change at a time

- **One table against two with another seed**, at 60M, then at one size
  under it and one above.
- **The first layer's values carried to every layer**, at 60M, then at the
  ends of the series if it gains.
- **A learning rate that falls to zero**, at 20 tokens per parameter.
- **The number of experts by size**: 8, 16 and 32 at two sizes, to find
  the tokens an expert needs.
- **A lookup table against layers at an equal total**, at two sizes, to
  find where the table stops paying.
- **A share of masked tokens in the loss**, at 60M, with the copying test.

### The vocabulary by size

- **A sweep of vocabularies at each width**: 4,096 to 65,536 at widths of
  128, 256 and 512, at an equal total of parameters and equal bytes of
  text, measured per byte. It gives the vocabulary a width wants, and the
  floor of width under each vocabulary.

### Other ways to read the text

- **Bytes cut at spaces against tokens at an equal total**, at 30M, 60M
  and 150M: where the advantage of bytes ends.
- **Learned cuts against cuts at spaces**, at the size where the first
  measure favours bytes.

### Other bodies

- **A loop with passes chosen by token** against a plain model and a
  plain loop, at an equal number of parameters and equal compute, at 30M
  and 60M.
- **One depth against a depth that rises with size**, each with its own
  learning rate, over a series.
- **Layers that keep a state in place of layers that read a short
  window**, at 60M and 150M.

### Going larger

- **Whether a choice made at 60M holds at 500M.** The changes above that
  gain are measured again at the largest size before a series keeps them:
  the published gains of a table, of a loop and of an optimizer all
  shrink with size.
- **Width 512 at 20 tokens per parameter**, the first size where the
  tables are a third of the model and not a half.
