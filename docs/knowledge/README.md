# Knowledge

This folder holds the reasons behind WPT's choices and how sure each one
is: why the model has the parts it has, why a recipe is written the way it
is, why a number has one value and not another. One file per question, each
with the answer the evidence supports and the evidence itself, from
published sources and from measures made with WPT.

A question belongs here when its answer decides something in the model, in
training or in a recipe. Among them:

- the architecture: why a part is in the model, what it costs, and at which
  sizes it pays;
- the shape of a model: its width, its depth, its vocabulary, its experts,
  its loops;
- training: how many tokens, in which windows, with which settings;
- what a trained model can do, and how that is measured.

It serves two readers: someone who writes a recipe or changes the model and
wants the rule to apply, and someone who wants to know what is settled,
what is disputed and what nobody has measured.

The other documents of `docs/` say what WPT is and does. A file here says
why, and how sure that is. A document that takes a decision from a question
names the file instead of repeating its reasons.

A question stays here whether it is answered or open. An open question says
what a measure would have to show to close it, so that it can be taken up
as an experiment. A question nobody has studied yet is worth a file as soon
as it is asked: the file then holds the question, the working assumption
and the measure that would answer it.

## The shape of a file

Every file has the same parts, in this order:

1. **The question**, in its title and its first sentence.
2. **Answer**: the rule to apply now, each rule with its mark (below). An
   open question says so here and gives the working assumption.
3. **Evidence**: what supports the answer and what speaks against it, source
   by source. A published source is named by its title, its year and its
   arXiv id or address. A measure made with WPT names the recipe's numbers,
   the machine, the options of the command and the commit.
4. **Open**: what is not measured, and what a measure would need.

A file gives the numbers a reader needs to judge the answer, and leaves the
rest to the sources.

## Marks

Each rule and each figure carries how far it can be trusted:

| Mark | Meaning |
| --- | --- |
| **measured here** | Measured with WPT's code, under the conditions the file names. It holds for those conditions. |
| **published, agreed** | Several independent published sources measure the same thing. |
| **published, one source** | One published source measures it. |
| **hypothesis** | WPT's own reading of the evidence. Nothing measures it directly. |

A figure from a published source is marked **unchecked** until it has been
compared with the source's own text: a figure can come from a summary of the
source. A rule never rests on an unchecked figure alone without saying so.

## The files

- [How wide must a model be for its vocabulary?](width-and-vocabulary.md)
- [How should parameters be split between depth and width?](depth-and-width.md)
- [How large a model for how many training tokens?](size-and-tokens.md)
- [What does a model's shape cost in training speed?](shape-and-speed.md)
- [What makes a training run faster without changing the model it trains?](training-speed.md)
- [At small sizes, are parameters better spent on experts or compute on loops?](experts-and-loops.md)
- [Can the vocabulary's two tables shrink without losing precision on the vocabulary?](vocabulary-tables.md)
- [Does a small model learn more from other models than from the text alone?](teachers-and-specialists.md)
- [At an equal number of bytes, few precise weights or many coarse ones?](bits-and-parameters.md)
- [Does a parameter do more in a lookup table than in a layer?](lookup-memory.md)
- [With the same tokens, does a small model gain from more, smaller updates?](batch-and-updates.md)
