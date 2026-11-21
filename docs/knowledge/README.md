# Knowledge

This folder holds what the project knows about sizing a model and writing a
recipe: one file per question, each with the answer the evidence supports
and the evidence itself, from published sources and from measures made with
WPT. It serves two readers: someone who writes a recipe and wants the rule
to apply, and someone who wants to know what is settled, what is disputed
and what nobody has measured.

The other documents of `docs/` say what WPT is and does. A file here says
why a number of a recipe should have one value and not another, and how
sure that is. A document that takes a decision from a question names the
file instead of repeating its reasons.

A question stays here whether it is answered or open. An open question says
what a measure would have to show to close it, so that it can be taken up
as an experiment.

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
