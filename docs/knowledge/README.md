# Knowledge

This folder holds what is known about building and training a small
language model, and how sure each piece is. One file per question, each
with the answer the evidence supports and the evidence itself.

A statement belongs here when it holds on any machine and with any code. It
is one of three kinds:

- it follows from definitions or from a derivation: a count of parameters,
  a bound on a rank, a bound on information;
- published sources measure it;
- WPT measures it, and it is a relation about models and data: how many
  training tokens a model of a size needs, below which width a vocabulary
  of a size cannot be told apart.

A statement whose value changes with the GPU or with a commit does not
belong here: a speed in tokens per second, a memory in GiB, a timing, what
the code or a recipe does. Neither does an account of what the project did.

Where there is no knowledge there is no file. A question stays here while
published sources or measures say something about it, whether they close it
or not; its file then says what a measure would have to show to close it.

It serves two readers: someone who writes a recipe or changes the model and
wants the relation to apply, and someone who wants to know what is settled,
what is disputed and what nobody has measured.

The other documents of `docs/` say what WPT is and does. A document that
takes a decision from a question names the file instead of repeating its
reasons.

## The shape of a file

Every file has the same parts, in this order:

1. **The question**, in its title and its first sentence.
2. **Answer**: the statements the evidence supports, each with its mark
   (below). An open question says so here.
3. **Evidence**: what supports the answer and what speaks against it, source
   by source. A published source is named by its title, its year and its
   arXiv id or address. A measure made with WPT names the recipe's numbers,
   how many runs support it, and the machine and the commit it was made
   with. The machine and the commit say where the measure comes from: the
   statement does not depend on them.
4. **Open**: what is not measured, and what a measure would need.

A file gives the numbers a reader needs to judge the answer, and leaves the
rest to the sources.

## Marks

Each statement and each figure carries how far it can be trusted:

| Mark | Meaning |
| --- | --- |
| **proved** | It follows from definitions or from a derivation. No measure is needed. |
| **measured here** | A relation about models and data, measured with WPT's code, with the number of runs that support it. |
| **published, agreed** | Several independent published sources measure the same thing. |
| **published, one source** | One published source measures it. |
| **hypothesis** | WPT's own reading of the evidence. Nothing measures it directly. |

A figure from a published source is marked **unchecked** until it has been
compared with the source's own text: a figure can come from a summary of the
source. A statement never rests on an unchecked figure alone without saying
so.

## The files

- [How wide must a model be for its vocabulary?](width-and-vocabulary.md)
- [How should parameters be split between depth and width?](depth-and-width.md)
- [How large a model for how many training tokens?](size-and-tokens.md)
- [How is a faster training code shown to train the same model?](training-speed.md)
- [At small sizes, are parameters better spent on experts or compute on loops?](experts-and-loops.md)
- [Can the vocabulary's two tables shrink without losing precision on the vocabulary?](vocabulary-tables.md)
- [Can a model learn where to cut its text instead of reading a fixed vocabulary?](learned-chunking.md)
- [Does a small model learn more from other models than from the text alone?](teachers-and-specialists.md)
- [At an equal number of bytes, few precise weights or many coarse ones?](bits-and-parameters.md)
- [Does a parameter do more in a lookup table than in a layer?](lookup-memory.md)
- [With the same tokens, does a small model gain from more, smaller updates?](batch-and-updates.md)
