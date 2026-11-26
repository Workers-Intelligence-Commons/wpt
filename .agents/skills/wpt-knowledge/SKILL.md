---
name: wpt-knowledge
description: How to use and extend docs/knowledge, the folder of what is known about building and training language models, transformers above all. Use it before choosing a model's shape, an order, a training setting or the next trial, to read what is already known; after a trial or a reading of papers, to write down what it showed; and whenever asked what is settled, what is open, or what to try at a given model size.
---

# The knowledge folder

`docs/knowledge/` holds what is known about language models and how sure
each piece is: statements proved, published, or measured, one file per
question. It is about language models in general, transformers above all,
not about WPT's code. Its [README](../../../docs/knowledge/README.md)
defines what belongs there, the shape of a file and the marks; read it
before writing.

## Reading it

- Before a choice about a model or its training, read the file of that
  question, and
  [choices-by-size.md](../../../docs/knowledge/choices-by-size.md), which
  gathers what every file says about size and lists the measures still to
  make, with the sizes each needs.
- A statement is as sure as its mark. A **hypothesis** is a reading, not a
  result. A figure marked **unchecked** came through a summary of its
  source: compare it with the source before a decision rests on it alone.
- A measure made once says less than its number suggests:
  [run-to-run-spread.md](../../../docs/knowledge/run-to-run-spread.md)
  says how far two runs of one order differ and which gaps one run can
  show.

## What goes in

A statement goes in when it would be true for someone else's language
model on someone else's machine:

- it follows from definitions or a derivation;
- published sources measure it;
- a measure made here shows a relation about models and data.

A statement names the architecture it was measured on when it may not hold
on another: "with routed experts", "with a lookup table read by the last
tokens". A statement that holds only for one implementation stays out.

What stays out: a speed, a memory use, a timing, anything whose value
changes with the GPU or the commit; what the code or an order does today;
an account of what the project did or in what order. Where there is no
knowledge there is no file.

## Writing in it

- **One question, one file.** Add to the file whose question the statement
  answers. Make a file only for a question no file asks, and add it to the
  README's list.
- **After a trial**, write its result in the Answer of its question, with
  its mark and the number of runs, the table of what was compared under
  "Measured with WPT", and what the trial does not separate. Remove from
  Open the measure it made, and add the measure it calls for.
- **When a measure refutes a hypothesis**, replace the hypothesis. The
  file says what is known, not what was believed.
- **After reading papers**, name each source by title, year and arXiv id
  or address, give the numbers a reader needs to judge it, and say what it
  was not measured on: the sizes, the amount of text. Mark every figure
  not compared with the source's own text **unchecked**. Put the sources
  that speak against the answer beside those that speak for it.
- **A relation with size** goes in the file of its question and, in one
  line, in `choices-by-size.md`.
- **An idea for a trial** is written as an open measure: what it would
  train, against what, at which sizes, and what result would close the
  question.
- The machine and the commit of a measure made here are named once, to say
  where it comes from. No statement depends on them.
- Write as the rest of the repository does: the current state only, no
  dates, no history, plain words (CONTRIBUTING.md, "Writing about the
  project").

## Before a comparison is written as a result

- Compare two models at the end of training, on the same tokens, never at
  an early update.
- Read the gap in validation loss against the spread between two runs of
  one order. A gap in a behavioural test, such as copying from the
  context, needs more than one run of each model before it ranks them.
- Say what differs between the two models besides the thing tested: the
  number of parameters, the tokens per parameter, the compute per token.
