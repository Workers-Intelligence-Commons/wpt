# How is a faster training code shown to train the same model?

This file answers one question: what proof a change to the training code
owes when it claims to make a run faster without changing the model the run
trains. Its marks are defined in [README.md](README.md).

## Answer

A change belongs to one of two kinds, and each kind has its own proof.

- **Work removed.** The change runs fewer operations and leaves every
  remaining operation, its inputs and its order as they were: a result
  computed once instead of twice, a block skipped where nothing reads its
  output. The same operations on the same numbers in the same order give
  the same numbers, so the proof is equality bit for bit: on a CPU, the
  weights after some updates are identical with and without the change.
  One differing bit refutes the change.
- **The same mathematics in another order.** The change computes the same
  expression through other operations or in another order: several
  operations fused in one, a sum taken in another grouping. Addition and
  multiplication of floating-point numbers are not associative, so the
  result may differ in its last bits and bit equality cannot be asked for.
  The proof is equality within rounding on one pass: the gradients with and
  without the change agree to the precision of the number format.
- **A difference is read against the spread of the unchanged code.** Where
  two runs of the same code already differ, as on a GPU whose operations
  do not run in a fixed order, a comparison between two codes says nothing
  below the difference between two runs of one code. That spread is
  measured first.
- Equality within rounding on one pass does not bound the difference after
  many updates: training amplifies small differences. Over a whole run, the
  two codes are compared as two runs of one code are, by their loss at the
  same update. **hypothesis**: a change that passes the proof of its kind
  moves the loss of a whole run by no more than a second run of the
  unchanged code does.

## Evidence

- That floating-point addition is not associative, and that the order of a
  sum therefore changes its last bits, is a property of the number format
  (IEEE 754), not a measure.
- That the same operations on the same inputs in the same order give the
  same bits holds on one machine with one build of the libraries, on a CPU
  with a fixed number of threads. It is the condition of the first proof,
  not a general property of hardware.

## Open

- **The bound over a whole run.** Nothing here bounds how far two codes
  equal within rounding on one pass may drift apart over a run. A measure
  would train the same recipe several times with each code and compare the
  spread of the final loss within a code with the gap between the codes.
