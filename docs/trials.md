# Trials: how a change is judged

A change to the model or to its training is kept when it is shown to help.
This document says how WPT shows it without training the same model
twice. What is known about the spread between two runs of one order is in
[knowledge/run-to-run-spread.md](knowledge/run-to-run-spread.md).

## A change is tried on a series

A change is tried on a model series ([model.md](model.md), "Model
series"): the series as it is, and the series with the change, each size
trained once. No size is trained again with another seed to confirm a
result. The sizes confirm each other:

- a change that helps lowers the final validation loss at every size, by a
  gap that grows or shrinks steadily with the size;
- a gap that changes sign from one size to the next, or shows at one size
  only, is not a result.

The series gives its own measure of the noise. The final losses of one
series lie on a smooth curve against the size; how far the points stand
from that curve is what one run moves by chance. A gap between the two
series smaller than that is not shown.

The series also says where the change stops paying: a gap that shrinks as
the model grows is a change for small models, and the largest size trained
says how far it holds.

## What makes two runs comparable

- **The same tokens.** Both runs read the same corpus and the same prefix
  of it, with the same stages.
- **The same seed.** Both then draw the same training windows.
- **The same validation windows.** A stage's validation windows depend only
  on the training tokens and the stage ([training.md](training.md),
  "Stages"), so two runs of one size are measured on the same windows.
  The losses of two sizes of a series are measured on different windows,
  since each size reads its own prefix: they place a point on the curve and
  are not compared one to one.
- **The end of training.** Two runs are compared at their last update. The
  order of two models at an earlier update says nothing about their order
  at the end.

## What a result states

A result gives, for each size, the final validation loss of both runs and,
beside it, what differs besides the change: the parameters, the compute for
one token and the tokens read. A change that lowers the loss with more
compute for one token is stated as that trade, not as a gain.

The tests of `wpt eval` ([evaluation.md](evaluation.md)) are given with the
result and do not rank two models: a count of right answers moves more
between two runs of one order than the loss does.

A result that holds for language models in general is written in
[knowledge/](knowledge/README.md), in the file of its question.
