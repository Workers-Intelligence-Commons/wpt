# Evaluation

`wpt eval` runs tests on a trained model and records their results in its
model card.

```sh
uv run wpt eval runs/default                      # every built-in test
uv run wpt eval runs/default path/to/my_test.py   # any Python file
```

It reads the model folder that `wpt train` wrote, and the corpus the model
was trained on from the dock named by `DOCK`, the one its
`provenance.json` names. Each result goes to `<model folder>/evals/<name>.json`
and the model card (`README.md`) is written again to show it. Training a new
model into the same folder removes the old results: they describe other
weights.

## The model on a gpu worker

The tests run on the machine that runs `wpt eval`, which holds the
tokenizer, the dock and the sandboxes. The model runs on a gpu worker
(the first key of `WPT_GPU_WORKERS`, else a local worker), which the
command sends the checkout and the folder's weight files; what a test asks
of the model goes through the link, as token ids, and the losses come back
as numbers
([machines.md](machines.md), "Evaluation").

`--kernels FORM` (or `WPT_KERNELS`) names what computes the model's
attention on the worker: `tilelang`, `triton` or `pytorch`, as for
`wpt train` ([model.md](model.md), "Attention"). A name that is none of
these is refused. The worker decides the form before it loads the model,
and a form that cannot work there refuses the evaluation before any test
runs. Unnamed, the worker chooses by what is installed for its GPU.

## Writing a test

A test is one Python file that defines `run`:

```python
from wpt.eval import Result

def run(context):
    losses = context.model.losses([[1, 2, 3, 4]])
    return Result({"windows": len(losses)}, f"{len(losses)} windows measured")
```

`context` gives:

- `model`: the model, which runs on a gpu worker (below): `model.config`,
  `model.generator(seed)`, `model.writer(prompt, rows, temperature=...,
  top_k=..., stops=...)`, `model.greedy(prompts, steps)` and
  `model.losses(windows)`, the next-token loss in nats of each token of each
  window after its first, read with only the window's earlier tokens before
  it; all in token ids, and it holds no tensors on this machine;
- `device`: where the model runs, in words;
- `tokenizer`, `folder`, and `provenance` (the folder's
  `provenance.json`);
- `dock`: the URL of the dock holding the corpus, read through `dock.api`
  ([api/dock.md](api/dock.md));
- `telemetry`: the run's record, for the test's own events
  ([observability.md](observability.md)).

Every test returns the same type, `Result`: `data`, numbers and names for
programs (anything JSON can hold), and `text`, the same findings written for
people, which the model card shows. A test is plain code, not an order file,
so it can measure whatever the question needs. The result file also records
the test file's sha256, so a result names the exact code that produced it.

Built-in tests live in `apps/wpt/src/wpt/eval/evals/`; a new file there
is a new built-in test, and a run that names none runs them all:

| Test | What it measures |
| --- | --- |
| `general_knowledge` | What everyone knows: the questions of `general_knowledge.json`, sentences to complete with one answer that leaves no doubt ("The capital of Italy is" Rome), each in French and in English with three wrong answers of the same kind. For each language and topic, how often the right answer costs the model the fewest nats of the four (chance is 25%), and how often the model writes it. |
| `copying` | Whether the model reuses what its context says, on prompts made from a seed, whose answer is in the prompt alone. Sequences of 8 whole-word tokens written twice then half a third time: the share of the second half the model writes, and the cost in nats of a word the first and the third time it is read. A value, a word or a number of 4 digits, stated in a sentence and asked for in the next, in French and in English: how often it costs the fewest nats of four (chance is 25%), and how often the model writes it. |
