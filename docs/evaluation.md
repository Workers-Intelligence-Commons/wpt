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
of the model goes through the link, as token ids, and the losses and ranks
come back as numbers
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
  top_k=..., stops=...)`, `model.greedy(prompts, steps)`,
  `model.losses(windows)`, the next-token loss in nats of each token of each
  window after its first, read with only the window's earlier tokens before
  it, and `model.ranks(windows)`, the rank of each of those tokens among
  the model's scores at its place (1: no token scores higher); all in token
  ids, and it holds no tensors on this machine;
- `device`: where the model runs, in words;
- `tokenizer`, `folder`, and `provenance` (the folder's
  `provenance.json`);
- `dock`: the URL of the dock holding the corpus, read through `dock.api`
  ([api/dock.md](api/dock.md));
- `telemetry`: the run's record, for the test's own events
  ([observability.md](observability.md)).

Every test returns the same type, `Result`: `data`, numbers and names for
programs (anything JSON can hold), and `text`, the same findings written for
people, which the model card shows. A test that cannot measure this model
raises `NotMeasured(reason)` (from `wpt.eval`), the reason about the model,
never naming a path of this machine. In a run of every built-in test (no
test named), its result is `{"measured": false, "reason": ...}` with the
text "Not measured: " and the reason, so the model card shows no figure
for it, and the run goes on; a test named refuses the model as any
failure does, and the run stops. Any other exception a test raises stops
the run, and the results already written stay.

A test is plain code, not an order file, so it can measure whatever the
question needs. The result file also records the test file's sha256, so a
result names the exact code that produced it.

Built-in tests live in `apps/wpt/src/wpt/eval/evals/`; a new file there
is a new built-in test, and a run that names none runs them all:

| Test | What it measures |
| --- | --- |
| `general_knowledge` | What everyone knows: the questions of `general_knowledge.json`, sentences to complete with one answer that leaves no doubt ("The capital of Italy is" Rome), each in French and in English with three wrong answers of the same kind. For each language and topic, how often the right answer costs the model the fewest nats of the four (chance is 25%), and how often the model writes it. Then, for the right answer's first token right after the sentence: the median of its rank among the model's scores (1: the likeliest), how often it is the likeliest (top 1) or among the 10 likeliest (top 10), and its mean probability. These figures move before the model writes the answer, so a small model that does not know it yet still shows how close it comes. |
| `bits_per_byte` | How well the model predicts texts no training read: the held-out portion of its corpus ([data-selection.md](data-selection.md), "The portions of a corpus"), read from the corpus delivery its `provenance.json` names. Its texts are taken each once, in the order of their document ids, as many as fit in `MAX_BYTES` (twice `HELD_OUT_BYTES`, the size the portion is filled to, so the whole portion), and read as one stream, END first, then each text's tokens and the END that closes it. The stream is cut into windows of the model's context, its `block_size` B: B + 1 tokens from token 0, B, 2B, and so on, each window starting on the last token of the one before it, so every token but the first END has its loss counted once, with the window's earlier tokens before it. Bits per byte is the nats of those tokens over ln 2 and over the bytes they decode to, each token decoded alone: the END that closes a text counts its loss and no byte, and the first END, which has no loss, stands for no byte either, so the bytes counted are every byte of the texts taken, the same for two vocabularies, which makes two models of different vocabularies trained on corpora of one snapshot and one set of shares comparable. It gives bits per byte, with its error, the standard error of a ratio of two sums, from the spread between windows (sqrt(sum((bits_i - r bytes_i)^2) / (n (n - 1))) over the mean bytes of a window, r the bits per byte, n the windows; none under two windows), the spread itself (the standard deviation of each window's bits over its bytes), the nats per token, the texts, tokens, bytes and windows measured, and bits per byte for each share's tag. A model whose corpus has no held-out portion (a corpus the depot built under `corpora/`, a delivery without portions, an order whose roles leave it out) is not measured: in a run of every built-in test, its result says so and why, with no figure, and the other tests run; named, the test refuses it and the run stops. A delivery the dock no longer holds or an erasure reached is refused either way, and the refusal stops the run. |
| `copying` | Whether the model reuses what its context says, on prompts made from a seed, whose answer is in the prompt alone. Sequences of 8 whole-word tokens written twice then half a third time: the share of the second half the model writes, and the cost in nats of a word the first and the third time it is read. A value, a word or a number of 4 digits, stated in a sentence and asked for in the next, in French and in English. The sentence ends the value with a period, which the tokenizer may join to its last letters or digits, so the value is judged as text, never by its tokens: how often the value followed by the period costs the fewest nats of four values followed by it (chance is 25%); how often the model writes it, the text it writes greedily, leading spaces aside, starting with the value and going on with neither a letter nor a digit; and, where the answer starts, the median rank and the mean probability of the first token of the value as the prompt cuts it. |
