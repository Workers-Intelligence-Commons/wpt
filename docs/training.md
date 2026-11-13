# Training and repeatability

## Training

```sh
uv run wpt depot build DATASET            # the corpus of the recipe's dataset; prints its id
uv run wpt train classics-test
```

A run takes minutes to days, so start it detached from the terminal or SSH
session, which would otherwise end it and lose its output:
`.\scripts\detach.ps1 uv run wpt train ...` on Windows,
`scripts/detach.sh uv run wpt train ...` on Linux and macOS. Both write its
output to `runs/logs/<date-time>.log` and print how to follow and stop it.

`wpt train` takes a model recipe (`data/recipes/models/<name>.json`,
`SERIES:SIZE`, a size of a model series, or `SERIES:SIZE:RUN`, another run
of that size; see [model.md](model.md), "Model recipes") and the id of a
corpus in the store, or a start of it, and never builds: `wpt depot build`
builds the corpus of the recipe's dataset, on the machine that holds the
depot. Without an id, it reads the corpus last built for the recipe's depot
and dataset, and with none it is refused. To find that corpus, it opens the
depot to read the dataset's id and current revision, so without an id
(`WPT_CORPUS` or the corpus option) the depot must be readable where
training runs. A corpus of another depot,
dataset or tokenizer recipe than the recipe names is refused. Everything that changes the
resulting model comes from the recipe: the architecture's numbers, the
seed, the steps ("Steps", below) with their settings, and the drafter. The
command-line options say only how the hardware runs it; each also comes
from an environment variable (`WPT_` + the option name in capitals,
`--micro-batch` -> `WPT_MICRO_BATCH`; the recipe and the corpus id from
`WPT_RECIPE` and `WPT_CORPUS`). `uv run wpt train --help` lists them.

The corpus holds everything the steps read: the token files of its plain
texts, and its conversations and tasks. Its record gives a fingerprint (a
sha256) for each of these parts, `text`, `conversations` and `tasks`,
which the steps' fingerprints read ("Step results and reuse", below). A
text added to the depot after the build reaches training only once the
corpus is built again.

`wpt train` trains through gpu workers, one per machine
([machines.md](machines.md)): each worker's job runs the steps on its GPUs, and `wpt train` answers it from
the store, the tokenizer and the run's record. The job's main process
records the run in the store, through that link: what it read, every step
and every update's loss, duration and learning rates, every evaluation and
checkpoint, every step result kept or reused, and every file it wrote
([observability.md](observability.md)). A record the store refuses stops
the run with exit code 3, which is never restarted. When the training
process's record breaks (a record call refused, or on S3 a group of its
lines refused in the background), it finishes its update unrecorded and
every process stops after it, as for Ctrl+C: the checkpoint is saved, and
`--resume` continues from it without losing an update. The broken record
does not name that checkpoint (nor, with `--run`, the store's record of it). When
`wpt train`'s own record breaks while the job trains (on S3, a group of
its lines refused in the background), it stops the job as Ctrl+C does: its
processes finish their update and save the checkpoint, and `wpt train`
exits with code 3.

## Steps

A recipe's `steps` list says what is done to the model, in order, on one
model: each step starts from the weights the one before it left. The recipe
has one `seed`; each step mixes its number in the list into it, so two
identical steps in one recipe draw different windows, conversations and
tasks. A step that continues the one before it draws as that step does
("Continued steps").

```json
"seed": 1337,
"steps": [
  {"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.002}},
   "settings": {"stages": [{"block_size": 4096, "steps": 2000, "batch_size": 256}], "eval_every": 200,
                "eval_batches": 20},
   "locked": []},
  {"step": "answer-as-assistant"},
  {"step": "finish-tasks", "settings": {"rounds": 4}}
]
```

A step is made of four roles, each a Python file of
`packages/wpt-training/src/wpt_training/`:

- the **method** (`methods/`) makes a loss from a batch of examples;
- the **environment** (`environments/`) gives the examples, batch by
  batch, and says when to evaluate and when the step has failed;
- the **optimizer** (`optimizers/`) moves the weights the loss reaches,
  with its schedule, and judges the evaluations;
- the **step** file (`steps/`) names its method, environment and
  optimizer with that optimizer's settings, the parts of the model it
  locks, and its own defaults for its method's and environment's settings.

The **runner** (`wpt_training.train`) is the training loop every step
shares: it freezes the locked parts, starts the optimizer, asks the
environment for each update's batch, lets the method learn from it,
updates, evaluates, writes checkpoints, and stops on a signal.

An entry is `{"step": NAME}` with, optionally, `optimizer` (`{"name",
"settings"}`), `settings`, `locked` and `continues` ("Continued steps");
each part left out takes the step file's own, and `continues` is false. `NAME` is a built-in step or the path of a Python file. The
settings are the defaults of the step's method and environment, then the
step file's, then the entry's, each of its default's type (an integer
where a number is expected is taken, text is converted;
`wpt_utils.settings.typed`). A setting belongs to the method or to the
environment by its name, never to both. The optimizer the entry names
takes the step file's settings for it when it is the step's own, else its
own defaults, then the entry's. Every step file is read and every setting
checked before any step runs: an unknown step or setting, a value of
another type, settings that cannot go together, a method that reads
another kind of examples than its environment gives, and an unknown part
to lock all refuse the run (exit code 2), naming the file. So do drafter
settings that cannot be. Once the corpus is read, and still before any
step runs, a step whose environment finds nothing to learn from in it
(no conversation, no task, or fewer tokens than its prefix) refuses the run the same way,
naming the step. `provenance.json` records each step: its files and their
sha256, its method, environment, optimizer and their settings, the kind
of examples it reads (`reads`), and its locked parts.

The built-in steps:

| step | method | environment | optimizer | locked |
| --- | --- | --- | --- | --- |
| `read-text` | `next-token` | `text-windows` | `by-part`, its defaults | none |
| `answer-as-assistant` | `assistant-tokens` | `conversations` | `by-part`, lr 1e-4, no weight decay, constant schedule, early stop on (no evaluations, so it never fires), no average | `balancing` |
| `finish-tasks` | `grpo` | `agent-tasks` | `by-part`, lr 1e-6, no weight decay, constant schedule, early stop on (no evaluations, so it never fires), no average | `routers`, `balancing` |
| `finish-tasks-with-less-writing` | `grpo` | `agent-tasks` | `by-part`, lr 1e-6, no weight decay, constant schedule, early stop on (no evaluations, so it never fires), no average | `routers`, `balancing` |

A recipe whose steps are `read-text` alone makes a base model. The model's
context (`config.json`'s `block_size`) is the last window length of the
last step that reads text windows; a recipe with no such step is refused.

### Locked parts

`locked` names parts of the model by the names the model gives them
([model.md](model.md), "Parts"); a name the model does not give is refused,
naming the parts there are. A locked part's weights do not change during
the step: they are left out of its optimizer and of the processes'
gradient exchange. `balancing` is the routers' balancing biases, which no
gradient trains: locked, they stay as they are after each update. A
method also names the parts its loss never reaches (`Method.unreached`:
the multi-token prediction block for `assistant-tokens` and `grpo`), which
are left out the same way. A step whose locked and unreached parts leave
nothing to train is refused.

### Step results and reuse

After each step, the store keeps its result as
`cache/steps/<fingerprint>`: a JSON record holding what the run recorded of the
step and its state's file (`{"key", "sha256"}`), the weights (and those of its best
evaluation) with the optimizer's, method's and environment's state: the
step's last checkpoint, as it is. The fingerprint covers the fingerprint
of the step before it (for the first step, the model's shape and
vocabulary size, without the context `block_size`, which no weight
depends on: a recipe that adds a later step reading longer windows still
reuses the first), the step as `provenance.json` records it, the
fingerprint of the corpus part it reads (the record's `fingerprints`: `text`
for `read-text`, `conversations` for `answer-as-assistant`, `tasks` for
`finish-tasks` and `finish-tasks-with-less-writing`), the seed and the
code (the commit it runs from and a hash of its uncommitted changes;
[architecture.md](architecture.md), "The store"), so any
commit makes every kept result unknown. A change to the
conversations leaves the fingerprint of `read-text` as it was, and so its kept result stands. The
options, which only say how the hardware runs it, stay out, `--stream` and
`--stream-shards` included.

A run whose first steps have a kept result starts from the result of the
last of them and runs the steps after it. So a recipe that shares its
first steps with an earlier run reuses them, and a run that failed after
some steps continues from the first step not kept. The main process reads
the result and sends its weights to every process. Reuse means the same
inputs, not the same bits ("What is exactly repeatable", below). Each step
starts its own optimizer, except a step that continues the one before it. `provenance.json` says, for each step, its
fingerprint, its result and whether this run reused it.

A step that cannot do its part, such as one whose loss is not finite or
whose attempts Docker drops every one, fails (`wpt_training.StepFailed`):
the run stops with exit code 2 and the reason, the step keeps no result,
and the results of the steps before it stay in the store. A checkpoint
the step wrote before it failed stays, and the message names it: the
same command with `--resume` continues the step from it, and deleting it
starts the step over (a run with other settings for that step needs it
deleted, "Checkpoints and resuming", below).

### Continued steps

A step with `continues: true` goes on from the whole state the step before
it left, as one step that stopped and was resumed: its weights, the
optimizer's, method's and environment's state, and its count of updates.
Its first update is the one after the last update of the step before it,
and its schedules (the learning rate, the weight of multi-token
prediction) count its own updates in all. Its draws read the number of the
step it continues, so the two draw the same windows as one step would.
When the step before it is reused, it goes on from that step's kept
state.

It has the same step, method, environment, optimizer and locked parts as
the step before it, and, both listing stages, more updates; a recipe that
breaks this is refused before any step runs. It forgets the best
evaluation of the steps before it, as a new stage does: it keeps the
weights of its own best evaluation. Its fingerprint chains to the step
before it as every step's does, so two recipes that share a step's whole
chain share its kept result. The decay branches of a model series are
built this way ([model.md](model.md), "Model series").

### Methods

A method file defines `READS`, the kind of examples it learns from,
`SETTINGS` and `start(settings)`, which returns a `wpt_training.Method`.
Its `learn` runs one pass's forward and backward passes; the runner adds
the passes of an update up, and `update_loss` gives the update's loss,
the same in every process. `begin` may refuse options the
method cannot train with, and `unreached` names the parts its loss never
reaches.

- `next-token` (reads `windows`): the next-token loss of every position,
  plus the multi-token prediction loss at `mtp_weight`, then at
  `mtp_late_weight` after `mtp_switch` of the step's updates (DeepSeek-V3:
  after 10T of 14.8T tokens). Evaluations compare the next-token loss
  alone. Settings: `mtp_weight` 0.3, `mtp_late_weight`
  0.1, `mtp_switch` 0.676.
- `assistant-tokens` (reads `conversations`): the next-token loss of the
  tokens the assistant wrote only ("The answer-as-assistant step", below).
- `grpo` (reads `attempts`): group relative policy optimization on scored
  attempts ("The finish-tasks step", below).

A method on text windows is a `wpt_training.TextMethod`: its
`loss(model, x, y)` returns the training loss (`model(x, next_tokens=y)`
returns the scores and the multi-token prediction scores). With
`targets=(y, after_next)` in that call, the model returns the two
cross-entropies summed over the tokens instead of the scores, 4,096 tokens
at a time: the built-in loss never holds every token's scores at once. On
a GPU where Triton imports, a pass with gradients runs the fused head and
loss on each 4,096 tokens, which computes the gradients in its forward
pass, the losses and the scores' gradients in one kernel. Each chunk's
head gradient is then one BF16 product, as autocast gives any weight's,
and the chunks' add up in FP32. Elsewhere
each chunk's scores are computed again in the backward pass.
A method that defines its own `loss` defines `part_loss(model, x, y,
following, window)` too, the share of a part of the windows (`following`:
the token after `y`'s last, or None at the windows' end; `window`: the
windows' length) in what `loss` gives over whole windows, or it trains
whole windows only (the run refuses `--segment` and `split` with it).
`--strategy split` is refused for a recipe with a step that reads
anything else than text windows.

### Environments

An environment file defines `GIVES`, the kind of examples it gives,
`SETTINGS`, `start(settings, data)`, which returns a
`wpt_training.Environment`, and optionally `check(settings)`, which raises
`ValueError` on settings that cannot go together, `context(settings)`,
the window length the step leaves the model reading at, and
`check_corpus(settings, record, store)`, which raises `ValueError` when
the corpus holds nothing for the step, before any step runs. `data` gives
the step's name, the store, the corpus's record, the tokenizer, the
model's configuration, the strategy, the run's record, the log, the seed
and the step's number in the recipe. The
environment gives its stages (one for most environments; text windows
have one per window length), the tokens of its examples, each update's
passes (or none, when the update has nothing to learn: it is then
skipped), when to evaluate, what happens after each update and at the
step's end, the state a resumed step needs, and what of the corpus it
learned (`learned`: the tokens of plain text, the conversations or the
tasks), from which the model's `CREDITS` is written.

- `text-windows` gives `windows` of the corpus's plain texts ("The read-text
  step", below).
- `conversations` gives the corpus's conversations ("The
  answer-as-assistant step", below).
- `agent-tasks` gives the model's own attempts at the corpus's tasks,
  scored ("The finish-tasks step", below).

### Optimizers

An optimizer file defines `SETTINGS` and `start(model, settings, run)`,
which returns a `wpt_training.Optimizer`: its torch optimizer over the
parameters the step trains, and optionally what happens just before each
update (a learning-rate schedule, gradient clipping) and just after it, at
the start of each stage after the first (`stage`), which evaluation is the
best and when to stop, the weights the step keeps in place of those
(`kept`), and the state a resumed step needs. `run` gives the
tokens of the step's examples, the parameters it trains, its number of
updates and the log. The file may also define `check(settings)`, which
raises `ValueError` for settings that cannot go together, so the run is
refused before any step trains. The step keeps the weights its optimizer
gives once the updates are done; without them, the weights of the
evaluation its optimizer judged best in its last stage, or its last
weights when it has no evaluation.

The built-in optimizer is `by-part`: each kind of weight under its own
update rule, the rule the model names for each of its parameters (model.md,
"Training"). It needs a model that names its parameters' roles.

- NorMuon for matrix weights: DeepSeek-V4's 8 then 2 Newton-Schulz steps,
  each output neuron's row normalized by a running mean of its squares,
  updates of RMS 0.18. It runs per head for the attention's query
  projections and output gates and per expert for routed experts; no
  QK-Clip (every attention layer normalizes its queries and keys).
- AdamW for the embedding, the output head, norms, the queries of
  Attention Residuals and attention sinks.
- Adam at `table_lr_factor` times the rate, without weight decay, for
  Engram's tables.

Weight decay is cautious: an entry of a matrix or vector decays only where
its update and its weight have the same sign.

The whole update runs as PyTorch's foreach operations, one call over every
tensor of a group, and the matrices of the same rows and columns go
through the Newton-Schulz steps in one batch. No value comes back from the
GPU to the host during an update. The update is the same on every device
within rounding, and the optimizer's state is the same on every device, so
a run resumes on another.

`schedule` chooses how the learning rate moves. With `steps`, it warms up
over `warmup` updates to `lr`, then falls to `first_factor` of it after
`first_drop` of the step's updates and to `second_factor` after
`second_drop`. With `constant`, every update uses `lr`.

With `average_states` above 0, the step keeps an average of the model's
last states (IMU-1). The states are the model's whole state at the step's
last update and every `average_every` updates before it, at most
`average_states` of them; a step with fewer updates takes the states it
has. The average starts from the oldest state and takes in each later one
with weight 1 - `average_beta`, an exponential moving average: with 10
states and 0.8, the newest weighs 0.2 and the oldest 0.8^9. It is held in
FP32 on the model's device from the oldest state on, one copy of the
weights, and saved with the optimizer's state, so a resumed step goes on
with it. With the average, no evaluation is the best. A continued step
whose oldest state comes before its first update has no whole average and
keeps its last weights. The average and early stop cannot go together,
since the states count back from the step's last update: a step with both
is refused.

Early stop and the best evaluation need evaluations, and only an
environment that makes them gives them: among the built-in ones,
`text-windows`. Under `conversations` and `agent-tasks`, a step runs
every update and keeps its last weights, whatever `patience` says.

Early stop is off while `patience` is 0. Then the best evaluation is the
lowest validation loss. With a `patience` above 0, the validation loss is
smoothed over the last `smoothing` evaluations, and the best evaluation
is the last one that improved it by `min_improve`. A stage stops early
once `patience` evaluations in a row have not improved it. The smaller
the tokens-per-parameter ratio, the smaller the improvement it asks for:
`chinchilla` is the tokens per parameter at which `min_improve` applies in
full.

Settings and defaults: `lr` 0.002, `weight_decay` 0.1, `momentum` 0.95,
`beta1` 0.9, `beta2` 0.95, `eps` 1e-8, `table_lr_factor` 5, `schedule`
`steps`, `warmup` 2000, `first_drop` 0.8, `first_factor` 0.316,
`second_drop` 0.9, `second_factor` 0.1, `average_states` 10,
`average_every` 500, `average_beta` 0.8, `patience` 0, `min_improve` 0.01,
`smoothing` 3, `chinchilla` 20.

### Writing a step

A step file defines `METHOD`, `ENVIRONMENT` and `OPTIMIZER` (`{"name":
..., "settings": {...}}`), and optionally `LOCKED` and `SETTINGS`, its
defaults for its method's and environment's settings, and
`check(settings)`. It names built-in roles or paths of Python files, so a
new kind of training is a new method, environment or optimizer file and
a step file that puts them together, without changes to the runner.

## The read-text step

The `read-text` step reads the corpus's plain texts: next-token prediction
(`next-token`) on windows of the training shards (`text-windows`), the
windows growing stage by stage, under `by-part` on its multi-step
schedule, without early stop. Its settings are `stages`,
`eval_every` (200), `eval_batches` (20) and `prefix_tokens` (0, "Prefix"
below), with the method's. Each update
draws its windows at random places of the training tokens, seeded by the
seed, the step's number in the recipe and the update only: a second
`read-text` step draws other windows than the first. Dropout reads the
same three.

Validation draws windows of the training tokens apart from the
training's. The tokens are read from disk as training needs them, so a
corpus larger than memory trains:
before the step's first update, the shards are checked against their
token hashes and their tokens written once into one file per part, in the
store's `scratch/` folder (`<store folder>/scratch/` for a local store,
the cache for S3). A later run reuses that file when it still matches the
sha256 recorded when it was written. `scratch/` can be deleted at any time.

### Prefix

`prefix_tokens`, when above 0, makes the step's training tokens the first
`prefix_tokens` of the corpus's, in the shards' order, and nothing after
them. Only the shards that hold them are copied (or streamed), the windows
are drawn from them alone, and the step learned from the texts of that
start. With `--stream`, the walk over the shards starts again after the
prefix. A corpus with fewer tokens than the prefix is refused before any
step runs. A `read-text` step that sets no `prefix_tokens` trains on the
first `total_tokens` of the recipe's data part, the same prefix; and
`wpt train` refuses a corpus that holds fewer tokens than `total_tokens`,
and a corpus that is stale (an admitted erasure reached a text it was built
from, [erasure.md](erasure.md), "What it covers") until its dataset is built
again. Each size of a model series reads its prefix of the series' one
corpus (model.md, "Model series"), and the setting is part of the step's
fingerprint.

### Streaming: a corpus larger than the disk

`--stream` (or `WPT_STREAM=1`) never copies every shard to disk. The
shards are walked in order at the speed training uses tokens: before each
update, the shard holding the next unused token and the shards after it,
`--stream-shards` in all (8 by default, about 2 GB), are the ones in use,
and the update draws its windows from them only. The shard after them is
prepared while training goes on, and a shard that leaves the set loses
its token file and, from an S3 store, its local copy. After the last shard
comes the first again.

Which shards are in use depends only on the update, so a resumed run and
any number of processes take the same windows, but they are not the
windows a run without `--stream` takes: a streamed run mixes texts within
the shards in use, not over the whole corpus. Both options only say
how the hardware reads the shards, so they are no part of the step's
fingerprint: a run reuses a step another run kept with other values of
them.

### The batch is the same on any hardware

A step's batch is the number of examples in one update, over every process
together: for text windows, each stage's `batch_size` ("Stages", below).
Each process takes `--micro-batch` examples per pass (by default its whole
share of the batch) and adds the passes up until the batch is complete.
Which windows an update uses depends only on the recipe's `seed`, the
step's number in the recipe and the update, never on the number of
processes: one GPU with two passes and two
GPUs with one pass each make the same updates. When a GPU runs out of
memory, lower `--micro-batch`, then add `--recompute`; the training stays
the same. Every stage's batch must split evenly: its `batch_size` a
multiple of `--micro-batch` times the number of processes (with
`--strategy split`, every process takes every window, so a multiple of
`--micro-batch`).

### Stages

Training targets 1M tokens and reaches it by stages of the `read-text`
step; the window lengths follow DeepSeek-V4 (section 4.2.2: 4K, then 16K,
64K and 1M tokens); V4 states no tokens or updates per stage, so each
recipe sets them. The step's `stages` setting lists them in order, each
with its window length, its length in updates and its batch:

```json
"stages": [
  {"block_size": 4096, "steps": 2000, "batch_size": 256},
  {"block_size": 16384, "steps": 500, "batch_size": 64},
  {"block_size": 65536, "steps": 200, "batch_size": 16},
  {"block_size": 1048576, "steps": 50, "batch_size": 1}
]
```

(an example of the shape, not a recipe's numbers). The stages are one
step: their updates are counted together, update 1 being the first
stage's first, and the learning-rate schedule and the multi-token
prediction weight's switch (model.md, "Training") run over them all. Each
stage draws batches of its own `batch_size` windows of its own length, so
a stage puts `batch_size` times `block_size` tokens in each update. A
stage's `batch_size` is global, over every process together ("The batch is
the same on any hardware", above), and its evaluations average
`eval_batches` batches of its own size.

Each stage is evaluated at its first and last updates and every
`eval_every` updates, on windows of its length: a longer window predicts
better, so losses are compared within a stage only. When a stage after
the first starts, the optimizer hears of it (`Optimizer.stage`) and the
best evaluation so far is forgotten: `by-part` starts counting again, and
the step keeps the best weights of its last stage when it keeps no
average. An optimizer's early
stop (`by-part`'s `patience`) ends its stage, and the next
stage starts.

Checkpoints and `--resume` cross stage boundaries: the checkpoint holds
the update to continue from, and a resumed step starts within its stage.
Each stage is a span of the run's record (`training.stage`, with its
number, window length, first and last updates, and the update a resumed
run started at), each update's event names its window length, and
`provenance.json` lists, for the step, the stages with the last update
each made (`stages`). The recipes keep one stage (model.md, "Model
recipes").

The sparse layers read their chosen tokens from the first update, at every
stage (model.md, "Attention").

The log shows, at each evaluation, the update's loss over the whole batch,
every micro-batch of every process, then the validation loss.

## Recomputing layers: long windows on a small GPU

`--recompute` (or `WPT_RECOMPUTE=1`) keeps only each sublayer's input
during a pass and runs the sublayer again during the backward pass, instead
of keeping everything the sublayer computed. The result is the same; memory drops and
each update takes longer. Its memory and time with the window, exact and
sparse layers are not measured.

On an integrated Radeon 860M (gfx1152, which ROCm supports only through
`HSA_OVERRIDE_GFX_VERSION`), Muon's 10 Newton-Schulz steps on the
32-expert recipes crash the process (a segmentation fault in the GPU
stack) in the first update unless `HIP_LAUNCH_BLOCKING=1` makes the CPU
wait for each GPU kernel; 8 steps do not, and the small `classics-test`
does not crash.

## Compiling the passes

`--compile` (or `WPT_COMPILE=1`) runs each layer, the sums of Attention
Residuals between them and the multi-token prediction block through
`torch.compile`, forward and backward, which joins
the model's many small operations into few GPU kernels. It trains the same
way within rounding. On a GPU, the sums of Attention Residuals run as
Triton kernels of their own instead of through `torch.compile`
(docs/model.md, "Attention Residuals"). It covers the passes over whole
text windows, whose shape stays the same for a stage. Evaluations, windows in segments or
slices, conversations and task attempts run uncompiled, since their length
or their place in a window changes from pass to pass and each new one would
compile again. The run's log says so when the option is on. The option needs
a C++ compiler where the run happens: a gpu worker installs `g++` when it
starts, and on Windows the run starts from the Visual Studio build tools' shell.

The exact and sparse layers' attention, the routing and the routed experts
stay out of the compiled graphs: each is marked with
`torch.compiler.disable`, and computes what it computes without the option.

On a CUDA GPU the whole pass over a window, forward to the two summed losses
and backward, is then recorded once per window shape as two CUDA graphs,
which every later window of that shape replays (`wpt_model/graphs.py`): the
GPU launches the pass's kernels itself, so the CPU's time per window no
longer grows with their number. Nothing in the pass waits on the host. The
losses, the gradients and the routers' balancing counts are kept apart from
the graphs' memory, which the next replay writes over, so a run trains the
same way with the graphs or without them. The graphs hold one pass's memory
between windows. A pass with `--recompute`, or that records or replays
routes, runs without them.

The first update takes longer, and so does the first update at a new window
length. `--stop-after-updates` with and without the option compares the
speeds, since what it measures leaves the first update out.

The weights keep their names, so a checkpoint resumes with the option or
without it. On anything but a CUDA device the option refuses the run.

## Long windows: segments and slices

A window too long for one GPU's memory, even with `--recompute`, runs in
parts, which give exactly the loss and gradients of the whole window at
once (model.md, "Long windows"):

- `--segment N` (or `WPT_SEGMENT`), on one GPU: a window longer than `N`
  tokens runs as consecutive segments of `N` (the last may be shorter). A
  first pass without gradients runs every segment but the last, each
  handing the next what its layers need of the earlier tokens; then the
  segments run again from the last to the first with gradients, each
  starting from what the one before handed on, and its backward pass adds
  the gradient the later segments sent back to what it handed on. Memory
  holds one segment's activations plus what the segments hand on, which
  grows with the window (below); every segment but the last runs twice,
  up to twice the time (15% more than whole with `--recompute`, below). `N` is a multiple of 128. Windows of `N` tokens or fewer run
  whole.
- `--strategy split`, on several processes (GPUs of one or several
  machines, started as for `ddp`): every process takes the same windows,
  and process `r` trains the `r`-th of equal, consecutive slices of each
  (DeepSeek-V4's contextual parallelism); the processes exchange what
  crosses the slices, and their gradients are added up before each update.
  Each stage's window length must split into slices of a multiple of 128
  tokens.
- Both, `--strategy split` with `--segment N`, when a slice is longer
  than `N`: every process trains its slice in segments of `N` tokens. The
  window runs as consecutive parts of `N` tokens per process (`N` times
  the number of processes; the last part may be shorter), each part split
  across the processes as above, so process `r` trains the `r`-th segment
  of every part, and each part hands the next what segments hand on, with
  the same two passes. Memory holds one segment's activations per
  process; the result stays exact. Windows whose slices are `N` tokens or
  fewer are split without segments.

These are options of the hardware, not of the recipe: a run may resume with
another `--segment`, or none. The training loss is the parts' share of the whole window's (`TextMethod.part_loss`, "Methods", above); the
validation loss in parts is each part's validation loss weighted by its
share of the window's tokens. The tests check, on the CPU, that 448 tokens
in segments of 128 (three boundaries, the last segment shorter), 512 tokens
split over 2 and 4 processes (gloo), and 768 tokens split over 2 processes
in segments of 128 (three parts) give the whole window's losses, gradients,
validation loss and expert balancing, each with the layers' activations
kept and with `--recompute`, for a model with every layer kind (window
layers, exact layers choosing 8 tokens per query, sparse layers, Engram,
the experts, In-Place TTT in chunks of 96 tokens over texts whose documents
end within the window, Attention Residuals, two passes of the loop, the
MTP block). A run of `wpt train` with `--strategy split --processes
2 --segment 128` trains the same model as one process on whole windows.

Not measured with the window, exact and sparse layers: the memory and
time of segments and slices on a GPU, `split` on GPUs (NCCL, rented NVIDIA
machines), and real runs on real data; `split` was only run on the CPU, by the tests.

## In-Place TTT and loops in training

In-Place TTT (model.md, "In-Place TTT") trains with the rest of the model:
its convolution and `W_target` get their gradients through every chunk's
update that a later chunk reads, from the first update. Its chunks of
1,024 tokens are counted from the window's first token, and a window that
packs several documents starts each one, after its end-of-document token,
from the trained matrix. A window no longer than 1,024 tokens completes
one chunk at most and reads no update, so TTT changes nothing in it and
its parts get a zero gradient. Segments and slices hand on the unfinished
chunk and the sum of updates, so a window in parts trains as the whole
window, also where a chunk crosses their boundaries.

A model that loops (model.md, "Loops") runs every pass in each training
pass, and the loss reads the last one. Its cost per token is `loops + 1`
times one pass's, and so is its memory of activations, with `--recompute`
too. Segments and slices hand on each pass's values apart. R3 records and
replays each pass's routed experts apart.

## Comparing routing in tests

Two passes over the same tokens that compute in another order (whole or
in segments, other CPU instructions, other thread counts) give router
scores that differ by rounding: on x86 CPUs, whole and in segments, up to
6e-8 at the first routed layer and 7.6e-6 at the last, the FP32 rounding
of every earlier layer added up. A choice whose two deciding values lie
within that rounding of each other can then go either way: a token whose
k-th and (k+1)-th biased scores nearly tie can go to other experts, and a
token whose `s_j - alpha` lies on the edge between two of balancing's
4096 bins (model.md, "Experts") can be counted in the next bin. Every
token's (k+1)-th expert has an offset of 0, on an edge, by construction.

So a test that compares routing or balancing counts between two such passes
keeps them exact for every token whose choice is clear, and lets only near
ties differ. A token is near a tie when the two values that decide a choice
are no farther apart than the two passes moved them. The bound on those
moves is the one the test asserts: the scores each pass counted agree
within ROUNDING, 2^-14 (about 6.1e-5, the `routing` fixture in
`conftest.py`). That score bound is what limits the differences to near
ties. The other checks document it and do not enforce it: they test, for
the scores at hand, that the experts differ only where the k-th and
(k+1)-th biased scores lie within two such moves of each other (plus 4
units of FP32 rounding, ARITHMETIC, for the comparison's own arithmetic),
that each pass's counts are exactly the bins of the scores it counted, and
that a count moves only where its `s_j - alpha` lies that close to a bin's
edge, by one bin at most (`assert_routed_alike` in
`packages/wpt-training/tests/test_parts.py`; split windows compare each
process's counts with those of its tokens in the whole window, and the sum
of the processes' counts with the whole window's: the processes' scores,
put back in the window's order, are compared with the whole window's under
the same bounds, so a sum differs from the whole window's counts only where
a token's bin may differ). A test of that comparison builds scores one unit
of rounding apart, with an exact edge and a near tie, so the near-tie path
runs the same on any processor.

A test that reads text through the generation cache and compares the result
with the whole text (`test_cache.py`, `test_sampling.py`) records the
cache's routes, reads the text whole recording its own
(`routing.first_difference`), and accepts a position whose routes differ
only when the whole pass's k-th and (k+1)-th biased scores lie within two
ROUNDING plus ARITHMETIC of each other. It then compares scores or tokens
exactly before the first position that differs, and in full when none does.
Two passes that run the same operations on the same shapes, such as
`--recompute`'s (`test_model.py`), give the same scores bit for bit, and
their counts are compared exactly. An exact comparison of balancing counts
in segments fails on GitHub's runners; the same failure shows on an x86
machine with `ATEN_CPU_CAPABILITY=avx512 OMP_NUM_THREADS=2` (or `avx2`),
where one to six tokens' counts in a layer move to the next bin and no
token goes to other experts.

A test that trains in segments and compares the gradients with the whole
window's (`assert_same` in `test_parts.py`) accepts a difference of 2e-5 of
the largest gradient, beyond a relative 1e-3. The difference is the
rounding of each math library path: on GitHub's runners, up to 8.3e-7 on
AMD EPYC processors and 8.2e-6 on Intel Xeon ones, which their math
libraries compute by other routes; on AMD, 4.4e-6 with oneDNN switched off
and 4.3e-5 with `MKL_CBWR=COMPATIBLE`. A limit of 1e-6 fails on every Intel
runner.

The drafter test (`test_sampling.py`) checks only that training works: the
losses of a few steps are finite, every weight they reach changes, and the
writer verifies the trained drafter's drafts. It does not ask that they be
kept more often than uniform drafts: the test's model has random weights,
so its next-token distribution is nearly uniform (entropy 3.75 of at most
3.91), uniform drafts are already kept at about 0.78 per position, and over
6 seeds a drafter trained for 40 or 200 steps was kept no more often than
uniform drafts on average (+0.0003), and less often after 400. With one
seed the margin went from -0.031 on Intel runners to +0.075 on AMD ones.

## Strategies: how the hardware is used

`--strategy` names a built-in or a Python file:

- `single` (default): one process on the best device of the machine;
- `ddp`: several processes, one per GPU. Each process holds a full copy of
  the model, and their gradients are averaged after each pass (PyTorch's
  `DistributedDataParallel`).
- `split`: several processes, one per GPU, that split each window between
  them ("Long windows", above). Each holds a full copy of the model, and
  their gradients are added up before each update.

Every strategy runs inside the gpu worker ([machines.md](machines.md)),
whose job is the training's processes; `--processes` starts them on the
worker's machine:

```sh
uv run wpt train <recipe> <corpus id> --strategy ddp --processes 2
```

(`split` starts the same way.) A run on several gpu workers starts
`--processes` processes on each machine, and they train as one run
([machines.md](machines.md), "Several machines"); across places, only
`ddp` runs. PyTorch's processes on Windows talk only through the CPU
library (gloo), so several GPUs gain less than on Linux.

`single` refuses to start among several processes, since each would train
alone and write the same folder. One process per
machine prepares the token files; only the main process logs and writes
the model folder.

A strategy file defines `start()`, which returns a `wpt_training.Strategy`:
the device, which process this is among how many, how the model is
wrapped, and how processes average a number, share a value and wait for
each other; a strategy that splits windows sets `splits_windows` and adds
the gradients up in `synchronize`. A model too large for one GPU needs a strategy that splits the
model itself (PyTorch's FSDP, or a library such as DeepSpeed): it is
written as such a file, without changes to the training loop. None ships.

## A limit and a measure

`--stop-after-minutes` and `--stop-after-updates` stop the run as a stop
signal does: the update finishes, a checkpoint is saved and comes back,
the command exits with code 130, and `--resume` continues. With both, the
first reached stops it. They count over every step the command trains. The
updates count from its first, and the minutes from the end of its first,
since that update also compiles the kernels and, with `--compile`, the
passes.

Such a run ends by printing what it measured from the end of that first
update, which is the short run that says what a whole one costs: its
updates and seconds, the tokens per second, and how many hours the recipe's
text windows take at that speed, in all and from where it stopped. The
event `training.measured` records the same numbers in the run's record. Evaluations in that
span are part of the speed. Longer windows run slower per token than the
stage measured, and only steps that list their window lengths have tokens
known in advance: the message names the steps it leaves out, and the
drafter is not in it.

`--profile-update N` (`WPT_PROFILE_UPDATE`, 0: none) runs update N of each
step, numbered in the step from 1, under `torch.profiler` (CPU and CUDA
activities, without stacks or shapes) in the main process, and gives the run's
record the file `profile.txt`; a step that does not reach N, or a resumed
step that starts past it, has none. Evaluation is part of the update when one is due at it. The file holds, in this
order: the update's number, its wall seconds, the sum of its self device time
and its count of device kernel entries; the seconds of its parts, taken with a
device synchronize at each time stamp (the batch, each pass's `learn`, the
optimizer's step with the gradient exchange before it, and everything after);
the machine's facts (CPUs and those the process may use, the cgroup's
`cpu.max`, the versions of PyTorch, CUDA and Triton, the GPU, its clock and
power draw when PyTorch reads them); the allocator's counts (retries, out of
memory errors, segment allocations) and the peaks of allocated and reserved
memory; then the 40 operations with the most self CUDA time and the 40 with the
most self CPU time. A failure of the profiler is a warning, and the training
goes on. The other updates run without any of it.

## Checkpoints and resuming

Inside a step, every `--checkpoint-every` updates (default 500, 0 for none
during a step), the gpu worker's main process writes a checkpoint: the step's
fingerprint, the update, the weights, the optimizer's, method's and
environment's state and the best weights so far (from the step's last
update on, the weights its optimizer keeps, such as an average). After the step's last
update it always writes one, unless that update's own was just written,
and that checkpoint, as it is, becomes the step's kept state ("Step
results and reuse", above); the checkpoint's place is then emptied: the
folder is removed, and a store checkpoint loses its record while its file
stays as the step's state. The same command with `--resume`
continues the step that stopped from its checkpoint, on other hardware
too: the number of processes, `--micro-batch`, `--recompute` and
`--segment` may change; a checkpoint written by another step, or for
other inputs (its fingerprint differs), stops the run with
`SettingsMismatch`. The steps before it are reused. Without `--resume`, a
leftover checkpoint stops the run, so a crashed run is never overwritten
by mistake.

The gpu worker writes every checkpoint on its own disk, where a restart of
its processes finds it. What comes back to the wpt manager is a setting,
`WPT_LINK_CHECKPOINTS`: `step-ends` (the default) sends only each step's
last checkpoint, with the step's end, which becomes its kept state;
`every` sends each one as it is written, into the place below. With
`step-ends`, a step that fails leaves no checkpoint there, so `--resume`
starts it over. A stop always leaves one: Ctrl+C or SIGTERM lets the
current update finish, the worker writes a checkpoint, whatever
`--checkpoint-every` says, and sends it to that place, under either
setting. A stop that comes after a step's last update keeps that step's
result instead, and the run stops before the next step. The first
start of a job removes the checkpoint folders that earlier runs' jobs
left on the worker, which no later run reads.

Where it lives, on the manager's side:

- without `--run`, in `<model folder>/checkpoint/checkpoint.safetensors`, on the
  wpt manager's machine; a run that fails leaves `checkpoint/` in an
  earlier folder at that place, and a model folder that did not exist
  before the run and holds nothing else goes with the checkpoint;
- with `--run NAME` (or `WPT_RUN`), in the store: the record
  `models/NAME/checkpoint.json` names the state's file, which lies beside
  it in `models/NAME/`. Any machine using the same `WPT_STORE` continues
  the run with the same command, `--run NAME --resume` and any `--out`: a
  run can move from one machine to another. Each save
  replaces the previous record of that name, and the previous file is then
  deleted; a crash while saving leaves the previous one. The record's change and every
  deletion are recorded with the file's key, and that line is in the
  store before the deletion: a deletion whose record breaks leaves the
  file in the store. A save once the record broke still writes the checkpoint's
  record, unrecorded, and leaves the previous file beside the new one. With an
  S3 store, every save is an upload of about four times the model's size.

### Stopping a run

Ctrl+C, `SIGTERM` (a shutdown, a rented machine being taken back) or, on
Windows, Ctrl+Break, sent to any process of the run, lets every process
finish its update; they then write a checkpoint, which comes back to the
wpt manager, and exit with code 130. `--resume` continues from that update. A second
signal stops the run at once, losing the updates since the last checkpoint,
as `kill -9` or a power cut does.

### Restarting after a crash

When a process crashes, the processes the gpu worker's job started are
stopped and all start again, up to `--max-restarts` times (default 3,
`WPT_MAX_RESTARTS`); a restarted process continues from the worker's own
checkpoint without `--resume`. A one-process run is started this way too,
by a small parent process in the job, unless `--max-restarts 0`, which runs
the training in the job's first process and never restarts. Three exits are never restarted: a refusal (exit code
2: a missing or damaged corpus, a checkpoint written by another step,
settings or options that do not fit, a step that failed, a drafter that
could not be trained), which would refuse again, a record that could not be
written (exit code 3), and a stop asked for (exit code 130). A bug that
crashes every time stops after the last restart, and `wpt train` exits
with the crash's exit code. Restarts stay on one machine: on several gpu
workers, a crash ends the job on every one, and `wpt train` exits with
code 5; `wpt manager` resumes the run from the checkpoint it holds, as for
a lost worker ([machines.md](machines.md), "A lost gpu worker").

A gpu worker lost (it cannot be reached, or its link breaks) stops the run
with exit code 4; `--resume` continues it from the checkpoint the manager
holds, and `wpt manager` does so itself on a replacement for a worker it
rented ([machines.md](machines.md), "A lost gpu worker").

## The drafter

After the last step, the gpu worker's main process trains the model's drafter
([model.md](model.md), "The drafter"): a model and its drafter are one
unit, so every run trains one, against the final weights, the model frozen,
with the recipe's `drafter` section. Its prompts are the beginnings of the
documents of the training shards the run's steps read: when every step that
reads text windows reads a prefix, the shards that hold the longest prefix,
never a shard past it, so the drafter fetches nothing the model never
trained on (a step that reads every training token, or a run with no such
step, leaves every shard). The model alone writes the answers, each
up to the end of a document and, when a step trained on conversations or
tasks, `<|im_end|>`, the end of a turn: no text people wrote as an answer,
and no conversation of a step, reaches it. No option skips it. A drafter
that cannot be trained (too few prompts, a loss that is not finite, no
draft verified) refuses the run (exit code 2) and writes no model folder;
every step's result is kept, so the same command trains the drafter again
without training. The drafter reads those shards once at the end of
the run, one shard at a time, to draw its prompts; from a remote store,
each shard's local copy is released once read. The disk then holds at most
one shard only when the shards were not already local: with `--stream`,
which released each one once its tokens were in the token file, or when
every step that reads text windows was reused. Otherwise the shards
training fetched to build its token file are still on disk when the
drafter starts, and each one goes as the drafter reads it.

Then the run measures the drafter scheduler's pass times once and writes
them into its record as `scheduler-costs.json`, with the setup they were
measured on: the model's config, the drafter's block (the drafts per
round), the rows (1) and the device (its kind and name; on the CPU, the
processor's). `--scheduler-costs FOLDER` replays those an earlier run
recorded in its record folder `FOLDER` instead; times measured on another
setup are refused before training starts. [model.md](model.md), "The
drafter", says exactly what a replay guarantees, and
[observability.md](observability.md) what the file holds.

## The model folder

The model folder holds:

- `config.json`, the model's numbers;
- the weights (safetensors, optionally split with `--max-shard-size`); each
  file's header carries `license` (AGPL-3.0-or-later), `output_license`
  (CC0-1.0) and `notice`, which points to the folder's `NOTICE` and the
  project, and `tokenizer` (the tokenizer recipe and version, such as
  `classics version 1`) and `tokenizer_sha256`;
- `tokenizer.json`, the corpus's vocabulary, exactly the bytes those
  headers name: `wpt chat` and `wpt eval` refuse another (see
  [tokenizer.md](tokenizer.md));
- `provenance.json`, what produced the model: the corpus (its id, depot
  and dataset), each step, its fingerprint, result, what it reached and how
  much of the corpus it learned (`learned`), the most any step learned, and
  how many texts that credits;
- `CREDITS`, every text the steps learned from, with its credit, license
  and notice, copied from the corpus's tables;
- `LICENSE`, the AGPL text; `NOTICE`, the licenses of the weights and of
  generated text, `CREDITS`, and the output notice; `README.md`, the model card, with the licenses at the top and
  the steps in order;
- `drafter/`, the drafter for speculative decoding trained against these
  weights, with its `config.json`, weights (the same header) and
  `provenance.json`, which names the command and recipe that trained it,
  the corpus, the sha256 of every weight file it was trained against, its
  settings, its losses, the calibration and the measure: `load_model`
  refuses a folder without it or whose drafter names other weights
  ([model.md](model.md), "The drafter");
- `evals/`, results written by `wpt eval` (see [evaluation.md](evaluation.md)),
  shown in the model card.

The main process writes the folder beside its target, as a hidden folder
named `.<folder name>-<random>`, records the sha256 of every file in it,
then renames it into place. A run that stops before (a crash, a refused
record, a failed step) leaves the target as it was. Training into a folder
that already exists still works, but replaces it as a whole: everything in
it goes, not only the earlier weights and the `evals/` results, which
describe other weights, but also its `checkpoint/` and any file put there
by hand. Keep such files elsewhere. The checkpoint's removal is recorded
before it is removed, and the record is in the store before the folder is
renamed. The earlier folder is first moved aside, then removed once the
new one is in place. When the new one cannot take its place, or the run is
interrupted meanwhile, the earlier folder is moved back; when that fails
too, the run stops with `RestoreFailed`, which names the earlier folder
(`.<folder name>-replaced-<random>`) and the new one, both kept: move one
of them into place by hand. Whenever the rename was tried and the target is
missing afterward, whatever error stopped the run (a first rename refused,
or a record that broke meanwhile), the new model is kept and named on the
error output, with the earlier folder when there is one. Without `--run`,
when the target holds only this run's checkpoint folder and the new folder
cannot take its place, the new model is kept beside it and named on the
error output, as when there was no target, since no other model is there
to go back to: move it into place by hand. When the earlier folder cannot
be removed, or its removal is interrupted, it stays beside the new one, as
`.<folder name>-replaced-<random>`, named in a warning and in the record:
remove it by hand. The run still succeeds, unless it was interrupted.

`NOTICE` and `README.md` are generated from the folder's own records, so they
cannot disagree with it. `CREDITS` lists every text the steps learned
from: a plain text whose tokens start inside what a step learned, and the
conversations and tasks a step read. Run
`wpt eval` on the folder to measure it.

`wpt chat <model folder>` continues text typed by the user, until the model
ends the document. Its start line names the licenses and where the notice
is; `wpt chat <model folder> --about` prints the folder's `NOTICE`. A model
trained on text only continues text. `wpt chat <model folder> --chat`
talks to a model trained on conversations: it keeps the history, writes it
with the chat template ([chat-format.md](chat-format.md)), and shows the
model's thinking and tool calls with their tags, without running the
calls. It measures the drafter scheduler's pass times when it loads the
model; `--scheduler-costs FOLDER` reads instead those a run recorded in its
record folder `FOLDER` of the store, refused when they were measured on
another setup ([model.md](model.md), "The drafter").

## Records written for training

The `answer-as-assistant` and `finish-tasks` steps learn from recorded
human demonstrations and from interactive environments.

### Rules

These records follow the corpus rules. Each example counts, so review
is systematic.

- Text written by people: existing sources under allowed licenses, original
  texts written for the project, and contributions under the DCO.
- No output of another company's language model, and no synthetic data that
  a WPT model did not make. WPT's own models may be used in training, their
  output included ([data-selection.md](data-selection.md), "Models in
  training"), such as a model's own attempt as the reference of a reward
  ("Rewards", below).
- A demonstration's `assistant` message may start as a draft written by a
  WPT model: the person edits it
  or confirms it as it is, and the message's `metadata.draft` records the
  model folder, the sha256 of its weights and of the draft, and whether the
  person changed it ([chat-format.md](chat-format.md), "Model drafts"). An
  unedited confirmed draft is allowed.
- An AI assistant (a coding agent helping a contributor) never reads or
  writes people's texts (CONTRIBUTING.md, "Models and generated text").
- Environments use no fake web pages or apps: they replay recorded content,
  only when the project has the authorization to use it. A task's replies
  to the `ask` tool are content a person wrote and the task records, which
  the environment replays: no person waits for the model's question
  ("Tools", below).
- Scores come from programs or from humans, never from another company's
  model.

### Where records live

Records are items of the depot, like any text: they get the same license,
credit, notice and decisions. Conversations are items of kind
`conversation`, in the chat format ([chat-format.md](chat-format.md)), and
tasks for agentic training are items of kind `task` ("Tasks", below); the
depot checks each against its kind's JSON Schema when it is added. Neither
is ever a plain text that `read-text` reads. A corpus keeps the
conversations of its dataset as tokens of the chat template, in token
files of their own, and its tasks as lines of `tasks.jsonl`, and the steps
read them from there ([data-selection.md](data-selection.md), "Corpora").

- A record holds only what a person made: commands typed, edits, messages
  written. From a session with an AI assistant, only the person's own
  messages may be kept, never the assistant's replies.
- No other person's personal data, and no secrets: terminal output and
  paths are checked before a record is added to a depot.

### Record format

The shape of recorded demonstrations and environment recordings is not
settled yet. The code reads two kinds of records: conversations, turned
into tokens by a template before training
([chat-format.md](chat-format.md)), and tasks ("Tasks", below). Whatever
the shape, a record's license, credit and decisions are its item's.

## The answer-as-assistant step

The `answer-as-assistant` step learns to write the assistant's turns of
the corpus's conversations tagged `tag` (the step's setting, by default
`answer-as-assistant`; a conversation's tags are in the corpus's `texts`
table). Each conversation becomes tokens with the
chat template ([chat-format.md](chat-format.md)), and the loss
(`assistant-tokens`) is the next-token loss of the tokens the assistant
wrote only (thinking, text, tool calls and the closing `<|im_end|>`),
averaged over every such token of the update's conversations: the other
turns are context the model reads and does not learn to write. The multi-token prediction block is not
used, so it does not change, and the routers' balancing biases are locked.

Each epoch (`epochs`, 3) goes over every conversation once, in an order
drawn from the recipe's `seed` and the step's number in the recipe, `batch` conversations per update (8; the last
update of an epoch may hold fewer). Conversation `j` of an update goes to
process `j` modulo the number of processes, and a process left with fewer
passes than the others makes passes that learn nothing, so that every
process makes as many. A corpus that holds no conversation with the tag refuses the
run before any step trains. The step fails when a
conversation does not fit the model's context (cutting it would drop the
answers it holds), when one has no assistant message, and when the loss
is not finite. It needs a vocabulary
of tokenizer 0.3.0 or later, which reserves the chat template's ids. Talk
to the result with `uv run wpt chat FOLDER --chat`.

## The finish-tasks step

The `finish-tasks` step teaches the model to finish tasks as an agent, by
reinforcement learning (group relative policy optimization). The model
works in a small harness, on tasks that people write, with four built-in
tools, `read` (a file), `write` (a file), `edit` (replace a string that
occurs exactly once in a file) and `bash` (run a command), and the tools a
task adds ("Tools", below). Its code is the `wpt-agent` package (tools,
containers, images, the agent loop and the scoring), the `agent-tasks`
environment and `wpt_training/agentic.py` (the attempts and their
rewards), and the `grpo` method (the learning). The wpt manager plays
every attempt, the gpu worker's model writing its tokens, so it needs
Docker where the manager's `docker` command reaches it: its own machine or
the Docker host `DOCKER_HOST` names ([machines.md](machines.md)); every
process on the worker then trains on its share of them. It reads the
corpus's tasks tagged `tag` (the step's setting, by default
`finish-tasks`); a corpus that holds none with that tag refuses the run,
with "no tasks", before any step trains.

### Tasks

A task is the content of a depot item of kind `task`, written by people
(`wpt_agent.tasks`). Its content holds three parts:

```text
task     what the model is asked, the image, the limits, the tools it needs and, if needed,
         larger containers, its own tools, the user's replies and the Git repository it works in
files    the starting files, by their path under files/: {mode, content in base64}, copied into
         the container's /work (those tool_files names go where only the tools' programs read them)
check    any program, the same way (its first line names its interpreter, such as #!/bin/sh)
```

```json
{"instructions": "Fix the failing test in calc.py.", "image": "python",
 "limits": {"max_turns": 20, "max_tokens": 4000, "command_seconds": 30},
 "container": {"memory": 8192}, "needs": [["read", "bash"], ["edit", "write"]],
 "replies": [{"pattern": "(?i)python.*version", "reply": "Python 3.13."}],
 "answered_by": ["read", "bash"], "answered_in": ["NOTES.md"], "area": ["calc.py", "tests/**"],
 "gates": ["read_before_write"]}
```

- `instructions`: the user message that starts every attempt. No task's
  instructions state an answer block: the model's final reply is its
  answer, as text, and the harness asks for its confidence after it
  ([chat-format.md](chat-format.md), "The answer"). The harness adds no
  sentence to the opening.
- `terse` (optional): the same request in one line, for the measure of
  terse against full requests: a task can hold its request in two forms,
  its full `instructions` and this one-line form. The agent loop opens an attempt
  with `instructions`, or with `terse` when its caller asks
  (`wpt_agent.attempts(..., terse=True)`; a task without the field is
  refused there). It holds no line break.
- `image`: the folder in `data/images/` the attempt's container comes from.
- `limits`: `max_turns`, the assistant turns in one attempt; `max_tokens`,
  the tokens the model writes in one attempt, over every turn;
  `command_seconds`, how long one tool call may run.
- `container` (optional): sizes above the defaults (Attempts, below):
  `memory` and `disk` in MiB, `processes`, `cpus`. Each can only raise its
  default; a smaller value leaves the default. Since `/work` and `/tmp` are
  kept in memory, the sizes that result must leave room for the processes:
  2 x `disk` + 1024 MiB (`HEADROOM`) must stay below `memory`. A task that
  raises `disk` must raise `memory` with it; the defaults leave 1024 MiB
  more than that.
- `needs`: the tools the task cannot be done without, each need the list of
  its variants (two different tools that write a file, say); every attempt
  is offered exactly one variant of each need ("Tools", below). `[]` when
  the task needs none.
- `replies` (optional): the user's replies, written in advance, to the
  `ask` tool, each a `pattern` and its `reply`; only a task with replies
  offers `ask`.
- `answered_by` (required with `replies`): the looking tools that can
  answer what the task leaves open, `read`, `bash` or a declared tool
  marked `looks` (the task's or its image's pool's), through which an
  attempt opens the files of `answered_in`.
- `answered_in` (required with `replies`): where the task's answers live,
  as glob patterns of paths under `/work` (as `area` reads them). An
  attempt asks the user only after it opened, through a tool of
  `answered_by`, every file it starts with that they match, and did not
  find the answer there (the check can name its text to `asked_last`); a
  question counts at once when they match none of its files, the answer
  being absent (`asked_last`, "The check library", below).
- `tools` (optional): the task's own tools beyond the built-in ones and
  `ask`, each with its `name`, `description`, `parameters` (the JSON Schema
  of its arguments), `program` (the absolute path, in the image, of the
  program that answers it) and, optionally, `names` (other names an attempt
  may offer it under) and `looks` (`true` for a tool that looks at sources
  as `read` does, such as a search or a fetch; "The check library", below).
- `tool_files` (optional): paths under `files/`, each a file or a folder
  with everything in it, that only the tools' programs read: they go to
  `/tmp/wpt-tools/` instead of `/work`.
- `absent` (optional): absolute folders of the image, outside `/work` and
  `/tmp`, their names made of letters, digits, `.`, `_`, `+` and `-`, that
  the attempts do not see: each is covered by an empty, read-only folder
  kept in memory (Docker's `--tmpfs`), so a task removes what its image
  installs there. The check's container shows the image as it is.
- `repository` (optional): the Git repository the task works in, its
  `address` and the full id of a `commit`. When the task runs, the harness
  fetches that one commit (`wpt_agent.repository.fetch`, a shallow fetch,
  once per process), on the machine that plays the attempts, since the
  containers have no network; `/work` starts as the repository checked out
  at it on the branch `main`, its `.git` folder included, with the task's
  files laid over it.
- `area` (optional): the paths under `/work` an attempt may change, as glob
  patterns (`*`, `**`, `?`, `[...]`), `[]` when the task only answers and may change nothing. An attempt
  that changed, added or removed a path no pattern matches breaks the
  `within_area` part ("Rewards", below). A task without `area` gets no such
  part.
- `gates` (optional): the parts that must be 1 for an attempt to score at
  all, by name: the check's own (such as `asked_last`) or `within_area`. A
  gated part below 1, or missing, makes the attempt worth 0 whatever the
  rest ("Rewards", below). There is no gate for every task beyond the
  missing answer.

Reading a task (`wpt_agent.tasks.from_item`) refuses one whose image is
not a folder of `data/images/` holding a `Dockerfile`, and one whose
fields break a rule across them: sizes that break the rule on `disk` and
`memory` above; a declared tool taking a built-in name or `ask`, or
declared twice; a need naming a tool the task does not offer (`ask`
without replies included); a tool in two needs; a task with replies and no
`answered_by`, or an `answered_by` naming a tool that is not a looking
tool; a reply pattern that is not a regular expression; a `terse` that
holds a line break; a `tool_files` entry that names nothing under
`files/`; an `absent` entry that is `/work` or `/tmp`, lies under them, or
holds `.` or `..`. The depot's schema of the kind checks each field's
shape when the item is added.

### Tools

Every attempt is offered the tools of its task, chosen for that attempt
("Variation", below). A task's tools are:

- the four built-in ones, `read`, `write`, `edit` and `bash`;
- `ask`, when the task has `replies`;
- the tools the task declares;
- and the working tools of the image's pool (`tools.json`, "Images",
  below), whose names the task does not take.

**`ask`**. The model calls it with its question, and the harness answers
from the task's replies, written in advance; the attempt goes on. Each
reply's `pattern` is a Python regular expression searched for anywhere in
the question, and the first reply whose pattern is found is given. A
question that no pattern matches gets "I don't know that", so a precise
question pays and a vague one does not. No person waits for the model's
question: the reply is written ahead and the harness gives it, as content a
person wrote that the task records (Rules, above). `ask` is offered only in
a task that declares replies. It serves asking the user last: a turn
without a tool call still ends the attempt, so a question goes through
`ask`.

**Declared tools**. A task declares each tool it offers beyond the built-in
ones and `ask`: its name, description, parameters, and the program in the
agent image that answers it. The program runs inside the attempt's
container, under its limits and `command_seconds`: it reads the call's
arguments as JSON on its standard input (as ASCII, any other character
escaped) and runs from `/work`; its standard output is the result, and an
exit code other than 0 makes the call fail, with its standard error (or,
when it printed none there, its output) as the result. Like a check, a
program must not load code from its current folder: the model's files are
there. Most of these tools are emulations, such as a web search answered
from local files; an emulation gives the real tool's kind and shape of
answer: a search returns ranked snippets with their sources, a fetch
returns a page's text. So the model learns what kind of answer to expect
from a kind of tool and adapts to whatever harness it is put in. Real MCP
servers are not used in training.

**Hidden files**. An emulated tool's backing files are hidden from the
model's own tools (`read`, `bash`): only the tools' programs read them. The
task names them in `tool_files`; they are unpacked into `/tmp/wpt-tools/`,
owned by root and readable only by group 1001 (folders 750, files 640). A
program runs as user 1000, the model's, in group 1001, so it reads them and
changes `/work` as the model's own tools do; the model's commands run in
group 1000, so they can neither list nor read them, and since the groups
differ the kernel lets them neither look into the program's process
(`ptrace`, `/proc/<pid>/fd`, `/proc/<pid>/mem`) nor gain group 1001 (no
setgid). The files count against what `/tmp` may hold. The check's
container does not get the task's `tool_files`, but it does get the files
an attempt's variant hid (a variant's `tool_files`, "Variation", below):
they are back at their place in its `/work`, as the tools' programs left
them, in place of whatever the model put there.

**Variation**. Every task lists the tools it absolutely needs (`needs`),
and a needed tool may have variants. For each attempt the step draws, from a generator seeded with the recipe's `seed` and the
step's number in the recipe, the task,
the round, the group and the attempt, so runs are reproducible
(`wpt_agent.offer`):

- one variant of each need, under its own name; a variant may be tools
  offered together with paths under `files/` that it hides (its
  `tool_files`, `{"tools": [...], "tool_files": [...]}`), which go to
  `/tmp/wpt-tools/` in that attempt only;
- each other tool of the task and of the pool (the variants not drawn
  aside) left out or kept, with even odds; a kept one is offered under
  one of its other names, with even odds, when one is free (the built-in
  tools' and `ask`'s other names are `OTHER_NAMES` in
  `packages/wpt-agent/src/wpt_agent/tools.py`, a declared tool's its
  `names`);
- every offered tool in a shuffled order.

The tools of an image's pool pad the offer with tools the task never needs;
each really works, answered by its own program in the image. The map from
each name the model saw to the real tool goes into the conversation the
check reads ("Scores", below). The aim is asking the user last without
habits tied to particular tools.

### Images

The containers come from Dockerfiles in the repository,
`data/images/<name>/Dockerfile`. Each machine builds them itself; a
committed lock pins what goes in, not the bytes that come out, since
builds differ between machines. A Dockerfile names its base as
`ARG BASE=<image>` before `FROM ${BASE}` and installs `${PACKAGES}`
(`ARG PACKAGES="<names>"`) with apt, so an image is Debian or Ubuntu. The
supported bases are Debian 13 (trixie) or later and Ubuntu 24.04 (noble) or
later, whose apt reads the dated archive (below); an older apt ignores it
and would read today's archive, which no lock pins:

```dockerfile
ARG BASE=debian:trixie-slim
FROM ${BASE}
ARG PACKAGES="bash coreutils python3"
RUN apt-get update && apt-get install -y --no-install-recommends ${PACKAGES} && rm -rf /var/lib/apt/lists/*
```

Only the base's apt is checked against the snapshot (below): the
Dockerfile's own steps are trusted not to add package sources or to
download anything, and nothing checks them. A person reviews them like
any other change.

An image's pool of tools is `data/images/<name>/tools.json` (optional),
written by people like the Dockerfile: `{"tools": [...]}`,
each tool declared as a task declares its own, its program a file the
Dockerfile puts in the image. The recipes' check (`ci/data.py`) checks it
against `image-tools.schema.json` and refuses a tool declared twice or taking the
name of a built-in tool or `ask`. A changed pool is a changed image
folder, which needs a new build.

`uv run wpt agent lock-image NAME` pulls the base, pins it by digest,
takes the current moment as the snapshot, checks that apt on the base,
after `apt-get update`, reads its package lists only from the snapshot
hosts (it refuses the base otherwise), builds the image, and writes
next to the Dockerfile `lock.json` (the base by digest, the snapshot and
the version of every installed package) and `licenses.json` (every
installed package, its version and its licenses; [licenses.md](licenses.md),
"Container images of agentic tasks"). Commit both. It refuses to replace
an existing lock unless given `--force`: locking again is the person's
choice. The recipes' check checks both files against their schemas
(`image-lock.schema.json`, `image-licenses.schema.json`).

Every build reads its packages from the dated archive of the base's
distribution (snapshot.debian.org or snapshot.ubuntu.com) as it was at the
lock's snapshot, so a lock keeps building for years. The base gets one
layer that sets apt's `APT::Snapshot` before the Dockerfile runs; apt
finds the archive itself. apt fetches it over HTTPS without checking the
server's certificate, since the base images hold no certificates; it
checks the archive's signatures and every package's checksum itself, and
accepts the dated archive's expired Release files.

`uv run wpt agent build-image NAME` builds the image from its lock on this
machine (the base by digest, the snapshot, every package as
`name=version`), and stops when the build installed other packages than the
lock pins. Like `lock-image`, and through the same code, it first checks
that apt, after `apt-get update`, reads its package lists only from
snapshot.debian.org or snapshot.ubuntu.com, and refuses to build otherwise.
A failed `docker build` (in `lock-image` or `build-image`) and a failed
`apt-get update` in that check are tried again by the same rule, 6 times in
all, with a growing pause before each new try: 10 s, 30 s, 1 min, 2 min and
5 min, about 9 minutes of waiting, because snapshot.ubuntu.com was once
down for over 2 hours. The error of a failed build ends with the output of
the step that failed. No step builds an image: the `agent-tasks`
environment stops before any attempt when a task's image is not built for
the folder's current files, and names the command that builds it. A changed Dockerfile
or lock needs a new build.

### The agent image

`data/images/agent/` is the first image of agentic tasks: Debian 13 with
`bash`, `git` and Python 3, and three parts of its own, which its Dockerfile
copies in:

- `wpt_check.py`, the shared check library ("The check library", below), in
  `/usr/lib/python3/dist-packages/`, where every `python3` finds it, `-I`
  and `-P` included;
- the programs that answer tools, in `/usr/local/lib/wpt-tools/`, written
  in Python and started as `python3 -I`, with the helpers they share,
  `wpt_tools.py`, next to the library;
- `tools.json`, its pool of working tools.

The pool (`tools.json`) pads the offer with tools that really work in
`/work`: `list_files` (a folder's entries, as `ls -p`), `find_files` (paths
matching a glob pattern), `search_text` (lines matching a regular
expression, as `grep -rn`) and `calculator` (an arithmetic expression); the
first three are marked `looks`. Since a program runs in the group that
reads the hidden files, these tools open a path only inside `/work`, one
folder at a time and never through a symbolic link, so neither a path
leading out of `/work` nor a link the model made reaches the hidden files.

A change to the library or to a program changes the image's folder, so
every machine builds the image again (`wpt agent build-image agent`); a
change to its packages also locks it again (`lock-image agent --force`).
The library and the programs are tested in the repository
(`packages/wpt-agent/tests/test_agent_image.py`), once for every check
that calls them, and in the built image itself.

### The check library

Rules many checks share live in one library inside the agent images, tested
once in the repository, and each check calls it. A check started as
`python3 -I` imports it:

```python
#!/usr/bin/python3 -I
import wpt_check
conversation = wpt_check.load()  # conversation.json, next to the check
right = ...  # the task's own judgement of the files and the answer
print(wpt_check.report(conversation, right, sources_first=wpt_check.sources_first(conversation)))
```

A check prints the named parts of its score ("Scores", below), and the
step weighs them ("Rewards", below); the library weighs nothing.

The rules read the attempt's trace by real tool names (`metadata.tools`
maps each name the model saw to the real one):

- `report(conversation, correct, **rules)`: what a check prints, the JSON
  object of its parts: `correct`, the task's own judgement through
  `scored`, and each rule given by name, as a number (`True` is 1). A
  task's check may report any part of its own the same way, such as a
  `conventions` part for the conventions its workspace shows (placement,
  naming, framework, existing commands): those are checked only by each
  task's own check, never by a rule of this library.
- `scored(conversation, score)`: the missing-answer rule. The attempt's
  score is 0 when it has no answer, whatever its files hold: no final
  reply, or a reply to the confidence question that is not a number from 0
  to 100 ([chat-format.md](chat-format.md), "The answer").
- `brier(conversation, correct)`: the Brier score of the confidence,
  `(confidence - 1)²` for a right result and `confidence²` for a wrong one;
  1, the worst, without an answer.
- `read_before_write(conversation, new=())`: every file written or edited
  was read first, by `read`, by an earlier `bash` command that names it, or
  written by the attempt itself; `new` lists the files the task expects the
  model to create.
- `sources_first(conversation, changes=("write", "edit", "ask"))`: the
  attempt looked at sources (below) before its first change or question,
  and looked at all when it did neither.
- `asked_last(conversation, found="")`: every `ask` came after the attempt
  opened every file where the answer lives: the files of `/work` it started
  with that the task's `answered_in` matches (`metadata.answered_in`), each
  opened by a call that looks (below) to one of the task's `answered_by`
  (`metadata.answered_by`) and did not fail, whose `path` argument is the
  file or, for `bash`, whose command names it and runs a program that shows
  what a file holds (`opened`, `SHOWING`: the reading programs but `wc`,
  `csplit`, `find`, `ls`, `dir`, `vdir`, `stat`, `du` and `cmp`, which show
  only its name, size or counts). Given `found`, the answer's text, a
  question no longer counts once a call that looks showed it, its words in
  order as `holds` reads them: the answer was found. A question counts at
  once when no such file exists (the answer is absent), and when the
  attempt was offered none of those tools. A look that does not name the
  file, such as `grep -r` over a folder, does not open it.
- `before(conversation, first, then)`: every call to a tool of `then` after
  a call to one of `first`; `calls` gives the trace, each call by its real
  tool's name and whether it looks, `answer` the answer the loop built,
  `{"answer": ..., "confidence": ...}` or `None`, and `answer_text` its
  text, the final reply's, `""` without one.
- `holds(text, expected)`: `text` holds the words of `expected` one after
  the other, letter case, punctuation, Markdown marks and spacing aside.
- `beside(name)` and `file_text(path)`: the text of a file next to the
  check, and of a regular file the attempt left (`""` for none, a folder, a
  pipe or a link).
- `programs(command)`: the programs a `bash` command runs, as the looking
  rule finds them (below).

Looking is one rule, which `sources_first` and `asked_last` share: every
looking tool counts as opening a source. A call looks when it calls `read`,
a declared tool marked `"looks": true` (the image's pool has `list_files`,
`find_files` and `search_text`), or `bash` with a command that runs one of
the image's reading programs. Those programs are listed once, as `READERS`
in `wpt_check.py`: the programs of the image that show what files or
folders hold, such as `cat`, `head`, `grep`, `sed`, `ls` and `find`, and a
test checks that the built image holds each of them. The command is parsed
as bash reads it, never searched for a name: each simple command of its
pipelines and lists (`|`, `;`, `&`, `&&`, `||`, a line break, parentheses,
a function's body) runs the program named by its first word, after any
`VAR=value` and bash's reserved words, and after programs that run another
program (`env`, `nice`, `nohup`, `stdbuf`, `time`, `timeout`, `xargs` and
the like) with their options and the words they take; a comment does not
count. So `cd docs && cat a.md` looks and `echo cat a.md` does not. Which
offered tools look is decided in one place, `wpt_agent.tasks.looks`
(`read`, `bash`, and a declaration marked `looks`): the agent loop writes
their real names into the conversation's `metadata.looking`, which the
library reads, and reading a task refuses an `answered_by` naming any
other tool.

The trace rules of #37 are `read_before_write`, `sources_first` and
`asked_last`, each reported as a part of that name by the checks that apply
it. Each task states where its answers live,
its `answered_in`, and the looking tools that can open them, its
`answered_by`; `asked_last` waits for both.

### Attempts

For each task, the model makes several attempts (`group`), written
together: the model reads the start the attempts' openings share once,
through its generation cache ([model.md](model.md), "Generation"), copies
it into one row per attempt, reads in each row the rest of that attempt's
opening (its offered tools differ), and writes the attempts' turns in one
batch; while one attempt's tool calls run, the others wait. Each attempt
runs in a fresh container of the task's image, removed after the attempt;
a group's containers run at the same time.
The container has no network and a read-only root: it can write only
`/work`, its working folder holding the task's `files/`, and `/tmp`, two
folders kept in memory. Its sizes are the defaults of the task format
(`SIZES` in `packages/wpt-agent/src/wpt_agent/tasks.py`), each raised
by the task's `container` field when it asks for more: `memory` (4096 MiB,
the files in `/work` and `/tmp` included), `disk` (1024 MiB, what `/work`
and `/tmp` may each hold), `processes` (256, processes and threads at
once) and `cpus` (2). The check's container gets the same sizes, and must
hold a full copy of `/work`: that is why `HEADROOM` (1024 MiB) of memory
stays free for the processes when `/work` and `/tmp` are both full. The
model's commands run as user 1000, not root, and nothing in the container
can gain privileges (no `setuid` program works).
The container keeps only three capabilities, `CHOWN`, `DAC_OVERRIDE` and
`FOWNER`, which root uses to unpack the task's `files/` into `/work` with
user 1000 as their owner; the model's commands, as user 1000, hold none of
them. The container's first process is Docker's init (`--init`), which
collects every process that ended, so background commands that finished
never count against the process limit; the check's container starts the
same way.

The conversation follows [chat-format.md](chat-format.md): the tools the
attempt is offered, then the instructions as the user's message. The model
writes an assistant turn; each tool call in it runs in the container (a
relative path starts at `/work`, and a call that cannot run returns its
error to the model); the results come back as a tool turn; and the model
writes the next turn. The attempt ends when a turn closes without a tool
call, when the model writes the end of a document, after `max_turns` turns
(the last turn's calls still run), or when `max_tokens` is reached in the
middle of a turn, whose calls then do not run. The turn that closes without
a tool call is the final reply, the answer; the loop then adds one user
turn asking for the confidence, a number from 0 to 100 and nothing else,
and the model replies; that turn counts against `max_tokens` and the
context, not `max_turns` ([chat-format.md](chat-format.md), "The answer").
The loop builds `{"answer": <the final reply's text>, "confidence": <number
/ 100>}`, or `null` when there is no final reply or the reply is not such a
number, and keeps it in the conversation's `metadata.answer`, next to
`metadata.tools`, the map from each name the model saw to the real tool,
`metadata.looking`, the real names of the offered tools that look at
sources, `metadata.answered_by`, the task's, and `metadata.answered_in`,
the files of `/work` the attempt starts with that the task's `answered_in`
matches ("The check library", above). The model's `block_size` does not
limit an attempt: the cache reads it past `block_size`, and the loss pass
reads it whole. The attempt also ends when the model's own processes or
files fill its container, so that Docker can start no command there:
processes left running reach the process limit, or files in `/work` and
`/tmp` exhaust the memory. That turn's calls get no result, and Docker's
own error never reaches the model. The attempt is scored like any other, on
what `/work` holds; when running out of memory also stopped the container,
`/work` is lost and the check scores an empty folder. A command that runs
longer than `command_seconds` is stopped, and its result says so (one that
ignores the signal to stop is killed 5 seconds later); a command that only
exits with the code `timeout` uses is reported with that exit code. A tool
result longer than `result_bytes` (4000) keeps its first and last 2000
bytes, with a line between them saying how many bytes were cut; the output
is cut while it is read, so a command that prints without end never fills
the machine's memory. `edit` rewrites only a UTF-8 file of at most 1 MiB,
and tells the model to use `bash` otherwise. No tool argument may hold a
NUL character or text that is not UTF-8, `write`'s content and `edit`'s
strings included: such a call fails with a message to the model. A file
holding a NUL byte, or other bytes that are not UTF-8 text, is written
through `bash`, with a command that prints them from escapes, such as
`printf '\0'`.

### Scores

When the attempt has ended, every process in its container is stopped
(the container is paused). The task's `check` then runs in a fresh
container of the same image and sizes, holding only a copy of the
attempt's `/work`: what the model started, or left anywhere else, cannot
reach the check. The check is copied in, in a folder of its own, and runs
as root from `/work` for at most `check_seconds` (300).

Next to the check, in its folder, is `conversation.json`: the attempt's
conversation in the chat format, as the attempts file keeps it (Runs,
below) but with only the attempt's own `metadata`: `tools`, the map from
each name the model saw to the real tool; `looking`, the real names of the
offered tools that look at sources ("The check library", below);
`answered_by`, the task's; `answered_in`, the files of `/work` the attempt
started with that the task's `answered_in` matches; and `answer`, the
answer the loop built (the final reply's text and the confidence) or
`null`. It is written as ASCII JSON, any other character escaped. A check
finds it as `"$(dirname "$0")/conversation.json"`, and scores everything:
the files, the trace of tool calls and the confidence.

Since the check runs from `/work`, the files the model left are in its
current folder. A check must not start an interpreter that loads code from
the current folder: run Python as `python3 -P` (or `-I`), and call programs
and scripts by absolute paths. Otherwise a `conftest.py`, `unittest.py` or
similar file the model planted would run as root inside the check and could
print any score it likes. Its standard output, without the white space
around it, must be one JSON object of named parts, each a number from 0 to
1, `correct` among them: the task's own judgement of the files and the
answer, next to the rules it reports, such as `{"correct": 0.8,
"read_before_write": 1, "sources_first": 0.5}` (`wpt_check.report` prints
it). A bare number from 0 to 1 reads as `{"correct": n}`, so a check that
prints one number keeps working. Its exit code and standard error are not
read. A check that prints anything else (a part outside 0 to 1, a `true`,
no `correct`), crashes or runs out of time gives no score; the attempt
counts as 0, and the step prints a line naming the task and why. A task
with an `area` gets one more part, `within_area`, from the harness
(`wpt_agent.play`): 1 when every path under `/work` the attempt changed,
added or removed matches one of the area's patterns, 0 otherwise. The
attempt's `score`, what its line keeps, is its `correct`
part, or 0 when a gate of the task is broken (`wpt_agent.gated`).

When Docker or the container fails during an attempt or its check, which
is not the model's doing, the attempt is dropped from its group and the
step prints a line naming the task and the reason. Each command starts
by printing one byte before it runs, so a command that never started
(no such container, a container that is not running, a refusal of
Docker's daemon or runtime) is told apart from a command that exited
with an error. When Docker could not start it, the sandbox asks Docker
whether the container is full (memory ran out in it, or fewer than 16
processes are free): if so, the model filled it (Attempts, above);
otherwise Docker failed. A group left with fewer than 2 attempts teaches
nothing and is skipped. The step goes on, unless every attempt of a
round was dropped: that round makes no update, and the step fails, naming
the round, how many attempts Docker failed and its last error. A round
with at least one update goes on as usual. A step that made no update at
all fails too: every group was skipped, and the step says how many because
their rewards were all equal and how many because they kept fewer than 2
attempts. An update whose groups were all skipped has nothing to learn and
is skipped (`training.skipped`).

### Rewards

An attempt's reward is its parts weighed together, with gates. The check
reports its parts; the step combines them, in `wpt_training/agentic.py`
(`combined`), which both task steps share, never the check:

- The reward is 0, whatever the rest, when the attempt has no answer (no
  final reply, or a reply to the confidence question that is not a number
  from 0 to 100), when its check gave no score, or when a gate of the
  task is broken: a part its `gates` names is below 1, or missing.
- Otherwise it is the weighted sum of `correct`; the calibration of the
  confidence, 1 - (confidence - correct)², one minus its Brier score
  (#40; paper 12 of the plan); each other part, a rule (the check's, and
  `within_area`); the number of tool calls, a cost on every call (#37,
  so the model does not open everything blindly); and, for a terse
  attempt, its closeness to the full attempt (below).
- The calibration and the tool calls are computed by the step from the
  attempt, not by the check: the confidence the harness collected
  (`metadata.answer`) and the tool calls of its conversation, so every
  attempt gets them, whatever its check reports.
- The weights are the step's settings (`weight_correct` 1,
  `weight_calibration` 0.5, `weight_rule` 0.25 for each rule,
  `weight_tool_call` -0.01 for each call, `weight_closeness` 0.25),
  starting values until experiments on the 1B model set them (#53).
  `weight_correct` must stay above `weight_calibration`, so stating a low
  confidence never pays more than being right: otherwise the run stops
  before any step runs (`agentic.check`). The `agent-tasks` environment
  takes these defaults from one place, `agentic.WEIGHTS`.

Terse against full (#39): a task that holds a one-line form (`terse`) is
played twice in each round of `finish-tasks`, first `group` attempts with its full
instructions, then `group` attempts with its one-line form, offered the
same tools attempt by attempt. Both forms' attempts are one group for the
advantage: with `group` 8, each of the 16 attempts' advantage is its reward
minus the mean reward of all 16. Only the terse attempts carry the
closeness part. Terse attempt *i* is compared with full attempt *i*, the
one of the same index: its reward adds, weighed by `weight_closeness`, how
alike their changes to `/work` are (twice the lines their diffs share, in
order, over the lines of both, `wpt_agent.changes`; 1 when neither changed
anything), next to passing the task's check as any attempt does. The
model's own full result is the reference, which the rule on WPT models in
training allows ([data-selection.md](data-selection.md), "Models in
training"). A task whose `area` is empty only answers, and gets no
closeness part. On any other task, a terse attempt whose full attempt
Docker dropped has nothing to be compared with: it is kept in the attempts
file with a `null` reward but left out of the group, since its reward would
lack the part every other terse attempt of the group has. As for any group,
one left with fewer than 2 attempts is skipped, and one whose rewards are
all equal teaches nothing.

The group's size matters (#40; paper 13 of the plan): normalizing by the
group can erase a penalty the weights were designed to give when few
distinct answers appear in a group. The advantage here is the reward
minus the group's mean, never divided by its standard deviation, and a
group left with fewer than 2 attempts is skipped (below); the group size
is set by experiment with the weights.

### Learning

Each attempt's advantage is its reward minus the mean reward of its group
(the attempts at the same task in the same round, both its forms for
`finish-tasks`'s terse tasks). It is not divided by the group's
standard deviation: that would make a group whose scores barely differ
weigh as much as one whose scores differ widely, and give tasks that are
almost always passed or failed the largest steps. The loss is minus the
advantage times the log-probability of each token the model wrote, the
same tokens `answer-as-assistant` learns (never the instructions, the tool results or the
template's own tokens), averaged over every token the model wrote in the
update's attempts; a skipped group (below) adds none. The tokens are the ones the model sampled, kept as they
were, since writing the conversation as tokens again would not always give
the same ones. There is no frozen reference model and no KL penalty: each
update learns from attempts the current model just made. A group whose
rewards are all equal teaches nothing and is skipped. The model samples
from its own probabilities (`temperature` 1, `top_k` 0: every token),
which the loss assumes.

The routed expert layers ([model.md](model.md), "Experts") choose experts
for each token while the model writes an attempt. The step records those
choices, each attempt's apart, as the generation cache reads its tokens,
and the loss pass routes each token through the same experts
instead of choosing again (routing replay, "R3"), so the loss scores the
tokens through the experts that wrote them. The routers' weights do not
change during the step, and their balancing biases do not either: the
step locks both. The
hash-routed layers choose from the token alone and need neither. The multi-token prediction block is not used, so it
does not change.

### Runs

The order of the tasks in each round and every sampled token come from
generators seeded with the recipe's `seed` and the step's number in the
recipe, so a run gives the same attempts again on
the same machine and setup (the same model, tasks, images and devices); the
checks themselves may differ, as any program may. The model alone writes
the attempts, without a drafter: the weights change at every update, the
drafter is trained only at the end of the run ("The drafter", above), and
the attempts then depend on the seed alone, never on this machine's times.
A task's attempts are written together in one batch, from one reading of
the task's opening (model.md, "Generation"). The step prints one line per
round (`rounds`, passes over every task): the mean reward, the number of
attempts and of updates.

Every attempt is kept in the run's record folder
([observability.md](observability.md)), as one line of
`<step>-attempts.jsonl` (`finish-tasks-attempts.jsonl`), written through
the run's record as soon as the attempt is scored, so a line on a local store is on the disk before the next attempt
is scored. A group's attempts are written together, then scored one by one,
in order, each attempt's container removed and its line kept before the
next attempt is scored (each attempt's line is kept as soon as that attempt
is scored, even though attempts are written together). Each line is a
conversation in the chat format ([chat-format.md](chat-format.md)) whose
`metadata` says which attempt it is, so the file passes the same check as
any `chat` file:

```json
{"tools": [...], "messages": [...],
 "metadata": {"tools": {"shell": "bash", "...": "..."}, "answer": {"answer": "...", "confidence": 0.5},
              "step": "finish-tasks", "task": "<task id>", "round": 1, "group": 3, "attempt": 2, "terse": false,
              "dropped": false, "error": null, "ended": "answered",
              "parts": {"correct": 1.0, "sources_first": 1.0}, "score": 1.0, "reward": 1.66,
              "written": 412, "tool_calls": 4}}
```

- `tools`, `messages`: the attempt: the tools it was offered, as the
  model saw them, the instructions, every assistant turn as the model's
  tokens read, every tool result as the model received it, and after the
  final reply the confidence question and the model's reply to it. A dropped
  attempt has only the tools and the instructions, the conversation it
  would have started from.
- `metadata.tools`: each name the model saw, mapped to the real tool's;
  `metadata.looking`: the real names of the offered tools that look at
  sources; `metadata.answered_by`: the task's; `metadata.answered_in`:
  the starting files the task's `answered_in` matches; `metadata.answer`: the
  answer the loop built, the final reply's text and the confidence, `null`
  without a final reply or a confidence from 0 to 100, and absent for a
  dropped attempt.
- `step`, `task`: the step and the task's id.
- `round`: the round, from 1; `null` for an attempt made before the rounds
  (the `finish-tasks-with-less-writing` step's reference, below).
- `group`: the group within the round, from 1, in the order played (for a
  reference attempt, the task's place in the list of tasks); `attempt`:
  the attempt within its group, from 1; `terse`: whether it opened with
  the task's one-line form ("Rewards", above).
- `dropped`: `true` when Docker failed; then `error` says how, and
  `ended`, `parts`, `score`, `reward`, `written` and `tool_calls` are
  `null`.
- `ended`: why the attempt ended (`answered`, `ended`, `turns`, `tokens`,
  `context` or `full`; Attempts, above).
- `parts`: the check's parts, with `within_area` for a task with an area,
  `null` when the check gave no score; `score`: its `correct` part, 0
  with a broken gate ("Scores", above); `reward`: what the attempt is
  worth to the step (for `finish-tasks`, its parts weighed together, "Rewards",
  above), `null` for an attempt never trained on.
- `written`: the tokens the model wrote, the tokens the loss learns;
  `tool_calls`: the tool calls it made.

The trace names each line: the attempts of a group are written, scored
and kept in a span `<step>.play` (`finish-tasks.play`: `task`, `round`, `group`, `attempts`,
`terse`);
inside it, each attempt's line is kept in a span `<step>.attempt` (`task`, `round`, `group`, `attempt`, `terse`, then `file` and
`line`, its line number from 1, with `dropped`, `ended`, `score`,
`reward` and `written`); each group is an event `<step>.group` (`task`,
`round`, `group`, `terse`: whether it holds the one-line form's attempts too, `file`, `lines`, and `outcome`: `learned`, or why it was
skipped); each round is an event `<step>.round` (`file`, `lines`: its first
and last, `attempts`, `dropped`, `updates`, and the value `mean_reward`). A
recipe that runs the step twice adds to the same file, and the line
numbers go on. A line the store refuses stops the command at once with
exit code 3, as any part of the record does.

Its settings are the `agent-tasks` environment's, their defaults: `rounds`
(10), `tasks_per_update` (4), `group` (8),
`terse` (true for this step: a task with a one-line form is played in
both forms), `reward` (`weighed`), `reference` (0), `temperature` (1),
`top_k` (0, every token), `check_seconds` (300), `result_bytes` (4000)
and the five weights ("Rewards", above). Its optimizer is
`by-part` at a constant learning rate of 1e-6, without weight decay, with
its early stop on ("Optimizers", above). The `agent-tasks` environment
makes no evaluations, so the early stop never stops the step and it keeps
its last weights. A recipe's
entry changes them ("Steps", above). The containers' sizes are not among
them: they belong to each task.

## The finish-tasks-with-less-writing step

The `finish-tasks-with-less-writing` step comes after `finish-tasks` in a
recipe's steps. Its goal is the same result as the model it starts from,
with fewer tokens. It trains on the tasks `finish-tasks` reads, those
tagged `finish-tasks`, with the same images, containers, limits and
checks, and the same learning (Attempts, Scores, Learning and Runs
above): groups of `group` attempts, in each task's full form only (it
plays no one-line form), the advantage as the reward minus the group's
mean, the loss over the tokens the model wrote, R3 routes and routers that
do not change, dropped attempts and the same failures. Only the reward
differs: the step is the `agent-tasks` environment with `reward`
`shorter` and `reference` 8.

First, the reference. Before any update, the incoming model makes
`reference` attempts at each task, written together, run and scored as
`finish-tasks`'s are. An attempt's score here is its result: its
`correct` part, 0 when a gate is broken, its check gave no score or it has
no answer (`agentic.result`). A task's best start score is the best score
among them. An attempt passes when its score is at least that best score,
and the task's reference length is the mean length of its passing
attempts. A length counts only the tokens the model wrote, the same tokens
the loss learns, never the instructions, the tool results or the
template's own tokens. The reference stays fixed for the whole step and is
kept with the step's state, so a resumed step does not play it again; the
step does not train on its attempts. A task whose every reference attempt
scored 0 has no result to keep and is left out, and so is a task whose
every reference attempt Docker failed; the step prints a line for each.
When no task is left, the step fails.

Then the rounds, as `finish-tasks`'s, with this reward for each attempt:

- 0 when it fails: its result, the gated `correct` part
  (`agentic.result`), is below the task's best start score;
- when it passes, its parts weighed together as `finish-tasks` weighs them
  ("Rewards", above, with this step's own weight settings; the parts are
  combined in the step that learns, never in the check), times the
  task's reference length divided by the attempt's length: the weighed
  parts once for an attempt as long as the reference, twice for one half
  as long, half for one twice as long.

Every passing attempt whose weighed parts are above 0 is worth more than
every failing one, and of two passing attempts whose parts weigh the same
the shorter one is worth more. The step prints one line per task for the
reference (its best start score and reference length) and one line per
round (the mean reward, the number of attempts and of updates). Sampling
draws from one generator seeded with the recipe's `seed` and the step's number, for the reference and then
the rounds. Every attempt is kept as `finish-tasks` keeps its own (Runs,
above), in `finish-tasks-with-less-writing-attempts.jsonl`, under the spans
and events `finish-tasks-with-less-writing.attempt`, `.group` and
`.round`: the reference attempts first, with `round` and `reward` `null`
in their `metadata`, then the rounds' attempts with their reward (the
round's value is `mean_reward`). Its settings are those of `finish-tasks`
("Runs", above), with `terse` false, `reward` `shorter` and `reference` 8.

## What is exactly repeatable

**The corpus is what a model trained on**: its record and tables name
every text, and its fingerprints cover its parts. Shards are compared by their token hash, which covers
only their rows, not the bytes of the Parquet file (a newer pyarrow may write
different bytes for the same tokens).

**Tool versions**: the repository pins a reference version of each tool
(`uv.lock` for Python packages, `.python-version`, and
`packages/wpt-tokenizer/rust-toolchain.toml` for Rust). Each build
records the versions it actually used, and reports where they differ from
the reference; it never refuses to run for that. The data hashes catch any
real change. Most upgrades change nothing; for example a newer Unicode
version could only affect newly added characters in steps that use Unicode
tables, and the tokenizer uses none.

**Training is not bit-exact**: GPU arithmetic differs between GPUs, drivers
and PyTorch versions, so the same inputs give equivalent but not identical
weights. On a CUDA device, training also computes FP32 matrix products in
TF32, and `--compile` joins operations in its own order; on the CPU, training
computes them in FP32 and repeats bit for bit. On a GPU, the attention kernels also add up the entries' gradients
in the order the GPU's atomic additions take, so two runs on one GPU can
differ slightly too (docs/model.md, "Attention"); `wpt train` logs whether
TileLang, Triton or PyTorch computes the attention, which `--kernels` names. The updates themselves are the same: each update's
examples, and any random draw a model makes, come from the seed and the update, and a resumed step
continues the same updates; a reused step's result stands for its inputs, whatever machine trained
it. The attempts of a task step depend on the model's own sampling, and the checks may differ, as
any program may. Around it, everything possible is checked:

- before training, every input is checked against its hash: each shard
  (file sha256 and token hash) and the vocabulary;
- `provenance.json` records what produced the model: the corpus (its id,
  depot and dataset), how much of it the steps learned, the vocabulary, the
  recipe's name and content, the command-line options, the strategy (file,
  its sha256), the model shape and parameter count, each step (its files
  and their sha256, its settings, its optimizer's, its locked parts, its
  fingerprint and result, whether it was reused, its stages with the last
  update each made, the updates made and skipped, the update whose
  weights it left, which the model card shows, its best validation loss
  at that update, and what it learned), the
  repository commit (and whether the working tree had
  uncommitted changes), the device and the tool versions.
