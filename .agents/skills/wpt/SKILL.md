---
name: wpt
description: How to work in the WPT repository (Workers' Pre-trained Transformer) with its own tools. Use it before changing the depot, a recipe, the model, training, an evaluation or a training step, before running any `wpt` subcommand, and before every push. It names the subcommand for each job, what each one reads and writes, how to extend evaluation and training, and the rules that keep the checks of `ci/` green.
---

# WPT tooling

WPT trains a language model only on content whose license is clear. The
texts live in a depot, a folder known by its address, that holds the texts
and items, their licenses and notices, the people's decisions on them, the
libraries and the datasets (`wpt depot`; docs/architecture.md). The
repository holds the recipes and the images of agentic tasks, never texts.
Everything built or trained lives in a store named by the `WPT_STORE`
setting (environment, or `.env` at the root): `corpora/`, `models/`,
`runs/` and `cache/`, plain named things. The documents in `docs/` hold
every decision; read the one that covers your change before changing it.
Every text in the repository, this skill included, describes only the
current state: no decision dates, no history, and a change may break
compatibility (CONTRIBUTING.md, "Writing about the project").

## Before every push

```sh
uv run ci/all.py
```

`ci/` holds one script per part, and `ci/all.py` runs them all: `lock`
(`uv.lock` matches the project files), `tests`, `rust`, `data` (the rules
on every recipe, tokenizer lock and image of `data/`, read through the
service layer) and `commits` (the author's sign-off on every commit a push
would send).

Only what a change touches runs, the files as they are, committed or not, compared
with the upstream branch: `--all` runs everything, `--base REV` and `--range RANGE`
name other revisions. pytest-testmon chooses the tests from a record in the user's
cache folder that every checkout of a machine shares. `--cpu` adds CI's software
conditions (the GPU hidden, Triton blocked). A green run here is no promise of a
green CI: only CI shows the timing of its 4 cores (docs/architecture.md, "Checks").

Commits are signed off (`git commit -s`): CI refuses any commit without a
`Signed-off-by` line.

## Setting up a machine

```sh
uv run scripts/setup.py          # installs the PyTorch build for this machine's GPU
```

A plain `uv sync` removes PyTorch, because no build is part of the default
install; run `scripts/setup.py` again after it. `uv run` does not remove it.
It detects the GPU, and says which settings a chip needs in the environment.
A Rust toolchain is needed: the tokenizer is built from source.

On Windows, work in Windows itself (PowerShell), never inside WSL: clone,
set up, build, train and run every `wpt` subcommand there. Linux containers,
where a step needs them, run through Docker Desktop.

## The commands

Every job is a subcommand of one command, `wpt` (`apps/wpt`; `uv run wpt
--help` lists them); `wpt depot web` serves the operations of `wpt depot`
in a browser, on 127.0.0.1 only, and training, evaluation and the other
jobs run from the command line (docs/architecture.md, "The app" and "Two
interfaces").

| Job | Command | Reads | Writes |
| --- | --- | --- | --- |
| the depot: texts, items, decisions, libraries, datasets, corpora, erasures | `uv run wpt depot new ADDRESS ...`, `clone ADDRESS TO`, `pull`, `upgrade` (brings a depot of an older format up to the code's `FORMAT`; every other command, `wpt depot web`, clone and pull refuse a depot of another format, and `backup` and `restore` copy an older one as it is and refuse a newer one; run it again after a crash), `sql QUERY` (`texts`, `libraries`, `datasets`, `about`, `erasures`: the current state; `<name>_revisions`: every revision; `contents`: every content version; `memberships`, `pins`), `add FILE [--kind KIND] --license L ...`, `license add ID NAME FILE`, `show ID`, `decide admit\|refuse WHERE`, `decide admit\|refuse --erasure ID`, `relicense ID ... --license L ...`, `content ID FILE` (a new content version; like `relicense`, the text's decisions start over and it leaves every open library, put back by hand once admitted), `credits ID [--title T] [--author A ...] [--tag T ...] [--address A] [--notice FILE] [--clear CREDIT ...]`, `library new\|put\|lock\|rename\|fork [--without WHERE]\|report LIBRARY PROBLEM [WHERE]`, `dataset new\|edit\|export\|import`, `about [--maintainer M] [--takedown-contact C]`, `tokenizer RECIPE`, `build DATASET`, `erase ID [START END]`, `erasures` (`wpt depot --help`; every decision, an add, an erasure request, a reported problem, a change of content and a change of credits included, takes `--name`, `--email` and `--accept`, the decider's declaration, public; a refusal ends with exit code 2 and writes nothing; a clone refuses every change to a row its upstream owns, docs/architecture.md, "Following an upstream") | the depot `WPT_DEPOT` names; for `build`, the dataset's tokenizer recipe and its lock, and the store; for `tokenizer`, the recipe, its depot and the store | the depot; for `build`, the corpus in the store's `corpora/`; for `tokenizer`, the vocabulary in the store and the recipe's lock |
| work on the depot in a browser | `uv run wpt depot web [--port N]` (needs Bun and the design-system submodule, `git submodule update --init`), then http://127.0.0.1:8410/ (HTTP API described at `/api/openapi.json`; the data pages "Texts" with each text's page, "Libraries", "Datasets" and "Erasures", the setup pages "Licenses", "Depot" and "Settings", each row of their lists opening a view of that thing with its own address, the "Overview" and the "Add a text" button do what the `wpt depot` commands do, except `new`, `clone` and `tokenizer`) | the depot `WPT_DEPOT` names, `apps/wpt/ui`, `vendor/wic-design-system`; to build a corpus, the store and the tokenizer recipes | the depot through `wpt_depot`; `node_modules/` at the root, the design system's `packages/*/dist`, `ui/dist` |
| what the store holds | `uv run wpt store list` | the store | prints every corpus and model |
| copy a kept model out of the store | `uv run wpt store copy-out NAME FOLDER` (NAME: the run's name; FOLDER must not exist; every file checked against its sha256) | the store's `models/NAME/` | FOLDER |
| follow a run | `uv run wpt store runs`, `uv run wpt store run FOLDER [--follow]` | the store's run records | prints the runs, or one run's stage, counts, last events and errors |
| train on a built corpus in one command | `docker compose up` with `COMPOSE_FILE=deploy/manager/compose.yaml` in `.env` (or `uv run wpt manager`), `WPT_RECIPE` required, `WPT_CORPUS` the corpus (unset: the one last built from any revision of the recipe's dataset, as `wpt train` finds it; the manager builds nothing, and a run with no corpus is refused), `WPT_RUN` the run's name (unset: a new one, written into the git-ignored `wpt-run.txt` with its copy-out line), `WPT_EVALUATE=1` to evaluate at the end; `WPT_GPU_WORKERS`: the gpu workers' keys, or how many to rent (rented workers run the training and the evaluation with `WPT_KERNELS=triton` unless the settings set `WPT_KERNELS`; `WPT_RENTAL_GPUS`, `WPT_RENTAL_KIND`, `WPT_RENTAL_PROVIDER`; a lost rented worker, or a crash on several workers, resumes the run up to `WPT_REPLACEMENTS` times; `WPT_BUDGET` in dollars, `WPT_MAX_HOURS`, or both: the first reached stops the run, its checkpoint saved, and the cost is printed before each rental; `wpt manager stop [RUN]`, or a SIGTERM (`docker compose stop`), stops a run on purpose the same way, and `WPT_STOP_DEADLINE` seconds later (default 600) the manager ends what has not ended and gives the machines back, docs/machines.md; the container exits when the runs are over, and `docker compose up` again continues a failed run); goes on first with its own runs a crash cut short; builds the images of the corpus's tasks where `DOCKER_HOST` reaches Docker; settings from the environment, else from `.env` (docs/machines.md) | the recipe, the store | the checkpoint and the model under `models/NAME/` (the manager's own copy of the folder removed); the tasks' images on the Docker host; its run record `runs/<run>/`; its runs' gpu workers, tokens included, in `WPT_MANAGER_STATE` (never the store); rentals at the provider, stopped once the evaluation or training ends, and tried again until stopped |
| a gpu worker on a machine with GPUs | `docker compose -f compose.yaml -f compose.nvidia.yaml up` (or `compose.amd.yaml`; the files of `deploy/gpu-worker/` plus `wpt_worker/` and `wpt_utils/` copied beside them; prints its key `wpt-worker://TOKEN@HOST:PORT#PIN`; give it, one per machine, to `wpt train` in `WPT_GPU_WORKERS`; docs/machines.md) | what the wpt manager sends it | its own folder (certificate, checkout, libraries, checkpoints); never the store |
| check a machine before work | `uv run wpt doctor` (the store answers, and its credentials, the GPU and the PyTorch build, the attention kernels and nvcc, the Rust toolchain, free disk, uncommitted or untracked files; exit code 1 when something is missing) | the settings, the store, the checkout | prints what is missing and how to fix it |
| train | `uv run wpt train RECIPE [CORPUS_ID] [--out FOLDER] [--stream] [--recompute] [--compile] [--segment N] [--run NAME] [--scheduler-costs RECORD_FOLDER] [--kernels FORM] [--stop-after-minutes M] [--stop-after-updates N] [--profile-update N] [options]`, started detached (skill `wpt-training-run`; RECIPE: a model recipe, or `SERIES:SIZE`, a size of a model series, which reads the start of the corpus every size of the series shares, or `SERIES:SIZE:RUN`, another run of that size (`branch-SHARE`, a decay branch that goes on from the size's trunk; `lr-FACTOR`, at the smallest size, the learning-rate check); CORPUS_ID (or `WPT_CORPUS`), a corpus id or a start of it; without it, the corpus last built from any revision of the recipe's dataset (matched by its depot and its id, so a renamed dataset keeps its corpora; it opens the depot to read the dataset's id and current revision, so the depot must be readable where training runs), with a warning when it was built from an older revision than the current one, and none refuses the run, naming `wpt depot build`; a stale corpus (an erasure reached a text it holds) and one with fewer tokens than the recipe's `data.total_tokens` are refused; `uv run wpt train series NAME` prints the numbers of every size of a series and trains nothing; a corpus of another depot, dataset or tokenizer recipe is refused; `--stop-after-minutes`, `--stop-after-updates`: stop as a signal does, the first reached, and print the tokens per second and the projected time of the run (docs/training.md, "A limit and a measure"); `--profile-update N`: update N of each step runs under the profiler and its run record receives `profile.txt` (same section); trains through gpu workers, the keys of `WPT_GPU_WORKERS` (one per machine), or else a local one it starts; exit code 4 when a gpu worker is lost; `WPT_LINK_CHECKPOINTS`: `step-ends`, the default, or `every`, the checkpoints that come back from the worker; docs/machines.md; runs the recipe's `steps` in order, starting from the result of the last of its first steps the store already holds, so a run reuses the steps it shares with an earlier one; a corpus that holds nothing for a step refuses the run before any step trains; a step that cannot do its part (`StepFailed`) stops the run with exit code 2, no result of it kept but the checkpoint it wrote, which `--resume` continues (with `WPT_LINK_CHECKPOINTS=every`; with `step-ends` it stays on the worker and `--resume` starts the step over), the results of the steps before it kept; ends by training the model's drafter; `--scheduler-costs`: replay the drafter scheduler's pass times an earlier run recorded on the same setup; a checkpoint is always written just before the drafter, so a drafter failure resumes with `--resume`; `--stream`: a few shards on disk at a time; `--recompute`: much less memory, about a third slower; `--compile`: the passes over whole text windows through `torch.compile`, on a CUDA device only (docs/training.md, "Compiling the passes"); `--segment N`: windows longer than N tokens run as segments of N, the same training in the memory of N tokens plus what the segments hand on, up to twice the time; with `--strategy split`, each process's slice runs in segments of N; `--run`: the checkpoint in the store, so another machine can continue; `--kernels`: `triton`, `tilelang` or `pytorch` computes the attention, one that cannot work refusing the run (unset: the first kernels installed for the GPU; `wpt manager` names `triton` on rented workers); docs/model.md, "Attention") | the model recipe, the store | a model folder, with `CREDITS`, the texts it learned from; each step's result, the store's `cache/steps/<fingerprint>`; its run record `runs/<run>/` (one per command, over restarts); `scratch/` token files in the store's local folder; with `--run`, the record `models/NAME/checkpoint.json` and the state it names, beside it in `models/NAME/`, until the model is saved |
| train on several GPUs | `uv run wpt train RECIPE CORPUS_ID --strategy ddp --processes N [options]`, or `--strategy split` to split each window between the GPUs (windows too long for one; DeepSeek-V4's contextual parallelism) (on the gpu worker's machine) | the store | the same |
| evaluate | `uv run wpt eval FOLDER [EVAL ...] [--kernels FORM]` (from inside the repository; the model runs on the first gpu worker of `WPT_GPU_WORKERS`, else a local one; `--kernels tilelang`, `triton` or `pytorch`, or `WPT_KERNELS`, names what computes the attention on the gpu worker, and one that cannot work there refuses the evaluation) | the folder, the store | `FOLDER/evals/<name>.json`, `README.md`; its run record `runs/<run>/` |
| lock an image of agentic tasks | `uv run wpt agent lock-image NAME [--force]` (needs Docker; `--force` replaces an existing lock) | `data/images/NAME/Dockerfile`, the base image's registry, the dated package archive | `data/images/NAME/lock.json` and `licenses.json` (commit them) |
| build an image of agentic tasks | `uv run wpt agent build-image NAME` (needs Docker; before `wpt train` with `finish-tasks` or `finish-tasks-with-less-writing`, and before `wpt eval`'s agent measures) | `data/images/NAME/`, the base image's registry, the dated package archive | a local Docker image |
| continue text | `uv run wpt chat FOLDER` (`--kernels tilelang`, `triton` or `pytorch`, or `WPT_KERNELS`, names what computes the attention, and one that cannot work stops the command; `--about` prints its NOTICE; `--chat` talks to a model trained on conversations; `--scheduler-costs RECORD_FOLDER`, its `<run>`, reads the scheduler's pass times a run recorded on the same setup instead of measuring them) | the folder; with `--scheduler-costs`, the store | |

Everything that changes a model comes from its model recipe
(`data/recipes/models/NAME.json`: data, architecture, seed, the training
`steps` in order, the drafter; docs/model.md), or from a model series
(`data/recipes/series/NAME.json`: the rules its sizes share and the list
of sizes), from which `wpt_recipes.series` derives each size's model
recipe (docs/model.md, "Model series"). Each step is
`{"step": NAME}` with, when they differ from the step file's, its
`optimizer` and that optimizer's settings, its `settings` and the parts it
keeps `locked`, named as the model names them (docs/model.md, "Parts";
docs/training.md, "Steps"), and `continues: true` when it goes on from the
whole state the step before it left (docs/training.md, "Continued
steps"). The built-in steps are `read-text`,
`answer-as-assistant`, `finish-tasks` and `finish-tasks-with-less-writing`.
The `read-text` step's `stages` list the window lengths in order, each
with its updates (`steps`) and its `batch_size`, which is global (over
every process) (docs/training.md, "Stages"); the model's context is the
last window length of the last step that reads text windows. Segments and split windows give
the whole window's loss and gradients, so they are options, never part of
the recipe (docs/training.md, "Long windows"). The `wpt train` options only
say how the hardware runs it, and each also comes from `WPT_<OPTION>` in the
environment (`--micro-batch` -> `WPT_MICRO_BATCH`); `--help` on any command
lists them. A crash restarts the processes on the gpu worker on their own, up to
`--max-restarts` times (default 3), from the last checkpoint; Ctrl+C or
`kill` finishes the current update and saves first, and that checkpoint reaches
the manager whatever `WPT_LINK_CHECKPOINTS` says; after a stop, the same command
with `--resume` continues from `FOLDER/checkpoint/` (or, with `--run NAME`,
from the store, on any machine). Exit codes: 2 refused, 3 the run's record
could not be written, 130 stopped.

Every run of `wpt train`, `wpt eval` and `wpt manager` writes its record
into its own folder of the store, `runs/<UTC time>-<command>-<main
id>-<random>/` (main id: a short corpus id, a short weights hash or the
run's name): `start.json`, `events.jsonl` (spans, events, every printed
line, with its level and without the time and level it prints with),
`ids.json` (the ids read and written, which chain the runs together),
`end.json`. With `OTEL_EXPORTER_OTLP_ENDPOINT` set (environment or
`.env`), the same goes live to an OpenTelemetry backend
(`deploy/otel-lgtm/compose.yaml`). A backend failure is one warning; a
record the store refuses stops the command with exit code 3 (`wpt train`
first finishes its update and saves its checkpoint). A training step's
environment records its own events and files through `data.telemetry`, and
an evaluation through `context.telemetry`.
Details: `docs/observability.md`.

## Texts in the depot

Every rule on licenses lives in the documentation: `docs/licenses.md`
and `CONTRIBUTING.md`. Read them before adding or reviewing texts. Never
apply a rule they do not hold: a case they do not settle is a question for
the maintainer, and the answer goes into the documentation as a general
rule about the current state, with no date and no source named
(CONTRIBUTING.md, "Writing about the project"). A new text starts under
review; only a person admits or refuses it, with a declaration of their
legal name and email (`wpt depot decide`), and no program decides.

## The service layer

`wpt_recipes/recipes.py` (`Recipes`) is the only code that reads or
writes `data/` (docs/architecture.md, "The service layer"): the model,
tokenizer and series recipes, the tokenizer locks and the images of
agentic tasks (Dockerfiles, their locks, license lists and `tools.json`);
it writes only the generated locks and license lists, for the command
that makes them. Code elsewhere asks it for documents and bytes, never for
a path under `data/`; `packages/wpt-recipes/tests/test_data_access.py`
fails otherwise, and the audit hook of `wpt_recipes/guard.py` refuses such
an access at run time (`DataAccessRefused`), except from tests. A new
operation on the depot gets both interfaces, a `wpt depot` command and a
`wpt depot web` route with its JSON Schema, both thin controllers over
`wpt_depot`.

## A person's texts

As an AI assistant, never read or write people's texts (CONTRIBUTING.md,
"Models and generated text"). A person may hand you raw text to turn into
files for the depot, asking you to change only its form. The text must
stay theirs, and no output of another company's model may enter the data,
so:

- Never open, read or print the content of those files.
- Learn their shape only through scripts that report structure, never
  words: line counts and lengths, blank lines, character classes,
  encodings, line endings.
- Change them only through scripts that transform the whole file
  mechanically (line endings, spacing, splitting into files), never by
  writing their text yourself.
- From a session with an AI assistant, keep only the person's own
  messages.

## Vocabularies

A tokenizer recipe's lock freezes its vocabulary: adding texts never
retrains it (docs/tokenizer.md). Raising the recipe's `version` and running
`wpt depot tokenizer RECIPE` trains a new one and replaces the lock; it is a person's decision (it makes later
models incompatible with earlier ones), so never raise it unasked. A change
of the tokenizer's code asks for no `version` of the recipe. Commit the
lock file.

## The model folder

`wpt train` writes `config.json`, the weights (each safetensors header
carries `license`, `output_license`, `notice`, `tokenizer` and
`tokenizer_sha256`), `tokenizer.json` (loaded only through
`wpt_model.load_tokenizer`, which checks it against those headers),
`provenance.json` (what produced it: the corpus, its depot and dataset,
and how much of it each step learned), `CREDITS` (every text it learned
from), and `LICENSE`, `NOTICE` and `README.md` (the model card).
`NOTICE` and `README.md` come from `wpt_model.notice.write_documents`, which
reads only the folder's own files: change what they say there, never by
editing a generated folder. `wpt train` writes the folder beside its
target and renames it into place; training into an existing folder replaces
it as a whole (its `evals/`, checkpoint and any other file go with it).
Every model folder holds `drafter/` (`config.json`, weights with the same
header, `provenance.json`): the drafter for speculative decoding, which
`wpt train` trains at the end of every run against the final weights. A model and its drafter are one unit: `load_model` and
`load_drafter` refuse a folder whose drafter is missing or names other
weights' sha256 (`DrafterMismatch`), and nothing skips the check; a
folder written by hand gets one through `save_drafter`, whose provenance
names `{"target": {"weights": weight_hashes(folder)}}`. `NOTICE` and the
card describe it (docs/model.md, "The drafter").

Weights are AGPL-3.0-or-later; generated text is CC0-1.0 under
`OUTPUT-NOTICE`. `packages/wpt-model/src/wpt_model/LICENSE` and
`OUTPUT-NOTICE` are copies of the root files, and a test fails when they
differ: change both together.

## Writing an evaluation

One Python file, a built-in in `apps/wpt/src/wpt/eval/evals/` or any
path given to `wpt eval`:

```python
from wpt.eval import Result

def run(context):  # context: model (on a gpu worker: config, generator,
                   # writer, greedy, losses), tokenizer, device, folder, provenance, store,
                   # telemetry (the run's record)
    ...
    return Result(data={...}, text="the same findings, for people")
```

Every evaluation returns `Result`: `data` that JSON can hold, and `text`
that the model card shows. Details: `docs/evaluation.md`.

## Writing a training strategy or step

A strategy (how the hardware is used) is one Python file, a built-in in
`packages/wpt-training/src/wpt_training/strategies/` or any path given to
`wpt train --strategy`:

```python
from wpt_training import Strategy

def start():  # a subclass overrides wrap, no_sync, synchronize, mean, reduce, broadcast, barrier, close;
                # splits_windows = True: every process trains its slice of the same windows
    return Strategy(device, rank=0, world_size=1, local_rank=0)
```

A step is made of separate parts, each one Python file, a built-in in its
folder of `packages/wpt-training/src/wpt_training/` or any path:

```python
# methods/NAME.py: how a loss is made from a batch of examples
READS = "windows"  # the kind of examples it learns from
SETTINGS = {...}
def start(settings): return Mine()  # a wpt_training.Method (TextMethod on text windows): learn, update_loss,
                                    # optionally begin, unreached (parts its loss never reaches), after_update, state_dict

# environments/NAME.py: where the examples come from
GIVES = "windows"  # the kind of examples it gives
SETTINGS = {...}
def start(settings, data): ...  # a wpt_training.Environment: stage_list, tokens, learned, batch (None: the update
                                # is skipped), evaluates, evaluate, after_update, finished, state_dict, close;
                                # data: the step's name, store, corpus record, tokenizer, config, strategy,
                                # telemetry, log, seed, the step's number, step_seed (both mixed:
                                # what the step's draws are seeded by); optionally check(settings),
                                # context(settings) and check_corpus(settings, record, store): ValueError
                                # when the corpus holds nothing for it, before any step runs

# optimizers/NAME.py: how the weights move, with its schedule
SETTINGS = {...}
def start(model, settings, run): ...  # a wpt_training.Optimizer over run's parameters; optionally check(settings)

# steps/NAME.py: puts them together
METHOD = "next-token"
ENVIRONMENT = "text-windows"
OPTIMIZER = {"name": "by-part", "settings": {...}}
LOCKED = ()        # model parts left as they are (wpt_model.PARTS)
SETTINGS = {...}   # defaults for its method's and environment's settings
```

A method that reads what its environment does not give, an unknown part or
setting, an optimizer that refuses the model, and a corpus that holds
nothing for a step are refused before any step runs. The runner in `wpt_training/runner.py` runs one step: its
stages, each update's global batch, windows in parts (`parts.py`),
evaluation and checkpoints; a part never seeds randomness itself, so runs
stay the same on any hardware. Anything a resumed step needs goes in the
`state_dict` of its method, environment or optimizer. A part that finds it
cannot do its job (a loss that is not finite, every attempt dropped)
raises `wpt_training.StepFailed(reason)`. The step keeps its result in the store
under a fingerprint of everything that decides it: the corpus part the
step reads (the record's `fingerprints`: text, conversations or tasks), not
the whole corpus, and never the hardware options (`--stream` included). After the last step, every run trains a drafter for the final
weights; the drafter is not a step. Details: `docs/training.md`, "Steps".

The `finish-tasks` steps play their attempts through `wpt_agent.play` with
the model alone (no drafter), and keep every attempt as a line of `<step>-attempts.jsonl` in the run's
record folder: a chat-format conversation whose `metadata` identifies the
attempt. A task's check prints named parts (`correct` and the rules it
reports), and the step weighs them into the reward with its weight
settings, gated by the task's `gates` (docs/training.md, "Rewards").

## Generating text

Anything that makes a model write (a chat, an evaluation, a step's
attempts) goes through `wpt_inference`, never through a loop of its own
over `model(ids)`:

```python
from wpt_inference import Writer, next_message, sample_tokens

for token in sample_tokens(model, ids, max_new_tokens=200):  # one text
    ...
reply = next_message(model, tokenizer, conversation, max_new_tokens=512)  # the next chat message, as wpt chat --chat
writer = Writer(model, prompt, rows=8, stops=[...], routes=True)  # several attempts from one prompt
written = writer.write({row: tokens_it_reads_first}, {row: most_tokens_it_may_write})
```

They read through the model's generation cache (`WPT.new_cache`,
`extend`, `undo`; docs/model.md, "Generation"): each token once, the
prompt once for every row, the rows in one batch, with no length limit
(past `block_size`). A `drafter` (`wpt_model.Drafter`, from
`load_drafter`) turns on speculative decoding: the same distribution,
fewer passes. Its scheduler needs `costs=`, the pass times a run measures
once (`measure_costs`) and records with the setup they were measured on
(`costs_setup`, `costs_file`), or replays on the same setup only
(`replayed_costs`); without them every draft is verified. For agentic
tasks, `wpt_agent.attempts` runs several attempts in one batch, and
`wpt_agent.play(...)` (what the task steps use) scores them one by one and hands each to the caller as soon as it
is scored. What a layer passes
on to later tokens is said once, in its `passes_on()` (`carry.py`), which
training's segments and slices (`sequence.py`) and the cache (`cache.py`)
both read: a change to a layer's computation changes its description
there, and `test_cache.py` and `test_parts.py` check that both still give
what the whole text gives (docs/model.md, "What a layer passes on").

## Rules that keep the checks green

- Write text files with `encoding="utf-8", newline="\n"`: Windows must
  produce the same bytes as Linux.
- Every change to the depot's layout (a table or a column of
  `wpt_depot/schema.py`) raises `FORMAT` there and adds to `STEPS` the
  upgrade step from the format before, with its test. A step names the
  columns it reads and writes itself, never from `SCHEMAS` (a later
  format's layout). It changes only the tables its layout change touches,
  and run again on the layout it leaves it changes nothing more. The
  current code must still apply an older format's undone journal changes
  on that older layout, since the upgrade finishes them first. A step that
  changes the journal table keeps backups and restores of older formats
  working: they read the journal's `id` and `done`, `about`'s `change` and
  `name`, and every table's storage versions (docs/architecture.md,
  "Format").
- Start external programs through `wpt_utils.programs.program(name)`,
  which reports a missing tool clearly.
- Print a line through `telemetry.log(text, level)` in a recorded run, else
  through `wpt_utils.lines.say(text, level)`, never `print`: every line starts
  with its time and level (docs/observability.md, "Printed lines"). Code that
  reads a line a command printed takes its text with `wpt_utils.lines.bare`.
- A training step's result holds `wpt_utils.programs.code()` (the commit
  and a hash of the uncommitted changes) in its fingerprint.
- Parallel work sizes itself with `wpt_utils.parallel.Sizing`, which
  measures each item's memory as it runs: never give a memory figure per
  item.
- Test isolation: a test never writes into the real store, cache or user
  settings; set `WPT_STORE`, `XDG_CACHE_HOME` and `LOCALAPPDATA` to the
  test's own folder when it touches them.
- A test passes both alone and under pytest-xdist. One that compares two
  trainings bit for bit, or runs several processes of a training on this
  machine, takes the `on_the_cpu` fixture of the root `conftest.py`
  (docs/architecture.md, "Checks").
- A test that must see something happen waits for that thing, not for a
  length of time, and a limit in seconds stands far above what it takes.
- The app's tests share their setup (a corpus built from a small depot,
  a tiny trained model) through the `built` and `trained` fixtures in
  `apps/wpt/tests/conftest.py`.
- A new third-party package in `uv.lock` or `Cargo.lock` gets a row in
  `docs/licenses.md` (version and license, read from its metadata), and a
  changed version gets its row checked again.
- The repository is public: never commit personal paths, machine names,
  credentials, or texts.
