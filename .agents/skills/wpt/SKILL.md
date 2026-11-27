---
name: wpt
description: How to work in the WPT repository (Workers' Pre-trained Transformer) with its own tools. Use it before changing the depot, the freight, the dock, an order, the model, training, an evaluation or a training step, before running any `wpt`, `depot`, `freight` or `dock` command, and before every push. It names the subcommand for each job, what each one reads and writes, how to extend evaluation and training, and the rules that keep the checks of `ci/` green.
---

# WPT tooling

WPT trains a language model only on content whose license is clear. The
texts live in a depot, an SQLite file known by its address, that holds the texts
and items, each of one kind, their licenses, notices and tags, the people's
decisions on them, and the snapshots, each a change number, named by the
UTC time it was made, and the texts admitted then (`depot`; docs/architecture.md). A hand-picked
selection of texts is a tag. A model family's data section names a
snapshot, with its change and digest, and the roles of the portions its
corpus is cut into: `held_out` (plain texts no training reads, always
first, its size in bytes fixed in `wpt_corpus.roles`, only with a
`total_tokens` of at least `HELD_OUT_MIN_TOKENS` there), `pretrain` (sized
by `total_tokens`) and `finetune` (conversations and tasks, sized by
`finetune_tokens`); left out, `held_out` and `pretrain`
(docs/data-selection.md, "The portions of a corpus"). The freight (`freight`;
docs/api/freight.md) reads a request, chooses items in a snapshot through
`depot.api`, runs one preparation program on them (`wpt-vocabulary`,
`wpt-tokens`) and leaves the result on the dock as a delivery. WPT asks it
for an order's vocabulary and corpus and trains from the dock; it never
reads the depot. The depot, the freight and the dock are separate parts,
meant to serve other models and data and to leave for a repository of
their own: each offers the others its `api` module alone, and
`packages/wic-utils/tests/test_part_imports.py` fails on any other import
between them (docs/architecture.md, "The parts"). The
repository holds the orders and the images of agentic tasks, never texts.
Everything built or trained lives in a dock named by the `DOCK`
setting (environment, or `.env` at the root): `deliveries/`, `corpora/`
(the corpora the depot built before the freight), `models/`, `runs/`,
`cache/` and `backups/`, plain named things. The documents in `docs/` hold
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
on every order and image of `data/`, read through the
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
--help` lists them); `depot web` serves the operations of `depot`
in a browser, on 127.0.0.1 only, and training, evaluation and the other
jobs run from the command line (docs/architecture.md, "The app" and "Two
interfaces").

| Job | Command | Reads | Writes |
| --- | --- | --- | --- |
| the depot: texts, items, decisions, snapshots, erasures | `uv run depot new ADDRESS ...`, `clone URL TO` (URL: the upstream's `depot web`; snapshots travel with their sets, checked against their digests; an upstream of an older format is refused: upgrade it first, then its clones), `pull`, `upgrade [--check | --yes] [--to FILE] [--upstream URL]` (alone or with `--check`, writes nothing in the depot: reads the depot's file read-only, may run while `depot web` serves, saves what it measured beside the file in `<depot file>.upgrade-check.json`, and says what bringing it up to the code's `FORMAT` would do: each locked library, each dataset revision the corpora of `DOCK` were built from and each current one made a snapshot, every revision each snapshot reads checked against what the library read, the names made snapshots' names, an order data section for each dataset revision, the cost of every step left (seconds, WAL, disk added, the next backup's full copy) and what to do before: `depot backup` right before, which only the old wpt restores, the upstream before its clones, every process that has the depot open (`depot web`, its jobs, a backup) stopped first; `--yes` runs that check again, taking the saved one while nothing was written since, and upgrades only when it passes, and refuses while another connection has the file open; a folder of format 3 or older becomes a new SQLite file, `<folder>.sqlite` by default, the folder left as it is, and a clone of those formats needs `--upstream`; an SQLite file is brought up in place, a long step in several transactions, the depot marked half upgraded until it ends; open libraries are not converted and the tables of libraries and datasets stay, unwritten; every other command, `depot web`, clone and pull refuse a depot of another format; run it again after a crash), `backup [--full]` (sends the change files waiting beside the depot, and a full copy on the file's first backup or with `--full`, onto the dock under `backups/<name>/`), `backup --off` (nothing backs the depot up: no change file is recorded until the next backup, which sends a full copy), `restore TO [--at MOMENT] [--depot NAME]` (a new depot file from the dock's backups, at a change number or a UTC time; never writes over a file), `sql QUERY` (SQLite's SQL; a list column is JSON text, `EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'book')`; `texts`, `about`, `erasures`, `tokenizers`: the current state; `<name>_revisions`: every revision; `contents`: every content version, whole (a query that reads the text of one bigger than one SQLite value fails; `depot show` reads it); `content_lists`: each version as the list of its parts; `parts`; `quotes`: the texts each content version of a conversation quotes; `changes`; the legacy tables of a depot brought up from format 12 or older (`libraries_revisions`, `memberships`, `pins`, `library_moves`, `datasets_revisions`), unwritten, which only the upgrade to format 13 reads; `text_decisions`: each decision on each text; `counts`: how many current texts have each status, tag, license, score name, score value and decision, and what the texts of each status hold by kind and plain tag set; `snapshots`, `snapshot_sets`: each snapshot, a name on a change number, and its texts as roaring bitmaps (depot.snapshots); `taggings`: each text's tags, each row a text and a tag with the change it joined it at and the change it left it at (null while it has it), the `tags` column of `texts` read from it; `text_tags`, `text_licenses`: the current texts of each tag and license; `text_titles`: an FTS5 index of the trigrams of the current titles, which stores no title), `add FILE [--kind KIND] --license L ...` (KIND `plain` by default, a text read as it is; any other kind is an item, JSON its kind's schema checks; a text's kind is fixed when it is added, and kinds are formats only: what a text is or where it comes from is a tag; docs/data-selection.md, "Kinds and items"), `license add ID NAME FILE`, `show ID [--from N --characters M]` (`--from`: one passage of a plain text, read from the parts it reaches alone, whatever the text's size), `decide admit\|refuse\|back-to-review WHERE` (`back-to-review`: admitted or refused texts under review again, their decisions starting over), `decide admit\|refuse --erasure ID`, `reason DECISION REASON` (an admission's or a refusal's public reason, given or replaced afterwards by the person who made it, the same name and email), `relicense ID ... --license L ...`, `content ID FILE [--kind KIND]` (a new content version, of the text's own kind: another kind is refused, add a new text instead; like `relicense`, the text's decisions start over), `content-file FILE` (FILE: one JSON line per plain text, `{"id": ID, "text": T, "was": V}`, `was` optional, the content version the line was written from, the text changing only while it is still its current one, at most 350,000 lines; each text changed as `content` changes one, its decisions starting over, its credits, licenses, tags and scores kept; one decision under review for each pair of old and new versions; a text whose content is that already, or whose `was` no longer holds, left out and counted; an item refused; docs/data-selection.md, "Texts"), `credits ID [--title T] [--author A ...] [--tag T ...] [--address A] [--notice FILE] [--clear CREDIT ...]`, `scores FILE` (FILE: one JSON line per text, `{"id": ID, "scores": [{"name": N, "value": V, "content_version": C}], "remove": [N]}`; one decision under review for every text it changes, which keep their status; docs/data-selection.md, "Scores"), `credits-file FILE` (FILE: one JSON line per text, `{"id": ID, "title": T, "authors": [A], "tags": [T], "address": A, "notice": N, "was": {CREDIT: VALUE}}`, each optional, null clearing it, `was` the values the line was written from, the text changing only while they hold, at most 350,000 lines; one decision under review for every text it changes, which keep their status; docs/data-selection.md, "Texts"), `score NAME (--value V \| --remove) WHERE` (one score on, or off, every text WHERE matches, as `scores` records it), `retag WHERE [--add T ...] [--remove T ...] [--yes]` (tags given to, and taken from, every text WHERE matches, as one decision under review, each text keeping its status; says the rows, bytes and seconds it would write, and writes only with `--yes`), `measure WHERE --out FOLDER [--lines N]` (reads only: the scores `symbols` and `running-text` of every plain text WHERE matches, from its text alone, as scores files for `scores` to record; docs/data-selection.md, "Measures"), `score-names` (every score name with how many texts have it, from the counts), `score-values NAME` (the values the score has across every current text, from the counts, each to three significant digits: how many have it, its smallest, largest and most frequent), `about [--maintainer M] [--takedown-contact C]`, `snapshot new [--note TEXT]` (the texts admitted now, as depot.snapshots makes them, named by the UTC time, `YYYY-MM-DDTHH-MM-SSZ`), `snapshot list`, `snapshot show NAME [--owner DEPOT]` (its row, summary and what an erasure did to it since), `jobs`, `jobs cancel ID` (the web app's queue, `<depot file>.jobs`), `erase ID [START END]`, `erasures`, `recount` (makes the depot's counts again from its rows, after copying again onto each revision the kind, length and bytes of its content version and onto each current one the copy of its text's tags) (`depot --help`; every decision, an add, an erasure request, a change of content, of credits and of scores included, takes `--name`, `--email` and `--accept`, the decider's declaration, public; a refusal ends with exit code 2 and writes nothing; a clone refuses every change to a row its upstream owns, docs/architecture.md, "Following an upstream") | the depot `DEPOT` names | the depot; for `backup`, the dock's `backups/`; an erasure carried out (`decide admit --erasure`, `pull`) marks nothing on the dock: `freight check` marks the corpora it reached |
| work on the depot in a browser | `uv run depot web [--port N]` (needs Bun and the design-system submodule, `git submodule update --init`), then http://127.0.0.1:8410/ (HTTP API described at `/api/openapi.json`; the data pages "Texts" (quick filters, the filter bar and a menu of the statuses, kinds, tags, licenses and saved filters, a Kinds menu in the filter bar, and a bulk bar that decides, scores, adds or removes a tag over the ticked texts; the list of the open texts at the left edge, out on hover; the current text's content, always editable, saved when the page moves to another text or decides on it, or by Ctrl+S; a review panel of score choices following the scale the depot's texts hold, a number pad that takes any number, tag toggles, presets kept in the browser and the moves and decisions, with keys; and a blob mode reading the open texts as one strip, each text's length in characters in the margin, "Biggest first" the longest first from the index `texts_length`), "Snapshots" (each with its summary, its name, change and digest, what an order's data section names it by; "Read it whole", its texts as one strip the longest first, from its order by length made once by a job of the queue's read lane, a click opening a text in a pane to edit or refuse it), the "Size map" (which tags, kinds and licenses of one status's texts, or of a snapshot, take the most characters, and the biggest texts, each opening its texts as a strip), "Jobs" and "Erasures", the "Add a text" page (the kind picked from cards, a conversation written in a chat editor whose excerpts are selected in a plain text), the setup pages "Licenses", "Depot" and "Settings", and "Help", what each page does, each row of their lists opening a view of that thing with its own address, the "Overview" and the "Add a text" button do what the `depot` commands do, except `new`, `clone`, `pull`, `upgrade`, `backup`, `restore`, `recount`, the change of `about` and cancelling a job, which are commands only; a snapshot and a change of tags over a selection each start only after their estimate is shown and confirmed (the route takes `confirmed: true` and the estimate it showed, 409 when it changed); a snapshot runs as a job of the queue's heavy lane, one at a time, the writes over many texts as jobs of its write lane and a snapshot's order by length as a job of its read lane, one at a time in each lane, which survive a restart; the depot reads no code repository and builds no corpus and no vocabulary, the freight does; it answers any client, with no login; a query or a filter's condition only reads the depot's file, on a read-only connection, and runs in a reader process, stopped after `SECONDS` and capped at `MEMORY` on Linux, `apps/depot/src/depot_app/web/readers.py`; the read-only routes `/api/follow/*` serve the clones and pulls of other depots) | the depot `DEPOT` names, `apps/depot/ui`, `vendor/wic-design-system`; the Translate button's models on the dock, which it keeps there the first time it fetches one from Mozilla | the depot through `depot`; the queue `<depot file>.jobs` and the jobs' folder `<depot file>.work` (an erasure carried out marks nothing on the dock: `freight check` marks stale a corpus it reached); `node_modules/` at the root, the design system's `packages/*/dist`, `ui/dist` |
| the old corpora and the models on the dock | `uv run wpt list` (the deliveries: `uv run freight deliveries`) | the dock | prints the corpora the depot built under `corpora/` before the freight, and the models |
| a file on the dock, by its key | `uv run dock find PREFIX`, `dock fetch KEY [--sha256 S] [--out FILE]`, `dock put KEY FILE` (docs/api/dock.md) | the dock `DOCK` names | `put`: the file under KEY; `find` prints `{key, bytes}` lines, `fetch` the bytes |
| a request to the freight: items chosen in a snapshot and prepared by one program (`wpt-vocabulary`: a vocabulary, its sample a size in bytes; `wpt-tokens`: a corpus) | `uv run freight request FILE [--wait]`, `work`, `jobs`, `job ID [--follow]`, `cancel ID`, `deliveries`, `delivery KEY`, `check` (docs/api/freight.md; the programs and their messages: docs/api/preparation.md) | the depot `DEPOT` names, read through `depot.api`; the request's inputs on the dock; `FREIGHT_HOME`, `FREIGHT_CAP` | the queue and the jobs' work in `FREIGHT_HOME`; the delivery under the dock's `deliveries/<delivery key>/`; for `check`, and before every job, what the depot's erasures since the last check did to the deliveries (stale, or noted in the note when its program says `on_erasure` `note`) and the stale marks of the corpora under `corpora/`, which nothing else marks |
| an order's vocabulary and corpus, asked of the freight | `uv run wpt build-resources FAMILY:SIZE` (the vocabulary first when the data section learns one, `{"size", "sample_bytes"}`, then the corpus, its input the vocabulary's delivery, cut into the portions of the order's roles, the held-out one first, `wpt-tokens` writing each portion's files under the folder of its name; a vocabulary named `{"sha256", "size"}` must be on the dock already; waits for both jobs, running the freight's queue here while no other process does; docs/tokenizer.md, "A vocabulary in an order") | the order; through the freight, the depot `DEPOT` names; `FREIGHT_HOME`; the dock `DOCK` names | the two deliveries on the dock; prints both delivery notes as JSON |
| copy a kept model out of the dock | `uv run wpt copy-out NAME FOLDER` (NAME: the run's name; FOLDER must not exist; every file checked against its sha256) | the dock's `models/NAME/` | FOLDER |
| follow a run | `uv run wpt runs`, `uv run wpt run FOLDER [--follow]` | the dock's run records | prints the runs, or one run's stage, counts, last events and errors |
| train on a built corpus in one command | `docker compose up` with `COMPOSE_FILE=deploy/manager/compose.yaml` in `.env` (or `uv run wpt manager`), `WPT_ORDER` required, `WPT_CORPUS` the corpus (unset: the order's, as `wpt train` finds it; the manager builds nothing, and a run with no corpus is refused, saying to run `wpt build-resources`), `WPT_RUN` the run's name (unset: a new one, written into the git-ignored `wpt-run.txt` with its copy-out line), `WPT_EVALUATE=1` to evaluate at the end; `WPT_GPU_WORKERS`: the gpu workers' keys, or how many to rent (rented workers run the training and the evaluation with `WPT_KERNELS=triton` unless the settings set `WPT_KERNELS`; `WPT_RENTAL_GPUS`, `WPT_RENTAL_KIND`, `WPT_RENTAL_PROVIDER`; a lost rented worker, or a crash on several workers, resumes the run up to `WPT_REPLACEMENTS` times; `WPT_BUDGET` in dollars, `WPT_MAX_HOURS`, or both: the first reached stops the run, its checkpoint saved, and the cost is printed before each rental; `wpt manager stop [RUN]`, or a SIGTERM (`docker compose stop`), stops a run on purpose the same way, and `WPT_STOP_DEADLINE` seconds later (default 600) the manager ends what has not ended and gives the machines back, docs/machines.md; the container exits when the runs are over, and `docker compose up` again continues a failed run); goes on first with its own runs a crash cut short; builds the images of the corpus's tasks where `DOCKER_HOST` reaches Docker; settings from the environment, else from `.env` (docs/machines.md) | the order, the dock | the checkpoint and the model under `models/NAME/` (the manager's own copy of the folder removed); the tasks' images on the Docker host; its run record `runs/<run>/`; its runs' gpu workers, tokens included, in `WPT_MANAGER_STATE` (never the dock); rentals at the provider, stopped once the evaluation or training ends, and tried again until stopped |
| a gpu worker on a machine with GPUs | `docker compose -f compose.yaml -f compose.nvidia.yaml up` (or `compose.amd.yaml`; the files of `deploy/gpu-worker/` plus `wpt_worker/` and `wic_utils/` copied beside them; prints its key `wpt-worker://TOKEN@HOST:PORT#PIN`; give it, one per machine, to `wpt train` in `WPT_GPU_WORKERS`; docs/machines.md) | what the wpt manager sends it | its own folder (certificate, checkout, libraries, checkpoints); never the dock |
| check a machine before work | `uv run wpt doctor` (the dock answers, and its credentials, the GPU and the PyTorch build, the attention kernels and nvcc, the Rust toolchain, free disk, uncommitted or untracked files; exit code 1 when something is missing) | the settings, the dock, the checkout | prints what is missing and how to fix it |
| train | `uv run wpt train ORDER [CORPUS_ID] [--out FOLDER] [--stream] [--recompute [--recompute-from N] | --no-recompute] [--compile | --no-compile] [--no-drafter] [--segment N] [--run NAME] [--scheduler-costs RECORD_FOLDER] [--kernels FORM] [--stop-after-minutes M] [--stop-after-updates N] [--profile-update N] [options]`, started detached (skill `wpt-training-run`; ORDER: `FAMILY:SIZE`, a size of a model family, which reads the start of the corpus the family's sizes share, or `FAMILY:SIZE:RUN`, another run of that size (`branch-SHARE`, a decay branch that goes on from the size's trunk; `lr-FACTOR`, at the smallest size, the learning-rate check); CORPUS_ID (or `WPT_CORPUS`), a corpus id or a start of it, of a corpus delivery or of the dock's `corpora/`, which must be made from the order's snapshot (name, change, digest) with its portions, shares and vocabulary, at any size (a corpus without portions, delivered without them or under `corpora/`, is all of it the pretraining portion and needs the order's shares); without it, the corpus delivery whose key the order gives (`wpt.resources.corpus_key`: the snapshot's change and digest, the portions of its roles, shares, `data.total_tokens` as the pretraining portion's size in tokens, a series its largest size's, and the key of the vocabulary's delivery, worked out from the order alone), else, for a vocabulary named by its sha256, the latest corpus the depot built into `corpora/` for that snapshot, depot, data section and size; no depot is read; none fresh refuses the run ("run `wpt build-resources ORDER`") and starts nothing; the vocabulary is the corpus's (a delivery's: the file of the vocabulary delivery it names); an order that names a dataset instead of a snapshot, and a corpus of `corpora/` built from a dataset, are refused (docs/training.md, "Models trained before snapshots"); each step reads its role's portion (text windows the pretraining one, conversations and tasks the fine-tuning one) and nothing reads the held-out portion: training's record names none of its files, and `CREDITS` none of its texts; `data.total_tokens` is the size of the pretraining portion the order asks for and a ceiling on what it reads: a corpus built at that size (its `max_tokens`) may hold fewer tokens of plain text (what the snapshot leaves the portion; a corpus without portions gives its conversations and tasks their share of that size), and each step reads at most what it holds (a series: each size reads the smaller of its prefix and what the corpus holds), the run's record saying how many when fewer; a stale corpus (an erasure reached a text it holds) and one with fewer tokens of plain text than `data.total_tokens` built at another size are refused; `uv run wpt train series NAME` prints the numbers of every size of a series and trains nothing; a corpus of another snapshot, portions, shares or vocabulary is refused, and the vocabulary is the corpus's; `--stop-after-minutes`, `--stop-after-updates`: stop as a signal does, the first reached, and print the tokens per second and the projected time of the run (docs/training.md, "A limit and a measure"); `--profile-update N`: update N of each step runs under the profiler and its run record receives `profile.txt` (same section); trains through gpu workers, the keys of `WPT_GPU_WORKERS` (one per machine), or else a local one it starts; exit code 4 when a gpu worker is lost; `WPT_LINK_CHECKPOINTS`: `step-ends`, the default, or `every`, the checkpoints that come back from the worker; docs/machines.md; runs the order's `steps` in order, starting from the result of the last of its first steps the dock already holds, so a run reuses the steps it shares with an earlier one; a corpus that holds nothing for a step refuses the run before any step trains; a step that cannot do its part (`StepFailed`) stops the run with exit code 2, no result of it kept but the checkpoint it wrote, which `--resume` continues (with `WPT_LINK_CHECKPOINTS=every`; with `step-ends` it stays on the worker and `--resume` starts the step over), the results of the steps before it kept; ends by training the model's drafter; `--scheduler-costs`: replay the drafter scheduler's pass times an earlier run recorded on the same setup; a checkpoint is always written just before the drafter, so a drafter failure resumes with `--resume`; `--stream`: a few shards on disk at a time; unasked, each stage of text windows chooses how its passes run from four short probe passes that measure the GPU's memory without recomputing and with it (5 % more per token, a tenth of the GPU kept free): every activation kept where its pass fits, else the layers recomputed, else fewer windows per pass (the largest number that divides the process's share of the batch; never with `--micro-batch`), else one window per pass in segments (the longest multiple of 128 that fits; never with `--segment`), and it logs the choice and its numbers; a stage no pass fits is refused with exit code 2 before it starts; `--recompute`: recompute in every stage, much less memory, slower (the window layers' attention and the experts run again in the backward pass; the exact and sparse layers' attention runs once); `--no-recompute`: in none; `--no-drafter` (`WPT_DRAFTER=0`): end without training the drafter, the weights the same, the folder's `provenance.json` saying so (`wpt chat` then writes alone, `load_drafter` refuses it), and the same command without it later trains the drafter from the kept steps; `--recompute-from N`: with `--recompute`, only the stages of windows of at least `N` tokens run their layers again, shorter ones keep every activation; the passes over whole text windows run through `torch.compile` wherever they can (a CUDA device with Triton and a C++ compiler), else uncompiled with one log line; `--compile`: refuse the run where they cannot; `--no-compile`: uncompiled (docs/training.md, "Compiling the passes"); `--segment N`: windows longer than N tokens run as segments of N, the same training in the memory of N tokens plus what the segments hand on, up to twice the time; with `--strategy split`, each process's slice runs in segments of N; `--run`: the checkpoint on the dock, so another machine can continue; `--kernels`: `triton`, `tilelang` or `pytorch` computes the attention, one that cannot work refusing the run (unset: the first kernels installed for the GPU; `wpt manager` names `triton` on rented workers); docs/model.md, "Attention") | the model order, the dock | a model folder, with `CREDITS`, the texts it learned from; each step's result, the dock's `cache/steps/<fingerprint>`; its run record `runs/<run>/` (one per command, over restarts); token files in the `scratch/` folder of the gpu worker that reads the corpus; with `--run`, the record `models/NAME/checkpoint.json` and the state it names, beside it in `models/NAME/`, until the model is saved |
| train on several GPUs | `uv run wpt train ORDER CORPUS_ID --strategy ddp --processes N [options]`, or `--strategy split` to split each window between the GPUs (windows too long for one; DeepSeek-V4's contextual parallelism) (on the gpu worker's machine) | the dock | the same |
| evaluate | `uv run wpt eval FOLDER [EVAL ...] [--kernels FORM]` (from inside the repository; the built-in `bits_per_byte` measures the model on the held-out portion of its corpus; a model whose corpus has none is not measured, its result saying so with no figure, when every built-in test runs, and refused, stopping the run, when the test is named; the model runs on the first gpu worker of `WPT_GPU_WORKERS`, else a local one; `--kernels tilelang`, `triton` or `pytorch`, or `WPT_KERNELS`, names what computes the attention on the gpu worker, and one that cannot work there refuses the evaluation) | the folder, the dock | `FOLDER/evals/<name>.json`, `README.md`; its run record `runs/<run>/` |
| lock an image of agentic tasks | `uv run wpt agent lock-image NAME [--force]` (needs Docker; `--force` replaces an existing lock) | `data/images/NAME/Dockerfile`, the base image's registry, the dated package archive | `data/images/NAME/lock.json` and `licenses.json` (commit them) |
| build an image of agentic tasks | `uv run wpt agent build-image NAME` (needs Docker; before `wpt train` with `finish-tasks` or `finish-tasks-with-less-writing`, and before `wpt eval`'s agent measures) | `data/images/NAME/`, the base image's registry, the dated package archive | a local Docker image |
| continue text | `uv run wpt chat FOLDER` (`--kernels tilelang`, `triton` or `pytorch`, or `WPT_KERNELS`, names what computes the attention, and one that cannot work stops the command; `--about` prints its NOTICE; `--chat` talks to a model trained on conversations; `--scheduler-costs RECORD_FOLDER`, its `<run>`, reads the scheduler's pass times a run recorded on the same setup instead of measuring them) | the folder; with `--scheduler-costs`, the dock | |

Everything that changes a model comes from its model family
(`data/orders/NAME.json`: data, architecture, seed, the training `steps`
in order, the drafter, written once, then the table `sizes` of the
settings that vary between sizes, each column a dotted path such as
`model.d_model` or `steps.1.optimizer.settings.lr`), from which
`wpt_orders.family` resolves each size's model order (docs/model.md,
"Model families"). `uv run wpt train family NAME` prints each size's
parameters, updates per stage, tokens and compute, and trains nothing;
read it after changing a family. Each step is
`{"step": NAME}` with, when they differ from the step file's, its
`optimizer` and that optimizer's settings, its `settings` and the parts it
keeps `locked`, named as the model names them (docs/model.md, "Parts";
docs/training.md, "Steps"), and `continues: true` when it goes on from the
whole state the step before it left (docs/training.md, "Continued
steps"). The built-in steps are `read-text`,
`answer-as-assistant`, `answer-as-assistant-dft`, `answer-as-assistant-neftune`,
`finish-tasks` and `finish-tasks-with-less-writing`.
The `read-text` step's `stages` list the window lengths in order, each
with its `share` of the tokens; `tokens_per_update` is global (over every
process), and the step computes each stage's updates and batch from the
shares, the size's tokens and `readings`, so a family writes neither
(docs/training.md, "Stages"); the size gives `prefix_tokens`; the model's context is the
last window length of the last step that reads text windows. Segments and split windows give
the whole window's loss and gradients, so they are options, never part of
the order (docs/training.md, "Long windows"). The `wpt train` options only
say how the hardware runs it, and each also comes from `WPT_<OPTION>` in the
environment (`--micro-batch` -> `WPT_MICRO_BATCH`); `--help` on any command
lists them. A crash restarts the processes on the gpu worker on their own, up to
`--max-restarts` times (default 3), from the last checkpoint; Ctrl+C or
`kill` finishes the current update and saves first, and that checkpoint reaches
the manager whatever `WPT_LINK_CHECKPOINTS` says; after a stop, the same command
with `--resume` continues from `FOLDER/checkpoint/` (or, with `--run NAME`,
from the dock, on any machine). Exit codes: 2 refused, 3 the run's record
could not be written, 130 stopped.

Every run of `wpt train`, `wpt eval` and `wpt manager` writes its record
into its own folder of the dock, `runs/<UTC time>-<command>-<main
id>-<random>/` (main id: a short corpus id, a short weights hash or the
run's name): `start.json`, `events.jsonl` (spans, events, every printed
line, with its level and without the time and level it prints with),
`ids.json` (the ids read and written, which chain the runs together),
`end.json`. With `OTEL_EXPORTER_OTLP_ENDPOINT` set (environment or
`.env`), the same goes live to an OpenTelemetry backend
(`deploy/otel-lgtm/compose.yaml`). A backend failure is one warning; a
record the dock refuses stops the command with exit code 3 (`wpt train`
first finishes its update and saves its checkpoint). A training step's
environment records its own events and files through `data.telemetry`, and
an evaluation through `context.telemetry`.
Details: `docs/observability.md`.

## Upgrading a depot

A depot of an older format is brought up with `depot upgrade`, which
runs every step between in one run. Run it only when the person asks:
it rewrites the depot, and the app must be stopped meanwhile.

1. `uv run depot upgrade --check` (or with no flag). It writes
   nothing. Read its output to the person: what converts (each locked
   library and each dataset revision a corpus was built from becomes a
   snapshot; open libraries stay in the legacy tables), its checks, the
   names, the order data sections it prints, and the cost (seconds, WAL,
   disk, the full copy the next backup sends). Set `DOCK` first, so
   the dataset revisions its corpora were built from convert too.
2. `uv run depot backup`, right before the upgrade. Only the wpt of
   the old format opens that backup: restoring it means checking out a
   commit of that format first.
3. `uv run depot upgrade --yes`. It runs the check again and upgrades
   only when the check passes; a failed check writes nothing (exit code
   2). Run it again after a crash: it goes on from the stage it reached.
4. The server first, then its clones: a clone of the new format refuses
   an upstream of the old one.

The server's runbook, with its containers, is docs/depot-web.md,
"Setting up the host".

## Texts in the depot

Every rule on licenses lives in the documentation: `docs/licenses.md`
and `CONTRIBUTING.md`. Read them before adding or reviewing texts. Never
apply a rule they do not hold: a case they do not settle is a question for
the maintainer, and the answer goes into the documentation as a general
rule about the current state, with no date and no source named
(CONTRIBUTING.md, "Writing about the project"). A new text starts under
review; only a person admits or refuses it, with a declaration of their
legal name and email (`depot decide`), and no program decides.

## The service layer

`wpt_orders/orders.py` (`Orders`) is the only code that reads or
writes `data/` (docs/architecture.md, "The service layer"): the model
families and the images of agentic tasks (Dockerfiles, their locks, license lists and `tools.json`);
it writes only the generated locks and license lists, for the command
that makes them. Code elsewhere asks it for documents and bytes, never for
a path under `data/`; `packages/wpt-orders/tests/test_data_access.py`
fails otherwise, and the audit hook of `wpt_orders/guard.py` refuses such
an access at run time (`DataAccessRefused`), except from tests. A new
operation on the depot gets both interfaces, a `depot` command and a
`depot web` route with its JSON Schema, both thin controllers over
`depot`. A pull, cancelling a job and a change of
the depot's maintainer or takedown contact are commands only, with no route (docs/architecture.md,
"Two interfaces").

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

A vocabulary never changes once learned: adding texts never retrains it
(docs/tokenizer.md). An order's data section names its vocabulary by its
sha256, a vocabulary the dock holds, or learns one, `{"size",
"sample_bytes"}`, which `wpt build-resources` asks the freight for.
Changing either changes every later model of the order and makes it
incompatible with earlier ones, so it is a person's decision: never change
it unasked. A change of the tokenizer's code asks for no new vocabulary.
The tokenizer recipes and vocabularies a depot may hold stay read only.
Some docks hold some of those vocabularies as the delivery
`adopted-<sha256>`, which an order finds by its sha256; nothing makes
new ones, and an order naming one a dock lacks is refused. A vocabulary
delivery is never made stale by an erasure: the erasure goes into its note.

## The model folder

`wpt train` writes `config.json`, the weights (each safetensors header
carries `license`, `output_license`, `notice`, `tokenizer` and
`tokenizer_sha256`), `tokenizer.json` (loaded only through
`wpt_model.load_tokenizer`, which checks it against those headers),
`provenance.json` (what produced it: `data`, the order's data section
with the depot, the snapshot, its change and digest; `deliveries`, the
key, id and preparation of the vocabulary's and the corpus's deliveries,
and the corpus's id and delivery key (a model trained on a corpus the depot
built: its id and build key); the order and the repository commit; and
how much of the corpus each step learned; a folder written before
snapshots names its corpus's id, depot and dataset instead, and every
reader takes the three shapes through `wpt_model.provenance`, never
rewriting a folder), `CREDITS` (every text it
learned from), and `LICENSE`, `NOTICE` and `README.md` (the model card).
`NOTICE` and `README.md` come from `wpt_model.notice.write_documents`, which
reads only the folder's own files: change what they say there, never by
editing a generated folder. `wpt train` writes the folder beside its
target and renames it into place; training into an existing folder replaces
it as a whole (its `evals/`, checkpoint and any other file go with it).
Every model folder holds `drafter/` (`config.json`, weights with the same
header, `provenance.json`): the drafter for speculative decoding, which
`wpt train` trains at the end of every run against the final weights. A model and its drafter are one unit: `load_model` and
`load_drafter` refuse a folder whose drafter is missing or names other
weights' sha256 (`DrafterMismatch`); the one exception is a folder that
`wpt train --no-drafter` saved, whose `provenance.json` says
`"drafter": false` among its `options` (`load_model` loads it,
`load_drafter` refuses it, `wpt chat` writes alone); a
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
                   # writer, greedy, losses, ranks), tokenizer, device, folder, provenance, dock,
                   # telemetry (the run's record)
    ...
    return Result(data={...}, text="the same findings, for people")
```

Every evaluation returns `Result`: `data` that JSON can hold, and `text`
that the model card shows. One that cannot measure this model raises
`wpt.eval.NotMeasured(reason)`: a run of every built-in test records it as
not measured and goes on, a run that names it stops. Details:
`docs/evaluation.md`.

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
KINDS = ("plain",)  # the kinds of texts it can read (depot.kinds)
SETTINGS = {...}
def start(settings, data): ...  # a wpt_training.Environment: stage_list, tokens, learned, batch (None: the update
                                # is skipped), evaluates, evaluate, after_update, finished, state_dict, close;
                                # learned: by kind, {kind: how much}; data: the step's name, the kinds it
                                # reads, dock, corpus record, tokenizer, config, strategy,
                                # telemetry, log, seed, the step's number, step_seed (both mixed:
                                # what the step's draws are seeded by); optionally check(settings),
                                # context(settings) and check_corpus(settings, record, dock, kinds):
                                # ValueError when the corpus holds nothing of those kinds for it, before
                                # any step runs

# optimizers/NAME.py: how the weights move, with its schedule
SETTINGS = {...}
def start(model, settings, run): ...  # a wpt_training.Optimizer over run's parameters; optionally check(settings)

# steps/NAME.py: puts them together
METHOD = "next-token"
ENVIRONMENT = "text-windows"
KINDS = ("plain",)  # the kinds of texts it reads, each one its environment reads: read-text plain,
                    # answer-as-assistant conversation, finish-tasks* task
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
raises `wpt_training.StepFailed(reason)`. The step keeps its result on the dock
under a fingerprint of everything that decides it: the kinds of texts the
step reads (the record's `fingerprints`, by kind: plain, conversation or
task; the corpus keeps its token files by kind, `shards`), not the whole
corpus, and never the hardware options (`--stream` included). After the last step, every run trains a drafter for the final
weights, unless `--no-drafter`; the drafter is not a step. Details: `docs/training.md`, "Steps".

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
- Every change to the depot's layout (a table, a column or an index of
  `depot/schema.py`, or the schema of a kind of `depot/kinds/`) raises `FORMAT` there and adds the upgrade step
  from the format before (`depot/upgrade.py`), with its test. A step
  names the columns it reads and writes itself, never through the current
  layout (a later format's). It changes only the tables its layout change
  touches, and run again on the layout it leaves it changes nothing more
  (docs/architecture.md, "Format").
- Every table of the depot declares a primary key (`KEYS` in
  `depot/schema.py`): the sessions that record each change for the
  backups name a row by it.
- Start external programs through `wic_utils.programs.program(name)`,
  which reports a missing tool clearly.
- Print a line through `telemetry.log(text, level)` in a recorded run, else
  through `wic_utils.lines.say(text, level)`, never `print`: every line starts
  with its time and level (docs/observability.md, "Printed lines"). Code that
  reads a line a command printed takes its text with `wic_utils.lines.bare`.
- A training step's result holds `wic_utils.programs.code()` (the commit
  and a hash of the uncommitted changes) in its fingerprint.
- Parallel work sizes itself with `wic_utils.parallel.Sizing`, which
  measures each item's memory as it runs: never give a memory figure per
  item.
- Test isolation: a test never writes into the real dock, cache or user
  settings; set `DOCK`, `XDG_CACHE_HOME` and `LOCALAPPDATA` to the
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
