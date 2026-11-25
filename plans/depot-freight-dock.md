# Plan: depot, freight, dock and WPT

This file is a plan, not a description of the repository. It says what to
build. Delete it in the change that finishes the work, once `docs/` and the
skills describe the result (CONTRIBUTING.md, "Writing about the project").

Each statement below is one of two kinds:

- **Settled**: the maintainer decided it. Do not change it without asking.
- **Proposed**: the writer of this plan chose it to leave no gap. It may be
  changed when the code shows a better way, and the change is reported.

A heading marked (settled) or (proposed) covers everything under it, unless
a line says otherwise.

## 1. The problem (settled)

A vocabulary of 16,384 entries, learned on the mix 35 % research papers,
15 % books, 5 % government, 5 % code, 40 % encyclopedia, cannot be made
today. A tokenizer recipe names a snapshot and a number of bytes, with no
shares, and the snapshot `balanced` holds 1,175 GB of code out of 1,472 GB.
The shares live only in the data section of a model recipe.

The cause is wider than the tokenizer:

- The package `wpt-depot` holds two jobs: keeping what people gave and
  decided, and knowing how to build what a model trains on (shares, groups,
  tokens, vocabularies, corpora).
- The heavy work runs inside the depot's web app, which was once killed for
  taking 62 GB of memory.
- One word, `balanced`, names a snapshot, another snapshot (`balanced-r2`),
  a dataset of the legacy tables and a tokenizer recipe. None of them is a
  mix. The mix itself has no place of its own.
- "Tokenizer recipe", "version" and "vocabulary" are three words for one
  result.

## 2. The picture (settled)

A person writes an order. WPT reads it and sends requests to the freight.
The freight takes what each request needs from the depot, runs one
preparation on it, and leaves the result on the dock with a delivery note.
WPT trains the model from what is on the dock.

| Part | Name in code | What it does | What it never knows |
| --- | --- | --- | --- |
| Depot | `depot` | Keeps what people gave and decided: texts and items, licenses, tags, scores, decisions, snapshots, erasures. Backed up, compressed, read in SQL. Nothing in it can be made again. | Models, tokens, vocabularies, the freight, the dock. |
| Freight | `freight` | Reads a request, chooses in a snapshot (kinds, shares, the depot's order), runs one preparation, delivers on the dock, finds a delivery again, marks what an erasure reached. | What a token or a vocabulary is. |
| Dock | `dock` | Keeps what machines made: deliveries, models, run records, cached results. Three gestures: find, fetch, put. Everything in it can be made again. | Everything else. |
| Factory | WPT (keeps its name) | Builds the model: training, evaluation, generation. Reads the order, sends the requests, provides its preparations. | The depot: WPT never talks to it. |

`depot`, `freight` and `dock` are meant to serve other kinds of models
(diffusion, world models, convolutional networks) and other kinds of data
(image, video, sound), and to leave this repository for their own. They may
be written again in another language. The documentation says so. For now
they live here, as independent as one repository allows, each with its own
command.

### The dependency rule (settled)

- `depot` imports none of the other three.
- `dock` imports none of the other three.
- `freight` imports `depot.api` and `dock.api`, nothing else of them.
- WPT imports `freight.api` and `dock.api`, never `depot`.
- None of `depot`, `freight`, `dock` imports a WPT package.

A test enforces it, written like
`packages/wpt-recipes/tests/test_data_access.py`: it reads the imports of
every module and fails on a forbidden one.

### The module `api` of each part (settled)

What one part offers the others is one module named `api`: `depot.api`,
`freight.api`, `dock.api`. It is the only module of a part that code
outside the part may import, and the same test fails on any other import
between parts. Every function of an `api` takes and gives only values that
JSON and byte streams can carry, and hands out no connection, row object or
path into the part's own files, so that it can become REST routes without
its callers changing. A part's own command and web app are inside the part
and may import all of it.

Each contract is described in a document of its own: `docs/api/depot.md`,
`docs/api/freight.md`, `docs/api/dock.md`, and `docs/api/preparation.md`
for the messages between the freight and a preparation (section 7.2). A
change to an `api` changes its document in the same commit.

Proposed: the helpers the three share today through `wpt_utils`
(`parallel`, `lines`, `files`, `programs`, `web`, the reading of `.env`)
move to a neutral package `wic-utils` (module `wic_utils`), which any part
may import. `wpt_utils` keeps what only WPT uses (`device`,
`settings.typed`). One exception to the rule is listed in the test:
`depot`'s backup writes through `dock`'s storage code until the backups
have a place of their own (section 12).

## 3. The words (settled)

| Today | Becomes | Why |
| --- | --- | --- |
| model recipe, series recipe (`recipe`) | order (`order`) | The JSON file holds parameters only. It names the ways of doing and holds none. |
| (nothing) | request | One piece of work WPT asks of the freight: a selection and one preparation. |
| (nothing) | preparation (`preparation`) | The program the freight runs on the items it chose. A way of doing, not a parameter. |
| training step (`step`) | unchanged | |
| store (`wpt-store`, `WPT_STORE`) | dock (`dock`) | "Store" did not say how it differs from the depot. |
| corpus record, build key | delivery note, delivery key | What the freight delivered for a request, and where. |
| corpus, vocabulary | unchanged | Each is a delivery. Neither has a free name. |
| tokenizer recipe, its version | gone | A vocabulary is asked for in the order. |
| snapshot named by a person | snapshot named by the depot, by its UTC date and time | A snapshot says when, nothing else. |
| mix | no name | It is read from the delivery note. |

A delivery is shown by what made it, never by a free name: `vocabulary
16,384 · snapshot 2026-11-23T02-27-38Z · research-paper 35 / books 15 /
government 5 / code 5 / encyclopedia 40 · 10 GB`.

## 4. The order

### 4.1 The file (settled, names proposed)

`data/recipes/` becomes `data/orders/` (`models/`, `series/`), the package
`wpt-recipes` becomes `wpt-orders` (module `wpt_orders`, class `Orders`),
and every text of the repository says "order" where it said "recipe" for
these files. The service layer rule is unchanged: `Orders` is the only code
that reads `data/`.

### 4.2 The data section

Settled: the data section keeps its shape, and its `vocabulary` takes one of
two forms.

```json
"data": {
  "depot": "<the depot's public address>",
  "snapshot": "<name>", "change": 32118, "digest": "<sha256>",
  "kinds": ["conversation", "plain", "task"],
  "shares": [{"tag": "research-paper", "share": 0.35}, {"tag": "books", "share": 0.15},
             {"tag": "government", "share": 0.05}, {"tag": "code", "share": 0.05},
             {"tag": "encyclopedia", "share": 0.4}],
  "total_tokens": 1179648000,
  "vocabulary": {"size": 16384, "sample_bytes": 10000000000}
}
```

- `{"size": N, "sample_bytes": B}`: a vocabulary of N entries learned on B
  bytes of this section's own snapshot and shares. The order never holds its
  sha256: the delivery note gives it.
- `{"sha256": S, "size": N}`: the vocabulary of that sha256, which the dock
  must hold. This is how an order keeps the vocabulary of an earlier model,
  and how the orders written before this plan stay valid.

Proposed: there is no third form that gives the vocabulary a mix of its
own. A vocabulary on another mix is made by a request of its own (section
6.2) and named by its sha256.

The legacy shape that names a dataset stays as it is
(docs/training.md, "Models trained before snapshots").

## 5. What the freight asks of the depot (settled)

The freight never opens the depot's file. It calls the functions of one
module of `depot`, `depot.api`, which are today's
calls and tomorrow's REST routes: each takes and gives only values that
JSON and byte streams can carry, and none hands out a connection, a row
object or a path.

1. **A snapshot.** By name or id: its id, name, change (`at_change`),
   digest, the time it was made, and its summary (texts, characters and
   bytes by kind, tag, plain tag set and license, and texts by owner). Today:
   `wpt_depot.snapshots.snapshot`.
2. **Its listing.** Every text of the snapshot in the depot's order, without
   content, from a place on: owner, number, id, piece (a long plain text
   comes as its pieces), revision, kind, tags as of the snapshot, bytes.
   The order and the cutting into pieces belong to the depot. Today:
   `wpt_depot.order.of` and `walk`, without `pool` and `taken`, which are
   the freight's (`pool_of` moves to the freight).
3. **Content.** For entries of the listing, the content of each as bytes, in
   the order asked: a whole text, one piece, or a span of characters of a
   content version (what a conversation's excerpts quote). Read from the
   text's parts, never whole in memory for a long text.
4. **Changes since a change number.** The erasures carried out after it:
   for each, its id and the texts it removed or cut. Today:
   `wpt_depot.snapshots.erased_since` and the erasures' rows.
5. **Credits.** For texts at given revisions: title, authors, address,
   licenses, notice id, scores; and the text of the licenses and notices
   they name.

The depot's summary gives the census (`wpt_depot.corpus.census`) with no
further request.

## 6. Requests and deliveries

### 6.1 A request (settled in substance, shape proposed)

A request is one JSON document: where to choose, what to choose, and the one
preparation to run on it.

```json
{
  "depot": "<public address>",
  "snapshot": {"name": "<name>", "change": 32118, "digest": "<sha256>"},
  "selection": {
    "kinds": ["plain"],
    "shares": [{"tag": "research-paper", "share": 0.35}],
    "size": {"unit": "bytes", "amount": 10000000000}
  },
  "preparation": {
    "name": "wpt-vocabulary",
    "settings": {"vocab_size": 16384},
    "inputs": {}
  }
}
```

- `selection.size` is `null` (everything the rules take), a number of bytes,
  or a number in the preparation's own unit (`"unit": "tokens"` for WPT),
  which only the preparation can measure.
- `preparation.inputs` names earlier deliveries by their delivery key. The
  freight fetches them from the dock and hands their folders to the
  program. It knows nothing else of the link between two requests.
- A request names exactly one preparation. The freight never chains two.

### 6.2 The delivery key (proposed)

The delivery key is the SHA-256 of the canonical JSON of: the snapshot's
digest and change, the selection in canonical form (`canonical` of today's
`corpus.py`, with the size's unit), the preparation's name, the version the
program answers (section 7.2), its settings, the delivery keys of its
inputs, and the version of the freight's selection rules (`RULES`). It
never covers the depot's address, the snapshot's name, the code's commit or
the machine. So:

- the same request lands in the same folder wherever it runs;
- WPT works out the key of each request from the order alone, with no
  depot and no freight at hand, which is how `wpt train` finds its corpus.

A request whose delivery the dock holds, fresh, is not run again: the
freight answers its delivery note at once (settled).

### 6.3 A delivery on the dock (proposed)

```
deliveries/<delivery key>/
    delivery.json          the delivery note, written last
    items/part-NNNNN.parquet   the items taken, in order, with their credits
    licenses.parquet, notices.parquet
    files/...              what the preparation wrote
```

The delivery note holds: `key`; `id` (covers the key and the content of
every file, as a corpus id does today); the request as given; the snapshot's
id, name, change and digest; the preparation's name, version and unit;
`result`, the JSON the preparation answered; every file with its key,
sha256 and bytes; the count of items and pieces; how much each share and
each group of a share holds; what was left out and why (`left_out`); the
erasures that had reached the snapshot; `made_at`; the code's commit; and
later `stale` or `deleted`, as a corpus record has them today.

The items tables are today's texts table of a corpus (`CORPUS_SCHEMAS`),
with `weight` in the preparation's unit where it says `tokens`. They are
how a vocabulary keeps the list of the texts it read, in the order it read
them, and how an erasure finds the deliveries that hold a text.

A failed job leaves no delivery note.

### 6.4 The selection rules (settled: unchanged)

The freight chooses exactly as `wpt depot build` chooses today
(docs/data-selection.md, "Shares", "Filling a part", "Long texts",
"Corpora", "Conversations that quote texts"): the first share whose tag a
plain text has, the groups of a share weighed by the square root of their
bytes, a part closed at a thousandth or after a thousand texts that do not
fit, the items' natural share of a size, a conversation only with every
passage it quotes, a text once at most. This plan moves that code and
changes none of its results.

A size in bytes (a vocabulary's sample) fills each share's proportion of
it, by group, on bytes, as a corpus without a size chooses on bytes. This is
the rule of `wpt_depot.legacy.sample` applied to a snapshot. The current
`wpt_depot.corpus.sample`, which ignores shares, goes.

## 7. Preparations

### 7.1 A program (settled)

A preparation is a program, not a module the freight loads. The freight
starts it on its own machine, as a child process, and talks to it through
its standard input and output. One job starts one program. The program
never reads the depot and never writes the dock: it reads what the freight
sends, writes files into the local folder the freight names, and says what
it wrote.

The freight finds a preparation by its name among the programs installed
beside it (proposed: a console script of that name on the path, which a
Docker image of the freight with WPT's preparations provides).

### 7.2 The messages (proposed)

Each message is one line of JSON in UTF-8, ended by a line feed. A message
that carries content names its length in `bytes`, and exactly that many
bytes follow the line. The format is independent of the channel.

From the freight:

| Message | Meaning |
| --- | --- |
| `{"type": "start", "settings": {...}, "inputs": {"<name>": "<local folder>"}, "out": "<local folder>", "threads": N}` | first message; `threads`: how many the program may use |
| `{"type": "weigh", "ref": R, "kind": K, "bytes": B}` + content | how much does this item weigh in your unit? |
| `{"type": "take", "ref": R, "id": I, "piece": P, "kind": K, "pool": T, "bytes": B}` + content | prepare this item; items come in the order of the delivery |
| `{"type": "end"}` | no more items |

From the program:

| Message | Meaning |
| --- | --- |
| `{"type": "ready", "name": "...", "version": "...", "unit": "tokens", "kinds": [...], "weighs": true}` | answer to `start`; `kinds`: the kinds it prepares; `weighs`: whether it answers `weigh` |
| `{"type": "weight", "ref": R, "weight": N}` | answer to `weigh`, in any order |
| `{"type": "taken", "ref": R, "weight": N, "at": {...}}` | answer to `take`, in the order of the `take` messages; `at`: where the item lies in the files, copied into its row |
| `{"type": "file", "path": "...", "items": N}` | a file under `out` is complete; the freight uploads it, deletes the local copy and keeps a checkpoint |
| `{"type": "log", "level": "...", "text": "..."}` | a line for the job's log |
| `{"type": "done", "result": {...}}` | answer to `end`, after its last `file` |
| `{"type": "failed", "reason": "..."}` | the program cannot do its part; the job fails with that reason |

Rules of the exchange:

- The same messages in the same order give the same files, byte for byte
  where the format allows, and always the same content hashes, on any
  machine and any number of threads.
- A `weight` never depends on the other items.
- A program that exits without `done` or `failed`, or writes a line that is
  no message, fails the job.
- A conversation is sent with each excerpt written as the passage it
  quotes, as the build hands it to `wpt_tokenizer.chat.encode` today.
- To resume, the freight starts the program again and sends `start` with
  `"resume": {"files": [...], "after": R}`: the files already uploaded and
  the last item they hold. The program goes on after it and writes the same
  files a run that never stopped would write.

Before the format is fixed, measure the cost of `weigh` through this channel
on real texts against today's in-process count: it is the most talkative
part of a build with a size in tokens. If it is too slow, batch the items of
one `weigh` message; do not change the channel.

### 7.3 WPT's preparations (settled: three, names proposed)

They live in a new package `wpt-corpus` (module `wpt_corpus`), which also
takes the shard format and its checked reader from `wpt_store.shards`,
since a token file is WPT's and not the dock's.

| Program | Settings | Inputs | Unit | Writes | Result |
| --- | --- | --- | --- | --- | --- |
| `wpt-vocabulary` | `vocab_size` | none | none (`weighs: false`) | `vocabulary.json`, the canonical vocabulary file | `{sha256, size, tokenizer_version}` |
| `wpt-tokens` | none | `vocabulary` | tokens | the token files by kind, `tasks.jsonl` | `{total_tokens, items_tokens, shards by kind, tasks, fingerprints by kind, vocabulary {sha256, size}}` |
| `wpt-count` | none | `vocabulary` | tokens | nothing | the total weight |

- `wpt-vocabulary` takes only plain texts and reads them once, in the
  order sent, as `Tokenizer.train` reads a stream today.
- `wpt-tokens` does what the workers of `wpt_depot.corpus` do today: a
  plain text or a piece becomes tokens followed by END, encoded a block at
  a time; a conversation goes through the chat template; a task becomes a
  line of `tasks.jsonl` and weighs the tokens of that line.
- `wpt-count` is `wpt-tokens` without files, for an estimate.

The programs are Python entry points over the Rust tokenizer for now.
Settled as a direction: what turns text into tokens belongs in the Rust
code of `wpt-tokenizer`. The chat template, the encoding by blocks and the
writing of token files are still Python and may move there later; that move
is no part of this plan.

## 8. How the freight runs (settled)

- The freight is told what to do, never how or where.
- It keeps a queue and runs one job at a time, in the order the requests
  came, giving the whole machine to the running job. This is a hard
  constraint.
- A job is one request, so one preparation.
- It runs on one machine. Several machines are not planned. To go faster,
  the freight's Docker image runs on a bigger machine.
- It takes no memory limit written by hand. It asks the system what is
  available and sizes itself, measuring again while the job runs
  (`parallel.Sizing` already works so).
- It has a command line and no web page.

Proposed details:

- The queue is a small SQLite file in the freight's own folder
  (`FREIGHT_HOME`), taken from the heavy lane of `apps/wpt/src/wpt/depot/jobs.py`:
  states queued, running, done, refused, failed, cancelled; at most one job
  queued or running for a delivery key; a job cut short runs again and goes
  on from its checkpoint, a limited number of times.
- `freight work` runs the queue until stopped. `freight request FILE`
  queues a request and prints its job; with `--wait` it also runs the queue
  when no other process does, until its own job ends, and prints the
  delivery note. `freight jobs`, `freight job ID [--follow]`,
  `freight cancel ID`, `freight deliveries`, `freight delivery KEY`,
  `freight estimate FILE` (today's estimate, through `weigh`),
  `freight check` (section 9).
- WPT calls the same operations as Python functions of `freight.api`
  (`request`, `wait`, `delivery`).
- The cap on the bytes the deliveries take on the dock (`WPT_CORPORA_CAP`,
  `wpt_depot.corpora.Room`) moves with the corpora's code and becomes the
  freight's setting.

## 9. Erasures (settled in substance)

The depot no longer marks anything on the dock. The freight asks the depot
what changed since the last change number it saw for that depot (request 4)
and marks stale every fresh delivery whose items hold a text an erasure
reached, as `wpt_depot.corpora.mark_stale` does today.

Proposed: it asks before every job, when `freight check` is run, and the
runbook has a timer run `freight check`. `wpt train` refuses a stale corpus
as it does today. `docs/erasure.md` says that a delivery is known stale
only once the freight has asked.

Settled: an erasure does not make a vocabulary delivery stale. The freight
writes the erasure into the vocabulary's delivery note, so every model that
names the vocabulary keeps working and the documented rule stays (the
merges learned from an erased text stay, docs/tokenizer.md).

## 10. What changes in each part

### 10.1 The depot

Settled:

- The tokenizer recipes go: `wpt depot tokenizer ...`, the routes
  `/api/tokenizers*`, the page "Tokenizers", the vocabulary job,
  `Depot.new_tokenizer`, `edit_tokenizer`, `add_vocabulary` and their
  readers.
- The builds go: `wpt depot build`, the routes `/api/recipes/*` and
  `/api/builds/{key}`, the page "Build", the composer of a data section on
  a snapshot's page, the corpus job, the checkpoints of builds in the
  queue's file.
- `wpt depot snapshot new` takes no name. The depot names the snapshot by
  the UTC time it is made, `YYYY-MM-DDTHH-MM-SSZ`. The web page offers no
  name either.
- The command becomes `depot` (`depot web`, `depot add`, ...).

Proposed:

- The tables `tokenizers_revisions`, `vocabularies` and `vocabulary_texts`
  stay in the file, unwritten, as the other legacy tables do, so the
  layout does not change and `FORMAT` stays. `depot sql` still reads them.
- The snapshots that exist keep their names: a snapshot's row never
  changes. Only new ones are named by their time.
- The frozen routes `/api/datasets*` stay.
- The queue keeps its write lane, its read lane and the snapshot job.
- Modules that leave `wpt_depot` for `freight`: `corpus.py`, `groups.py`,
  `fill.py`, `estimate.py`, `corpora.py`, and `pool_of` of `order.py`.
  `notice.py` goes to WPT. `order.py`, `pieces.py`, `parts.py`, `quotes.py`
  and `snapshots.py` stay.
- The package becomes `packages/depot` (module `depot`) and its
  controllers `apps/depot`. Settings: `DEPOT` (was `WPT_DEPOT`),
  `DEPOT_URL`.

### 10.2 The dock

- `wpt-store` becomes `packages/dock` (module `dock`), its setting `DOCK`
  (was `WPT_STORE`), its command `dock` with `find`, `fetch` and `put`
  (settled). `dock.api` holds those three gestures and what opens a dock
  from the settings.
- Proposed: the layout keeps `models/`, `runs/` and `cache/`, and gains
  `deliveries/`. The corpora already under `corpora/` stay where they are
  and stay readable (section 11). `wpt_store.models` (a model folder copied
  out) and the listing of runs are WPT's and move to it: `wpt store
  copy-out`, `wpt store runs` and `wpt store run` become WPT commands that
  read the dock.

### 10.3 The freight

New: `packages/freight` (module `freight`) and `apps/freight`. It holds the
selection (section 6.4), the queue (section 8), the exchange with a
preparation (section 7), the deliveries on the dock and their cap, and the
check for erasures (section 9). It holds no token, no vocabulary and no
chat template.

### 10.4 WPT

- Settled: a WPT command reads an order and sends the requests in order,
  first the vocabulary when the order asks for one to be learned, then the
  corpus, whose request names the vocabulary's delivery. Proposed name:
  `wpt build-resources ORDER` (`SERIES:` as today reads a series at its
  largest size). It prints both delivery notes.
- `wpt train ORDER` works out the delivery keys from the order, finds the
  deliveries on the dock, and refuses the run when one is missing ("run
  `wpt build-resources`") or stale. It reads the vocabulary from its
  delivery and writes it into the model folder as `tokenizer.json`, as
  today.
- `provenance.json` keeps the order's data section and, for each delivery,
  its key, id and preparation. `wpt_model.provenance` reads the three
  shapes.
- `CREDITS` and `NOTICE` are written from the corpus delivery's items
  tables (`notice.py`, moved).
- `wpt train` no longer reads a depot (`apps/wpt/src/wpt/train/recipe_depot.py`
  keeps only the legacy dataset shape).
- `wpt manager` names the corpus by its delivery.

## 11. What exists already (proposed)

- **The two vocabularies in the depot** (`4e264a0d…`, 32,768 entries, and
  `312d8d3f…`, 65,536 entries). A one-time command of WPT writes each on
  the dock as a delivery whose note says it was adopted, not made by the
  freight, with what the depot recorded: its recipe's name and version, the
  dataset it named, its sample bytes, its count of texts, and the list of
  the texts it read as the items table. Its delivery key is
  `adopted-<sha256>`. An order that names a vocabulary by sha256 finds it
  so.
- **The corpora under `corpora/<build key>/`** stay and stay readable by
  `wpt train`, whose reader takes both record shapes. The orders that name
  them keep working. They are not copied into `deliveries/`.
- **The snapshot `balanced`** (`wic-4`, change 32118) keeps its name.
  Settled: the first vocabulary and corpus of this plan are made from a new
  snapshot, made once step 5 names it by its time.

## 12. Set aside (settled)

- The depot's backups in a bucket of their own, kept under the 3-2-1 rule.
  Until then they stay under `backups/` of the dock.
- Kinds for image, video and sound.
- A repository of their own for `depot`, `freight` and `dock`.
- Several people working on one depot at once.
- Several machines for one freight.

## 13. Decided by the maintainer (settled)

1. An erasure is written into a vocabulary's delivery note and does not
   make it stale (section 9).
2. The first vocabulary reads a new snapshot, not `balanced` (section 11).
3. The names `wic-utils`, `wpt-corpus`, `wpt-orders`, `wpt build-resources`,
   `wpt-vocabulary`, `wpt-tokens` and `wpt-count` stand. The depot's request
   module is `depot.api`, and every part has its `api` (section 2).

## 14. The order of the work (proposed)

Each step ends with `uv run ci/all.py` green and the documentation of what
it touched describing the new state. Steps 1 to 4 deliver the vocabulary
the problem started from; the renames come after.

1. **The depot's request module** (`wpt_depot.api` until the rename of
   step 6, with `docs/api/depot.md`). Add it over today's code, and make
   `corpus.py`, `groups.py` and `estimate.py` read the depot only through
   it. No behavior changes: the existing tests pass unchanged.
2. **The freight, with the preparations.** Create `freight` and
   `wpt-corpus`; move the selection, the corpora and the heavy queue; write
   the exchange and the programs `wpt-tokens` and `wpt-count`. Check:
   on the test depot of `apps/wpt/tests/conftest.py`, a corpus request
   gives token files with the same token hashes, and items tables with the
   same rows, as `wpt depot build` gives for the same snapshot, data
   section and vocabulary.
3. **The vocabulary.** Write `wpt-vocabulary` and the selection on bytes.
   Check: a snapshot whose shares are given holds, in the items of the
   vocabulary's delivery, each share's proportion of the sample's bytes
   within a thousandth, and the same request gives the same sha256 twice.
4. **WPT's side.** `wpt build-resources`, the two forms of
   `data.vocabulary`, `wpt train` on deliveries, provenance, the adoption
   of the two existing vocabularies. Check: the `trained` fixture trains
   from an order with a learned vocabulary, and from one that names a
   vocabulary by sha256.
5. **The depot loses what moved.** Remove the tokenizer recipes, the
   builds, their routes and pages, and the names of new snapshots.
6. **The renames.** `wpt-store` to `dock`, `wpt-depot` to `depot`,
   `wpt-recipes` to `wpt-orders`, `data/recipes` to `data/orders`, the
   settings, the three commands, `wic-utils`, the `api` module of each
   part with its document under `docs/api/`, and the import test of
   section 2.
7. **The documentation and the skills**, rewritten to describe the result:
   `docs/architecture.md`, `tokenizer.md`, `data-selection.md`,
   `training.md`, `depot-web.md`, `erasure.md`, `machines.md`, `model.md`,
   and the skills under `.agents/skills/`. `docs/architecture.md` states
   that `depot`, `freight` and `dock` are meant for other models and data
   and for a repository of their own. Delete this file.

## 15. Done

On the server, with the code of step 4 at least:

1. The dock holds a vocabulary delivery of 16,384 entries learned on
   10,000,000,000 bytes of a new snapshot with the shares
   0.35 / 0.15 / 0.05 / 0.05 / 0.40 of section 4.2, and its note lists the
   texts it read.
2. The dock holds a corpus delivery built with it from the same snapshot,
   kinds `conversation`, `plain`, `task`, the same shares, and
   `total_tokens` 1,179,648,000.
3. An order in `data/` asks for both, and `wpt train` finds them from it.
4. The thread that asked for them has received: the vocabulary's sha256,
   the corpus's id and delivery key, its tokens and texts, the exact data
   section to put in an order, and what changed in the code and on main.

The whole plan is done when step 7 is merged and this file is gone.

## 16. Limits for whoever does the work (settled)

- The depot is public and shared. Nothing in this plan writes to the
  depot's file: the vocabulary and the corpus go to the dock.
- Restarting or redeploying the depot's web app, and anything else that
  changes the server beyond new deliveries on the dock, needs the
  maintainer's yes first.
- Never read or print the texts of the depot or of a corpus, a settings
  file or a key. Counts, ids, sizes and logs are fine.
- Watch the server's memory while a job runs.
