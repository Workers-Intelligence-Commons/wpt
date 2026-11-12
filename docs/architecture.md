# Architecture

**The texts live in a depot; the repository holds none.** The repository
holds the code, the model, tokenizer and series recipes, and the images of
agentic tasks. The texts, the decisions on them and what to train on live
in depots, and everything built or trained lives in a store.

- A depot is one folder, known by its address, of tables: the texts and the
  structured items (conversations, tasks), each with its license, credits
  and notice; the people's decisions on them; the libraries that group
  admitted texts; the erasures; and the datasets, each naming libraries and
  the share of each tag. A depot can follow an upstream depot and pull its
  updates (`wpt_depot`).
- A corpus is what a dataset is built into, in the store: the token files,
  the conversations and tasks, and copies of every text's credits,
  licenses and notices, so it says everything about what it holds even once
  the depots it read change or vanish (`wpt_depot.corpus`).
- A model recipe names a depot address and a dataset in it; training reads
  that dataset's corpus from the store, and every size of a model series
  reads the same corpus, the smaller sizes its first part.

## Components and services

Each component has one role. Every operation is a service: a function with
its inputs, its result and its named failures. The `wpt depot` commands
and the `wpt depot web` routes are controllers: each reads its input
(arguments, a request), calls the same services, and presents the result.
A controller holds no rule of its own, so the two interfaces cannot
disagree.

| Component | Role | Code |
| --- | --- | --- |
| depot | every change to a depot's tables, under its rules; a dataset's corpus built into the store | `wpt_depot` |
| service layer | the only code that reads or writes `data/` ([The service layer](#the-service-layer)) | `wpt_recipes/recipes.py` |
| store | files, compressed, on a folder or a bucket ([The store](#the-store)) | `wpt_store` |
| training | the steps a recipe names, through gpu workers | `wpt.train`, `wpt_training`, `wpt_worker` |
| agent harness | the containers, tools and checks of agentic tasks, and the Git repository a task works in | `wpt_agent` |
| evaluation | tests on a trained model, into its model card | `wpt.eval` |

## The depot

A depot is one folder, or a prefix in an S3-compatible bucket
(`s3://bucket/prefix`), given as a path or an address and known by that
address. One app writes it, and people copy it whole. A depot in a bucket
reads its own settings, never the store's: `WPT_DEPOT_ACCESS_KEY_ID` and
`WPT_DEPOT_SECRET_ACCESS_KEY` (both needed), `WPT_DEPOT_SESSION_TOKEN`,
`WPT_DEPOT_ENDPOINT_URL` and `WPT_DEPOT_REGION` (default `us-east-1`), in
the environment or in `.env`. It reads no `AWS_*` setting. The setting
`WPT_DEPOT`, in the environment or in `.env`, names the depot that
`wpt depot` and `wpt depot web` work on, as `WPT_STORE` names the store.
`wpt depot new ADDRESS` makes an empty one, with its name, its maintainer
and its takedown contact.

Its tables are `about` (the depot's name, maintainer, takedown contact
and upstream), `licenses`, `notices`, `kinds`, `decisions`, `texts`,
`contents` (every content version of each text), `libraries`,
`memberships` (when each text joined and left each library), `pins` (the
revision of each text a locked library reads), `erasures`, `datasets`,
`pulls` (for a depot that follows another, how its change numbers meet
its upstream's) and `journal`. What each holds
and the rules on it are in [data-selection.md](data-selection.md) (texts,
items, tags, libraries, datasets and corpora),
[licenses.md](licenses.md) (licenses, notices and decisions) and
[erasure.md](erasure.md) (erasures).

**Ids.** Every id (of a text, a decision, a library, an erasure, a notice
or a dataset) is the depot's name, a hyphen and a number: `my-depot-17`. The
depot's name is made of letters, digits, `.`, `_` and `-`, and begins with a letter or a digit. A reference to an
id in another table, in a dataset, in a corpus or in a tokenizer lock holds
the whole id, so an id names one thing across depots. The number comes from
the highest number this depot's ids carry in that table, an erased text's id
included. A clone takes a name of its own, since it never reuses its
upstream's.

Each table is a Delta Lake table in its own folder, written through
delta-rs and read as SQL through DuckDB (`wpt_depot/tables.py`). Delta
Lake checks only NOT NULL and CHECK constraints, so `wpt_depot.depot`
enforces keys and references itself. It is the only code that changes a
depot: each operation checks the whole change against the depot's rules
first, so a refused change writes nothing.

**Revisions and change numbers.** Every change of a depot has a change
number, one past the last. `about`, `texts`, `libraries`, `erasures` and
`datasets` keep each change of a row as a new row, its revision, keyed by
the id and the revision number and recording its change number; the
highest revision is the current one. A text's content versions are
rows of `contents`, keyed by the text's id and the version number and
recording their change number, and each revision of the text names the
content version it reads. A revision of a text also holds its scores,
each with the content version it was made from. Memberships record the change
numbers a text joined and left a library at. A depot can so be read as it
was at any change number, which a dataset names for each library it
reads. Delta Lake's own table history plays no part: an admitted erasure
drops it.

**SQL.** A read query sees each revised table's current state under its
plain name (`texts`, `libraries`, `datasets`, `about`, `erasures`) and
every one of its rows under `<name>_revisions`. The `texts` view joins
each text with the content version its current revision reads and the
libraries that hold it now. The other
tables, `contents`, `memberships`, `pins` and `pulls` included, are seen
as they are.

**The journal.** Delta Lake commits one table at a time, so a change that
touches several tables is written into the journal first, and that write
makes it certain. The journal holds no text's content: the content
versions a change adds are written to `contents` in one commit right
after its entry, numbered by its change number. A change of scores holds
only its decision and the number of texts it changes: its decision and
then the texts' new revisions, made by one query over the lines of the
change and the current texts, are written right after its entry, the
revisions in one streamed commit numbered by its change number. The depot then applies
the change to each table and marks it done. Opening a depot applies
again every change left undone, and cuts short a change whose content
versions, or whose revisions of a change of scores, were never written,
writing nothing of it, so a crash never
leaves a change half applied. A change that fails while the depot stays
open is finished the same way before its next write, so no later change
takes the ids or the change number of an undone one (`wpt_depot/journal.py`). The decision that
admits an erasure and the erasure's carrying out are one change. Only the
app that owns a depot finishes its journal: reading another depot (a
dataset's library, a corpus, an upstream) writes nothing.

**Writing rows.** A new row is appended without reading the rows its
table holds, so adding texts costs no more for the text the depot holds
already. A change applied again after a crash compares keys instead, so it
never writes a row twice. Only a row that changes in place is rewritten
through a merge, which reads the whole table: a membership that ends, a
content version an erasure cuts, a row a pull replaces, a journal entry
marked done. Each write adds files that every later read of its table
opens, so at regular intervals of writes a table's log is summed up in
a checkpoint and its small files are merged: a read opens a bounded
number of small files whatever the number of writes before. A merge
that fails leaves the write it follows as committed. The files a merge
replaces stay with the older states, and every checkpoint records them,
so opening a table still grows slowly with the files it replaced.
`contents` is never merged: its replaced files would hold each content a
second time, so its file count grows with the writes that add contents.
A query opens only the tables it reads. A text's content is stored once,
in `contents`, whose content columns are written without statistics, so
no file footer or table log holds a copy of a text's first characters.

**Format.** The depot's layout, its tables and their columns, has a
number: its format. The code holds the format it reads and writes
(`FORMAT` in `wpt_depot/schema.py`). Each `about` row records the depot's
format when it was written, so the current row gives the depot's format. A
depot whose `about` table has no `format` column is of format 0.

Each format adds one change of layout to the one before. Format 1 adds the
`format` column of `about`. Format 2 adds the `scores` column of `texts`,
null for a text without scores. Its upgrade step adds the column to the
table's log alone and rewrites no data file, so it costs the same whatever
the number of texts. A data file written before a column was added lacks
that column, and every read gives it as null in that file's rows: the SQL
view of a table a step widened this way (`WIDENED` in
`wpt_depot/schema.py`) matches the columns of its files by name, which
reads the footer of each of its files before a query runs, and every other
table's view reads its files by the first one's columns.

Every way into a depot that reads or changes its rows refuses one of
another format than the code's, older or newer: every `wpt depot` command
but backup, restore and upgrade, `wpt depot web`, the reading of another
depot's library, and a clone or a pull, which need both the upstream and
the clone at the code's format. The refusal names the depot's address,
its format and the code's, and says to run `wpt depot upgrade` for an
older depot or to update wpt for a newer one.

A backup and a restore copy a depot of the code's format or of an older
one as it is, so the copy keeps its format, and they say when it needs
`wpt depot upgrade`. They refuse a newer format.

`wpt depot upgrade` runs only when asked. It first finishes every change
the journal holds undone, with the current code, on the layout the depot
has then. Then it runs, in order, the step from the depot's format to the
next up to the code's. It records each format reached as a new revision of
`about`, a change of the journal like any other. An upgrade cut short is
finished by running the command again, which resumes after the last format
it recorded. A newer depot is never brought down: the command refuses it.

Every change to the depot's layout raises `FORMAT` and adds the upgrade
step from the format before, with its test. A step names the columns it
reads and writes itself, never through the current layout, which is a
later format's. It changes only the tables its layout change touches, and
run again on the layout it leaves it changes nothing more. A layout change
keeps the current code able to apply an older format's undone journal
changes on that older layout. A step that changes the journal table keeps
backups and restores of older formats working: they read the journal's id
and done, the `about` table's change and name, and every table's storage
versions.

**Following an upstream.** A depot can follow another, its upstream, the
way a Git clone follows its remote. `wpt depot clone ADDRESS TO --name NAME --maintainer M --takedown-contact C` copies
every row of every table but the journal, revisions, contents,
memberships and pins included, with their change numbers, into a new
depot whose `about` row names the upstream. `wpt depot pull` fetches the
upstream's updates: each upstream row replaces the row of the same key,
and a row only the depot holds stays. Its change number is past the
upstream's, so the depot's later changes come after every change it
received. The texts the upstream's admitted erasures removed are removed
too, and an erasure the depot had not carried out is carried out there
(`wpt_depot/follow.py`). A pull runs when someone asks for it.

The rows a clone receives keep the upstream's change numbers, so the
`pulls` table says how the two numberings meet: for the clone itself and
for each pull, the depot that pulled, its change number then, its
upstream's name and the upstream's change number the pull brought it up
to. That number is the upstream's settled change number: the highest
change its journal holds done with no change undone below it, read before
any other table, so every upstream change up to it was written whole and
a change the upstream is still writing, or holds undone after a crash,
stays past it until a later pull. The pull's journal change carries that
number, so a pull applied again after a crash records it the same way. A
dataset that names a library of another depot names it at that depot's
settled change number too. A read of the clone's own
library at a past change of the clone sees the upstream's rows (their
text revisions) only up to the upstream change the last pull at or
before it brought, and the clone's own rows by its own numbers, so the
library reads exactly as it stood then. A library of the upstream is
read by the upstream's numbers, so a dataset naming it at an upstream
change reads the same in the clone as in the upstream, and a dataset of
the clone that names it by the clone's address names it at the upstream
change its pulls brought. A clone copies its upstream's `pulls` rows
too, so a clone of a clone follows the numbers back through each depot.

A clone is read-only on the rows its upstream owns, those whose id does
not carry the clone's name. An id carries a name when it is exactly that
name, a hyphen and digits, so `wpt-eu-3` carries `wpt-eu` and never
`wpt`, and a clone whose name begins its upstream's name owns none of the
upstream's rows. Every operation that would change one is refused and
writes nothing, with a message that it belongs to the upstream: a
decision, a license change, a content change, an edit of credits or
scores, or an erasure request on
an upstream text, a decision on an erasure request that is the upstream's
or names an upstream text,
renaming, locking, reporting on or putting texts into an upstream
library, and editing an upstream dataset. An operation on several texts
is refused whole when one of them is the upstream's, and the review of a
decision lists those texts apart. Only the upstream changes its rows, so
a pull never meets a clone's change to one. The clone's own texts,
libraries and datasets stay fully changeable. Its own library may hold
upstream texts, and locking it pins their revisions; a fork of an
upstream library is a library of the clone's own. An upstream text that a
pull brings refused, or with its licenses or content changed, leaves the clone's own
open libraries at the pull's change number, as that decision made in the
clone would; its locked libraries keep their pins. A dataset reads one of
the upstream's libraries in the clone whether it names it by the clone's
address or by the upstream's, so the head of the library in the workspace
lists both. The pages show quietly
that a text, library, dataset or erasure request belongs to the upstream,
and offer no action on it that the depot would refuse.

**Backups.** `wpt depot backup TO` writes a new full copy of the depot
that `WPT_DEPOT` names into a new folder under `TO`, a folder or a bucket
as a depot's address is. The folder's name is the depot's name, the UTC
time and the change number the copy holds:
`my-depot-20261110T153000Z-change1234`. The copy holds every table with
its log and its kept older storage states, as of the depot's settled
change number, so a change written during the copy is not in it. It is
an ordinary depot of the format the depot had, "Format" above
(`wpt_depot/backup.py`).

The app never cuts, erases or removes a backup, and keeps no list of
backups. An erasure does not reach them ([erasure.md](erasure.md)).
Removing a backup is done by hand, by deleting its folder. Backups on a
schedule come from a timer outside the app, such as this cron line:

    0 3 * * * cd /path/to/wpt && uv run wpt depot backup /backups

`wpt depot restore BACKUP TO` copies a backup as it is to `TO`, which
must hold nothing: it never overwrites. The restored depot reads as the
backup does. Erasures carried out after the backup are not carried out
in it.

## The store

The store is set by one setting, `WPT_STORE`, in the environment or in
`.env` at the repository root:

```
WPT_STORE=file:///home/me/wpt-store      a local folder
WPT_STORE=s3://bucket/prefix             an S3-compatible bucket
```

For S3, the credentials and a non-AWS endpoint come from the store's own
settings, in the environment or in `.env`: `WPT_STORE_ACCESS_KEY_ID` and
`WPT_STORE_SECRET_ACCESS_KEY` (both needed), `WPT_STORE_SESSION_TOKEN`,
`WPT_STORE_ENDPOINT_URL` and `WPT_STORE_REGION` (default `us-east-1`). The
store reads no `AWS_*` setting (not even `AWS_ENDPOINT_URL` or `AWS_PROFILE`)
and no AWS configuration or credentials file, so the keys a cloud provider's
other tools use never reach it.

The setting's prefix picks the store's kind, a class of `wpt-store`
(`wpt_store/kinds.py`). A kind supplies five operations on files: write a
whole file, read a byte range, replace a file only if it is unchanged
since it was read, list, delete. Everything else (the layout and
compression) is built on them, alike for every kind.

A store holds real named things, the same on a folder and on a bucket:

```
corpora/<name>/                    a dataset's corpus: corpus.json, token files, vocabulary, tables, tasks
models/<name>/                     the model's real files
runs/<run>/                        each command's record
cache/<stage>/<fingerprint>        only what can be rebuilt
```

`wpt store list` lists the corpora and models.

Hashes do two jobs. Every read checks the bytes against their sha256: a
record names each file it points at by the one shape `{"key", "sha256"}` (the
file's key in the store and its sha256), a record holds the
sha256 of its own fields, and a result holds its own in its first line.
Finished work is found from its inputs: a result lies under its
fingerprint, such as a training step's result in `cache/steps/`. A file
known only by its content, such as the kept state of a step that nothing
but the step's result names, lies in `cache/files/<sha256>`.

`cache/` holds only what can be rebuilt, so emptying it leaves every corpus
and every model usable: a corpus's files and a model's files, a run's
checkpoint included, lie in their own folders, and a record never points
into `cache/` for a file it needs to be read.

### Compression

Every file is compressed with zstd while it is written, in frames of
about 1 MB compressed apart (zstd's seekable format, which plain zstd
reads, its seek table a skippable frame that standard tools pass over), so
a byte range is read without the rest. It is kept compressed, under its key
plus `.zst`, only when that is smaller, and as it is otherwise, at level 3.
An S3 answer cut while it is read is asked again, up to 5 tries in all.
Large files are compressed several frames at once, as `wpt_utils.parallel`
allows. They go to S3 in parts of 64 MB, and every upload of a process
sends its parts through one shared limit of 32 at once: S3-compatible
stores such as R2 give each connection a few MiB/s, so a link fills only
with many connections at once, and uploads that run together do not
multiply them. That limit is set for network work, not by the cores; a
part starts only while the memory rule of `wpt_utils.parallel` allows.

Reads from S3 keep the same many connections busy. A compressed file's
frames are fetched in blocks of 8 frames, one request each, and a read
requests the blocks it needs together, at most 8 requests in flight per
process. When a reader goes through a file in order, the blocks after the
one it works on are requested ahead, so its next data arrives while it
processes the current data. The compressed bytes requested ahead and not
yet taken stay within 64 MB per process. A request ahead that fails raises
at the read that takes its block. What a read returns does not depend on
the timing. A local folder reads in place, one request at a time.

### What else it keeps

A named run's checkpoint lies beside its model: `models/<name>/checkpoint.json`
is a record that names the state's file, which lies in the same folder
([training.md](training.md), "Checkpoints and resuming").

The store also holds the record of every run of `wpt train`, `wpt eval`
and `wpt manager`: one folder per run under `runs/`, written by name, each
file compressed (a local store adds a record file's lines to one plain file
while the record runs, and compresses it into one part when the record
ends; [observability.md](observability.md)).

## The service layer

`packages/wpt-recipes/src/wpt_recipes/recipes.py` (`Recipes`) is the only
code that reads or writes the repository's data. It hands out documents
and bytes, never a path under `data/`. It holds the model, tokenizer and
series recipes, the tokenizer locks, and the images of agentic tasks:
their Dockerfiles, locks, license lists and pools of tools (`tools.json`).
`wpt train`, `wpt manager`, `wpt eval`, `wpt doctor`, the
image commands of `wpt agent` and the task reader of `wpt_agent` read them
through it. It checks every file it reads against its schema, and its
check adds the rules across files: a recipe names a tokenizer recipe that
exists, a series derives every size, an image's pool takes no built-in
tool's name. A recipe whose depot address or dataset name is still empty
reads, and the check names it for a person to fill in. People edit the
recipes and `tools.json` as files; `Recipes` writes only the generated
files (a tokenizer's `.lock.json`, an image's `lock.json` and
`licenses.json`), for the command that makes them, after their schema,
through a temporary file and a rename.

Two checks hold the rule. A test
(`packages/wpt-recipes/tests/test_data_access.py`) reads the source code of
every package, application and script and fails when code outside the
service layer names the data folder in a way that could open a file under it: the name
`data` as a part of a path (either side of `/`, or an argument of a call
that builds or reaches a path, such as `Path()`, `joinpath()`,
`os.path.join()`, `os.walk()`, `os.listdir()`, `glob()` or `open()`), a
name bound to `"data"` or a comparison with it, a string or an f-string
that starts with `data/` or holds `/data/` or `:data/`, or a private member
of the service layer used from outside. A docstring, a command's help text
and a message (inside a `raise`, or given to `print()`, a log call or
`sys.exit()`) may name the folder for people. A syntax scan cannot follow
every way to build a path, so the service layer also sets an audit hook
(Python's `sys.addaudithook`, `wpt_recipes/guard.py`) in every process,
when the first `Recipes` is made: any opening, listing, removal
or renaming of a path under a guarded data folder, as written or with its
symbolic links resolved, whose nearest caller in the project's own code
(not Python's library, not an installed package or its command script) is
neither the service layer nor a test fails with
`DataAccessRefused` before it happens, in every command and in the test
suite. Inside `shutil.rmtree`, a relative name is a child of the folder being
removed, not of the current folder, so only the folder `rmtree` was given is
checked. A test is a
`conftest.py` or a file under a `tests` folder, both inside the project's
checkout: a folder named `tests` above the checkout, or a `conftest.py`
outside it, does not make its code a test. The tests are left out of both:
they build the repositories they check and read what a command wrote.

### Two interfaces

The depot has two interfaces, both thin layers over `wpt_depot`: the
`wpt depot` commands, and the HTTP API of `wpt depot web`, described in
OpenAPI 3.1 at `/api/openapi.json` (the server checks every request body
against the same description). The depot applies every rule, and a
refused change writes nothing.

| Operation | `wpt depot` | `wpt depot web` |
| --- | --- | --- |
| a new depot, a clone | `new ADDRESS`, `clone ADDRESS TO` | none |
| a pull | `pull` | `POST /api/pull`; page "Depot" |
| the depot's layout brought up to the code's format | `upgrade` | none |
| the depot's maintainer and takedown contact | `about [--maintainer M] [--takedown-contact C]` | `POST /api/about`; page "Depot" |
| a read query | `sql QUERY` | `POST /api/sql` |
| the depot at a glance | none | `GET /api/decisions/latest`; pages "Overview" and "Settings" |
| a decision, shown | `sql QUERY` | `GET /api/decisions/{id}`, with the texts it is on, the erasure requests it decides or the problem it records; the view of a decision, and of a reported problem |
| a tag, shown | `sql QUERY` | `POST /api/texts/search` with the tag; the view of a tag |
| texts found by a filter, a page at a time | `sql QUERY` | `POST /api/texts/search`, with `GET /api/tags`; the playlist of the "Workspace" page |
| texts found by a filter, with their text, a page at a time | `sql QUERY` | `POST /api/texts/passages`, each page bounded by a count of texts and by characters; the blob of the "Workspace" page |
| a text or an item, added | `add FILE [--kind KIND]` | `POST /api/texts`, with `/api/kinds` and `/api/licenses`; the "Add a text" button |
| a text or an item, shown | `show ID` | `GET /api/texts/{id}`; the editor of the "Workspace" page, with the text's record |
| a license | `license add ID NAME FILE` | `/api/licenses`, `GET /api/licenses/{id}`; page "Licenses" with the view of each license, and the license picker of the "Add a text" page, of a license change and of the review of an admission for a missing one |
| a decision on texts | `decide admit\|refuse WHERE` | `POST /api/decisions`, on named texts, and `POST /api/texts/blocked`, which of named texts an admission cannot cover and why; the "Workspace" page, on ticked texts, on the current text, and by the keys A and R |
| a change of texts' licenses | `relicense ID ... --license L ...` | `POST /api/texts/licenses`, on named texts; the review of an admission, for texts with no allowed license, and the current text of the "Workspace" page |
| a change of a text's content | `content ID FILE` | `POST /api/texts/{id}/content`; saving in the editor of the "Workspace" page |
| a change of a text's credits | `credits ID [--title T] [--author A ...] [--tag T ...] [--address A] [--notice FILE] [--clear CREDIT ...]` | `POST /api/texts/{id}/credits`; the current text of the "Workspace" page |
| a change of the scores of many texts | `scores FILE` | `POST /api/texts/scores` |
| libraries | `library new`, `put`, `take-out`, `lock`, `fork [--without WHERE]`, `report`, `rename` | `/api/libraries[/{id}/texts\|taken-out\|pins\|lock\|forks\|problems\|name]`; the "Workspace" page lists them and opens each, with its head (a locked one's with its pinned revisions), and the view of each reported problem |
| datasets and remotes | `dataset new`, `edit`, `export`, `import` | `/api/datasets`, `GET /api/tokenizers`, `GET /api/datasets/{id}` (with the corpora the store holds built from it), `POST /api/datasets/{id}` (its edit), `GET /api/datasets/{name}/remote`, `POST /api/remotes`; page "Datasets" with the view of each dataset |
| a corpus | `build DATASET` | `POST /api/datasets/{name}/corpus`; page "Datasets" |
| a tokenizer's vocabulary | `tokenizer RECIPE` | none |
| erasures | `erase ID [START END]`, `erasures`, `decide admit\|refuse --erasure ID` | `/api/erasures`, `GET /api/erasures/{id}`, what one request asks to erase with the text around each passage, and `POST /api/erasures/decisions` on one or several requests; page "Erasures" asks, shows, lists and decides, and the current text of the "Workspace" page asks |

Each row of a list on the pages opens a view of that thing, with its own
address in the page's hash so it can be linked: a text (`#/texts/ID`, the
workspace with that text in its editor), a library (`#/libraries/ID`, the
workspace with that library open), a reported problem
(`#/libraries/ID/problems/DECISION`), a dataset (`#/datasets/ID`), a
license (`#/licenses/ID`), an erasure request (`#/erasures?request=ID`), a
decision (`#/decisions/ID`) and a tag (`#/tags/TAG`).

Every decision, an added text's, an erasure request's, a reported
problem's, a change of content's, of credits' and of scores' included, carries
the decider's declaration: a legal name, an email, and the confirmation
that they read the requirements, accept them, and that what they submit
meets them. The commands take it as `--name`, `--email` and `--accept`.
The "Settings" page keeps the name and email in the browser and says that
both are public. Each decision is sent only by a click on its own confirming
button, or by a deciding key of the workspace. Beside the button stand the
confirmation that `GET /api/about` gives, a link to the requirements, and
the name and email the decision records. A form without a name or an email
says so and links to the settings.

The "Workspace" page replaces separate pages of texts and libraries. Its
left column lists what can be opened: every text, each status, the
libraries, the tags, the licenses, and the filters saved in the browser.
What is open is a filter, kept in the address, which the filter bar
refines. The middle column is the playlist of what is open, a page at a
time; the right column is the current text in a text editor, with its
record and its actions beside it. Saving an edit is a content change only:
the text goes back under review and leaves every open library, and is
admitted again apart. In the playlist, keys move from text to text, and A
and R admit or refuse the current text and open the next one. A legend
beside the playlist lists the keys and shows the confirmation they make
with the name and email they record. A line names the text and the
decision after each press. Without a name or an email, the key asks for
them and decides nothing. The keys do nothing while the cursor is in a
field or the editor. The blob mode shows the open texts as one continuous
text, read a page at a time as it scrolls; a click on a passage opens
that text in the editor.

The web pages decide on texts by their ids, never by a filter. A filter
finds the texts; the review window lists every id it reached, each ticked,
and the decision, the put, the take-out, the fork or the report takes exactly
the ids that stay ticked. A text that matches the filter after the review is not part of
the decision.

The review of an admission sets apart the texts the depot would refuse to
admit because none of their licenses is allowed. The depot's own check
gives that list, the one a decision applies, so the two cannot disagree.
Those texts are never part of the admission. From the list a person
refuses them or changes their licenses. After a refusal or a license change
made there, the review reads its summary again and checks its texts again,
so a text that now has an allowed license joins the admission, ticked.

The HTTP API answers 400 for a query or a remote that is not one, 404 when
the thing does not exist, 422 for a
change the depot's rules refuse (with their messages), and 503 when the
store or a locked vocabulary is missing on this machine. A command
prints the same failure's kind and message and ends with exit code 2.

`wpt depot web` serves the depot `WPT_DEPOT` names. It listens only on
127.0.0.1 and checks neither the `Host` header nor a request's
`Content-Type`: like the commands, it asks for no login and no header. What
that leaves open while it runs: a web page open in the same browser can
send it a cross-origin `POST` with `Content-Type: text/plain`, which the
browser sends without asking first, so such a page can add texts and licenses,
make libraries and datasets, build a corpus, or send a decision with a
declaration it made up; and a site that makes its own name resolve to
127.0.0.1 (DNS rebinding) can reach every route, reading every text of the
depot as well.

## Packages

| Package | Role |
| --- | --- |
| `wpt-depot` | the depot: texts and structured items, the decisions on them, libraries, erasures and datasets, in Delta Lake tables of one folder written through delta-rs and read as SQL through DuckDB (`depot.py`, the only code that changes them, `tables.py`); the app's policy on decisions (`policy.py`); a filter of texts turned into an SQL condition (`filters.py`); the journal of changes across tables (`journal.py`); erasures carried out (`erasure.py`); the kinds of items and their schemas (`kinds.py`, `kinds/`); clone and pull (`follow.py`); a dataset as a remote (`remote.py`); a dataset's corpus built into the store (`corpus.py`) and a model's notice from it (`notice.py`); `WPT_DEPOT` (`settings.py`) |
| `wpt-recipes` | the model, tokenizer and series recipes and the images of agentic tasks: the service layer that reads and checks them (`recipes.py`), the sizes a series derives (`series.py`), and the audit hook that keeps `data/` to the service layer (`guard.py`) |
| `wpt-store` | the store (its kinds: a local folder or S3), its compression, its named files (corpora, models, run folders) and its results, and the token shard format with its checked reader, so training and evaluation read a corpus without depending on the depot; a kept model folder copied out, checked against its sha256s (`models.py`) |
| `wpt-telemetry` | the record of one command run in the store, sent live to an OpenTelemetry backend when one is set ([observability.md](observability.md)) |
| `wpt-tokenizer` | the byte-level BPE tokenizer, Rust inside |
| `wpt-model` | the WPT network ([model.md](model.md)) and its drafter, its safetensors model folders, and their license documents (`LICENSE`, `NOTICE`, model card) |
| `wpt-training` | the training loop (global batch, evaluation, checkpoints, resuming) that runs one training step, with its plug points: strategies (how the hardware is used), and the parts of a step ([training.md](training.md), "Steps"): methods (how the model learns from an example), environments (where the examples come from), optimizers (how an update changes the weights) and the steps that name one of each, with the model parts they lock |
| `wpt-inference` | sampling through the model's generation cache: several rows from one prompt, the model's next chat message (`next_message`, which `wpt chat --chat` writes with), speculative decoding with a drafter and its scheduler, the scheduler's pass times (measured, recorded, replayed); training a model's drafter (`drafting.py`), which `wpt train` runs at the end of every run, so it uses `wpt-model` |
| `wpt-utils` | helpers several packages share, one submodule each: `wpt_utils.device` picks the best GPU, else the CPU (`WPT_CPU=1` forces the CPU); `wpt_utils.programs` starts the outside programs WPT runs (`program(name)`, with a clear error when one is missing), identifies the code (`code()`: the checkout's commit and a hash of its uncommitted changes, the one place that reads the commit, which records and a training step's fingerprint hold; `code_name()`: the same as one line, which a stored result keeps as what made it) and gives that commit, and whether there are changes, in the shape the records name them (`repository()`), without importing PyTorch; `wpt_utils.files.sha256_file` is the one way a file is hashed, by the store, the model folders and the run records alike; `wpt_utils.files.lock_file` is the lock the operating system holds on a file for one process, which a local store's replace of a file uses; `wpt_utils.settings.typed` types the settings of a training step and of its method, environment and optimizer from the defaults each file declares; `wpt_utils.parallel` is the one rule that sizes parallel work (at most 75% of the cores; before each new item it measures again and starts it only while a quarter of the memory, of the cores and, where the items write to a disk, of the free disk stays free, one item's memory being what running items are measured taking, never a given figure; it never stops a running item) and runs items that way with their results in the items' order, an error an item may fail with becoming that item's result; `wpt_utils.lines` writes every printed line, with its time and level |
| `wpt-agent` | the agent harness of agentic training: a task read from a depot item (`tasks.py`) and the Git repository it works in, fetched when it runs (`repository.py`), the tools (the four built-in ones, `ask`, and those a task or its image's pool declares, answered by programs in the image) and each attempt's choice of them, one Docker container per attempt, the images of `data/images/` and their locks, the agent loop and the task's check, and the playing of a task's attempts (`wpt_agent.play`), which the `agent-tasks` environment of the `finish-tasks` and `finish-tasks-with-less-writing` steps uses; an image's lock and build (`images.py`) |
| `wpt-worker` | the gpu worker, which runs the GPU work of a training on its machine, the link to it from the wpt manager's side, and the gradients exchanged between gpu workers in different places ([machines.md](machines.md)) |
| `wpt` (`apps/wpt`) | the app: the `wpt` command and its web server, two controllers over the services above ([The app](#the-app)) |

The libraries hold services only, and no command. Those that run without
the repository, `wpt-worker` (on a gpu worker's machine), `wpt-model` and
`wpt-inference` (a model folder and the text it writes), install alone.

## The app

One project, `wpt` (`apps/wpt`; the workspace's root project is
`wpt-workspace`), holds the one command, `wpt`:

- the subcommands: `wpt depot`, `wpt store`, `wpt agent`, `wpt train`,
  `wpt eval`, `wpt chat`, `wpt manager` and `wpt doctor`,
  each with its own options (`wpt SUBCOMMAND --help`). The module `wpt`
  (`apps/wpt/src/wpt/__init__.py`) names the module whose `main` runs each
  one and imports it only when it runs, so a light subcommand never loads
  PyTorch; `python -m wpt` is the same command. The controllers of the
  depot, the store and the agent harness are `wpt/depot/`, `wpt/store.py`
  and `wpt/agent.py`; the jobs are the subpackages `wpt.train`, `wpt.eval`, `wpt.chat` and `wpt.manager`.
- the web server, `wpt depot web` (`wpt/depot/web/`: `server.py`,
  `routes.py`, and `ui.py`, which builds the interface of `apps/wpt/ui`
  with Bun). The root `package.json` makes one Bun workspace of the
  interface and the packages of the organization's design system, the git
  submodule `vendor/wic-design-system`, whose tokens and components the
  interface is built on; `git submodule update --init` fetches it, and
  `bun install` builds its packages. What its routes and the subcommands share is
  `wpt/depot/shared.py`. It listens on 127.0.0.1 only; another machine
  reaches it through an SSH tunnel or Tailscale ([Two
  interfaces](#two-interfaces)).

The app's tests live in `apps/wpt/tests/`, one folder per controller or
job, and share their setup through `apps/wpt/tests/conftest.py`.

## Checks

The tests prove that things work, not that they are correct.

`ci/` holds one script per part: `lock.py`, `tests.py`, `rust.py`, `data.py`
and `commits.py`, each run with `uv run ci/NAME.py`. `ci/all.py`
runs every part, each even after one failed, and lists what passed, what failed
and what was left out. Each part prints its time, and the tests part prints its
slowest tests. Only what a change touches runs; `--all` runs everything.

A change is the difference between the files as they are, committed or not, and
a base: where the branch left its upstream branch on a machine, the last green
commit in CI, or `--base REV`. The parts:

- `lock`: `uv.lock` matches every `pyproject.toml`. It runs when one of them
  changed.
- `tests`: the tests of every package, below.
- `rust`: the tokenizer's Rust tests. It runs when the tokenizer's Rust changed.
- `data`: the recipes' check (`wpt_recipes.Recipes.check`): every recipe,
  tokenizer lock and image file matches its schema, and the rules across
  files hold (a recipe names a tokenizer recipe that exists, a series
  derives every size, an image's pool takes no built-in tool's name, a
  recipe names its depot and dataset). It runs when `data/` changed, or
  `wpt-recipes`, the root `pyproject.toml` or `uv.lock`.
- `commits`: the author's sign-off (`Signed-off-by: <author>`) on every
  commit, merge commits aside. It always runs, on what a push sends:
  `--range`, by default from the base to `HEAD`.

There is no local hook: run `uv run ci/all.py` before pushing.

### Which tests run

pytest-testmon keeps a record of the source lines each test ran, and runs a test
again only when one of those lines changed or when it failed. The record reads
what coverage measures in the test's own process, and `ci/startup/` extends it to
the programs a test starts: a startup file on `PYTHONPATH` makes each of them
write every source file of ours it loads into a folder named per test, and a
pytest plugin makes the test depend on all the lines of those files. A gpu
worker's job keeps that search path behind its checkout's for this reason. The
startup file acts only in the project's Python, since other Python programs
run with the same environment.

Coverage sees only Python, and not the files of `ci/startup/`, which run before
coverage starts or only in a child pytest. A change to any other file under
`packages/` or `apps/`, a Rust file included, to any file of `ci/startup/`, or to
the root `pyproject.toml` or `uv.lock` runs every test, and so does a missing record. The hashes of those files stay beside
the record, written when a run passed. A change elsewhere in the repository,
under `data/` for example, selects no test.

testmon sees only code that runs. A test that reads files as text instead of
running them is selected only when the code it runs changes, so CI can be green
with such a test broken. `packages/wpt-recipes/tests/test_data_access.py` reads
every source file to enforce that only the service layer reaches `data/`, and it
does not run on most changes. A change to `data/`, `LICENSE` or `scripts/`
selects no test. This is accepted.

A machine's record is in the user's cache folder, or the folder
`WPT_TEST_RECORD` names, and every checkout of the machine shares it, so a new
checkout does not run everything. It stays outside the checkout because its
changes would change the hash of uncommitted files that a test reads. A run with
`--cpu` and a run without it each keep their own record, since their tests do
not see the same machine.

The image build tests (marked `image_build`, in
`apps/wpt/tests/agent/test_images.py`, and the test of the agent
image's tools and check library in
`packages/wpt-agent/tests/test_agent_image.py`) lock and build
real images, which takes minutes. They run apart and keep no record, with `--all`
or when the change touches image code:
`packages/wpt-agent/src/wpt_agent/images.py`,
`apps/wpt/tests/agent/test_images.py`,
`packages/wpt-agent/tests/test_agent_image.py`, the root `conftest.py`
(the `image` fixture),
`packages/wpt-recipes/src/wpt_recipes/recipes.py` (its
`Recipes.image_files` decides which files an image's tag covers), the
schemas `image-lock.schema.json` and `image-licenses.schema.json` in
`packages/wpt-recipes/src/wpt_recipes/schemas/`, or anything under
`data/images/`. Otherwise the part leaves them out and says so; a plain
`uv run pytest` runs them with every other test.

The two task step tests that run attempts in a real image
(`apps/wpt/tests/train/test_steps.py`) keep its lock in
`.pytest_cache/` and the built image in Docker, for as long as the
image's files and `wpt_agent/images.py` stay the same. Only the first
run on a machine, and every CI run that runs them (which starts empty), pays for
locking and building it; when those files change, the next run locks
and builds again and removes the old image.

### How the tests run

None needs a server: the S3 test runs against moto, which answers S3 requests
inside the test process. The tests run in parallel (pytest-xdist), each process
limited to one PyTorch thread and kept off the GPU, since a GPU that shares the
machine's memory gives every process its own share of it; a plain
`uv run pytest` still uses the GPU. A test passes in both. One whose
claim a GPU cannot hold keeps its programs off the GPU in both (the
`on_the_cpu` fixture of the root `conftest.py`): a comparison of two
trainings bit for bit, since two runs on one GPU differ
([training.md](training.md), "What is exactly repeatable"), and several
processes of a training on one machine, since NCCL refuses two processes on
one GPU. A limit in seconds on what a test waits for stands far above what
that takes, so other work on the machine does not reach it.

The number of processes is fixed before they start, so it comes from
`wpt_utils.parallel.limit`, the rule's form for a count fixed in advance. The
part first runs a sample of the chosen tests (every 40th, in collection order)
alone, under a worker's conditions, and measures the memory that run held on
average (`wpt_utils.parallel.average`: the mean of the readings taken while it
ran, of the program and every program it starts, what they share counted once
where the system says how it is shared, by each page's share in
`/proc/PID/smaps_rollup`, and by resident memory elsewhere). One test process is
taken to hold that much, and the count is the smaller of 75% of the
cores and 75% of the memory available divided by it, at least 1, and never more
than the tests. Where the memory of the sample cannot be measured, or the
available memory cannot be read, the cores alone decide. The part prints the
number, the sample's result and both inputs.

### The `--cpu` option

`--cpu` runs the tests under CI's software conditions, on any machine, with no
second environment: `WPT_CPU=1`, the setting of `wpt_utils.device.select_device`
that forces the CPU, every GPU hidden, and Triton blocked at startup in the
project's Python. The last two go together, since a visible GPU under ROCm cannot
start without Triton. CI always passes it. A run with `--cpu` shows what CI's
software shows: the same tests collected, run and skipped. Only CI shows what
depends on its machine: the timing of its 4 cores. A green run here is no promise
of a green CI.

### In CI

`.github/workflows/ci.yml` runs on every pull request and push, and compares
with the last green commit: `ci/last_green.py` finds the last successful run of
the branch, of the same kind of event, through the GitHub API, and the test
record of that commit comes from the Actions cache, where only a green run saves
it. A part that failed therefore runs again until it is fixed, since its change
stays in the comparison. With no green run or no record of it, everything runs.
The `commits` part keeps reading the pushed range.

### What each part takes

Each figure comes from one run. A laptop's times depend on the workers
its free memory allows, and a CI run on its 3 workers.

A full run is meant to take 25 minutes on CI and 15 minutes on a local machine.
Nothing fails when a run takes longer: the check prints its times so that growth
gets noticed.

| Part | 16 cores, 30 GB of memory | CI, 4 cores |
| --- | --- | --- |
| `lock` | under 1 s | under 1 s |
| `tests`, every test | 12 min on 5 workers, 41 min on 1 | 21 min on 3 workers |
| `tests`, nothing changed | 7 s | |
| `rust` | 3 s, its build done | 12 s |
| `data` | under 1 s | |
| `commits` | as long as the range; under 1 s for one ordinary commit | |
