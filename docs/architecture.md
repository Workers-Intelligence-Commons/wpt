# Architecture

**The texts live in a depot; the repository holds none.** The repository
holds the code, the model and series orders, and the images of agentic
tasks. The texts, the decisions on them and what to train on live in
depots, and everything built or trained lives in a dock.

- A depot is one SQLite file, known by its address, of tables: the texts and the
  structured items (conversations, tasks), each of one kind and with its
  status, tags, scores, licenses, credits and notice; the people's
  decisions on them; the snapshots, each a change number, named by the
  UTC time it was made, and the texts admitted then; the erasures; and,
  kept read only, the tokenizer recipes and vocabularies a depot may hold.
  A hand-picked selection of texts is a tag. A depot can follow an upstream depot and pull its updates
  (`depot`).
- A corpus is what a model order's data section is built into, on the
  dock: the token files, the conversations and tasks, and copies of every
  text's credits, licenses and notices, so it says everything about what it
  holds even once the depots it read change or vanish (the freight's
  delivery, [api/freight.md](api/freight.md)). It is cut into portions,
  each named after its role, which WPT gives it: the held-out portion,
  filled first, that no training reads; the pretraining one; the
  fine-tuning one ([data-selection.md](data-selection.md), "The portions of
  a corpus").
  A corpus is a cache: the same build gives it again, so nothing keeps one
  for reproducibility.
- A model order names a depot by its public address, a snapshot of it (with the change
  and digest it reads), the roles of its corpus's portions, its shares, sizes and vocabulary;
  training reads the corpus of that request's delivery key from the dock, and every size of a
  model series reads the same corpus, the one of its largest size, the smaller sizes the first
  part of its pretraining portion;
  `total_tokens` is the size the pretraining portion is made at and a ceiling on what training
  reads, and training reads what a corpus made at that size holds. An order of
  the older dataset shape names a dataset of a depot of format 12 or older, read
  through the frozen legacy reader ([training.md](training.md), "Models trained
  before snapshots").

## The parts

Four parts make a model, each with its own packages and its own command:

| Part | Packages | What it never knows |
| --- | --- | --- |
| depot | `depot`, `depot-app` (`depot`) | models, tokens, vocabularies, the freight, the dock |
| freight | `freight`, `freight-app` (`freight`) | what a token or a vocabulary is |
| dock | `dock`, `dock-app` (`dock`) | everything else |
| WPT | every other package of `packages/`, `apps/wpt` (`wpt`) and `scripts/` | the depot, which only `wpt adopt-vocabularies` reads (below) |

The depot, the freight and the dock are meant to serve other kinds of
models than WPT's transformers, such as diffusion models, world models and
convolutional networks, and other kinds of data than text, such as images,
video and sound. They are meant to leave this repository for one of their
own, and may be written again in another language. So each knows nothing
of WPT. The aim is that what one offers the others goes through its `api`
module alone, whose calls carry only values JSON and byte streams can
carry, so that each `api` can become web routes without its callers
changing. The code is not all there yet: the exceptions below say where a
part still reaches into another. Until they leave, they live here as three parts, each with
its own packages, its own command and its own document under `docs/api/`.

What one part offers the others is one module named `api`: `depot.api`,
`freight.api` and `dock.api`, each described in a document of its own
([api/depot.md](api/depot.md), [api/freight.md](api/freight.md),
[api/dock.md](api/dock.md)). It is the only module of a part that code
outside the part imports. A part's own command and web app may import all
of it. The parts import one another only so:

- the depot imports none of the other three;
- the dock imports none of the other three;
- the freight imports `depot.api` and `dock.api`, nothing else of them;
- WPT imports `freight.api` and `dock.api`, never the depot;
- `wic_utils`, the helpers every part shares, imports none of them.

`packages/wic-utils/tests/test_part_imports.py` reads the imports of every
module and fails on any other, except those its list `EXCEPTIONS` names,
each with its reason: the depot's own command and web app write its
backups to the dock, keep the Translate button's models there, and read
the corpora built before the freight there, which its frozen dataset
routes send and an upgrade from a format before snapshots reads;
and `wpt adopt-vocabularies` reads the vocabularies the depot's tokenizer
recipes trained through `depot.api`. Every call of `dock.api` names the
dock by its URL and exchanges only values and bytes: no part holds a dock
object or a path into the dock's files, and a part that needs a local file
has the dock fetch it into a folder of its own ([api/dock.md](api/dock.md),
"The calls").

## Components and services

Each component has one role. Every operation is a service: a function with
its inputs, its result and its named failures. The `depot` commands
and the `depot web` routes are controllers: each reads its input
(arguments, a request), calls the same services, and presents the result.
A controller holds no rule of its own, so the two interfaces cannot
disagree.

| Component | Role | Code |
| --- | --- | --- |
| depot | every change to a depot's tables, under its rules, and the snapshots, listings and content the freight reads ([api/depot.md](api/depot.md)) | `depot` |
| service layer | the only code that reads or writes `data/` ([The service layer](#the-service-layer)) | `wpt_orders/orders.py` |
| freight | a request's items chosen in a snapshot and prepared by one program into a delivery on the dock ([api/freight.md](api/freight.md)) | `freight` |
| dock | files, compressed, on a folder or a bucket ([The dock](#the-dock)) | `dock` |
| training | the steps an order names, through gpu workers | `wpt.train`, `wpt_training`, `wpt_worker` |
| agent harness | the containers, tools and checks of agentic tasks, and the Git repository a task works in | `wpt_agent` |
| evaluation | tests on a trained model, into its model card | `wpt.eval` |

## The depot

A depot is one SQLite file, given as a path or a `file://` address and
known by that address. The setting `DEPOT`, in the environment or in
`.env`, names the depot that `depot` and `depot web` work on, as
`DOCK` names the dock. `depot new ADDRESS` makes an empty one,
with its name, its maintainer and its takedown contact. The setting
`DEPOT_URL`, read the same way, is the depot's public address, the web
address of its `depot web`. `/api/about` answers it as the depot's
address, and the frozen dataset routes and `depot upgrade` name the depot
by it, never by its file; a depot without it is named by its name. A
corpus the freight delivers names the depot by the address its order's
`data.depot` gives, and each of its items by the depot's name
([data-selection.md](data-selection.md), "Corpora").

Its tables are `about` (the depot's name, maintainer, takedown contact
and upstream), `changes` (every change: its number, its kind and its
time), `numbers` (the last number of each kind of the depot's own ids),
`licenses`, `notices`, `kinds`, `decisions`, `texts`, `content_lists`
(every content version of each text, as the list of its parts, with its
kind, never null), `parts`
(the parts of each text's content versions), `snapshots` and
`snapshot_sets` (each snapshot and the texts it holds), `erasures` and
`pulls` (for a depot that follows another, how its change numbers meet its
upstream's). Every table declares a primary key. A depot brought up from
format 12 or older also keeps the legacy tables of its libraries and
datasets (`libraries_revisions`, `memberships`, `pins`, `library_moves`,
`datasets_revisions`) as they were: no write touches them, and only the
frozen legacy reader reads them (`depot/legacy.py`). What each table
holds and the rules on it are in [data-selection.md](data-selection.md)
(texts, items, tags, snapshots and corpora), [licenses.md](licenses.md)
(licenses, notices and decisions) and [erasure.md](erasure.md)
(erasures).

**Ids.** Every id (of a text, a decision, a snapshot, an erasure or a
notice) is the depot's name, a hyphen and a number: `my-depot-17`. The
depot's name is made of letters, digits, `.`, `_` and `-`, and begins with a letter or a digit. A reference to an
id in another table, in a snapshot, in a corpus or in a vocabulary's list of texts holds
the whole id, so an id names one thing across depots. The number is one
past the last number of that kind of id the depot gave (`numbers`), an
erased text's included, so a number is never given twice. A clone takes a
name of its own, since it never reuses its upstream's.

**Writing.** The file is in WAL mode. Several processes may open it: each
reads at any time, and SQLite's lock lets one write at a time, the others
waiting for it. `depot.depot` is the only code that changes a depot.
Each operation is one change: one transaction, under the next change
number, that checks the whole change against the depot's rules inside it,
so a refused or failed change writes nothing and a crash never leaves a
change half written (`depot/database.py`).

A session of SQLite records every row a change writes or deletes. Before
the transaction commits, that record goes to a change file in the waiting
folder beside the depot's file (`<file>.waiting`), named by the change
number, the number of the change before it and its time. The file is
synced and kept only once the change is committed, so the waiting folder
holds exactly the depot's committed changes since its last backup. A
deleted row's bytes are overwritten in the file, so an erasure leaves no
copy of what it removed in the file's free space.

**Revisions and change numbers.** Every change of a depot has a change
number, one past the last. `about`, `texts`, `erasures` and `tokenizers`
keep each change of a row as a new row, its revision, keyed by
the id and the revision number and recording its change number; the
highest revision is the current one. A text's content versions are
rows of `content_lists`, keyed by the text's id and the version number and
recording their change number, and each revision of the text names the
content version it reads and keeps that version's kind, length in
characters and UTF-8 bytes (`content_kind`, `content_length`,
`content_bytes`), copied from it by every write that makes a revision or a
content version and set again by an erasure's cut, which changes a
version's length in place. The counts, a decision, a change of licenses or
tags, a snapshot's read and a listing's walk read them from the revision
alone, never from `content_lists`. The table `quotes` records, for each content
version of a conversation, the texts its excerpts quote, so an erasure
finds the conversations quoting a text it reaches without reading every
conversation; every write of a conversation keeps it (an add, a change of
content, an erasure's cut, a clone or a pull, which makes it from the
content it receives and never receives it). A revision of a text also holds its scores,
each with the content version it was made from. A decision on a text is a
row of `text_decisions`, with the change number it was made at, so a
decision on a million texts writes a million small rows and no revision.
A text's tags are rows of `taggings`, never a revision of their own: one
for each text and tag, with the change the text joined the tag at and the
change it left it at, so a change of tags writes a small row for each tag a
text gains or loses (`depot/taggings.py`). A depot can so be read as it
was at any change number, which a snapshot names, its texts' tags
included. The current revision of each text also holds a copy of the tags
it has now, in the order of their names (its column `current_tags`): every
change of tags writes it again in place, and a new revision takes it over,
so a decision, a count and the `texts` view read a text's tags on the row
they read already, as fast as when a revision held them.

**Status and counts.** The current revision of each text stores the status
its decisions make, and the index `texts_status` holds the current texts by
status, then in the numeric order of their ids. The index `texts_length`
holds them by status, then by the length in characters of the content
version each reads, then by id, so the texts of any filter can be read the
longest first, a page at a time. The table `counts` holds
how many current texts the depot has, and how many have each status, each
tag, each license and each score name and each decision is on, and, for
each score name, how many have each of its values, to three significant
digits, so the review's choices follow the scale the whole depot holds,
read from one range of its rows. It splits
every text, and the texts of each tag and each license, by status. The
tables `text_tags` and `text_licenses` list
the current texts of each tag and of each license, in the numeric order of
their ids.
The title index `text_titles` is an SQLite FTS5 table of the trigrams of
each current text's title, which stores no title. Its rows are numbered by
the texts' ids, through the slot `title_owners` gives each depot name the
ids carry, so one name's texts come in the numeric order of their ids. A
text whose id is not a name, a hyphen and a number without leading zeros
cannot be numbered, and a write that would index one is refused. `counts`
also holds what the current texts of each status hold, in texts,
characters and UTF-8 bytes: by kind (kinds are a closed list, the rows of
`kinds`), by plain tag set (the plain texts whose set of tags is exactly
that one) and, through the splits of each tag and license, by tag and by
license. A snapshot's summary is read from these, so making one reads no
text. Every
write keeps the stored statuses, the counts, the two listings and the title
index right in its own transaction, so a page reads them at once whatever the depot's
size; one module writes them (`depot/counts.py`), and `depot
recount` makes them again from the rows, after copying again onto each
revision the kind, length and bytes of its content version, and onto each
current one the copy of its text's tags, where they differ. The title index is written through
`depot/titles.py`, which that module calls.

**Snapshots.** A snapshot (`depot/snapshots.py`) is a name on a change
number: the texts admitted at that change, each read at its revision as of
it. Its name is unique among the snapshots of the depot that made it, so a
clone's snapshot never collides with its upstream's of the same name. The
table `snapshots` holds its row (the change it reads, its totals, a SHA-256
digest of the text numbers it holds and a summary by kind, tag, plain tag
set, license and owner, read from the counts), and `snapshot_sets` holds
its texts, one row for each part and each depot whose texts it holds: the
change of that depot's numbering they are read at and their numbers as a
portable roaring bitmap (pyroaring), a few bytes for many texts with
neighbouring numbers. Making one reads the index `texts_status` a page of
50,000 ids a read transaction, then, in one short read transaction, reads
again the texts that moved meanwhile (through the indexes of change numbers
of `texts_revisions` and `text_decisions` and the erasures written
meanwhile) and the summary, and writes the rows in one short change, so no
daily write and no erasure's checkpoint waits for it. A pull in between
starts the reading over. The digest is computed from the sorted numbers,
never from a roaring library's bytes, so anyone can check it
([data-selection.md](data-selection.md), "A snapshot's digest"). A snapshot is never changed nor removed: an erasure
cuts the texts it holds in place, and a read skips and counts a text an
erasure removed whole.

**Reads that do not grow with the depot.** A page, a count step and a write
of a few texts read only the rows they name, through an index: the
listings above and `text_decisions_order` of each decision's texts in the
numeric order of their ids. The words of a search are read from the title
index when one has 3 characters or more and the search does not read the
texts themselves. A page reads a bounded window of candidates, but a filter
no listing narrows reads up to every text over its pages: a search whose
words are all shorter, a search in the texts, and two parts of a filter
whose texts meet in few. A kind is matched on each candidate, and the
counts answer how many texts of each kind and status there are. A pull reads each table's
rows past a change number through an index of their change numbers, the
rows of every depot numbered in that range. A query names the index or the join
order it needs where SQLite would otherwise choose to read a whole table.
`packages/depot/tests/test_reads.py` runs each such route on two depots,
one twice the size of the other, and fails when its reads grow.

**A snapshot's order.** The depot gives a snapshot's texts, in its
listing (`depot.api.listing`), in the order of their export keys
(`depot/order.py`), which every delivery of the freight follows. The order is made in memory from the
snapshot's sets, never from stored keys: each text's key is worked out from
its id, a number several parts hold is read in the lowest part, and the
whole is sorted, at 8 bytes a text once sorted. A plain text longer than
1,048,576 characters comes as pieces (`depot/pieces.py`), each at the
key of its own name, so a long text spreads over the order; the walk finds
those texts from the index `content_lists_long` of long content versions,
so finding them grows with the depot's long contents, not with the
snapshot's texts. A walk reads each text at its revision as of its part's
bound, a page at a time, and only the texts it reaches, so a delivery at a
small size starts at once whatever the snapshot's size. A request with
shares is the exception: each share splits its part over the tags of its
texts ([data-selection.md](data-selection.md), "Filling a part"), so the
freight's first walk reads every entry of the listing, keeping numpy arrays
and no object per text (`freight/groups.py`), before the stream takes the
texts it chose. What a delivery needs of the whole snapshot (the bytes of
each share and of the conversations and tasks) comes from the snapshot's
summary, never from a recount, so the freight plans the same after an
erasure.

**Contents.** A content version, a plain text or an item's JSON, is the
ordered list of its parts, and its row of `content_lists` holds that list,
each part's SHA-256 and length, with the version's length in characters
and in UTF-8 bytes. The same rule cuts every text, whatever its size, so
a text of many gigabytes, such as a repository with all its files and its
history, is one text (`depot/parts.py`). A text shorter than a part's
least size is one part. A cut falls only right after a line end, at a line
end that a hash of the bytes ending there chooses, so a part holds a few
megabytes on average, and a cut depends only on the bytes around it. A
part never holds more than a fixed maximum, far under the most bytes one
SQLite value holds: past it, the part ends at its last line end, or, with
none, at its last whole character. A row of `parts` holds one part,
compressed with zstd, keyed by the text's id and the part's hash, stored
once for that text and shared by none other. An edit changes only the
parts around it, so a new content version stores only the parts no
version of the text held before. An erasure writes again only the parts
that hold a character it cuts, and the versions list the new parts
instead. A part that no version lists any longer goes. A text bigger than
memory is added from a file read as a stream. An erasure, a clone and a
pull read and write a text's parts a few at a time. The content the
freight asks for (`depot.api.content`) is read from the parts, a piece of a
long text from the parts its characters lie in alone.
`depot show` gives a text whole, so it holds it in memory, and with
`--from` and `--characters` it reads one passage from the parts that passage
reaches alone. `GET /api/texts/{id}` gives a plain text of more than
2,000,000 characters as its start alone, and `GET /api/texts/{id}/passage`
reads any passage of it, so the web app never holds a long text whole.

**SQL.** Every query is SQLite's SQL. A list column (a text's licenses,
authors, tags, decisions and scores, a snapshot's summary, and the others
`depot/schema.py` names) is JSON text, so a tag test is `EXISTS
(SELECT 1 FROM json_each(tags) WHERE value = 'book')`. A read query sees
each revised table's current state under its plain name (`texts`, `about`,
`erasures`, `tokenizers`) and every one of its rows under
`<name>_revisions`. The `texts` view joins each text with the content
version its current revision reads, its kind and its length. The
`contents` view gives each content version's whole text, its parts joined
in order, and `content_lists` its rows as they are. The other tables,
`parts`, `snapshots`, `snapshot_sets`, `text_decisions`, `taggings`, `counts` and
`pulls` included, are seen as they are, the legacy tables of a depot
brought up from an older format too.
The `texts` view gives each text's decisions as a list read from
`text_decisions`, in order. The column `decisions` of `texts_revisions` is
unused: every revision the current format writes holds `[]` there, and a
past revision brought up from an older format keeps the list it held, so a query over
past revisions reads the decisions from `text_decisions` and its change
numbers. The column stays because removing it rewrites the whole table.
The column `tags` of `texts_revisions` is unused the same way, read only by
the frozen routes of the legacy tables, which read the tags a revision held
under an older format: the `texts` view gives each text's tags from the copy
on its current revision, the same as the rows of `taggings` it has not left,
and the tags a text had at a change are the rows of `taggings` that cover it.
A query that names no text column
reads the views without joining any part, so a query over the texts'
other columns costs the same whatever their size. A query that reads the
text of a content version bigger than one SQLite value holds fails with an
error that names the version, and such a text is read from its parts, as
`depot show` reads it.

A person's SQL (`depot sql`, the condition of a filter) runs on a
connection that opens the file read-only and whose authorizer lets a
statement only read rows and call functions: a statement that writes,
attaches another file, runs a pragma or loads an extension is refused, and
a query is one statement. The one pragma it may run is reading
`data_version`, which the title index reads to know whether the file
changed.

**Format.** The depot's layout, its tables and their columns, has a
number: its format. The code holds the format it reads and writes
(`FORMAT` in `depot/schema.py`). Each `about` row records the depot's
format when it was written, so the current row gives the depot's format.
A depot of format 3 or older is a folder of Delta Lake tables, which only
`depot upgrade` reads (`depot/delta.py`).

Every way into a depot that reads or changes its rows refuses one of
another format than the code's, older or newer: every `depot` command
but upgrade, `depot web`, and a clone or a pull, which need both the upstream and the clone at the
code's format. The refusal names the depot's address, its format and the
code's, and says to run `depot upgrade` for an older depot or to
update wpt for a newer one.

`depot upgrade` runs only when asked (`depot/upgrade.py`). On a
folder of format 3 or older, it first runs in the folder the steps from
its format up to format 3, then writes a new SQLite file of format 4 from
it: by default the folder's path with `.sqlite` added, or the file `--to`
names. It never changes the folder after format 3 and never
writes over a file. It reads each table's data files a batch at a time, so
its memory does not grow with the depot. The file is written as
`<file>.upgrading` and renamed once whole; an upgrade cut short goes on
from the last data file it wrote when run again. A folder whose journal
holds a change never finished is refused, since only the code of its
format could finish it. A clone of those formats named its upstream by
another address than a web address, so its upgrade takes the upstream's
web address (`--upstream`) and is refused without it. A newer depot is
never brought down: the command refuses it.

An SQLite file of format 4 or newer, the one named or the one just
written from a folder, is brought up in place, one step per format. A
step records a new `about` revision at its format under a change of kind
`upgrade`. A step that fits one transaction leaves the file as it was when
it is cut short, and runs again whole. A step that rewrites every current
text runs in several transactions, so its WAL stays the size of one batch:
the first marks the depot as half upgraded, each next one writes one batch
of rows and records where it stopped, and the last removes the mark. Every
way into a half-upgraded depot but `depot upgrade` refuses it, and the
upgrade run again goes on from the batch after the last one written. It
says how far it is as it goes. The change files record no change of layout, so
before a step the file is marked for a full copy, under a new timeline, at
its next backup. The step to format 5 builds the index `texts_order`, the
current texts in the numeric order of their ids. It is one `CREATE INDEX`
that reads each current text's row once and rewrites no table, and it says
what it does before it starts and how long it took. The step to format 6
writes each current text's decisions as rows of `text_decisions` and its
status, a batch of rows at a time, then builds the index `texts_status`
and fills `counts`. It also makes the tables of the tokenizer recipes, of
their vocabularies and of the texts each vocabulary trained on, empty, and
the column `max_tokens` of the datasets. The step to format 7 makes the
column `export_key` of the memberships and of the pins, and the column
`vocabulary` of the datasets. It writes the export key of every membership,
then of every pin, a batch of rows at a time, saying how many rows a second it writes; run again after
a stop, it goes on in the table and at the row it reached. Then it builds
the indexes `memberships_key` and `pins_key`. The step to format 8 builds
the index `text_decisions_decision`, the rows of `text_decisions` by
decision, in one `CREATE INDEX` that reads each row once and rewrites no
table. It marks the depot as half upgraded before the build and removes the
mark after it, so a stop during the build leaves a depot that only the
upgrade opens, and the upgrade run again builds the index whole. It says
what it does before it starts and how long it took. Format 8 is also the
format of a depot that may hold decisions that put texts back under review,
which code of format 7 does not know, so that code refuses such a depot.
The step to format 9 makes the columns `characters` and `bytes` of
`counts` and fills what each library holds by kind and by tag. It reads
the memberships that have not ended, the current revisions and the content
versions in one sequential scan each, sorted by text id, and merges them,
so no row is looked up through an index and the time grows with the rows
whatever order the file stores them in. SQLite sorts the three scans at
once in its temporary folder, which `SQLITE_TMPDIR` names, and they take
about 100 bytes of free disk a membership; the step says how much before
it starts. It writes the sums a batch of
memberships at a time, marked half upgraded until the last, saying how
many rows a second it counts; run again after a stop, it goes on after the
last text it wrote. The same merge gives the latest change each library's
current texts were revised at, and one read of the index `memberships_key`
gives the latest change its memberships joined or left at, which make
`library_moves`; in a clone, whose rows carry several numberings, the change
of each library is unknown. Its last transaction gives each dataset
revision the figures those prove exact.
The step to format 10 makes the listings `text_tags` and `text_licenses`,
the title index, the counts of score names and the splits by status and by
locked library, from the current revisions read in rowid order, a batch at a time, marked
half upgraded until the last, saying how many rows a second it reads; run
again after a stop, it goes on after the last batch it wrote. It then
builds `memberships_order`, `pins_order`, `text_decisions_order` and the
indexes of change numbers, each in one `CREATE INDEX` that a run after a
stop builds again if it is missing. Its last transaction drops
`text_decisions_decision`, which `text_decisions_order` serves in its
place, and counts the pins older than their text's current revision.
The step to format 11 builds the partial index `content_lists_long`, the
plain content versions longer than 1,048,576 characters that a corpus
build cuts into pieces, in one `CREATE INDEX` that reads each row of
`content_lists` once, marked half upgraded while it builds as the step to
format 8 is.
The step to format 12 gives every text a kind: a content version with no
kind is a plain text, and becomes of kind `plain`; it counts what each
library holds of each kind on its own; it gives the kind `conversation`
the schema that takes the `excerpt` block (docs/chat-format.md,
"Excerpts"); and it fills the table `quotes`, the texts each content
version of a conversation quotes, through which an erasure finds the
conversations that quote the text it reaches. Its first transaction marks
the depot half upgraded, drops `content_lists_long`, whose condition names
the kind, adds the row `plain` to `kinds`, writes the conversation schema,
makes the empty table `quotes` and its index, and deletes the counts of
what libraries hold by kind but those of their plain texts, which stay
right. Its next transaction counts, for each library that holds their
text, the current content versions of the other kinds, by kind, so the
step needs no recount: the items are copied from one sequential scan of
`content_lists`, then found among the current revisions and the
memberships that have not ended by one sequential scan of each, or, when
they are fewer than one revision in 40, each looked up through an index,
which then reads fewer pages. It then writes the kind `plain` on the
content versions with none, in rowid order, a batch of 250,000 rowids a
transaction, and writes the rows of `quotes` of the conversations among
those rowids, each read from its parts, one lookup in `parts` for each
conversation. Each transaction records the last rowid it reached and the
step says how many rows a second it writes; run again after a stop, it
goes on after the last batch it wrote. Its last transaction checks that no
content version is left without a kind, declares the column `kind` of
`content_lists` `NOT NULL` by changing the table's statement in
`sqlite_schema` (with `PRAGMA writable_schema`, as SQLite's documentation
allows for a constraint every row already meets, so no row is rewritten),
builds `content_lists_long` again over the kind `plain` (built last, so no
batch has an index to keep up as it writes a kind), gives the dataset
revisions' figures a count of each kind (a library's figures that hold no
item become its plain texts and every other kind at 0, those that hold
items, of a library the depot owns, are read again from the counts when
the counts prove them exact, and are unknown otherwise) and drops the
mark. It reads `content_lists` four times and writes again each page that
holds a content version of no kind. On the synthetic depot of format 11 of
`packages/depot/bench/large.py` (4,624,673 content versions, 8% of them
items and 6% conversations, 18,322,341 revisions, 21 GB, with 1.5 GB of
memory) it took 99 to 107 seconds: 32 to 41 to count the kinds, 48 to 55
for the batches and 11 to 12 for the last transaction.
The step to format 13 turns the libraries and datasets into snapshots and
the tags of the revisions into rows of `taggings`. It
runs in stages, each recorded in the table that marks the depot half
upgraded and struck off as it ends, so a run after a stop starts again at
the first stage not ended. Its first transaction also records the dataset
revisions the corpora of the dock were built from, so a run after a stop
needs no dock. Its stage `tables`, in one transaction, makes the tables
`snapshots`, `snapshot_sets`, `legacy_snapshot_of` and `taggings`, empty,
the index of the snapshots' names by owner and the columns of the kind,
length and bytes of its content version and of the copy of its text's tags
on each revision of a text, empty,
and counts from every current text what the
texts of each status hold by kind and by plain tag set, and the characters
and bytes of the splits of each tag and license. Its stage `taggings`
writes each tag a text's revisions held as a row of `taggings`, joined at
the change of the revision that first holds it and left at the change of
the next one that lacks it, a batch of texts at a time, each batch in
its own transaction recording how far it went, so a run after a stop goes on
after the last batch written. Its stage `sizes` copies
onto every revision of a text the kind, length and bytes of the content
version it reads, and onto each current one the copy of its text's tags,
the same way, a batch of revisions in each transaction. No stage looks a
row up through an index: on a file far bigger than the memory each lookup
is a random read of the disk. Each reads the tables it needs in sequential
scans, sorted by text id where it must, into tables of SQLite's temporary
database or straight into a merge, so SQLite's temporary folder
(`SQLITE_TMPDIR`) needs free disk, up to about 120 bytes a revision of a
text, which the check says. On a synthetic depot of 18,322,341
revisions, 21 GB, with 1.5 GB of memory (`packages/depot/bench/large.py`),
the stage `tables` took 50 to 65 seconds, `taggings` 75 to 82 and `sizes` 66 to
76, and the upgrade added 1.8 GB. Its stage `locked` makes
a snapshot of each locked library the depot owns: the texts it pinned,
read at the change that locked it, with the bounds of every depot's
numbering the depot held then. Its stage `datasets` makes one of each
dataset revision a corpus was built from and of each dataset's current
revision, with one part for each library it reads, in its order, each
read as the dataset read it (a library locked by then by its pins at its
lock; another by the texts it held at the change the revision names); a
revision that read one locked library alone gets that library's snapshot,
and `legacy_snapshot_of` names, for each revision converted, its snapshot.
A snapshot takes the library's or dataset's name, every character a
snapshot name cannot hold made `-`, with `-r<revision>` for a dataset
revision other than the current one and a number added on a clash; its
`origin` keeps the name it had. A clone converts only what it owns: its
upstream's snapshots arrive by pull. Open libraries are not converted. For
every text of every snapshot, the revision the snapshot reads must be the
one the library read: a text that differs stops the stage, the snapshot
unwritten. The stage then says, for each dataset revision converted, the
data section of a model order that builds it. Its stage `counts` deletes
the counts of what libraries hold and folds the splits by whether a locked
library held the text, and its stage `views` drops the views `libraries`
and `datasets` an older file may hold and deletes the strip reader's
folder beside the file. The tables of libraries and datasets stay in the
file, unwritten; nothing is dropped and the file is not vacuumed.

The step to format 14 counts the values of every score of the current
texts into `counts`, in one statement that reads every revision once, then
builds the index `texts_length` in one statement,
which reads every current text's row once and sorts the entries in
SQLite's temporary folder, between a transaction that marks the depot
half upgraded and one that drops the mark. On the synthetic depot of
`packages/depot/bench/large.py`, 1,500,000 current texts in a 17.7 GB
file, it took 6 seconds and added 102 MB, about 70 bytes a text; the check
of a depot of format 13 says the seconds, the disk and the full copy the
next backup sends.

`depot upgrade` alone, or with `--check`, writes nothing in the depot: it opens the
file read-only and says what the upgrade would do. For a depot of format
11 it checks what the step to format 12 checks before it ends; for one of
format 11 or 12 it makes in memory every snapshot the step to format 13
would make and checks every revision each one reads, so a difference is
found before anything is written. It says the names each library and
dataset would take, the order data sections, the open libraries left as
they are, what is not converted and why, and the cost of every step left
together: the seconds (from the rates `packages/depot/bench/upgrade.py`
and `packages/depot/bench/large.py` measure, plus the time its own
reads took), the free disk SQLite's temporary folder needs, the WAL at its
largest, the disk added and the full copy the next backup sends. It reads
every snapshot at once in one merge of four sequential scans sorted by text
id (`depot.conversion`), each sort holding only the columns the merge
needs, then the tags and licenses of the revisions the snapshots read, in
the order of the file, all in one read transaction; on the synthetic depot
above it took 86 to 87 seconds and 1.4 GB of temporary files. It saves
what it measured beside the depot's file, in
`<depot file>.upgrade-check.json`, with the depot's state: its latest
change, the upgrade's own aside, and the largest rowid of each table it
reads. A later check, `--yes`'s included, takes those measures while the
state is the same, and reads again otherwise, so the check can run while
the web app serves and `--yes` then reads no library. A check that runs
beside the web app holds its read transaction to the end: a checkpoint
cannot copy into the file what the app writes meanwhile, so the WAL keeps
every page the app writes until the check ends, and the check's reads
take disk time and page cache from the app. It ends with what to
do before: run `depot backup` right before the upgrade, knowing that
only the wpt of the depot's old format opens that backup, upgrade the
upstream before its clones, and stop every process that has the depot open
first (`depot web`, its jobs, a backup): the code of the old format
checks the format only when it opens a depot, so a web app left running
would go on writing into the half-upgraded file. `depot upgrade --yes`
runs the check again and upgrades only when it passes, and refuses while
another connection still has the file open; the upgrade then writes the
snapshots that check measured, without reading the libraries again, when
the depot did not change in between, and deletes the saved check when it
ends. The upgrade puts a file that is not in WAL mode in it first.

Every change to the depot's layout raises `FORMAT` and adds the upgrade
step from the format before, with its test. So does a change of the schema
of a kind a new depot starts with, since each depot keeps its own copy in
`kinds`. A step names the columns it
reads and writes itself, never through the current layout, which is a
later format's, and run again on the layout it leaves it changes nothing
more.

**Following an upstream.** A depot can follow another, its upstream, the
way a Git clone follows its remote. The upstream is named by the web
address of its web app (`depot web`), and a clone reads nothing but
the answers of its follow routes, which only read. `depot clone URL TO
--name NAME --maintainer M --takedown-contact C` copies every row of every
table that travels, revisions, content lists, snapshots and their sets
included, with their change numbers, into a new depot whose `about` row
names the upstream. The tables of libraries and datasets an older format
left do not travel. `depot pull` fetches the upstream's updates: each
upstream row replaces the row of the same key, and a row only the depot
holds stays. A snapshot never changes, so one the depot holds already is
kept as it is. A snapshot's set travels with its snapshot, chosen by the
snapshot's change, its bitmap as base64 text, and the sets of every
snapshot a clone or a pull brings are checked against its digest before
anything is kept: sets that do not match write nothing. Clone and pull
read only an upstream of the code's format, so the upstream is upgraded
first, then its clones. What travels is paged, so no answer holds more
than a bounded number of rows and bytes and fits the size and time limits
of a web server in front of the upstream. A part never changes once
stored, so a clone and a pull copy only the parts the depot lacks that the
content lists they bring list, a few at a time, and a pull drops the
parts that the versions an erasure it brings cut no longer list. Its
change number is past the upstream's, so the depot's later changes come
after every change it received. The texts the upstream's admitted
erasures removed are removed too, and an erasure the depot had not
carried out is carried out there (`depot/follow.py`). A pull runs
when someone asks for it. A clone is written as `<TO>.cloning` and
renamed once whole; a clone cut short goes on from its last page when run
again.

The rows a clone receives keep the upstream's change numbers, so the
`pulls` table says how the two numberings meet: for the clone itself and
for each pull, the depot that pulled, its change number then, its
upstream's name and the upstream's change number the pull brought it up
to, the change number of the upstream's head when the pull began. A
change the upstream commits while the pull reads it stays past that
number until a later pull. An erasure the upstream carries out meanwhile
makes the pull start over, and a pull writes nothing until it ends whole.
A snapshot of the upstream reads its texts by the upstream's numbers, so
it reads the same in the clone as in the upstream, and an order naming it
gets the same corpus under the same delivery key from both. A snapshot the
clone makes holds the upstream's texts in a set of their own, read at the
upstream change the clone's pulls had brought, and its own texts by its
own numbers. A clone copies
its upstream's `pulls` rows too, so a clone of a clone follows the numbers
back through each depot.

A clone is read-only on the rows its upstream owns, those whose id does
not carry the clone's name. An id carries a name when it is exactly that
name, a hyphen and digits, so `wpt-eu-3` carries `wpt-eu` and never
`wpt`, and a clone whose name begins its upstream's name owns none of the
upstream's rows. Every operation that would change one is refused and
writes nothing, with a message that it belongs to the upstream: a
decision, a license change, a content change, an edit of credits or
scores, or an erasure request on
an upstream text, a decision on an erasure request that is the upstream's
or names an upstream text, and a change of tags over upstream texts. An operation on several texts
is refused whole when one of them is the upstream's, and the review of a
decision lists those texts apart. Only the upstream changes its rows, so
a pull never meets a clone's change to one. The clone's own texts and
snapshots stay fully changeable, and its own snapshot may hold upstream
texts. A snapshot's name is unique among the snapshots of the depot that
made it, so a clone's snapshot and its upstream's named by the same second
stand side by side. The pages show quietly that a text or erasure request
belongs to the upstream, and offer no action on it that the depot would
refuse.

**Backups.** The backups of a depot are on the dock, under
`backups/<the depot's name>/` (`depot/backup.py`). `depot backup`
sends each change file waiting beside the depot onto the dock and deletes
it from the waiting folder once the dock holds it. Its first run on a
file a command made whole (a new depot, a clone, an upgrade, a restore),
and a run with `--full`, also sends a full copy of the file, made with
SQLite's backup into a temporary file beside the depot's file while the
depot goes on being written. Each run records what it sent. Such a file
starts a timeline of its own, a random id its backups record, so the
changes of a depot restored to an earlier moment, which number on from
there again, never mix with the changes of the same numbers its origin
made.

`depot backup --off` marks a depot that nothing backs up. Its writes
then record no change file, so the waiting folder does not grow, and its
waiting change files are deleted. Its next backup records again and sends
a full copy.

The app never cuts, erases or removes a backup. An erasure does not reach
them ([erasure.md](erasure.md)). Backups on a schedule come from a timer
outside the app, such as this cron line:

    0 3 * * * cd /path/to/wpt && uv run depot backup

`depot restore TO --at MOMENT` makes a new depot file at `TO` from the
backups of one timeline: the latest by default; for a change number, the
latest that reaches it; for a UTC time, the one that made the latest change
or full copy at or before it. It takes that timeline's last full copy at or
before the moment, then applies every change file after it in order, up to
the moment. A change that does not apply exactly stops the restore, which
leaves nothing at `TO`; it never writes over a file. It reads the backups of the
only depot name the dock holds backups of, or of the name `--depot` gives. Erasures
carried out after the moment are not carried out in the restored depot.

## The dock

The dock is set by one setting, `DOCK`, in the environment or in
`.env` at the repository root:

```
DOCK=file:///home/me/dock      a local folder
DOCK=s3://bucket/prefix             an S3-compatible bucket
```

For S3, the credentials and a non-AWS endpoint come from the dock's own
settings, in the environment or in `.env`: `DOCK_ACCESS_KEY_ID` and
`DOCK_SECRET_ACCESS_KEY` (both needed), `DOCK_SESSION_TOKEN`,
`DOCK_ENDPOINT_URL` and `DOCK_REGION` (default `us-east-1`). The
dock reads no `AWS_*` setting (not even `AWS_ENDPOINT_URL` or `AWS_PROFILE`)
and no AWS configuration or credentials file, so the keys a cloud provider's
other tools use never reach it.

The setting's prefix picks the dock's kind, a class of `dock`
(`dock/kinds.py`). A kind supplies five operations on files: write a
whole file, read a byte range, replace a file only if it is unchanged
since it was read, list, delete. Everything else (the layout and
compression) is built on them, alike for every kind.

A dock holds real named things, the same on a folder and on a bucket:

```
deliveries/<delivery key>/         what the freight delivered for a request: its note, items tables and files
corpora/<build key>/               a corpus the depot built before the freight: corpus.json, token files, tables
models/<name>/                     the model's real files
runs/<run>/                        each command's record
cache/<stage>/<fingerprint>        only what can be rebuilt
backups/<depot name>/              the depot's backups
```

`freight deliveries` lists the deliveries, and `wpt list` the corpora and
models.

A delivery lies under its delivery key: a hash of the request (the
snapshot's change and digest, the selection, the preparation and its
inputs, the selection rules), so the same request always lands in the same
folder, on a server and on its clones alike. A selection can cut its
delivery into named portions, filled in the order given, a text in one
portion at most ([data-selection.md](data-selection.md), "Portions"). The freight writes and finds
them, and marks them stale after an erasure ([api/freight.md](api/freight.md),
"A delivery on the dock"). A delivery whose files a cap deleted keeps its
note, marked deleted. The corpora under `corpora/` stay readable by WPT
(`wpt_corpus/corpora.py`). Only the freight marks them stale after an
erasure, at its next check (`freight/erasures.py`, `freight/corpora.py`);
the depot marks nothing on the dock, so its frozen dataset routes send one
an erasure reached until that check ([erasure.md](erasure.md), "What it
covers"). Nothing writes a new one.

Hashes do two jobs. Every read checks the bytes against their sha256: a
record names each file it points at by the one shape `{"key", "sha256"}` (the
file's key on the dock and its sha256), a record holds the
sha256 of its own fields, and a result holds its own in its first line.
Finished work is found from its inputs: a result lies under its
fingerprint, such as a training step's result in `cache/steps/`. A file
known only by its content, such as the kept state of a step that nothing
but the step's result names, lies in `cache/files/<sha256>`.

`cache/` holds only what can be rebuilt, so emptying it leaves every delivery
and every model usable: a delivery's files and a model's files, a run's
checkpoint included, lie in their own folders, and a record never points
into `cache/` for a file it needs to be read.

### Compression

Every file is compressed with zstd while it is written, in frames of
about 1 MB compressed apart (zstd's seekable format, which plain zstd
reads, its seek table a skippable frame that standard tools pass over), so
a byte range is read without the rest. It is kept compressed, under its key
plus `.zst`, only when that is smaller, and as it is otherwise, at level 3.
An S3 answer cut while it is read is asked again, up to 5 tries in all.
Large files are compressed several frames at once, as `wic_utils.parallel`
allows. They go to S3 in parts of 64 MB, and every upload of a process
sends its parts through one shared limit of 32 at once: S3-compatible
stores such as R2 give each connection a few MiB/s, so a link fills only
with many connections at once, and uploads that run together do not
multiply them. That limit is set for network work, not by the cores; a
part starts only while the memory rule of `wic_utils.parallel` allows.

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

The web app keeps in `cache/files/` the files of Mozilla's translation
models that its Texts page asked for, each under its sha256.

The dock also holds the record of every run of `wpt train`, `wpt eval`
and `wpt manager`: one folder per run under `runs/`, written by name, each
file compressed (a local dock adds a record file's lines to one plain file
while the record runs, and compresses it into one part when the record
ends; [observability.md](observability.md)).

## The service layer

`packages/wpt-orders/src/wpt_orders/orders.py` (`Orders`) is the only
code that reads or writes the repository's data. It hands out documents
and bytes, never a path under `data/`. It holds the model and series
orders and the images of agentic tasks:
their Dockerfiles, locks, license lists and pools of tools (`tools.json`).
`wpt train`, `wpt manager`, `wpt eval`, `wpt doctor`, the
image commands of `wpt agent` and the task reader of `wpt_agent` read them
through it. It checks every file it reads against its schema, and its
check adds the rules across files: a series derives every size, and an
image's pool takes no built-in tool's name. An order's data part has the snapshot shape
or the legacy dataset shape ([model.md](model.md), "Model orders"); the check refuses
shares that name a tag twice or sum above 1. An order whose depot, snapshot, digest,
vocabulary or dataset is still empty reads, and the check names it for a person to fill in. People edit the
orders and `tools.json` as files; `Orders` writes only the generated
files (an image's `lock.json` and `licenses.json`), for the command that makes them, after their schema,
through a temporary file and a rename.

Two checks hold the rule. A test
(`packages/wpt-orders/tests/test_data_access.py`) reads the source code of
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
(Python's `sys.addaudithook`, `wpt_orders/guard.py`) in every process,
when the first `Orders` is made: any opening, listing, removal
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

The depot has two interfaces, both thin layers over `depot`: the
`depot` commands, and the HTTP API of `depot web`, described in
OpenAPI 3.1 at `/api/openapi.json` (the server checks every request body
against the same description). The depot applies every rule, and a
refused change writes nothing.

| Operation | `depot` | `depot web` |
| --- | --- | --- |
| a new depot, a clone | `new ADDRESS`, `clone ADDRESS TO` | none |
| a pull | `pull` | none |
| the depot's layout brought up to the code's format | `upgrade` | none |
| the depot's maintainer and takedown contact | `about [--maintainer M] [--takedown-contact C]` | `GET /api/about` reads them; page "Depot" |
| a read query | `sql QUERY` | `POST /api/sql` |
| the depot at a glance | none | `GET /api/counts`, the depot's counts, and `GET /api/decisions/latest`; pages "Overview" and "Settings" |
| the depot's counts made again from its rows | `recount` | none |
| a decision, shown | `sql QUERY` | `GET /api/decisions/{id}`, with the texts it is on or the erasure requests it decides; the view of a decision |
| a tag, shown | `sql QUERY` | `POST /api/texts/search` and `POST /api/texts/counts` with the tag; the view of a tag |
| texts found by a filter, a page at a time | `sql QUERY` | `POST /api/texts/search`, with `GET /api/tags`; the list of the "Texts" page |
| the texts a filter matches, counted one slice of the depot at a time, or at once from the depot's counts when they hold the answer | `sql QUERY` | `POST /api/texts/held`, then `POST /api/texts/counts` for what the depot's counts do not give; the counts of the "Texts" page, its list and its review, of the "Overview" and of the views of a tag, a license and a decision |
| texts found by a filter, with their text, a page at a time | `sql QUERY` | `POST /api/texts/passages`, each page bounded by a count of texts and by characters; the blob of the "Texts" page |
| a text or an item, added | `add FILE [--kind KIND]` (`plain` by default) | `POST /api/texts`, with `/api/kinds` and `/api/licenses`; the "Add a text" button |
| a text or an item, shown | `show ID` | `GET /api/texts/{id}`; the content and review panel of the "Texts" page, with the text's record |
| a passage of a plain text | `show ID --from N --characters M` | `GET /api/texts/{id}/passage`; the reader of a text too long to edit on the "Texts" page |
| a license | `license add ID NAME FILE` | `/api/licenses`, `GET /api/licenses/{id}`; page "Licenses" with the view of each license, and the license picker of the "Add a text" page, of a license change and of the review of an admission for a missing one |
| a decision on texts | `decide admit\|refuse\|back-to-review WHERE` | `POST /api/decisions`, on a selection, and `POST /api/texts/blocked`, which texts of a selection a decision cannot cover and why; the "Texts" page, on ticked texts, on the current text, and by the keys A, R and U |
| the reason of an admission or a refusal, given afterwards by the person who made it | `reason DECISION REASON` | `POST /api/decisions/{id}/reason`; the reasons the "Texts" page offers after a refusal |
| a change of texts' licenses | `relicense ID ... --license L ...` | `POST /api/texts/licenses`, on a selection; the review of an admission, for texts with no allowed license, and the current text of the "Texts" page |
| a change of a text's content | `content ID FILE [--kind KIND]` (a kind other than the text's refused) | `POST /api/texts/{id}/content`; saving the content on the "Texts" page |
| a change of the plain texts of many texts | `content-file FILE` | `POST /api/texts/content` |
| a change of a text's credits | `credits ID [--title T] [--author A ...] [--tag T ...] [--address A] [--notice FILE] [--clear CREDIT ...]` | `POST /api/texts/{id}/credits`; the current text of the "Texts" page, whose tags its review panel toggles |
| a change of the credits of many texts | `credits-file FILE` | `POST /api/texts/credits` |
| a change of the scores of many texts | `scores FILE` | `POST /api/texts/scores`; the score rows of the review panel of the "Texts" page |
| one score set on, or removed from, texts | `score NAME (--value V \| --remove) WHERE` | `POST /api/texts/score`, on a selection; the "Texts" page, on ticked texts |
| the names of the scores | `score-names` | `GET /api/scores` |
| the values a score has across every current text | `score-values NAME` | `GET /api/scores/{name}/values`: how many have it, its smallest and largest, its `SCORE_VALUES` most frequent (`depot/depot.py`), each to three significant digits, read from the counts; the choices of the score rows of the review panel of the "Texts" page |
| a change of tags over a selection | `retag WHERE [--add T ...] [--remove T ...]`, its cost, then the change with `--yes` | `POST /api/texts/retag/estimate`, its cost (texts changed, rows and bytes written, seconds), then `POST /api/texts/retag` with the estimate it showed; the bulk bar of the "Texts" page |
| the times of changes | `sql QUERY` | `POST /api/changes/times` |
| a snapshot, named by the UTC time it is made | `snapshot new [--note TEXT]`, `snapshot list`, `snapshot show NAME [--owner DEPOT]` | `GET /api/snapshots/new` (the cost of making one now, and the snapshot job open), then `POST /api/snapshots` with the estimate it showed, a job, one at a time; `GET /api/snapshots`, `GET /api/snapshots/{id}` (its summary and the texts erased since), `GET /api/snapshots/{id}/sets` (its bounds and bitmaps, base64, and its digest), `POST /api/snapshots/{id}/texts` (a page of its texts at their revision then); page "Snapshots" |
| a dataset of a depot of format 12 or older, and a tokenizer recipe, for models trained before snapshots | none | the frozen routes of `depot_app/web/legacy.py`: `GET /api/datasets`, `GET /api/datasets/{id}`, `GET /api/datasets/{name}/remote`, `GET /api/datasets/{id}/download` and its corpora, files and `everything.tar`, `GET /api/datasets/{id}/export` and its parts, `GET /api/datasets/{dataset}/sample`, `GET /api/tokenizers/{name}` and its `vocabularies/{version}/file`; no page |
| the jobs | `jobs`, `jobs cancel ID` | `GET /api/jobs`, `GET /api/jobs/{id}`; page "Jobs" |
| a plain text read translated, never saved | none | `GET /api/translations`, the language pairs of Mozilla's translation models, and `GET /api/translations/files/{sha256}`, one model file, fetched from Mozilla the first time and kept on the dock; the Translate button of the "Texts" page, whose translation runs in the browser |
| erasures | `erase ID [START END]`, `erasures`, `decide admit\|refuse --erasure ID` | `/api/erasures`, `GET /api/erasures/{id}`, what one request asks to erase with the text around each passage, and `POST /api/erasures/decisions` on one or several requests; page "Erasures" asks, shows, lists and decides, and the current text of the "Texts" page asks |

Each row of a list on the pages opens a view of that thing, with its own
address in the page's hash so it can be linked: a text (`#/texts/ID`, the
Texts page with that text open), a snapshot (`#/snapshots/ID`), a license (`#/licenses/ID`), an erasure request (`#/erasures?request=ID`), a
decision (`#/decisions/ID`) and a tag (`#/tags/TAG`).

Every decision, an added text's, an erasure request's, a change of
content's, of credits' and of scores' included, carries
the decider's declaration: a legal name, an email, and the confirmation
that they read the requirements, accept them, and that what they submit
meets them. The commands take it as `--name`, `--email` and `--accept`.
The "Settings" page keeps the name and email in the browser and says that
both are public. Each decision is sent only by a click on its own confirming
button, or by a deciding key of the Texts page. Beside the button stand the
confirmation that `GET /api/about` gives, a link to the requirements, and
the name and email the decision records. A form without a name or an email
says so and links to the settings.

The "Texts" page is where texts are read, scored, tagged, edited and
decided one after the other. What is open is a filter, kept in the
address: quick filters and the filter bar set it, and a menu lists what
can be opened, every text, each status, each kind, the tags, the
licenses and the filters saved in the browser. The list of the open
texts, a page at a time, hides at the left edge and comes out while the
pointer rests there, or after a button on a phone.

The content of the current text fills the middle and is always editable.
An edit has no button: it is saved as a content change when the page
moves to another text, before a decision on the text (so Admit & next
admits the version just saved), with Ctrl+S, and when the text is left any
other way; the text goes back under review, an admitted one too, and a
save refused keeps the text open and says why. Words typed while a save
is on its way are sent before that save ends, and the editor keeps them.
Back to review on an edited text records its decision first, with its
reason, then saves the edit. An edit that cannot be saved when the text
is left by a way the page does not guard (the browser's back button) is
said in a line every page shows, and kept in the tab, so the text opens
again with it. A reload or a closed tab with an edit not saved asks
first.

The review panel at the right holds a row for each score name, its
choices read from `GET /api/scores/{name}/values`, the scale the depot's
texts hold (whole grades, 0 to 10 for a person's 0 to 10, each a button),
a number pad for a value between the grades (kept to 0 to 10 on a scale of whole grades) or, on
another scale, − and + and the value, as toggles the tags this browser put on a text last and then the most used,
a picker of tags and score names that reads the depot's again as words are typed in it, the moves and decisions,
presets of scores and tags kept in the browser, and the text's facts and
other actions. Clicks on scores and tags are gathered and saved as one
change of scores and one of credits a moment after the last, or when
another text opens. Admit & next and Refuse decide at once; after a
refusal, a line at the bottom names the text and its decision, offers
Undo, which puts it back under review at once and opens it again, and its reasons as chips (the usual ones,
then those kept in the browser, each added by a click from a reason typed
once) and a field for another: the one picked becomes that refusal's
public reason (`POST /api/decisions/{id}/reason`), and none is needed. A
return to review and the admission of a refused text ask first, in a line
at the bottom that holds the confirmation and the name the decision
records. A line names the text and the decision after each one.

Keys outside the editor do the same: J and K move, A and R admit or
refuse and open the next text, 1 to 9 pick a refusal's reason, Z undoes
it, U puts the
text back under review and X ticks it. Escape leaves the editor.
Without a name or an email, a decision asks for them and decides nothing.
The blob mode shows the open texts as one continuous text, read a page at
a time as it scrolls (a first page of 20 texts, then 50), each text's length in characters in the margin
with a Refuse button that decides at once and then offers Undo, three lines of each text or a longer start; a
click on a passage opens that text. "Biggest first" reads them the longest
first, to find what takes the most room. A snapshot reads whole the same
way, its longest texts first, each as the snapshot holds it, a click
opening the text as it is now in a pane beside the strip, to edit, admit
or refuse it: its order by length is made once by a job of the read lane
(`depot.sizes`), which the page follows until it ends; a job that
ended without the order is shown with its problem and a "Try again"
button, and no other is queued until it is pressed. The size map
shows which tags, kinds and licenses of the texts of one status, or of a
snapshot, take the most characters, from the counts or the snapshot's
summary, with the biggest texts beside them, each with a bar and a Refuse button; each part opens its texts as a
strip, the longest first.

The "Help" page, last in the top bar, says how the depot works and what
each page does.

A list of texts comes a page at a time with no count, since a count reads
every text it matches. A page starts after the last row of the page
before it, in its order, or ends before the first row of the page after
it, so a far page is read as fast as the first and the pages are read one
after another, never by number. A page can also start at a text's place in
its order, which the server finds, so the list opens on a text that a
link or the blob opened. A page reads its candidates from the narrowest
listing its filter names, chosen by the counts: the texts of a decision,
of one tag, or of one or several licenses or statuses, from
their indexes in the numeric order of ids, the windows of several values
merged; the texts the title index gives for every word of a search of 3
characters or more, each depot name's from the index in that order, with
the text whose id a word is; the ids named, sorted, when they are at most `SORTED`
(`depot/depot.py`); else every current text, from `texts_order`. It
tests each candidate against the rest of the filter. In the numeric order
of ids it reads at most `WINDOW` candidates after its place (`TEXT_WINDOW`
with a search in the texts themselves), in that order, and stops at the
page's last match, so a first page of every text reads about its own rows;
in the order of statuses it reads at most that many from `texts_status`,
and in the order of lengths at most that many from `texts_length`, the
range of each status merged, so a page of the longest texts of 22 million
admitted ones reads about its own rows; past `SORTED` texts in its listing
each candidate is tested on the whole filter. A listing of at most `SORTED`
texts in the order of lengths walks `texts_length` too when its first
window is expected to fill the page, its texts lying evenly among the
lengths: it reads the listing's keys once through the listing's own index,
then at most `KEPT_WINDOW` entries of `texts_length` (shared among the
statuses it reads), keeps those the listing holds and looks up only the
texts of the page, about 130 ms for a tag of 48,000 among 1.2 million
admitted texts with the file out of memory. Otherwise (a few texts among
many) it is sorted on the lengths alone, read through `texts_latest`,
before any text is read; that costs one lookup a text of the listing,
about 2 seconds for 20,000 texts with the file out of memory.
So a page is bounded whatever the
depot's size and however few texts match. A page cut at that bound is
partial: it can hold fewer rows than asked, or none, and the next page
reads on from its last candidate. The list and the blob say so; the
Texts page reads on by itself while no text of the filter is open yet,
until a text matches, the filter's texts end or its count finds none. No index serves the other orders (title, kind, tags,
licenses), which sort every
candidate, so the server gives them only when the listing holds at most
`SORTED` texts and refuses them past it; a page says whether its listing
allows them, and the list offers them only then. The counts of a filter
come apart, in steps: each step counts the matches in one slice of at most
a million texts of the page's listing, in its order, so each one stays well
under the time limit of a reader. A filter that names its texts is counted
in one step over those texts. A filter of statuses and of at most one tag
or license, or of statuses and kinds, is counted at once from the depot's
counts (`POST /api/texts/held`), with no step: how many texts it matches
and their split by status, and for a filter of statuses alone its tags and
licenses too, from the split of each tag and license by status. A page asks
that answer the moment its filter changes, beside any step running, and
names the parts it shows besides the statuses (by tag, by license, the
blocked texts): only those the depot's counts leave out are counted by
steps, so the Texts page, which shows the statuses, steps through texts
only for a filter the counts do not hold, or while its Tags or Licenses
menu counts within the open texts. The pages run one step at a time, the counts
open on a page taking turns, and add the steps up. The numbers grow as the
steps answer ("at least"), with how many texts are counted out of the
listing's texts, a number `counts` gives. A count
ends when its page leaves it or its filter changes, and it starts again
after a change of the depot made from the page. A step starts at least two
seconds after the one before it. A step refused by a proxy's rate limit
(429), or that found no free reader, is sent again after a wait that
doubles. A step over a slice that the time limit stopped is sent again over
a slice half as big. A step that answers within a quarter of the time limit
lets the next one go over twice as many texts, up to a million.

The web pages act on texts by a selection, never by a list of every id the
browser read: the filter the page shows, the change number its page was
read at, and the ids unticked by hand. "Tick all" reads nothing more. The
review window counts the selection and lists its texts a page at a time,
each ticked. The decision, the license change or the change of tags
matches the filter when it writes, leaves out the ids
unticked and every text changed after that change number, and its answer
says how many it left out. A text that matches the filter only after the
page was read is left out with them. A text's own page names that one text
in the filter. A filter reaches the server as fields, which
`depot.filters` turns into its SQL condition, and every figure a page
shows comes from a route of its own, so no page sends SQL but the page
where a person types it. The commands and the web reach the same SQL in `depot`.

The server runs its long work as jobs (`depot_app/jobs.py`): a write over
a selection, the making of a snapshot and the order by length of a
snapshot. The queue is a SQLite
file beside the depot, `<depot file>.jobs`, which nothing backs up. Jobs
run in three lanes, one at a time in each, each job in a process of its own,
which runs the same `depot` operation as one change and sends the name
of each stage it starts and its progress: the write lane runs the writes,
the heavy lane the snapshots, which hold the depot's write lock for one
short transaction at most, so a write never
waits behind them, and the read lane the orders by length of the snapshots
read whole, which write nothing in the depot. A write over at most `INLINE` texts runs in the request
instead, while no job of the write lane is queued or running: a selection
whose filter names at most `INLINE` ids, or a change of tags that changes
at most `INLINE` texts. Otherwise every such write is a job, since it waits for the running
write's lock and only a job shows that it waits. Either way the answer is
the job's state, and `GET /api/jobs/{id}` reads it again while it runs, so
the page shows the stage of a long write and its end. A job survives a
restart of the server: a queued job stays queued, and a job a restart cut
short runs again, at most `RESTARTS` times. At most one snapshot job is
open, and each snapshot has at most one order by length (409 otherwise).
Only `depot jobs cancel ID` cancels a
job. The queue's file also holds what the downloads need
(`depot_app/downloads.py`): the streams being sent, how many texts each legacy export counts, and each
export part's size, sha256 and conversations left out once it was sent
whole; and
how many texts of each snapshot an erasure reached since, by last erasure.

The review of an admission sets apart the texts the depot would refuse to
admit because none of their licenses is allowed. The depot's own check
gives that list, the one a decision applies, so the two cannot disagree.
Those texts are never part of the admission. From the list a person
refuses them or changes their licenses. After a refusal or a license change
made there, the review reads its summary again and checks its texts again,
so a text that now has an allowed license joins the admission, ticked.

The HTTP API answers 400 for a query that is not one read of the depot,
404 when the thing does not exist, 422 for a change the depot's rules
refuse (with their messages) and for a person's SQL stopped by the limits
below, and 503
when the dock is missing on this machine. A command prints the same
failure's kind and message and ends with exit code 2.

A person's SQL reaches only the depot's file. It runs on the read-only
connection whose authorizer lets a statement only read ("SQL" in
[The depot](#the-depot)), so no query writes, attaches another file, runs
a pragma (but reading `data_version`) or loads an extension, and SQLite has no function that reads
another file or an address on the network. A filter's condition is a query
as well, under the same rule. `depot sql` follows it too.

A route that runs a person's SQL, or that reads many texts (`POST /api/sql`, `POST /api/texts/search`, `POST
/api/texts/passages`, `POST /api/texts/counts` and `POST /api/texts/blocked`),
runs it in a
reader process of its own (`depot_app/web/readers.py`), never in the
server's process, with at most `READERS` reader processes at a time.
SQLite stops a person's statement `SECONDS` after its reader takes it, and the
server kills the reader once the request has run `SECONDS`, the wait for
a free reader included; both answer 422 `TimeLimitReached`. A request that
finds no free reader within `SECONDS` answers 422 `NoReaderFree`, its reads
never started. Up to `KEPT` readers stay alive between requests, and a
reader encodes its answer as JSON once, which the server sends as it is.
On Linux, the reader's memory is
capped at `MEMORY` bytes more than it holds once started, and a query past
it is refused with 422. On another system, the reader has no memory cap.
A query's answer holds at most `ANSWER_ROWS` rows and `ANSWER_BYTES` bytes
of JSON, counted as its rows are read, and a larger one is refused with
422. The server's process goes on after any of these, and the next request
gets a new reader. The commands have none of these limits.

The follow routes (`POST /api/follow/head`, `POST /api/follow/rows` and
`POST /api/follow/parts`) answer the clones and pulls of other depots
("Following an upstream" in [The depot](#the-depot)). They only read,
and each answer is one bounded page, streamed as it is read.

`depot web` serves the depot `DEPOT` names. It listens only on
127.0.0.1 and checks neither the `Host` header nor a request's
`Content-Type`: like the commands, it asks for no login and no header, and
it answers any client that reaches it, through a proxy in front of it
included. Every such client reads and changes the depot through every
route: it adds texts and licenses, makes snapshots, and sends
decisions with a declaration it made up. A web page open in the same
browser can do the same with a cross-origin `POST` with `Content-Type:
text/plain`, which the browser sends without asking first, and so can a
site that makes its own name resolve to 127.0.0.1 (DNS rebinding).

A pull (which makes the machine fetch another depot) and a change of the
depot's maintainer or takedown contact are commands only: the server has no route for them, and nothing is opened
at an address a client gives. `POST /api/erasures/decisions` carries out
an erasure in the depot alone: it marks nothing on the dock, and the frozen
dataset routes still send a corpus that holds the erased text until the
freight's next check marks it stale ([erasure.md](erasure.md), "What it
covers"). No route reads a code repository: the depot lives apart from
any, and builds no corpus.
A legacy dataset's export reads its texts from the depot, never from the
dock.

## Packages

| Package | Role |
| --- | --- |
| `depot` | the depot: texts and structured items, the decisions on them, snapshots and erasures, in the tables of one SQLite file (`depot.py`, the only code that changes them, `schema.py`); its connections and the write that records each change (`database.py`); the app's policy on decisions (`policy.py`); a filter of texts turned into an SQL condition (`filters.py`); backup and restore onto the dock (`backup.py`); the upgrade from an older format (`upgrade.py`, `delta.py`, and `conversion.py`, which makes snapshots of the libraries and datasets of format 12); erasures carried out (`erasure.py`); snapshots, made, read and checked against their digest (`snapshots.py`); the kinds of texts and their schemas (`kinds.py`, `kinds/`); the excerpts of a conversation and where they match (`quotes.py`); clone and pull (`follow.py`); what code outside the depot asks of it, a snapshot, its listing, its texts' content and credits and the erasures, in values JSON carries, the only module code outside the depot imports ([api/depot.md](api/depot.md), `api.py`); a snapshot's texts in the order of their export keys (`order.py`), a long text taken as pieces (`pieces.py`); the libraries and datasets of format 12, read-only, for models trained before snapshots, with a dataset's export in parts and its sample, each share split over its texts' tags (`legacy.py`, `groups.py`, `fill.py`); the table of counts (`counts.py`); `DEPOT` (`settings.py`) |
| `wpt-orders` | the model and series orders and the images of agentic tasks: the service layer that reads and checks them (`orders.py`), the sizes a series derives (`series.py`), and the audit hook that keeps `data/` to the service layer (`guard.py`) |
| `dock` | the dock (its kinds: a local folder or S3), its compression, its named files (deliveries, corpora, models, run folders) and its results; what code outside it asks of it, the gestures find, fetch and put, the only module code outside the dock imports ([api/dock.md](api/dock.md), `api.py`) |
| `freight` | reads a request, chooses items in a depot's snapshot by the selection rules ([data-selection.md](data-selection.md)), reading the depot only through `depot.api` (`selection.py`, `groups.py`, `fill.py`), runs one preparation on them, a program it talks to through its standard input and output (`exchange.py`, `messages.py`; [api/preparation.md](api/preparation.md)), and leaves the result on the dock with a delivery note (`job.py`, `deliveries.py`); what the depot's erasures did to the deliveries (`erasures.py`); its queue, one job at a time (`jobs.py`); what code outside it asks of it ([api/freight.md](api/freight.md), `api.py`) |
| `wpt-corpus` | WPT's preparations, the programs `wpt-vocabulary` (a vocabulary learned from plain texts, `vocabulary.py`), `wpt-tokens` (a corpus's token files and `tasks.jsonl`) and `wpt-count` (the same weights, no file) (`programs.py`; [api/preparation.md](api/preparation.md)); how an item becomes tokens (`encoding.py`); the token shard format with its checked reader, so training and evaluation read a corpus without depending on the depot (`shards.py`); the reading of the corpora the depot built before the freight, under `corpora/` of the dock (`corpora.py`); `DockFiles`, the reader of a corpus's files on the dock that `wpt train`, the shards and the environments take, which fetches each file it is asked for as a file into a folder of the caller's (`files.py`) |
| `wpt-telemetry` | the record of one command run on the dock, sent live to an OpenTelemetry backend when one is set ([observability.md](observability.md)) |
| `wpt-tokenizer` | the byte-level BPE tokenizer, Rust inside |
| `wpt-model` | the WPT network ([model.md](model.md)) and its drafter, its safetensors model folders, their license documents (`LICENSE`, `NOTICE`, model card), and what a folder's `provenance.json` says of its corpus in either of its shapes (`provenance.py`) |
| `wpt-training` | the training loop (global batch, evaluation, checkpoints, resuming) that runs one training step, with its plug points: strategies (how the hardware is used), and the parts of a step ([training.md](training.md), "Steps"): methods (how the model learns from an example), environments (where the examples come from), optimizers (how an update changes the weights) and the steps that name one of each, with the model parts they lock |
| `wpt-inference` | sampling through the model's generation cache: several rows from one prompt, the model's next chat message (`next_message`, which `wpt chat --chat` writes with), speculative decoding with a drafter and its scheduler, the scheduler's pass times (measured, recorded, replayed); training a model's drafter (`drafting.py`), which `wpt train` runs at the end of every run, so it uses `wpt-model` |
| `wpt-utils` | the helpers only WPT uses: `wpt_utils.device` picks the best GPU, else the CPU (`WPT_CPU=1` forces the CPU); `wpt_utils.settings.typed` types the settings of a training step and of its method, environment and optimizer from the defaults each file declares |
| `wic-utils` | the helpers every part may import, the depot, the freight, the dock and WPT alike, one submodule each: the reading of `.env` files (`wic_utils.env`); `wic_utils.programs` starts the outside programs WPT runs (`program(name)`, with a clear error when one is missing), identifies the code (`code()`: the checkout's commit and a hash of its uncommitted changes, every changed or untracked file read by git and never by this process, so the audit hook of the service layer's guard has nothing to refuse under `data/`; the one place that reads the commit, which records and a training step's fingerprint hold; `code_name()`: the same as one line, which a stored result keeps as what made it) and gives that commit, and whether there are changes, in the shape the records name them (`repository()`), without importing PyTorch; `wic_utils.files.sha256_file` is the one way a file is hashed, by the dock, the model folders and the run records alike; `wic_utils.files.lock_file` is the lock the operating system holds on a file for one process, which a local dock's replace of a file uses; `wic_utils.parallel` is the one rule that sizes parallel work (at most 75% of the cores; before each new item it measures again and starts it only while a quarter of the memory, of the cores and, where the items write to a disk, of the free disk stays free, one item's memory being what running items are measured taking, never a given figure; it never stops a running item) and runs items that way with their results in the items' order, an error an item may fail with becoming that item's result; `wic_utils.lines` writes every printed line, with its time and level |
| `wpt-agent` | the agent harness of agentic training: a task read from a depot item (`tasks.py`) and the Git repository it works in, fetched when it runs (`repository.py`), the tools (the four built-in ones, `ask`, and those a task or its image's pool declares, answered by programs in the image) and each attempt's choice of them, one Docker container per attempt, the images of `data/images/` and their locks, the agent loop and the task's check, and the playing of a task's attempts (`wpt_agent.play`), which the `agent-tasks` environment of the `finish-tasks` and `finish-tasks-with-less-writing` steps uses; an image's lock and build (`images.py`) |
| `wpt-worker` | the gpu worker, which runs the GPU work of a training on its machine, the link to it from the wpt manager's side, and the gradients exchanged between gpu workers in different places ([machines.md](machines.md)) |
| `wpt` (`apps/wpt`) | the app: the `wpt` command, a controller over the services above ([The app](#the-app)) |
| `depot-app` (`apps/depot`) | the `depot` command and the depot's web app, the depot's own controllers, which may import all of `depot` |
| `freight-app` (`apps/freight`) | the `freight` command, a controller over `freight.api` ([api/freight.md](api/freight.md)) |
| `dock-app` (`apps/dock`) | the `dock` command, `find`, `fetch` and `put`, a controller over `dock.api` ([api/dock.md](api/dock.md)) |

The packages hold services and no command, with one exception: the
programs a preparation runs as, `wpt-vocabulary`, `wpt-tokens` and `wpt-count`, are console
scripts of `wpt-corpus`, which the freight starts. Those that run without
the repository, `wpt-worker` (on a gpu worker's machine), `wpt-model` and
`wpt-inference` (a model folder and the text it writes), install alone.

## The app

Four projects of `apps/` hold the four commands, each a controller over
its part (the workspace's root project, `wpt-workspace`, holds none):

- `wpt` (`apps/wpt`), the command `wpt`: the subcommands `wpt list`,
  `wpt copy-out`, `wpt runs`, `wpt run`, `wpt agent`,
  `wpt build-resources`, `wpt adopt-vocabularies`, `wpt train`, `wpt eval`,
  `wpt chat`, `wpt manager` and `wpt doctor`, each with its own options
  (`wpt SUBCOMMAND --help`). The module `wpt`
  (`apps/wpt/src/wpt/__init__.py`) names the module whose `main` runs each
  one and imports it only when it runs, so a light subcommand never loads
  PyTorch; `python -m wpt` is the same command. What WPT reads of the dock
  (the corpora and models it holds, a model copied out, the runs) is
  `wpt/docked.py` and `wpt/models.py`, the agent harness `wpt/agent.py`;
  WPT's side of the freight, the requests an order makes and the
  deliveries training reads, is `wpt/resources.py` (with `wpt/adopt.py`,
  the depot's vocabularies put on the dock); the jobs are the subpackages
  `wpt.train`, `wpt.eval`, `wpt.chat` and `wpt.manager`.
- `depot-app` (`apps/depot`, module `depot_app`), the command `depot` and
  the depot's web server, `depot web` (`depot_app/web/`: `server.py`,
  `routes.py`, `queue.py` for the routes of the jobs, `snapshots.py` for
  the snapshots, `legacy.py` for the frozen dataset and tokenizer routes,
  `translations.py` for the translation models of the Texts page, and
  `ui.py`, which builds the interface of `apps/depot/ui` with Bun). The
  root `package.json` makes one Bun workspace of the interface and the
  packages of the organization's design system, the git submodule
  `vendor/wic-design-system`, whose tokens and components the interface is
  built on; `git submodule update --init` fetches it, and `bun install`
  builds its packages. What its routes and the subcommands share is
  `depot_app/shared.py`. It listens on 127.0.0.1 only; another machine
  reaches it through an SSH tunnel, Tailscale, or a proxy in front of it
  ([Two interfaces](#two-interfaces); [depot-web.md](depot-web.md)).
- `freight-app` (`apps/freight`), the command `freight`, a controller
  over `freight.api` ([api/freight.md](api/freight.md)).
- `dock-app` (`apps/dock`), the command `dock`, a controller over
  `dock.api` ([api/dock.md](api/dock.md)).

Each app's tests live in its `tests/` folder; those of `wpt` in
`apps/wpt/tests/`, one folder per controller or job, sharing their setup
through `apps/wpt/tests/conftest.py`.

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
- `data`: the orders' check (`wpt_orders.Orders.check`): every order
  and image file matches its schema, and the rules across files hold (a
  series derives every size, an image's pool takes no built-in tool's name, a
  order names its depot and its snapshot, or its dataset in the legacy
  shape). It runs when `data/` changed, or
  `wpt-orders`, the root `pyproject.toml` or `uv.lock`.
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
with such a test broken. `packages/wpt-orders/tests/test_data_access.py` reads
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
`packages/wpt-orders/src/wpt_orders/orders.py` (its
`Orders.image_files` decides which files an image's tag covers), the
schemas `image-lock.schema.json` and `image-licenses.schema.json` in
`packages/wpt-orders/src/wpt_orders/schemas/`, or anything under
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
`wic_utils.parallel.limit`, the rule's form for a count fixed in advance. The
part first runs a sample of the chosen tests (every 40th, in collection order)
alone, under a worker's conditions, and measures the memory that run held on
average (`wic_utils.parallel.average`: the mean of the readings taken while it
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
