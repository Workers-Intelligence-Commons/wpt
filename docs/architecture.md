# Architecture

**The texts live in a depot; the repository holds none.** The repository
holds the code, the model and series recipes, and the images of agentic
tasks. The texts, the decisions on them, what to train on and the
tokenizer recipes live in depots, and everything built or trained lives in
a store.

- A depot is one SQLite file, known by its address, of tables: the texts and the
  structured items (conversations, tasks), each with its license, credits
  and notice; the people's decisions on them; the libraries that group
  admitted texts; the erasures; the datasets, each naming libraries and
  the share of each tag; and the tokenizer recipes, each with the
  vocabulary of each of its versions. A depot can follow an upstream depot
  and pull its updates (`wpt_depot`).
- A corpus is what a dataset is built into, in the store: the token files,
  the conversations and tasks, and copies of every text's credits,
  licenses and notices, so it says everything about what it holds even once
  the depots it read change or vanish (`wpt_depot.corpus`).
- A model recipe names a depot by its public address and a dataset in it; training reads
  the latest fresh corpus of that dataset, the vocabulary the dataset publishes and
  the recipe's `total_tokens` from the store, and every size of a model series reads
  the same corpus, the one of its largest size, the smaller sizes its first part;
  `total_tokens` is the size the corpus is built at and a ceiling on what training
  reads, and training reads what a corpus built at that size holds.

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

A depot is one SQLite file, given as a path or a `file://` address and
known by that address. The setting `WPT_DEPOT`, in the environment or in
`.env`, names the depot that `wpt depot` and `wpt depot web` work on, as
`WPT_STORE` names the store. `wpt depot new ADDRESS` makes an empty one,
with its name, its maintainer and its takedown contact. The setting
`WPT_DEPOT_URL`, read the same way, is the depot's public address, the web
address of its `wpt depot web`: its corpora name the depot by it, never by
its file, and a depot without it is named by its name. A corpus names
another local depot it reads by that depot's name too
([data-selection.md](data-selection.md), "Corpora").

Its tables are `about` (the depot's name, maintainer, takedown contact
and upstream), `changes` (every change: its number, its kind and its
time), `numbers` (the last number of each kind of the depot's own ids),
`licenses`, `notices`, `kinds`, `decisions`, `texts`, `content_lists`
(every content version of each text, as the list of its parts), `parts`
(the parts of each text's content versions), `libraries`, `memberships`
(when each text joined and left each library), `pins` (the revision of
each text a locked library reads), `erasures`, `datasets` and `pulls`
(for a depot that follows another, how its change numbers meet its
upstream's). Every table declares a primary key. What each holds and the
rules on it are in [data-selection.md](data-selection.md) (texts, items,
tags, libraries, datasets and corpora), [licenses.md](licenses.md)
(licenses, notices and decisions) and [erasure.md](erasure.md)
(erasures).

**Ids.** Every id (of a text, a decision, a library, an erasure, a notice
or a dataset) is the depot's name, a hyphen and a number: `my-depot-17`. The
depot's name is made of letters, digits, `.`, `_` and `-`, and begins with a letter or a digit. A reference to an
id in another table, in a dataset, in a corpus or in a vocabulary's list of texts holds
the whole id, so an id names one thing across depots. The number is one
past the last number of that kind of id the depot gave (`numbers`), an
erased text's included, so a number is never given twice. A clone takes a
name of its own, since it never reuses its upstream's.

**Writing.** The file is in WAL mode. Several processes may open it: each
reads at any time, and SQLite's lock lets one write at a time, the others
waiting for it. `wpt_depot.depot` is the only code that changes a depot.
Each operation is one change: one transaction, under the next change
number, that checks the whole change against the depot's rules inside it,
so a refused or failed change writes nothing and a crash never leaves a
change half written (`wpt_depot/database.py`).

A session of SQLite records every row a change writes or deletes. Before
the transaction commits, that record goes to a change file in the waiting
folder beside the depot's file (`<file>.waiting`), named by the change
number, the number of the change before it and its time. The file is
synced and kept only once the change is committed, so the waiting folder
holds exactly the depot's committed changes since its last backup. A
deleted row's bytes are overwritten in the file, so an erasure leaves no
copy of what it removed in the file's free space.

**Revisions and change numbers.** Every change of a depot has a change
number, one past the last. `about`, `texts`, `libraries`, `erasures` and
`datasets` keep each change of a row as a new row, its revision, keyed by
the id and the revision number and recording its change number; the
highest revision is the current one. A text's content versions are
rows of `content_lists`, keyed by the text's id and the version number and
recording their change number, and each revision of the text names the
content version it reads. A revision of a text also holds its scores,
each with the content version it was made from. Memberships record the change
numbers a text joined and left a library at. A decision on a text is a
row of `text_decisions`, with the change number it was made at, so a
decision on a million texts writes a million small rows and no revision. A
depot can so be read as it was at any change number, which a dataset names
for each library it reads.

**Status and counts.** The current revision of each text stores the status
its decisions make, and the index `texts_status` holds the current texts by
status, then in the numeric order of their ids. The table `counts` holds
how many current texts the depot has, and how many have each status, each
tag, each license and each score name, each library holds and each decision
is on, and how many texts each locked library pinned at a revision older
than their current one. It splits every text, and the texts of each tag,
each license and each library, by status and by whether a locked library
holds them. The tables `text_tags` and `text_licenses` list
the current texts of each tag and of each license, in the numeric order of
their ids.
The title index `text_titles` is an SQLite FTS5 table of the trigrams of
each current text's title, which stores no title. Its rows are numbered by
the texts' ids, through the slot `title_owners` gives each depot name the
ids carry, so one name's texts come in the numeric order of their ids. A
text whose id is not a name, a hyphen and a number without leading zeros
cannot be numbered, and a write that would index one is refused. `counts`
also
holds what each library holds, each text at its current revision, in
texts, characters and UTF-8 bytes: by kind (plain texts, the conversations
and tasks a corpus takes, and items of other kinds) and, for its plain
texts, by tag, a text with several tags counting once for each. Every
write keeps the stored statuses, the counts, the two listings and the title
index right in its own transaction, so a page reads them at once whatever the depot's
size; one module writes them (`wpt_depot/counts.py`), and `wpt depot
recount` makes them again from the rows. The title index is written through
`wpt_depot/titles.py`, which that module calls.

**Reads that do not grow with the depot.** A page, a count step and a write
of a few texts read only the rows they name, through an index: the
listings above, the index `memberships_order` of each library's current
texts in the numeric order of their ids, `pins_order` of each locked
library's pins in that order, `text_decisions_order` of each decision's
texts in that order, and `memberships_held` of each text's open
memberships. The words of a search are read from the title index when one
has 3 characters or more and the search does not read the texts
themselves. A page reads a bounded window of candidates, but a filter no
listing narrows reads up to every text over its pages: a search whose
words are all shorter, a search in the texts, texts outside some libraries, texts no locked library holds, and
two parts of a filter whose texts meet in few. A pull reads each table's
rows past a change number through an index of their change numbers, the
rows of every depot numbered in that range. A query names the index or the join
order it needs where SQLite would otherwise choose to read a whole table.
`packages/wpt-depot/tests/test_reads.py` runs each such route on two depots,
one twice the size of the other, and fails when its reads grow.

**A dataset revision's figures.** A dataset reads each library at a change
number, so what a build of a revision takes is what the libraries held then,
not what they hold now. The same module keeps, in `library_moves`, the
change at which what each library holds last moved. A library that has not
moved since the change a revision reads it at holds now what it held then,
so its counts are exact for that change. The write that makes a dataset
revision copies those counts into the revision, in its own transaction, for
each library they prove exact, and stores nothing for the others: a library
of another depot, an upstream's library in a clone, whose change numbers
are of another numbering, and a library that moved since the change an edit
keeps reading it at. The dataset's pages show the stored figures, with how
many more or fewer texts the libraries hold now, and "unknown" where nothing
is stored. An edit keeps each library at the change it was first read at.
Reading a dataset again (`wpt depot dataset read-again`) is the one write
that moves its libraries: a new revision, the same otherwise, that reads
every library of this depot at the change number now, a locked one as its
lock leaves it, and stores their figures; a library of another depot keeps
its change. It reads the dataset's current row inside its own write. It writes
nothing when no library's change number would change, or when the depot's
latest change is the dataset's own read again, since nothing moved after it. The
dataset's page offers it only where it changes what the page shows. An erasure carried out after the figures were read reaches the past
a revision reads, so a revision's figures are unknown once one reached a
text its library holds or held, or removed a text whole, which leaves no
row naming its libraries. In a clone, an upstream's erasure counts from the
pull that brought it. The upgrade to format 9 gives figures to a
revision only when nothing touched its libraries after the change it reads
them at, so a stored figure is never wrong.

**A dataset's order.** A corpus build, its estimate and the sample that
trains a vocabulary read a dataset's texts in the order of their export
keys, the order its export reads too (`wpt_depot/order.py`). Each library
is read as its own rows by key, from the index `memberships_key`, or
`pins_key` for a library locked at the change the dataset reads it at, a
page at a time, each row's revision looked up as it is read, and the
libraries' reads are merged by key. A plain text longer than 1,048,576
characters comes as pieces (`wpt_depot/pieces.py`), each at the key of its
own name, so a long text spreads over the order. The walk finds those
texts before it starts, from the index `content_lists_long` of long content
versions, each looked up in the library, so finding them grows with the
depot's long contents, not with the library's texts. Only the texts the walk reaches are
read, so a build at a small size and an estimate start at once whatever
the dataset's size. A dataset with shares is the exception for builds and
vocabulary samples: each share splits its part over the tags of its texts
([data-selection.md](data-selection.md), "Filling a part"), so a first walk
reads every row of the dataset, keeping numpy arrays and no object per
text (`wpt_depot/groups.py`), before the stream takes the texts it chose. What a build needs of the whole dataset (the bytes of
each share and of the conversations and tasks, and how many texts it holds
of each kind) comes from the figures its revision stores. When they are not
known, the build and the estimate count the same figures from every text
of the dataset's libraries instead.

**Contents.** A content version, a plain text or an item's JSON, is the
ordered list of its parts, and its row of `content_lists` holds that list,
each part's SHA-256 and length, with the version's length in characters
and in UTF-8 bytes. The same rule cuts every text, whatever its size, so
a text of many gigabytes, such as a repository with all its files and its
history, is one text (`wpt_depot/parts.py`). A text shorter than a part's
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
pull read and write a text's parts a few at a time, and the corpus build
reads the parts of all the texts it takes in one pass over `parts`. The corpus build
reads a piece of a long text from the parts its characters lie in alone,
and holds the tokens of one text or piece whole in memory while it encodes it, and
`wpt depot show` and the web pages give a text whole, so they hold it in
memory.

**SQL.** Every query is SQLite's SQL. A list column (a text's licenses,
authors, tags, decisions and scores, a library's problems, a dataset's
libraries and shares, and the others `wpt_depot/schema.py` names) is JSON
text, so a tag test is `EXISTS (SELECT 1 FROM json_each(tags) WHERE value
= 'book')`. A read query sees each revised table's current state under its
plain name (`texts`, `libraries`, `datasets`, `about`, `erasures`) and
every one of its rows under `<name>_revisions`. The `texts` view joins
each text with the content version its current revision reads, its length
and the libraries that hold it now. The `contents` view gives each content
version's whole text, its parts joined in order, and `content_lists` its
rows as they are. The other tables, `parts`, `memberships`, `pins`,
`text_decisions`, `counts` and `pulls` included, are seen as they are.
The `texts` view gives each text's decisions as a list read from
`text_decisions`, in order. The column `decisions` of `texts_revisions` is
unused: every revision the current format writes holds `[]` there, and a
past revision brought up from an older format keeps the list it held, so a query over
past revisions reads the decisions from `text_decisions` and its change
numbers. The column stays because removing it rewrites the whole table.
A query that names no text column
reads the views without joining any part, so a query over the texts'
other columns costs the same whatever their size. A query that reads the
text of a content version bigger than one SQLite value holds fails with an
error that names the version, and such a text is read from its parts, as
`wpt depot show` reads it.

A person's SQL (`wpt depot sql`, the condition of a filter) runs on a
connection that opens the file read-only and whose authorizer lets a
statement only read rows and call functions: a statement that writes,
attaches another file, runs a pragma or loads an extension is refused, and
a query is one statement. The one pragma it may run is reading
`data_version`, which the title index reads to know whether the file
changed.

**Format.** The depot's layout, its tables and their columns, has a
number: its format. The code holds the format it reads and writes
(`FORMAT` in `wpt_depot/schema.py`). Each `about` row records the depot's
format when it was written, so the current row gives the depot's format.
A depot of format 3 or older is a folder of Delta Lake tables, which only
`wpt depot upgrade` reads (`wpt_depot/legacy.py`).

Every way into a depot that reads or changes its rows refuses one of
another format than the code's, older or newer: every `wpt depot` command
but upgrade, `wpt depot web`, the reading of another depot's library, and
a clone or a pull, which need both the upstream and the clone at the
code's format. The refusal names the depot's address, its format and the
code's, and says to run `wpt depot upgrade` for an older depot or to
update wpt for a newer one.

`wpt depot upgrade` runs only when asked (`wpt_depot/upgrade.py`). On a
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
way into a half-upgraded depot but `wpt depot upgrade` refuses it, and the
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

Every change to the depot's layout raises `FORMAT` and adds the upgrade
step from the format before, with its test. A step names the columns it
reads and writes itself, never through the current layout, which is a
later format's, and run again on the layout it leaves it changes nothing
more.

**Following an upstream.** A depot can follow another, its upstream, the
way a Git clone follows its remote. The upstream is named by the web
address of its web app (`wpt depot web`), and a clone reads nothing but
the answers of its follow routes, which only read. `wpt depot clone URL TO
--name NAME --maintainer M --takedown-contact C` copies every row of every
table but `changes`, `numbers` and `about`, revisions, content lists, memberships
and pins included, with their change numbers, into a new depot whose
`about` row names the upstream. `wpt depot pull` fetches the upstream's
updates: each upstream row replaces the row of the same key, and a row
only the depot holds stays. What travels is paged, so no answer holds more
than a bounded number of rows and bytes and fits the size and time limits
of a web server in front of the upstream. A part never changes once
stored, so a clone and a pull copy only the parts the depot lacks that the
content lists they bring list, a few at a time, and a pull drops the
parts that the versions an erasure it brings cut no longer list. Its
change number is past the upstream's, so the depot's later changes come
after every change it received. The texts the upstream's admitted
erasures removed are removed too, and an erasure the depot had not
carried out is carried out there (`wpt_depot/follow.py`). A pull runs
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
A dataset that names a library of another depot names it at that depot's
change number. A read of the clone's own library at a past change of the
clone sees the upstream's rows (their text revisions) only up to the
upstream change the last pull at or before it brought, and the clone's
own rows by its own numbers, so the library reads exactly as it stood
then. A library of the upstream is read by the upstream's numbers, so a
dataset naming it at an upstream change reads the same in the clone as in
the upstream, and a dataset of the clone that names it by the clone's
address names it at the upstream change its pulls brought. A clone copies
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
the upstream's libraries in the clone once a pull has brought it, whether
it names it with no address or by the upstream's web address, so the head
of the library on the Texts page lists both. A library of the upstream that
no pull brought yet is asked of the upstream's web app when a dataset
names it. The pages show quietly
that a text, library, dataset or erasure request belongs to the upstream,
and offer no action on it that the depot would refuse.

**Backups.** The backups of a depot are in the store, under
`backups/<the depot's name>/` (`wpt_depot/backup.py`). `wpt depot backup`
sends each change file waiting beside the depot into the store and deletes
it from the waiting folder once the store holds it. Its first run on a
file a command made whole (a new depot, a clone, an upgrade, a restore),
and a run with `--full`, also sends a full copy of the file, made with
SQLite's backup into a temporary file beside the depot's file while the
depot goes on being written. Each run records what it sent. Such a file
starts a timeline of its own, a random id its backups record, so the
changes of a depot restored to an earlier moment, which number on from
there again, never mix with the changes of the same numbers its origin
made.

`wpt depot backup --off` marks a depot that nothing backs up. Its writes
then record no change file, so the waiting folder does not grow, and its
waiting change files are deleted. Its next backup records again and sends
a full copy.

The app never cuts, erases or removes a backup. An erasure does not reach
them ([erasure.md](erasure.md)). Backups on a schedule come from a timer
outside the app, such as this cron line:

    0 3 * * * cd /path/to/wpt && uv run wpt depot backup

`wpt depot restore TO --at MOMENT` makes a new depot file at `TO` from the
backups of one timeline: the latest by default; for a change number, the
latest that reaches it; for a UTC time, the one that made the latest change
or full copy at or before it. It takes that timeline's last full copy at or
before the moment, then applies every change file after it in order, up to
the moment. A change that does not apply exactly stops the restore, which
leaves nothing at `TO`; it never writes over a file. It reads the backups of the
only depot name the store holds backups of, or of the name `--depot` gives. Erasures
carried out after the moment are not carried out in the restored depot.

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

The store keeps the latest fresh corpus of each dataset, vocabulary and
size, and the record of every corpus (`wpt_depot/corpora.py`). A build
deletes the corpus before it of the same dataset, vocabulary and size once
its own is whole, and, under the cap
`WPT_CORPORA_CAP`, deletes the files of folders without a record, then
those of the corpus downloaded least recently, until its own fit, never
those of a corpus being built or downloaded. A deleted corpus's record stays, marked deleted with the
reason, and building it again checks that it gives the same id.

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
and bytes, never a path under `data/`. It holds the model and series
recipes and the images of agentic tasks:
their Dockerfiles, locks, license lists and pools of tools (`tools.json`).
`wpt train`, `wpt manager`, `wpt eval`, `wpt doctor`, the
image commands of `wpt agent` and the task reader of `wpt_agent` read them
through it. It checks every file it reads against its schema, and its
check adds the rules across files: a series derives every size, and an
image's pool takes no built-in tool's name. A recipe whose depot or dataset is still empty
reads, and the check names it for a person to fill in. People edit the
recipes and `tools.json` as files; `Recipes` writes only the generated
files (an image's `lock.json` and `licenses.json`), for the command that makes them, after their schema,
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
| a pull | `pull` | none |
| the depot's layout brought up to the code's format | `upgrade` | none |
| the depot's maintainer and takedown contact | `about [--maintainer M] [--takedown-contact C]` | `GET /api/about` reads them; page "Depot" |
| a read query | `sql QUERY` | `POST /api/sql` |
| the depot at a glance | none | `GET /api/counts`, the depot's counts, and `GET /api/decisions/latest`; pages "Overview" and "Settings" |
| the depot's counts made again from its rows | `recount` | none |
| a decision, shown | `sql QUERY` | `GET /api/decisions/{id}`, with the texts it is on, the erasure requests it decides or the problem it records; the view of a decision, and of a reported problem |
| a tag, shown | `sql QUERY` | `POST /api/texts/search` and `POST /api/texts/counts` with the tag; the view of a tag |
| texts found by a filter, a page at a time | `sql QUERY` | `POST /api/texts/search`, with `GET /api/tags`; the list of the "Texts" page |
| the texts a filter matches, counted one slice of the depot at a time, or at once from the depot's counts when they hold the answer | `sql QUERY` | `POST /api/texts/counts`; the counts of the "Texts" page, its list and its review, of the "Overview" and of the views of a tag, a license, a decision and a problem |
| texts found by a filter, with their text, a page at a time | `sql QUERY` | `POST /api/texts/passages`, each page bounded by a count of texts and by characters; the blob of the "Texts" page |
| a text or an item, added | `add FILE [--kind KIND]` | `POST /api/texts`, with `/api/kinds` and `/api/licenses`; the "Add a text" button |
| a text or an item, shown | `show ID` | `GET /api/texts/{id}`; the content and review panel of the "Texts" page, with the text's record |
| a license | `license add ID NAME FILE` | `/api/licenses`, `GET /api/licenses/{id}`; page "Licenses" with the view of each license, and the license picker of the "Add a text" page, of a license change and of the review of an admission for a missing one |
| a decision on texts | `decide admit\|refuse\|back-to-review WHERE` | `POST /api/decisions`, on a selection, and `POST /api/texts/blocked`, which texts of a selection a decision cannot cover and why; the "Texts" page, on ticked texts, on the current text, and by the keys A, R and U |
| a change of texts' licenses | `relicense ID ... --license L ...` | `POST /api/texts/licenses`, on a selection; the review of an admission, for texts with no allowed license, and the current text of the "Texts" page |
| a change of a text's content | `content ID FILE` | `POST /api/texts/{id}/content`; saving the content on the "Texts" page |
| a change of the plain texts of many texts | `content-file FILE` | `POST /api/texts/content` |
| a change of a text's credits | `credits ID [--title T] [--author A ...] [--tag T ...] [--address A] [--notice FILE] [--clear CREDIT ...]` | `POST /api/texts/{id}/credits`; the current text of the "Texts" page, whose tags its review panel toggles |
| a change of the credits of many texts | `credits-file FILE` | `POST /api/texts/credits` |
| a change of the scores of many texts | `scores FILE` | `POST /api/texts/scores`; the score rows of the review panel of the "Texts" page |
| one score set on, or removed from, texts | `score NAME (--value V \| --remove) WHERE` | `POST /api/texts/score`, on a selection; the "Texts" page, on ticked texts |
| the names of the scores | `score-names` | `GET /api/scores` |
| the values a score has on the texts changed last | `score-values NAME` | `GET /api/scores/{name}/values`: how many have it, its smallest and largest, its most frequent, over at most `SCORE_READ` revisions (`wpt_depot/depot.py`); the choices of the score rows of the review panel of the "Texts" page |
| libraries | `library new`, `put`, `take-out`, `lock`, `fork [--without WHERE]`, `report`, `rename` | `/api/libraries[/{id}/texts\|taken-out\|pins\|lock\|forks\|problems\|name]`, the put, take-out, fork and report on a selection; page "Libraries" lists them and makes one, the "Texts" page lists them too and opens each, with its head (a locked one's with its pinned revisions), and the view of each reported problem |
| a library's size | `sql QUERY` | `GET /api/libraries/{id}/size`; the head of an open library on the "Texts" page |
| a library read whole, as one strip of text | none | `GET /api/libraries/{id}/strip`, its size and whether its map is made, and `POST /api/libraries/{id}/strip`, its characters after or before a place, or at a position; page "Read whole", under the head of the library with its actions, linked as "read whole" wherever a library is named |
| what libraries hold, and the mix of a dataset's shares over them | `sql QUERY` | `POST /api/libraries/measure`: texts, characters and bytes, by share and by kind, read from the counts, and the part of each share the mix keeps; the dataset form, the list of datasets, the page "Libraries" and the head of a library |
| the times of changes | `sql QUERY` | `POST /api/changes/times`; the moment each library of a dataset is read at |
| datasets and remotes | `dataset new`, `edit`, `read-again`, `export`, `import` | `/api/datasets`, `GET /api/datasets/{id}` (with the corpora the store holds built from it and the figures its revision stores), `POST /api/datasets/{id}` (its edit, refused when made from a revision older than the current one), `POST /api/datasets/{id}/read-again` (its libraries of this depot read at the change number now, refused the same way), `GET /api/datasets/{name}/remote`; page "Datasets" with the view of each dataset; through the server a dataset reads only this depot's libraries, and a remote's import is a command only |
| tokenizer recipes | `tokenizer new`, `edit`, `import` | `/api/tokenizers`, `/api/tokenizers/{name}`; page "Tokenizers" with the page of each recipe; a recipe file's import is a command only |
| a tokenizer's vocabulary | `tokenizer train NAME` | `GET /api/tokenizers/{name}/estimate`, then `POST /api/tokenizers/{name}/train`, confirmed, a job; `GET /api/vocabularies`; `GET /api/tokenizers/{name}/vocabularies/{version}`, its `file` and the `texts` it trained on |
| a corpus | `build DATASET [--vocabulary SHA256] (--max-tokens N \| --all)`, run in the command | `GET /api/datasets/{id}/estimate`, then `POST /api/datasets/{id}/build` with a trained vocabulary and a size, confirmed, a job; the dataset's page "Corpora" |
| a corpus downloaded | none | `GET /api/datasets/{id}/download` (the dataset's corpora), `GET /api/datasets/{id}/download/{vocabulary}/{size}`, its `files/{name}` and its `everything.tar`; the dataset's page "Corpora" and the page of each corpus |
| a dataset's texts exported | none | `GET /api/datasets/{id}/export` and its parts `{name}`; the dataset's page "Export" |
| a dataset's sample, for another depot's vocabulary | none | `GET /api/datasets/{dataset}/sample?bytes=N` |
| the jobs | `jobs`, `jobs cancel ID` | `GET /api/jobs`, `GET /api/jobs/{id}`; page "Jobs" |
| erasures | `erase ID [START END]`, `erasures`, `decide admit\|refuse --erasure ID` | `/api/erasures`, `GET /api/erasures/{id}`, what one request asks to erase with the text around each passage, and `POST /api/erasures/decisions` on one or several requests; page "Erasures" asks, shows, lists and decides, and the current text of the "Texts" page asks |

Each row of a list on the pages opens a view of that thing, with its own
address in the page's hash so it can be linked: a text (`#/texts/ID`, the
Texts page with that text open), a library (`#/libraries/ID`, the
Texts page with that library open), a reported problem
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
button, or by a deciding key of the Texts page. Beside the button stand the
confirmation that `GET /api/about` gives, a link to the requirements, and
the name and email the decision records. A form without a name or an email
says so and links to the settings.

The "Texts" page is where texts are read, scored, tagged, edited and
decided one after the other. What is open is a filter, kept in the
address: quick filters and the filter bar set it, and a menu lists what
can be opened, every text, each status, the libraries, the tags, the
licenses and the filters saved in the browser. The list of the open
texts, a page at a time, hides at the left edge and comes out while the
pointer rests there, or after a button on a phone.

The content of the current text fills the middle and is always editable.
An edit is saved only by its button or Ctrl+S, as a content change: the
text goes back under review and leaves every open library, and an
admitted text asks once more first.

The review panel at the right holds a row for each score name, its
choices read from `GET /api/scores/{name}/values` with a number pad for
any other value, the most used tags as toggles, the moves and decisions,
presets of scores and tags kept in the browser, and the text's facts and
other actions. Clicks on scores and tags are gathered and saved as one
change of scores and one of credits a moment after the last, or when
another text opens. Admit & next decides at once; a refusal, a return to
review and the admission of a refused text ask first, in a line at the
bottom that holds the confirmation and the name the decision records. A
line names the text and the decision after each one.

Keys outside the editor do the same: J and K move, A and R admit or
refuse and open the next text, U puts the text back under review, T takes
it out of the open library and X ticks it. Escape leaves the editor.
Without a name or an email, a decision asks for them and decides nothing.
The blob mode shows the open texts as one continuous text, read a page at
a time as it scrolls; a click on a passage opens that text.

"Read it whole", in the head of a library, and "read whole" wherever a
library is named (the Libraries page, Open ▾, a text's panel, the
Overview), open the page "Read whole", under the library's head and its
actions:
the library as one strip of text, its texts one after another in the
numeric order of their ids (wpt_depot.strips). A library can hold more
characters than a browser can lay out, so the page holds only what is
near the screen: the server gives the strip in reads of about 64 KB of
characters after or before a place (a text and a character in it), each
ending with a line, cut from the parts of the texts it reaches, never a
whole text; the page keeps about 1 MB of characters, and lays out only
the blocks of lines that cover the screen and two screens on each side,
taking off those that scroll away and putting on those that scroll in,
the view held where it was. Its own scrollbar stands for the whole strip
in characters. A jump to a position needs the library's map: the
characters before every 4096th text, made by `wpt depot web` in a thread
of its own, one library at a time, in one walk of the library's index
that reads no text, kept in `<depot file>.work/strips/` and made again
once the library holds other texts, the last map answering meanwhile; a
jump then reads at most 4096 rows to find its text (twice that with a map
of an earlier state, then landing near), and lands at the start of that
line. The server keeps the last 256 MB of characters of parts it
unpacked, so reads on through one part unpack it once.

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
of one tag, or of one or several licenses, libraries or statuses, from
their indexes in the numeric order of ids, the windows of several values
merged; the texts the title index gives for every word of a search of 3
characters or more, each depot name's from the index in that order, with
the text whose id a word is; the ids named, sorted, when they are at most `SORTED`
(`wpt_depot/depot.py`); else every current text, from `texts_order`. It
tests each candidate against the rest of the filter. In the numeric order
of ids it reads at most `WINDOW` candidates after its place (`TEXT_WINDOW`
with a search in the texts themselves), and in the order of statuses at
most that many from `texts_status`, so a page is bounded whatever the
depot's size and however few texts match. A page cut at that bound is
partial: it can hold fewer rows than asked, or none, and the next page
reads on from its last candidate. The list and the blob say so and
read on when asked. No index serves the other orders, which sort every
candidate, so the server gives them only when the listing holds at most
`SORTED` texts and refuses them past it; a page says whether its listing
allows them, and the list offers them only then. The counts of a filter
come apart, in steps: each step counts the matches in one slice of at most
a million texts of the page's listing, in its order, so each one stays well
under the time limit of a reader. A filter that names its texts is counted
in one step over those texts. A filter of statuses, of at most one tag,
license or library, and of whether a locked library holds the texts is
counted at once from the depot's counts, with no step. The pages run one step at a time, the counts
open on a page taking turns, and add the steps up. The numbers grow as the
steps answer, with how many texts are counted out of the listing's texts, a
number `counts` gives. A count
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
each ticked. The decision, the license change, the put, the take-out, the
fork or the report matches the filter when it writes, leaves out the ids
unticked and every text changed after that change number, and its answer
says how many it left out. A text that matches the filter only after the
page was read is left out with them. A text's own page names that one text
in the filter. A filter reaches the server as fields, which
`wpt_depot.filters` turns into its SQL condition, and every figure a page
shows comes from a route of its own, so no page sends SQL but the page
where a person types it. The commands and the web reach the same SQL in `wpt_depot`.

The server runs its long work as jobs (`wpt/depot/jobs.py`): a write over
a selection, a fork or a lock of a library, the training of a vocabulary
and the build of a corpus. The
queue is a SQLite file beside the depot, `<depot file>.jobs`, which nothing
backs up. One job runs at a time, in a process of its own, which runs the
same `wpt_depot` operation as one change and sends the name of each stage
it starts and its progress. A write over at most `INLINE` texts runs in
the request instead, while no job is queued or running: a selection whose
filter names at most `INLINE` ids, or a fork or a lock of a library that
holds at most `INLINE` texts. Otherwise every such write is a job, since
it waits for the running job's write lock and only a job shows that it
waits. Either way the answer is the job's state, and `GET /api/jobs/{id}`
reads it again while it runs, so the page shows the stage of a long write
and its end. A job survives a restart of the server: a queued job stays
queued, and a job a restart cut short runs again, at most `RESTARTS`
times. Each dataset has at most one corpus job open and each recipe one
vocabulary job (409 otherwise). Only `wpt depot jobs cancel ID` cancels a
job. The queue's file also holds what the downloads need
(`wpt/depot/downloads.py`): when each corpus was last downloaded, the
streams being sent, how many texts each export counts, and each export
part's size and sha256 once it was sent whole.

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
when the store is missing on this machine. A command prints the same
failure's kind and message and ends with exit code 2.

A person's SQL reaches only the depot's file. It runs on the read-only
connection whose authorizer lets a statement only read ("SQL" in
[The depot](#the-depot)), so no query writes, attaches another file, runs
a pragma (but reading `data_version`) or loads an extension, and SQLite has no function that reads
another file or an address on the network. A filter's condition is a query
as well, under the same rule. `wpt depot sql` follows it too.

A route that runs a person's SQL, or that reads many texts or
memberships (`POST /api/sql`, `POST /api/texts/search`, `POST
/api/texts/passages`, `POST /api/texts/counts` and `POST /api/texts/blocked`),
runs it in a
reader process of its own (`wpt/depot/web/readers.py`), never in the
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

`wpt depot web` serves the depot `WPT_DEPOT` names. It listens only on
127.0.0.1 and checks neither the `Host` header nor a request's
`Content-Type`: like the commands, it asks for no login and no header, and
it answers any client that reaches it, through a proxy in front of it
included. Every such client reads and changes the depot through every
route: it adds texts and licenses, makes libraries and datasets, and sends
decisions with a declaration it made up. A web page open in the same
browser can do the same with a cross-origin `POST` with `Content-Type:
text/plain`, which the browser sends without asking first, and so can a
site that makes its own name resolve to 127.0.0.1 (DNS rebinding).

A pull (which makes the machine fetch another depot), a remote's import,
a tokenizer recipe file's import, and a change of the depot's maintainer or
takedown contact are commands only: the server has no route for them. A dataset made or edited through the server reads only
this depot's libraries: a library named by another depot's address is
refused (422) before anything is opened, and only `wpt depot dataset new`
and `edit` name one. Some routes reach past the depot. `POST
/api/erasures/decisions`, once it carries out an erasure, marks stale the
store's corpora that hold a text it cut, in the background, and sends no
corpus until that ends. `GET /api/datasets/{id}` lists
the store's corpora built from the dataset. The build of a corpus writes
it into the store, and the downloads read it from there. A dataset's
export reads its texts from the depots, never from the store. The training of
a recipe whose dataset is another depot's reads that depot's sample route.

## Packages

| Package | Role |
| --- | --- |
| `wpt-depot` | the depot: texts and structured items, the decisions on them, libraries, erasures and datasets, in the tables of one SQLite file (`depot.py`, the only code that changes them, `schema.py`); its connections and the write that records each change (`database.py`); the app's policy on decisions (`policy.py`); a filter of texts turned into an SQL condition (`filters.py`); backup and restore into the store (`backup.py`); the upgrade from an older format (`upgrade.py`, `legacy.py`); erasures carried out (`erasure.py`); the kinds of items and their schemas (`kinds.py`, `kinds/`); clone and pull (`follow.py`); a dataset as a remote (`remote.py`); a dataset's corpus built into the store as a stream (`corpus.py`), in the order of its texts' export keys (`order.py`), a long text taken as pieces (`pieces.py`), estimated before it is built (`estimate.py`), what the store keeps of corpora (`corpora.py`), a corpus's texts made into parts to download (`exports.py`) and a model's notice from a corpus (`notice.py`); the table of counts (`counts.py`); `WPT_DEPOT` (`settings.py`) |
| `wpt-recipes` | the model and series recipes and the images of agentic tasks: the service layer that reads and checks them (`recipes.py`), the sizes a series derives (`series.py`), and the audit hook that keeps `data/` to the service layer (`guard.py`) |
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
  `routes.py`, `builds.py` for the routes of the tokenizer recipes, the jobs, the builds and the downloads, and `ui.py`, which builds the interface of `apps/wpt/ui`
  with Bun). The root `package.json` makes one Bun workspace of the
  interface and the packages of the organization's design system, the git
  submodule `vendor/wic-design-system`, whose tokens and components the
  interface is built on; `git submodule update --init` fetches it, and
  `bun install` builds its packages. What its routes and the subcommands share is
  `wpt/depot/shared.py`. It listens on 127.0.0.1 only; another machine
  reaches it through an SSH tunnel, Tailscale, or a proxy in front of it
  ([Two interfaces](#two-interfaces); [depot-web.md](depot-web.md)).

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
- `data`: the recipes' check (`wpt_recipes.Recipes.check`): every recipe
  and image file matches its schema, and the rules across files hold (a
  series derives every size, an image's pool takes no built-in tool's name, a
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
