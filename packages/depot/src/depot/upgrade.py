"""Bringing a depot of an older format up to the code's format
(schema.FORMAT), the only way a depot's format changes (`depot
upgrade`).

A depot of format 3 or older is a folder of Delta Lake tables
(depot.delta). The upgrade first runs, in that folder, the step from
its format to the next until it reaches format 3 (delta.STEPS), each
recorded as it was in those formats. Then it writes a new SQLite file of
format 4 from the folder, which it never changes again: by default the
folder's own path with `.sqlite` added, or the file given (`--to`). It
refuses a folder whose journal holds a change never finished, since only
the code of its format knew how to finish it.

An SQLite file of format 4 or newer, the one given or the one just written
from a folder, is brought up in place by the step from its format to the
next (FILE_STEPS) until it reaches the code's format. It first refuses
while any other connection has the file open (_alone): the code of an
older format checks the format only when it opens a depot, so a web app or
a job left running would go on writing into a half-upgraded file. A step changes the
layout and adds a revision of the about row at its new format under a
change of its own, kind 'upgrade'. A step that fits one transaction leaves
the file as it was when it is cut short, and runs again whole. A step that
rewrites every current text runs in several transactions: its first one
makes the table UNDER_WAY, which marks the depot as half upgraded, with
the new layout and the about revision; each next one handles one batch of
rows and records in UNDER_WAY where it stopped; the last one drops it. A
half-upgraded depot is at no format the code reads: every way in but the
upgrade refuses it (errors.HalfUpgraded), and the upgrade run again goes
on from the batch after the last one written, saying how far it is every
SAY_EVERY seconds. Before a step, the file
is marked to send a full copy at its next backup, under a new timeline
(depot.database), since the change files record no change of layout.
The step to format 5 builds the index texts_order (depot.schema) in
one statement, which reads every current text's row once and rewrites no
table; it says what it does before it starts and how long it took. The
step to format 6 makes, in its first transaction, the tables of the
tokenizer recipes, their vocabularies and the texts each vocabulary trained
on, and the column max_tokens of the datasets, all empty; then it writes
each current text's decisions as rows of text_decisions and its status,
BATCH rows of texts_revisions a transaction, then builds the index
texts_status and fills the counts of format 6 (COUNTS_6). The step to
format 7 adds, in its first transaction, the column export_key of
memberships and pins and the column vocabulary of the datasets, empty;
then it writes the export key (depot.ids.export_key) of every row of
those two tables, in that order (KEYED_7: every membership, then every
pin, by the text it names), BATCH rows a transaction, read and written by their rowid with
memory bounded by the batch, saying how many rows a second it writes; the
table upgrade_under_way keeps, for each table, the last rowid written, so a
run after a stop goes on in the table and at the row it reached; then it
builds the indexes memberships_key and pins_key, each in one
statement. The step to format 8 builds the index text_decisions_decision
(depot.schema) in one statement, which reads every row of
text_decisions once and rewrites no table, between a first transaction
that marks the depot half upgraded and a last one that drops the mark, so
a stop while it builds leaves a depot that only the upgrade opens; it says
what it does before it starts and how long it took. Format 8 is also the
format of a depot that may hold decisions that put texts back under review
(depot.policy), which the code of format 7 does not know: that code
refuses a depot of format 8. The step to format 9 marks the depot half
upgraded and adds the columns characters and bytes of counts, the column
figures of the dataset revisions and the table library_moves (MOVES_9) in
its first transaction; then it fills the counts of what each library holds by kind
and by tag (LIBRARY_KIND_9, LIBRARY_TAG_9): the memberships that have not
ended, the current revisions and the content versions are each read in
one sequential scan (SCANS_9), sorted by text id by SQLite in its
temporary folder, and merged (_merged_9), so the time grows with the rows
whatever order the file stores them in and no row is looked up through an
index. SQLite sorts the three scans before the first row comes back, so
its temporary folder (SQLITE_TMPDIR) holds the three sorts at once, about
SORTED_9 bytes a membership, which the step says before it starts. Every
BATCH memberships or so, at the end of a text, one transaction adds their
sums to counts, raises the change of library_moves of each library to
the latest change a text it holds among them was revised at, and records
that text's id in upgrade_under_way; a run after a stop scans only the
texts after it. It says how many memberships a second it counts every
SAY_EVERY seconds. Its last transaction raises library_moves to the latest
change each library's memberships joined or left at, read from the index
memberships_key alone (JOINED_OR_LEFT_9), or, in a depot that follows an
upstream, whose rows carry several numberings, makes every change of it
null; then it gives each dataset revision its figures (_figures_9), read
from the counts it filled, for each library they prove exact and null for
the others, and drops the mark. A library whose memberships and texts no
change touched after the change a revision reads it at holds what it held
then, so only those get figures, and an upgraded revision never holds
wrong ones. The step to format 10 marks the depot half upgraded and makes
the tables text_tags and text_licenses (TABLES_10) and the title index
with its owners (TITLES_10) in its first transaction; then it lists the
current texts by tag and by license, puts their titles into the title
index (TITLE_ROWS_10, a slot in title_owners for each depot name first, and
refused for an id the index cannot number, ODD_TITLE_10) and counts their
score names and the splits of every text and of each tag, license and
library by status and by whether a locked library holds them (LISTED_10,
SCORES_10, SPLITS_10), reading the current
revisions in rowid order, BATCH rowids a transaction, each recording the
last rowid in upgrade_under_way, so a run after a stop goes on after it; it
says how many rows a second it reads every SAY_EVERY seconds. Then it
builds each index of INDEXES_10 in one statement, skipping one that
exists, so a run after a stop builds only those missing. Its last
transaction drops the index text_decisions_decision, which
text_decisions_order serves in its place (DROPPED_10), counts the pins
older than their text's current revision (NEWER_COUNTS_10) and drops the
mark.
The step to format 11 builds the partial index content_lists_long
(LONG_11: the plain content versions a corpus build cuts into pieces) in
one statement, which reads every row of content_lists once and rewrites no
table, between a first transaction that marks the depot half upgraded and
a last one that drops the mark, as the step to format 8 does; it says what
it does before it starts and how long it took.
The step to format 12 gives every text a kind: a content version of no
kind is a plain text, which format 12 names plain; it counts what each
library holds of each kind on its own, where format 11 counts the
conversations and tasks together (items) and every other kind together
(other); it gives the kind conversation the schema that takes the excerpt
block (depot.quotes), which the step holds itself (CONVERSATION_12),
as a new depot of format 12 starts with it; and it writes the table
quotes, the texts each content version of a conversation quotes. Its first
transaction marks the depot half upgraded, drops the index
content_lists_long, whose condition names the kind, adds the row plain to
the table kinds (PLAIN_12), writes the conversation schema, deletes the
counts of what each library holds by kind but those of its plain texts,
which stay right (GROUPS_12), and makes the table quotes and its index
quotes_text, empty (QUOTES_12). Then, in one transaction, it adds to the
counts what the current content versions of the other kinds give each
library that holds their text, by kind (ITEMS_TABLE_12: the items copied
from one sequential scan of content_lists, then found among the current
revisions and the memberships that have not ended by one sequential scan
of each, SCANNED_12, or, when they are few, ITEMS_SHARE_12, each looked up
through their indexes, LOOKED_UP_12). Then it writes the kind plain on
every content version of no kind, in rowid order, BATCH rowids of
content_lists a transaction, and the rows of quotes of the conversations
among those rowids, each read from its parts (_quotes_12, a lookup in
parts for each conversation); each transaction records the last rowid in
upgrade_under_way, so a run after a stop goes on after it, the quotes it
added kept (a run of the code that counted each batch's kinds with
KINDS_12, stopped half way, goes on counting them that way); it
says how many rows a second it writes every SAY_EVERY seconds. Each batch
rewrites the rows it changes, and only those: no index of content_lists
but content_lists_long names the column kind. Its last transaction checks
that no content version is left without a kind, which reads the table
once, declares the column kind NOT NULL (NOT_NULL_12) by changing the
table's statement in sqlite_schema alone, as SQLite's documentation allows
for a constraint every row meets (a table's statement names no layout of
its rows on disk, so no row is rewritten), builds the index
content_lists_long again over kind plain (LONG_12), in one statement that
reads the table once, gives every dataset revision's figures the layout of
format 12 (_figures_12) and drops the mark. A library's figures of format
11 that hold no item become its plain texts and every other kind at 0;
those that hold items, of a library this depot owns, are read again from
the counts it filled when they prove them exact, as the step to format 9
decides, and are null otherwise, since format 11 never split them by kind.
The step to format 13 runs in stages (STAGES_13): its first transaction
marks the depot half upgraded with a row of upgrade_under_way for each
stage and one for each dataset revision a corpus record names (NAMED_13:
the records given, those of the corpora under corpora/ of the dock, read once, so a
run after a stop needs no dock), and each stage whose row is left runs,
deleting its row in its last transaction, so a run after a stop starts
again at the first stage not ended; the last transaction drops the mark.
Its stage tables, in one transaction, makes the tables of the snapshots,
snapshots and snapshot_sets, the index of their names by owner, the
table legacy_snapshot_of and the table taggings with its index
taggings_change, empty (TABLES_13), adds the column sample of
tokenizers_revisions (SAMPLE_13), null on every revision it holds, and
the columns content_kind, content_length, content_bytes and current_tags of
texts_revisions (SIZES_13), null until the stage sizes fills them, and
counts from every current text what the texts of each status hold by kind
and by plain tag set, and the characters and bytes of each tag's and
license's split (COUNTS_13), their rows before deleted. It reads them from
tables of the writer's temporary database, filled by one sequential scan
each: the kind, length and bytes of every content version
(SIZES_TABLE_13), the texts locked libraries hold (LOCKED_TABLE_13) and
what the counts read of every current text (CURRENT_TABLE_13), so no row
of the file is looked up through an index. Every stage reads that way: on
a file far bigger than the memory, each lookup is a random read of the
disk. Its stage taggings turns the tags every revision of a text holds
into rows of taggings (_tagged_13: a tag joined at the change of the
revision that first holds it, after one that lacks it, and left at the
change of the next revision that lacks it), reading the revisions copied,
in one sequential scan sorted by text id, into a table of the writer's
temporary database (TAGGED_TABLE_13, COPY_TAGGED_13), in SQLite's
temporary folder (SQLITE_TMPDIR, else /var/tmp, /usr/tmp or /tmp), about
TAGGINGS_SORTED_13 bytes a revision; the rows of the texts of about BATCH
revisions are written in a transaction that records the last text's id in
upgrade_under_way, so a run after a stop copies only the texts after it,
and a row written twice is ignored; no read of the file stays open across
a commit, so the WAL is cut back after each. The revisions keep their column
tags, which nothing of format 13 reads; the counts of
tags, read from the revisions before this stage, stay right. Its stage
sizes copies onto every revision of a text the kind, length and bytes of
the content version it reads, looked up in the temporary table of the
sizes (SIZES_TABLE_13, made again when the stage tables ran in another
run), and writes onto each current revision the tags it holds in the order
of their names, each once, as the copy of its text's tags (current_tags;
FILL_SIZES_13), reading the revisions in rowid order, BATCH rowids a
transaction, each recording the last rowid in upgrade_under_way, so a run
after a stop goes on after it. Its stage locked makes a snapshot of each
locked library the depot owns, and its stage datasets one of each dataset
revision a corpus names and of each dataset's current revision, or names
for a revision that read one locked library alone that library's
snapshot, in legacy_snapshot_of (depot.conversion says what each
holds, how it is named, how it is checked and how it is read: every
snapshot at once, in one sorted merge of sequential scans). The first of
the two stages that runs measures every snapshot both have left in that
one merge (_measured_13), or takes what the check right before measured
when the depot did not change since (Check.state, _state_13: `depot
upgrade --yes` runs the check, then the upgrade, which then reads no
library again); each snapshot is then written in a transaction of its own,
under the step's change, and a run after a stop skips those written. A
text the snapshot would read at another revision than the library read it
stops the stage (Refused), the snapshot unwritten: `depot upgrade
--check` (check) finds it before anything is written. The stage datasets
then says, for each converted revision, the data section of a model order on its snapshot with the revision's
shares, size and vocabulary (conversion.order_data). Open libraries are not converted. Its stage counts, in
one transaction that reads only the table counts, deletes the counts of
what libraries hold (LIBRARY_FACETS_13) and folds the rows of the facets
split by status and by whether a locked library held the text (SPLIT_13)
into one row of each status and name. Its stage views drops the views
libraries and datasets an older file may hold and deletes the strip
reader's folder of format 12 beside the file (STRIPS_13). The tables of the
libraries and datasets stay as they are, rows included; no write touches
them any more, and nothing is dropped nor vacuumed.
The step to format 14 builds the index texts_length (LENGTH_14: the
current texts by status, then by the length of the content version each
reads, then by id) in one statement, which reads every current text's row
once, sorts in SQLite's temporary folder and rewrites no table, and makes
the rows of counts of each score's values (SCORE_VALUES_14, the facet
`score value` of depot.counts) in one statement that reads every
revision's row once, between a first transaction that marks the depot half
upgraded and a last one that drops the mark, as the step to format 11
does; it says what it does before it starts and how long it took.
The step to format 15 gives the kind conversation the schema that takes
what an agent harness records (docs/chat-format.md, "Recorded from a
harness"), which the step holds itself (CONVERSATION_15), as a new depot
of format 15 starts with it. It writes that one row of kinds and the
about revision in one transaction, so a stop leaves the depot at format
14 and the step runs again whole; it needs no half-upgraded mark, and no
conversation is read or rewritten, since every conversation of format 14
matches the new schema.

The check (check) opens the file read-only and writes nothing: for a depot
of format 11 it checks what the step to format 12 checks before it ends,
then, for one of format 11 or 12, it works out and reads every snapshot
the step to format 13 would make, checks every revision each reads (in
the one sorted merge depot.conversion.measure_all makes, which only
reads, its sorts in SQLite's temporary folder), and says the names it
gives, the order data sections, the cost of every step left together
(seconds from the rates RATE_12, RATE_13_TABLES, RATE_13_TAGGINGS and
RATE_13_SIZES, the last three measured on bench/large.py's file, bigger than the memory,
and the time its own reads took, the free disk the temporary folder needs,
the WAL at its largest, the disk added, the full copy the next backup
sends) and what to do before (RUNBOOK). `depot upgrade --yes` runs it
again and upgrades only when it passes, handing the upgrade what it
measured (Check.measured) with the depot's state when it read it
(Check.state). `depot upgrade --check` saves what it measured with
that state beside the depot's file (saved_check_path, _save_check), and a
later check takes it while the depot's state is the same (_saved_check),
so the check can run while the web app serves and `--yes` reads no
library again; the upgrade deletes the file when it ends.

An SQLite file is put in WAL mode before its first step (_in_place), as
Database.create makes every depot file: the steps to formats 9 and 13
write what a sorted scan found in transactions of their own while the scan
is still open, which a file in another journal mode would keep waiting.

A file written from a folder is made with the code's tables, so the
content versions of no kind it copies are written of kind plain at once,
and the step to format 9 reads kind plain as no kind (PLAIN_9).

A clone of those formats named its upstream by the upstream's folder
or bucket address; a clone of format 4 or newer follows its upstream over the
upstream's web app (depot.follow), so upgrading a clone takes that web
address (`--upstream`), which its about row then names, and is refused
without it.

The new file is written as `<file>.upgrading` and renamed once whole, so an
upgrade cut short (a crash, a full disk) leaves no depot at the file's
path. Run again, it goes on: each data file of each table goes into the
new file in one transaction that also records it in the table
upgrade_progress, so a data file is read once and a restart picks up after
the last one written. The last transaction makes the upgrade's change and
drops upgrade_progress; a file that holds that change (an about revision
of format 4) is whole, and a restart only renames it. A table's rows are read a batch at a time, so the
memory it takes does not grow with the depot. Lists become JSON text,
times ISO text, a part's text is compressed (depot.parts), and the
format 4 columns the folder lacks are made from it: a revision is the
current one when no higher revision of its id exists, a decision or a
notice records the journal entry that first named it (the upgrade's change
when none did), a pin the change that locked its library, each journal
entry becomes a row of changes (made at its decision's time, when it has
one), and numbers holds the highest number of each kind of the depot's own
ids, erasures' needles included. The upgrade itself is the last change, a
new revision of the about row at format 4. The new file's next backup
sends a full copy (depot.backup).
"""

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import apsw
from wic_utils.lines import say

from . import ids, policy
from .database import JOURNAL_LIMIT, Database, now
from .errors import DepotExists, OtherFormat, Refused
from .follow import Upstream
from .delta import LAST, STEPS, DeltaFolder
from .parts import Parted, Reader, pack
from .schema import FORMAT

UPGRADING = ".upgrading"  # what the name of the file being written adds to the new file's name
ROWS = 4096  # rows of a data file read at a time
PARTS_ROWS = 16  # rows of parts read at a time: each holds up to 64 MiB of text
WRITTEN = 4  # the format of the SQLite file written from a folder
# the index of format 5: the current texts in the numeric order of their ids (ids.order_sql)
ORDER = ("((CASE WHEN (rtrim(id, '0123456789') <> id "
               "AND rtrim(id, '0123456789') GLOB '*-') THEN substr(id, 1, length(rtrim(id, '0123456789')) - 1) ELSE "
               "id END) || char(1) || CASE WHEN (CASE WHEN (rtrim(id, '0123456789') <> id AND rtrim(id, '0123456789') "
               "GLOB '*-') THEN CAST(substr(id, length(rtrim(id, '0123456789')) + 1) AS INTEGER) END) IS NULL THEN "
               "'1' ELSE '0' || printf('%020d', (CASE WHEN (rtrim(id, '0123456789') <> id AND rtrim(id, "
               "'0123456789') GLOB '*-') THEN CAST(substr(id, length(rtrim(id, '0123456789')) + 1) AS INTEGER) "
               "END)) END || char(1) || id)")
TEXTS_ORDER = f"CREATE INDEX IF NOT EXISTS texts_order ON texts_revisions ({ORDER}) WHERE latest"
# format 6: the index of the current texts by status, then in the order of their ids, and its tables (a file
# written from a folder is made with the code's tables, so they may exist already, the column status too)
TEXTS_STATUS = f"CREATE INDEX IF NOT EXISTS texts_status ON texts_revisions (status, {ORDER}) WHERE latest"
TABLES_6 = ("CREATE TABLE IF NOT EXISTS text_decisions (text TEXT NOT NULL, decision TEXT NOT NULL, "
            "change INTEGER NOT NULL, PRIMARY KEY (text, decision)) WITHOUT ROWID",
            "CREATE TABLE IF NOT EXISTS counts (facet TEXT NOT NULL, name TEXT NOT NULL, texts INTEGER NOT NULL, "
            "PRIMARY KEY (facet, name)) WITHOUT ROWID",
            "CREATE TABLE IF NOT EXISTS tokenizers_revisions (id TEXT NOT NULL, revision INTEGER NOT NULL, "
            "change INTEGER NOT NULL, latest INTEGER NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL, depot TEXT, "
            "dataset TEXT NOT NULL, sample_bytes INTEGER NOT NULL, vocab_size INTEGER NOT NULL, note TEXT, "
            "PRIMARY KEY (id, revision))",
            "CREATE UNIQUE INDEX IF NOT EXISTS tokenizers_latest ON tokenizers_revisions (id) WHERE latest",
            "CREATE TABLE IF NOT EXISTS vocabularies (recipe TEXT NOT NULL, version INTEGER NOT NULL, "
            "change INTEGER NOT NULL, sha256 TEXT NOT NULL, merges TEXT NOT NULL, texts INTEGER NOT NULL, "
            "bytes INTEGER, made_at TEXT NOT NULL, PRIMARY KEY (recipe, version))",
            "CREATE TABLE IF NOT EXISTS vocabulary_texts (recipe TEXT NOT NULL, version INTEGER NOT NULL, "
            "position INTEGER NOT NULL, depot TEXT NOT NULL, text TEXT NOT NULL, change INTEGER NOT NULL, "
            "PRIMARY KEY (recipe, version, position)) WITHOUT ROWID")
# format 6: each facet of counts, with the query of its (name, texts) rows over the whole depot
COUNTS_6 = {
    "texts": "SELECT '', count(*) FROM texts_revisions WHERE latest",
    "status": "SELECT status, count(*) FROM texts_revisions WHERE latest GROUP BY status",
    "tag": "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.tags) j WHERE t.latest GROUP BY j.value",
    "license": "SELECT j.value, count(*) FROM texts_revisions t, json_each(t.licenses) j WHERE t.latest "
               "GROUP BY j.value",
    "library": "SELECT library, count(*) FROM memberships WHERE left_change IS NULL GROUP BY library",
    "decision": "SELECT decision, count(*) FROM text_decisions GROUP BY decision",
}
# format 6 columns added to tables of format 5: (table, column, its declaration)
COLUMNS_6 = (("texts_revisions", "status", "TEXT"), ("datasets_revisions", "max_tokens", "INTEGER"))
# format 7 columns added to tables of format 6: (table, column, its declaration)
COLUMNS_7 = (("memberships", "export_key", "INTEGER"), ("pins", "export_key", "INTEGER"), ("datasets_revisions", "vocabulary", "TEXT"))
# format 7: the tables whose rows get their text's export key, in the order they are filled, and the column naming it
KEYED_7 = (("memberships", "text"), ("pins", "text"))
INDEXES_7 = (("memberships_key", "CREATE INDEX IF NOT EXISTS memberships_key ON "
                                 "memberships (library, export_key, joined_change, left_change, text)"),
             ("pins_key", "CREATE INDEX IF NOT EXISTS pins_key ON pins (library, export_key, text, revision)"))
# format 8: the index of text_decisions by decision
TEXT_DECISIONS_DECISION = "CREATE INDEX IF NOT EXISTS text_decisions_decision ON text_decisions (decision)"
# format 9: the columns added to counts, the facets of its new rows, the kinds of item a corpus takes, and the three
# scans the counts are filled from, each in the order of the texts' ids, after the text `{after}` names when given (+
# keeps SQLite from reading an index for that order, and NOT INDEXED from reading the rows through one)
COLUMNS_9 = ("characters", "bytes")
LIBRARY_KIND_9, LIBRARY_TAG_9 = "library kind", "library tag"
FACETS_9 = f"('{LIBRARY_KIND_9}', '{LIBRARY_TAG_9}')"
ITEMS_9 = ("conversation", "task")
SCANS_9 = ("SELECT text, library FROM memberships NOT INDEXED WHERE left_change IS NULL{after} ORDER BY +text",
           "SELECT id, content_version, tags, change FROM texts_revisions NOT INDEXED WHERE latest{after} ORDER BY +id",
           "SELECT id, version, kind, length, bytes FROM content_lists NOT INDEXED WHERE 1{after} ORDER BY +id, +version")
SCANNED_9 = ("text", "id", "id")  # the column of each scan of SCANS_9 that names the text
SORTED_9 = 100  # bytes of temporary files the three sorts of SCANS_9 take for each membership, about
# format 9: the table library_moves, and the latest change each library's memberships joined or left at, read from the
# index memberships_key alone
MOVES_9 = ("CREATE TABLE IF NOT EXISTS library_moves (library TEXT NOT NULL, change INTEGER, PRIMARY KEY (library)) "
           "WITHOUT ROWID")
JOINED_OR_LEFT_9 = ("SELECT library, max(max(joined_change, coalesce(left_change, 0))) FROM memberships "
                    "INDEXED BY memberships_key GROUP BY library")
KINDS_9 = ("plain", "items", "other")
PLAIN_9 = "plain"  # the kind a file written from a folder gives a plain text, read as no kind (the module says why)
# format 10: the listings of the current texts by tag and by license, each row a name and the text's place in the
# numeric order of ids (ORDER over the column text, ORDER_TEXT_10), the facets of counts it adds, and the indexes it
# builds, by name, in the order they are built
ORDER_TEXT_10 = ORDER.replace("id", "text")
TABLES_10 = ("CREATE TABLE IF NOT EXISTS text_tags (tag TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY (tag, key)) "
             "WITHOUT ROWID",
             "CREATE TABLE IF NOT EXISTS text_licenses (license TEXT NOT NULL, key TEXT NOT NULL, "
             "PRIMARY KEY (license, key)) WITHOUT ROWID")
SCORE_10, NEWER_10 = "score", "pinned newer"
# what one batch of current revisions, the rowids from :low (excluded) to :high, adds: the rows of each listing, and
# the count of each score name (NOT INDEXED: the rows are read in rowid order, from the table itself)
LISTED_10 = (("text_tags", "tag", "tags"), ("text_licenses", "license", "licenses"))
BATCH_ROWS_10 = "FROM texts_revisions t NOT INDEXED WHERE t.latest AND t.rowid > :low AND t.rowid <= :high"
SCORES_10 = f"SELECT s.value ->> 'name', count(*) {BATCH_ROWS_10.replace('WHERE', ', json_each(t.scores) s WHERE')} GROUP BY 1"
# the facets that split every text and the texts of each tag, license and library by status and by whether a locked
# library holds them, each row named `<status> <locked> <name>` (locked 1 or 0): what one batch adds to each, the
# locked test (LOCKED_10) read from the text's side through the index memberships_held
LOCKED_10 = ("EXISTS (SELECT 1 FROM memberships lm INDEXED BY memberships_held CROSS JOIN libraries_revisions ll "
             "ON ll.id = lm.library AND ll.latest AND ll.locked_at IS NOT NULL WHERE lm.text = t.id "
             "AND lm.left_change IS NULL)")
SPLIT_NAME_10 = f"t.status || ' ' || {LOCKED_10} || ' ' || "
SPLITS_10 = (
    ("texts split", f"SELECT {SPLIT_NAME_10}'', count(*) {BATCH_ROWS_10} GROUP BY 1"),
    ("tag split", f"SELECT {SPLIT_NAME_10}j.value, count(*) "
                  f"{BATCH_ROWS_10.replace('WHERE', ', json_each(t.tags) j WHERE')} GROUP BY 1"),
    ("license split", f"SELECT {SPLIT_NAME_10}j.value, count(*) "
                      f"{BATCH_ROWS_10.replace('WHERE', ', json_each(t.licenses) j WHERE')} GROUP BY 1"),
    ("library split", f"SELECT {SPLIT_NAME_10}m.library, count(*) "
                      + BATCH_ROWS_10.replace("WHERE", "CROSS JOIN memberships m INDEXED BY memberships_held "
                                                       "ON m.text = t.id AND m.left_change IS NULL WHERE")
                      + " GROUP BY 1"))
# the title index (depot.titles at format 10): an FTS5 table of the trigrams of each current title, each row
# numbered by its text's id, its owner's slot (title_owners) times SPAN_10 plus its number; what one batch adds to it,
# in the order of the row numbers (each lower number than the one before writes a new segment out), and the test of an
# id the index cannot number
SPAN_10 = 1 << 44
TITLES_10 = ("CREATE TABLE IF NOT EXISTS title_owners (owner TEXT NOT NULL, slot INTEGER NOT NULL, "
             "PRIMARY KEY (owner))",
             "CREATE VIRTUAL TABLE IF NOT EXISTS text_titles USING fts5(title, content='', columnsize=0, detail=none, "
             "tokenize='trigram')")
OWNER_10 = (f"(CASE WHEN rtrim(t.id, '0123456789') <> t.id AND rtrim(t.id, '0123456789') GLOB '*-' "
            f"THEN substr(t.id, 1, length(rtrim(t.id, '0123456789')) - 1) ELSE t.id END)")
NUMBER_10 = (f"(CASE WHEN rtrim(t.id, '0123456789') <> t.id AND rtrim(t.id, '0123456789') GLOB '*-' "
             f"THEN CAST(substr(t.id, length(rtrim(t.id, '0123456789')) + 1) AS INTEGER) END)")
TITLED_10 = BATCH_ROWS_10 + " AND t.title IS NOT NULL"
ODD_TITLE_10 = (f"SELECT t.id {TITLED_10} AND NOT ({NUMBER_10} IS NOT NULL AND {NUMBER_10} < {SPAN_10} "
                f"AND t.id = {OWNER_10} || '-' || {NUMBER_10}) LIMIT 1")
OWNERS_10 = ("INSERT INTO title_owners (owner, slot) SELECT n.owner, "
             "(SELECT coalesce(max(slot), -1) FROM title_owners) + row_number() OVER (ORDER BY n.owner) "
             f"FROM (SELECT DISTINCT {OWNER_10} AS owner {TITLED_10}) n "
             "WHERE n.owner NOT IN (SELECT owner FROM title_owners)")
TITLE_ROWS_10 = (f"INSERT INTO text_titles (rowid, title) SELECT (SELECT o.slot FROM title_owners o WHERE o.owner = "
                 f"{OWNER_10}) * {SPAN_10} + {NUMBER_10}, t.title {TITLED_10} ORDER BY 1")
NEWER_COUNTS_10 = (f"INSERT INTO counts (facet, name, texts) SELECT '{NEWER_10}', p.library, count(*) FROM pins p "
                   "JOIN texts_revisions t ON t.id = p.text AND t.latest WHERE t.revision > p.revision "
                   "GROUP BY p.library")
INDEXES_10 = (
    ("memberships_order", f"CREATE INDEX IF NOT EXISTS memberships_order ON memberships (library, {ORDER_TEXT_10}) "
                          "WHERE left_change IS NULL"),
    ("pins_order", f"CREATE INDEX IF NOT EXISTS pins_order ON pins (library, {ORDER_TEXT_10})"),
    ("memberships_joined", "CREATE INDEX IF NOT EXISTS memberships_joined ON memberships (joined_change)"),
    ("memberships_left", "CREATE INDEX IF NOT EXISTS memberships_left ON memberships (left_change) "
                         "WHERE left_change IS NOT NULL"),
    ("content_lists_change", "CREATE INDEX IF NOT EXISTS content_lists_change ON content_lists (change)"),
    ("text_decisions_change", "CREATE INDEX IF NOT EXISTS text_decisions_change ON text_decisions (change)"),
    ("pins_change", "CREATE INDEX IF NOT EXISTS pins_change ON pins (change)"),
    ("text_decisions_order", f"CREATE INDEX IF NOT EXISTS text_decisions_order ON text_decisions (decision, "
                             f"{ORDER_TEXT_10})"))
# the index of format 9 that text_decisions_order serves in its place, dropped once that one is built
DROPPED_10 = "DROP INDEX IF EXISTS text_decisions_decision"
# format 11: the partial index of the plain content versions longer than 1,048,576 characters
LONG_11 = ("CREATE INDEX IF NOT EXISTS content_lists_long ON content_lists (id, version) "
           "WHERE kind IS NULL AND length > 1048576")
# format 12: the kind of a plain text, its row of kinds (id, meaning, schema), the column kind of content_lists as
# format 11 declares it and as format 12 does, and the index content_lists_long over kind plain
PLAIN_12 = ("plain", "a text read as it is, which the read-text step learns from as text windows",
            '{\n  "$schema": "https://json-schema.org/draft/2020-12/schema",\n'
            '  "$id": "https://github.com/Workers-Intelligence-Commons/wpt/schemas/plain.schema.json",\n'
            '  "title": "WPT plain text (a text read as it is: its content is the text itself, any Unicode string, '
            'stored as it is and never as JSON; this schema is that string as a JSON value)",\n'
            '  "type": "string"\n}\n')
# format 12: the schema of the kind conversation, which takes the excerpt block (depot.quotes)
CONVERSATION_12 = (
    "{\n"
    "  \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",\n"
    "  \"$id\": \"https://github.com/Workers-Intelligence-Commons/wpt/schemas/conversation.schema.json\",\n"
    "  \"title\": \"WPT conversation (one line of a file in the chat format; docs/chat-format.md)\",\n"
    "  \"type\": \"object\",\n"
    "  \"additionalProperties\": false,\n"
    "  \"required\": [\"messages\"],\n"
    "  \"properties\": {\n"
    "    \"metadata\": {\"type\": \"object\", \"description\": \"what identifies the conversation where it was"
    " recorded, such as a training attempt's step, task and round; never read as tokens\"},\n"
    "    \"tools\": {\n"
    "      \"type\": \"array\",\n"
    "      \"description\": \"the tools offered in the conversation\",\n"
    "      \"items\": {\n"
    "        \"type\": \"object\",\n"
    "        \"additionalProperties\": false,\n"
    "        \"required\": [\"name\", \"description\", \"parameters\"],\n"
    "        \"properties\": {\n"
    "          \"name\": {\"type\": \"string\", \"minLength\": 1},\n"
    "          \"description\": {\"type\": \"string\"},\n"
    "          \"parameters\": {\"type\": \"object\", \"description\": \"the JSON Schema of the call's"
    " arguments\"}\n"
    "        }\n"
    "      }\n"
    "    },\n"
    "    \"messages\": {\n"
    "      \"type\": \"array\",\n"
    "      \"minItems\": 1,\n"
    "      \"items\": {\n"
    "        \"type\": \"object\",\n"
    "        \"additionalProperties\": false,\n"
    "        \"required\": [\"role\", \"content\"],\n"
    "        \"properties\": {\n"
    "          \"role\": {\"enum\": [\"system\", \"user\", \"assistant\", \"tool\"]},\n"
    "          \"content\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/$defs/block\"}},\n"
    "          \"metadata\": {\n"
    "            \"type\": \"object\",\n"
    "            \"description\": \"what is known about the message where it was recorded; never read as"
    " tokens\",\n"
    "            \"properties\": {\"draft\": {\"$ref\": \"#/$defs/draft\"}}\n"
    "          }\n"
    "        },\n"
    "        \"allOf\": [\n"
    "          {\"if\": {\"properties\": {\"role\": {\"const\": \"tool\"}}},\n"
    "           \"then\": {\"properties\": {\"content\": {\"items\": {\"properties\": {\"type\": {\"const\":"
    " \"tool_result\"}}}}}},\n"
    "           \"else\": {\"properties\": {\"content\": {\"items\": {\"properties\": {\"type\": {\"not\": {\"const\":"
    " \"tool_result\"}}}}}}}},\n"
    "          {\"if\": {\"properties\": {\"role\": {\"const\": \"assistant\"}}},\n"
    "           \"else\": {\"properties\": {\"content\": {\"minItems\": 1, \"items\": {\"properties\": {\"type\":"
    " {\"not\": {\"enum\": [\"thinking\", \"tool_call\"]}}}}},\n"
    "                                   \"metadata\": {\"not\": {\"required\": [\"draft\"]}}}}},\n"
    "          {\"if\": {\"properties\": {\"role\": {\"enum\": [\"assistant\", \"system\"]}}},\n"
    "           \"then\": {\"properties\": {\"content\": {\"items\": {\"properties\": {\"type\": {\"not\": {\"const\":"
    " \"excerpt\"}}}}}}}}\n"
    "        ]\n"
    "      }\n"
    "    }\n"
    "  },\n"
    "  \"$defs\": {\n"
    "    \"draft\": {\n"
    "      \"type\": \"object\",\n"
    "      \"additionalProperties\": false,\n"
    "      \"required\": [\"model\", \"weights\", \"sha256\", \"edited\"],\n"
    "      \"description\": \"an assistant message that started as a reply drafted by a WPT model, which a"
    " person then confirmed, edited or not (docs/chat-format.md, \\\"Model drafts\\\")\",\n"
    "      \"properties\": {\n"
    "        \"model\": {\"type\": \"string\", \"minLength\": 1, \"description\": \"the name of the model folder"
    " that drafted it\"},\n"
    "        \"weights\": {\"type\": \"object\", \"minProperties\": 1, \"description\": \"the sha256 of each of"
    " that folder's weight files, by file name\",\n"
    "                    \"additionalProperties\": {\"type\": \"string\", \"pattern\": \"^[0-9a-f]{64}$\"}},\n"
    "        \"sha256\": {\"type\": \"string\", \"pattern\": \"^[0-9a-f]{64}$\", \"description\": \"the sha256 of the"
    " drafted content as the model wrote it, of its JSON text written as docs/chat-format.md, Model"
    " drafts, says\"},\n"
    "        \"edited\": {\"type\": \"boolean\", \"description\": \"true when the person changed the draft before"
    " confirming it: the content no longer gives sha256\"}\n"
    "      }\n"
    "    },\n"
    "    \"block\": {\n"
    "      \"type\": \"object\",\n"
    "      \"required\": [\"type\"],\n"
    "      \"properties\": {\"type\": {\"enum\": [\"text\", \"thinking\", \"tool_call\", \"tool_result\", \"json\","
    " \"file\", \"excerpt\"]}},\n"
    "      \"allOf\": [\n"
    "        {\"if\": {\"properties\": {\"type\": {\"enum\": [\"text\", \"thinking\"]}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"text\"],\n"
    "                  \"properties\": {\"type\": true, \"text\": {\"type\": \"string\"}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"tool_call\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"id\", \"name\", \"arguments\"],\n"
    "                  \"properties\": {\"type\": true, \"id\": {\"type\": \"string\", \"minLength\": 1},\n"
    "                                 \"name\": {\"type\": \"string\", \"minLength\": 1}, \"arguments\": {\"type\":"
    " \"object\"}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"tool_result\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"call\", \"content\"],\n"
    "                  \"properties\": {\"type\": true, \"call\": {\"type\": \"string\", \"minLength\": 1}, \"error\":"
    " {\"type\": \"boolean\"},\n"
    "                                 \"content\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/$defs/block\",\n"
    "                                             \"properties\": {\"type\": {\"enum\": [\"text\", \"json\","
    " \"file\", \"excerpt\"]}}}}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"json\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"value\"],\n"
    "                  \"properties\": {\"type\": true, \"value\": true}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"file\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"sha256\", \"media_type\"],\n"
    "                  \"properties\": {\"type\": true, \"sha256\": {\"type\": \"string\", \"pattern\":"
    " \"^[0-9a-f]{64}$\"},\n"
    "                                 \"media_type\": {\"type\": \"string\", \"pattern\":"
    " \"^[a-z]+/[A-Za-z0-9.+-]+$\"}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"excerpt\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"source\", \"selector\"],\n"
    "                  \"description\": \"a passage of a plain text of the depot, quoted by its content"
    " (docs/chat-format.md, \\\"Excerpts\\\"): the corpus build writes the passage in its place\",\n"
    "                  \"properties\": {\"type\": true, \"source\": {\"type\": \"string\", \"minLength\": 1,"
    " \"description\": \"the id of the quoted text\"},\n"
    "                                 \"selector\": {\"type\": \"object\", \"additionalProperties\": false,"
    " \"required\": [\"type\", \"exact\"],\n"
    "                                              \"description\": \"a W3C Web Annotation"
    " TextQuoteSelector (https://www.w3.org/TR/annotation-model/#text-quote-selector)\",\n"
    "                                              \"properties\": {\"type\": {\"const\":"
    " \"TextQuoteSelector\"},\n"
    "                                                             \"exact\": {\"type\": \"string\","
    " \"description\": \"the passage, character for character\"},\n"
    "                                                             \"prefix\": {\"type\": \"string\","
    " \"description\": \"the characters right before it\"},\n"
    "                                                             \"suffix\": {\"type\": \"string\","
    " \"description\": \"the characters right after it\"}}}}}}\n"
    "      ]\n"
    "    }\n"
    "  }\n"
    "}\n"
)
NULLABLE_12, NOT_NULL_12 = "kind TEXT,", "kind TEXT NOT NULL,"
LONG_12 = ("CREATE INDEX IF NOT EXISTS content_lists_long ON content_lists (id, version) "
           "WHERE kind = 'plain' AND length > 1048576")
# format 12: the counts of what each library holds by kind of format 11 that format 12 counts again (every one but
# those of the plain texts), and what the current content versions of a batch of rowids of content_lists that are not
# plain add to each library that holds their text, by kind
GROUPS_12 = (f"DELETE FROM counts WHERE facet = '{LIBRARY_KIND_9}' AND substr(name, instr(name, ' ') + 1) <> "
             "'plain'")
KINDS_12 = (f"INSERT INTO counts (facet, name, texts, characters, bytes) SELECT '{LIBRARY_KIND_9}', "
            "m.library || ' ' || c.kind, count(*), sum(c.length), sum(c.bytes) FROM content_lists c "
            "CROSS JOIN texts_revisions t ON t.id = c.id AND t.latest AND t.content_version = c.version "
            "CROSS JOIN memberships m INDEXED BY memberships_held ON m.text = c.id AND m.left_change IS NULL "
            "WHERE c.rowid > ? AND c.rowid <= ? AND c.kind <> 'plain' GROUP BY m.library, c.kind "
            "ON CONFLICT (facet, name) DO UPDATE SET texts = texts + excluded.texts, "
            "characters = characters + excluded.characters, bytes = bytes + excluded.bytes")
# format 12: the same, for every content version at once, read without a lookup per row of the file when there are
# many: the content versions of a kind (every one but plain, of no kind in format 11) copied into a table of the
# writer's temporary database in one sequential scan of content_lists (ITEMS_TABLE_12); then, when they are more than
# one revision of a text in ITEMS_SHARE_12, those a text's current revision reads among them found in one sequential
# scan of texts_revisions and the memberships that have not ended in one of memberships (each looking rows up in the
# temporary tables only; SCANNED_12), else each looked up through the index of the current revisions and
# memberships_held (LOOKED_UP_12), which reads fewer pages; what the counts of a library's kinds add up to is KINDS_12's
# over every batch
ITEMS_TABLE_12 = ("CREATE TEMP TABLE upgrade_items (id TEXT NOT NULL, version INTEGER NOT NULL, kind TEXT NOT NULL, "
                  "length INTEGER, bytes INTEGER, PRIMARY KEY (id, version)) WITHOUT ROWID",
                  "INSERT INTO temp.upgrade_items (id, version, kind, length, bytes) SELECT id, version, kind, length, "
                  "bytes FROM content_lists NOT INDEXED WHERE kind IS NOT NULL AND kind <> 'plain' "
                  "ORDER BY +id, +version",
                  "CREATE TEMP TABLE upgrade_current_items (id TEXT NOT NULL PRIMARY KEY, kind TEXT NOT NULL, "
                  "length INTEGER, bytes INTEGER) WITHOUT ROWID")
ITEMS_SHARE_12 = 40  # bench/large.py's depot: 86 microseconds an item looked up, 2.2 a revision scanned
_ITEM_COUNTS_12 = (f"INSERT INTO counts (facet, name, texts, characters, bytes) SELECT '{LIBRARY_KIND_9}', "
                   "m.library || ' ' || c.kind, count(*), sum(c.length), sum(c.bytes) FROM {joined} "
                   "WHERE m.left_change IS NULL GROUP BY m.library, c.kind ON CONFLICT (facet, name) DO UPDATE SET "
                   "texts = texts + excluded.texts, characters = characters + excluded.characters, "
                   "bytes = bytes + excluded.bytes")
SCANNED_12 = ("INSERT INTO temp.upgrade_current_items (id, kind, length, bytes) SELECT t.id, i.kind, i.length, "
              "i.bytes FROM texts_revisions t NOT INDEXED CROSS JOIN temp.upgrade_items i ON i.id = t.id "
              "AND i.version = t.content_version WHERE t.latest",
              _ITEM_COUNTS_12.format(joined="memberships m NOT INDEXED CROSS JOIN temp.upgrade_current_items c "
                                            "ON c.id = m.text"))
LOOKED_UP_12 = ("INSERT INTO temp.upgrade_current_items (id, kind, length, bytes) SELECT i.id, i.kind, i.length, "
                "i.bytes FROM temp.upgrade_items i CROSS JOIN texts_revisions t ON t.id = i.id AND t.latest "
                "AND t.content_version = i.version",
                _ITEM_COUNTS_12.format(joined="temp.upgrade_current_items c CROSS JOIN memberships m INDEXED BY "
                                              "memberships_held ON m.text = c.id"))
ITEMS_DROPPED_12 = ("DROP TABLE temp.upgrade_current_items", "DROP TABLE temp.upgrade_items")
# format 12: the table of the texts each content version of a conversation quotes, and its index by quoted text
QUOTES_12 = ("CREATE TABLE IF NOT EXISTS quotes (conversation TEXT NOT NULL, version INTEGER NOT NULL, "
             "text TEXT NOT NULL, PRIMARY KEY (conversation, version, text)) WITHOUT ROWID",
             "CREATE INDEX IF NOT EXISTS quotes_text ON quotes (text)")
UNDER_WAY = "upgrade_under_way"  # the table that marks a depot whose upgrade has not ended, and how far it went
BATCH = 250_000  # rows a transaction of the upgrades to formats 6 (texts_revisions), 7 (memberships, pins) and 9
                 # (memberships) goes over
SAY_EVERY = 30  # seconds between two lines of progress
# each format 3 table, by its folder's name: the table of format 4 its rows go into
TARGETS = {"about": "about_revisions", "licenses": "licenses", "notices": "notices", "kinds": "kinds",
           "decisions": "decisions", "texts": "texts_revisions", "contents": "content_lists", "parts": "parts",
           "libraries": "libraries_revisions", "memberships": "memberships", "pins": "pins",
           "erasures": "erasures_revisions", "datasets": "datasets_revisions", "pulls": "pulls"}
LISTS = {"licenses", "authors", "tags", "decisions", "scores", "problems", "needles", "shares", "libraries"}
_ID = re.compile(r'"([A-Za-z0-9][A-Za-z0-9._-]*-[0-9]+)"')


def upgrade(address: str, to: str | None = None, upstream: str | None = None, log=say,
            records: list[dict] | None = None, public: str | None = None, checked: "Check | None" = None):
    """(the depot at address brought up to this code's format, the format it
    had). A depot of the code's format is left as it is. upstream: the web
    address of the upstream's web app, for a clone. log: where a step says
    what it does. records: the records of the corpora of the dock
    (the records under corpora/, which `depot upgrade` reads), whose dataset revisions the step to format
    13 makes snapshots of; public: the depot's public address, which the
    order data sections that step says name. `depot upgrade` runs check
    first and acts only when it passes; checked: that check, whose measures
    of the snapshots the step to format 13 writes as they are when the depot
    did not change since (_to_13). OtherFormat for a newer format, which no code brings down;
    Refused for a folder whose journal holds a change never finished, for a
    clone without upstream and for upstream given to a depot that follows
    none; DepotExists when the new file's path holds a file already."""
    from .depot import Depot, format_of, location, other_format
    path = location(address)
    start = format_of(address)
    if start > FORMAT:
        raise OtherFormat(other_format(str(path), start))
    if not path.is_dir():
        if upstream is not None:
            raise Refused(f"{path} is a depot file already: --upstream is for the upgrade of a clone of format 3 "
                          "or older")
        db = Database(path)
        try:
            start = _format(db)  # a depot whose upgrade has not ended: the format it started from
        finally:
            db.close()
        if start < FORMAT:
            _alone(path)
    else:
        path = _from_folder(path, to, upstream)
    _in_place(path, log, records, public, checked)
    saved_check_path(path).unlink(missing_ok=True)
    return Depot.open(str(path)), start


def _alone(path: Path):
    """Refused while any other connection, of this process or another, has
    the depot file at path open: one connection asks for an exclusive lock
    in SQLite's exclusive locking mode, which every open connection of a
    file in WAL mode keeps from being granted once it has read the file,
    then lets it go. A depot file is in WAL mode (Database.create); in a
    file in another journal mode an idle connection holds no lock, so the
    check does not see it, and the runbook's word to stop every process is
    the only guard. A process that opens the file later finds a format it
    refuses, since the first transaction of a step writes the new format."""
    con = apsw.Connection(str(path))
    try:
        con.set_busy_timeout(ALONE_WAIT_MS)
        con.execute("PRAGMA locking_mode = EXCLUSIVE").fetchall()
        try:
            con.execute("BEGIN EXCLUSIVE")
        except (apsw.BusyError, apsw.LockedError):
            raise Refused(f"{path}: another process has the depot open (`depot web`, a job, a backup or another "
                          "`depot` command): stop it, then upgrade; nothing was written") from None
        con.execute("ROLLBACK")
    finally:
        con.close()


def _from_folder(path: Path, to: str | None, upstream: str | None) -> Path:
    """The SQLite file of format 4 written from the folder at path, as the module says."""
    from .depot import location
    legacy = DeltaFolder(path)
    undone = legacy.undone()
    if undone:
        raise Refused(f"{path}: its journal holds {len(undone)} changes never finished (the first: {undone[0]}); "
                      "finish them with the wpt of the depot's format, by opening the depot once, then upgrade")
    followed = max(legacy.read("about").to_pylist(), key=lambda one: one["revision"]).get("upstream")
    if followed is None and upstream is not None:
        raise Refused(f"{path} follows no upstream: --upstream is for the upgrade of a clone")
    if followed is not None:
        if upstream is None:
            raise Refused(f"{path} is a clone of {followed}: a clone of format {FORMAT} follows its upstream over "
                          "the upstream's web app, so give its web address (--upstream https://...)")
        upstream = Upstream(upstream).url
    target = location(to) if to is not None else path.with_name(path.name + ".sqlite")
    if target.exists():
        raise DepotExists(f"{target} already exists: the upgrade writes a new file")
    while (reached := legacy.format()) < LAST:
        STEPS[reached](legacy)
        legacy.record_format(reached + 1)
    _write(legacy, target, upstream)
    return target


def _in_place(path: Path, log, records: list[dict] | None = None, public: str | None = None,
              checked: "Check | None" = None):
    """The SQLite file at path brought up to the code's format, a step at a time, as the module says. The file is
    put in WAL mode first, as Database.create makes every depot file (a copy made by another tool may not be): a step
    reads one scan while it writes what it found in transactions of its own, which a file in another journal mode
    would keep waiting."""
    wal = apsw.Connection(str(path))
    try:
        wal.execute("PRAGMA journal_mode = WAL").fetchall()
    finally:
        wal.close()
    db = Database(path)
    try:
        while (reached := _format(db)) < FORMAT:
            db.settle()
            db.mark_full_copy()
            if reached == 12:
                FILE_STEPS[reached](db, log, records, public, checked)
            else:
                FILE_STEPS[reached](db, log)
    finally:
        db.close()


def _format(db: Database) -> int:
    """The depot's format, or for a depot under an upgrade that has not
    ended, the format that upgrade starts from."""
    from .depot import half_upgraded
    con = db.reader()
    found = con.execute("SELECT format FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]
    return found - 1 if half_upgraded(con) else found


def _to_5(db: Database, log):
    """Format 4 to 5: the index texts_order."""
    texts = db.reader().execute("SELECT count(*) FROM texts_revisions WHERE latest").fetchall()[0][0]
    log(f"{db.path}: building the index texts_order over its {texts} current texts, in one statement that reads "
        "each of their rows once and writes no row of a table")
    began = time.monotonic()
    with db.bulk() as con:
        con.execute(TEXTS_ORDER)
        _new_format(con, 5)
    log(f"{db.path}: the index texts_order is built, in {time.monotonic() - began:.0f} seconds; the depot is at "
        "format 5")


def _new_format(con, reached: int):
    """The upgrade's own change: a new revision of the about row at the format reached."""
    number = con.execute("SELECT coalesce(max(id), 0) + 1 FROM changes").fetchall()[0][0]
    con.execute("INSERT INTO changes (id, kind, made_at) VALUES (?, 'upgrade', ?)", (number, now()))
    con.execute("INSERT INTO about_revisions (revision, change, name, maintainer, takedown_contact, upstream, format) "
                "SELECT revision + 1, ?, name, maintainer, takedown_contact, upstream, ? FROM about_revisions "
                "ORDER BY revision DESC LIMIT 1", (number, reached))


def _to_6(db: Database, log):
    """Format 5 to 6, in several transactions, as the module says: each
    text's decisions as rows of text_decisions, its stored status, the
    index texts_status and the counts."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('texts', 0)")
            for statement in TABLES_6:
                con.execute(statement)
            for table, column, declared in COLUMNS_6:
                if column not in {one[1] for one in con.execute(f"PRAGMA table_info({table})")}:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declared}")
            _new_format(con, 6)
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'texts'").fetchall()[0][0]
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    log(f"{db.path}: writing the decisions and the status of each current text, rows {after} to {last} of "
        f"texts_revisions, {BATCH} rows a transaction; run again after a stop, it goes on from the last one")
    began = said = time.monotonic()
    while after < last:
        upto = min(after + BATCH, last)
        with db.bulk() as con:
            con.execute("INSERT OR IGNORE INTO text_decisions (text, decision, change) "
                        "SELECT t.id, j.value, d.change FROM texts_revisions t, json_each(t.decisions) j "
                        "JOIN decisions d ON d.id = j.value WHERE t.latest AND t.rowid > ? AND t.rowid <= ?",
                        (after, upto))
            con.execute(f"UPDATE texts_revisions SET status = {policy.status_sql('decisions')}, decisions = '[]' "
                        "WHERE latest AND rowid > ? AND rowid <= ?", (after, upto))
            con.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'texts'", (upto,))
        after = upto
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: {after} of {last} rows of texts_revisions, {said - began:.0f} seconds")
    log(f"{db.path}: building the index texts_status over the current texts, in one statement")
    with db.bulk() as con:
        con.execute(TEXTS_STATUS)
    log(f"{db.path}: counting the texts of each status, tag, license, library and decision")
    with db.bulk() as con:
        _recount_6(con)
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the depot is at format 6, in {time.monotonic() - began:.0f} seconds")


def _recount_6(con):
    """The counts of format 6 made from the depot's rows (COUNTS_6)."""
    con.execute("DELETE FROM counts")
    for facet, query in COUNTS_6.items():
        con.execute(f"INSERT INTO counts (facet, name, texts) SELECT '{facet}', * FROM ({query})")
    con.execute("DELETE FROM counts WHERE texts = 0")


def _to_7(db: Database, log):
    """Format 6 to 7, in several transactions, as the module says: the
    export key of every row of memberships and pins, then their indexes."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.executemany(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES (?, 0)", [(t,) for t, _ in KEYED_7])
            for table, column, declared in COLUMNS_7:
                if column not in {one[1] for one in con.execute(f"PRAGMA table_info({table})")}:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declared}")
            _new_format(con, 7)
    began = time.monotonic()
    for table, text in KEYED_7:
        _keys(db, table, text, log)
    for name, statement in INDEXES_7:
        log(f"{db.path}: building the index {name}, in one statement")
        indexed = time.monotonic()
        with db.bulk() as con:
            con.execute(statement)
        log(f"{db.path}: the index {name} is built, in {time.monotonic() - indexed:.0f} seconds")
    with db.bulk() as con:
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the depot is at format 7, in {time.monotonic() - began:.0f} seconds")


def _to_8(db: Database, log):
    """Format 7 to 8, as the module says: the index text_decisions_decision,
    built while the depot is marked half upgraded."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            _new_format(con, 8)
    # the count kept in counts, since counting the table itself reads every row as the build does
    rows = db.reader().execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'decision'").fetchall()[0][0]
    log(f"{db.path}: building the index text_decisions_decision over its {rows} rows of text_decisions, in one "
        "statement that reads each row once and writes no row of a table; run again after a stop, it builds it again")
    began = time.monotonic()
    with db.bulk() as con:
        con.execute(TEXT_DECISIONS_DECISION)
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the index text_decisions_decision is built, in {time.monotonic() - began:.0f} seconds; the "
        "depot is at format 8")


def _keys(db: Database, table: str, text: str, log):
    """Write the export key of each row of table, from the text its column
    text names, BATCH rows a transaction from the rowid upgrade_under_way
    keeps for it, which each transaction moves on."""
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = ?", (table,)).fetchall()[0][0]
    last = con.execute(f"SELECT coalesce(max(rowid), 0) FROM {table}").fetchall()[0][0]
    if after >= last:
        return
    log(f"{db.path}: writing the export key of each row of {table}, rows {after} to {last}, {BATCH} rows a "
        "transaction; run again after a stop, it goes on from the last one")
    began = said = time.monotonic()
    written = 0
    while after < last:
        upto = min(after + BATCH, last)
        found = db.reader().execute(f"SELECT rowid, {text} FROM {table} WHERE rowid > ? AND rowid <= ?",
                                    (after, upto)).fetchall()
        with db.bulk() as con:
            con.executemany(f"UPDATE {table} SET export_key = ? WHERE rowid = ?",
                            [(ids.export_key(one), rowid) for rowid, one in found])
            con.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = ?", (upto, table))
        after, written = upto, written + len(found)
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: row {after} of {last} of {table}, {written / max(said - began, 1e-9):,.0f} rows "
                f"a second, {said - began:.0f} seconds")


def _to_9(db: Database, log):
    """Format 8 to 9, in several transactions, as the module says: the
    columns characters and bytes of counts, and the counts of what each
    library holds by kind and by tag, from one sorted merge of three
    sequential scans (_merged_9)."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after TEXT)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('memberships', NULL)")
            held = {one[1] for one in con.execute("PRAGMA table_info(counts)")}
            for column in COLUMNS_9:
                if column not in held:
                    con.execute(f"ALTER TABLE counts ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            if "figures" not in {one[1] for one in con.execute("PRAGMA table_info(datasets_revisions)")}:
                con.execute("ALTER TABLE datasets_revisions ADD COLUMN figures TEXT")
            con.execute(MOVES_9)
            con.execute("DELETE FROM library_moves")
            con.execute(f"DELETE FROM counts WHERE facet IN {FACETS_9}")
            _new_format(con, 9)
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'memberships'").fetchall()[0][0]
    total = con.execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'library'").fetchall()[0][0]
    log(f"{db.path}: counting what each library holds by kind and by tag, in characters and bytes: its {total:,} "
        "memberships that have not ended, the current revisions and the content versions, each read in one "
        "sequential scan sorted by text id, then merged, the sums written about every "
        f"{BATCH:,} memberships. SQLite sorts the three scans at once in its temporary folder (SQLITE_TMPDIR, "
        f"else /var/tmp, /usr/tmp or /tmp), which needs about {total * SORTED_9 / 1e9:,.1f} GB of free disk"
        + ("" if after is None else f"; going on after the text {after}, where the last run stopped"))
    began = said = time.monotonic()
    params = () if after is None else (after,)
    read = 0
    for sums, last, revised in _merged_9(*(con.execute(scan.format(after="" if after is None else f" AND {column} > ?"),
                                                       params) for scan, column in zip(SCANS_9, SCANNED_9))):
        with db.bulk() as written:
            written.executemany("INSERT INTO counts (facet, name, texts, characters, bytes) VALUES (?, ?, ?, ?, ?) "
                                "ON CONFLICT (facet, name) DO UPDATE SET texts = texts + excluded.texts, "
                                "characters = characters + excluded.characters, bytes = bytes + excluded.bytes",
                                [(facet, name, *by) for (facet, name), by in sums.items()])
            _moved_9(written, revised.items())
            written.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'memberships'", (last,))
        read += sum(by[0] for (facet, _), by in sums.items() if facet == LIBRARY_KIND_9)
        if time.monotonic() - said >= SAY_EVERY:
            said = time.monotonic()
            log(f"{db.path}: {read:,} memberships counted of {total:,}, the last text {last}, "
                f"{read / max(said - began, 1e-9):,.0f} rows a second, {said - began:.0f} seconds")
    counted = time.monotonic()
    log(f"{db.path}: {read:,} memberships counted, {read / max(counted - began, 1e-9):,.0f} rows a second, in "
        f"{counted - began:.0f} seconds; reading the latest change each library's memberships joined or left at "
        "from the index memberships_key, and the figures of each dataset revision the counts prove exact")
    with db.bulk() as written:
        _moved_9(written, written.execute(JOINED_OR_LEFT_9).fetchall())
        if written.execute("SELECT upstream FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]:
            written.execute("UPDATE library_moves SET change = NULL")  # its rows carry several numberings
        figured = _figures_9(written)
        written.execute(f"DROP TABLE {UNDER_WAY}")
    took = time.monotonic() - began
    log(f"{db.path}: {figured[0]:,} of {figured[1]:,} libraries read by a dataset revision have figures, in "
        f"{time.monotonic() - counted:.0f} seconds; the depot is at format 9, in {took:.0f} seconds")


def _moved_9(con, found):
    """library_moves raised to each (library, change) of found where it is lower, or made."""
    con.executemany("INSERT INTO library_moves (library, change) VALUES (?, ?) ON CONFLICT (library) "
                    "DO UPDATE SET change = max(change, excluded.change)", list(found))


def _figures_9(con) -> tuple[int, int]:
    """The figures of every dataset revision (the layout of format 9, as
    depot.schema says), read at the upgrade's change from the counts
    of format 9: those of a library this depot owns, read here, whose row
    of library_moves names a change at or before the change it is read at
    (the change that locked it, for a library locked then), or that has no
    row; null for any other. (libraries given figures, libraries read)."""
    name, at = con.execute("SELECT name, change FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0]
    given = read = 0
    for dataset, revision, libraries, shares in con.execute(
            "SELECT id, revision, libraries, shares FROM datasets_revisions").fetchall():
        tags = [one["tag"] for one in json.loads(shares or "[]")]
        found = []
        for ref in json.loads(libraries):
            read += 1
            library, change = ref["library"], ref["change"]
            if ref.get("depot") is not None or ids.owner(library) != name:
                found.append(None)
                continue
            state = con.execute("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                                "ORDER BY revision DESC LIMIT 1", (library, change)).fetchall()
            if state and state[0][0] is not None:
                change = con.execute("SELECT min(change) FROM libraries_revisions WHERE id = ? "
                                     "AND locked_at IS NOT NULL", (library,)).fetchall()[0][0]
            moved = con.execute("SELECT change FROM library_moves WHERE library = ?", (library,)).fetchall()
            if moved and (moved[0][0] is None or moved[0][0] > change):
                found.append(None)
                continue
            names = ([(LIBRARY_KIND_9, f"{library} {kind}") for kind in KINDS_9]
                     + [(LIBRARY_TAG_9, f"{library} {tag}") for tag in tags])
            sized = []
            for facet, row_name in names:
                row = con.execute("SELECT texts, characters, bytes FROM counts WHERE facet = ? AND name = ?",
                                  (facet, row_name)).fetchall()
                sized.append(dict(zip(("texts", "characters", "bytes"), row[0] if row else (0, 0, 0))))
            found.append(dict(zip(KINDS_9, sized)) | {"tags": sized[len(KINDS_9):]})
            given += 1
        con.execute("UPDATE datasets_revisions SET figures = ? WHERE id = ? AND revision = ?",
                    (json.dumps({"at": at, "libraries": found}, ensure_ascii=False, separators=(",", ":")),
                     dataset, revision))
    return given, read


def _merged_9(held, current, versions):
    """What the memberships held adds to the counts of format 9's new
    facets, from three streams in the order of the texts' ids (SCANS_9):
    held, (text, library) of each membership that has not ended; current,
    (id, content version, tags as JSON text, change) of each current
    revision; versions, (id, version, kind, length, bytes) of each content
    version. It goes through the three at once, as one sorted merge, and
    gives (sums: {(facet, name): [texts, characters, bytes]}, the id of the
    last text they cover, revised: {library: the latest change a text it
    holds among them was revised at}) each time about BATCH memberships are
    summed, at the end of a text, then the rest. Each membership adds to `<library>
    <kind>` (plain, items: ITEMS_9, other) and, for a plain text, to
    `<library> <tag>` for each of its tags; one of a text with no current
    revision or no content version counts for nothing."""
    sums, summed, seen, size, revised, change = {}, 0, None, None, {}, None
    text_row, version_row = next(current, None), next(versions, None)
    for text, library in held:
        if text != seen:
            if summed >= BATCH:
                yield sums, seen, revised
                sums, summed, revised = {}, 0, {}
            seen, size, change = text, None, None
            while text_row is not None and text_row[0] < text:
                text_row = next(current, None)
            if text_row is not None and text_row[0] == text:
                change = text_row[3]
                wanted = (text, text_row[1])
                while version_row is not None and (version_row[0], version_row[1]) < wanted:
                    version_row = next(versions, None)
                if version_row is not None and (version_row[0], version_row[1]) == wanted:
                    kind = version_row[2]
                    plain = kind is None or kind == PLAIN_9
                    what = "plain" if plain else "items" if kind in ITEMS_9 else "other"
                    size = (what, version_row[3] or 0, version_row[4] or 0,
                            json.loads(text_row[2] or "[]") if plain else ())
        summed += 1
        if change is not None and revised.get(library, -1) < change:
            revised[library] = change
        if size is None:
            continue
        what, length, nbytes, tags = size
        for key in ((LIBRARY_KIND_9, f"{library} {what}"), *((LIBRARY_TAG_9, f"{library} {tag}") for tag in tags)):
            one = sums.get(key)
            if one is None:
                sums[key] = [1, length, nbytes]
            else:
                one[0] += 1
                one[1] += length
                one[2] += nbytes
    if sums or seen is not None:
        yield sums, seen, revised


def _to_10(db: Database, log):
    """Format 9 to 10, in several transactions, as the module says: the
    listings text_tags and text_licenses, the counts of score names and the
    splits, from the current revisions read in rowid order, BATCH rowids a
    transaction; then each index of INDEXES_10, one statement each; then
    the index text_decisions_decision dropped and the counts of pins whose
    text has a newer revision."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('texts', 0)")
            for statement in TABLES_10 + TITLES_10:
                con.execute(statement)
            for table, _, _ in LISTED_10:
                con.execute(f"DELETE FROM {table}")
            con.execute("INSERT INTO text_titles (text_titles) VALUES ('delete-all')")
            con.execute("DELETE FROM title_owners")
            con.execute(f"DELETE FROM counts WHERE facet IN ('{SCORE_10}', '{NEWER_10}', "
                        + ", ".join(f"'{facet}'" for facet, _ in SPLITS_10) + ")")
            _new_format(con, 10)
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'texts'").fetchall()[0][0]
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    total = con.execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'texts'").fetchall()[0][0]
    log(f"{db.path}: listing the {total:,} current texts by tag and by license, indexing their titles and counting "
        f"their score names, "
        f"from the revisions read in rowid order, rows {after:,} to {last:,}, {BATCH:,} rowids a transaction"
        + ("" if after == 0 else "; going on after the last transaction the last run wrote"))
    began = said = time.monotonic()
    first, titled = after, 0.0
    while after < last:
        upto = min(after + BATCH, last)
        bounds = {"low": after, "high": upto}
        with db.bulk() as written:
            for table, name, column in LISTED_10:
                written.execute(f"INSERT OR IGNORE INTO {table} ({name}, key) SELECT j.value, {ORDER.replace('id', 't.id')} "
                                f"{BATCH_ROWS_10.replace('WHERE', f', json_each(t.{column}) j WHERE')}", bounds)
            added = [(SCORE_10, name, count) for name, count in written.execute(SCORES_10, bounds)]
            for facet, query in SPLITS_10:
                added += [(facet, name, count) for name, count in written.execute(query, bounds)]
            written.executemany("INSERT INTO counts (facet, name, texts) VALUES (?, ?, ?) ON CONFLICT (facet, name) "
                                "DO UPDATE SET texts = texts + excluded.texts", added)
            indexing = time.monotonic()
            odd = written.execute(ODD_TITLE_10, bounds).fetchall()
            if odd:
                raise Refused(f"the title index numbers a text by its depot's name, a hyphen and its number, which "
                              f"{odd[0][0]} is not: the upgrade stops before this batch, the depot half upgraded, "
                              "which every other depot command refuses; give that text such an id, or no title, "
                              "with SQL on the depot's file through the sqlite3 program, then run depot upgrade "
                              "again, which goes on with this batch")
            written.execute(OWNERS_10, bounds)
            written.execute(TITLE_ROWS_10, bounds)
            titled += time.monotonic() - indexing
            written.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'texts'", (upto,))
        after = upto
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: revision row {after:,} of {last:,}, {(after - first) / max(said - began, 1e-9):,.0f} rows a "
                f"second, {said - began:.0f} seconds")
    listed = time.monotonic()
    for name, statement in INDEXES_10:
        if db.reader().execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)).fetchall():
            continue
        log(f"{db.path}: building the index {name}, in one statement that reads its table once; run again after a "
            "stop, it builds it again")
        indexed = time.monotonic()
        with db.bulk() as written:
            written.execute(statement)
        log(f"{db.path}: the index {name} is built, in {time.monotonic() - indexed:.0f} seconds")
    with db.bulk() as written:
        written.execute(DROPPED_10)
        written.execute(f"DELETE FROM counts WHERE facet = '{NEWER_10}'")
        written.execute(NEWER_COUNTS_10)
        written.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the listings took {listed - began:.0f} seconds, {titled:.0f} of them the title index, and the "
        f"indexes {time.monotonic() - listed:.0f}; "
        f"the depot is at format 10, in {time.monotonic() - began:.0f} seconds")


def _to_11(db: Database, log):
    """Format 10 to 11, as the module says: the index content_lists_long,
    built while the depot is marked half upgraded."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            _new_format(con, 11)
    log(f"{db.path}: building the index content_lists_long of the plain content versions longer than 1,048,576 "
        "characters, in one statement that reads each row of content_lists once and writes no row of a table; run "
        "again after a stop, it builds it again")
    began = time.monotonic()
    with db.bulk() as con:
        con.execute(LONG_11)
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the index content_lists_long is built, in {time.monotonic() - began:.0f} seconds; the depot is "
        "at format 11")


def _to_12(db: Database, log):
    """Format 11 to 12, in several transactions, as the module says: the
    conversation kind's schema and the table quotes, the kind plain on
    every content version of no kind and the quotes of every conversation,
    then the column kind declared NOT NULL and the index content_lists_long
    built again."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('contents', 0)")
            con.execute(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES ('kinds', 0)")
            con.execute("DROP INDEX IF EXISTS content_lists_long")
            con.execute("INSERT OR IGNORE INTO kinds (id, meaning, schema) VALUES (?, ?, ?)", PLAIN_12)
            con.execute("UPDATE kinds SET schema = ? WHERE id = 'conversation'", (CONVERSATION_12,))
            con.execute(GROUPS_12)
            for statement in QUOTES_12:
                con.execute(statement)
            _new_format(con, 12)
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'contents'").fetchall()[0][0]
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM content_lists").fetchall()[0][0]
    # a run of the code before ITEMS_12 stopped half way counts the kinds of each batch as it goes on (KINDS_12)
    each_batch = not con.execute(f"SELECT 1 FROM {UNDER_WAY} WHERE stage = 'kinds'").fetchall() and after > 0
    started = began = said = time.monotonic()
    if con.execute(f"SELECT 1 FROM {UNDER_WAY} WHERE stage = 'kinds'").fetchall():
        log(f"{db.path}: counting what each library holds of each kind but plain, in one transaction that reads "
            "content_lists, texts_revisions and memberships once each, in the order of the file; run again after a "
            "stop, it does it again")
        with db.bulk() as written:
            for statement in ITEMS_TABLE_12:
                written.execute(statement)
            items = written.execute("SELECT count(*) FROM temp.upgrade_items").fetchall()[0][0]
            revisions = written.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
            for statement in SCANNED_12 if items * ITEMS_SHARE_12 > revisions else LOOKED_UP_12:
                written.execute(statement)
            for statement in ITEMS_DROPPED_12:
                written.execute(statement)
            written.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'kinds'")
        log(f"{db.path}: the kinds counted in {time.monotonic() - began:.0f} seconds")
    started, began = began, time.monotonic()
    said = began
    log(f"{db.path}: writing the kind plain on every content version of no kind and the texts each conversation "
        f"quotes, rows {after:,} to {last:,} of content_lists, {BATCH:,} rowids a transaction"
        + ("" if after == 0 else "; going on after the last transaction the last run wrote"))
    first = after
    while after < last:
        upto = min(after + BATCH, last)
        with db.bulk() as written:
            written.execute(f"UPDATE content_lists SET kind = '{PLAIN_12[0]}' WHERE rowid > ? AND rowid <= ? "
                            "AND kind IS NULL", (after, upto))
            if each_batch:
                written.execute(KINDS_12, (after, upto))
            _quotes_12(written, after, upto)
            written.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'contents'", (upto,))
        after = upto
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: content row {after:,} of {last:,}, {(after - first) / max(said - began, 1e-9):,.0f} rows "
                f"a second, {said - began:.0f} seconds")
    log(f"{db.path}: checking that every content version has a kind, declaring the column kind NOT NULL, "
        "building the index content_lists_long of the plain content versions longer than 1,048,576 characters and "
        "giving the dataset revisions' figures a count of each kind, in one transaction that reads content_lists "
        "twice; run again after a stop, it does it again")
    finished = time.monotonic()
    with db.bulk() as written:
        left = written.execute("SELECT rowid FROM content_lists WHERE kind IS NULL LIMIT 1").fetchall()
        if left:
            raise Refused(f"the content version of rowid {left[0][0]} of content_lists has no kind after every batch "
                          "wrote one: the upgrade stops, the depot half upgraded; run depot upgrade again")
        statement = written.execute("SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'content_lists'"
                                    ).fetchall()[0][0]
        if NOT_NULL_12 not in statement:
            if statement.count(NULLABLE_12) != 1:
                raise Refused(f"content_lists is declared as {statement!r}, where format 11 declares the column "
                              f"kind once as {NULLABLE_12!r}: the upgrade stops, the depot half upgraded")
            version = written.execute("PRAGMA schema_version").fetchall()[0][0]
            written.execute("PRAGMA writable_schema = ON")
            try:
                written.execute("UPDATE sqlite_schema SET sql = ? WHERE type = 'table' AND name = 'content_lists'",
                                (statement.replace(NULLABLE_12, NOT_NULL_12),))
                written.execute(f"PRAGMA schema_version = {int(version) + 1}")
            finally:
                written.execute("PRAGMA writable_schema = OFF")
        written.execute(LONG_12)
        figured = _figures_12(written)
        written.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the kinds took {finished - began:.0f} seconds and the last transaction "
        f"{time.monotonic() - finished:.0f}; {figured[0]:,} of {figured[1]:,} libraries with figures keep them; the "
        f"depot is at format 12, in {time.monotonic() - started:.0f} seconds")


def _quotes_12(con, after: int, upto: int):
    """Write the rows of quotes of the content versions of conversations
    among the rowids after..upto of content_lists: the ids each one's
    excerpts quote (blocks of type excerpt in its messages' content and in
    their tool results' content, each naming its text by its source), read
    from its parts."""
    reader = Reader(lambda: con)
    for text, version, parts in con.execute("SELECT id, version, parts FROM content_lists WHERE rowid > ? AND "
                                            "rowid <= ? AND kind = 'conversation'", (after, upto)).fetchall():
        content = json.loads(str(Parted(reader, text, json.loads(parts))))
        blocks = [block for message in content.get("messages", []) for block in message.get("content", [])]
        blocks += [inner for block in blocks if block.get("type") == "tool_result"
                   for inner in block.get("content", [])]
        quoted = dict.fromkeys(block["source"] for block in blocks if block.get("type") == "excerpt")
        con.executemany("INSERT OR IGNORE INTO quotes (conversation, version, text) VALUES (?, ?, ?)",
                        [(text, version, one) for one in quoted])


def _figures_12(con) -> tuple[int, int]:
    """The figures of every dataset revision in the layout of format 12
    (depot.schema), from those of format 11 ({plain, items, other,
    tags}), as the module says: {kinds: {kind: sizes} of every row of
    kinds, plain first, then the others by id, tags}. (libraries that keep
    figures, libraries that had them)."""
    zero = {"texts": 0, "characters": 0, "bytes": 0}
    kinds = ["plain"] + [kind for (kind,) in con.execute("SELECT id FROM kinds WHERE id <> 'plain' ORDER BY id")]
    kept = had = 0
    for dataset, revision, libraries, figures in con.execute(
            "SELECT id, revision, libraries, figures FROM datasets_revisions WHERE figures IS NOT NULL").fetchall():
        figures, refs = json.loads(figures), json.loads(libraries)
        found = []
        for ref, one in zip(refs, figures["libraries"], strict=True):
            if one is None or "kinds" in one:
                found.append(one)
                continue
            had += 1
            if not one["items"]["texts"] and not one["other"]["texts"]:
                found.append({"kinds": {kind: dict(one["plain"] if kind == "plain" else zero) for kind in kinds},
                              "tags": one["tags"]})
            elif _proved_12(con, ref):
                sized = dict.fromkeys(kinds)
                library = ref["library"]
                for kind in kinds:
                    row = con.execute("SELECT texts, characters, bytes FROM counts WHERE facet = ? AND name = ?",
                                      (LIBRARY_KIND_9, f"{library} {kind}")).fetchall()
                    sized[kind] = dict(zip(zero, row[0])) if row else dict(zero)
                found.append({"kinds": sized, "tags": one["tags"]})
            else:
                found.append(None)
                continue
            kept += 1
        con.execute("UPDATE datasets_revisions SET figures = ? WHERE id = ? AND revision = ?",
                    (json.dumps({"at": figures["at"], "libraries": found}, ensure_ascii=False,
                                separators=(",", ":")), dataset, revision))
    return kept, had


def _proved_12(con, ref: dict) -> bool:
    """Whether the counts now are the figures of the library a dataset reads
    (ref: {depot, library, change}) at the change it reads it at, as the
    step to format 9 decides (_figures_9): a library this depot owns, read
    here (never one of another depot, whose counts this depot does not
    keep), the change that locked it for a library locked then, and its
    row of library_moves at or before that change, or no row."""
    library, change = ref["library"], ref["change"]
    name = con.execute("SELECT name FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]
    if ref.get("depot") is not None or ids.owner(library) != name:
        return False
    state = con.execute("SELECT locked_at FROM libraries_revisions WHERE id = ? AND change <= ? "
                        "ORDER BY revision DESC LIMIT 1", (library, change)).fetchall()
    if state and state[0][0] is not None:
        change = con.execute("SELECT min(change) FROM libraries_revisions WHERE id = ? AND locked_at IS NOT NULL",
                             (library,)).fetchall()[0][0]
    moved = con.execute("SELECT change FROM library_moves WHERE library = ?", (library,)).fetchall()
    return not moved or (moved[0][0] is not None and moved[0][0] <= change)


# format 13: the tables of the snapshots and the index of their names by owner
TABLES_13 = (
    "CREATE TABLE IF NOT EXISTS snapshots (id TEXT NOT NULL, change INTEGER NOT NULL, name TEXT NOT NULL, "
    "at_change INTEGER NOT NULL, made_at TEXT NOT NULL, origin TEXT NOT NULL, note TEXT, texts INTEGER NOT NULL, "
    "characters INTEGER NOT NULL, bytes INTEGER NOT NULL, digest TEXT NOT NULL, summary TEXT NOT NULL, "
    "PRIMARY KEY (id))",
    "CREATE TABLE IF NOT EXISTS snapshot_sets (snapshot TEXT NOT NULL, part INTEGER NOT NULL, owner TEXT NOT NULL, "
    "bound INTEGER NOT NULL, bitmap BLOB NOT NULL, texts INTEGER NOT NULL, PRIMARY KEY (snapshot, part, owner))",
    "CREATE UNIQUE INDEX IF NOT EXISTS snapshots_name ON snapshots ((CASE WHEN (rtrim(id, '0123456789') <> id AND "
    "rtrim(id, '0123456789') GLOB '*-') THEN substr(id, 1, length(rtrim(id, '0123456789')) - 1) ELSE id END), name)",
    "CREATE TABLE IF NOT EXISTS legacy_snapshot_of (dataset TEXT NOT NULL, revision INTEGER NOT NULL, "
    "snapshot TEXT NOT NULL, PRIMARY KEY (dataset, revision))",
    "CREATE TABLE IF NOT EXISTS taggings (text TEXT NOT NULL, tag TEXT NOT NULL, joined_change INTEGER NOT NULL, "
    "left_change INTEGER, change INTEGER NOT NULL, PRIMARY KEY (text, tag, joined_change)) WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS taggings_change ON taggings (change)")
# format 13: every revision of the texts after the text ?, with its tags, copied into a table of the writer's
# temporary database in one sequential scan sorted by text id then revision, which the stage taggings then reads a
# batch at a time and turns into rows of taggings (_tagged_13): a scan of the file left open while the stage commits
# would keep every page it writes in the WAL until the end. NOT INDEXED and the unary plus keep SQLite from reading
# the table through an index, whose order is not the file's. The bytes of temporary files the sort and the table take
# for each revision, about
TAGGED_TABLE_13 = ("CREATE TEMP TABLE upgrade_tagged (id TEXT NOT NULL, revision INTEGER NOT NULL, "
                   "change INTEGER NOT NULL, tags TEXT NOT NULL, PRIMARY KEY (id, revision)) WITHOUT ROWID")
COPY_TAGGED_13 = ("INSERT INTO temp.upgrade_tagged (id, revision, change, tags) SELECT id, revision, change, tags "
                  "FROM texts_revisions NOT INDEXED WHERE id > ? ORDER BY +id, +revision")
TAGGED_13 = "SELECT id, revision, change, tags FROM temp.upgrade_tagged WHERE id > ? ORDER BY id, revision LIMIT ?"
TAGGED_REST_13 = ("SELECT id, revision, change, tags FROM temp.upgrade_tagged WHERE id = ? AND revision > ? "
                  "ORDER BY revision")
TAGGINGS_SORTED_13 = 120
# format 13: what the name of a row of upgrade_under_way that records a dataset revision a corpus names begins with
# (then the dataset's id, a space and the corpus's id; its after holds the revision)
NAMED_13 = "named "
# format 13: the folder of the strip reader of format 12: what the depot's work folder adds to the depot file's name,
# then the folder inside it
STRIPS_13 = (".work", "strips")
# format 13: the column of a tokenizer recipe's revision that names its sample (a snapshot and kinds)
SAMPLE_13 = "ALTER TABLE tokenizers_revisions ADD COLUMN sample TEXT"
# format 13: the columns of a revision of texts that keep the kind, length and bytes of the content version it reads
# and, on a current revision, a copy of its text's tags, with their types; and the statement that copies the sizes
# from content_lists into the revisions of rowids in (?, ?] and writes the tags of each current one among them in the
# order of their names, each once, as its copy (the rows of taggings the stage taggings made from them)
SIZES_13 = (("content_kind", "TEXT"), ("content_length", "INTEGER"), ("content_bytes", "INTEGER"),
            ("current_tags", "TEXT"))
# the kind, length and bytes of every content version, copied in one sequential scan of content_lists into a table
# of the writer's temporary database, in the order of its key, which the statement looks them up in: a lookup in
# content_lists itself would read its pages in the order of the revisions, one random read each on a file bigger
# than the memory
SIZES_TABLE_13 = ("CREATE TEMP TABLE upgrade_sizes (id TEXT NOT NULL, version INTEGER NOT NULL, kind TEXT, "
                  "length INTEGER, bytes INTEGER, PRIMARY KEY (id, version)) WITHOUT ROWID")
COPY_SIZES_13 = ("INSERT INTO temp.upgrade_sizes (id, version, kind, length, bytes) SELECT id, version, kind, length, "
                 "bytes FROM content_lists NOT INDEXED ORDER BY +id, +version")
FILL_SIZES_13 = ("UPDATE texts_revisions SET (content_kind, content_length, content_bytes) = (SELECT c.kind, c.length, "
                 "c.bytes FROM temp.upgrade_sizes c WHERE c.id = texts_revisions.id AND "
                 "c.version = texts_revisions.content_version), current_tags = CASE WHEN latest THEN (SELECT "
                 "json_group_array(g.value ORDER BY g.value) FROM (SELECT DISTINCT value FROM json_each(tags)) g) END "
                 "WHERE rowid > ? AND rowid <= ?")
# format 13: the facets of counts of what libraries hold, which go, and the facets split by status and by whether a
# locked library held the text, whose locked part is folded away
LIBRARY_FACETS_13 = ("library", "pinned newer", "library kind", "library tag", "library split")
SPLIT_13 = ("texts split", "tag split", "license split")
# format 13: what the counts of COUNTS_13 read of every current text, copied into a table of the writer's temporary
# database in one sequential scan of texts_revisions (each lookup in the temporary tables, which the memory holds):
# its status, the kind, length and bytes of its content version (has: whether there is one), its tags and licenses,
# and whether a locked library holds it (locked, 1 or 0, as LOCKED_10 says), the texts locked libraries hold copied
# first from one sequential scan of memberships
LOCKED_TABLE_13 = ("CREATE TEMP TABLE upgrade_locked (text TEXT NOT NULL PRIMARY KEY) WITHOUT ROWID",
                   "INSERT OR IGNORE INTO temp.upgrade_locked (text) SELECT text FROM memberships NOT INDEXED "
                   "WHERE left_change IS NULL AND library IN (SELECT id FROM libraries_revisions WHERE latest "
                   "AND locked_at IS NOT NULL) ORDER BY +text")
CURRENT_TABLE_13 = ("CREATE TEMP TABLE upgrade_current (status TEXT, has INTEGER, kind TEXT, length INTEGER, "
                    "bytes INTEGER, tags TEXT, licenses TEXT, locked INTEGER)",
                    "INSERT INTO temp.upgrade_current (status, has, kind, length, bytes, tags, licenses, locked) "
                    "SELECT t.status, c.id IS NOT NULL, c.kind, c.length, c.bytes, t.tags, t.licenses, "
                    "EXISTS (SELECT 1 FROM temp.upgrade_locked k WHERE k.text = t.id) FROM texts_revisions t "
                    "NOT INDEXED LEFT JOIN temp.upgrade_sizes c ON c.id = t.id AND c.version = t.content_version "
                    "WHERE t.latest")
# format 13: the counts of what the current texts of each status hold, by kind and by plain tag set, and the splits
# of each tag and license with their characters and bytes, each made from every current text (CURRENT_TABLE_13);
# the facets they make
SUMS_13 = "count(*), coalesce(sum(c.length), 0), coalesce(sum(c.bytes), 0)"
CURRENT_13 = "temp.upgrade_current c"
COUNTS_13 = (
    ("kind", f"SELECT c.status || ' ' || c.kind, {SUMS_13} FROM {CURRENT_13} WHERE c.has GROUP BY 1"),
    ("plain tagset", "SELECT c.status || ' ' || (SELECT json_group_array(g.value ORDER BY g.value) FROM "
                     f"(SELECT DISTINCT value FROM json_each(c.tags)) g), {SUMS_13} "
                     f"FROM {CURRENT_13} WHERE c.has AND c.kind = 'plain' GROUP BY 1"),
    ("tag split", f"SELECT c.status || ' ' || c.locked || ' ' || j.value, {SUMS_13} FROM {CURRENT_13}, "
                  "json_each(c.tags) j GROUP BY 1"),
    ("license split", f"SELECT c.status || ' ' || c.locked || ' ' || j.value, {SUMS_13} FROM {CURRENT_13}, "
                      "json_each(c.licenses) j GROUP BY 1"))


def _to_13(db: Database, log, records: list[dict] | None = None, public: str | None = None,
           checked: "Check | None" = None):
    """Format 12 to 13, as the module says: the depot marked half upgraded
    with a row of upgrade_under_way for each stage of STAGES_13 and one for
    each dataset revision a corpus record of records names (NAMED_13), then
    each stage whose row is left, each deleting its row as it ends, then the
    mark dropped. public: the depot's public address, which the order data
    sections it says name (its name without one). checked: the check that
    ran right before (check), whose measures of the snapshots the stages
    locked and datasets write when the depot did not change since it read
    them (_state_13)."""
    from .depot import half_upgraded
    known = {}
    if checked is not None and checked.state is not None and checked.state == _state_13(db.reader()):
        known = dict(checked.measured)
    if not half_upgraded(db.reader()):
        own = _owner_13(db.reader())
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            con.executemany(f"INSERT INTO {UNDER_WAY} (stage, after) VALUES (?, 0)", [(name,) for name in STAGES_13])
            con.executemany(f"INSERT OR IGNORE INTO {UNDER_WAY} (stage, after) VALUES (?, ?)",
                            [(f"{NAMED_13}{dataset} {corpus}", revision)
                             for (dataset, revision), corpora in named_revisions(records or [], own).items()
                             for corpus in corpora])
            _new_format(con, 13)
    began = time.monotonic()
    for name, stage in STAGES_13.items():
        if db.reader().execute(f"SELECT 1 FROM {UNDER_WAY} WHERE stage = ?", (name,)).fetchall():
            if name == "datasets":
                stage(db, log, known, public)
            elif name == "locked":
                stage(db, log, known)
            else:
                stage(db, log)
    with db.bulk() as con:
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the depot is at format 13, in {time.monotonic() - began:.0f} seconds")


def named_revisions(records: list[dict], own: str) -> dict[tuple[str, int], list[str]]:
    """{(dataset id, revision): the ids of the corpora whose record names
    it} of the corpus records (those under corpora/ of the dock) built from a
    dataset of the depot named own."""
    found = {}
    for record in records:
        dataset = record.get("dataset") or {}
        if dataset.get("id") and dataset.get("revision") is not None and ids.owner(dataset["id"]) == own:
            found.setdefault((dataset["id"], int(dataset["revision"])), []).append(record["id"])
    return found


def _owner_13(con) -> str:
    return con.execute("SELECT name FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]


def _named_13(con) -> dict[tuple[str, int], list[str]]:
    """The dataset revisions a corpus names, as the first transaction of the step to format 13 recorded them."""
    found = {}
    if not con.execute(f"SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '{UNDER_WAY}'").fetchall():
        return found
    for stage, revision in con.execute(f"SELECT stage, after FROM {UNDER_WAY} WHERE stage LIKE '{NAMED_13}%'"):
        dataset, corpus = stage[len(NAMED_13):].split(" ", 1)
        found.setdefault((dataset, revision), []).append(corpus)
    return found


def _reads_here(path: Path):
    """reads_here of depot.conversion.plan for the depot's file at path, opened read-only when a dataset first
    names a library by a depot's address: whether that library is one of this depot's rows. It is when the address
    leads here (in any form, as Depot.place compares them), or, in a depot that follows an upstream, when the
    address is the upstream's web address and the upstream owns the id, a library a pull brought here."""
    opened = []

    def reads_here(address: str, library: str) -> bool:
        if not opened:
            from .depot import Depot
            opened.append(Depot(str(path), readonly=True))
        depot = opened[0]
        place = depot.place(address)
        if place is None:
            return True
        upstream = depot.upstream()
        return upstream is not None and place == depot.place(upstream) and bool(depot.upstream_owned([library]))
    return reads_here


def _write_snapshot(con, planned, measured, change: int) -> str:
    """Write the converted snapshot's row and sets under the upgrade's change: its id."""
    from .snapshots import COLUMNS
    own = _owner_13(con)
    last = (con.execute("SELECT last FROM numbers WHERE name = 'snapshots'").fetchall() or [(0,)])[0][0]
    con.execute("INSERT INTO numbers (name, last) VALUES ('snapshots', ?) ON CONFLICT (name) DO UPDATE SET "
                "last = excluded.last", (last + 1,))
    snapshot = ids.make(own, last + 1)
    con.execute(f"INSERT INTO snapshots ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (snapshot, change, planned.name, planned.at_change, now(), _json(planned.origin), None, measured.texts,
                 measured.characters, measured.bytes, measured.digest, _json(measured.summary)))
    con.executemany("INSERT INTO snapshot_sets (snapshot, part, owner, bound, bitmap, texts) VALUES (?, ?, ?, ?, ?, ?)",
                    [(snapshot, part, owner, bound, bitmap.serialize(), len(bitmap))
                     for part, owner, bound, bitmap in measured.sets])
    return snapshot


def _upgrade_change(con) -> int:
    return con.execute("SELECT change FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0][0]


def _differ_refused(planned, measured) -> Refused:
    return Refused(f"the snapshot {planned.name} ({_origin_label(planned.origin)}) would read {measured.differ} of its "
                   "texts at another revision than the libraries read them: " + "; ".join(measured.examples)
                   + ". The upgrade stops, the depot half upgraded; nothing of this snapshot was written")


def _origin_label(origin: dict) -> str:
    if origin["kind"] == "library":
        return f"the locked library {origin['name']}, {origin['library']}, locked at the change {origin['lock_change']}"
    return f"revision {origin['revision']} of the dataset {origin['name']}, {origin['dataset']}"


def _state_13(con) -> tuple:
    """What tells whether the depot changed since a check read it: its
    latest change and the largest rowid of each table a conversion reads
    (STATE_13), the upgrade's own changes and the revisions of the about
    row they make left out: the step to format 12 between a check and the
    step to format 13 writes only the kind plain on content versions of no
    kind, which a conversion reads as plain already."""
    upgrades = "SELECT id FROM changes WHERE kind = 'upgrade'"
    return tuple(con.execute(
        f"SELECT (SELECT coalesce(max(id), 0) FROM changes WHERE id NOT IN ({upgrades})), "
        f"(SELECT coalesce(max(revision), 0) FROM about_revisions WHERE change NOT IN ({upgrades})), "
        + ", ".join(f"(SELECT coalesce(max(rowid), 0) FROM {table})" for table in STATE_13
                    if table != "about_revisions")).fetchall()[0])


def _measured_13(db: Database, log, known: dict) -> tuple[list, list, list, set]:
    """What the stages locked and datasets have to write: (the locked
    libraries' snapshots left, every dataset revision's snapshot, why each
    dataset revision left out is, the keys of the snapshots written), as
    depot.conversion.plan plans them, with the measure of each left in
    known ({conversion.key_of: Measured}): those known lacks are measured
    in one sorted merge (conversion.measure_all)."""
    from . import conversion
    con = db.reader()
    libraries, datasets, left = conversion.plan(con, _named_13(con), _reads_here(db.path))
    done = {json.loads(origin).get("library") for (origin,) in con.execute(
        "SELECT origin FROM snapshots WHERE origin ->> 'kind' = 'library'")}
    done |= set(con.execute("SELECT dataset, revision FROM legacy_snapshot_of").fetchall())
    libraries = [one for one in libraries if conversion.key_of(one) not in done]
    wanted = [one for one in [*libraries, *datasets] if one.reuses is None and conversion.key_of(one) not in done]
    missing = [one for one in wanted if not _fits(one, known)]
    if missing:
        log(f"{db.path}: reading what the {len(missing)} snapshots left hold and checking each revision they read "
            f"against what the libraries read, in one sorted merge (depot.conversion); SQLite sorts its scans in "
            f"its temporary folder (SQLITE_TMPDIR, else /var/tmp, /usr/tmp or /tmp), about "
            f"{_size(_sorted_bytes(con))} of free disk")
        known.update(conversion.measure_all(con, missing, lambda line: log(f"{db.path}: {line}")))
    elif wanted:
        log(f"{db.path}: the {len(wanted)} snapshots left are those the check right before read and checked, the "
            "depot unchanged since: each is written as the check read it")
    return libraries, datasets, left, done


def _fits(planned, known: dict) -> bool:
    """Whether known holds a measure of the planned snapshot, of the same parts."""
    from . import conversion
    found = known.get(conversion.key_of(planned))
    return found is not None and [(one["library"], one["locked"], one["change"]) for one in found.parts] == [
        (part.library, part.locked, part.change) for part in planned.parts]


def _sorted_bytes(con) -> int:
    """About the bytes of temporary files conversion.measure_all's sorts take on the depot of con."""
    from . import conversion
    rows = con.execute("SELECT (SELECT coalesce(max(rowid), 0) FROM texts_revisions) + "
                       "(SELECT coalesce(max(rowid), 0) FROM content_lists)").fetchall()[0][0]
    return rows * conversion.SORTED


def _locked_13(db: Database, log, known: dict):
    """The stage locked of the step to format 13: each locked library the
    depot owns made a snapshot of one part (depot.conversion), each in
    a transaction of its own, from its measure (_measured_13), stopping on
    a revision the snapshot would read otherwise, those a run before wrote
    skipped."""
    from . import conversion
    libraries, _, _, _ = _measured_13(db, log, known)
    log(f"{db.path}: making a snapshot of each locked library it owns, {len(libraries)} left, each from its pins "
        "read at the change that locked it, checked against them")
    change = _upgrade_change(db.reader())
    for planned in libraries:
        measured = known[conversion.key_of(planned)]
        if measured.differ:
            raise _differ_refused(planned, measured)
        with db.bulk() as written:
            _write_snapshot(written, planned, measured, change)
        log(f"{db.path}: the snapshot {planned.name} of the locked library {planned.origin['name']}: "
            f"{measured.texts:,} texts, {measured.rows:,} pins read")
    with db.bulk() as written:
        written.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'locked'")


def _datasets_13(db: Database, log, known: dict, public: str | None = None):
    """The stage datasets of the step to format 13: each dataset revision a
    corpus names and each dataset's current revision made a snapshot with a
    part for each library it reads (depot.conversion), from its measure
    (_measured_13), or given the snapshot of the locked library it reads
    alone, recorded in legacy_snapshot_of, each in a transaction of its
    own, stopping on a revision the snapshot would read otherwise, those a
    run before wrote skipped; then, for each one, the data section of a
    model order on its snapshot said."""
    from . import conversion
    _, datasets, left, done = _measured_13(db, log, known)
    con = db.reader()
    log(f"{db.path}: making a snapshot of each of {len(datasets)} dataset revisions, "
        f"{sum(conversion.key_of(one) not in done for one in datasets)} left"
        + "".join(f"; not converted: {one}" for one in left))
    change = _upgrade_change(con)
    for planned in datasets:
        key = conversion.key_of(planned)
        if key in done:
            continue
        if planned.reuses is not None:
            [(snapshot,)] = con.execute("SELECT id FROM snapshots WHERE origin ->> 'kind' = 'library' AND "
                                        "origin ->> 'library' = ?", (planned.reuses,)).fetchall()
            with db.bulk() as written:
                written.execute("INSERT INTO legacy_snapshot_of (dataset, revision, snapshot) VALUES (?, ?, ?)",
                                (*key, snapshot))
            log(f"{db.path}: {_origin_label(planned.origin)} reads the locked library {planned.reuses} alone: it is "
                f"the snapshot {planned.name}")
            continue
        measured = known[key]
        if measured.differ:
            raise _differ_refused(planned, measured)
        with db.bulk() as written:
            snapshot = _write_snapshot(written, planned, measured, change)
            written.execute("INSERT INTO legacy_snapshot_of (dataset, revision, snapshot) VALUES (?, ?, ?)",
                            (*key, snapshot))
        log(f"{db.path}: the snapshot {planned.name} of {_origin_label(planned.origin)}: {measured.texts:,} texts, "
            f"{measured.rows:,} rows of its libraries read")
    for planned in datasets:
        found = con.execute("SELECT s.digest FROM legacy_snapshot_of l JOIN snapshots s ON s.id = l.snapshot "
                            "WHERE l.dataset = ? AND l.revision = ?",
                            (planned.origin["dataset"], planned.origin["revision"])).fetchall()
        log(f"{db.path}: the data section of a model order on the snapshot of {_origin_label(planned.origin)}, "
            "with its shares, size and vocabulary: "
            + _json(_data_section(con, planned, found[0][0], public)))
    with db.bulk() as written:
        written.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'datasets'")


def _data_section(con, planned, digest: str, public: str | None) -> dict:
    """The data section of a model order on the snapshot of the converted dataset revision, with its shares, size
    and vocabulary (conversion.order_data)."""
    from . import conversion
    vocabulary = (planned.row or {}).get("vocabulary")
    size = con.execute("SELECT r.vocab_size FROM vocabularies v JOIN tokenizers_revisions r ON r.id = v.recipe AND "
                       "r.version = v.version WHERE v.sha256 = ? LIMIT 1", (vocabulary,)).fetchall() if vocabulary else []
    measured = conversion.Measured([], {}, 0, 0, 0, digest, 0, 0)
    return conversion.order_data(planned, measured, public or _owner_13(con), size[0][0] if size else None)


def _tables_13(db: Database, log):
    """The stage tables of the step to format 13, in one transaction: the
    tables snapshots and snapshot_sets, the index snapshots_name and the
    table taggings, empty (TABLES_13), the column sample of
    tokenizers_revisions and the columns of SIZES_13 of texts_revisions,
    empty (the stage sizes fills them), and the counts of COUNTS_13 made
    again from every current text, their rows before deleted, read from
    the tables of the writer's temporary database that one sequential scan
    of content_lists, of memberships and of texts_revisions fill
    (SIZES_TABLE_13, LOCKED_TABLE_13, CURRENT_TABLE_13)."""
    log(f"{db.path}: making the tables of the snapshots and of the tags and counting what the texts of each status "
        "hold by kind, by set of tags and by tag and license in characters and bytes, in one transaction that reads "
        "content_lists, memberships and texts_revisions once each, in the order of the file; run again after a "
        "stop, it does it again")
    began = time.monotonic()
    with db.bulk() as con:
        for statement in TABLES_13:
            con.execute(statement)
        if "sample" not in [one[1] for one in con.execute("PRAGMA table_info(tokenizers_revisions)")]:
            con.execute(SAMPLE_13)
        held = {one[1] for one in con.execute("PRAGMA table_info(texts_revisions)")}
        for column, kind in SIZES_13:
            if column not in held:
                con.execute(f"ALTER TABLE texts_revisions ADD COLUMN {column} {kind}")
        _sizes_copied(con)
        for statement in (*LOCKED_TABLE_13, *CURRENT_TABLE_13):
            con.execute(statement)
        for facet, query in COUNTS_13:
            con.execute("DELETE FROM counts WHERE facet = ?", (facet,))
            con.execute(f"INSERT INTO counts (facet, name, texts, characters, bytes) SELECT '{facet}', * FROM "
                        f"({query})")
        con.execute("DROP TABLE temp.upgrade_current")
        con.execute("DROP TABLE temp.upgrade_locked")
        con.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'tables'")
    log(f"{db.path}: the stage tables took {time.monotonic() - began:.0f} seconds")


def _sizes_copied(con):
    """The kind, length and bytes of every content version in the table upgrade_sizes of the temporary database of
    con, the writer, copied once (SIZES_TABLE_13, COPY_SIZES_13) for the stages tables and sizes, in con's
    transaction."""
    if not con.execute("SELECT 1 FROM temp.sqlite_master WHERE name = 'upgrade_sizes'").fetchall():
        con.execute(SIZES_TABLE_13)
        con.execute(COPY_SIZES_13)


def _counts_13(db: Database, log):
    """The stage counts of the step to format 13, in one transaction that
    reads only the table counts: the rows of the facets of libraries
    (LIBRARY_FACETS_13) deleted, and each row of the facets split by status
    and by whether a locked library held the text (SPLIT_13), named
    `<status> <locked> <name>`, folded into the row `<status> <name>`, the
    two rows of a name added up."""
    log(f"{db.path}: folding the counts of the libraries away, in one transaction that reads only the counts")
    with db.bulk() as con:
        con.execute(f"DELETE FROM counts WHERE facet IN ({', '.join(repr(one) for one in LIBRARY_FACETS_13)})")
        for facet in SPLIT_13:
            for status in ("under review", "admitted", "refused"):
                marks = [f"{status} {locked} " for locked in (0, 1)]
                width = len(marks[0])
                folded = con.execute(
                    "SELECT ? || substr(name, ?), sum(texts), sum(characters), sum(bytes) FROM counts "
                    "WHERE facet = ? AND substr(name, 1, ?) IN (?, ?) GROUP BY 1",
                    (f"{status} ", width + 1, facet, width, *marks)).fetchall()
                con.execute("DELETE FROM counts WHERE facet = ? AND substr(name, 1, ?) IN (?, ?)",
                            (facet, width, *marks))
                con.executemany("INSERT INTO counts (facet, name, texts, characters, bytes) VALUES (?, ?, ?, ?, ?) "
                                "ON CONFLICT (facet, name) DO UPDATE SET texts = texts + excluded.texts, "
                                "characters = characters + excluded.characters, bytes = bytes + excluded.bytes",
                                [(facet, *one) for one in folded])
        con.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'counts'")


def _views_13(db: Database, log):
    """The stage views of the step to format 13, in one transaction: the
    views libraries and datasets an older file may hold dropped (the code
    makes its views on each connection, VIEWS, the texts view without
    libraries), then the files of the strip reader of format 12 deleted
    (STRIPS_13, beside the depot's file)."""
    with db.bulk() as con:
        for view in ("libraries", "datasets"):
            con.execute(f"DROP VIEW IF EXISTS main.{view}")
        con.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'views'")
    strips = db.path.with_name(db.path.name + STRIPS_13[0]) / STRIPS_13[1]
    if strips.is_dir():
        shutil.rmtree(strips)
        log(f"{db.path}: deleted {strips}, the strip reader's files, which nothing reads any more")


def _taggings_13(db: Database, log):
    """The stage taggings of the step to format 13: the tags every revision
    of a text holds become rows of taggings (_tagged_13), the revisions
    copied, in one sequential scan sorted by text id, into a table of the
    writer's temporary database (TAGGED_TABLE_13, COPY_TAGGED_13), then read
    from it about BATCH revisions at a time, whole texts, the rows of each
    batch written in a transaction that records the last text's id in
    upgrade_under_way (0 before the first), so a run after a stop copies
    only the texts after it. No read of the file stays open across a
    commit, so the WAL is cut back after each. The revisions keep their
    column tags as it is."""
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'taggings'").fetchall()[0][0]
    total = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    log(f"{db.path}: writing the tags of every revision of the texts as rows of taggings, the {total:,} revisions "
        "copied in one sequential scan sorted by text id into SQLite's temporary folder (SQLITE_TMPDIR, else "
        f"/var/tmp, /usr/tmp or /tmp), about {_size(total * TAGGINGS_SORTED_13)} of free disk"
        + ("" if after == 0 else f"; going on after the text {after}, where the last run stopped"))
    began = said = time.monotonic()
    after = "" if after == 0 else after
    with db.bulk() as written:
        written.execute("DROP TABLE IF EXISTS temp.upgrade_tagged")
        written.execute(TAGGED_TABLE_13)
        written.execute(COPY_TAGGED_13, (after,))
    log(f"{db.path}: the revisions copied and sorted in {time.monotonic() - began:.0f} seconds")
    read = 0
    while True:
        with db.bulk() as written:
            batch = written.execute(TAGGED_13, (after, BATCH)).fetchall()
            if not batch:
                break
            if len(batch) == BATCH:  # the rest of the last text
                batch += written.execute(TAGGED_REST_13, (batch[-1][0], batch[-1][1])).fetchall()
            for rows, last, revisions in _tagged_13(batch):
                written.executemany("INSERT OR IGNORE INTO taggings (text, tag, joined_change, left_change, change) "
                                    "VALUES (?, ?, ?, ?, ?)", rows)
                read += revisions
            after = batch[-1][0]
            written.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'taggings'", (after,))
        if time.monotonic() - said >= SAY_EVERY:
            said = time.monotonic()
            log(f"{db.path}: the tags of {read:,} revisions written, the last text {after}, "
                f"{read / max(said - began, 1e-9):,.0f} revisions a second, {said - began:.0f} seconds")
    with db.bulk() as written:
        written.execute("DROP TABLE IF EXISTS temp.upgrade_tagged")
        written.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'taggings'")
    log(f"{db.path}: the stage taggings took {time.monotonic() - began:.0f} seconds")


def _tagged_13(revisions):
    """The rows of taggings (text, tag, joined_change, left_change, change)
    of the revisions (id, revision, change, tags as JSON text) in the order
    of their ids, then of their revisions: for each tag a revision holds
    that the revision before it of its text lacks (or with none before it),
    a row joined at its change and left at the lowest change of the later
    revisions of its text that lack the tag (null for none), its change
    that one, else the change it joined at. It gives (rows, the last text's
    id, the revisions read) each time the rows of about BATCH revisions are
    made, at the end of a text, then the rest."""
    rows, count, held, seen = [], 0, [], None
    parsed = {}  # a JSON array as text: its values, in order, and as a set

    def tags_of(listed):
        found = parsed.get(listed)
        if found is None:
            values = json.loads(listed)
            found = parsed[listed] = (values, set(values))
            if len(parsed) > 1_000_000:
                parsed.clear()
        return found

    def flush():
        for at, (_, change, (values, _)) in enumerate(held):
            before = held[at - 1][2][1] if at else ()
            for tag in values:
                if tag in before:
                    continue
                gone = min((later[1] for later in held[at + 1:] if tag not in later[2][1]), default=None)
                rows.append((seen, tag, change, gone, change if gone is None else gone))

    for text, revision, change, tags in revisions:
        if text != seen:
            if seen is not None:
                flush()
                if count >= BATCH:
                    yield rows, seen, count
                    rows, count = [], 0
            seen, held = text, []
        held.append((revision, change, tags_of(tags)))
        count += 1
    if seen is not None:
        flush()
        yield rows, seen, count


def _sizes_13(db: Database, log):
    """The stage sizes of the step to format 13: every revision of a text
    keeps the kind, length and bytes of the content version it reads, copied
    from content_lists through the temporary table of the sizes
    (_sizes_copied), and each current revision a copy of its text's tags
    (FILL_SIZES_13), the revisions read in rowid order,
    BATCH rowids a transaction, the row of upgrade_under_way recording the
    last rowid each transaction ended at, so a run after a stop goes on
    after it."""
    con = db.reader()
    after = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'sizes'").fetchall()[0][0]
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    log(f"{db.path}: copying the kind, length and bytes of each revision's content version onto the revision, rows "
        f"{after:,} to {last:,} of texts_revisions, {BATCH:,} rowids a transaction"
        + ("" if after == 0 else "; going on after the last transaction the last run wrote"))
    began = said = time.monotonic()
    first = after
    if after < last:
        with db.bulk() as written:
            _sizes_copied(written)
    while after < last:
        upto = min(after + BATCH, last)
        with db.bulk() as written:
            written.execute(FILL_SIZES_13, (after, upto))
            written.execute(f"UPDATE {UNDER_WAY} SET after = ? WHERE stage = 'sizes'", (upto,))
        after = upto
        if time.monotonic() - said >= SAY_EVERY or after == last:
            said = time.monotonic()
            log(f"{db.path}: revision row {after:,} of {last:,}, {(after - first) / max(said - began, 1e-9):,.0f} "
                f"rows a second, {said - began:.0f} seconds")
    with db.bulk() as written:
        written.execute(f"DELETE FROM {UNDER_WAY} WHERE stage = 'sizes'")
        written.execute("DROP TABLE IF EXISTS temp.upgrade_sizes")
    log(f"{db.path}: the stage sizes took {time.monotonic() - began:.0f} seconds")


# the tables whose largest rowid, with the latest change, tells whether the depot changed since a check (_state_13):
# what the conversions read
STATE_13 = ("texts_revisions", "content_lists", "pins", "memberships", "libraries_revisions", "datasets_revisions",
            "pulls", "about_revisions")
# the stages of the step to format 13, in order, by the name of their row of upgrade_under_way
STAGES_13 = {"tables": _tables_13, "taggings": _taggings_13, "sizes": _sizes_13, "locked": _locked_13,
             "datasets": _datasets_13, "counts": _counts_13, "views": _views_13}

# format 14: the index of the current texts by status, then the length of the content version each reads, then id
LENGTH_14 = (f"CREATE INDEX IF NOT EXISTS texts_length ON texts_revisions (status, content_length, {ORDER}) "
             "WHERE latest")


# format 14: the rows of counts of the values each score takes on the current texts, each value to three significant
# digits, named by the JSON array of the score's name and that value
SCORE_VALUES_14 = (
    "INSERT INTO counts (facet, name, texts) SELECT 'score value', json_array(s.value ->> 'name', "
    "printf('%.3g', CAST(s.value ->> 'value' AS REAL))), count(*) FROM texts_revisions t, json_each(t.scores) s "
    "WHERE t.latest GROUP BY 2")


def _to_14(db: Database, log):
    """Format 13 to 14, as the module says: the rows of counts of the
    scores' values, then the index texts_length, built while the depot is
    marked half upgraded."""
    from .depot import half_upgraded
    if not half_upgraded(db.reader()):
        with db.bulk() as con:
            con.execute(f"CREATE TABLE {UNDER_WAY} (stage TEXT NOT NULL PRIMARY KEY, after INTEGER NOT NULL)")
            _new_format(con, 14)
    texts = db.reader().execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'status'").fetchall()[0][0]
    revisions = db.reader().execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    log(f"{db.path}: counting the values of the scores of its {texts:,} current texts, in one statement that reads "
        f"each of its {revisions:,} revisions once and writes a few rows of counts; run again after a stop, it counts "
        "them again")
    began = time.monotonic()
    with db.bulk() as con:
        con.execute("DELETE FROM counts WHERE facet = 'score value'")
        con.execute(SCORE_VALUES_14)
    log(f"{db.path}: the values of the scores are counted, in {time.monotonic() - began:.0f} seconds")
    log(f"{db.path}: building the index texts_length of its {texts:,} current texts by status and length, in one "
        "statement that reads each of their rows once, sorts them in SQLite's temporary folder (SQLITE_TMPDIR, else "
        "/var/tmp, /usr/tmp or /tmp) and writes no row of a table; run again after a stop, it builds it again")
    began = time.monotonic()
    with db.bulk() as con:
        con.execute(LENGTH_14)
        con.execute(f"DROP TABLE {UNDER_WAY}")
    log(f"{db.path}: the index texts_length is built, in {time.monotonic() - began:.0f} seconds; the depot is at "
        "format 14")


# format 15: the schema of the kind conversation, which takes what an agent harness records (docs/chat-format.md,
# "Recorded from a harness")
CONVERSATION_15 = (
    "{\n"
    "  \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",\n"
    "  \"$id\": \"https://github.com/Workers-Intelligence-Commons/wpt/schemas/conversation.schema.json\",\n"
    "  \"title\": \"WPT conversation (one line of a file in the chat format; docs/chat-format.md)\",\n"
    "  \"type\": \"object\",\n"
    "  \"additionalProperties\": false,\n"
    "  \"required\": [\"messages\"],\n"
    "  \"properties\": {\n"
    "    \"metadata\": {\"type\": \"object\", \"description\": \"what identifies the conversation where it was"
    " recorded, such as a training attempt's step, task and round; never read as tokens\"},\n"
    "    \"harness\": {\n"
    "      \"type\": \"object\",\n"
    "      \"additionalProperties\": false,\n"
    "      \"required\": [\"name\"],\n"
    "      \"description\": \"the agent harness whose forms the conversation follows; never read as tokens\",\n"
    "      \"properties\": {\n"
    "        \"name\": {\"type\": \"string\", \"minLength\": 1},\n"
    "        \"version\": {\"type\": \"string\", \"minLength\": 1}\n"
    "      }\n"
    "    },\n"
    "    \"tools\": {\"$ref\": \"#/$defs/tools\", \"description\": \"the tools defined from the start of the"
    " conversation\"},\n"
    "    \"messages\": {\n"
    "      \"type\": \"array\",\n"
    "      \"minItems\": 1,\n"
    "      \"items\": {\n"
    "        \"type\": \"object\",\n"
    "        \"additionalProperties\": false,\n"
    "        \"required\": [\"role\", \"content\"],\n"
    "        \"properties\": {\n"
    "          \"role\": {\"enum\": [\"system\", \"user\", \"assistant\", \"tool\"]},\n"
    "          \"content\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/$defs/block\"}},\n"
    "          \"end\": {\"enum\": [\"stop\", \"tool_calls\", \"limit\"], \"description\": \"how an assistant's turn"
    " ended: the model closed it with no call, the model closed it holding calls, or a limit cut it;"
    " never read as tokens\"},\n"
    "          \"metadata\": {\n"
    "            \"type\": \"object\",\n"
    "            \"description\": \"what is known about the message where it was recorded; never read as"
    " tokens\",\n"
    "            \"properties\": {\"draft\": {\"$ref\": \"#/$defs/draft\"}}\n"
    "          }\n"
    "        },\n"
    "        \"allOf\": [\n"
    "          {\"if\": {\"properties\": {\"role\": {\"const\": \"system\"}}},\n"
    "           \"then\": {\"properties\": {\"content\": {\"items\": {\"properties\": {\"type\": {\n"
    "             \"enum\": [\"text\", \"json\", \"file\", \"tools\"], \"description\": \"the blocks a system message"
    " holds\"}}}}}}},\n"
    "          {\"if\": {\"properties\": {\"role\": {\"const\": \"user\"}}},\n"
    "           \"then\": {\"properties\": {\"content\": {\"items\": {\"properties\": {\"type\": {\n"
    "             \"enum\": [\"text\", \"json\", \"file\", \"excerpt\"], \"description\": \"the blocks a user message"
    " holds\"}}}}}}},\n"
    "          {\"if\": {\"properties\": {\"role\": {\"const\": \"assistant\"}}},\n"
    "           \"then\": {\"properties\": {\"content\": {\"items\": {\"properties\": {\n"
    "             \"type\": {\"enum\": [\"text\", \"thinking\", \"tool_call\", \"json\", \"file\"], \"description\":"
    " \"the blocks an assistant message holds\"},\n"
    "             \"by\": {\"not\": {}, \"description\": \"an assistant's text is the model's own: the harness"
    " adds no text to it\"}}}}}}},\n"
    "          {\"if\": {\"properties\": {\"role\": {\"const\": \"tool\"}}},\n"
    "           \"then\": {\"properties\": {\"content\": {\"items\": {\"properties\": {\"type\": {\n"
    "             \"enum\": [\"tool_result\"], \"description\": \"the blocks a tool message holds\"}}}}}}},\n"
    "          {\"if\": {\"properties\": {\"role\": {\"const\": \"assistant\"}}},\n"
    "           \"else\": {\"properties\": {\"content\": {\"minItems\": 1},\n"
    "                                   \"metadata\": {\"not\": {\"required\": [\"draft\"]}},\n"
    "                                   \"end\": {\"not\": {}, \"description\": \"only an assistant's turn has"
    " an end\"}}}}\n"
    "        ]\n"
    "      }\n"
    "    }\n"
    "  },\n"
    "  \"$defs\": {\n"
    "    \"tools\": {\n"
    "      \"type\": \"array\",\n"
    "      \"items\": {\n"
    "        \"type\": \"object\",\n"
    "        \"additionalProperties\": false,\n"
    "        \"required\": [\"name\"],\n"
    "        \"description\": \"a tool: its name, and its description and the JSON Schema of its parameters"
    " when they are known\",\n"
    "        \"properties\": {\n"
    "          \"name\": {\"type\": \"string\", \"minLength\": 1},\n"
    "          \"description\": {\"type\": \"string\"},\n"
    "          \"parameters\": {\"type\": \"object\", \"description\": \"the JSON Schema of the call's"
    " arguments\"}\n"
    "        }\n"
    "      }\n"
    "    },\n"
    "    \"draft\": {\n"
    "      \"type\": \"object\",\n"
    "      \"additionalProperties\": false,\n"
    "      \"required\": [\"model\", \"weights\", \"sha256\", \"edited\"],\n"
    "      \"description\": \"an assistant message that started as a reply drafted by a WPT model, which a"
    " person then confirmed, edited or not (docs/chat-format.md, \\\"Model drafts\\\")\",\n"
    "      \"properties\": {\n"
    "        \"model\": {\"type\": \"string\", \"minLength\": 1, \"description\": \"the name of the model folder"
    " that drafted it\"},\n"
    "        \"weights\": {\"type\": \"object\", \"minProperties\": 1, \"description\": \"the sha256 of each of"
    " that folder's weight files, by file name\",\n"
    "                    \"additionalProperties\": {\"type\": \"string\", \"pattern\": \"^[0-9a-f]{64}$\"}},\n"
    "        \"sha256\": {\"type\": \"string\", \"pattern\": \"^[0-9a-f]{64}$\", \"description\": \"the sha256 of the"
    " drafted content as the model wrote it, of its JSON text written as docs/chat-format.md, Model"
    " drafts, says\"},\n"
    "        \"edited\": {\"type\": \"boolean\", \"description\": \"true when the person changed the draft before"
    " confirming it: the content no longer gives sha256\"}\n"
    "      }\n"
    "    },\n"
    "    \"block\": {\n"
    "      \"type\": \"object\",\n"
    "      \"required\": [\"type\"],\n"
    "      \"properties\": {\"type\": {\"enum\": [\"text\", \"thinking\", \"tool_call\", \"tool_result\", \"json\","
    " \"file\", \"excerpt\", \"tools\"]}},\n"
    "      \"allOf\": [\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"text\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"text\"],\n"
    "                  \"properties\": {\"type\": true, \"text\": {\"type\": \"string\"},\n"
    "                                 \"by\": {\"const\": \"harness\", \"description\": \"who wrote the text when"
    " it is not the message's author: the harness that added it\"}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"thinking\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"text\"],\n"
    "                  \"properties\": {\"type\": true, \"text\": {\"type\": \"string\"}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"tool_call\"}}},\n"
    "         \"then\": {\"if\": {\"required\": [\"raw\"]},\n"
    "                  \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"id\", \"raw\"],\n"
    "                           \"description\": \"a call whose arguments could not be parsed: it has no"
    " arguments, and raw holds, with name, the text of its arguments as the model wrote them, without"
    " name, the whole call as the model wrote it\",\n"
    "                           \"properties\": {\"type\": true, \"id\": {\"type\": \"string\", \"minLength\":"
    " 1},\n"
    "                                          \"name\": {\"type\": \"string\", \"minLength\": 1}, \"raw\":"
    " {\"type\": \"string\"}}},\n"
    "                  \"else\": {\"additionalProperties\": false, \"required\": [\"type\", \"id\", \"name\","
    " \"arguments\"],\n"
    "                           \"description\": \"a call with its arguments, or with raw, as the model"
    " wrote them, when they could not be parsed\",\n"
    "                           \"properties\": {\"type\": true, \"id\": {\"type\": \"string\", \"minLength\":"
    " 1},\n"
    "                                          \"name\": {\"type\": \"string\", \"minLength\": 1}, \"arguments\":"
    " {\"type\": \"object\"}}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"tool_result\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"call\", \"content\"],\n"
    "                  \"properties\": {\"type\": true, \"call\": {\"type\": \"string\", \"minLength\": 1}, \"error\":"
    " {\"type\": \"boolean\"},\n"
    "                                 \"value\": {\"description\": \"what the tool returned, in structured"
    " form, when it is known; never read as tokens\"},\n"
    "                                 \"conversation\": {\"type\": \"string\", \"minLength\": 1, \"description\":"
    " \"the id of the depot conversation of the sub-agent that produced the result\"},\n"
    "                                 \"content\": {\"type\": \"array\", \"items\": {\"$ref\": \"#/$defs/block\",\n"
    "                                             \"properties\": {\"type\": {\"enum\": [\"text\", \"json\","
    " \"file\", \"excerpt\", \"tools\"],\n"
    "                                                                     \"description\": \"the blocks a"
    " tool result holds\"}}}}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"json\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"value\"],\n"
    "                  \"properties\": {\"type\": true, \"value\": true}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"file\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"sha256\", \"media_type\"],\n"
    "                  \"properties\": {\"type\": true, \"sha256\": {\"type\": \"string\", \"pattern\":"
    " \"^[0-9a-f]{64}$\"},\n"
    "                                 \"media_type\": {\"type\": \"string\", \"pattern\":"
    " \"^[a-z]+/[A-Za-z0-9.+-]+$\"}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"excerpt\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"source\", \"selector\"],\n"
    "                  \"description\": \"a passage of a plain text of the depot, quoted by its content"
    " (docs/chat-format.md, \\\"Excerpts\\\"): the corpus build writes the passage in its place\",\n"
    "                  \"properties\": {\"type\": true, \"source\": {\"type\": \"string\", \"minLength\": 1,"
    " \"description\": \"the id of the quoted text\"},\n"
    "                                 \"selector\": {\"type\": \"object\", \"additionalProperties\": false,"
    " \"required\": [\"type\", \"exact\"],\n"
    "                                              \"description\": \"a W3C Web Annotation"
    " TextQuoteSelector (https://www.w3.org/TR/annotation-model/#text-quote-selector)\",\n"
    "                                              \"properties\": {\"type\": {\"const\": \"TextQuoteSelector\"},\n"
    "                                                             \"exact\": {\"type\": \"string\","
    " \"description\": \"the passage, character for character\"},\n"
    "                                                             \"prefix\": {\"type\": \"string\","
    " \"description\": \"the characters right before it\"},\n"
    "                                                             \"suffix\": {\"type\": \"string\","
    " \"description\": \"the characters right after it\"}}}}}},\n"
    "        {\"if\": {\"properties\": {\"type\": {\"const\": \"tools\"}}},\n"
    "         \"then\": {\"additionalProperties\": false, \"required\": [\"type\", \"tools\"],\n"
    "                  \"description\": \"the tools defined from this point of the conversation on\",\n"
    "                  \"properties\": {\"type\": true, \"tools\": {\"$ref\": \"#/$defs/tools\", \"minItems\":"
    " 1}}}}\n"
    "      ]\n"
    "    }\n"
    "  }\n"
    "}\n"
)


def _to_15(db: Database, log):
    """Format 14 to 15, as the module says: the conversation kind's
    schema, in one transaction."""
    log(f"{db.path}: writing the schema of the kind conversation that takes what an agent harness records, in one "
        "transaction that writes one row of kinds")
    with db.bulk() as con:
        con.execute("UPDATE kinds SET schema = ? WHERE id = 'conversation'", (CONVERSATION_15,))
        _new_format(con, 15)
    log(f"{db.path}: the depot is at format 15")


# each format of an SQLite file below the code's: the step that brings it to the next
FILE_STEPS = {4: _to_5, 5: _to_6, 6: _to_7, 7: _to_8, 8: _to_9, 9: _to_10, 10: _to_11, 11: _to_12, 12: _to_13,
              13: _to_14, 14: _to_15}


# what `depot upgrade --check` takes for the time of what it does not read itself, measured on bench/large.py's
# depot (2,500,000 texts, 18,322,341 revisions, 4,624,673 content versions, 8% of them items and 6% conversations,
# 21 GB, with 1.5 GB of memory): rows of content_lists the step to format 12 writes a second, its last transaction
# included (4,624,673 in 66 seconds, the quotes of its conversations read from their parts included), revisions a
# second its count of the items' kinds reads at most (18,322,341 in 40 seconds when it scans texts_revisions; the
# 370,000 items looked up one by one took 32), and revisions a second the stage tables of the step to format 13 reads
# and counts (280,000 and 330,000 in two runs)
RATE_12, RATE_12_KINDS, RATE_13_TABLES = 70_000, 450_000, 250_000
# the same for the stage taggings: revisions it copies, sorts and turns into rows a second (230,000 there), and the
# bytes of the file a row of taggings takes, its index's entry included, besides twice its text's and tag's names
# (_sampled_taggings: 11,512,394 rows in 999 MB on bench/large.py's depot, 3,508,771 in 218 MB on bench/upgrade.py's)
RATE_13_TAGGINGS, TAGGING_ROW_BYTES = 200_000, 30
# the same for the stage sizes: revisions it writes a second (240,000 and 280,000 on bench/large.py's depot), and
# the bytes of the file each revision grows by, the pages its rows no longer fit in split included (43 on
# bench/large.py's depot, whose texts_revisions grew from 4,407 MB to 5,193; 20 on bench/upgrade.py's)
RATE_13_SIZES, SIZE_BYTES = 200_000, 43
# the bytes of the file the copy of a text's tags on its current revision takes (current_tags): for each current
# text, and for each tag it has (bench/upgrade.py measures 5.4 on the three-letter tags of its synthetic depot, whose
# texts_revisions grows by 31.6 bytes a revision with SIZE_BYTES; longer names take more)
COPY_BYTES, COPY_TAG_BYTES = 3, 8
# the same for the step to format 14: current texts a second whose entry it adds to the index texts_length (1,500,000
# in 6 seconds on bench/large.py's depot of 17.7 GB, the file partly in the page cache; taken lower, for a file read
# from the disk), and the bytes of the file an entry takes (101,588,992 bytes for those 1,500,000)
RATE_14, LENGTH_ENTRY_14 = 200_000, 70
# and the revisions a second its count of the scores' values reads (SCORE_VALUES_14: 11,247,035 in 12 seconds on
# bench/large.py's depot, read from the disk, its texts holding 3 million scores; taken lower, for texts of more scores)
RATE_14_SCORES = 400_000
REVISION_BYTES = 200  # bytes of the file a revision of a text takes, about: what the stage sizes rewrites of it
SAMPLED = 2_000  # rows the check reads to measure what it does not read whole (content rows' size, tag sets)
COUNT_ROW_BYTES = 64  # bytes a row of counts takes besides its name, about
KIND_BYTES = 6  # bytes the kind plain adds to a content version of no kind
# what the check's file beside the depot's file adds to its name (saved_check_path)
SAVED_CHECK = ".upgrade-check.json"
# how long the upgrade waits for every other connection to the depot file to close before it refuses (_alone)
ALONE_WAIT_MS = 1000
RUNBOOK = (
    "Before: run `depot backup` right before the upgrade; restoring that backup needs the wpt of the depot's "
    f"format now, since this wpt (format {FORMAT}) cannot open a depot of format {FORMAT - 1} or older.",
    f"Order: upgrade the server first, then each clone: a clone of format {FORMAT} refuses an upstream of format "
    f"{FORMAT - 1}.",
    "Stop: stop every process that has the depot open (`depot web`, its jobs, a backup, any other `depot` "
    "command) before `--yes`; the upgrade refuses while one still has the file open, since the code of the old "
    "format checks the format only when it opens the depot and would go on writing into the half-upgraded file. "
    "`--check` may run while the web app serves: `--yes` takes what it saved if nothing was written since, and "
    "the WAL keeps what the app writes until the check ends.")


@dataclass
class Check:
    """What `depot upgrade --check` found: whether the upgrade may run
    (passed), what it says (lines), the snapshots the step to format 13
    would make ({name, origin, at_change, digest, texts, reuses}), what
    would stop it (failures), its cost ({seconds, wal, added, backup},
    seconds by step), whether there is anything to upgrade (needed), what
    it measured of each snapshot ({conversion.key_of: Measured}) and the
    depot's state when it read them (_state_13), which the upgrade right
    after writes when the state is the same."""
    passed: bool
    lines: list[str]
    snapshots: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    needed: bool = True
    measured: dict = field(default_factory=dict)
    state: tuple | None = None


def saved_check_path(path: Path) -> Path:
    """The file beside the depot's file where the check saves what it measured (SAVED_CHECK)."""
    return path.with_name(path.name + SAVED_CHECK)


def _save_check(path: Path, state: tuple, measured: dict):
    """What the check measured of each snapshot and the depot's state when it read them, written beside the
    depot's file (saved_check_path) through a temporary file and a rename."""
    import base64
    found = []
    for key, one in measured.items():
        found.append({"key": list(key) if isinstance(key, tuple) else key,
                      "sets": [[k, owner, bound, base64.b64encode(bitmap.serialize()).decode("ascii")]
                               for k, owner, bound, bitmap in one.sets],
                      "summary": one.summary, "texts": one.texts, "characters": one.characters, "bytes": one.bytes,
                      "digest": one.digest, "rows": one.rows, "differ": one.differ, "examples": one.examples,
                      "seconds": one.seconds, "parts": one.parts})
    target = saved_check_path(path)
    temporary = target.with_name(target.name + ".part")
    temporary.write_text(json.dumps({"format": FORMAT, "state": list(state), "saved_at": now(), "measured": found}),
                         encoding="utf-8", newline="\n")
    temporary.replace(target)


def _saved_check(path: Path, state: tuple) -> tuple[dict, str]:
    """({conversion.key_of: Measured} the saved check holds when it was made by this code's format on the depot in
    the state given, else {}; what to say of it, '' when there is none)."""
    import base64
    from pyroaring import BitMap
    from . import conversion
    target = saved_check_path(path)
    if not target.is_file():
        return {}, ""
    try:
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved.get("format") != FORMAT or tuple(saved["state"]) != tuple(state):
            return {}, (f"the check saved in {target} at {saved.get('saved_at')} read the depot in another state "
                        "(a change written since, or another format): every snapshot is measured again")
        known = {}
        for one in saved["measured"]:
            key = tuple(one["key"]) if isinstance(one["key"], list) else one["key"]
            known[key] = conversion.Measured(
                [(k, owner, bound, BitMap.deserialize(base64.b64decode(bitmap))) for k, owner, bound, bitmap
                 in one["sets"]], one["summary"], one["texts"], one["characters"], one["bytes"], one["digest"],
                one["rows"], one["differ"], one["examples"], one["seconds"], one["parts"])
    except (ValueError, KeyError, TypeError) as error:
        return {}, f"the check saved in {target} cannot be read ({error}): every snapshot is measured again"
    return known, (f"the check saved in {target} at {saved['saved_at']} read the depot as it is now: its measures "
                   "are taken, and only a snapshot it lacks is measured")


def check(address: str, records: list[dict] | None = None, public: str | None = None, log=None,
          save: bool = False) -> Check:
    """What upgrading the depot at address would do, read through a
    connection that opens its file read-only, so nothing is written: the
    steps left; for a depot of format 11, the check the step to format 12
    makes before it ends (its column kind declared as format 11 declares
    it); for one of format 11 or 12, what the step to format 13 converts
    (depot.conversion: locked libraries, dataset revisions a corpus of
    records names and current ones, their parts, open libraries left as
    they are, what is not converted and why), every check of the revisions
    each part reads, the names made snapshots' names, the order data
    section of each dataset revision (public: the depot's public address),
    and the cost of every step left together (seconds, the full copy the
    next backup sends, the WAL at its largest, the disk added), then the
    runbook (RUNBOOK) and the command that runs it. A depot older than
    format 11 or a folder gets the steps alone: each stage of the step to
    format 13 checks its conversions itself and stops on a difference; a
    depot of format 13 gets the cost of the steps to formats 14 and 15
    alone, one of format 14 that of the step to format 15.
    It reads in one read transaction. A check saved beside the depot's file
    (saved_check_path) of the depot in the same state (_state_13) gives the
    measures it holds, and save writes what this one measured there, so a
    check run while the depot is in use spares `--yes` its reads.
    OtherFormat for a newer format."""
    from . import conversion
    from .depot import format_of, half_upgraded, location, other_format
    path = location(address)
    command = "depot upgrade --yes"
    if path.is_dir():
        start = format_of(address)
        return Check(True, [f"{path}: a folder of format {start}: the upgrade runs each step up to format 3 in it, then "
                            f"writes a new depot file from it and brings that file up to format {FORMAT}; each stage of the "
                            "step to format 13 checks the snapshots it makes and stops on a difference", *RUNBOOK,
                            f"Run: {command}"])
    db = Database(path, readonly=True)
    con = None
    try:
        con = db.reader()
        start, half = _format(db), half_upgraded(con)
        if start > FORMAT:
            raise OtherFormat(other_format(str(path), start))
        if start >= FORMAT:
            return Check(True, [f"{path}: at format {FORMAT} already, nothing to upgrade"], needed=False)
        lines = [f"{path}: format {start}" + (f", an upgrade to format {start + 1} not ended" if half else "")
                 + f"; the upgrade runs the steps {', '.join(f'{one} to {one + 1}' for one in range(start, FORMAT))}"]
        if start >= 13:
            cost = _cost_14(con, path) if start == 13 else _cost_15(path)
            return Check(True, [*lines, *cost.pop("lines"), *RUNBOOK, f"Run: {command}"], cost=cost)
        if start < 11:
            lines.append("The libraries and datasets are converted by the step to format 13, after the steps before it: "
                         "this check reads a depot of format 11 or 12, so each stage checks its conversions itself and "
                         "stops on a difference, writing nothing of the snapshot it was making.")
            return Check(True, [*lines, *RUNBOOK, f"Run: {command}"])
        failures = []
        con.execute("BEGIN")  # one read transaction: the plan, the state and the measures of one depot
        if start == 11:
            statement = con.execute("SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'content_lists'"
                                    ).fetchall()[0][0]
            if NOT_NULL_12 not in statement and statement.count(NULLABLE_12) != 1:
                failures.append(f"content_lists is declared as {statement!r}, where format 11 declares the column kind "
                                f"once as {NULLABLE_12!r}: the step to format 12 would stop")
            else:
                lines.append("Step 11 to 12: content_lists declares its column kind as format 11 does, so the step can "
                             "declare it NOT NULL once every content version has a kind.")
        own = _owner_13(con)
        named = named_revisions(records or [], own)
        for key, corpora in _named_13(con).items():
            named.setdefault(key, []).extend(one for one in corpora if one not in named.get(key, []))
        if records is None and not _named_13(con):
            lines.append("No dock given: only each dataset's current revision converts, not the revisions corpora "
                         "were built from (set DOCK to the dock that holds them).")
        libraries, datasets, left = conversion.plan(con, named, _reads_here(path))
        state = _state_13(con)
        wanted = [*libraries, *(one for one in datasets if one.reuses is None)]
        known, said = _saved_check(path, state)
        measured = {conversion.key_of(one): known[conversion.key_of(one)] for one in wanted if _fits(one, known)}
        if said:
            lines.append(f"Saved check: {said}.")
            if log is not None:
                log(f"{path}: {said}")
        missing = [one for one in wanted if conversion.key_of(one) not in measured]
        if missing:
            if log is not None:
                log(f"{path}: reading what each snapshot the step to format 13 would make holds and checking each "
                    "revision it reads against what the libraries read, in one sorted merge (depot.conversion); "
                    "SQLite sorts its scans in its temporary folder (SQLITE_TMPDIR, else /var/tmp, /usr/tmp or "
                    f"/tmp), about {_size(_sorted_bytes(con))} of free disk")
            measured.update(conversion.measure_all(con, missing,
                                                   None if log is None else lambda line: log(f"{path}: {line}")))
        if save:
            _save_check(path, state, measured)
            lines.append(f"Saved: what this check measured, in {saved_check_path(path)}; `depot upgrade --yes` "
                         "takes it while the depot stays as it is now, and measures again after any change.")
        made = []
        lines.append("What converts (step 12 to 13):")
        for planned in libraries:
            found = measured[planned.origin["library"]]
            made.append(_made(planned, found))
            lines.append(f"  {_origin_label(planned.origin)} -> snapshot {planned.name}: {found.texts:,} texts, "
                         f"digest {found.digest}")
        for planned in datasets:
            if planned.reuses is not None:
                found = measured[planned.reuses]
                made.append(_made(planned, found))
                lines.append(f"  {_origin_label(planned.origin)} ({planned.why}) reads the locked library "
                             f"{planned.reuses} alone -> the snapshot {planned.name}, no second one")
                continue
            found = measured[(planned.origin["dataset"], planned.origin["revision"])]
            made.append(_made(planned, found))
            parts = ", ".join(f"{part.library} ({'pins at ' if part.locked else 'members at '}{part.change})"
                              for part in planned.parts)
            lines.append(f"  {_origin_label(planned.origin)} ({planned.why}) -> snapshot {planned.name}, parts {parts}: "
                         f"{found.texts:,} texts, digest {found.digest}")
        for one in left:
            lines.append(f"  not converted: {one} (models trained on it stay verifiable through the legacy routes)")
        held = con.execute(f"SELECT id, name, locked_at FROM libraries_revisions WHERE latest ORDER BY "
                           f"{ids.order_sql('id')}").fetchall()
        opened = [f"{name} ({library})" for library, name, locked in held if locked is None and ids.owner(library) == own]
        if opened:
            lines.append(f"  open libraries left in the legacy tables, not converted: {', '.join(opened)}")
        theirs = [f"{name} ({library})" for library, name, locked in held if locked is not None and ids.owner(library) != own]
        if theirs:
            lines.append(f"  locked libraries of other depots, converted by their own upgrade and received by a pull: "
                         f"{', '.join(theirs)}")
        lines.append("Checks of the revisions each part reads (the legacy read's against the snapshot's):")
        for planned in [*libraries, *(one for one in datasets if one.reuses is None)]:
            key = (planned.origin["library"] if planned.origin["kind"] == "library"
                   else (planned.origin["dataset"], planned.origin["revision"]))
            found = measured[key]
            for part in found.parts:
                said = f"  snapshot {planned.name}, library {part['library']}: {part['texts']:,} texts"
                if part["differ"]:
                    failures.append(f"snapshot {planned.name}, library {part['library']}: {part['differ']:,} texts "
                                    f"read at another revision; " + "; ".join(found.examples))
                    lines.append(f"{said}, FAILED: {part['differ']:,} differ")
                else:
                    lines.append(f"{said}, every revision equal")
        renamed = [(one.origin["name"], one.name) for one in [*libraries, *datasets] if one.reuses is None
                   and one.origin["name"] != one.name]
        lines.append("Names made snapshots' names: " + ("; ".join(f"{old!r} -> {new}" for old, new in renamed)
                                                       if renamed else "every name kept as it is"))
        for planned in datasets:
            lines.append(f"Order data section for {_origin_label(planned.origin)}: "
                         + _json(_data_section(con, planned, measured[planned.reuses if planned.reuses is not None else (
                             planned.origin["dataset"], planned.origin["revision"])].digest, public)))
        cost = _cost(con, path, start, half, [measured[key] for key in measured], libraries, datasets)
        lines += cost.pop("lines")
    finally:
        if con is not None and not con.getautocommit():
            con.execute("ROLLBACK")
        db.close()
    if failures:
        lines.append("The check FAILED: the upgrade would stop at what is named above. Nothing was written. Send these "
                     "lines to the maintainers before upgrading; the depot stays as it is.")
        lines += [f"  {one}" for one in failures]
    else:
        lines += [*RUNBOOK, f"Run: {command}"]
    return Check(not failures, lines, made, failures, cost, measured=measured, state=state)


def _made(planned, measured) -> dict:
    return {"name": planned.name, "origin": planned.origin, "at_change": planned.at_change, "digest": measured.digest,
            "texts": measured.texts, "reuses": planned.reuses}


def _cost(con, path: Path, start: int, half: bool, measured: list, libraries: list, datasets: list) -> dict:
    """The cost of the steps left (Check.cost) and its lines."""
    from . import conversion
    rows = con.execute("SELECT coalesce(max(rowid), 0) FROM content_lists").fetchall()[0][0]
    after = 0
    if start == 11 and half:
        found = con.execute(f"SELECT after FROM {UNDER_WAY} WHERE stage = 'contents'").fetchall()
        after = found[0][0] if found else 0
    contents = max(0, rows - after) if start == 11 else 0
    step = max(1, rows // SAMPLED)
    row_bytes = _sampled_row_bytes(con, rows, step)
    # from the counts, which every write keeps: counting an index would read its pages in the order of its keys
    current = con.execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'status'").fetchall()[0][0]
    revisions = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    tagged = con.execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'tag'").fetchall()[0][0]
    tagsets = _tagsets(con, current)
    taggings, tagging_bytes = _sampled_taggings(con, current)
    converted = list(measured)
    snapshot_bytes = [conversion.set_bytes(one) + len(_json(one.summary).encode()) + 200 for one in converted]
    # the stages locked and datasets write what this check measured, when the upgrade runs right after it
    again = sum(one.seconds for one in converted)
    seconds = {"11 to 12": contents / RATE_12 + (revisions / RATE_12_KINDS if start == 11 else 0),
               "12 to 13, tables": revisions / RATE_13_TABLES,
               "12 to 13, taggings": revisions / RATE_13_TAGGINGS, "12 to 13, sizes": revisions / RATE_13_SIZES,
               "12 to 13, locked and datasets": 0.0 if start == 12 else again, "12 to 13, counts and views": 0.0,
               "13 to 14": current / RATE_14 + revisions / RATE_14_SCORES, "14 to 15": 0.0}
    temporary = max(_sorted_bytes(con), revisions * TAGGINGS_SORTED_13)
    added = (current * LENGTH_ENTRY_14 + contents * KIND_BYTES + sum(snapshot_bytes) + 50 * len(datasets)
             + COUNT_ROW_BYTES * (tagsets + 50) + tagging_bytes + revisions * SIZE_BYTES
             + current * COPY_BYTES + tagged * COPY_TAG_BYTES)
    wal = max([int(min(contents, BATCH) * row_bytes * 1.1), *(int(one * 1.1) for one in snapshot_bytes),
               COUNT_ROW_BYTES * (tagsets + 50) * 2, int(min(revisions, BATCH) * 3 * tagging_bytes / max(1, taggings) * 1.1),
               int(min(revisions, BATCH) * REVISION_BYTES * 1.1), 1 << 20])
    size = path.stat().st_size
    total = sum(seconds.values())
    lines = ["Cost of the steps left together:",
             *(f"  {name}: about {value:,.0f} seconds" for name, value in seconds.items() if start == 11
               or name != "11 to 12"),
             f"  in all about {total:,.0f} seconds ({total / 60:,.1f} minutes); `--yes` runs this check again first: "
             "it takes what `depot upgrade --check` saved while the depot stays as it was then, and otherwise "
             f"reads again, about {again:,.0f} seconds more (the libraries' {sum(one.rows for one in converted):,} "
             f"rows read and every revision they read checked in {again:,.1f} seconds); the stages locked and "
             "datasets then write the snapshots it measured, the depot unchanged since but for the upgrade's own steps",
             f"  temporary folder: SQLite sorts in it (SQLITE_TMPDIR, else /var/tmp, /usr/tmp or /tmp), about "
             f"{_size(temporary)} of free disk at most, for this check and for the upgrade; its files are deleted "
             "as they are made, so nothing stays after a stop",
             f"  disk added: about {_size(added)} ({_size(sum(snapshot_bytes))} of snapshots"
             + (f", {_size(contents * KIND_BYTES)} of kinds on {contents:,} content versions" if contents else "")
             + f", about {tagsets:,} rows of counts of plain tag sets, {_size(tagging_bytes)} of "
               f"taggings, about {taggings:,} rows, one for each tag a text has and one more for each tag an older "
               f"revision held (from the revisions of a sample of texts), {_size(revisions * SIZE_BYTES)} of the kind, length and bytes kept on "
               f"{revisions:,} revisions, {_size(current * COPY_BYTES + tagged * COPY_TAG_BYTES)} of the copy of "
               f"each current text's tags, {_size(current * LENGTH_ENTRY_14)} of the index texts_length"
             + "); the tables of libraries and datasets stay, nothing is freed",
             f"  WAL at its largest: about {_size(wal)}, cut back to {_size(JOURNAL_LIMIT)} after each "
             "transaction",
             f"  backup: the next `depot backup` sends a full copy of the file, about {_size(size + added)}, "
             "onto the dock (each step marks the file for a full copy, since change files record no layout)"]
    return {"seconds": seconds, "again": again, "added": added, "wal": wal, "backup": size + added,
            "tagsets": tagsets, "temporary": temporary, "lines": lines}


def _cost_15(path: Path) -> dict:
    """The cost of the step to format 15 alone (Check.cost) and its lines: one row of kinds written."""
    size = path.stat().st_size
    lines = ["Cost of the step 14 to 15:",
             "  writing the schema of the kind conversation: one row of kinds, in one transaction, under a second",
             "  disk added: none to speak of; WAL: a few pages",
             f"  backup: the next `depot backup` sends a full copy of the file, about {_size(size)}, onto the dock "
             "(the step marks the file for a full copy, since change files record no layout)"]
    return {"seconds": {"14 to 15": 0.0}, "again": 0.0, "added": 0, "wal": 0, "backup": size, "temporary": 0,
            "lines": lines}


def _cost_14(con, path: Path) -> dict:
    """The cost of the steps to formats 14 and 15 (Check.cost) and its lines, from the counts of the current texts:
    the step to format 15 writes one row, which takes no time to speak of."""
    current = con.execute("SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'status'").fetchall()[0][0]
    revisions = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    counting, building = revisions / RATE_14_SCORES, current / RATE_14
    seconds, added = {"13 to 14": counting + building, "14 to 15": 0.0}, current * LENGTH_ENTRY_14
    size = path.stat().st_size
    lines = ["Cost of the steps 13 to 14 and 14 to 15:",
             f"  counting the values of the scores over {revisions:,} revisions: about {counting:,.0f} seconds "
             f"({counting / 60:,.1f} minutes)",
             f"  building the index texts_length of {current:,} current texts: about {building:,.0f} "
             f"seconds ({building / 60:,.1f} minutes)",
             "  writing the schema of the kind conversation (14 to 15): one row of kinds, under a second",
             f"  temporary folder: SQLite sorts the index's entries in it (SQLITE_TMPDIR, else /var/tmp, /usr/tmp or "
             f"/tmp), about {_size(added)} of free disk",
             f"  disk added: about {_size(added)}, the index",
             f"  WAL at its largest: about {_size(added)}, the index's pages, cut back to {_size(JOURNAL_LIMIT)} "
             "after the step",
             f"  backup: the next `depot backup` sends a full copy of the file, about {_size(size + added)}, "
             "onto the dock (the step marks the file for a full copy, since change files record no layout)"]
    return {"seconds": seconds, "again": 0.0, "added": added, "wal": added, "backup": size + added,
            "temporary": added, "lines": lines}


def _sampled_row_bytes(con, rows: int, step: int) -> float:
    found = [one for (one,) in con.execute(
        "SELECT length(CAST(parts AS BLOB)) + 40 FROM content_lists WHERE rowid IN (SELECT value FROM json_each(?))",
        (_json([at * step + 1 for at in range(SAMPLED + 1) if at * step + 1 <= rows]),))]
    return sum(found) / len(found) if found else 0.0


def _sampled_taggings(con, texts: int) -> tuple[int, int]:
    """About (the rows of taggings, their bytes) the stage taggings writes for the texts, from the revisions of the
    texts of SAMPLED revisions spread over texts_revisions by rowid, each text weighed by one over its revisions,
    since a text of many revisions is picked more often; a row's bytes, its index entry's included, about twice its
    text's and tag's names and TAGGING_ROW_BYTES."""
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    step = max(1, last // SAMPLED)
    picked = {text for (text,) in con.execute(
        "SELECT id FROM texts_revisions WHERE rowid IN (SELECT value FROM json_each(?))",
        (_json(list(range(1, last + 1, step))[:SAMPLED + 1]),))}
    weights = rows = size = 0.0
    for text in picked:
        revisions = con.execute("SELECT id, revision, change, tags FROM texts_revisions WHERE id = ? "
                                "ORDER BY revision", (text,)).fetchall()
        made = [row for found, _, _ in _tagged_13(revisions) for row in found]
        weight = 1 / max(1, len(revisions))
        weights += weight
        rows += weight * len(made)
        size += weight * sum(2 * (len(text.encode()) + len(tag.encode())) + TAGGING_ROW_BYTES for _, tag, *_ in made)
    if not weights:
        return 0, 0
    return int(texts * rows / weights), int(texts * size / weights)


def _tagsets(con, current: int) -> int:
    """About how many sets of tags the current plain texts have, from SAMPLED of them: those seen, and those seen
    once standing for as many unseen in the rest (Good and Turing's estimate)."""
    last = con.execute("SELECT coalesce(max(rowid), 0) FROM texts_revisions").fetchall()[0][0]
    step = max(1, last // SAMPLED)
    found = {}
    for (tagset,) in con.execute(
            "SELECT (SELECT json_group_array(g.value ORDER BY g.value) FROM (SELECT DISTINCT value FROM "
            "json_each(t.tags)) g) FROM texts_revisions t JOIN content_lists c ON c.id = t.id AND "
            "c.version = t.content_version WHERE t.rowid IN (SELECT value FROM json_each(?)) AND t.latest AND "
            "coalesce(c.kind, 'plain') = 'plain'", (_json(list(range(1, last + 1, step))[:SAMPLED + 1]),)):
        found[tagset] = found.get(tagset, 0) + 1
    seen = sum(found.values())
    if not seen:
        return 0
    once = sum(1 for count in found.values() if count == 1)
    return min(current, int(len(found) + once / seen * max(0, current - seen)))


def _size(value: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:,.0f} {unit}" if unit == "bytes" else f"{value:,.1f} {unit}"
        value /= 1000
    return f"{value:,.1f} TB"


def _write(legacy: DeltaFolder, target: Path, upstream: str | None):
    """Write the folder's tables into the new file target, as the module says."""
    working = target.with_name(target.name + UPGRADING)
    db = Database(working)
    if not working.exists():
        db.create()
    try:
        if not db.reader().execute("SELECT 1 FROM about_revisions WHERE format = ?", (WRITTEN,)).fetchall():
            _fill(db, legacy, upstream)
    finally:
        db.close()  # the last connection to close folds the WAL into the file and removes it
    for suffix in ("-wal", "-shm"):
        if working.with_name(working.name + suffix).exists():
            raise Refused(f"{working} is still open by another process")
    working.rename(target)
    waiting = working.with_name(working.name + ".waiting")
    if waiting.exists():
        waiting.rename(target.with_name(target.name + ".waiting"))
    Database(target).mark_full_copy()


def _fill(db: Database, legacy: DeltaFolder, upstream: str | None):
    """Every data file not yet in, then the upgrade's own change."""
    with db.bulk() as con:
        con.execute("CREATE TABLE IF NOT EXISTS upgrade_progress (table_name TEXT NOT NULL, file TEXT NOT NULL, "
                    "PRIMARY KEY (table_name, file))")
    done = {(table, file) for table, file in db.reader().execute("SELECT table_name, file FROM upgrade_progress")}
    for table, into in TARGETS.items():
        rows = PARTS_ROWS if table == "parts" else ROWS
        for file in legacy.files(table) if table in legacy.exist() else []:
            name = Path(file).name
            if (table, name) in done:
                continue
            with db.bulk() as con:
                for batch in legacy.batches(table, file, rows):
                    _insert(con, into, [_converted(table, one) for one in batch])
                con.execute("INSERT INTO upgrade_progress VALUES (?, ?)", (table, name))
    with db.bulk() as con:
        _finish(con, legacy, upstream)
        con.execute("DROP TABLE upgrade_progress")


def _converted(table: str, row: dict) -> dict:
    """A row of a format 3 table as its format 4 table holds it."""
    found = {}
    for name, value in row.items():
        if table == "contents" and name == "parts":
            value = json.dumps([{"hash": one["hash"].hex(), "length": one["length"]} for one in value],
                               separators=(",", ":"))
        elif table == "erasures" and name == "erased" and value is not None:
            value = _json([{**one, "hash_a": one["hash_a"].hex(), "hash_b": one["hash_b"].hex()} for one in value])
        elif table == "parts" and name == "value":
            value = pack(value)
        elif table == "contents" and name == "kind" and value is None:
            value = PLAIN_12[0]  # the code's content_lists holds a kind on every row
        elif name in LISTS:
            value = None if value is None else _json(value)
        elif isinstance(value, datetime):
            value = value.isoformat()
        elif isinstance(value, bool):
            value = int(value)
        found[name] = value
    if table in ("texts", "libraries", "erasures", "datasets"):
        found["latest"] = 0
    if table in ("decisions", "notices", "pins"):
        found["change"] = 0  # made from the journal and the libraries once every table is in (_finish)
    return found


def _insert(con, table: str, found: list[dict]):
    if found:
        names = list(found[0])
        con.executemany(f"INSERT OR IGNORE INTO {table} ({', '.join(names)}) "
                        f"VALUES ({', '.join(':' + one for one in names)})", found)


def _finish(con, legacy: DeltaFolder, upstream: str | None):
    """Make the format 4 columns and tables the folder lacks, then the upgrade's own change."""
    for table in ("texts_revisions", "libraries_revisions", "erasures_revisions", "datasets_revisions"):
        con.execute(f"UPDATE {table} SET latest = 1 WHERE revision = "
                    f"(SELECT max(r.revision) FROM {table} r WHERE r.id = {table}.id)")
    con.execute("UPDATE pins SET change = coalesce((SELECT min(l.change) FROM libraries_revisions l "
                "WHERE l.id = pins.library AND l.locked_at IS NOT NULL), 0)")
    named = {one: None for (one,) in con.execute("SELECT id FROM decisions UNION ALL SELECT id FROM notices")}
    made = {}
    entries = []
    for batch in _journal(legacy):
        for entry, change in batch:
            text = json.loads(change)
            kind = text.get("kind", "change")
            decision = text.get("decision") if isinstance(text.get("decision"), dict) else None
            entries.append((entry, kind, decision.get("made_at") if decision else None))
            for one in _ID.findall(change):
                if one in named and named[one] is None:
                    named[one] = entry
    about = con.execute("SELECT change FROM about_revisions").fetchall()
    highest = con.execute("""SELECT max(n) FROM (
        SELECT max(change) AS n FROM about_revisions UNION ALL SELECT max(change) FROM texts_revisions
        UNION ALL SELECT max(change) FROM content_lists UNION ALL SELECT max(change) FROM libraries_revisions
        UNION ALL SELECT max(change) FROM erasures_revisions UNION ALL SELECT max(change) FROM datasets_revisions
        UNION ALL SELECT max(joined_change) FROM memberships UNION ALL SELECT max(left_change) FROM memberships
        UNION ALL SELECT max(change) FROM pulls)""").fetchall()[0][0] or 0
    number = max([highest, *(entry for entry, _, _ in entries)]) + 1
    for one, entry in named.items():
        made[one] = number if entry is None else entry
    con.executemany("UPDATE decisions SET change = ? WHERE id = ?", [(entry, one) for one, entry in made.items()])
    con.executemany("UPDATE notices SET change = ? WHERE id = ?", [(entry, one) for one, entry in made.items()])
    known = {entry for entry, _, _ in entries}
    con.executemany("INSERT OR IGNORE INTO changes (id, kind, made_at) VALUES (?, ?, ?)", entries)
    con.executemany("INSERT OR IGNORE INTO changes (id, kind, made_at) VALUES (?, 'about', NULL)",
                    [(change,) for (change,) in about if change not in known])
    current = con.execute("SELECT * FROM about_revisions ORDER BY revision DESC LIMIT 1").fetchall()[0]
    columns = [one[1] for one in con.execute("PRAGMA table_info(about_revisions)")]
    row = dict(zip(columns, current))
    name = row["name"]
    con.execute("INSERT INTO changes (id, kind, made_at) VALUES (?, 'upgrade', ?)",
                (number, now()))
    con.execute("INSERT INTO about_revisions (revision, change, name, maintainer, takedown_contact, upstream, format) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", (row["revision"] + 1, number, name, row["maintainer"],
                                                 row["takedown_contact"], upstream, WRITTEN))
    own = ids.carries_sql(name)
    counted = {
        "texts": f"SELECT max({ids.number_sql()}) FROM (SELECT id FROM texts_revisions UNION SELECT j.value ->> 'text' "
                 "AS id FROM erasures_revisions e, json_each(e.needles) j) WHERE " + own,
        **{kind: f"SELECT max({ids.number_sql()}) FROM {table} WHERE {own}" for kind, table in (
            ("decisions", "decisions"), ("notices", "notices"), ("libraries", "libraries_revisions"),
            ("erasures", "erasures_revisions"), ("datasets", "datasets_revisions"))}}
    for kind, query in counted.items():
        last = con.execute(query).fetchall()[0][0]
        if last is not None:
            con.execute("INSERT INTO numbers (name, last) VALUES (?, ?)", (kind, last))


def _journal(legacy: DeltaFolder):
    """(id, change) of every entry of the journal, a batch at a time."""
    for file in legacy.files("journal"):
        for batch in legacy.batches("journal", file, ROWS):
            yield [(one["id"], one["change"]) for one in batch]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_plain)


def _plain(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"{type(value).__name__} is no JSON value")
