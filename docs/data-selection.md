# Data selection

This document covers what WPT trains on: the texts of a depot, the
libraries that group them, the datasets that say what to train on, and
the corpus a dataset is built into. How a depot stores them is in
[architecture.md](architecture.md), "The depot"; which licenses a text may
carry and how decisions work is in [licenses.md](licenses.md).

## Texts

A text is one piece of final training text, such as an article, a book
or a code file, with its metadata. Each text is a row of a depot's
`texts` table:

- its text, or, for a structured item, its kind and its content
  ("Items", below);
- its licenses: one or more ids of the depot's licenses, as a choice, so
  one allowed license is enough ([licenses.md](licenses.md), "Licenses
  and notices");
- its notice: the copyright notice exactly as the source gives it, or the
  proof that the work is in the public domain, stored once in the depot
  and shared by every text that gives it;
- its credits: a title, its authors, and the address it came from, each
  optional;
- its tags ("Tags", below);
- its scores ("Scores", below);
- its decisions, in order, and the libraries that hold it.

A text holds final text only. Plain text stays as written, with no
Markdown escaping and no backslash line breaks. A structured page keeps
its headings, paragraphs and lists, and code keeps its exact file. Where a
text came from survives in its address, so changing how a text was made
means making it again from its origin.

A text never changes by itself. A newer version of a work is another
text, under review. An older one stays where it is until someone refuses
it or asks for its erasure ([erasure.md](erasure.md)).

Every change of a text makes a new revision of it: a new row under the
same id, with the next revision number (1, 2, 3...) and the depot's
change number ("Change numbers", below). Its rows before stay as they
were, and the highest revision is the current one. A revision holds the
small fields (licenses, notice, credits, tags, scores, decisions) and the
number of the content version it reads, never the content itself.

A text's content is stored apart, as content versions numbered 1, 2,
3..., each with the change number it was made at. A content version never
changes but by an erasure, which cuts its passages from every version
([erasure.md](erasure.md)).

Its licenses change only through `wpt depot relicense`, which starts its
decisions over ([licenses.md](licenses.md), "Who decides").

Its content changes through `wpt depot content ID FILE` or the editor of
the workspace of `wpt depot web`: a plain text in a text editor whose find
and replace, by words or a regular expression, works inside that text
only, an item through the form of its kind's schema, which its new content
must still match. Saving is the content change alone, and says so before
the click. The change is a new content version and a new revision that
reads it. It is a decision under review, with the person's declaration,
whose reason the depot writes, `content changed from version OLD to version NEW`. It starts
the text's decisions over, as a license change does
([licenses.md](licenses.md), "Who decides"): the text is under review
again and leaves every open library, to be put back once it is admitted. A
locked library keeps reading the content version of the revision it
pinned. A content that changes nothing writes nothing.

Its credits, tags and notice change through `wpt depot credits ID` or
"Edit credits" beside the text in the workspace. The change is a decision under
review, with the person's declaration, whose reason the depot writes,
naming each credit's old and new value. The decisions before it keep
counting, so the text keeps its status and its libraries
([licenses.md](licenses.md), "Who decides"). A changed notice is stored
once, as on an add.

### Scores

A text's scores are pairs of a name and a decimal number, under any name,
such as `machine-made` 0.03 or `source-rank` 4. Each score also records the
content version of the text it was made from. A text has at most one score
of each name, and a text without scores has none at all.

Scores change through `wpt depot scores FILE`, or `POST /api/texts/scores`
([architecture.md](architecture.md), "Two interfaces"). One change covers
many texts: FILE holds one JSON object per line, one line per text,

    {"id": "my-depot-17", "scores": [{"name": "source-rank", "value": 4, "content_version": 1}], "remove": ["old-rank"]}

with the scores to set and the names of the scores to remove, each part
optional. A score set under a name the text has already replaces it. Each
text the change changes gets a new revision, and the change is one
decision under review on all of them, with the person's declaration, whose
reason the depot writes naming the scores set and removed. Scoring a whole
depot is so one decision and one new revision per text. As for credits,
the decisions before it keep counting, so the texts keep their status and
their libraries, and a locked library keeps reading the revision it
pinned. A text whose scores would stay as they are is left out of the
change, and the command and the route report how many texts changed and
how many were left out. The depot refuses the whole change, writing
nothing, when a line is not of that shape, when no text would change, when
a text is named twice or is no text of the depot, when a value is not a
finite number, when a score names a content version the text does not
have, or when a text belongs to the upstream of a clone. The depot checks
and applies a change where its lines are, so scoring every text of a large
depot takes no more of the program's memory and no larger record of the
change than scoring a few.

A score's value comes from a person or from a program a person chose,
never from the judgment of another company's AI assistant ("Models in
training", below).

Every other change of a text, a new content version included, keeps its
scores as they are, each still naming the content version it was made
from.

### Change numbers

Every change of a depot has a number, one past the last: a text added, a
decision, a revision, a library made, changed or locked, a dataset, an
erasure. Each revision and each membership records the number of the
change that made it, so the depot can be read as it was at any change
number.

`wpt depot add FILE` adds one text, under review, with its licenses,
notice, tags and credits; `wpt depot show ID` shows it with its decisions
and libraries; `wpt depot sql "SELECT ..."` answers any read query over
the depot's tables. `wpt depot web` does the same as pages
([architecture.md](architecture.md), "Two interfaces").

Duplicate texts are kept. Nothing judges a text's content: people decide
on it ([licenses.md](licenses.md), "Who decides").

### Items

A structured item is a text whose content is JSON instead of plain text:
its row holds a `kind` and its `content`, and no `text`. A binary file
inside the content is written in base64. The depot's `kinds` table holds
each kind's meaning and the JSON Schema its content must match, and the
depot checks every item against it when it is added. A new depot starts
with four kinds:

| Kind | What it is | Who reads it |
| --- | --- | --- |
| `conversation` | a conversation in the chat format ([chat-format.md](chat-format.md)) | the `answer-as-assistant` step |
| `task` | a task for agentic training: what it asks, its starting files and its check ([training.md](training.md), "Tasks") | the `finish-tasks` and `finish-tasks-with-less-writing` steps |
| `rating` | a person's pass or fail on one attempt of a `wpt eval` run | no step: a measure only |
| `probe` | a question to ask a model and the patterns that flag a reply as alarming | no step: a measure only |

An item gets the same licenses, notice, credits, tags and decisions as a
plain text.

## Tags

Tags say what a text is. A tag is a language code or a word. Names are
flat: a tag has no parent, and a share for a wider area names one of its
tags ("Shares", below).

Tags are free text for everyone, AI assistants included: a text takes any
tag that says what it is, and a new tag needs no change anywhere first.
The table below lists tags in use, and the notes after it give the meaning
each of them has.

| Family | What it says | Tags |
| --- | --- | --- |
| Form | the kind of document | `novel`, `short-story`, `poetry`, `drama`, `essay`, `letters`, `speech`, `biography`, `memoir`, `diary`, `scripture`, `encyclopedia`, `dictionary`, `textbook`, `research-paper`, `news`, `periodical`, `government`, `manual`, `guidebook`, `conversation`, `web`, `code` |
| Subject | what it is about | `history`, `philosophy`, `religion`, `mythology`, `politics`, `economics`, `society`, `law`, `education`, `language`, `art`, `music`, `travel`, `nature`, `agriculture`, `cookery`, `medicine`, `science`, `biology`, `physics`, `chemistry`, `mathematics`, `astronomy`, `earth-science`, `engineering`, `computing`, `technical`, `warfare` |
| Fiction genre | the kind of story and its readers | `adventure`, `romance`, `mystery`, `science-fiction`, `fantasy`, `horror`, `historical-fiction`, `humour`, `children` |
| Broad group | a wide grouping of books | `literature`, `books` |
| Period | when the text was written | `ancient`, `medieval`, `early-modern`, `19th-century`, `20th-century`, `21st-century` |
| Text state | how the text reached the depot | `scanned`, `translated`, `verse-numbers` |
| Code language | the programming language of a file | `code-<language>` |
| Length | how big the work is | `short`, `medium`, `long`, `huge` |
| File format | the format of the file the text came from | `format-html`, `format-xhtml`, `format-text`, `format-markdown`, `format-code`, `format-records` |
| Structure | what the text holds | `with-images`, `with-tables`, `with-footnotes`, `with-dialogue` |

A word that is also a language code means the language, so a topic never
takes a language code in use: `warfare` has that name because `war` is
Waray's code. The exceptions are `law` and `web`, which are topics.

Meanings the word alone does not give:

- `memoir`: a first-person account of the writer's own life.
- `periodical`: one issue of a magazine or a journal.
- `government`: published by a government or a public body.
- `science`: science with no narrower subject stated.
- `technical`: technical writing outside the other subjects.
- `literature`: literary works, and writing about language and literature.
- `books`: books outside `literature`.
- `ancient`: written before 500. `medieval`: from 500 to 1499.
  `early-modern`: from 1500 to 1799. Each century tag covers its own
  hundred years.
- `scanned`: machine-read from paper, so it can hold reading errors.
- `translated`: a translation of a text first written in another language.
- `verse-numbers`: each verse starts with its reference.
- `code-<language>`: the language's name in Linguist, GitHub's list of
  languages and their file endings, in lower case. `+` becomes `p`, `#`
  becomes `sharp`, and any other character that is not a letter, a digit
  or `-` becomes `-`.
- `short`: under 100 KB. `medium`: from 100 KB to 1 MB. `long`: from 1 MB
  to 1 GB. `huge`: over 1 GB.
- `with-tables`: holds at least one table. `with-footnotes`: holds at
  least one footnote. `with-dialogue`: a large part of it is speech
  between people.

Tags are plain stored data. Nothing computes the Length, File format or
Structure tags: whoever adds a text writes the tags it has, and the depot
keeps them as written.

## Libraries

A library is a named collection of admitted texts inside a depot. Only an
admitted text joins one: a text under review sits in the depot and in no
library. A refusal or a change of its licenses takes a text out of every
open library that holds it ([licenses.md](licenses.md), "Who decides").

A library records when each text joined it and when it left, by change
number. An open library reads each of its texts at its current revision,
and at a past change number it reads the texts it held then, each at the
revision it had then. A locked library reads, for good, the exact revision
of each text it held when it was locked: a newer revision of one of its
texts does not reach it. The page of a locked library lists those
revisions and marks each text that has a newer one, and the page of a text
shows how many revisions it has and which revision each locked library
reads.

- `wpt depot library new NAME` makes an open library.
- `wpt depot library put LIBRARY WHERE` puts into an open library every
  admitted text the SQL condition `WHERE` matches, and names the matched
  texts it left out because they are not admitted.
- `wpt depot library take-out LIBRARY WHERE` takes out of an open library
  every text the SQL condition `WHERE` matches, and names the matched texts
  it did not hold. A text taken out keeps its status: it stays admitted and
  can be put back.
- `wpt depot library rename LIBRARY NAME` gives a library, open or
  locked, a new name. The name is not what it holds, so the change needs
  no declaration and no decision.
- `wpt depot library lock LIBRARY` locks it for good, pinning the current
  revision of each text it holds. No text joins or leaves a locked
  library; only an erasure still takes a text out or cuts its content
  ([erasure.md](erasure.md)). Its texts can still get new revisions,
  which it does not read. A dataset built from a locked library can be
  built again later.
- `wpt depot library fork LIBRARY NAME [--without WHERE]` makes a new open
  library holding the same texts, less those the SQL condition `WHERE`
  matches, and records which library it started from (`forked_from`). It
  starts with no problem.
- `wpt depot library report LIBRARY PROBLEM [WHERE]` reports a problem with
  a library, locked or not, with the reporter's declaration. A library's
  `problems` lists them: the ids of the texts concerned (the texts of the
  library `WHERE` matches, none without `WHERE`), the problem in words,
  and the decision, under review, that records who reported it. A report
  changes no membership and no text, so a locked library takes it. To
  correct an open library, take the texts concerned out of it; to correct
  a locked one, fork it without them.

## Datasets

A dataset is a row of a depot that says what to train on: its name, the
tokenizer recipe whose vocabulary its corpus uses, the libraries it reads,
and optional shares. Each library it reads is named by the address of its
depot (this depot or another), its id there, and the change number of
the depot that owns the library it reads it at, which is that depot's
change number when the library was named. In a clone, a library of the
upstream named by the clone's address is named at the upstream change the
clone's pulls brought ([architecture.md](architecture.md), "Following an
upstream").
A dataset never picks single texts: it takes every text of its libraries
that its shares pick.

`wpt depot dataset new NAME --tokenizer RECIPE --library [ADDRESS:]ID ...
[--share TAG=SHARE ...]` makes one.

`wpt depot dataset edit NAME`, or "Edit it" on the view of the dataset,
changes its name, its libraries and its shares as a new revision, its id
and its tokenizer recipe kept, with the checks of a new one. A library it
read already, named by its depot's address in any form (in a clone, a
library of the upstream by either depot's address), stays as the
dataset holds it, its address and its change number included, and its
depot is not opened, so a dataset imported from a remote whose depots this
machine cannot reach still changes; a library added is read at its depot's
change number now. The change needs no declaration and no decision, since it
decides nothing about a text. A corpus built before keeps its own copy of
the dataset's row, its id and revision included, so it stays what it was;
the view of a dataset lists every corpus built from any of its revisions,
with the revision each was built from.

### Shares

Shares are optional and ordered, each a tag and a share of the tokens
(`0.3` is 30%), each share above 0 and at most 1, each tag once. A text
counts for the first share whose tag it has. A text that has none of the
shares' tags is left out. Without shares, every text of the libraries is
taken.

The build brings the shares to their proportions without repeating a text.
It counts each share's tokens, finds the share that is scarcest relative to
its proportion, and keeps every text of that share. From each other share it
takes a seeded random subset of texts, whose tokens match the proportion
against the scarcest share. A text enters a corpus once at most. The seed
is part of the dataset, so the same dataset builds the same corpus. The
corpus's record keeps how many tokens each share holds (`shares_tokens`) and
how many texts the build left out (`left_out`: texts no share took, texts
trimmed from a share, items of the kinds that shares do not apply to).

A share with no text refuses the build. Shares apply to plain texts only:
conversations and tasks are not trimmed.

### Remotes

A dataset written out as JSON is a remote: the dataset's row without its
id, pointing to the depots and the change numbers its data comes from, so anyone
can build its corpus without the depot that made it.
`wpt depot dataset export NAME` writes one, and
`wpt depot dataset import FILE` makes a dataset from one.

### Building a dataset again

A dataset names each library with a change number, and a depot keeps
every revision and membership, so the dataset builds the same corpus
again whatever its libraries and texts did since. Only an admitted
erasure changes that corpus: the texts it removes are gone, and the
passages it cuts are cut in every revision ([erasure.md](erasure.md)).

## Corpora

`wpt depot build DATASET` builds the dataset's corpus into the store
(`WPT_STORE`): what training reads. It runs on the machine that holds the
depot. A rented GPU machine only reads the corpus and trains
([machines.md](machines.md)). The build needs the lock of the dataset's
tokenizer recipe at its version and the vocabulary that lock names in the
store ([tokenizer.md](tokenizer.md)); without them it builds nothing.

The build takes every text of the dataset's libraries, each library at
the change number the dataset names; a text two libraries hold is taken
once. It chooses nothing by score. It then keeps:

- every plain text, as its tokens followed by the end-of-document token,
  in token files;
- every conversation, as the tokens of the chat template
  ([chat-format.md](chat-format.md), "The template"), in token files of
  their own;
- every task, as one line of `tasks.jsonl`.

Items of other kinds are not training data: the build leaves them out and
counts them in the record. The texts go in a seeded order drawn from the
dataset's row, so that any first part of the corpus is drawn from all of
it. Every size of a model series reads the same corpus, the smaller sizes
its first part ([model.md](model.md), "Model series").

A corpus says everything about what it holds, even once the depots it
read change or vanish. Beside its token files and `tasks.jsonl`, it holds
these tables:

- `dataset`: the dataset's row, copied, with the time of the build;
- `texts`: every text it took, with its depot's address, its id there,
  its kind, its credits, licenses and tags copied, its scores copied with
  the content version each was made from, its notice, and where its
  tokens start (the steps that read conversations and tasks pick them by
  tag from this table);
- `licenses` and `notices`: the license texts and notices those texts
  name, copied from their depots (ids carry their depot's name, so they never
  repeat across depots).

Everything goes under `corpora/<dataset name>-<first 8 hex characters of
the corpus id>/` in the store, with the record, `corpus.json`, written
last, so a build that fails leaves no corpus. The corpus id covers the
dataset's id and its row (without its revision and change), the
vocabulary and every file's content (of each text: its depot, id, notice,
where its tokens start and, when it has any, its scores), so building the
same corpus again finds it and writes nothing, and another dataset with
the same content
gets a corpus of its own. The record also gives a
fingerprint of each part a training step reads, `text`, `conversations`
and `tasks` ([training.md](training.md), "Step results and reuse").

A model recipe's data part names a depot address, a dataset in it, the
tokenizer recipe and the total tokens ([model.md](model.md), "Model
recipes"). `wpt train` reads the corpus last built for that depot and
dataset, or the corpus it is given. A text added to the depot after a
build reaches training only once the corpus is built again.

## Languages and translation

As many languages as possible, plus translation as a capability. Clean
parallel text is scarce. It comes from:

- public-domain works and their public-domain translations, aligned by
  sentence;
- texts published in several languages under a license on the allowed list
  ([licenses.md](licenses.md));
- translations people add to a depot, under their declaration
  ([licenses.md](licenses.md), "Who decides").

## Models in training

WPT's own models may be used in training, their output included: as a
reference a reward compares with, such as a model's own full attempt in
the terse-versus-full reward of the `finish-tasks` step
([training.md](training.md), "Rewards"), as a draft a person edits or
confirms ([chat-format.md](chat-format.md), "Model drafts"), or as
`answer-as-assistant` conversations made of its attempts that passed
their task's check. Every such use says exactly what the WPT model did, so
it is open to review.

Other companies' models stay forbidden as a source and as a judge: no
text they write of their own enters the data or the training, and they
take no role in it as data, a simulator or a judge. Another company's AI
assistant may carry out work on the data that a person directs: set the
scores or other fields that person decided, run the program the person
chose to compute scores, apply an edit the person described exactly (for
example removing stray backslashes), find texts and show them for review,
or sum up what a text says. It may also choose a text's tags. It never
acts as a judge: it does not choose a score by its own judgment, admit or
refuse a text, or decide what a text is worth. It writes no text into the
data unless a person said exactly what to write, and a person reviews what
it wrote. Synthetic data not made by a WPT model is forbidden too; puzzles
and tasks a program generates are not that kind of synthetic data
([licenses.md](licenses.md), "No distillation from another company's
model").

## Model size

The repository is recipes, scripts and environment setups usable for any
model size. Flags and features let a smaller model train on a subset (for
example a computer-use subset for every training step).
