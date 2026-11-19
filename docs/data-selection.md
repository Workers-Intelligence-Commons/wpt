# Data selection

This document covers what WPT trains on: the texts of a depot, the
libraries that group them, the datasets that say what to train on, and
the corpus a dataset is built into. How a depot stores them is in
[architecture.md](architecture.md), "The depot"; which licenses a text may
carry and how decisions work is in [licenses.md](licenses.md).

## Texts

A text is one piece of final training text, such as an article, a book
or a repository with all its files, with its metadata. Each text is a row of a depot's
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

A text holds final text only. A text its source publishes as text, such
as a code file, a record's text field or a plain-text edition, is kept
exactly as the source gives it. When a source publishes a work both as
text and as a page, the text edition is taken. Only a page a source gives
as HTML or wikitext alone is converted, to Markdown that marks its
headings, paragraphs, lists, emphasis, quotes and tables. That Markdown
escapes no character and breaks no line with a backslash. Where a
text came from survives in its address, so changing how a text was made
means making it again from its origin.

A text never changes by itself. A newer version of a work is another
text, under review. An older one stays where it is until someone refuses
it or asks for its erasure ([erasure.md](erasure.md)).

Every change of a text but a decision on it makes a new revision of it: a
new row under the same id, with the next revision number (1, 2, 3...) and
the depot's change number ("Change numbers", below). Its rows before stay
as they were, and the highest revision is the current one. A revision
holds the small fields (licenses, notice, credits, tags, scores) and the
number of the content version it reads, never the content itself. A
decision on a text is linked to it with the change number it was made at,
and the current revision stores the status its decisions make, so a
decision changes that status and makes no revision.

A text's content is stored apart, as content versions numbered 1, 2,
3..., each with the change number it was made at. A content version never
changes but by an erasure, which cuts its passages from every version
([erasure.md](erasure.md)). Each content version is stored as the ordered
list of its parts, so a text of any size is one text
([architecture.md](architecture.md), "Contents").

Its licenses change only through `wpt depot relicense`, which starts its
decisions over ([licenses.md](licenses.md), "Who decides").

Its content changes through `wpt depot content ID FILE` or the content of
the Texts page of `wpt depot web`: a plain text in a text editor whose find
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

The plain texts of many texts change at once through `wpt depot
content-file FILE`, or `POST /api/texts/content`
([architecture.md](architecture.md), "Two interfaces"). FILE holds one
JSON object per line, one line per text,

    {"id": "my-depot-17", "text": "The text as its source publishes it.", "was": 1}

with `text` the text's new plain text and `was`, optional, the content
version the line was written from: the depot changes the text only while
that is still its current content version, and otherwise leaves it out, so
a file written from an earlier reading never undoes a change made since.
Each text the change changes gets what a change of its content alone gives
it: a new content version, a new revision that reads it, its decisions
started over, so it is under review again and leaves every open library,
and a locked library keeps reading the content version of the revision it
pinned. Its credits, licenses, tags and scores stay as they are. The
change is one decision under review for each pair of old and new content
versions, on the texts it moves, with the person's declaration and the
reason `content changed from version OLD to version NEW`. A text whose
content is that already is left out, and the command and the route report
each decision, how many texts changed, how many were left out because
their content is that already, and how many because their `was` no longer
holds, with the first of those. An item's content changes one item at a
time, so a change that names an item is refused. One change names at most
350,000 texts, and is refused whole, writing nothing, for the same reasons
as a change of the credits of many texts below, and when a new text holds a
passage an erasure removed from that text. The depot reads the lines a few
at a time and stores each text in turn, so the program's memory is that of
one text whatever the number of texts.

Its credits, tags and notice change through `wpt depot credits ID` or
"Edit credits" on the Texts page, whose review panel also toggles
tags. The change is a decision under
review, with the person's declaration, whose reason the depot writes,
naming each credit's old and new value. The decisions before it keep
counting, so the text keeps its status and its libraries
([licenses.md](licenses.md), "Who decides"). A changed notice is stored
once, as on an add.

The credits of many texts change at once through `wpt depot credits-file
FILE`, or `POST /api/texts/credits` ([architecture.md](architecture.md),
"Two interfaces"). FILE holds one JSON object per line, one line per text,

    {"id": "my-depot-17", "title": "owner/repo/src/main.py", "authors": ["Ann Lee"]}

with any of the credits `title`, `authors`, `tags`, `address` and
`notice`. A credit given replaces the text's, `null` clears it, and a
credit left out stays as it is. Each text the change changes gets a new
revision, and the change is one decision under review on all of them, with
the person's declaration, whose reason the depot writes naming the credits
that changed (`credits changed: title, authors`). As for one text's
credits, the decisions before it keep counting, so the texts keep their
status and their libraries, and a locked library keeps reading the
revision it pinned. A text whose credits would stay as they are is left
out of the change, and the command and the route report how many texts
changed and how many were left out.

A line may also name the values its writer read of some credits, under
`was`:

    {"id": "my-depot-17", "title": "owner/repo/src/main.py", "was": {"title": null}}

The depot changes the text only while each credit named there still has
that value, `null` meaning none, and otherwise leaves the text out. A file
written from an earlier reading of the depot therefore never undoes a
change made since.

One change names at most 350,000 texts. The change holds the depot's
lock while it writes, so a larger set of texts is sent as several changes,
each its own decision, and the depot's other writers wait for one change
at a time only. The depot refuses the whole change, writing nothing, when
a line is not of that shape, when it names more texts than that, when no
text would change, when a text is named twice or is no text of the depot,
or when a text belongs to the upstream of a clone. The depot checks and
applies a change where its lines are, so the program's memory stays the
same whatever the number of texts.

### Titles from a text's source

A text's title comes from its source, by the rule of the kind of source
it came from. Most rules read the text's own row: its address, its notice
or its own text; Project Gutenberg's rule reads the project's catalogue. A
rule that finds nothing leaves the title empty; no title is made up. A
text that has a title keeps it.

- A code file from a dataset of source files (Stack V2 Edu, Stack V2
  HTML), whose address is
  `https://raw.githubusercontent.com/<owner>/<repo>/<commit>/<path>`:
  `<owner>/<repo>/<path>`, as the address writes it.
- A GitHub repository, whose address is `https://github.com/<owner>/<name>`:
  `<owner>/<name>`.
- A GitHub issue or pull request (GitHub Archive): the first line of its
  text, when a blank line or the end of the text follows it.
- A research paper of peS2o: the first line of its text, when a blank
  line or the end of the text follows it. When every line of the text's
  first paragraph but the last ends in a backslash, a hard line break, the
  paragraph is the title, its lines joined by a space.
- A Python Enhancement Proposal: the `Title:` field of the header that
  starts its text, up to the header's next field.
- A Standard Ebooks book: the first `dc:title` of the
  `=== src/epub/content.opf` block inside its text.
- A Bible translation of eBible.org: its notice's text before
  `, eBible.org`.
- A Project Gutenberg ebook: the title Project Gutenberg's catalogue
  `pg_catalog.csv` (https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv)
  gives the ebook number in the text's address, the last folder of the
  address made of digits alone. A catalogue title of several lines is
  its lines, each trimmed, the empty ones skipped, joined by `: `.
- A PubMed Central article has no title.

A title read from a text is read as its Markdown shows it: a backslash
that escapes a punctuation character is dropped, a hard line break (a
backslash at the end of a line) is a space, and spaces at both ends are
trimmed.

A title read from the first line or paragraph of a text is left empty when
it is longer than 300 characters or holds `\documentclass`: such a line
runs into the text's body.

An author stored as the text of a Python dict, `{'first': 'Jean M',
'last': 'Feugang'}`, is the name its parts make, `Jean M Feugang`: the
parts that are not empty, in the order first, last. An author whose parts
are both empty is dropped, and a text left with no author has none.

`scripts/source_credits.py` reads a depot, read only, and writes the
files of `wpt depot credits-file` by these rules, one family of sources
at a time, in files of at most 350,000 lines. Each line names under
`was` the title or authors it read, so a title set after the reading
stays. It learns which source each text came from from the ledgers of the
run that added them.

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

One score is set on, or removed from, every text a condition matches
through `wpt depot score NAME (--value V | --remove) WHERE`, or `POST
/api/texts/score` on a selection. A score it sets is made from each text's
current content version. It is recorded as a change of scores is, one
decision under review on every text it changes. `wpt depot score-names`,
or `GET /api/scores`, lists every name of a score a text has, with how
many texts have it, read from the depot's counts at once.
`wpt depot score-values NAME`, or `GET /api/scores/{name}/values`, gives
the values the score has on the current texts among the latest revisions,
at most `SCORE_READ` of them (`wpt_depot/depot.py`): how many have it,
its smallest and largest value, and its most frequent values. The review
panel of the Texts page offers them as its choices.

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
and optional shares. Each library it reads is named by its id, the
change number it is read at, and the address of its depot. A library the
depot holds has no address, which means the depot that holds the row, so
the row reads the same in a clone. A library of another depot has that
depot's address: a file, or the web address of its `wpt depot web`, which
is asked about the library when it is named. The change number is the
owning depot's change number when the library was named. In a clone, a
library of the upstream is read at the upstream change the clone's pulls
brought ([architecture.md](architecture.md), "Following an upstream").
A dataset never picks single texts: it takes every text of its libraries
that its shares pick.

`wpt depot dataset new NAME --tokenizer RECIPE --library [ADDRESS:]ID ...
[--share TAG[=SHARE] ...] [--max-tokens N]` makes one. The tokenizer recipe
is one of the depot's ([tokenizer.md](tokenizer.md)); it may be made after
the dataset, and a build needs it.

A corpus is built at a size, `max_tokens`: the most tokens the corpus
holds, or none. The dataset's own `max_tokens`, optional, is only the size
the web app's build form starts at; nothing else reads it.
The size counts the tokens of its plain texts, conversations and tasks
together, and a corpus of any size keeps the dataset's mix of items and
plain text.

Conversations and tasks together have their natural share of the size:
the share they hold of the UTF-8 bytes of the corpus built without a size
(every conversation and task, and the plain texts the shares keep, see
"Shares"), rounded down to a whole token. So every size has the mix of the
corpus of every token. A share no text has counts for nothing there. The
share is measured in bytes because bytes are known for every text before
anything is tokenized. The bytes are those of the figures the dataset's
revision stores, which count a text two libraries hold once for each, and a
plain text with several of the shares' tags once for each tag. When a
revision's figures are not known, the build counts the same figures from
every text of its libraries, which takes long on a big dataset, so the same
depot state builds the same corpus either way. The
items fill that share in the corpus's order (see "Filling a part" below).
A conversation counts the tokens of its chat template. A task is kept as a JSON line and
never tokenized, so it counts the tokens of its content written as the JSON
text of that line.

Each share then fills its proportion of what the items left, or, without
shares, the plain texts fill all of it, in the corpus's order. Without a
size, every conversation and task is taken. The corpus's record keeps the
tokens its conversations and tasks hold (`items_tokens`).

`wpt depot dataset edit NAME`, or "Edit it" on the view of the dataset,
changes its name, its libraries, its shares and its `max_tokens` as a new
revision, its id and its tokenizer recipe kept, with the checks of a new one. A library it
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

A dataset publishes at most one vocabulary: the sha256 of one of the
vocabularies of the depot that holds the dataset, whatever depots its
libraries are of. `wpt depot dataset edit NAME --vocabulary SHA256` (or
`--no-vocabulary`), or the vocabulary part of the dataset's view, sets or
clears it as a new revision; a sha256 the depot holds no vocabulary of is
refused. Training takes the corpus of that vocabulary (see "Corpora"
below). A build never reads it.

### Shares

Shares are optional and ordered, each a tag and a share of the tokens
(`0.3` is 30%), each share above 0 and at most 1, each tag once. A share
given without a number takes an even part of what the shares given as
numbers leave of 1, and the dataset stores that number: `research-paper`
0.35, `books` 0.15, `government` 0.05, `code` 0.05 and eight language tags
without a number store 0.05 for each language. The parts are rounded to 12
decimals. A share without a number is refused when the numbers reach 1
already. A text counts for the first share whose tag it has. A text that has none of the
shares' tags is left out. Without shares, every text of the libraries is
taken.

Inside a share, its part splits evenly over every tag its texts carry,
the share's own tag excepted: each such tag is a group, and the texts that
carry no other tag are one group more. A text counts for every group of
its tags. Every tag counts alike, length tags such as `long` or `huge`
included: the rule reads the texts' tags and nothing else. A share whose
texts make a single group fills in the corpus's order (see "Filling a
part"), and so does the share the build keeps whole without a size.

The build brings the shares to their proportions, and each share's groups
to even parts of it, without repeating a text.
Without `max_tokens`, it finds the share with the fewest bytes relative to
its proportion and keeps every text of that share; each other share fills
its proportion of the scarcest share's bytes, in the corpus's order (see
"Filling a part"), a piece of a long text counting for its share of the
text's bytes, split evenly among its pieces. The bytes of each share come from
the figures, as the items' share does. A text the figures count twice
makes its shares hold more bytes there than the build can take of them, so
the shares and the items' share then come out off their proportions. With a size, each share fills
its part of it. A text enters a corpus once at most, whole or as some of
its pieces. The order depends on
the texts' ids alone, so the same dataset builds the same corpus, and a
change of its name, of its `max_tokens` or of the build's size keeps its
order. The
corpus's record keeps how many tokens each share holds (`shares_tokens`),
with each of its groups, and how many texts the build left out (`left_out`: texts no share took, texts
of which a share took nothing, items `max_tokens` left out, items of the kinds that
are no training data).

A share with no text refuses the build. Shares apply to plain texts only:
only a size trims conversations and tasks, to their natural share of it.

### Filling a part

Every part of a corpus fills the same way: the items' share, each share
of the plain texts, and each share of a vocabulary's sample. The texts
come in the corpus's order. A text that fits what is left of its part is
taken. A text that would pass it is left out, and the next text is tried.
A part closes once what is left of it is at most a thousandth of it, or
once a thousand of its texts in a row did not fit. The build stops reading
once every part is closed, so a nearly full corpus never reads the rest of
the dataset for its last tokens. A smaller size takes the same texts as a
larger one up to the first text it leaves out; past it, it takes only what
fits what is left of it.

A share whose texts make two groups or more (see "Shares") fills by
group instead. Each group's texts wait in the corpus's order. The share
always takes the next text of the group that holds the fewest tokens so
far; when several groups hold the fewest, the group whose next text comes
first in the corpus's order goes first. A text taken counts for every one
of its groups, and is taken once. A text that does not fit what is left of
the share is left out for good. A group whose texts run out drops out, and
so does a group that meets a thousand texts in a row that do not fit. The
share closes once what is left of it is at most a thousandth of it, or
once every group dropped out. So a group with few texts gives all it has,
and the other groups share what it leaves.

To choose so, the build first walks the whole order, reading each text's
tags and size but not its content, and keeps what the choice needs in
compact arrays. With a size, it chooses on the texts' exact tokens, which
worker processes count a few texts ahead of each group; the texts it
chooses are tokenized again as the build writes them. Without a size, and
for a vocabulary's sample, it chooses on bytes. The build keeps the list of
the texts chosen in its work folder, so a build cut short takes the same
texts when it goes on, and chooses them again, the same, when the list is
lost. A smaller size takes about the same share of each group as a larger
one, but not the start of the larger one's texts.

### Long texts

A plain text longer than 1,048,576 characters, such as a repository the
depot holds as one text, comes into a corpus as pieces of about that
length (`wpt_depot/pieces.py`). The depot keeps the text whole. A piece
ends right before a line that begins with `=== `, the line that begins
each file of a repository, when one comes within the next piece's length;
otherwise before the first line start there, or at the length itself when
the text has no line break there. Each piece has its own place in the
corpus's order and counts as a text of its own for every part, so a long
text spreads through the corpus, and a corpus of any size takes about its
share of it. Each piece waits in the groups of its text's tags as a text of
its own. An item is never cut.

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

`wpt depot build DATASET [--vocabulary SHA256] (--max-tokens N | --all)`
builds the dataset's corpus into the store
(`WPT_STORE`): what training reads. It runs on the machine that holds the
depot, or as a job of `wpt depot web` ([depot-web.md](depot-web.md)). A
rented GPU machine only reads the corpus and trains
([machines.md](machines.md)). A build is one dataset, one vocabulary and
one size. The vocabulary is one the depot holds already, by its sha256
([tokenizer.md](tokenizer.md)): by default the one the dataset
publishes, and a dataset that publishes none refuses a build that names
none. Without it nothing is built: a build never trains a vocabulary. The size is always named: a number of tokens,
or `--all`.

The build is a stream. It reads each library in the corpus's order
through an index, so it reads only the texts it reaches, and a build at a
small size starts at once whatever the dataset's size. A dataset with
shares is walked once whole before the stream, to choose each share's
texts by group (see "Filling a part"), which reads every text's row but
only the contents of the texts whose tokens the choice needs. Worker processes
tokenize the texts, and the build takes their results in that order. It
uploads each token file to the store as soon as it is full, with the part of the `texts` table that
lists its texts, and deletes its local copies, so the memory and the local
disk it takes stay bounded whatever the dataset's size. A build cut short
goes on from the first text after the last token file uploaded and writes
the same bytes as a build that never stopped.

The build takes every text of the dataset's libraries, each library at
the change number the dataset names; a text two libraries hold is taken
once. It chooses nothing by score. It then keeps:

- every plain text, as its tokens followed by the end-of-document token,
  in token files; a long text as its pieces, each a row of its own, named
  by its text's id, `#` and its number (`depot-12#3`);
- every conversation, as the tokens of the chat template
  ([chat-format.md](chat-format.md), "The template"), in token files of
  their own;
- every task, as one line of `tasks.jsonl`.

Items of other kinds are not training data: the build leaves them out and
counts them in the record. The texts go in the order of their export keys,
a hash of each text's id that its depot stores, the order the dataset's
export uses too, so that any first part of the corpus is drawn from all of
it. A piece's key is the same hash of its name. The build finds the long
texts before it reads, through an index of the depot's long content
versions, so it never reads every text to find them. The order depends on no dataset and not on where a depot's file is.
Texts of one key go by the place of their depot among the depots the
dataset reads, then by their id. Every size of a model series reads the same corpus, the smaller sizes
its first part ([model.md](model.md), "Model series").

A corpus names a depot by what no place of its file changes. The depot
that holds the dataset is named by its public address, the web address of
its `wpt depot web` that the setting `WPT_DEPOT_URL` gives, or by its name
when it has none; a clone names a dataset its upstream owns by the
upstream's web address. Another local depot is named by the name in its
own settings, never by its file. The corpus's record, its build's key and
the erasures it counts, its copy of the dataset's row, its token files,
its `texts` table and `tasks.jsonl` all carry those names, so the same
depot state builds the same corpus wherever its files are, and a clone
builds its upstream's corpus again.

Two depots a corpus reads never carry one name: a build where another
depot's name is the one the corpus gives the dataset's own depot, or where
two other depots share a name, is refused. A build of the same corpus
again finds each other depot's file from the dataset's libraries at the
revision the corpus took.

A corpus says everything about what it holds, even once the depots it
read change or vanish. Beside its token files and `tasks.jsonl`, it holds
these tables:

- `dataset`: the dataset's row, copied, each library of another depot named by that depot's name, its
  `max_tokens` the build's size, with the time of the build;
- `texts`: every text it took, in parts, one for each token file, with its depot's name (above), its id there,
  its kind, its credits, licenses and tags copied, its scores copied with
  the content version each was made from, its notice, and where its
  tokens start (the steps that read conversations and tasks pick them by
  tag from this table); a piece taken has a row of its own with its
  text's credits and its number (`piece`, empty for a whole text);
- `licenses` and `notices`: the license texts and notices those texts
  name, copied from their depots (ids carry their depot's name, so they never
  repeat across depots).

Everything goes under `corpora/<dataset name>-<first 8 hex characters of
the build's key>/` in the store, with the record, `corpus.json`, written
last, so a build that fails leaves no corpus. The build's key covers the
depot, the dataset's id and row, the vocabulary, the size and the erasures
carried out on a text the dataset's libraries hold or held, so another
vocabulary or size builds into another folder, building a deleted corpus
again writes into its folder, and an erasure of a text the dataset never
read changes neither. The record names the vocabulary's sha256 and the
size. Each share of its `shares_tokens` lists its groups (see "Shares"),
each with its tag, null for the texts that carry no other tag, how many
texts it holds and their tokens; a text counts for each of its groups, so
the groups' tokens add up to more than the share's when texts carry
several tags. The corpus id covers the
dataset's id and its row (without its revision and change), the
vocabulary and every file's content (of each text: its depot, id, notice,
where its tokens start and, when it has any, its scores), so building the
same corpus again finds it and writes nothing, and another dataset with
the same content
gets a corpus of its own. The record also gives a
fingerprint of each part a training step reads, `text`, `conversations`
and `tasks` ([training.md](training.md), "Step results and reuse").

The store keeps only the latest fresh corpus of each dataset, vocabulary
and size: a new corpus deletes the one before of the same three once it is
whole, and leaves the corpora of the dataset's other vocabularies and sizes
as they are. A corpus of an older revision of the dataset, built again,
deletes none of a newer revision, and the latest corpus of a dataset is
the last built of its newest revision. A cap (`WPT_CORPORA_CAP`) bounds the bytes all corpora take
together, the files of a folder left without a record by a build that
stopped included. To make room, a build deletes those files first, then
the corpus downloaded least recently, a file at a time, never one being
built or downloaded. Before its first upload, a build estimates its whole
size from its first files, and a corpus bigger than the cap fails then,
before any other corpus is deleted for it. A build whose uploaded files
were deleted meanwhile writes no record, and started again it starts over.
A deleted corpus keeps its record, and building it again from that record
(its dataset revision, vocabulary and size) says when an erasure changed
its id.

An erasure carried out while a corpus is built is checked once the build
ends: when it reached a text the corpus took, the corpus is written stale
and deletes no other corpus.

A model recipe's data part names a depot by the name its corpora carry, a
dataset in it by id or name, and the total tokens ([model.md](model.md),
"Model recipes"). `wpt train` reads the corpus it is given, or else the
latest fresh corpus of the triple that dataset, the vocabulary it publishes
and the total tokens as the size. A dataset that publishes no vocabulary,
or a triple not built, refuses the run with the triple and the dataset's
download page in the depot's web app (`<depot address>/#/download/<dataset
id>`); training never builds or queues anything. A text added to the depot
after a build reaches training only once the corpus is built again.

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
