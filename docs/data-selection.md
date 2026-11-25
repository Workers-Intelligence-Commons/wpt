# Data selection

This document covers what WPT trains on: the texts of a depot, the tags
that group them, the snapshots that fix which texts a model may read, and
the corpus a model recipe's data section is built into from a snapshot. How a depot stores them is in
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
- its decisions, in order.

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

Every change of a text but a decision on it or a change of its tags makes
a new revision of it: a new row under the same id, with the next revision
number (1, 2, 3...) and the depot's change number ("Change numbers",
below). Its rows before stay as they were, and the highest revision is the
current one. A revision holds the small fields (licenses, notice, credits,
scores), the number of the content version it reads and that version's
kind, length in characters and bytes, never the content itself. A
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
again. A snapshot made before keeps reading the content version of the
revision it read. A content that changes nothing writes nothing.

The Texts page can also show a plain text translated, read only, in place
of its editor, so a person can judge a text in a language they do not
read. The translation runs in the browser with Mozilla's models, from the
language Firefox's detector finds in the text, else the one a tag names.
Nothing of it is saved: it is never content, data or training material.

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
started over, so it is under review again, and a snapshot made before keeps
reading the content version of the revision it read. Its credits, licenses, tags and scores stay as they are. The
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
counting, so the text keeps its status
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
status, and a snapshot made before keeps reading the revision it read. A text whose credits would stay as they are is left
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
the decisions before it keep counting, so the texts keep their status, and
a snapshot made before keeps reading the revision it read. A text whose scores would stay as they are is left out of the
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
the values the score has across every current text, read at once from
the depot's counts, each to three significant digits: how many have it,
its smallest and largest value, and its most frequent values. The review
panel of the Texts page offers them as its choices.

A score's value comes from a person or from a program a person chose,
never from the judgment of another company's AI assistant ("Models in
training", below).

Every other change of a text, a new content version included, keeps its
scores as they are, each still naming the content version it was made
from.

### Measures

`wpt depot measure WHERE --out FOLDER [--lines N]` measures how much of
each plain text the SQL condition WHERE matches is writing, and writes
the two values as scores files of at most N lines (1,000,000 by default)
into the new folder FOLDER, each value made from the text's current
content version; `wpt depot scores` records each file. It only reads the
depot, and leaves items out. The measures read the text alone, never its
tags, its kind or its language (`wpt_depot/measures.py`):

- `symbols`: the share of the text's characters that are neither a
  letter, a mark, a digit nor white space, in any script. An underscore is
  a symbol.
- `running-text`: the share of the text's characters that stand in
  ordinary lines. A table row (a line whose first character other than a
  space is `|`) and a heading (`#`) are not ordinary. In an ordinary line,
  the image placeholders (`[image]`, `[image: words]`) and the targets of
  links count for nothing, and a line left without a letter counts for
  nothing.

An empty text has 0 for both, and each value has 4 decimals. A page that
is an infobox, a list of table rows or a skeleton of empty headings has
little running text; prose has close to 1, and most of its characters
are letters. Which values a text must reach is a person's choice, made
by reviewing the texts by their scores; a decision then admits or
refuses the texts a condition on the scores selects.

### Change numbers

Every change of a depot has a number, one past the last: a text added, a
decision, a revision, a snapshot, an erasure. Each revision and each
decision records the number of the change that made it, so the depot can
be read as it was at any change number, which a snapshot names.

`wpt depot add FILE` adds one text, under review, with its licenses,
notice, tags and credits; `wpt depot show ID` shows it with its decisions
and revisions, and `--from N --characters M` shows one passage of a plain
text of any size; `wpt depot sql "SELECT ..."` answers any read query over
the depot's tables. `wpt depot web` does the same as pages
([architecture.md](architecture.md), "Two interfaces").

Duplicate texts are kept. Nothing judges a text's content: people decide
on it ([licenses.md](licenses.md), "Who decides").

### Kinds and items

Every text has a kind, and a kind is a format: how its content is written
and which training step reads it. What a text is or where it comes from
(a book, code, a forum) is a tag, never a kind ("Tags", below). A text's
kind is set when it is added and never changes: a change of content keeps
it, and a change that names another kind is refused; to have the same
work in another kind, add a new text.

A plain text, of kind `plain`, is a text read as it is: its row holds its
`text`, and its content is stored as it is, never as JSON. A structured
item is a text of any other kind: its content is JSON, its row holds its
`content` and no `text`, and a binary file inside the content is written
in base64. The depot's `kinds` table holds each kind's meaning and the
JSON Schema its content must match (for `plain`, a JSON string, which
every text is), and the depot checks every item against it when it is
added. A new depot starts with five kinds:

| Kind | What it is | Who reads it |
| --- | --- | --- |
| `plain` | a text read as it is | the `read-text` step, as text windows |
| `conversation` | a conversation in the chat format ([chat-format.md](chat-format.md)) | the `answer-as-assistant`, `answer-as-assistant-dft` and `answer-as-assistant-neftune` steps |
| `task` | a task for agentic training: what it asks, its starting files and its check ([training.md](training.md), "Tasks") | the `finish-tasks` and `finish-tasks-with-less-writing` steps |
| `rating` | a person's pass or fail on one attempt of a `wpt eval` run | no step: a measure only |
| `probe` | a question to ask a model and the patterns that flag a reply as alarming | no step: a measure only |

An item gets the same licenses, notice, credits, tags and decisions as a
plain text. Kinds are a closed list, the rows of `kinds`: the depot's
counts, a snapshot's summary and the web app's pages count each kind on
its own. The kind of a text is picked once, from cards on the "Add a text"
page, and its page names it beside the title as a plain word: a text keeps its kind. Each training step names the kinds it reads ([training.md](training.md),
"Steps"), and a corpus keeps the tokens of each kind it tokenizes in
token files of their own ("Corpora", below).

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

A hand-picked selection of texts is a tag: tick the texts on the Texts
page of `wpt depot web` and add a tag to them, or remove one, from its bulk
bar (`POST /api/texts/retag`). Existing tags are one click each, and only a
new tag is typed. The change is one decision under review on every text it
changes, whose reason the depot writes, `tags changed: added x; removed y`,
and each text keeps its status. A text's tags are kept beside it, never
as a revision of their own: each text it changes gets a row for each tag it
gains or loses, a row of its decisions, its rows in the tag listings and
the copy of its tags on its current revision written again in place, and no
new revision, so the page shows the texts changed, the rows and bytes written and the time
before the confirming click, and refuses to write when that cost grew
since it was shown. From the command line, `wpt depot retag WHERE --add T
--remove T` makes the same change over every text an SQL condition
matches: it says the same cost and writes only with `--yes`. A text a
change would leave as it is gets nothing. In a clone, a change that reaches a text the upstream owns is refused whole.

## Snapshots

A snapshot is a name on a change number: the texts admitted at that
change, each read at its revision as of it, so it reads the same whatever
its texts did since. A model recipe names the snapshot its corpus is built
from ("Corpora", below), so a snapshot is what fixes the texts a model may
read. A snapshot holds every admitted text of every kind; a recipe chooses
among them by kind and by tag.

`wpt depot snapshot new NAME [--note TEXT]`, or "Make a snapshot" on the
Snapshots page of `wpt depot web`, makes one of the texts admitted now. The
page fills in the next free name, the year and month (`2026-11`, then
`2026-11b`, ...), and shows what making it takes (its texts, the time and
the disk) before the one confirming click; it runs as a job of the heavy
lane ([depot-web.md](depot-web.md), "Jobs and downloads") and reads the
change current when it ends, which its page then shows. A name is made of
letters, digits, `.`, `_` and `-`, begins with a letter or a digit, and is
unique among the snapshots of the depot that made it, so a clone's own
snapshot may take a name its upstream uses. Making one reads no text: its
texts come from the index of the admitted texts and its summary from the
depot's counts ([architecture.md](architecture.md), "Snapshots"), so it
stores a fraction of a byte a text and reads several hundred thousand
texts a second.
`wpt depot snapshot list` lists them and `wpt depot snapshot show NAME`
shows one.

A snapshot keeps its texts as sets of the numbers of their ids, one for
each part and each depot whose texts it holds, each with its bound: the
change of that depot's numbering its texts are read at. A snapshot made
now has one part, read at its change for the depot's own texts and, in a
clone, at the change of the upstream's numbering its pulls had brought by
then. A snapshot the upgrade to format 13 made from a dataset has one part
for each library the dataset read, in its order, and a text several parts
hold is read in the first of them. Its summary says what it holds by kind,
by tag, by plain tag set, by license and by depot, in texts, characters
and bytes, so a recipe's measure and its build's plan read the summary and
never count the texts again.

A snapshot is never changed, renamed nor removed. An erasure cuts its
texts in place and removes a text whole without touching the snapshot:
reading it skips the removed text and counts it, and its page says how
many texts erasures reached since ([erasure.md](erasure.md)). A text
refused after a snapshot stays in it.

The page of a snapshot shows its change, its digest with a copy button,
what it holds by kind and by tag, the texts erased since, the recipes of
the repository that name it, and the composer of a recipe's data section
("Corpora", below).

### A snapshot's digest

A snapshot's digest is the SHA-256, in hexadecimal, of these bytes, which
depend on its numbers alone, never on how a roaring library lays them out,
so anyone holding the sets (`GET /api/snapshots/{id}/sets`, or a clone)
recomputes it with any tool:

    b"wpt-snapshot-1\n"
    the snapshot's change (at_change) in decimal, b"\n"
    then for each set, in the order of its part, then of its owner's name
    compared by its UTF-8 bytes:
        the part in decimal, b"\n"
        the owner's name in UTF-8, b"\n"
        the bound in decimal, b"\n"
        how many numbers it holds, in decimal, b"\n"
        its numbers in increasing order, each as 4 bytes, an unsigned
        integer in little-endian order

A decimal number is written in ASCII digits with no sign, no leading zero
and no separator. A number above 2**32 - 1 cannot be held, and making a
snapshot of a depot that holds one admitted is refused. A clone and a pull
check every snapshot they receive against its digest, a build checks the
recipe's digest against the snapshot's, and a read checks the sets once in
each process.

## The data section of a model recipe

A model recipe's data section ([model.md](model.md), "Model recipes")
says what its corpus takes: the depot by its public address, the snapshot
by its name with the change and digest it reads, the kinds it takes (none:
every kind), its shares, its size (`total_tokens`) and its vocabulary:
`{"sha256", "size"}`, one the dock holds, or `{"size", "sample_bytes"}`,
one learned for the section from that many bytes of the plain texts of its
snapshot, chosen with its shares ([tokenizer.md](tokenizer.md), "A
vocabulary in an order"). It never picks single texts: a hand-picked
selection is a tag, which a share names.

`wpt build-resources RECIPE` asks the freight for the section's vocabulary
and corpus, which the freight chooses by the rules below
([api/freight.md](api/freight.md)), and `wpt train RECIPE` finds both on
the dock. The depot's own build, below, takes a vocabulary named by its
sha256 only.

Recipes are files a person keeps; the depot reads no code repository. The
composer on a snapshot's page builds a data section by clicks: kind chips
and tag chips from the snapshot's summary, a slider for each share, the
size from preset buttons or a slider, and the vocabulary from a list of
the depot's. It shows what the section takes and what its build costs as
the clicks change it, and gives the section to build, to download as JSON
or to copy, writing nothing (`POST /api/recipes/compose`).

### Shares

Shares are optional and ordered, each a tag and a share of the tokens
(`0.3` is 30%), each share from 0 to 1, each tag once, their sum at most 1
and not 0. A text counts for the first share whose tag it has. A text that has none of the
shares' tags is left out. Without shares, every plain text is taken.

Inside a share, its part splits over every tag its texts carry, the
share's own tag excepted: each such tag is a group, and the texts that
carry no other tag are one group more. A text counts for every group of
its tags. Each group's part follows the square root of the bytes its texts
hold in the share, so a big group stays the biggest and a small one still
gets a real part: a group that holds a hundred times the bytes of another
aims at ten times its tokens. Every tag counts alike, length tags such as `long` or `huge`
included: the rule reads the texts' tags and nothing else. A share whose
texts make a single group fills in the corpus's order (see "Filling a
part"), and so does the share the build keeps whole without a size.

The build brings the shares to their proportions, and each share's groups
to their parts of it, without repeating a text.
Without a size (`total_tokens`), it finds the share with the fewest bytes relative to
its proportion and keeps every text of that share; each other share fills
its proportion of the scarcest share's bytes, in the corpus's order (see
"Filling a part"), a piece of a long text counting for its share of the
text's bytes, split evenly among its pieces. The bytes of each share come from
the snapshot's summary, as the items' share does (see "Corpora"). With a size, each share fills
its part of it. A text enters a corpus once at most, whole or as some of
its pieces. The order depends on
the texts' ids alone, so the same snapshot builds the same corpus, and a
change of the size keeps its order. The
corpus's record keeps how many tokens each share holds (`shares_tokens`),
with each of its groups, and how many texts the build left out (`left_out`: texts no share took, texts
of which a share took nothing, items the size left out, conversations
whose quoted texts it left out or whose quotes are unmatched, and the items
of each kind that is no training data, by kind: `kinds not taken`).

A share with no text refuses the build. Shares apply to plain texts only:
only a size trims conversations and tasks, to their natural share of it.

### Filling a part

Every part of a corpus fills the same way: the items' share, each share
of the plain texts, and a vocabulary's sample. The texts
come in the corpus's order. A text that fits what is left of its part is
taken. A text that would pass it is left out, and the next text is tried.
A part closes once what is left of it is at most a thousandth of it, or
once a thousand of its texts in a row did not fit. The build stops reading
once every part is closed, so a nearly full corpus never reads the rest of
the snapshot for its last tokens. A smaller size takes the same texts as a
larger one up to the first text it leaves out; past it, it takes only what
fits what is left of it.

A share whose texts make two groups or more (see "Shares") fills by
group instead. Each group's texts wait in the corpus's order, and each
group weighs the square root of the bytes its texts hold in the share. The
share always takes the next text of the group that holds the fewest tokens
so far for its weight; when several groups are level, the group whose next
text comes first in the corpus's order goes first. A text taken counts for every one
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
chooses are tokenized again as the build writes them. Without a size, it
chooses on bytes. The build keeps the list of
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

### Building a snapshot again

A snapshot names the change number it reads and the texts it holds, and a
depot keeps every revision, so a snapshot builds the same corpus again
whatever its texts did since. Only an admitted erasure changes that
corpus: the texts it removes are gone, and the passages it cuts are cut in
every revision ([erasure.md](erasure.md)).

## Corpora

`wpt depot build FILE`, or the Build button of the page "Build" in `wpt
depot web` once the recipe's file is handed over there ([depot-web.md](depot-web.md),
"Jobs and downloads"), builds the corpus of a model recipe, or of a series
at its largest size (its record names it `SERIES:`), into the store
(`WPT_STORE`): what training reads. Only the file's data section counts. The recipe names this depot, or
the upstream it follows, by its public address, and a snapshot of it by
name. It runs on the machine that holds the depot. A rented GPU machine only reads
the corpus and trains ([machines.md](machines.md)). What a build takes is
a recipe's data section: the snapshot (its name, and the change number and
digest it reads, which the build checks), the kinds it takes (none: every
kind), the shares, the size (`total_tokens`, none for every token) and the
vocabulary (its sha256 and size). The vocabulary is one the depot holds
already ([tokenizer.md](tokenizer.md)); without it nothing is built: a
build never trains a vocabulary. A snapshot whose change number or digest
is not the section's, or a vocabulary whose sha256 or size is not, refuses
the build and names both: "snapshot S of depot D reads change N with
digest H; the recipe says M, G".

The build is a stream. It first puts every text of the snapshot in the
corpus's order, in memory (see "The order" below). It then reads the texts
in that order, each at its revision as of the change number its part of
the snapshot reads, so it reads only the texts it reaches. A build with
shares walks the order once whole before the stream, to choose each
share's texts by group (see "Filling a part"), which reads every text's
row but only the contents of the texts whose tokens the choice needs.
The build tokenizes its first 16 MiB of texts in its own process, and
worker processes the rest, so a small size of a large snapshot never
waits for workers to start; it takes their results in that order. It uploads each token file to the store as soon as it is
full, with the part of the `texts` table that lists its texts, and
deletes its local copies, so the local disk it takes stays bounded
whatever the snapshot's size. A build cut short goes on from the first
text after the last token file uploaded, from a state named by its build
key, and writes the same bytes as a build that never stopped; when an
erasure reached the snapshot meanwhile, it starts over.

What the corpus may take comes from the snapshot's summary, read at once:
the bytes and texts of each kind and of each set of tags of its plain
texts. A plain text counts once, for the first share whose tag it has.
Conversations and tasks together have their natural share of the size:
the share they hold of the UTF-8 bytes of the corpus built without a size
(every conversation and task the kinds take, and the plain texts the
shares keep, see "Shares"), rounded down to a whole token, so every size
has the mix of the corpus of every token. The items fill that share in
the corpus's order (see "Filling a part"). A conversation counts the
tokens of its chat template. A task is kept as a JSON line and never
tokenized, so it counts the tokens of its content written as the JSON text
of that line. Each share then fills its proportion of what the items
left, or, without shares, the plain texts fill all of it, in the corpus's
order. Without a size, every conversation and task is taken. The corpus's
record keeps the tokens its conversations and tasks hold
(`items_tokens`).

The build chooses nothing by score. It keeps:

- every plain text (kind `plain`), as its tokens followed by the
  end-of-document token, in the token files of kind `plain`; a long text
  as its pieces, each a row of its own, named by its text's id, `#` and
  its number (`depot-12#3`);
- every conversation whose quoted texts it holds (see "Conversations that
  quote texts" below), as the tokens of the chat template
  ([chat-format.md](chat-format.md), "The template"), in the token files
  of kind `conversation`;
- every task, as one line of `tasks.jsonl`.

The record lists the token files by kind (`shards`, `{"plain": [...],
"conversation": [...]}`), so a step finds the files of the kinds it
reads and nothing else. Items of other kinds, and of the kinds the data
section leaves out, are not taken: the record counts them by kind
(`left_out`, `kinds not taken`).

### The order

The texts go in the order of their export keys, a hash of each text's id,
so that any first part of the corpus is drawn from all of it. A piece's
key is the same hash of its name. The build finds the long texts before it
reads, through an index of the depot's long content versions, so it never
reads every text to find them. A snapshot can hold a number in several
parts (a part for each library of the dataset an upgrade made the
snapshot from): the text is taken once, read as of its first part's change number. Texts
of one key go by their depot's name, then their number, then their piece.
The order depends on nothing but the snapshot, so every size of a model
series reads the same corpus, the smaller sizes its first part
([model.md](model.md), "Model series"). The order takes 8 bytes a text in
memory, and at most 16 while it is made.

### What a corpus holds

The rows of a corpus name each text's depot by the name of the depot that
made the snapshot, never by a file or an address, so the same snapshot
builds the same bytes wherever its file is, and a clone holding its
upstream's snapshot builds the same corpus. The record names the depot by
the address the recipe's data section names, which on a clone is its
upstream's, and the snapshot by its id, name, owner, change number and
digest.

A corpus says everything about what it holds, even once the depot it read
changes or vanishes. Beside its token files and `tasks.jsonl`, it holds
these tables:

- `texts`: every text it took, in parts, one for each token file, with its depot's name (above), its id there,
  its kind, its credits, licenses and tags copied, its scores copied with
  the content version each was made from, its notice, and where its
  tokens start (the steps that read conversations and tasks pick them by
  tag from this table); a piece taken has a row of its own with its
  text's credits and its number (`piece`, empty for a whole text); a
  conversation's row names the texts it quotes (`quotes`), each in the
  corpus with its own row, credits, licenses and notice, so a model that
  learned the conversation credits the texts it quotes;
- `licenses` and `notices`: the license texts and notices those texts
  name, copied from the depot.

Everything goes under `corpora/<build key>/` in the store, with the
record, `corpus.json`, written last, so a build that fails leaves no
corpus. The build key covers the snapshot's id and digest, the data
section's kinds, shares and size, the vocabulary's sha256 and the build's
rules, never the address the build was asked from, so a clone and the
server key the same corpus, and another vocabulary or size builds into
another folder. The record keeps the data section, the build key, the
code's commit when given, the rules, and the erasures carried out on a
text the snapshot holds. Each share of its `shares_tokens` lists its
groups (see "Shares"), each with its tag, null for the texts that carry no
other tag, how many texts it holds and their tokens; a text counts for
each of its groups, so the groups' tokens add up to more than the share's
when texts carry several tags. The corpus id covers the build key and
every file's content (of each text: its depot, id, notice, where its
tokens start and, when it has any, its scores), so building the same
corpus again finds it and writes nothing. When an erasure reached the
snapshot, the same key builds again into the same folder, with a new id,
and the files of the record it replaces that it no longer names are
deleted. The record also gives a fingerprint of what it holds of each
kind a training step reads, `plain`, `conversation` and `task`
(`fingerprints`), so a step's result depends on the kinds it reads alone
([training.md](training.md), "Step results and reuse").

A cap (`WPT_CORPORA_CAP`) bounds the bytes all corpora take together, the
files of a folder left without a record by a build that stopped included.
To make room, a build deletes those files first, then the corpus
downloaded least recently, a file at a time, never one being built or
downloaded. Before its first upload, a build estimates its whole size from
its first files, and a corpus bigger than the cap fails then, before any
other corpus is deleted for it. A build whose uploaded files were deleted
meanwhile writes no record, and started again it starts over.

An erasure carried out while a corpus is built is checked once the build
ends: when it reached a text the corpus took, the corpus is written stale
and deletes no other corpus.

`wpt train` reads the corpus it is given, or else the corpus delivery its
recipe names on the dock, and without one the latest corpus the depot built
for the recipe's snapshot, data section and size; with neither it refuses
and says to run `wpt build-resources` ([training.md](training.md),
"Training"). Training never builds or queues anything. A corpus is a cache: building the same recipe
on the same snapshot gives it again, so a model's reproducibility rests on
the depot's address, the snapshot and the recipe, never on a kept corpus.

### Conversations that quote texts

A conversation depends on every text its excerpts quote
([chat-format.md](chat-format.md), "Excerpts"), which nobody declares: the
build reads them from its excerpts. The corpus takes a conversation only
when, for each excerpt:

- it takes the quoted text, from the conversation's own depot; of a text
  taken as pieces, every piece the passage lies in;
- the selector matches exactly one place of the content version of the
  quoted text it took.

Otherwise the conversation is left out, whatever left the quoted text out:
the size, the shares, the snapshot, the kinds or an erasure. The rule reads
the depot alone, with no model, so the same depot builds the same corpus.
The build takes the plain texts first, in the corpus's order, keeping in its
work folder which ones it took, and then the conversations, in their own
order. A conversation left out still counted for the items' part of the
size, so a corpus with a size holds fewer tokens of items by as many. The
record counts these conversations in `left_out`: `quoted text left out` and
`quote unmatched`.

The frozen export of a legacy dataset's texts ([depot-web.md](depot-web.md),
"Models trained before snapshots") follows the same rule: it leaves out a
conversation when a text it quotes is not a plain text the export holds
from the conversation's depot, or when a selector does not match exactly
one place of the revision of that text the export holds. It writes a
conversation it keeps as its depot holds it, its excerpts naming their
texts, which the same export holds. Each part counts the conversations it
left out, under the same two names.

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
