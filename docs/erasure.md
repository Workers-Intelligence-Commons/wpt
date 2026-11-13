# Erasure requests

**Every erasure request is honored**, including a copyright holder's request
about content that is properly licensed.

Backups of a depot are the one exception. An erasure does not reach a
backup, so a backup keeps an erased passage for good, and a depot
restored to a moment before the erasure holds the passage again ([architecture.md](architecture.md),
"Backups").

## How to ask

- Through the depot: `wpt depot erase ID [START END] [--field POINTER] --reason TEXT` with
  the asker's declaration, or `wpt depot web`: the "Ask for an erasure"
  action of the "Erasures" page, which finds the text by a part of its
  title or id, or by its exact id, or "Ask to erase" beside the text in
  the workspace. Both show the text, where a selection with the mouse fills the positions, which can
  also be typed, or the whole text can be named.
  Anyone can make a request.
- To a depot's maintainer, at the takedown contact its `about` table
  names.
- For this project, a public GitHub issue on this repository, or
  privately, through GitHub's private vulnerability reporting: the
  repository's **Security** tab, then **Report a vulnerability**. It needs a
  GitHub account. Use it when the request itself should not be public.

Requests under the GDPR are answered within one month.

## What a request holds

An erasure request is a row of the depot's `erasures` table, kept for
good: its reason, its decisions, and its needles. A needle names one
passage by its place, never by its content: the id of a text or an item
(a depot's name and a number, "Ids" in [architecture.md](architecture.md)),
and the start and end of the passage, counted in characters (Unicode code
points), the end excluded. On an item ([data-selection.md](data-selection.md),
"Items"), whose content is JSON, a needle also names a `field`, a JSON
pointer to a string of the content (for example `/messages/0/text`), and
its start and end count inside that string. A needle without a start and an
end names the whole text or item.

A text's content can change, each change a new content version kept next
to the old ones ([data-selection.md](data-selection.md), "Texts"). A needle
counts its places in one content version of its text, and the request
records which: the one it names, or the text's current one when the
request is asked.

The needles of one request that name the same string, the same text and
content version and for an item the same field, and that overlap or touch are one passage, from
the first start to the last end. The request is carried out on that
passage, so no character of any of its needles remains.

The "Erasures" page of `wpt depot web` shows what a request asks to
erase, as a code review shows a removal: its needles grouped by text, and
for each one its place and the text around it, at most 300 characters on
each side, the passage struck through, read from the content version the
needle names. A long passage shows only its two
ends, and a whole text its head. Once the request is carried out, the depot
keeps only each passage's length and the place it left, so a marker with
that length stands there. A later erasure of the same string can still move
the text around that place. A request carried out before places were
recorded has none: its marker's place is reckoned from the needle's start
less the request's other passages before it in the same string, and the
view says so. `GET /api/erasures/{id}` gives the same view to any client,
never a whole text.

A request is decided like a text: it starts with one decision under
review, the asker's, and it is admitted when at least one decision admits
it and none refuses it ([licenses.md](licenses.md), "Who decides").
`wpt depot erasures` lists every request.

`wpt depot decide admit|refuse --erasure ID` decides a request, with the
decider's declaration, and so does the "Erasures" page of `wpt depot web`.
Both call `Depot.decide_erasure`, which refuses a request that was carried
out already.

## What happens once it is admitted

The depot carries out an admitted request at once:

- From the content version each passage is in, it removes the passage
  itself. Then, from every content version of that text, and from no other
  text, it removes every place where the passage still appears, the
  passages read from the text as it was before any cut of the request. On
  an item, the passage is removed from every string of each content
  version, and every cut version must still match its kind's JSON Schema:
  a cut that breaks one is refused, and the request stays undecided. The
  removed text leaves no marker.
- Every revision of a text reads one of its content versions, so a cut
  reaches every revision, and a locked library that pinned an older
  revision reads its version cut. A passage in a small field of the
  text, such as its title, is cut from every revision that holds it.
- A needle without a start and an end removes its text or item from the
  depot: every revision of it, every content version, and its place in
  every library, locked or open.
- A text removed whole leaves its id unused for good: no later text takes
  it, so an id names one text everywhere it is recorded.
- A later content change of that text is refused when its new content
  holds a passage erased from it, matched by the passage's public hash.
  The check covers only the text the passage was erased from.
- A cut text keeps its id, its libraries and its decisions, in a locked
  library too: the admitted erasure is its admission again.
- The request's `erased` gets one entry per passage: its place once cut
  in the content version its needle names (`at`: how many characters of
  its string stand before it, in characters, inside the field for an item;
  none for a whole text), the first of the
  request's needles it covers (`needle`), its length and two public
  hashes: the SHA-256 of its UTF-8 bytes, and a polynomial hash of its
  characters whose base and numbers are fixed, public parameters. No
  secret is involved, so any depot or clone can match an erased passage
  in a text and refuse it. The cost is that anyone can confirm a guess at a
  short passage, such as a name or a phone number, by hashing it; the
  second hash can be slid over a later text to find the passage there.
- The depot's file keeps no copy of the passage: a deleted row's bytes
  are overwritten, and the file's write-ahead log is emptied into it.
- The erasure's change file, like every change file, holds the rows it
  removed as they were, the passage included. It waits in the waiting
  folder beside the depot's file (`<file>.waiting`) until the next
  `wpt depot backup` sends it to the store and deletes it from the
  folder. The
  backups keep it for good.

A depot that follows an upstream carries out the upstream's admitted
erasures when it pulls, with their recorded places ([architecture.md](architecture.md), "The depot").
It takes no erasure request that names an upstream text and decides no
upstream request: those are asked for and decided at the upstream, and
the pull brings the result.

## What it covers

Erasure applies to the depot and to every corpus built after it, never
to the depot's backups. A corpus
already in the store is not rewritten: it keeps the passage until its
dataset is built again, and it is marked stale (its record names the
erasures), so `wpt train` refuses it until it is built again. Built again,
a dataset gives what its libraries held at its change numbers, less what
the erasure removed or cut ([data-selection.md](data-selection.md),
"Building a dataset again"). Models already trained and published are not
changed.

A locked vocabulary is not retrained when a text changes (see
[tokenizer.md](tokenizer.md)), so merges learned from erased text, such as
pieces of a rare name, stay in it until someone raises the tokenizer
recipe's `version`. When an erasure must reach the vocabulary too, raise
the version together with the erasure.

## The repository's history

The repository holds no texts, so an erasure never touches it. Its git
history is rewritten only for accidents, such as a leaked secret, and a
rewrite is incomplete: clones, forks and cached views keep the data until
GitHub Support purges it.
