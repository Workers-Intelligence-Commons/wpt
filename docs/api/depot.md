# The depot's api

`wpt_depot.api` is what code outside the depot asks of it. The corpus
build, its estimate and the sample that trains a vocabulary
(`wpt_depot.corpus`, `wpt_depot.estimate`, `wpt_depot.groups`) read the
depot only through it. A change to the module changes this document in
the same commit.

## Rules of every call

- The first argument is the depot asked, by its address: a `str`, the path
  or `file://` address of its SQLite file, as `WPT_DEPOT` names it.
- Every other argument, and every answer, is a value JSON can carry
  (`null`, a boolean, a number, a string, a list, an object with string
  keys), or a stream of bytes. No call hands out a connection, a row
  object or a path into the depot's files. So each call can become a
  route of the depot's web app without its callers changing.
- Nothing is written. The depot is opened read-only, once in each process,
  and kept while 8 other depots are not asked for after it. A file
  replaced at the same path is opened again.
- A depot of another format than the code's is refused (`OtherFormat`,
  `HalfUpgraded`), an address that holds no depot too (`NotADepot`), and a
  snapshot whose sets do not match its digest (`Refused`). A snapshot,
  text or revision the depot does not hold is `NotFound`.

## The calls

### The depot

`about(depot)` gives the depot's about row: `name` (which begins its ids),
`maintainer`, `takedown_contact`, `upstream` (the web address of the depot
it follows, `null` for none), `format`, `revision` and `change`.

### A snapshot

`snapshot(depot, snapshot_id)`, or `snapshot(depot, name=NAME,
owner=OWNER)` (the depot asked by default), gives the snapshot's row:
`id`, `owner`, `name`, `change`, `at_change` (the change number it reads),
`made_at`, `origin`, `note`, `texts`, `characters`, `bytes`, `digest` and
`summary` (texts, characters and bytes by kind, tag, plain tag set and
license, and texts by owner; docs/architecture.md, "The depot"). The
summary gives a corpus's census with no further call.

### Its listing

`listing(depot, snapshot, after=None, spread=None)` gives every text of
the snapshot in the depot's order, without content, one object at a time:

| Field | What it holds |
| --- | --- |
| `key` | the export key the order sorts by |
| `owner`, `number` | the depot whose numbering the id is in, and the number |
| `id` | the text's id |
| `piece` | the piece's number, `null` for a whole text |
| `revision` | the text's revision as of the snapshot's bound |
| `kind` | the kind of the content version that revision reads |
| `tags` | the text's tags as of the bound, a list |
| `bytes` | the content's UTF-8 bytes; a piece's, its share of its text's, split evenly among its pieces |

The order is the order of the export keys of the texts' ids, and of the
pieces' names for a plain text longer than a piece, which comes as its
pieces (docs/data-selection.md, "Long texts"). The order and the cutting
into pieces belong to the depot: the same snapshot gives the same listing
whatever asks, and an erasure that removed a text whole passes it by.

`after` is the place of an entry, `[key, owner, number, piece]` with `-1`
for a whole text's piece: the listing goes on right after it, which is how
a build goes on from its last upload. `spread` gives at most that many
places spread evenly over the listing instead of every one.

The listing of a snapshot is made once from its sets and kept until an
erasure reaches the depot or another snapshot is listed, so the walks of
one build read the same listing and make it once.

`listing_size(depot, snapshot)` gives `{entries, long_texts, pieces}`: the
places of the listing (a text an erasure removed whole since still
counting), the plain texts that come as pieces, and their pieces in all.

`sampled(depot, snapshot, points)` gives at most `points` whole texts of
the snapshot, as `listing` gives them and in its order, spread over the
snapshot's numbers rather than its order, so it never works out every
text's key: what an estimate reads of a big snapshot.

`texts(depot, snapshot, ids)` gives `{id: {revision, kind, tags, bytes}}`
of the texts named that the snapshot holds, each as its listing reads it
whole: at its revision as of the bound of the lowest part of the snapshot
that holds it.

`holds(depot, snapshot, ids)` gives those of the texts named that any set
of the snapshot holds, in the order given, whether or not an erasure
removed them since.

### Content

`content(depot, text, revision, piece=None, span=None)` gives the content
of a text at a revision as a stream of UTF-8 bytes: the whole content
version that revision reads, one part at a time, so a long text is never
whole in memory (a plain text's characters, an item's JSON); with
`piece`, that piece of a long plain text; with `span`, `[start, end]`,
its characters from `start` to `end`, excluded.

`matched(depot, text, revision, selectors, pieced=False)` says where quote
selectors (docs/chat-format.md, "Excerpts") match the content version a
text's revision reads, reading it once for all of them: for each selector,
in order, `null` when it matches no place or several, else the numbers of
the pieces its passage lies in when `pieced` (the text is taken as
pieces), `[]` when not.

`quoted(depot, text, revision)` gives the ids of the texts a conversation
quotes at that revision.

### Erasures

`erasures(depot)` gives the erasures the depot has carried out, in the
order of their ids: `{id, texts}`, the texts each removed or cut, each
once, in the order its request names them.

### Credits

`credits(depot, texts)` takes `[[id, revision], ...]` and gives, for each
in order, the text's credits at that revision: `kind`, `title`, `authors`,
`address`, `licenses`, `tags` (the text's now: a revision holds none; the
listing gives them as of the snapshot), `notice` (its id) and `scores`
(each with the content version it was made from).

`licenses(depot, ids)` gives `{id, name, text}` of each license named that
the depot holds, and `notices(depot, ids)` gives `{id, text}` of each
notice, both in the order given.
