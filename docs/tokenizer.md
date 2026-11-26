# Tokenizer

WPT has its own byte-level BPE tokenizer (`packages/wpt-tokenizer`), with its
hot loops in Rust (PyO3, built by maturin when the workspace is installed).

- **Every byte has a token** (ids 0-255), so encoding never drops anything.
- Id 256 is the end-of-document token.
- Ids 257 to 512 are reserved for special tokens, such as the chat
  template's tags ([chat-format.md](chat-format.md)). Encoding text never
  produces them, and decoding gives them no bytes.
- Each learned merge adds the next id from 513 on.
- Text is first cut into chunks by ASCII whitespace only (a chunk is a word
  with at most one leading space, or a run of whitespace), so results never
  depend on a Unicode version.
- Each thread that encodes keeps its own cache of encoded chunks (up to
  about a million, then it starts again empty), so threads encoding at
  once never wait for each other, and the tokens never depend on the cache.
  A thread's cache goes when the thread ends.
- Training reads every document, but counts the chunks in a table of a
  fixed number of slots (Space-Saving: Metwally, Agrawal and El Abbadi,
  2005), so the memory does not grow with the corpus. A chunk already in
  the table has its count raised; a new chunk takes a free slot with count 1;
  when the table is full, the slot with the smallest count is evicted and
  the new chunk takes it with the evicted count plus 1, recording the evicted
  count as its error. A chunk's true count is then between its count less
  its error and its count, and every chunk more frequent than the smallest
  count is in the table. Equal counts are told apart by the chunk's bytes,
  so the same documents in the same order always give the same table and
  the same vocabulary. The merges are learned from the table's chunks and
  counts. While the documents have no more distinct chunks than the slots,
  no chunk is evicted, every count is exact and the order of the documents
  does not matter. The number of slots (50,000,000) is fixed in the
  tokenizer's code, never measured from the machine, so a vocabulary never
  depends on the machine that trained it; it keeps reading and learning
  together within about 25 GB.
- The learning step uses every core of the machine: the merges come one
  after the other, and the work inside each one is shared between threads.
  It gives the same vocabulary as on one core, whatever the number of cores.

## Vocabularies

A vocabulary (the list of merges) is learned once and shared by every
order and model that names it, by its sha256.

The freight learns a vocabulary outside the depot: a request names a
snapshot, its shares and a sample in bytes, and the program
`wpt-vocabulary` learns from the plain texts the freight chooses, read once
in the order of the snapshot's listing. Each share fills its proportion of
the sample's bytes, by group, as a corpus fills its shares
([data-selection.md](data-selection.md), "Shares" and "Filling a part").
The delivery on the dock holds `vocabulary.json`, and its items tables
list the texts it read, in that order ([api/freight.md](api/freight.md),
[api/preparation.md](api/preparation.md), "wpt-vocabulary"). An erasure is
written into its delivery note and does not make it stale.

A depot may also hold tokenizer recipes and the vocabularies they trained,
which it keeps and never writes: no command or page of the depot
makes, edits, trains or imports one, and a clone and a pull carry them
with the rest of its rows ([architecture.md](architecture.md), "Following
an upstream"). A tokenizer recipe is a revised row (`tokenizers_revisions`):
its name, its sample (the snapshot whose plain texts trained its
vocabulary, and the kinds it read) or, for a revision written before
snapshots, a dataset, `sample_bytes`, `vocab_size` and a note, its
`version` raised by each change of what trains it. `vocabularies` holds at
most one vocabulary for each version of a tokenizer recipe, and `vocabulary_texts`
every text it trained on, in the order they were read, by its depot (empty
for the depot that holds the recipe, the web address for another depot)
and its id there. Each of those vocabularies is on the dock as an adopted
delivery (below), and the web app's frozen routes `GET /api/tokenizers/{name}` and its
vocabulary file serve the models trained before snapshots
([training.md](training.md), "Models trained before snapshots").

### A vocabulary in an order

A model order's data section names its vocabulary in one of two forms
([data-selection.md](data-selection.md), "The data section of a model
order"):

- `{"size": N, "sample_bytes": B}`: a vocabulary of N entries learned on B
  bytes of the plain texts of the section's own snapshot, chosen with its
  shares. The order never holds its sha256: the vocabulary's delivery note
  gives it.
- `{"sha256": S, "size": N}`: the vocabulary of that sha256, which the dock
  must hold. This is how an order keeps the vocabulary of an earlier model.

`wpt build-resources ORDER` (`SERIES:` for a series) sends the order's
requests to the freight in order: the vocabulary first when the order
learns one, then the corpus, whose request names the vocabulary's delivery
as its input. It waits for both jobs and prints both delivery notes
(`wpt.resources`). Run again after it was stopped, it waits for the job of
the same delivery that is still queued or running, which goes on from its
checkpoint. A vocabulary named by its sha256 makes no request: the
corpus names the delivery of that vocabulary the dock holds, the adopted
one (`adopted-<sha256>`) when there is one, else the vocabulary delivery of
that sha256 whose key comes first. The key of each delivery comes from the
order alone, with the version of the program `wpt-corpus` holds, so `wpt
train ORDER` finds both on the dock with no depot and no freight at hand
([training.md](training.md), "Training").

A delivery is shown by what made it, never by a free name: `vocabulary
16,384 · snapshot 2026-11-23T02-27-38Z · research-paper 35 / books 15 /
government 5 / code 5 / encyclopedia 40 · 10,000,000,000 bytes`. That is
the `tokenizer` a model's weight files name for a vocabulary the freight
learned (`wpt.resources.label`).

Every vocabulary the depot's tokenizer recipes trained is on the dock, so
an order naming one by its sha256 finds it: the delivery
`adopted-<sha256>`, its file `vocabulary.json`, the result a
`wpt-vocabulary` delivery gives (`sha256`, `size`, `tokenizer_version`),
what the depot recorded of it (`adopted`: its tokenizer recipe's name and version,
the dataset it named, its sample and sample bytes, its count of texts and
their bytes), and as its items table the texts it trained on, in the order
it read them (`depot`, `id`, `change`). Nothing in WPT makes such a
delivery: a new vocabulary is one the freight learns. The weight
files of a model trained on it name it by its tokenizer recipe and version
(`classics version 1`).

An erasure never makes a vocabulary delivery stale, adopted or learned: the
freight writes it into the delivery's `erasures_since` and the delivery
stays fresh, so every order and model that names the vocabulary keeps
working and the merges learned from the erased text stay
([erasure.md](erasure.md)). The corpus that holds the text is stale, and
`wpt build-resources` makes it again with the same vocabulary.

### One vocabulary, for good

A vocabulary never changes once learned. Every order that names it by its
sha256 uses it:

- **Adding or changing texts never retrains it.** Many orders and
  models, whatever their training settings, share one vocabulary, so a
  model can keep training on new data. A text the vocabulary was not
  trained on still encodes exactly, since every byte has a token, only less
  compactly.
- **A new vocabulary takes a new request.** For a big change (much more
  data, another size), an order asks for another one to learn. A change of
  the tokenizer's code asks for none: the vocabulary stays.
- **Erasure does not reach a vocabulary**: merges learned from text that
  was erased later stay in it (see [erasure.md](erasure.md)).

The tokenizer's version is in `packages/wpt-tokenizer/rust/Cargo.toml`; a
vocabulary file names it (see below). Bump it when a change makes the ids
mean other tokens.

### Where the vocabulary is checked

- A corpus the depot built onto the dock's `corpora/` names in its
  record its vocabulary's tokenizer recipe, version and sha256 and the
  vocabulary's file (`{"key", "sha256"}`, in the corpus's folder), and its
  id covers its sha256.
- A corpus delivery names its vocabulary's delivery as its input, and its
  result names the vocabulary's sha256 and size; the delivery key covers
  the input's key.
- Training reads the vocabulary from the corpus (a delivery's: the file of
  the vocabulary delivery it names), checked against that sha256, and
  writes it into the model folder as `tokenizer.json`. Every weight file's
  header names it: `tokenizer` (such as `classics version 1`) and
  `tokenizer_sha256`.
- `wpt chat` and `wpt eval` load `tokenizer.json` only when its sha256
  is the one in the weights' headers (`TokenizerMismatch` otherwise), so a
  model never runs with another vocabulary.

A vocabulary file is canonical JSON:

```json
{"format":"wpt-bpe","merges":[[a,b],...],"tokenizer_version":"0.3.0"}
```

A vocabulary file written by another tokenizer version is refused, since
its ids would mean other tokens: tokenizer 0.3.0 reserved ids 257 to 512,
so vocabularies of 0.2.0, and the models trained on them, need a new
vocabulary version and a new model.
