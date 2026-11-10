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

A vocabulary (the list of merges) is a build result: one per version of a
tokenizer recipe, shared by every dataset and model that names it.

`data/recipes/tokenizers/<name>.json`:

```json
{ "version": 1, "depot": "<address>", "dataset": "<name>", "sample_bytes": 10000000000, "vocab_size": 32000, "note": "" }
```

`depot` is the address of a depot and `dataset` the name of a dataset
there: their texts train the vocabulary. `sample_bytes` is the most bytes of
text the sample that trains it holds.

### Training it

`wpt depot tokenizer RECIPE` trains the vocabulary of a recipe. It opens the
recipe's depot, takes the plain texts of the recipe's dataset in the order
its corpus would take them (the dataset's libraries at the states it names,
its shares), and keeps each text whole while the bytes kept stay within
`sample_bytes`; a text that no longer fits is passed over, and items are no
sample. It trains the vocabulary on that sample, puts it in the store
(`WPT_STORE`), and writes the recipe's lock. A recipe whose `depot` or
`dataset` is still empty is refused (`InvalidRecipe`), and so is a dataset
with no plain text within `sample_bytes` (`NoSample`).

### The lock freezes it

The vocabulary of a recipe version is trained once, and its lock,
`data/recipes/tokenizers/<name>.lock.json`, freezes it: the recipe, the
vocabulary's sha256, and every text it was trained on, by its depot's
address and its id there (`texts`). Commit the lock.
From then on, every build of that version uses that vocabulary:

- **Adding or changing texts never retrains it.** Many datasets and
  models, whatever their training settings, share one vocabulary, so a
  model can keep training on new data. A text the vocabulary was not
  trained on still encodes exactly, since every byte has a token, only less
  compactly.
- **A new vocabulary takes a new version.** For a big change (much more
  data, another size) or a change of the tokenizing algorithm, raise
  `version` in the recipe and run `wpt depot tokenizer RECIPE` again: it
  replaces the lock (git history keeps the old one). While the lock is of
  the recipe's version, the command refuses to train (`TokenizerLocked`), so
  one name and version always mean one vocabulary. A change of the
  tokenizer's code asks for no `version` of the recipe: the lock keeps the
  vocabulary.
- **The vocabulary is shared through the store.** A locked vocabulary is
  never trained again from the texts its lock lists, on this machine or
  another: every build reads it from the store, checked against the sha256
  the lock keeps. A build stops (`NoVocabulary`) when the recipe has no lock
  of its version, or when the store does not hold the vocabulary, so a
  machine without access to the store that holds it cannot use it.
- **Erasure does not reach a locked vocabulary** until `version` is raised:
  merges learned from text that was erased later stay in it (see
  [erasure.md](erasure.md)).

The tokenizer's version is in `packages/wpt-tokenizer/rust/Cargo.toml`; a
vocabulary file names it (see below). Bump it when a change makes the ids
mean other tokens.

### Where the vocabulary is checked

- The corpus build uses only the locked vocabulary; the corpus record
  names the recipe, its version and the vocabulary's file (`{"key",
  "sha256"}`, in the corpus's folder), and the corpus id covers its
  sha256.
- Training reads the vocabulary from the store, checked against that
  sha256, and writes it into the model folder as `tokenizer.json`. Every
  weight file's header names it: `tokenizer` (such as
  `classics version 1`) and `tokenizer_sha256`.
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
