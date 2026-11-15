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

A vocabulary (the list of merges) is trained once for each version of a
tokenizer recipe and shared by every dataset and model that names it. The
recipes and their vocabularies live in the depot, so a clone and a pull
carry them with the rest of its rows ([architecture.md](architecture.md),
"Following an upstream").

A tokenizer recipe is a revised row of the depot: its name, the dataset
whose sample trains its vocabulary, `sample_bytes` (the most UTF-8 bytes
that sample holds), `vocab_size` and a note. The dataset is one of the
depot's own, or, with `depot`, a dataset of another depot named by its name
and by the web address of that depot's web app. A change of the dataset,
the depot, `sample_bytes` or `vocab_size` makes a new revision with the
next `version`; a change of the note alone makes a new revision at the same
version.

    wpt depot tokenizer new NAME --dataset D [--depot URL] --sample-bytes N --vocab-size V [--note T]
    wpt depot tokenizer edit NAME [--dataset D] [--depot URL] [--sample-bytes N] [--vocab-size V] [--note T]

The page "Tokenizers" of `wpt depot web` makes and changes them the same
way.

### Training it

`wpt depot tokenizer train NAME`, the button "Train" on the recipe's page,
or the build of a dataset that names the recipe trains the vocabulary of
the recipe's current version. The web app runs the training as a job of
its queue ([depot-web.md](depot-web.md)).

The sample is the plain texts of the recipe's dataset in the order its
corpus takes them (the dataset's libraries at the states it names, then
its shares), each share stopping at its first text that would pass its
part of `sample_bytes`. Items are no sample. The training reads the sample
once, as a stream: the texts of the depot's own dataset from the depot, and
the texts of another depot's dataset through that depot's sample route
(`GET /api/datasets/{dataset}/sample?bytes=N`). A dataset with no plain
text within `sample_bytes` trains nothing (`NoSample`).

The depot keeps the vocabulary with the list of every text it trained on,
in the order they were read, by its depot (empty for the depot that holds
the recipe, the web address for another depot) and its id there.

### One vocabulary per version

The depot holds at most one vocabulary for each version of a recipe, and
refuses a second one. From then on, every build of that version uses that
vocabulary:

- **Adding or changing texts never retrains it.** Many datasets and
  models, whatever their training settings, share one vocabulary, so a
  model can keep training on new data. A text the vocabulary was not
  trained on still encodes exactly, since every byte has a token, only less
  compactly.
- **A new vocabulary takes a new version.** For a big change (much more
  data, another size), change what trains it, which raises the version,
  and train again. A change of the tokenizer's code asks for no new
  version: the depot keeps the vocabulary.
- **A build without the vocabulary of its version builds nothing**
  (`NoVocabulary`) from the command line; the web app's build queues the
  training first.
- **Erasure does not reach a trained vocabulary** until the version is
  raised: merges learned from text that was erased later stay in it (see
  [erasure.md](erasure.md)).

### Importing one

`wpt depot tokenizer import NAME FILE --dataset D` makes a recipe at
version 1 from a recipe file (JSON with `sample_bytes`, `vocab_size` and an
optional `note`), trained on the depot's dataset D. With `--lock LOCK`, it
also keeps the vocabulary a lock file names (JSON with the recipe's fields,
`vocabulary_sha256` and `texts`), read from the store, with its list of
texts; the lock's `sample_bytes` and `vocab_size` must be the recipe
file's. With `--vocabulary FILE`, it keeps that vocabulary file, with no
list of texts. Without either, the recipe has no vocabulary until it is
trained.

The tokenizer's version is in `packages/wpt-tokenizer/rust/Cargo.toml`; a
vocabulary file names it (see below). Bump it when a change makes the ids
mean other tokens.

### Where the vocabulary is checked

- The corpus build uses only the vocabulary the depot holds for the
  recipe's version; the corpus record names the recipe, its version and
  the vocabulary's file (`{"key", "sha256"}`, in the corpus's folder), and
  the corpus id covers its sha256.
- Training reads the vocabulary from the corpus, checked against that
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
