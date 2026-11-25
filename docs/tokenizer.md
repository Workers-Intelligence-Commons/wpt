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
tokenizer recipe and shared by every recipe and model that names it. The
recipes and their vocabularies live in the depot, so a clone and a pull
carry them with the rest of its rows ([architecture.md](architecture.md),
"Following an upstream").

A tokenizer recipe is a revised row of the depot: its name, its sample
(the snapshot whose texts train its vocabulary, by its id, and the kinds it
reads: plain, the one kind a sample takes), `sample_bytes` (the most UTF-8
bytes that sample holds), `vocab_size` and a note. A change of the sample,
`sample_bytes` or `vocab_size` makes a new revision with the next
`version`; a change of the note alone makes a new revision at the same
version. A revision written before snapshots names a dataset instead of a
sample and keeps it; such a recipe trains nothing until an edit names a
snapshot, and its page says so. When the upgrade to snapshots made a
snapshot of that dataset (`legacy_snapshot_of`), the web app offers it: "Use
snapshot NAME", on the recipe's row and page, is that edit in one click,
the sample and sizes kept, which makes the next version.

    wpt depot tokenizer new NAME --snapshot S [--kind plain] --sample-bytes N --vocab-size V [--note T]
    wpt depot tokenizer edit NAME [--snapshot S] [--kind plain] [--sample-bytes N] [--vocab-size V] [--note T]

`--snapshot` takes a snapshot's id, or the name of one of the depot's own.
The page "Tokenizers" of `wpt depot web` makes and changes them the same
way, by clicks: a snapshot from a list (the latest first), the kinds as
chips, the sizes as preset buttons, and the name filled in.

### Training it

`wpt depot tokenizer train NAME`, or the button "Train its vocabulary" on
the tokenizer recipe's page, trains the vocabulary of the recipe's current version;
a corpus build never trains one. The web app shows the training's estimate
first, the sample's bytes and a time taken in proportion to the one
training measured, and runs the training as a job of its queue only once
it is confirmed ([depot-web.md](depot-web.md)).

The sample is the plain texts of the recipe's snapshot in the order a
corpus takes them (the order of their export keys, a long text as its
pieces), each at its revision as of the snapshot, filling `sample_bytes`
as a corpus fills a part ([data-selection.md](data-selection.md), "Filling
a part" and "Long texts"). Items are no sample. The list of the texts a
vocabulary trained on, and its count, name each text once, a long text
whose pieces the sample took included. The training reads the sample
once, as a stream. A snapshot with no plain text within `sample_bytes`
trains nothing (`NoSample`).

The depot keeps the vocabulary with the list of every text it trained on,
in the order they were read, by its depot (empty for the depot that holds
the recipe, the web address for another depot) and its id there.

The freight learns a vocabulary too, outside the depot: a request names a
snapshot, its shares and a sample in bytes, and the program
`wpt-vocabulary` learns from the plain texts the freight chooses, read once
in the order of the snapshot's listing. Each share fills its proportion of
the sample's bytes, by group, as a corpus fills its shares
([data-selection.md](data-selection.md), "Shares" and "Filling a part").
The delivery on the dock holds `vocabulary.json`, and its items tables
list the texts it read, in that order ([api/freight.md](api/freight.md),
[api/preparation.md](api/preparation.md), "wpt-vocabulary"). An erasure is
written into its delivery note and does not make it stale.

### One vocabulary per version

The depot holds at most one vocabulary for each version of a recipe, and
refuses a second one. From then on, every build that names that version's
vocabulary, by its sha256, uses it:

- **Adding or changing texts never retrains it.** Many recipes and
  models, whatever their training settings, share one vocabulary, so a
  model can keep training on new data. A text the vocabulary was not
  trained on still encodes exactly, since every byte has a token, only less
  compactly.
- **A new vocabulary takes a new version.** For a big change (much more
  data, another size), change what trains it, which raises the version,
  and train again. A change of the tokenizer's code asks for no new
  version: the depot keeps the vocabulary.
- **A build whose vocabulary the depot does not hold builds nothing**
  (`NoVocabulary`), from the command line and from the web app alike.
- **Erasure does not reach a trained vocabulary** until the version is
  raised: merges learned from text that was erased later stay in it (see
  [erasure.md](erasure.md)).

### Importing one

`wpt depot tokenizer import NAME FILE --snapshot S [--kind plain]` makes
a recipe at version 1 from a recipe file (JSON with `sample_bytes`,
`vocab_size` and an optional `note`), whose sample reads the snapshot S. With `--lock LOCK`, it
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

- The corpus build uses only a vocabulary the depot holds; the corpus
  record names its recipe, its version, its sha256 and the vocabulary's
  file (`{"key", "sha256"}`, in the corpus's folder), and the corpus id
  covers its sha256.
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
