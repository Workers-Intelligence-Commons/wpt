# A preparation and its messages

A preparation is a program the freight runs on the items it chose. The
freight starts it on its own machine as a child process, one program for
one job, and talks to it through the program's standard input and output.
The program never reads the depot and never writes the dock: it reads what
the freight sends, writes files into the local folder the freight names,
and says what it wrote. The freight knows nothing of what a preparation
makes.

The freight finds a program by its name: an executable of that name beside
the Python the freight runs on (the console scripts installed with it),
else on the search path (`freight.exchange.find`). WPT's programs,
`wpt-vocabulary`, `wpt-tokens` and `wpt-count`, are console scripts of the
package `wpt-corpus` (`wpt_corpus.vocabulary`, `wpt_corpus.programs`). A change to these messages changes
this document in the same commit.

## The format

Each message is one line of JSON in UTF-8, ended by a line feed. Only the
freight's `weigh` and `take` carry content: each names its length in
`bytes`, and exactly that many bytes follow the line. No message of the
program carries content, so in its messages `bytes` is a field like any
other. The format does not depend on the channel (`freight.messages`,
`wpt_corpus.channel`).

## From the freight

| Message | Meaning |
| --- | --- |
| `{"type": "start", "settings": {...}, "inputs": {"<name>": "<local folder>"}, "out": "<local folder>", "threads": N}` | the first message; `inputs`: the folder of each delivery the request names, holding its files; `threads`: how many the program may use; `resume`, when the job goes on (below) |
| `{"type": "weigh", "ref": R, "kind": K, "bytes": B}` + content | how much does this item weigh in your unit? |
| `{"type": "take", "ref": R, "depot": D, "id": I, "name": N, "piece": P, "kind": K, "pool": T, "bytes": B}` + content | prepare this item; the items come in the order of the delivery. `depot`: the depot that owns the item; `name`: the item's id, or `<id>#<n>` for the piece `n` of a long text; `piece`: `n`, or null; `pool`: the tag of the share a plain text was taken for, `""` for an item of another kind |
| `{"type": "end"}` | no more items |

The content of a plain text is its UTF-8 text, a piece's the piece. An
item's content is its JSON, and a conversation's excerpts each come
written as the passage it quotes.

## From the program

| Message | Meaning |
| --- | --- |
| `{"type": "ready", "name": "...", "version": "...", "unit": "tokens", "kinds": [...], "weighs": true, "on_erasure": "stale"}` | the answer to `start`; `version` changes whenever the same messages give other files or weights, and the delivery key covers it; `unit`: what its weights count, `null` for a program that gives none; `kinds`: the kinds it prepares; `weighs`: whether it answers `weigh`, which needs a unit; `on_erasure`, optional: what an erasure that reaches one of the delivery's items does, `stale` (the default) or `note` (the erasure written into the delivery note, which stays fresh; [freight.md](freight.md), "Erasures") |
| `{"type": "weight", "ref": R, "weight": N}` | the answer to `weigh` |
| `{"type": "taken", "ref": R, "weight": N, "at": {...}}` | the answer to `take`, in the order of the `take` messages; `weight`: a whole number in its unit, absent or `null` for a program with no unit; `at`: where the item lies in the files, each field a whole number copied into its row of the items table, and named as none of the fields the freight gives that row (`depot`, `id`, `revision`, `kind`, `title`, `authors`, `address`, `licenses`, `tags`, `notice`, `scores`, `piece`, `quotes`, `bytes`, `weight`; [freight.md](freight.md), "A delivery on the dock"): such a name fails the job |
| `{"type": "file", "path": "...", "items": N, "hash": "...", ...}` | a file under `out` is complete; `items`: the items it holds; `hash`, optional: the hash of its content the delivery id covers in place of the file's sha256, for a file whose bytes may differ where its content does not; any other field is the program's own. The delivery note keeps the whole message apart from the file's place on the dock, and the freight sends it back on a resume |
| `{"type": "log", "level": "...", "text": "..."}` | a line for the job's log; `level` one of `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `{"type": "done", "result": {...}}` | the answer to `end`, after its last `file` |
| `{"type": "failed", "reason": "..."}` | the program cannot do its part; the job fails with that reason |

## Rules of the exchange

- The same messages in the same order give the same files, byte for byte
  where the format allows, and always the same content hashes, on any
  machine and with any number of threads.
- A `weight` never depends on the other items, and a `taken` gives the
  item the weight its `weigh` gave.
- The freight may send many messages before it reads an answer: the
  program reads them while it works and answers in order. The freight
  sends from a thread of its own, so neither side waits on a full pipe.
- A program that exits without `done` or `failed`, or writes a line that
  is no message, fails the job.
- A file is uploaded once the next `taken` or the `done` after its `file`
  message arrives: the freight then uploads every file said, writes the
  items rows taken so far, and keeps a checkpoint. The program does not
  touch a file after it said it.
- To go on after a job was cut short, the freight starts the program again
  with the same `out` folder and sends `start` with `"resume": {"files":
  [...], "after": R}`: the `file` messages already uploaded, as the program
  sent them, and the `ref` of the last item they hold. The freight then
  sends the items after that one. The program goes on from what those
  messages say and writes the same files a run that never stopped writes.
  A `file` message therefore carries whatever the program needs to go on.

## The cost of `weigh`

A size in tokens makes the freight weigh every text before it takes it:
the most talkative part of a job. Measured on the texts of the test depot
(`apps/wpt/tests/conftest.py`, its stories and fables and a
conversation), 4,600 items of 1.2 KB on average, `weigh` through the
channel with every message of a batch sent before its answers are read
takes 1.3 times the in-process count with one thread (59 against 47
microseconds an item) and 1.6 times with four, the program's threads not
helping texts this short. With texts a hundred times longer (92 items of
119 KB), the channel takes 0.6 times the in-process count: the program
encodes them on its threads while the freight reads the next. The cost of
the channel is some 15 to 30 microseconds an item, small beside reading
the texts from the depot, so the freight sends one `weigh` message an
item, a batch of them at a time (`freight.job`: at most 8 MB or 4,096
texts), and does not put several items in one message.

## wpt-vocabulary

`wpt-vocabulary` learns a vocabulary. Its one setting, `vocab_size`, is
the size of the vocabulary, at least 513. It prepares plain texts only and
takes no input. It has no unit and weighs nothing (`unit: null`,
`weighs: false`): its sample is a size in bytes, which the freight
measures itself on the bytes its listing gives each text
([freight.md](freight.md), "A request"). Each `taken` carries no weight
and no `at`, so the delivery's shares count their texts' `bytes` and
their `weight` is `null`.

- The texts go to the tokenizer's training as they come, read once in the
  order sent, as `wpt_tokenizer.Tokenizer.train` reads a stream, so the
  same texts in the same order give the same vocabulary
  ([tokenizer.md](../tokenizer.md)).
- At `end` it writes `vocabulary.json`, the canonical vocabulary file,
  says it, and answers `{"sha256", "size", "tokenizer_version"}`.
- Its version names the tokenizer's version, so the delivery key changes
  with it.
- It learns from every text at once and cannot go on after a stop: it
  answers a `start` that resumes with `failed`. Since it says its one file
  at the end, the freight keeps no checkpoint before it, and a job cut
  short starts over.
- `on_erasure` is `note`: an erasure that reaches a text it read is
  written into its delivery note, which stays fresh, and the merges
  learned from that text stay.

## wpt-tokens

`wpt-tokens` writes a corpus. Its input `vocabulary` is a folder holding
`vocabulary.json`, the canonical vocabulary file (`wpt_tokenizer`). Its
one setting, `shard_tokens`, is how many tokens a token file holds before
it closes (about 64 million by default). Its unit is tokens, and it
prepares conversations, plain texts and tasks.

- A plain text or a piece becomes its tokens followed by END, encoded a
  block at a time; a conversation goes through the chat template
  (`wpt_corpus.encoding`). Each goes into a token file of its kind
  (`plain-NNNNN.parquet`, `conversation-NNNNN.parquet`; the format:
  `wpt_corpus.shards`), one row per item: its `source` the item's depot,
  its `document` the item's `name`, its `pool` the share's tag.
- A task becomes one line of `tasks.jsonl`, `{"depot", "id", "content"}`.
  It weighs the tokens of its content alone, written as JSON text with
  its keys sorted (`wpt_corpus.encoding.task_text`), not those of the
  whole line.
- Once a token file holds `shard_tokens` tokens, the token file of each
  kind closes, and each is said as it closes; `tasks.jsonl` is said at the
  end. Its `file` messages carry `kind`, `tokens`, `token_sha256`, `rows`,
  `sources`, `hash` (its `token_sha256`) and `so_far` (what was taken up
  to then), which a resume reads back.
- An item's `at` is `{"tokens": N}`: the position of its first token over
  the token files of its kind in order, a task's line in `tasks.jsonl`.
- Its result: `total_tokens` (plain text), `items_tokens` (conversations
  and tasks), `shards` (by kind, each file's `path`, `token_sha256`,
  `rows`, `tokens`, `sources`), `tasks` (`path`, `count`, or null),
  `fingerprints` (by kind, as a training step reading that kind names it)
  and `vocabulary` (`sha256`, `size`).

The files and hashes are those `wpt depot build` writes for the same
snapshot, data section and vocabulary
(`apps/wpt/tests/freight/test_requests.py`).

## wpt-count

`wpt-count` weighs as `wpt-tokens` does and writes nothing: no `at`, no
file. Its result is `{"total", "plain", "items"}`, the tokens of what it
took.
