# The freight's api

The freight reads a request, chooses items in a depot's snapshot, runs one
preparation on what it chose, and leaves the result on the dock with a
delivery note. It finds a delivery again by its key, and marks stale what
an erasure reached. It reads the depot only through `wpt_depot.api`
([depot.md](depot.md)) and knows nothing of what its preparations make: a
preparation is a program it starts and talks to
([preparation.md](preparation.md)).

`freight.api` is what code outside the freight asks of it, and the only
module of the package `freight` that code outside it imports. The command
`freight` (`apps/freight`) is a controller over the same functions. A
change to the module changes this document in the same commit.

The dock is, for now, the store `wpt_store` opens: `WPT_STORE` names it.

## Settings

Each function takes the settings it needs as keyword arguments; one not
given is read from the environment, else from the `.env` file of the
working folder:

| Setting | What it names |
| --- | --- |
| `FREIGHT_HOME` (`home`) | the freight's own folder: its queue (`queue.sqlite`) and the work of its jobs (`work/<delivery key>/`), which nothing backs up |
| `WPT_DEPOT` (`depot`) | the depot it reads, by its address |
| `WPT_STORE` (`dock`) | the dock it delivers on, by its URL |
| `FREIGHT_CAP` (`cap`) | the bytes its deliveries may take on the dock together; none by default |

## Rules of every call

Every argument and every answer is a value JSON can carry. No call hands
out a connection, a row object or a path into the freight's own files, so
each can become a route without its callers changing.

## A request

```json
{
  "depot": "<the depot's public address>",
  "snapshot": {"name": "<name>", "change": 32118, "digest": "<sha256>"},
  "selection": {
    "kinds": ["conversation", "plain", "task"],
    "shares": [{"tag": "research-paper", "share": 0.35}, {"tag": "books", "share": 0.65}],
    "size": {"unit": "tokens", "amount": 1179648000}
  },
  "preparation": {
    "name": "wpt-tokens",
    "settings": {},
    "inputs": {"vocabulary": "<delivery key>"}
  }
}
```

- `snapshot`: the snapshot to choose in, its change number and digest
  checked against the depot's; `owner`, the depot that made it, when it
  is the upstream's (the depot asked by default).
- `selection.kinds`: the kinds to take, `[]` or absent for every kind the
  preparation prepares. `shares`: the share of each tag, absent for every
  text. `size`: `null` for everything the rules take, or an amount in the
  preparation's unit (`tokens` for WPT's). A size in bytes is refused for
  now.
- `preparation.inputs` names earlier deliveries by their delivery key. The
  freight fetches their files from the dock and hands the program a
  folder for each.
- A request names exactly one preparation.

The freight chooses as `wpt depot build` chooses
([data-selection.md](../data-selection.md), "Shares", "Filling a part",
"Long texts", "Corpora", "Conversations that quote texts";
`freight.selection`, `freight.groups`, `freight.fill`): the first share
whose tag a plain text has, the groups of a share weighed by the square
root of their bytes, a part closed at a thousandth or after a thousand
texts that do not fit, the items' natural share of a size, a conversation
only with every passage it quotes, a text once at most. With a size, a
text is weighed (`weigh`) before it is taken.

## The delivery key

`key(request, version)` gives the delivery key of a request whose
program answers `version`: the SHA-256 of the canonical JSON of the
snapshot's change and digest, the selection in canonical form (kinds
sorted, shares in order, size with its unit), the preparation's name,
version, settings and input keys, and the version of the selection rules
(`freight.selection.RULES`). It never covers the depot's address, the
snapshot's name, the code's commit or the machine: the same request lands
in the same folder wherever it runs, and a caller works the key out with
no depot and no freight at hand.

## The calls

`request(request, home=, depot=, dock=, cap=, log=)` queues a request and
gives its job's state. It first checks the request against the depot (its
snapshot, change and digest), the dock (its inputs) and the program (it
starts it once to read its version). A request whose delivery the dock
holds, fresh, is a job done at once, its answer that delivery's note.
`Refused` for a request that cannot be met; `Busy` when a job of the same
key is queued or running.

`wait(job, home=, seconds=)` gives the job's state once it ended, or after
`seconds`. While no other process runs the queue, this one runs it, the
jobs queued before first.

`work(home=, stop=)` runs the queue until `stop()` says so; `False` when
another process runs it.

`job(job)`, `jobs(limit=)` and `cancel(job)` give a job's state, every
job queued or running followed by the latest that ended, and cancel one.
A job's state is `{id, key, request, state, stages, progress, answer,
problem, created, started, ended, restarts}`.

`delivery(key, dock=)` gives the note of the delivery of that key, `null`
when the dock holds none; `deliveries_on(dock=)` every note, the last made
first; `items(key, dock=)` the rows of its items tables, in order.

`check(home=, depot=, dock=)` asks the depot which erasures it carried
out since the mark the freight saw last there, and marks stale every fresh
delivery whose items hold a text one of them reached: `{mark, erasures,
marked}`.

`adopt(key, files, result, about, dock=)` puts files made elsewhere on the
dock as the delivery of `key`, which starts with `adopted-`: `files`
maps a path under `files/` to a local file, `result` is what a preparation
would have answered, `about` what the note says of where they come from.

## The queue

The queue is the SQLite file `queue.sqlite` of the freight's home
(`freight.jobs`). One process at a time runs it (the lock
`worker.lock`), one job at a time, in the order the requests came, each
job in a process of its own that has the whole machine. A job's state is
`queued`, `running`, `done` (its answer, the delivery note), `refused`
(the request cannot be met as it is), `failed` or `cancelled`. Each
delivery key has at most one job queued or running. A job cut short goes
on from its checkpoint when the queue runs again, at most three times. A
cancelled job never runs; a running one's process is stopped.

## A delivery on the dock

```
deliveries/<delivery key>/
    delivery.json              the delivery note, written last
    items/part-NNNNN.parquet   the items taken, in order, with their credits
    licenses.parquet, notices.parquet
    files/...                  what the preparation wrote
```

A failed job leaves no delivery note. A job uploads the program's files as
it says them, with the items rows taken so far, and keeps a checkpoint;
run again, it goes on from there.

The delivery note holds:

| Field | What it holds |
| --- | --- |
| `key` | the delivery key |
| `id` | covers the key, the content hash of every file, the items rows, the licenses and the notices |
| `depot`, `request` | the request as given |
| `snapshot` | its `id`, `name`, `owner`, `change` and `digest` |
| `preparation` | its `name`, `version`, `unit`, `settings` and `inputs` (by delivery key) |
| `result` | the JSON the preparation answered with `done` |
| `files` | each file the program said: `path` (under `files/`), its key, `sha256`, `bytes`, and the fields of its `file` message |
| `tables` | the items parts (`rows`, `text_bytes`, `fingerprint` each), `licenses`, `notices` |
| `items` | `texts`, `pieces`, and the items taken by kind |
| `shares` | for each share, its `tag`, its `weight` and its groups (`tag`, `texts`, `weight`); `null` without shares |
| `left_out` | what was left out and why, as a corpus record counts it |
| `erasures` | the erasures that had reached the snapshot |
| `made_at`, `code_commit`, `rules` | when, from which code, under which selection rules |
| `stale`, `deleted` | once an erasure reached it, or once its files are deleted |

The items tables are the texts table of a corpus (`depot`, `id`,
`revision`, `kind`, `title`, `authors`, `address`, `licenses`, `tags`,
`notice`, `scores`, `piece`, `quotes`), followed by the fields of each
item's `at` and its `weight` in the preparation's unit. They are how an
erasure finds the deliveries that hold a text.

With a cap, a job that would pass it alone is refused before its first
upload. Otherwise it makes room as it uploads, deleting a file at a time:
first the files of a folder that holds no note (left by a job that failed),
then those of a delivery whose deletion is not whole, then those of the
delivery made least recently. A delivery whose files go is marked
deleted.

A request whose delivery the dock holds, fresh, is never run again. A
stale delivery is made again into the same folder, and the files its note
no longer names are deleted.
