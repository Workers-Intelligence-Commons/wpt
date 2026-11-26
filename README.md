# WPT: Workers' Pre-trained Transformer

WPT is an open language model, trained from scratch only on content whose
license is clear, and published under copyleft (AGPL-3.0-or-later). It is
one of the projects of Workers' Intelligence Commons, which works on
artificial intelligence in general, whatever its form: intelligence is a
means of production, and it should stay a common good that anyone can use,
host, modify and improve. Anyone may serve WPT, for money too; whoever serves a
modified version must share it. The principles are in the
[charter](docs/charter.md).

This repository is everything needed to build a WPT model from nothing:
the code of the depot that holds the texts and the decisions on them, of
the freight that prepares a vocabulary and a corpus from it, of the dock
that keeps what machines made, the code that trains, evaluates and runs
the model, and the orders of the models. The depot, the freight and the
dock are meant to serve other models and other kinds of data too, and to
move to a repository of their own
([architecture](docs/architecture.md), "The parts"). No checkpoint has been published yet.

## How it works

**The texts live in a depot; the repository holds none.** A depot is an
SQLite file, known by its address, that holds texts and items (conversations,
tasks), each of one kind and with its license, credit, notice and tags,
the human decisions that admit or refuse it, and its snapshots, each a
change number, named by the UTC time it was made, and the texts admitted
then (`depot`). A model order names a depot by its public address, a
snapshot of it with the change and digest it reads, the kinds and the
shares by tag it takes, its size and its vocabulary. `wpt build-resources
ORDER` asks the freight to learn that vocabulary and prepare that corpus,
which it delivers on a dock you control (a folder or an S3 bucket): the
token files, with copies of every text's credits, licenses and notices.
`wpt train` reads that corpus from the dock. A corpus is a cache: the
depot's address, the snapshot and the order make it again, the same. A depot can follow
another and pull its updates, the way a Git clone follows its remote.
WPT hosts no data.

- **Clear licenses only.** A text enters only when its license lets the
  weights be distributed under the AGPL; when in doubt, it stays out
  ([docs/licenses.md](docs/licenses.md)). Owners' "no AI training" wishes
  are honored.
- **People decide.** Every decision on a text carries the decider's
  declared name and email, which are public; a text trains only when
  someone admitted it and nobody refused it.
- **No distillation.** Trained from scratch, with no output of another
  company's model in the data and no existing base model.
- **Provenance travels with the model.** Every model folder carries a
  `NOTICE` and `CREDITS` naming each text it learned from, with its credit, and a
  `provenance.json` recording exactly what produced it.
- **Erasure on request**: an admitted erasure cuts the passage from the
  depot and drops every older state of it, so every later corpus is made
  without it ([docs/erasure.md](docs/erasure.md)).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), git and a Rust toolchain
(the tokenizer is built from source). On Windows, run WPT in Windows itself (PowerShell), not
inside WSL: a second system adds failures of its own and keeps files on a
disk Windows does not see. Linux containers, where a step needs them, run
through Docker Desktop.

```sh
uv run scripts/setup.py                            # the PyTorch build for this machine's GPU
echo "DOCK=file://$HOME/dock" >> .env    # where deliveries, models and runs live
echo "DEPOT=$HOME/depot.sqlite" >> .env    # the depot the depot commands work on
echo "FREIGHT_HOME=$HOME/freight" >> .env  # the freight's queue and the work of its jobs
uv run wpt doctor                                  # what this machine still lacks, and how to fix it

uv run depot snapshot new                      # the texts admitted now, named by the UTC time
cp data/orders/models/classics-test.json data/orders/models/first.json
uv run depot web                               # the snapshot's page gives its name, change and digest
# put them in the "data" of data/orders/models/first.json with a vocabulary to learn, then commit the file
uv run wpt build-resources first                   # its vocabulary and corpus, from the freight, on the dock
scripts/detach.sh uv run wpt train first --stream --out runs/first
uv run wpt eval runs/first                         # the tests, into the model card
uv run wpt chat runs/first                         # continue text with the model
```

Every job of WPT is a subcommand of one command, `wpt` (`uv run wpt --help`
lists them, and `uv run wpt SUBCOMMAND --help` gives each one's options);
the depot, the freight and the dock have their own commands, `depot`,
`freight` and `dock`. The dock holds plain named things: `deliveries/`,
`corpora/` (the corpora the depot built before the freight), `models/` and
`runs/` (`freight deliveries` lists the deliveries, `wpt list` the corpora
and models).

A model order (`data/orders/models/ORDER.json`) holds everything that
changes the model: its data, architecture and training steps
([model.md](docs/model.md)); the command-line options only say how the
hardware runs it. Its data section names the depot, the snapshot (its
name, change number and digest) and the vocabulary (its sha256 and size,
or the size and sample of one to learn), so it is made only from the depot
it names: a snapshot's page in `depot web` gives its name, change and
digest, to put in the order file and commit, since the corpus and the
model name the commit they were built from. `classics-test` is a
small model order, enough to see every step work on one machine, which
the quick start copies for its model part. Like every order of the repository, it names the dataset
`test1` (id `wic-1`) of the public depot
https://depot.workers-intelligence-commons.org, in the legacy shape of
orders written before snapshots ([model.md](docs/model.md), "Model
orders"), which `wpt build-resources` refuses: it trains only on a corpus
of that dataset a dock holds.

**Streaming** keeps disk use small at each step, for data larger than the
disk:

- `wpt train --stream` reads the training shards a few at a time (8 by
  default, about 2 GB, `--stream-shards`) instead of copying them all to
  disk first.

**Long runs** start detached (`scripts/detach.sh` on Linux and macOS,
`.\scripts\detach.ps1` on Windows), so closing the terminal or the SSH
session does not stop them; Ctrl+C or a shutdown writes a checkpoint, and
the same command with `--resume` continues.

**Bigger setups:**

| To | Use |
| --- | --- |
| share data and runs between machines | `DOCK=s3://bucket/prefix` (any S3-compatible service) |
| move a run to another machine | `--run NAME`: the checkpoint lives on the dock; `--run NAME --resume` there |
| use several GPUs | `--strategy ddp --processes N`, on the gpu worker's machine |
| train on another machine's GPUs | a gpu worker on each machine (`deploy/gpu-worker/compose.yaml`, with `compose.nvidia.yaml` or `compose.amd.yaml` for GPUs), their keys in `WPT_GPU_WORKERS` ([machines](docs/machines.md)) |
| rent the GPUs | `WPT_GPU_WORKERS=N` and `WPT_RENTAL_GPUS` for `wpt manager`, which rents N gpu workers through SkyPilot (as one cluster in one request), replaces a lost one and gives them back at the end ([machines](docs/machines.md)) |
| train on a built corpus in one command | `docker compose up`, with `COMPOSE_FILE=deploy/manager/compose.yaml` and `WPT_ORDER` set in `.env`; then `wpt copy-out NAME runs/NAME` ([machines](docs/machines.md)) |
| teach the model to chat | conversations in a new snapshot, its corpus made, and an order whose `steps` add `answer-as-assistant` after `read-text`: `wpt train ORDER CORPUS_ID --out runs/chat`, then `wpt chat runs/chat --chat` ([training](docs/training.md)) |
| teach the model to finish tasks as an agent | tasks in a new snapshot, its corpus made, their images in `data/images/` locked with `wpt agent lock-image NAME` and built with `wpt agent build-image NAME`, and an order whose `steps` hold `finish-tasks` (needs Docker; [training](docs/training.md)) |
| teach the agent to reach the same result with fewer tokens | `finish-tasks-with-less-writing` after `finish-tasks` in the order's `steps`: it trains on the same tasks ([training](docs/training.md)) |
| train long windows on a small GPU | nothing: each stage whose pass does not fit the GPU's memory runs its layers again in the backward pass instead of keeping their values; `--recompute` does it in every stage, `--no-recompute` in none |
| train without `torch.compile` | `--no-compile`: the passes over whole text windows, compiled by default on a CUDA device (an NVIDIA or AMD GPU) with Triton and a C++ compiler, run uncompiled ([training](docs/training.md), "Compiling the passes") |
| add texts, conversations and tasks, decide on them, tag them and make snapshots, in a browser | `depot web` (needs Bun and `git submodule update --init`), then http://127.0.0.1:8410/ (this machine only; another one reaches it through an SSH tunnel, Tailscale, or a proxy in front of it ([depot-web](docs/depot-web.md))); the same through the `depot` commands ([architecture](docs/architecture.md), "Two interfaces") |

Every option also comes from the environment (`--micro-batch` is
`WPT_MICRO_BATCH`), and `--help` lists them all. Details:
[training](docs/training.md), [architecture](docs/architecture.md).

## Contributing

Every commit is signed off (`git commit -s`), and `uv run ci/all.py` runs the
checks your change touches before you push; see
[CONTRIBUTING.md](CONTRIBUTING.md). License questions the project cannot
settle alone are listed for legal review in
[LEGAL-QUESTIONS.html](LEGAL-QUESTIONS.html).

## Documentation

- [Charter](docs/charter.md)
- [Architecture](docs/architecture.md)
- [Licenses and what the corpus accepts](docs/licenses.md)
- [Questions for legal review, in English and French](LEGAL-QUESTIONS.html)
- [Erasure requests](docs/erasure.md)
- [Data selection: texts, tags, snapshots, corpora](docs/data-selection.md)
- [The depot's api: what code outside the depot asks of it](docs/api/depot.md)
- [The freight's api: requests, deliveries and the `freight` command](docs/api/freight.md)
- [A preparation and its messages: the programs the freight runs](docs/api/preparation.md)
- [The dock's api: find, fetch, put and the `dock` command](docs/api/dock.md)
- [Tokenizer](docs/tokenizer.md)
- [Model architecture: the research it comes from and the orders that size it](docs/model.md)
- [Training and repeatability](docs/training.md)
- [Knowledge: what sizes a model, the evidence and what is open](docs/knowledge/README.md)
- [Machines for training: the wpt manager, gpu workers, sandboxes](docs/machines.md)
- [The public depot web app: its containers, limits and host](docs/depot-web.md)
- [Evaluation](docs/evaluation.md)
- [Observability: the record of every run, and its live view](docs/observability.md)
- [Chat format: conversations as JSON, turned into tokens by a template](docs/chat-format.md)
- [Output notice and attribution](docs/output-notice.md)
- [Publishing models: releases, names, where](docs/publishing.md)

## License

Code, documents and the weights of WPT models: the
[GNU Affero General Public License, version 3 or later](LICENSE). Text a
WPT model generates: the authors of WPT waive their rights in it under
CC0-1.0 ([OUTPUT-NOTICE](OUTPUT-NOTICE), pending legal review), but
anything it repeats from its training data or its context keeps its own
license. Data built by WPT's tools (corpora and vocabularies) is under the
AGPL as well, while each text's content inside it keeps its own license. [NOTICE](NOTICE) says where to find a given model's
notice.
