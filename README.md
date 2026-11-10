# WPT: Workers' Pre-trained Transformer

WPT is an open language model, trained from scratch only on content whose
license is clear, and published under copyleft (AGPL-3.0-or-later). It is
the first project of Workers' Intelligence Commons: intelligence is a means
of production, and it should stay a common good that anyone can use, host,
modify and improve. Anyone may serve WPT, for money too; whoever serves a
modified version must share it. The principles are in the
[charter](docs/charter.md).

This repository is everything needed to build a WPT model from nothing:
the code of the depot that holds the texts and the decisions on them, the
code that builds a corpus from it, trains, evaluates and runs the model, and the
recipes of the models. No checkpoint has been published yet.

## How it works

**The texts live in a depot; the repository holds none.** A depot is a
folder, known by its address, that holds texts and items (conversations,
tasks), each with its license, credit and notice, the human decisions that
admit or refuse it, the libraries that group admitted texts, and the
datasets that say what to train on (`wpt depot`). `wpt depot build DATASET`
builds a dataset into a corpus in a store you control (a folder or an S3
bucket): the token files, with copies of every text's credits, licenses and
notices. A model recipe names a depot address and a dataset in it, and
`wpt train` reads that dataset's corpus from the store. A depot can follow
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
  depot and drops every older state of it, so every later corpus is built
  without it ([docs/erasure.md](docs/erasure.md)).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), git and a Rust toolchain
(the tokenizer is built from source). On Windows, run WPT in Windows itself (PowerShell), not
inside WSL: a second system adds failures of its own and keeps files on a
disk Windows does not see. Linux containers, where a step needs them, run
through Docker Desktop.

```sh
uv run scripts/setup.py                            # the PyTorch build for this machine's GPU
echo "WPT_STORE=file://$HOME/wpt-store" >> .env    # where corpora, models and runs live
echo "WPT_DEPOT=$HOME/wpt-depot" >> .env           # the depot the wpt depot commands work on
uv run wpt doctor                                  # what this machine still lacks, and how to fix it

uv run wpt depot build DATASET                     # the corpus of the recipe's dataset, into the store; prints its id
scripts/detach.sh uv run wpt train classics-test --stream --out runs/first
uv run wpt eval runs/first                         # the tests, into the model card
uv run wpt chat runs/first                         # continue text with the model
```

Every job is a subcommand of one command, `wpt` (`uv run wpt --help`
lists them, and `uv run wpt SUBCOMMAND --help` gives each one's options).
The store holds plain named things: `corpora/`, `models/` and `runs/`
(`wpt store list` lists the corpora and models).

`classics-test` is a small model recipe, enough to see every step work on
one machine; its depot address and dataset name are for a person to fill
in. A model recipe
(`data/recipes/models/`) holds everything that changes the model: its
data, architecture and training steps
([model.md](docs/model.md)); the command-line options only say how the
hardware runs it.

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
| share data and runs between machines | `WPT_STORE=s3://bucket/prefix` (any S3-compatible service) |
| move a run to another machine | `--run NAME`: the checkpoint lives in the store; `--run NAME --resume` there |
| use several GPUs | `--strategy ddp --processes N`, on the gpu worker's machine |
| train on another machine's GPUs | a gpu worker on each machine (`deploy/gpu-worker/compose.yaml`, with `compose.nvidia.yaml` or `compose.amd.yaml` for GPUs), their keys in `WPT_GPU_WORKERS` ([machines](docs/machines.md)) |
| rent the GPUs | `WPT_GPU_WORKERS=N` and `WPT_RENTAL_GPUS` for `wpt manager`, which rents N gpu workers through SkyPilot (as one cluster in one request), replaces a lost one and gives them back at the end ([machines](docs/machines.md)) |
| train on a built corpus in one command | `docker compose up`, with `COMPOSE_FILE=deploy/manager/compose.yaml` and `WPT_RECIPE` set in `.env`; then `wpt store copy-out NAME runs/NAME` ([machines](docs/machines.md)) |
| teach the model to chat | conversations in the dataset, the corpus built again, and a recipe whose `steps` add `answer-as-assistant` after `read-text`: `wpt train RECIPE CORPUS_ID --out runs/chat`, then `wpt chat runs/chat --chat` ([training](docs/training.md)) |
| teach the model to finish tasks as an agent | tasks in the dataset, the corpus built again, their images in `data/images/` locked with `wpt agent lock-image NAME` and built with `wpt agent build-image NAME`, and a recipe whose `steps` hold `finish-tasks` (needs Docker; [training](docs/training.md)) |
| teach the agent to reach the same result with fewer tokens | `finish-tasks-with-less-writing` after `finish-tasks` in the recipe's `steps`: it trains on the same tasks ([training](docs/training.md)) |
| train long windows on a small GPU | `--recompute`: layers run again in the backward pass instead of keeping their values |
| train faster on a CUDA device (an NVIDIA or AMD GPU) | `--compile`: the passes over whole text windows run through `torch.compile` ([training](docs/training.md), "Compiling the passes") |
| add texts, conversations and tasks, decide on them, and make libraries and datasets, in a browser | `wpt depot web` (needs Bun and `git submodule update --init`), then http://127.0.0.1:8410/ (this machine only; another one reaches it through an SSH tunnel or Tailscale); the same through the `wpt depot` commands ([architecture](docs/architecture.md), "Two interfaces") |

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
- [Data selection: texts, tags, libraries, datasets, corpora](docs/data-selection.md)
- [Tokenizer](docs/tokenizer.md)
- [Model architecture: the research it comes from and the recipes that size it](docs/model.md)
- [Training and repeatability](docs/training.md)
- [Machines for training: the wpt manager, gpu workers, sandboxes](docs/machines.md)
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
