# Data descriptions

This folder describes data; it never holds data itself. The texts live in
a depot (`wpt depot`).

- `recipes/tokenizers/`: tokenizer recipes and the lock files their
  training writes.
- `recipes/models/`: model recipes, each everything that changes one model:
  the depot and dataset it trains on, its architecture and training steps.
  See [docs/model.md](../docs/model.md).
- `recipes/series/`: model series, each deriving the model recipes of its
  sizes.
- `images/<name>/`: the container images of agentic training: a
  `Dockerfile`, and the `lock.json` and `licenses.json` that
  `wpt agent lock-image <name>` writes; `wpt agent build-image <name>`
  builds the image from them on each machine. See
  [docs/training.md](../docs/training.md).
