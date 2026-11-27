# Data descriptions

This folder describes data; it never holds data itself. The texts live in
a depot (`depot`).

- `orders/<name>.json`: model families, each everything that changes the
  models of its sizes: the depot and the snapshot they train on, with the
  roles of the corpus's portions, the shares, tokens and vocabulary they
  take, their architecture and training steps, written once, then a table
  of what varies between sizes. `FAMILY:SIZE` names one size.
  See [docs/model.md](../docs/model.md), "Model families".
- `images/<name>/`: the container images of agentic training: a
  `Dockerfile`, and the `lock.json` and `licenses.json` that
  `wpt agent lock-image <name>` writes; `wpt agent build-image <name>`
  builds the image from them on each machine. See
  [docs/training.md](../docs/training.md).
