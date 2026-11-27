"""What a model folder's provenance.json says of the corpus it learned
from, read from any of its three shapes, so every reader (the NOTICE, the
model card, wpt eval) says the same and never rewrites a folder.

A model trained on deliveries of the freight (docs/training.md, "What a
model records") names, in `data`, the data section of its order: the
depot's address, the snapshot's name, its change and digest; in
`deliveries`, the key, id and preparation of its vocabulary's delivery and
of its corpus's; in `corpus`, the corpus delivery's id, key and rules; and
`order_name` and `repository_commit`; a model trained by a size of a
family also names its `family`, `size` and, for another run of the size,
`run`, and `order` is that size's resolved model order. A model trained on a corpus the
depot built from a snapshot names the same `data`, and in `corpus` the
corpus's id, build key and rules. A model trained before snapshots names,
in `corpus`, the corpus's id, its depot and the name of its dataset.
A folder written before the keys `order_name` and `order` names them
`recipe_name` and `recipe`, which every reader here takes as well.

    corpus_id(provenance)      what names the corpus: its id, or its build key when the folder names no id
    corpus_label(provenance)   one sentence of what the model learned from, in any shape
    on_snapshot(provenance)    whether the folder names a snapshot (the two newer shapes)
    deliveries(provenance)     {vocabulary, corpus: {key, id, preparation or adopted}}, {} for the older shapes
    order_name(named)          the order's name a provenance, or a drafter's trained_by, gives, under either key
    trained_by(provenance)     the family and size the model was trained by, or its order's name, for people
    order(provenance)          the order the model was trained by, under either key; {} without one
"""


def on_snapshot(provenance: dict) -> bool:
    """Whether provenance names the snapshot it was trained on, rather
    than a dataset (a folder written before snapshots)."""
    return "snapshot" in _data(provenance)


def corpus_id(provenance: dict) -> str:
    """The id of the corpus the model learned from, or its build key when
    the folder names no id."""
    corpus = provenance["corpus"]
    return corpus.get("id") or corpus["build_key"]


def corpus_label(provenance: dict) -> str:
    """What the model learned from, in one sentence without its final
    full stop: "snapshot S (change N, digest H) of the depot D, family F,
    size Z (or order R), code at commit X" for a model trained on a snapshot (a delivery's or a
    corpus the depot built), "corpus C, built
    from dataset T of the depot D" for one trained before snapshots."""
    if not on_snapshot(provenance):
        corpus = provenance["corpus"]
        return f"corpus {corpus['id']}, built from dataset {corpus['dataset']} of the depot {corpus['depot']}"
    data = _data(provenance)
    commit = provenance.get("repository_commit")
    code = (f"code at commit {commit}" + (" with uncommitted changes" if provenance.get("repository_dirty") else "")
            if commit else "code outside a git checkout")
    return (f"snapshot {data['snapshot']} (change {data['change']}, digest {data['digest']}) of the depot "
            f"{data['depot']}, {trained_by(provenance)}, {code}")


def trained_by(provenance: dict) -> str:
    """The order the model was trained by, for people: "family F, size S"
    (and ", run R" for another run of the size) for a folder that names
    them, else "order NAME"."""
    if provenance.get("family"):
        return (f"family {provenance['family']}, size {provenance['size']}"
                + (f", run {provenance['run']}" if provenance.get("run") else ""))
    return f"order {order_name(provenance)}"


def _data(provenance: dict) -> dict:
    """The data section the model's order gave: provenance's own `data`,
    else the order's."""
    return provenance.get("data") or order(provenance).get("data") or {}


def order_name(named: dict) -> str | None:
    """The name of the order a provenance.json, or a drafter's
    `trained_by`, names: `order_name`, or `recipe_name` in a folder
    written before it."""
    return named.get("order_name", named.get("recipe_name"))


def order(provenance: dict) -> dict:
    """The order the model was trained by: `order`, or `recipe` in a
    folder written before it; {} when the folder names neither."""
    return provenance.get("order") or provenance.get("recipe") or {}


def deliveries(provenance: dict) -> dict:
    """The deliveries the model learned from, by role (`vocabulary`,
    `corpus`), each its key, id and preparation (or, for an adopted one,
    where it comes from); {} for a folder trained on a corpus the depot
    built, or before snapshots."""
    return provenance.get("deliveries") or {}
