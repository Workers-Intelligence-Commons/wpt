"""What a model folder's provenance.json says of the corpus it learned
from, read from any of its three shapes, so every reader (the NOTICE, the
model card, wpt eval) says the same and never rewrites a folder.

A model trained on deliveries of the freight (docs/training.md, "What a
model records") names, in `data`, the data section of its order: the
depot's address, the snapshot's name, its change and digest; in
`deliveries`, the key, id and preparation of its vocabulary's delivery and
of its corpus's; in `corpus`, the corpus delivery's id, key and rules; and
`recipe_name` and `repository_commit`. A model trained on a corpus the
depot built from a snapshot names the same `data`, and in `corpus` the
corpus's id, build key and rules. A model trained before snapshots names,
in `corpus`, the corpus's id, its depot and the name of its dataset.

    corpus_id(provenance)      what names the corpus: its id, or its build key when the folder names no id
    corpus_label(provenance)   one sentence of what the model learned from, in any shape
    on_snapshot(provenance)    whether the folder names a snapshot (the two newer shapes)
    deliveries(provenance)     {vocabulary, corpus: {key, id, preparation or adopted}}, {} for the older shapes
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
    full stop: "snapshot S (change N, digest H) of the depot D, order R,
    code at commit X" for a model trained on a snapshot (a delivery's or a
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
            f"{data['depot']}, order {provenance.get('recipe_name')}, {code}")


def _data(provenance: dict) -> dict:
    """The data section the model's order gave: provenance's own `data`,
    else the order's."""
    return provenance.get("data") or (provenance.get("recipe") or {}).get("data") or {}


def deliveries(provenance: dict) -> dict:
    """The deliveries the model learned from, by role (`vocabulary`,
    `corpus`), each its key, id and preparation (or, for an adopted one,
    where it comes from); {} for a folder trained on a corpus the depot
    built, or before snapshots."""
    return provenance.get("deliveries") or {}
