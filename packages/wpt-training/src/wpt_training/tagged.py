"""Which texts of a corpus carry a tag: the corpus's texts table holds each
text's tags (wpt_depot.corpus), and a step that learns from conversations or
tasks picks the ones with its `tag` setting."""

import pyarrow.parquet as pq


def tagged(record, store, tag) -> set[tuple[str, str]]:
    """(depot address, text id) of every text of the corpus carrying tag."""
    table = pq.read_table(store.path(record["tables"]["texts"]), columns=["depot", "id", "tags"])
    return {(depot, text) for depot, text, tags in zip(*(table[name].to_pylist() for name in ("depot", "id", "tags")))
            if tag in tags}
