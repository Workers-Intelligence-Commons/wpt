"""Which texts of a corpus carry a tag: the corpus's texts table holds each
text's tags (wpt_corpus.encoding), in parts, and a step that learns from conversations or
tasks picks the ones with its `tag` setting."""

import pyarrow.parquet as pq


def tagged(record, store, tag) -> set[tuple[str, str]]:
    """(depot address, text id) of every text of the corpus carrying tag."""
    found = set()
    for part in record["tables"]["texts"]:
        table = pq.read_table(store.path(part), columns=["depot", "id", "tags"])
        found |= {(depot, text) for depot, text, tags in zip(*(table[name].to_pylist()
                                                                for name in ("depot", "id", "tags"))) if tag in tags}
    return found
