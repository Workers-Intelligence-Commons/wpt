"""Which texts of a corpus carry a tag: the corpus's texts table holds each
text's tags (wpt_corpus.encoding), in parts, and a step that learns from conversations or
tasks picks the ones with its `tag` setting. Of a corpus cut into portions,
only the rows of the portions training reads count (the record's
`trains_on`)."""

import pyarrow.parquet as pq


def tagged(record, dock, tag) -> set[tuple[str, str]]:
    """(depot address, text id) of every text of the corpus carrying tag."""
    found, trains_on = set(), record.get("trains_on")
    for part in record["tables"]["texts"]:
        table = pq.read_table(dock.path(part), columns=["depot", "id", "tags"] + (["portion"] if trains_on else []))
        portions = table["portion"].to_pylist() if trains_on else [None] * table.num_rows
        found |= {(depot, text) for depot, text, tags, portion in zip(
            *(table[name].to_pylist() for name in ("depot", "id", "tags")), portions)
            if tag in tags and (not trains_on or portion in trains_on)}
    return found
