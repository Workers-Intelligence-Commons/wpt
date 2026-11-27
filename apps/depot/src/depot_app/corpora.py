"""The corpora the dock keeps under corpora/, as an upgrade from a format
before snapshots reads their records (depot.upgrade): the dataset
revisions they were built from become snapshots. The depot only reads
them: the freight marks stale those an erasure reached (freight.erasures).
The depot reaches the dock through dock.api alone (docs/architecture.md,
"The parts"); dock is the dock's URL.

    records(dock)                        every corpus the dock holds, newest first
"""

from dock import api

FOLDER = "corpora"
RECORD = "corpus.json"


def records(dock: str) -> list[dict]:
    """The record of every corpus the dock holds, each checked by the
    dock, the last built first."""
    found = []
    for name in api.folders(dock, f"{FOLDER}/"):
        if (record := api.get_json(dock, f"{FOLDER}/{name}/{RECORD}")) is not None:
            found.append(record)
    return sorted(found, key=lambda record: record["built_at"], reverse=True)
