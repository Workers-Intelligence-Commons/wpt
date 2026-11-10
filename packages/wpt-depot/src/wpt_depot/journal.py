"""The journal: the depot's changes. Delta Lake commits one table at a
time, so every change is written here whole first, and that write makes it
certain; the depot then applies it to each table and marks it done. An
entry's id is the change's number, the depot-wide change number that the
rows it writes record (wpt_depot.schema). A change left undone (a crash, a
failed write) is applied again when the depot is next opened, so applying
a change twice must leave the tables as applying it once does.

Rows numbered at a change left undone may already be written in some
tables and not in others, so a reader of a depot that another app writes
(a clone pulling its upstream) reads it only up to its settled change
number (Journal.settled): every row numbered up to it belongs to a change
written whole.

A change is a JSON object whose `kind` names how it applies (depot.py);
times in it are ISO 8601 text and bytes are hexadecimal text.
"""

import json

import pyarrow as pa

from .schema import rows
from .tables import Tables


class Journal:
    def __init__(self, tables: Tables):
        self.tables = tables

    def write(self, change: dict, number: int):
        """Write change as the entry number, not done."""
        self.tables.insert("journal", rows("journal", [{"id": number, "change": _text(change), "done": False}]))

    def done(self, entry: int):
        self.tables.update("journal", {"done": "true"}, f"id = {int(entry)}")

    def pending(self) -> list[tuple[int, dict]]:
        """(id, change) of every entry not done, in order."""
        found = self.tables.query("SELECT id, change FROM journal WHERE NOT done ORDER BY id")
        return [(one["id"], json.loads(one["change"])) for one in found.to_pylist()]

    def settled(self, version: int | None = None) -> int:
        """The highest change number up to which every change is written
        whole: the highest id of an entry done with no entry not done below
        it, or, with none, the change of the depot's first about row (its
        creation, or a clone's, written outside the journal). Read from the
        journal's current state alone (or its storage version `version`), so
        the states of the other tables taken after it hold every row numbered
        up to it, and a row numbered above it (of a change undone then, or
        begun later) is past it."""
        found = self.tables.read("journal", version, columns=["id", "done"]).to_pylist()
        first = min((one["id"] for one in found if not one["done"]), default=None)
        base = min(self.tables.read("about", columns=["change"])["change"].to_pylist(), default=0)
        return max([base, *(one["id"] for one in found if one["done"] and (first is None or one["id"] < first))])

    def entries(self, kind: str) -> list[tuple[int, dict]]:
        """(id, change) of every entry of that kind, in order."""
        found = self.tables.query(
            "SELECT id, change FROM journal WHERE json_extract_string(change, '$.kind') = ? ORDER BY id", [kind])
        return [(one["id"], json.loads(one["change"])) for one in found.to_pylist()]

    def rewrite(self, changes: dict[int, dict]):
        """Replace the change of each entry named, keeping whether it is done."""
        if changes:
            source = pa.table({"id": pa.array(list(changes), pa.int64()),
                               "change": pa.array([_text(change) for change in changes.values()], pa.string())})
            self.tables.merge_update("journal", source, {"change": "source.change"})


def _text(change: dict) -> str:
    return json.dumps(change, ensure_ascii=False, sort_keys=True)
