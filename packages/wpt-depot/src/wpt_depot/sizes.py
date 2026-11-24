"""A snapshot's texts by length: its texts from the longest to the
shortest (or the other way), each at its revision as of its bound
(wpt_depot.snapshots), read a page at a time with their text, so a person
reads a whole snapshot as one strip, the texts that take the most room
first.

A snapshot never changes, so its order by length is made once (build) and
kept in a file (save, load): one entry a text, its length in characters
(the content_length of its revision as of its bound, schema.SIZES), the
index of its owner among the snapshot's owners in the order of their
UTF-8 bytes, and its number, sorted by length, then in the numeric order
of ids (ids.order_sql: owner, then number), 14 bytes an entry. Making it
reads, for each owner's numbers in increasing order, BATCH texts a
statement: first those whose current revision is the one at their bound,
from the index texts_order, then the others at their bound; it reads no
title, tag or text, and no read transaction stays open between two
statements. A text an erasure removed whole is left out; an erasure that
cuts a text later leaves its entry where it was, and a page shows its
length now.

A page (page) reads the entries after a position, the place a page gives
as next, each text at its revision as of its bound with its kind, title,
length and text cut from its parts, at most limit texts and characters
characters together (a first text longer than that cut), and the status
the text has now, so a text refused since shows as such. Positions count
from the longest text; a page in ascending order reads them backwards.

    build(depot, snapshot_id, progress)   the order, as Sizes
    save(sizes, path), load(path)         kept in a file (.npy, read through a memory map)
    page(depot, sizes, after, ...)        a page of the strip
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ids, snapshots
from .errors import Refused
from .parts import Parted

BATCH = 20_000  # texts a statement of build reads
ENTRY = np.dtype([("length", "<i8"), ("owner", "<u2"), ("number", "<u4")])
# each text asked ([place (ids.order_sql), bound] pairs as JSON) whose current revision is the one at its bound: its
# place among those asked and its length, from the index texts_order
CURRENT_LENGTHS = ("WITH a AS MATERIALIZED (SELECT key, value ->> 0 AS place, value ->> 1 AS bound FROM json_each(?)) "
                   "SELECT a.key, t.content_length FROM a CROSS JOIN texts_revisions t INDEXED BY texts_order "
                   f"ON {ids.order_sql('t.id')} = a.place AND t.latest AND t.change <= a.bound "
                   "WHERE t.content_kind IS NOT NULL")
# the same for any text asked ([id, bound] pairs as JSON), at its revision as of its bound
LENGTHS_AT_BOUND = ("WITH a AS MATERIALIZED (SELECT key, value ->> 0 AS text, value ->> 1 AS bound FROM json_each(?)) "
                    "SELECT a.key, t.content_length FROM a CROSS JOIN texts_revisions t "
                    "ON t.id = a.text AND t.revision = (SELECT max(r.revision) FROM texts_revisions r "
                    f"INDEXED BY {snapshots.TEXTS_KEY} WHERE r.id = a.text AND r.change <= a.bound) "
                    "WHERE t.content_kind IS NOT NULL")


@dataclass(frozen=True)
class Sizes:
    """A snapshot's order by length: snapshot, its id; digest, its digest;
    owners, its owners in the order of their UTF-8 bytes; entries, the
    array of ENTRY, the longest first."""
    snapshot: str
    digest: str
    owners: list[str]
    entries: np.ndarray

    def __len__(self) -> int:
        return len(self.entries)

    def text(self, at: int) -> str:
        entry = self.entries[at]
        return ids.make(self.owners[int(entry["owner"])], int(entry["number"]))


def build(depot, snapshot_id: str, progress: Callable[[int], None] | None = None) -> Sizes:
    """The snapshot's order by length, as the module says. progress is
    called with how many texts were read so far."""
    read = snapshots._read(depot, snapshot_id)
    digest = depot.db.reader().execute("SELECT digest FROM snapshots WHERE id = ?", (snapshot_id,)).fetchall()[0][0]
    owners = sorted(read.by_owner, key=str.encode)
    found, done = [], 0
    for rank, owner in enumerate(owners):
        numbers = np.frombuffer(read.union(owner).to_array(), dtype=np.uint32)
        for start in range(0, len(numbers), BATCH):
            batch = numbers[start:start + BATCH]
            bounds = [read.bound(owner, int(number))[1] for number in batch]
            head = f"{owner}\x01"
            con = depot.db.reader()
            lengths = np.full(len(batch), -1, dtype=np.int64)
            for at, length in con.execute(CURRENT_LENGTHS, (_json(
                    [[f"{head}0{int(number):020d}\x01{ids.make(owner, int(number))}", bound]
                     for number, bound in zip(batch, bounds)]),)):
                lengths[at] = length
            rest = np.flatnonzero(lengths < 0)
            if len(rest):
                for at, length in con.execute(LENGTHS_AT_BOUND, (_json(
                        [[ids.make(owner, int(batch[at])), bounds[at]] for at in rest]),)):
                    lengths[rest[at]] = length
            held = lengths >= 0  # a text with no row left was removed whole by an erasure
            part = np.empty(int(held.sum()), dtype=ENTRY)
            part["length"], part["owner"], part["number"] = lengths[held], rank, batch[held]
            found.append(part)
            done += len(batch)
            if progress is not None:
                progress(done)
    entries = np.concatenate(found) if found else np.empty(0, dtype=ENTRY)
    order = np.lexsort((entries["number"], entries["owner"], -entries["length"]))
    return Sizes(snapshot_id, digest, owners, entries[order])


def save(sizes: Sizes, path: Path):
    """The order written at path (its entries, .npy) and beside it (path with .json: snapshot, digest, owners),
    each through a temporary file and a rename, the entries last, so a file found is whole."""
    head = path.with_suffix(".json")
    for target, write in ((head, lambda f: f.write(json.dumps(
            {"snapshot": sizes.snapshot, "digest": sizes.digest, "owners": sizes.owners}).encode())),
                          (path, lambda f: np.save(f, sizes.entries, allow_pickle=False))):
        partial = target.with_name(target.name + ".partial")
        with open(partial, "wb") as file:
            write(file)
        os.replace(partial, target)


def load(path: Path) -> Sizes | None:
    """The order kept at path, its entries read through a memory map; None when there is none whole."""
    try:
        head = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        entries = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, KeyError):
        return None
    if entries.dtype != ENTRY:
        return None
    return Sizes(head["snapshot"], head["digest"], list(head["owners"]), entries)


def page(depot, sizes: Sizes, after: int | None = None, limit: int = 50, characters: int | None = 200_000,
         descending: bool = True, each: int | None = None) -> dict:
    """{rows: the texts after the position after (from the first without
    it), at most limit holding at most characters characters together, each
    {id, kind, title, length, bytes, status: its status now, revision, passage:
    its text or an item's content from its start, whole, or cut at
    characters for a first text longer than that; with characters None,
    no passage and no bound but limit; each, the most characters a text
    shows, its start}, next: the position to
    give as after for the next page, None after the last; total: the
    snapshot's texts in the order}. Positions count from the longest text,
    descending; from the shortest otherwise."""
    if limit < 1 or (characters is not None and characters < 1):
        raise Refused("a page holds at least one text and one character")
    total = len(sizes)
    start = 0 if after is None else int(after) + 1
    read = snapshots._read(depot, sizes.snapshot)
    con = depot.db.reader()
    rows, left, at = [], float("inf") if characters is None else characters, start
    while at < total and len(rows) < limit:
        positions = list(range(at, min(total, at + max(limit - len(rows), 1) * 2)))
        asked = []
        for position in positions:
            text = sizes.text(position if descending else total - 1 - position)
            asked.append((position, text, *read.bound(ids.owner(text), ids.number(text))))
        found = _rows(con, asked)
        stop = False
        for position, text, _, _ in asked:
            at = position + 1
            row = found.get(text)
            if row is None:
                continue  # removed whole by an erasure since
            wanted = row["length"] if each is None else min(row["length"], each)
            if rows and wanted > left:
                at = position
                stop = True
                break
            shown = min(wanted, left)
            left -= shown
            rows.append({**row, "shown": shown, "position": position})
            if len(rows) >= limit or left <= 0:
                stop = True
                break
        if stop:
            break
    reader = depot.reader()
    for row in rows:
        parts, shown = row.pop("parts"), row.pop("shown")
        if characters is not None:
            row["passage"] = Parted(reader, row["id"], json.loads(parts))[:shown]
    for row in rows:
        row.pop("position")
    return {"rows": rows, "next": at - 1 if at < total else None, "total": total}


def _rows(con, asked: list[tuple[int, str, int, int]]) -> dict[str, dict]:
    """{id: the row of each text asked ((position, id, part, bound)) at its revision as of its bound, with the
    parts of the content version it reads and its status now}."""
    found = {}
    for one in con.execute(
            "WITH a AS MATERIALIZED (SELECT value ->> 0 AS text, value ->> 1 AS bound FROM json_each(?)) "
            "SELECT t.id, t.revision, c.kind, t.title, c.length, c.bytes, c.parts, "
            "(SELECT n.status FROM texts_revisions n INDEXED BY texts_latest WHERE n.id = t.id AND n.latest) "
            "FROM a CROSS JOIN texts_revisions t ON t.id = a.text AND t.revision = (SELECT max(r.revision) FROM "
            f"texts_revisions r INDEXED BY {snapshots.TEXTS_KEY} WHERE r.id = a.text AND r.change <= a.bound) "
            "JOIN content_lists c ON c.id = t.id AND c.version = t.content_version",
            (_json([[text, bound] for _, text, _, bound in asked]),)):
        text, revision, kind, title, length, size, parts, status = one
        found[text] = {"id": text, "revision": revision, "kind": kind, "title": title, "length": length,
                       "bytes": size, "status": status, "parts": parts}
    return found


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
