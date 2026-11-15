"""The texts of a dataset in its corpus's seeded order, kept in an SQLite
file of their own (the order file), so a dataset of any size is read in that
order a text at a time, from any position, with little memory.

The order takes every text of the dataset's libraries, each library as it
stood at the change number of its depot the dataset names
(Depot.library_members: its members then, each at its revision then), a
text two libraries hold once, at its revision in the first of those
libraries the dataset names. Each text's place is its key, the first 8
bytes of the SHA-256 of the seed, its depot's name in the order and its id
(the depot's name: '' for the depot that holds the dataset, another
depot's own name, the one in its settings, so the order does not depend on
where the depot's file is, and a clone orders its upstream's texts as the
upstream does), with the
seed a fingerprint of the dataset's libraries (each with its change
number) and shares alone, so the same libraries and shares always give the
same order, whatever the dataset's name, tokenizer recipe and limit, and a
limit only cuts it short: every size of a model series reads the same corpus, the
smaller sizes its first part. Texts of equal keys go by their depot's
place in the dataset and their id.

The file is made in two passes: every member of every library goes into
the table members as it is read; then one statement sorts and dedupes them
into the table ordered, whose rowid is a text's position (1, 2, 3...), and
the table members goes. SQLite sorts in temporary files in the folder that
holds the order file, never in memory or the system's temporary folder.
An order file is whole once its meta row 'done' is written; a file that is
not whole is made again from the start.

Each text of the order has its position, its source (the index of its depot
in sources()), id, revision, kind (null: a plain text), pool (with shares:
the index of the first share whose tag a plain text has, null when it has
none; without shares, 0 for every plain text; null for an item), bytes
(its content's UTF-8 bytes) and taken: whether the corpus takes it at
all, that is, a plain text with a pool or a conversation or a task (an item
of another kind is no training data).

With max_tokens, a build first counts the items' tokens and keeps in the
table limited the positions of the items the limit leaves out, with what
that limit was counted for (its vocabulary and max_tokens, wpt_depot.corpus);
the texts the corpus takes then leave those out. The order itself depends on
neither, so builds of one dataset with other vocabularies or sizes share it.
"""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import apsw

TRAINING_KINDS = ("conversation", "task")
BATCH = 50_000  # members written to the order file in one statement
CACHE = 256 << 20  # bytes of pages the order file's connection keeps in memory


_NOT_LIMITED = "rowid NOT IN (SELECT position FROM limited)"  # an ordered row the limit does not leave out


class Entry(NamedTuple):
    position: int
    source: int
    id: str
    revision: int
    kind: str | None
    pool: int | None
    bytes: int


def seed_of(dataset: dict) -> str:
    """The seed of a dataset's order: a fingerprint of its row's libraries and shares."""
    kept = {"libraries": dataset["libraries"], "shares": dataset["shares"]}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def key_of(seed: str, depot: str, text: str) -> int:
    """A text's place in the order: the first 8 bytes of SHA-256(seed, depot's name, id), as a signed integer."""
    return int.from_bytes(hashlib.sha256(f"{seed}\0{depot}\0{text}".encode()).digest()[:8], "big", signed=True)


class Order:
    """The order file at path."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.con = apsw.Connection(str(self.path))
        self.con.execute(f"PRAGMA cache_size = -{CACHE // 1024}")
        self.con.execute("PRAGMA journal_mode = OFF")
        self.con.execute("PRAGMA synchronous = OFF")
        # sorting goes into files beside the order file (a process-wide setting of SQLite)
        self.con.execute(f"PRAGMA temp_store_directory = '{str(self.path.parent).replace(chr(39), chr(39) * 2)}'")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.con.execute("CREATE TABLE IF NOT EXISTS limited (position INTEGER PRIMARY KEY)")

    def close(self):
        self.con.close()

    def _meta(self, name: str):
        found = list(self.con.execute("SELECT value FROM meta WHERE name = ?", (name,)))
        return json.loads(found[0][0]) if found else None

    def _set(self, name: str, value):
        self.con.execute("INSERT OR REPLACE INTO meta (name, value) VALUES (?, ?)", (name, json.dumps(value)))

    def done(self) -> bool:
        return bool(self._meta("done"))

    @classmethod
    def made(cls, path: Path, members, sources: list[str], seed: str, shares: list[dict] | None,
             identity: str = "", progress=lambda count: None, names: list[str] | None = None) -> "Order":
        """The order file at path, made from members unless it is whole
        already and made for the same identity (what decides it, as its
        caller names it): members, for each library in the dataset's order, gives
        (source index, rows of (id, revision, kind, tags as JSON text,
        bytes)), each source index one of sources (the names a corpus
        records for the depots). names: each source's name in the keys (the
        module says; default: sources). progress gets how many members were
        read, now and then."""
        order = cls(path)
        if order.done() and order._meta("identity") == identity:
            return order
        con = order.con
        for table in ("members", "ordered", "named", "limited"):
            con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute("DELETE FROM meta")
        con.execute("CREATE TABLE limited (position INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE members (key INTEGER NOT NULL, source INTEGER NOT NULL, id TEXT NOT NULL, "
                    "library INTEGER NOT NULL, revision INTEGER NOT NULL, kind TEXT, pool INTEGER, "
                    "bytes INTEGER NOT NULL, taken INTEGER NOT NULL)")
        tags = None if shares is None else [one["tag"] for one in shares]
        read = 0
        names = sources if names is None else names
        for library, (source, rows) in enumerate(members):
            address = names[source]
            batch = []
            for text, revision, kind, held_tags, size in rows:
                if kind is None:
                    pool = 0 if tags is None else next((at for at, tag in enumerate(tags)
                                                        if tag in json.loads(held_tags)), None)
                    taken = pool is not None
                else:
                    pool, taken = None, kind in TRAINING_KINDS
                batch.append((key_of(seed, address, text), source, text, library, revision, kind, pool, size,
                              int(taken)))
                if len(batch) >= BATCH:
                    order._insert(batch)
                    read += len(batch)
                    batch = []
                    progress(read)
            order._insert(batch)
            read += len(batch)
            progress(read)
        with con:
            # one row per text: the one of the first library the dataset names (SQLite gives the bare columns of
            # the row whose library is the least)
            con.execute("CREATE TABLE ordered AS SELECT key, source, id, min(library) AS library, revision, kind, "
                        "pool, bytes, taken FROM members GROUP BY key, source, id ORDER BY key, source, id")
            con.execute("DROP TABLE members")
            order._set("sources", sources)
            order._set("identity", identity)
            order._set("done", True)
        return order

    def _insert(self, batch: list[tuple]):
        if batch:
            with self.con:
                self.con.executemany("INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)

    def sources(self) -> list[str]:
        """The names a corpus records for the depots, by the index a text's source names."""
        return self._meta("sources")

    def count(self) -> int:
        """How many texts the order holds."""
        return list(self.con.execute("SELECT coalesce(max(rowid), 0) FROM ordered"))[0][0]

    def entries(self, start: int = 1, taken: bool = True) -> Iterator[Entry]:
        """The texts from position start on, in order; only those the corpus
        takes (an item the limit leaves out is not), unless taken is False."""
        where = f" AND taken AND {_NOT_LIMITED}" if taken else ""
        cursor = self.con.cursor().execute(
            f"SELECT rowid, source, id, revision, kind, pool, bytes FROM ordered WHERE rowid >= ?{where} "
            "ORDER BY rowid", (start,))
        for row in cursor:
            yield Entry(*row)

    def last_item(self) -> int:
        """The position of the last conversation or task the corpus takes, 0 without one."""
        return list(self.con.execute("SELECT coalesce(max(rowid), 0) FROM ordered WHERE taken AND kind IS NOT NULL "
                                     f"AND {_NOT_LIMITED}"))[0][0]

    def items(self) -> Iterator[Entry]:
        """The conversations and tasks the corpus may take, in order, the limit aside."""
        cursor = self.con.cursor().execute("SELECT rowid, source, id, revision, kind, pool, bytes FROM ordered "
                                           "WHERE taken AND kind IS NOT NULL ORDER BY rowid")
        for row in cursor:
            yield Entry(*row)

    def limit_items(self, left_out: list[int], tokens: int | None, limit):
        """Keep the positions of the items the limit leaves out, the tokens
        of the items it takes (items_tokens) and what the limit was counted
        for (limit: any JSON value its caller names it by)."""
        with self.con:
            self.con.execute("DELETE FROM limited")
            self.con.executemany("INSERT INTO limited (position) VALUES (?)", ((one,) for one in left_out))
            self._set("items_tokens", tokens)
            self._set("limit", limit)

    def items_tokens(self, limit) -> int | None:
        """The tokens of the items the limit takes, when limit_items last
        kept that limit; None otherwise."""
        if self._meta("limit") != limit:
            return None
        return self._meta("items_tokens")

    def limited(self) -> int:
        """How many items the limit leaves out."""
        return list(self.con.execute("SELECT count(*) FROM limited"))[0][0]

    def pool_bytes(self) -> dict[int, int]:
        """{pool: the bytes of the plain texts of that pool the corpus may take}."""
        return dict(self.con.execute("SELECT pool, sum(bytes) FROM ordered WHERE taken AND kind IS NULL GROUP BY pool"))

    def left_out(self) -> dict:
        """What the corpus leaves out whatever the limits: {"no share": the plain
        texts no share takes, "kinds": {kind: the items of a kind that is no
        training data}}."""
        no_share = list(self.con.execute("SELECT count(*) FROM ordered WHERE kind IS NULL AND NOT taken"))[0][0]
        kinds = dict(self.con.execute("SELECT kind, count(*) FROM ordered WHERE kind IS NOT NULL AND NOT taken "
                                      "GROUP BY kind ORDER BY kind"))
        return {"no share": no_share, "kinds": kinds}

    def taken_bytes(self) -> int:
        """The bytes of every text the corpus may take, plain texts and items."""
        return list(self.con.execute("SELECT coalesce(sum(bytes), 0) FROM ordered WHERE taken"))[0][0]

    def items_bytes(self) -> int:
        """The bytes of the conversations and tasks the corpus takes, those the limit leaves out aside."""
        return list(self.con.execute("SELECT coalesce(sum(bytes), 0) FROM ordered WHERE taken AND kind IS NOT NULL "
                                     f"AND {_NOT_LIMITED}"))[0][0]

    def plain(self) -> int:
        """How many plain texts the corpus may take."""
        return list(self.con.execute("SELECT count(*) FROM ordered WHERE taken AND kind IS NULL"))[0][0]
