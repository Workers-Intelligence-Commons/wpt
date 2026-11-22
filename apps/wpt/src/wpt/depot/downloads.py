"""What a depot's web app knows of its downloads, kept in the queue's file
beside the depot (wpt.depot.jobs): when each corpus was last downloaded,
the streams being sent now, how many texts each export of a dataset counts
and the size, sha256 and conversations left out of each part of an export
once it was sent whole (wpt_depot.legacy), and the erasures whose corpora are being marked stale.

An erasure decided through the web app is written into marking before it
is carried out, and leaves it once every corpus holding one of its texts
is marked stale (wpt.depot.shared.settle_marks, in the background, since
it reads the texts table of every corpus the store keeps): while marking
holds an erasure, the web app sends no corpus, so a corpus an erasure
reached is never sent, even when the marking failed or the web app
stopped before it ended.

A build's cap (wpt_depot.corpora.Room) reads the first two: it deletes the
corpus downloaded least recently first, never one being downloaded. The
web app reads the streams to bound the text exports: at most EXPORTS at
once in all, and VISITOR_EXPORTS for one visitor (TooMany otherwise).
An export's count of texts and a part's size and conversations left out
are what reading them again would give, kept only to save that work.
"""

import json
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

from wpt_depot import errors
from wpt_depot.legacy import MAKER

from .jobs import connected

EXPORTS = 2  # text exports (and samples) sent at once, in all
VISITOR_EXPORTS = 1  # text exports (and samples) sent at once to one visitor
LIMITED = ("text", "sample")  # the kinds of streams the bounds count


class TooMany(errors.DepotError):
    """As many text exports run as the web app sends at once, in all or to
    this visitor: nothing was sent; the same request can come again once one ended."""


class Downloads:
    """The downloads of the queue's file at path."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def streaming(self, corpus: str | None, visitor: str, kind: str):
        """Keep a stream of that kind (a token file: "file"; LIMITED's kinds
        are bounded) to visitor, of the corpus, as being sent while the
        block runs, and the corpus as downloaded now when it ends. TooMany
        past a bound."""
        stream = secrets.token_hex(8)
        with connected(self.path, write=True) as con:
            if kind in LIMITED:
                running = con.execute("SELECT count(*), coalesce(sum(visitor = ?), 0) FROM streams WHERE kind IN "
                                      "('text', 'sample')", (visitor,)).fetchone()
                if running[0] >= EXPORTS:
                    raise TooMany(f"{EXPORTS} text exports are being sent already: ask again once one ended")
                if running[1] >= VISITOR_EXPORTS:
                    raise TooMany(f"a text export is being sent to you already ({VISITOR_EXPORTS} at a time): ask "
                                  "again once it ended")
            con.execute("INSERT INTO streams (id, corpus, visitor, kind, started) VALUES (?, ?, ?, ?, ?)",
                        (stream, corpus, visitor, kind, time.time()))
        try:
            yield
        finally:
            with connected(self.path, write=True) as con:
                con.execute("DELETE FROM streams WHERE id = ?", (stream,))
                if corpus is not None:
                    con.execute("INSERT OR REPLACE INTO downloads (corpus, last) VALUES (?, ?)", (corpus, time.time()))

    def last_download(self, corpus: str) -> float:
        """When the corpus was last downloaded, in seconds since the epoch; 0 for never."""
        with connected(self.path) as con:
            found = con.execute("SELECT last FROM downloads WHERE corpus = ?", (corpus,)).fetchone()
        return 0.0 if found is None else found["last"]

    def busy(self) -> set[str]:
        """The ids of the corpora being downloaded now."""
        with connected(self.path) as con:
            return {row["corpus"] for row in con.execute("SELECT DISTINCT corpus FROM streams WHERE corpus IS NOT NULL")}

    def export_texts(self, export: str) -> int | None:
        """How many texts the export (its identity, wpt_depot.legacy.Export) counts, once kept; None before."""
        with connected(self.path) as con:
            found = con.execute("SELECT texts FROM export_texts WHERE export = ?", (export,)).fetchone()
        return None if found is None else found["texts"]

    def counted(self, export: str, texts: int):
        """Keep how many texts the export counts."""
        with connected(self.path, write=True) as con:
            con.execute("INSERT OR REPLACE INTO export_texts (export, texts) VALUES (?, ?)", (export, texts))

    def export_sizes(self, export: str, parts: int) -> dict[tuple[int, str], dict]:
        """{(part, form): {bytes, sha256, left_out}} of each part of the
        parts of the export sent whole in that form by the same versions of
        the libraries that make it (wpt_depot.legacy.MAKER); left_out: how
        many conversations the part left out, by why (Export.part)."""
        with connected(self.path) as con:
            found = con.execute("SELECT part, format, bytes, sha256, left_out FROM exported_parts WHERE export = ? "
                                "AND parts = ? AND maker = ?", (export, parts, MAKER)).fetchall()
        return {(row["part"], row["format"]): {"bytes": row["bytes"], "sha256": row["sha256"],
                                               "left_out": json.loads(row["left_out"])} for row in found}

    def exported(self, export: str, parts: int, part: int, form: str, size: int, sha256: str,
                 left_out: dict[str, int]):
        """Keep the size, sha256 and conversations left out (by why) of a part
        of the export in that form, sent whole by these versions."""
        with connected(self.path, write=True) as con:
            con.execute("INSERT OR REPLACE INTO exported_parts (export, parts, part, format, maker, bytes, sha256, "
                        "left_out) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (export, parts, part, form, MAKER, size, sha256, json.dumps(left_out, sort_keys=True)))

    def marking(self, erasures: list[str]):
        """Keep the erasures as being marked: no corpus is sent until they leave (marked)."""
        with connected(self.path, write=True) as con:
            con.executemany("INSERT OR IGNORE INTO marking (erasure, since) VALUES (?, ?)",
                            [(one, time.time()) for one in erasures])

    def marked(self, erasures: list[str]):
        """The erasures leave marking: every corpus holding one of their texts is marked, or they were not carried out."""
        with connected(self.path, write=True) as con:
            con.executemany("DELETE FROM marking WHERE erasure = ?", [(one,) for one in erasures])

    def pending(self) -> list[str]:
        """The erasures being marked."""
        with connected(self.path) as con:
            return [row["erasure"] for row in con.execute("SELECT erasure FROM marking ORDER BY erasure")]
