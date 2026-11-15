"""What a depot's web app knows of the downloads of corpora, kept in the
queue's file beside the depot (wpt.depot.jobs): when each corpus was last
downloaded, the streams being sent now, the size and sha256 of each part
of a corpus's texts once it was sent whole, and the erasures whose corpora
are being marked stale.

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
"""

import secrets
import time
from contextlib import contextmanager
from pathlib import Path

from wpt_depot import errors
from wpt_depot.exports import MAKER

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

    def export(self, corpus: str, part: int, form: str) -> dict | None:
        """{bytes, sha256} of a part of the corpus's texts in that form, once
        it was sent whole by the same versions of the libraries that make it
        (wpt_depot.exports.MAKER); None before."""
        with connected(self.path) as con:
            found = con.execute("SELECT bytes, sha256 FROM exports WHERE corpus = ? AND part = ? AND format = ? AND "
                                "maker = ?", (corpus, part, form, MAKER)).fetchone()
        return None if found is None else {"bytes": found["bytes"], "sha256": found["sha256"]}

    def exported(self, corpus: str, part: int, form: str, size: int, sha256: str):
        """Keep the size and sha256 of a part of the corpus's texts in that form, sent whole by these versions."""
        with connected(self.path, write=True) as con:
            con.execute("INSERT OR REPLACE INTO exports (corpus, part, format, maker, bytes, sha256) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (corpus, part, form, MAKER, size, sha256))

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
