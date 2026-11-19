"""Two measures of how much of a plain text is writing, made from the text
alone (docs/data-selection.md, "Measures"): never from its tags, its kind
or its language, so they read every text alike and a person chooses, from
their values, which texts to review or refuse.

- `symbols` (SYMBOLS): the share of the text's characters that are neither
  a letter, a mark, a digit nor white space, in any script (Unicode
  categories L, M, N and white space); an underscore is a symbol.
- `running-text` (RUNNING): the share of the text's characters that stand
  in ordinary lines. A line is ordinary unless it is a table row (its first
  character other than a space is `|`) or a heading (`#`); in an ordinary
  line, the image placeholders (`[image]`, `[image: words]`) and the
  targets of links (`](<address>)`, `](address)`) count for nothing, and
  a line left without a letter counts for nothing either.

An empty text has 0 for both. Each value is rounded to DIGITS decimals.

A depot's texts are measured in worker processes, BATCH texts at a time,
read in the order of their ids; an item is no plain text and is left out.

    Measure()                        fed a text part after part (feed), its values at the end (values)
    measured(text)                   {SYMBOLS: v, RUNNING: v} of a whole text
    score_lines(depot, where, ...)   the scores-file lines of the texts where selects, with what was left out
"""

import json
import re
import unicodedata
from collections.abc import Callable, Iterator

from wpt_utils.parallel import Sizing, in_order, process_pool

from .parts import scan

SYMBOLS, RUNNING = "symbols", "running-text"
DIGITS = 4  # decimals of a value
BATCH = 1000  # texts read and measured at a time
IN_PROCESS = 20_000  # texts a walk measures in its own process before it starts worker processes
_NOT_WRITING = re.compile(r"[^\w\s]|_")  # every symbol is one of these; a mark is too, and is told apart
_PLACED = re.compile(r"\[image(?::[^\]\n]*)?\]|\]\(<[^>\n]*>\)|\]\([^)\s]*\)")  # placeholders and link targets
_LETTER = re.compile(r"[^\W\d_]")


class Measure:
    """The measures of one text fed part after part: a part may end inside
    a line, which waits for the next part or for values."""

    def __init__(self):
        self.characters = self.symbols = self.running = 0
        self.line = []  # the pieces of the line a part ended inside, joined once its end comes

    def feed(self, part: str):
        self.characters += len(part)
        if part.isascii():  # no mark: every match is a symbol
            self.symbols += _NOT_WRITING.subn("", part)[1]
        else:
            self.symbols += sum(1 for found in _NOT_WRITING.findall(part)
                                if found < "\x80" or unicodedata.category(found)[0] != "M")
        lines = part.split("\n")
        if len(lines) == 1:
            self.line.append(part)
            return
        self.running += _running("".join([*self.line, lines[0]]))
        for line in lines[1:-1]:
            self.running += _running(line)
        self.line = [lines[-1]]

    def values(self) -> dict[str, float]:
        if self.line:
            self.running += _running("".join(self.line))
            self.line = []
        if not self.characters:
            return {SYMBOLS: 0.0, RUNNING: 0.0}
        return {SYMBOLS: round(self.symbols / self.characters, DIGITS),
                RUNNING: round(min(self.running, self.characters) / self.characters, DIGITS)}


def _running(line: str) -> int:
    """The characters line counts for in running text (the module says which)."""
    first = line.lstrip()[:1]
    if first in ("|", "#"):
        return 0
    kept = _PLACED.sub("", line)
    return len(kept) if _LETTER.search(kept) else 0


def measured(text: str) -> dict[str, float]:
    one = Measure()
    one.feed(text)
    return one.values()


class Walked:
    """What a walk of score_lines met: measured, the plain texts it gave a
    line; items, the items it left out."""

    def __init__(self):
        self.measured = self.items = 0


def score_lines(depot, where: str, walked: Walked | None = None,
                progress: Callable[[dict], None] = lambda state: None) -> Iterator[dict]:
    """For each text of depot (a Depot, opened to read) the SQL condition
    where (over the texts view) selects, in the order of the ids, its line
    of a scores file ("Scores"): {"id", "scores": [SYMBOLS, RUNNING, each
    made from the text's current content version]}; an item is left out
    and counted in walked. progress gets {"stage": "measuring", "texts":
    measured so far} after each batch."""
    walked = walked if walked is not None else Walked()
    batches = _batches(depot, where, walked)
    first = []
    for batch in batches:  # a small selection is measured here, a large one in worker processes
        first.append(batch)
        if len(first) * BATCH >= IN_PROCESS:
            break
    if len(first) * BATCH < IN_PROCESS:
        _start_worker(depot.address)
        found = (_measured_batch(batch) for batch in first)
    else:
        def every():
            yield from first
            yield from batches
        sizing = Sizing("measuring workers")
        pool = process_pool(sizing, sizing.limit, _start_worker, (depot.address,))
        found = in_order(_measured_batch, every(), pool, sizing)
    try:
        for lines in found:
            yield from lines
            walked.measured += len(lines)
            progress({"stage": "measuring", "texts": walked.measured})
    finally:
        if len(first) * BATCH >= IN_PROCESS:
            pool.shutdown(cancel_futures=True)


def _batches(depot, where: str, walked: Walked) -> Iterator[list[tuple]]:
    """The plain texts where selects, BATCH at a time: (id, content version, its parts)."""
    after = ""
    while True:
        # the next ids on the index of the current texts, then the condition over those alone, each looked up
        # by its id (CROSS JOIN keeps that order), so a batch costs the same wherever it stands and whichever
        # index the condition alone would make SQLite walk
        ids = [row["id"] for row in depot.rows("SELECT id FROM texts_revisions INDEXED BY texts_latest WHERE latest "
                                                "AND id > ? ORDER BY id LIMIT ?", (after, BATCH))]
        if not ids:
            return
        after = ids[-1]
        rows = depot.rows(
            "SELECT t.id AS id, t.content_version AS version, c.kind AS kind, c.parts AS parts FROM "
            f"(WITH wanted (wanted_id) AS MATERIALIZED (SELECT value FROM json_each(?)) SELECT texts.id, "
            f"texts.content_version FROM wanted CROSS JOIN texts ON texts.id = wanted_id WHERE ({where})) t "
            "JOIN content_lists c ON c.id = t.id AND c.version = t.content_version ORDER BY t.id",
            (json.dumps(ids),))
        plain = [(row["id"], row["version"], row["parts"]) for row in rows if row["kind"] is None]
        walked.items += len(rows) - len(plain)
        if plain:
            yield plain


# What one measuring worker holds: the depot it reads, opened once in each process
_HELD = {}


def _start_worker(address: str):
    from .depot import Depot
    _HELD.clear()
    _HELD["depot"] = Depot.view(address)


def _measured_batch(batch: list[tuple]) -> list[dict]:
    reader = _HELD["depot"].reader()
    lines = []
    for text, version, parts in batch:
        one = Measure()
        for _, value in scan(reader, {text: parts}):
            if value is not None:
                one.feed(value)
        lines.append({"id": text, "scores": [{"name": name, "value": value, "content_version": version}
                                             for name, value in one.values().items()]})
    return lines
