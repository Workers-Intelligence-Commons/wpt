"""Where a corpus build cuts a long plain text into pieces, and the text of
one piece read from the text's parts.

A plain text longer than LENGTH characters (schema.LONG, the texts the
index content_lists_long holds) is cut into count(length) pieces, numbered
from 0, most about LENGTH characters long (one can be a few characters or
nearly twice that, where the cuts fall): piece n runs from the cut of n
to the cut of n + 1, the cut of 0 being the text's start and the cut after
the last piece its end. The cut of n, from 1 on, is the first line start
from n × LENGTH on, before (n + 1) × LENGTH and the text's end, whose line
begins with MARK (the line that begins each file of a repository the depot
holds as one text); without one, the first line start there; without one,
n × LENGTH itself. So a cut falls between two files whenever one is in
reach, every piece holds at least one character, and how many pieces a
text has follows from its length alone, known before it is read.

A piece is read from the characters from n × LENGTH − 1 to (n + 2) ×
LENGTH alone, fetching only the parts they lie in (depot.parts.Parted),
so a text of any size is never read whole. Its name, `<id>#<n>`, is what
its export key is made from (depot.order) and the document its row of
a token file names; no id holds '#' (depot.ids).

    count(length)                  how many pieces a plain text of that length is cut into
    name(text, n)                  the name of the piece n of the text
    read(content, n)               the text of piece n of a content version (Parted)
    holding(content, start, end)   the pieces that hold some of a span of its characters
"""

from .parts import Parted
from .schema import LONG

LENGTH = LONG  # characters: a plain text longer than this is cut into pieces of about this length
MARK = "=== "  # what begins the line of each file of a repository held as one text


def count(length: int) -> int:
    """How many pieces a plain text of length characters is cut into: 1, the whole text, up to LENGTH."""
    return -(-length // LENGTH) if length > LENGTH else 1


def name(text: str, n: int) -> str:
    """The name of the piece n of the text whose id is text."""
    return f"{text}#{n}"


def read(content: Parted, n: int) -> str:
    """The text of the piece n of a plain content version, read as the module says."""
    length = len(content)
    low, high = max(n * LENGTH - 1, 0), min((n + 2) * LENGTH, length)
    held = content[low:high]
    start = _cut(held, low, n, length)
    end = length if n + 1 == count(length) else _cut(held, low, n + 1, length)
    return held[start - low:end - low]


def holding(content: Parted, start: int, end: int) -> list[int]:
    """The numbers of the pieces of a long plain content version that hold
    at least one of its characters from start to end (excluded), in order,
    each cut read from the characters around it alone. The cut of n lies
    from n × LENGTH on and before (n + 1) × LENGTH, so a character at place
    p lies in piece p // LENGTH or the one before it."""
    length = len(content)
    last = count(length) - 1

    def cut(n: int) -> int:
        if n == 0:
            return 0
        if n > last:
            return length
        low = n * LENGTH - 1
        return _cut(content[low:min((n + 1) * LENGTH, length)], low, n, length)

    first = max(0, start // LENGTH - 1)
    return [n for n in range(first, min(last, max(start, end - 1) // LENGTH) + 1)
            if cut(n) < end and cut(n + 1) > start]


def _cut(held: str, low: int, n: int, length: int) -> int:
    """The cut of n in a text of length characters, from held, its characters from low on."""
    if n == 0:
        return 0
    start = n * LENGTH
    window = held[start - 1 - low:min(start + LENGTH, length) - low]  # the character before start, then the reach
    found = window.find("\n" + MARK)
    if found < 0:
        found = window.find("\n", 0, len(window) - 1)
    return start if found < 0 else start + found
