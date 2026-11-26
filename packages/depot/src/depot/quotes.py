"""The excerpts of a conversation: passages of the depot's plain texts it
quotes by their content instead of copying them (docs/chat-format.md,
"Excerpts").

An excerpt block names the quoted text by its id (`source`) and picks the
passage with a W3C Web Annotation TextQuoteSelector (`selector`): `exact`,
the passage itself, and optional `prefix` and `suffix`, the characters
right before and right after it, which tell apart two places that hold the
same passage. A place of a text matches the selector when the text holds
prefix, exact and suffix there one after the other; the passage is then
exact, at the place where it starts (find). Two matches may overlap, and
each counts. An excerpt counts only when its selector matches exactly one
place, so the quote finds its passage whatever moved it, and a passage
that is no longer there, or that is there twice, is no quote.

A text is read as its parts (depot.parts.Parted), so a quote is found
in a text of any size holding one part and the selector's length at a time.

    excerpts(conversation)       every excerpt block of a conversation, in order
    quoted(conversation)         the ids of the texts it quotes
    find(document, selector)     where the selector matches in a text
    find_each(document, selectors)  the same for several selectors, reading the text once
"""

from collections.abc import Iterable, Iterator


def excerpts(conversation: dict) -> Iterator[dict]:
    """Every excerpt block of the conversation, in order: in the content of
    its messages and in the content of their tool results."""
    for message in conversation.get("messages", []):
        for block in message.get("content", []):
            if block.get("type") == "excerpt":
                yield block
            elif block.get("type") == "tool_result":
                yield from (inner for inner in block.get("content", []) if inner.get("type") == "excerpt")


def quoted(conversation: dict) -> list[str]:
    """The ids of the texts the conversation's excerpts quote, each once, in order."""
    return list(dict.fromkeys(block["source"] for block in excerpts(conversation)))


def find(document: str | Iterable[str], selector: dict, most: int = 2) -> list[int]:
    """The places, in characters, where the passage starts of each place of
    document (a str, or the str blocks it is made of in order, such as a
    Parted gives them) that matches the selector, in order, at most most of
    them: the reading stops at the most-th. A selector whose exact is empty
    matches nothing."""
    prefix, exact, suffix = selector.get("prefix", ""), selector["exact"], selector.get("suffix", "")
    if not exact:
        return []
    pattern = prefix + exact + suffix
    found, held, offset = [], "", 0  # offset: the place of held's first character
    for block in [document] if isinstance(document, str) else document:
        held += block
        at = held.find(pattern)
        while at != -1:
            found.append(offset + at + len(prefix))
            if len(found) >= most:
                return found
            at = held.find(pattern, at + 1)
        kept = max(0, len(held) - len(pattern) + 1)  # a match not found yet starts at kept or later
        held, offset = held[kept:], offset + kept
    return found


def find_each(document: str | Iterable[str], selectors: list[dict], most: int = 2) -> list[list[int]]:
    """For each selector, in order, what find(document, selector, most)
    gives, reading document once for all of them, so the excerpts that
    quote the same text read it once."""
    skips, patterns = [], []
    for selector in selectors:
        prefix, exact = selector.get("prefix", ""), selector["exact"]
        skips.append(len(prefix))
        patterns.append(prefix + exact + selector.get("suffix", "") if exact else None)
    found = [[] for _ in selectors]
    after = [0] * len(selectors)  # each selector's place, in characters, where its next match may start
    reading = [at for at, pattern in enumerate(patterns) if pattern is not None]
    held, offset = "", 0  # offset: the place of held's first character
    for block in [document] if isinstance(document, str) else document:
        if not reading:
            break
        held += block
        for at in reading:
            pattern = patterns[at]
            place = held.find(pattern, after[at] - offset)
            while place != -1 and len(found[at]) < most:
                found[at].append(offset + place + skips[at])
                after[at] = offset + place + 1
                place = held.find(pattern, place + 1)
            after[at] = max(after[at], offset + len(held) - len(pattern) + 1)  # a match not found yet starts there
        reading = [at for at in reading if len(found[at]) < most]
        kept = max(0, len(held) - max((len(patterns[at]) for at in reading), default=1) + 1)
        held, offset = held[kept:], offset + kept
    return found
