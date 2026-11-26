"""Carrying out an erasure on texts, and the two public hashes that name an
erased passage without its content.

A text's content can change, each change a new content version kept next
to the old ones (depot.depot), so a needle names a passage by its place
in one content version of one text, counted in characters (Unicode code
points), the end excluded. In a plain text it has a start and an end. In an item (a kind and its JSON content) it also has a field: a JSON
Pointer (RFC 6901, such as `/messages/0/content/1/text`) to a string of the
content, and the start and the end count characters of that string. A needle
with no start, no end and no field names the whole text or item.

The needles of one request that name the same string (the same text and
content version, and for an item the same field) and overlap or touch are
merged into one passage, from the first start to the last end, so no
character of any of them remains (merged). Carrying out an erasure removes,
from the content version each passage is in, the passage itself, then from
every content version of that text (and of no other text) every place where
it still appears: in a plain text, in the text; in an item, in every string
value of its JSON (not in its keys). The passages are read before any of the
erasure's cuts. The removed text leaves no marker, and every cut version of
an item is checked against its kind's JSON Schema again, before anything is
written.

A conversation whose excerpts quote a plain text the erasure reaches holds
copies of that text's passages, so it is cut too, every version of it
(cut_quotes), a conversation the erasure names included, on top of its own
cut. Only its strings of text are cut (the text of its text and thinking
blocks, the values of its tool calls' arguments and json blocks, its tools'
descriptions, its excerpts' selectors), never a string that names or
structures it (a role, a type, an excerpt's source, an id, a name, a call,
metadata), so it still matches its kind and names what it named: each
passage leaves those strings, and each excerpt, placed where its selector
matches the quoted text before the cut, loses every character that stands
over one the erasure removes there, so a passage that only overlaps it or
spans its prefix, exact and suffix leaves nothing of itself either. An
excerpt of a text removed whole, or one that matches no version of its
text at exactly one place, keeps none of its characters. A clone cuts its
own such conversations from the public hashes of the passages its upstream
erased, before it carries the erasure out (cut_hashed).

What is kept of each passage is where it stood in its string once cut, in
the content version its needle named (its place: how many characters stand
before it), its length in characters and
two hashes
with fixed, public parameters and no secret, so that any depot or clone
can match the passage in a text and refuse it: hash_a, the SHA-256 of its
UTF-8 bytes, and hash_b, a polynomial hash of its characters modulo
MODULUS whose base and whose number for each character are SHA-256 of
DOMAIN and a name. Anyone can compute them for a guess, so they confirm a
guess at a short passage; hash_b can be slid over a later text to find the
passage there.

A plain text's content version may be far bigger than memory: it is read
as its parts (depot.parts.Parted), each cut a pass over its parts that
finds a passage across part boundaries, and what is cut of it is told as the
places of its characters that stay (pieces), so only the parts that lose a
character are written again.

What a request asks to erase can be looked at without sending a whole text:
for each needle, at most CONTEXT characters on each side of its passage, and
the passage (only its two ends when it is long), or once the request is
carried out, the passage's length at the place it left (shown).
"""

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from typing import NamedTuple

from . import quotes
from .errors import Refused
from .kinds import PLAIN

MODULUS = (1 << 61) - 1  # a Mersenne prime
DOMAIN = b"depot erasure hash 1\0"  # the fixed parameter every number of hash_b is made from
CONTEXT = 300  # the characters shown on each side of a passage, and twice as many of a whole text's head


class Needle(NamedTuple):
    text: str  # the id of the text or item
    start: int | None
    end: int | None
    field: str | None  # a JSON Pointer into an item's content
    version: int  # the content version it counts in


def check_needle(row: dict, needle: Needle):
    """Refused unless the needle names a place of the row ({text, kind,
    content}, the content version it names): a plain text with a start and an end, or
    neither; an item with a field that names a string of its content and a
    start and an end within it, or nothing at all."""
    if needle.start is None and needle.end is None:
        if needle.field is not None:
            raise Refused(f"a needle with a field also has a start and an end: {needle.text}")
        return
    if row["kind"] == PLAIN:
        if needle.field is not None:
            raise Refused(f"{needle.text} is a plain text: its needle has no field")
        value = row["text"]
    else:
        if needle.field is None:
            raise Refused(f"{needle.text} is an item: its needle names a field of its content, or the whole item")
        value = resolve(json.loads(row["content"]), needle.field)
    start, end = needle.start, needle.end
    if start is None or end is None or not 0 <= start < end <= len(value):
        raise Refused(f"a needle is a start and an end with 0 <= start < end <= {len(value)}, or neither: "
                      f"({start}, {end})")


def resolve(content, pointer: str) -> str:
    """The string at a JSON Pointer in content. Refused when the pointer is
    not one, leads nowhere or does not lead to a string."""
    if pointer and not pointer.startswith("/"):
        raise Refused(f"{pointer!r} is not a JSON Pointer: it is empty or begins with /")
    value = content
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise Refused(f"the field {pointer} is not in the item")
    if not isinstance(value, str):
        raise Refused(f"the field {pointer} is not a string")
    return value


class Passage(NamedTuple):
    """One passage a request erases: the needles of one string that overlap
    or touch merged into one, or a needle that names a whole text or item."""
    text: str
    start: int | None
    end: int | None
    field: str | None
    needles: tuple[int, ...]  # the places in the request's needles of the needles it covers, in order
    version: int  # the content version it is read from


def merged(needles: list[Needle]) -> list[Passage]:
    """The passages needles name, in the order of the first needle each
    covers: needles of the same text, content version and field (the same
    string) whose places overlap or touch make one passage from the first
    start to the last end; a needle with no start is a passage of its own."""
    found, open_ = [], {}  # open_: the passages of each string, built in the order of their starts
    for index in sorted(range(len(needles)), key=lambda at: (needles[at].start is None, needles[at].start or 0, at)):
        one = needles[index]
        if one.start is None:
            found.append(Passage(one.text, None, None, None, (index,), one.version))
            continue
        runs = open_.setdefault((one.text, one.version, one.field), [])
        if runs and one.start <= runs[-1][1]:
            runs[-1] = [runs[-1][0], max(runs[-1][1], one.end), runs[-1][2] + [index]]
        else:
            runs.append([one.start, one.end, [index]])
    for (text, version, field), runs in open_.items():
        found.extend(Passage(text, start, end, field, tuple(sorted(covered)), version)
                     for start, end, covered in runs)
    return sorted(found, key=lambda one: one.needles[0])


def carried_out(documents: dict[str, dict[int, object]], needles: list[Needle], items=frozenset()
                ) -> tuple[dict[str, dict[int, object]], list[str], list[tuple[Passage, object, int | None]]]:
    """(the cut content versions, {id: {version: what is left of it}} for
    each version the erasure changes, the ids of texts removed whole,
    (passage, its text, its place after the cut) for each passage of
    merged(needles) in order): what an erasure of needles leaves of
    documents, every content version of every id they name ({id: {version:
    its text}}: a plain text as a str or a depot.parts.Parted; for an id
    of items, its content as JSON text). What is left of an item is its cut
    content as JSON text; of a plain text, the places [(start, end)] of its
    characters that stay, in order (pieces). A passage's text is a str, or
    for a whole plain text the text itself, as documents gives it.

    In the content version a needle names, the string it counts in first loses the characters of its named
    passages, so none of them remains; then every passage of a text is
    removed from every content version of that text wherever it still
    appears, one passage after the other in order, each at its leftmost
    places that do not overlap (as str.replace removes them), from every
    string of an item. A passage's place is how many characters of its
    string stand before it once the cut is done, in the version it names;
    None for a whole text, and for a passage of a text removed whole."""
    passages = merged(needles)
    removed = sorted({one.text for one in passages if one.start is None})
    texts = []
    for one in passages:
        document = documents[one.text][one.version]
        if one.start is None:
            texts.append(document)
        else:
            value = resolve(json.loads(document), one.field) if one.text in items else document
            texts.append(value[one.start:one.end])
    cut, places = {}, {}
    for text in dict.fromkeys(one.text for one in passages):
        if text in removed:
            continue
        mine = [(at, one) for at, one in enumerate(passages) if one.text == text]
        removing = [texts[at] for at, _ in mine]
        for version, document in documents[text].items():
            here = [(at, one) for at, one in mine if one.version == version]
            if text in items:
                content = json.loads(document)
                named = {}
                for at, one in here:
                    named.setdefault(_path(content, one.field), []).append((one.start, one.end, at))
                value = json.dumps(_cut_json(content, (), named, removing, places), ensure_ascii=False)
                if value != json.dumps(content, ensure_ascii=False):
                    cut.setdefault(text, {})[version] = value
            else:
                pieces, found = _cut_pieces(_blocks_of(document), len(document),
                                            [(one.start, one.end, at) for at, one in here], removing)
                places.update(found)
                if pieces != [(0, len(document))]:
                    cut.setdefault(text, {})[version] = pieces
    return cut, removed, [(one, texts[at], places.get(at)) for at, one in enumerate(passages)]


def cut_quotes(documents: dict[str, dict[int, str]], versions: dict[str, dict[int, object]],
               erased: dict[str, dict[int, list[tuple[int, int]]]], gone: set[str], removing: list[str]
               ) -> dict[str, dict[int, str]]:
    """What an erasure leaves of the conversations that quote the plain
    texts it reaches (documents: {id: {version: its content as JSON text}},
    every content version of each): {id: {version: its cut content as JSON
    text}} for each version it changes, as _cut_quoting cuts it. versions:
    every content version, before the cut, of each plain text it reaches
    and keeps ({id: {version: a str or a depot.parts.Parted}}); erased:
    the places [(start, end)] of the characters it removes from each
    version it cuts, in that version before the cut; gone: the texts it
    removes whole; removing: the text of every passage it erases."""
    def erased_in(text: str, version: int, start: int, end: int) -> list[tuple[int, int]]:
        return [(a, b) for a, b in erased.get(text, {}).get(version, []) if a < end and b > start]
    return _cut_quoting(documents, versions, erased_in, gone, lambda value: _cut_string(value, [], removing)[0])


def erased_places(pieces: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """The places [(start, end)] of a string of length characters that the
    pieces it keeps (carried_out) leave out, in order."""
    found, at = [], 0
    for a, b in pieces:
        if a > at:
            found.append((at, a))
        at = b
    if at < length:
        found.append((at, length))
    return found


def cut_hashed(documents: dict[str, dict[int, str]], versions: dict[str, dict[int, object]], erased: list[dict],
               gone: set[str]) -> dict[str, dict[int, str]]:
    """What erasures leave of the conversations that quote the texts they
    reached ({id: {version: its content as JSON text}}), as cut_quotes
    gives it, where each passage is known only by an entry of an erasure's
    erased ({length, hash_a in hexadecimal}). In a string of text, every
    window whose length and hash_a are an entry's goes, leftmost first and
    not overlapping, the longest first where several start at one place. In
    the versions of a quoted text, as the depot held them before the
    erasures (versions), every window whose length and hash_a are an
    entry's counts as erased, overlapping ones included, so an excerpt
    loses every character such a window covers. gone: the texts removed
    whole."""
    wanted: dict[int, set[bytes]] = {}
    for one in erased:
        wanted.setdefault(one["length"], set()).add(bytes.fromhex(one["hash_a"]))
    lengths = sorted(wanted, reverse=True)
    longest = lengths[0] if lengths else 0

    def hit(value: str, at: int) -> int | None:
        return next((length for length in lengths if at + length <= len(value) and hashlib.sha256(
            value[at:at + length].encode("utf-8", "surrogatepass")).digest() in wanted[length]), None)

    def without(value: str) -> str:
        kept, start, at = [], 0, 0
        while at < len(value):
            length = hit(value, at)
            if length is None:
                at += 1
                continue
            kept.append(value[start:at])
            at = start = at + length
        return "".join(kept) + value[start:]

    def erased_in(text: str, version: int, start: int, end: int) -> list[tuple[int, int]]:
        low = max(0, start - longest + 1)
        window = versions[text][version][low:end + longest - 1]
        found = []
        for at in range(len(window)):
            for length in lengths:
                if at + length <= len(window) and hashlib.sha256(window[at:at + length].encode(
                        "utf-8", "surrogatepass")).digest() in wanted[length]:
                    found.append((low + at, low + at + length))
        return [(a, b) for a, b in found if a < end and b > start]
    return _cut_quoting(documents, versions, erased_in, gone, without)


def _cut_quoting(documents: dict[str, dict[int, str]], versions: dict[str, dict[int, object]],
                 erased_in: Callable[[str, int, int, int], list[tuple[int, int]]], gone: set[str],
                 cut_string: Callable[[str], str]) -> dict[str, dict[int, str]]:
    """{id: {version: its cut content as JSON text}} of each version of the
    conversations documents gives that the cut changes. Only the strings
    that carry text are cut (_texts_mapped): never a block's type, a role,
    an excerpt's source, a tool call's id or name, a tool result's call, a
    file's sha256 or media type, or any metadata, so a cut version still
    matches its kind and still names what it named. Each such string loses
    what cut_string removes of it. An excerpt of a text removed whole (gone)
    keeps none of its characters: its exact, prefix and suffix become
    empty. An excerpt of a text of versions is first found in each of that
    text's versions (depot.quotes.find): where its selector matches
    exactly one place, its prefix, exact and suffix stand over the
    characters of that place, and each of them loses the characters that
    stand over a character the erasure removes from that version
    (erased_in(text, version, start, end): the erased places that meet
    [start, end)), so an erased passage that only overlaps an excerpt, or
    spans its prefix, exact and suffix, leaves nothing of itself either;
    an excerpt whose selector matches no version at exactly one place
    cannot be placed and becomes empty, as for a text removed whole. Then
    cut_string cuts its three strings too."""
    cut, placed = {}, {}  # placed: the erased places of the selector of each (text, selector), by version found
    for text, conversation_versions in documents.items():
        for version, document in conversation_versions.items():
            content = json.loads(document)
            value = _texts_mapped(content, cut_string)
            for block in quotes.excerpts(value):
                selector, source = block["selector"], block["source"]
                if source in gone:
                    _set_selector(selector, "", "", "")
                elif source in versions:
                    prefix, exact, suffix = (selector.get(key, "") for key in ("prefix", "exact", "suffix"))
                    key = (source, prefix, exact, suffix)
                    if key not in placed:
                        placed[key] = _erased_offsets(versions[source], source, selector, erased_in)
                    offsets = placed[key]
                    if offsets is None:
                        _set_selector(selector, "", "", "")
                    else:
                        joined = prefix + exact + suffix
                        bounds = (len(prefix), len(prefix) + len(exact))
                        kept = [at for at in range(len(joined)) if at not in offsets]
                        _set_selector(selector, *(
                            "".join(joined[at] for at in kept if low <= at < high)
                            for low, high in ((0, bounds[0]), bounds, (bounds[1], len(joined)))))
                _set_selector(selector, *(cut_string(selector.get(key, "")) for key in ("prefix", "exact", "suffix")))
            written = json.dumps(value, ensure_ascii=False)
            if written != json.dumps(content, ensure_ascii=False):
                cut.setdefault(text, {})[version] = written
    return cut


def _erased_offsets(versions: dict[int, object], source: str, selector: dict,
                    erased_in: Callable[[str, int, int, int], list[tuple[int, int]]]) -> set[int] | None:
    """The places, in prefix + exact + suffix, of the characters an excerpt's
    selector loses (_cut_quoting): those over an erased character of a
    version of its text where it matches exactly one place; None when it
    matches no version at exactly one place."""
    prefix, joined = selector.get("prefix", ""), "".join(selector.get(key, "") for key in ("prefix", "exact", "suffix"))
    offsets, located = set(), False
    for version, document in versions.items():
        places = quotes.find(document, selector)
        if len(places) != 1:
            continue
        located = True
        start = places[0] - len(prefix)
        end = start + len(joined)
        for a, b in erased_in(source, version, start, end):
            offsets.update(range(max(a, start) - start, min(b, end) - start))
    return offsets if located else None


def _set_selector(selector: dict, prefix: str, exact: str, suffix: str):
    """Write the selector's prefix, exact and suffix: exact always, prefix and suffix where it had them or not empty."""
    selector["exact"] = exact
    for key, value in (("prefix", prefix), ("suffix", suffix)):
        if key in selector or value:
            selector[key] = value


def _texts_mapped(conversation: dict, mapped: Callable[[str], str]) -> dict:
    """A copy of the conversation (the chat format, docs/chat-format.md)
    whose strings that carry text are replaced by what mapped gives of
    them: each tool's description, the text of a text or thinking block,
    the values (never the keys) of a tool call's arguments and of a json
    block, and the same within a tool result's blocks. Everything else is
    copied as it is, the excerpts' selectors included."""
    value = copy.deepcopy(conversation)
    for tool in value.get("tools", []):
        if isinstance(tool, dict) and isinstance(tool.get("description"), str):
            tool["description"] = mapped(tool["description"])
    for message in value.get("messages", []):
        for block in message.get("content", []):
            _block_mapped(block, mapped)
    return value


def _block_mapped(block: dict, mapped: Callable[[str], str]):
    kind = block.get("type")
    if kind in ("text", "thinking") and isinstance(block.get("text"), str):
        block["text"] = mapped(block["text"])
    elif kind == "tool_call" and isinstance(block.get("arguments"), dict):
        block["arguments"] = _strings_mapped(block["arguments"], mapped)
    elif kind == "json" and "value" in block:
        block["value"] = _strings_mapped(block["value"], mapped)
    elif kind == "tool_result":
        for inner in block.get("content", []):
            _block_mapped(inner, mapped)


def _strings_mapped(value, mapped: Callable[[str], str]):
    """A new JSON value: value with each of its strings (not its keys) replaced by what mapped gives of it."""
    if isinstance(value, str):
        return mapped(value)
    if isinstance(value, list):
        return [_strings_mapped(one, mapped) for one in value]
    if isinstance(value, dict):
        return {key: _strings_mapped(one, mapped) for key, one in value.items()}
    return value


def _path(content, pointer: str) -> tuple:
    """The steps from content to the string at a JSON Pointer: a key, or a list's index as a number."""
    steps, value = [], content
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        step = int(part) if isinstance(value, list) else part
        steps.append(step)
        value = value[step]
    return tuple(steps)


def _cut_json(value, path: tuple, named: dict, removing: list[str], places: dict):
    """value with every string cut (_cut_string): the named passages of the
    string at each path, then each passage of removing; the places found go
    into places."""
    if isinstance(value, str):
        value, found = _cut_string(value, named.get(path, []), removing)
        places.update(found)
        return value
    if isinstance(value, list):
        return [_cut_json(one, (*path, at), named, removing, places) for at, one in enumerate(value)]
    if isinstance(value, dict):
        return {key: _cut_json(one, (*path, key), named, removing, places) for key, one in value.items()}
    return value


def _cut_string(value: str, named: list[tuple[int, int, int]], removing: list[str]) -> tuple[str, dict[int, int]]:
    """(value cut, {number: place}), as _cut_pieces cuts it."""
    pieces, places = _cut_pieces(_blocks_of(value), len(value), named, removing)
    return "".join(value[a:b] for a, b in pieces), places


def _blocks_of(document) -> Callable[[], Iterator[tuple[int, str]]]:
    """How to read a str or a depot.parts.Parted again and again, as
    (its first character's place, a block of text), in order: a str whole,
    a Parted part by part."""
    if isinstance(document, str):
        return lambda: iter([(0, document)])

    def blocks():
        for start, value in zip(document.starts, document):
            yield start, value
    return blocks


def _cut_pieces(blocks: Callable[[], Iterable[tuple[int, str]]], length: int, named: list[tuple[int, int, int]],
                removing: list[str]) -> tuple[list[tuple[int, int]], dict[int, int]]:
    """(pieces, {number: place}): what is left of a string of length
    characters, read as blocks() gives it, once it loses the characters of
    each named passage (start, end, number), then each passage of removing
    at its leftmost places that do not overlap (as str.replace finds them),
    one passage after the other: the places [(start, end)] in the string of
    the characters that stay, in order. Each named passage's place is how
    many characters that stay stood before its start. Each passage of
    removing reads the string once, a block at a time."""
    pieces = [(0, length)]
    for start, end, _ in sorted(named):
        pieces = _without_range(pieces, start, end)
    for passage in removing:
        spans = _spans(_kept(blocks(), pieces), passage)
        if spans:
            pieces = _without_spans(pieces, spans)
    places = {number: sum(max(0, min(b, start) - a) for a, b in pieces) for start, _, number in named}
    return pieces, places


def _kept(blocks: Iterable[tuple[int, str]], pieces: list[tuple[int, int]]) -> Iterator[str]:
    """The characters of the blocks that pieces keep, in order, a piece of a block at a time."""
    first = 0
    for start, value in blocks:
        end = start + len(value)
        while first < len(pieces) and pieces[first][1] <= start:
            first += 1
        at = first
        while at < len(pieces) and pieces[at][0] < end:
            a, b = pieces[at]
            yield value[max(a, start) - start:min(b, end) - start]
            at += 1


def _spans(segments: Iterable[str], passage: str) -> list[tuple[int, int]]:
    """The places [(start, end)] of passage in the string the segments make
    one after the other, leftmost first and not overlapping, as repeated
    str.find gives them; at most a segment and the passage's length are
    held at a time."""
    spans, held, offset, at = [], "", 0, 0  # offset: the place of held's first character; at: where to look on
    for segment in segments:
        held += segment
        found = held.find(passage, at)
        while found != -1:
            spans.append((offset + found, offset + found + len(passage)))
            at = found + len(passage)
            found = held.find(passage, at)
        kept = max(at, len(held) - len(passage) + 1)  # a place before it cannot start a passage not found yet
        held, offset, at = held[kept:], offset + kept, 0
    return spans


def _without_spans(pieces: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """pieces less the characters of each span [start, end) of the string
    they make now, the spans in order and apart."""
    found, offset, next_ = [], 0, 0
    for a, b in pieces:
        low_end, at = offset + b - a, a  # at: the first place of this piece not handled yet
        while next_ < len(spans) and spans[next_][0] < low_end:
            start, end = spans[next_]
            low, high = a + max(start, offset) - offset, a + min(end, low_end) - offset
            if low > at:
                found.append((at, low))
            at = max(at, high)
            if end > low_end:
                break
            next_ += 1
        if at < b:
            found.append((at, b))
        offset = low_end
    return found


def _without_range(pieces: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """pieces less the places [start, end) of the original value."""
    found = []
    for a, b in pieces:
        if b <= start or a >= end:
            found.append((a, b))
            continue
        if a < start:
            found.append((a, start))
        if b > end:
            found.append((end, b))
    return found


def shown(needles: list[Needle], documents: dict[str, dict[int, str]], items=frozenset(),
          erased: list[dict] | None = None, context: int = CONTEXT) -> list[dict]:
    """What each needle of one request asks to erase, in order, read from
    the content version it names in documents, every content version of every id they name that the depot
    still holds ({id: {version: its text}}; for an id of items, its content
    as JSON text), with erased, the
    request's erased entries once it is carried out (None before). Each is
    {state, before, passage, after, from, length, erased}, where
    the string a needle counts in is its text, the string at its field, or
    for a whole item its content as JSON text, and length is that string's
    length now; omitted and passage_end are 0 and "" unless the passage is
    cut:

    - "passage": the request is not carried out; passage is the needle's
      passage, before and after at most context characters on each side of
      it, from the position where before starts. A passage longer than twice
      context is sent as its first context characters (passage), how many
      characters follow them unsent (omitted) and its last context
      characters (passage_end). For a needle with no start, passage is the
      head of the text, at most twice context characters.
    - "erased": the request is carried out, so the passage is no longer
      there; erased is the length of the passage the needle was part of
      (merged with the needles of the same string it overlaps or touches),
      and before and after are the text around the place it left. That
      place is the one recorded when the request was carried out; a later
      erasure of the same string can still move the text around it. For a
      needle with no start, the whole text went: erased is its length (None
      when another erasure had removed it before) and nothing is around
      it.
    - "gone": the depot no longer holds the text (another needle or another
      erasure removed it whole).
    - "misplaced": the request is not carried out and its place no longer
      fits the string (another erasure cut it since), or its field no longer
      names a string."""
    carried = erased is not None
    entries = _entries(needles, erased) if carried else [None] * len(needles)
    found = []
    for index, one in enumerate(needles):
        entry = entries[index]
        place = {"state": "gone", "before": "", "passage": None, "omitted": 0, "passage_end": "", "after": "", "from": 0,
                 "length": None, "erased": None}
        if one.start is None and carried:
            found.append({**place, "state": "erased", "erased": None if entry is None else entry["length"]})
            continue
        if one.text not in documents:
            found.append(place)
            continue
        value = documents[one.text][one.version]
        if one.field is not None:
            try:
                value = resolve(json.loads(value), one.field)
            except Refused:
                found.append({**place, "state": "misplaced"})
                continue
        place["length"] = len(value)
        if one.start is None:
            found.append({**place, "state": "passage", "passage": value[:2 * context]})
        elif carried:
            at, length = max(0, min(len(value), entry["at"])), entry["length"]
            found.append({**place, "state": "erased", "erased": length,
                          "from": max(0, at - context), "before": value[max(0, at - context):at],
                          "after": value[at:at + context]})
        elif one.end > len(value):
            found.append({**place, "state": "misplaced"})
        else:
            long = one.end - one.start > 2 * context  # only its two ends are sent
            found.append({**place, "state": "passage", "from": max(0, one.start - context),
                          "before": value[max(0, one.start - context):one.start],
                          "passage": value[one.start:one.start + context] if long else value[one.start:one.end],
                          "omitted": one.end - one.start - 2 * context if long else 0,
                          "passage_end": value[one.end - context:one.end] if long else "",
                          "after": value[one.end:one.end + context]})
    return found


def _entries(needles: list[Needle], erased: list[dict]) -> list[dict | None]:
    """The erased entry of each needle: the entry that names the first
    needle of its passage (merged(needles)); None for a needle whose text
    another erasure had removed before, which has none."""
    by_first = {one["needle"]: one for one in erased}
    found = [None] * len(needles)
    for passage in merged(needles):
        for index in passage.needles:
            found[index] = by_first.get(passage.needles[0])
    return found


def named(passage) -> dict:
    """{length, hash_a, hash_b} of a passage: a str, or an iterable of the
    str blocks it is made of (as a depot.parts.Parted gives them), read
    once, a block at a time."""
    blocks = [passage] if isinstance(passage, str) else passage
    length, digest, value = 0, hashlib.sha256(), 0
    for block in blocks:
        length += len(block)
        digest.update(block.encode())
        value = rolling(block, value)
    return {"length": length, "hash_a": digest.digest(), "hash_b": value.to_bytes(8, "big")}


def holds_erased(strings: Iterable, erased: list[dict]) -> bool:
    """Whether one of the strings holds a passage whose {length, hash_a}
    is one of erased: every window of each length is hashed and compared.
    Each string is a str, or an iterable of the str blocks it is made of,
    read once, holding a block and the longest length at a time."""
    wanted: dict[int, set[bytes]] = {}
    for one in erased:
        wanted.setdefault(one["length"], set()).add(bytes(one["hash_a"]))
    if not wanted:
        return False
    longest = max(wanted)
    for string in strings:
        held, seen = "", 0  # seen: the characters of held whose windows were compared already
        for block in [string] if isinstance(string, str) else string:
            held += block
            if _holds(held, seen, wanted):
                return True
            kept = max(0, len(held) - longest + 1)
            held = held[kept:]
            seen = len(held)
    return False


def _holds(held: str, seen: int, wanted: dict[int, set[bytes]]) -> bool:
    """Whether a window of held of a length wanted names, that ends past its
    first seen characters, hashes to one of its hash_a."""
    return any(hashlib.sha256(held[start:start + length].encode("utf-8", "surrogatepass")).digest() in hashes
               for length, hashes in wanted.items()
               for start in range(max(0, seen - length + 1), len(held) - length + 1))


def strings_of(value) -> list[str]:
    """Every string of a JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for key, inner in value.items() for s in [key, *strings_of(inner)]]
    if isinstance(value, list):
        return [s for inner in value for s in strings_of(inner)]
    return []


def rolling(passage: str, value: int = 0) -> int:
    """The polynomial hash of passage: the sum of number(c_k) x
    base^(L-1-k) modulo MODULUS, for its L characters c_k; with value, the
    hash of a string that goes on from one whose hash is value."""
    base = _number(b"base") or 1
    numbers = {}
    for character in passage:
        if character not in numbers:
            numbers[character] = _number(b"character" + character.encode("utf-8", "surrogatepass"))
        value = (value * base + numbers[character]) % MODULUS
    return value


def _number(what: bytes) -> int:
    return int.from_bytes(hashlib.sha256(DOMAIN + what).digest()[:8], "big") % MODULUS
