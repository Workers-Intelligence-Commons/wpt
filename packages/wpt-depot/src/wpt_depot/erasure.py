"""Carrying out an erasure on texts, and the two public hashes that name an
erased passage without its content.

A text's content can change, each change a new content version kept next
to the old ones (wpt_depot.depot), so a needle names a passage by its place
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

What a request asks to erase can be looked at without sending a whole text:
for each needle, at most CONTEXT characters on each side of its passage, and
the passage (only its two ends when it is long), or once the request is
carried out, the passage's length at the place it left (shown).
"""

import hashlib
import json
from typing import NamedTuple

from .errors import Refused

MODULUS = (1 << 61) - 1  # a Mersenne prime
DOMAIN = b"wpt-depot erasure hash 1\0"  # the fixed parameter every number of hash_b is made from
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
    if row["kind"] is None:
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


def carried_out(documents: dict[str, dict[int, str]], needles: list[Needle], items=frozenset()
                ) -> tuple[dict[str, dict[int, str]], list[str], list[tuple[Passage, str, int | None]]]:
    """(the cut content versions, {id: {version: its text cut}} for each
    version the erasure changes, the ids of texts removed whole, (passage,
    its text, its place after the cut) for each passage of merged(needles)
    in order): what an erasure of needles leaves of documents, every content
    version of every id they name ({id: {version: its text}}; for an id of
    items, its content as JSON text).

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
                value, found = _cut_string(document, [(one.start, one.end, at) for at, one in here], removing)
                places.update(found)
                if value != document:
                    cut.setdefault(text, {})[version] = value
    return cut, removed, [(one, texts[at], places.get(at)) for at, one in enumerate(passages)]


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
    """(value cut, {number: place}): value without the characters of each
    named passage (start, end, number), then without each passage of
    removing at its leftmost places that do not overlap, in order; each
    named passage's place is how many characters of the cut value stood
    before its start."""
    if not named and not any(one in value for one in removing):
        return value, {}
    # what is left, as the places in value of its pieces, in order: (start, end)
    pieces = [(0, len(value))]
    for start, end, _ in sorted(named):
        pieces = _without_range(pieces, start, end)
    for passage in removing:
        current = "".join(value[a:b] for a, b in pieces)
        spans, at = [], current.find(passage)
        while at != -1:
            spans.append((at, at + len(passage)))
            at = current.find(passage, at + len(passage))
        for start, end in reversed(spans):  # each span in the places of current, the last first
            pieces = _without_current(pieces, start, end)
    places = {number: sum(max(0, min(b, start) - a) for a, b in pieces) for start, _, number in named}
    return "".join(value[a:b] for a, b in pieces), places


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


def _without_current(pieces: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """pieces less the characters [start, end) of the string they make now."""
    found, offset = [], 0
    for a, b in pieces:
        low, high = max(start - offset, 0), min(end - offset, b - a)
        if low >= high:
            found.append((a, b))
        else:
            if low > 0:
                found.append((a, a + low))
            if high < b - a:
                found.append((a + high, b))
        offset += b - a
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


def named(passage: str) -> dict:
    """{length, hash_a, hash_b} of a passage."""
    return {"length": len(passage), "hash_a": hashlib.sha256(passage.encode()).digest(),
            "hash_b": rolling(passage).to_bytes(8, "big")}


def holds_erased(strings: list[str], erased: list[dict]) -> bool:
    """Whether one of the strings holds a passage whose {length, hash_a}
    is one of erased: every window of each length is hashed and compared."""
    wanted: dict[int, set[bytes]] = {}
    for one in erased:
        wanted.setdefault(one["length"], set()).add(bytes(one["hash_a"]))
    return any(hashlib.sha256(string[start:start + length].encode("utf-8", "surrogatepass")).digest() in hashes
               for string in strings for length, hashes in wanted.items()
               for start in range(len(string) - length + 1))


def strings_of(value) -> list[str]:
    """Every string of a JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for key, inner in value.items() for s in [key, *strings_of(inner)]]
    if isinstance(value, list):
        return [s for inner in value for s in strings_of(inner)]
    return []


def rolling(passage: str) -> int:
    """The polynomial hash of passage: the sum of number(c_k) x
    base^(L-1-k) modulo MODULUS, for its L characters c_k."""
    base = _number(b"base") or 1
    value, numbers = 0, {}
    for character in passage:
        if character not in numbers:
            numbers[character] = _number(b"character" + character.encode("utf-8", "surrogatepass"))
        value = (value * base + numbers[character]) % MODULUS
    return value


def _number(what: bytes) -> int:
    return int.from_bytes(hashlib.sha256(DOMAIN + what).digest()[:8], "big") % MODULUS
