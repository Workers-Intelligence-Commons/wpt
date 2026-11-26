"""What a request chooses in a snapshot, read the same whatever the
preparation (docs/data-selection.md, "Shares", "Filling a part", "Long
texts", "Corpora", "Conversations that quote texts"), and the key of its
delivery.

A request's selection names the kinds it takes (none: every kind the
preparation prepares), its shares by tag ([{tag, share}] in order; none:
every plain text, in the snapshot's own proportions) and its size (None:
everything the rules take; {unit, amount}: a number of bytes (unit
BYTES), which the freight measures on the listing's bytes, or a number of
the preparation's unit, which only the preparation can weigh). canonical
gives it in the form the delivery key covers.

A plain text counts for the first share whose tag it has, and a plain
text that has none of their tags is left out. An item (any other kind) is
taken whole when the preparation prepares its kind, and is never cut. The
listing of the snapshot (depot's api, docs/api/depot.md) gives each text,
or each piece of a long plain text, with its place, revision, kind, tags
and bytes; walk reads it as Entry, each with its pool and whether a
delivery takes it at all (pool_of).

What a delivery of a snapshot may take whatever its size (census) comes
from the snapshot's summary alone, so it never changes, an erasure after it
included.

    canonical(selection)                              a selection in the form the key covers
    delivery_key(snapshot, selection, preparation)    the key of a delivery
    census(summary, shares, kinds, prepared)          what a delivery of the snapshot may take
    walk(source, snapshot, shares, kinds, prepared)   the snapshot's listing under a selection, as Entry
    pool_of(kind, tags, shares, kinds, prepared)      a text's pool and whether a delivery takes it
    excerpts(conversation)                            the excerpts of a conversation, in order
"""

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from typing import NamedTuple

from depot import api

PLAIN, CONVERSATION = "plain", "conversation"
BYTES = "bytes"  # the unit of a size the freight measures itself, on the bytes the listing gives each entry
RULES = 1  # the version of the rules of what a job takes, in the delivery key: raised whenever they change
INTERNED = 10_000  # different lists of tags kept as their JSON text, at most


class Refused(ValueError):
    """A request that cannot be met as it is: nothing was delivered."""


def canonical(selection: dict) -> dict:
    """What a selection says, in the form the delivery key covers: {kinds:
    the kinds named, sorted, each once ([]: every kind), shares: [{tag,
    share}] in order ([]: none), size: None or {unit, amount}, amount a
    whole number above 0}. Refused for a selection that cannot be met:
    kinds not a list of names, shares whose tags repeat or whose share is
    below 0 or not a number, or whose shares are all 0, a size that is not
    a unit and a whole number above 0."""
    kinds = selection.get("kinds") or []
    if not isinstance(kinds, list) or not all(isinstance(one, str) and one for one in kinds):
        raise Refused(f"the selection's kinds are a list of kinds' names: {kinds!r}")
    shares = selection.get("shares") or []
    if not isinstance(shares, list) or not all(isinstance(one, dict) and isinstance(one.get("tag"), str)
                                               and isinstance(one.get("share"), (int, float))
                                               and not isinstance(one.get("share"), bool) and one["share"] >= 0
                                               for one in shares):
        raise Refused(f"the selection's shares are a list of {{tag, share}}, each share a number of 0 or more: "
                      f"{shares!r}")
    tags = [one["tag"] for one in shares]
    if len(set(tags)) != len(tags):
        raise Refused(f"the selection's shares name a tag twice: {', '.join(tags)}")
    if shares and not sum(one["share"] for one in shares):
        raise Refused("the selection's shares are all 0: give one a share above 0, or give no shares")
    size = selection.get("size")
    if size is not None:
        amount = size.get("amount") if isinstance(size, dict) else None
        if not isinstance(size, dict) or not isinstance(size.get("unit"), str) or not size["unit"] \
                or not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            raise Refused(f"a selection's size is {{unit, amount}}, amount a whole number above 0, or null: {size!r}")
        size = {"unit": size["unit"], "amount": amount}
    return {"kinds": sorted(set(kinds)), "shares": [{"tag": one["tag"], "share": float(one["share"])}
                                                    for one in shares], "size": size}


def delivery_key(snapshot: dict, selection: dict, preparation: dict) -> str:
    """The key of the delivery of a request: the SHA-256 of the canonical
    JSON of the snapshot's digest and change, the selection (canonical),
    the preparation's name, the version its program answers, its settings,
    the delivery keys of its inputs, and RULES. Never the depot's address,
    the snapshot's name, the code's commit or the machine, so the same
    request lands in the same folder wherever it runs, and a caller works
    it out with no depot at hand. snapshot: {change, digest}; preparation:
    {name, version, settings, inputs: {name: delivery key}}."""
    return fingerprint({"snapshot": {"change": snapshot["change"], "digest": snapshot["digest"]},
                        "selection": canonical(selection),
                        "preparation": {"name": preparation["name"], "version": preparation["version"],
                                        "settings": preparation.get("settings") or {},
                                        "inputs": dict(sorted((preparation.get("inputs") or {}).items()))},
                        "rules": RULES})


def fingerprint(value) -> str:
    """The sha256 of value as canonical JSON."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


class Entry(NamedTuple):
    """A text or a piece of a snapshot's listing under a selection: its key
    (as an int, which orders places), owner, number, id, piece (None for a
    whole text), revision, kind, bytes, tags (as JSON text) and name (the
    text's id or the piece's name), and pool and taken (pool_of)."""
    key: int
    owner: str
    number: int
    id: str
    piece: int | None
    revision: int
    kind: str
    pool: int | None
    bytes: int
    taken: bool
    tags: str
    name: str

    def place(self) -> list:
        """The entry's place in the listing, [key, owner, number, piece (-1 for a whole text)], what a walk's
        after names."""
        return [self.key, self.owner, self.number, -1 if self.piece is None else self.piece]


def items_of(prepared: Sequence[str], kinds: Sequence[str] | None = None) -> list[str]:
    """The kinds of items (every kind but plain) a delivery takes: those the
    preparation prepares, of the kinds named (None or empty: every kind)."""
    return [kind for kind in sorted(prepared) if kind != PLAIN and (not kinds or kind in kinds)]


def pool_of(kind: str, tags: str, shares: Sequence[str] | None, kinds: Sequence[str] | None,
            prepared: Sequence[str]) -> tuple[int | None, bool]:
    """(pool, taken) of a text of that kind with tags (JSON text), under
    the tags of a selection's shares (None without shares), the kinds it
    names (None or empty: every kind) and the kinds the preparation
    prepares: pool, with shares, the index of the first share whose tag a
    plain text has, None when it has none; without shares, 0 for every
    plain text; None for an item; taken, whether a delivery takes it at
    all, that is, a text of a kind named and prepared that is a plain text
    with a pool or an item."""
    if (kinds and kind not in kinds) or kind not in prepared:
        return None, False
    if kind != PLAIN:
        return None, True
    if shares is None:
        return 0, True
    held = json.loads(tags or "[]")
    pool = next((at for at, tag in enumerate(shares) if tag in held), None)
    return pool, pool is not None


_TAGS_TEXT = {}  # a list of tags as a tuple: its JSON text, a few kept


def tags_text(tags: list[str]) -> str:
    """A list of tags as JSON text, the form an entry keeps them in."""
    key = tuple(tags)
    found = _TAGS_TEXT.get(key)
    if found is None:
        if len(_TAGS_TEXT) >= INTERNED:
            _TAGS_TEXT.clear()
        found = _TAGS_TEXT[key] = json.dumps(tags, ensure_ascii=False, separators=(",", ":"))
    return found


def entries(listed: Iterable[dict], shares: Sequence[str] | None, kinds: Sequence[str] | None,
            prepared: Sequence[str]) -> Iterator[Entry]:
    """The entries of a listing (the depot's api), each as Entry under the
    tags of a selection's shares (None without shares), the kinds it names
    and the kinds the preparation prepares."""
    for one in listed:
        tags = tags_text(one["tags"])
        pool, taken = pool_of(one["kind"], tags, shares, kinds, prepared)
        yield Entry(int(one["key"]), one["owner"], one["number"], one["id"], one["piece"], one["revision"],
                    one["kind"], pool, one["bytes"], taken, tags, one["name"])


def walk(source: str, snapshot: str, shares: Sequence[str] | None, kinds: Sequence[str] | None,
         prepared: Sequence[str], after: Sequence | None = None) -> Iterator[Entry]:
    """The listing of the snapshot whose id is snapshot of the depot at the
    address source, from the place after on, as entries (Entry)."""
    if after is not None:
        after = [str(after[0]), *after[1:]]
    return entries(api.listing(source, snapshot, after), shares, kinds, prepared)


def tags_of(shares: list[dict] | None) -> list[str] | None:
    """The tags of a selection's shares, in order; None without shares."""
    return [one["tag"] for one in shares] if shares else None


def proportions(shares: list[dict] | None, total: int) -> dict[int, int]:
    """{pool: its part of total}: each share's proportion of it, or all of it for the one pool without shares."""
    if not shares:
        return {0: total}
    whole = sum(share["share"] for share in shares)
    return {at: int(total * share["share"] / whole) for at, share in enumerate(shares)}


def scarcest_trim(available: dict[int, float], shares: list[dict]) -> dict[int, float]:
    """{pool: what it may take} without a size, from what each pool of
    shares holds (bytes): the scarcest share all it holds, each other its
    proportion of the scarcest's, at most all it holds. A share that holds
    nothing takes nothing, and the others are measured among themselves."""
    held = [pool for pool in available if available[pool]]
    if not held:
        return dict.fromkeys(available, 0)
    whole = sum(share["share"] for share in shares)
    proportion = {pool: shares[pool]["share"] / whole for pool in available}
    scarcest = min(held, key=lambda pool: available[pool] / proportion[pool])
    total = available[scarcest] / proportion[scarcest]
    return {pool: available[pool] if pool == scarcest else min(available[pool], proportion[pool] * total)
            for pool in available}


def census(summary: dict, shares: list[dict] | None, kinds: list[str] | None, prepared: Sequence[str]) -> dict:
    """What a delivery of a snapshot whose summary is summary (the depot's
    api) may take whatever its size, under the shares ([{tag, share}] in
    order, none: every plain text) and kinds (none: every kind) of a
    selection and the kinds the preparation prepares: {pool_bytes: {pool:
    the bytes of the plain texts of that pool}, pool_texts: {pool: how many
    such texts}, plain: how many in all, item_bytes and items: the bytes
    and number of the items taken, no_share: the plain texts no share
    takes, other: the texts of the kinds not taken, kinds: {kind: {texts,
    bytes}} of each kind on its own, texts: every text}, from the summary
    alone: a plain text counts for the first share whose tag it has."""
    allowed = set(kinds) if kinds else None
    sizes = {kind: {"texts": one[0], "bytes": one[2]} for kind, one in summary["kinds"].items()}
    taken_plain = PLAIN in prepared and (allowed is None or PLAIN in allowed)
    tags = tags_of(shares)
    pools = [0] if tags is None else list(range(len(tags)))
    pool_bytes, pool_texts, no_share = dict.fromkeys(pools, 0), dict.fromkeys(pools, 0), 0
    if taken_plain and tags is None:
        plain = sizes.get(PLAIN, {"texts": 0, "bytes": 0})
        pool_bytes[0], pool_texts[0] = plain["bytes"], plain["texts"]
    elif taken_plain:
        for joined, (texts, _, size) in summary["plain_tagsets"].items():
            held = set(joined.split("\0")) if joined else set()
            pool = next((at for at, tag in enumerate(tags) if tag in held), None)
            if pool is None:
                no_share += texts
            else:
                pool_bytes[pool] += size
                pool_texts[pool] += texts
    items = items_of(prepared, kinds)
    other = sum(one["texts"] for kind, one in sizes.items()
                if kind not in items and not (kind == PLAIN and taken_plain))
    return {"pool_bytes": pool_bytes, "pool_texts": pool_texts, "plain": sum(pool_texts.values()),
            "item_bytes": sum(sizes.get(kind, {"bytes": 0})["bytes"] for kind in items),
            "items": sum(sizes.get(kind, {"texts": 0})["texts"] for kind in items),
            "no_share": no_share, "other": other, "kinds": sizes,
            "texts": sum(one["texts"] for one in sizes.values())}


def excerpts(conversation: dict) -> Iterator[dict]:
    """Every excerpt block of a conversation (docs/chat-format.md,
    "Excerpts"), in order: in the content of its messages and in the
    content of their tool results."""
    for message in conversation.get("messages", []):
        for block in message.get("content", []):
            if block.get("type") == "excerpt":
                yield block
            elif block.get("type") == "tool_result":
                yield from (inner for inner in block.get("content", []) if inner.get("type") == "excerpt")
