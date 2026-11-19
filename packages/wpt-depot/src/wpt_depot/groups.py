"""How each share of a dataset splits its budget evenly over the tags its
texts carry (docs/data-selection.md, "Shares"): a first pass over the
dataset's order, the choice of texts it makes, and the chosen set a corpus
build or a vocabulary's sample follows as it walks the order again.

A plain text belongs to the first share whose tag it has (wpt_depot.order).
Inside a share, every tag its texts carry other than the share's own is a
group, and the texts that carry no other tag are one group more, whose tag
is None. Nothing here knows a tag's name or kind: the groups come from the
texts' tags alone. A piece of a long text is an entry of its own in the
groups of its text's tags.

The first pass (plan) walks the whole order once, before anything is
chosen, and keeps numpy arrays only, never an object per text: for each
entry of a share's plain texts, its depot and id (an id is a depot's name,
a hyphen and a number, wpt_depot.ids: the name interned and the number
kept; an id of another shape is kept apart), its revision, its piece, its
bytes and the ids of its groups; then each group's queue, its entries in
the order. That is about 25 bytes an entry and 8 bytes for each pair of an
entry and one of its groups; 8 bytes a pair more for a moment while the
queues are made, and 5 bytes an entry more while it chooses (what is
decided of each entry, and its tokens when sizes are tokens).

A share with one group, or with no limit (a budget of None), is no part of
the plan: it fills as wpt_depot.fill.Fill says. Every other share chooses
its texts so (choose): it takes the next text, in the order, of the group
that holds the fewest tokens (or bytes) so far, ties going to the group
whose next text comes first in the order, then to the group seen first in
the order; a text taken counts for every one of its groups and is taken
once; a text that does not fit what is left of the share's budget is left
out for good; a group drops out once its queue is empty, or once SKIPS of
the texts it met in a row did not fit; the share closes once what is left
of its budget is at most a FULL-th of it, or once every group dropped out.

Sizes are tokens when a count is given (count: the tokens of a list of
entries, (depot file, id, revision, piece) each, that wpt_depot.corpus
works out in worker processes, asked for the next texts of every group of
the share at once, LOOKAHEAD of each at first and twice as many each time
after, at most LOOKAHEAD_MOST); else bytes. Tokens never depend on what is
chosen, so the same depot state chooses the same texts.

A plan's entries are the plain texts of the shares it covers, in the order;
an entry's place is its rank among them. The chosen set is the sorted array
of the places of the entries chosen, so a walk that counts the entries of
those shares as it meets them knows whether each one is chosen (Picker).

    plan(libraries, tags, budget, long, addresses, count, progress)   the texts the shares choose, as a Plan
    groups_of(tags, share)                                            the groups of a text of a share
"""

import array
import hashlib
import heapq
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .fill import FULL, SKIPS
from .order import walk

LOOKAHEAD = 16  # texts of each group whose tokens are asked for at first, ahead of the group's head
LOOKAHEAD_MOST = 512  # and at most, the number doubling each time
ASKED_BYTES = 64 << 20  # bytes of texts one request for tokens names at most, beyond the text it is for
TAKEN, LEFT_OUT = 1, 2  # what is decided of an entry (0: nothing yet)
MEASURED = 100_000  # entries walked, or decided, between two reports of the progress
INTERNED = 10_000  # different (share, tags) kept with their groups' ids, at most
BIG_NUMBER = 1 << 32  # an id's number from which it is kept apart, as an id of another shape
QUEUED = 1 << 20  # pairs of an entry and a group placed in the queues at a time


def groups_of(tags: str, share: str) -> list[str | None]:
    """The groups of a plain text whose tags are tags (JSON text) in the
    share of tag share: each of its other tags once, in its order, or
    [None] when it has none."""
    return [tag for tag in dict.fromkeys(json.loads(tags)) if tag != share] or [None]


class Plan:
    """The texts the shares a plan covers chose: last, {pool: the place of
    its last chosen entry, -1 for none}, one key for each share covered,
    and chosen, the sorted places of every chosen entry."""

    def __init__(self, last: dict[int, int], chosen: np.ndarray):
        self.last, self.chosen = last, chosen

    def save(self, path: Path) -> str:
        """Write the chosen places into path, as little-endian 64-bit
        integers: the sha256 of those bytes."""
        data = self.chosen.astype("<i8").tobytes()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path: Path, last: dict[int, int], sha256: str) -> "Plan | None":
        """The plan whose chosen places save wrote into path, or None when
        the file is missing or holds other bytes than those of sha256."""
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        if hashlib.sha256(data).hexdigest() != sha256:
            return None
        return cls(last, np.frombuffer(data, "<i8").astype(np.int64))

    def picker(self, seen: int = 0) -> "Picker":
        return Picker(self, seen)


class Picker:
    """A walk of the order through a plan's entries, from the entry whose
    place is seen: pick says of each entry of a covered share whether it is
    chosen, and done whether every covered share is past its last chosen
    entry."""

    def __init__(self, plan: Plan, seen: int = 0):
        self.last = plan.last
        self.chosen = memoryview(plan.chosen)
        self.seen = seen
        self.at = int(np.searchsorted(plan.chosen, seen))
        self.open = {pool for pool, last in plan.last.items() if last >= seen}

    def covers(self, pool) -> bool:
        return pool in self.last

    def pick(self, pool: int) -> tuple[int, bool]:
        """(the place of the next entry, of the covered share pool, whether it is chosen)."""
        place = self.seen
        self.seen += 1
        chosen = self.at < len(self.chosen) and self.chosen[self.at] == place
        if chosen:
            self.at += 1
        if place >= self.last[pool]:
            self.open.discard(pool)
        return place, chosen

    def done(self) -> bool:
        return not self.open


def plan(libraries: Sequence[tuple], tags: list[str], budget: dict, long: list,
         addresses: Sequence[str] | None = None, count: Callable[[list[tuple]], list[int]] | None = None,
         progress=lambda state: None) -> Plan:
    """The texts each share of a dataset chooses, as the module says:
    libraries and long as wpt_depot.order.walk takes them, tags the tags of
    the shares in order, budget {pool: the most it takes, None for no
    limit}; addresses, the depot file of each source, and count, the tokens
    of a list of entries, when sizes are tokens (else bytes). progress gets
    {stage: "choosing", ...} now and then."""
    covered = {pool for pool, limit in budget.items() if limit is not None}
    if not covered:
        return Plan({}, np.zeros(0, np.int64))
    entries = _first_pass(libraries, tags, covered, long, progress)
    if entries is None:
        return Plan({}, np.zeros(0, np.int64))
    return _choose(entries, budget, addresses, count, progress)


class _Entries:
    """What the first pass keeps of the entries of the shares a plan covers
    with two groups or more, in the order: whos (each interned (source, id
    without its number)), odd (the (source, id) of an id kept apart, whose
    who is -1 - its index), and arrays: who, number, revision, piece (-1:
    a whole text), bytes, offsets (an entry's groups are groups[offsets[e]:
    offsets[e + 1]]), groups, and for each group its pool, its tag and its
    queue (queue[starts[g]:starts[g + 1]], its entries in the order)."""


def _first_pass(libraries, tags, covered, long, progress) -> _Entries | None:
    """The entries of the shares covered that have two groups or more
    (_Entries), from one walk of the whole order; None when no share has."""
    share, who, number = array.array("h"), array.array("i"), array.array("I")
    revision, piece, size = array.array("I"), array.array("i"), array.array("I")
    counts, pairs = array.array("H"), array.array("i")
    whos, odd, interned_who = [], [], {}
    group_key, group_ids, interned = [], {}, {}
    walked = 0
    for entry in walk(libraries, tags, long=long):
        walked += 1
        if walked % MEASURED == 0:
            progress({"stage": "choosing", "position": walked})
        if entry.kind is not None or entry.pool not in covered:
            continue
        ids = interned.get((entry.pool, entry.tags))
        if ids is None:
            if len(interned) >= INTERNED:
                interned.clear()
            ids = []
            for tag in groups_of(entry.tags, tags[entry.pool]):
                key = (entry.pool, tag)
                if key not in group_ids:
                    group_ids[key] = len(group_key)
                    group_key.append(key)
                ids.append(group_ids[key])
            interned[entry.pool, entry.tags] = ids
        name, dash, digits = entry.id.rpartition("-")
        if dash and digits.isascii() and digits.isdigit() and (digits == "0" or digits[0] != "0") \
                and int(digits) < BIG_NUMBER:
            held = interned_who.get((entry.source, name))
            if held is None:
                held = interned_who[entry.source, name] = len(whos)
                whos.append((entry.source, f"{name}-"))
            who.append(held)
            number.append(int(digits))
        else:
            who.append(-1 - len(odd))
            odd.append((entry.source, entry.id))
            number.append(0)
        share.append(entry.pool)
        revision.append(entry.revision)
        piece.append(-1 if entry.piece is None else entry.piece)
        size.append(entry.bytes)
        counts.append(len(ids))
        pairs.extend(ids)
    per_pool = {}
    for pool, _ in group_key:
        per_pool[pool] = per_pool.get(pool, 0) + 1
    many = sorted(pool for pool, groups in per_pool.items() if groups >= 2)
    if not many:
        return None
    found = _Entries()
    found.whos, found.odd = whos, odd
    mask = np.isin(np.frombuffer(share, np.int16), many)
    whole = bool(mask.all())  # every entry kept: the arrays are read where the walk wrote them, never copied

    def kept(values: array.array, kind) -> np.ndarray:
        found = np.frombuffer(values, kind)
        return found if whole else found[mask]

    per_entry = kept(counts, np.uint16)
    pair_mask = None if whole else np.repeat(mask, np.frombuffer(counts, np.uint16))
    found.who, found.number = kept(who, np.int32), kept(number, np.uint32)
    found.revision, found.piece, found.bytes = kept(revision, np.uint32), kept(piece, np.int32), kept(size, np.uint32)
    del share, mask, who, number, revision, piece, size
    entries = len(per_entry)
    total = int(per_entry.sum(dtype=np.int64))
    found.offsets = np.zeros(entries + 1, np.uint32 if total < BIG_NUMBER else np.int64)
    np.cumsum(per_entry, out=found.offsets[1:])
    del per_entry, counts
    # the groups of the shares kept, numbered again from 0 in the order they were first seen
    renumbered = np.full(len(group_key), -1, np.int32)
    found.pool, found.tag = [], []
    for old, (pool, tag) in enumerate(group_key):
        if pool in many:
            renumbered[old] = len(found.pool)
            found.pool.append(pool)
            found.tag.append(tag)
    pairs_read = np.frombuffer(pairs, np.int32)
    found.groups = renumbered[pairs_read if pair_mask is None else pairs_read[pair_mask]]
    del pairs_read, pairs, pair_mask
    order = np.argsort(found.groups, kind="stable")  # each group's entries stay in the order
    found.queue = np.empty(total, np.int32 if entries < (1 << 31) else np.int64)
    for start in range(0, total, QUEUED):  # each pair's entry, from its place among the pairs
        found.queue[start:start + QUEUED] = np.searchsorted(found.offsets, order[start:start + QUEUED], "right") - 1
    del order
    found.starts = np.concatenate([[0], np.cumsum(np.bincount(found.groups, minlength=len(found.pool)))]
                                  ).astype(np.int64)
    found.entries = entries
    progress({"stage": "choosing", "position": walked, "entries": entries})
    return found


def _choose(entries: _Entries, budget: dict, addresses, count, progress) -> Plan:
    """The texts each share of entries chooses, as the module says, one
    share after the other."""
    decided = np.zeros(entries.entries, np.int8)
    tokens = None if count is None else np.full(entries.entries, -1, np.int32)
    done, queue, offsets, groups = memoryview(decided), memoryview(entries.queue), memoryview(entries.offsets), \
        memoryview(entries.groups)
    sizes = memoryview(entries.bytes if tokens is None else tokens)
    starts = entries.starts.tolist()
    ends = starts[1:]
    look = [LOOKAHEAD]
    asked = {}  # each group: the place in its queue up to which tokens were asked for

    def describe(entry: int) -> tuple:
        held = int(entries.who[entry])
        if held >= 0:
            source, name = entries.whos[held]
            text = f"{name}{int(entries.number[entry])}"
        else:
            source, text = entries.odd[-1 - held]
        number = int(entries.piece[entry])
        return addresses[source], text, int(entries.revision[entry]), None if number < 0 else number

    def ask(entry: int, active: set, head: dict):
        """The tokens of entry, and of the next texts of the groups active, worked out by count: the
        groups name one text each in turn, at most look of them each and ASKED_BYTES of texts in all,
        so a request never holds more texts than the workers can tokenize at once, whatever the
        number of groups and the length of their texts."""
        wanted, seen, room = [entry], {entry}, ASKED_BYTES
        places = {group: max(head[group], asked.get(group, 0)) for group in sorted(active)}
        left = dict.fromkeys(places, look[0])
        while room > 0 and left:
            for group in list(left):
                place, end = places[group], ends[group]
                while place < end and (done[queue[place]] or sizes[queue[place]] >= 0 or queue[place] in seen):
                    place += 1
                if place == end or room <= 0:
                    del left[group]
                else:
                    one = queue[place]
                    wanted.append(one)
                    seen.add(one)
                    room -= int(entries.bytes[one])
                    place += 1
                    left[group] -= 1
                    if not left[group]:
                        del left[group]
                places[group] = place
        asked.update(places)
        for one, found in zip(wanted, count([describe(one) for one in wanted]), strict=True):
            sizes[one] = found
        look[0] = min(look[0] * 2, LOOKAHEAD_MOST)

    last, decisions = {}, 0
    for pool in sorted(set(entries.pool)):
        limit = budget[pool]
        floor = limit // FULL
        members = [group for group, owner in enumerate(entries.pool) if owner == pool]
        used = dict.fromkeys(members, 0)
        run = dict.fromkeys(members, 0)
        head = {group: starts[group] for group in members}
        active = set(members)
        heap = [(0, queue[starts[group]], group) for group in members]
        heapq.heapify(heap)
        total, chosen = 0, -1
        while heap and limit - total > floor:
            had, first, group = heapq.heappop(heap)
            if group not in active or had != used[group]:
                continue
            place, end = head[group], ends[group]
            while place < end and done[queue[place]]:
                place += 1
            head[group] = place
            if place == end:
                active.discard(group)
                continue
            entry = queue[place]
            if entry != first:  # its head was decided through another group: it waits with its next text
                heapq.heappush(heap, (had, entry, group))
                continue
            if tokens is not None and sizes[entry] < 0:
                ask(entry, active, head)
            size = sizes[entry]
            decisions += 1
            if decisions % MEASURED == 0:
                progress({"stage": "choosing", "decided": decisions})
            if total + size <= limit:
                done[entry] = TAKEN
                total += size
                chosen = max(chosen, entry)
                run[group] = 0
                for other in groups[offsets[entry]:offsets[entry + 1]]:
                    used[other] += size
                    if size and other in active:
                        heapq.heappush(heap, (used[other], queue[head[other]], other))
                if not size:
                    heapq.heappush(heap, (had, entry, group))
            else:
                done[entry] = LEFT_OUT
                run[group] += 1
                if run[group] >= SKIPS:
                    active.discard(group)
                    continue
                heapq.heappush(heap, (had, entry, group))
        last[pool] = chosen
    return Plan(last, np.flatnonzero(decided == TAKEN).astype(np.int64))
