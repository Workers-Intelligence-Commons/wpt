"""One job: a request run from the depot's snapshot to a delivery on the
dock, through one preparation (docs/api/freight.md).

A job reads the depot only through its api module, by the depot's
address (docs/api/depot.md): the snapshot's row, its listing, each text's
content and credits, where a quote matches, the licenses and notices, and
the erasures carried out. It reads the inputs the request names (earlier
deliveries) from the dock, starts the preparation's program
(freight.exchange), and hands it the items it chooses, in the order of the
snapshot's listing. It knows nothing of what the program makes of them:
only their weights, in the program's unit, and the files it says it wrote.

The program is started once to learn its version (ready), which the
delivery key covers (freight.selection.delivery_key); a delivery of that
key the dock holds, fresh, is answered at once. Otherwise the program is
started again with the folder it writes into, and the job walks the
snapshot's listing under the selection (freight.selection.walk), reading
only the texts it reaches:

- a plain text counts for the first share whose tag it has; within a share
  that splits its budget over the tags its texts carry (freight.groups), a
  first pass over the whole listing chooses its texts before the walk;
  every other limit fills as freight.fill.Fill says;
- with a size, the items (every kind but plain) first take their natural
  share of it: the share of the bytes of the delivery without a size that
  the items hold, filled by their weights in the order, as one pool; each
  share then fills its proportion of what the items left, by weight. So a
  text whose pool is open is weighed (`weigh`) before it is taken;
- a size in bytes (BYTES, a vocabulary's sample) is the freight's own
  measure, whatever the program's unit: the same rule, every weight the
  bytes the listing gives the entry (a piece's: its share of its text's),
  and nothing weighed;
- without a size, every item is taken, and the plain texts go in order:
  with shares, the scarcest share (the one with the fewest bytes for its
  proportion) takes every text, each other share its proportion of the
  scarcest's bytes; with neither, every plain text;
- a conversation is taken only once the walk knows every plain text it
  took, and only when each passage its excerpts quote lies in a plain text
  taken, in pieces taken for a text taken as pieces, and its selector
  matches exactly one place (`quoted text left out`, `quote unmatched` in
  left_out otherwise). The walk keeps the conversations it passes in the
  work folder (named.sqlite) and takes them once it ends, HELD_PAGE at a
  time.

Each item taken is sent to the program (`take`), with its content; the
program's answers come in order, each with its weight and where the item
lies in its files (`at`), which its row copies. As the program says a file
is complete, the job uploads it under files/ of the delivery's folder and
deletes the local copy; once the files it said are uploaded, it uploads
the part of the items table that lists the items they hold and keeps the
state that resumes it (checkpoint): a job started again with that state
and the same work folder goes on from the item after the last one the files
hold, the program told which files it wrote (`resume`), and gives the same
delivery as a job that never stopped. A state of another key, or from
before an erasure that reached the snapshot, starts over.

The delivery note (freight.deliveries) is written last: {key, id (covers
the key and every file's content: a file by the hash its program gives,
else its sha256; each part of the items table by a hash of its rows; the
licenses and notices), depot (the address the request names), request (as
given), snapshot {id, name, owner, change, digest}, preparation {name,
version, unit, settings, inputs}, result (what the program answered),
files (each {path, key, sha256, bytes} and said, its file message),
tables {items: [parts], licenses, notices}, items {texts (a text taken as
pieces counting once), pieces, kinds: {kind: items taken}}, shares (with
shares: [{tag, weight, groups: [{tag, texts, weight}]}]; a text counts for
each of its groups, a text with no other tag for the group null), left_out,
erasures (those the depot had carried out that reached a text of the
snapshot), on_erasure, made_at, code_commit, rules}. Each row of the items table is a
text taken, with its revision, its credits, its tags as of the snapshot,
its scores, `piece` (null for a whole text), `quotes` (a conversation's
quoted texts), `bytes` (what the listing gives it, which a size in bytes
counts), the fields of its `at`, and its `weight`.

An erasure carried out while the job ran is checked before the note is
written and again after: when it reached an item taken, the note is
written stale, or with the erasure in it when the program answered
`on_erasure` "note" (freight.deliveries.reached).

    Job(request, source, dock, work, ...).run()   the delivery note
"""

import json
import os
import shutil
import sqlite3
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from wpt_depot import api
from wpt_store.names import File
from wpt_utils.lines import say

from . import deliveries, groups
from .exchange import Preparation, PreparationFailed
from .fill import Fill
from .selection import (BYTES, CONVERSATION, PLAIN, RULES, Refused, canonical, census, delivery_key, excerpts,
                        fingerprint, items_of, proportions, scarcest_trim, tags_of, walk)

HELD_PAGE = 1000  # conversations the job takes from its work folder at a time once the walk ends
LEFT_OUT, UNMATCHED = "quoted text left out", "quote unmatched"  # why a conversation is left out
BATCH_BYTES = 8 << 20  # bytes of texts read and sent to the program at a time
BATCH_TEXTS = 4096  # texts one batch holds at most
SHARE = 0.75  # the share of the cores the program may use
NAMES = pa.list_(pa.string())
SCORES = pa.list_(pa.struct([("name", pa.string()), ("value", pa.float64()), ("content_version", pa.int64())]))
ITEM_FIELDS = [pa.field("depot", pa.string(), False), pa.field("id", pa.string(), False),
               pa.field("revision", pa.int64(), False), pa.field("kind", pa.string(), False),
               pa.field("title", pa.string()), pa.field("authors", NAMES), pa.field("address", pa.string()),
               pa.field("licenses", NAMES, False), pa.field("tags", NAMES, False), pa.field("notice", pa.string()),
               pa.field("scores", SCORES), pa.field("piece", pa.int64()), pa.field("quotes", NAMES),
               pa.field("bytes", pa.int64(), False)]
TABLES = {"licenses": pa.schema([pa.field("id", pa.string(), False), pa.field("name", pa.string(), False),
                                 pa.field("text", pa.string(), False)]),
          "notices": pa.schema([pa.field("id", pa.string(), False), pa.field("text", pa.string(), False)])}
REBUILT = "made again over its stale note"  # why the files of a note a job replaces are deleted


def items_schema(at: list[str]) -> pa.Schema:
    """The schema of a part of the items table whose rows' `at` hold those fields."""
    return pa.schema([*ITEM_FIELDS, *(pa.field(name, pa.int64()) for name in at), pa.field("weight", pa.int64())])


def threads() -> int:
    """How many threads the program may use: SHARE of the cores."""
    return max(1, int((os.cpu_count() or 1) * SHARE))


def inputs_of(dock, keys: dict[str, str], folder: Path) -> dict[str, str]:
    """{name: the local folder of the delivery of that key's files}, each
    fetched from the dock into folder/<name>. Refused for a delivery the
    dock does not hold fresh."""
    found = {}
    for name, key in sorted(keys.items()):
        note = deliveries.of_key(dock, key)
        if note is None or "deleted" in note:
            raise Refused(f"the input {name} names the delivery {key}, which the dock does not hold")
        if "stale" in note:
            raise Refused(f"the input {name} names the delivery {key}, which is stale: an erasure reached it")
        target = folder / name
        if target.exists():
            shutil.rmtree(target)
        found[name] = str(deliveries.fetch(dock, note, target))
    return found


def ready_of(name: str, settings: dict, inputs: dict[str, str], folder: Path, log=say) -> dict:
    """The answer of the program name to start (ready), the program stopped right after."""
    with Preparation(name, settings, inputs, folder, 1, log=log) as program:
        return program.ready


class Job:
    """One request run, as the module says. source: the address of the
    depot the job reads; dock: the dock it delivers on; work: its local
    folder; checkpoint: the state a former run of the same request handed
    save; progress: called with {stage, ...} as it goes, and may raise to
    stop the job; cap: the dock's cap in bytes (None: none); code_commit:
    the code's commit, recorded only."""

    def __init__(self, request: dict, source: str, dock, work: Path, *, checkpoint: dict | None = None,
                 save=lambda state: None, progress=lambda state: None, cap: int | None = None,
                 code_commit: str | None = None, log=say):
        self.request, self.source, self.dock, self.work = request, source, dock, Path(work)
        self.state, self.save, self.progress, self.log = checkpoint or {}, save, progress, log
        self.cap, self.code_commit = cap, code_commit
        self.selection = canonical(request.get("selection") or {})
        self.preparation = request.get("preparation") or {}
        if not isinstance(self.preparation.get("name"), str):
            raise Refused(f"a request names its preparation as {{name, settings, inputs}}: {self.preparation!r}")
        self.settings = self.preparation.get("settings") or {}
        self.input_keys = self.preparation.get("inputs") or {}

    def snapshot_row(self) -> dict:
        """The snapshot the request names, once its change and digest are the request's."""
        named = self.request.get("snapshot") or {}
        if not isinstance(named.get("name"), str):
            raise Refused(f"a request names its snapshot as {{name, change, digest}}: {named!r}")
        row = api.snapshot(self.source, name=named["name"], owner=named.get("owner"))
        if named.get("change") != row["at_change"] or named.get("digest") != row["digest"]:
            raise Refused(f"snapshot {row['name']} of depot {row['owner']} reads change {row['at_change']} with digest "
                          f"{row['digest']}; the request says {named.get('change')}, {named.get('digest')}")
        return row

    def plan(self) -> str:
        """Read the snapshot, fetch the inputs and ask the program its
        version: the delivery key."""
        self.snapshot = self.snapshot_row()
        self.work.mkdir(parents=True, exist_ok=True)
        self.inputs = inputs_of(self.dock, self.input_keys, self.work / "inputs")
        name = self.preparation["name"]
        self.ready = ready_of(name, self.settings, self.inputs, self.work / "probe", self.log)
        self.version, self.unit = str(self.ready.get("version")), self.ready.get("unit")
        self.on_erasure = self.ready.get("on_erasure", deliveries.STALE)
        if self.on_erasure not in deliveries.ON_ERASURE:
            raise PreparationFailed(f"{name} answered on_erasure {self.on_erasure!r}, not one of "
                                    f"{', '.join(deliveries.ON_ERASURE)}")
        self.prepared = tuple(self.ready.get("kinds") or ())
        self.key = delivery_key({"change": self.snapshot["at_change"], "digest": self.snapshot["digest"]},
                                self.selection, {"name": name, "version": self.version, "settings": self.settings,
                                                 "inputs": self.input_keys})
        self.folder = deliveries.folder_of(self.key)
        return self.key

    def run(self) -> dict:
        """The delivery note of the request: the one on the dock when it is fresh, else the one this job makes."""
        self.plan()
        ready, name = self.ready, self.preparation["name"]
        known = deliveries.of_key(self.dock, self.key)
        if known is not None and deliveries.fresh(known):
            self.log(f"delivery {known['key']} of snapshot {self.snapshot['name']} is on the dock already")
            return known
        self.replaced = known
        size = self.selection["size"]
        self.limit = None if size is None else size["amount"]
        self.by_bytes = size is not None and size["unit"] == BYTES  # the freight's own measure: nothing to weigh
        if size is not None and not self.by_bytes and size["unit"] != self.unit:
            raise Refused(f"the size is in {size['unit']}, and {name} weighs in {self.unit or 'nothing'}: give it "
                          f"in {self.unit or BYTES}" + (f" or in {BYTES}" if self.unit else ""))
        if size is not None and not self.by_bytes and not ready.get("weighs"):
            raise Refused(f"{name} weighs nothing, so a size in {size['unit']} cannot be met")
        self.shares = self.selection["shares"] or None
        self.kinds = self.selection["kinds"] or None
        self.holder = self.snapshot["owner"]  # the depot's name every row carries
        self.census = census(self.snapshot["summary"], self.shares, self.kinds, self.prepared)
        self.room = deliveries.Room(self.dock, self.cap, self.folder, self.log, inputs=self.input_keys.values())
        self.erasures = _erasures(self.source, self.snapshot["id"])
        named = sqlite3.connect(self.work / "named.sqlite", isolation_level=None)
        try:
            named.execute("CREATE TABLE IF NOT EXISTS named (kind TEXT NOT NULL, id TEXT NOT NULL, "
                          "PRIMARY KEY (kind, id)) WITHOUT ROWID")
            # the conversations the walk passed, by their place in it, and the plain texts it took (piece -1: whole)
            named.execute("CREATE TABLE IF NOT EXISTS held (walked INTEGER NOT NULL PRIMARY KEY, id TEXT NOT NULL, "
                          "revision INTEGER NOT NULL, bytes INTEGER NOT NULL, tags TEXT NOT NULL)")
            named.execute("CREATE TABLE IF NOT EXISTS taken (id TEXT NOT NULL, piece INTEGER NOT NULL, "
                          "revision INTEGER NOT NULL, PRIMARY KEY (id, piece)) WITHOUT ROWID")
            return self._stream(named)
        finally:
            named.close()

    # The stream

    def _fresh_state(self) -> dict:
        return {"key": self.key, "erasures": self.erasures, "after": None, "walked": 0, "limited": None,
                "files": [], "said": [], "parts": [], "ref": 0, "last_ref": None, "pools": {}, "fill": None,
                "taken": 0, "whole": 0, "pieces": 0, "kinds": {}, "plan": None, "seen": 0, "covered": {},
                "groups": {}, "resolved": 0, "unquoted": {LEFT_OUT: 0, UNMATCHED: 0},
                "read": {"plain": 0, "items": 0}, "result": None}

    def _stream(self, named_file) -> dict:
        state = self.state
        if state.get("key") == self.key and (gone := deliveries.missing(self.dock, self.folder, _uploaded(state))):
            self.log(f"{len(gone)} files this job uploaded are no longer on the dock: it starts over")
            state = {}
        if state.get("key") == self.key and state.get("erasures") != self.erasures:
            self.log("an erasure reached a text of the snapshot since this job stopped: it starts over")
            state = {}
        out = self.work / "out"
        if state.get("key") != self.key:  # a state of another request starts nothing
            state = self._fresh_state()
            for table in ("named", "held", "taken"):
                named_file.execute(f"DELETE FROM {table}")
            if out.exists():
                shutil.rmtree(out)
        self.state = state
        shares, limit = self.shares, self.limit
        if state.get("result") is not None:  # the program was done before this job stopped: only the note is left
            return self._finish(state, state["result"], named_file)
        resume = {"files": state["said"], "after": state["last_ref"]} if state["said"] or state["ref"] else None
        with ExitStack() as stack:
            program = stack.enter_context(Preparation(self.preparation["name"], self.settings, self.inputs, out,
                                                      threads(), resume, self.log))
            if str(program.ready.get("version")) != self.version:
                raise PreparationFailed(f"{self.preparation['name']} answered version {program.ready.get('version')} "
                                        f"and then {self.version}")
            self.program = program
            if state["limited"] is None:  # with a size, the items take their share of it first (_limit_items)
                state["limited"] = self._limit_items(shares) if limit is not None else {
                    "weight": None, "texts": self.census["items"], "bytes": self.census["item_bytes"],
                    "last": None, "skipped": []}
            limited = state["limited"]
            last_item = None if limited["last"] is None else tuple(limited["last"])
            skipped_items = {tuple(one) for one in limited["skipped"]}

            def item_taken(entry) -> bool:
                """Whether the job takes the item: every one without a size, else those _limit_items took."""
                place = tuple(entry.place())
                return limit is None or (last_item is not None and place <= last_item and place not in skipped_items)

            if limit is not None:  # the items took their share; the shares split what is left
                budget = proportions(shares, limit - limited["weight"])
                counted = "bytes" if self.by_bytes else "weights"
            elif shares is not None:
                budget, counted = self._byte_budget(shares), "bytes"
            else:
                budget, counted = None, None
            self.budget, self.counted = budget, counted
            chosen = self._plan(state, budget, counted, shares)
            fill = None if budget is None else Fill({pool: most for pool, most in budget.items()
                                                     if pool not in chosen.last}, state["fill"])
            self.fill = fill
            tags = [""] if shares is None else [one["tag"] for one in shares]
            self.group_counts = {share: {tag: [texts, weight] for tag, texts, weight in found}
                                 for share, found in state["groups"].items()}
            self.named_groups = {}
            self.rows, self.named, self.row_bytes = [], set(), 0
            self.held_rows, self.taken_rows = [], []
            self.keep_taken = CONVERSATION in items_of(self.prepared, self.kinds) and self.census["items"] > 0
            self.named_file = named_file
            ahead = None if fill is None else fill.copy()  # the decisions, made ahead of the program's answers

            def batches():
                """The texts the job takes from the state's place on, in batches of [entry, read, walked, place]:
                read is whether it is taken (None: once weighed), which a closed pool, a limit that leaves it out,
                or a share's choice by group saves; walked how many texts the walk passed up to it; place the
                entry's place among the plan's entries, None for an entry of a share the plan does not cover.
                With a size, an item _limit_items did not take is passed, and the batches end once every pool is
                closed, every share the plan covers is past its last chosen text, and no item taken follows."""
                picker = chosen.picker(state["seen"])
                batch, size, walked = [], 0, state["walked"]
                for entry in walk(self.source, self.snapshot["id"], tags_of(shares), self.kinds, self.prepared,
                                  state["after"]):
                    walked += 1
                    if limit is not None and ahead.done() and picker.done() and (
                            last_item is None or tuple(entry.place()) > last_item):
                        break
                    if not entry.taken or (entry.kind != PLAIN and not item_taken(entry)):
                        continue
                    place = None
                    if entry.kind != PLAIN or counted is None:
                        read = True
                    elif picker.covers(entry.pool):
                        place, read = picker.pick(entry.pool)
                    elif counted == "bytes":
                        read = ahead.take(entry.pool, entry.bytes)
                    else:
                        read = None if entry.pool not in ahead.closed else False
                    batch.append([entry, read, walked, place])
                    size += entry.bytes if read is not False else 0
                    if size >= BATCH_BYTES or len(batch) >= BATCH_TEXTS:
                        yield batch
                        batch, size = [], 0
                if batch:
                    yield batch

            for batch in batches():
                self._batch(batch, ahead, tags)
                self.progress({"stage": "reading", "position": state["walked"],
                               "texts": state["taken"], "weight": sum(state["pools"].values()),
                               "files": len(state["files"])})
            self._flush_named()
            last = state["resolved"]
            while page := list(named_file.execute("SELECT walked, id, revision, bytes, tags FROM held "
                                                  "WHERE walked > ? ORDER BY walked LIMIT ?", (last, HELD_PAGE))):
                self._conversations(page)
                last = state["resolved"]
                self.progress({"stage": "taking conversations", "conversations": state["kinds"].get(CONVERSATION, 0),
                               "left out": sum(state["unquoted"].values())})
            state["result"] = program.end()
            self._checkpoint(final=True)
        return self._finish(state, state["result"], named_file)

    def _batch(self, batch: list[list], ahead: Fill | None, tags: list[str]):
        """Decide, send and take one batch of the walk, in order."""
        state = self.state
        wanted = [one for one in batch if one[1] is not False and one[0].kind != CONVERSATION]
        contents = dict(zip((id(one) for one in wanted), api.contents(
            self.source, [(one[0].id, one[0].revision, one[0].piece) for one in wanted])))
        weighed = [one for one in wanted if one[1] is None]
        weights = {}
        if weighed:
            found = self.program.weigh([(PLAIN, contents[id(one)]) for one in weighed], self._refs(len(weighed)))
            weights = {id(one): weight for one, weight in zip(weighed, found)}
            for one in weighed:  # in the order: the same decisions the fill makes once the answers come
                one[1] = ahead.take(one[0].pool, weights[id(one)])
        sent = [one for one in batch if one[1] and one[0].kind != CONVERSATION]
        credits = dict(zip((id(one) for one in sent), api.credits(
            self.source, [(one[0].id, one[0].revision) for one in sent]))) if sent else {}
        refs = {}
        for one in sent:
            entry = one[0]
            refs[id(one)] = ref = self._refs(1)
            self.program.take({"ref": ref, "id": entry.id, "name": entry.name, "depot": self.holder,
                               "piece": entry.piece, "kind": entry.kind,
                               "pool": tags[entry.pool] if entry.kind == PLAIN else ""}, contents[id(one)])
        for one in batch:
            entry, read, walked, place = one
            if entry.kind == CONVERSATION:  # taken once the walk knows every plain text taken (_conversations)
                self.held_rows.append((walked, entry.id, entry.revision, entry.bytes, entry.tags))
                self._passed(entry, walked)
                continue
            answer = self._taken(refs[id(one)]) if read else None
            weight = answer["weight"] if answer is not None else weights.get(id(one))
            if answer is not None and id(one) in weights and weights[id(one)] != weight:
                raise PreparationFailed(f"{self.preparation['name']} weighed {entry.id} {weights[id(one)]} and then "
                                        f"took it at {weight}: a weight depends on the item alone")
            if place is not None:  # a share that chose its texts by group: the chosen ones, within its budget
                state["seen"] = place + 1
                if read:
                    used = state["covered"].get(str(entry.pool), 0) + (
                        weight if self.counted == "weights" else entry.bytes)
                    if used > self.budget[entry.pool]:
                        raise AssertionError(f"the share {tags[entry.pool]} chose {entry.id}, which passes its budget")
                    state["covered"][str(entry.pool)] = used
            elif entry.kind == PLAIN and (self.counted == "bytes" or (self.counted == "weights" and weight is not None)):
                # the same take the decision made ahead (a text whose pool was closed then was never weighed)
                took = self.fill.take(entry.pool, weight if self.counted == "weights" else entry.bytes)
                if took != bool(read):
                    raise AssertionError(f"the fill left {entry.id} out on one pass and not the other")
            if read:
                self._took(entry, answer, credits[id(one)], tags)
            self._passed(entry, walked)

    def _refs(self, count: int) -> int:
        """The first of count new refs."""
        first = self.state["ref"]
        self.state["ref"] += count
        return first

    def _passed(self, entry, walked: int):
        self.state["after"], self.state["walked"] = entry.place(), walked

    def _taken(self, ref: int) -> dict:
        """The program's answer to the take of ref, once the files it said before it are kept (_checkpoint)."""
        answer = self.program.taken()
        if self.program.files:  # said before this answer: they hold the items answered before it
            self._checkpoint(before=answer)
        if answer["ref"] != ref:
            raise PreparationFailed(f"{self.preparation['name']} answered the take of {answer['ref']} where "
                                    f"{ref} was due")
        return answer

    def _took(self, entry, answer: dict, credit: dict, tags: list[str], quotes: list[str] | None = None):
        """Count one item taken and add its row."""
        state = self.state
        weight = answer["weight"]
        pool_tag = tags[entry.pool] if entry.kind == PLAIN else ""
        if entry.kind == PLAIN:
            state["pools"][pool_tag] = state["pools"].get(pool_tag, 0) + weight
            if self.shares is not None:
                held = self.named_groups.get((pool_tag, entry.tags))
                if held is None:
                    if len(self.named_groups) >= groups.INTERNED:
                        self.named_groups.clear()
                    held = self.named_groups[pool_tag, entry.tags] = groups.groups_of(entry.tags, pool_tag)
                into = self.group_counts.setdefault(pool_tag, {})
                for tag in held:
                    counted_so = into.setdefault(tag, [0, 0])
                    counted_so[0] += 1
                    counted_so[1] += weight
            if entry.piece is None:
                state["whole"] += 1
            else:
                state["pieces"] += 1
                self.named.add(("long text", entry.id))
            if self.keep_taken:
                self.taken_rows.append((entry.id, -1 if entry.piece is None else entry.piece, entry.revision))
        state["kinds"][entry.kind] = state["kinds"].get(entry.kind, 0) + 1
        state["taken"] += 1
        state["last_ref"] = answer["ref"]
        row = {"depot": self.holder, "id": entry.id, "revision": entry.revision, **credit,
               "tags": json.loads(entry.tags), "piece": entry.piece, "bytes": entry.bytes, **(answer.get("at") or {}),
               "weight": weight}
        if quotes is not None:
            row["quotes"] = quotes
        self.rows.append(row)
        self.row_bytes += entry.bytes
        state["read"]["plain" if entry.kind == PLAIN else "items"] += entry.bytes
        self.named.update(("license", one) for one in credit["licenses"])
        if credit["notice"] is not None:
            self.named.add(("notice", credit["notice"]))

    def _conversations(self, page: list[tuple]):
        """Take the conversations of a page of the work folder's table held,
        in order: each only when every passage its excerpts quote lies in a
        plain text taken (of a text taken as pieces, in pieces taken) and its
        selector matches exactly one place of the content version taken;
        else left out, for its first excerpt that fails: LEFT_OUT, or
        UNMATCHED for a selector that matches no place or several. Each
        quoted content version is read once for all the selectors on it."""
        state = self.state
        raw = list(api.contents(self.source, [(text, revision, None) for _, text, revision, _, _ in page]))
        read = [json.loads(one) for one in raw]
        quoted = [[(block["source"], block["selector"]) for block in excerpts(one)] for one in read]
        taken = {}  # id: its rows of taken, (piece, revision)
        asked = {}  # (id, revision, whether taken as pieces): {the selector as JSON: its place in asked}
        for found in quoted:
            for source, selector in found:
                if source not in taken:
                    taken[source] = list(self.named_file.execute(
                        "SELECT piece, revision FROM taken WHERE id = ?", (source,)))
                rows = taken[source]
                if rows:
                    on = asked.setdefault((source, rows[0][1], rows[0][0] != -1), {})
                    on.setdefault(json.dumps(selector, sort_keys=True), len(on))
        lying = {}
        for (source, revision, pieced), selectors in asked.items():
            matched = api.matched(self.source, source, revision, [json.loads(one) for one in selectors], pieced)
            for selector, at in selectors.items():
                lying[(source, revision, pieced), selector] = matched[at]
        kept = []
        for (walked, text, revision, size, tags), found in zip(page, quoted):
            why = None
            for source, selector in found:
                rows = taken[source]
                if not rows:
                    why = LEFT_OUT
                    break
                pieced = rows[0][0] != -1
                held = lying[(source, rows[0][1], pieced), json.dumps(selector, sort_keys=True)]
                if held is None:
                    why = UNMATCHED
                    break
                if pieced and not set(held) <= {piece for piece, _ in rows}:
                    why = LEFT_OUT
                    break
            kept.append(why)
        sent = [at for at, why in enumerate(kept) if why is None]
        credits = dict(zip(sent, api.credits(self.source, [(page[at][1], page[at][2]) for at in sent]))) if sent else {}
        refs = {}
        for at in sent:
            walked, text, revision, size, tags = page[at]
            content = raw[at]
            refs[at] = ref = self._refs(1)
            self.program.take({"ref": ref, "id": text, "name": text, "depot": self.holder, "piece": None,
                               "kind": CONVERSATION, "pool": ""}, content)
        for at, ((walked, text, revision, size, tags), why) in enumerate(zip(page, kept)):
            if why is not None:
                state["unquoted"][why] += 1
            else:
                answer = self._taken(refs[at])
                entry = _Held(text, revision, size, tags)
                self._took(entry, answer, credits[at], [""],
                           quotes=list(dict.fromkeys(source for source, _ in quoted[at])))
            state["resolved"] = walked

    def _flush_named(self):
        """Keep in the work folder the ids named, the conversations passed and the plain texts taken so far:
        the walk gives them again, the same, after a stop."""
        con = self.named_file
        con.execute("BEGIN")
        con.executemany("INSERT OR IGNORE INTO named (kind, id) VALUES (?, ?)", sorted(self.named))
        con.executemany("INSERT OR IGNORE INTO held (walked, id, revision, bytes, tags) VALUES (?, ?, ?, ?, ?)",
                        self.held_rows)
        con.executemany("INSERT OR IGNORE INTO taken (id, piece, revision) VALUES (?, ?, ?)", self.taken_rows)
        con.execute("COMMIT")
        self.named.clear()
        self.held_rows.clear()
        self.taken_rows.clear()

    def _upload(self, path: Path, key: str) -> dict:
        size = path.stat().st_size
        self.room.make(size)
        file = self.dock.put_named(key, path)
        path.unlink()
        return {**file.reference(), "bytes": size}

    def _checkpoint(self, before: dict | None = None, final: bool = False):
        """Upload the files the program said and the part of the items table
        begun, and hand the state that resumes after them; before a job's
        first upload, the cap's check of its estimated size (Room.expect).
        before: the answer read right after the files, whose item is not
        counted yet."""
        state = self.state
        said, self.program.files = self.program.files, []
        # a file's entry: the dock's reference and size, and the program's file message apart (said), so no field
        # the program names takes the place of the freight's
        ready = [(self.work / "out" / one["path"], f"{self.folder}/files/{one['path']}", state["files"],
                  {"path": one["path"], "said": one}) for one in said]
        if self.rows:
            number = len(state["parts"])
            path = self.work / f"items-{number:05d}.parquet"
            at = sorted({name for row in self.rows for name in row} - {field.name for field in ITEM_FIELDS} - {"weight"})
            pq.write_table(pa.Table.from_pylist(self.rows, schema=items_schema(at)), path, compression="zstd")
            ready.append((path, f"{self.folder}/items/part-{number:05d}.parquet", state["parts"],
                          {"rows": len(self.rows), "text_bytes": self.row_bytes, "fingerprint": fingerprint(self.rows)}))
            self.rows = []
            self.row_bytes = 0
        if ready and not (state["files"] or state["parts"]):
            self.room.expect(self._estimate(sum(path.stat().st_size for path, *_ in ready), final))
        for path, key, kept, added in ready:
            kept.append({**self._upload(path, key), **added})
        state["said"] += said
        self._flush_named()
        state["groups"] = {share: [[tag, *counted_so] for tag, counted_so in found.items()]
                           for share, found in self.group_counts.items()}
        state["fill"] = None if self.fill is None else self.fill.state()
        self.save(json.loads(json.dumps(state)))

    def _estimate(self, local: int, final: bool) -> int:
        """The delivery's whole size in bytes, from the local files of its first upload: those files when the
        job ends with them, else their size times the bytes the delivery may take over the bytes they hold (the
        order is shuffled, so its start stands for the whole)."""
        if final:
            return local
        state, read = self.state, self.state["read"]
        plain_all = sum(self.census["pool_bytes"].values())
        if self.counted == "bytes":
            plain_bytes = min(plain_all, sum(self.census["pool_bytes"].get(pool, 0) if value is None else value
                                             for pool, value in self.budget.items()))
        elif self.counted == "weights":
            weight = sum(state["pools"].values())
            plain_bytes = plain_all if not weight else min(plain_all, int(sum(self.budget.values())
                                                                          * read["plain"] / weight))
        else:
            plain_bytes = plain_all
        return int(local * (plain_bytes + state["limited"]["bytes"]) / max(read["plain"] + read["items"], 1))

    def _weigh(self, described: list[tuple]) -> list[int]:
        """The weights of entries (the depot's address, id, revision, piece) of plain texts, in order."""
        found = list(api.contents(self.source, [(text, revision, piece) for _, text, revision, piece in described]))
        return self.program.weigh([(PLAIN, one) for one in found], self._refs(len(found)))

    def _plan(self, state: dict, budget: dict | None, counted: str | None, shares: list[dict] | None
              ) -> groups.Plan:
        """The texts the shares choose by group (freight.groups), sizes counted as the budget counts them
        (weights or bytes): read again from the work folder when the state names them and the file still holds
        them, else chosen again, the same for the same depot state, and kept there, its sha256 in the state."""
        if shares is None or budget is None:
            return groups.Plan({}, np.zeros(0, np.int64))
        path = self.work / "chosen.bin"
        known = state.get("plan")
        if known is not None:
            found = groups.Plan.load(path, {int(pool): last for pool, last in known["last"].items()}, known["sha256"])
            if found is not None:
                return found
        self.progress({"stage": "choosing"})
        chosen = groups.plan(walk(self.source, self.snapshot["id"], tags_of(shares), self.kinds, self.prepared),
                             tags_of(shares), budget, self._weigh if counted == "weights" else None, self.progress,
                             self.source)
        sha256 = chosen.save(path)
        if known is not None and known["sha256"] != sha256:
            raise AssertionError("the shares chose other texts than this job chose before it stopped")
        state["plan"] = {"sha256": sha256, "last": {str(pool): last for pool, last in chosen.last.items()}}
        if chosen.last:
            self.log(f"the shares {', '.join(shares[pool]['tag'] for pool in sorted(chosen.last))} split their "
                     f"budgets over the tags of their texts, by the square root of what each holds: "
                     f"{len(chosen.chosen):,} texts chosen")
        return chosen

    def _limit_items(self, shares: list[dict] | None) -> dict:
        """The items a job with a size takes: their share of it (the items'
        part of the bytes of the delivery without a size, from the census),
        filled by their weights in the order (their bytes for a size in
        bytes), as one pool (Fill). {weight,
        texts, bytes: of the items taken, last: the place of the last item
        taken (None: none), skipped: the places of the items before it left
        out}; the walk ends once the pool closes, or at the end of the
        snapshot."""
        item_bytes = self.census["item_bytes"]
        found = {"weight": 0, "texts": 0, "bytes": 0, "last": None, "skipped": []}
        if not self.census["items"]:
            return found
        plain = sum((self.census["pool_bytes"] if not shares else self._byte_budget(shares, False)).values())
        part = self.limit * item_bytes // (item_bytes + plain) if item_bytes else 0
        fill = Fill({0: part})

        def batches():
            batch, size = [], 0
            for entry in walk(self.source, self.snapshot["id"], tags_of(shares), self.kinds, self.prepared):
                if not entry.taken or entry.kind == PLAIN:
                    continue
                batch.append(entry)
                size += entry.bytes
                if size >= BATCH_BYTES or len(batch) >= BATCH_TEXTS:
                    yield batch
                    batch, size = [], 0
            if batch:
                yield batch

        done, skipped = 0, []
        for batch in batches():
            if self.by_bytes:
                weights = [entry.bytes for entry in batch]
            else:
                contents = api.contents(self.source, [(entry.id, entry.revision, None) for entry in batch])
                weights = self.program.weigh([(entry.kind, content) for entry, content in zip(batch, contents)],
                                             self._refs(len(batch)))
            for entry, weight in zip(batch, weights):
                if fill.done():
                    break
                if not fill.take(0, weight):
                    skipped.append(entry.place())
                    continue
                found["weight"] += weight
                found["texts"] += 1
                found["bytes"] += entry.bytes
                found["last"] = entry.place()
                found["skipped"] += skipped
                skipped = []
            done += len(batch)
            self.progress({"stage": "weighing items", "items": done, "weight": found["weight"]})
            if fill.done():
                break
        return found

    def _byte_budget(self, shares: list[dict], refuse: bool = True) -> dict[int, int | None]:
        """{pool: the bytes it may take} without a size: the scarcest share
        all of its bytes, each other its proportion of the scarcest's (from
        the census). A share with no text is refused, or with refuse False
        takes nothing and the others are measured among themselves. With
        refuse, a share that may take all its bytes has no limit (None), so
        it takes every text."""
        available = {at: 0 for at in range(len(shares))} | self.census["pool_bytes"]
        empty = [shares[at]["tag"] for at, size in available.items() if not size]
        if empty and refuse:
            raise Refused(f"the shares cannot be met: no plain text of the snapshot {self.snapshot['name']} the "
                          f"request takes has the tag {', '.join(empty)}")
        trimmed = {at: int(size) for at, size in scarcest_trim(available, shares).items()}
        return trimmed if not refuse else {at: None if size >= available[at] else size
                                           for at, size in trimmed.items()}

    # The note

    def _finish(self, state: dict, result: dict, named_file) -> dict:
        self.progress({"stage": "finishing", "texts": state["taken"]})
        named = {kind: [one for (one,) in named_file.execute("SELECT id FROM named WHERE kind = ? ORDER BY id",
                                                             (kind,))] for kind in ("license", "notice")}
        license_rows = api.licenses(self.source, named["license"])
        notice_rows = api.notices(self.source, named["notice"])
        long_taken = next(named_file.execute("SELECT count(*) FROM named WHERE kind = 'long text'"))[0]
        plain_taken = state["whole"] + long_taken  # a text taken as pieces counts once

        def put(table: str, rows: list[dict]) -> dict:
            path = self.work / f"{table}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=TABLES[table]), path, compression="zstd")
            return self._upload(path, f"{self.folder}/{table}.parquet")

        tables = {"items": state["parts"], "licenses": put("licenses", license_rows),
                  "notices": put("notices", notice_rows)}
        delivery_id = fingerprint({"key": self.key,
                                   "files": [[one["path"], one["said"].get("hash") or one["sha256"]]
                                             for one in state["files"]],
                                   "items": [part["fingerprint"] for part in state["parts"]],
                                   "licenses": license_rows, "notices": notice_rows})
        unquoted = sum(state["unquoted"].values())
        items_taken = sum(count for kind, count in state["kinds"].items() if kind != PLAIN)
        left_out = {"no share": self.census["no_share"], "kinds not taken": self._not_taken(),
                    "trimmed": max(0, self.census["plain"] - plain_taken),
                    "trimmed items": max(0, self.census["items"] - items_taken - unquoted), **state["unquoted"]}
        note = {
            "key": self.key, "id": delivery_id, "depot": self.request.get("depot"), "request": self.request,
            "snapshot": {"id": self.snapshot["id"], "name": self.snapshot["name"], "owner": self.holder,
                         "change": self.snapshot["at_change"], "digest": self.snapshot["digest"]},
            "preparation": {"name": self.preparation["name"], "version": self.version, "unit": self.unit,
                            "settings": self.settings, "inputs": self.input_keys},
            "result": result, "files": state["files"], "tables": tables,
            "items": {"texts": state["taken"] - state["pieces"] + long_taken, "pieces": state["pieces"],
                      "kinds": dict(sorted(state["kinds"].items()))},
            "shares": None if self.shares is None else [
                {"tag": share["tag"], "weight": state["pools"].get(share["tag"], 0),
                 "groups": [{"tag": tag, "texts": texts, "weight": weight} for tag, texts, weight in sorted(
                     state["groups"].get(share["tag"], []), key=lambda one: (one[0] is not None, one[0] or ""))]}
                for share in self.shares],
            "left_out": left_out, "erasures": self.erasures, "on_erasure": self.on_erasure,
            "made_at": datetime.now(UTC).isoformat(),
            "code_commit": self.code_commit, "rules": RULES,
        }
        gone = deliveries.missing(self.dock, self.folder, deliveries.files(note))
        if gone:
            raise deliveries.FilesGone(f"{len(gone)} files of this job were deleted while it ran, {gone[0]} first: "
                                       "no note was written; run the request again")
        # an erasure carried out while the job ran may have reached an item it took after the job passed it:
        # checked before the note is written, and again after, for one carried out in between; it makes the
        # delivery stale, or is written into its note (on_erasure)
        now, hit = self._erased_since(self.erasures, note)
        note = deliveries.reached(note, hit)
        note = self.dock.put_json(f"{self.folder}/{deliveries.NOTE}", note)
        _, hit = self._erased_since(now, note)
        if hit:
            note = self.dock.put_json(f"{self.folder}/{deliveries.NOTE}", deliveries.reached(note, hit))
        self.log(f"delivery {note['key']} of snapshot {self.snapshot['name']} ({self.preparation['name']}): "
                 f"{note['items']['texts']:,} texts"
                 + (f" ({state['pieces']:,} pieces of {long_taken:,} of them)" if state["pieces"] else "")
                 + f", {len(state['files'])} files, id {delivery_id}")
        self._drop_replaced(note)
        if not deliveries.fresh(note):
            self.log(f"delivery {note['key']} is stale already: the erasures {', '.join(note['stale']['erasures'])}, "
                     "carried out while it was made, reached an item it holds; run the request again")
        return note

    def _drop_replaced(self, note: dict):
        """Delete the files of the note this job replaced (a stale or deleted one) that the new one does not name."""
        if self.replaced is None:
            return
        kept = {one["key"] for one in deliveries.files(note)}
        for one in deliveries.files(self.replaced):
            if one["key"] not in kept and self.dock.has(File.of(one)):
                self.dock.delete(File.of(one))
                self.log(f"{one['key']} deleted: {REBUILT}")

    def _not_taken(self) -> dict[str, int]:
        """{kind: the texts of the snapshot of that kind the job does not take}: kinds the preparation does not
        prepare and kinds the selection does not name."""
        allowed = self.kinds
        return {kind: one["texts"] for kind, one in self.census["kinds"].items() if one["texts"] and (
            kind not in self.prepared or (allowed is not None and kind not in allowed))}

    def _erased_since(self, before: list[str], note: dict) -> tuple[list[str], list[str]]:
        """(the erasures the depot has carried out now that reached a text of the snapshot, the ids of those not in
        before that reached an item the delivery holds)."""
        now = _erasures(self.source, self.snapshot["id"])
        new = set(now) - set(before)
        hit = [one["id"] for one in api.erasures(self.source)["erasures"] if one["id"] in new
               and deliveries.holds(self.dock, note, set(one["texts"]))]
        return now, sorted(hit)


class _Held:
    """A conversation the walk passed, as _took reads an entry."""

    kind, piece, pool = CONVERSATION, None, None

    def __init__(self, text: str, revision: int, size: int, tags: str):
        self.id, self.revision, self.bytes, self.tags = text, revision, size, tags


def _erasures(source: str, snapshot: str) -> list[str]:
    """The ids of the erasures the depot at the address source has carried
    out that reached a text the snapshot whose id is snapshot holds: an
    erasure of a text it does not hold changes nothing of its delivery."""
    found = api.erasures(source)["erasures"]
    held = set(api.holds(source, snapshot, sorted({text for one in found for text in one["texts"]})))
    return [one["id"] for one in found if any(text in held for text in one["texts"])]


def _uploaded(state: dict) -> list[dict]:
    """The references of the files a job's state says it uploaded."""
    return [*state["files"], *state["parts"]]
