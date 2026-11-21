"""A dataset's texts as files to download, made when asked and never
stored, with no tokenizer and no build: every text of its current
revision a corpus may take, split into numbered parts.

The texts are those of the dataset's libraries, each library as it stood
at the change number of its depot the dataset names, a text two libraries
hold once, at its revision in the first of those libraries the dataset
names; of them, the plain texts a share takes (without shares, every plain
text) and the conversations and tasks, as a corpus takes them
(wpt_depot.order), with no limit: neither max_tokens nor the trimming of
the shares to their proportions applies, so an export is always the whole
dataset. A conversation that quotes texts (wpt_depot.quotes) follows the
rule of a corpus (wpt_depot.corpus): each excerpt's text must be a plain
text the export holds from the same depot, and its selector must match
exactly one place of the revision the export holds; otherwise the export
leaves the conversation out, as LEFT_OUT or UNMATCHED, and counts it in
the part it would be in (Export.part's left_out).

Each text's place is its export key, which its depot stores with it
(wpt_depot.ids.export_key: a signed 64-bit integer made from its id
alone, so it does not depend on where a file is, on the depot's name or
on its public address). The export has parts_for(texts) parts, texts
being how many texts its libraries held at their change numbers
(Depot.library_count), and part n holds the texts whose key falls in the
n-th of that many equal ranges of the keys (bounds), in the order of their
keys. Each library is read as that range of its own rows by key (the
index memberships_key, or pins_key for a library locked then), each row
kept when the library held its text at its change number
(Depot.library_range), and the libraries' reads are merged in key order
as they go: no pass orders the dataset, none reads the texts of the other
parts nor the rows of other libraries, and a part's first text is sent
before its last is read. Every
text is in exactly one part.

A part comes in two forms, each the same bytes every time it is made for
the same export (identity: the dataset's revision and the erasures its
depots carried out), by the same version of what its rows hold (ROWS) and
the same versions of Python's zstd and of pyarrow (MAKER):

- "jsonl.zst": one JSON line per text, {"depot", "id", "revision", "kind",
  "text"} (its kind, wpt_depot.kinds; an item's text is its JSON content;
  depot: the name a corpus records for it, wpt_depot.corpus.depot_name),
  compressed with zstd at
  LEVEL as one frame; a text is read and written a part at a time, so a
  text of any size streams through;
- "parquet": a Parquet file of the same columns (TEXT_SCHEMA), zstd
  compressed, in row groups of at most GROUP_TEXTS texts or about
  GROUP_BYTES bytes of text, each group sent once written; a text is held
  whole while its group is written.

    FORMS                                        the forms, by name
    MAKER                                        the versions a part's bytes depend on
    Export(depot, dataset, url)                  the export of a dataset's current revision
    Export.part(number, parts, form, left_out)   the bytes of a part, as a stream
"""

import heapq
import itertools
import json
import math
from collections.abc import Iterator
from compression import zstd

import pyarrow as pa
import pyarrow.parquet as pq

from . import quotes
from .corpus import LEFT_OUT, UNMATCHED, _depots, _erasures, _fingerprint, _named, depot_name
from .depot import Depot
from .ids import EXPORT_SEED, export_key
from .kinds import PLAIN
from .order import TRAINING_KINDS
from .parts import Parted, scan

PART_TEXTS = 200_000  # texts of the dataset's libraries per part, about: near 1 GB of text at 5 KB a text
SPACE = 1 << 64  # how many keys there are
LOWEST = -(1 << 63)  # the first key
LEVEL = 6  # the zstd level of a part in jsonl.zst
GROUP_TEXTS = 4096  # texts in one row group of a part in parquet, at most
GROUP_BYTES = 64 << 20  # characters of text after which a row group of a part in parquet ends
TEXT_SCHEMA = pa.schema([pa.field("depot", pa.string(), False), pa.field("id", pa.string(), False),
                         pa.field("revision", pa.int64(), False), pa.field("kind", pa.string(), False),
                         pa.field("text", pa.large_string(), False)])
FORMS = {"jsonl.zst": "application/zstd", "parquet": "application/vnd.apache.parquet"}
# what makes a part's bytes beside the export: a size or sha256 kept for a part holds only for the same MAKER
ROWS = 2  # the version of what a part's rows hold: raised whenever their fields, their values or the rule of
         # which texts a part holds change
MAKER = f"rows {ROWS}, pyarrow {pa.__version__}, zstd {zstd.zstd_version}"


def parts_for(texts: int) -> int:
    """How many parts an export of a dataset whose libraries hold texts texts has: one per PART_TEXTS, at least one."""
    return max(1, math.ceil(texts / PART_TEXTS))


def bounds(number: int, parts: int) -> tuple[int, int]:
    """The keys of part number of parts: (the first, the one after the last)."""
    return LOWEST + number * SPACE // parts, LOWEST + (number + 1) * SPACE // parts


class Export:
    """The export of the current revision of depot's dataset (its row),
    whose depot url names (wpt_depot.corpus.depot_name). Reading what
    identifies it opens the depots the dataset reads and reads their
    erasures; counting its texts (texts) reads the revision's figures, and
    an index of each library whose figure is not known."""

    def __init__(self, depot: Depot, dataset: dict, url: str | None):
        self.depot, self.dataset = depot, dataset
        self.holder = depot_name(depot, dataset, url)
        self.names, self.opened = _depots(depot, dataset, self.holder)
        self.named = _named(dataset, self.names, self.holder)
        self.identity = _fingerprint({"seed": EXPORT_SEED, "depot": self.holder, "dataset": dataset["id"],
                                      "revision": dataset["revision"], "libraries": self.named["libraries"],
                                      "erasures": _erasures(dataset, self.names, self.opened)})

    def texts(self) -> int:
        """How many texts the dataset's libraries held at their change
        numbers, each library counted apart (a text two hold counts twice,
        and so does one no share takes): what sets the number of parts.
        Each library's count is the one the dataset's revision stores in its
        figures (Depot.dataset_figures), read at once, and is counted from
        the library's index only where that figure is not known."""
        stored = self.depot.dataset_figures(self.dataset["id"], self.dataset["revision"])["libraries"]
        return sum(self.opened[name].library_count(ref["library"], ref["change"]) if one["stored"] is None
                   else sum(sized["texts"] for sized in one["stored"]["kinds"].values())
                   for name, ref, one in zip(self.names, self.dataset["libraries"], stored, strict=True))

    def members(self, number: int, parts: int) -> Iterator[tuple[str, str, int, str]]:
        """The texts of part number of parts, in key order (texts of one key
        in the order of their ids, then of their depots' names), a text two
        libraries hold once, at its revision in the first of them: (the name
        a corpus records for its depot, id, revision, kind), given as the
        libraries' reads go, each library read once in key order and the
        reads merged, so nothing waits for a whole part."""
        low, high = bounds(number, parts)

        def read(name: str, ref: dict) -> Iterator[tuple]:
            for key, text, revision, kind, held_tags, _ in self.opened[name].library_range(
                    ref["library"], ref["change"], low, high - 1):
                yield key, name, text, revision, kind, held_tags

        reads = [read(name, ref) for name, ref in zip(self.names, self.dataset["libraries"])]
        # heapq.merge keeps the libraries' order among rows of one key, so the first library's revision comes first
        for _, rows in itertools.groupby(heapq.merge(*reads, key=lambda row: row[0]), key=lambda row: row[0]):
            found = {}
            for _, name, text, revision, kind, held_tags in rows:
                if (name, text) in found:
                    continue
                found[name, text] = (revision, kind, self._taken(kind, held_tags))
            yield from ((name, text, revision, kind) for (text, name), (revision, kind, taken)
                        in sorted(((text, name), value) for (name, text), value in found.items()) if taken)

    def _taken(self, kind: str, held_tags: str) -> bool:
        """Whether the export takes a text of that kind and tags (JSON text) in
        the first library holding it: a plain text a share takes (every plain
        text without shares), an item of a kind a corpus trains on."""
        if kind != PLAIN:
            return kind in TRAINING_KINDS
        shares = self.dataset["shares"]
        return shares is None or any(one["tag"] in json.loads(held_tags) for one in shares)

    def _held(self, name: str, text: str) -> int | None:
        """The revision of text at which the export holds it as a plain text
        of the depot named name, or None when the export does not hold it so:
        read from the first of the dataset's libraries in that depot holding
        it, as that library's row of its export key."""
        key = export_key(text)
        for library, ref in zip(self.names, self.dataset["libraries"]):
            if library != name:
                continue
            for _, held, revision, kind, held_tags, _ in self.opened[name].library_range(
                    ref["library"], ref["change"], key, key):
                if held == text:
                    return revision if kind == PLAIN and self._taken(kind, held_tags) else None
        return None

    def _left_out(self, name: str, conversation: dict) -> str | None:
        """Why the export leaves out a conversation of the depot named name,
        or None when it holds it: LEFT_OUT when a text it quotes is not a
        plain text the export holds from that depot, UNMATCHED when an
        excerpt's selector does not match exactly one place of the revision
        the export holds."""
        source = self.opened[name]
        for block in quotes.excerpts(conversation):
            revision = self._held(name, block["source"])
            if revision is None:
                return LEFT_OUT
            quoted = source.listed_text(block["source"], revision)
            if len(quotes.find(Parted(source.reader(), block["source"], quoted["parts"]), block["selector"])) != 1:
                return UNMATCHED
        return None

    def _rows(self, number: int, parts: int, left_out: dict[str, int]) -> Iterator[tuple[dict, Iterator[str]]]:
        """For each text of the part, in order: (its row without its text, its
        text's pieces in order); a conversation is read whole, and one the
        export leaves out is counted in left_out instead."""
        for name, text, revision, kind in self.members(number, parts):
            source = self.opened[name]
            listed = source.listed_text(text, revision)
            pieces = (value for _, value in scan(source.reader(), {text: listed["parts"]}) if value is not None)
            if kind == "conversation":
                whole = "".join(pieces)
                why = self._left_out(name, json.loads(whole))
                if why is not None:
                    left_out[why] += 1
                    continue
                pieces = iter([whole])
            yield {"depot": name, "id": text, "revision": revision, "kind": kind}, pieces

    def part(self, number: int, parts: int, form: str, left_out: dict[str, int] | None = None) -> Iterator[bytes]:
        """The bytes of part number (from 0) of parts in that form (FORMS),
        as they are made. left_out, when given, ends up holding how many
        conversations of the part the export left out, by why (LEFT_OUT,
        UNMATCHED), once the bytes were all read. KeyError for a form
        FORMS does not name, IndexError for a part outside parts."""
        if form not in FORMS:
            raise KeyError(form)
        if not 0 <= number < parts:
            raise IndexError(number)
        if left_out is None:
            left_out = {}
        left_out.update({LEFT_OUT: 0, UNMATCHED: 0})
        rows = self._rows(number, parts, left_out)
        return _jsonl(rows) if form == "jsonl.zst" else _parquet(rows)


def _jsonl(rows) -> Iterator[bytes]:
    compressor = zstd.ZstdCompressor(level=LEVEL)
    for row, pieces in rows:
        head = json.dumps(row, ensure_ascii=False)[:-1] + ', "text": "'
        out = compressor.compress(head.encode("utf-8"))
        if out:
            yield out
        for piece in pieces:
            out = compressor.compress(json.dumps(piece, ensure_ascii=False)[1:-1].encode("utf-8"))
            if out:
                yield out
        out = compressor.compress(b'"}\n')
        if out:
            yield out
    yield compressor.flush()


class _Sink:
    """A file a Parquet writer writes to, whose bytes are taken as they come."""

    def __init__(self):
        self.held, self.position, self.closed = [], 0, False

    def write(self, data) -> int:
        data = bytes(data)
        self.held.append(data)
        self.position += len(data)
        return len(data)

    def tell(self) -> int:
        return self.position

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def taken(self) -> bytes:
        found = b"".join(self.held)
        self.held.clear()
        return found


def _parquet(rows) -> Iterator[bytes]:
    sink = _Sink()
    writer = pq.ParquetWriter(pa.PythonFile(sink, mode="w"), TEXT_SCHEMA, compression="zstd")
    group, size = [], 0
    for row, pieces in rows:
        text = "".join(pieces)
        group.append({**row, "text": text})
        size += len(text)
        if len(group) >= GROUP_TEXTS or size >= GROUP_BYTES:
            writer.write_table(pa.Table.from_pylist(group, schema=TEXT_SCHEMA))
            group, size = [], 0
            yield sink.taken()
    if group:
        writer.write_table(pa.Table.from_pylist(group, schema=TEXT_SCHEMA))
    writer.close()
    yield sink.taken()
