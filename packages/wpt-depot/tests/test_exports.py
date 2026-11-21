import io
import json
import re
import sqlite3
from compression import zstd
from pathlib import Path

import pyarrow.parquet as pq

from wpt_depot import NewText, exports
from wpt_depot.exports import Export
from wpt_depot.ids import export_key

CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


def rows_of(data: bytes) -> list[dict]:
    return [json.loads(line) for line in zstd.decompress(data).decode().splitlines()]


def test_an_export_holds_every_text_a_corpus_may_take_once_in_parts_by_key_the_same_bytes_each_time(
        make_depot, me, monkeypatch):
    monkeypatch.setattr(exports, "PART_TEXTS", 4)  # several parts from a few texts
    depot = make_depot()
    english = depot.add([NewText(text=f"text {n} in english", licenses=("CC0-1.0",), tags=("en",))
                         for n in range(12)], me)
    [untagged] = depot.add([NewText(text="no share takes it", licenses=("CC0-1.0",))], me)
    [talk, probe] = depot.add([
        NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",), tags=("en",)),
        NewText(kind="probe", content=json.dumps({"question": "q?", "patterns": ["x"]}), licenses=("CC0-1.0",),
                tags=("en",))], me)
    depot.decide("admitted", "true", me)
    first, second = depot.new_library("first"), depot.new_library("second")
    depot.put(first, "true")
    depot.put(second, f"id IN ('{english[0]}', '{english[1]}')")  # two texts both libraries hold
    # a limit and shares that a corpus trims to: an export takes every text a share takes, whatever they say
    dataset = depot.new_dataset("all", "classics", [(None, first), (None, second)], [("en", 1.0)], max_tokens=3)

    export = Export(depot, depot.dataset_by_id(dataset), "https://depot.example")
    parts = exports.parts_for(export.texts())
    found = [rows_of(b"".join(export.part(number, parts, "jsonl.zst"))) for number in range(parts)]

    assert export.texts() == 17 and parts == 5  # 15 texts in the first library, 2 in the second
    taken = [row["id"] for part in found for row in part]
    assert sorted(taken) == sorted([*english, talk]) and untagged not in taken and probe not in taken
    for number, part in enumerate(found):
        keys = [export_key(row["id"]) for row in part]
        low, high = exports.bounds(number, parts)
        assert keys == sorted(keys) and all(low <= key < high for key in keys)
    assert {row["depot"] for part in found for row in part} == {"https://depot.example"}
    assert json.loads([row for part in found for row in part if row["id"] == talk][0]["text"]) == CONVERSATION
    again = Export(depot, depot.dataset_by_id(dataset), "https://depot.example")
    assert [b"".join(again.part(number, parts, "jsonl.zst")) for number in range(parts)] == [
        b"".join(export.part(number, parts, "jsonl.zst")) for number in range(parts)]
    table = b"".join(export.part(1, parts, "parquet"))
    assert pq.read_table(io.BytesIO(table)).to_pylist() == found[1]
    assert b"".join(again.part(1, parts, "parquet")) == table


def quoting(source, exact, prefix=None):
    """A conversation whose question quotes exact of the text source (docs/chat-format.md, "Excerpts")."""
    selector = {"type": "TextQuoteSelector", "exact": exact, **({} if prefix is None else {"prefix": prefix})}
    return NewText(kind="conversation", licenses=("CC0-1.0",), tags=("en",), content=json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                     {"type": "excerpt", "source": source, "selector": selector}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A fox."}]}]}))


def test_an_export_leaves_out_and_counts_a_conversation_whose_quote_it_cannot_hold(make_depot, me):
    # the corpus's rule: the quoted text must be a plain text the export holds, and the quote must find one place
    depot = make_depot()
    fox, outside, french, twice = depot.add([
        NewText(text=words, licenses=("CC0-1.0",), tags=(tag,))
        for words, tag in (("the quick brown fox jumps", "en"), ("the lazy dog sleeps", "en"),
                           ("le renard dort", "fr"), ("a fox and a dog", "en"))], me)
    kept, left_out, untaken, unmatched = depot.add([quoting(fox, "brown fox", "quick "), quoting(outside, "lazy dog"),
                                                    quoting(french, "renard"), quoting(twice, "dog")], me)
    depot.change_content(twice, "a dog and a dog", None, me)  # its quote now finds two places
    depot.decide("admitted", "true", me)
    library = depot.new_library("mine")
    depot.put(library, f"id <> '{outside}'")  # the quoted text left out of the dataset
    dataset = depot.new_dataset("all", "classics", [(None, library)], [("en", 1.0)])  # no share takes french

    export = Export(depot, depot.dataset_by_id(dataset), "https://depot.example")
    left = {}
    rows = rows_of(b"".join(export.part(0, 1, "jsonl.zst", left)))

    assert sorted(row["id"] for row in rows) == sorted([fox, twice, kept])
    assert left == {"quoted text left out": 2, "quote unmatched": 1}
    # the conversation is exported as it is in the depot: the excerpt names its text, the passage is not put in
    assert json.loads(next(row["text"] for row in rows if row["id"] == kept)) == json.loads(depot.content(kept))
    assert [row["id"] for row in pq.read_table(io.BytesIO(b"".join(export.part(0, 1, "parquet")))).to_pylist()] == [
        row["id"] for row in rows]


def test_an_export_reading_another_depot_reads_its_moved_file_and_names_it_by_its_name(make_depot, me, tmp_path):
    depot, other = make_depot("depot"), make_depot("other")
    [mine] = depot.add([NewText(text="the quick fox", licenses=("CC0-1.0",))], me)
    [theirs] = other.add([NewText(text="the lazy dog", licenses=("CC0-1.0",))], me)
    for one in (depot, other):
        one.decide("admitted", "true", me)
    library, their_library = depot.new_library("mine"), other.new_library("theirs")
    depot.put(library, "true")
    other.put(their_library, "true")
    dataset = depot.new_dataset("all", "classics", [(None, library), (other.address, their_library)])
    moved = tmp_path / "elsewhere" / "moved.sqlite"
    moved.parent.mkdir()
    with sqlite3.connect(other.address) as source, sqlite3.connect(moved) as target:
        source.backup(target)
    Path(other.address).unlink()
    depot.edit_dataset(dataset, "all", [(None, library), (str(moved), their_library)])

    export = Export(depot, depot.dataset_by_id(dataset), None)
    data = b"".join(export.part(0, 1, "jsonl.zst"))

    rows = rows_of(data)
    assert {(row["depot"], row["id"], row["text"]) for row in rows} == {
        (depot.name(), mine, "the quick fox"), ("other", theirs, "the lazy dog")}
    assert str(tmp_path) not in zstd.decompress(data).decode()


def test_a_part_reads_its_library_s_rows_of_its_range_of_keys_alone_for_open_and_locked_libraries(make_depot, me):
    # a part reads only the rows of each library whose keys fall in its range: the plan of each read of a library
    # walks that range of the index of that library's rows by key, so a small library beside a large one never reads
    # the large one's rows, nor a text outside the range
    depot = make_depot()
    texts = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",)) for n in range(40)], me)
    depot.decide("admitted", "true", me)
    large, small, locked = depot.new_library("large"), depot.new_library("small"), depot.new_library("locked")
    depot.put(large, f"id IN ({', '.join(repr(one) for one in texts[:30])})")
    depot.put(small, [texts[0], texts[35]])
    depot.put(locked, f"id IN ({', '.join(repr(one) for one in texts[20:])})")
    depot.lock(locked)
    export = Export(depot, depot.dataset_by_id(depot.new_dataset("all", "classics", [
        (None, large), (None, small), (None, locked)])), None)
    reader = depot.db.reader()
    traced = []
    reader.exec_trace = lambda cursor, sql, bindings: traced.append((sql, bindings)) or True

    found = [row["id"] for number in range(4) for row in rows_of(b"".join(export.part(number, 4, "jsonl.zst")))]

    reader.exec_trace = None
    assert sorted(found) == sorted(texts)  # each text once, those several libraries hold included
    ranges = [(sql, bindings) for sql, bindings in traced if "export_key BETWEEN" in sql]
    assert len(ranges) == 12  # each library, for each part
    plans = set()
    for sql, bindings in ranges:
        plan = " ".join(row[3] for row in reader.execute(f"EXPLAIN QUERY PLAN {sql}", bindings))
        index = "pins_key" if bindings["library"] == locked else "memberships_key"
        # the index alone gives each row: no read of the table's rows, which lie in no order of the keys
        assert plan.startswith(f"SEARCH h USING COVERING INDEX {index} (library=? AND export_key>? AND export_key<?)")
        assert "TEMP B-TREE" not in plan  # read in key order, never sorted
        plans.add(index)
    assert plans == {"pins_key", "memberships_key"}
    # a library is counted from the index alone too, never through the table's rows
    for library, table in ((large, "memberships"), (locked, "pins")):
        held, params = depot._held(library, depot.change())
        plan = " ".join(row[3] for row in reader.execute(f"EXPLAIN QUERY PLAN SELECT count(*) FROM ({held})", params))
        assert re.search(rf"USING COVERING INDEX {table}_\w+ \(library=\?\)", plan), plan
    # the small library's read of a part gives only its own texts of that part
    low, high = exports.bounds(0, 1)
    assert sorted(row[1] for row in depot.library_range(small, depot.change(), low, high - 1)) == sorted(
        [texts[0], texts[35]])


def test_every_way_a_library_gets_a_text_stores_the_text_s_export_key_with_it(make_depot, me):
    depot = make_depot()
    texts = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",)) for n in range(4)], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, texts)
    fork = depot.fork(library, "fork", without=[texts[0]])
    depot.lock(fork)

    for table, count in (("memberships", 7), ("pins", 3)):
        rows = depot.rows(f"SELECT text, export_key FROM {table}")
        assert len(rows) == count and all(row["export_key"] == export_key(row["text"]) for row in rows)
