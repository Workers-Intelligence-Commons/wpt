import hashlib
import json
import random
from pathlib import Path

import pytest

from wpt_corpus.files import DockFiles
from wpt_corpus.shards import read_verified
from wpt_tokenizer import END, Tokenizer
from depot import About, NewText, clone, erasure, parts
from corpus_build import _Encoder, build
from depot.follow import pull
from depot.parts import FileText, Sizes, listed, parted, split

SMALL = Sizes(minimum=64, target=256, maximum=1024)  # parts of a few lines, so a test text of a few KB has many
TOKENIZER = Tokenizer.train([b"the quick brown fox jumps over the lazy dog\n"] * 4, vocab_size=560)


@pytest.fixture
def small_parts(monkeypatch):
    monkeypatch.setattr(parts, "SIZES", SMALL)


def lines(count: int, start: int = 0) -> str:
    """count distinct lines of hex digits, which compression leaves as they are."""
    return "".join(hashlib.sha256(str(number).encode()).hexdigest() + "\n" for number in range(start, start + count))


def plain(text):
    return NewText(text=text, licenses=("CC0-1.0",))


def files_holding(folder, passage: str) -> list[Path]:
    return [path for path in Path(folder).rglob("*") if path.is_file() and passage.encode() in path.read_bytes()]


def test_a_text_cut_into_parts_joins_back_whole_with_the_same_parts_however_it_is_read():
    # the parts depend on the bytes only, so a file read in blocks of any size stores the same parts
    rng = random.Random(7)
    text = "".join(rng.choice(["short\n", "a longer line of words\n", "é\n", "x" * 3000, "\n"]) for _ in range(2000))
    data = text.encode()

    whole = list(split([data], SMALL))
    for size in (1, 7, 100, 4096):
        assert list(split([data[at:at + size] for at in range(0, len(data), size)], SMALL)) == whole
    assert b"".join(whole) == data and len(whole) > 10
    assert all(len(one) <= SMALL.maximum for one in whole)
    assert all(one.endswith(b"\n") or len(one) > SMALL.maximum - 4 for one in whole[:-1])  # a long line is cut
    assert "".join(one.value for one in parted(text, SMALL)) == text
    assert list(parted("", SMALL)) == [] and len(list(parted("one line", SMALL))) == 1


def test_an_edit_in_the_middle_changes_only_the_parts_around_it():
    before = lines(400)
    after = before.replace(hashlib.sha256(b"200").hexdigest(), "an edited line")

    old, new = listed(before, SMALL).parts, listed(after, SMALL).parts

    assert len(old) > 20
    assert len(set(new) - set(old)) <= 3


def test_a_text_bigger_than_a_part_is_added_from_a_file_read_shown_and_built_into_a_corpus(
        make_depot, me, small_parts, tmp_path):
    text = lines(300)
    path = tmp_path / "text.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    depot = make_depot()

    [one] = depot.add([NewText(text=FileText(path), licenses=("CC0-1.0",))], me)

    assert depot.show(one)["text"]["text"] == text
    assert depot.sql(f"SELECT text, length FROM texts WHERE id = '{one}'") == [
        {"text": text, "length": len(text)}]
    assert [row["n"] for row in depot.sql(f"SELECT count(*) AS n FROM parts WHERE id = '{one}'")] == [
        len(listed(text, SMALL).parts)] and len(listed(text, SMALL).parts) > 10
    assert depot.passages("true", characters=100)["rows"][0]["passage"] == text[:100]
    cut = depot.show(one, characters=100)
    assert cut["cut"] and cut["text"]["text"] == text[:100] and cut["text"]["length"] == len(text)
    assert not depot.show(one, characters=len(text))["cut"]
    middle = len(text) // 2
    assert depot.passage(one, middle, 5000) == {"id": one, "version": 1, "length": len(text), "from": middle,
                                                 "text": text[middle:middle + 5000]}  # across several parts
    assert depot.passage(one, len(text) - 10, 5000)["text"] == text[-10:]
    assert depot.passage(one, len(text) + 10, 5000)["text"] == ""
    depot.decide("admitted", "true", me)
    snapshot = depot.create_snapshot()
    dock = (tmp_path / "dock").as_uri()
    vocabulary = {"sha256": hashlib.sha256(TOKENIZER.to_bytes()).hexdigest(), "size": TOKENIZER.vocab_size}

    record = build(depot, snapshot["id"], {"vocabulary": vocabulary}, dock, TOKENIZER, log=lambda *_: None)

    [(tokens, _, document, _, _)] = [row for shard in record["shards"]["plain"] for row in read_verified(DockFiles(dock, tmp_path / "reader"), shard)]
    assert document == one
    assert list(tokens) == [*TOKENIZER.encode(text.encode()), END]  # read in parts, tokenized as one text


def test_a_new_content_version_stores_only_the_parts_no_version_held_before(make_depot, me, small_parts):
    depot = make_depot()
    text = lines(400)
    [one] = depot.add([plain(text)], me)
    first = depot.sql("SELECT count(*) AS n FROM parts")[0]["n"]
    changed = text.replace(hashlib.sha256(b"200").hexdigest(), "an edited line")

    depot.change_content(one, changed, None, me)

    added = depot.sql("SELECT count(*) AS n FROM parts")[0]["n"] - first
    assert 0 < added <= 3
    assert depot.show(one)["text"]["text"] == changed
    assert depot.content(one, 1) == text and depot.content(one, 2) == changed


def test_an_erasure_across_two_parts_cuts_only_those_two_and_leaves_no_copy(make_depot, me, small_parts):
    depot = make_depot()
    text = lines(200)
    [one] = depot.add([plain(text)], me)
    found = listed(text, SMALL).parts
    boundary = found[0][1] + found[1][1]  # the end of the second part
    start, end = boundary - 40, boundary + 40
    left, right = text[start:boundary], text[boundary:end]
    depot.decide("admitted", "true", me)
    request = depot.ask_erasure("a passage", [(one, start, end)], me)

    depot.decide_erasure(request, "admitted", me)

    assert depot.show(one)["text"]["text"] == text[:start] + text[end:]
    now = {bytes(row["hash"]) for row in depot.sql("SELECT hash FROM parts")}
    assert now - {hash_ for hash_, _ in found} != set()
    assert len({hash_ for hash_, _ in found} - now) == 2  # the two parts that held the passage, and no other
    assert files_holding(depot.address, left) == [] and files_holding(depot.address, right) == []


def test_cutting_a_text_read_in_blocks_cuts_as_cutting_it_whole():
    # an erasure reads a big text a part at a time: a passage across two blocks is found as in the whole string
    rng = random.Random(3)
    for _ in range(3000):
        value = "".join(rng.choice("ab\n") for _ in range(rng.randint(0, 40)))
        named = []
        if value:
            start = rng.randrange(len(value))
            named = [(start, rng.randint(start + 1, len(value)), 0)]
        removing = ["".join(rng.choice("ab\n") for _ in range(rng.randint(1, 4))) for _ in range(rng.randint(0, 3))]
        cuts = sorted(rng.sample(range(1, len(value)), min(len(value) - 1, rng.randint(0, 6)))) if len(value) > 1 else []
        bounds = [0, *cuts, len(value)]
        blocks = [(a, value[a:b]) for a, b in zip(bounds, bounds[1:])]

        whole = erasure._cut_pieces(lambda: iter([(0, value)]), len(value), named, removing)

        assert erasure._cut_pieces(lambda: iter(blocks), len(value), named, removing) == whole
        assert erasure.holds_erased([iter(block for _, block in blocks)], [erasure.named(one) for one in removing]) \
            == any(one in value for one in removing)


def test_a_text_fed_to_the_corpus_encoder_in_pieces_gives_the_tokens_of_the_whole_text_and_holds_little():
    # indented JSON and one long line: no line starts with a byte that is not whitespace
    rng = random.Random(5)
    words = ["the", "quick", "fox", "{", "}", '"key":', "1,", "\u00e9t\u00e9", "\u4e2d"]
    spaces = [" ", "  ", "\t", "\n    ", "\r\n  ", "\x0b", "\x0c", "\n\n   "]
    for _ in range(200):
        text = "".join(rng.choice(words) + rng.choice(spaces) for _ in range(rng.randint(0, 300)))
        text = rng.choice(["", " ", "  "]) + text
        bounds = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(0, 40)))) if len(text) > 1 else []
        encoder, held = _Encoder(TOKENIZER), 0
        for start, end in zip([0, *bounds], [*bounds, len(text)]):
            encoder.feed(text[start:end])
            held = max(held, sum(map(len, encoder.pending)) - len(text[start:end].encode()))
        assert list(encoder.finish()) == TOKENIZER.encode(text.encode())
        assert held <= 2 * max(map(len, words + spaces)) * 4  # never more than the last chunk or two before a piece


def test_a_clone_of_a_clone_pulls_the_parts_of_a_text_added_upstream_after_its_upstream_s_own_changes(serve, 
        make_depot, me, tmp_path):
    first = make_depot("first")
    first.add([plain("first")], me)
    second = clone(serve(first), str(tmp_path / "second.sqlite"), About("second", "Grace Hopper", "grace@example.org"))
    for number in range(8):  # the second depot's change numbers run past the first's
        second.add([plain(f"own {number}")], me)
    third = clone(serve(second), str(tmp_path / "third.sqlite"), About("third", "Ada Lovelace", "ada@example.org"))
    [text] = first.add([plain("second text from the first")], me)

    pull(second)
    pull(third)

    assert second.show(text)["text"]["text"] == "second text from the first"
    assert third.show(text)["text"]["text"] == "second text from the first"
    assert [row["n"] for row in third.sql(f"SELECT count(*) AS n FROM parts WHERE id = '{text}'")] == [1]


def test_adding_again_the_text_an_erasure_left_makes_no_new_version_and_its_parts_are_cut_as_any_text(
        make_depot, me, small_parts):
    depot = make_depot()
    text = lines(200)
    [one] = depot.add([plain(text)], me)
    depot.decide("admitted", [one], me)
    start = text.index(hashlib.sha256(b"50").hexdigest()) + 7
    end = text.index(hashlib.sha256(b"52").hexdigest()) + 3
    depot.decide_erasure(depot.ask_erasure("a passage", [(one, start, end)], me), "admitted", me)
    cut = depot.content(one)

    stored = json.loads(depot.sql(f"SELECT parts FROM content_lists WHERE id = '{one}'")[0]["parts"])

    assert cut == text[:start] + text[end:]
    assert [(bytes.fromhex(part["hash"]), part["length"]) for part in stored] == listed(cut, SMALL).parts
    assert depot.change_content(one, cut, None, me) is None
    assert [row["n"] for row in depot.sql(f"SELECT count(*) AS n FROM content_lists WHERE id = '{one}'")] == [1]
