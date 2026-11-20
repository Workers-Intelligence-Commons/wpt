import pytest

from wpt_depot import NewText, parts, strips
from wpt_depot.parts import Sizes

SMALL = Sizes(minimum=64, target=256, maximum=1024)  # parts of a few lines, so a read reaches inside a text's parts


@pytest.fixture
def strip(make_depot, me, monkeypatch):
    """A depot whose library holds texts of 0 to 2000 characters, with the strip they make."""
    monkeypatch.setattr(parts, "SIZES", SMALL)
    depot = make_depot()
    sizes = [30, 0, 2000, 5, 700, 1, 1200, 0, 64]
    texts = [f"{n:03d} " * (size // 4) + "x" * (size % 4) for n, size in enumerate(sizes)]
    added = depot.add([NewText(text=text, licenses=("CC0-1.0",), title=f"t{n}") for n, text in enumerate(texts)], me)
    depot.add([NewText(text="left outside", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("all")
    depot.put(library, list(added))
    return depot, library, added, texts


def joined(found: dict) -> str:
    return "".join(piece["text"] for piece in found["pieces"])


def test_reads_after_one_another_give_the_whole_strip_in_the_order_of_ids_and_its_end(strip, monkeypatch):
    monkeypatch.setattr(strips, "BATCH", 2)  # the index read a few rows at a time
    depot, library, added, texts = strip
    whole, anchor, seen, edge = "", (None, 0), [], False
    while not edge:
        found = strips.read(depot, library, *anchor, characters=333)
        assert found["pieces"] or found["edge"]
        whole += joined(found)
        seen += [piece["id"] for piece in found["pieces"] if piece["start"] == 0]
        if found["pieces"]:
            last = found["pieces"][-1]
            anchor = (last["id"], last["start"] + len(last["text"]))
        edge = found["edge"]
    assert whole == "".join(texts)
    assert seen == list(added)  # every text once, the empty ones too


def test_reads_backward_from_the_end_give_the_same_strip(strip, monkeypatch):
    monkeypatch.setattr(strips, "BATCH", 3)
    depot, library, added, texts = strip
    whole, anchor, seen, edge = "", (None, 0), [], False
    while not edge:
        found = strips.read(depot, library, *anchor, characters=250, forward=False)
        whole = joined(found) + whole
        seen = [piece["id"] for piece in found["pieces"] if piece["start"] + len(piece["text"]) == piece["length"]] + seen
        if found["pieces"]:
            anchor = (found["pieces"][0]["id"], found["pieces"][0]["start"])
        edge = found["edge"]
    assert whole == "".join(texts)
    assert seen == list(added)


def test_a_read_stops_at_its_count_of_texts(strip):
    depot, library, added, _ = strip
    found = strips.read(depot, library, None, 0, characters=10_000, most_texts=3)
    assert [piece["id"] for piece in found["pieces"]] == list(added[:3]) and not found["edge"]


def test_the_map_locates_every_position_as_the_strip_counts_it(strip, monkeypatch):
    monkeypatch.setattr(strips, "WALKED", 3)  # a walk of several statements
    depot, library, added, texts = strip
    found = strips.build(depot, library, every=2)
    assert (found.texts, found.characters) == (len(texts), sum(map(len, texts)))
    assert [text for _, text, _ in found.marks] == list(added[::2])
    assert strips.total(depot, library) == {"texts": found.texts, "characters": found.characters}
    strip_text = "".join(texts)
    for position in range(0, len(strip_text), 37):
        text, offset = strips.locate(depot, library, found, position)
        assert texts[added.index(text)][offset] == strip_text[position]
        read = strips.read(depot, library, text, offset, characters=50, inclusive=True)
        start = position - offset  # the texts hold no line end: a read at a place starts with its text
        assert (read["pieces"][0]["id"], read["pieces"][0]["start"]) == (text, 0)
        assert joined(read) == strip_text[start:start + len(joined(read))]
    assert strips.locate(depot, library, found, len(strip_text) + 10) == (added[-1], len(texts[-1]))
    assert strips.last(depot, library) == (added[-1], len(texts[-1]))


def test_a_library_s_state_moves_with_what_it_holds_and_a_locked_one_reads_what_it_pinned(strip, me):
    depot, library, added, texts = strip
    before = strips.state(depot, library)
    depot.take_out(library, [added[0]])
    assert strips.state(depot, library) != before
    depot.lock(library)
    assert strips.state(depot, library) == "locked"
    depot.change_content(added[2], "changed after the lock", None, me)
    assert joined(strips.read(depot, library, None, 0, characters=100_000)) == "".join(texts[1:])


def test_an_empty_library_reads_nothing(make_depot):
    depot = make_depot()
    library = depot.new_library("empty")
    assert strips.read(depot, library, None, 0, characters=10) == {"pieces": [], "edge": True}
    assert strips.locate(depot, library, strips.build(depot, library), 5) is None
    assert strips.last(depot, library) is None


def test_a_read_ends_with_a_line_and_a_read_at_a_place_starts_with_its_line(make_depot, me, monkeypatch):
    monkeypatch.setattr(strips, "LINE", 40)
    depot = make_depot()
    text = "first line\nsecond line\n" + "x" * 100 + "\nlast"
    [one] = depot.add([NewText(text=text, licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("one")
    depot.put(library, [one])

    assert joined(strips.read(depot, library, None, 0, characters=3)) == "first line\n"
    assert joined(strips.read(depot, library, one, 15, characters=1, inclusive=True)) == "second line\n"
    assert joined(strips.read(depot, library, one, 4, characters=1, inclusive=True)) == "first line\n"
    assert joined(strips.read(depot, library, one, 30, characters=1)) == "x"  # a line past LINE is cut
    assert joined(strips.read(depot, library, one, 18, characters=2, forward=False)) == "second "
    assert joined(strips.read(depot, library, one, len(text), characters=2, forward=False)) == "last"
