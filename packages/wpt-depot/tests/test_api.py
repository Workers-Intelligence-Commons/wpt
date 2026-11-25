import json

import pytest

from wpt_depot import NewText, api, pieces
from wpt_depot.errors import NotFound
from wpt_depot.schema import LONG


def admitted(depot, me, *texts):
    """The ids of texts added and admitted, and a snapshot of them."""
    added = depot.add(list(texts), me)
    depot.decide("admitted", "true", me)
    return added, depot.create_snapshot("s1")


def carried(value) -> bool:
    """Whether JSON carries the value as it is."""
    return json.loads(json.dumps(value)) == value


def test_every_answer_is_a_value_json_carries_and_the_listing_names_every_text_and_piece(make_depot, me):
    depot = make_depot()
    long_text = "a line of a long text\n" * (LONG // 20)
    (short, long), made = admitted(depot, me, NewText(text="the dog", licenses=("CC0-1.0",), tags=("en",)),
                                   NewText(text=long_text, licenses=("CC0-1.0",), title="Long"))
    address = depot.address

    row = api.snapshot(address, made["id"])
    listed = list(api.listing(address, made["id"]))
    size = api.listing_size(address, made["id"])

    assert row == api.snapshot(address, name="s1") and row["digest"] == made["digest"]
    assert size == {"entries": len(listed), "long_texts": 1, "pieces": pieces.count(len(long_text)),
                    "piece_characters": pieces.LENGTH}
    assert sorted((one["id"], one["piece"], one["name"]) for one in listed) == sorted(
        [(short, None, short)] + [(long, n, pieces.name(long, n)) for n in range(size["pieces"])])
    # a key is a decimal string, which a JSON reader of doubles does not round
    assert all(isinstance(one["key"], str) and str(int(one["key"])) == one["key"] for one in listed)
    assert all(carried(value) for value in (row, listed, size, api.about(address),
                                            api.credits(address, [[short, 1]]), api.erasures(address)))
    # a walk goes on right after any place
    at = listed[2]
    assert list(api.listing(address, made["id"], [at["key"], at["owner"], at["number"], at["piece"]])) == listed[3:]
    assert api.texts(address, made["id"], [short, "depot-999"]) == {
        short: {"revision": 1, "kind": "plain", "tags": ["en"], "bytes": 7}}


def test_content_comes_as_bytes_whole_by_piece_or_by_span(make_depot, me):
    depot = make_depot()
    long_text = "a line of a long text\n" * (LONG // 20)
    (long,), _ = admitted(depot, me, NewText(text=long_text, licenses=("CC0-1.0",)))
    address = depot.address

    assert b"".join(api.content(address, long, 1)) == long_text.encode()
    assert b"".join(b"".join(api.content(address, long, 1, n)) for n in range(pieces.count(len(long_text)))) == \
        long_text.encode()
    assert b"".join(api.content(address, long, 1, span=[2, 6])) == b"line"
    asked = [[long, 1, n] for n in reversed(range(pieces.count(len(long_text))))]
    assert list(api.contents(address, asked)) == [b"".join(api.content(address, *one)) for one in asked]
    with pytest.raises(NotFound):
        list(api.content(address, long, 2))


def test_an_erasure_carried_out_names_the_texts_it_reached_and_the_snapshot_still_holds_them(make_depot, me):
    depot = make_depot()
    (fox, dog), made = admitted(depot, me, NewText(text="the quick fox", licenses=("CC0-1.0",)),
                                NewText(text="the lazy dog", licenses=("CC0-1.0",)))
    erased = depot.ask_erasure("a word", [(fox, 4, 9)], me)
    depot.decide_erasure(erased, "admitted", me)

    found = api.erasures(depot.address)
    [carried_out] = depot.rows("SELECT change FROM erasures WHERE id = ?", (erased,))
    assert found["erasures"] == [{"id": erased, "change": carried_out["change"], "texts": [fox]}]
    assert api.holds(depot.address, made["id"], [dog, fox, "depot-999"]) == [dog, fox]
    # asked from the mark it gave, nothing new; a later erasure, alone
    assert api.erasures(depot.address, found["mark"]) == found | {"erasures": []}
    later = depot.ask_erasure("a word", [(dog, 4, 8)], me)
    depot.decide_erasure(later, "admitted", me)
    after = api.erasures(depot.address, found["mark"])
    assert after["mark"] > found["mark"] and [one["id"] for one in after["erasures"]] == [later]
    assert [one["id"] for one in api.erasures(depot.address)["erasures"]] == sorted([erased, later])
