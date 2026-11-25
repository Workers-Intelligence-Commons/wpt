"""A vocabulary asked of the freight, its program wpt-vocabulary learning
it from a sample given in bytes: each share holds its proportion of the
sample's bytes within a thousandth, by group; the vocabulary is the one the
tokenizer learns from the items in the order the delivery lists them; the
same request gives the same vocabulary again; and without shares the
sample takes the plain texts of the listing in order, each that fits what
is left of the size, until a thousandth of it is left or a thousand texts
in a row did not fit (docs/data-selection.md, "Filling a part")."""

import hashlib
import json

import pytest

from freight import api
from wpt_depot import NewText
from wpt_depot import api as depot_api
from wpt_store import open_store
from wpt_store.names import File
from wpt_tokenizer import Tokenizer

SAMPLE = 60_000  # bytes
SHARES = [("en", 0.5), ("de", 0.3), ("fr", 0.2)]
VOCAB_SIZE = 600
WORDS = {"en": "the fox ran over the hill and", "de": "der Fuchs lief über den Hügel und",
         "fr": "le renard courut sur la colline et"}


def texts() -> list[NewText]:
    """About 1.5 times each share's part of SAMPLE in plain texts of 10 to 400 bytes, the English ones in groups (a
    second tag, or none), and texts of a tag no share names."""
    made = []
    for tag, share in SHARES:
        words, held, n = WORDS[tag], 0, 0
        while held < 1.5 * share * SAMPLE:
            size = 10 + (n * 37) % 390
            text = f"{n} " + (words + " ") * (size // len(words) + 1)
            text = text[:size]
            group = ("fable", "news", None)[n % 3] if tag == "en" else None
            made.append(NewText(text=text, licenses=("MIT",), tags=(tag, group) if group else (tag,)))
            held += len(text.encode())
            n += 1
    made += [NewText(text=f"nobody's text {n}", licenses=("MIT",), tags=("xx",)) for n in range(20)]
    return made


def request(built, name, shares, size=SAMPLE) -> dict:
    name = built.names.get(name, name)  # a test's label of a snapshot, the depot's name of it
    snapshot = built.depot.snapshot(name)
    return {"depot": built.depot.name(),
            "snapshot": {"name": name, "change": snapshot["at_change"], "digest": snapshot["digest"]},
            "selection": {"kinds": ["plain"], "shares": [{"tag": tag, "share": share} for tag, share in shares],
                          "size": {"unit": "bytes", "amount": size}},
            "preparation": {"name": "wpt-vocabulary", "settings": {"vocab_size": VOCAB_SIZE}}}


def asked(built, folder, asking) -> dict:
    """The delivery note of the request, run by a freight whose home and dock are in folder."""
    where = {"home": str(folder / "freight"), "depot": built.depot.address, "dock": (folder / "dock").as_uri()}
    done = api.wait(api.request(asking, **where)["id"], home=where["home"])
    assert done["state"] == "done", done["problem"]
    return done["answer"]


def vocabulary_of(folder, note) -> bytes:
    (file,) = note["files"]
    assert file["path"] == "vocabulary.json"
    return open_store((folder / "dock").as_uri()).path(File.of(file)).read_bytes()


def test_each_share_holds_its_proportion_of_the_samples_bytes_and_the_same_request_gives_the_same_vocabulary(
        built, tmp_path):
    built.snapshot("v", texts())
    asking = request(built, "v", SHARES)
    note = asked(built, tmp_path / "a", asking)
    rows = api.items(note["key"], dock=(tmp_path / "a" / "dock").as_uri())
    tags = [tag for tag, _ in SHARES]
    held = dict.fromkeys(tags, 0)
    for row in rows:
        assert row["kind"] == "plain"
        held[next(tag for tag in tags if tag in row["tags"])] += row["bytes"]
    for tag, share in SHARES:
        assert abs(held[tag] / SAMPLE - share) <= 0.001, (tag, held[tag])
    assert [one["bytes"] for one in note["shares"]] == [held[tag] for tag in tags]
    groups = {one["tag"]: one for one in note["shares"][0]["groups"]}
    assert set(groups) == {None, "fable", "news"} and all(one["texts"] for one in groups.values())
    assert sum(one["bytes"] for one in groups.values()) == held["en"]  # each English text is in one group here
    assert note["on_erasure"] == "note" and note["preparation"]["unit"] is None
    assert {row["weight"] for row in rows} == {None} and {one["weight"] for one in note["shares"]} == {None}

    # the vocabulary the tokenizer learns from the items, read once in the order the delivery lists them
    data = vocabulary_of(tmp_path / "a", note)
    contents = list(depot_api.contents(built.depot.address,
                                       [(row["id"], row["revision"], row["piece"]) for row in rows]))
    assert Tokenizer.train(contents, VOCAB_SIZE).to_bytes() == data
    assert note["result"] == {"sha256": hashlib.sha256(data).hexdigest(), "size": Tokenizer.from_bytes(data).vocab_size,
                              "tokenizer_version": json.loads(data)["tokenizer_version"]}

    # the same request, run again on another dock
    again = asked(built, tmp_path / "b", asking)
    assert (again["key"], again["id"], again["result"]["sha256"]) == (note["key"], note["id"], note["result"]["sha256"])
    assert vocabulary_of(tmp_path / "b", again) == data


def test_without_shares_the_sample_takes_the_first_plain_texts_of_the_listing_that_fit(built, tmp_path):
    built.snapshot("v", texts())
    size = SAMPLE // 3
    note = asked(built, tmp_path, request(built, "v", [], size=size))
    rows = api.items(note["key"], dock=(tmp_path / "dock").as_uri())
    expected, used, missed = [], 0, 0
    for entry in depot_api.listing(built.depot.address, built.depot.snapshot(built.names["v"])["id"]):
        if size - used <= size // 1000 or missed >= 1000:
            break
        if entry["kind"] != "plain":
            continue
        if used + entry["bytes"] > size:
            missed += 1
            continue
        expected.append((entry["id"], entry["piece"]))
        used, missed = used + entry["bytes"], 0
    assert [(row["id"], row["piece"]) for row in rows] == expected
    assert size * 0.999 <= sum(row["bytes"] for row in rows) <= size


@pytest.mark.parametrize("settings, why", [({}, "vocab_size"), ({"vocab_size": 600, "merges": 3}, "no setting merges")])
def test_a_request_without_the_vocabularys_size_or_with_a_setting_it_does_not_take_fails(built, tmp_path, settings, why):
    asking = request(built, "d", [("en", 1.0)])
    asking["preparation"]["settings"] = settings
    where = {"home": str(tmp_path / "freight"), "depot": built.depot.address, "dock": (tmp_path / "dock").as_uri()}
    with pytest.raises(Exception, match=f"wpt-vocabulary: Failed: .*{why}"):  # the program refuses its start as the freight asks its version
        api.request(asking, **where)
