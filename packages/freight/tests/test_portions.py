"""A delivery cut into portions, with the tests' program fake
(conftest.py): a selection without portions keeps the key and the bytes
it had; a text lies in one portion at most, a long text's pieces and the
passages a conversation quotes included; a first portion sized in bytes
takes the same texts whatever the preparation, its settings and the
portions after it; the program is told each item's portion and the note
counts each portion; a job cut short goes on to the same delivery; and the
canonical form and the key cover the portions."""

import json

import pytest

from depot import NewText
from dock import api as dock_api
from freight import api, deliveries
from freight.job import EARLIER, LEFT_OUT, Job
from freight.selection import Refused, canonical

LINE = "the quick brown fox jumps over the lazy dog\n"
SHARES = [{"tag": "en", "share": 0.7}, {"tag": "de", "share": 0.3}]


class Stop(Exception):
    """A job cut short."""


def quoting(source: str, exact: str) -> NewText:
    """A conversation whose question quotes exact of the text source (docs/chat-format.md, "Excerpts")."""
    return NewText(kind="conversation", licenses=("MIT",), tags=("en",), content=json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                     {"type": "excerpt", "source": source,
                                      "selector": {"type": "TextQuoteSelector", "exact": exact}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A crow."}]}]}))


@pytest.fixture
def rich(freight):
    """The freight fixture's depot with a second snapshot: its 18 texts, 8 fables tagged en and fable, a text
    longer than a piece tagged en and code, and a conversation quoting each fable and one quoting the long text.
    request(selection, **settings): the request of that snapshot, prepared by fake with those settings;
    quotes: {conversation id: the text it quotes}."""
    depot = freight.depot
    plain = [NewText(text=f"fable {n}: a crow found cheese number {n}. " + "it sang and the fox took it. " * (3 + n),
                     licenses=("MIT",), tags=("en", "fable")) for n in range(8)]
    plain.append(NewText(text="".join(f"=== src/file{n}.c\n" + LINE * 3000 for n in range(10)), licenses=("MIT",),
                         tags=("en", "code")))
    added = depot.add(plain, freight.me)
    fables, repository = added[:8], added[8]
    conversations = [quoting(fable, f"found cheese number {n}") for n, fable in enumerate(fables)]
    conversations.append(quoting(repository, "=== src/file7.c"))
    talks = depot.add(conversations, freight.me)
    depot.decide("admitted", f"id IN ({', '.join(repr(one) for one in added + talks)})", freight.me)
    name = depot.create_snapshot()["name"]

    def request(selection: dict, **settings) -> dict:
        snapshot = depot.snapshot(name)
        return {"depot": depot.name(),
                "snapshot": {"name": name, "change": snapshot["at_change"], "digest": snapshot["digest"]},
                "selection": selection, "preparation": {"name": "fake", "settings": settings, "inputs": {}}}

    freight.rich = request
    freight.quotes = dict(zip(talks, [*fables, repository], strict=True))
    freight.repository = repository
    return freight


def held_out(amount: int = 900) -> dict:
    return {"name": "held_out", "kinds": ["plain"], "shares": SHARES, "size": {"unit": "bytes", "amount": amount}}


def run(freight, request, folder: str) -> tuple[dict, list[dict]]:
    dock = (freight.tmp_path / f"{folder}-dock").as_uri()
    note = Job(request, freight.depot.address, dock, freight.tmp_path / folder).run()
    return note, deliveries.read_items(dock, note).to_pylist()


def chosen(rows: list[dict], portion: str) -> list[tuple]:
    return [(row["id"], row["piece"], row["revision"]) for row in rows if row["portion"] == portion]


# what a selection without portions was keyed and delivered as before portions existed, for the freight
# fixture's request (shares en 0.7 / de 0.3, a size of 1,500 bytes of text) and three others
PINNED = {"shares and size": ("5adfb2d8f552f5f58f8792d558c146bfb3add79288443e535613e4e149351994",
                              "979aa3ccb4e1a0607786502bd77ac0bdb9754fdeaa2e682bea125270ab902224"),
          "in bytes": ("714627a0465ce998419c052c3ba9a78f06b72828d275e35792ea27a9ef84f3ac",
                       "b3859315bf2f9f5e83d12b7f90e7f919ecf84d6691121d4b170eb234b4b85f8b"),
          "no size": ("4db0375ffdc64d4012ebba07a55656625c4a7ff720923d82b9e9c9b2b3d4f505",
                      "be5cf7ba94a95e290aca5845031bda32d52222590bad1235e9def515ac0a9760"),
          "no shares": ("80e0a880f5b42049ad8bc9fc7fc884e98646d7dcba8ab0e3e28222a5a674ae8d",
                        "33df2749d03647f4b9ea4bbb099af748942e8b3626313a36028de5574ea816dd")}


def test_a_selection_without_portions_keeps_its_key_and_its_bytes(freight):
    requests = {name: freight.request() for name in PINNED}
    requests["in bytes"]["selection"]["size"]["unit"] = "bytes"
    requests["no size"]["selection"]["size"] = None
    requests["no shares"]["selection"] = {"kinds": ["plain"], "size": {"unit": "bytes", "amount": 700}}
    for name, request in requests.items():
        note, rows = run(freight, request, name.replace(" ", "-"))
        assert (note["key"], note["id"]) == PINNED[name], name
        assert "portions" not in note and all("portion" not in row for row in rows)
        assert api.key(request, "1") == note["key"]


def test_each_text_lies_in_one_portion_at_most(rich):
    request = rich.rich({"portions": [held_out(), {"name": "rest", "kinds": [], "shares": [], "size": None}]})
    note, rows = run(rich, request, "work")
    held = {row["id"] for row in rows if row["portion"] == "held_out"}
    rest = {row["id"] for row in rows if row["portion"] == "rest"}
    assert held and rest and not held & rest
    assert [row["portion"] for row in rows] == sorted((row["portion"] for row in rows), key=["held_out", "rest"].index)
    assert {row["kind"] for row in rows if row["portion"] == "held_out"} == {"plain"}
    pieces = {row["portion"] for row in rows if row["id"] == rich.repository}
    assert len(pieces) == 1 and len([row for row in rows if row["id"] == rich.repository]) > 1
    taken = {row["id"]: row for row in rows if row["kind"] == "conversation"}
    for conversation, quoted in rich.quotes.items():
        if quoted in held:  # its passage went to the held-out portion: the conversation stays out of the rest
            assert conversation not in taken
        else:
            assert taken[conversation]["portion"] == "rest" and taken[conversation]["quotes"] == [quoted]
    counted = {one["name"]: one for one in note["portions"]}
    assert note["shares"] is None and note["left_out"] is None
    assert counted["rest"]["left_out"][LEFT_OUT] == sum(quoted in held for quoted in rich.quotes.values())
    assert counted["rest"]["left_out"][EARLIER] == len(held)
    assert counted["held_out"]["left_out"][EARLIER] == 0
    for name, ids in (("held_out", held), ("rest", rest)):
        mine = [row for row in rows if row["portion"] == name]
        assert counted[name]["items"]["texts"] == len(ids)
        assert counted[name]["bytes"] == sum(row["bytes"] for row in mine)
        assert counted[name]["weight"] == sum(row["weight"] for row in mine)
    assert counted["held_out"]["bytes"] <= 900 and [one["tag"] for one in counted["held_out"]["shares"]] == ["en", "de"]
    assert counted["rest"]["shares"] is None
    # the program was told each item's portion: fake writes it on the item's line
    lines = b"".join(dock_api.fetch(dock_api_url(rich, "work"), one["key"]) for one in note["files"]).splitlines()
    assert [line.split(b" ")[2].decode() for line in lines] == [row["portion"] for row in rows]


def dock_api_url(freight, folder: str) -> str:
    return (freight.tmp_path / f"{folder}-dock").as_uri()


def test_a_first_portion_sized_in_bytes_takes_the_same_texts_whatever_follows(rich):
    tokens = {"name": "rest", "kinds": [], "shares": [], "size": {"unit": "bytes-of-text", "amount": 3000}}
    base_note, base = run(rich, rich.rich({"portions": [held_out(), tokens]}), "base")
    first = chosen(base, "held_out")
    assert 0 < len(first) < len({row["id"] for row in base})
    variants = {
        "other settings": rich.rich({"portions": [held_out(), tokens]}, per_file=100, at_name="row"),
        "a larger rest": rich.rich({"portions": [held_out(), {**tokens, "size": {"unit": "bytes-of-text",
                                                                                   "amount": 9000}}]}),
        "a rest without size": rich.rich({"portions": [held_out(), {**tokens, "size": None}]}),
        "a program with no unit": rich.rich({"portions": [held_out(), {**tokens, "size": {"unit": "bytes",
                                                                                            "amount": 2000}}]},
                                            unit=None),
        "a third portion": rich.rich({"portions": [held_out(), tokens, {"name": "talk", "kinds": ["conversation"],
                                                                        "shares": [], "size": None}]}),
        "alone": rich.rich({"portions": [held_out()]}),
    }
    for name, request in variants.items():
        note, rows = run(rich, request, name.replace(" ", "-"))
        assert note["key"] != base_note["key"], name
        assert chosen(rows, "held_out") == first, name
    # and the same texts as a selection without portions that asks for what the first portion asks for
    alone = held_out()
    _, rows = run(rich, rich.rich({"kinds": alone["kinds"], "shares": alone["shares"], "size": alone["size"]}), "none")
    assert [(row["id"], row["piece"], row["revision"]) for row in rows] == first


@pytest.mark.parametrize("cut", [1, 3, 6])
def test_a_job_with_portions_cut_short_goes_on_to_the_same_delivery(rich, cut):
    request = rich.rich({"portions": [held_out(), {"name": "rest", "kinds": [], "shares": SHARES,
                                                   "size": {"unit": "bytes-of-text", "amount": 4000}}]}, per_file=2)
    whole, whole_rows = run(rich, request, "whole")
    assert len(whole["files"]) > cut
    saved = []

    def save(state):
        saved.append(state)
        if len(saved) == cut:
            raise Stop()
    work = rich.tmp_path / "work"
    with pytest.raises(Stop):
        Job(request, rich.depot.address, rich.dock, work, save=save).run()
    note = Job(request, rich.depot.address, rich.dock, work, checkpoint=saved[-1]).run()
    assert (note["key"], note["id"], note["result"], note["portions"]) == (
        whole["key"], whole["id"], whole["result"], whole["portions"])
    assert deliveries.read_items(rich.dock, note).to_pylist() == whole_rows
    assert [one["sha256"] for one in note["files"]] == [one["sha256"] for one in whole["files"]]


def test_the_canonical_form_and_the_key_cover_the_portions(freight):
    portions = [held_out(), {"name": "rest", "kinds": ["task", "plain", "plain"], "shares": [],
                             "size": {"unit": "tokens", "amount": 10}}]
    assert canonical({"portions": portions}) == {"portions": [
        {"name": "held_out", "kinds": ["plain"], "shares": [{"tag": "en", "share": 0.7}, {"tag": "de", "share": 0.3}],
         "size": {"unit": "bytes", "amount": 900}},
        {"name": "rest", "kinds": ["plain", "task"], "shares": [], "size": {"unit": "tokens", "amount": 10}}]}
    request = freight.request()
    keys = set()
    for selection in ({"portions": portions}, {"portions": portions[::-1]}, {"portions": portions[:1]},
                      {"portions": [{**portions[0], "name": "other"}, portions[1]]}, {"portions": []}):
        keys.add(api.key({**request, "selection": selection}, "1"))
    assert len(keys) == 5
    assert canonical({"portions": []}) == canonical({})
    for wrong, why in (({"portions": portions, "kinds": ["plain"]}, "not beside"),
                       ({"portions": [portions[0], portions[0]]}, "twice"),
                       ({"portions": [{**portions[0], "name": "a b"}]}, "name"),
                       ({"portions": {"name": "x"}}, "list")):
        with pytest.raises(Refused, match=why):
            canonical(wrong)
