"""A job run with the tests' program fake (conftest.py): a job cut short
goes on from its checkpoint to the delivery a job that never stopped
makes; the note keeps the program's file messages apart from the dock's
references; an erasure makes a delivery stale or is noted in it, as its
program says; the dock's cap never deletes what nothing can make
again or what another delivery needs; a size in bytes is measured by
the freight, whatever the program's unit; the rows of a part wait in local
files past a bound in memory and give the same part; and a program's `at`
cannot take the name of a field the freight fills."""

import pytest

from freight import api, deliveries
from freight import job
from freight.exchange import PreparationFailed
from freight.job import Job
from wpt_store import open_store
from wpt_store.names import File


class Stop(Exception):
    """A job cut short."""


def items(store, note) -> list[dict]:
    return deliveries.read_items(store, note).to_pylist()


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_a_job_cut_short_goes_on_from_its_checkpoint_to_the_same_delivery(freight, cut):
    request = freight.request()
    whole_dock = open_store((freight.tmp_path / "whole").as_uri())
    whole = Job(request, freight.depot.address, whole_dock, freight.tmp_path / "whole-work").run()
    assert len(whole["files"]) > cut  # a file is a checkpoint: the cut falls inside the job
    saved = []

    def save(state):
        saved.append(state)
        if len(saved) == cut:
            raise Stop()
    work = freight.tmp_path / "work"
    with pytest.raises(Stop):
        Job(request, freight.depot.address, freight.store, work, save=save).run()
    note = Job(request, freight.depot.address, freight.store, work, checkpoint=saved[-1]).run()
    assert (note["key"], note["id"], note["result"]) == (whole["key"], whole["id"], whole["result"])
    assert items(freight.store, note) == items(whole_dock, whole)
    assert [one["sha256"] for one in note["files"]] == [one["sha256"] for one in whole["files"]]


def test_the_note_keeps_the_programs_file_messages_apart_from_the_docks_references(freight):
    note = Job(freight.request(), freight.depot.address, freight.store, freight.tmp_path / "work").run()
    for one in note["files"]:
        said = one["said"]
        assert (said["bytes"], said["key"], said["sha256"]) == (1 << 20, "mine", "mine")
        assert one["key"] == f"deliveries/{note['key']}/files/{one['path']}" and one["sha256"] != "mine"
        assert one["bytes"] == freight.store.size(File.of(one))
    rows = items(freight.store, note)
    assert [row["line"] for row in rows] == list(range(len(rows)))
    assert sum(one["said"]["items"] for one in note["files"]) == len(rows) == note["result"]["lines"]


def test_an_erasure_makes_a_delivery_stale_or_is_noted_in_it_as_its_program_says(freight):
    where = {"home": freight.home, "depot": freight.depot.address, "dock": freight.dock}
    stale_request, noted_request = freight.request(), freight.request(on_erasure="note")
    stale = Job(stale_request, freight.depot.address, freight.store, freight.tmp_path / "a").run()
    noted = Job(noted_request, freight.depot.address, freight.store, freight.tmp_path / "b").run()
    assert api.check(**where)["erasures"] == []
    taken = items(freight.store, stale)[0]["id"]
    erased = freight.depot.ask_erasure("a word", [(taken, 0, 5)], freight.me)
    freight.depot.decide_erasure(erased, "admitted", freight.me)
    # a request asks the depot first: its delivery, stale now, is made again; the noted one is answered at once
    assert api.request(stale_request, **where)["state"] == "queued"
    answered = api.request(noted_request, **where)
    assert answered["state"] == "done" and answered["answer"]["erasures_since"] == [erased]
    assert api.delivery(stale["key"], dock=freight.dock)["stale"] == {"erasures": [erased]}
    assert "stale" not in api.delivery(noted["key"], dock=freight.dock)
    assert api.check(**where) | {"mark": None} == {"mark": None, "erasures": [], "stale": [], "noted": []}


def test_the_cap_never_deletes_an_adopted_delivery_nor_an_input(freight):
    dock = freight.store

    def put(key, made_at, inputs=None):
        file = dock.put_named(f"deliveries/{key}/files/f", b"x" * 100)
        dock.put_json(f"deliveries/{key}/delivery.json", {
            "key": key, "id": key, "files": [{**file.reference(), "path": "f", "bytes": 100}], "tables": {},
            "made_at": made_at, "preparation": {"inputs": inputs or {}}})
    put("adopted-v", "2000-01-01")  # the oldest, which nothing can make again
    put("input", "2000-01-02")  # named as an input by "user"
    put("asked", "2000-01-03")  # named as an input by the delivery being made
    put("old", "2000-01-04")
    put("user", "2000-01-05", {"vocabulary": "input"})
    room = deliveries.Room(dock, 500, "deliveries/new", inputs=["asked"])
    room.make(100)
    room.make(100)
    with pytest.raises(deliveries.NoRoom):
        room.make(100)
    assert [key for key in ("adopted-v", "input", "asked", "old", "user")
            if "deleted" in deliveries.of_key(dock, key)] == ["old", "user"]


def test_adopted_files_come_as_bytes_and_a_second_adoption_keeps_the_first(freight):
    note = api.adopt("adopted-x", {"vocabulary.json": b"{}"}, {"size": 0}, {"by": "a test"}, on_erasure="note",
                     dock=freight.dock)
    assert (note["on_erasure"], note["files"][0]["bytes"]) == ("note", 2)
    assert api.adopt("adopted-x", {"vocabulary.json": b"[]"}, {}, {}, dock=freight.dock)["id"] == note["id"]
    with pytest.raises(deliveries.DeliveryError):
        api.adopt("x", {}, {}, {}, dock=freight.dock)


def test_a_size_in_bytes_is_the_freights_own_measure_whatever_the_programs_unit(freight):
    # fake weighs a text by its bytes, which are those the listing gives a whole text: the same choice either way
    weighed = Job(freight.request(), freight.depot.address, freight.store, freight.tmp_path / "a").run()
    request = freight.request()
    request["selection"]["size"]["unit"] = "bytes"
    measured = Job(request, freight.depot.address, freight.store, freight.tmp_path / "b").run()
    assert measured["key"] != weighed["key"]
    rows = items(freight.store, measured)
    assert rows == items(freight.store, weighed) and [row["bytes"] for row in rows] == [row["weight"] for row in rows]
    assert [share["tag"] for share in measured["shares"]] == ["en", "de"]
    assert all(0 < share["weight"] <= part for share, part in zip(measured["shares"], (1050, 450)))  # 0.7, 0.3 of 1,500
    assert [share["bytes"] for share in measured["shares"]] == [
        sum(row["bytes"] for row in rows if "en" in row["tags"]), sum(row["bytes"] for row in rows if "de" in row["tags"])]


def test_the_rows_of_a_part_held_in_local_files_past_the_bound_give_the_same_part(freight, monkeypatch):
    request = freight.request(per_file=100)  # one file, so one part of every row
    held = Job(request, freight.depot.address, freight.store, freight.tmp_path / "a").run()
    monkeypatch.setattr(job, "SPILL_ROWS", 2)
    dock = open_store((freight.tmp_path / "spilled").as_uri())
    spilled = Job(request, freight.depot.address, dock, freight.tmp_path / "b").run()
    assert len(held["tables"]["items"]) == 1 and held["tables"]["items"][0]["rows"] > 2
    assert (spilled["key"], spilled["id"], spilled["tables"]["items"][0]["fingerprint"]) == (
        held["key"], held["id"], held["tables"]["items"][0]["fingerprint"])
    assert items(dock, spilled) == items(freight.store, held)
    assert not list((freight.tmp_path / "b").glob("items-spill-*"))


@pytest.mark.parametrize("name", ["bytes", "weight", "id"])
def test_an_at_field_named_as_a_field_the_freight_fills_fails_the_job(freight, name):
    with pytest.raises(PreparationFailed, match=f"at fields {name}"):
        Job(freight.request(at_name=name), freight.depot.address, freight.store, freight.tmp_path / "work").run()
