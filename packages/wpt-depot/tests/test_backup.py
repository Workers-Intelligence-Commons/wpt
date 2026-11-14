from pathlib import Path

import apsw
import pytest
from wpt_store import open_store

from wpt_depot import Depot, NewText, backup, restore
from wpt_depot.database import PENDING
from wpt_depot.errors import BackupRefused, DepotExists
from wpt_depot.schema import KEYS

READ = "SELECT id, revision, text, decisions, libraries FROM texts ORDER BY id"


def plain(text):
    return NewText(text=text, licenses=("CC0-1.0",))


def table_rows(path) -> dict[str, list[tuple]]:
    """Every row of every table of the file, in the order of the table's key: what a depot holds, its internal
    row numbers aside."""
    con = apsw.Connection(f"file:{Path(path).as_posix()}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    try:
        return {table: con.execute(f"SELECT * FROM {table} ORDER BY {', '.join(keys)}").fetchall()
                for table, keys in KEYS.items()}
    finally:
        con.close()


@pytest.fixture
def store(tmp_path):
    return open_store((tmp_path / "store").as_uri())


def test_a_restore_holds_every_row_the_depot_held_after_a_full_copy_and_its_changes(make_depot, me, tmp_path,
                                                                                        store):
    depot = make_depot()
    texts = depot.add([plain("one"), plain("two")], me)
    first = backup(depot, store)
    depot.decide("admitted", texts, me)
    depot.put(depot.new_library("all"), texts)
    second = backup(depot, store)

    restored = restore(store, str(tmp_path / "restored.sqlite"))

    assert first["full"] is not None and second["full"] is None and len(second["changes"]) == 3
    assert table_rows(restored.address) == table_rows(depot.address)
    assert list(depot.db.waiting.glob("*.changeset")) == []


def test_a_restore_at_a_change_number_holds_the_depot_as_it_was_then(make_depot, me, tmp_path, store):
    depot = make_depot()
    texts = depot.add([plain("one")], me)
    backup(depot, store)
    held = depot.sql(READ)
    then = depot.change()
    depot.decide("refused", texts, me)
    backup(depot, store)

    restored = restore(store, str(tmp_path / "then.sqlite"), at=str(then))

    assert restored.sql(READ) == held
    assert restored.change() == then


def test_a_restored_depot_backs_up_on_a_timeline_of_its_own(make_depot, me, tmp_path, store):
    # the restored depot numbers its changes on from an earlier one, as its origin did: the restore never mixes them
    depot = make_depot()
    depot.add([plain("text 1")], me)
    early = depot.change()
    backup(depot, store)
    depot.add([plain("text 2")], me)
    depot.add([plain("text 3")], me)
    backup(depot, store)

    branch = restore(store, str(tmp_path / "branch.sqlite"), at=str(early))
    branch.add([plain("only in the branch")], me)
    backup(branch, store)
    latest = restore(store, str(tmp_path / "latest.sqlite"))

    assert branch.change() < depot.change()
    assert table_rows(latest.address) == table_rows(branch.address)


def test_a_restore_keeps_no_copy_of_a_passage_erased_after_the_full_copy(make_depot, me, tmp_path, store):
    depot = make_depot()
    [text] = depot.add([plain("alpha secret beta " * 50)], me)
    backup(depot, store)
    depot.decide_erasure(depot.ask_erasure("asked", [(text, 6, 12)], me), "admitted", me)
    backup(depot, store)

    restored = restore(store, str(tmp_path / "restored.sqlite"))
    restored.db.close()

    assert table_rows(restored.address) == table_rows(depot.address)
    assert b"secret" not in Path(restored.address).read_bytes()


def test_a_restore_without_a_change_of_the_chain_is_refused_and_writes_nothing(make_depot, me, tmp_path, store):
    depot = make_depot()
    depot.add([plain("one")], me)
    backup(depot, store)
    depot.add([plain("two")], me)
    lost = depot.change()
    depot.add([plain("three")], me)
    [missing] = [path for path in depot.db.waiting_files() if path.name.startswith(f"{lost:012d}-")]
    missing.unlink()
    backup(depot, store)

    with pytest.raises(BackupRefused):
        restore(store, str(tmp_path / "restored.sqlite"))
    assert not (tmp_path / "restored.sqlite").exists()


def test_a_restore_never_overwrites_a_file(make_depot, me, tmp_path, store):
    depot = make_depot()
    depot.add([plain("one")], me)
    backup(depot, store)
    taken = tmp_path / "taken.sqlite"
    taken.write_text("mine", encoding="utf-8")

    with pytest.raises(DepotExists):
        restore(store, str(taken))
    assert taken.read_text(encoding="utf-8") == "mine"


def test_a_pending_change_file_stays_when_its_change_is_in_the_depot_and_goes_otherwise(make_depot, me):
    # a crash between writing the change file and the commit, or between the commit and the rename
    depot = make_depot()
    depot.add([plain("one")], me)
    [held] = [path for path in depot.db.waiting_files() if path.name.startswith(f"{depot.change():012d}-")]
    kept = held.rename(held.with_name(held.name.removesuffix(".changeset") + PENDING))
    never = depot.db.waiting / f"{depot.change() + 1:012d}-{depot.change():012d}-20260101T000000.000000Z{PENDING}"
    never.write_bytes(b"never committed")

    again = Depot.open(depot.address)

    assert held.exists() and not kept.exists() and not never.exists()
    assert again.change() == depot.change()


def test_a_depot_backed_up_by_nothing_records_no_change_file_until_its_next_backup_sends_a_full_copy(
        make_depot, me, tmp_path, store):
    depot = make_depot()
    texts = depot.add([plain("one")], me)
    assert depot.db.waiting_files()
    depot.db.backup_off()
    assert depot.db.waiting_files() == []  # no backup will send them
    depot.decide("admitted", texts, me)
    depot.add([plain("two")], me)
    assert depot.db.waiting_files() == []

    record = backup(depot, store)
    assert record["full"] is not None and record["full"]["change"] == depot.change()
    assert depot.db.recording()
    depot.add([plain("three")], me)
    assert len(depot.db.waiting_files()) == 1
    backup(depot, store)
    restored = restore(store, str(tmp_path / "restored.sqlite"))
    assert table_rows(restored.db.path) == table_rows(depot.db.path)
