import sys
from pathlib import Path

import apsw
import pytest
from depot_app import DockBackups

from depot import Depot, NewText, backup, restore, upgrade
from depot.database import FULL_COPY_NEXT, PENDING
from depot.errors import BackupRefused, DepotExists, OtherFormat
from depot.filters import Filter
from depot.schema import KEYS

READ = "SELECT id, revision, text, decisions FROM texts ORDER BY id"


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
def dock(tmp_path):
    return DockBackups((tmp_path / "dock").as_uri())


def test_a_restore_holds_every_row_the_depot_held_after_a_full_copy_and_its_changes(make_depot, me, tmp_path,
                                                                                        dock):
    depot = make_depot()
    texts = depot.add([plain("one"), plain("two")], me)
    first = backup(depot, dock)
    depot.decide("admitted", texts, me)
    depot.retag(texts, ["all"], [], me)
    depot.edit_credits(texts[0], {"title": "One"}, me)
    second = backup(depot, dock)

    restored = restore(dock, str(tmp_path / "restored.sqlite"))

    assert first["full"] is not None and second["full"] is None and len(second["changes"]) == 3
    assert table_rows(restored.address) == table_rows(depot.address)
    assert list(depot.db.waiting.glob("*.changeset")) == []


def test_a_restore_holds_the_taggings_of_tags_given_and_taken_after_the_full_copy(make_depot, me, tmp_path, dock):
    depot = make_depot()
    texts = depot.add([NewText(text="one", licenses=("CC0-1.0",), tags=("a",)), plain("two")], me)
    backup(depot, dock)
    depot.retag(texts, ["b"], ["a"], me)
    depot.edit_credits_many([{"id": texts[1], "tags": ["c"]}], me)  # b, given by the retag, taken back
    at = depot.change()
    depot.retag(texts, ["d"], [], me)
    backup(depot, dock)

    restored = restore(dock, str(tmp_path / "restored.sqlite"))
    then = restore(dock, str(tmp_path / "then.sqlite"), at=str(at))

    assert table_rows(restored.address)["taggings"] == table_rows(depot.address)["taggings"]
    assert [restored.show(one)["text"]["tags"] for one in texts] == [["b", "d"], ["c", "d"]]
    assert [then.show(one)["text"]["tags"] for one in texts] == [["b"], ["c"]]
    assert then.search(Filter(tags=("d",)))["rows"] == []


def test_a_restore_holds_the_title_index_the_depot_held(make_depot, me, tmp_path, dock, title_index):
    # the index is an FTS5 table, whose own tables the change files carry like any other
    depot = make_depot()
    texts = depot.add([NewText(text=n, licenses=("CC0-1.0",), title=f"Old title {n}") for n in ("one", "two")], me)
    backup(depot, dock)
    depot.edit_credits_many([{"id": texts[0], "title": "New name"}, {"id": texts[1], "title": None}], me)
    depot.add([NewText(text="three", licenses=("CC0-1.0",), title="Another name")], me)
    backup(depot, dock)

    restored = restore(dock, str(tmp_path / "restored.sqlite"))

    assert title_index(restored) == title_index(depot) and title_index(restored)["unfound"] == []
    assert [one["id"] for one in restored.search(Filter(search="name"))["rows"]] == [texts[0], "depot-3"]
    assert restored.search(Filter(search="old"))["rows"] == []


def test_a_restore_at_a_change_number_holds_the_depot_as_it_was_then(make_depot, me, tmp_path, dock):
    depot = make_depot()
    texts = depot.add([plain("one")], me)
    backup(depot, dock)
    held = depot.sql(READ)
    then = depot.change()
    depot.decide("refused", texts, me)
    backup(depot, dock)

    restored = restore(dock, str(tmp_path / "then.sqlite"), at=str(then))

    assert restored.sql(READ) == held
    assert restored.change() == then


def test_a_restored_depot_backs_up_on_a_timeline_of_its_own(make_depot, me, tmp_path, dock):
    # the restored depot numbers its changes on from an earlier one, as its origin did: the restore never mixes them
    depot = make_depot()
    depot.add([plain("text 1")], me)
    early = depot.change()
    backup(depot, dock)
    depot.add([plain("text 2")], me)
    depot.add([plain("text 3")], me)
    backup(depot, dock)

    branch = restore(dock, str(tmp_path / "branch.sqlite"), at=str(early))
    branch.add([plain("only in the branch")], me)
    backup(branch, dock)
    latest = restore(dock, str(tmp_path / "latest.sqlite"))

    assert branch.change() < depot.change()
    assert table_rows(latest.address) == table_rows(branch.address)


def conversation_schema(path) -> str:
    con = apsw.Connection(f"file:{Path(path).as_posix()}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    try:
        return con.execute("SELECT schema FROM kinds WHERE id = 'conversation'").fetchall()[0][0]
    finally:
        con.close()


def test_the_step_to_format_15_is_a_change_the_backups_carry_and_restore(make_depot, me, tmp_path, dock, monkeypatch):
    # the step changes no layout, so a depot backed up onto the dock goes on with change files: its next backup
    # sends no full copy of a file that may weigh terabytes, and its backups still give the depot back as it is after
    # the step, at format 15 with the new conversation schema, and as it was before it, at format 14, for the wpt of
    # that format
    depot = make_depot()
    depot.add([plain("one")], me)
    with depot.db.bulk() as con:  # the depot as a depot of format 14 holds it: the conversation schema of format 14
        con.execute("UPDATE kinds SET schema = ? WHERE id = 'conversation'", (upgrade.CONVERSATION_12,))
        con.execute("UPDATE about_revisions SET format = 14")
    assert backup(depot, dock)["full"] is not None
    depot.add([plain("two")], me)
    before, at_14, timeline = depot.change(), table_rows(depot.address), depot.db.timeline()
    depot.db.close()

    upgraded, start = upgrade.upgrade(depot.address, log=lambda line: None)

    assert start == 14 and upgraded.format() == 15 and not (upgraded.db.waiting / FULL_COPY_NEXT).exists()
    sent = backup(upgraded, dock)
    assert sent["full"] is None and sent["timeline"] == timeline
    assert [one["number"] for one in sent["changes"]] == [before, upgraded.change()]
    restored = restore(dock, str(tmp_path / "after.sqlite"))
    assert table_rows(restored.address) == table_rows(upgraded.address)
    assert restored.format() == 15 and conversation_schema(restored.address) == upgrade.CONVERSATION_15
    with pytest.raises(OtherFormat, match="restore it with the wpt of format 14"):
        restore(dock, str(tmp_path / "before.sqlite"), at=str(before))
    assert list(tmp_path.glob("before.sqlite*")) == []

    # the wpt of format 14 restores that moment: the same chain, its format and its opening those of format 14
    restoring = sys.modules["depot.backup"]
    monkeypatch.setattr(restoring, "FORMAT", 14)
    monkeypatch.setattr(restoring, "Depot", type("Format14", (), {"open": staticmethod(str)}))
    restore(dock, str(tmp_path / "before.sqlite"), at=str(before))
    assert table_rows(tmp_path / "before.sqlite") == at_14
    assert conversation_schema(tmp_path / "before.sqlite") == upgrade.CONVERSATION_12


def test_a_restore_keeps_no_copy_of_a_passage_erased_after_the_full_copy(make_depot, me, tmp_path, dock):
    depot = make_depot()
    [text] = depot.add([plain("alpha secret beta " * 50)], me)
    backup(depot, dock)
    depot.decide_erasure(depot.ask_erasure("asked", [(text, 6, 12)], me), "admitted", me)
    backup(depot, dock)

    restored = restore(dock, str(tmp_path / "restored.sqlite"))
    restored.db.close()

    assert table_rows(restored.address) == table_rows(depot.address)
    assert b"secret" not in Path(restored.address).read_bytes()


def test_a_restore_without_a_change_of_the_chain_is_refused_and_writes_nothing(make_depot, me, tmp_path, dock):
    depot = make_depot()
    depot.add([plain("one")], me)
    backup(depot, dock)
    depot.add([plain("two")], me)
    lost = depot.change()
    depot.add([plain("three")], me)
    [missing] = [path for path in depot.db.waiting_files() if path.name.startswith(f"{lost:012d}-")]
    missing.unlink()
    backup(depot, dock)

    with pytest.raises(BackupRefused):
        restore(dock, str(tmp_path / "restored.sqlite"))
    assert not (tmp_path / "restored.sqlite").exists()


def test_a_restore_never_overwrites_a_file(make_depot, me, tmp_path, dock):
    depot = make_depot()
    depot.add([plain("one")], me)
    backup(depot, dock)
    taken = tmp_path / "taken.sqlite"
    taken.write_text("mine", encoding="utf-8")

    with pytest.raises(DepotExists):
        restore(dock, str(taken))
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
        make_depot, me, tmp_path, dock):
    depot = make_depot()
    texts = depot.add([plain("one")], me)
    assert depot.db.waiting_files()
    depot.db.backup_off()
    assert depot.db.waiting_files() == []  # no backup will send them
    depot.decide("admitted", texts, me)
    depot.add([plain("two")], me)
    assert depot.db.waiting_files() == []

    record = backup(depot, dock)
    assert record["full"] is not None and record["full"]["change"] == depot.change()
    assert depot.db.recording()
    depot.add([plain("three")], me)
    assert len(depot.db.waiting_files()) == 1
    backup(depot, dock)
    restored = restore(dock, str(tmp_path / "restored.sqlite"))
    assert table_rows(restored.db.path) == table_rows(depot.db.path)
