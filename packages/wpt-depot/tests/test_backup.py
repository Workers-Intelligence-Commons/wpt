from pathlib import Path

import pyarrow.fs as pafs
import pytest
from deltalake.exceptions import DeltaError

from wpt_depot import Depot, NewText, backup, restore
from wpt_depot.errors import BackupCut, DepotExists, JournalUnfinished

READ = "SELECT id, revision, text, decisions, libraries FROM texts ORDER BY id"


def plain(text):
    return NewText(text=text, licenses=("CC0-1.0",))


def files_holding(folder, passage: str) -> list[Path]:
    return [path for path in Path(folder).rglob("*") if path.is_file() and passage.encode() in path.read_bytes()]


def test_a_backup_opens_as_a_depot_reading_the_same_rows_and_older_states(make_depot, me, tmp_path):
    depot = make_depot()
    texts = depot.add([plain("one"), plain("two")], me)
    depot.decide("admitted", texts, me)
    depot.put(depot.new_library("all"), texts)

    copy = backup(depot, str(tmp_path / "backups"))

    assert Path(copy.address).parent == tmp_path / "backups"
    assert Path(copy.address).name.startswith("depot-") and Path(copy.address).name.endswith(f"-change{depot.change()}")
    assert Depot.open(copy.address).sql(READ).to_pylist() == depot.sql(READ).to_pylist()
    assert copy.tables.table("texts").history() == depot.tables.table("texts").history()


def test_a_change_half_written_during_the_backup_is_not_in_it(make_depot, me, tmp_path, monkeypatch):
    # Delta Lake writes one table at a time: the refusal revises the text before it ends its membership
    depot = make_depot()
    texts = depot.add([plain("one")], me)
    depot.decide("admitted", texts, me)
    depot.put(depot.new_library("all"), texts)
    settled = depot.sql(READ).to_pylist()

    def fails(*args, **kwargs):
        raise OSError("the disk went away")

    with monkeypatch.context() as patch:
        patch.setattr(Depot, "_leave", fails)
        with pytest.raises(JournalUnfinished):
            depot.decide("refused", texts, me)
    copy = backup(Depot.view(depot.address), str(tmp_path / "backups"))

    assert Depot.view(copy.address).journal.pending() == []
    assert Depot.open(copy.address).sql(READ).to_pylist() == settled


def test_an_erasure_carried_out_after_a_backup_leaves_the_backup_untouched(make_depot, me, tmp_path):
    depot = make_depot()
    [text] = depot.add([plain("a secret passage")], me)
    copy = backup(depot, str(tmp_path / "backups"))
    held = copy.sql(READ).to_pylist()

    depot.decide_erasure(depot.ask_erasure("asked", [(text, 2, 8)], me), "admitted", me)

    assert files_holding(depot.address, "secret") == []
    assert files_holding(copy.address, "secret") != []
    assert Depot.open(copy.address).sql(READ).to_pylist() == held


def test_a_restore_reads_as_the_backup_did_and_never_overwrites(make_depot, me, tmp_path):
    depot = make_depot()
    [text] = depot.add([plain("a secret passage")], me)
    copy = backup(depot, str(tmp_path / "backups"))
    held = copy.sql(READ).to_pylist()
    depot.decide_erasure(depot.ask_erasure("asked", [(text, 2, 8)], me), "admitted", me)
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "note.txt").write_text("mine", encoding="utf-8")

    restored = restore(copy.address, str(tmp_path / "restored"))

    assert Depot.open(restored.address).sql(READ).to_pylist() == held
    with pytest.raises(DepotExists):
        restore(copy.address, str(tmp_path / "taken"))
    assert [path.name for path in (tmp_path / "taken").iterdir()] == ["note.txt"]


def test_a_backup_cut_short_before_its_last_about_file_is_no_depot(make_depot, me, tmp_path, monkeypatch):
    depot = make_depot()
    depot.edit_about(maintainer="Bob")
    copied = pafs.copy_files

    def cut(source, destination, **kwargs):
        if destination.endswith(f"about/_delta_log/{0:020d}.json"):
            raise OSError("the disk went away")
        return copied(source, destination, **kwargs)

    monkeypatch.setattr(pafs, "copy_files", cut)
    with pytest.raises(BackupCut):
        backup(depot, str(tmp_path / "backups"))

    [begun] = (tmp_path / "backups").iterdir()
    with pytest.raises(DeltaError):  # its about log lacks its first entry
        Depot.view(str(begun))
