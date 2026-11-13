import hashlib
import shutil
from pathlib import Path

import apsw
import pytest

from wpt_depot import About, Depot, NewText, clone, upgrade
from wpt_depot.errors import DepotExists, OtherFormat, Refused
from wpt_depot.legacy import Legacy
from wpt_depot.schema import FORMAT, KEYS


def files_of(folder: Path) -> dict[str, str]:
    """{path under folder: sha256 of its bytes} of every file under folder."""
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def table_rows(path) -> dict[str, list[tuple]]:
    """Every row of every table of the file in the order of its key, the time of the upgrade's own change aside."""
    con = apsw.Connection(f"file:{Path(path).as_posix()}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    try:
        found = {table: con.execute(f"SELECT * FROM {table} ORDER BY {', '.join(keys)}").fetchall()
                 for table, keys in KEYS.items()}
    finally:
        con.close()
    found["changes"] = [(number, kind, None if kind == "upgrade" else made_at)
                        for number, kind, made_at in found["changes"]]
    return found


def test_a_new_depot_and_a_clone_are_of_the_code_s_format(make_depot, me, serve, tmp_path):
    upstream = make_depot("upstream")
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace", "grace@example.org"))

    assert upstream.format() == FORMAT and follower.format() == FORMAT


def test_an_upgrade_of_a_format_3_folder_writes_a_new_file_holding_what_the_folder_held(make_legacy, me):
    folder = make_legacy(3)
    before = files_of(folder)

    depot, start = Depot.upgrade(str(folder))

    assert start == 3 and depot.format() == FORMAT and depot.address == str(folder) + ".sqlite"
    assert files_of(folder) == before  # the folder is never changed by the step to format 4
    shown = depot.show("old-1")
    assert (shown["text"]["text"], shown["status"], shown["notice"]) == ("hello world", "admitted", "a notice")
    assert [one["id"] for one in shown["libraries"]] == ["old-1", "old-2"] and shown["revisions"] == 2
    assert [one["revision"] for one in depot.library_texts("old-2", depot.change())] == [2]  # the revision it pinned
    assert depot.sql("SELECT id, revision FROM texts WHERE EXISTS (SELECT 1 FROM json_each(tags) WHERE value = "
                     "'book')") == [{"id": "old-1", "revision": 2}]
    assert [one["format"] for one in depot.sql("SELECT format FROM about_revisions ORDER BY revision")] == [2, 3, 4]
    assert depot.erasures()[0]["needles"][0]["text"] == "old-2"
    # the ids go on from the highest number of each kind the folder held
    assert depot.add([NewText(text="new", licenses=("CC0-1.0",))], me) == ["old-3"]
    assert depot.new_library("next") == "old-3"


@pytest.mark.parametrize("start", [0, 1, 2])
def test_an_upgrade_of_an_older_folder_runs_each_step_and_reaches_the_same_rows(make_legacy, start):
    folder = make_legacy(start)

    depot, found = Depot.upgrade(str(folder))

    assert found == start and depot.format() == FORMAT
    assert depot.content("old-1") == "hello world" and depot.show("old-1")["text"]["scores"] is None
    assert depot.sql("SELECT format FROM about")[0]["format"] == FORMAT


def test_an_upgrade_cut_short_goes_on_when_run_again_and_writes_what_one_run_writes(make_legacy, monkeypatch,
                                                                                     tmp_path):
    folder = make_legacy(3)
    shutil.copytree(folder, tmp_path / "copy")
    whole, _ = Depot.upgrade(str(tmp_path / "copy"))
    real, calls = upgrade._insert, []

    def stops(*args):
        calls.append(args)
        if len(calls) == 5:
            raise OSError("the machine stopped")
        return real(*args)

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "_insert", stops)
        with pytest.raises(OSError):
            Depot.upgrade(str(folder))
    assert not Path(str(folder) + ".sqlite").exists()

    depot, _ = Depot.upgrade(str(folder))

    assert table_rows(depot.address) == table_rows(whole.address)


def test_an_upgrade_stopped_once_its_file_is_whole_only_renames_it_when_run_again(make_legacy, monkeypatch):
    folder = make_legacy(3)
    real = Path.rename

    def stops(self, target):
        if self.name.endswith(".upgrading"):
            raise OSError("the machine stopped")
        return real(self, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", stops)
        with pytest.raises(OSError):
            Depot.upgrade(str(folder))

    depot, _ = Depot.upgrade(str(folder))

    assert [one["kind"] for one in depot.sql("SELECT kind FROM changes WHERE kind = 'upgrade'")] == ["upgrade"]
    assert [one["format"] for one in depot.sql("SELECT format FROM about_revisions ORDER BY revision")] == [2, 3, 4]


def test_the_upgrade_of_a_clone_takes_its_upstream_s_web_address(make_legacy):
    folder = make_legacy(3, upstream="s3://bucket/upstream")

    with pytest.raises(Refused, match="--upstream"):
        upgrade.upgrade(str(folder))
    assert not Path(str(folder) + ".sqlite").exists() and not Path(str(folder) + ".sqlite.upgrading").exists()
    depot, _ = Depot.upgrade(str(folder), upstream="https://depot.example.org/")

    assert depot.upstream() == "https://depot.example.org"


def test_an_upgrade_never_writes_over_a_file(make_legacy):
    folder = make_legacy(3)
    taken = Path(str(folder) + ".sqlite")
    taken.write_text("mine", encoding="utf-8")

    with pytest.raises(DepotExists):
        Depot.upgrade(str(folder))
    assert taken.read_text(encoding="utf-8") == "mine"


def test_a_folder_of_an_older_format_is_refused_with_the_command_that_upgrades_it(make_legacy):
    folder = make_legacy(3)

    for way_in in (Depot.view, Depot.open):
        with pytest.raises(OtherFormat) as refused:
            way_in(str(folder))
        assert str(refused.value) == (f"{folder} is a depot of format 3, older than the format {FORMAT} this wpt "
                                      f"reads: run `wpt depot upgrade` to bring it up to format {FORMAT}")
    assert Legacy(folder).format() == 3


def test_a_depot_of_a_newer_format_is_refused_everywhere_and_never_brought_down(make_depot):
    depot = make_depot()
    with depot.db.write("test") as (con, _):
        con.execute("UPDATE about_revisions SET format = ?", (FORMAT + 1,))
    depot.db.close()

    for way_in in (Depot.view, Depot.open, Depot.upgrade):
        with pytest.raises(OtherFormat) as refused:
            way_in(depot.address)
        assert str(refused.value) == (f"{depot.address} is a depot of format {FORMAT + 1}, newer than the format "
                                      f"{FORMAT} this wpt reads: update wpt to read it; a depot is never brought "
                                      "down to an older format")
