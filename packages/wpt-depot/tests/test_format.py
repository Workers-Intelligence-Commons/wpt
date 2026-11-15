import hashlib
import re
import shutil
from pathlib import Path

import apsw
import pytest

from wpt_depot import About, Depot, NewText, clone, counts, ids, upgrade
from wpt_depot.errors import DepotExists, HalfUpgraded, OtherFormat, Refused
from wpt_depot.legacy import Legacy
from wpt_depot.database import FULL_COPY_NEXT
from wpt_depot.schema import FORMAT, KEYS, VIEWS


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
    assert [one["format"] for one in depot.sql("SELECT format FROM about_revisions ORDER BY revision")] == [2, 3, 4, 5, 6, 7]
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

    assert [one["kind"] for one in depot.sql("SELECT kind FROM changes WHERE kind = 'upgrade'")] == ["upgrade"] * 4
    assert [one["format"] for one in depot.sql("SELECT format FROM about_revisions ORDER BY revision")] == [2, 3, 4, 5, 6, 7]


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


def schema_of(path) -> list[tuple]:
    """Every table and index of the file, with the statement that makes it, its spacing made one space."""
    con = apsw.Connection(f"file:{Path(path).as_posix()}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    try:
        return [(kind, name, sql and re.sub(r"\s+", " ", sql)) for kind, name, sql in
                con.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()]
    finally:
        con.close()


def made_older(depot, format_: int):
    """The depot as a depot of format_ (4, 5 or 6) holds what it holds: no export keys, no indexes
    memberships_key and pins_key and no vocabulary of the datasets; for 5 and 4 also each text's decisions as the JSON list of its current revision, no
    status, text_decisions, counts, tokenizer recipes, vocabularies or max_tokens; for 4 also no index texts_order.
    Its rows before (table_rows), with the status of the revisions that are not current, which no format records,
    left out."""
    with depot.db.bulk() as con:
        con.execute("UPDATE texts_revisions SET status = NULL WHERE NOT latest")
    before = table_rows(depot.address)
    with depot.db.bulk() as con:
        for view in VIEWS:  # the connection's own views read the columns
            con.execute(f"DROP VIEW temp.{view}")
        for index in ("memberships_key", "pins_key"):
            con.execute(f"DROP INDEX {index}")
        for table in ("memberships", "pins"):
            con.execute(f"ALTER TABLE {table} DROP COLUMN export_key")
        con.execute("ALTER TABLE datasets_revisions DROP COLUMN vocabulary")
        con.execute("UPDATE about_revisions SET format = ?", (format_,))
    if format_ == 6:
        depot.db.close()
        (depot.db.waiting / FULL_COPY_NEXT).unlink()
        return before
    with depot.db.bulk() as con:
        con.execute("UPDATE texts_revisions SET decisions = (SELECT json_group_array(l.decision ORDER BY "
                    "length(l.decision), l.decision) FROM text_decisions l WHERE l.text = texts_revisions.id) "
                    "WHERE latest")
        con.execute("DROP INDEX texts_status")
        con.execute("ALTER TABLE texts_revisions DROP COLUMN status")
        con.execute("ALTER TABLE datasets_revisions DROP COLUMN max_tokens")
        for table in ("text_decisions", "counts", "tokenizers_revisions", "vocabularies", "vocabulary_texts"):
            con.execute(f"DROP TABLE {table}")
        if format_ == 4:
            con.execute("DROP INDEX texts_order")
        con.execute("UPDATE about_revisions SET format = ?", (format_,))
    depot.db.close()
    (depot.db.waiting / FULL_COPY_NEXT).unlink()
    return before


def decided(depot, me):
    """A depot whose texts have decisions of every kind: admitted, refused, and started over by a license change."""
    depot.add_license("MIT", "MIT License", "text")
    texts = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",), tags=("a",) if n % 2 else ())
                       for n in range(12)], me)
    depot.decide("admitted", texts[:8], me)
    depot.decide("refused", texts[6:8], me)
    depot.relicense(texts[:2], ["MIT"], me)
    depot.put(depot.new_library("first"), texts[2:6])
    return depot


def test_an_upgrade_of_a_format_4_file_builds_its_indexes_tables_and_statuses_in_place_and_keeps_every_row(
        make_depot, me):
    new = make_depot("new")
    depot = decided(make_depot(), me)
    before = made_older(depot, 4)
    with pytest.raises(OtherFormat):
        Depot.open(depot.address)
    said = []

    upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 4 and upgraded.format() == FORMAT and upgraded.address == depot.address
    assert schema_of(depot.address) == schema_of(new.address)
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [4, 5, 6, 7] and after["changes"][-1][1] == "upgrade"
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    assert (depot.db.waiting / FULL_COPY_NEXT).exists()  # the change files hold no change of layout
    assert "texts_order" in said[0] and "12 current texts" in said[0]
    assert [one["id"] for one in upgraded.search("true", limit=3)["rows"]] == ["depot-1", "depot-2", "depot-3"]
    assert upgraded.count("true")["statuses"] == {"under review": 6, "admitted": 4, "refused": 2}

    _, start = upgrade.upgrade(depot.address, log=said.append)
    assert start == FORMAT and table_rows(depot.address) == after


def test_an_upgrade_to_format_6_cut_short_marks_the_depot_and_goes_on_when_run_again(make_depot, me, monkeypatch):
    # the step runs in several transactions, so a stop between two leaves a depot only the upgrade reads
    depot = decided(make_depot(), me)
    before = made_older(depot, 5)
    real = counts.recount

    def stops(con):
        raise OSError("the machine stopped")

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 5)
        patch.setattr(counts, "recount", stops)
        with pytest.raises(OSError):
            upgrade.upgrade(depot.address, log=lambda line: None)
    for way_in in (Depot.view, Depot.open):  # at format 6 half-way, so older than this code's too
        with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
            way_in(depot.address)
    assert counts.recount is real
    said = []

    upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 5 and upgraded.format() == FORMAT
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [5, 6, 7]
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    assert any("at format 6" in line for line in said)


def test_an_upgrade_to_format_7_writes_every_export_key_goes_on_from_its_last_batch_and_builds_the_indexes(
        make_depot, me, monkeypatch):
    depot = decided(make_depot(), me)
    held = depot.rows("SELECT id FROM texts WHERE status = 'admitted' ORDER BY id")
    locked = depot.new_library("locked")
    depot.put(locked, [one["id"] for one in held])
    depot.lock(locked)  # 8 memberships (two libraries of 4 texts) and 4 pins
    before = made_older(depot, 6)
    real = ids.export_key
    made = []

    def stops_in_the_second_batch_of_memberships(text):
        made.append(text)
        if len(made) == 5 + 1:  # a first batch of 5 memberships
            raise OSError("the machine stopped")
        return real(text)

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 5)
        patch.setattr(ids, "export_key", stops_in_the_second_batch_of_memberships)
        with pytest.raises(OSError):
            upgrade.upgrade(depot.address, log=lambda line: None)
    with pytest.raises(HalfUpgraded, match="wpt depot upgrade"):
        Depot.open(depot.address)
    said = []

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 5)
        patch.setattr(ids, "export_key", lambda text: made.append(text) or real(text))
        upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 6 and upgraded.format() == FORMAT
    # the first batch of memberships written before the stop is kept: it goes on in memberships
    assert "rows 5 to 8" in said[0] and "of memberships" in said[0]
    assert len(made) == 6 + 3 + 4  # the 3 memberships left, then the 4 pins
    assert any("rows a second" in line for line in said)
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [6, 7]
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    for table, count in (("memberships", 8), ("pins", 4)):
        keys = upgraded.rows(f"SELECT text, export_key FROM {table}")
        assert len(keys) == count and all(one["export_key"] == real(one["text"]) for one in keys)
    assert {"memberships_key", "pins_key"} <= {name for _, name, _ in schema_of(depot.address)}
    joined = upgraded.new_library("after")
    upgraded.put(joined, [held[0]["id"]])
    assert upgraded.rows("SELECT export_key FROM memberships WHERE library = ?", (joined,)) == [
        {"export_key": ids.export_key(held[0]["id"])}]
