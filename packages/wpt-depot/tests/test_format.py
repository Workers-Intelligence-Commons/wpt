import hashlib
import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from wpt_depot import About, Depot, NewText, backup, clone, format3, parts, pull, restore
from wpt_depot import depot as depots
from wpt_depot.errors import JournalUnfinished, OtherFormat
from wpt_depot.schema import FORMAT, STEPS
from wpt_depot.scoring import lines
from wpt_depot.tables import Tables

ABOUT = "SELECT * EXCLUDE (format) FROM about_revisions ORDER BY revision"


def at_format_2(depot):
    """The depot as format 2 lays it out: each row of contents holding its
    version's whole text (text, or an item's content), no parts table, and
    every about row of format 2."""
    found = depot.tables.query("SELECT id, version, change, text, kind, content FROM contents")
    schema = pa.schema([pa.field("id", pa.string(), False), pa.field("version", pa.int64(), False),
                        pa.field("change", pa.int64(), False), pa.field("text", pa.string()),
                        pa.field("kind", pa.string()), pa.field("content", pa.string())])
    found = found.cast(schema)
    depot.tables.rewrite("contents", found.slice(0, 1))
    for at in range(1, found.num_rows):  # a data file for each row, as appends at format 2 left them
        depot.tables.append("contents", found.slice(at, 1))
    depot.tables.table("contents").alter.add_constraint({format3.CONSTRAINT_2: (
        "(kind IS NULL AND text IS NOT NULL AND content IS NULL) "
        "OR (kind IS NOT NULL AND text IS NULL AND content IS NOT NULL)")})
    depot.tables.drop_history(["contents"])
    shutil.rmtree(Path(depot.address) / "parts")
    at_format(depot, 2)


def at_format_1(depot):
    """The depot as format 1 lays it out: format 2's layout, its texts table
    without the column scores, and every about row of format 1."""
    at_format_2(depot)
    depot.tables.rewrite("texts", depot.tables.read("texts").drop_columns(["scores"]))
    at_format(depot, 1)


def at_format_0(depot):
    """The depot as format 0 lays it out: format 1's layout, its about table
    without the column format."""
    at_format_1(depot)
    depot.tables.rewrite("about", depot.tables.read("about").drop_columns(["format"]))


def at_format(depot, number):
    """The depot's about rows all of the format number."""
    depot.tables.update("about", {"format": str(number)}, "true")


def fails_on(table):
    """Tables.append, failing for the table named."""
    real = Tables.append

    def append(self, name, found):
        if name == table:
            raise OSError("the disk went away")
        return real(self, name, found)
    return append


def follower_of(make_depot, tmp_path) -> tuple[Depot, Depot]:
    upstream = make_depot("upstream")
    follower = clone(upstream.address, str(tmp_path / "follower"), About("follower", "Grace Hopper", "grace@example.org"))
    return upstream, follower


def test_a_new_depot_and_a_clone_are_of_the_code_s_format_and_every_older_format_has_its_step(make_depot, tmp_path):
    upstream, follower = follower_of(make_depot, tmp_path)

    assert upstream.format() == FORMAT and follower.format() == FORMAT
    assert sorted(STEPS) == list(range(FORMAT))  # each layout change raised FORMAT with its upgrade step


def test_a_depot_of_an_older_format_is_refused_by_every_way_in_with_the_command_that_upgrades_it(
        make_depot, tmp_path):
    upstream, follower = follower_of(make_depot, tmp_path)
    at_format_0(upstream)

    for way_in in (lambda: Depot.view(upstream.address), lambda: Depot.open(upstream.address),
                   lambda: clone(upstream.address, str(tmp_path / "second"), About("second", "Ada", "ada@example.org")),
                   lambda: pull(follower)):
        with pytest.raises(OtherFormat) as refused:
            way_in()
        assert str(refused.value) == (f"{upstream.address} is a depot of format 0, older than the format {FORMAT} "
                                      f"this wpt reads: run `wpt depot upgrade` to bring it up to format {FORMAT}")
    assert not (tmp_path / "second").exists()
    assert follower.sql("SELECT count(*) AS n FROM pulls")["n"].to_pylist() == [1]  # the clone's own, no pull


def test_an_upgrade_keeps_every_about_revision_adds_one_at_a_new_change_and_run_again_changes_nothing(make_depot, me):
    depot = make_depot()
    [text] = depot.add([NewText(text="one", licenses=("CC0-1.0",))], me)
    depot.edit_about(maintainer="Grace Hopper")
    before, change = depot.sql(ABOUT).to_pylist(), depot.change()
    at_format_0(depot)

    upgraded, start = Depot.upgrade(depot.address)

    assert start == 0 and upgraded.format() == FORMAT
    found = upgraded.sql("SELECT * FROM about_revisions ORDER BY revision").to_pylist()
    assert [{name: value for name, value in row.items() if name != "format"} for row in found] == [
        *before, *({**before[-1], "revision": 2 + reached, "change": change + reached}
                   for reached in range(1, FORMAT + 1))]
    assert [row["format"] for row in found] == [0, 0, *range(1, FORMAT + 1)]  # each format reached, in turn
    opened = Depot.open(depot.address)
    assert opened.show(text)["text"]["text"] == "one"

    again, start = Depot.upgrade(depot.address)

    assert start == FORMAT and again.change() == change + FORMAT
    assert again.sql("SELECT count(*) AS n FROM about_revisions")["n"].to_pylist() == [2 + FORMAT]


def test_an_upgrade_whose_change_was_left_undone_is_finished_by_running_it_again(make_depot, monkeypatch):
    depot = make_depot()
    change = depot.change()
    at_format_0(depot)

    with monkeypatch.context() as patch:
        patch.setattr(Tables, "append", fails_on("about"))  # the step is done, the new format's change is not
        with pytest.raises(JournalUnfinished):
            Depot.upgrade(depot.address)
    with pytest.raises(OtherFormat, match="run `wpt depot upgrade`"):
        Depot.open(depot.address)

    upgraded, start = Depot.upgrade(depot.address)

    assert start == 0 and upgraded.format() == FORMAT
    assert upgraded.sql("SELECT revision, change, format FROM about_revisions ORDER BY revision").to_pylist() == [
        {"revision": 1, "change": 1, "format": 0},
        *({"revision": 1 + reached, "change": change + reached, "format": reached} for reached in range(1, FORMAT + 1))]
    assert upgraded.sql("SELECT bool_and(done) AS done FROM journal")["done"].to_pylist() == [True]


def test_an_upgrade_finishes_a_change_its_older_format_left_undone_before_recording_the_new_format(
        make_depot, monkeypatch):
    depot = make_depot()
    with monkeypatch.context() as patch:
        patch.setattr(Tables, "append", fails_on("about"))
        with pytest.raises(JournalUnfinished):
            depot.edit_about(maintainer="Grace Hopper")
    at_format_0(depot)

    upgraded, _ = Depot.upgrade(depot.address)

    assert upgraded.sql("SELECT revision, maintainer, format FROM about_revisions ORDER BY revision").to_pylist() == [
        {"revision": 1, "maintainer": "Ada Lovelace", "format": 0},
        {"revision": 2, "maintainer": "Grace Hopper", "format": 0},
        *({"revision": 2 + reached, "maintainer": "Grace Hopper", "format": reached}
          for reached in range(1, FORMAT + 1))]


def test_a_depot_of_a_newer_format_is_refused_everywhere_and_never_brought_down(make_depot, tmp_path):
    upstream, follower = follower_of(make_depot, tmp_path)
    at_format(upstream, FORMAT + 1)
    version = upstream.tables.version("about")

    for way_in in (lambda: Depot.view(upstream.address), lambda: Depot.open(upstream.address),
                   lambda: Depot.upgrade(upstream.address),
                   lambda: clone(upstream.address, str(tmp_path / "second"), About("second", "Ada", "ada@example.org")),
                   lambda: pull(follower)):
        with pytest.raises(OtherFormat) as refused:
            way_in()
        assert str(refused.value) == (f"{upstream.address} is a depot of format {FORMAT + 1}, newer than the format "
                                      f"{FORMAT} this wpt reads: update wpt to read it; a depot is never brought "
                                      "down to an older format")
    assert upstream.tables.version("about") == version


def test_an_upgrade_stopped_after_a_step_resumes_at_the_next_one_and_runs_no_step_twice(make_depot, monkeypatch):
    depot = make_depot()
    at_format_0(depot)
    ran = []

    def first(tables):
        ran.append(0)
        STEPS[0](tables)

    def second(tables):
        ran.append(1)
        if ran.count(1) == 1:
            raise OSError("the machine stopped")

    monkeypatch.setattr(depots, "FORMAT", 2)
    monkeypatch.setattr(depots, "STEPS", {0: first, 1: second})
    with pytest.raises(OSError):
        Depot.upgrade(depot.address)
    assert Depot.find(depot.address).format() == 1

    upgraded, start = Depot.upgrade(depot.address)

    assert start == 1 and ran == [0, 1, 1]
    assert upgraded.sql("SELECT format FROM about_revisions ORDER BY revision")["format"].to_pylist() == [0, 1, 2]


def test_about_read_at_its_storage_version_from_before_the_upgrade_has_a_null_format(make_depot):
    depot = make_depot()
    at_format_0(depot)
    version = depot.tables.version("about")
    Depot.upgrade(depot.address)

    old = depot.tables.read("about", version)

    assert old.to_pylist() == [{"revision": 1, "change": 1, "name": "depot", "maintainer": "Ada Lovelace",
                                "takedown_contact": "ada@example.org", "upstream": None, "format": None}]


def test_a_depot_of_an_older_format_is_backed_up_and_restored_as_it_is_and_the_copy_then_upgrades(
        make_depot, me, tmp_path):
    depot = make_depot()
    [text] = depot.add([NewText(text="one", licenses=("CC0-1.0",))], me)
    at_format_0(depot)

    copy = backup(Depot.find(depot.address), str(tmp_path / "backups"))
    restored = restore(copy.address, str(tmp_path / "restored"))

    assert (copy.format(), restored.format(), Depot.find(depot.address).format()) == (0, 0, 0)
    with pytest.raises(OtherFormat):
        Depot.open(restored.address)
    upgraded, start = Depot.upgrade(restored.address)
    assert start == 0 and upgraded.format() == FORMAT
    assert upgraded.show(text)["text"]["text"] == "one"


def test_backup_and_restore_refuse_a_depot_of_a_newer_format(make_depot, tmp_path):
    depot = make_depot()
    copy = backup(depot, str(tmp_path / "backups"))
    at_format(depot, FORMAT + 1)
    at_format(copy, FORMAT + 1)

    with pytest.raises(OtherFormat, match="newer than the format"):
        backup(Depot.find(depot.address), str(tmp_path / "backups"))
    with pytest.raises(OtherFormat, match="newer than the format"):
        restore(copy.address, str(tmp_path / "restored"))

    assert len(list((tmp_path / "backups").iterdir())) == 1 and not (tmp_path / "restored").exists()


def test_an_upgrade_from_format_1_adds_scores_to_every_text_without_writing_its_rows_again_then_scores_are_set(
        make_depot, me):
    depot = make_depot()
    first, second = depot.add([NewText(text="one", licenses=("CC0-1.0",), tags=("x",)),
                               NewText(text="two", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, "true")
    at_format_1(depot)
    before = depot.sql("SELECT * FROM texts_revisions ORDER BY id, revision").to_pylist()
    files = depot.tables.table("texts").file_uris()

    upgraded, start = Depot.upgrade(depot.address)

    assert start == 1 and upgraded.format() == FORMAT
    assert sorted(upgraded.tables.table("texts").file_uris()) == sorted(files)  # only the table's log changed
    assert upgraded.sql("SELECT * FROM texts_revisions ORDER BY id, revision").to_pylist() == before
    assert {row["scores"] for row in before} == {None}
    version = upgraded.tables.version("texts")
    STEPS[1](upgraded.tables)  # run again, as after a stop just before its format was recorded
    assert upgraded.tables.version("texts") == version

    upgraded.edit_scores(lines([{"id": first, "scores": [{"name": "rank", "value": 2, "content_version": 1}]}]), me)

    assert upgraded.show(first)["text"]["scores"] == [{"name": "rank", "value": 2.0, "content_version": 1}]
    assert (upgraded.show(first)["status"], upgraded.show(first)["text"]["libraries"]) == ("admitted", [library])
    assert upgraded.show(second)["text"]["scores"] is None
    again, start = Depot.upgrade(depot.address)
    assert start == FORMAT and again.show(first)["revisions"] == 3


def test_an_upgrade_from_format_1_finishes_an_add_left_undone_on_the_layout_without_scores(make_depot, me, monkeypatch):
    depot = make_depot()
    with monkeypatch.context() as patch:
        patch.setattr(Tables, "append", fails_on("texts"))
        with pytest.raises(JournalUnfinished):
            depot.add([NewText(text="one", licenses=("CC0-1.0",))], me)
    at_format_1(depot)

    upgraded, _ = Depot.upgrade(depot.address)

    assert upgraded.sql("SELECT id, text, scores FROM texts").to_pylist() == [
        {"id": "depot-1", "text": "one", "scores": None}]


def test_a_depot_of_format_1_is_backed_up_and_restored_as_it_is_and_the_copy_then_upgrades(make_depot, me, tmp_path):
    depot = make_depot()
    [text] = depot.add([NewText(text="one", licenses=("CC0-1.0",))], me)
    at_format_1(depot)

    copy = backup(Depot.find(depot.address), str(tmp_path / "backups"))
    restored = restore(copy.address, str(tmp_path / "restored"))

    assert (copy.format(), restored.format()) == (1, 1)
    assert "scores" not in restored.tables.columns("texts")
    upgraded, start = Depot.upgrade(restored.address)
    assert start == 1 and upgraded.show(text)["text"]["scores"] is None


def hex_lines(count: int) -> str:
    return "".join(hashlib.sha256(str(number).encode()).hexdigest() + "\n" for number in range(count))


def format_2_depot(make_depot, me, monkeypatch):
    """A depot at format 2 holding a text of many parts in two versions, a small text and an item, and what
    each version reads as before: {(id, version): (text, content)}."""
    monkeypatch.setattr(parts, "SIZES", parts.Sizes(64, 256, 1024))
    depot = make_depot()
    big, small = depot.add([NewText(text=hex_lines(200), licenses=("CC0-1.0",)),
                            NewText(text="one line", licenses=("CC0-1.0",))], me)
    depot.add([NewText(kind="probe", content='{"question": "q?", "patterns": ["x"]}', licenses=("CC0-1.0",))], me)
    depot.change_content(big, hex_lines(200).replace(hashlib.sha256(b"100").hexdigest(), "edited"), None, me)
    before = {(row["id"], row["version"]): (row["text"], row["content"]) for row in depot.sql(
        "SELECT id, version, text, content FROM contents").to_pylist()}
    at_format_2(depot)
    return depot, big, before


def test_an_upgrade_from_format_2_stores_every_content_version_as_its_parts_each_part_once(make_depot, me, monkeypatch):
    depot, big, before = format_2_depot(make_depot, me, monkeypatch)
    assert "text" in depot.tables.columns("contents") and "parts" not in depot.tables.exist()

    upgraded, start = Depot.upgrade(depot.address)

    assert start == 2 and upgraded.format() == FORMAT
    assert {(row["id"], row["version"]): (row["text"], row["content"]) for row in upgraded.sql(
        "SELECT id, version, text, content FROM contents").to_pylist()} == before
    stored = upgraded.sql("SELECT count(*) AS n, count(DISTINCT (id, hash)) AS distinct_ FROM parts").to_pylist()
    listed = upgraded.sql("SELECT count(DISTINCT (id, hash)) AS n FROM "
                          "(SELECT id, unnest(parts).hash AS hash FROM content_lists)")["n"].to_pylist()
    assert stored == [{"n": listed[0], "distinct_": listed[0]}]  # every part listed is stored once, none else
    versions = upgraded.sql(f"SELECT version, parts FROM content_lists WHERE id = '{big}' ORDER BY version").to_pylist()
    shared = {bytes(one["hash"]) for one in versions[0]["parts"]} & {bytes(one["hash"]) for one in versions[1]["parts"]}
    assert len(shared) >= len(versions[0]["parts"]) - 3  # the second version reuses the parts it did not change
    assert "text" not in upgraded.tables.columns("contents")
    assert files_holding_text(depot.address, hashlib.sha256(b"100").hexdigest()) == ["parts"]  # no old copy kept

    format3.to_parts(upgraded.tables)  # run again, as after a stop just before its format was recorded

    assert upgraded.sql("SELECT count(*) AS n FROM parts").to_pylist() == [{"n": listed[0]}]
    assert Depot.open(depot.address).show(big)["text"]["text"] == before[big, 2][0]


def files_holding_text(folder, passage: str) -> list[str]:
    """The tables some file of which holds passage."""
    return sorted({path.relative_to(folder).parts[0] for path in Path(folder).rglob("*")
                   if path.is_file() and passage.encode() in path.read_bytes()})


def test_an_upgrade_from_format_2_cut_short_between_two_batches_resumes_and_stores_no_part_twice(
        make_depot, me, monkeypatch, tmp_path):
    depot, big, before = format_2_depot(make_depot, me, monkeypatch)
    monkeypatch.setattr(format3, "BATCH_BYTES", 1)  # a batch for each data file of contents
    assert len(depot.tables.table("contents").file_uris()) > 1
    real = Tables.append_stream
    calls = []

    def stops(self, name, found, transaction=None, **written):
        calls.append(transaction)
        if len(calls) == 2:
            raise OSError("the machine stopped")
        return real(self, name, found, transaction, **written)

    with monkeypatch.context() as patch:
        patch.setattr(Tables, "append_stream", stops)
        with pytest.raises(OSError):
            Depot.upgrade(depot.address)
    assert Depot.find(depot.address).format() == 2
    assert len(list((Path(depot.address) / format3.LISTS).glob("*.parquet"))) == 1  # the first batch's lists, kept

    upgraded, start = Depot.upgrade(depot.address)

    assert start == 2 and upgraded.format() == FORMAT
    assert {(row["id"], row["version"]): (row["text"], row["content"]) for row in upgraded.sql(
        "SELECT id, version, text, content FROM contents").to_pylist()} == before
    assert upgraded.sql("SELECT count(*) = count(DISTINCT (id, hash)) AS once FROM parts")["once"].to_pylist() == [True]
    assert not (Path(depot.address) / format3.LISTS).exists()


def test_an_upgrade_from_format_2_stopped_after_contents_is_replaced_removes_the_kept_lists_when_run_again(
        make_depot, me, monkeypatch):
    depot, big, before = format_2_depot(make_depot, me, monkeypatch)
    real = format3._remove

    def stops(system, folder):
        if (Path(depot.address) / format3.LISTS).exists():
            raise OSError("the machine stopped")
        real(system, folder)

    with monkeypatch.context() as patch:
        patch.setattr(format3, "_remove", stops)
        with pytest.raises(OSError):
            Depot.upgrade(depot.address)
    assert (Path(depot.address) / format3.LISTS).exists()

    upgraded, _ = Depot.upgrade(depot.address)

    assert not (Path(depot.address) / format3.LISTS).exists()
    assert upgraded.format() == FORMAT and upgraded.content(big, 2) == before[big, 2][0]
