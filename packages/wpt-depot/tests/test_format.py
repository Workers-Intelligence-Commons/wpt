import pytest

from wpt_depot import About, Depot, NewText, backup, clone, pull, restore
from wpt_depot import depot as depots
from wpt_depot.errors import JournalUnfinished, OtherFormat
from wpt_depot.schema import FORMAT, STEPS
from wpt_depot.tables import Tables

ABOUT = "SELECT * EXCLUDE (format) FROM about_revisions ORDER BY revision"


def at_format_0(depot):
    """The depot's about table as format 0 lays it out: without the column format."""
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
        *before, {**before[-1], "revision": 3, "change": change + 1}]
    assert [row["format"] for row in found] == [0, 0, FORMAT]
    opened = Depot.open(depot.address)
    assert opened.show(text)["text"]["text"] == "one"

    again, start = Depot.upgrade(depot.address)

    assert start == FORMAT and again.change() == change + 1
    assert again.sql("SELECT count(*) AS n FROM about_revisions")["n"].to_pylist() == [3]


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
        {"revision": 1, "change": 1, "format": 0}, {"revision": 2, "change": change + 1, "format": FORMAT}]
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
        {"revision": 3, "maintainer": "Grace Hopper", "format": FORMAT}]


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
