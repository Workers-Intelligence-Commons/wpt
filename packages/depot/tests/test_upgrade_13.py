"""The step to format 13 on the fixture depots (fixtures/README.md), written by the code of formats 11 and 12:
what `depot upgrade --check` says and never writes, and what the upgrade makes of the libraries and datasets."""

import hashlib
import json
import shutil
from pathlib import Path

import apsw
import pytest

from depot import Depot, conversion, upgrade
from freight.corpora import records
from depot.errors import HalfUpgraded, OtherFormat, Refused
from depot.schema import FORMAT, KEYS, LEGACY

from .conftest import ME, sizes_apart, tags_apart

FIXTURES = Path(__file__).parent / "fixtures"
PUBLIC = "https://depot.example.org"
LIBRARY_FACETS = ("library", "pinned newer", "library kind", "library tag", "library split")


def copied(tmp_path: Path, format_: int, name: str = "depot") -> tuple[Path, list[dict]]:
    """(a copy of the fixture depot of that format in tmp_path, the records of its dock's corpora)."""
    path = tmp_path / f"{name}.sqlite"
    shutil.copy(FIXTURES / f"format-{format_}.sqlite", path)
    dock = tmp_path / f"{name}-dock"
    if not dock.exists():
        shutil.copytree(FIXTURES / f"format-{format_}-dock", dock)
    return path, records(dock.as_uri())


def connected(path: Path) -> apsw.Connection:
    return apsw.Connection(f"file:{path.as_posix()}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)


def rows_of(path: Path, query: str, params=()) -> list[tuple]:
    con = connected(path)
    try:
        return con.execute(query, params).fetchall()
    finally:
        con.close()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(folder: Path) -> dict[str, str]:
    return {one.name: sha256(one) for one in sorted(folder.iterdir()) if one.is_file()}


def every_row(path: Path, tables) -> dict[str, list[tuple]]:
    """Every row of each table, in the order of its key; the time of the upgrade's change and of each snapshot
    aside."""
    con = connected(path)
    try:
        found = {table: con.execute(f"SELECT * FROM {table} ORDER BY {', '.join(KEYS[table])}").fetchall()
                 for table in tables}
    finally:
        con.close()
    if "changes" in found:
        found["changes"] = [(number, kind, None if kind == "upgrade" else at) for number, kind, at in found["changes"]]
    if "snapshots" in found:
        found["snapshots"] = [(*row[:4], None, *row[5:]) for row in found["snapshots"]]
    return found


def texts_of(depot: Depot, snapshot: str) -> dict[str, tuple[int, int]]:
    """{id: (revision, part)} of every text the snapshot reads (its sets checked against its digest)."""
    found, after = {}, None
    while True:
        page = depot.snapshot_texts(snapshot, after=after)
        found |= {one["id"]: (one["revision"], one["part"]) for one in page["texts"]}
        if page["after"] is None:
            return found
        after = page["after"]


def library_id(path: Path, name: str) -> str:
    return rows_of(path, "SELECT id FROM libraries_revisions WHERE name = ? AND latest", (name,))[0][0]


def pinned(path: Path, library: str) -> dict[str, int]:
    """{text: revision} of the library's pins whose revision the depot still holds."""
    return dict(rows_of(path, "SELECT p.text, p.revision FROM pins p WHERE p.library = ? AND EXISTS (SELECT 1 FROM "
                              "texts_revisions r WHERE r.id = p.text AND r.revision = p.revision)", (library,)))


def lock_change(path: Path, library: str) -> int:
    return rows_of(path, "SELECT min(change) FROM libraries_revisions WHERE id = ? AND locked_at IS NOT NULL",
                   (library,))[0][0]


@pytest.mark.parametrize("format_", [11, 12])
def test_the_check_reads_without_writing_and_says_what_converts_what_it_checked_and_what_it_costs(tmp_path, format_):
    path, found = copied(tmp_path, format_)
    before = files(tmp_path)

    checked = upgrade.check(str(path), found, PUBLIC)

    assert files(tmp_path) == before  # the file, and no WAL or other file beside it
    assert checked.passed and checked.needed and not checked.failures
    said = "\n".join(checked.lines)
    assert sorted(one["name"] for one in checked.snapshots) == ["ds-mixed", "ds-two", "ds-two-r2", "lib-a", "lib-b"]
    assert "snapshot ds-mixed, library fixture-3: 40 texts, every revision equal" in said
    assert said.count("every revision equal") == 9 and "FAILED" not in said
    assert "open libraries left in the legacy tables, not converted: lib-c (fixture-3), lib-d (fixture-4)" in said
    assert "locked libraries of other depots" in said and "up-a (up-1)" in said
    assert "Names made snapshots' names: 'ds-two' -> ds-two-r2" in said
    section = next(line for line in checked.lines if line.startswith("Order data section for revision 3 of the "
                                                                     "dataset ds-two"))
    data = json.loads(section.split(": ", 1)[1])
    two = next(one for one in checked.snapshots if one["name"] == "ds-two")
    assert (data["depot"], data["snapshot"], data["digest"]) == (PUBLIC, "ds-two", two["digest"])
    assert data["shares"] == [{"tag": "en", "share": 0.3}, {"tag": "letter", "share": 0.7}]
    assert data["vocabulary"]["sha256"] and data["vocabulary"]["size"] > 0
    # the steps left and their cost together, then what to do before and the command
    assert ("11 to 12: about" in said) == (format_ == 11) and "12 to 13, tables: about" in said
    assert {"seconds", "wal", "added", "backup"} <= set(checked.cost)
    assert "full copy of the file" in said and "WAL at its largest" in said and "disk added" in said
    assert any(line.startswith("Before: run `depot backup`") for line in checked.lines)
    assert checked.lines[-1] == "Run: depot upgrade --yes"


def test_a_depot_of_format_11_is_brought_to_13_in_one_run_as_the_check_said(tmp_path):
    path, found = copied(tmp_path, 11)
    checked = upgrade.check(str(path), found, PUBLIC)
    said = []

    depot, start = upgrade.upgrade(str(path), log=said.append, records=found, public=PUBLIC)

    assert start == 11 and depot.format() == FORMAT
    assert [one for (one,) in rows_of(path, "SELECT format FROM about_revisions ORDER BY revision")][-4:] == [11, 12, 13, 14]
    made = {one["name"]: one for one in depot.snapshots()}
    assert {name: (one["digest"], one["texts"], one["at_change"]) for name, one in made.items()} == {
        one["name"]: (one["digest"], one["texts"], one["at_change"]) for one in checked.snapshots}
    # each dataset revision converted names its snapshot; a clone converts only what it owns
    assert {(dataset, revision): snapshot for dataset, revision, snapshot in rows_of(
        path, "SELECT dataset, revision, snapshot FROM legacy_snapshot_of")} == {
        ("fixture-1", 2): made["ds-two-r2"]["id"], ("fixture-1", 3): made["ds-two"]["id"],
        ("fixture-2", 1): made["ds-mixed"]["id"]}
    assert all(not one["origin"].get("library", "").startswith("up-") for one in made.values())
    assert made["ds-two-r2"]["origin"] == {"kind": "dataset", "dataset": "fixture-1", "revision": 2, "name": "ds-two"}
    assert any(line.startswith(f"{path}: the data section of a model order on the snapshot of revision 2 of the "
                               "dataset")
               for line in said)
    for one in made.values():
        assert texts_of(depot, one["id"])  # its sets match its digest
    # the figures of the datasets stay, in the layout of format 12
    assert rows_of(path, "SELECT count(*) FROM datasets_revisions WHERE figures IS NOT NULL AND figures LIKE "
                         "'%\"kinds\"%'")[0][0] > 0


@pytest.mark.parametrize("format_", [11, 12])
def test_the_upgrade_writes_what_the_check_right_before_measured_and_measures_again_after_a_change(
        tmp_path, monkeypatch, format_):
    alone, found = copied(tmp_path, format_, "alone")
    upgrade.upgrade(str(alone), records=found, public=PUBLIC)[0].db.close()
    path, _ = copied(tmp_path, format_)
    checked = upgrade.check(str(path), found, PUBLIC)
    real = conversion.measure_all

    def refused(*args, **kwargs):
        raise AssertionError("measured again")
    monkeypatch.setattr(conversion, "measure_all", refused)

    upgrade.upgrade(str(path), records=found, public=PUBLIC, checked=checked)[0].db.close()

    assert every_row(path, KEYS) == every_row(alone, KEYS)
    # a change written between the check and the upgrade: the upgrade measures again
    changed, _ = copied(tmp_path, format_, "changed")
    monkeypatch.setattr(conversion, "measure_all", real)
    checked = upgrade.check(str(changed), found, PUBLIC)
    con = apsw.Connection(str(changed))
    con.execute("INSERT INTO changes SELECT id + 1, 'decision', made_at FROM changes ORDER BY id DESC LIMIT 1")
    con.close()
    calls = []
    monkeypatch.setattr(conversion, "measure_all", lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs))

    upgrade.upgrade(str(changed), records=found, public=PUBLIC, checked=checked)[0].db.close()

    assert calls == [1]


def test_the_step_to_12_counts_the_kinds_alike_whether_it_scans_the_tables_or_looks_the_items_up(
        tmp_path, monkeypatch):
    rows = []
    for share in (0, 10 ** 9):  # never scanned, always scanned
        monkeypatch.setattr(upgrade, "ITEMS_SHARE_12", share)
        path, found = copied(tmp_path, 11, f"share-{share}")
        upgrade.upgrade(str(path), records=found, public=PUBLIC)[0].db.close()
        rows.append(every_row(path, KEYS))
    assert rows[0] == rows[1]


def test_a_saved_check_spares_the_next_check_its_reads_until_the_depot_changes(tmp_path, monkeypatch):
    path, found = copied(tmp_path, 12)
    first = upgrade.check(str(path), found, PUBLIC, save=True)
    saved = upgrade.saved_check_path(path)
    assert saved.is_file() and first.passed
    real, calls = conversion.measure_all, []
    monkeypatch.setattr(conversion, "measure_all", lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs))

    again = upgrade.check(str(path), found, PUBLIC)

    assert calls == [] and again.snapshots == first.snapshots and again.state == first.state
    assert any(line.startswith("Saved check: ") for line in again.lines)
    # a change written since: measured again; a file that cannot be read: measured again
    con = apsw.Connection(str(path))
    con.execute("INSERT INTO changes SELECT id + 1, 'decision', made_at FROM changes ORDER BY id DESC LIMIT 1")
    con.close()
    changed = upgrade.check(str(path), found, PUBLIC)
    assert calls == [1] and "another state" in next(line for line in changed.lines if line.startswith("Saved check"))
    saved.write_text("{", encoding="utf-8")
    upgrade.check(str(path), found, PUBLIC)
    assert calls == [1, 1]
    upgrade.check(str(path), found, PUBLIC, save=True)
    upgrade.upgrade(str(path), records=found, public=PUBLIC)[0].db.close()
    assert not saved.exists()


def test_a_locked_library_becomes_a_snapshot_of_its_pins_read_at_the_change_that_locked_it(tmp_path):
    path, found = copied(tmp_path, 12)
    libraries = {name: library_id(path, name) for name in ("lib-a", "lib-b")}
    pins = {name: pinned(path, library) for name, library in libraries.items()}
    locks = {name: lock_change(path, library) for name, library in libraries.items()}

    depot, _ = upgrade.upgrade(str(path), log=lambda line: None, records=found)

    for name, library in libraries.items():
        snapshot = depot.snapshot(name)
        assert snapshot["origin"] == {"kind": "library", "library": library, "name": name,
                                      "lock_change": locks[name]}
        assert snapshot["at_change"] == locks[name] and snapshot["texts"] == len(pins[name])
        assert {text: revision for text, (revision, _) in texts_of(depot, snapshot["id"]).items()} == pins[name]
        assert rows_of(path, "SELECT part, owner, bound FROM snapshot_sets WHERE snapshot = ?", (snapshot["id"],)) == [
            (0, "fixture", locks[name])]


def test_a_dataset_of_two_libraries_locked_at_different_changes_reads_each_at_its_own_lock(tmp_path):
    path, found = copied(tmp_path, 12)
    a, b = library_id(path, "lib-a"), library_id(path, "lib-b")
    first, second = pinned(path, a), pinned(path, b)

    depot, _ = upgrade.upgrade(str(path), log=lambda line: None, records=found)

    snapshot = depot.snapshot("ds-two")
    assert rows_of(path, "SELECT part, owner, bound FROM snapshot_sets WHERE snapshot = ? ORDER BY part",
                   (snapshot["id"],)) == [(0, "fixture", lock_change(path, a)), (1, "fixture", lock_change(path, b))]
    read = texts_of(depot, snapshot["id"])
    # a text both libraries pinned is read in the first, as a dataset read it in the first library holding it
    assert read == {text: (revision, 0) for text, revision in first.items()} | {
        text: (revision, 1) for text, revision in second.items() if text not in first}


def test_a_dataset_reading_an_open_library_at_a_change_and_a_locked_one_above_its_lock(tmp_path):
    # ds-mixed reads lib-a at a change above the change that locked it, and a text lib-a pinned has a newer
    # revision: the snapshot reads the pinned revision; it reads the open lib-c by the texts it held at its change
    path, found = copied(tmp_path, 12)
    a, c = library_id(path, "lib-a"), library_id(path, "lib-c")
    [refs] = rows_of(path, "SELECT libraries FROM datasets_revisions WHERE name = 'ds-mixed' AND latest")[0]
    refs = json.loads(refs)
    assert refs[0]["library"] == a and refs[0]["change"] > lock_change(path, a)
    pins = pinned(path, a)
    edited = [text for text, revision in pins.items()
              if rows_of(path, "SELECT max(revision) FROM texts_revisions WHERE id = ?", (text,))[0][0] > revision]
    assert edited
    change = refs[1]["change"]
    members = dict(rows_of(path, "SELECT m.text, (SELECT max(r.revision) FROM texts_revisions r WHERE r.id = m.text "
                                 "AND r.change <= ?) FROM memberships m WHERE m.library = ? AND m.joined_change <= ? "
                                 "AND (m.left_change IS NULL OR m.left_change > ?)", (change, c, change, change)))

    depot, _ = upgrade.upgrade(str(path), log=lambda line: None, records=found)

    snapshot = depot.snapshot("ds-mixed")
    read = texts_of(depot, snapshot["id"])
    for text in edited:
        assert read[text] == (pins[text], 0)
    assert {text: revision for text, (revision, part) in read.items() if part == 1} == {
        text: revision for text, revision in members.items() if text not in pins and revision is not None}
    sets = rows_of(path, "SELECT part, owner, bound FROM snapshot_sets WHERE snapshot = ? ORDER BY part, owner",
                   (snapshot["id"],))
    assert sets[0] == (0, "fixture", lock_change(path, a)) and sets[1][:3] == (1, "fixture", change)
    assert {owner for part, owner, _ in sets if part == 2} == {"up"}  # the upstream's locked library, received


def test_names_become_snapshot_names_the_original_kept_in_origin_with_a_suffix_on_a_clash(tmp_path):
    path, found = copied(tmp_path, 12)
    con = apsw.Connection(str(path))
    con.execute("UPDATE libraries_revisions SET name = 'lib-a' WHERE name = 'lib-b'")
    con.execute("UPDATE datasets_revisions SET name = 'lib-a' WHERE name = 'ds-mixed'")
    con.execute("UPDATE datasets_revisions SET name = '  ds two!' WHERE name = 'ds-two'")
    con.close()
    b = rows_of(path, "SELECT id FROM libraries_revisions WHERE name = 'lib-a' AND latest ORDER BY id DESC")[0][0]

    checked = upgrade.check(str(path), found)
    depot, _ = upgrade.upgrade(str(path), log=lambda line: None, records=found)

    names = {one["name"]: one["origin"] for one in depot.snapshots()}
    assert set(names) == {"lib-a", f"lib-a-{b.rsplit('-', 1)[1]}", "lib-a-3", "ds-two-", "ds-two--r2"}
    assert names["ds-two-"]["name"] == names["ds-two--r2"]["name"] == "  ds two!"
    assert names["lib-a-3"]["name"] == "lib-a" and names["lib-a-3"]["kind"] == "dataset"
    renamed = next(line for line in checked.lines if line.startswith("Names made snapshots' names: "))
    assert "'  ds two!' -> ds-two-" in renamed and "'lib-a' -> lib-a-3" in renamed


def test_sanitized_names():
    assert conversion.sanitized("2026/11 (b)", "library") == "2026-11--b-"
    assert conversion.sanitized("", "library") == "library"
    assert conversion.sanitized("._-", "dataset") == "dataset"
    assert conversion.sanitized("Ok.v1_2-x", "library") == "Ok.v1_2-x"


STOPS = [*(f"before {name}" for name in upgrade.STAGES_13), "in taggings", "in sizes", "in locked", "in datasets"]


@pytest.mark.parametrize("stop", STOPS)
def test_an_upgrade_stopped_in_any_stage_goes_on_when_run_again_and_writes_what_one_run_writes(
        tmp_path, monkeypatch, stop):
    whole_path, found = copied(tmp_path, 12, "whole")
    upgrade.upgrade(str(whole_path), log=lambda line: None, records=found)
    path, _ = copied(tmp_path, 12, "stopped")
    written = []
    real = upgrade._write_snapshot

    def stops(*args):
        raise OSError("the machine stopped")

    def stops_at(count):
        def write(*args):
            written.append(args)
            if len(written) == count:
                raise OSError("the machine stopped")
            return real(*args)
        return write

    log = lambda line: None  # noqa: E731

    def stops_after_a_transaction(line):  # the stage logs each transaction once it is written
        if (" of texts_revisions" not in line and "revision row " in line) or " revisions written, the last text " in line:
            raise OSError("the machine stopped")

    in_sizes = []

    def stops_in_sizes(line):  # the same, once the stage sizes has begun
        if line.endswith(f"{upgrade.BATCH:,} rowids a transaction") and "copying the kind" in line:
            in_sizes.append(line)
        elif in_sizes:
            stops_after_a_transaction(line)

    with monkeypatch.context() as patch:
        if stop in ("in taggings", "in sizes"):  # after its first transaction of 50 revisions
            patch.setattr(upgrade, "BATCH", 50)
            patch.setattr(upgrade, "SAY_EVERY", 0)
            log = stops_after_a_transaction if stop == "in taggings" else stops_in_sizes
        elif stop.startswith("before "):
            patch.setitem(upgrade.STAGES_13, stop.removeprefix("before "), stops)
        else:  # once the first snapshot of the stage is written
            patch.setattr(upgrade, "_write_snapshot", stops_at(2 if stop == "in locked" else 4))
        with pytest.raises(OSError):
            upgrade.upgrade(str(path), log=log, records=found)
    if stop == "in sizes":
        assert rows_of(path, f"SELECT after FROM {upgrade.UNDER_WAY} WHERE stage = 'sizes'") == [(50,)]
    if stop == "in taggings":  # the texts of about 50 revisions written, in the order of their ids, the last recorded
        [(last,)] = rows_of(path, f"SELECT after FROM {upgrade.UNDER_WAY} WHERE stage = 'taggings'")
        assert rows_of(path, "SELECT count(*) > 0, count(*) FILTER (WHERE text > ?) FROM taggings", (last,)) == [(1, 0)]
        assert 50 <= rows_of(path, "SELECT count(*) FROM texts_revisions WHERE id <= ?", (last,))[0][0] < 60
    for way_in in (Depot.view, Depot.open):
        with pytest.raises((HalfUpgraded, OtherFormat), match="depot upgrade"):
            way_in(str(path))

    depot, start = upgrade.upgrade(str(path), log=lambda line: None, records=[])  # the dock's revisions were kept

    assert start == 12 and depot.format() == FORMAT
    assert every_row(path, KEYS) == every_row(whole_path, KEYS)


def test_the_upgrade_refuses_while_another_connection_has_the_depot_open_and_writes_nothing(tmp_path):
    # a web app of format 12 left running checks the format only when it opens the depot: it would write on
    path, found = copied(tmp_path, 12)
    wal = apsw.Connection(str(path))  # a depot file is in WAL mode (Database.create); the fixture's copy is not
    wal.execute("PRAGMA journal_mode = WAL").fetchall()
    wal.close()
    before = sha256(path)
    running = apsw.Connection(str(path))  # as a web app holds it, once it has read the file
    try:
        running.execute("SELECT count(*) FROM about_revisions").fetchall()
        with pytest.raises(Refused, match="another process has the depot open"):
            upgrade.upgrade(str(path), log=lambda line: None, records=found)
    finally:
        running.close()
    assert sha256(path) == before

    depot, start = upgrade.upgrade(str(path), log=lambda line: None, records=found)

    assert start == 12 and depot.format() == FORMAT


@pytest.mark.parametrize("format_", [11, 12])
def test_every_revision_upgraded_keeps_the_kind_length_and_bytes_of_the_content_version_it_reads(tmp_path, format_):
    path, found = copied(tmp_path, format_)
    assert rows_of(path, "SELECT count(*) FROM pragma_table_info('texts_revisions') WHERE name = 'content_kind'"
                   ) == [(0,)]

    depot, _ = upgrade.upgrade(str(path), log=lambda line: None, records=found)

    assert sizes_apart(depot) == [] and tags_apart(depot) == []
    assert rows_of(path, "SELECT count(DISTINCT content_kind) FROM texts_revisions")[0][0] > 1  # items among them
    # a decision and an erasure's cut after the upgrade keep them as every write does
    text = depot.rows("SELECT id FROM texts WHERE kind = 'plain' AND length > 10 AND id LIKE 'fixture-%' "
                      "ORDER BY id LIMIT 1")[0]["id"]
    depot.decide("refused", [text], ME)
    depot.decide_erasure(depot.ask_erasure("a passage", [(text, 0, 5)], ME), "admitted", ME)
    assert sizes_apart(depot) == [] and tags_apart(depot) == []


def test_the_tags_each_revision_held_stay_through_later_changes_of_tags(tmp_path):
    # a pinned revision keeps its own column tags, order included, as the older formats left them; the copy of a
    # text's tags that later changes rewrite lives in current_tags
    path, found = copied(tmp_path, 12)
    held = rows_of(path, "SELECT id, revision, tags FROM texts_revisions ORDER BY id, revision")

    depot, _ = upgrade.upgrade(str(path), log=lambda line: None, records=found)
    texts = [row["id"] for row in depot.rows("SELECT id FROM texts WHERE json_array_length(tags) > 0 "
                                             "AND id LIKE 'fixture-%' ORDER BY id LIMIT 3")]
    first = depot.rows("SELECT tags FROM texts WHERE id = ?", (texts[0],))[0]["tags"][0]
    depot.retag(texts, ["added-later"], [first], ME)
    depot.relicense(texts[:1], ["CC0-1.0"], ME)

    assert rows_of(path, "SELECT id, revision, tags FROM texts_revisions WHERE (id, revision) IN "
                   "(SELECT value ->> 0, value ->> 1 FROM json_each(?)) ORDER BY id, revision",
                   (json.dumps([row[:2] for row in held]),)) == held
    assert tags_apart(depot) == []
    assert "added-later" in depot.rows("SELECT tags FROM texts WHERE id = ?", (texts[0],))[0]["tags"]


def test_the_legacy_tables_are_left_as_they_were(tmp_path):
    path, found = copied(tmp_path, 12)
    before = every_row(path, LEGACY)

    upgrade.upgrade(str(path), log=lambda line: None, records=found)

    after = every_row(path, LEGACY)
    assert {table: len(rows) for table, rows in after.items()} == {table: len(rows) for table, rows in before.items()}
    assert hashlib.sha256(repr(after).encode()).hexdigest() == hashlib.sha256(repr(before).encode()).hexdigest()


def test_the_counts_of_libraries_go_and_the_splits_are_folded_into_what_a_recount_makes(tmp_path):
    path, found = copied(tmp_path, 12)
    assert rows_of(path, "SELECT count(*) FROM counts WHERE facet IN ('library', 'library kind')")[0][0] > 0
    assert rows_of(path, "SELECT count(*) FROM counts WHERE facet = 'tag split' AND name LIKE 'admitted 1 %'")[0][0] > 0

    depot, _ = upgrade.upgrade(str(path), log=lambda line: None, records=found)

    upgraded = rows_of(path, "SELECT facet, name, texts, characters, bytes FROM counts ORDER BY facet, name")
    assert not [row for row in upgraded if row[0] in LIBRARY_FACETS]
    depot.recount()
    assert rows_of(path, "SELECT facet, name, texts, characters, bytes FROM counts ORDER BY facet, name") == upgraded


def test_a_revision_the_snapshot_would_read_otherwise_stops_the_check_and_the_upgrade(tmp_path):
    # a pin naming a revision newer than the lock (which no write of format 12 made) is what a snapshot cannot
    # read at one change: the check names it and writes nothing, the upgrade writes no snapshot of it
    path, found = copied(tmp_path, 12)
    a = library_id(path, "lib-a")
    pins = pinned(path, a)
    text = next(text for text, revision in pins.items()
                if rows_of(path, "SELECT max(revision) FROM texts_revisions WHERE id = ?", (text,))[0][0] > revision)
    con = apsw.Connection(str(path))
    con.execute("UPDATE pins SET revision = (SELECT max(revision) FROM texts_revisions WHERE id = ?) "
                "WHERE library = ? AND text = ?", (text, a, text))
    con.close()
    before = files(tmp_path)

    checked = upgrade.check(str(path), found)

    assert files(tmp_path) == before
    assert not checked.passed and any(text in one for one in checked.failures)
    assert "The check FAILED" in "\n".join(checked.lines) and "Run: depot upgrade --yes" not in checked.lines
    with pytest.raises(Refused, match=text):
        upgrade.upgrade(str(path), log=lambda line: None, records=found)
    assert not rows_of(path, "SELECT 1 FROM snapshots WHERE origin ->> 'library' = ?", (a,))


def test_the_check_of_a_depot_of_the_code_s_format_says_there_is_nothing_to_do(make_depot):
    depot = make_depot()

    checked = upgrade.check(depot.address)

    assert checked.passed and not checked.needed
    assert checked.lines == [f"{depot.address}: at format {FORMAT} already, nothing to upgrade"]
