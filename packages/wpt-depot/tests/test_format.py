import hashlib
import json
import re
import shutil
from pathlib import Path

import apsw
import pytest

from wpt_depot.filters import Filter
from wpt_depot import About, Depot, NewText, clone, ids, upgrade
from wpt_depot.errors import DepotExists, HalfUpgraded, OtherFormat, Refused
from wpt_depot.filters import KEY_INDEXES
from wpt_depot.legacy import Legacy
from wpt_depot.database import FULL_COPY_NEXT
from wpt_depot.schema import FORMAT, KEYS, VIEWS

CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}


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


def test_the_key_indexes_queries_name_hold_their_table_s_key_in_a_new_depot_and_a_clone(make_depot, serve, tmp_path):
    # a query that names an index SQLite named fails with no such index, or reads another order, once a layout
    # rebuilds the table with another key
    upstream = make_depot("upstream")
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace", "grace@example.org"))

    for depot in (upstream, follower):
        for index, table in KEY_INDEXES.items():
            columns = [one[2] for one in depot.db.reader().execute(f"PRAGMA index_info({index})")]
            assert columns == list(KEYS[table]), (depot.address, index)


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
    assert [one["format"] for one in depot.sql("SELECT format FROM about_revisions ORDER BY revision")] == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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

    assert [one["kind"] for one in depot.sql("SELECT kind FROM changes WHERE kind = 'upgrade'")] == ["upgrade"] * 9
    assert [one["format"] for one in depot.sql("SELECT format FROM about_revisions ORDER BY revision")] == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


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


OLD_CONVERSATION = json.dumps({"type": "object", "description": "a conversation kind of a format before excerpts"})


def made_older(depot, format_: int):
    """The depot as a depot of format_ (4 to 11) holds what it holds: no kind on its plain texts' content versions,
    the column kind of content_lists nullable, no row plain in kinds, the index content_lists_long over the
    content versions of no kind, a conversation kind whose schema takes no excerpt, no table quotes nor its
    index, and what each library holds counted, and its dataset revisions' figures written,
    by group of kinds (plain, items: conversations and tasks, other), not by kind; for 10 and older no index
    content_lists_long; for 9 and older
    also no tables text_tags and text_licenses, no title
    index text_titles nor its title_owners, no
    indexes of the format 10 upgrade (INDEXES_10) but the index text_decisions_decision, and no counts of scores, of
    pins with a newer revision and of the splits by status and by locked library; for 8
    and older also no columns characters and bytes of counts and
    none of its rows of what libraries hold by kind and by tag, no column figures of the dataset revisions and no
    table library_moves; for 7 and older also no index text_decisions_decision; for 6
    and older also no export keys, no indexes
    memberships_key and pins_key and no vocabulary of the datasets; for 5 and 4 also each text's decisions as the JSON list of its current revision, no
    status, text_decisions, counts, tokenizer recipes, vocabularies or max_tokens; for 4 also no index texts_order.
    Its rows before (table_rows), with the status of the revisions that are not current, which no format records,
    left out."""
    with depot.db.bulk() as con:
        con.execute("UPDATE texts_revisions SET status = NULL WHERE NOT latest")
    before = table_rows(depot.address)
    with depot.db.bulk() as con:
        statement = con.execute("SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'content_lists'"
                                ).fetchall()[0][0]
        version = con.execute("PRAGMA schema_version").fetchall()[0][0]
        con.execute("PRAGMA writable_schema = ON")
        con.execute("UPDATE sqlite_schema SET sql = ? WHERE type = 'table' AND name = 'content_lists'",
                    (statement.replace(upgrade.NOT_NULL_12, upgrade.NULLABLE_12),))
        con.execute(f"PRAGMA schema_version = {int(version) + 1}")
        con.execute("PRAGMA writable_schema = OFF")
    with depot.db.bulk() as con:
        con.execute("UPDATE kinds SET schema = ? WHERE id = 'conversation'", (OLD_CONVERSATION,))
        con.execute("DROP TABLE quotes")
        grouped = {}
        for name, *sized in con.execute("SELECT name, texts, characters, bytes FROM counts "
                                        "WHERE facet = 'library kind'").fetchall():
            library, kind = name.split(" ", 1)
            if kind != "plain":
                into = grouped.setdefault(f"{library} {'items' if kind in ('conversation', 'task') else 'other'}",
                                          [0, 0, 0])
                into[:] = [a + b for a, b in zip(into, sized)]
        con.execute("DELETE FROM counts WHERE facet = 'library kind' AND substr(name, instr(name, ' ') + 1) <> "
                    "'plain'")
        con.executemany("INSERT INTO counts (facet, name, texts, characters, bytes) VALUES ('library kind', ?, ?, ?, ?)",
                        [(name, *sized) for name, sized in grouped.items()])
        for dataset, revision, figures in con.execute("SELECT id, revision, figures FROM datasets_revisions "
                                                      "WHERE figures IS NOT NULL").fetchall():
            figures = json.loads(figures)

            def group(kinds, chosen):
                return {key: sum(one[key] for kind, one in kinds.items() if kind in chosen)
                        for key in ("texts", "characters", "bytes")}

            figures["libraries"] = [None if one is None else {
                "plain": one["kinds"]["plain"], "items": group(one["kinds"], {"conversation", "task"}),
                "other": group(one["kinds"], set(one["kinds"]) - {"plain", "conversation", "task"}),
                "tags": one["tags"]} for one in figures["libraries"]]
            con.execute("UPDATE datasets_revisions SET figures = ? WHERE id = ? AND revision = ?",
                        (json.dumps(figures, ensure_ascii=False, separators=(",", ":")), dataset, revision))
        con.execute("DROP INDEX content_lists_long")
        con.execute("UPDATE content_lists SET kind = NULL WHERE kind = 'plain'")
        con.execute("DELETE FROM kinds WHERE id = 'plain'")
        if format_ == 11:
            con.execute(upgrade.LONG_11)
        con.execute("UPDATE about_revisions SET format = ?", (format_,))
    if format_ == 11:
        depot.db.close()
        (depot.db.waiting / FULL_COPY_NEXT).unlink()
        return before
    if format_ == 10:
        depot.db.close()
        (depot.db.waiting / FULL_COPY_NEXT).unlink()
        return before
    with depot.db.bulk() as con:
        for name, _ in upgrade.INDEXES_10:
            con.execute(f"DROP INDEX {name}")
        con.execute(upgrade.TEXT_DECISIONS_DECISION)
        for table in ("text_tags", "text_licenses", "text_titles", "title_owners"):
            con.execute(f"DROP TABLE {table}")
        con.execute("DELETE FROM counts WHERE facet IN ('score', 'pinned newer', "
                    + ", ".join(f"'{facet}'" for facet, _ in upgrade.SPLITS_10) + ")")
        con.execute("UPDATE about_revisions SET format = ?", (format_,))
    if format_ == 9:
        depot.db.close()
        (depot.db.waiting / FULL_COPY_NEXT).unlink()
        return before
    with depot.db.bulk() as con:
        con.execute("DELETE FROM counts WHERE facet IN ('library kind', 'library tag')")
        for column in ("characters", "bytes"):
            con.execute(f"ALTER TABLE counts DROP COLUMN {column}")
        for view in VIEWS:  # the connection's own views read the columns
            con.execute(f"DROP VIEW IF EXISTS temp.{view}")
        con.execute("ALTER TABLE datasets_revisions DROP COLUMN figures")
        con.execute("DROP TABLE library_moves")
        con.execute("UPDATE about_revisions SET format = ?", (format_,))
    if format_ == 8:
        depot.db.close()
        (depot.db.waiting / FULL_COPY_NEXT).unlink()
        return before
    with depot.db.bulk() as con:
        con.execute("DROP INDEX text_decisions_decision")
        con.execute("UPDATE about_revisions SET format = ?", (format_,))
    if format_ == 7:
        depot.db.close()
        (depot.db.waiting / FULL_COPY_NEXT).unlink()
        return before
    with depot.db.bulk() as con:
        for view in VIEWS:  # the connection's own views read the columns
            con.execute(f"DROP VIEW IF EXISTS temp.{view}")
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
    texts = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",), tags=("a",) if n % 2 else (),
                               title=f"Text number {n}" if n % 3 else None) for n in range(12)], me)
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
    assert [one[-1] for one in after["about_revisions"]] == [4, 5, 6, 7, 8, 9, 10, 11, 12] and after["changes"][-1][1] == "upgrade"
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
    real = upgrade._recount_6

    def stops(con):
        raise OSError("the machine stopped")

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 5)
        patch.setattr(upgrade, "_recount_6", stops)
        with pytest.raises(OSError):
            upgrade.upgrade(depot.address, log=lambda line: None)
    for way_in in (Depot.view, Depot.open):  # at format 6 half-way, so older than this code's too
        with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
            way_in(depot.address)
    assert upgrade._recount_6 is real
    said = []

    upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 5 and upgraded.format() == FORMAT
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [5, 6, 7, 8, 9, 10, 11, 12]
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
    with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
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
    assert [one[-1] for one in after["about_revisions"]] == [6, 7, 8, 9, 10, 11, 12]
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


def test_an_upgrade_to_format_8_stopped_while_it_builds_the_decision_index_builds_it_when_run_again(
        make_depot, me, monkeypatch):
    new = make_depot("new")
    depot = decided(make_depot(), me)
    before = made_older(depot, 7)
    with pytest.raises(OtherFormat):
        Depot.open(depot.address)
    with monkeypatch.context() as patch:  # the build fails as a stop in its statement leaves it
        patch.setattr(upgrade, "TEXT_DECISIONS_DECISION", "CREATE INDEX text_decisions_decision ON no_table (x)")
        with pytest.raises(apsw.SQLError):
            upgrade.upgrade(depot.address, log=lambda line: None)
    for way_in in (Depot.view, Depot.open):  # at format 8 half-way, so older than this code's too
        with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
            way_in(depot.address)
    said = []

    upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 7 and upgraded.format() == FORMAT
    assert schema_of(depot.address) == schema_of(new.address)
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [7, 8, 9, 10, 11, 12]
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    assert "text_decisions_decision" in said[0] and any("at format 8" in line for line in said)

    _, start = upgrade.upgrade(depot.address, log=said.append)
    assert start == FORMAT and table_rows(depot.address) == after


def test_an_upgrade_to_format_9_fills_what_libraries_hold_exactly_and_goes_on_after_a_stop(
        make_depot, me, monkeypatch, library_figures):
    # the dataset form of an upgraded depot reads these counts, so the fill must give what every write keeps, and on
    # the public depot it runs for a long time, so a stop must not start it over
    depot = decided(make_depot(), me)
    texts = [one["id"] for one in depot.rows("SELECT id FROM texts WHERE status = 'admitted'")]
    depot.put(depot.new_library("second"), texts)  # texts two libraries hold
    stored, direct = library_figures(depot)
    assert stored == direct and stored
    before = made_older(depot, 8)
    real = upgrade._merged_9
    given = []

    def stops_after_its_first_sums(*streams):
        for one in real(*streams):
            given.append(one)
            yield one
            raise OSError("the machine stopped")

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 2)
        patch.setattr(upgrade, "_merged_9", stops_after_its_first_sums)
        with pytest.raises(OSError):
            upgrade.upgrade(depot.address, log=lambda line: None)
    with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
        Depot.open(depot.address)
    reader = apsw.Connection(f"file:{Path(depot.address).as_posix()}?mode=ro",
                             flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    last_text = reader.execute("SELECT max(text) FROM memberships").fetchall()[0][0]
    reader.close()
    assert given[0][1] < last_text  # it stopped with texts left to count
    said = []

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 2)
        upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 8 and upgraded.format() == FORMAT
    assert f"going on after the text {given[0][1]}" in said[0] and any("rows a second" in line for line in said)
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [8, 9, 10, 11, 12]
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    assert library_figures(upgraded)[0] == stored


def test_an_upgrade_to_format_9_gives_a_dataset_revision_figures_only_when_they_are_exact(make_depot, me, figures_at):
    # the pages show a revision's stored figures as what a build of it takes, so the upgrade gives them only to a
    # revision whose libraries nothing changed after the change it reads them at, and leaves the others unknown
    depot = decided(make_depot(), me)
    first = depot.rows("SELECT id FROM libraries WHERE name = 'first'")[0]["id"]
    texts = [one["id"] for one in depot.rows("SELECT id FROM texts WHERE status = 'admitted' ORDER BY id")]
    second = depot.new_library("second")
    depot.put(second, texts[:2])
    still = depot.new_dataset("still", "tok", [(None, first)], [("a", 1.0)])
    moved = depot.new_dataset("moved", "tok", [(None, second), (None, first)])
    depot.put(second, texts[2:4])  # after the change moved reads it at
    # a text the library holds gets another tag after the change the dataset reads it at: no membership moves
    tagged, kept = depot.add([NewText(text="tagged", licenses=("CC0-1.0",), tags=("a",)),
                              NewText(text="kept", licenses=("CC0-1.0",), tags=("a",))], me)
    depot.decide("admitted", [tagged, kept], me)
    third, fourth = depot.new_library("third"), depot.new_library("fourth")
    depot.put(third, [tagged])
    retagged = depot.new_dataset("retagged", "tok", [(None, third)], [("a", 1.0)])
    depot.edit_credits(tagged, {"tags": ["b"]}, me)
    # a locked library is read by its pins, at the change that locked it: a pinned text's tag changed after that
    depot.put(fourth, [kept])
    depot.lock(fourth)
    depot.edit_credits(kept, {"tags": ["b"]}, me)
    pinned = depot.new_dataset("pinned", "tok", [(None, fourth)], [("a", 1.0)])
    exact = {one: figures_at(depot, one) for one in (still, moved, retagged, pinned)}
    assert exact[retagged][0]["tags"][0]["texts"] == exact[pinned][0]["tags"][0]["texts"] == 1  # read before the change
    made_older(depot, 8)

    upgraded, _ = upgrade.upgrade(depot.address, log=lambda line: None)

    assert [one["stored"] for one in upgraded.dataset_figures(still)["libraries"]] == exact[still]
    assert [one["stored"] for one in upgraded.dataset_figures(moved)["libraries"]] == [None, exact[moved][1]]
    assert [one["stored"] for one in upgraded.dataset_figures(retagged)["libraries"]] == [None]
    assert [one["stored"] for one in upgraded.dataset_figures(pinned)["libraries"]] == [None]
    assert {one: figures_at(upgraded, one) for one in exact} == exact  # what a build reads is what it was


def test_an_upgrade_to_format_9_of_a_clone_gives_its_own_dataset_revisions_no_figures(make_depot, me, serve, tmp_path):
    # a clone's rows carry its upstream's numbering beside its own, so the upgrade cannot prove when its libraries last
    # moved, and leaves the figures of its own datasets unknown rather than risk wrong ones
    upstream = decided(make_depot("upstream"), me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace", "grace@example.org"))
    texts = [one["id"] for one in follower.rows("SELECT id FROM texts WHERE status = 'admitted'")]
    library = follower.new_library("mine")
    follower.put(library, texts)
    dataset = follower.new_dataset("mine", "tok", [(None, library)])
    assert follower.dataset_figures(dataset)["libraries"][0]["stored"] is not None
    made_older(follower, 8)

    upgraded, _ = upgrade.upgrade(follower.address, log=lambda line: None)

    assert [one["stored"] for one in upgraded.dataset_figures(dataset)["libraries"]] == [None]


def test_an_upgrade_to_format_10_lists_and_counts_what_the_writes_keep_and_goes_on_after_a_stop(
        make_depot, me, monkeypatch, title_index):
    # the pages read these listings and counts instead of the texts, so the fill must give what every write keeps,
    # and on the public depot it runs for over an hour, so a stop must not start it over
    new = make_depot("new")
    depot = decided(make_depot(), me)
    texts = [one["id"] for one in depot.rows("SELECT id FROM texts ORDER BY id")]
    library = depot.rows("SELECT id FROM libraries")[0]["id"]
    depot.edit_scores([{"id": one, "scores": [{"name": "rank", "value": 1, "content_version": 1}]}
                       for one in texts[:5]], me)
    depot.lock(library)
    pinned = depot.rows("SELECT text FROM pins")[0]["text"]
    depot.edit_credits(pinned, {"tags": ["poems"]}, me)  # a newer revision than the locked library's pin
    assert depot.counts()["score"] == {"rank": 5} and depot.counts()["pinned newer"] == {library: 1}
    depot.edit_credits_many([{"id": texts[1], "title": "Renamed"}, {"id": texts[2], "title": None}], me)
    indexed = title_index(depot)
    before = made_older(depot, 9)
    said = []

    def stops_after_its_first_batch(line):
        said.append(line)
        if "revision row" in line:
            raise OSError("the machine stopped")

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 3)
        patch.setattr(upgrade, "SAY_EVERY", 0)
        with pytest.raises(OSError):
            upgrade.upgrade(depot.address, log=stops_after_its_first_batch)
    with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
        Depot.open(depot.address)
    said.clear()

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 3)
        upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 9 and upgraded.format() == FORMAT
    assert "rows 3 to" in said[0] and "going on after the last transaction" in said[0]
    assert schema_of(depot.address) == schema_of(new.address)
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [9, 10, 11, 12]
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    assert title_index(upgraded) == indexed and indexed["terms"] and indexed["unfound"] == []
    assert [one["id"] for one in upgraded.search(Filter(search="number 1"))["rows"]] == ["depot-2", "depot-12"]


def test_an_upgrade_to_format_11_stopped_while_it_builds_the_long_content_index_builds_it_when_run_again(
        make_depot, me, monkeypatch):
    # a corpus build finds a library's long texts through this index, so an upgraded depot must hold it as a new
    # one does, and a stop during its build, which reads every content version, must leave a depot only the upgrade
    # opens
    new = make_depot("new")
    depot = decided(make_depot(), me)
    before = made_older(depot, 10)
    with pytest.raises(OtherFormat):
        Depot.open(depot.address)
    with monkeypatch.context() as patch:  # the build fails as a stop in its statement leaves it
        patch.setattr(upgrade, "LONG_11", "CREATE INDEX content_lists_long ON no_table (x)")
        with pytest.raises(apsw.SQLError):
            upgrade.upgrade(depot.address, log=lambda line: None)
    for way_in in (Depot.view, Depot.open):
        with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
            way_in(depot.address)
    said = []

    upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 10 and upgraded.format() == FORMAT
    assert schema_of(depot.address) == schema_of(new.address)
    after = table_rows(depot.address)
    assert [one[-1] for one in after["about_revisions"]] == [10, 11, 12]
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    assert "content_lists_long" in said[0] and any("at format 11" in line for line in said)


def test_an_upgrade_to_format_12_gives_every_text_a_kind_goes_on_after_a_stop_and_declares_the_kind_not_null(
        make_depot, me, monkeypatch):
    # on the public depot it rewrites over a hundred million content versions, so a stop must not start it over,
    # and a depot it leaves must refuse a content version of no kind as a new one does; what each library holds is
    # counted by kind, each kind on its own, as a new depot counts it, and so are the figures of the datasets; and a
    # depot keeps each kind's schema in its own table, so the conversation kind must take the schema with excerpts
    # a new depot starts with, and the table quotes through which an erasure finds the conversations quoting a
    # text must hold each conversation's quotes, as a new depot writes them
    new = make_depot("new")
    depot = decided(make_depot(), me)
    [quoted] = depot.rows("SELECT id, text FROM texts WHERE kind = 'plain' ORDER BY id LIMIT 1")
    quoting = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Read this."}, {
            "type": "excerpt", "source": quoted["id"],
            "selector": {"type": "TextQuoteSelector", "exact": quoted["text"]}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Read."}]}]}
    probe = json.dumps({"question": "What is your name?", "patterns": ["(?i)password"]})
    items = depot.add([NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",)),
                       NewText(kind="probe", content=probe, licenses=("CC0-1.0",)),
                       NewText(kind="conversation", content=json.dumps(quoting), licenses=("CC0-1.0",))], me)
    depot.decide("admitted", items, me)
    mixed = depot.new_library("mixed")
    depot.put(mixed, items)
    first = depot.rows("SELECT id FROM libraries WHERE name = 'first'")[0]["id"]
    dataset = depot.new_dataset("both", "classics", [(None, first), (None, mixed)])
    before = made_older(depot, 11)
    said = []

    def stops_after_its_first_batch(line):
        said.append(line)
        if "content row" in line:
            raise OSError("the machine stopped")

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 3)
        patch.setattr(upgrade, "SAY_EVERY", 0)
        with pytest.raises(OSError):
            upgrade.upgrade(depot.address, log=stops_after_its_first_batch)
    for way_in in (Depot.view, Depot.open):
        with pytest.raises((HalfUpgraded, OtherFormat), match="wpt depot upgrade"):
            way_in(depot.address)
    said.clear()

    with monkeypatch.context() as patch:
        patch.setattr(upgrade, "BATCH", 3)
        upgraded, start = upgrade.upgrade(depot.address, log=said.append)

    assert start == 11 and upgraded.format() == FORMAT
    assert "rows 3 to" in said[0] and "going on after the last transaction" in said[0]
    # the column kind NOT NULL, the index over plain, the table quotes and its index
    assert schema_of(depot.address) == schema_of(new.address)
    after = table_rows(depot.address)
    assert after["kinds"] == table_rows(new.address)["kinds"]
    assert [row[-1] for row in after["quotes"]] == [quoted["id"]]
    # the step writes the schema it holds itself, never the kinds folder's, and it is a new depot's of format 12
    [(written,)] = upgraded.db.reader().execute("SELECT schema FROM kinds WHERE id = 'conversation'").fetchall()
    [(fresh,)] = new.db.reader().execute("SELECT schema FROM kinds WHERE id = 'conversation'").fetchall()
    assert written == upgrade.CONVERSATION_12 == fresh
    assert [one[-1] for one in after["about_revisions"]] == [11, 12]
    assert {table: rows for table, rows in after.items() if table not in ("about_revisions", "changes")} == {
        table: rows for table, rows in before.items() if table not in ("about_revisions", "changes")}
    kinds = upgraded.sql("SELECT kind, count(*) AS n FROM content_lists GROUP BY kind ORDER BY kind")
    assert [one["kind"] for one in kinds] == ["conversation", "plain", "probe"]
    plain_only, both = (one["stored"]["kinds"] for one in upgraded.dataset_figures(dataset)["libraries"])
    assert [(kind, one["texts"]) for kind, one in plain_only.items()] == [
        ("plain", 4), ("conversation", 0), ("probe", 0), ("rating", 0), ("task", 0)]
    assert {kind: one["texts"] for kind, one in both.items() if one["texts"]} == {"conversation": 2, "probe": 1}
    assert upgraded.db.reader().execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    with pytest.raises(apsw.ConstraintError):
        with upgraded.db.bulk() as con:
            con.execute("UPDATE content_lists SET kind = NULL")


def test_the_upgrade_to_format_12_never_takes_another_depot_s_library_as_proved_by_its_counts(make_depot, me):
    # the counts hold only what this depot's own libraries hold, so they prove nothing about a library a dataset
    # reads from another depot, nor about one whose id another depot owns
    depot = decided(make_depot(), me)
    first = depot.rows("SELECT id FROM libraries WHERE name = 'first'")[0]["id"]
    change = depot.change()
    con = depot.db.reader()

    assert upgrade._proved_12(con, {"depot": None, "library": first, "change": change})
    assert not upgrade._proved_12(con, {"depot": "https://elsewhere.example/", "library": first, "change": change})
    assert not upgrade._proved_12(con, {"depot": None, "library": f"elsewhere-{ids.number(first)}",
                                        "change": change})
