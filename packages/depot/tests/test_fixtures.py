"""The checked-in fixture depots (tests/fixtures, README.md there) hold every case the upgrade and the legacy API's
tests read from them. Read with plain SQL on a read-only connection, never through depot, so the check stays
the same whatever format the code is at."""

import json
import sqlite3
from compression import zstd
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def rows(con, query: str, params=()) -> list[tuple]:
    return con.execute(query, params).fetchall()


@pytest.fixture(params=[11, 12])
def fixture(request):
    con = sqlite3.connect(f"{(FIXTURES / f'format-{request.param}.sqlite').as_uri()}?mode=ro", uri=True)
    yield request.param, con
    con.close()


def test_each_fixture_is_of_its_format_and_follows_an_upstream_whose_texts_and_libraries_it_received(fixture):
    form, con = fixture
    [(written, upstream)] = rows(con, "SELECT format, upstream FROM about_revisions ORDER BY revision DESC LIMIT 1")
    assert written == form and upstream.startswith("http://127.0.0.1:")
    owners = dict(rows(con, "SELECT substr(id, 1, instr(id, '-') - 1), count(*) FROM texts_revisions WHERE latest "
                            "GROUP BY 1"))
    assert owners["up"] > 0 and owners["fixture"] > 0
    assert rows(con, "SELECT count(*) FROM libraries_revisions WHERE latest AND id LIKE 'up-%' "
                     "AND locked_at IS NOT NULL") == [(2,)]
    assert rows(con, "SELECT count(*) FROM pulls")[0][0] >= 1


def test_libraries_locked_at_different_changes_an_open_one_and_a_text_edited_after_a_lock(fixture):
    _, con = fixture
    libraries = {name: (library, locked) for library, name, locked in rows(
        con, "SELECT id, name, locked_at IS NOT NULL FROM libraries_revisions WHERE latest")}
    locks = {name: rows(con, "SELECT min(change) FROM pins WHERE library = ?", (library,))[0][0]
             for name, (library, locked) in libraries.items() if locked and library.startswith("fixture")}
    assert set(locks) == {"lib-a", "lib-b"} and locks["lib-a"] < locks["lib-b"]
    assert not libraries["lib-c"][1] and rows(con, "SELECT count(*) FROM memberships WHERE library = ? AND "
                                                   "left_change IS NULL", (libraries["lib-c"][0],))[0][0] > 0
    edited = rows(con, "SELECT count(*) FROM pins p JOIN texts_revisions t ON t.id = p.text AND t.latest "
                       "WHERE p.library = ? AND t.revision > p.revision", (libraries["lib-a"][0],))[0][0]
    assert edited >= 1
    problems = {name: json.loads(found) for name, found in rows(
        con, "SELECT name, problems FROM libraries_revisions WHERE latest")}
    assert problems["lib-a"] and problems["lib-c"]


def test_two_datasets_one_of_two_locked_libraries_one_reading_a_locked_library_past_its_lock(fixture):
    _, con = fixture
    lock = dict(rows(con, "SELECT library, min(change) FROM pins GROUP BY library"))
    datasets = {name: (revision, json.loads(libraries)) for name, revision, libraries in rows(
        con, "SELECT name, revision, libraries FROM datasets_revisions WHERE latest")}
    revision, two = datasets["ds-two"]
    assert revision >= 3 and len(two) == 2 and all(ref["library"] in lock for ref in two)
    assert len({lock[ref["library"]] for ref in two}) == 2
    _, mixed = datasets["ds-mixed"]
    assert any(ref["library"] in lock and ref["change"] > lock[ref["library"]] for ref in mixed)
    assert any(ref["library"] not in lock for ref in mixed)  # an open library
    assert any(ref["library"].startswith("up-") for ref in mixed)  # the upstream's


def test_tokenizer_recipes_a_vocabulary_and_erasures_one_carried_out(fixture):
    _, con = fixture
    assert rows(con, "SELECT count(*) FROM tokenizers_revisions WHERE latest") == [(2,)]
    assert rows(con, "SELECT max(version) FROM tokenizers_revisions") == [(2,)]
    assert rows(con, "SELECT count(*) FROM vocabularies") == [(1,)]
    erasures = rows(con, "SELECT erased FROM erasures_revisions WHERE latest")
    assert len(erasures) == 2 and sum(found not in (None, "[]") for (found,) in erasures) == 1


def test_each_dock_holds_corpora_of_two_revisions_of_ds_two(fixture):
    form, con = fixture
    folders = sorted((FIXTURES / f"format-{form}-store" / "corpora").iterdir())
    assert [folder.name.rsplit("-", 1)[0] for folder in folders] == ["ds-two", "ds-two"]
    records = [json.loads(zstd.decompress((folder / "corpus.json.zst").read_bytes())) for folder in folders]
    [current] = rows(con, "SELECT revision FROM datasets_revisions WHERE latest AND name = 'ds-two'")
    assert sorted(record["dataset"]["revision"] for record in records) == [current[0] - 1, current[0]]
