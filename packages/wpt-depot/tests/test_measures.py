import json

import pytest

from wpt_depot import NewText
from wpt_depot.measures import RUNNING, SYMBOLS, Measure, Walked, measured, score_lines

PROSE = "The river rises in the hills and runs north to the sea."
TABLE = "| Population | 14 (2011) [image: Edit this on Wikidata] |\n|---|---|\n| Country | Lithuania |"


@pytest.mark.parametrize("text, symbols, running", [
    (PROSE, 0.0182, 1.0),  # one full stop among 55 characters
    (TABLE, 0.2198, 0.0),  # only table rows: 20 symbols among 91 characters
    ("# 1674\n\n## Events\n\n## Births", 0.1786, 0.0),  # only headings: an empty skeleton page
    ("[image: icon]\n- \n" + PROSE, 0.0694, 0.7639),  # a placeholder and an empty list item count for nothing
    ("See [the map](<https://example.org/a-b>) here.", 0.2826, 0.3913),  # its link's target counts for nothing
    ("हिन्दी भाषा", 0.0, 1.0),  # Devanagari's marks are writing, not symbols
    ("snake_case", 0.1, 1.0),  # an underscore is a symbol
    ("", 0.0, 0.0),
])
def test_a_text_s_symbols_and_running_text_are_shares_of_its_characters(text, symbols, running):
    assert measured(text) == {SYMBOLS: symbols, RUNNING: running}


def test_a_text_fed_in_parts_cut_inside_its_lines_measures_as_it_does_whole():
    text = TABLE + "\n" + PROSE + "\n# Heading\n" + PROSE
    for cut in range(1, len(text)):
        one = Measure()
        one.feed(text[:cut])
        one.feed(text[cut:])
        assert one.values() == measured(text)


def test_the_lines_of_a_depot_s_texts_are_a_scores_change_it_records_and_items_are_left_out(make_depot, me,
                                                                                          monkeypatch):
    monkeypatch.setattr("wpt_depot.measures.BATCH", 2)  # several batches
    depot = make_depot()
    prose, table, other = depot.add([NewText(text=PROSE, licenses=("CC0-1.0",), tags=("encyclopedia",)),
                                     NewText(text=TABLE, licenses=("CC0-1.0",), tags=("encyclopedia",)),
                                     NewText(text=PROSE, licenses=("CC0-1.0",))], me)
    depot.add([NewText(kind="conversation", content=json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]}), licenses=("CC0-1.0",))], me)
    walked = Walked()

    lines = list(score_lines(depot, "true", walked))

    assert [line["id"] for line in lines] == sorted([prose, table, other])
    assert walked.measured == 3 and walked.items == 1
    assert {line["id"]: {one["name"]: one["value"] for one in line["scores"]} for line in lines}[table] == (
        measured(TABLE))
    assert [line["id"] for line in score_lines(depot, "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = "
                                                      "'encyclopedia')")] == sorted([prose, table])
    _, _, changed, _ = depot.edit_scores(lines, me)
    assert changed == 3
    assert depot.rows("SELECT scores FROM texts WHERE id = ?", (table,))[0]["scores"] == [{"name": RUNNING, "value": 0.0, "content_version": 1},
                                           {"name": SYMBOLS, "value": measured(TABLE)[SYMBOLS],
                                            "content_version": 1}]


def test_texts_measured_in_worker_processes_give_the_lines_measured_here(make_depot, me, monkeypatch):
    monkeypatch.setattr("wpt_depot.measures.BATCH", 2)
    depot = make_depot()
    depot.add([NewText(text=text, licenses=("CC0-1.0",)) for text in [PROSE, TABLE, PROSE + "\n" + TABLE] * 3], me)
    here = list(score_lines(depot, "true"))

    monkeypatch.setattr("wpt_depot.measures.IN_PROCESS", 4)  # past two batches, the rest goes to workers
    assert list(score_lines(depot, "true")) == here
