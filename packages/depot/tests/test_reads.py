"""What a page, a count step and a write of one text read does not grow with the depot: each route is run on two
depots that differ only in how many texts they hold, and the SQLite steps its statements take are compared. A route
that reads a whole table, or every row of a large tag or license, takes about twice the steps on the depot
twice as large; one that reads a listing's window, an index range or the rows of the texts it names takes the same."""

import apsw

from depot import NewText, follow, sizes
from depot import depot as depots
from depot.filters import Filter

BOUND = 50  # the window, the count step and SORTED of both depots: far below the texts of each
SOME = 60  # texts of the decision on them, the same in both depots: more than BOUND


def steps_of(depot, action) -> int:
    """The SQLite virtual machine steps every statement of the depot's connections takes while action runs."""
    taken = []

    def counted(event):
        taken.append(event.get("stmt_status", {}).get("SQLITE_STMTSTATUS_VM_STEP", 0))

    connections = [depot.db.reader(), depot.db.person(), depot.db._write_connection()]
    for one in connections:
        one.trace_v2(apsw.SQLITE_TRACE_PROFILE, counted)
    try:
        action()
    finally:
        for one in connections:
            one.trace_v2(0, None)
    return sum(taken)


def holding(make_depot, me, legacy_rows, texts: int, name: str):
    """A depot of texts admitted texts, every one with the tag common, the license CC0-1.0 and a score, 5 more with
    the tag rare and the license MIT, and SOME more admitted by a decision of their own; the legacy rows of an open
    library holding the first texts, which a pull reads; the same rows but for the number of texts."""
    depot = make_depot(name)
    depot.add_license("MIT", "MIT License", "text")
    ids = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",), tags=("common",)) for n in range(texts)], me)
    rare = depot.add([NewText(text=f"rare {n}", licenses=("MIT",), tags=("rare",)) for n in range(5)], me)
    depot.decide("admitted", ids + rare, me)
    depot.edit_scores([{"id": one, "scores": [{"name": "rank", "value": 1, "content_version": 1}]} for one in ids], me)
    legacy_rows.library(depot, "every", ids)
    some_ids = depot.add([NewText(text=f"some {n}", licenses=("CC0-1.0",)) for n in range(SOME)], me)
    decision = depot.decide("admitted", some_ids, me)[0]
    snapshot = depot.create_snapshot()["id"]
    return depot, {"order": sizes.build(depot, snapshot), "ids": ids, "rare": rare, "some_ids": some_ids,
                   "decision": decision}


def routes(depot, me, held) -> dict:
    """{route: the action that runs it} of each page, count step, count and write of one text that the depot's
    pages and common commands run."""
    held_ = held
    ids = held["ids"]
    one, other = ids[len(ids) // 2], ids[len(ids) // 2 + 1]

    def far_page(selection, **options):
        found = depot.search(selection, **options)
        if found["next"]:
            depot.search(selection, after=found["next"], **options)

    def walk(selection, expected):
        found = depot.search(selection, limit=BOUND)
        rows = [one["id"] for one in found["rows"]]
        while found["next"] is not None:
            found = depot.search(selection, after=found["next"], limit=BOUND)
            rows += [one["id"] for one in found["rows"]]
        assert sorted(rows) == sorted(expected)

    def held(selection):
        assert depot.held_count(selection) is not None

    def pull_rows():
        change = depot.change()
        for table in follow.BY_CHANGE:
            list(follow.rows(depot, table, {depot.about()["name"]: [change - 1, change]}))

    return {
        "page of every text": lambda: far_page(Filter()),
        "page of every text by status": lambda: far_page(Filter(), order="status"),
        "page of every text by length": lambda: far_page(Filter(), order="length", descending=True),
        "page of a common tag by length": lambda: far_page(Filter(tags=("common",)), order="length"),
        "passages of a status by length": lambda: depot.passages(Filter(statuses=("admitted",)), limit=10,
                                                                 order="length", descending=True),
        "page of a snapshot's strip": lambda: sizes.page(depot, held_["order"], 100, limit=10),
        "page of a common tag": lambda: far_page(Filter(tags=("common",))),
        "page of a rare tag": lambda: far_page(Filter(tags=("rare",))),
        "page of a rare tag by title": lambda: far_page(Filter(tags=("rare",)), order="title"),
        "page of a license": lambda: far_page(Filter(licenses=("CC0-1.0",))),
        "page of a status": lambda: far_page(Filter(statuses=("admitted",))),
        "page of a search": lambda: far_page(Filter(search="nothing like it")),
        "page of a search in the text": lambda: far_page(Filter(search="nothing like it", in_text=True)),
        "every page of a decision on many texts": lambda: walk(Filter(decision=held_["decision"]), held_["some_ids"]),
        "count of a tag's statuses": lambda: held(Filter(tags=("common",))),
        "count of a license's statuses": lambda: held(Filter(licenses=("MIT",))),
        "count step of every text": lambda: depot.count(Filter(), size=BOUND),
        "count step of a tag": lambda: depot.count(Filter(tags=("common",)), size=BOUND),
        "count step of a decision": lambda: depot.count(Filter(decision=held_["decision"]), size=BOUND),
        "texts a license blocks": lambda: depot.blocked(Filter()),
        "score names": depot.score_names,
        "score values": lambda: depot.score_values("rank"),
        "rows a pull reads": pull_rows,
        "refusal of one text": lambda: depot.decide("refused", [one], me),
        "admission of one text by filter": lambda: depot.decide("admitted", Filter(ids=(one,)), me),
        "retag of one text": lambda: depot.retag([one], ["y"], [], me),
        "relicense of one text": lambda: depot.relicense([other], ["MIT"], me),
        "content of one text": lambda: depot.change_content(other, "a new text", None, me),
        "credits of one text": lambda: depot.edit_credits(one, {"tags": ["x"]}, me),
        "scores of one text": lambda: depot.edit_scores(
            [{"id": one, "scores": [{"name": "rank", "value": 2, "content_version": 1}]}], me),
        "score of one text": lambda: depot.score(Filter(ids=(one,)), "other", 1.0, me),
        "erasure of one text": lambda: depot.decide_erasure(
            depot.ask_erasure("all of it", [(other, None, None)], me), "admitted", me),
    }


def test_each_page_count_and_write_of_one_text_reads_as_much_on_a_depot_twice_as_large(make_depot, me, monkeypatch,
                                                                                        legacy_rows):
    # the public depot holds 166 million texts: a route whose reads grow with the depot times out there
    for constant in ("WINDOW", "TEXT_WINDOW", "SORTED"):
        monkeypatch.setattr(depots, constant, BOUND)
    small, held_small = holding(make_depot, me, legacy_rows, 400, "small")
    large, held_large = holding(make_depot, me, legacy_rows, 800, "large")
    on_small, on_large = routes(small, me, held_small), routes(large, me, held_large)
    grown = {}
    for route in on_small:
        steps = steps_of(small, on_small[route]), steps_of(large, on_large[route])
        if steps[1] > steps[0] * 1.2 + 200:
            grown[route] = steps
    assert grown == {}, "routes whose steps grow with the depot (small, large): " + repr(grown)


def test_a_page_in_the_order_of_ids_looks_up_its_texts_only_up_to_its_last_row_not_its_whole_window(make_depot, me,
                                                                                                    legacy_rows):
    # the window (depot.WINDOW, 10,000 texts) holds every text of both depots: a first page of 5 rows still reads as
    # much on the larger one, so a workspace opens as fast whatever the window holds
    small, _ = holding(make_depot, me, legacy_rows, 400, "small")
    large, _ = holding(make_depot, me, legacy_rows, 800, "large")
    grown = {}
    for name, selection in (("every text", Filter()), ("a tag", Filter(tags=("common",))),
                            ("a status", Filter(statuses=("admitted",)))):
        for descending in (False, True):
            steps = [steps_of(one, lambda: one.search(selection, limit=5, descending=descending))
                     for one in (small, large)]
            if steps[1] > steps[0] * 1.2 + 200:
                grown[(name, descending)] = steps
    assert grown == {}, "pages whose steps grow with the window (small, large): " + repr(grown)
