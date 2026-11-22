import hashlib
import json
import re
import time
import tracemalloc
from pathlib import Path

import pytest

from wpt_depot import About, Declaration, Depot, Needle, NewText, clone, pull
from wpt_depot import depot as depots
from wpt_depot import counts, crediting, erasure, follow, policy, quotes, scoring
from wpt_depot.errors import (BadContent, BadQuery, BelongsToUpstream, NoDepotSetting, NotFound,
                              Refused, TimeLimitReached, UpstreamUnreachable)
from wpt_depot.depot import BELONGS_TO_UPSTREAM, NO_ALLOWED_LICENSE
from wpt_depot.filters import Filter
from wpt_depot.schema import FORMAT, KEYS
from wpt_depot.settings import depot_address

from .conftest import sizes_apart, tags_apart


def column(rows: list[dict], name: str) -> list:
    """The values of one column of rows, in order."""
    return [one[name] for one in rows]


def pick(rows: list[dict], names: list[str]) -> list[dict]:
    """rows with only the columns named."""
    return [{name: one[name] for name in names} for one in rows]


def every_row(depot) -> dict[str, list[dict]]:
    """Every row of every table of the depot, in the order of the table's key."""
    return {table: depot.db.query(f"SELECT * FROM {table} ORDER BY {', '.join(keys)}") for table, keys in KEYS.items()}


def plain(text, *tags, notice=None):
    return NewText(text=text, licenses=("CC0-1.0",), tags=tags, notice=notice)


def with_text(depot, table) -> list[dict]:
    """Rows of texts, each with the text of the content version it reads."""
    return [{**one, "text": depot.content(one["id"], one["content_version"])} for one in table]


def texts_of(depot, table) -> list[dict]:
    return [{"id": one["id"], "text": one["text"]} for one in with_text(depot, table)]


def files_holding(address, passage: str) -> list[Path]:
    """The depot's file and its WAL, those whose bytes hold passage: what an erasure must leave none of (its
    change files wait for the next backup, which sends them away, wpt_depot.backup)."""
    path = Path(address)
    return [one for one in (path, path.with_name(path.name + "-wal"))
            if one.is_file() and passage.encode() in one.read_bytes()]


def test_an_item_is_kept_only_when_its_content_matches_its_kind(make_depot, me):
    depot = make_depot()
    probe = json.dumps({"question": "What is your name?", "patterns": ["(?i)password"]})

    [item] = depot.add([NewText(kind="probe", content=probe, licenses=("CC0-1.0",))], me)

    assert depot.show(item)["text"]["content"] == probe
    for wrong in (json.dumps({"question": "Name?"}), "not JSON"):
        with pytest.raises(BadContent):
            depot.add([NewText(kind="probe", content=wrong, licenses=("CC0-1.0",))], me)
    with pytest.raises(Refused):
        depot.add([NewText(kind="poem", content="{}", licenses=("CC0-1.0",))], me)
    with pytest.raises(Refused):
        depot.add([NewText(text="no license")], me)
    assert column(depot.sql("SELECT count(*) AS n FROM texts"), "n") == [1]
    assert sorted(column(depot.sql("SELECT id FROM kinds"), "id")) == ["conversation", "plain", "probe", "rating", "task"]


def test_a_text_s_kind_is_fixed_when_it_is_made_and_a_change_of_kind_is_refused(make_depot, me):
    # each training step reads the kinds it names, so a text that changed kind would move between steps unseen
    depot = make_depot()
    probe = json.dumps({"question": "What is your name?", "patterns": ["(?i)password"]})
    text, item = depot.add([NewText(kind="plain", text="words", licenses=("CC0-1.0",)),
                            NewText(kind="probe", content=probe, licenses=("CC0-1.0",))], me)
    assert [depot.show(one)["text"]["kind"] for one in (text, item)] == ["plain", "probe"]

    for one, plain_, content, kind in ((text, None, probe, "probe"), (item, "words", None, "plain")):
        with pytest.raises(Refused, match="a text's kind is fixed when it is made"):
            depot.change_content(one, plain_, content, me, kind=kind)

    assert depot.change_content(text, "other words", None, me, kind="plain") is not None  # its own kind passes
    assert [depot.show(one)["text"]["kind"] for one in (text, item)] == ["plain", "probe"]
    assert depot.show(item)["text"]["content"] == probe


def test_a_decision_without_the_confirmation_writes_nothing(make_depot):
    depot = make_depot()

    with pytest.raises(Refused):
        depot.add([plain("text")], Declaration("Ada Lovelace", "ada@example.org", False))

    assert column(depot.sql("SELECT count(*) AS n FROM decisions"), "n") == [0]


def test_an_admitted_erasure_cuts_its_passage_from_that_text_only_and_drops_every_older_state(make_depot, me):
    depot = make_depot()
    secret = "555-0199"
    named, other = depot.add([plain(f"Élodie: {secret}, again {secret}.", "a"), plain(f"Not this one: {secret}")], me)
    start = len("Élodie: ")  # characters: É is two bytes in UTF-8
    request = depot.ask_erasure("a phone number", [(named, start, start + len(secret))], me)

    assert depot.decide_erasure(request, "admitted", me) is True

    assert depot.show(named)["text"]["text"] == "Élodie: , again ."
    assert depot.show(other)["text"]["text"] == f"Not this one: {secret}"
    [erased] = depot.erasures()[0]["erased"]
    assert erased["length"] == len(secret) and len(erased["hash_a"]) == 32 and len(erased["hash_b"]) == 8
    assert files_holding(depot.address, f"{secret}, again") == []


def test_an_erasure_that_removes_a_text_whole_deletes_its_row(make_depot, me):
    depot = make_depot()
    gone, kept = depot.add([plain("a whole text to remove"), plain("kept")], me)
    request = depot.ask_erasure("asked by its author", [(gone, None, None)], me)

    depot.decide_erasure(request, "admitted", me)

    assert column(depot.sql("SELECT id FROM texts"), "id") == [kept]
    assert files_holding(depot.address, "a whole text to remove") == []


def test_the_hashes_of_an_erased_passage_are_public_so_anyone_can_match_a_guess(make_depot, me):
    # the same passage gives the same hashes in two depots, and a guess at it is confirmed without any secret
    first, second = make_depot("first"), make_depot("second")
    passage = "555-0199"
    hashes = []
    for depot in (first, second):
        [text] = depot.add([plain(f"call {passage} now")], me)
        depot.decide_erasure(depot.ask_erasure("a number", [(text, 5, 13)], me), "admitted", me)
        [erased] = depot.erasures()[0]["erased"]
        hashes.append(erased)

    assert hashes[0] == hashes[1]
    assert hashes[0]["hash_a"] == hashlib.sha256(passage.encode()).digest()
    assert hashes[0]["hash_b"] == erasure.rolling(passage).to_bytes(8, "big")
    assert erasure.rolling("another guess") != erasure.rolling(passage)


def test_a_change_that_fails_partway_writes_nothing(make_depot, me, monkeypatch):
    # a change is one transaction: the relicensing writes its revision and decision before it counts them, and the
    # failure undoes it
    depot = make_depot()
    [text] = depot.add([plain("one", "a")], me)
    depot.decide("admitted", "true", me)
    depot.add_license("MIT", "MIT", "text")
    before = depot.change()

    def fails(*args, **kwargs):
        raise OSError("the disk went away")

    with monkeypatch.context() as patch:
        patch.setattr(counts, "shift", fails)
        with pytest.raises(OSError):
            depot.relicense("true", ["MIT"], me)

    assert depot.change() == before
    assert depot.show(text)["text"]["licenses"] == ["CC0-1.0"] and depot.show(text)["status"] == "admitted"
    assert column(depot.sql("SELECT count(*) AS n FROM decisions"), "n") == [2]
    assert depot.add([plain("two")], me) == ["depot-2"]


def test_a_text_s_content_is_held_once_by_the_depot_s_file(make_depot, me):
    # hex digits, which compression leaves as they are; the version an erasure cut holds the passage still
    depot = make_depot()
    passage = hashlib.sha256(b"a passage held once").hexdigest()[:48]
    texts = depot.add([plain(f"{passage}in the first text"), plain("another text")], me)
    depot.decide("admitted", texts, me)
    depot.change_content(texts[1], "another text, changed", None, me)
    request = depot.ask_erasure("a word", [(texts[0], len(passage) + 3, len(passage) + 12)], me)
    depot.decide_erasure(request, "admitted", me)
    depot.db.close()

    assert Path(depot.address).read_bytes().count(passage.encode()) == 1
    assert depot.show(texts[0])["text"]["text"] == f"{passage}in  text"


def test_sql_reads_and_refuses_anything_else(make_depot, me):
    depot = make_depot()
    depot.add([plain("one", "wikipedia", "en")], me)

    assert column(depot.sql("SELECT count(*) AS n FROM texts WHERE EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'en')"), "n") == [1]
    assert column(depot.sql("SELECT count(*) AS n FROM TEXTS t JOIN Contents c ON c.id = t.id"), "n") == [1]
    for query in ("DELETE FROM texts", "SELECT 1; SELECT 2", "COPY (SELECT 1) TO 'out.csv'"):
        with pytest.raises(BadQuery):
            depot.sql(query)


def test_the_depot_s_queries_read_only_its_file_and_change_nothing(make_depot, me, tmp_path):
    # a person's query (wpt depot sql, and the web routes that take a query or a condition) may only read the depot
    depot = make_depot()
    depot.add([plain("one")], me)
    outside = tmp_path / "outside.sqlite"

    for statement in (f"ATTACH '{outside}' AS other", "PRAGMA journal_mode = DELETE", "PRAGMA query_only = 0",
                      "DELETE FROM texts_revisions", "UPDATE licenses SET name = 'x'", "CREATE TABLE t (a)",
                      "SELECT load_extension('x')", "VACUUM INTO 'copy.sqlite'",
                      "WITH x AS (SELECT 1) DELETE FROM licenses", "SELECT 1; DELETE FROM licenses"):
        with pytest.raises(BadQuery):
            depot.sql(statement)
        with pytest.raises(BadQuery):  # a condition is a query as well
            depot.search(f"id IN ({statement})")
        with pytest.raises(BadQuery):
            depot.count(f"id IN ({statement})")

    assert not outside.exists() and not (tmp_path / "copy.sqlite").exists()
    assert column(depot.sql("SELECT count(*) AS n FROM texts"), "n") == [1]
    assert column(depot.sql("SELECT count(*) AS n FROM licenses WHERE name != 'x'"), "n") == [1]


def test_a_person_s_query_stops_at_its_deadline(make_depot, me):
    depot = make_depot()
    depot.db.deadline = time.monotonic() - 1

    with pytest.raises(TimeLimitReached):
        depot.sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n")


def test_an_erasure_cuts_what_every_revision_shows_and_no_revision_row_holds_content(make_depot, me):
    depot = make_depot()
    secret = "555-0199"
    [text] = depot.add([NewText(text=f"call {secret} now", licenses=("CC0-1.0",), title="Call")], me)
    depot.decide("admitted", "true", me)
    depot.edit_credits(text, {"title": "A call"}, me)
    assert "text" not in depot.sql("SELECT * FROM texts_revisions")[0]
    request = depot.ask_erasure("a number", [(text, 5, 13)], me)

    depot.decide_erasure(request, "admitted", me)

    assert column(depot.sql(f"SELECT DISTINCT c.text FROM texts_revisions r JOIN contents c ON c.id = r.id "
                     f"WHERE r.id = '{text}'"), "text") == ["call  now"]
    assert column(depot.sql(f"SELECT count(*) AS n FROM texts_revisions WHERE id = '{text}'"), "n") == [2]
    assert files_holding(depot.address, secret) == []


def test_the_sql_names_show_the_current_state_and_their_revisions_every_row(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    depot.edit_credits(text, {"title": "First"}, me)
    depot.edit_about(maintainer="Grace Hopper")

    assert depot.sql("SELECT id, text, revision, title FROM texts") == [
        {"id": text, "text": "one", "revision": 2, "title": "First"}]
    assert column(depot.sql("SELECT title FROM texts_revisions ORDER BY revision"), "title") == [None, "First"]
    assert column(depot.sql("SELECT maintainer FROM about_revisions ORDER BY revision"), "maintainer") == [
        "Ada Lovelace", "Grace Hopper"]
    for name in ("contents", "erasures_revisions", "snapshots", "snapshot_sets"):
        depot.sql(f"SELECT * FROM {name}")
    # the tables of libraries and datasets stay readable, as an older format left them
    for name in ("libraries_revisions", "memberships", "pins", "datasets_revisions"):
        assert depot.sql(f"SELECT * FROM {name}") == []
    for name in ("libraries", "datasets"):
        with pytest.raises(BadQuery):
            depot.sql(f"SELECT * FROM {name}")
    changes = column(depot.sql("SELECT change FROM texts_revisions ORDER BY revision"), "change")
    assert changes[0] < changes[1] <= depot.change()


def test_a_clone_follows_its_upstream_and_a_pull_brings_its_erasures(serve, make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    [text] = upstream.add([plain("my address is 1 Main Street")], me)
    address = serve(upstream)
    follower = clone(address, str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    assert follower.about()["upstream"] == address
    [added] = upstream.add([plain("a new text")], me)
    request = upstream.ask_erasure("an address", [(text, 14, 27)], me)
    upstream.decide_erasure(request, "admitted", me)

    upstream.decide("admitted", [added], me)
    upstream.edit_credits(added, {"title": "New"}, me)

    assert pull(follower) == [request]

    assert follower.sql(f"SELECT revision, change, title, decisions, status FROM texts WHERE id = '{added}'") \
        == upstream.sql(f"SELECT revision, change, title, decisions, status FROM texts WHERE id = '{added}'")
    assert follower.sql(f"SELECT decision, change FROM text_decisions WHERE text = '{added}' ORDER BY decision") \
        == upstream.sql(f"SELECT decision, change FROM text_decisions WHERE text = '{added}' ORDER BY decision")
    assert follower.change() > upstream.change()  # the pull itself is a change of the follower

    assert follower.sql("SELECT id, text FROM texts ORDER BY id") == [
        {"id": text, "text": "my address is "}, {"id": added, "text": "a new text"}]
    assert files_holding(follower.address, "1 Main Street") == []
    [place] = column(follower.sql(f"SELECT j.value ->> 'at' AS at FROM erasures e, json_each(e.erased) j "
                                  f"WHERE e.id = '{request}'"), "at")
    assert place == len("my address is ")  # the place recorded upstream travels with the pull


def test_overlapping_needles_of_one_string_are_cut_as_one_passage_so_nothing_of_either_remains(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("name: Ann Lee Smith, born 1901")], me)
    start = len("name: ")
    # "Ann Lee" and "Lee Smith" overlap: cut one after the other, the second would no longer be found whole
    request = depot.ask_erasure("a name", [(text, start, start + 7), (text, start + 4, start + 13)], me)

    depot.decide_erasure(request, "admitted", me)

    left = depot.show(text)["text"]["text"]
    assert left == "name: , born 1901"
    assert all(word not in left for word in ("Ann", "Lee", "Smith"))
    [erased] = depot.erasures()[0]["erased"]
    assert erased["length"] == len("Ann Lee Smith")
    assert erased["hash_a"] == hashlib.sha256(b"Ann Lee Smith").digest()


def test_touching_needles_are_one_passage_with_one_erased_entry(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("call 555-0199 now")], me)
    request = depot.ask_erasure("a number", [(text, 5, 9), (text, 9, 13)], me)

    depot.decide_erasure(request, "admitted", me)

    assert depot.show(text)["text"]["text"] == "call  now"
    [erased] = depot.erasures()[0]["erased"]
    assert (erased["length"], erased["hash_a"], erased["at"]) == (8, hashlib.sha256(b"555-0199").digest(), 5)
    shown = depot.erasure_shown(request)["needles"]
    assert [(one["state"], one["erased"], one["before"], one["after"]) for one in shown] == [
        ("erased", 8, "call ", " now"), ("erased", 8, "call ", " now")]


def test_the_place_a_passage_left_is_recorded_exactly_when_it_also_appears_earlier(make_depot, me):
    depot = make_depot()
    talk = {"messages": [{"role": "user", "content": [{"type": "text", "text": "id 42, then id 42 here"}]}]}
    [text, item] = depot.add([plain("id 42, then id 42 here"),
                              NewText(kind="conversation", content=json.dumps(talk), licenses=("CC0-1.0",))], me)
    second = len("id 42, then id ")
    field = "/messages/0/content/0/text"
    request = depot.ask_erasure("an id", [(text, second, second + 2), (item, second, second + 2, field)], me)

    depot.decide_erasure(request, "admitted", me)

    # both copies go, so the second one's place is two characters before its start
    assert depot.show(text)["text"]["text"] == "id , then id  here"
    assert [one["at"] for one in depot.erasures()[0]["erased"]] == [second - 2, second - 2]
    for one in depot.erasure_shown(request)["needles"]:
        assert (one["before"], one["after"], one["erased"]) == ("id , then id ", " here", 2)


def test_the_depot_address_comes_from_wpt_depot(tmp_path):
    env = tmp_path / ".env"
    env.write_text("WPT_DEPOT=/from/file\n", encoding="utf-8")

    assert depot_address({"WPT_DEPOT": "/from/environment"}, env) == "/from/environment"
    assert depot_address({}, env) == "/from/file"
    with pytest.raises(NoDepotSetting):
        depot_address({}, tmp_path / "absent")


def test_a_text_id_is_never_used_again_after_an_erasure_removed_its_text(serve, make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    upstream.add([plain("kept")], me)
    [gone] = upstream.add([plain("to remove whole")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    request = upstream.ask_erasure("asked", [(gone, None, None)], me)
    upstream.decide_erasure(request, "admitted", me)

    [added] = upstream.add([plain("new")], me)
    pull(follower)

    assert added != gone
    assert column(follower.sql("SELECT id FROM texts ORDER BY id"), "id") == ["upstream-1", added]


def upstream_and_clone_holding(serve, make_depot, me, tmp_path):
    """An upstream with two admitted texts, and a clone of it: (upstream, clone, texts)."""
    upstream = make_depot("upstream")
    upstream.add_license("MIT", "MIT License", "text")
    texts = upstream.add([plain("to be refused"), plain("to be relicensed")], me)
    upstream.decide("admitted", texts, me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    return upstream, follower, texts


def test_a_pull_that_fails_partway_writes_nothing_and_the_next_pull_brings_everything(serve, 
        make_depot, me, tmp_path, monkeypatch):
    upstream, follower, (refused, kept) = upstream_and_clone_holding(serve, make_depot, me, tmp_path)
    upstream.decide("refused", [refused], me)
    before = follower.change()

    def fails(*args, **kwargs):
        raise OSError("the disk went away")

    with monkeypatch.context() as patch:
        patch.setattr(follow._Tracked, "settle", fails)
        with pytest.raises(OSError):
            pull(follower)
    assert follower.change() == before and follower.show(refused)["status"] == "admitted"

    pull(follower)

    assert follower.show(refused)["status"] == "refused" and follower.show(kept)["status"] == "admitted"
    assert follower.sql("SELECT depot, change, upstream, upstream_change FROM pulls ORDER BY change")[-1] \
        == {"depot": "follower", "change": follower.change(), "upstream": "upstream", "upstream_change": upstream.change()}


def test_the_decision_that_admits_an_erasure_and_its_carrying_out_are_one_change(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("a secret")], me)
    request = depot.ask_erasure("why", [(text, 2, 8)], me)
    before = depot.change()

    depot.decide_erasure(request, "admitted", me)

    assert depot.change() == before + 1
    assert depot.erasures()[0]["erased"] is not None


def test_a_problem_reported_before_reports_went_is_still_shown_by_its_decision(tmp_path):
    # the fixture depot holds two problems reported by the code of format 12; nothing writes one any more, and the
    # decision that recorded one is shown as any other decision, its reason in its own words
    copy = tmp_path / "fixture.sqlite"
    copy.write_bytes((Path(__file__).parent / "fixtures" / "format-12.sqlite").read_bytes())
    depot, _ = Depot.upgrade(str(copy))

    found = depot.decision("fixture-8")

    assert found["decision"]["status"] == "under review" and set(found) == {"decision", "texts", "erasures"}
    assert any(one["id"] == "fixture-8" for one in depot.latest_decisions(1000))


def test_every_id_carries_the_depot_s_name_so_a_clone_adds_its_own_rows_beside_the_upstream_s(serve, make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    [theirs] = upstream.add([plain("upstream text", "a")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))

    [mine] = follower.add([plain("follower text", "a")], me)
    [more] = upstream.add([plain("another upstream text", "a")], me)
    pull(follower)

    assert (theirs, mine, more) == ("upstream-1", "follower-1", "upstream-2")
    assert column(follower.sql("SELECT id FROM texts ORDER BY id"), "id") == [mine, theirs, more]
    assert column(follower.sql("SELECT id FROM decisions ORDER BY id"), "id") == [
        "follower-1", "upstream-1", "upstream-2"]
    with pytest.raises(Refused):
        clone(serve(upstream), str(tmp_path / "same.sqlite"), About("upstream", "Ada", "ada@example.org"))
    with pytest.raises(Refused):
        Depot.new(str(tmp_path / "bad"), About("not a name", "Ada", "ada@example.org"))


def test_admitting_a_text_none_of_whose_licenses_is_allowed_is_refused_and_one_allowed_license_is_enough(make_depot, me):
    depot = make_depot()
    for license_id in ("GPL-2.0-only", "MIT AND CC-BY-4.0", "MIT AND GPL-2.0-only", "Apache-2.0 WITH LLVM-exception"):
        depot.add_license(license_id, license_id, "text")
    [refused] = depot.add([NewText(text="one", licenses=("GPL-2.0-only",))], me)
    [choice] = depot.add([NewText(text="two", licenses=("GPL-2.0-only", "CC0-1.0"))], me)
    [together] = depot.add([NewText(text="three", licenses=("MIT AND CC-BY-4.0",))], me)
    [not_together] = depot.add([NewText(text="four", licenses=("MIT AND GPL-2.0-only",))], me)
    [exception] = depot.add([NewText(text="five", licenses=("Apache-2.0 WITH LLVM-exception",))], me)

    for text in (refused, not_together):
        with pytest.raises(Refused):
            depot.decide("admitted", f"id = '{text}'", me)
    depot.decide("admitted", f"id IN ('{choice}', '{together}', '{exception}')", me)

    assert depot.show(refused)["text"]["decisions"] == ["depot-1"]
    assert [one["status"] for one in depot.show(choice)["decisions"]] == ["under review", "admitted"]
    depot.decide("refused", f"id = '{refused}'", me)  # a refusal needs no allowed license


def test_blocked_names_each_text_an_admission_cannot_cover_with_its_reasons_and_a_decision_agrees(make_depot, me):
    # the review of an admission lists these texts apart before the click, so each must be exactly a text the
    # depot then refuses to admit, and every other text must be admitted
    depot = make_depot()
    for license_id in ("CC-BY-NC-4.0", "MIT AND CC-BY-NC-4.0"):
        depot.add_license(license_id, license_id, "text")
    held, under_review = depot.add([plain("held"), plain("under review, its license allowed")], me)
    depot.decide("admitted", [held], me)
    unlicensed, combined, choice, free = depot.add([
        NewText(text="one", licenses=("CC-BY-NC-4.0",)), NewText(text="two", licenses=("MIT AND CC-BY-NC-4.0",)),
        NewText(text="three", licenses=("CC-BY-NC-4.0", "CC0-1.0")), plain("four")], me)

    found = depot.blocked([free, choice, combined, unlicensed, under_review, held])["rows"]

    assert found == [{"id": unlicensed, "reasons": [NO_ALLOWED_LICENSE]},
                     {"id": combined, "reasons": [NO_ALLOWED_LICENSE]}]
    assert depot.blocked([under_review, held])["rows"] == []
    first = depot.blocked("true", limit=1)
    assert first["rows"] == found[:1] and depot.blocked("true", after=first["next"])["rows"] == found[1:]
    for one in found:
        with pytest.raises(Refused):
            depot.decide("admitted", [one["id"], free], me)
    assert depot.show(free)["status"] == "under review"
    assert depot.decide("admitted", [free, choice, under_review, held], me)[1] == 4
    assert depot.decide("refused", [unlicensed, combined], me)[1] == 2
    assert depot.decide("admitted", "true", me, leave_blocked=True)[1:] == (4, {NO_ALLOWED_LICENSE: 2})
    with pytest.raises(NotFound):
        depot.blocked([free, "depot-99"])


def test_a_decision_may_carry_a_public_reason(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)

    depot.decide("admitted", f"id = '{text}'", me, reason="  public domain in 1923  ")

    reasons = [one["reason"] for one in depot.show(text)["decisions"]]
    assert reasons == [None, "public domain in 1923"]
    with pytest.raises(Refused):
        depot.decide("refused", f"id = '{text}'", me, reason=" ")


def test_a_needle_on_an_item_names_a_field_and_the_cut_item_must_match_its_kind_again(make_depot, me):
    depot = make_depot()
    number = "555-0199"
    talk = {"messages": [{"role": "user", "content": [{"type": "text", "text": f"call {number} now"}]},
                         {"role": "assistant", "content": [{"type": "text", "text": f"I will call {number}."}]}]}
    [item] = depot.add([NewText(kind="conversation", content=json.dumps(talk), licenses=("CC0-1.0",))], me)
    field = "/messages/0/content/0/text"

    for wrong in ((item, 5, 13, "/messages/0/content/9/text"), (item, 5, 13, "/messages/0/role"),
                  (item, 5, 99, field), (item, 5, 13, None)):
        with pytest.raises(Refused):
            depot.ask_erasure("a number", [wrong], me)
    request = depot.ask_erasure("a number", [(item, 5, 13, field)], me)
    depot.decide_erasure(request, "admitted", me)

    cut = json.loads(depot.show(item)["text"]["content"])
    assert cut["messages"][0]["content"][0]["text"] == "call  now"
    assert cut["messages"][1]["content"][0]["text"] == "I will call ."  # every place the passage appears
    assert files_holding(depot.address, number) == []


def test_a_request_is_shown_as_its_passages_in_bounded_context_then_as_their_erased_lengths(make_depot, me):
    depot = make_depot()
    long = "a" * 1000 + "call 555-0199 or 555-0123 now" + "z" * 1000
    talk = {"messages": [{"role": "user", "content": [{"type": "text", "text": "my number is 555-0100."}]}]}
    [text, item, whole] = depot.add([plain(long), NewText(kind="conversation", content=json.dumps(talk),
                                                          licenses=("CC0-1.0",)), plain("all of it, " * 200)], me)
    huge = depot.erasure_shown(depot.ask_erasure("most of it", [(text, 10, 1990)], me))["needles"][0]
    assert (huge["before"], huge["passage"], huge["omitted"], huge["passage_end"]) == ("a" * 10, "a" * 300, 1380,
                                                                                      long[1690:1990])
    first = 1000 + len("call ")
    second = 1000 + len("call 555-0199 or ")
    field = "/messages/0/content/0/text"
    request = depot.ask_erasure("numbers", [(text, first, first + 8), (text, second, second + 8),
                                            (item, 13, 21, field), (whole, None, None)], me)

    asked = depot.erasure_shown(request)
    assert (asked["status"], asked["carried_out"]) == ("under review", False)
    one, two, three, four = asked["needles"]
    assert (one["state"], one["passage"], one["from"], one["length"]) == ("passage", "555-0199", first - 300, len(long))
    assert one["before"] == "a" * 295 + "call " and one["after"] == " or 555-0123 now" + "z" * 284
    assert (two["passage"], three["passage"], three["field"], three["kind"]) == ("555-0123", "555-0100", field,
                                                                                "conversation")
    assert three["before"] == "my number is " and three["after"] == "."
    assert four["passage"] == ("all of it, " * 200)[:600] and four["length"] == 2200

    depot.decide_erasure(request, "admitted", me)
    done = depot.erasure_shown(request)
    assert (done["status"], done["carried_out"]) == ("admitted", True)
    one, two, three, four = done["needles"]
    assert (one["state"], one["passage"], one["erased"]) == ("erased", None, 8)
    assert one["before"] == "a" * 295 + "call " and one["after"] == " or  now" + "z" * 292
    assert two["before"].endswith("call  or ") and two["after"] == " now" + "z" * 296
    assert (three["before"], three["after"], three["erased"]) == ("my number is ", ".", 8)
    assert (four["state"], four["erased"], four["title"]) == ("erased", 2200, None)
    with pytest.raises(NotFound):
        depot.erasure_shown("depot-99")


def test_a_needle_that_leaves_an_item_breaking_its_kind_is_refused_at_the_request(make_depot, me):
    depot = make_depot()
    probe = {"question": "abc", "patterns": ["x"]}
    [item] = depot.add([NewText(kind="probe", content=json.dumps(probe), licenses=("CC0-1.0",))], me)

    with pytest.raises(BadContent):  # the question may not be empty
        depot.ask_erasure("all of it", [(item, 0, 3, "/question")], me)

    assert depot.erasures() == []


def test_an_operation_on_named_texts_takes_exactly_those_and_refuses_an_id_that_is_no_text(make_depot, me):
    # what a person reviewed is what is decided: a text added after the review is not, and an id that is not
    # a text writes nothing
    depot = make_depot()
    ids = depot.add([plain(f"text {n}", "a") for n in range(1, 12)], me)
    reviewed = [ids[9], ids[1]]
    depot.add([plain("added after the review", "a")], me)

    decision, decided, _ = depot.decide("admitted", reviewed, me)

    assert decided == 2 and depot.decision(decision)["texts"] == 2
    assert depot.search("true", limit=3)["rows"][1]["id"] == ids[1]
    assert [one["status"] for one in depot.search(f"id = '{ids[0]}'")["rows"]] == ["under review"]
    with pytest.raises(NotFound):
        depot.decide("admitted", [ids[0], "depot-99"], me)
    assert depot.show(ids[0])["status"] == "under review"
    with pytest.raises(NotFound):
        depot.retag([ids[1], "depot-99"], ["a"], [], me)
    assert depot.retag([ids[1], ids[2]], ["b"], [], me)[1] == 2
    assert column(depot.search("EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'b')")["rows"], "id") == [
        ids[1], ids[2]]


def test_pages_of_texts_follow_one_another_in_the_numeric_order_of_ids_forward_and_back(make_depot, me):
    # a page starts after the last row of the one before it, so depot-10 comes after depot-9, not after depot-1
    depot = make_depot()
    ids = depot.add([plain(f"text {n}") for n in range(1, 13)], me)
    pages, after = [], None
    while True:
        page = depot.search("true", after=after, limit=5)
        pages.append([one["id"] for one in page["rows"]])
        if page["next"] is None:
            break
        after = page["next"]

    assert pages == [ids[:5], ids[5:10], ids[10:]]
    back = depot.search("true", before=depot.search("true", after=after, limit=5)["previous"], limit=5)
    assert [one["id"] for one in back["rows"]] == ids[5:10]
    assert [one["id"] for one in depot.search("true", before=back["previous"], limit=5)["rows"]] == ids[:5]
    shown = depot.passages("true", limit=7, characters=1000)
    assert [one["id"] for one in shown["rows"] + depot.passages("true", after=shown["next"])["rows"]] == ids

    # the last page read again after its texts left the filter (decided from it) is empty, with a way back to the
    # rows before the place it starts after
    left = f"id NOT IN ({', '.join(repr(one) for one in ids[10:])})"
    emptied = depot.search(left, after=after, limit=5)
    assert (emptied["rows"], emptied["next"]) == ([], None)
    assert [one["id"] for one in depot.search(left, before=emptied["previous"], limit=5)["rows"]] == ids[4:9]


def test_a_page_at_a_text_starts_with_that_text_in_any_order_and_direction(make_depot, me):
    # the workspace opens a text the playlist's page does not hold on the page that starts with it
    depot = make_depot()
    ids = depot.add([plain("x" * n) for n in range(1, 13)], me)  # depot-10 sorts after depot-9, the lengths by id
    page = depot.search("true", at=ids[9], limit=3)
    assert [one["id"] for one in page["rows"]] == ids[9:12] and page["next"] is None
    back = depot.search("true", before=page["previous"], limit=3)
    assert [one["id"] for one in back["rows"]] == ids[6:9]
    assert [one["id"] for one in depot.search("true", at=ids[9], limit=3, descending=True)["rows"]] == [
        ids[9], ids[8], ids[7]]
    assert [one["id"] for one in depot.search("true", at=ids[4], limit=2, order="length")["rows"]] == ids[4:6]

    # a text the condition leaves out: the page starts where it would stand; an id the depot lacks: the first page
    assert [one["id"] for one in depot.search(f"id <> '{ids[4]}'", at=ids[4], limit=2)["rows"]] == ids[5:7]
    assert [one["id"] for one in depot.search("true", at="nowhere-1", limit=2)["rows"]] == ids[:2]
    with pytest.raises(Refused):
        depot.search("true", after=page["previous"], at=ids[1])


def test_several_erasure_requests_are_decided_together_only_when_each_can_be(make_depot, me):
    depot = make_depot()
    first, second = depot.add([plain("call 555-0100 now"), plain("my name is Ann")], me)
    one = depot.ask_erasure("a number", [(first, 5, 13)], me)
    other = depot.ask_erasure("a name", [(second, 11, 14)], me)

    with pytest.raises(NotFound):
        depot.decide_erasures([one, "depot-9"], "admitted", me)
    assert [row["status"] for row in depot.erasures()] == ["under review", "under review"]

    assert depot.decide_erasures([one, other], "admitted", me) == {one: True, other: True}
    assert (depot.show(first)["text"]["text"], depot.show(second)["text"]["text"]) == ("call  now", "my name is ")
    with pytest.raises(Refused):
        depot.decide_erasures([other], "refused", me)


def statuses_everywhere(depot, text) -> set[str]:
    """The text's status as its page, a search's row and a search's filter by status say it: one value when
    the policy and its SQL form agree."""
    from wpt_depot.filters import Filter, condition
    by_filter = [status for status in ("under review", "admitted", "refused")
                 if depot.search(condition(Filter(ids=(text,), statuses=(status,))))["rows"]]
    return {depot.show(text)["status"], depot.search(f"id = '{text}'")["rows"][0]["status"], *by_filter}


def test_a_refused_text_whose_licenses_change_starts_over_under_review_and_can_then_be_admitted(make_depot, me):
    # a text refused for its license, once given an allowed one, must be reviewable again: the refusal made
    # before the change no longer counts
    depot = make_depot()
    depot.add_license("CC-BY-NC-4.0", "Creative Commons Attribution Non Commercial 4.0", "text")
    depot.add_license("MIT", "MIT License", "text")
    [text] = depot.add([NewText(text="one", licenses=("CC-BY-NC-4.0",))], me)
    depot.decide("refused", [text], me, reason="non-commercial")

    [(decision, changed)], skipped = depot.relicense([text], ["MIT"], me)

    assert changed == 1 and skipped == 0
    shown = depot.show(text)
    assert shown["text"]["licenses"] == ["MIT"]
    assert [(one["status"], one["reason"], one["counts"]) for one in shown["decisions"]] == [
        ("under review", None, False), ("refused", "non-commercial", False),
        ("under review", "licenses changed from CC-BY-NC-4.0 to MIT", True)]
    assert shown["decisions"][-1]["id"] == decision and shown["decisions"][-1]["name"] == me.name
    assert statuses_everywhere(depot, text) == {"under review"}
    depot.decide("admitted", [text], me)
    assert statuses_everywhere(depot, text) == {"admitted"}


def test_an_admitted_text_whose_licenses_change_is_no_longer_admitted(make_depot, me):
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    changed, kept = depot.add([plain("one"), plain("two")], me)
    depot.decide("admitted", [changed, kept], me)

    depot.relicense([changed], ["MIT", "CC0-1.0"], me)

    assert statuses_everywhere(depot, changed) == {"under review"}
    assert statuses_everywhere(depot, kept) == {"admitted"}
    assert depot.count("true")["statuses"] == {"under review": 1, "admitted": 1, "refused": 0}


def test_texts_with_different_old_licenses_each_get_the_reason_true_for_them(make_depot, me):
    depot = make_depot()
    for license_id in ("CC-BY-NC-4.0", "MIT", "GPL-2.0-only"):
        depot.add_license(license_id, license_id, "text")
    first, second, third = depot.add([NewText(text="one", licenses=("CC-BY-NC-4.0",)),
                                      NewText(text="two", licenses=("CC-BY-NC-4.0", "GPL-2.0-only")),
                                      NewText(text="three", licenses=("CC-BY-NC-4.0",))], me)

    found, _ = depot.relicense("true", ["MIT"], me)

    assert [texts for _, texts in found] == [2, 1]
    assert [depot.show(text)["decisions"][-1]["reason"] for text in (first, second, third)] == [
        "licenses changed from CC-BY-NC-4.0 to MIT", "licenses changed from CC-BY-NC-4.0, GPL-2.0-only to MIT",
        "licenses changed from CC-BY-NC-4.0 to MIT"]


def test_a_license_change_that_breaks_a_rule_writes_nothing(make_depot, me):
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    held, free = depot.add([plain("held"), plain("free")], me)
    depot.decide("admitted", [held], me)
    before = depot.change(), depot.sql("SELECT count(*) AS n FROM decisions")[0]["n"]

    with pytest.raises(Refused):
        depot.relicense([free], ["MIT", "CC-BY-NC-4.0"], me)  # a license the depot does not hold
    with pytest.raises(NotFound):
        depot.relicense([free, "depot-99"], ["MIT"], me)
    with pytest.raises(Refused):
        depot.relicense([free], [], me)
    with pytest.raises(Refused):
        depot.relicense([free], ["MIT"], Declaration("Ada Lovelace", "ada@example.org", False))

    assert (depot.change(), depot.sql("SELECT count(*) AS n FROM decisions")[0]["n"]) == before
    assert depot.show(held)["text"]["licenses"] == ["CC0-1.0"]


def test_a_text_that_names_the_new_licenses_already_is_left_out_and_reported(make_depot, me):
    # a change wipes the decisions before it, so a change to the same licenses would only clear a refusal
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    same, other = depot.add([NewText(text="one", licenses=("MIT", "CC0-1.0")), plain("two")], me)
    depot.decide("refused", [same, other], me)
    before = depot.change()

    assert depot.relicense([same], ["CC0-1.0", "MIT"], me) == ([], 1)
    assert depot.change() == before and depot.show(same)["status"] == "refused"

    [(decision, changed)], skipped = depot.relicense([other, same], ["MIT", "CC0-1.0"], me)

    assert (changed, skipped) == (1, 1)
    assert depot.show(same)["status"] == "refused" and depot.show(other)["status"] == "under review"


def test_only_the_depot_writes_a_reason_in_the_form_of_a_license_change(make_depot, me):
    # the policy recognises a license change by its reason, so a person writing one could wipe a text's decisions
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    request = depot.ask_erasure("private", [(text, 0, 1)], me)

    with pytest.raises(Refused):
        depot.decide("refused", [text], me, reason=" licenses changed from CC0-1.0 to MIT")
    with pytest.raises(Refused):
        depot.decide_erasure(request, "refused", me, reason="licenses changed from CC0-1.0 to MIT")

    assert [one["status"] for one in depot.show(text)["decisions"]] == ["under review"]
    depot.decide("refused", [text], me, reason="the licenses changed from what the source said")


def test_a_texts_credits_change_in_place_as_a_decision_under_review_and_it_stays_admitted(make_depot, me):
    depot = make_depot()
    [text] = depot.add([NewText(text="The fox.", licenses=("CC0-1.0",), title="Fox", tags=("x", "y"),
                                notice="(c) Ann")], me)
    depot.decide("admitted", "true", me)

    decision, reason = depot.edit_credits(text, {"title": "The fox", "tags": ["x", "z"], "authors": ["Ann"],
                                                 "notice": "(c) Ann Lee"}, me)

    shown = depot.show(text)
    assert reason == ("title changed from Fox to The fox; authors changed from none to Ann; tags changed from x, y to "
                      "x, z; notice changed from depot-1 to depot-2")
    assert (shown["text"]["title"], shown["text"]["tags"], shown["text"]["authors"]) == ("The fox", ["x", "z"], ["Ann"])
    assert shown["notice"] == "(c) Ann Lee"
    assert shown["status"] == "admitted"
    assert [(one["id"], one["status"], one["reason"], one["counts"]) for one in shown["decisions"]][-1] == (
        decision, "under review", reason, True)
    assert all(one["counts"] for one in shown["decisions"])
    # the same notice text is stored once
    depot.edit_credits(text, {"notice": "(c) Ann", "title": None}, me)
    assert column(depot.sql("SELECT count(*) AS n FROM notices"), "n") == [2]
    assert depot.show(text)["text"]["title"] is None


def test_a_credits_change_that_changes_nothing_writes_nothing(make_depot, me):
    depot = make_depot()
    [text] = depot.add([NewText(text="The fox.", licenses=("CC0-1.0",), title="Fox")], me)
    with pytest.raises(Refused):
        depot.edit_credits(text, {"title": "Fox"}, me)
    with pytest.raises(Refused):
        depot.edit_credits(text, {"licenses": ["MIT"]}, me)
    with pytest.raises(NotFound):
        depot.edit_credits("depot-99", {"title": "x"}, me)

    assert depot.show(text)["revisions"] == 1
    assert column(depot.sql("SELECT count(*) AS n FROM decisions"), "n") == [1]


def score(name, value, version=1):
    return {"name": name, "value": value, "content_version": version}


def scores_of(depot, changes, me):
    """depot.edit_scores over changes given as objects, as the web route gives them."""
    return depot.edit_scores(list(changes), me)


def test_one_scores_change_is_one_decision_on_many_texts_which_keep_their_status(make_depot, me):
    depot = make_depot()
    first, second, third = depot.add([plain("one"), plain("two"), plain("three")], me)
    depot.decide("admitted", "true", me)

    decision, reason, changed, left_out = scores_of(
        depot, [{"id": second, "scores": [score("machine-made", 0.03), score("source-rank", 4)]},
                {"id": first, "scores": [score(" machine-made ", 0.5)]}], me)

    assert (reason, changed, left_out) == ("scores changed: set machine-made, source-rank", 2, 0)
    assert depot.show(second)["text"]["scores"] == [
        {"name": "machine-made", "value": 0.03, "content_version": 1},
        {"name": "source-rank", "value": 4.0, "content_version": 1}]
    assert depot.show(third)["text"]["scores"] is None
    for text in (first, second):
        shown = depot.show(text)
        assert shown["status"] == "admitted"
        assert [(one["id"], one["status"], one["reason"]) for one in shown["decisions"]][-1] == (
            decision, "under review", reason)
    assert column(depot.sql(f"SELECT count(*) AS n FROM texts WHERE EXISTS (SELECT 1 FROM json_each(decisions) WHERE value = '{decision}')"), "n") \
        == [2]
    assert column(depot.sql("SELECT scores FROM texts_revisions WHERE revision = 1 ORDER BY id"), "scores") == [
        None, None, None]  # the revisions before stay as they were
    assert depot.sql("SELECT id, (SELECT json_group_array(s.value ->> 'name') FROM json_each(scores) s) AS names "
                     "FROM texts WHERE scores IS NOT NULL ORDER BY id") == [
        {"id": first, "names": '["machine-made"]'}, {"id": second, "names": '["machine-made","source-rank"]'}]


def test_the_values_of_a_score_are_those_of_the_current_texts_changed_last_most_frequent_first_kept(
        make_depot, me, monkeypatch):
    depot = make_depot()
    ids = depot.add([plain(f"text {n}") for n in range(6)], me)
    assert depot.score_values("quality") == {"name": "quality", "read": 6, "texts": 0, "low": None, "high": None,
                                             "common": []}
    scores_of(depot, [{"id": one, "scores": [score("quality", value)]}
                      for one, value in zip(ids, (4, 4, 2, 4, 2, 5))], me)
    scores_of(depot, [{"id": ids[5], "scores": [score("quality", 1)]}], me)  # its earlier 5 no longer counts

    found = depot.score_values(" quality ")

    assert (found["name"], found["texts"], found["low"], found["high"]) == ("quality", 6, 1.0, 4.0)
    assert found["common"] == [{"value": 1.0, "texts": 1}, {"value": 2.0, "texts": 2}, {"value": 4.0, "texts": 3}]
    # only the most frequent are given, by value, and only the revisions of the latest changes are read
    monkeypatch.setattr(depots, "SCORE_VALUES", 2)
    assert depot.score_values("quality")["common"] == [{"value": 2.0, "texts": 2}, {"value": 4.0, "texts": 3}]
    monkeypatch.setattr(depots, "SCORE_READ", 1)
    assert (depot.score_values("quality")["read"], depot.score_values("quality")["texts"]) == (1, 1)


def test_a_score_set_again_replaces_it_one_is_removed_by_name_and_an_unchanged_text_is_left_out(make_depot, me):
    depot = make_depot()
    first, second = depot.add([plain("one"), plain("two")], me)
    scores_of(depot, [{"id": first, "scores": [score("a", 1), score("b", 2)]},
                      {"id": second, "scores": [score("a", 1)]}], me)

    _, reason, changed, left_out = scores_of(
        depot, [{"id": first, "scores": [score("a", 5)], "remove": ["b", "never-set"]},
                {"id": second, "scores": [score("a", 1)]}], me)

    assert (reason, changed, left_out) == ("scores changed: set a; removed b", 1, 1)
    assert depot.show(first)["text"]["scores"] == [{"name": "a", "value": 5.0, "content_version": 1}]
    # a search gives each row its scores, so a list shows them
    assert [one["scores"] for one in depot.search("true")["rows"]] == [
        [{"name": "a", "value": 5.0, "content_version": 1}], [{"name": "a", "value": 1.0, "content_version": 1}]]
    scores_of(depot, [{"id": first, "remove": ["a"]}], me)
    assert depot.show(first)["text"]["scores"] is None
    assert depot.show(second)["revisions"] == 2


def test_scores_keep_the_content_version_they_were_made_from_across_a_content_licenses_and_credits_change(
        make_depot, me):
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    [text] = depot.add([plain("one")], me)
    scores_of(depot, [{"id": text, "scores": [score("rank", 3)]}], me)

    depot.change_content(text, "one, again", None, me)
    depot.relicense([text], ["MIT"], me)
    depot.edit_credits(text, {"title": "One"}, me)

    shown = depot.show(text)["text"]
    assert (shown["content_version"], shown["scores"]) == (2, [{"name": "rank", "value": 3.0, "content_version": 1}])
    scores_of(depot, [{"id": text, "scores": [score("rank", 4, 2)]}], me)
    assert depot.show(text)["text"]["scores"] == [{"name": "rank", "value": 4.0, "content_version": 2}]


def test_the_depot_s_counts_answer_a_filter_of_statuses_and_one_value_as_the_steps_count_it(make_depot, me):
    depot = make_depot()
    depot.add([plain("one", "en"), plain("two", "en", "fr"), plain("three"), plain("four", "fr")], me)
    depot.decide("admitted", "id <> 'depot-4'", me)
    depot.decide("refused", "id = 'depot-3'", me)
    shapes = [Filter(), Filter(tags=("en",)), Filter(tags=("fr",)), Filter(licenses=("CC0-1.0",)),
              Filter(statuses=("refused",)), Filter(tags=("fr",), statuses=("admitted", "under review")),
              Filter(tags=("none",))]

    for found in shapes:
        held, stepped = depot.held_count(found), depot.count(found)
        assert held["held"] is True and (held["texts"], held["total"], held["next"]) == (4, 4, None), found
        assert held["statuses"] == stepped["statuses"], found
    # the tags, licenses and blocked texts among the matches are known for every text or none
    whole, stepped = depot.held_count(Filter()), depot.count("true")
    assert {key: whole[key] for key in ("blocked", "tags", "licenses")} == {
        key: stepped[key] for key in ("blocked", "tags", "licenses")}
    assert (depot.held_count(Filter(tags=("en",)))["tags"], depot.held_count(Filter(tags=("none",)))["tags"]) == (
        None, [])
    # anything else is counted by steps
    for other_shape in (Filter(tags=("en", "fr")), Filter(tags=("en",), licenses=("CC0-1.0",)), Filter(search="one"),
                        Filter(ids=("depot-1",)), Filter(change=depot.change() - 1), Filter(decision="depot-1")):
        assert depot.held_count(other_shape) is None


def test_one_score_set_on_a_selection_takes_each_text_s_content_version_and_removing_it_leaves_the_others(
        make_depot, me):
    depot = make_depot()
    first, second, third = depot.add([plain("one", "en"), plain("two", "en"), plain("three")], me)
    depot.decide("admitted", "true", me)
    depot.change_content(second, "two, again", None, me)  # under review again, at content version 2
    scores_of(depot, [{"id": first, "scores": [score("rank", 1)]}], me)

    decision, changed, left_out = depot.score(Filter(tags=("en",)), "machine-made", 0.25, me)

    assert (changed, left_out) == (2, 0)
    assert depot.show(first)["text"]["scores"] == [score("machine-made", 0.25), score("rank", 1.0)]
    assert depot.show(second)["text"]["scores"] == [score("machine-made", 0.25, 2)]
    assert depot.show(third)["text"]["scores"] is None
    assert [depot.show(one)["status"] for one in (first, second)] == ["admitted", "under review"]
    assert depot.decision(decision)["decision"]["reason"] == "scores changed: set machine-made"
    assert depot.score_names() == [{"name": "machine-made", "texts": 2}, {"name": "rank", "texts": 1}]
    # the same again changes nothing and writes nothing; removing it leaves the other score
    before = depot.change()
    assert depot.score(Filter(tags=("en",)), "machine-made", 0.25, me) == (None, 0, 2)
    assert depot.change() == before
    removed, changed, left_out = depot.score([first, third], "machine-made", None, me)
    assert (changed, left_out) == (1, 1)
    assert depot.show(first)["text"]["scores"] == [score("rank", 1.0)]
    assert depot.decision(removed)["decision"]["reason"] == "scores changed: removed machine-made"
    with pytest.raises(Refused):
        depot.score("true", " ", 1, me)
    with pytest.raises(Refused):
        depot.score("true", "rank", float("nan"), me)


@pytest.mark.parametrize("change, error", [
    ([], Refused),
    ([{"id": "depot-99", "scores": [score("a", 1)]}], NotFound),
    ([{"id": "depot-1", "scores": [score("a", 1, 2)]}], Refused),  # no content version 2
    ([{"id": "depot-1", "scores": [score("a", float("nan"))]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", float("inf"))]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", "high")]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", True)]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", 1, 1.5)]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", 1, 0)]}], Refused),
    ([{"id": "depot-1", "scores": [score("", 1)]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", 1), score("a", 2)]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", 1)], "remove": ["a"]}], Refused),
    ([{"id": "depot-1", "scores": [score("a", 1)]}, {"id": "depot-1", "scores": [score("b", 1)]}], Refused),
    ([{"id": "depot-1", "scores": [{"name": "a", "value": 1}]}], Refused),
    ([{"id": "depot-1", "remvoe": ["a"]}], Refused),
    ([{"id": 1, "scores": [score("a", 1)]}], Refused),
    ([{"id": "depot-1", "remove": [""]}], Refused),
    ([{"id": "depot-1", "remove": ["a"]}], Refused),  # nothing changes
    ([{"id": "depot-1", "scores": [score("a", 1)]}, {"id": "depot-99", "scores": [score("a", 1)]}], NotFound),
])
def test_a_scores_change_that_breaks_a_rule_or_changes_nothing_writes_nothing(make_depot, me, change, error):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    assert text == "depot-1"
    before = depot.change(), depot.sql("SELECT count(*) AS n FROM decisions")

    with pytest.raises(error):
        scores_of(depot, change, me)

    assert (depot.change(), depot.sql("SELECT count(*) AS n FROM decisions")) == before
    assert depot.show(text)["revisions"] == 1


def test_scores_read_from_a_json_lines_file_and_a_file_that_is_not_json_lines_is_refused(make_depot, me, tmp_path):
    depot = make_depot()
    first, second = depot.add([plain("one"), plain("two")], me)
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"id": first, "scores": [score("a", 1)]}) + "\n\n"
                    + json.dumps({"id": second, "scores": [score("a", 2)]}) + "\n", encoding="utf-8")

    assert depot.edit_scores(path, me)[2:] == (2, 0)

    assert [depot.show(one)["text"]["scores"][0]["value"] for one in (first, second)] == [1.0, 2.0]
    path.write_text(json.dumps({"id": first, "remove": ["a"]}) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(Refused, match="not JSON lines"):
        depot.edit_scores(path, me)
    assert depot.show(first)["revisions"] == 2


def test_a_scores_change_over_many_texts_keeps_python_memory_the_same_size(
        make_depot, me, tmp_path):
    depot = make_depot()
    texts = depot.add([plain(f"text {number}") for number in range(4 * scoring.BATCH)], me)

    def scored(count: int) -> int:
        """The peak of Python's memory while scoring the first count texts."""
        path = tmp_path / f"{count}.jsonl"
        path.write_text("".join(json.dumps({"id": one, "scores": [score(f"rank-{count}", 1.5), score("other", 2)]})
                                + "\n" for one in texts[:count]), encoding="utf-8")
        tracemalloc.start()
        try:
            depot.edit_scores(path, me)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return peak

    scored(10)  # the first call prepares the statements it reuses
    small, large = scored(scoring.BATCH), scored(4 * scoring.BATCH)  # both read the lines a full batch at a time

    assert column(depot.sql("SELECT count(*) AS n FROM texts WHERE EXISTS (SELECT 1 FROM json_each(scores) "
                            f"WHERE value ->> 'name' = 'rank-{4 * scoring.BATCH}')"), "n") == [4 * scoring.BATCH]
    assert large < 2 * small  # no Python object per text


def test_a_clone_receives_the_upstream_s_scores_and_a_pull_brings_their_changes(serve, make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    first, second = upstream.add([plain("one"), plain("two")], me)
    scores_of(upstream, [{"id": first, "scores": [score("rank", 1)]}], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    assert follower.show(first)["text"]["scores"] == [{"name": "rank", "value": 1.0, "content_version": 1}]

    scores_of(upstream, [{"id": first, "remove": ["rank"]}, {"id": second, "scores": [score("rank", 2)]}], me)
    pull(follower)

    assert follower.show(first)["text"]["scores"] is None
    assert follower.show(second)["text"]["scores"] == [{"name": "rank", "value": 2.0, "content_version": 1}]
    assert follower.show(second)["decisions"][-1]["reason"] == "scores changed: set rank; removed rank"
    [mine] = follower.add([plain("mine")], me)
    assert scores_of(follower, [{"id": mine, "scores": [score("rank", 3)]}], me)[2] == 1


def test_one_credits_change_sets_each_text_s_own_credits_and_the_texts_stay_admitted(make_depot, me):
    depot = make_depot()
    first, second, third = depot.add([
        NewText(text="one", licenses=("CC0-1.0",), title="Kept", authors=("{'first': 'Ann', 'last': 'Lee'}",),
                tags=("en",), address="https://example.org/1", notice="(c) Ann"),
        plain("two", "en"), plain("three", "en")], me)
    depot.decide("admitted", "true", me)
    tags_before = depot.counts()["tag"]

    decision, reason, changed, left_out = depot.edit_credits_many(
        [{"id": first, "authors": ["Ann Lee"]},
         {"id": second, "title": "owner/repo/src/main.py", "notice": "(c) Bo"},
         {"id": third, "title": None, "authors": None}], me)

    assert (reason, changed, left_out) == ("credits changed: title, authors, notice", 2, 1)
    one, two = depot.show(first), depot.show(second)
    assert (one["text"]["title"], one["text"]["authors"], one["text"]["tags"], one["text"]["address"],
            one["notice"]) == ("Kept", ["Ann Lee"], ["en"], "https://example.org/1", "(c) Ann")
    assert (two["text"]["title"], two["text"]["authors"], two["notice"]) == ("owner/repo/src/main.py", None, "(c) Bo")
    assert depot.show(third)["revisions"] == 1
    for shown in (one, two):
        assert shown["status"] == "admitted"
        assert [(one["id"], one["status"], one["reason"]) for one in shown["decisions"]][-1] == (
            decision, "under review", reason)
        assert all(one["counts"] for one in shown["decisions"])
    assert pick(depot.sql("SELECT id, title, authors FROM texts ORDER BY id"), ["id", "title", "authors"]) == [
        {"id": first, "title": "Kept", "authors": '["Ann Lee"]'},
        {"id": second, "title": "owner/repo/src/main.py", "authors": None},
        {"id": third, "title": None, "authors": None}]
    # the revisions before stay as they were
    assert depot.sql("SELECT id, title, authors FROM texts_revisions WHERE revision = 1 ORDER BY id") == [
        {"id": first, "title": "Kept", "authors": '["{\'first\': \'Ann\', \'last\': \'Lee\'}"]'},
        {"id": second, "title": None, "authors": None},
        {"id": third, "title": None, "authors": None}]
    assert depot.counts()["decision"][decision] == 2
    assert depot.counts()["tag"] == tags_before


def test_tags_changed_in_one_credits_change_move_the_counts_of_tags(make_depot, me):
    depot = make_depot()
    first, second = depot.add([plain("one", "en"), plain("two", "en")], me)
    depot.decide("admitted", "true", me)

    revisions = depot.sql("SELECT count(*) AS n FROM texts_revisions")

    depot.edit_credits_many([{"id": first, "tags": ["fr", "fr", "book"]}, {"id": second, "tags": None}], me)

    assert depot.show(first)["text"]["tags"] == ["book", "fr"]  # in the order of their names
    assert depot.show(second)["text"]["tags"] == []
    assert depot.counts()["tag"] == {"fr": 1, "book": 1}
    # a change of tags alone writes no revision, only rows of taggings
    assert depot.sql("SELECT count(*) AS n FROM texts_revisions") == revisions
    assert [one["decisions"][-1]["reason"] for one in map(depot.show, (first, second))] == ["credits changed: tags"] * 2


@pytest.mark.parametrize("change, error", [
    ([], Refused),
    ([{"id": "depot-99", "title": "x"}], NotFound),
    ([{"id": "depot-1", "title": "one"}, {"id": "depot-99", "title": "x"}], NotFound),
    ([{"id": "depot-1", "title": "One"}], Refused),  # nothing changes
    ([{"id": "depot-1"}], Refused),  # nothing changes
    ([{"id": "depot-1", "title": "a"}, {"id": "depot-1", "title": "b"}], Refused),
    ([{"id": "depot-1", "title": ""}], Refused),
    ([{"id": "depot-1", "title": "   "}], Refused),
    ([{"id": "depot-1", "title": 3}], Refused),
    ([{"id": "depot-1", "authors": "Ann"}], Refused),
    ([{"id": "depot-1", "authors": ["Ann", ""]}], Refused),
    ([{"id": "depot-1", "licenses": ["MIT"]}], Refused),
    ([{"title": "x"}], Refused),
    ([{"id": "depot-1", "title": "x", "was": {"title": None}}], Refused),  # its was no longer holds
    ([{"id": "depot-1", "title": "x", "was": "One"}], Refused),
    ([{"id": "depot-1", "title": "x", "was": {"licenses": None}}], Refused),
    ([{"id": "depot-1", "title": "x", "was": {"title": ""}}], Refused),
])
def test_a_credits_change_that_breaks_a_rule_or_changes_nothing_writes_nothing(make_depot, me, change, error):
    depot = make_depot()
    [text] = depot.add([NewText(text="one", licenses=("CC0-1.0",), title="One")], me)
    assert text == "depot-1"
    before = depot.change(), depot.sql("SELECT count(*) AS n FROM decisions")

    with pytest.raises(error):
        depot.edit_credits_many(change, me)

    assert (depot.change(), depot.sql("SELECT count(*) AS n FROM decisions")) == before
    assert depot.show(text)["revisions"] == 1


def test_a_credits_line_changes_its_text_only_while_the_credits_its_was_names_still_hold(make_depot, me):
    depot = make_depot()
    untitled, retitled, authored, tagged = depot.add([
        plain("one"), plain("two"), NewText(text="three", licenses=("CC0-1.0",), authors=("{'first': 'Ann'}",),
                                            notice="(c) Ann"),
        plain("four", "en", "book")], me)
    depot.edit_credits(retitled, {"title": "Set by a person"}, me)  # after the lines below were written

    decision, reason, changed, left_out = depot.edit_credits_many([
        {"id": untitled, "title": "From the source", "was": {"title": None}},
        {"id": retitled, "title": "From the source", "was": {"title": None}},
        {"id": authored, "authors": None, "was": {"authors": ["{'first': 'Ann'}"], "notice": "(c) Ann"}},
        {"id": tagged, "title": "Tagged", "was": {"tags": ["book", "en"]}}], me)

    assert (reason, changed, left_out) == ("credits changed: title, authors", 3, 1)
    assert [depot.show(one)["text"]["title"] for one in (untitled, retitled, tagged)] == [
        "From the source", "Set by a person", "Tagged"]
    assert depot.show(authored)["text"]["authors"] is None


def test_credits_read_from_a_json_lines_file_and_more_lines_than_one_change_takes_are_refused(
        make_depot, me, tmp_path, monkeypatch):
    depot = make_depot()
    first, second, third = depot.add([plain("one"), plain("two"), plain("three")], me)
    path = tmp_path / "credits.jsonl"
    path.write_text(json.dumps({"id": first, "title": "One"}) + "\n\n" + json.dumps({"id": second, "title": "Two"})
                    + "\n", encoding="utf-8")
    monkeypatch.setattr(crediting, "MOST", 2)

    assert depot.edit_credits_many(path, me)[2:] == (2, 0)

    assert [depot.show(one)["text"]["title"] for one in (first, second)] == ["One", "Two"]
    path.write_text("".join(json.dumps({"id": one, "title": "x"}) + "\n" for one in (first, second, third)),
                    encoding="utf-8")
    with pytest.raises(Refused, match="at most 2 texts"):
        depot.edit_credits_many(path, me)
    path.write_text(json.dumps({"id": first, "title": "x"}) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(Refused, match="not JSON lines"):
        depot.edit_credits_many(path, me)
    assert depot.show(first)["revisions"] == 2


def test_a_clone_refuses_a_credits_change_of_its_upstream_s_texts(serve, make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    [theirs] = upstream.add([plain("one")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    [mine] = follower.add([plain("mine")], me)

    with pytest.raises(BelongsToUpstream):
        follower.edit_credits_many([{"id": mine, "title": "Mine"}, {"id": theirs, "title": "Theirs"}], me)

    assert follower.edit_credits_many([{"id": mine, "title": "Mine"}], me)[2] == 1
    assert follower.show(theirs)["text"]["title"] is None


def test_the_depots_maintainer_and_takedown_contact_change_and_its_name_does_not(make_depot):
    depot = make_depot()

    depot.edit_about(maintainer="Grace Hopper", takedown_contact="takedown@example.org")

    assert {key: value for key, value in depot.about().items() if key not in ("change", "revision")} == {
        "name": "depot", "maintainer": "Grace Hopper", "takedown_contact": "takedown@example.org", "upstream": None,
        "format": FORMAT}
    assert depot.about()["revision"] == 2
    depot.edit_about(maintainer="Ada Lovelace")
    assert depot.about()["takedown_contact"] == "takedown@example.org"
    for wrong in ({}, {"maintainer": " "}):
        with pytest.raises(Refused):
            depot.edit_about(**wrong)


@pytest.fixture
def upstream_and_clone(serve, make_depot, me, tmp_path):
    """An upstream holding an admitted text, a snapshot of it, a tokenizer
    recipe of that snapshot and an erasure request under review, and a clone
    of it."""
    upstream = make_depot("upstream")
    [text] = upstream.add([plain("upstream text 123", "a")], me)
    upstream.decide("admitted", [text], me)
    snapshot = upstream.create_snapshot("first")["id"]
    tokenizer = upstream.new_tokenizer("tok", snapshot, None, 1000, 600)
    request = upstream.ask_erasure("a number", [(text, 14, 17)], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    return upstream, follower, {"text": text, "snapshot": snapshot, "tokenizer": tokenizer, "request": request}


REFUSED_ON_UPSTREAM_ROWS = {
    "admit": lambda depot, me, ids: depot.decide("admitted", [ids["text"]], me),
    "refuse by a condition": lambda depot, me, ids: depot.decide("refused", "true", me),
    "relicense": lambda depot, me, ids: depot.relicense([ids["text"]], ["MIT"], me),
    "edit credits": lambda depot, me, ids: depot.edit_credits(ids["text"], {"title": "Mine"}, me),
    "change content": lambda depot, me, ids: depot.change_content(ids["text"], "my text", None, me),
    "score": lambda depot, me, ids: depot.edit_scores(list(
        [{"id": ids["mine"], "scores": [{"name": "rank", "value": 1, "content_version": 1}]},
         {"id": ids["text"], "scores": [{"name": "rank", "value": 2, "content_version": 1}]}]), me),
    "ask to erase": lambda depot, me, ids: depot.ask_erasure("why", [(ids["text"], 0, 8)], me),
    "decide an erasure": lambda depot, me, ids: depot.decide_erasure(ids["request"], "admitted", me),
    "retag": lambda depot, me, ids: depot.retag([ids["mine"], ids["text"]], ["b"], [], me),
    "edit a tokenizer recipe": lambda depot, me, ids: depot.edit_tokenizer(ids["tokenizer"], ids["snapshot"], None,
                                                                           2000, 600),
}


@pytest.mark.parametrize("operation", REFUSED_ON_UPSTREAM_ROWS)
def test_a_clone_refuses_every_change_to_its_upstream_s_rows_and_writes_nothing(upstream_and_clone, me, operation):
    upstream, follower, ids = upstream_and_clone
    follower.add_license("MIT", "MIT License", "The MIT text.")
    [ids["mine"]] = follower.add([plain("my own text")], me)
    follower.decide("admitted", [ids["mine"]], me)
    before = follower.change(), every_row(follower)

    with pytest.raises(BelongsToUpstream, match="belong.* to the upstream"):
        REFUSED_ON_UPSTREAM_ROWS[operation](follower, me, ids)

    assert (follower.change(), every_row(follower)) == before


def test_a_decision_on_texts_of_both_depots_is_refused_whole_and_names_the_upstream_s(upstream_and_clone, me):
    upstream, follower, ids = upstream_and_clone
    [mine] = follower.add([plain("my own text")], me)

    with pytest.raises(BelongsToUpstream, match=ids["text"]):
        follower.decide("admitted", [mine, ids["text"]], me)

    assert follower.show(mine)["status"] == "under review"
    assert follower.blocked([mine, ids["text"]], "refused")["rows"] == [
        {"id": ids["text"], "reasons": [BELONGS_TO_UPSTREAM]}]
    assert follower.show(ids["text"])["upstream"] and not follower.show(mine)["upstream"]
    assert upstream.blocked([ids["text"]])["rows"] == [] and not upstream.show(ids["text"])["upstream"]


def test_a_clone_s_own_rows_stay_changeable_beside_the_upstream_s(upstream_and_clone, me):
    upstream, follower, ids = upstream_and_clone
    [mine] = follower.add([plain("my own text")], me)
    follower.edit_credits(mine, {"title": "Mine"}, me)
    follower.decide("admitted", [mine], me)
    follower.retag([mine], ["mine"], [], me)
    upstream.edit_credits(ids["text"], {"title": "Newer"}, me)
    pull(follower)

    assert pick(follower.search("true")["rows"], ["id", "title", "tags", "status"]) == [
        {"id": mine, "title": "Mine", "tags": ["mine"], "status": "admitted"},
        {"id": ids["text"], "title": "Newer", "tags": ["a"], "status": "admitted"}]


def test_a_pull_after_the_clone_s_own_work_keeps_both_depots_rows(upstream_and_clone, me):
    upstream, follower, ids = upstream_and_clone
    [mine] = follower.add([plain("my own text")], me)
    follower.decide("admitted", [mine], me)
    own_request = follower.ask_erasure("my own", [(mine, 0, 3)], me)
    [more] = upstream.add([plain("upstream text later")], me)
    upstream.decide("admitted", [more], me)
    upstream.decide_erasure(ids["request"], "admitted", me)

    assert pull(follower) == [ids["request"]]

    assert follower.sql("SELECT id, text FROM texts ORDER BY id") == [
        {"id": mine, "text": "my own text"}, {"id": ids["text"], "text": "upstream text "},
        {"id": more, "text": "upstream text later"}]
    assert follower.decide_erasure(own_request, "admitted", me)  # its own request stays its to decide
    assert column(follower.sql(f"SELECT text FROM texts WHERE id = '{mine}'"), "text") == ["own text"]


def test_a_clone_whose_name_begins_its_upstream_s_name_owns_none_of_the_upstream_s_rows(serve, make_depot, me, tmp_path):
    upstream = make_depot("wpt-eu")
    [text] = upstream.add([plain("upstream text")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("wpt", "Grace Hopper", "grace@example.org"))
    [mine] = follower.add([plain("my own text")], me)
    before = follower.change(), every_row(follower)

    with pytest.raises(BelongsToUpstream, match=text):
        follower.decide("refused", [text], me)

    assert (follower.change(), every_row(follower)) == before
    assert follower.upstream_owned([text, mine]) == [text]
    assert follower.show(text)["upstream"] and follower.show(text)["status"] == "under review"
    assert column(follower.sql(f"SELECT id FROM texts WHERE {follower.upstream_sql()}"), "id") == [text]
    follower.decide("admitted", [mine], me)
    assert follower.show(mine)["status"] == "admitted"


def test_a_clone_s_own_erasure_request_naming_an_upstream_text_is_never_carried_out(upstream_and_clone, me):
    upstream, follower, ids = upstream_and_clone
    # such a request can no longer be asked; recorded as asking used to record it
    with follower.db.write("ask erasure") as (con, number):
        decision = follower._decision(con, number, "under review", me)
        row = depots._request("theirs", [Needle(ids["text"], 0, 8, None, 1)],
                              follower._id(follower._take(con, "erasures")), [decision])
        depots._insert(con, "erasures_revisions", [depots._erasure_row(row, number)])
    before = follower.change(), every_row(follower)

    with pytest.raises(BelongsToUpstream, match=ids["text"]):
        follower.decide_erasures([row["id"]], "admitted", me)

    assert (follower.change(), every_row(follower)) == before
    assert column(follower.sql(f"SELECT text FROM texts WHERE id = '{ids['text']}'"), "text") == [
        "upstream text 123"]


def contents_of(depot, text) -> list[str]:
    """Every content version of a plain text, in order."""
    return column(depot.sql(f"SELECT text FROM contents WHERE id = '{text}' ORDER BY version"), "text")


def snapshot_texts(depot, snapshot) -> list[dict]:
    """The texts of a snapshot with the text of the content version each reads."""
    return texts_of(depot, depot.snapshot_texts(snapshot)["texts"])


def test_a_content_change_starts_the_text_over_and_a_snapshot_keeps_reading_the_old_content(make_depot, me):
    depot = make_depot()
    changed, kept = depot.add([plain("The fox jumps."), plain("The dog sleeps.")], me)
    depot.decide("admitted", "true", me)
    old = depot.create_snapshot("26.10")["id"]

    decision, version = depot.change_content(changed, "The fox jumps high.", None, me)

    shown = depot.show(changed)
    assert (version, shown["text"]["content_version"], shown["content_versions"]) == (2, 2, 2)
    assert shown["status"] == "under review"
    assert shown["decisions"][-1]["id"] == decision
    assert shown["decisions"][-1]["reason"] == "content changed from version 1 to version 2"
    assert [one["counts"] for one in shown["decisions"]] == [False, False, True]
    assert column(depot.snapshot_texts(depot.create_snapshot("26.11")["id"])["texts"], "id") == [kept]
    depot.decide("admitted", [changed], me)
    assert snapshot_texts(depot, depot.create_snapshot("26.12")["id"]) == [
        {"id": changed, "text": "The fox jumps high."}, {"id": kept, "text": "The dog sleeps."}]
    assert snapshot_texts(depot, old) == [
        {"id": changed, "text": "The fox jumps."}, {"id": kept, "text": "The dog sleeps."}]
    assert contents_of(depot, changed) == ["The fox jumps.", "The fox jumps high."]
    assert column(depot.sql(f"SELECT text FROM texts WHERE id = '{changed}'"), "text") == ["The fox jumps high."]


def test_an_erasure_after_a_content_change_cuts_the_passage_from_every_version_and_stays_exact_for_the_one_named(
        make_depot, me):
    depot = make_depot()
    secret = "555-0199"
    [text] = depot.add([plain(f"call {secret} now")], me)
    depot.decide("admitted", [text], me)
    snapshot = depot.create_snapshot("26.10")["id"]
    depot.change_content(text, f"Please, call {secret} or {secret} today.", None, me)
    request = depot.ask_erasure("a number", [(text, 5, 13, None, 1)], me)  # the place in the first version
    assert depot.erasure_shown(request)["needles"][0]["passage"] == secret

    depot.decide_erasure(request, "admitted", me)

    assert contents_of(depot, text) == ["call  now", "Please, call  or  today."]
    assert [one["text"] for one in snapshot_texts(depot, snapshot)] == ["call  now"]
    [erased] = depot.erasures()[0]["erased"]
    assert (erased["length"], erased["at"]) == (len(secret), 5)
    [needle] = depot.erasure_shown(request)["needles"]
    assert (needle["version"], needle["state"], needle["before"], needle["after"]) == (1, "erased", "call ", " now")
    assert files_holding(depot.address, secret) == []


def test_a_content_change_that_brings_back_a_passage_erased_from_that_text_is_refused(make_depot, me):
    depot = make_depot()
    secret = "TOKEN = 'abc123'"
    [text, other] = depot.add([plain(f"a = 1\n{secret}\nb = 2\n"), plain("x = 1\n")], me)
    depot.decide("admitted", [text, other], me)
    request = depot.ask_erasure("a secret", [(text, 6, 6 + len(secret))], me)
    depot.decide_erasure(request, "admitted", me)
    change = depot.change()

    with pytest.raises(Refused, match="an erasure removed"):
        depot.change_content(text, f"a = 1\nb = 2\n{secret}\n", None, me)

    assert depot.change() == change
    assert contents_of(depot, text) == ["a = 1\n\nb = 2\n"]
    depot.change_content(text, "a = 1\nb = 3\n", None, me)  # other new content passes
    depot.change_content(other, f"{secret}\n", None, me)  # the check covers the text it was erased from


def test_a_needle_names_the_current_content_version_when_it_names_none(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one two")], me)
    depot.change_content(text, "three four", None, me)

    request = depot.ask_erasure("a word", [(text, 0, 5)], me)

    assert depot.erasures()[0]["needles"][0]["version"] == 2
    assert depot.erasure_shown(request)["needles"][0]["passage"] == "three"
    with pytest.raises(Refused):
        depot.ask_erasure("a word", [(text, 0, 3, None, 3)], me)


def test_a_content_that_changes_nothing_writes_nothing_and_an_item_still_matches_its_kind(make_depot, me):
    depot = make_depot()
    probe = {"question": "What is your name?", "patterns": ["(?i)password"]}
    text, item = depot.add([plain("same"), NewText(kind="probe", content=json.dumps(probe), licenses=("CC0-1.0",))], me)
    before = depot.change()

    assert depot.change_content(text, "same", None, me) is None
    assert depot.change_content(item, None, json.dumps({"patterns": ["(?i)password"], "question": "What is your name?"},
                                                       indent=2), me) is None
    with pytest.raises(BadContent):
        depot.change_content(item, None, json.dumps({"question": "Name?"}), me)
    for wrong in [(text, None, '{"x": 1}'), (text, None, None), (item, "words", None), (item, None, None)]:
        with pytest.raises(Refused):
            depot.change_content(*wrong, me)

    assert depot.change() == before
    decision, version = depot.change_content(item, None, json.dumps({**probe, "question": "Your name?"}), me)
    assert json.loads(depot.show(item)["text"]["content"])["question"] == "Your name?" and version == 2


def test_only_the_depot_writes_a_reason_in_the_form_of_a_content_change(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)

    with pytest.raises(Refused):
        depot.decide("refused", [text], me, reason="content changed from version 1 to version 2")

    assert [one["status"] for one in depot.show(text)["decisions"]] == ["under review"]


def test_rows_an_upstream_sends_with_a_column_the_table_lacks_are_refused_and_never_run_as_sql(make_depot, tmp_path):
    # a column's name comes from the upstream's answer: it must never become a statement of the follower
    depot = make_depot()
    planted = tmp_path / "planted.sqlite"
    strange = (f"text) SELECT 'MIT', 'x', 'y'; DELETE FROM changes; ATTACH '{planted}' AS p; "
               "CREATE TABLE p.t (a); --")
    before = every_row(depot)

    with pytest.raises(UpstreamUnreachable, match="columns"):
        with depot.db.write("pull") as (con, _):
            follow._put(con, "licenses", [{"id": "MIT", "name": "n", strange: "z"}])

    assert every_row(depot) == before and not planted.exists()


def counted_again(depot, title_index) -> tuple[dict, dict]:
    """The depot's counts, its listings of texts by tag and by license, the title index (title_index, the fixture),
    and what the texts of each status hold by kind, tag set, tag and license as every write left them, and as a
    recount from its rows makes them: equal when each write kept them right."""
    def stored():
        split = {facet: {one["name"]: one["texts"] for one in depot.rows(
            "SELECT name, texts FROM counts WHERE facet = ? ORDER BY name", (facet,))} for facet in counts.SPLIT}
        sized = depot.rows("SELECT facet, name, texts, characters, bytes FROM counts WHERE facet IN "
                           f"{counts.TEXT_SIZED} ORDER BY 1, 2")
        return {**depot.counts(), **split, **{table: depot.rows(f"SELECT * FROM {table} ORDER BY 1, 2")
                                              for table, _ in counts.LISTINGS.values()},
                "titles": title_index(depot), "sized": sized}
    kept = stored()
    depot.recount()
    return kept, stored()


def stored_and_ruled(depot) -> tuple[dict, dict]:
    """Each current text's status as the depot stores it, and as the policy's rule makes it of the text's
    decisions in order: equal when each write kept the stored status right."""
    decided = {one["id"]: one for one in depot.rows("SELECT id, status, reason FROM decisions")}
    found = depot.rows("SELECT id, status, decisions FROM texts")
    return ({one["id"]: one["status"] for one in found},
            {one["id"]: policy.status([decided[d] for d in one["decisions"]]) for one in found})


def test_every_write_keeps_the_counts_and_statuses_equal_to_what_the_rows_make(make_depot, me, title_index):
    # the tags, licenses and overview pages read only the counts, a plain search reads the title index,
    # and every filter by status reads the stored status, so each write must keep them right
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    texts = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",), tags=("a",) if n % 2 else ("b",),
                               title=f"Title {n}" if n % 3 else None) for n in range(8)], me)
    writes = [
        lambda: depot.decide("admitted", "true", me),
        lambda: depot.decide("refused", texts[:2], me),
        lambda: depot.decide("under review", texts[:3], me),  # two refused texts and an admitted one
        lambda: depot.relicense(texts[3:5], ["MIT"], me),
        lambda: depot.change_content(texts[5], "text 5, changed", None, me),
        lambda: depot.edit_credits(texts[6], {"tags": ["a", "c"]}, me),
        lambda: depot.edit_credits(texts[6], {"title": "Renamed"}, me),
        lambda: depot.edit_scores([{"id": texts[7], "scores": [score("rank", 1)]}], me),
        lambda: depot.score(Filter(ids=tuple(texts[5:8])), "length", 2.0, me),
        lambda: depot.edit_credits(texts[3], {"tags": ["d"]}, me),
        lambda: depot.relicense([texts[4]], ["CC0-1.0", "MIT"], me),
        lambda: depot.change_content(texts[5], "text 5, changed again", None, me),
        lambda: depot.edit_scores([{"id": texts[5], "remove": ["length"]}], me),
        # many texts' content at once: an admitted text, a text under review named at its content version, one at
        # another content version, and one whose content is that already
        lambda: depot.change_content_many([{"id": texts[6], "text": "text 6, from its source"},
                                           {"id": texts[0], "text": "text 0, from its source", "was": 1},
                                           {"id": texts[5], "text": "text 5, from its source"},
                                           {"id": texts[1], "text": "text 1"}], me),
        # many texts' credits at once: tags alone, titles alone (one given, one cleared) and authors alone
        lambda: depot.edit_credits_many([{"id": texts[2], "tags": ["e"]}, {"id": texts[4], "tags": ["a", "e"]},
                                         {"id": texts[5], "tags": []}], me),
        lambda: depot.edit_credits_many([{"id": texts[0], "title": "First"}, {"id": texts[1], "title": None},
                                         {"id": texts[3], "title": "Title 3, longer"}], me),
        lambda: depot.edit_credits_many([{"id": texts[0], "authors": ["Ann Lee"]},
                                         {"id": texts[2], "authors": ["Bo"]}], me),
        lambda: depot.decide_erasure(depot.ask_erasure("all of it", [(texts[7], None, None)], me), "admitted", me),
    ]
    for write in writes:
        write()
        assert sizes_apart(depot) == [] and tags_apart(depot) == []  # before the recount, which copies them again
        kept, made = counted_again(depot, title_index)
        assert kept == made
        stored, ruled = stored_and_ruled(depot)
        assert stored == ruled
    assert made["texts"] == {"": 7} and made["tag"] == {"a": 3, "b": 1, "c": 1, "d": 1, "e": 2}
    assert made["score"] == {"length": 1}
    assert made["license"] == {"CC0-1.0": 6, "MIT": 2} and len(made["text_licenses"]) == 8
    assert made["titles"]["terms"] and made["titles"]["unfound"] == []
    assert [one["id"] for one in depot.search(Filter(search="title"))["rows"]] == texts[2:6]


def every_match(depot, where) -> list[str]:
    """The ids a selection matches, page after page in the order of ids, each page read as a person's page is."""
    found, after = [], None
    while True:
        page = depot.search(where, after=after, limit=7)
        found += [one["id"] for one in page["rows"]]
        if page["next"] is None:
            return found
        after = page["next"]


TITLED = ["Hello World", "the DRIVERS of rivers", "Rivers and hello", "École normale", "ölmühle", "50% off_line",
          "it's a \"quoted\" name", "ab", "a riverside walk", "Hello again, world", "hell", "x-12 notes"]


@pytest.mark.parametrize("words, whole", [
    ("hello", False), ("rivers", False), ("rivers", True), ("RIVER hello", False), ("école", False),
    ("ÖLM", False), ("% off_", False), ('"quoted"', False), ("it's", False), ("ab", False), ("ab hello", False),
    ("depot-3", False), ("depot-3 rivers", False), ("depot-30 depot-3", False), ("x-12", False), ("zzz", False),
    ("hel wor", True)])
def test_a_plain_search_finds_through_the_title_index_what_a_reading_of_every_text_finds(make_depot, me, words, whole):
    # a search reads its candidates from the title index; reading every text with the same condition is the measure
    depot = make_depot()
    texts = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",),
                               title=TITLED[n % len(TITLED)] if n % 5 else None) for n in range(40)], me)
    depot.edit_credits_many([{"id": texts[n], "title": f"renamed {TITLED[n % len(TITLED)]}"} for n in range(0, 40, 3)],
                            me)
    found = Filter(search=words, whole_words=whole)

    assert every_match(depot, found) == every_match(depot, depot.condition(found))
    counted = depot.count(found)
    assert sum(counted["statuses"].values()) == len(every_match(depot, found))
    assert counted["total"] == counted["texts"] and counted["next"] is None
    backward = depot.search(found, before=["zzz"], limit=100)["rows"]
    assert [one["id"] for one in reversed(backward)] == list(reversed(every_match(depot, found)))


def test_a_plain_search_of_a_word_of_three_characters_reads_the_title_index_and_never_every_text(make_depot, me):
    depot = make_depot()
    depot.add([NewText(text="one", licenses=("CC0-1.0",), title="Hello World")] + [plain(f"{n}") for n in range(5)], me)
    con = depot.db.reader()

    plan = depot._plan(Filter(search="hello wo depot-1"))
    assert plan.listing.parts and not plan.listing.scans
    for compare, direction in ((None, ""), (">", ""), ("<=", " DESC")):
        window, uses = depots._window(plan.listing, compare, direction, 10)
        steps = [one[3] for one in con.execute(f"EXPLAIN QUERY PLAN {window}", ("x",) * uses)]
        # the words matched and the place's bounds both read by the index (M, then < and >), at most 10 rows of it
        # sorted afterwards; the text named by its id looked up by its key
        assert any(re.fullmatch(r"SCAN text_titles VIRTUAL TABLE INDEX \d+:M\d*[<>]{2}", one) for one in steps), steps
        assert "SEARCH texts_revisions USING INDEX texts_latest (id=?)" in steps
        assert not any(one.startswith("SCAN texts_revisions") for one in steps), steps
    assert depot._plan(Filter(search="he wo")).listing.scans  # no trigram: the bounded reading of every text
    assert depot._plan(Filter(search="hello", in_text=True)).listing.scans


def test_a_search_in_a_smaller_listing_counts_the_title_index_only_as_far_as_that_listing(make_depot, me):
    # the index counted up to the tag's size + 1: past it the tag's texts are read, below it the index
    depot = make_depot()
    depot.add([NewText(text=f"{n}", licenses=("CC0-1.0",), title="River walk" if n < 8 else "Lake",
                       tags=("few",) if n >= 5 else ()) for n in range(10)], me)

    for words, read_from in (("river", "text_tags"), ("lake", "text_titles")):
        found = Filter(search=words, tags=("few",))
        plan = depot._plan(found)
        assert read_from in plan.listing.rows
        assert every_match(depot, found) == every_match(depot, depot.condition(found))
    assert depot._plan(Filter(search="lake", tags=("few",))).listing.texts == 2
    assert depot._titled(["river"], 5).texts == 6 and depot._titled(["river"], 50).texts == 8


def test_a_count_of_a_search_names_no_total_when_the_title_index_gives_too_many_to_count_at_once(make_depot, me,
                                                                                                 monkeypatch):
    # counting every text a common word gives would take as long as the steps, past a reader's time limit
    depot = make_depot()
    depot.add([NewText(text=f"{n}", licenses=("CC0-1.0",), title=f"River {n}" if n < 5 else "Lake" if n == 5 else None)
               for n in range(8)], me)
    monkeypatch.setattr(depots, "COUNT_STEP", 3)

    first = depot.count(Filter(search="river"), size=2)
    second = depot.count(Filter(search="river"), start=first["next"], size=2)
    last = depot.count(Filter(search="river"), start=second["next"], size=2)

    assert first["total"] is None and (first["texts"], second["texts"], last["texts"]) == (2, 2, 1)
    assert last["next"] is None and depot.count(Filter(search="lake"))["total"] == 1


def test_a_clone_and_its_pulls_keep_the_title_index_of_the_upstream_s_texts_and_its_own(serve, make_depot, me,
                                                                                         tmp_path, title_index):
    upstream = make_depot("upstream")
    theirs = upstream.add([NewText(text=f"u{n}", licenses=("CC0-1.0",), title=f"Book {n}") for n in range(3)], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"),
                     About("follower", "Grace Hopper", "grace@example.org"))
    mine = follower.add([NewText(text="mine", licenses=("CC0-1.0",), title="My book")], me)
    upstream.edit_credits_many([{"id": theirs[0], "title": "Renamed"}, {"id": theirs[1], "title": None}], me)
    [later] = upstream.add([NewText(text="later", licenses=("CC0-1.0",), title="A later book")], me)
    upstream.decide_erasure(upstream.ask_erasure("gone", [(theirs[2], None, None)], me), "admitted", me)

    pull(follower)

    kept, made = counted_again(follower, title_index)
    assert kept == made and made["titles"]["unfound"] == []
    assert every_match(follower, Filter(search="book")) == ["follower-1", later]
    assert every_match(follower, Filter(search="book")) == every_match(follower, "title LIKE '%book%'")
    assert mine == ["follower-1"]


def test_a_pull_brings_the_upstream_s_decisions_as_rows_and_keeps_the_clone_s_statuses_and_counts(
        serve, make_depot, me, tmp_path, title_index):
    upstream, follower, (refused, kept) = upstream_and_clone_holding(serve, make_depot, me, tmp_path)
    decision, _, _ = upstream.decide("refused", [refused], me)

    pull(follower)

    assert follower.show(refused)["status"] == "refused" and follower.show(kept)["status"] == "admitted"
    assert follower.decision(decision)["texts"] == 1
    assert follower.search("status = 'refused'")["rows"][0]["id"] == refused
    kept_counts, made = counted_again(follower, title_index)
    assert kept_counts == made and made["status"] == {"admitted": 1, "refused": 1}
    # a text it brings later counts as it is upstream
    [later] = upstream.add([plain("admitted later")], me)
    upstream.decide("admitted", [later], me)
    pull(follower)
    kept_counts, made = counted_again(follower, title_index)
    assert kept_counts == made and follower.held_count(Filter(statuses=("admitted",)))["statuses"]["admitted"] == 2


def test_an_operation_on_a_filter_read_at_a_change_leaves_out_the_texts_changed_since_and_says_how_many(
        make_depot, me):
    # what a page showed at a change is what a bulk action takes: never a text changed or added after it
    from wpt_depot.depot import CHANGED_SINCE
    from wpt_depot.filters import Filter
    depot = make_depot()
    first, edited, unticked = depot.add([plain("one", "x"), plain("two", "x"), plain("three", "x")], me)
    read_at = depot.change()
    depot.edit_credits(edited, {"title": "Two"}, me)
    [added] = depot.add([plain("four", "x")], me)

    decision, decided, left = depot.decide("admitted", Filter(tags=("x",), unticked=(unticked,), change=read_at), me)

    assert (decided, left) == (1, {CHANGED_SINCE: 2})
    assert [depot.show(one)["status"] for one in (first, edited, unticked, added)] == [
        "admitted", "under review", "under review", "under review"]
    retag_left = {}
    assert depot.retag(Filter(change=read_at), ["y"], [], me, left=retag_left)[1] == 1
    assert retag_left == {CHANGED_SINCE: 3}


def test_a_page_and_a_count_step_of_a_decision_read_its_rows_when_few_and_test_each_text_reached_when_many(
        make_depot, me, monkeypatch):
    # the live depot holds hundreds of millions of decision rows: a page or a count step of a decision reads only its
    # rows, in order, never every text, and one of a decision on many texts tests each text it reads instead of
    # reading the whole decision at each page and each step
    from wpt_depot.filters import Filter
    depot = make_depot()
    texts = depot.add([plain(str(n)) for n in range(12)], me)
    decision = depot.decide("admitted", texts[8:] + texts[2:4], me)[0]
    depot.decide("refused", texts[3:5], me)
    read = depot.db.person_rows

    def planned(where):
        """The ids of a page and the count of the texts where matches, with the
        lines of the query plans of the page and of the first count step that
        name text_decisions, its rows l or the slice of a count step."""
        queries = []
        monkeypatch.setattr(depot.db, "person_rows",
                            lambda sql, params=None: queries.append((sql, params)) or read(sql, params))
        page = [one["id"] for one in depot.search(where)["rows"]]
        searched = len(queries)
        counted = depot.count(where, size=5)
        stepped = len(queries)
        steps = [counted]
        while steps[-1]["next"] is not None:
            steps.append(depot.count(where, steps[-1]["next"], size=5))
        monkeypatch.setattr(depot.db, "person_rows", read)
        plans = [[one[3] for sql, params in part for one in depot.db.reader().execute(
            f"EXPLAIN QUERY PLAN {sql}", params) if "text_decisions" in one[3] or "texts_latest" in one[3]
            or one[3].startswith("SEARCH l ")]
            for part in (queries[:searched], queries[searched:stepped])]
        return page, sum(sum(one["statuses"].values()) for one in steps), plans

    page, counted, [paged, stepped] = planned(Filter(decision=decision))
    assert (page, counted) == (texts[2:4] + texts[8:], 6)
    assert "SEARCH text_decisions USING COVERING INDEX text_decisions_order (decision=?)" in paged, paged
    assert "SEARCH text_decisions USING COVERING INDEX text_decisions_order (decision=?)" in stepped, stepped

    monkeypatch.setattr(depots, "SORTED", 5)
    page, counted, [paged, stepped] = planned(Filter(decision=decision))
    assert (page, counted) == (texts[2:4] + texts[8:], 6)
    for plan in (paged, stepped):  # its rows read in order as the page's candidates, each text tested by its key
        assert "SEARCH text_decisions USING COVERING INDEX text_decisions_order (decision=?)" in plan, plan
        tested = r"SEARCH l EXISTS USING .*\((decision=\? AND text=\?|text=\? AND decision=\?)\)"
        assert any(re.fullmatch(tested, one) for one in plan), plan


def test_a_page_in_an_order_no_index_serves_is_refused_past_sorted_matches_with_text_or_without(
        make_depot, me, monkeypatch):
    # such an order sorts every match, so past SORTED matches both the rows and the passages refuse it clearly
    # instead of sorting until the reader's time limit stops them
    depot = make_depot()
    first, _ = depot.add([plain("one"), plain("two")], me)
    monkeypatch.setattr(depots, "SORTED", 1)

    for read in (depot.search, depot.passages):
        with pytest.raises(Refused, match="order by id or status"):
            read("true", order="title")
        assert len(read("true", order="id")["rows"]) == 2
        assert len(read(Filter(ids=(first,)), order="title")["rows"]) == 1


def recipe_depot(make_depot, me):
    """A depot holding an admitted text and a snapshot of it: (the depot, the snapshot's id)."""
    depot = make_depot()
    depot.decide("admitted", depot.add([plain("one", "en")], me), me)
    return depot, depot.create_snapshot("first")["id"]


def test_a_tokenizer_recipe_takes_a_new_version_only_when_what_trains_its_vocabulary_changes(make_depot, me):
    depot, snapshot = recipe_depot(make_depot, me)
    recipe = depot.new_tokenizer("admitted", "first", None, 1000, 600, "the first note")
    shown = depot.tokenizer("admitted")
    assert (shown["version"], shown["sample"], shown["needs_snapshot"]) == (
        1, {"snapshot": snapshot, "kinds": ["plain"]}, False)
    assert (shown["depot"], shown["dataset"]) == (None, "")

    assert depot.edit_tokenizer(recipe, snapshot, None, 1000, 600, "another note") == 1
    assert depot.edit_tokenizer(recipe, snapshot, ["plain"], 1000, 600, "another note") == 1  # no change, no revision
    assert depot.edit_tokenizer(recipe, snapshot, None, 2000, 600, "another note") == 2
    second = depot.create_snapshot("second")["id"]
    assert depot.edit_tokenizer(recipe, "second", None, 2000, 600, None) == 3

    assert [one["revision"] for one in depot.sql(f"SELECT revision FROM tokenizers_revisions WHERE id = '{recipe}' "
                                                 "ORDER BY revision")] == [1, 2, 3, 4]
    assert depot.tokenizer_by_id(recipe)["sample"]["snapshot"] == second
    with pytest.raises(Refused):
        depot.new_tokenizer("admitted", snapshot, None, 1000, 600)  # a name is taken for good
    with pytest.raises(Refused):
        depot.new_tokenizer("small", snapshot, None, 1000, 100)  # under the 513 tokens every vocabulary holds
    with pytest.raises(Refused, match="kind plain"):
        depot.new_tokenizer("items", snapshot, ["conversation"], 1000, 600)
    with pytest.raises(NotFound):
        depot.new_tokenizer("none", "no such snapshot", None, 1000, 600)


def test_a_tokenizer_recipe_written_before_snapshots_keeps_its_dataset_and_needs_a_snapshot_to_train(make_depot, me):
    depot, snapshot = recipe_depot(make_depot, me)
    recipe = depot.new_tokenizer("old", snapshot, None, 1000, 600)
    with depot.db.write("an older revision") as (con, number):
        con.execute("UPDATE tokenizers_revisions SET sample = NULL, dataset = 'datasets-1' WHERE id = ?", (recipe,))

    shown = depot.tokenizer("old")
    assert (shown["sample"], shown["dataset"], shown["needs_snapshot"]) == (None, "datasets-1", True)
    assert depot.tokenizers()[0]["needs_snapshot"]
    assert depot.edit_tokenizer(recipe, snapshot, None, 1000, 600) == 2  # naming a snapshot changes what trains it
    assert not depot.tokenizer("old")["needs_snapshot"]


def test_a_vocabulary_is_kept_once_per_version_with_the_texts_it_trained_on(make_depot, me):
    depot, snapshot = recipe_depot(make_depot, me)
    recipe = depot.new_tokenizer("admitted", snapshot, None, 1000, 600)
    merges = b"a vocabulary file"

    sha = depot.add_vocabulary(recipe, 1, merges, [("", "depot-1"), ("https://other.example.org", "other-7")], 12)

    found = depot.vocabulary(recipe, 1)
    assert sha == hashlib.sha256(merges).hexdigest() == found["sha256"]
    assert found["merges"].encode() == merges and (found["texts"], found["bytes"]) == (2, 12)
    assert depot.vocabulary_texts(recipe, 1) == [{"position": 0, "depot": "", "text": "depot-1"},
                                                 {"position": 1, "depot": "https://other.example.org",
                                                  "text": "other-7"}]
    assert depot.tokenizers()[0]["trained"]
    with pytest.raises(Refused, match="already"):
        depot.add_vocabulary(recipe, 1, b"another", [], 0)
    with pytest.raises(Refused, match="never had"):
        depot.add_vocabulary(recipe, 2, merges, [], 0)
    depot.edit_tokenizer(recipe, snapshot, None, 1000, 700)
    assert not depot.tokenizers()[0]["trained"]  # the new version has none yet
    depot.add_vocabulary(recipe, 2, b"the second", [], 0)
    assert depot.vocabulary(recipe, 1)["merges"] == "a vocabulary file"  # the first never changes


def test_a_clone_receives_the_upstream_s_tokenizer_recipes_and_vocabularies_and_a_pull_their_changes(
        serve, make_depot, me, tmp_path):
    upstream, snapshot = recipe_depot(make_depot, me)
    recipe = upstream.new_tokenizer("admitted", snapshot, None, 1000, 600)
    upstream.add_vocabulary(recipe, 1, b"first", [("", "upstream-1")], 3)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"),
                     About("follower", "Grace Hopper", "grace@example.org"))
    assert follower.tokenizer("admitted")["version"] == 1
    assert follower.vocabulary(recipe, 1)["merges"] == "first"

    upstream.edit_tokenizer(recipe, snapshot, None, 2000, 600)
    upstream.add_vocabulary(recipe, 2, b"second", [("", "upstream-1")], 3)
    pull(follower)

    assert follower.tokenizer("admitted")["version"] == 2
    assert follower.vocabulary(recipe, 2)["merges"] == "second"
    assert follower.vocabulary_texts(recipe, 2) == [{"position": 0, "depot": "", "text": "upstream-1"}]
    with pytest.raises(BelongsToUpstream):
        follower.edit_tokenizer(recipe, snapshot, None, 3000, 600)


def test_an_admitted_text_put_back_under_review_can_be_decided_again(make_depot, me):
    # an admission made by mistake must be undone by a recorded decision, and the text then reviewed afresh
    depot = make_depot()
    undone, kept = depot.add([plain("one"), plain("two")], me)
    depot.decide("admitted", "true", me)

    decision, decided, left = depot.decide("under review", [undone], me, reason="admitted by mistake")

    assert (decided, left) == (1, {})
    shown = depot.show(undone)
    assert [(one["status"], one["reason"], one["counts"], one["put_back"]) for one in shown["decisions"]] == [
        ("under review", None, False, False), ("admitted", None, False, False),
        ("under review", "put back under review: admitted by mistake", True, True)]
    assert shown["decisions"][-1]["id"] == decision
    assert statuses_everywhere(depot, undone) == {"under review"}
    assert depot.show(kept)["status"] == "admitted"
    depot.decide("admitted", [undone], me)
    assert statuses_everywhere(depot, undone) == {"admitted"}


def test_a_refused_text_put_back_under_review_no_longer_blocks_an_admission(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    depot.decide("refused", [text], me)

    depot.decide("under review", [text], me)

    assert depot.show(text)["decisions"][-1]["reason"] == "put back under review"
    assert statuses_everywhere(depot, text) == {"under review"}
    depot.decide("admitted", [text], me)
    assert statuses_everywhere(depot, text) == {"admitted"}
    assert depot.count("true")["statuses"] == {"under review": 0, "admitted": 1, "refused": 0}


def test_putting_back_under_review_leaves_out_texts_under_review_already_and_never_only_wipes_decisions(
        make_depot, me):
    depot = make_depot()
    decided, waiting = depot.add([plain("one"), plain("two")], me)
    depot.decide("admitted", [decided], me)
    depot.decide("admitted", [waiting], me)
    depot.decide("under review", [waiting], me)

    assert depot.decide("under review", [decided, waiting], me)[1:] == (1, {depots.ALREADY_UNDER_REVIEW: 1})
    with pytest.raises(NotFound, match="under review already"):
        depot.decide("under review", [decided, waiting], me)
    assert [len(depot.show(one)["decisions"]) for one in (decided, waiting)] == [3, 3]


def test_a_person_s_reason_cannot_pass_for_the_depot_s_own_put_back(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    depot.decide("admitted", [text], me)
    before = every_row(depot)

    for status in ("refused", "under review"):
        with pytest.raises(Refused, match="put back under review"):
            depot.decide(status, [text], me, reason="put back under review: really")

    assert every_row(depot) == before


def test_a_filter_finds_texts_on_a_decision_and_by_whole_words(make_depot, me):
    from wpt_depot.filters import Filter, condition
    depot = make_depot()
    river, drivers, other = depot.add([NewText(text="by the river bank", licenses=("CC0-1.0",), title="Rivers"),
                                       NewText(text="the drivers", licenses=("CC0-1.0",), title="Drivers"),
                                       plain("nothing here")], me)
    decision, _, _ = depot.decide("admitted", [river, drivers], me)

    def found(**fields):
        return column(depot.search(condition(Filter(**fields)))["rows"], "id")

    assert found(decision=decision) == [river, drivers]
    assert found(search="river") == [river, drivers]  # the titles
    assert found(search="river", whole_words=True) == []  # neither title holds the whole word
    assert found(search="river", in_text=True, whole_words=True) == [river]
    assert found(search="RIVER", in_text=True, whole_words=True) == [river]  # in any case of its ASCII letters
    assert found(search="RIVER BANK", in_text=True) == [river]


def test_change_times_gives_the_time_of_each_change_named_the_depot_holds(make_depot, me):
    depot = make_depot()
    depot.add([plain("one")], me)
    now = depot.change()

    times = depot.change_times([now, now + 100])

    assert list(times) == [now] and times[now] == depot.sql(f"SELECT made_at FROM changes WHERE id = {now}")[0]["made_at"]


def test_a_content_change_of_many_texts_reads_only_the_texts_it_names(make_depot, me, monkeypatch):
    # every read of the depot's tables goes from the change's own lines: no step scans the texts, their content
    # lists or their decisions in the whole depot, as one that SQLite turned around would; on a depot of
    # millions of texts each such scan reads the whole file for a change of a few
    depot = make_depot()
    texts = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",)) for n in range(40)], me)
    depot.decide("admitted", "true", me)
    monkeypatch.setattr(depots.rewriting, "close", lambda con: None)  # its temporary tables kept to plan on
    writer = depot.db._write_connection()
    traced = []
    writer.exec_trace = lambda cursor, sql, bindings: traced.append((sql, bindings)) or True

    depot.change_content_many([{"id": texts[3], "text": "text 3, from its source"},
                               {"id": texts[7], "text": "text 7, from its source"}], me)

    writer.exec_trace = None
    # the erasures are few, whatever the number of texts: read whole, they cost nothing
    tables = {row[0] for row in writer.execute("SELECT name FROM main.sqlite_master WHERE type IN ('table', 'index') "
                                               "AND tbl_name NOT LIKE 'erasures%'")}
    scans = []
    for sql, bindings in traced:
        if not re.match(r"\s*(SELECT|INSERT|UPDATE|DELETE|WITH)", sql, re.I):
            continue
        for row in writer.execute(f"EXPLAIN QUERY PLAN {sql}", bindings or ()):
            if row[3].startswith("SCAN") and set(re.findall(r"\w+", row[3])) & tables:
                scans.append((row[3], sql))
    assert not scans, "\n".join(f"{plan}   <- {sql[:150]}" for plan, sql in scans)


def excerpt(source, exact, prefix=None, suffix=None):
    """An excerpt block of the chat format quoting source (docs/chat-format.md, "Excerpts")."""
    selector = {"type": "TextQuoteSelector", "exact": exact}
    selector.update({key: value for key, value in (("prefix", prefix), ("suffix", suffix)) if value is not None})
    return {"type": "excerpt", "source": source, "selector": selector}


def quoting(*blocks, role="user"):
    """A conversation whose first message, of that role, holds the blocks, and an assistant's answer."""
    return NewText(kind="conversation", licenses=("CC0-1.0",), content=json.dumps({"messages": [
        {"role": role, "content": [{"type": "text", "text": "Read this."}, *blocks]},
        {"role": "assistant", "content": [{"type": "text", "text": "Read."}]}]}))


def test_an_excerpt_is_refused_unless_it_quotes_one_place_of_a_plain_text_the_depot_holds_now(make_depot, me):
    # a conversation depends on what it quotes, so the depot takes only a quote the text holds once: a quote of
    # nothing, of an item or of two places would leave the build nothing it can put in its place
    depot = make_depot()
    text, other = depot.add([plain("the dog sat; the dog ran"), plain("a cat")], me)
    [talk] = depot.add([NewText(kind="conversation", licenses=("CC0-1.0",), content=json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}))], me)

    for wrong, problem in ((excerpt(f"{text}9", "dog"), "no text of the depot"),
                           (excerpt(talk, "hi"), "an item of kind conversation"),
                           (excerpt(text, ""), "quotes no character"),
                           (excerpt(text, "horse"), "matches no place"),
                           (excerpt(text, "the dog"), "more than one place"),
                           (excerpt(text, "dog", prefix="the "), "more than one place")):
        with pytest.raises(BadContent, match=problem):
            depot.add([quoting(wrong)], me)
    with pytest.raises(BadContent, match="does not match its schema"):  # an assistant never quotes
        depot.add([quoting(excerpt(text, "dog", suffix=" ran"), role="assistant")], me)
    in_result = {"type": "tool_result", "call": "c1", "content": [excerpt(other, "cat")]}
    assert depot.add([quoting(excerpt(text, "dog", suffix=" ran")),
                      quoting(excerpt(text, "dog sat", prefix="the "), excerpt(other, "a cat"))], me)
    [quotes_in_a_result] = depot.add([NewText(kind="conversation", licenses=("CC0-1.0",), content=json.dumps(
        {"messages": [{"role": "tool", "content": [in_result]}]}))], me)

    # a change of the conversation's content is checked against the quoted text as it is then
    depot.change_content(other, "a cat and a cat", None, me)
    with pytest.raises(BadContent, match="more than one place"):
        depot.change_content(quotes_in_a_result, None, json.dumps({"messages": [{"role": "tool", "content": [
            {**in_result, "content": [excerpt(other, "cat"), {"type": "text", "text": "seen"}]}]}]}), me)


def test_the_selectors_on_one_text_read_it_once_and_find_what_each_finds_alone():
    # a build matches every excerpt of a text in one reading, so each must find the places it finds alone, however
    # the text's parts cut it, and whatever the longer or overlapping selectors beside it find
    text = "abab cab abab dog ab cat ab"
    selectors = [{"exact": "ab"}, {"exact": "abab", "prefix": " "}, {"exact": "dog", "prefix": "abab ", "suffix": " ab"},
                 {"exact": "cat"}, {"exact": ""}, {"exact": "bab"}, {"exact": "zebra"}, {"exact": "b", "suffix": " "}]
    for size in (1, 2, 3, 5, len(text)):
        blocks = [text[at:at + size] for at in range(0, len(text), size)]
        for most in (1, 2, 5):
            assert quotes.find_each(blocks, selectors, most) == [quotes.find(blocks, one, most) for one in selectors]


def test_an_erasure_of_a_quoted_text_cuts_the_conversations_that_quote_it(make_depot, me):
    # an excerpt holds the passage it quotes, so an erasure of the quoted text reaches it as it reaches the text:
    # an erased passage leaves every string of the conversation, a text removed whole leaves its excerpts empty
    depot = make_depot()
    secret = "555-0199"
    text, gone, apart = depot.add([plain(f"call {secret} today"), plain("a whole text"), plain(f"{secret} again")],
                                  me)
    talk, untouched = depot.add([quoting(excerpt(text, secret, prefix="call "), excerpt(gone, "whole"),
                                         {"type": "text", "text": f"so {secret} it is"}),
                                 quoting(excerpt(apart, secret))], me)

    depot.decide_erasure(depot.ask_erasure("a number", [(text, 5, 5 + len(secret))], me), "admitted", me)
    depot.decide_erasure(depot.ask_erasure("asked by its author", [(gone, None, None)], me), "admitted", me)

    blocks = json.loads(depot.show(talk)["text"]["content"])["messages"][0]["content"]
    assert blocks[1:] == [excerpt(text, "", prefix="call "), excerpt(gone, "", prefix=None),
                          {"type": "text", "text": "so  it is"}]
    assert secret in depot.content(untouched)  # it quotes another text, which the erasure never named
    assert depot.content(text) == "call  today"


def test_an_erased_passage_leaves_no_character_in_an_excerpt_it_only_overlaps(make_depot, me):
    # an excerpt stands over its place in the quoted text, so a passage that starts in its prefix and ends in its
    # exact leaves each of them the characters it covers, and the quote still finds its place in the cut text
    depot = make_depot()
    text = depot.add([plain("Please call Jane at 555-1234 now or later.")], me)[0]
    [talk] = depot.add([quoting(excerpt(text, "555-1234 now", prefix="call Jane at "))], me)
    start = "Please call ".__len__()

    depot.decide_erasure(depot.ask_erasure("a person", [(text, start, start + len("Jane at 555-1234"))], me),
                         "admitted", me)

    assert depot.content(text) == "Please call  now or later."
    block = json.loads(depot.content(talk))["messages"][0]["content"][1]
    assert block == excerpt(text, " now", prefix="call ")
    assert quotes.find(depot.content(text), block["selector"]) == [len("Please call ")]


def test_an_erasure_cuts_only_the_strings_of_text_of_a_conversation_it_reaches_through_a_quote(make_depot, me):
    # a quoted passage may also be part of a name, an id or a role: cutting those would break the conversation's
    # schema or what it names, so the erasure leaves them as they are and cuts its text
    depot = make_depot()
    [text] = depot.add([plain("user 12345 wrote")], me)
    source_like = depot.add([plain("12345")], me)[0]
    talk_content = {"messages": [
        {"role": "user", "content": [excerpt(text, "user 12345 wrote"), {"type": "text", "text": "id 12345?"}]},
        {"role": "assistant", "content": [{"type": "tool_call", "id": "call-12345", "name": "look12345",
                                           "arguments": {"q": "12345"}}]},
        {"role": "tool", "content": [{"type": "tool_result", "call": "call-12345",
                                      "content": [{"type": "text", "text": "found 12345"}]}]}],
        "metadata": {"run": "12345"}}
    [talk] = depot.add([NewText(kind="conversation", licenses=("CC0-1.0",), content=json.dumps(talk_content))], me)

    depot.decide_erasure(depot.ask_erasure("a number", [(text, 5, 10)], me), "admitted", me)

    cut = json.loads(depot.content(talk))
    assert cut["messages"][0]["content"] == [excerpt(text, "user  wrote"), {"type": "text", "text": "id ?"}]
    assert cut["messages"][1]["content"][0] == {"type": "tool_call", "id": "call-12345", "name": "look12345",
                                                "arguments": {"q": ""}}
    assert cut["messages"][2]["content"][0]["call"] == "call-12345"
    assert cut["messages"][2]["content"][0]["content"] == [{"type": "text", "text": "found "}]
    assert cut["metadata"] == {"run": "12345"}
    assert depot.content(source_like) == "12345"  # a text the erasure never named


def test_an_erasure_naming_a_conversation_and_the_text_it_quotes_cuts_both_ways(make_depot, me):
    # the erasure's own cut of the conversation's field must not stand in for the cut of its quote
    depot = make_depot()
    [text] = depot.add([plain("call 555-0199 today")], me)
    [talk] = depot.add([quoting(excerpt(text, "555-0199", prefix="call "))], me)

    depot.decide_erasure(depot.ask_erasure("a number", [(text, 5, 13), (talk, 0, 4, "/messages/0/content/0/text")],
                                           me), "admitted", me)

    blocks = json.loads(depot.content(talk))["messages"][0]["content"]
    assert blocks == [{"type": "text", "text": " this."}, excerpt(text, "", prefix="call ")]


def test_the_table_quotes_follows_every_write_of_a_conversation(make_depot, me):
    # an erasure finds the conversations quoting a text through it, so it must hold what each version quotes
    depot = make_depot()
    first, second = depot.add([plain("one text"), plain("another text")], me)
    [talk] = depot.add([quoting(excerpt(first, "one"))], me)

    def held():
        return [tuple(row.values()) for row in depot.sql("SELECT conversation, version, text FROM quotes "
                                                          "ORDER BY version, text")]
    assert held() == [(talk, 1, first)]
    depot.change_content(talk, None, json.dumps(json.loads(quoting(excerpt(second, "another"),
                                                                   excerpt(first, "one")).content)), me)
    assert held() == [(talk, 1, first), (talk, 2, first), (talk, 2, second)]
    depot.decide_erasure(depot.ask_erasure("asked", [(talk, None, None)], me), "admitted", me)
    assert held() == []


def test_a_pull_cuts_the_conversations_that_quote_a_text_the_upstream_erased(serve, make_depot, me, tmp_path):
    # the upstream cut its own conversations, which the pull reads again; the clone cuts its own, finding the
    # passage by its length and public hash, since the clone never holds the request's passage once pulled
    upstream = make_depot("upstream")
    secret = "555-0199"
    [text] = upstream.add([plain(f"call {secret} today")], me)
    [theirs] = upstream.add([quoting(excerpt(text, secret))], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"),
                     About("follower", "Grace Hopper", "grace@example.org"))
    [own] = follower.add([quoting(excerpt(text, secret, suffix=" today"))], me)
    upstream.decide_erasure(upstream.ask_erasure("a number", [(text, 5, 5 + len(secret))], me), "admitted", me)

    pull(follower)

    for talk in (theirs, own):
        assert secret not in follower.content(talk)
    assert json.loads(follower.content(own))["messages"][0]["content"][1] == excerpt(text, "", suffix=" today")
    assert files_holding(follower.address, secret) == []


def test_every_revision_keeps_the_kind_length_and_bytes_of_the_content_version_it_reads(
        serve, make_depot, me, tmp_path):
    # the counts, a decision, a change of licenses or tags, a snapshot's read and a build's walk read them from the
    # revision alone, so every write that makes a revision or a content version, and every cut, keeps them exact
    upstream = make_depot("upstream")
    upstream.add_license("MIT", "MIT License", "text")
    secret = "555-0199"
    first, second, whole = upstream.add([plain(f"call {secret} today", "a"), plain("ünïcödé text", "b"),
                                         plain("all of it")], me)
    talk, item = upstream.add([quoting(excerpt(first, secret, prefix="call ")), probe("Name?")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"),
                     About("follower", "Grace Hopper", "grace@example.org"))
    [own] = follower.add([quoting(excerpt(first, secret, suffix=" today"))], me)
    writes = [
        lambda: upstream.decide("admitted", "true", me),
        lambda: upstream.relicense([first, item], ["MIT"], me),
        lambda: upstream.retag(Filter(ids=(first, second)), ["c"], ["a"], me),
        lambda: upstream.edit_credits(second, {"title": "Second", "tags": ["d"]}, me),
        lambda: upstream.edit_scores([{"id": second, "scores": [score("rank", 1)]}], me),
        lambda: upstream.change_content(first, f"call {secret} today, and {secret} tomorrow", None, me),
        lambda: upstream.change_content(item, None, json.dumps({"question": "Age, in years?",
                                                                "patterns": ["(?i)x"]}), me),
        lambda: upstream.change_content_many([{"id": second, "text": "a longer text, ünïcödé"},
                                              {"id": whole, "text": "all of it, longer"}], me),
        lambda: upstream.decide("refused", [second], me),
        # a cut of every version of the text and of the conversation that quotes it, an item's field, a whole text
        lambda: upstream.decide_erasure(upstream.ask_erasure("a number", [(first, 5, 5 + len(secret))], me),
                                        "admitted", me),
        lambda: upstream.decide_erasure(upstream.ask_erasure("a field", [(item, 0, 3, "/question")], me),
                                        "admitted", me),
        lambda: upstream.decide_erasure(upstream.ask_erasure("asked", [(whole, None, None)], me), "admitted", me),
    ]
    for write in writes:
        write()
        assert sizes_apart(upstream) == [] and tags_apart(upstream) == []

    pull(follower)  # carries out the erasures: the clone's cut of its own conversation and of the pulled versions

    for depot in (upstream, follower):
        assert sizes_apart(depot) == [] and tags_apart(depot) == []
        assert depot.rows("SELECT count(*) AS n FROM texts_revisions")[0]["n"] > 5
    assert secret not in follower.content(own)
    [cut] = follower.rows("SELECT content_length, content_bytes FROM texts_revisions WHERE id = ? AND latest", (own,))
    assert (cut["content_length"], cut["content_bytes"]) == (len(follower.content(own)),
                                                             len(follower.content(own).encode()))


def probe(question):
    return NewText(kind="probe", content=json.dumps({"question": question, "patterns": ["(?i)x"]}),
                   licenses=("CC0-1.0",))


def test_a_text_s_current_revision_holds_its_tags_now_through_retags_new_revisions_and_pulls(
        serve, make_depot, me, tmp_path):
    # the counts, the texts view and a count step read a text's tags from its current revision, which a change of tags
    # alone rewrites in place, with no new change of its own that a pull would see
    upstream = make_depot("upstream")
    first, second = upstream.add([plain("one", "b", "a", "b"), plain("two")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"),
                     About("follower", "Grace Hopper", "grace@example.org"))
    upstream.retag(Filter(ids=(first, second)), ["c"], ["a"], me)
    upstream.decide("admitted", [first], me)  # stored on the current revision, with no new one
    upstream.relicense([second], ["CC0-1.0"], me)  # a new revision, which copies the tags
    upstream.edit_credits(first, {"title": "One"}, me)  # a new revision: the follower's past one drops its copy
    pull(follower)

    for depot in (upstream, follower):
        assert tags_apart(depot) == []
        assert {row["id"]: row["tags"] for row in depot.rows("SELECT id, tags FROM texts")} == {
            first: ["b", "c"], second: ["c"]}
        assert depot.count(Filter())["tags"] == [{"tag": "c", "texts": 2}, {"tag": "b", "texts": 1}]


def test_a_kind_filter_matches_the_kind_of_each_text_s_current_content_and_pages_can_be_ordered_by_kind(
        make_depot, me):
    depot = make_depot()
    first, item, second, other = depot.add([plain("one", "en"), probe("Name?"), plain("two"), probe("Age?")], me)
    depot.decide("admitted", [first, item], me)

    def ids_of(found, **options):
        return column(depot.search(found, **options)["rows"], "id")

    assert ids_of(Filter(kinds=("probe",))) == [item, other]
    assert ids_of(Filter(kinds=("plain",))) == [first, second]
    assert ids_of(Filter(kinds=("plain", "probe", "plain"))) == [first, item, second, other]
    assert ids_of(Filter(kinds=("conversation",))) == []
    assert ids_of(Filter(kinds=("probe",), statuses=("admitted",))) == [item]
    assert ids_of(Filter(kinds=("plain",), tags=("en",))) == [first]
    assert ids_of(Filter(), order="kind") == [first, second, item, other]
    assert ids_of(Filter(), order="kind", descending=True) == [other, item, second, first]
    # a decision on a kind filter takes those texts only
    depot.decide("refused", Filter(kinds=("probe",), statuses=("under review",)), me)
    assert depot.show(other)["status"] == "refused" and depot.show(second)["status"] == "under review"


def test_the_depot_s_counts_answer_a_kind_filter_as_the_steps_count_it(make_depot, me):
    depot = make_depot()
    first, item, second, other = depot.add([plain("one", "en"), probe("Name?"), plain("two"), probe("Age?")], me)
    depot.decide("admitted", [first, item, second], me)
    depot.decide("refused", [second], me)
    shapes = [Filter(kinds=("plain",)), Filter(kinds=("probe",)), Filter(kinds=("plain", "probe")),
              Filter(kinds=("conversation",)), Filter(kinds=("probe",), statuses=("admitted",)),
              Filter(kinds=("plain",), statuses=("refused", "under review"))]

    for found in shapes:
        held, stepped = depot.held_count(found), depot.count(found)
        assert held is not None and held["held"] is True, found
        assert held["statuses"] == stepped["statuses"], found
    assert depot.held_count(Filter(kinds=("conversation",)))["tags"] == []
    # with a tag or a license, the counts do not split by kind: the steps count it
    for other_shape in (Filter(kinds=("plain",), tags=("en",)), Filter(kinds=("plain",), licenses=("CC0-1.0",))):
        assert depot.held_count(other_shape) is None


def test_a_retag_writes_rows_of_taggings_for_the_texts_it_changes_with_one_decision_no_revision_and_keeps_status(
        make_depot, me, title_index):
    depot = make_depot()
    has_both, has_old, has_none, has_new = depot.add(
        [plain("one", "old", "new"), plain("two", "x", "old"), plain("three"), plain("four", "new", "y")], me)
    depot.decide("admitted", [has_both, has_old], me)
    revisions = {one: depot.show(one)["text"]["revision"] for one in (has_both, has_old, has_none, has_new)}
    change = depot.change()

    decision, changed, left = depot.retag(Filter(), ["new"], ["old"], me)

    assert (changed, left) == (3, 1)
    assert {one: depot.show(one)["text"]["tags"] for one in (has_both, has_old, has_none, has_new)} == {
        has_both: ["new"], has_old: ["new", "x"], has_none: ["new"], has_new: ["new", "y"]}
    assert {one: depot.show(one)["text"]["revision"] for one in revisions} == revisions
    number = change + 1  # the retag's change: a tag a text left ends its row there, a tag it joined starts one
    written = {(one["text"], one["tag"], one["joined_change"] == number, one["left_change"]) for one in depot.sql(
        f"SELECT text, tag, joined_change, left_change FROM taggings WHERE change = {number}")}
    assert written == {(has_both, "old", False, number), (has_old, "old", False, number), (has_old, "new", True, None),
                       (has_none, "new", True, None)}
    assert depot.sql(f"SELECT text FROM text_decisions WHERE decision = '{decision}' ORDER BY text") == [
        {"text": one} for one in sorted((has_both, has_old, has_none))]
    assert depot.sql(f"SELECT status, reason FROM decisions WHERE id = '{decision}'") == [
        {"status": "under review", "reason": "tags changed: added new; removed old"}]
    assert [depot.show(one)["status"] for one in (has_both, has_old, has_none)] == [
        "admitted", "admitted", "under review"]
    assert depot.sql("SELECT count(*) AS n FROM changes")[-1]["n"] == change + 1
    # a retag that changes no text writes nothing
    assert depot.retag([has_new], ["new"], ["old"], me) == (None, 0, 1)
    assert depot.change() == change + 1
    kept, again = counted_again(depot, title_index)
    assert kept == again


@pytest.mark.parametrize("add, remove", [([], []), (["a"], ["a"]), ([""], []), (["  "], []), ("a", []),
                                         ([3], [])])
def test_a_retag_that_breaks_a_rule_writes_nothing(make_depot, me, add, remove):
    depot = make_depot()
    depot.add([plain("one", "a")], me)
    change = depot.change()
    with pytest.raises(Refused):
        depot.retag(Filter(), add, remove, me)
    with pytest.raises(Refused):
        depot.retag_estimate(Filter(), add, remove)
    assert depot.change() == change


def test_a_retag_estimate_names_the_rows_the_retag_then_writes(make_depot, me):
    depot = make_depot()
    depot.add([plain("one", "a", "b"), plain("two", "a"), plain("three"), plain("four", "c"),
               plain("five", "b", "c")], me)
    tables = ("texts_revisions", "text_decisions", "decisions")

    def rows():
        found = {table: depot.sql(f"SELECT count(*) AS n FROM {table}")[0]["n"] for table in tables}
        found["taggings"] = {tuple(one.values()) for one in depot.sql("SELECT * FROM taggings")}
        return found, {(one["tag"], one["key"]) for one in depot.sql("SELECT tag, key FROM text_tags")}

    for selection, add, remove in ((Filter(), ["c", "d"], ["a"]), (Filter(tags=("b",)), [], ["b", "z"]),
                                   ("id <> 'depot-3'", ["e"], []), (["depot-1", "depot-3"], ["e"], ["c"])):
        estimate = depot.retag_estimate(selection, add, remove)
        (before, tags_before) = rows()
        _, changed, left = depot.retag(selection, add, remove, me)
        (after, tags_after) = rows()
        assert (estimate["changed"], estimate["texts"]) == (changed, changed + left), selection
        assert after["texts_revisions"] == before["texts_revisions"], selection  # no revision
        assert estimate["rows"] == {"text_decisions": after["text_decisions"] - before["text_decisions"],
                                    "decisions": after["decisions"] - before["decisions"],
                                    "taggings": len(after["taggings"] - before["taggings"]),
                                    "text_tags": len(tags_after ^ tags_before)}, selection
        assert (estimate["tags_added"], estimate["tags_removed"]) == (len(tags_after - tags_before),
                                                                      len(tags_before - tags_after)), selection
        assert estimate["written"] == sum(estimate["rows"].values())
        assert estimate["bytes"] == round(changed * depots.RETAG_BYTES) and estimate["seconds"] > 0
    nothing = depot.retag_estimate(Filter(tags=("e",)), ["e"], [])
    assert (nothing["changed"], nothing["written"], nothing["bytes"], nothing["seconds"]) == (0, 0, 0, 0)


def test_a_retag_estimate_is_not_below_the_change_file_the_retag_then_records(make_depot, me):
    # the next backup sends that file: a retag rewrites each text's copy of its tags on its current revision too
    depot = make_depot()
    chosen = depot.add([plain(f"text {number}", "t00", "t01") for number in range(1000)], me)

    def recorded():
        return sum(path.stat().st_size for path in depot.db.waiting_files())

    before = recorded()
    estimate = depot.retag_estimate(chosen, ["bench-tag"], [])
    depot.retag(chosen, ["bench-tag"], [], me)
    assert recorded() - before <= estimate["change_bytes"] == round(1000 * depots.RETAG_CHANGE_BYTES)


def test_a_retag_refuses_texts_the_upstream_owns_and_a_clone_retags_its_own(serve, make_depot, me, tmp_path):
    upstream = make_depot("up")
    [theirs] = upstream.add([plain("theirs", "a")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"),
                     About("follower", "Grace Hopper", "grace@example.org"))
    [own] = follower.add([plain("own", "a")], me)

    assert follower.retag_estimate(Filter(), ["b"], [])["upstream"] == 1
    with pytest.raises(BelongsToUpstream):
        follower.retag(Filter(), ["b"], [], me)
    assert follower.retag([own], ["b"], [], me)[1:] == (1, 0)
    assert follower.show(theirs)["text"]["tags"] == ["a"] and follower.show(own)["text"]["tags"] == ["a", "b"]


def taggings_rows(depot) -> list[tuple]:
    return [tuple(one.values()) for one in depot.sql("SELECT * FROM taggings ORDER BY text, tag, joined_change")]


def test_a_clone_receives_the_upstream_s_taggings_and_each_pull_the_tags_given_and_taken_since(
        serve, make_depot, me, tmp_path, title_index):
    upstream = make_depot("up")
    first, second = upstream.add([plain("one", "a", "b"), plain("two", "a")], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"),
                     About("follower", "Grace Hopper", "grace@example.org"))
    assert taggings_rows(follower) == taggings_rows(upstream)

    upstream.retag([first, second], ["c"], ["a"], me)
    upstream.edit_credits_many([{"id": second, "tags": ["d"]}], me)  # c, given by the retag, taken back
    pull(follower)
    assert taggings_rows(follower) == taggings_rows(upstream)
    upstream.retag([first], [], ["b"], me)  # a tag the follower holds, taken after its pull
    pull(follower)

    assert taggings_rows(follower) == taggings_rows(upstream)
    assert [follower.show(one)["text"]["tags"] for one in (first, second)] == [["c"], ["d"]]
    assert follower.counts()["tag"] == upstream.counts()["tag"] == {"c": 1, "d": 1}
    assert column(follower.search(Filter(tags=("c",)))["rows"], "id") == [first]
    assert follower.search(Filter(tags=("b",)))["rows"] == []
    kept, again = counted_again(follower, title_index)
    assert kept == again


def test_an_erasure_that_removes_a_text_whole_deletes_its_taggings_and_its_tags_counts(make_depot, me, title_index):
    depot = make_depot()
    gone, kept = depot.add([plain("a whole text to remove", "a", "b"), plain("kept", "a")], me)
    depot.retag([gone], [], ["b"], me)  # an ended row too
    request = depot.ask_erasure("asked by its author", [(gone, None, None)], me)

    depot.decide_erasure(request, "admitted", me)

    assert {row[0] for row in taggings_rows(depot)} == {kept}
    assert depot.counts()["tag"] == {"a": 1}
    kept_counts, again = counted_again(depot, title_index)
    assert kept_counts == again


def test_a_filter_of_tags_matches_only_texts_with_every_tag_whichever_listing_a_read_takes(make_depot, me):
    # a read through one tag's listing does not test that tag again: the other tags, and a listing chosen in its
    # place (the texts of a refused license, for blocked), must still be tested
    depot = make_depot()
    depot.add_license("CC-BY-NC-4.0", "CC-BY-NC-4.0", "text")
    both, only_a, only_b = depot.add([plain("both", "a", "b"), plain("only a", "a"), plain("only b", "b")], me)
    depot.retag([only_b], ["rare"], [], me)
    [unlicensed_rare, unlicensed] = depot.add([NewText(text="nc rare", licenses=("CC-BY-NC-4.0",), tags=("rare",)),
                                               NewText(text="nc", licenses=("CC-BY-NC-4.0",), tags=("a",))], me)

    assert column(depot.search(Filter(tags=("a", "b")))["rows"], "id") == [both]
    assert column(depot.search(Filter(tags=("rare",)))["rows"], "id") == [only_b, unlicensed_rare]
    assert column(depot.search(Filter(tags=("rare", "b")))["rows"], "id") == [only_b]
    assert depot.count(Filter(tags=("a", "b")))["statuses"] == {"under review": 1, "admitted": 0, "refused": 0}
    assert column(depot.blocked(Filter(tags=("a",)))["rows"], "id") == [unlicensed]
    assert column(depot.blocked(Filter(tags=("rare",)))["rows"], "id") == [unlicensed_rare]
