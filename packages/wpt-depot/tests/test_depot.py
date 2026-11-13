import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import pytest

from wpt_depot import About, Declaration, Depot, Needle, NewText, clone, pull
from wpt_depot import depot as depots
from wpt_depot import erasure, follow, scoring
from wpt_depot.errors import (BadContent, BadQuery, BadRemote, BelongsToUpstream, Locked, NoDepotSetting, NotFound,
                              Refused, TimeLimitReached, UpstreamUnreachable)
from wpt_depot.depot import BELONGS_TO_UPSTREAM, NO_ALLOWED_LICENSE
from wpt_depot.schema import FORMAT, KEYS
from wpt_depot.settings import depot_address


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
    """The rows of a library's texts (Depot.library_texts) with the text of the content version each reads."""
    return [{**one, "text": depot.content(one["id"], one["content_version"])} for one in table]


def texts_of(depot, table) -> list[dict]:
    return [{"id": one["id"], "text": one["text"]} for one in with_text(depot, table)]


def files_holding(address, passage: str) -> list[Path]:
    """The depot's file and its WAL, those whose bytes hold passage: what an erasure must leave none of (its
    change files wait for the next backup, which sends them away, wpt_depot.backup)."""
    path = Path(address)
    return [one for one in (path, path.with_name(path.name + "-wal"))
            if one.is_file() and passage.encode() in one.read_bytes()]


def test_only_admitted_texts_never_refused_join_a_library_and_a_refusal_takes_them_out(make_depot, me):
    depot = make_depot()
    first, second, third = depot.add([plain("one", "a"), plain("two", "a"), plain("three", "b")], me)
    depot.decide("admitted", "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'a')", me)
    depot.decide("refused", f"id = '{second}'", me)
    library = depot.new_library("first")

    joined, left_out = depot.put(library, "true")

    assert joined == [first]
    assert left_out == [second, third]
    depot.decide("refused", f"id = '{first}'", me)
    assert depot.show(first)["text"]["libraries"] == []
    assert [one["status"] for one in depot.show(first)["decisions"]] == ["under review", "admitted", "refused"]


def test_a_locked_library_takes_no_text_and_a_fork_is_an_open_library_holding_its_admitted_texts(make_depot, me):
    depot = make_depot()
    held, other, refused_later = depot.add([plain("kept", "a"), plain("later", "a"), plain("refused", "a")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("first model")
    depot.put(library, [held, refused_later])
    depot.lock(library)

    with pytest.raises(Locked):
        depot.put(library, f"id = '{other}'")
    depot.decide("refused", [refused_later], me)  # a locked library keeps it, at the revision it pinned
    assert column(depot.library_texts(library, depot.change()), "id") == [held, refused_later]
    fork = depot.fork(library, "next model")
    assert depot.put(fork, f"id = '{other}'") == ([other], [])

    assert column(depot.sql(f"SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = '{library}') ORDER BY id"), "id") == [held, refused_later]
    assert column(depot.sql(f"SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = '{fork}') ORDER BY id"), "id") \
        == [held, other]
    assert depot.sql(f"SELECT forked_from, locked_at FROM libraries WHERE id = '{fork}'") \
        == [{"forked_from": library, "locked_at": None}]


def test_a_text_taken_out_of_an_open_library_leaves_it_stays_admitted_and_can_be_put_back(make_depot, me):
    depot = make_depot()
    leaving, staying, outside = depot.add([plain("leaving"), plain("staying"), plain("outside")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, [leaving, staying])
    before = depot.change()

    assert depot.take_out(library, [leaving, outside]) == ([leaving], [outside])

    assert column(depot.library_texts(library, depot.change()), "id") == [staying]
    assert column(depot.library_texts(library, before), "id") == [leaving, staying]
    assert depot.show(leaving)["status"] == "admitted"
    assert depot.put(library, [leaving]) == ([leaving], [])
    assert column(depot.library_texts(library, depot.change()), "id") == [leaving, staying]


def test_a_locked_library_refuses_a_take_out_and_writes_nothing(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("pinned")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("locked")
    depot.put(library, [text])
    depot.lock(library)
    before = depot.change()

    with pytest.raises(Locked):
        depot.take_out(library, [text])

    assert depot.change() == before
    assert column(depot.library_texts(library, depot.change()), "id") == [text]


def test_a_pull_brings_a_take_out_made_upstream(serve, make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    texts = upstream.add([plain("leaving"), plain("staying")], me)
    upstream.decide("admitted", texts, me)
    library = upstream.new_library("open")
    upstream.put(library, texts)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    upstream.take_out(library, [texts[0]])

    pull(follower)

    assert column(follower.library_texts(library, follower.change()), "id") == [texts[1]]
    assert follower.show(texts[0])["status"] == "admitted"


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
    assert sorted(column(depot.sql("SELECT id FROM kinds"), "id")) == ["conversation", "probe", "rating", "task"]


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
    # a change is one transaction: the fork's library row is written before its texts, and the failure undoes it
    depot = make_depot()
    [text] = depot.add([plain("one", "a")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("first")
    depot.put(library, "true")
    before = depot.change()

    def fails(*args, **kwargs):
        raise OSError("the disk went away")

    with monkeypatch.context() as patch:
        patch.setattr(Depot, "_join", fails)
        with pytest.raises(OSError):
            depot.fork(library, "fork")

    assert column(depot.sql("SELECT count(*) AS n FROM libraries"), "n") == [1]
    assert depot.change() == before
    assert depot.show(text)["text"]["libraries"] == [library]
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

    assert not outside.exists() and not (tmp_path / "copy.sqlite").exists()
    assert column(depot.sql("SELECT count(*) AS n FROM texts"), "n") == [1]
    assert column(depot.sql("SELECT count(*) AS n FROM licenses WHERE name != 'x'"), "n") == [1]


def test_a_person_s_query_stops_at_its_deadline(make_depot, me):
    depot = make_depot()
    depot.db.deadline = time.monotonic() - 1

    with pytest.raises(TimeLimitReached):
        depot.sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n")


def test_a_dataset_exported_as_a_remote_is_imported_as_the_same_dataset(make_depot, me):
    depot = make_depot()
    depot.add([plain("one", "en")], me)
    library = depot.new_library("first")
    change = depot.change()
    depot.new_dataset("first", "classics", [(None, library)], [("en", 0.7), ("fr", 0.3)])
    remote = depot.export_dataset("first")
    other = make_depot("other")

    other.import_dataset(remote)

    assert other.export_dataset("first") == remote
    assert remote["libraries"] == [{"depot": None, "library": library, "change": change}]
    with pytest.raises(Refused):
        other.import_dataset(remote)  # the name is taken
    with pytest.raises(BadRemote):
        other.import_dataset({**remote, "id": 3})


def test_an_open_library_read_at_a_past_change_gives_its_members_and_revisions_then_after_an_erasure_too(make_depot, me):
    depot = make_depot()
    first, second, third = depot.add([plain("one 123", "a"), plain("two"), plain("three")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, [first, second])
    then = depot.change()
    depot.put(library, [third])  # joins after
    depot.decide("refused", [second], me)  # leaves after
    depot.edit_credits(first, {"title": "One"}, me)  # a newer revision of a member

    def read(change):
        return [(one["id"], one["revision"], one["title"], one["text"]) for one in
                with_text(depot, depot.library_texts(library, change))]

    assert read(then) == [(first, 2, None, "one 123"), (second, 2, None, "two")]
    assert read(depot.change()) == [(first, 3, "One", "one 123"), (third, 2, None, "three")]
    request = depot.ask_erasure("a number", [(first, 4, 7)], me)
    depot.decide_erasure(request, "admitted", me)
    assert read(then) == [(first, 2, None, "one "), (second, 2, None, "two")]  # the cut reaches every revision
    removal = depot.ask_erasure("all of it", [(second, None, None)], me)
    depot.decide_erasure(removal, "admitted", me)
    assert read(then) == [(first, 2, None, "one ")]  # a text removed whole is in no past
    with pytest.raises(NotFound):
        depot.library_texts(depot.new_library("later"), then)


def test_a_credits_edit_of_a_text_a_locked_library_holds_is_a_new_revision_the_library_does_not_read(make_depot, me):
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    [text] = depot.add([NewText(text="The fox.", licenses=("CC0-1.0",), title="Fox")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("first")
    depot.put(library, "true")
    depot.lock(library)
    pinned = depot.show(text)["text"]["revision"]

    depot.edit_credits(text, {"title": "The fox"}, me)
    depot.relicense([text], ["MIT"], me)

    shown = depot.show(text)
    assert (shown["text"]["title"], shown["text"]["licenses"], shown["revisions"]) == ("The fox", ["MIT"], pinned + 2)
    assert shown["pins"] == [{"library": library, "name": "first", "revision": pinned}]
    assert pick(depot.library_texts(library, depot.change()), ["id", "revision", "title", "licenses"]) \
        == [{"id": text, "revision": pinned, "title": "Fox", "licenses": ["CC0-1.0"]}]
    assert depot.pinned(library) == {"total": 1, "newer": 1, "rows": [
        {"id": text, "title": "Fox", "revision": pinned, "current": pinned + 2}]}
    assert column(depot.sql(f"SELECT revision, title FROM texts_revisions WHERE id = '{text}' ORDER BY revision"), "title") == ["Fox", "Fox", "The fox", "The fox"]  # added, admitted, credits, licenses


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
    assert column(depot.sql(f"SELECT count(*) AS n FROM texts_revisions WHERE id = '{text}'"), "n") == [3]
    assert files_holding(depot.address, secret) == []


def test_the_sql_names_show_the_current_state_and_their_revisions_every_row(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    library = depot.new_library("first")
    depot.rename_library(library, "second")
    depot.edit_about(maintainer="Grace Hopper")

    assert depot.sql("SELECT id, text, revision, libraries FROM texts") == [
        {"id": text, "text": "one", "revision": 1, "libraries": "[]"}]
    assert depot.sql("SELECT name FROM libraries") == [{"name": "second"}]
    assert column(depot.sql("SELECT name FROM libraries_revisions ORDER BY revision"), "name") == ["first", "second"]
    assert column(depot.sql("SELECT maintainer FROM about_revisions ORDER BY revision"), "maintainer") == [
        "Ada Lovelace", "Grace Hopper"]
    for name in ("contents", "memberships", "pins", "datasets_revisions", "erasures_revisions"):
        depot.sql(f"SELECT * FROM {name}")
    changes = column(depot.sql("SELECT change FROM libraries_revisions ORDER BY revision"), "change")
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
    library = upstream.new_library("open")
    upstream.put(library, [added])
    upstream.edit_credits(added, {"title": "New"}, me)

    assert pull(follower) == [request]

    assert follower.sql(f"SELECT revision, change, title, decisions FROM texts_revisions WHERE id = '{added}' "
                        "ORDER BY revision") == upstream.sql(
        f"SELECT revision, change, title, decisions FROM texts_revisions WHERE id = '{added}' ORDER BY revision"
    )
    assert pick(follower.library_texts(library, follower.change()), ["id", "revision", "title"]) == [
        {"id": added, "revision": 3, "title": "New"}]
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


def test_a_clone_reads_its_locked_library_at_its_state_after_a_pulled_erasure(serve, make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    [text] = upstream.add([plain("one 123")], me)
    upstream.decide("admitted", "true", me)
    library = upstream.new_library("locked")
    upstream.put(library, "true")
    upstream.lock(library)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    state = follower.change()
    request = upstream.ask_erasure("a number", [(text, 4, 7)], me)
    upstream.decide_erasure(request, "admitted", me)

    pull(follower)

    assert texts_of(follower, follower.library_texts(library, state)) == [{"id": text, "text": "one "}]


def upstream_and_clone_holding(serve, make_depot, me, tmp_path):
    """An upstream with two admitted texts, and a clone whose own open and
    own locked libraries hold both: (upstream, clone, texts, open, locked)."""
    upstream = make_depot("upstream")
    upstream.add_license("MIT", "MIT License", "text")
    texts = upstream.add([plain("to be refused"), plain("to be relicensed")], me)
    upstream.decide("admitted", texts, me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    open_library, locked_library = follower.new_library("open"), follower.new_library("locked")
    follower.put(open_library, texts)
    follower.put(locked_library, texts)
    follower.lock(locked_library)
    return upstream, follower, texts, open_library, locked_library


def test_a_pulled_refusal_or_license_change_takes_the_text_out_of_the_clone_s_own_open_libraries_only(serve, 
        make_depot, me, tmp_path):
    upstream, follower, (refused, relicensed), open_library, locked_library = upstream_and_clone_holding(
        serve, make_depot, me, tmp_path)
    before = follower.change()
    upstream.decide("refused", [refused], me)
    upstream.relicense([relicensed], ["MIT"], me)

    pull(follower)

    pulled = follower.change()
    assert follower.library_texts(open_library, pulled) == []
    assert column(follower.library_texts(open_library, before), "id") == [refused, relicensed]
    assert column(follower.library_texts(locked_library, pulled), "id") == [refused, relicensed]
    assert column(follower.sql(f"SELECT DISTINCT left_change FROM memberships WHERE library = '{open_library}'"), "left_change") == [pulled]


def test_a_pull_that_fails_partway_writes_nothing_and_the_next_pull_brings_everything(serve, 
        make_depot, me, tmp_path, monkeypatch):
    upstream, follower, (refused, kept), open_library, _ = upstream_and_clone_holding(serve, make_depot, me, tmp_path)
    upstream.decide("refused", [refused], me)
    before = follower.change()

    def fails(*args, **kwargs):
        raise OSError("the disk went away")

    with monkeypatch.context() as patch:
        patch.setattr(follow, "_leave", fails)
        with pytest.raises(OSError):
            pull(follower)
    assert follower.change() == before and follower.show(refused)["status"] == "admitted"

    pull(follower)

    assert follower.show(refused)["status"] == "refused"
    assert column(follower.library_texts(open_library, follower.change()), "id") == [kept]
    assert follower.sql("SELECT depot, change, upstream, upstream_change FROM pulls ORDER BY change")[-1] \
        == {"depot": "follower", "change": follower.change(), "upstream": "upstream", "upstream_change": upstream.change()}


def test_a_clone_s_own_open_library_read_before_a_pull_holds_the_revisions_it_held_then(serve, make_depot, me, tmp_path):
    # the refusal keeps the upstream's change number, lower than the clone's change read here
    upstream, follower, (refused, kept), open_library, _ = upstream_and_clone_holding(serve, make_depot, me, tmp_path)
    before = follower.change()
    held = pick(follower.library_texts(open_library, before), ["id", "revision", "decisions"])
    upstream.decide("refused", [refused], me)
    assert upstream.sql(f"SELECT change FROM texts WHERE id = '{refused}'")[0]["change"] <= before

    pull(follower)

    assert pick(follower.library_texts(open_library, before), ["id", "revision", "decisions"]) == held
    assert column(follower.library_texts(open_library, follower.change()), "id") == [kept]


def test_a_dataset_naming_an_upstream_library_at_an_upstream_change_reads_the_same_in_the_clone(serve, 
        make_depot, me, tmp_path):
    upstream = make_depot("upstream")
    texts = upstream.add([plain("one"), plain("two"), plain("three")], me)
    upstream.decide("admitted", texts, me)
    library = upstream.new_library("theirs")
    upstream.put(library, texts[:2])
    dataset = upstream.new_dataset("theirs", "tok", [(None, library)])
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    follower.new_library("mine")  # the clone's numbers run past the upstream's
    follower.new_library("more")
    named = follower.new_dataset("mine", "tok", [(None, library)])  # by the clone's address, at the upstream's number
    upstream.put(library, texts[2:])
    upstream.edit_credits(texts[0], {"title": "Later"}, me)
    upstream.decide("refused", [texts[1]], me)

    pull(follower)

    for ref in (upstream.dataset_by_id(dataset)["libraries"][0], follower.dataset_by_id(named)["libraries"][0]):
        assert follower.library_texts(library, ref["change"]) == upstream.library_texts(
            library, ref["change"])
        assert pick(follower.library_texts(library, ref["change"]), ["id", "title"]) == [
            {"id": texts[0], "title": None}, {"id": texts[1], "title": None}]


def test_a_clone_of_a_clone_reads_its_upstream_s_library_as_the_upstream_did_at_that_change(serve, make_depot, me, tmp_path):
    first = make_depot("first")
    [text] = first.add([plain("theirs")], me)
    first.decide("admitted", [text], me)
    second = clone(serve(first), str(tmp_path / "second.sqlite"), About("second", "Grace Hopper", "grace@example.org"))
    library = second.new_library("mine")
    second.put(library, [text])
    second.new_library("more")
    before = second.change()
    first.decide("refused", [text], me)
    pull(second)
    third = clone(serve(second), str(tmp_path / "third.sqlite"), About("third", "Ada Lovelace", "ada@example.org"))

    assert third.library_texts(library, before) == second.library_texts(library, before)
    assert pick(third.library_texts(library, before), ["id", "revision"]) == [
        {"id": text, "revision": 2}]


def test_the_decision_that_admits_an_erasure_and_its_carrying_out_are_one_change(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("a secret")], me)
    request = depot.ask_erasure("why", [(text, 2, 8)], me)
    before = depot.change()

    depot.decide_erasure(request, "admitted", me)

    assert depot.change() == before + 1
    assert depot.erasures()[0]["erased"] is not None


def test_a_fork_without_a_condition_holds_the_texts_that_condition_does_not_match(make_depot, me):
    depot = make_depot()
    kept, left_out, also_kept = depot.add([plain("one", "a"), plain("two", "b"), plain("three", "a")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("first")
    depot.put(library, "true")
    depot.lock(library)

    fork = depot.fork(library, "without b", without="EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'b')")

    assert column(depot.sql(f"SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = '{fork}') ORDER BY id"), "id") \
        == [kept, also_kept]
    assert column(depot.sql(f"SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = '{library}') ORDER BY id"), "id") \
        == [kept, left_out, also_kept]
    assert depot.sql(f"SELECT forked_from, locked_at FROM libraries WHERE id = '{fork}'") \
        == [{"forked_from": library, "locked_at": None}]


def test_a_problem_is_reported_on_a_locked_library_and_changes_nothing_else(make_depot, me):
    depot = make_depot()
    held, other = depot.add([plain("one", "a"), plain("two", "b")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("first")
    depot.put(library, f"id = '{held}'")
    depot.lock(library)
    before = depot.show(held)
    reporter = Declaration("Grace Hopper", "grace@example.org", True)

    decision, texts = depot.report(library, "it is not in English", "true", reporter)
    whole, nothing = depot.report(library, "the library is too small", None, reporter)

    assert (texts, nothing) == ([held], [])  # only the texts the library holds count
    assert [json.loads(one) for one in column(depot.sql(f"SELECT problems FROM libraries WHERE id = '{library}'"), "problems")] == [[
        {"texts": [held], "problem": "it is not in English", "decision": decision},
        {"texts": [], "problem": "the library is too small", "decision": whole}]]
    assert depot.sql(f"SELECT status, name FROM decisions WHERE id = '{decision}'") == [
        {"status": "under review", "name": "Grace Hopper"}]
    assert depot.show(held)["text"] == before["text"] and depot.show(held)["decisions"] == before["decisions"]


def test_a_problem_that_names_no_text_of_the_library_or_has_no_confirmation_writes_nothing(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    library = depot.new_library("first")

    with pytest.raises(NotFound):
        depot.report(library, "a problem", f"id = '{text}'", me)  # the text is in no library
    with pytest.raises(Refused):
        depot.report(library, "a problem", None, Declaration("Ada Lovelace", "ada@example.org", False))

    assert [json.loads(one) for one in column(depot.sql(f"SELECT problems FROM libraries WHERE id = '{library}'"), "problems")] == [[]]
    assert column(depot.sql("SELECT count(*) AS n FROM decisions"), "n") == [1]


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
    library = depot.new_library("first")
    depot.put(library, [held])
    depot.lock(library)
    unlicensed, combined, choice, free = depot.add([
        NewText(text="one", licenses=("CC-BY-NC-4.0",)), NewText(text="two", licenses=("MIT AND CC-BY-NC-4.0",)),
        NewText(text="three", licenses=("CC-BY-NC-4.0", "CC0-1.0")), plain("four")], me)

    found = depot.blocked([free, choice, combined, unlicensed, under_review, held])

    assert found == [{"id": unlicensed, "reasons": [NO_ALLOWED_LICENSE]},
                     {"id": combined, "reasons": [NO_ALLOWED_LICENSE]}]
    assert depot.blocked([under_review, held]) == []  # a locked library holding a text blocks nothing
    for one in found:
        with pytest.raises(Refused):
            depot.decide("admitted", [one["id"], free], me)
    assert depot.show(free)["status"] == "under review"
    assert depot.decide("admitted", [free, choice, under_review, held], me)[1] == [held, under_review, choice, free]
    assert depot.decide("refused", [unlicensed, combined], me)[1] == [unlicensed, combined]
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
    # a text, or a text outside the library a problem is reported on, writes nothing
    depot = make_depot()
    ids = depot.add([plain(f"text {n}", "a") for n in range(1, 12)], me)
    reviewed = [ids[9], ids[1]]
    depot.add([plain("added after the review", "a")], me)

    decision, decided = depot.decide("admitted", reviewed, me)

    assert decided == [ids[1], ids[9]]  # in the numeric order of their ids: depot-2 before depot-10
    assert depot.search("true", 0, 3)["rows"][1]["id"] == ids[1]
    assert [one["status"] for one in depot.search(f"id = '{ids[0]}'")["rows"]] == ["under review"]
    with pytest.raises(NotFound):
        depot.decide("admitted", [ids[0], "depot-99"], me)
    assert depot.show(ids[0])["status"] == "under review"
    library = depot.new_library("first")
    assert depot.put(library, [ids[1], ids[2]]) == ([ids[1]], [ids[2]])
    with pytest.raises(Refused):
        depot.report(library, "a problem", [ids[1], ids[2]], me)
    assert depot.report(library, "a problem", [ids[1]], me)[1] == [ids[1]]
    fork = depot.fork(library, "without", [ids[1]])
    assert depot.search(f"EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = '{fork}')")["total"] == 0


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
                 if text in depot.matching(condition(Filter(ids=(text,), statuses=(status,))))]
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

    assert changed == [text] and skipped == []
    shown = depot.show(text)
    assert shown["text"]["licenses"] == ["MIT"]
    assert [(one["status"], one["reason"], one["counts"]) for one in shown["decisions"]] == [
        ("under review", None, False), ("refused", "non-commercial", False),
        ("under review", "licenses changed from CC-BY-NC-4.0 to MIT", True)]
    assert shown["decisions"][-1]["id"] == decision and shown["decisions"][-1]["name"] == me.name
    assert statuses_everywhere(depot, text) == {"under review"}
    depot.decide("admitted", [text], me)
    assert statuses_everywhere(depot, text) == {"admitted"}


def test_an_admitted_text_whose_licenses_change_leaves_its_open_libraries_and_is_no_longer_admitted(make_depot, me):
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    changed, kept = depot.add([plain("one"), plain("two")], me)
    depot.decide("admitted", [changed, kept], me)
    library = depot.new_library("open")
    depot.put(library, [changed, kept])

    depot.relicense([changed], ["MIT", "CC0-1.0"], me)

    assert depot.show(changed)["text"]["libraries"] == []
    assert depot.show(kept)["text"]["libraries"] == [library]
    assert statuses_everywhere(depot, changed) == {"under review"}
    assert depot.put(library, [changed]) == ([], [changed])
    assert depot.search("true")["statuses"] == {"under review": 1, "admitted": 1, "refused": 0}


def test_texts_with_different_old_licenses_each_get_the_reason_true_for_them(make_depot, me):
    depot = make_depot()
    for license_id in ("CC-BY-NC-4.0", "MIT", "GPL-2.0-only"):
        depot.add_license(license_id, license_id, "text")
    first, second, third = depot.add([NewText(text="one", licenses=("CC-BY-NC-4.0",)),
                                      NewText(text="two", licenses=("CC-BY-NC-4.0", "GPL-2.0-only")),
                                      NewText(text="three", licenses=("CC-BY-NC-4.0",))], me)

    found, _ = depot.relicense("true", ["MIT"], me)

    assert [texts for _, texts in found] == [[first, third], [second]]
    assert [depot.show(text)["decisions"][-1]["reason"] for text in (first, second, third)] == [
        "licenses changed from CC-BY-NC-4.0 to MIT", "licenses changed from CC-BY-NC-4.0, GPL-2.0-only to MIT",
        "licenses changed from CC-BY-NC-4.0 to MIT"]


def test_a_license_change_that_breaks_a_rule_writes_nothing(make_depot, me):
    depot = make_depot()
    depot.add_license("MIT", "MIT License", "text")
    held, free = depot.add([plain("held"), plain("free")], me)
    depot.decide("admitted", [held], me)
    library = depot.new_library("first")
    depot.put(library, [held])
    depot.lock(library)
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

    assert depot.relicense([same], ["CC0-1.0", "MIT"], me) == ([], [same])
    assert depot.change() == before and depot.show(same)["status"] == "refused"

    [(decision, changed)], skipped = depot.relicense([other, same], ["MIT", "CC0-1.0"], me)

    assert (changed, skipped) == ([other], [same])
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
    library = depot.new_library("first")
    depot.put(library, "true")

    decision, reason = depot.edit_credits(text, {"title": "The fox", "tags": ["x", "z"], "authors": ["Ann"],
                                                 "notice": "(c) Ann Lee"}, me)

    shown = depot.show(text)
    assert reason == ("title changed from Fox to The fox; authors changed from none to Ann; tags changed from x, y to "
                      "x, z; notice changed from depot-1 to depot-2")
    assert (shown["text"]["title"], shown["text"]["tags"], shown["text"]["authors"]) == ("The fox", ["x", "z"], ["Ann"])
    assert shown["notice"] == "(c) Ann Lee"
    assert (shown["status"], shown["text"]["libraries"]) == ("admitted", [library])
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


def test_one_scores_change_is_one_decision_on_many_texts_which_keep_their_status_and_libraries(make_depot, me):
    depot = make_depot()
    first, second, third = depot.add([plain("one"), plain("two"), plain("three")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("open")
    depot.put(library, "true")
    locked = depot.new_library("locked")
    depot.put(locked, "true")
    depot.lock(locked)

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
        assert (shown["status"], shown["text"]["libraries"]) == ("admitted", [library, locked])
        assert [(one["id"], one["status"], one["reason"]) for one in shown["decisions"]][-1] == (
            decision, "under review", reason)
    assert column(depot.sql(f"SELECT count(*) AS n FROM texts WHERE EXISTS (SELECT 1 FROM json_each(decisions) WHERE value = '{decision}')"), "n") \
        == [2]
    assert column(depot.library_texts(locked, depot.change()), "scores") == [None, None, None]
    assert depot.sql("SELECT id, (SELECT json_group_array(s.value ->> 'name') FROM json_each(scores) s) AS names "
                     "FROM texts WHERE scores IS NOT NULL ORDER BY id") == [
        {"id": first, "names": '["machine-made"]'}, {"id": second, "names": '["machine-made","source-rank"]'}]


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


def test_a_library_open_or_locked_takes_a_new_name_and_keeps_its_texts(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("one")], me)
    depot.decide("admitted", "true", me)
    library = depot.new_library("first")
    depot.put(library, "true")
    depot.lock(library)

    depot.rename_library(library, "first model")

    assert depot.sql(f"SELECT name, locked_at IS NOT NULL AS locked FROM libraries WHERE id = '{library}'") \
        == [{"name": "first model", "locked": True}]
    assert depot.show(text)["text"]["libraries"] == [library]
    with pytest.raises(Refused):
        depot.rename_library(library, " ")
    with pytest.raises(NotFound):
        depot.rename_library("depot-99", "x")


def test_a_dataset_changes_as_a_new_revision_keeping_its_id_and_the_change_of_the_libraries_it_kept(make_depot, me):
    depot = make_depot()
    depot.add([plain("one", "en")], me)
    first, second = depot.new_library("first"), depot.new_library("second")
    dataset = depot.new_dataset("mix", "tok", [(None, first)])
    change = depot.dataset("mix")["libraries"][0]["change"]
    depot.add([plain("two")], me)  # the depot's change number moves on
    now = depot.change()

    depot.edit_dataset(dataset, "mix two", [(None, first), (None, second)], [("en", 1.0)])

    row = depot.dataset_by_id(dataset)
    assert (row["name"], row["tokenizer"], row["revision"]) == ("mix two", "tok", 2)
    assert [(one["library"], one["change"]) for one in row["libraries"]] == [(first, change), (second, now)]
    assert column(depot.sql(f"SELECT name FROM datasets_revisions WHERE id = '{dataset}' ORDER BY revision"), "name") == ["mix", "mix two"]
    assert row["shares"] == [{"tag": "en", "share": 1.0}]
    depot.new_dataset("other", "tok", [(None, first)])
    for name, libraries, shares in (("other", [(None, first)], None), ("mix two", [], None),
                                    ("mix two", [(None, first)], [("en", 2.0)])):
        with pytest.raises(Refused):
            depot.edit_dataset(dataset, name, libraries, shares)
    with pytest.raises(NotFound):
        depot.edit_dataset(dataset, "mix two", [(None, "depot-99")])
    assert depot.dataset_by_id(dataset) == row


def test_an_imported_dataset_keeps_each_library_as_its_remote_wrote_it_when_it_changes(make_depot, me, tmp_path):
    depot = make_depot()
    depot.add([plain("one", "en")], me)
    library = depot.new_library("first")
    written = Path(depot.address).as_uri()  # another form of the depot's address
    other = make_depot("other")
    dataset = other.import_dataset({"name": "mix", "tokenizer": "tok", "shares": None, "libraries": [
        {"depot": written, "library": library, "change": depot.change()},
        {"depot": str(tmp_path / "nowhere"), "library": "nowhere-1", "change": 7}]})
    stored = other.dataset_by_id(dataset)["libraries"]
    depot.add([plain("two")], me)  # the library's depot moves on

    # as the CLI does, with the stored addresses; then with the plain path for the first
    other.edit_dataset(dataset, "mix two", [(one["depot"], one["library"]) for one in stored])
    assert other.dataset_by_id(dataset)["libraries"] == stored
    other.edit_dataset(dataset, "mix three", [(depot.address, library), (str(tmp_path / "nowhere"), "nowhere-1")])

    row = other.dataset_by_id(dataset)
    assert row["name"] == "mix three"
    assert row["libraries"] == stored


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
    """An upstream holding an admitted text in an open and a locked library,
    an erasure request under review and a dataset, and a clone of it."""
    upstream = make_depot("upstream")
    [text] = upstream.add([plain("upstream text 123", "a")], me)
    upstream.decide("admitted", [text], me)
    library, locked = upstream.new_library("open"), upstream.new_library("locked")
    upstream.put(library, [text])
    upstream.put(locked, [text])
    upstream.lock(locked)
    request = upstream.ask_erasure("a number", [(text, 14, 17)], me)
    dataset = upstream.new_dataset("theirs", "tok", [(None, library)])
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace Hopper", "grace@example.org"))
    return upstream, follower, {"text": text, "library": library, "locked": locked, "request": request,
                                "dataset": dataset}


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
    "rename a library": lambda depot, me, ids: depot.rename_library(ids["library"], "mine"),
    "lock a library": lambda depot, me, ids: depot.lock(ids["library"]),
    "put into a library": lambda depot, me, ids: depot.put(ids["library"], [ids["mine"]]),
    "take out of a library": lambda depot, me, ids: depot.take_out(ids["library"], [ids["text"]]),
    "report on a library": lambda depot, me, ids: depot.report(ids["locked"], "a problem", None, me),
    "edit a dataset": lambda depot, me, ids: depot.edit_dataset(ids["dataset"], "renamed", [(None, ids["library"])]),
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
    assert follower.blocked([mine, ids["text"]], "refused") == [{"id": ids["text"], "reasons": [BELONGS_TO_UPSTREAM]}]
    assert follower.show(ids["text"])["upstream"] and not follower.show(mine)["upstream"]
    assert upstream.blocked([ids["text"]]) == [] and not upstream.show(ids["text"])["upstream"]


def test_a_clone_s_own_library_holds_and_pins_upstream_texts_and_its_own_rows_stay_changeable(upstream_and_clone,
                                                                                               me):
    upstream, follower, ids = upstream_and_clone
    [mine] = follower.add([plain("my own text")], me)
    follower.edit_credits(mine, {"title": "Mine"}, me)
    follower.decide("admitted", [mine], me)
    library = follower.new_library("my library")
    follower.rename_library(library, "my pinned library")

    assert follower.put(library, [ids["text"], mine]) == ([mine, ids["text"]], [])
    follower.lock(library)
    fork = follower.fork(ids["library"], "my copy of theirs")  # a fork of an upstream library is the clone's own
    follower.put(fork, [mine])
    upstream.edit_credits(ids["text"], {"title": "Newer"}, me)
    pull(follower)

    assert follower.pinned(library)["rows"] == [{"id": mine, "title": "Mine", "revision": 3, "current": 3},
                                                {"id": ids["text"], "title": None, "revision": 2, "current": 3}]
    assert [one["id"] for one in follower.library_texts(fork, follower.change())] == [mine, ids["text"]]
    assert column(follower.sql(f"SELECT name FROM libraries WHERE id = '{library}'"), "name") == [
        "my pinned library"]


def test_a_pull_after_the_clone_s_own_work_keeps_both_depots_rows(upstream_and_clone, me):
    upstream, follower, ids = upstream_and_clone
    [mine] = follower.add([plain("my own text")], me)
    follower.decide("admitted", [mine], me)
    library = follower.new_library("mine")
    follower.put(library, [mine, ids["text"]])
    dataset = follower.new_dataset("mine", "tok", [(None, library), (None, ids["library"])])
    own_request = follower.ask_erasure("my own", [(mine, 0, 3)], me)
    [more] = upstream.add([plain("upstream text later")], me)
    upstream.decide("admitted", [more], me)
    upstream.put(ids["library"], [more])
    upstream.decide_erasure(ids["request"], "admitted", me)

    assert pull(follower) == [ids["request"]]

    assert follower.sql("SELECT id, text FROM texts ORDER BY id") == [
        {"id": mine, "text": "my own text"}, {"id": ids["text"], "text": "upstream text "},
        {"id": more, "text": "upstream text later"}]
    assert column(follower.sql(f"SELECT id FROM texts WHERE EXISTS (SELECT 1 FROM json_each(libraries) WHERE value = '{library}') ORDER BY id"), "id") == [mine, ids["text"]]
    assert follower.dataset_by_id(dataset)["name"] == "mine"
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


def test_a_content_change_starts_the_text_over_and_a_locked_library_keeps_reading_the_old_content(make_depot, me):
    # the maintainer's 26.11 scenario: 26.10 is locked and trained on, 26.11 is its fork
    depot = make_depot()
    changed, kept = depot.add([plain("The fox jumps."), plain("The dog sleeps.")], me)
    depot.decide("admitted", "true", me)
    old = depot.new_library("good code 26.10")
    depot.put(old, "true")
    depot.lock(old)
    new = depot.fork(old, "good code 26.11")

    decision, version = depot.change_content(changed, "The fox jumps high.", None, me)

    shown = depot.show(changed)
    assert (version, shown["text"]["content_version"], shown["content_versions"]) == (2, 2, 2)
    assert shown["status"] == "under review" and shown["text"]["libraries"] == [old]
    assert shown["decisions"][-1]["id"] == decision
    assert shown["decisions"][-1]["reason"] == "content changed from version 1 to version 2"
    assert [one["counts"] for one in shown["decisions"]] == [False, False, True]
    assert column(depot.library_texts(new, depot.change()), "id") == [kept]
    assert depot.put(new, [changed]) == ([], [changed])
    depot.decide("admitted", [changed], me)
    assert depot.put(new, [changed]) == ([changed], [])
    assert texts_of(depot, depot.library_texts(new, depot.change())) == [
        {"id": changed, "text": "The fox jumps high."}, {"id": kept, "text": "The dog sleeps."}]
    assert texts_of(depot, depot.library_texts(old, depot.change())) == [
        {"id": changed, "text": "The fox jumps."}, {"id": kept, "text": "The dog sleeps."}]
    assert contents_of(depot, changed) == ["The fox jumps.", "The fox jumps high."]
    assert column(depot.sql(f"SELECT text FROM texts WHERE id = '{changed}'"), "text") == ["The fox jumps high."]


def test_an_open_library_read_before_a_content_change_gives_the_old_content(make_depot, me):
    depot = make_depot()
    [text] = depot.add([plain("first words")], me)
    depot.decide("admitted", [text], me)
    library = depot.new_library("open")
    depot.put(library, [text])
    before = depot.change()
    depot.change_content(text, "second words", None, me)
    left = depot.change()
    depot.decide("admitted", [text], me)
    depot.put(library, [text])

    assert [one["text"] for one in texts_of(depot, depot.library_texts(library, before))] == ["first words"]
    assert depot.library_texts(library, left) == []
    assert [one["text"] for one in texts_of(depot, depot.library_texts(library, depot.change()))] == ["second words"]


def test_an_erasure_after_a_content_change_cuts_the_passage_from_every_version_and_stays_exact_for_the_one_named(
        make_depot, me):
    depot = make_depot()
    secret = "555-0199"
    [text] = depot.add([plain(f"call {secret} now")], me)
    depot.decide("admitted", [text], me)
    library = depot.new_library("26.10")
    depot.put(library, [text])
    depot.lock(library)
    depot.change_content(text, f"Please, call {secret} or {secret} today.", None, me)
    request = depot.ask_erasure("a number", [(text, 5, 13, None, 1)], me)  # the place in the first version
    assert depot.erasure_shown(request)["needles"][0]["passage"] == secret

    depot.decide_erasure(request, "admitted", me)

    assert contents_of(depot, text) == ["call  now", "Please, call  or  today."]
    assert [one["text"] for one in texts_of(depot, depot.library_texts(library, depot.change()))] == ["call  now"]
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


def test_a_pulled_content_change_takes_the_text_out_of_the_clone_s_own_open_libraries(serve, make_depot, me, tmp_path):
    upstream, follower, (changed, kept), open_library, locked_library = upstream_and_clone_holding(
        serve, make_depot, me, tmp_path)
    upstream.change_content(changed, "new words", None, me)

    pull(follower)

    assert column(follower.library_texts(open_library, follower.change()), "id") == [kept]
    assert texts_of(follower, follower.library_texts(locked_library, follower.change())) == [
        {"id": changed, "text": "to be refused"}, {"id": kept, "text": "to be relicensed"}]


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
