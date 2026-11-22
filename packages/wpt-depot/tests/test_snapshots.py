import hashlib
import json
import re
import struct
from pathlib import Path

import apsw
import pytest

import wpt_depot
from wpt_depot import About, Depot, NewText, clone, ids, pull, snapshots
from wpt_depot.errors import NotFound, Refused

CONVERSATION = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
                             {"role": "assistant", "content": [{"type": "text", "text": "Hello there."}]}]}


def held(depot, snapshot_id: str) -> set[str]:
    """The ids of every text the snapshot's sets hold, every part together."""
    return {ids.make(owner, number) for _, owner, _, bitmap in snapshots.sets(depot, snapshot_id) for number in bitmap}


def admitted(depot) -> set[str]:
    return {one["id"] for one in depot.rows("SELECT id FROM texts WHERE status = 'admitted'")}


def every_text(depot, snapshot_id: str, **asked) -> tuple[list[dict], int]:
    """Every text of the snapshot, read a page of 2 at a time, and how many erased texts the pages went past."""
    rows, erased, after = [], 0, None
    while True:
        page = depot.snapshot_texts(snapshot_id, after=after, limit=2, **asked)
        rows += page["texts"]
        erased += page["erased"]
        after = page["after"]
        if after is None:
            return rows, erased


def mixed(depot, me) -> list[str]:
    """Texts of every kind of case: plain texts with tags given out of order, without tags and with two licenses,
    conversations, texts under review and refused ones. Their ids, in order."""
    depot.add_license("MIT", "MIT License", "text")
    texts = depot.add([NewText(text=f"text number {n} " + "é" * n, licenses=("CC0-1.0",) if n % 4 else ("CC0-1.0", "MIT"),
                               tags=(("b", "a") if n % 3 == 0 else ("c",) if n % 3 == 1 else ()))
                       for n in range(14)], me)
    texts += depot.add([NewText(kind="conversation", content=json.dumps(CONVERSATION), licenses=("CC0-1.0",),
                                tags=("a",)) for _ in range(2)], me)
    depot.decide("admitted", texts[:12] + texts[14:], me)
    depot.decide("refused", texts[10:12], me)
    return texts


def test_a_snapshot_made_while_texts_change_holds_what_was_admitted_at_its_change(make_depot, me):
    # a snapshot of the public depot reads for minutes, while people decide, add, edit and erase: none of it may wait
    # for the snapshot, and what it holds must be the texts admitted at the change it names, whatever moved meanwhile
    depot = make_depot()
    texts = mixed(depot, me)
    under_review = depot.add([NewText(text=f"late {n}", licenses=("CC0-1.0",)) for n in range(4)], me)
    other = Depot.open(depot.address)
    pages = []

    def meanwhile(done):
        pages.append(done)
        if len(pages) == 1:  # the first page read, the rest not yet
            other.decide("admitted", [under_review[0], under_review[3]], me)  # one read already, one not
            other.decide("refused", [texts[0], texts[9]], me)  # one read already, one not
            other.change_content(texts[1], "text number 1, changed", None, me)  # back under review
            added = other.add([NewText(text="added meanwhile", licenses=("CC0-1.0",))], me)
            other.decide("admitted", added, me)
            whole = other.ask_erasure("all of it", [(texts[2], None, None)], me)
            other.decide_erasure(whole, "admitted", me)  # its checkpoint waits for every reader
            cut = other.ask_erasure("a passage", [(texts[3], 0, 4)], me)
            other.decide_erasure(cut, "admitted", me)
        elif len(pages) == 2:
            other.decide("refused", [under_review[0]], me)

    measured = {}
    made = snapshots.create(depot, "2026-11", read=meanwhile, page=3, measured=measured)

    assert len(pages) >= 3 and measured["at_change"] > measured["started"] and measured["moved"] >= 8
    assert measured["longest_read"] < 1.0
    assert made["at_change"] == measured["at_change"] and made["change"] == depot.change()
    assert held(depot, made["id"]) == admitted(depot)
    assert made["texts"] == len(admitted(depot))
    assert texts[2] not in held(depot, made["id"]) and texts[3] in held(depot, made["id"])


def test_the_summary_holds_what_a_count_of_the_texts_themselves_gives(make_depot, me):
    # a build plans its shares from the summary alone, so it must say exactly what the snapshot's texts hold, after
    # content changes, a cut, a relicense and a retag kept the counts by every write
    depot = make_depot()
    texts = mixed(depot, me)
    depot.change_content(texts[4], "a longer text number four, changed", None, me)
    depot.relicense([texts[5]], ["MIT"], me)
    depot.edit_credits(texts[6], {"tags": ["d", "a"]}, me)
    depot.decide("admitted", texts[4:6], me)
    request = depot.ask_erasure("a passage", [(texts[7], 0, 5)], me)
    depot.decide_erasure(request, "admitted", me)

    made = depot.create_snapshot("2026-11")

    rows, erased = every_text(depot, made["id"])
    direct = {"kinds": {}, "tags": {}, "plain_tagsets": {}, "licenses": {}, "owners": {}}
    for row in rows:
        sizes = [1, row["length"], row["bytes"]]
        names = [("kinds", row["kind"])] + [("tags", tag) for tag in row["tags"]]
        names += [("licenses", license_) for license_ in row["licenses"]]
        if row["kind"] == "plain":
            names.append(("plain_tagsets", "\0".join(sorted(set(row["tags"])))))
        for into, name in names:
            found = direct[into].setdefault(name, [0, 0, 0])
            found[:] = [a + b for a, b in zip(found, sizes)]
        direct["owners"][ids.owner(row["id"])] = direct["owners"].get(ids.owner(row["id"]), 0) + 1
    summary = made["summary"]
    assert erased == 0 and summary.pop("version") == 1
    assert summary == {key: dict(sorted(value.items())) for key, value in direct.items()}
    assert summary["plain_tagsets"]["a\0b"][0] == 3 and summary["kinds"]["conversation"][0] == 2
    assert (made["texts"], made["characters"], made["bytes"]) == tuple(
        sum(one[at] for one in summary["kinds"].values()) for at in range(3))
    assert depot.snapshot_count(made["id"]) == dict(zip(("texts", "characters", "bytes"),
                                                       (made["texts"], made["characters"], made["bytes"])))
    assert depot.snapshot_count(made["id"], ["conversation", "task"]) == dict(
        zip(("texts", "characters", "bytes"), summary["kinds"]["conversation"]))


def test_the_digest_is_the_sha256_of_the_numbers_sorted_as_the_module_says(make_depot, me):
    # an outsider checks a snapshot with any roaring library or none, from the numbers alone
    depot = make_depot()
    mixed(depot, me)
    made = depot.create_snapshot("2026-11")

    numbers = sorted(ids.number(one) for one in admitted(depot))
    expected = hashlib.sha256(b"wpt-snapshot-1\n" + f"{made['at_change']}\n".encode()
                              + f"0\ndepot\n{made['at_change']}\n{len(numbers)}\n".encode()
                              + b"".join(struct.pack("<I", one) for one in numbers)).hexdigest()

    assert made["digest"] == expected
    assert snapshots.digest_of([(0, "depot", made["at_change"], reversed(numbers))], made["at_change"]) == expected
    # a set that does not match its digest is refused on read
    with depot.db.write("damage") as (con, _):
        con.execute("UPDATE snapshot_sets SET bitmap = ?", (snapshots.BitMap(numbers[1:]).serialize(),))
    snapshots._kept.clear()
    with pytest.raises(Refused, match="digest"):
        snapshots.sets(depot, made["id"])


def test_a_snapshot_reads_its_texts_at_their_revisions_then_whatever_changes_after(make_depot, me):
    # a recipe names a snapshot so that a build next year reads what it read today: an edit, a retag or a refusal
    # after it changes nothing it reads, and an erasure of a whole text is skipped and counted
    depot = make_depot()
    texts = mixed(depot, me)
    made = depot.create_snapshot("2026-11")
    before, _ = every_text(depot, made["id"])

    depot.edit_credits(texts[0], {"tags": ["z"], "title": "Retitled"}, me)
    depot.retag([texts[1], texts[2]], ["late"], ["a"], me)  # tags alone: no revision, rows of taggings
    depot.change_content(texts[3], "text number 3, much longer than it was", None, me)
    depot.decide("admitted", [texts[3]], me)
    depot.decide("refused", [texts[5]], me)
    request = depot.ask_erasure("all of it", [(texts[6], None, None)], me)
    depot.decide_erasure(request, "admitted", me)

    after, erased = every_text(depot, made["id"])
    assert erased == 1
    assert after == [one for one in before if one["id"] != texts[6]]
    assert [one["id"] for one in before] == sorted(held(depot, made["id"]), key=ids.order_key)
    first = next(one for one in after if one["id"] == texts[0])
    assert first["tags"] == ["a", "b"] and first["title"] is None and first["revision"] == 1
    plain, erased_plain = every_text(depot, made["id"], kinds=["plain"])
    assert [one["id"] for one in plain] == [one["id"] for one in after if one["kind"] == "plain"] and erased_plain == 1
    conversations, _ = every_text(depot, made["id"], kinds=["conversation"])
    assert [one["id"] for one in conversations] == texts[14:]
    assert depot.snapshot_count(made["id"]) == {"texts": made["texts"], "characters": made["characters"],
                                                "bytes": made["bytes"]}  # the summary, never a recount
    later = depot.create_snapshot("2026-12")
    third = next(one for one in every_text(depot, later["id"])[0] if one["id"] == texts[3])
    assert third["revision"] == 2 and texts[5] not in held(depot, later["id"])


def test_a_number_held_by_two_parts_is_read_in_the_first_part_at_its_bound(make_depot, me):
    # a snapshot an upgrade makes from a dataset has a part per library, each read at its own change, and a text two
    # libraries hold is read as the first one held it
    depot = make_depot()
    first, second = depot.add([NewText(text=f"text {n}", licenses=("CC0-1.0",)) for n in range(2)], me)
    depot.decide("admitted", [first, second], me)
    early = depot.change()
    depot.edit_credits(first, {"title": "Later"}, me)
    depot.edit_credits(second, {"title": "Later"}, me)
    late = depot.change()
    parts = [(0, "depot", early, snapshots.BitMap([ids.number(first)])),
             (1, "depot", late, snapshots.BitMap([ids.number(first), ids.number(second)]))]
    with depot.db.write("snapshot") as (con, number):
        con.execute(f"INSERT INTO snapshots ({snapshots.COLUMNS}) VALUES ('depot-1', ?, 'two', ?, 'now', ?, NULL, "
                    "2, 0, 0, ?, '{}')", (number, late, json.dumps({"kind": "made"}), snapshots.digest_of(parts, late)))
        con.executemany("INSERT INTO snapshot_sets (snapshot, part, owner, bound, bitmap, texts) "
                        "VALUES ('depot-1', ?, ?, ?, ?, 1)", [(p, o, b, m.serialize()) for p, o, b, m in parts])

    rows, _ = every_text(depot, "depot-1")

    assert [(one["id"], one["part"], one["revision"], one["title"]) for one in rows] == [
        (first, 0, 1, None), (second, 1, 2, "Later")]


def test_a_clone_s_snapshot_holds_each_owner_s_texts_at_its_own_bound_and_names_are_per_owner(
        make_depot, me, serve, tmp_path):
    # a clone holds its upstream's texts in the upstream's numbering and its own in its own, so each owner has a
    # set and a bound; a clone's "2026-11" never collides with its upstream's "2026-11"
    upstream = make_depot("up")
    theirs = upstream.add([NewText(text=f"upstream {n}", licenses=("CC0-1.0",)) for n in range(4)], me)
    upstream.decide("admitted", theirs[:3], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace", "g@example.org"))
    ours = follower.add([NewText(text=f"own {n}", licenses=("CC0-1.0",)) for n in range(2)], me)
    follower.decide("admitted", ours, me)

    made = follower.create_snapshot("2026-11")

    brought = follower.rows("SELECT upstream_change FROM pulls ORDER BY change DESC LIMIT 1")[0]["upstream_change"]
    assert [(part, owner, bound, len(bitmap)) for part, owner, bound, bitmap in snapshots.sets(follower, made["id"])] \
        == [(0, "follower", made["at_change"], 2), (0, "up", brought, 3)]
    assert made["summary"]["owners"] == {"follower": 2, "up": 3}
    upstream.edit_credits(theirs[0], {"title": "Edited upstream"}, me)
    pull(follower)
    rows, _ = every_text(follower, made["id"])
    assert [one["id"] for one in rows] == ours + theirs[:3]
    assert next(one for one in rows if one["id"] == theirs[0])["title"] is None

    # the same name in two owners: the upstream's own snapshot, which a pull brings, beside the clone's
    theirs_made = upstream.create_snapshot("2026-11")
    pull(follower)
    assert follower.snapshot("2026-11")["id"] == made["id"]
    assert follower.snapshot("2026-11", owner="up")["id"] == theirs_made["id"]
    assert [one["id"] for one in follower.snapshots()] == [made["id"], theirs_made["id"]]
    with pytest.raises(Refused, match="already"):
        follower.create_snapshot("2026-11")
    with pytest.raises(apsw.ConstraintError):
        with follower.db.write("twice") as (con, _):
            con.execute(f"INSERT INTO snapshots ({snapshots.COLUMNS}) SELECT 'follower-9', change, name, at_change, "
                        "made_at, origin, note, texts, characters, bytes, digest, summary FROM snapshots WHERE id = ?",
                        (made["id"],))
    with pytest.raises(NotFound):
        follower.snapshot("2026-11", owner="elsewhere")


def test_a_pull_while_a_snapshot_reads_starts_the_reading_over(make_depot, me, serve, tmp_path):
    # rows a pull brings carry the upstream's numbering, which the catching up cannot find by change number
    upstream = make_depot("up")
    theirs = upstream.add([NewText(text=f"upstream {n}", licenses=("CC0-1.0",)) for n in range(4)], me)
    upstream.decide("admitted", theirs[:2], me)
    follower = clone(serve(upstream), str(tmp_path / "follower.sqlite"), About("follower", "Grace", "g@example.org"))
    other = Depot.open(follower.address)
    pulled = []

    def meanwhile(_):
        if not pulled:
            upstream.decide("admitted", theirs[2:], me)
            upstream.decide("refused", theirs[:1], me)
            pulled.append(pull(other))

    measured = {}
    made = snapshots.create(follower, "2026-11", read=meanwhile, page=1, measured=measured)

    assert measured["tries"] == 2 and held(follower, made["id"]) == set(theirs[1:]) == admitted(follower)


def test_a_snapshot_s_name_and_numbers_are_checked_and_the_counts_must_hold_what_was_read(make_depot, me):
    depot = make_depot()
    mixed(depot, me)
    for name in ("", "a b", "-x", "é", None):
        with pytest.raises(Refused, match="name"):
            depot.create_snapshot(name)
    depot.create_snapshot("v1.0_a-b")
    with pytest.raises(Refused, match="already"):
        depot.create_snapshot("v1.0_a-b")
    # counts that drifted from the rows refuse it, and a recount mends them
    with depot.db.write("drift") as (con, _):
        con.execute("DELETE FROM counts WHERE facet = 'kind' AND name = 'admitted conversation'")
    with pytest.raises(Refused, match="recount"):
        depot.create_snapshot("drifted")
    depot.recount()
    assert depot.create_snapshot("drifted")["texts"] == len(admitted(depot))
    # a number no 32-bit set holds
    with depot.db.write("far") as (con, _):
        con.execute("UPDATE numbers SET last = ? WHERE name = 'texts'", (1 << 32,))
    far = depot.add([NewText(text="far", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", far, me)
    with pytest.raises(Refused, match="above"):
        depot.create_snapshot("far")
    assert [one["name"] for one in depot.snapshots()] == ["v1.0_a-b", "drifted"]


def test_no_write_changes_a_snapshot_once_written(make_depot, me):
    # a recipe and a trained model name a snapshot for good: nothing may change its row or its sets afterwards
    depot = make_depot()
    texts = mixed(depot, me)
    made = depot.create_snapshot("2026-11")

    def stored():
        con = depot.db.reader()
        return (con.execute("SELECT * FROM snapshots WHERE id = ?", (made["id"],)).fetchall(),
                con.execute("SELECT * FROM snapshot_sets WHERE snapshot = ? ORDER BY part, owner",
                            (made["id"],)).fetchall())

    before = stored()
    depot.retag(texts[:4], ["some"], [], me)
    depot.decide("refused", texts[:2], me)
    depot.relicense([texts[4]], ["MIT"], me)
    depot.change_content(texts[5], "changed", None, me)
    depot.edit_credits(texts[6], {"tags": ["new"]}, me)
    depot.edit_scores([{"id": texts[7], "scores": [{"name": "rank", "value": 1, "content_version": 1}]}], me)
    depot.decide_erasure(depot.ask_erasure("whole", [(texts[8], None, None)], me), "admitted", me)
    depot.decide_erasure(depot.ask_erasure("part", [(texts[9], 0, 3)], me), "admitted", me)
    depot.create_snapshot("2026-12")
    depot.recount()

    assert stored() == before
    # and no code of the depot holds a statement that would
    source = Path(wpt_depot.__file__).parent
    changing = re.compile(r"(UPDATE|DELETE\s+FROM|REPLACE\s+INTO|INSERT\s+OR\s+REPLACE\s+INTO)\s+snapshot", re.I)
    assert [path.name for path in source.glob("*.py") if changing.search(path.read_text(encoding="utf-8"))] == []


def test_the_last_erasure_rises_with_an_erasure_revision_pulled_at_a_lower_change_than_the_depot_s_own(make_depot,
                                                                                                         me):
    # a pull keeps the upstream's change numbers: a key on the highest change would not see the revision come
    depot = make_depot()
    [text] = depot.add([NewText(text="a text", licenses=("CC0-1.0",))], me)
    depot.decide("admitted", [text], me)
    request = depot.ask_erasure("asked", [(text, 0, 1)], me)
    assert depot.decide_erasure(request, "admitted", me) is True
    before = snapshots.last_erasure(depot)
    assert before > 0
    with depot.db.write("as a pull writes it") as (con, _):
        con.execute("INSERT INTO erasures_revisions (id, revision, change, latest, reason, decisions, needles) "
                    "VALUES ('upstream-1', 1, 1, 1, 'asked upstream', '[]', '[]')")

    assert snapshots.last_erasure(depot) > before
