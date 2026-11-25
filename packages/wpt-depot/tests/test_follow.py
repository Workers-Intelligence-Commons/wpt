import pytest

from wpt_depot import About, NewText, clone, follow, pull, snapshots
from wpt_depot.errors import OtherFormat, UpstreamUnreachable
from wpt_depot.schema import FORMAT


def admitted_texts(depot, me, label: str, count: int) -> list[str]:
    texts = depot.add([NewText(text=f"{label} {n}", licenses=("CC0-1.0",)) for n in range(count)], me)
    depot.decide("admitted", texts, me)
    return texts


def stored(depot) -> tuple[list, list]:
    """Every snapshot row and every set, bitmaps as their bytes."""
    con = depot.db.reader()
    return (con.execute("SELECT * FROM snapshots ORDER BY id").fetchall(),
            con.execute("SELECT * FROM snapshot_sets ORDER BY snapshot, part, owner").fetchall())


def read_whole(depot, snapshot_id: str) -> list[tuple]:
    """(id, revision, title) of every text the snapshot reads."""
    found, after = [], None
    while True:
        page = depot.snapshot_texts(snapshot_id, after=after, limit=3)
        found += [(one["id"], one["revision"], one["title"]) for one in page["texts"]]
        after = page["after"]
        if after is None:
            return found


def follower_of(serve, depot, tmp_path, name: str):
    return clone(serve(depot), str(tmp_path / f"{name}.sqlite"), About(name, "Grace Hopper", "grace@example.org"))


def test_snapshots_and_their_bitmaps_travel_by_clone_and_pull_and_read_the_same_on_the_follower(
        make_depot, me, serve, tmp_path):
    # a recipe names the server's snapshot: a clone must hold the very same row and sets, so it reads the same texts
    upstream = make_depot("up")
    texts = admitted_texts(upstream, me, "upstream", 5)
    first = upstream.create_snapshot()
    follower = follower_of(serve, upstream, tmp_path, "follower")

    assert stored(follower) == stored(upstream)
    assert snapshots.sets(follower, first["id"])  # read back and checked against its digest
    assert read_whole(follower, first["id"]) == read_whole(upstream, first["id"])

    upstream.edit_credits(texts[0], {"title": "Edited"}, me)
    upstream.decide("refused", texts[1:2], me)
    later = upstream.create_snapshot()
    pull(follower)

    assert stored(follower) == stored(upstream)
    assert follower.snapshot(later["name"], owner="up")["digest"] == later["digest"]
    assert read_whole(follower, later["id"]) == read_whole(upstream, later["id"])
    assert read_whole(follower, first["id"]) == read_whole(upstream, first["id"])  # still at its own bound
    assert follower.snapshot(later["name"], owner="up")["summary"] == later["summary"]


def test_a_snapshot_whose_sets_do_not_match_its_digest_is_refused_and_the_pull_writes_nothing(
        make_depot, me, serve, tmp_path):
    upstream = make_depot("up")
    admitted_texts(upstream, me, "upstream", 4)
    follower = follower_of(serve, upstream, tmp_path, "follower")
    made = upstream.create_snapshot()
    with upstream.db.write("damage") as (con, _):  # as a damaged file or a broken answer would send it
        con.execute("UPDATE snapshot_sets SET bitmap = ? WHERE snapshot = ?",
                    (snapshots.BitMap([1, 2]).serialize(), made["id"]))
    change = follower.change()

    with pytest.raises(UpstreamUnreachable, match="digest"):
        pull(follower)

    assert stored(follower) == ([], []) and follower.change() == change
    with pytest.raises(UpstreamUnreachable, match="digest"):
        follower_of(serve, upstream, tmp_path, "second")
    assert not (tmp_path / "second.sqlite").exists() and not (tmp_path / "second.sqlite.cloning").exists()


def test_bytes_that_are_not_base64_are_refused_and_never_written(make_depot):
    depot = make_depot()

    with pytest.raises(UpstreamUnreachable, match="base64"):
        with depot.db.write("pull") as (con, _):
            follow._put(con, "snapshot_sets", [{"snapshot": "up-1", "part": 0, "owner": "up", "bound": 1,
                                                "bitmap": "not base64!", "texts": 1}])

    assert depot.db.query("SELECT * FROM snapshot_sets") == []


def test_a_clone_of_a_clone_holds_both_depots_snapshots_and_reads_them_alike(make_depot, me, serve, tmp_path):
    # the middle depot holds the top one's snapshot in the top one's numbering, and one of its own over both owners
    top = make_depot("up")
    theirs = admitted_texts(top, me, "top", 3)
    top_made = top.create_snapshot()
    middle = follower_of(serve, top, tmp_path, "middle")
    admitted_texts(middle, me, "middle", 2)
    middle_made = middle.create_snapshot()
    bottom = follower_of(serve, middle, tmp_path, "bottom")

    assert stored(bottom) == stored(middle)
    for made, owner in ((top_made, "up"), (middle_made, "middle")):
        assert bottom.snapshot(made["name"], owner=owner)["id"] == made["id"]
        assert read_whole(bottom, made["id"]) == read_whole(middle, made["id"])
    assert {owner for _, owner, _, _ in snapshots.sets(bottom, middle_made["id"])} == {"up", "middle"}

    # what the top depot does later reaches the bottom through the middle
    top.edit_credits(theirs[0], {"title": "Later"}, me)
    top_later = top.create_snapshot()
    pull(middle)
    pull(bottom)

    assert stored(bottom) == stored(middle)
    assert read_whole(bottom, top_later["id"]) == read_whole(top, top_later["id"])
    assert read_whole(bottom, top_made["id"]) == read_whole(top, top_made["id"])


def test_a_clone_s_own_snapshot_named_like_the_upstream_s_does_not_break_a_pull(make_depot, me, serve, tmp_path):
    upstream = make_depot("up")
    admitted_texts(upstream, me, "upstream", 3)
    follower = follower_of(serve, upstream, tmp_path, "follower")
    admitted_texts(follower, me, "own", 2)
    ours = follower.create_snapshot()
    theirs = upstream.create_snapshot()
    with upstream.db.write("same name") as (con, _):  # made in the same second as the clone's: named alike
        con.execute("UPDATE snapshots SET name = ? WHERE id = ?", (ours["name"], theirs["id"]))

    pull(follower)
    pull(follower)  # a second pull brings nothing new and keeps both

    assert follower.snapshot(ours["name"])["id"] == ours["id"]
    assert follower.snapshot(ours["name"], owner="up")["id"] == theirs["id"]
    assert sorted(one["id"] for one in follower.snapshots()) == sorted([ours["id"], theirs["id"]])
    assert read_whole(follower, theirs["id"]) == read_whole(upstream, theirs["id"])


@pytest.mark.parametrize("way", ["clone", "pull"])
def test_a_clone_refuses_an_upstream_of_an_older_format_and_says_to_upgrade_it_first(
        make_depot, me, serve, tmp_path, monkeypatch, way):
    upstream = make_depot("up")
    admitted_texts(upstream, me, "upstream", 2)
    address = serve(upstream)
    follower = clone(address, str(tmp_path / "follower.sqlite"), About("follower", "G", "g@example.org")) \
        if way == "pull" else None
    head = follow.head
    monkeypatch.setattr(follow, "head", lambda depot: {**head(depot), "format": FORMAT - 1})

    with pytest.raises(OtherFormat, match="upgrade the upstream first"):
        if way == "pull":
            pull(follower)
        else:
            clone(address, str(tmp_path / "other.sqlite"), About("other", "G", "g@example.org"))
