"""The orders and the task images, read and checked through Orders: each
order here a family of one size, s, with every setting common, so its
size's model order is the order it writes (the tables of several sizes:
test_family.py)."""

import json
import re

import pytest

from wpt_orders import InvalidOrder, NotFound, Orders, dump_json
from wpt_orders.orders import HELD_OUT_MIN_TOKENS

MODEL = {"window_layers": 1, "sparse_groups": 1, "sparse_layers": 1, "d_model": 16, "attention_heads": 2,
         "attention_head_dim": 64, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
         "shared_width": 16, "engram_layers": [1], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
         "engram_table_size": 31, "residual_groups": 3, "loops": 0}
STEPS = [{"step": "read-text", "optimizer": {"name": "by-part", "settings": {}},
          "settings": {"stages": [{"block_size": 16, "share": 1}], "tokens_per_update": 64}, "locked": []}]
DEPOT = "https://depot.example"
DIGEST, SHA256 = "d" * 64, "5" * 64


@pytest.fixture
def orders(tmp_path):
    root = tmp_path / "wpt"
    (root / "data" / "orders").mkdir(parents=True)
    return Orders(root)


def write(orders, name, document):
    """The family `name` of one size, s, whose order is the document."""
    family = {**document, "sizes": {"columns": ["size"], "rows": [["s"]]}}
    (orders.root / "data" / "orders" / f"{name}.json").write_text(dump_json(family), encoding="utf-8")


def snapshot_model(**data):
    """An order of the snapshot shape; data replaces fields of its data part."""
    return {"data": {"depot": DEPOT, "snapshot": "2026-11", "change": 1234, "digest": DIGEST,
                     "roles": ["held_out", "pretrain", "finetune"],
                     "shares": [{"tag": "fiction", "share": 0.3}, {"tag": "law", "share": 0.2}],
                     "total_tokens": 10**9, "finetune_tokens": 40, "vocabulary": {"sha256": SHA256, "size": 4096},
                     **data},
            "model": MODEL, "seed": 1, "steps": STEPS, "drafter": {}}


def test_a_model_order_names_a_snapshot_of_a_depot_its_roles_shares_sizes_and_vocabulary(orders):
    write(orders, "m", snapshot_model())
    pretrain = snapshot_model(roles=["pretrain"], shares=[])
    del pretrain["data"]["finetune_tokens"]
    write(orders, "bare", pretrain)
    bare = snapshot_model()
    # left out: the roles held_out and pretrain, the snapshot's own proportions, every conversation and task
    del bare["data"]["roles"], bare["data"]["shares"], bare["data"]["finetune_tokens"]
    write(orders, "least", bare)
    # 0.33 + 0.56 + 0.11 is 1.0000000000000002 in floats: a share of the whole is still at most 1
    write(orders, "whole", snapshot_model(shares=[{"tag": "a", "share": 0.33}, {"tag": "b", "share": 0.56},
                                                            {"tag": "c", "share": 0.11}]))
    assert orders.model_order("m:s")["data"] == snapshot_model()["data"]
    assert orders.model_order("m:s")["steps"][0]["settings"]["prefix_tokens"] == 10**9  # its total_tokens
    assert orders.data_part("m:s") == orders.model_order("m:s")["data"]
    assert orders.check() == []


@pytest.mark.parametrize(("data", "place", "message"), [
    ({"digest": "D" * 64}, "data/digest", "does not match"),
    ({"digest": "d" * 63}, "data/digest", "does not match"),
    ({"vocabulary": {"sha256": "x" * 64, "size": 4096}}, "data/vocabulary/sha256", "does not match"),
    ({"vocabulary": {"sha256": SHA256}}, "data/vocabulary", "'size' is a required property"),
    ({"snapshot": "two words"}, "data/snapshot", "does not match"),
    ({"change": -1}, "data/change", "less than the minimum of 0"),
    ({"shares": [{"tag": "law", "share": 1.5}]}, "data/shares/0/share", "greater than the maximum of 1"),
    ({"shares": [{"tag": "law"}]}, "data/shares/0", "'share' is a required property"),
    ({"roles": ["pretrain", "pretrain"]}, "data/roles", "non-unique elements"),
    ({"roles": ["held_out", "finetune"]}, "data/roles", "does not contain items matching"),
    ({"roles": ["pretrain", "test"]}, "data/roles/1", "'test' is not one of"),
    ({"kinds": ["plain"]}, "data", "Additional properties are not allowed ('kinds' was unexpected)"),
    ({"dataset": "first"}, "data", "names a dataset; an order names a snapshot of a depot"),
])
def test_a_snapshot_data_part_of_the_wrong_shape_is_refused_naming_its_field(orders, data, place, message):
    write(orders, "m", snapshot_model(**data))
    with pytest.raises(InvalidOrder, match=rf"data/orders/m.json, size s: at {place}: .*{re.escape(message)}"):
        orders.model_order("m:s")
    assert orders.problems("family", "m")[0].startswith(f"data/orders/m.json, size s: at {place}: ")


def test_a_vocabulary_is_named_by_its_sha256_or_learned_from_a_sample_of_a_number_of_bytes(orders):
    # the second form names no sha256: the vocabulary's delivery note gives it once it is learned
    write(orders, "learned", snapshot_model(vocabulary={"size": 4096, "sample_bytes": 10**10}))
    assert orders.data_part("learned:s")["vocabulary"] == {"size": 4096, "sample_bytes": 10**10}
    assert orders.check() == []
    for wrong, message in (({"size": 4096}, "'sha256' is a required property"),
                           ({"size": 512, "sample_bytes": 10}, "less than the minimum of 513"),
                           ({"size": 4096, "sample_bytes": 10, "sha256": SHA256}, "Additional properties")):
        write(orders, "m", snapshot_model(vocabulary=wrong))
        with pytest.raises(InvalidOrder, match=f"at data/vocabulary.*{re.escape(message)}"):
            orders.model_order("m:s")


def test_a_snapshot_data_part_without_its_vocabulary_is_refused(orders):
    missing = snapshot_model()
    del missing["data"]["vocabulary"]
    write(orders, "m", missing)
    with pytest.raises(InvalidOrder, match="at data: 'vocabulary' is a required property"):
        orders.model_order("m:s")


def test_the_check_names_each_field_of_a_snapshot_data_part_a_person_has_not_filled_in(orders):
    # an empty depot, snapshot, digest or vocabulary reads, and the check names the file and fields to fill in
    write(orders, "m", snapshot_model(snapshot="", digest="", vocabulary={"sha256": "", "size": 4096}))
    assert orders.model_order("m:s")["data"]["snapshot"] == ""
    assert orders.check() == [
        "data/orders/m.json, size s: data.snapshot, data.digest, data.vocabulary.sha256 empty: a person names the "
        "depot's public address and its snapshot, with the snapshot's digest and, for a vocabulary named by its "
        "sha256, that sha256"]


@pytest.mark.parametrize(("shares", "message"), [
    ([{"tag": "law", "share": 0.2}, {"tag": "fiction", "share": 0.1}, {"tag": "law", "share": 0.3}],
     "data.shares name a tag more than once: law"),
    ([{"tag": "law", "share": 0.6}, {"tag": "fiction", "share": 0.5}],
     "data.shares sum to 1.1, above 1: each is a part of the plain text"),
    ([{"tag": "law", "share": 0}], "data.shares are all 0: give one a share above 0, or give no shares"),
])
def test_the_check_refuses_shares_a_corpus_cannot_be_built_with(orders, shares, message):
    write(orders, "m", snapshot_model(shares=shares))
    assert orders.model_order("m:s")["data"]["shares"] == shares  # the schema takes them: the check refuses them
    assert orders.problems("family", "m") == [f"data/orders/m.json, size s: {message}"]


def test_the_check_refuses_a_size_of_the_finetune_portion_its_roles_leave_out(orders):
    write(orders, "m", snapshot_model(roles=["held_out", "pretrain"]))
    assert orders.model_order("m:s")["data"]["finetune_tokens"] == 40  # the schema takes it: the check refuses it
    assert orders.problems("family", "m") == [
        "data/orders/m.json, size s: data.finetune_tokens sizes the finetune portion, which data.roles leave out: add "
        "finetune to the roles, or give no finetune_tokens"]


@pytest.mark.parametrize("named", [None, ["held_out", "pretrain"]])
def test_the_check_refuses_a_held_out_portion_in_a_corpus_too_small_for_it(orders, named):
    # the role held_out, named or left out, needs a total_tokens of at least HELD_OUT_MIN_TOKENS
    def sized(total, roles=named):
        document = snapshot_model(roles=roles, total_tokens=total)
        del document["data"]["finetune_tokens"]
        if roles is None:
            del document["data"]["roles"]
        return document

    write(orders, "m", sized(HELD_OUT_MIN_TOKENS - 1))
    assert orders.problems("family", "m") == [
        f"data/orders/m.json, size s: data.total_tokens {HELD_OUT_MIN_TOKENS - 1:,} is too small for the held-out "
        f"portion, which needs at least {HELD_OUT_MIN_TOKENS:,}: name data.roles without held_out, such as "
        '["pretrain"], or ask for more tokens']
    write(orders, "m", sized(HELD_OUT_MIN_TOKENS - 1, ["pretrain"]))
    write(orders, "enough", sized(HELD_OUT_MIN_TOKENS))
    assert orders.check() == []


def test_an_order_that_names_a_dataset_is_refused_saying_an_order_names_a_snapshot(orders):
    # the data part of the orders written before snapshots: a dataset of a depot, which no training reads
    write(orders, "m", {**snapshot_model(), "data": {"depot": DEPOT, "dataset": "first", "total_tokens": 100}})
    with pytest.raises(InvalidOrder, match="data/orders/m.json, size s: at data: names a dataset; an order names a "
                                           "snapshot of a depot, with its change and digest"):
        orders.model_order("m:s")
    assert orders.check() == [
        "data/orders/m.json, size s: at data: names a dataset; an order names a snapshot of a depot, with its change "
        "and digest (docs/data-selection.md, \"The data section of a model order\")"]


def test_a_order_of_an_earlier_shape_is_refused(orders):
    # an order with a field its shape lacks is refused naming that field, not as a whole value
    earlier = snapshot_model()
    earlier["data"] = {"mixture": "classics", "tokenizer": "t", "total_tokens": 100, "held_out": 0.05}
    write(orders, "earlier", earlier)
    named = snapshot_model()
    named["data"]["tokenizer"] = "t"  # the vocabulary is named by itself: an order names no tokenizer
    write(orders, "named", named)
    with pytest.raises(InvalidOrder, match="data/orders/earlier.json, size s: at data: "):
        orders.model_order("earlier:s")
    with pytest.raises(InvalidOrder, match="data/orders/named.json, size s: at data: Additional properties"):
        orders.model_order("named:s")
    with pytest.raises(NotFound, match="no family 'nothing'"):
        orders.model_order("nothing:s")
    with pytest.raises(NotFound, match="no size 'x' in the family named; its sizes are s"):
        orders.model_order("named:x")
    # an order is always a size of a family: the family's name alone names none
    with pytest.raises(NotFound, match="'named' names no size: an order is FAMILY:SIZE"):
        orders.model_order("named")
    with pytest.raises(NotFound, match="'named:' names no size.*its sizes are s"):
        orders.data_part("named:")


def test_the_check_reads_the_lock_and_licenses_of_each_image(orders):
    # data/images/<name>/: its two written files match their schemas, and every image folder has its Dockerfile
    folder = orders.root / "data" / "images" / "tiny"
    folder.mkdir(parents=True)
    (folder / "Dockerfile").write_text("ARG BASE=debian:trixie-slim\nFROM ${BASE}\n")
    lock = {"base": "debian@sha256:" + "a" * 64, "snapshot": "20261017T120000Z",
            "packages": {"bash": "5.2.37-2", "hello": "2.10-5"}}
    licenses = {"packages": {"bash": {"version": "5.2.37-2", "licenses": ["GPL-3+"]},
                             "hello": {"version": "2.10-5", "text": "Copyright: the authors\n"}}}
    (folder / "lock.json").write_text(dump_json(lock))
    (folder / "licenses.json").write_text(dump_json(licenses))
    assert orders.check() == []
    assert orders.generated("image-lock", "tiny") == lock

    (folder / "lock.json").write_text(dump_json({**lock, "snapshot": "2026-10-17"}))
    (orders.root / "data" / "images" / "other").mkdir()
    assert orders.check() == [
        "no image 'other': data/images/other/Dockerfile does not exist",
        "data/images/tiny/lock.json: at snapshot: '2026-10-17' does not match '^[0-9]{8}T[0-9]{6}Z$'"]


def test_an_image_s_pool_of_tools_is_read_after_the_check(orders):
    # tools.json is the image's shared pool; a tool that takes the name of one every task has is refused
    folder = orders.root / "data" / "images" / "tiny"
    folder.mkdir(parents=True)
    (folder / "Dockerfile").write_text("FROM debian:trixie-slim\n")
    fetch = {"name": "fetch", "description": "Fetch a page's text.", "program": "/usr/local/bin/fetch",
             "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
             "names": ["get_page"]}
    assert orders.image_tools("tiny") == []
    (folder / "tools.json").write_text(json.dumps({"tools": [fetch]}))
    assert orders.image_tools("tiny") == [fetch] and orders.problems("image-tools", "tiny") == []
    (folder / "tools.json").write_text(json.dumps({"tools": [{**fetch, "name": "bash"}]}))
    with pytest.raises(InvalidOrder, match="the tool 'bash' is declared twice, or takes the name of a tool every task"):
        orders.image_tools("tiny")
    with pytest.raises(NotFound, match="no image 'gone'"):
        orders.image_tools("gone")
