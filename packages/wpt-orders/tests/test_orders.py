"""The orders and the task images, read and checked through Orders."""

import json
import re

import pytest

from wpt_orders import InvalidOrder, NotFound, Orders, data_section, dump_json

MODEL = {"window_layers": 1, "sparse_groups": 1, "sparse_layers": 1, "d_model": 16, "attention_heads": 2,
         "attention_head_dim": 64, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
         "shared_width": 16, "engram_layers": [1], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
         "engram_table_size": 31, "residual_groups": 3, "loops": 0}
STEPS = [{"step": "read-text", "optimizer": {"name": "by-part", "settings": {}},
          "settings": {"stages": [{"block_size": 16, "steps": 1, "batch_size": 4}]}, "locked": []}]
DEPOT = "https://depot.example"
DIGEST, SHA256 = "d" * 64, "5" * 64


@pytest.fixture
def orders(tmp_path):
    root = tmp_path / "wpt"
    for folder in ("models", "series"):
        (root / "data" / "orders" / folder).mkdir(parents=True)
    return Orders(root)


def write(orders, kind, name, document):
    folder = {"model": "models"}[kind]
    (orders.root / "data" / "orders" / folder / f"{name}.json").write_text(dump_json(document), encoding="utf-8")


def model(depot=DEPOT, dataset="first"):
    """An order of the legacy dataset shape, written before snapshots."""
    return {"data": {"depot": depot, "dataset": dataset, "total_tokens": 100},
            "model": MODEL, "seed": 1, "steps": STEPS, "drafter": {}}


def snapshot_model(**data):
    """An order of the snapshot shape; data replaces fields of its data part."""
    return {"data": {"depot": DEPOT, "snapshot": "2026-11", "change": 1234, "digest": DIGEST,
                     "kinds": ["plain", "conversation"],
                     "shares": [{"tag": "fiction", "share": 0.3}, {"tag": "law", "share": 0.2}],
                     "total_tokens": 100, "vocabulary": {"sha256": SHA256, "size": 4096}, **data},
            "model": MODEL, "seed": 1, "steps": STEPS, "drafter": {}}


def test_a_model_order_names_a_snapshot_of_a_depot_its_kinds_shares_size_and_vocabulary(orders):
    write(orders, "model", "m", snapshot_model())
    write(orders, "model", "bare", snapshot_model(kinds=[], shares=[]))
    bare = snapshot_model()
    del bare["data"]["kinds"], bare["data"]["shares"]  # left out: every kind, the snapshot's own proportions
    write(orders, "model", "least", bare)
    # 0.33 + 0.56 + 0.11 is 1.0000000000000002 in floats: a share of the whole is still at most 1
    write(orders, "model", "whole", snapshot_model(shares=[{"tag": "a", "share": 0.33}, {"tag": "b", "share": 0.56},
                                                            {"tag": "c", "share": 0.11}]))
    assert orders.model_order("m")["data"] == snapshot_model()["data"]
    assert orders.data_part("m") == orders.model_order("m")["data"]
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
    ({"kinds": ["plain", "plain"]}, "data/kinds", "non-unique elements"),
    ({"dataset": "first"}, "data", "Additional properties are not allowed ('dataset' was unexpected)"),
])
def test_a_snapshot_data_part_of_the_wrong_shape_is_refused_naming_its_field(orders, data, place, message):
    write(orders, "model", "m", snapshot_model(**data))
    with pytest.raises(InvalidOrder, match=rf"data/orders/models/m.json: at {place}: .*{re.escape(message)}"):
        orders.model_order("m")
    assert orders.problems("model", "m")[0].startswith(f"data/orders/models/m.json: at {place}: ")


def test_a_vocabulary_is_named_by_its_sha256_or_learned_from_a_sample_of_a_number_of_bytes(orders):
    # the second form names no sha256: the vocabulary's delivery note gives it once it is learned
    write(orders, "model", "learned", snapshot_model(vocabulary={"size": 4096, "sample_bytes": 10**10}))
    assert orders.data_part("learned")["vocabulary"] == {"size": 4096, "sample_bytes": 10**10}
    assert orders.check() == []
    for wrong, message in (({"size": 4096}, "'sha256' is a required property"),
                           ({"size": 512, "sample_bytes": 10}, "less than the minimum of 513"),
                           ({"size": 4096, "sample_bytes": 10, "sha256": SHA256}, "Additional properties")):
        write(orders, "model", "m", snapshot_model(vocabulary=wrong))
        with pytest.raises(InvalidOrder, match=f"at data/vocabulary.*{re.escape(message)}"):
            orders.model_order("m")


def test_a_snapshot_data_part_without_its_vocabulary_is_refused(orders):
    missing = snapshot_model()
    del missing["data"]["vocabulary"]
    write(orders, "model", "m", missing)
    with pytest.raises(InvalidOrder, match="at data: 'vocabulary' is a required property"):
        orders.model_order("m")


def test_the_check_names_each_field_of_a_snapshot_data_part_a_person_has_not_filled_in(orders):
    # an empty depot, snapshot, digest or vocabulary reads, and the check names the file and fields to fill in
    write(orders, "model", "m", snapshot_model(snapshot="", digest="", vocabulary={"sha256": "", "size": 4096}))
    assert orders.model_order("m")["data"]["snapshot"] == ""
    assert orders.check() == [
        "data/orders/models/m.json: data.snapshot, data.digest, data.vocabulary.sha256 empty: a person names the "
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
    write(orders, "model", "m", snapshot_model(shares=shares))
    assert orders.model_order("m")["data"]["shares"] == shares  # the schema takes them: the check refuses them
    assert orders.problems("model", "m") == [f"data/orders/models/m.json: {message}"]


def test_a_legacy_model_order_names_a_depot_a_dataset_there_and_its_tokens(orders):
    write(orders, "model", "m", model())
    assert orders.model_order("m")["data"] == {"depot": DEPOT, "dataset": "first", "total_tokens": 100}
    assert orders.data_part("m") == orders.model_order("m")["data"]
    assert orders.check() == []


def test_the_check_names_each_legacy_order_whose_depot_or_dataset_a_person_has_not_named(orders):
    # an empty depot or dataset reads (training needs neither), and the check names the file to fill in
    write(orders, "model", "m", model(dataset=""))
    assert orders.model_order("m")["data"]["dataset"] == ""
    assert orders.check() == [
        "data/orders/models/m.json: data.dataset empty: a person names the depot's public address and the id or "
        "name of its dataset"]


def test_a_order_of_an_earlier_shape_is_refused(orders):
    # a legacy order with a field its shape lacks is refused naming that field, not as a whole value
    earlier = model()
    earlier["data"] = {"mixture": "classics", "tokenizer": "t", "total_tokens": 100, "held_out": 0.05}
    write(orders, "model", "earlier", earlier)
    named = model()
    named["data"]["tokenizer"] = "t"  # the tokenizer is the dataset's own: an order names none
    write(orders, "model", "named", named)
    with pytest.raises(InvalidOrder, match="data/orders/models/earlier.json: at data: Additional properties"):
        orders.model_order("earlier")
    with pytest.raises(InvalidOrder, match="data/orders/models/named.json: at data: Additional properties"):
        orders.model_order("named")
    with pytest.raises(NotFound, match="no model 'nothing'"):
        orders.model_order("nothing")


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


def test_a_order_handed_over_as_json_is_read_by_its_data_section_alone():
    # a depot's build reads an order a person hands over, never from data/: only its data section counts
    assert data_section(snapshot_model(), "m.json") == ("model", snapshot_model()["data"])
    assert data_section({"data": snapshot_model()["data"]}, "m.json") == ("model", snapshot_model()["data"])
    assert data_section({**snapshot_model(), "model": "anything"}, "m.json")[0] == "model"
    assert data_section(model(), "old.json") == ("model", model()["data"])  # the legacy shape reads; no build takes it
    for document, message in (([], "no data section"), ({"model": MODEL}, "no data section"),
                              (snapshot_model(change=-1), "at data/change: "),
                              (snapshot_model(shares=[{"tag": "a", "share": 0.7}, {"tag": "b", "share": 0.7}]),
                               "shares sum to 1.4"),
                              (snapshot_model(digest=""), "data.digest empty")):
        with pytest.raises(InvalidOrder, match=f"m.json: .*{re.escape(message)}"):
            data_section(document, "m.json")
