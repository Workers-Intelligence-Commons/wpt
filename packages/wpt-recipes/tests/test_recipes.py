"""The recipes and the task images, read and checked through Recipes."""

import json

import pytest

from wpt_recipes import InvalidRecipe, NotFound, Recipes, dump_json

MODEL = {"window_layers": 1, "sparse_groups": 1, "sparse_layers": 1, "d_model": 16, "attention_heads": 2,
         "attention_head_dim": 64, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
         "shared_width": 16, "engram_layers": [1], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
         "engram_table_size": 31, "residual_groups": 3, "loops": 0}
STEPS = [{"step": "read-text", "optimizer": {"name": "by-part", "settings": {}},
          "settings": {"stages": [{"block_size": 16, "steps": 1, "batch_size": 4}]}, "locked": []}]
DEPOT = "s3://depot-of-the-test"


@pytest.fixture
def recipes(tmp_path):
    root = tmp_path / "wpt"
    for folder in ("tokenizers", "models", "series"):
        (root / "data" / "recipes" / folder).mkdir(parents=True)
    return Recipes(root)


def write(recipes, kind, name, document):
    folder = {"tokenizer": "tokenizers", "model": "models"}[kind]
    (recipes.root / "data" / "recipes" / folder / f"{name}.json").write_text(dump_json(document), encoding="utf-8")


def tokenizer(depot=DEPOT, dataset="first"):
    return {"version": 1, "depot": depot, "dataset": dataset, "sample_bytes": 1000, "vocab_size": 300}


def model(tokenizer_name="t", depot=DEPOT, dataset="first"):
    return {"data": {"depot": depot, "dataset": dataset, "tokenizer": tokenizer_name, "total_tokens": 100},
            "model": MODEL, "seed": 1, "steps": STEPS, "drafter": {}}


def test_a_model_recipe_names_a_depot_a_dataset_there_its_tokenizer_and_its_tokens(recipes):
    write(recipes, "tokenizer", "t", tokenizer())
    write(recipes, "model", "m", model())
    assert recipes.model_recipe("m")["data"] == {"depot": DEPOT, "dataset": "first", "tokenizer": "t",
                                                 "total_tokens": 100}
    assert recipes.vocab_size(recipes.model_recipe("m")) == 300
    assert recipes.check() == []


def test_the_check_names_each_recipe_whose_depot_or_dataset_a_person_has_not_named(recipes):
    # an empty depot or dataset reads (training needs neither), and the check names the file to fill in
    write(recipes, "tokenizer", "t", tokenizer(depot=""))
    write(recipes, "model", "m", model(dataset=""))
    assert recipes.model_recipe("m")["data"]["dataset"] == ""
    assert recipes.check() == [
        "data/recipes/tokenizers/t.json: depot empty: a person names the depot's address and the name of its dataset",
        "data/recipes/models/m.json: data.dataset empty: a person names the depot's address and the name of its "
        "dataset"]


def test_a_recipe_of_the_earlier_shape_or_naming_no_tokenizer_recipe_is_refused(recipes):
    write(recipes, "tokenizer", "t", tokenizer())
    earlier = model()
    earlier["data"] = {"mixture": "classics", "tokenizer": "t", "total_tokens": 100, "held_out": 0.05}
    write(recipes, "model", "earlier", earlier)
    write(recipes, "model", "lost", model(tokenizer_name="gone"))
    with pytest.raises(InvalidRecipe, match="data/recipes/models/earlier.json: at data: Additional properties"):
        recipes.model_recipe("earlier")
    assert recipes.problems("model", "lost") == ["no tokenizer 'gone' (data/recipes/tokenizers/gone.json does not "
                                                 "exist)"]
    with pytest.raises(NotFound, match="no model 'nothing'"):
        recipes.model_recipe("nothing")


def test_a_lock_lists_the_texts_its_vocabulary_read_by_depot_and_id_and_a_wrong_one_writes_nothing(recipes):
    write(recipes, "tokenizer", "t", tokenizer())
    lock = {"recipe": tokenizer(), "texts": [{"depot": DEPOT, "id": "my-depot-12"}, {"depot": "other-depot", "id": "other-depot-12"}],
            "vocabulary_sha256": "b" * 64}
    assert recipes.generated("lock", "t") is None
    assert recipes.write_generated("lock", "t", lock) == "data/recipes/tokenizers/t.lock.json"
    assert recipes.generated("lock", "t") == lock
    with pytest.raises(InvalidRecipe, match="at texts/0: 'id' is a required property"):
        recipes.write_generated("lock", "t", {**lock, "texts": [{"depot": DEPOT}]})
    assert recipes.generated("lock", "t") == lock
    with pytest.raises(NotFound, match="no tokenizer 'other'"):  # a lock belongs to a tokenizer recipe
        recipes.write_generated("lock", "other", lock)
    assert recipes.names("lock") == ["t"] and recipes.names("tokenizer") == ["t"]


def test_the_check_reads_the_lock_and_licenses_of_each_image(recipes):
    # data/images/<name>/: its two written files match their schemas, and every image folder has its Dockerfile
    folder = recipes.root / "data" / "images" / "tiny"
    folder.mkdir(parents=True)
    (folder / "Dockerfile").write_text("ARG BASE=debian:trixie-slim\nFROM ${BASE}\n")
    lock = {"base": "debian@sha256:" + "a" * 64, "snapshot": "20261017T120000Z",
            "packages": {"bash": "5.2.37-2", "hello": "2.10-5"}}
    licenses = {"packages": {"bash": {"version": "5.2.37-2", "licenses": ["GPL-3+"]},
                             "hello": {"version": "2.10-5", "text": "Copyright: the authors\n"}}}
    (folder / "lock.json").write_text(dump_json(lock))
    (folder / "licenses.json").write_text(dump_json(licenses))
    assert recipes.check() == []
    assert recipes.generated("image-lock", "tiny") == lock

    (folder / "lock.json").write_text(dump_json({**lock, "snapshot": "2026-10-17"}))
    (recipes.root / "data" / "images" / "other").mkdir()
    assert recipes.check() == [
        "no image 'other': data/images/other/Dockerfile does not exist",
        "data/images/tiny/lock.json: at snapshot: '2026-10-17' does not match '^[0-9]{8}T[0-9]{6}Z$'"]


def test_an_image_s_pool_of_tools_is_read_after_the_check(recipes):
    # tools.json is the image's shared pool; a tool that takes the name of one every task has is refused
    folder = recipes.root / "data" / "images" / "tiny"
    folder.mkdir(parents=True)
    (folder / "Dockerfile").write_text("FROM debian:trixie-slim\n")
    fetch = {"name": "fetch", "description": "Fetch a page's text.", "program": "/usr/local/bin/fetch",
             "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
             "names": ["get_page"]}
    assert recipes.image_tools("tiny") == []
    (folder / "tools.json").write_text(json.dumps({"tools": [fetch]}))
    assert recipes.image_tools("tiny") == [fetch] and recipes.problems("image-tools", "tiny") == []
    (folder / "tools.json").write_text(json.dumps({"tools": [{**fetch, "name": "bash"}]}))
    with pytest.raises(InvalidRecipe, match="the tool 'bash' is declared twice, or takes the name of a tool every task"):
        recipes.image_tools("tiny")
    with pytest.raises(NotFound, match="no image 'gone'"):
        recipes.image_tools("gone")
