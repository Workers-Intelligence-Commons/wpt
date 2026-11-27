"""Model families (docs/model.md, "Model families"): what the sizes share
written once, a table of what varies, each size resolved to its model
order, the rules of the table, the corpora the sizes share, the other
runs of a size and the compute a size's numbers count."""

import copy
import json
from pathlib import Path

import pytest

from wpt_orders import InvalidOrder, NotFound, Orders, counts, dump_json, family

ROOT = Path(__file__).resolve().parents[3]
MODEL = {"window_layers": 1, "sparse_groups": 1, "sparse_layers": 1, "attention_heads": 2, "attention_head_dim": 64,
         "n_experts": 4, "top_k": 2, "engram_layers": [1], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
         "engram_table_size": 31, "residual_groups": 3, "loops": 0}
DATA = {"depot": "https://depot.example", "snapshot": "s", "change": 1, "digest": "d" * 64, "roles": ["pretrain"],
        "vocabulary": {"size": 4096, "sample_bytes": 10**9}}
STEP = {"step": "read-text", "optimizer": {"name": "by-part", "settings": {"lr": 0.004}},
        "settings": {"stages": [{"block_size": 16, "share": 1}, {"block_size": 32, "share": 1}],
                     "tokens_per_update": 64, "mtp_switch": 0.8}, "locked": []}


def table(columns, *rows):
    return {"columns": ["size", *columns], "rows": [list(row) for row in rows]}


def three():
    """A family of three widths, each reading its own tokens; its expert widths a column too."""
    return {"data": DATA, "model": MODEL, "seed": 1, "steps": [STEP], "drafter": {},
            "sizes": table(["model.d_model", "model.expert_width", "model.latent_width", "model.shared_width",
                            "data.total_tokens"],
                           ("16", 16, 8, 8, 16, 1000), ("32", 32, 16, 16, 32, 3000), ("24", 24, 12, 12, 24, 2000))}


@pytest.fixture
def orders(tmp_path):
    (tmp_path / "data" / "orders").mkdir(parents=True)
    return Orders(tmp_path)


def write(orders, name, document):
    (orders.root / "data" / "orders" / f"{name}.json").write_text(dump_json(document), encoding="utf-8")


def test_a_size_resolves_to_the_common_settings_with_its_own_values_in_place(orders):
    write(orders, "f", three())
    order = orders.model_order("f:32")
    assert order["model"] == {**MODEL, "d_model": 32, "expert_width": 16, "latent_width": 16, "shared_width": 32}
    assert {key: order[key] for key in ("seed", "drafter")} == {"seed": 1, "drafter": {}}
    assert order["steps"][0]["optimizer"] == STEP["optimizer"]
    assert orders.check() == []


def test_sizes_of_one_data_part_read_the_start_of_one_corpus_of_the_most_tokens_any_reads(orders):
    # each size's order asks for the same corpus, of 3,000 tokens, and its step reads its own tokens from the start
    write(orders, "f", three())
    for size, own in (("16", 1000), ("24", 2000), ("32", 3000)):
        order = orders.model_order(f"f:{size}")
        assert order["data"] == {**DATA, "total_tokens": 3000}
        assert order["steps"][0]["settings"]["prefix_tokens"] == own
        assert orders.data_part(f"f:{size}") == order["data"]


def test_sizes_of_another_vocabulary_read_a_corpus_of_their_own(orders):
    # the vocabulary is a column: the two sizes of the 8,192-token vocabulary share a corpus, the other reads its own
    found = three()
    found["data"] = {key: value for key, value in DATA.items() if key != "vocabulary"}
    found["sizes"]["columns"].append("data.vocabulary")
    small, large = {"size": 4096, "sample_bytes": 10**9}, {"size": 8192, "sample_bytes": 10**9}
    for row, vocabulary in zip(found["sizes"]["rows"], (small, large, large)):
        row.append(vocabulary)
    write(orders, "f", found)
    assert orders.data_part("f:16") == {**DATA, "vocabulary": small, "total_tokens": 1000}
    assert orders.data_part("f:24") == orders.data_part("f:32") == {**DATA, "vocabulary": large, "total_tokens": 3000}
    assert orders.model_order("f:24")["steps"][0]["settings"]["prefix_tokens"] == 2000


@pytest.mark.parametrize(("change", "message"), [
    # a setting in two places
    (lambda f: f["model"].update(d_model=16),
     "model.d_model is written both as a common setting and as a column: write it in one place"),
    (lambda f: f["data"].update(total_tokens=10),
     "data.total_tokens is written both as a common setting and as a column"),
    (lambda f: f["sizes"]["columns"].append("data.vocabulary.size") or [row.append(4096) for row in f["sizes"]["rows"]],
     "data.vocabulary.size is written both as a common setting and as a column"),
    (lambda f: f["sizes"]["columns"].append("model.d_model") or [row.append(8) for row in f["sizes"]["rows"]],
     "sizes.columns names model.d_model more than once"),
    (lambda f: f["sizes"]["columns"].append("steps.1.settings.stages.0") or [row.append(1) for row in f["sizes"]["rows"]],
     "is written both as a common setting and as a column"),
    # a column a size leaves empty
    (lambda f: f["sizes"]["rows"][1].__setitem__(3, None),
     "size 32 leaves the column model.latent_width empty: every size fills every column"),
    (lambda f: f["sizes"]["rows"][2].pop(),
     "size 24: its row has 4 values for 5 columns: every size fills every column"),
    # the tokens a size reads are its data.total_tokens alone
    (lambda f: f["steps"][0]["settings"].update(prefix_tokens=5),
     "steps.1.settings.prefix_tokens: a size reads its data.total_tokens"),
    # a path that names nothing of an order, a step it does not have, a table without its names
    (lambda f: f["sizes"]["columns"].__setitem__(1, "width"), "column 'width': a path starts with one of"),
    (lambda f: f["sizes"]["columns"].__setitem__(1, "steps.2.settings.readings"),
     "names step 2, and the family writes 1 steps"),
    (lambda f: f["sizes"]["columns"].__setitem__(0, "name"), "sizes.columns: the first column is 'size'"),
    (lambda f: f["sizes"]["rows"][2].__setitem__(0, "32"), "the sizes 32 are named more than once"),
])
def test_the_check_refuses_a_table_that_writes_a_setting_twice_or_leaves_a_value_empty(orders, change, message):
    found = copy.deepcopy(three())
    change(found)
    write(orders, "f", found)
    with pytest.raises(InvalidOrder, match=f"data/orders/f.json: .*{message}"):
        orders.model_order("f:16")
    [problem] = orders.check()
    assert problem.startswith("data/orders/f.json: ") and message.split(":")[0] in problem


def test_a_size_whose_order_the_model_order_s_schema_refuses_is_named_with_its_size(orders):
    found = three()
    found["sizes"]["rows"][1][1] = 0  # a width of 0
    write(orders, "f", found)
    with pytest.raises(InvalidOrder, match=r"data/orders/f.json, size 32: at model/d_model: 0 is less than"):
        orders.model_order("f:32")
    assert orders.model_order("f:16")["model"]["d_model"] == 16  # the other sizes still resolve


def test_a_training_setting_varies_as_a_column_like_any_other(orders):
    # the learning rate and the readings of the one step, by size
    found = three()
    found["steps"] = [{**STEP, "optimizer": {"name": "by-part", "settings": {}}}]
    found["sizes"]["columns"] += ["steps.1.optimizer.settings.lr", "steps.1.settings.readings"]
    for row, lr, readings in zip(found["sizes"]["rows"], (0.004, 0.003, 0.0035), (1, 2, 3)):
        row += [lr, readings]
    write(orders, "f", found)
    step = orders.model_order("f:24")["steps"][0]
    assert step["optimizer"]["settings"] == {"lr": 0.0035} and step["settings"]["readings"] == 3


def test_a_size_s_branches_go_on_from_one_trunk_cut_at_their_starts(orders):
    # the size makes 1,000 updates (its step's, as its environment counts them); its branches make 250, 500 and
    # 1,000, each its decay after 80% of them: 200, 400 and 800
    write(orders, "f", {**three(), "branches": [0.25, 0.5, 1]})
    updates = lambda entry: 1000  # noqa: E731 a size's updates, which wpt_training's stages count
    [own] = orders.model_order("f:24")["steps"]
    assert family.branch_numbers({"branches": [0.25, 0.5, 1]}, 1000) == [
        {"run": "branch-0.25", "updates": 250, "start": 200}, {"run": "branch-0.5", "updates": 500, "start": 400},
        {"run": "branch-1", "updates": 1000, "start": 800}]

    def trunk(end, continues):
        step = copy.deepcopy(own)
        step["settings"].update(last_update=end, mtp_switch=1.0)
        step["optimizer"]["settings"].update(first_drop=1.0, second_drop=1.0)
        return {**step, "continues": True} if continues else step

    half = orders.model_order("f:24:branch-0.5", updates)
    assert half["steps"] == [trunk(200, False), trunk(400, True),
                             {**own, "settings": {**own["settings"], "last_update": 500}, "continues": True}]
    full = orders.model_order("f:24:branch-1", updates)
    assert full["steps"][:3] == [trunk(200, False), trunk(400, True), trunk(800, True)]  # the trunk shared
    assert full["steps"][3] == {**own, "settings": {**own["settings"], "last_update": 1000}, "continues": True}
    assert {key: value for key, value in full.items() if key != "steps"} == \
        {key: value for key, value in orders.model_order("f:24").items() if key != "steps"}


def test_the_learning_rate_check_runs_the_smallest_size_at_multiples_of_its_learning_rate(orders):
    write(orders, "f", {**three(), "branches": [1], "learning_rate_check": [0.5, 2]})
    assert family.smallest(orders.family("f")) == "16"
    assert family.runs(orders.family("f"), "16") == ["branch-1", "lr-0.5", "lr-2"]
    assert family.runs(orders.family("f"), "32") == ["branch-1"]
    own = orders.model_order("f:16")
    for run, rate in (("lr-0.5", 0.002), ("lr-2", 0.008)):
        changed = copy.deepcopy(own)
        changed["steps"][0]["optimizer"]["settings"]["lr"] = rate
        assert orders.model_order(f"f:16:{run}") == changed  # everything else is the size's own run
    with pytest.raises(NotFound, match="no run 'lr-2' of the size 32; its runs are branch-1"):
        orders.model_order("f:32:lr-2")


def test_two_branches_of_the_same_updates_are_refused():
    with pytest.raises(ValueError, match="two branches have the same number of updates of the size's 3"):
        family.branch_numbers({"branches": [0.5, 0.6]}, 3)


@pytest.mark.parametrize("change", [lambda step: step["settings"].pop("mtp_switch"),
                                    lambda step: step["settings"].update(mtp_switch=0.676),
                                    lambda step: step["optimizer"]["settings"].update(first_drop=0.9)])
def test_a_family_with_branches_switches_and_drops_where_its_branches_decay(change):
    # a branch's trunk keeps the multi-token prediction's early weight and the peak rate to 80% of the branch's
    # updates: the size's own run is its longest branch only if it changes both there too
    found = copy.deepcopy({**three(), "branches": [1]})
    change(found["steps"][0])
    with pytest.raises(ValueError, match="size 16, step 1: a branch decays after 0.8 of its updates"):
        family.check(found)
    family.check({key: value for key, value in found.items() if key != "branches"})  # without branches: any


def test_the_compute_of_a_token_counts_every_pass_of_the_loop_and_every_read_of_the_output_head():
    # pocket's widest size, in 32,768-token windows
    model = {**MODEL, "d_model": 256, "attention_heads": 4, "expert_width": 128, "latent_width": 128,
             "shared_width": 256, "window_layers": 2, "sparse_layers": 5, "residual_groups": 22}
    vocab, window, head = 16384, 32768, 6 * 16384 * 256
    one = counts.compute_per_token(model, vocab, window)
    # one more pass of every layer costs the layers' work once more, and nothing else
    layers = counts.compute_per_token({**model, "loops": 1}, vocab, window) - one
    assert 0 < layers < one
    # a loss on both passes reads the output head once more: two sizes compared at equal compute pay for it
    weighed = counts.compute_per_token({**model, "loops": 1, "loop_losses": [1, 1]}, vocab, window)
    assert weighed - (one + layers) == head
    # a weight of 0 reads nothing through the head
    assert counts.compute_per_token({**model, "loops": 1, "loop_losses": [0, 1]}, vocab, window) == one + layers
    # with the loop over the first layers, the layers after it run again for the earlier pass the loss weighs
    first = {**model, "loops": 1, "loop_layers": 4}
    coda = counts.compute_per_token({**first, "loop_losses": [1, 1]}, vocab, window) \
        - counts.compute_per_token(first, vocab, window) - head
    assert 0 < coda < layers
    # the head, read for the next token and by the multi-token prediction block, is 2 x 6 x vocab x width of it
    assert one - counts.compute_per_token(model, 0, window) == 2 * head


def test_the_families_of_the_repository_pass_the_check_and_name_their_sizes_as_they_say():
    # pocket's sizes are its widths, read from one corpus; coherent and test have one size each
    repo = Orders(ROOT)
    assert repo.check() == []
    assert repo.names("family") == ["coherent", "pocket", "test"]
    pocket = repo.family("pocket")
    assert family.sizes(pocket) == ["64", "96", "128", "192", "256"]
    for size in family.sizes(pocket):
        order = repo.model_order(f"pocket:{size}")
        assert order["model"]["d_model"] == int(size) and order["data"]["total_tokens"] == 1179648000
    assert json.dumps(repo.model_order("pocket:256")["steps"]) == json.dumps(
        [{**pocket["steps"][0], "settings": {**pocket["steps"][0]["settings"], "prefix_tokens": 1179648000}}])
