"""Model folders of older shapes keep working, and wpt eval, the NOTICE
and the model card read them as they are, never rewriting them. Each
provenance.json here is frozen as such folders hold it: a folder written
before snapshots (old-provenance.json) names its corpus by id, depot and
dataset; a folder trained on deliveries that names its order recipe_name
and recipe (recipe-provenance.json), with a drafter whose trained_by
names it recipe_name (recipe-drafter-provenance.json); a folder trained by
an order file names no family or size, and its order's stages give their
updates and batches by hand."""

import json
import shutil
from pathlib import Path

from wpt_model import load_drafter, load_model
from wpt_model.notice import documents
from wpt_model.provenance import corpus_id, corpus_label, on_snapshot, order, order_name

from wpt.chat import main as chat
from wpt.eval import main

OLD = Path(__file__).with_name("old-provenance.json")
RECIPE = Path(__file__).with_name("recipe-provenance.json")
RECIPE_DRAFTER = Path(__file__).with_name("recipe-drafter-provenance.json")


def test_an_old_provenance_is_read_by_eval_inputs_the_notice_and_the_card(trained, tmp_path):
    folder = trained.folder
    shutil.copy(OLD, folder / "provenance.json")
    frozen = OLD.read_bytes()
    old = json.loads(frozen)
    label = f"corpus {'5ca1ab1e' * 8}, built from dataset classics of the depot https://depot.example.org"
    assert not on_snapshot(old) and corpus_id(old) == "5ca1ab1e" * 8 and corpus_label(old) == label

    written = documents(folder)
    assert f"From {label}:" in written["NOTICE"]
    assert (f"- Corpus: `{'5ca1ab1e' * 8}`, built from dataset `classics` of the depot `https://depot.example.org`"
            in written["README.md"]) and "# WPT model, corpus 5ca1ab1e5ca1\n" in written["README.md"]

    test = tmp_path / "corpus_of.py"
    test.write_text("from wpt.eval import Result\n\n"
                    "def run(context):\n"
                    "    return Result({'corpus': context.provenance['corpus']['id']}, 'read the old folder')\n")
    main([str(folder), str(test)])

    [run] = trained.records("wpt eval")
    assert run.ids["inputs"]["corpus"] == "5ca1ab1e" * 8  # what the run's record chains on
    assert json.loads((folder / "evals" / "corpus_of.json").read_text())["data"]["corpus"] == "5ca1ab1e" * 8
    card = (folder / "README.md").read_text()
    assert "read the old folder" in card and "built from dataset `classics`" in card
    assert (folder / "provenance.json").read_bytes() == frozen  # never rewritten



def test_a_provenance_that_names_its_order_recipe_is_read_by_eval_the_notice_and_the_card(trained, tmp_path):
    folder = trained.folder
    shutil.copy(RECIPE, folder / "provenance.json")
    # the drafter's provenance as frozen, but for the weights it was trained against: this folder's own
    drafter = folder / "drafter" / "provenance.json"
    older_drafter = {**json.loads(RECIPE_DRAFTER.read_text()), "target": json.loads(drafter.read_text())["target"]}
    assert older_drafter["trained_by"] == {"command": "wpt train", "recipe_name": "d"}
    drafter.write_text(json.dumps(older_drafter, indent=1) + "\n")
    frozen, frozen_drafter = RECIPE.read_bytes(), drafter.read_bytes()
    older = json.loads(frozen)
    assert "order" not in older and "order_name" not in older
    assert on_snapshot(older) and order_name(older) == "d" and order(older) == older["recipe"]
    corpus = older["corpus"]["id"]
    label = corpus_label(older)
    assert ", order d, code at commit 0123456789abcdef0123456789abcdef01234567" in label

    written = documents(folder)
    assert f"Trained on {label}, corpus {corpus}:" in written["NOTICE"]
    assert f"- Trained on {label}" in written["README.md"]
    assert "- Trained by `wpt train`, order `d`, the model frozen" in written["README.md"]

    test = tmp_path / "order_of.py"
    test.write_text("from wpt.eval import Result\n"
                    "from wpt_model.provenance import order_name\n\n"
                    "def run(context):\n"
                    "    return Result({'order': order_name(context.provenance)}, 'read the folder')\n")
    main([str(folder), str(test)])

    [run] = trained.records("wpt eval")
    assert run.ids["inputs"]["corpus"] == corpus
    assert json.loads((folder / "evals" / "order_of.json").read_text())["data"]["order"] == "d"
    card = (folder / "README.md").read_text()
    assert "read the folder" in card and ", order d, " in card and "order `d`, the model frozen" in card
    assert (folder / "provenance.json").read_bytes() == frozen  # never rewritten
    assert (folder / "drafter" / "provenance.json").read_bytes() == frozen_drafter


def test_a_folder_trained_by_an_order_file_before_families_loads_and_is_read_by_eval_chat_and_the_card(
        trained, tmp_path, capsys):
    # a folder trained by data/orders/models/d.json: no family and no size, its order's stages written by hand
    folder = trained.folder
    provenance = json.loads((folder / "provenance.json").read_text())
    older = {key: value for key, value in provenance.items() if key not in ("family", "size")}
    older["order_name"] = "d"
    older["order"]["steps"][0]["settings"] = {"stages": [{"block_size": 16, "steps": 1, "batch_size": 4}],
                                              "eval_every": 2, "eval_batches": 20, "prefix_tokens": 6002}
    (folder / "provenance.json").write_text(json.dumps(older, indent=1) + "\n")
    drafter = folder / "drafter" / "provenance.json"
    older_drafter = json.loads(drafter.read_text())
    older_drafter["trained_by"] = {"command": "wpt train", "order_name": "d"}
    drafter.write_text(json.dumps(older_drafter, indent=1) + "\n")
    frozen, frozen_drafter = (folder / "provenance.json").read_bytes(), drafter.read_bytes()
    assert ", order d, " in corpus_label(older)

    load_model(folder)
    load_drafter(folder)
    written = documents(folder)
    assert "- Trained by `wpt train`, order `d`, the model frozen" in written["README.md"]
    capsys.readouterr()
    chat([str(folder), "--about"])
    assert written["NOTICE"].splitlines()[0] in capsys.readouterr().out

    test = tmp_path / "order_of.py"
    test.write_text("from wpt.eval import Result\n"
                    "from wpt_model.provenance import order_name\n\n"
                    "def run(context):\n"
                    "    return Result({'order': order_name(context.provenance)}, 'read the folder')\n")
    main([str(folder), str(test)])
    assert json.loads((folder / "evals" / "order_of.json").read_text())["data"]["order"] == "d"
    assert (folder / "provenance.json").read_bytes() == frozen  # never rewritten
    assert drafter.read_bytes() == frozen_drafter
