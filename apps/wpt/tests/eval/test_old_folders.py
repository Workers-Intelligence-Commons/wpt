"""A model folder written before snapshots keeps working: its
provenance.json (old-provenance.json, frozen as such folders hold it)
names its corpus by id, depot and dataset, and wpt eval, the NOTICE and
the model card read it as it is, never rewriting it."""

import json
import shutil
from pathlib import Path

from wpt_model.notice import documents
from wpt_model.provenance import corpus_id, corpus_label, on_snapshot

from wpt.eval import main

OLD = Path(__file__).with_name("old-provenance.json")


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

