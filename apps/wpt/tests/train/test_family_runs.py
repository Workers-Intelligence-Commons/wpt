"""A decay branch of a family's size trained through wpt train
(docs/model.md, "Model families"): the trunk's steps train once, and a
branch goes on from the trunk's state at its start."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from wpt.train import main as train


def tiny(built) -> dict:
    """The application tests' tiny model (the family d's) as a family of one size reading 256 tokens of the
    snapshot d in 4 updates of 4 windows of 16 tokens, after a warm-up of 1; its branches make 2 updates (half),
    decaying after update 1, and 4, decaying after update 3. No average of the last states: by-part's 10 states,
    2% of the updates apart, lie inside the decay of a size of 50 updates or more, and before it at 4."""
    base = json.loads(Path("data/orders/d.json").read_text())
    optimizer = {"lr": 0.002, "warmup": 0.25, "average_states": 0}
    return {"data": {**base["data"], "total_tokens": 256}, "model": base["model"], "seed": 1337,
            "drafter": base["drafter"],
            "steps": [{"step": "read-text", "optimizer": {"name": "by-part", "settings": optimizer},
                       "settings": {"stages": [{"block_size": 16, "share": 1}], "tokens_per_update": 64,
                                    "mtp_switch": 0.8}}],
            "branches": [0.5, 1], "sizes": {"columns": ["size"], "rows": [["small"]]}}


def steps_of(folder):
    """(fingerprint, reused, updates reached) of each step the model folder's provenance lists."""
    provenance = json.loads((Path(folder) / "provenance.json").read_text())
    return [(step["fingerprint"], step["reused"], step["updates"]) for step in provenance["steps"]]


def test_branches_share_the_trunk_and_the_full_branch_trains_as_the_size_s_own_run(built, tmp_path):
    built.write("tiny", tiny(built))
    corpus = built.corpus_id
    # on the CPU, where the same updates give the same bits
    (tmp_path / "cpu.py").write_text("import torch\nfrom wpt_training import Strategy\n\n"
                                     "def start():\n    return Strategy(torch.device('cpu'))\n")
    cpu = ["--strategy", str(tmp_path / "cpu.py")]
    train(["tiny:small:branch-0.5", corpus, "--out", "half", *cpu])
    half = steps_of("half")
    assert [found[1:] for found in half] == [(False, 1), (False, 2)]
    trunk = half[0][0]
    # the full branch reuses the trunk's first step, goes on to update 3 and decays to update 4: the reused
    # state, then its own, carried on to the next step
    train(["tiny:small:branch-1", corpus, "--out", "full", *cpu])
    assert steps_of("full")[0] == (trunk, True, 1)
    assert [found[1:] for found in steps_of("full")[1:]] == [(False, 3), (False, 4)]
    provenance = json.loads((Path("full") / "provenance.json").read_text())
    assert (provenance["family"], provenance["size"], provenance["run"]) == ("tiny", "small", "branch-1")
    # the same weights as the size's own run of 4 updates: the same updates at the same rates on the same
    # windows (its switch of the multi-token prediction's weight, after 80% of the updates, is the branch's
    # too: it falls at update 4 in both)
    train(["tiny:small", corpus, "--out", "own", *cpu])
    full, own = load_file("full/model.safetensors"), load_file("own/model.safetensors")
    assert full.keys() == own.keys() and all(torch.equal(full[name], own[name]) for name in own)
