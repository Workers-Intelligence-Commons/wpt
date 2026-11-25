"""Shared setup for the app's tests: a corpus built from a snapshot of a
depot of 16 plain texts, and a tiny model trained through wpt train on the
corpus the freight delivers for the same order. Test files
reach them as fixtures (they cannot import this file under
--import-mode=importlib)."""

import hashlib
import json
import secrets
import socket
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wpt_depot import About, Declaration, Depot, NewText
from wpt_depot import corpora
from wpt_depot.errors import NotFound
from wpt_depot.corpus import build
from freight import api as freight
from wpt_store import open_store
from wpt_tokenizer import Tokenizer
from wpt.resources import main as build_resources
from wpt.train import main as train

ME = Declaration("Ada Writer", "ada@example.org", True)
LICENSE = ("MIT", "MIT License", "Copyright (c) 2026 Ada Writer\n\nPermission is hereby granted, free of charge, to "
           "any person.\n")
NOTICE = "Fox Stories, copyright (c) 2026 Ada Writer"
TAG = "en"  # every test text's tag, and the recipes' one share
TOKENIZER = "t"  # the tokenizer recipe every test corpus names as its vocabulary's
VOCAB_SIZE = 556


TINY_MODEL = {"window_layers": 1, "sparse_groups": 1, "sparse_layers": 1, "d_model": 16, "attention_heads": 2,
              "attention_head_dim": 64, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
              "shared_width": 16, "engram_layers": [2], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
              "engram_table_size": 31, "residual_groups": 3, "loops": 0}
# the read-text step of the recipe "d": the optimizer, its settings, the step's text windows, the recipe's seed
ONE_STEP = {"optimizer": "by-part", "settings": {}, "stages": [{"block_size": 16, "steps": 1, "batch_size": 4}],
            "eval_every": 2, "eval_batches": 20, "seed": 1337}
# the drafter every run trains at its end, as small and short as still trains and calibrates one
TINY_DRAFTER = {"epochs": 1, "prompts": 4, "prompt_tokens": 8, "answer_tokens": 8, "batch": 2, "anchors": 4,
                "held_out": 0.25}


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A corpus of a depot's snapshot "d": 16 plain texts. The working folder tmp_path is the
    repository, the store is WPT_STORE and the depot WPT_DEPOT. Gives store, depot (the Depot),
    corpus_id (the corpus of every token of "d", which the model recipe "d", a tiny model
    trained one step, names), tokenizer (the vocabulary every corpus here is built with), records(command) (the record folders of that
    command's runs, oldest first), notice (the texts' notice), vocabulary
    (the sha256 of the vocabulary every corpus here is built with),
    data(name, size) (the data section of a recipe of the snapshot name at
    that size, as recipe writes it),
    recipe(name, model=..., training=..., drafter=..., steps=...,
    snapshot="d"), which writes a model recipe of that snapshot (its data
    section: the depot by its name, the snapshot's change and digest, the
    one share TAG, the vocabulary, and as total_tokens the tokens of plain
    text of the snapshot's corpus of every token, 3000 before one is
    built), with those changes to "d" (training: to ONE_STEP, its
    read-text step; steps: the entries that follow it), and returns its
    name, snapshot(name, items=()), which adds the items (NewText, licensed
    MIT and tagged TAG unless they say otherwise) to the depot, admitted,
    makes the snapshot name of every admitted text, builds its corpus of
    every token and returns its id, and sized(name, size, url=None), which
    builds the corpus of the snapshot name at that size, its record naming
    the depot by url (default: its name), and returns its record."""
    store = open_store((tmp_path / "store").as_uri())
    depot = Depot.new(str(tmp_path / "depot"), About("fox", "Ada Writer", "ada@example.org"))
    depot.add_license(*LICENSE)
    stories = [f"story {i}: the quick brown fox jumps over the lazy dog. " * 30 for i in range(16)]
    tokenizer = Tokenizer.train([story.encode() for story in stories], vocab_size=VOCAB_SIZE)
    vocabulary = {"sha256": hashlib.sha256(tokenizer.to_bytes()).hexdigest(), "size": tokenizer.vocab_size}

    def data_of(name, size):
        """The data section of a recipe of the snapshot name at that size."""
        try:
            snapshot = depot.snapshot(name)
        except NotFound:  # a recipe of a snapshot not made: its change and digest are not known
            snapshot = {"at_change": 0, "digest": ""}
        return {"depot": depot.name(), "snapshot": name, "change": snapshot["at_change"],
                "digest": snapshot["digest"], "shares": [{"tag": TAG, "share": 1.0}], "total_tokens": size,
                "vocabulary": vocabulary}

    def sized(name, size, url=None) -> dict:
        snapshot = depot.snapshot(name)
        return build(depot, snapshot["id"], data_of(name, size), store, tokenizer, address=url or depot.name(),
                     recipe=name, tokenizer_recipe=TOKENIZER, tokenizer_version=1, log=lambda *_: None)

    def write(kind, name, document):
        path = tmp_path / "data" / "recipes" / f"{kind}s" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        return name

    def snapshot(name, items=()):
        items = [replace(item, licenses=item.licenses or (LICENSE[0],), tags=item.tags or (TAG,)) for item in items]
        if items:
            added = depot.add(items, ME)
            depot.decide("admitted", f"id IN ({', '.join(repr(one) for one in added)})", ME)
        depot.create_snapshot(name)
        return sized(name, None)["id"]

    def recipe(name="d", model=None, training=None, drafter=None, steps=(), snapshot="d"):
        text = {**ONE_STEP, **(training or {})}
        whole = next((one for one in corpora.records(store)
                      if one.get("snapshot", {}).get("name") == snapshot and one["max_tokens"] is None), None)
        total = whole["total_tokens"] if whole else 3000
        read_text = {"step": "read-text", "optimizer": {"name": text["optimizer"], "settings": text["settings"]},
                     "settings": {key: text[key] for key in ("stages", "eval_every", "eval_batches")}}
        return write("model", name, {"data": data_of(snapshot, total),
                                     "model": {**TINY_MODEL, **(model or {})}, "seed": text["seed"],
                                     "steps": [read_text, *steps], "drafter": {**TINY_DRAFTER, **(drafter or {})}})

    corpus_id = snapshot("d", [NewText(text=story, notice=NOTICE) for story in stories])
    recipe()
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())  # where each command run also writes its record
    monkeypatch.setenv("WPT_DEPOT", depot.address)  # the depot the recipes name (by its name: it has no public address)
    monkeypatch.delenv("WPT_DEPOT_URL", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)  # no live backend, whatever this machine sets
    monkeypatch.setenv("WPT_MAX_RESTARTS", "0")  # wpt train runs in the test's own process, unless a test says
    monkeypatch.chdir(tmp_path)

    def records(command):
        found = []
        for folder in store.record_folders():
            start = json.loads(store.read_record(folder, "start.json"))
            if start["command"] == command:
                found.append(SimpleNamespace(
                    folder=folder, start=start, ids=json.loads(store.read_record(folder, "ids.json")),
                    events=[json.loads(line) for line in store.read_record(folder, "events.jsonl").splitlines()],
                    end=json.loads(store.read_record(folder, "end.json")) if "end.json" in store.record_files(folder)
                    else None))
        return sorted(found, key=lambda run: run.start["started"])

    return SimpleNamespace(store=store, depot=depot, corpus_id=corpus_id, notice=NOTICE, recipe=recipe,
                           records=records, snapshot=snapshot, sized=sized, vocabulary=vocabulary["sha256"],
                           tokenizer=tokenizer, data=data_of)


@pytest.fixture
def worker_port():
    """Gives a port for a gpu worker that meets others on the port after its
    own (docs/machines.md, "Several machines"): an even one below the range
    the system takes the ports of its connections from, since a port there
    stays taken for a minute after a connection that used it, and PyTorch
    cannot listen on it. Even, so one worker's port is never another's next."""
    def free() -> int:
        while True:
            port = 20000 + 2 * secrets.randbelow(6000)
            try:
                with socket.socket() as first, socket.socket() as second:
                    first.bind(("127.0.0.1", port))
                    second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port

    return free


LEARNED = {"size": VOCAB_SIZE, "sample_bytes": 10**6}  # every byte of the 16 stories


@pytest.fixture(params=["named"])
def trained(request, built, tmp_path, monkeypatch):
    """tmp_path/model, trained by the order "d" on deliveries of the freight
    (its home FREIGHT_HOME, tmp_path/freight), as a person makes it:
    `wpt build-resources d`, then `wpt train d`. By default ("named") the
    order names its vocabulary by its sha256, the test vocabulary, which
    the dock holds as the adopted delivery `adopted-<sha256>` (put there by
    freight.api.adopt, its items every text of the depot, as the tokenizer
    recipe TOKENIZER version 1 read them); with the parameter "learned"
    (indirect), the order learns its vocabulary of VOCAB_SIZE entries from
    the stories instead (LEARNED). Gives what built gives, folder, notes
    (the two delivery notes `wpt build-resources` printed) and corpus_id,
    the corpus delivery's id."""
    monkeypatch.setenv("FREIGHT_HOME", str(tmp_path / "freight"))
    printed = request.getfixturevalue("capsys")
    if request.param == "named":
        change = built.depot.snapshot("d")["at_change"]
        items = [{"depot": built.depot.name(), "id": row["id"], "change": change}
                 for row in built.depot.sql("SELECT id FROM texts ORDER BY id")]
        freight.adopt(f"{freight.ADOPTED}{built.vocabulary}", {"vocabulary.json": built.tokenizer.to_bytes()},
                      {"sha256": built.vocabulary, "size": built.tokenizer.vocab_size,
                       "tokenizer_version": json.loads(built.tokenizer.to_bytes())["tokenizer_version"]},
                      {"by": "the test", "recipe": TOKENIZER, "version": 1}, on_erasure="note", items=items)
    else:
        path = tmp_path / "data" / "recipes" / "models" / "d.json"
        order = json.loads(path.read_text())
        order["data"]["vocabulary"] = LEARNED
        path.write_text(json.dumps(order))
    assert build_resources(["d"]) == 0
    out = printed.readouterr().out
    notes = json.loads(out[out.index("\n{\n") + 1:])  # printed last, after the lines of the jobs
    train(["d", "--out", "model"])
    return SimpleNamespace(**{**vars(built), "corpus_id": notes["corpus"]["id"]}, folder=tmp_path / "model",
                           notes=notes)


@pytest.fixture
def s3(tmp_path, monkeypatch):
    """An S3 store that WPT_STORE now names (moto answers its requests inside
    the test process).
    Gives store, copy(local), which puts a local store's files into it (the
    built corpus, for a command that reads it), and refuse(marker), after which the bucket refuses every group of
    record lines that holds marker."""
    import boto3
    from moto import mock_aws

    from wpt_store import S3Store

    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/store")
        monkeypatch.setenv("WPT_STORE", "s3://wpt-test/store")

        def copy(local):
            for key in local.kind.list(""):
                store.kind.write(key, local.root / key)

        def refuse(marker):
            real = S3Store._raw_flush_records

            def refusing(self):
                if any(marker in data for additions in self._pending.values() for *_, data in additions):
                    raise OSError("the bucket refused the request")
                real(self)

            monkeypatch.setattr(S3Store, "_raw_flush_records", refusing)

        yield SimpleNamespace(store=store, copy=copy, refuse=refuse)
