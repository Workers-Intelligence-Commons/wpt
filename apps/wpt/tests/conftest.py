"""Shared setup for the app's tests: a corpus built from a depot of 16
plain texts, and a tiny model trained on it through wpt train. Test files
reach them as fixtures (they cannot import this file under
--import-mode=importlib)."""

import json
import secrets
import socket
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wpt_depot import About, Declaration, Depot, NewText
from wpt_depot import corpus as corpora
from wpt_store import open_store
from wpt_tokenizer import Tokenizer
from wpt.train import main as train

ME = Declaration("Ada Writer", "ada@example.org", True)
LICENSE = ("MIT", "MIT License", "Copyright (c) 2026 Ada Writer\n\nPermission is hereby granted, free of charge, to "
           "any person.\n")
NOTICE = "Fox Stories, copyright (c) 2026 Ada Writer"
TAG = "en"  # every test text's tag, and the dataset's one share
TOKENIZER = "t"  # the tokenizer recipe every test dataset names
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
    """A corpus of a depot's dataset "d": 16 plain texts. The working folder tmp_path is the
    repository, and the store is WPT_STORE. Gives store, depot (the Depot),
    corpus_id (the corpus of "d", which the model recipe "d", a tiny model
    trained one step, names), records(command) (the record folders of that
    command's runs, oldest first), notice (the texts' notice),
    recipe(name, model=..., training=..., drafter=..., steps=...,
    dataset="d"), which writes a model recipe of that dataset, with those
    changes to "d" (training: to ONE_STEP, its read-text step; steps: the
    entries that follow it), and returns its name, and dataset(name,
    items=()), which adds the items (NewText, licensed MIT and tagged TAG
    unless they say otherwise) to the depot, admitted, makes the dataset
    name of every admitted text, builds its corpus and returns its id."""
    store = open_store((tmp_path / "store").as_uri())
    depot = Depot.new(str(tmp_path / "depot"), About("fox", "Ada Writer", "ada@example.org"))
    depot.add_license(*LICENSE)
    stories = [f"story {i}: the quick brown fox jumps over the lazy dog. " * 30 for i in range(16)]
    tokenizer = Tokenizer.train([story.encode() for story in stories], vocab_size=VOCAB_SIZE)

    def write(kind, name, document):
        path = tmp_path / "data" / "recipes" / f"{kind}s" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        return name

    def dataset(name, items=()):
        items = [replace(item, licenses=item.licenses or (LICENSE[0],), tags=item.tags or (TAG,)) for item in items]
        if items:
            added = depot.add(items, ME)
            depot.decide("admitted", f"id IN ({', '.join(repr(one) for one in added)})", ME)
        library = depot.new_library(name)
        depot.put(library, "true")
        depot.new_dataset(name, TOKENIZER, [(None, library)], [(TAG, 1.0)])
        return corpora.build(depot, name, store, tokenizer, 1, log=lambda *_: None)["id"]

    def recipe(name="d", model=None, training=None, drafter=None, steps=(), dataset="d"):
        text = {**ONE_STEP, **(training or {})}
        named = [one["id"] for one in depot.sql(f"SELECT id FROM datasets WHERE name = '{dataset}'")]
        found = corpora.latest(store, depot.address, named[0]) if named else None  # the whole corpus, once built
        total = found["total_tokens"] if found else 3000
        read_text = {"step": "read-text", "optimizer": {"name": text["optimizer"], "settings": text["settings"]},
                     "settings": {key: text[key] for key in ("stages", "eval_every", "eval_batches")}}
        return write("model", name, {"data": {"depot": depot.address, "dataset": dataset, "tokenizer": TOKENIZER,
                                              "total_tokens": total},
                                     "model": {**TINY_MODEL, **(model or {})}, "seed": text["seed"],
                                     "steps": [read_text, *steps], "drafter": {**TINY_DRAFTER, **(drafter or {})}})

    write("tokenizer", TOKENIZER, {"version": 1, "depot": depot.address, "dataset": "d", "sample_bytes": 10**9,
                                   "vocab_size": VOCAB_SIZE})
    corpus_id = dataset("d", [NewText(text=story, notice=NOTICE) for story in stories])
    recipe()
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())  # where each command run also writes its record
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
                           records=records, dataset=dataset)


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


@pytest.fixture
def trained(built, tmp_path):
    """The built corpus plus tmp_path/model, trained by the recipe "d"."""
    train(["d", built.corpus_id, "--out", "model"])
    return SimpleNamespace(**vars(built), folder=tmp_path / "model")


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
