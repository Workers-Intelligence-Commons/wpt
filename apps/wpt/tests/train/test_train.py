import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from wpt_inference import COSTS_FILE, DRAFT, costs_setup, replayed_costs
from wpt_model import load_drafter, load_model, load_tokenizer, tokenizer_identity, weight_hashes
import wpt_telemetry
from wpt_store import LocalStore, StoreError
from wpt_telemetry import FLUSH_THREAD
from wpt_tokenizer import Tokenizer
from wpt_utils.device import select_device
from wpt.train import InputError, StoreCheckpoint, check_record, main
from wpt_training import load_state
from wpt_utils.lines import bare
from wpt_depot import corpora
from wpt_store.shards import shard_file

ROOT = Path(__file__).resolve().parents[4]


def one_stage(steps, block_size=16, batch_size=4):
    """A read-text step of one stage (docs/training.md, "Stages")."""
    return {"stages": [{"block_size": block_size, "steps": steps, "batch_size": batch_size}]}


def checkpoint(store, name):
    """The file of the checkpoint of the run name, None without one."""
    return StoreCheckpoint(store, name, None, set()).state_file()


def forget_steps(store):
    """Forget every step result the store keeps, as a store that never saw
    these recipes: the next run trains every step again."""
    shutil.rmtree(store.root / "cache" / "steps", ignore_errors=True)


def test_training_writes_a_model_with_its_provenance(built, tmp_path, monkeypatch, capsys):
    store, corpus_id = built.store, built.corpus_id
    recipe = built.recipe("three", training=one_stage(3))
    monkeypatch.setenv("WPT_RECIPE", recipe)  # the recipe and the corpus come from the environment too
    monkeypatch.setenv("WPT_CORPUS", corpus_id)
    main(["--out", "model"])

    model = load_model(tmp_path / "model")
    assert model.config.n_layer == 5 and model.config.block_size == 16  # the recipe's numbers
    record = corpora.find(store, corpus_id)
    # the weights name the corpus's vocabulary, and only that file loads with them
    assert tokenizer_identity(tmp_path / "model") == ("t version 1", record["tokenizer"]["vocabulary"]["sha256"])
    tokenizer = Tokenizer.from_bytes(load_tokenizer(tmp_path / "model"))
    assert model.config.vocab_size == tokenizer.vocab_size
    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    # the snapshot, the recipe and the code build the corpus again: the corpus itself is a cache
    assert provenance["corpus"] == {"id": corpus_id, "build_key": record["build_key"], "rules": record["rules"],
                                    "depot": built.depot.name(), "snapshot": record["snapshot"],
                                    "data": record["data"]}
    assert provenance["data"] == json.loads((tmp_path / "data" / "recipes" / "models" / "three.json").read_text())["data"]
    assert (provenance["data"]["snapshot"], provenance["data"]["change"], provenance["data"]["digest"]) == (
        "d", record["snapshot"]["change"], record["snapshot"]["digest"])
    assert provenance["repository_commit"] is not None
    assert provenance["vocabulary"] == record["tokenizer"]["vocabulary"]
    assert provenance["learned"] == {"plain": record["total_tokens"]}  # the recipe trains on the whole corpus
    assert provenance["recipe_name"] == "three"
    assert provenance["recipe"]["steps"][0]["settings"]["stages"] == [{"block_size": 16, "steps": 3, "batch_size": 4}]
    [step] = provenance["steps"]
    assert step["step"]["name"] == "read-text" and step["updates"] == 3 and not step["reused"]
    # the step's result is kept in the store, under its fingerprint
    assert hashlib.sha256(store.get_result("steps", step["fingerprint"])[0]).hexdigest() == step["result"]
    assert [s["token_sha256"] for s in provenance["shards"]] == [s["token_sha256"] for s in record["shards"]["plain"]]


def test_a_recipe_with_one_vocabulary_table_trains_a_model_and_its_drafter(built, tmp_path):
    # tied_tables in the recipe's model: the run, its drafter (which reads the model's embedding and head)
    # and the saved folder hold one table
    main([built.recipe("tied", model={"tied_tables": True}), built.corpus_id, "--out", "tied"])
    model = load_model(tmp_path / "tied")
    assert model.config.tied_tables and "embedding.weight" not in model.state_dict()
    load_drafter(tmp_path / "tied")


def test_a_training_run_records_what_it_read_every_step_and_what_it_wrote(built, tmp_path, capsys):
    # afterwards one must know exactly what happened, from the store alone:
    # the corpus read, each step's loss, the evaluations, the files written
    main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    printed = bare(capsys.readouterr().out)

    [run] = built.records("wpt train")
    assert run.end["status"] == "succeeded" and run.end["exit_code"] == 0
    assert run.ids["inputs"]["recipe"] == "three" and run.ids["inputs"]["corpus"] == built.corpus_id
    assert f"-wpt-train-{built.corpus_id[:8]}-" in run.folder  # its folder is found by the corpus it trained on
    steps = [e for e in run.events if e.get("name") == "training.step"]
    assert [e["attributes"]["step"] for e in steps] == [1, 2, 3]
    assert all(math.isfinite(e["values"]["train_loss"]) for e in steps)
    evaluations = [e["attributes"]["step"] for e in run.events if e.get("name") == "training.evaluation"]
    assert evaluations == [1, 2, 3]  # eval_every 2, and the first and last steps
    folder = tmp_path / "model"
    weights = hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()
    assert run.ids["outputs"]["weights"] == {"model.safetensors": weights}
    [files] = [e["attributes"]["files"] for e in run.events if e.get("name") == "model.files"]
    assert files["NOTICE"] == hashlib.sha256((folder / "NOTICE").read_bytes()).hexdigest()
    assert {"LICENSE", "README.md", "provenance.json", "tokenizer.json", "config.json"} <= set(files)
    logged = "".join(e["text"] + "\n" for e in run.events if e["kind"] == "log")
    assert logged == printed  # every line it printed, nothing else


def test_what_the_gpu_worker_records_reaches_the_live_backend_once_through_this_side(built, otlp_backend, monkeypatch,
                                                                                     capsys):
    # the local worker's environment names the backend too: it must still send nothing itself, or its spans come twice
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", otlp_backend.endpoint)
    main(["d", built.corpus_id, "--out", "model"])

    run = built.records("wpt train")[-1]
    update = next(line for line in run.events if line["kind"] == "event" and line["name"] == "training.step")
    assert otlp_backend.gauges()["wpt.training.step.train_loss"] == update["values"]["train_loss"]
    in_the_record = sorted(line["span"] for line in run.events if line["kind"] == "span_end")
    assert sorted(span.span_id.hex() for _, span in otlp_backend.spans()) == in_the_record
    assert {resource["service.instance.id"] for resource, _ in otlp_backend.spans()} == {run.folder}
    printed = sum(line["kind"] in ("log", "event") for line in run.events)
    assert len(otlp_backend.logs()) == printed
    assert "OpenTelemetry" not in capsys.readouterr().err


def test_training_ends_with_the_drafter_against_the_final_weights_and_the_scheduler_s_times(built, tmp_path,
                                                                                             capsys):
    # a model and its drafter are one
    # unit; the run trains the drafter at its end, the model frozen, and
    # measures the scheduler's pass times once, into its record; a later run
    # replays them
    main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    folder = tmp_path / "model"
    drafter = load_drafter(folder).provenance
    assert drafter["target"] == {"weights": weight_hashes(folder)}
    assert drafter["trained_by"] == {"command": "wpt train", "recipe_name": "three"}
    assert drafter["corpus"] == built.corpus_id and drafter["prompts"]["count"] == 4
    [run] = built.records("wpt train")
    names = [e.get("name") for e in run.events]
    assert names.index("drafter.settings") > max(i for i, name in enumerate(names) if name == "training.step")
    assert run.ids["outputs"]["drafter"] == hashlib.sha256((folder / "drafter" / "model.safetensors").read_bytes()
                                                           ).hexdigest()
    recorded = built.store.read_record(run.folder, COSTS_FILE)
    # the times carry the setup they were measured on
    setup = costs_setup(load_model(folder).config, 1, DRAFT, select_device())
    assert json.loads(recorded)["setup"] == setup
    costs = replayed_costs(recorded, setup)
    assert len(costs) == DRAFT + 1 and all(seconds > 0 for seconds in costs)

    main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "again", "--scheduler-costs",
          run.folder])
    replayed = built.records("wpt train")[-1]
    assert built.store.read_record(replayed.folder, COSTS_FILE) == recorded
    # refused before training, not after: no such record, or times measured on another setup
    for recipe, record, why in ((built.recipe("three", training=one_stage(3)), "no-such-run", "MissingFile"),
                                (built.recipe("block", training=one_stage(3), drafter={"block": 4}), run.folder,
                                 "another setup: draft 5, here 4"),
                                (built.recipe("longer", training=one_stage(3, block_size=32)), run.folder,
                                 "another setup: model block_size 16, here 32")):
        with pytest.raises(SystemExit) as refused:
            main([recipe, built.corpus_id, "--out", "never", "--scheduler-costs", record])
        assert refused.value.code == 2 and not (tmp_path / "never").exists()
        assert why in capsys.readouterr().err
        assert not [e for e in built.records("wpt train")[-1].events if e.get("name") == "training.step"]


def test_drafter_feature_layers_the_model_does_not_have_are_refused_before_training(built, tmp_path, capsys):
    # checked against the model's layers at startup
    with pytest.raises(SystemExit) as refused:
        main([built.recipe("three", training=one_stage(3), drafter={"features": [9]}), built.corpus_id, "--out",
              "model"])
    assert refused.value.code == 2 and "9 is not a layer of a" in capsys.readouterr().err
    assert not [e for e in built.records("wpt train")[-1].events if e.get("name") == "training.step"]


@pytest.mark.parametrize("given", ["option", "environment"])
def test_compiling_asked_for_off_a_cuda_device_refuses_the_run_before_training(built, on_the_cpu, monkeypatch, capsys,
                                                                               given):
    # docs/training.md, "Compiling the passes": as --kernels names what cannot work here
    if given == "environment":
        monkeypatch.setenv("WPT_COMPILE", "1")
    with pytest.raises(SystemExit) as refused:
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model",
              *(["--compile"] if given == "option" else [])])
    assert refused.value.code == 2
    assert "--compile: the passes compile on a CUDA device only, and this process runs on cpu" in \
        capsys.readouterr().err
    assert not [e for e in built.records("wpt train")[-1].events if e.get("name") == "training.step"]


def test_a_run_that_cannot_compile_trains_uncompiled_and_says_why_in_one_line(built, on_the_cpu, capsys):
    # unasked, the passes compile wherever they can; here they cannot, and --no-compile is never refused
    main([built.recipe("two", training=one_stage(2)), built.corpus_id, "--out", "model"])
    said = [line for line in bare(capsys.readouterr().out).splitlines() if "compile:" in line]
    assert len(said) == 1 and "compile: off, since the passes compile on a CUDA device only" in said[0]
    forget_steps(built.store)  # it trains again
    main([built.recipe("two", training=one_stage(2)), built.corpus_id, "--out", "other", "--no-compile"])
    assert "compile:" not in capsys.readouterr().out
    from wpt.train import parse_args
    assert [parse_args(["r", *given], environ=environ).compile for given, environ in (
        ([], {}), (["--compile"], {}), (["--no-compile"], {}), ([], {"WPT_COMPILE": "0"}),
        ([], {"WPT_COMPILE": "yes"}), (["--no-compile"], {"WPT_COMPILE": "1"}))] == [None, True, False, False, True,
                                                                                     False]


def test_kernels_named_that_cannot_work_on_the_gpu_worker_refuse_the_run_before_training(built, on_the_cpu, monkeypatch,
                                                                                         capsys):
    # docs/model.md, "Attention": a named form is the one that runs or the
    # run stops with the reason, never another form; here a worker with no
    # GPU asked for Triton's kernels. A name that is no form is refused too
    with pytest.raises(SystemExit) as refused:
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model", "--kernels", "triton"])
    assert refused.value.code == 2
    assert "KernelUnavailable: Triton kernels run on a GPU, and this process runs on cpu" in capsys.readouterr().err
    assert not [e for e in built.records("wpt train")[-1].events if e.get("name") == "training.step"]

    monkeypatch.setenv("WPT_KERNELS", "fast")
    with pytest.raises(SystemExit) as refused:
        main(["d", built.corpus_id, "--out", "model"])
    assert refused.value.code == 2 and "WPT_KERNELS='fast': not one of tilelang, triton, pytorch" in \
        capsys.readouterr().err


def test_a_drafter_that_cannot_be_trained_refuses_the_run_and_its_steps_stay_kept(built, tmp_path, capsys):
    # held_out keeps both prompts apart: nothing is left to learn from. The
    # step's result was kept when it ended, even with no checkpoint during
    # training, so the same command, its drafter fixed, takes no update and
    # trains the drafter again
    failing = built.recipe("three", training=one_stage(3), drafter={"prompts": 2, "held_out": 0.9})
    with pytest.raises(SystemExit) as refused:
        main([failing, built.corpus_id, "--out", "model", "--checkpoint-every", "0"])
    assert refused.value.code == 2
    assert capsys.readouterr().err.strip().endswith(
        "DrafterFailed: held_out 0.9 keeps every one of the 2 prompts apart; nothing was written; every step's "
        "result is kept in the store: the same command trains the drafter again without training")
    assert not (tmp_path / "model" / "config.json").exists() and not (tmp_path / "model" / "checkpoint").exists()
    [run] = built.records("wpt train")
    names = [e.get("name") for e in run.events if e["kind"] != "span_start"]
    assert names.count("training.checkpoint") == 1 and names.count("store.result_set") == 1  # the step's result
    assert names.index("training.checkpoint") > max(i for i, name in enumerate(names) if name == "training.step")
    # the drafter's section takes no part in a step's fingerprint: a fixed one reuses the kept step
    main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    again = built.records("wpt train")[-1]
    assert not [e for e in again.events if e.get("name") == "training.step"]
    assert [e["attributes"]["step"] for e in again.events if e.get("name") == "training.step_reused"] == ["read-text"]
    assert load_drafter(tmp_path / "model").provenance["target"] == {"weights": weight_hashes(tmp_path / "model")}


def test_a_record_the_store_refuses_stops_the_run_once_its_checkpoint_is_saved(built, tmp_path, monkeypatch, capsys):
    # without its record a run must do nothing more than keep its work: it
    # finishes the step and saves its checkpoint, takes no further step,
    # writes no model, and says why
    real = LocalStore._raw_append_record

    def disk_full_at_step_3(self, folder, name, data):
        if b'"name": "training.step"' in data and b'"step": 3,' in data:
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full_at_step_3)
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")  # the checkpoint it saves comes to this machine
    with pytest.raises(SystemExit) as stopped:
        main([built.recipe("five", training=one_stage(5)), built.corpus_id, "--out", "model"])

    assert stopped.value.code == 3 and "No space left on device" in capsys.readouterr().err
    [run] = built.records("wpt train")
    # the lines held back when step 3's was refused are lost with it (they would have gone to the store
    # together); none comes after it
    steps = [e["attributes"]["step"] for e in run.events if e.get("name") == "training.step"]
    assert steps == [1, 2][:len(steps)]
    assert run.end is None and [path.name for path in (tmp_path / "model").iterdir()] == ["checkpoint"]
    assert load_state(tmp_path / "model" / "checkpoint" / "checkpoint.safetensors")["step"] == 3


def test_a_send_refused_in_the_background_keeps_the_step_in_the_store_checkpoint(built, tmp_path, monkeypatch, capsys):
    # a record that breaks while the process trains (here a send refused in
    # the background, which arrives as a stop signal) must not cost the steps
    # done: the step ends, the store's checkpoint holds it, unrecorded, and
    # --resume gives the model of an unbroken run. The checkpoint it replaced
    # is not deleted, since its deletion could not be recorded first
    real, now = LocalStore._raw_flush_records, tmp_path / "refuse-now"  # the gpu worker's job makes now at update 3

    def refused_in_the_background(self):
        if threading.current_thread().name == FLUSH_THREAD and now.exists():
            raise OSError(28, "No space left on device")
        real(self)

    monkeypatch.setattr(wpt_telemetry, "FLUSH_SECONDS", 0.02)
    monkeypatch.setattr(LocalStore, "_raw_flush_records", refused_in_the_background)
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")
    options = [*interrupting(built, tmp_path, "if os.environ.get('REFUSE'): open(os.environ['REFUSE'], 'w').close(); "
                                              "__import__('time').sleep(2)"), "--checkpoint-every", "1"]
    main([*options, "--out", "whole"])
    forget_steps(built.store)

    monkeypatch.setenv("REFUSE", str(now))
    with pytest.raises(SystemExit) as stopped:
        main([*options, "--out", "here", "--run", "fox-1"])
    monkeypatch.delenv("REFUSE")
    now.unlink()
    assert stopped.value.code == 3 and "No space left on device" in capsys.readouterr().err
    saved = checkpoint(built.store, "fox-1")
    assert load_state(built.store.path(saved))["step"] == 3
    run = built.records("wpt train")[-1]
    named = [e["attributes"]["state"] for e in run.events
             if e.get("name") == "training.checkpoint_saved"]
    assert len(named) == 2 and saved["key"] not in named and built.store.version(named[-1])  # steps 1 and 2; 2's file stays

    main([*options, "--out", "there", "--run", "fox-1", "--resume"])
    whole, there = weights(tmp_path / "whole"), weights(tmp_path / "there")
    assert all(torch.equal(whole[k], there[k]) for k in whole)
    assert checkpoint(built.store, "fox-1") is None and built.store.version(named[-1])


def test_training_into_a_model_folder_replaces_it_whole_and_only_once_recorded(built, tmp_path, monkeypatch):
    # the folder appears only after the record names its files, so a record
    # that fails leaves the earlier folder as it was; a run that succeeds
    # replaces it as a whole, results of other weights and stray files included
    recipe = built.recipe("two", training=one_stage(2))
    main([recipe, built.corpus_id, "--out", "model"])
    folder = tmp_path / "model"
    (folder / "evals").mkdir()
    (folder / "evals" / "memorization.json").write_text("{}\n")
    (folder / "notes.txt").write_text("mine\n")
    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}

    real = LocalStore._raw_append_record

    def disk_full_at_the_files(self, record, name, data):
        if b'"name": "model.files"' in data:
            raise OSError(28, "No space left on device")
        real(self, record, name, data)

    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "_raw_append_record", disk_full_at_the_files)
        with pytest.raises(SystemExit) as stopped:
            main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    assert stopped.value.code == 3
    assert {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()} == before
    assert (folder / "evals" / "memorization.json").exists()
    # the step's result is kept: the same command finishes the run without training
    assert not (folder / "checkpoint").exists()

    main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    assert not (folder / "evals").exists() and not (folder / "notes.txt").exists()
    assert json.loads((folder / "provenance.json").read_text())["recipe_name"] == "three"
    run = built.records("wpt train")[-1]
    assert run.ids["outputs"]["weights"] == {
        "model.safetensors": hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()}
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".model")] == []  # nothing left beside it


def test_on_s3_the_model_folder_is_replaced_only_once_its_record_is_in_the_bucket(built, s3, tmp_path):
    # S3 holds record lines back: the lines naming the new files must be in
    # the bucket before the earlier folder goes, so a refused send leaves it
    # as it was
    s3.copy(built.store)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine\n")
    s3.refuse(b'"training.checkpoint_removed"')
    with pytest.raises(SystemExit) as stopped:
        main(["d", built.corpus_id, "--out", "model"])
    assert stopped.value.code == 3
    # as it was: the step's checkpoint went once its result was kept
    assert sorted(path.name for path in folder.iterdir()) == ["notes.txt"]
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".model")] == []


def test_every_checkpoint_deleted_from_the_store_is_named_in_the_record_first(built, tmp_path, monkeypatch):
    # a file leaves the store only once the record says which one went:
    # each replaced checkpoint; the last one loses its name only, its file
    # kept as it is as the step's state
    real, deleted = LocalStore.delete, []

    def delete(self, state):
        [run] = built.records("wpt train")
        named = [e["attributes"]["state"] for e in run.events if e.get("name") == "training.checkpoint_deleted"]
        deleted.append((state["key"], state["key"] in named))
        real(self, state)

    monkeypatch.setattr(LocalStore, "delete", delete)
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")  # each checkpoint comes to the store
    main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model", "--checkpoint-every", "1",
          "--run", "fox-1"])
    assert len({sha for sha, _ in deleted}) == 2 and all(named for _, named in deleted)
    assert checkpoint(built.store, "fox-1") is None
    [step] = json.loads((tmp_path / "model" / "provenance.json").read_text())["steps"]
    state = json.loads(built.store.get_result("steps", step["fingerprint"])[0])["state"]
    assert built.store.has(state) and state["key"] not in {key for key, _ in deleted}
    assert state["key"].startswith("models/fox-1/")  # beside the run's checkpoint record


def test_on_s3_a_checkpoint_is_deleted_only_once_its_deletion_is_in_the_bucket(built, s3, tmp_path, monkeypatch):
    # S3 holds record lines back: a deletion the bucket did not record must
    # not happen, so the previous checkpoint stays
    from wpt_store import S3Store

    s3.copy(built.store)
    deleted = []
    monkeypatch.setattr(S3Store, "delete", lambda self, state: deleted.append(state))
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")
    s3.refuse(b'"training.checkpoint_deleted"')
    with pytest.raises(SystemExit) as stopped:
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model",
              "--checkpoint-every", "1", "--run", "fox-1"])
    assert stopped.value.code == 3 and deleted == []
    assert checkpoint(s3.store, "fox-1") is not None


@pytest.mark.parametrize("error", [PermissionError(13, "Permission denied"), KeyboardInterrupt()],
                         ids=["refused", "interrupted"])
def test_an_earlier_folder_that_cannot_be_removed_is_left_beside_with_a_warning(built, tmp_path, monkeypatch, capsys,
                                                                                error):
    # the new model is in place and complete: an earlier folder that the
    # system will not remove must not fail the run, but must be named; a
    # Ctrl+C during its removal names it the same way, and still stops
    main(["d", built.corpus_id, "--out", "model"])
    real = shutil.rmtree

    def refuse_the_replaced_one(path, *args, **options):
        if "-replaced-" in Path(path).name:
            raise error
        return real(path, *args, **options)

    monkeypatch.setattr(shutil, "rmtree", refuse_the_replaced_one)
    with pytest.raises(KeyboardInterrupt) if isinstance(error, KeyboardInterrupt) else contextlib.nullcontext():
        main(["d", built.corpus_id, "--out", "model"])

    [left] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-replaced-")]
    assert (left / "model.safetensors").exists() and (tmp_path / "model" / "model.safetensors").exists()
    run = built.records("wpt train")[-1]
    assert run.end["exit_code"] == (130 if isinstance(error, KeyboardInterrupt) else 0)
    assert [e["attributes"]["folder"] for e in run.events if e.get("name") == "model.replaced_folder_left"] == [
        left.name]  # as --out named the folder: relative to the working folder
    assert left.name in capsys.readouterr().err


def test_an_earlier_folder_is_put_back_when_the_new_one_cannot_take_its_place(built, tmp_path, monkeypatch):
    # moving the earlier folder aside and the new one in are one step: when
    # the second fails, the target must be the earlier folder again
    main(["d", built.corpus_id, "--out", "model"])
    before = {path.name: path.read_bytes() for path in (tmp_path / "model").iterdir() if path.is_file()}
    real = os.replace

    def refuse_the_new_one(source, target):
        if Path(target).name == "model" and not Path(source).name.startswith(".model-replaced-"):
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_new_one)
    with pytest.raises(PermissionError):
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    assert {path.name: path.read_bytes() for path in (tmp_path / "model").iterdir() if path.is_file()} == before
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".model")] == []


def test_an_earlier_folder_that_cannot_be_put_back_is_named_with_the_new_one(built, tmp_path, monkeypatch):
    # when the earlier folder cannot be moved back either, the target is
    # missing: neither model may be deleted, and the error names both
    from wpt.train import RestoreFailed

    main(["d", built.corpus_id, "--out", "model"])
    earlier = (tmp_path / "model" / "model.safetensors").read_bytes()
    real = os.replace

    def refuse_every_move_into_the_target(source, target):
        if Path(target).name == "model":
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_every_move_into_the_target)
    with pytest.raises(RestoreFailed) as failed:
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    assert not (tmp_path / "model").exists() and str(failed.value.aside) in str(failed.value)
    assert (failed.value.aside / "model.safetensors").read_bytes() == earlier
    assert json.loads((failed.value.new / "provenance.json").read_text())["recipe_name"] == "three"


def test_a_trained_model_folder_names_its_licenses_and_the_texts_it_learned_from(built, tmp_path):
    # whoever receives the folder, or only a weight file, must learn the licenses,
    # and each text the model learned from, with its notice and license text
    main(["d", built.corpus_id, "--out", "model"])
    folder = tmp_path / "model"

    credits = (folder / "CREDITS").read_text()
    learned = credits.split("\n\n", 1)[0].splitlines()
    texts = {(one["depot"], one["id"]): one
             for one in corpora.read_table(built.store, corpora.find(built.store, built.corpus_id), "texts").to_pylist()}
    assert len(learned) == len(texts) == 16  # the step read every token of the plain texts
    assert all(line.startswith(f"{built.depot.name()} ") and "; MIT; notice " in line for line in learned)
    assert f"\n{built.notice}\n" in credits and "MIT (MIT License)\nCopyright (c) 2026 Ada Writer" in credits
    notice = (folder / "NOTICE").read_text()
    assert "Weights: AGPL-3.0-or-later" in notice and "Generated text: CC0-1.0" in notice
    snapshot = corpora.find(built.store, built.corpus_id)["snapshot"]
    trained_on = (f"Trained on snapshot d (change {snapshot['change']}, digest {snapshot['digest']}) of the depot "
                  f"{built.depot.name()}, recipe d, code at commit ")
    assert trained_on in notice and f"corpus {built.corpus_id}:" in notice
    assert credits.rstrip("\n") in notice
    assert notice.endswith((ROOT / "OUTPUT-NOTICE").read_text())
    assert (folder / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()
    card = (folder / "README.md").read_text()
    assert card.startswith("---\nlicense: agpl-3.0\n---\n") and "Weights: AGPL-3.0-or-later" in card
    assert built.corpus_id in card and f"- {trained_on}" in card and "None yet: run `wpt eval" in card
    assert json.loads((folder / "provenance.json").read_text())["texts"] == len(learned)
    with safe_open(folder / "model.safetensors", "pt") as weights:
        assert weights.metadata()["license"] == "AGPL-3.0-or-later"
        assert weights.metadata()["output_license"] == "CC0-1.0"


def test_a_damaged_shard_stops_training_before_it_starts(built, tmp_path, capsys):
    store, corpus_id = built.store, built.corpus_id
    record = corpora.find(store, corpus_id)
    path = store.path(shard_file(record["shards"]["plain"][0]))  # the shard's own bytes, whether the store keeps it compressed or not
    path.write_bytes(path.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(SystemExit) as refused:
        main(["d", corpus_id, "--out", "model"])
    assert refused.value.code == 2 and "CorruptFile" in capsys.readouterr().err  # 2: never restarted
    assert not (tmp_path / "model").exists()


def test_a_corpus_of_another_snapshot_or_other_shares_than_the_recipe_s_is_refused(built, tmp_path, capsys):
    # the model must train on what its recipe names, so a corpus of another
    # snapshot, or of the same one with other shares, stops the run before it starts
    other = built.snapshot("other")
    with pytest.raises(SystemExit) as refused:
        main(["d", other, "--out", "model"])
    said = bare(capsys.readouterr().err)
    assert refused.value.code == 2 and "was built from snapshot other" in said and "this recipe names snapshot d " in said
    folder = tmp_path / "data" / "recipes" / "models"
    recipe = json.loads((folder / "d.json").read_text())
    recipe["data"]["shares"] = [{"tag": "en", "share": 0.5}]
    (folder / "half.json").write_text(json.dumps(recipe))
    with pytest.raises(SystemExit) as refused:
        main(["half", built.corpus_id, "--out", "model"])
    assert refused.value.code == 2 and "with kinds, shares or vocabulary other than the recipe's" in capsys.readouterr().err
    assert not (tmp_path / "model").exists()


def weights(folder):
    return load_file(folder / "model.safetensors")


def test_two_processes_train_the_same_model_as_one(built, on_the_cpu, tmp_path):
    # the batch is global: 2 processes of 2 windows each, or 1 process adding up
    # 2 passes of 2 windows, must take the same steps, dropout included
    recipe = built.recipe("four", training=one_stage(4))
    for out, command in [("one", ["-m", "wpt", "train", "--micro-batch", "2"]), ("two", ["-m", "wpt", "train", "--processes", "2", "--strategy", "ddp"])]:
        forget_steps(built.store)  # the hardware is no part of a step's fingerprint: each run trains
        subprocess.run([sys.executable, *command, recipe, built.corpus_id, "--out", out], cwd=tmp_path, check=True)

    one, two = weights(tmp_path / "one"), weights(tmp_path / "two")
    assert one.keys() == two.keys()
    # gloo adds gradients in varying order: up to 4e-6 apart; other dropout gives 2e-3
    assert all(torch.allclose(one[k], two[k], atol=1e-4) for k in one)
    provenance = json.loads((tmp_path / "two" / "provenance.json").read_text())
    assert provenance["strategy"]["processes"] == 2


def test_split_windows_in_segments_train_the_same_model_as_one_process(built, on_the_cpu, tmp_path):
    # docs/training.md, "Long windows": 2 processes split each 512-token
    # window, each running its slice in segments of 128 tokens, one window
    # per pass; the batch of 2 windows takes 2 passes in each process, and
    # the model is the one a single process trains on whole windows
    recipe = built.recipe("split", training=one_stage(2, 512, 2))
    for out, options in [("one", []), ("two", ["--processes", "2", "--strategy", "split", "--micro-batch", "1",
                                              "--segment", "128"])]:
        forget_steps(built.store)
        subprocess.run([sys.executable, "-m", "wpt", "train", recipe, built.corpus_id, "--out", out, *options],
                       cwd=tmp_path, check=True)

    one, two = weights(tmp_path / "one"), weights(tmp_path / "two")
    assert one.keys() == two.keys()
    assert all(torch.allclose(one[k], two[k], atol=1e-4) for k in one)
    provenance = json.loads((tmp_path / "two" / "provenance.json").read_text())
    assert provenance["strategy"]["processes"] == 2


def crash(capsys, argv):
    """Run argv, whose gpu worker's job crashes with RuntimeError('power
    cut'): the command ends with exit code 1 and the job's traceback on its
    error output."""
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 1 and "RuntimeError: power cut" in capsys.readouterr().err


def interrupting(built, tmp_path, at_step_3, at_step=3, best=None):
    """The recipe and options of a 5-update run on the CPU whose optimizer
    does at_step_3 (Python code) at update 3 (or at_step), just before the
    update; with best (an update), it judges only that update's evaluation
    the best."""
    strategy = tmp_path / "cpu.py"
    strategy.write_text("import torch\nfrom wpt_training import Strategy\n\n"
                        "def start():\n    return Strategy(torch.device('cpu'))\n")
    method = tmp_path / "interrupting.py"
    method.write_text(
        "import os\nimport signal\nimport torch\nfrom wpt_training import Optimizer, Verdict\n\n"
        "SETTINGS = {'lr': 0.01}\n\n"
        "class Interrupting(Optimizer):\n"
        "    def __init__(self, model, settings):\n"
        "        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], "
        "lr=settings['lr'])\n\n"
        "    def before_update(self, step):\n"
        f"        if step == {at_step}:\n"
        f"            {at_step_3}\n\n"
        + ("" if best is None else
           "    def evaluated(self, step, val_loss):\n"
           f"        return Verdict(best=step == {best}, stop=False)\n\n")
        + "def start(model, settings, run):\n    return Interrupting(model, settings)\n")
    return [built.recipe("interrupting", training={**one_stage(5), "optimizer": str(method)}), built.corpus_id,
            "--strategy", str(strategy)]


def test_a_crashed_run_resumes_where_its_checkpoint_left_it(built, tmp_path, monkeypatch, capsys):
    # after a crash, --resume must give the same model as a run never interrupted
    options = [*interrupting(built, tmp_path, "if os.environ.get('CRASH'): raise RuntimeError('power cut')"),
               "--checkpoint-every", "2"]
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")  # the checkpoint the crash leaves is on this machine
    main([*options, "--out", "whole"])
    forget_steps(built.store)

    monkeypatch.setenv("CRASH", "1")
    crash(capsys, [*options, "--out", "resumed"])
    assert (tmp_path / "resumed" / "checkpoint" / "checkpoint.safetensors").is_file()
    with pytest.raises(SystemExit) as refused:
        main([*options, "--out", "resumed"])
    assert refused.value.code == 2 and "--resume" in capsys.readouterr().err
    monkeypatch.delenv("CRASH")
    main([*options, "--out", "resumed", "--resume"])

    whole, resumed = weights(tmp_path / "whole"), weights(tmp_path / "resumed")
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    assert not (tmp_path / "resumed" / "checkpoint").exists()


def staged(built, tmp_path, crash_at, best=None):
    """The options of a run of two stages on the CPU (docs/training.md,
    "Stages"): 2 updates of 4 windows of 16 tokens, then 4 updates of 2
    windows of 256 tokens, whose optimizer raises at update crash_at when
    CRASH is set."""
    options = interrupting(built, tmp_path, "if os.environ.get('CRASH'): raise RuntimeError('power cut')", crash_at,
                           best)
    options[0] = built.recipe("staged", training={
        "stages": [{"block_size": 16, "steps": 2, "batch_size": 4}, {"block_size": 256, "steps": 4, "batch_size": 2}],
        "optimizer": str(tmp_path / "interrupting.py")})
    return [*options, "--checkpoint-every", "2"]


def stage_spans(run):
    return [event["attributes"] for event in run.events
            if event["kind"] == "span_start" and event["name"] == "training.stage"]


def step_windows(run):
    return [(event["attributes"]["step"], event["attributes"]["block_size"]) for event in run.events
            if event.get("name") == "training.step"]


def test_stages_train_in_order_and_a_run_resumes_across_their_boundary(built, tmp_path, monkeypatch, capsys):
    # a crash at the first step of the second stage resumes from the
    # checkpoint the first stage's end wrote, and gives the model of a run
    # never interrupted (the long windows in segments of 128 tokens, as a
    # 16 GB GPU would run them); each stage draws its own batch
    options = staged(built, tmp_path, 3)
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")
    main([*options, "--out", "whole", "--segment", "128"])
    forget_steps(built.store)
    monkeypatch.setenv("CRASH", "1")
    crash(capsys, [*options, "--out", "resumed"])
    monkeypatch.delenv("CRASH")
    main([*options, "--out", "resumed", "--segment", "128", "--resume"])

    whole, resumed = weights(tmp_path / "whole"), weights(tmp_path / "resumed")
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    assert load_model(tmp_path / "resumed").config.block_size == 256  # the model's context: the last stage's
    provenance = json.loads((tmp_path / "resumed" / "provenance.json").read_text())
    assert provenance["steps"][0]["stages"] == [
        {"block_size": 16, "steps": 2, "batch_size": 4, "first_step": 1, "last_step": 2},
        {"block_size": 256, "steps": 4, "batch_size": 2, "first_step": 3, "last_step": 6}]

    uninterrupted, crashed, again = built.records("wpt train")
    assert [(s["stage"], s["block_size"], s["batch_size"], s["first_step"], s["last_step"])
            for s in stage_spans(uninterrupted)] == [(1, 16, 4, 1, 2), (2, 256, 2, 3, 6)]
    assert step_windows(uninterrupted) == [(1, 16), (2, 16), (3, 256), (4, 256), (5, 256), (6, 256)]
    assert [(s["stage"], s["resumed_at"]) for s in stage_spans(again)] == [(2, 3)]
    assert step_windows(again) == [(3, 256), (4, 256), (5, 256), (6, 256)]


def test_a_run_resumed_in_the_middle_of_a_later_stage_keeps_its_best_weights(built, tmp_path, monkeypatch, capsys):
    # the method judges only step 4's evaluation the best: a crash at step 5
    # resumes from step 4's checkpoint, inside the second stage, with that
    # best kept, so the folder holds the same weights from the same step as
    # a run never interrupted, and the resumed run trains steps 5 and 6 only
    options = staged(built, tmp_path, 5, best=4)
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")
    main([*options, "--out", "whole"])
    forget_steps(built.store)
    monkeypatch.setenv("CRASH", "1")
    crash(capsys, [*options, "--out", "resumed"])
    monkeypatch.delenv("CRASH")
    main([*options, "--out", "resumed", "--resume"])

    whole, resumed = weights(tmp_path / "whole"), weights(tmp_path / "resumed")
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    for out in ("whole", "resumed"):
        provenance = json.loads((tmp_path / out / "provenance.json").read_text())
        assert (provenance["steps"][0]["weights_update"], provenance["steps"][0]["updates"]) == (4, 6)
    uninterrupted, crashed, again = built.records("wpt train")
    assert [(s["stage"], s["first_step"], s["resumed_at"]) for s in stage_spans(again)] == [(2, 3, 5)]
    assert step_windows(again) == [(5, 256), (6, 256)]


def test_the_model_card_names_the_updates_trained_and_the_update_of_its_weights(built, tmp_path):
    # a published model says how far each step trained and which update its
    # weights come from, even when its optimizer stopped early and kept an
    # earlier update
    strategy = tmp_path / "cpu.py"
    strategy.write_text("import torch\nfrom wpt_training import Strategy\n\n"
                        "def start():\n    return Strategy(torch.device('cpu'))\n")
    method = tmp_path / "early.py"
    method.write_text(
        "import torch\nfrom wpt_training import Optimizer, Verdict\n\nSETTINGS = {}\n\n"
        "class Early(Optimizer):\n"
        "    def __init__(self, model):\n"
        "        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)\n\n"
        "    def evaluated(self, step, val_loss):\n"
        "        return Verdict(best=step == 2, stop=step == 4)\n\n"
        "def start(model, settings, run):\n    return Early(model)\n")
    recipe = built.recipe("early", training={**one_stage(10), "optimizer": str(method)})
    main([recipe, built.corpus_id, "--out", "model", "--strategy", str(strategy)])

    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert (provenance["steps"][0]["updates"], provenance["steps"][0]["weights_update"]) == (4, 2)
    card = (tmp_path / "model" / "README.md").read_text()
    assert "1. `read-text`: method next-token, optimizer early, environment text-windows; 4 updates (limit 10), " \
           "the weights of update 2" in card


def test_streamed_training_never_copies_the_whole_corpus_to_disk(built, tmp_path, monkeypatch):
    # --stream trains from a few token files at a time, so the gpu worker
    # needs disk only for those, and the model folder names its texts as usual
    monkeypatch.setenv("WPT_WORKER_FOLDER", str(tmp_path / "worker"))  # the local worker's, kept
    main([built.recipe("two", training=one_stage(2)), built.corpus_id, "--out", "model", "--stream",
          "--stream-shards", "1"])

    assert not list((tmp_path / "worker" / "job" / "scratch" / "tokens").glob("*.u32"))  # no copy of the tokens
    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["corpus"]["id"] == built.corpus_id and provenance["texts"] > 0
    assert load_model(tmp_path / "model").config.block_size == 16


def test_a_stop_saves_a_checkpoint_even_when_the_run_writes_none_during_a_step(built, tmp_path, monkeypatch):
    # a stop keeps what the machines did, whatever --checkpoint-every says
    options = [*interrupting(built, tmp_path, "signal.raise_signal(signal.SIGTERM)"), "--checkpoint-every", "0"]
    with pytest.raises(SystemExit) as stopped:
        main([*options, "--out", "here"])
    assert stopped.value.code == 130
    assert load_state(tmp_path / "here" / "checkpoint" / "checkpoint.safetensors")["step"] == 3


def test_a_run_stops_at_its_limit_prints_what_it_measured_and_resumes(built, tmp_path, capsys):
    # a short run to measure speed needs nobody to stop it: the speed and the
    # time of the whole run come from the updates after the first, which also
    # compiles the kernels, and the run goes on from its checkpoint
    command = [built.recipe("five", training=one_stage(5)), built.corpus_id, "--out", "model"]
    with pytest.raises(SystemExit) as stopped:
        main([*command, "--stop-after-updates", "3"])
    assert stopped.value.code == 130
    printed = capsys.readouterr().out
    assert "measured 2 updates in " in printed and " tokens per second" in printed
    assert "the run's 320 tokens of text windows take " in printed  # 5 updates of 4 windows of 16 tokens
    assert load_state(tmp_path / "model" / "checkpoint" / "checkpoint.safetensors")["step"] == 3
    [run] = built.records("wpt train")
    [measured] = [e for e in run.events if e.get("name") == "training.measured"]
    assert measured["values"]["updates"] == 2 and measured["values"]["tokens_per_second"] > 0
    assert measured["values"]["projected_hours_left"] == pytest.approx(measured["values"]["projected_hours"] * 2 / 5)
    assert run.end["exit_code"] == 130

    main([*command, "--resume"])
    assert (tmp_path / "model" / "config.json").is_file()
    assert [e["attributes"]["step"] for e in built.records("wpt train")[-1].events
            if e.get("name") == "training.resumed"] == [3]


def test_the_update_asked_for_is_profiled_into_the_record_and_training_goes_on(built, tmp_path):
    # the profile of update 2 lands in the run's record, whatever the device; a run that does not ask has none
    recipe = built.recipe("three", training=one_stage(3))
    main([recipe, built.corpus_id, "--out", "model", "--profile-update", "2"])
    [run] = built.records("wpt train")
    text = built.store.read_record(run.folder, "profile.txt").decode()
    assert text.startswith("update 2: ") and "seconds by part:" in text and "data (environment.batch)" in text
    assert "learn 1" in text and "optimizer step" in text and "everything after" in text
    assert "top 40 by self CPU time:" in text and "os.cpu_count()" in text
    assert load_model(tmp_path / "model").config.block_size == 16  # the run finished

    main([recipe, built.corpus_id, "--out", "other"])
    assert "profile.txt" not in built.store.record_files(built.records("wpt train")[-1].folder)


def test_a_store_that_fails_while_the_last_corpus_is_found_refuses_the_run_in_one_line(built, monkeypatch, capsys):
    def failing(store, key):
        raise StoreError("the bucket does not answer")

    monkeypatch.setattr(corpora, "of_key", failing)  # the run asks the store for the corpus of its build key
    with pytest.raises(SystemExit) as refused:
        main(["d"])
    assert refused.value.code == 2 and bare(capsys.readouterr().err) == "StoreError: the bucket does not answer\n"


def test_wpt_train_without_a_corpus_id_reads_the_corpus_of_its_build_key(built, tmp_path, capsys):
    # the id goes from the build to the training through the store, not through a person's hands: the recipe's
    # snapshot, data section and vocabulary make the build's key, which names the corpus's folder
    whole = built.snapshot("later")  # every token
    recipe = built.recipe("three", training=one_stage(3), snapshot="later")  # total_tokens: every token of later
    size = json.loads((tmp_path / "data" / "recipes" / "models" / "three.json").read_text())["data"]["total_tokens"]
    with pytest.raises(SystemExit) as refused:  # only the corpus of every token is built, not that size
        main([recipe, "--out", "model"])
    said = bare(capsys.readouterr().err)
    assert refused.value.code == 2 and f"no fresh corpus of snapshot later of the depot fox with the vocabulary " \
        f"{built.vocabulary} at the size {size:,}" in said and "press Build on the recipe page" in said

    sized = built.sized("later", size)["id"]
    main([recipe, "--out", "model"])
    assert json.loads((tmp_path / "model" / "provenance.json").read_text())["corpus"]["id"] == sized

    main([recipe, whole, "--out", "given"])  # a corpus id given wins
    assert json.loads((tmp_path / "given" / "provenance.json").read_text())["corpus"]["id"] == whole
    with pytest.raises(SystemExit) as refused:  # a start of an id that names no corpus
        main([recipe, "0" * 8, "--out", "other"])
    assert refused.value.code == 2 and "no corpus 00000000" in capsys.readouterr().err


def test_a_recipe_whose_snapshot_reads_another_change_or_digest_or_is_missing_is_refused(built, tmp_path):
    # the change and the digest prove which texts the snapshot holds: a recipe that says others names other texts
    from wpt.train import InputError, find_corpus, load_data
    data = load_data(built.recipe("three", training=one_stage(3)))
    built.sized("d", data["total_tokens"])
    assert find_corpus(built.store, None, data)["build_key"]
    snapshot = built.depot.snapshot("d")
    for wrong in ({"change": data["change"] + 1}, {"digest": "0" * 64}):
        with pytest.raises(InputError, match=f"snapshot d of depot fox reads change {snapshot['at_change']} with "
                                             f"digest {snapshot['digest']}; the recipe says "):
            find_corpus(built.store, None, {**data, **wrong})
    with pytest.raises(InputError, match="no snapshot 'never' of the depot 'fox'"):
        find_corpus(built.store, None, {**data, "snapshot": "never"})
    with pytest.raises(InputError, match="cannot be read"):  # a depot neither WPT_DEPOT nor a web address names
        find_corpus(built.store, None, {**data, "depot": "elsewhere"})


def test_a_recipe_whose_only_corpus_lists_its_token_files_in_one_list_is_told_the_build_that_makes_one(built):
    # without a corpus id, an old corpus of the build key must not pass for one never built
    from wpt.train import InputError, find_corpus, load_data
    data = load_data(built.recipe("old-layout", training=one_stage(3)))
    record = built.sized("d", data["total_tokens"])
    built.store.put_json(f"{corpora.folder_of(record)}/{corpora.RECORD}",
                         {**record, "shards": corpora.token_files(record)})

    with pytest.raises(InputError, match=r"does not list its token files by kind.*press Build on the recipe page"):
        find_corpus(built.store, None, data)


def test_a_recipe_written_before_snapshots_trains_on_its_dataset_s_corpus_and_its_folder_says_so(built, tmp_path):
    # a recipe of a dataset (the legacy shape) still trains on a corpus built from that dataset, given by its id,
    # and the model folder names that corpus, its dataset and its depot as such folders always did
    record = corpora.find(built.store, built.corpus_id)
    legacy = {key: value for key, value in record.items() if key not in ("build_key", "snapshot", "data")}
    legacy.update(id="1e9acy" + record["id"][6:], dataset={"id": "fox-1", "name": "classics", "revision": 1})
    built.store.put_json(f"{corpora.FOLDER}/classics-1e9acy00/{corpora.RECORD}", legacy)  # its files stay where they are
    folder = tmp_path / "data" / "recipes" / "models"
    recipe = json.loads((folder / "d.json").read_text())
    recipe["data"] = {"depot": built.depot.name(), "dataset": "classics", "total_tokens": record["total_tokens"]}
    (folder / "old.json").write_text(json.dumps(recipe))

    main(["old", legacy["id"][:12], "--out", "model"])

    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["corpus"] == {"id": legacy["id"], "depot": built.depot.name(), "dataset": "classics"}
    assert "data" not in provenance
    assert (f"From corpus {legacy['id']}, built from dataset classics of the depot {built.depot.name()}:"
            in (tmp_path / "model" / "NOTICE").read_text())
    with pytest.raises(SystemExit) as refused:  # a corpus of a snapshot is not a dataset's
        main(["old", built.corpus_id, "--out", "other"])
    assert refused.value.code == 2


def test_a_recipe_naming_its_depot_by_a_web_address_finds_its_snapshot_through_the_depot_s_web_app(
        built, monkeypatch):
    # a machine without the depot's file reads the snapshot from the depot's web app (its own name, GET /api/about,
    # and its snapshots, GET /api/snapshots), then the corpus of the build key from the store
    from wpt.train import InputError, find_corpus, load_corpus, recipe_depot
    address = "https://depot.example.org"
    answers = {"/api/about": {"about": built.depot.about()},
               "/api/snapshots": [{**row, "owner": row["owner"]} for row in built.depot.snapshots()]}
    asked = []
    monkeypatch.setattr(recipe_depot.WebDepot, "_json", lambda self, route: asked.append(route) or answers[route])
    monkeypatch.delenv("WPT_DEPOT")
    size = corpora.find(built.store, built.corpus_id)["total_tokens"]
    data = {**built.data("d", size), "depot": address}
    public = built.sized("d", size, url=address)

    record = find_corpus(built.store, None, data)

    assert record["id"] == public["id"] and record["depot"] == address and asked == ["/api/about", "/api/snapshots"]
    load_corpus(built.store, record, {"data": data})
    with pytest.raises(InputError, match="press Build on the recipe page"):
        find_corpus(built.store, None, {**data, "total_tokens": size - 1})  # that size is not built
    answers["/api/about"] = {"about": {**built.depot.about(), "name": "other"}}  # another depot's snapshots
    with pytest.raises(InputError, match="no snapshot 'd' of the depot 'https://depot.example.org' \\(other\\)"):
        find_corpus(built.store, None, data)


def test_a_stopped_run_saves_its_step_and_continues_on_another_machine(built, tmp_path, monkeypatch):
    # a machine taken back (SIGTERM) or Ctrl+C loses no step; with --run the
    # checkpoint is in the store, so another model folder, as on another machine
    # using the same store, continues it and gives the model of an unbroken run,
    # though WPT_LINK_CHECKPOINTS leaves it at its default, step-ends.
    # SIGTERM, not SIGINT: a shell starts background commands with SIGINT
    # ignored, and an ignored signal stays ignored
    options = [*interrupting(built, tmp_path, "if os.environ.get('STOP'): signal.raise_signal(signal.SIGTERM)"),
               "--checkpoint-every", "100"]
    main([*options, "--out", "whole"])
    forget_steps(built.store)

    monkeypatch.setenv("STOP", "1")
    with pytest.raises(SystemExit) as stopped:
        main([*options, "--out", "here", "--run", "fox-1"])
    monkeypatch.delenv("STOP")
    assert stopped.value.code == 130 and not (tmp_path / "here" / "checkpoint").exists()
    saved = checkpoint(built.store, "fox-1")
    assert load_state(built.store.path(saved))["step"] == 3  # not a multiple of 100: the signal
    main([*options, "--out", "there", "--run", "fox-1", "--resume"])

    whole, there = weights(tmp_path / "whole"), weights(tmp_path / "there")
    assert all(torch.equal(whole[k], there[k]) for k in whole)
    assert checkpoint(built.store, "fox-1") is None and not built.store.has(saved)


def test_a_crashed_run_starts_again_on_its_own(built, on_the_cpu, tmp_path):
    # a long run must survive a crash nobody watches: the processes start again
    # and continue from the checkpoint, without anyone typing --resume
    options = [*interrupting(
        built, tmp_path, "if os.environ.get('CRASH') and os.environ['TORCHELASTIC_RESTART_COUNT'] == '0': "
                         "raise RuntimeError('power cut')"), "--checkpoint-every", "2"]
    subprocess.run([sys.executable, "-m", "wpt", "train", *options, "--out", "whole"], cwd=tmp_path, check=True)
    forget_steps(built.store)
    run = subprocess.run([sys.executable, "-m", "wpt", "train", *options, "--out", "restarted"],
                         cwd=tmp_path, env={**os.environ, "CRASH": "1", "WPT_MAX_RESTARTS": "1"},
                         capture_output=True, text=True)

    assert run.returncode == 0, run.stderr
    assert "power cut" in run.stderr and "starting every process again (1 of 1 restarts)" in run.stderr
    assert "read-text: resumed from update 2" in run.stdout
    whole, restarted = weights(tmp_path / "whole"), weights(tmp_path / "restarted")
    assert all(torch.equal(whole[k], restarted[k]) for k in whole)
    # one command, one record: both attempts and the restart between them
    record = built.records("wpt train")[-1]
    assert record.start["arguments"][-2:] == ["--out", "restarted"] and record.end["status"] == "succeeded"
    assert [e["attributes"]["step"] for e in record.events if e.get("name") == "training.step"] == [1, 2, 3, 4, 5]
    assert any("starting every process again" in e.get("text", "") for e in record.events)
    assert record.ids["outputs"]["weights"] == {
        "model.safetensors": hashlib.sha256((tmp_path / "restarted" / "model.safetensors").read_bytes()).hexdigest()}


def test_a_new_model_that_cannot_take_an_empty_place_is_kept_and_named(built, tmp_path, monkeypatch, capsys):
    # with no earlier folder, a refused rename must not delete the only copy of the new model; the checkpoint
    # lives in the store (--run), so no checkpoint folder takes the target's place first
    real = os.replace

    def refuse_the_target(source, target):
        if Path(target).name == "model":
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_target)
    with pytest.raises(PermissionError):
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model", "--run", "fox-2"])
    [kept] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-")]
    assert not (tmp_path / "model").exists()
    assert json.loads((kept / "provenance.json").read_text())["recipe_name"] == "three"
    assert str(kept) in capsys.readouterr().err


def test_a_new_model_that_cannot_take_the_place_of_its_own_checkpoint_folder_is_kept_and_named(built, tmp_path,
                                                                                             monkeypatch, capsys):
    # without --run, the target holds only what this run's checkpoint left
    # there: a refused rename keeps the new model, which has no other copy,
    # and names it
    real = os.replace

    def refuse_the_new_one(source, target):
        if Path(target).name == "model" and not Path(source).name.startswith(".model-replaced-"):
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_new_one)
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")  # the step's last checkpoint comes into the model folder
    with pytest.raises(PermissionError):
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    assert list((tmp_path / "model").iterdir()) == []
    [kept] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-")]
    assert json.loads((kept / "provenance.json").read_text())["recipe_name"] == "three"
    assert f"it is kept in {kept}" in capsys.readouterr().err
    monkeypatch.setattr(os, "replace", real)
    main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])  # reuses the step
    assert (tmp_path / "model" / "config.json").exists()
    assert kept.is_dir()  # the run wrote its own folder: the kept one is left to remove by hand


def test_a_restore_failure_inside_a_broken_record_still_keeps_both_models(built, tmp_path, monkeypatch, capsys):
    # a broken record turns RestoreFailed into RecordError on its way out of
    # the span: the folders, not the error's type, decide what is kept
    import wpt.train
    from wpt_telemetry import RecordError

    main(["d", built.corpus_id, "--out", "model"])
    earlier = (tmp_path / "model" / "model.safetensors").read_bytes()
    real_replace, real_folder = os.replace, wpt.train._replace_folder

    def refuse_every_move_into_the_target(source, target):
        if Path(target).name == "model":
            raise PermissionError(13, "Permission denied", str(source))
        real_replace(source, target)

    def replace_while_the_record_breaks(new, target, telemetry):
        try:
            real_folder(new, target, telemetry)
        except wpt.train.RestoreFailed as failed:
            raise RecordError("the record broke too") from failed

    monkeypatch.setattr(os, "replace", refuse_every_move_into_the_target)
    monkeypatch.setattr(wpt.train, "_replace_folder", replace_while_the_record_breaks)
    with pytest.raises(SystemExit) as stopped:
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", "model"])
    assert stopped.value.code == 3 and not (tmp_path / "model").exists()
    [aside] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-replaced-")]
    [new] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-") and path != aside]
    assert (aside / "model.safetensors").read_bytes() == earlier
    assert json.loads((new / "provenance.json").read_text())["recipe_name"] == "three"
    err = capsys.readouterr().err
    assert str(new) in err and str(aside) in err


def test_only_each_step_s_end_comes_back_from_the_gpu_worker_unless_every_checkpoint_is_asked(built, tmp_path,
                                                                                              monkeypatch, capsys):
    # by default (an empty value too, as a copied .env.example gives) the
    # store gets no checkpoint during a step, only the step's result at its
    # end; WPT_LINK_CHECKPOINTS=every sends each checkpoint; another value is
    # refused before the gpu worker starts
    def checkpoints_set(out, mode):
        monkeypatch.setenv("WPT_LINK_CHECKPOINTS", mode)
        forget_steps(built.store)
        main([built.recipe("three", training=one_stage(3)), built.corpus_id, "--out", out, "--checkpoint-every",
              "1", "--run", "fox-1"])
        run = built.records("wpt train")[-1]
        assert [e["attributes"]["step"] for e in run.events
                if e["kind"] == "span_start" and e["name"] == "training.step_kept"] == ["read-text"]
        return len([e for e in run.events
                    if e.get("name") == "training.checkpoint_saved"])

    assert checkpoints_set("ends", "") == 0
    assert checkpoints_set("every", "every") == 3

    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "sometimes")
    with pytest.raises(SystemExit) as refused:
        main(["d", built.corpus_id, "--out", "model"])
    assert refused.value.code == 2 and "WPT_LINK_CHECKPOINTS='sometimes': one of every, step-ends" in \
        capsys.readouterr().err


def refusal(capsys, argv) -> str:
    """What wpt train wrote on its error output when it refused argv (exit code 2)."""
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2
    return capsys.readouterr().err


def test_a_stale_corpus_is_refused_until_it_is_built_again(built, tmp_path, capsys):
    # an admitted erasure reached a text the corpus holds (docs/erasure.md): training on it would learn the erased passage
    record = corpora.find(built.store, built.corpus_id)
    text = corpora.read_table(built.store, record, "texts").to_pylist()[0]["id"]
    corpora.mark_stale(built.store, [text], [f"{built.depot.name()}-1"])

    said = refusal(capsys, [built.recipe("stale"), built.corpus_id, "--out", "model"])
    assert "is stale" in said
    assert "press Build on the recipe page" in said
    assert not (tmp_path / "model").exists()


def test_a_corpus_that_does_not_list_its_token_files_by_kind_is_refused_with_the_build_that_makes_one():
    # each step reads the token files of the kinds it names: a corpus of one list cannot say which are which; no
    # build makes a corpus of a dataset, so the message says to give the recipe a snapshot
    record = {"id": "old", "dataset": {"id": "d", "name": "classics"}, "shards": [{"path": "shard-0.bin"}],
              "total_tokens": 10}

    with pytest.raises(InputError, match=r"does not list its token files by kind.*no build makes a corpus of a "
                                         r"dataset.*press Build on the recipe page"):
        check_record(record, {"data": {"total_tokens": 10}})


def test_a_corpus_with_fewer_tokens_than_the_recipe_s_total_is_refused(built, tmp_path, capsys):
    record = corpora.find(built.store, built.corpus_id)
    recipe = json.loads((tmp_path / "data" / "recipes" / "models" / "d.json").read_text())
    recipe["data"]["total_tokens"] = record["total_tokens"] + 1
    (tmp_path / "data" / "recipes" / "models" / "long.json").write_text(json.dumps(recipe))

    assert "holds" in refusal(capsys, ["long", built.corpus_id, "--out", "model"])  # built at no size


def test_a_corpus_built_at_the_recipe_s_size_trains_on_what_it_holds_and_the_record_says_how_much(built, tmp_path):
    # total_tokens is the size the recipe asks for and a ceiling: a corpus built at that size never passes it, so
    # it may hold fewer tokens of plain text, and training reads them all, found by its triple or given by its id
    whole = corpora.find(built.store, built.corpus_id)["total_tokens"]
    asked = whole + 500  # more than the dataset holds: the corpus built at that size holds every token, fewer
    folder = tmp_path / "data" / "recipes" / "models"
    recipe = json.loads((folder / "d.json").read_text())
    recipe["data"]["total_tokens"] = asked
    (folder / "ceiling.json").write_text(json.dumps(recipe))
    sized = built.sized("d", asked)
    assert sized["max_tokens"] == asked and sized["total_tokens"] == whole

    main(["ceiling", "--out", "model"])

    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["corpus"]["id"] == sized["id"] and provenance["learned"]["plain"] == whole
    warned = [event["text"] for event in built.records("wpt train")[-1].events
              if event["kind"] == "log" and event["severity"] == "warning"]
    assert any(f"step 1 read-text reads the first {whole:,} tokens of plain text, all that corpus {sized['id']} "
               f"holds, not the {asked:,} it asks for" in text for text in warned)
    main(["ceiling", sized["id"], "--out", "given"])  # given by its id, the same rule
    assert json.loads((tmp_path / "given" / "provenance.json").read_text())["learned"]["plain"] == whole


def test_each_size_reads_the_smaller_of_its_prefix_and_what_a_corpus_built_at_the_recipe_s_size_holds():
    # a size of a series sets its own prefix; a corpus built at the series' size reads at most what it holds
    from wpt.train import with_prefixes

    def stepped(prefix):
        return {"step": "read-text", "settings": {"stages": [{"block_size": 16, "steps": 1, "batch_size": 1}],
                                                  "prefix_tokens": prefix}}
    recipe = {"data": {"total_tokens": 5000}, "steps": [stepped(4000), stepped(800)]}
    record = {"id": "c", "total_tokens": 900, "dataset": {"id": "d"}, "tokenizer": {"sha256": "0" * 64}}
    said = []

    at_size = with_prefixes(recipe, {**record, "max_tokens": 5000}, said.append)
    other = with_prefixes(recipe, {**record, "max_tokens": None}, said.append)

    assert [one["settings"]["prefix_tokens"] for one in at_size["steps"]] == [900, 800]
    assert len(said) == 1 and said[0].startswith("step 1 read-text reads the first 900 tokens")
    assert [one["settings"]["prefix_tokens"] for one in other["steps"]] == [4000, 800]  # check_corpus refuses it


def test_a_plain_recipe_trains_on_the_first_total_tokens_of_its_corpus(built, tmp_path):
    folder = tmp_path / "data" / "recipes" / "models"
    recipe = json.loads((folder / "d.json").read_text())
    recipe["data"]["total_tokens"] = 3000  # fewer than the corpus holds
    (folder / "short.json").write_text(json.dumps(recipe))

    main(["short", built.corpus_id, "--out", "model"])

    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["learned"]["plain"] == 3000 < corpora.find(built.store, built.corpus_id)["total_tokens"]


def test_recompute_from_narrows_recompute_and_is_refused_alone(capsys):
    # docs/training.md, "Recomputing layers": --recompute-from (or WPT_RECOMPUTE_FROM) goes with --recompute
    from wpt.train import parse_args

    assert parse_args(["r", "--recompute", "--recompute-from", "8192"], environ={}).recompute_from == 8192
    assert parse_args(["r", "--recompute"], environ={"WPT_RECOMPUTE_FROM": "4096"}).recompute_from == 4096
    assert parse_args(["r"], environ={}).recompute_from == 0
    for refused in (["r", "--recompute-from", "8192"], ["r", "--recompute", "--recompute-from", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(refused, environ={})
    said = capsys.readouterr().err
    assert "give --recompute too" in said and "0 (every stage) or a number of tokens" in said
