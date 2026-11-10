"""The wpt manager (docs/machines.md): one command that trains a recipe on
its corpus, already built into the store, through a local gpu worker, and
keeps the model in the store under the run's name."""

import base64
import json

import pytest

from wpt_depot import NewText
from wpt_depot import corpus as corpora
from wpt.manager import RUN_FILE, main, mount_point, task_images
from wpt.store import main as store_command
from wpt_telemetry import file_hashes


def test_the_manager_refuses_to_start_without_a_recipe(built, monkeypatch, capsys):
    monkeypatch.delenv("WPT_RECIPE", raising=False)
    with pytest.raises(SystemExit) as refused:
        main([])
    assert refused.value.code == 2 and "WPT_RECIPE is not set" in capsys.readouterr().err
    assert not built.records("wpt manager")


def test_one_command_trains_on_the_recipe_s_corpus_and_keeps_the_model_under_the_run_s_name(built, tmp_path,
                                                                                            monkeypatch, capsys):
    # with no run name, the manager makes one and writes it, with the
    # command to copy, into the git-ignored file; the model it trained comes
    # back out of the store as the folder wpt train wrote, file for file,
    # and the manager's own copy of that folder is gone; it trains on the
    # corpus last built for the recipe's dataset
    monkeypatch.setenv("WPT_RECIPE", "d")
    monkeypatch.delenv("WPT_RUN", raising=False)
    main([])
    [copy_out] = [line for line in (tmp_path / RUN_FILE).read_text().splitlines() if not line.startswith("#")]
    name = copy_out.split()[5]
    assert name.startswith("d-")
    assert copy_out == f"uv run wpt store copy-out {name} runs/{name}"
    assert copy_out in capsys.readouterr().out

    assert not (tmp_path / "runs" / "manager" / name).exists()
    assert store_command(["copy-out", name, str(tmp_path / "copy")]) == 0
    [train] = built.records("wpt train")
    [written] = [e["attributes"]["files"] for e in train.events if e.get("name") == "model.files"]
    assert file_hashes(tmp_path / "copy") == written
    assert (tmp_path / "copy" / "config.json").is_file()
    assert (tmp_path / "copy" / "drafter" / "model.safetensors").is_file()

    [run] = built.records("wpt manager")
    assert run.end["status"] == "succeeded"
    assert [e["name"] for e in run.events if e["kind"] == "span_start" and e["name"].startswith("manager.")] == [
        "manager.corpus", "manager.workers", "manager.train", "manager.model"]
    assert [e["attributes"]["model"] for e in run.events if e.get("name") == "store.model_kept"] == [name]
    assert train.start["arguments"][:2] == ["d", built.corpus_id] and "--run" in train.start["arguments"]


def test_a_corpus_the_settings_name_is_trained_on(built, monkeypatch):
    other = built.dataset("later")  # the corpus last built, of another dataset than the recipe's
    monkeypatch.setenv("WPT_RECIPE", "d")
    monkeypatch.setenv("WPT_CORPUS", built.corpus_id[:12])
    main([])
    [run] = built.records("wpt manager")
    assert run.end["status"] == "succeeded"
    [train] = built.records("wpt train")
    assert built.corpus_id in train.start["arguments"] and other not in train.start["arguments"]


def test_a_recipe_whose_corpus_is_not_built_stops_the_run_before_anything_starts(built, monkeypatch, capsys):
    built.recipe("unbuilt", dataset="unbuilt")
    monkeypatch.setenv("WPT_RECIPE", "unbuilt")
    with pytest.raises(SystemExit) as refused:
        main([])
    err = capsys.readouterr().err
    assert refused.value.code == 2 and "build it with `wpt depot build unbuilt`" in err
    [run] = built.records("wpt manager")
    assert [e["name"] for e in run.events if e["kind"] == "span_start" and e["name"].startswith("manager.")] == [
        "manager.corpus"]
    assert built.records("wpt train") == []


def test_the_images_to_build_are_those_the_corpus_s_tasks_name_each_once(built):
    def task(image):
        check = {"mode": "100755", "content": base64.b64encode(b"#!/bin/sh\necho 1\n").decode()}
        return NewText(kind="task", content=json.dumps({"task": {
            "instructions": "Say hello.", "image": image, "needs": [],
            "limits": {"max_turns": 2, "max_tokens": 6, "command_seconds": 5}}, "files": {}, "check": check}))

    assert task_images(built.store, corpora.find(built.store, built.corpus_id)) == []
    found = built.dataset("tasks", [task("tiny"), task("agent"), task("tiny")])
    assert task_images(built.store, corpora.find(built.store, found)) == ["agent", "tiny"]


def test_the_manager_reads_its_settings_from_dot_env_when_the_environment_leaves_them_unset(built, tmp_path,
                                                                                          monkeypatch, capsys):
    monkeypatch.setenv("WPT_RECIPE", "")  # empty, as Compose passes an unset setting: the .env file's counts
    (tmp_path / ".env").write_text("WPT_RECIPE=no-such-recipe\n", encoding="utf-8")
    with pytest.raises(SystemExit) as refused:
        main([])
    assert refused.value.code == 2 and "no-such-recipe" in capsys.readouterr().err


def test_a_store_folder_counts_as_mounted_only_under_a_mount_point_other_than_the_root():
    # /proc/self/mountinfo's fifth field is the mount point, with a space written as \040
    mountinfo = ("1 0 0:1 / / rw - overlay overlay rw\n"
                 "2 1 0:2 / /store rw - ext4 /dev/a rw\n"
                 "3 1 0:3 / /my\\040repo rw - ext4 /dev/b rw\n")
    assert mount_point("/store/objects", mountinfo) == "/store"
    assert mount_point("/my repo/store", mountinfo) == "/my repo"
    assert mount_point("/home/user/wpt-store", mountinfo) == "/"  # in the container's own layer
    assert mount_point("/storehouse", mountinfo) == "/"
