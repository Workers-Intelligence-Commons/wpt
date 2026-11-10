"""The steps that read conversations and tasks (docs/training.md), run
through wpt train on a corpus that holds them."""

import base64
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

import wpt_agent
import wpt.train
from wpt_agent import images, played
from wpt_depot import NewText
from wpt_depot import corpus as corpora
from wpt_model import PARTS
from wpt_store import LocalStore
from wpt_telemetry import RECORD_FAILED, Telemetry
from wpt_tokenizer import chat
from wpt_training import Context, Data, Strategy, in_process, load_step, step_seed
from wpt_training import agentic
from wpt.train import main as train


def one_stage(steps, block_size=16, batch_size=4):
    """A read-text step of one stage (docs/training.md, "Stages")."""
    return {"stages": [{"block_size": block_size, "steps": steps, "batch_size": batch_size}]}


def refused(capsys, argv) -> str:
    """What wpt train wrote on its error output when it refused argv (exit code 2)."""
    with pytest.raises(SystemExit) as stopped:
        train(argv)
    assert stopped.value.code == 2
    return capsys.readouterr().err


TALK = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}]}


def talks(built, name="chat", count=1) -> str:
    """The id of the corpus of a new dataset, name, of the depot's texts and
    count more conversations, each TALK."""
    return built.dataset(name, [NewText(kind="conversation", content=json.dumps(TALK), tags=("answer-as-assistant",))] * count)


def test_a_step_learns_only_the_conversations_carrying_its_tag(built, capsys):
    # the corpus holds a conversation, tagged for another use: the step picks items by tag from the corpus's texts
    other = built.dataset("other", [NewText(kind="conversation", content=json.dumps(TALK), tags=("review",))])
    chat_step = [{"step": "answer-as-assistant"}]
    err = refused(capsys, [built.recipe("other", training=one_stage(1, 64), steps=chat_step, dataset="other"), other,
                           "--out", "model"])
    assert "no conversations: the corpus holds none tagged answer-as-assistant" in err
    tagged = [{"step": "answer-as-assistant", "settings": {"tag": "review"}}]
    train([built.recipe("base", training=one_stage(1, 64), dataset="other"), other, "--out", "base"])
    train([built.recipe("tagged", training=one_stage(1, 64), steps=tagged, dataset="other"), other, "--out", "model"])


def credited(folder) -> list[str]:
    """The lines of the texts a model folder's CREDITS names."""
    return (folder / "CREDITS").read_text().split("\n\n", 1)[0].splitlines()


def test_answer_as_assistant_learns_from_the_corpus_s_conversations_and_credits_them(built, tmp_path, monkeypatch,
                                                                                     capsys):
    # the step learns the assistant's turns of the conversations of its
    # corpus; a corpus without any is refused before any step trains. A recipe
    # that shares its first step with an earlier run reuses that step's
    # result; the step changes the weights, the CSA indexers learning by
    # their own loss, but never the routers' balancing biases, which it
    # locks; the model's CREDITS name the conversation, and its drafter's
    # answers stop at the end of a turn too
    chat_step = [{"step": "answer-as-assistant"}]
    err = refused(capsys, [built.recipe("no-talk", training=one_stage(1, 64), steps=chat_step), built.corpus_id,
                           "--out", "model"])
    assert "InputError: step 2 answer-as-assistant: no conversations: the corpus holds none tagged answer-as-assistant" in err
    assert not (tmp_path / "model").exists() and not (built.store.root / "cache" / "steps").exists()

    dataset = talks(built)
    assert corpora.find(built.store, dataset)["conversations"] == 1
    recipe = built.recipe("chat", training=one_stage(1, 64), steps=chat_step, dataset="chat")
    train([built.recipe("base", training=one_stage(1, 64), dataset="chat"), dataset, "--out", "base"])

    train([recipe, dataset, "--out", "model"])
    [drafter] = [event for event in built.records("wpt train")[-1].events
                 if event["kind"] == "span_start" and event["name"] == "drafter"]
    assert drafter["attributes"]["stops"] == [wpt.train.Tokenizer.END, chat.IM_END]
    assert "step 1 read-text: reused" in capsys.readouterr().out
    base, tuned = load_file(tmp_path / "base" / "model.safetensors"), load_file(tmp_path / "model" / "model.safetensors")
    assert any(not torch.equal(base[name], tuned[name]) for name in base)
    indexers = [name for name in base if ".attention.indexer." in name]
    assert indexers and any(not torch.equal(base[name], tuned[name]) for name in indexers)
    biases = [name for name in base if name.endswith("experts.bias")]
    assert biases and all(torch.equal(base[name], tuned[name]) for name in biases)

    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    first, second = provenance["steps"]
    assert first["reused"] and first["result"] == json.loads(
        (tmp_path / "base" / "provenance.json").read_text())["steps"][0]["result"]
    assert second["step"]["name"] == "answer-as-assistant" and not second["reused"]
    assert second["locked"] == ["balancing"] and second["learned"] == {"conversations": 1}
    assert provenance["learned"]["conversations"] == 1
    lines = credited(tmp_path / "model")
    assert len(lines) == 17 == len(credited(tmp_path / "base")) + 1  # the plain texts stay credited
    assert "2. `answer-as-assistant`" in (tmp_path / "model" / "README.md").read_text()


@dataclass
class Shape:
    """Stands for ModelConfig: wpt.train.fingerprints reads its fields."""
    vocab_size: int


def test_each_step_s_fingerprint_names_only_the_part_of_the_corpus_it_reads(built):
    # wpt.train.fingerprints: text for read-text, conversations for
    # answer-as-assistant, tasks for the task steps; a later step also
    # follows the one before it
    steps = [load_step(entry, PARTS) for entry in ({"step": "read-text", "settings": one_stage(1)},
                                                   {"step": "answer-as-assistant"}, {"step": "finish-tasks"},
                                                   {"step": "finish-tasks-with-less-writing"})]
    config = Shape(vocab_size=10)
    parts = {"text": "a" * 64, "conversations": "b" * 64, "tasks": "c" * 64}

    def marks(**changed):
        return wpt.train.fingerprints(steps, config, {"fingerprints": {**parts, **changed}}, 1337, {})

    base = marks()
    assert len(set(base)) == 4
    only_chat = marks(conversations="d" * 64)
    assert only_chat[0] == base[0] and all(new != old for new, old in zip(only_chat[1:], base[1:]))
    only_tasks = marks(tasks="d" * 64)
    assert only_tasks[:2] == base[:2] and all(new != old for new, old in zip(only_tasks[2:], base[2:]))
    assert all(new != old for new, old in zip(marks(text="d" * 64), base))


def test_the_hardware_options_stay_out_of_a_step_s_fingerprint(built, tmp_path, capsys):
    # --stream and --stream-shards change how the windows are read, not the
    # step: a run with them reuses the step an earlier run kept without them
    train([built.recipe("two", training=one_stage(2)), built.corpus_id, "--out", "base"])
    capsys.readouterr()
    train([built.recipe("two", training=one_stage(2)), built.corpus_id, "--out", "streamed", "--stream",
           "--stream-shards", "1"])
    assert "step 1 read-text: reused" in capsys.readouterr().out


def test_the_steps_after_a_reused_one_train_as_in_a_run_of_every_step(built, on_the_cpu, tmp_path):
    # reusing a kept step gives its weights, so what follows learns the same
    # as in a run that trained every step itself; on the CPU, since two runs
    # on a GPU differ even with no step reused (docs/training.md, "What is exactly repeatable")
    dataset = talks(built)
    steps = [{"step": "answer-as-assistant", "settings": {"epochs": 1}}]
    train([built.recipe("base", training=one_stage(1, 64), dataset="chat"), dataset, "--out", "base"])
    train([built.recipe("chat", training=one_stage(1, 64), steps=steps, dataset="chat"), dataset, "--out", "reused"])
    shutil.rmtree(built.store.root / "cache" / "steps")  # a store that never saw these recipes
    train([built.recipe("chat", training=one_stage(1, 64), steps=steps, dataset="chat"), dataset, "--out", "fresh"])

    reused, fresh = (json.loads((tmp_path / name / "provenance.json").read_text())["steps"]
                     for name in ("reused", "fresh"))
    assert [step["reused"] for step in reused] == [True, False] and [step["reused"] for step in fresh] == [False, False]
    assert reused[1]["learned"] == {"conversations": 1}
    weights = [load_file(tmp_path / name / "model.safetensors") for name in ("reused", "fresh")]
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][name], weights[1][name]) for name in weights[0])


def test_a_recipe_that_adds_a_text_step_with_longer_windows_reuses_the_first_one(built, tmp_path, capsys):
    # the model's context comes from the last step that reads text, but no
    # weight depends on it: a recipe that goes on reading text at longer
    # windows starts from the first step an earlier run kept
    train([built.recipe("base", training=one_stage(1)), built.corpus_id, "--out", "base"])
    longer = {"step": "read-text", "settings": one_stage(1, 32)}
    train([built.recipe("long", training=one_stage(1), steps=[longer]), built.corpus_id, "--out", "long"])
    assert "step 1 read-text: reused" in capsys.readouterr().out
    steps = json.loads((tmp_path / "long" / "provenance.json").read_text())["steps"]
    assert [step["reused"] for step in steps] == [True, False]
    assert json.loads((tmp_path / "long" / "config.json").read_text())["block_size"] == 32


def test_two_text_steps_of_one_recipe_draw_other_windows(built):
    # a second read-text step of the same settings goes on learning from
    # other windows than the first, while each step draws the same windows
    # on every run
    record = corpora.find(built.store, built.corpus_id)
    step = load_step({"step": "read-text", "settings": one_stage(1)}, PARTS)
    strategy = Strategy(torch.device("cpu"))

    def first_windows(number):
        data = Data(step.name, built.store, record, None, strategy, Telemetry.off(), lambda _: None, 1337, number)
        environment = step.start_environment(data)
        try:
            [(x, _)] = environment.batch(1, None, strategy)
            return x
        finally:
            environment.close()

    assert not torch.equal(first_windows(1), first_windows(2))
    assert torch.equal(first_windows(2), first_windows(2))


def test_two_identical_steps_of_one_recipe_draw_other_conversations_and_tasks(built, tmp_path):
    # one seed per recipe; each step mixes its number in the list into it, so
    # a second answer-as-assistant or finish-tasks step of the same settings
    # goes through the conversations and the tasks in another order, while
    # one step draws the same order on every run
    record = corpora.find(built.store, talks(built, count=12))
    tokenizer = wpt.train.Tokenizer.from_bytes(built.store.get_bytes(record["tokenizer"]["vocabulary"]))
    strategy = Strategy(torch.device("cpu"))

    def conversation_orders(number):
        step = load_step({"step": "answer-as-assistant"}, PARTS)
        config = SimpleNamespace(block_size=64)
        part = step.start_manager(Context("x", built.store, record, tokenizer, config, Telemetry.off(),
                                          lambda _: None, 1337, number))
        data = Data("x", built.store, record, config, strategy, Telemetry.off(), lambda _: None, 1337, number,
                    call=in_process(part))
        return step.start_environment(data).orders

    assert conversation_orders(1) != conversation_orders(2)
    assert conversation_orders(2) == conversation_orders(2)

    def task_orders(number):
        found, _ = environment({"rounds": 4, "tasks_per_update": 2}, [fake_task(f"t{i}") for i in range(6)],
                               lambda *_: [])
        found.data = SimpleNamespace(step="x", log=lambda *_: None, telemetry=Telemetry.off(),
                                     step_seed=step_seed(1337, number))
        found.begin()
        return found.orders

    assert task_orders(1) != task_orders(2)
    assert task_orders(2) == task_orders(2)


FAILS_ONCE = """
import importlib.util
import os
from pathlib import Path

import wpt_training
from wpt_training import StepFailed

spec = importlib.util.spec_from_file_location(
    "text_windows", Path(wpt_training.__file__).parent / "environments" / "text-windows.py")
text = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text)
GIVES, SETTINGS, check, context = text.GIVES, text.SETTINGS, text.check, text.context


class FailsOnce(text.TextWindows):
    def after_update(self, step, loss, indexer):
        marker = Path(os.environ["WPT_TEST_FAIL"])
        if step == 2 and marker.exists():
            marker.unlink()
            raise StepFailed("the examples went away")


def start(settings, data):
    return FailsOnce(settings, data)
"""


def test_a_step_that_fails_after_a_checkpoint_keeps_it_for_resume(built, tmp_path, monkeypatch, capsys):
    # a step that fails once it wrote a checkpoint says that the checkpoint
    # stays and where; the same command with --resume continues the step
    # from it
    (tmp_path / "fails_once.py").write_text(FAILS_ONCE)
    environment = str(tmp_path / "fails_once.py")
    (tmp_path / "fails-once-step.py").write_text(f"METHOD = 'next-token'\nENVIRONMENT = {environment!r}\n"
                                                 "OPTIMIZER = {'name': 'by-part'}\n")
    (tmp_path / "fail").touch()
    monkeypatch.setenv("WPT_TEST_FAIL", str(tmp_path / "fail"))
    recipe = built.recipe("fails", steps=[{"step": str(tmp_path / "fails-once-step.py"), "settings": one_stage(3)}])
    command = [recipe, built.corpus_id, "--out", "model", "--checkpoint-every", "1"]
    monkeypatch.setenv("WPT_LINK_CHECKPOINTS", "every")  # the checkpoint the failure leaves comes to this machine
    err = refused(capsys, command)
    assert ("StepFailed: step 2 fails-once-step: the examples went away; its checkpoint stays in "
            f"{Path('model') / 'checkpoint'}: --resume continues the step from it, and deleting it starts the step "
            "over; the results of the 1 steps before it stay in the store") in err
    assert (tmp_path / "model" / "checkpoint" / "checkpoint.safetensors").is_file()
    train([*command, "--resume"])
    out = capsys.readouterr().out
    assert "step 1 read-text: reused" in out and "fails-once-step: resumed from update 1" in out
    assert [step["updates"] for step in json.loads((tmp_path / "model" / "provenance.json").read_text())["steps"]] \
        == [1, 3]


def test_finish_tasks_says_so_when_the_corpus_holds_no_tasks(built, tmp_path, capsys):
    # before any step trains: the read-text step before it would be trained for nothing
    err = refused(capsys, [built.recipe("agent", steps=[{"step": "finish-tasks"}]), built.corpus_id,
                           "--out", "model"])
    assert "InputError: step 2 finish-tasks: no tasks: the corpus holds none tagged finish-tasks" in err
    assert not (tmp_path / "model").exists() and not (built.store.root / "cache" / "steps").exists()


TASK = {"instructions": "Write hello into out.txt.", "image": "tiny",
        "limits": {"max_turns": 2, "max_tokens": 6, "command_seconds": 5}, "needs": []}


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def task_corpus(built, root, check='#!/bin/bash\necho "0.$RANDOM"\n') -> tuple[str, str]:
    """Writes data/images/tiny/Dockerfile and adds one task to the depot,
    whose check by default prints scores that differ: (the id of the
    corpus of a new dataset "tasks", of the depot's texts and the task; the
    task's id as the agent harness names it)."""
    (root / "data" / "images" / "tiny").mkdir(parents=True, exist_ok=True)
    (root / "data" / "images" / "tiny" / "Dockerfile").write_text("ARG BASE=ubuntu:24.04\nFROM ${BASE}\nARG PACKAGES\n")
    content = {"task": TASK, "files": {"notes.txt": {"mode": "100644", "content": encoded(b"start\n")}},
               "check": {"mode": "100755", "content": encoded(check.encode())}}
    corpus_id = built.dataset("tasks", [NewText(kind="task", content=json.dumps(content), tags=("finish-tasks",))])
    [line] = built.store.get_bytes(corpora.find(built.store, corpus_id)["tasks"]["file"]).decode().splitlines()
    return corpus_id, f"{built.depot.address} {json.loads(line)['id']}"


def attempt_lines(store, folder, name) -> list[dict]:
    """The attempts file's lines, each a conversation in the chat format."""
    lines = [json.loads(line) for line in store.read_record(folder, name).decode().splitlines()]
    assert all({"messages", "metadata"} <= set(line) for line in lines)
    return lines


def no_sandbox(monkeypatch):
    """Every container fails to start (a full disk), the image counted as built: no Docker needed."""
    def refused(*args, **kwargs):
        raise wpt_agent.SandboxError("docker create: no space left on device")

    monkeypatch.setattr(images, "built", lambda root, name: f"wpt-agent/{name}:test")
    monkeypatch.setattr(played, "Sandbox", refused)


def test_finish_tasks_fails_when_every_attempt_of_a_round_is_dropped(built, tmp_path, monkeypatch, capsys):
    # a sandbox that never starts is not the model's doing, but a round that
    # could make no update at all fails the step, naming the round and why;
    # the dropped attempts are kept too, each as the conversation it would
    # have started from, and the round's event names their lines
    dataset, _ = task_corpus(built, tmp_path)
    assert corpora.find(built.store, dataset)["tasks"]["count"] == 1
    no_sandbox(monkeypatch)
    recipe = built.recipe("agent", training=one_stage(1, 1024), steps=[{"step": "finish-tasks"}], dataset="tasks")
    err = refused(capsys, [recipe, dataset, "--out", "model"])
    assert ("StepFailed: step 2 finish-tasks: round 1: every attempt was dropped, Docker failed 8 times; the last "
            "time: docker create: no space left on device; nothing of it is kept") in err
    assert not (tmp_path / "model").exists()
    [run] = built.records("wpt train")
    lines = attempt_lines(built.store, run.folder, "finish-tasks-attempts.jsonl")
    assert [(line["metadata"]["round"], line["metadata"]["group"], line["metadata"]["attempt"]) for line in lines] == [
        (1, 1, i) for i in range(1, 9)]
    for line in lines:
        metadata = line["metadata"]
        assert metadata["dropped"] and "no space left" in metadata["error"]
        assert metadata["ended"] is metadata["score"] is metadata["reward"] is metadata["written"] is None
        assert [m["role"] for m in line["messages"]] == ["user"]
        assert line["messages"][0]["content"][0]["text"] == TASK["instructions"]
    [round_] = [e for e in run.events if e.get("name") == "finish-tasks.round"]
    assert round_["attributes"]["file"] == "finish-tasks-attempts.jsonl" and round_["attributes"]["lines"] == [1, 8]


def test_a_refused_attempt_line_stops_the_run_with_the_record_error_code(built, tmp_path, monkeypatch):
    # an attempt the record cannot keep stops the command at once, as any
    # part of the record does: exit code 3, no model written
    dataset, _ = task_corpus(built, tmp_path)
    no_sandbox(monkeypatch)  # a dropped attempt is kept too
    real = LocalStore._raw_append_record

    def disk_full(self, folder, name, data):
        if name == "finish-tasks-attempts.jsonl":
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full)
    with pytest.raises(SystemExit) as stopped:
        train([built.recipe("agent", training=one_stage(1, 1024), steps=[{"step": "finish-tasks"}], dataset="tasks"),
               dataset, "--out", "model"])
    assert stopped.value.code == RECORD_FAILED
    assert not (tmp_path / "model" / "config.json").exists()


def test_finish_tasks_fails_when_no_group_ever_teaches_anything(docker_runs, built, tmp_path, monkeypatch, capsys):
    # a check that always prints 0 gives every group equal rewards: the step
    # makes no update at all, and fails, saying why
    dataset, _ = task_corpus(built, tmp_path, check="#!/bin/sh\necho 0\n")
    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    monkeypatch.setattr(images, "built", lambda root, name: "ubuntu:24.04")  # a plain image is enough here
    step = {"step": "finish-tasks", "settings": {"rounds": 2, "group": 2, "tasks_per_update": 1}}
    err = refused(capsys, [built.recipe("agent", training=one_stage(1, 1024), steps=[step], dataset="tasks"),
                           dataset, "--out", "model"])
    assert ("StepFailed: step 2 finish-tasks: no update in 2 rounds: every group was skipped, 2 because its rewards "
            "were all equal, 0 because it kept fewer than 2 attempts") in err
    assert not (tmp_path / "model").exists()


def test_finish_tasks_with_less_writing_fails_when_no_task_is_left(docker_runs, built, tmp_path, monkeypatch, capsys):
    # the incoming model scores 0 on every reference attempt: the task has no
    # result to keep, and with no task left the step fails, saying why
    dataset, _ = task_corpus(built, tmp_path, check="#!/bin/sh\necho 0\n")
    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    monkeypatch.setattr(images, "built", lambda root, name: "ubuntu:24.04")
    step = {"step": "finish-tasks-with-less-writing", "settings": {"reference": 2}}
    err = refused(capsys, [built.recipe("agent", training=one_stage(1, 1024), steps=[step], dataset="tasks"),
                           dataset, "--out", "model"])
    assert ("StepFailed: step 2 finish-tasks-with-less-writing: no task left of the 1 tasks tagged finish-tasks: each scored 0 "
            "on every reference attempt, or Docker failed every one") in err
    assert not (tmp_path / "model").exists()


# a lock of data/images/tiny/ whose image no machine builds: its tag is absent
NEVER_BUILT = {"base": f"ubuntu@sha256:{'0' * 64}", "snapshot": "20000101T000000Z",
               "packages": {"hello": "0"}}
TINY_IMAGE = "wpt-train/tiny-image"  # in pytest's cache (.pytest_cache/): the pinned and built image of tiny


@pytest.fixture(scope="session")
def tiny_image(request, docker_runs):
    """Gives pinned(root), which pins and builds data/images/tiny/ of root,
    as task_corpus writes it, and returns the image's tag. The lock and the
    image are kept, in pytest's cache and in Docker, for as long as the
    folder's files (but its lock) and wpt_agent/images.py stay the same: a
    later pinned() writes the kept lock.json and licenses.json into the
    folder, and builds again only when Docker lost the image. When they
    change, the image kept for the old ones is removed."""
    cache = request.config.cache

    def pinned(root):
        digest = hashlib.sha256(Path(images.__file__).read_bytes())
        for path, (executable, content) in sorted(images.files(root, "tiny").items()):
            if path not in ("lock.json", "licenses.json"):
                digest.update(path.encode() + b"\0" + str(executable).encode() + b"\0" + content + b"\0")
        key, folder = digest.hexdigest(), root / "data" / "images" / "tiny"
        kept = cache.get(TINY_IMAGE, None)
        if kept and kept["key"] == key:
            (folder / "lock.json").write_text(kept["lock"])
            (folder / "licenses.json").write_text(kept["licenses"])
            return images.build(root, "tiny", log=lambda *_: None)  # at once when Docker has it
        if kept:
            subprocess.run(["docker", "rmi", "-f", kept["tag"]], capture_output=True)
        images.lock(root, "tiny", force=True, log=lambda *_: None)
        tag = images.build(root, "tiny", log=lambda *_: None)
        cache.set(TINY_IMAGE, {"key": key, "tag": tag, "lock": (folder / "lock.json").read_text(),
                               "licenses": (folder / "licenses.json").read_text()})
        return tag

    return pinned


def answered(monkeypatch):
    """Every attempt ends with an answer, its confidence 0.5, as a model
    that has learned the chat format gives one: a tiny model never does,
    and an attempt without an answer is worth 0 whatever its check says."""
    real = wpt_agent.loop._ended
    monkeypatch.setattr(wpt_agent.loop, "_ended", lambda conversation, ids, learn, ended, answer=None: real(
        conversation, ids, learn, ended, answer or {"answer": "", "confidence": 0.5}))


def unchanged_routers(before, after):
    routers = [name for name in before if name.endswith("experts.router.weight")]
    return routers and all(torch.equal(before[name], after[name]) for name in routers)


@pytest.mark.xdist_group("tiny-image")  # one worker: the first of these tests pins and builds the image
def test_finish_tasks_learns_from_scored_attempts_at_the_corpus_s_tasks(docker_runs, built, tmp_path, tiny_image,
                                                                       monkeypatch, capsys):
    # each attempt runs in a container of the task's image and its check
    # scores it; attempts that score apart change the weights, but not the
    # routers', and the model's CREDITS name the task; the
    # image must be built first, and the step says how. The opening alone
    # (the tools' descriptions) is longer than block_size: the model reads
    # the attempts through its cache, up to the task's max_tokens
    dataset, task_id = task_corpus(built, tmp_path)
    (tmp_path / "data" / "images" / "tiny" / "lock.json").write_text(json.dumps(NEVER_BUILT))  # no image built yet
    answered(monkeypatch)
    train([built.recipe("base", training=one_stage(1, 64), dataset="tasks"), dataset, "--out", "base"])
    step = {"step": "finish-tasks", "settings": {"rounds": 1, "group": 2, "tasks_per_update": 1},
            "optimizer": {"name": "by-part", "settings": {"lr": 1e-2, "weight_decay": 0.0}}}
    recipe = built.recipe("agent", training=one_stage(1, 64), steps=[step], dataset="tasks")
    err = refused(capsys, [recipe, dataset, "--out", "model"])
    assert "image 'tiny' is not built on this machine" in err and "wpt agent build-image tiny" in err
    tiny_image(tmp_path)
    train([recipe, dataset, "--out", "model"])

    before, after = load_file(tmp_path / "base" / "model.safetensors"), load_file(tmp_path / "model" / "model.safetensors")
    assert unchanged_routers(before, after)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    steps = json.loads((tmp_path / "model" / "provenance.json").read_text())["steps"]
    assert steps[0]["reused"] and steps[1]["step"]["name"] == "finish-tasks" and steps[1]["learned"] == {"tasks": 1}
    environment = load_step({"step": "finish-tasks"}, PARTS).environment.module
    assert steps[1]["settings"] == {**load_step({"step": "finish-tasks"}, PARTS).settings, **step["settings"]}
    assert set(environment.SETTINGS) <= set(steps[1]["settings"])
    assert any(line.startswith(f"{task_id}: ") for line in credited(tmp_path / "model"))
    # every attempt is kept in the run's record, as a conversation in the chat format, and its span names its line
    run = built.records("wpt train")[-1]
    lines = attempt_lines(built.store, run.folder, "finish-tasks-attempts.jsonl")
    assert [tuple(line["metadata"][key] for key in ("step", "task", "round", "group", "attempt")) for line in lines] == [
        ("finish-tasks", task_id, 1, 1, 1), ("finish-tasks", task_id, 1, 1, 2)]
    for line in lines:
        metadata = line["metadata"]
        assert not metadata["dropped"] and metadata["error"] is None
        # the check prints one number, its correct part; the reward weighs it with the calibration and tool calls
        assert metadata["parts"] == {"correct": metadata["score"]} and 0 <= metadata["score"] <= 1
        assert metadata["reward"] == pytest.approx(metadata["score"] + 0.5 * (1 - (0.5 - metadata["score"]) ** 2)
                                                   - 0.01 * metadata["tool_calls"])
        assert metadata["written"] > 0 and metadata["terse"] is False
        assert metadata["ended"] in ("answered", "ended", "turns", "tokens", "context", "full")
        assert line["messages"][0]["content"][0]["text"] == TASK["instructions"]
        assert line["messages"][1]["role"] == "assistant"
    spans = [e for e in run.events if e["kind"] == "span_end" and e["name"] == "finish-tasks.attempt"]
    assert [(span["attributes"]["file"], span["attributes"]["line"]) for span in spans] == [
        ("finish-tasks-attempts.jsonl", 1), ("finish-tasks-attempts.jsonl", 2)]
    [group] = [e for e in run.events if e.get("name") == "finish-tasks.group"]
    assert group["attributes"]["lines"] == [1, 2]


@pytest.mark.xdist_group("tiny-image")
def test_finish_tasks_with_less_writing_learns_from_the_tasks_finish_tasks_reads(docker_runs, built, tmp_path,
                                                                                 tiny_image, monkeypatch):
    # the incoming model's attempts at finish-tasks's own tasks set the
    # reference, then attempts that pass and fail it change the weights, but
    # not the routers', and the model credits the task
    dataset, task_id = task_corpus(built, tmp_path, check="#!/bin/bash\necho $((RANDOM % 2))\n")
    tiny_image(tmp_path)
    answered(monkeypatch)
    train([built.recipe("base", training=one_stage(1, 1024), dataset="tasks"), dataset, "--out", "base"])
    # 16 reference attempts all fail 1 time in 65536; 8 attempts all equal, in each of 2 rounds, 1 time in 16384
    step = {"step": "finish-tasks-with-less-writing",
            "settings": {"rounds": 2, "group": 8, "reference": 16, "tasks_per_update": 1},
            "optimizer": {"name": "by-part", "settings": {"lr": 1e-2, "weight_decay": 0.0}}}
    train([built.recipe("agent", training=one_stage(1, 1024), steps=[step], dataset="tasks"), dataset, "--out",
           "model"])

    before, after = load_file(tmp_path / "base" / "model.safetensors"), load_file(tmp_path / "model" / "model.safetensors")
    assert unchanged_routers(before, after)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    steps = json.loads((tmp_path / "model" / "provenance.json").read_text())["steps"]
    assert steps[1]["step"]["name"] == "finish-tasks-with-less-writing" and steps[1]["learned"] == {"tasks": 1}
    assert any(line.startswith(f"{task_id}: ") for line in credited(tmp_path / "model"))
    # the reference attempts are kept first, with no round and no reward, then the rounds' attempts
    run = built.records("wpt train")[-1]
    metadata = [line["metadata"] for line in
                attempt_lines(built.store, run.folder, "finish-tasks-with-less-writing-attempts.jsonl")]
    assert [(m["round"], m["attempt"]) for m in metadata] == (
        [(None, i) for i in range(1, 17)] + [(r, i) for r in (1, 2) for i in range(1, 9)])
    assert all(m["step"] == "finish-tasks-with-less-writing" for m in metadata)
    assert all(m["reward"] is None for m in metadata[:16])
    assert all(m["reward"] is not None for m in metadata[16:] if not m["dropped"])


# what follows checks the rules of the agent-tasks environment on attempts
# made up here, without a model that writes them or Docker

AGENT_TASKS = load_step({"step": "finish-tasks"}, PARTS).environment.module


def fake_task(name, **fields):
    """A task as the environment reads it, without files or a check."""
    return type("Task", (), {"id": name, "image": "tiny", "needs": (), "replies": (), "tools": (), "answered_by": (),
                             "answered_in": (), "files": {}, "tool_files": {}, "absent": (), "repository": None,
                             "terse": "", "area": None, "gates": (), "instructions": "Say hello.", **fields})


def scripted(calls=0, confidence=0.5, written=1):
    """An attempt that made `calls` tool calls, wrote `written` tokens and
    answered with that confidence (None: no answer)."""
    messages = [{"role": "assistant", "content": [{"type": "tool_call", "id": f"c{n}", "name": "bash",
                                                    "arguments": {}}]} for n in range(calls)]
    answer = None if confidence is None else {"answer": "Done.", "confidence": confidence}
    return wpt_agent.Attempt({"messages": messages, "metadata": {"answer": answer}}, [0] * (written + 1),
                             [False] + [True] * written, "answered")


def test_an_attempt_s_parts_add_up_by_their_weights_and_a_broken_gate_or_no_answer_scores_0():
    # correct 1, calibration 0.5 (Brier:
    # 1 - (confidence - correct)²), each rule 0.25, each tool call -0.01; a
    # gated rule below 1, or no answer, makes the whole attempt worth 0
    settings = AGENT_TASKS.SETTINGS
    task, gated = fake_task("t"), fake_task("t", gates=("asked_last",))
    parts = {"correct": 1.0, "sources_first": 1.0, "asked_last": 0.0}
    assert agentic.combined(task, scripted(3, 0.8), parts, settings) == pytest.approx(
        1 + 0.5 * (1 - 0.2 ** 2) + 0.25 * 1 - 0.01 * 3)
    assert agentic.combined(gated, scripted(3, 0.8), parts, settings) == 0.0
    assert agentic.combined(task, scripted(3, None), parts, settings) == 0.0
    assert agentic.combined(task, scripted(), None, settings) == 0.0  # the check gave no score
    # stating a low confidence never pays more than being right
    right = agentic.combined(task, scripted(0, 0.0), {"correct": 1.0}, settings)
    assert right > agentic.combined(task, scripted(0, 0.0), {"correct": 0.0}, settings) == 0.5


def test_stating_a_low_confidence_may_never_pay_more_than_being_right_and_is_refused_before_any_step(built, tmp_path,
                                                                                                    capsys):
    step = {"step": "finish-tasks", "settings": {"weight_calibration": 2}}
    err = refused(capsys, [built.recipe("agent", steps=[step]), built.corpus_id, "--out", "model"])
    assert "finish-tasks.py: weight_correct (1.0) must be above weight_calibration (2.0)" in err
    assert not (tmp_path / "model").exists()


class FakePlayer:
    """Stands for agentic.Player: plays(task, round, form) gives each
    attempt's (score, written tokens, changes), or None for one Docker
    dropped; every attempt has parts {"correct": score} and answers with
    confidence 0.5. It keeps the calls it got."""

    def __init__(self, plays):
        self.plays, self.calls, self.lines, self.file = plays, [], 0, "x-attempts.jsonl"

    def attempts(self, task, round_, group, count, reward=None, terse=False, with_changes=False):
        self.calls.append((task.id, round_, terse, with_changes))
        kept = []
        for index, outcome in enumerate(self.plays(task, round_, terse)[:count], 1):
            self.lines += 1
            if outcome is None:
                kept.append(agentic.Kept(self.lines, None, None, None, None, wpt_agent.SandboxError("lost")))
                continue
            score, length, changes = outcome
            one = wpt_agent.Played(scripted(written=length), score, SimpleNamespace(chosen={}),
                                   changes if with_changes else None, {"correct": score})
            kept.append(agentic.Kept(self.lines, one.attempt, one.routes, score,
                                     None if reward is None else reward(task, index, one), None, one.parts,
                                     one.changes))
        return kept


def environment(settings, tasks, plays):
    """The agent-tasks environment of these settings and tasks, with a
    FakePlayer of plays in place of the model's attempts, before its begin()."""
    found = AGENT_TASKS.AgentTasks.__new__(AGENT_TASKS.AgentTasks)
    found.settings = {**AGENT_TASKS.SETTINGS, **settings}
    found.tasks, found.references, found.round = tasks, None, AGENT_TASKS._Round()
    found.updates = found.equal = found.small = 0
    found.data = SimpleNamespace(step="x", log=lambda *_: None, telemetry=Telemetry.off(), step_seed=0)
    found.lines = {"folder": None, "count": 0}
    player = FakePlayer(plays)
    found._player = lambda: player
    return found, player


def test_a_group_whose_rewards_are_all_equal_never_dilutes_the_update():
    # the update learns from the tokens written in the groups that teach
    # something: a skipped group, whose rewards are all equal, adds none
    scores = {"differ": [(1.0, 3, None), (0.0, 3, None)], "equal": [(0.5, 6, None), (0.5, 6, None)]}
    counted = []
    for names in (["differ", "equal"], ["differ"]):
        found, _ = environment({"rounds": 1, "group": 2, "tasks_per_update": 2, "terse": False},
                               [fake_task(name) for name in names], lambda task, round_, terse: scores[task.id])
        found.begin()
        payload = found.play(1)
        counted.append((payload["count"], len(payload["items"])))
        assert found.equal == len(names) - 1
    assert counted == [(6, 2), (6, 2)]


def test_each_terse_attempt_is_rewarded_for_its_closeness_to_the_full_attempt_of_the_same_index():
    # a task with a one-line form is
    # played in full, then terse, both forms one group, so each advantage is
    # taken from the mean of all their attempts; terse attempt i adds,
    # weighed, how alike its changes to /work are to full attempt i's; an
    # answer-only task (an empty area) gets no closeness; where closeness
    # counts, a terse attempt whose full attempt Docker dropped is not
    # trained on
    full = [(1.0, 1, "+a\n+b\n"), (1.0, 1, "+c\n"), None]
    terse = [(1.0, 1, "+a\n+b\n"), (1.0, 1, "+x\n"), (1.0, 1, "+c\n")]
    base = agentic.combined(fake_task("t"), scripted(), {"correct": 1.0}, AGENT_TASKS.SETTINGS)
    weight = AGENT_TASKS.SETTINGS["weight_closeness"]
    learned = []
    for area, rewards in ((None, [base, base, base + weight, base]), ([], [base, base, base, base, base])):
        found, player = environment({"rounds": 1, "group": 3, "tasks_per_update": 1, "terse": True},
                                    [fake_task("t", terse="Say hi.", area=area)],
                                    lambda task, round_, form: terse if form else full)
        found.begin()
        payload = found.play(1)
        assert player.calls == [("t", 1, False, area is None), ("t", 1, True, area is None)]
        learned.append(len(payload["items"]))
        mean = sum(rewards) / len(rewards)
        assert sorted(advantage for *_, advantage in payload["items"]) == pytest.approx(
            sorted(reward - mean for reward in rewards if reward != mean))
    # with closeness, the rewards differ and the group is learned from; without it, they are all equal
    assert learned[0] > 0 and learned[1] == 0


def test_less_writing_rewards_a_passing_attempt_by_how_few_tokens_it_wrote():
    # below the best start score an attempt earns nothing, whatever its
    # weighed parts; at or above it, its weighed parts times the reference
    # length over the tokens the model wrote
    found, _ = environment(load_step({"step": "finish-tasks-with-less-writing"}, PARTS).environment_settings,
                           [], lambda *_: [])
    found.references = {"t": (0.5, 12.0)}  # best start score, reference length
    task = fake_task("t")

    def reward(score, written):
        return found._reward(task, 1, wpt_agent.Played(scripted(written=written), score, parts={"correct": score}))

    weighed = agentic.combined(task, scripted(), {"correct": 0.5}, found.settings)
    assert reward(0.4, 2) == 0.0  # short, but below the best start score
    assert reward(0.5, 12) == pytest.approx(weighed)  # as long
    assert reward(0.5, 6) == pytest.approx(2 * weighed)  # half as long
    assert reward(0.5, 24) == pytest.approx(weighed / 2)  # twice as long
    assert reward(1.0, 12) == pytest.approx(agentic.combined(task, scripted(), {"correct": 1.0}, found.settings))


def test_less_writing_takes_the_mean_length_of_the_attempts_that_reach_the_best_start_score():
    # per task: the best result the incoming model reached (its correct
    # part, never the weighed parts, which calibration and tool calls move),
    # and the mean length of its attempts that reach it; a task the model
    # was always wrong on, or whose every attempt Docker dropped, is left out
    plays = {"mixed": [(0.5, 4, None), (0.75, 10, None), (0.25, 2, None), (0.75, 20, None)],
             "zero": [(0.0, 3, None), (0.0, 7, None)], "lost": [None, None]}
    found, _ = environment({"reward": "shorter", "reference": 4, "terse": False},
                           [fake_task(name) for name in plays], lambda task, round_, terse: plays[task.id])
    assert found._references() == {"mixed": (0.75, pytest.approx(15.0))}


def test_each_attempt_s_line_is_kept_as_soon_as_it_is_scored(monkeypatch):
    # a task's attempts are written
    # together, but each attempt's line is kept right after it is scored,
    # before the next one is, as when attempts ran one by one; a dropped
    # attempt takes its place in the order
    class Box:
        made = 0

        def __init__(self, *args, **kwargs):
            Box.made += 1
            if Box.made == 2:
                raise wpt_agent.SandboxError("docker create: no space left on device")

        def close(self):
            pass

    def attempts(task, boxes, tokenizer, start, result_bytes, offers, terse):
        start([0], len(boxes), ())
        return [wpt_agent.Attempt({}, [0, 1], [False, True], "answered") for _ in boxes]

    kept, scored = [], []
    monkeypatch.setattr(played, "Sandbox", Box)
    monkeypatch.setattr(played, "attempts", attempts)
    monkeypatch.setattr(played, "score",
                        lambda box, check, seconds, conversation, hidden: scored.append(list(kept)) or
                        ({"correct": 1.0}, None))
    player = agentic.Player("x", lambda ids, rows, **kwargs: type("W", (), {"routes": [None] * rows}), None,
                            {"tiny": None}, {}, AGENT_TASKS.SETTINGS, 0)
    monkeypatch.setattr(player, "_keep", lambda task, round_, group, index, one, offer, terse: kept.append(index)
                        or len(kept))
    task = type("Task", (), {"id": "t", "image": "tiny", "files": None, "check": None, "needs": (), "replies": (),
                             "tools": (), "tool_files": {}, "answered_by": (), "repository": None,
                             "answered_in": (), "absent": (), "area": None, "gates": (),
                             "container": {"memory": 1, "disk": 1, "processes": 1, "cpus": 1}})
    outcomes = player.attempts(task, 1, 1, 3, lambda task, index, one: one.score)
    assert scored == [[], [1, 2]]  # attempt 1 scored before any line, attempt 3 after the lines of 1 and 2
    assert kept == [1, 2, 3] and [one.attempt is None for one in outcomes] == [False, True, False]
