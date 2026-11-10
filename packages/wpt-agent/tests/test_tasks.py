"""Reading a task from a depot item's content (wpt_agent.tasks) and the Git
repository a task works in, fetched when it runs (wpt_agent.repository)."""

import base64
import json
import os
import subprocess

import pytest

from wpt_agent.files import FILE
from wpt_agent.repository import BRANCH, RepositoryFailed, fetch
from wpt_agent.tasks import InvalidTask, from_item, from_line, prepared
from wpt_recipes import Recipes

GIT = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "A. Keeper",
       "GIT_AUTHOR_EMAIL": "keeper@example.org", "GIT_COMMITTER_NAME": "A. Keeper",
       "GIT_COMMITTER_EMAIL": "keeper@example.org"}


def encoded(content: bytes) -> dict:
    return {"mode": FILE, "content": base64.b64encode(content).decode()}


def content(files=None, **fields) -> dict:
    """A task item's content of the image tiny; fields replace the task's."""
    task = {"instructions": "Write hello into out.txt.", "image": "tiny",
            "limits": {"max_turns": 2, "max_tokens": 6, "command_seconds": 5}, "needs": [["bash"]], **fields}
    return {"task": task, "files": {path: encoded(data) for path, data in (files or {}).items()},
            "check": encoded(b"#!/bin/sh\necho 1\n")}


@pytest.fixture
def recipes(tmp_path):
    (tmp_path / "data" / "images" / "tiny").mkdir(parents=True)
    (tmp_path / "data" / "images" / "tiny" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return Recipes(tmp_path)


def commits(folder) -> list[str]:
    """Makes a repository of two commits in folder; returns their ids, the first first."""
    folder.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=folder, env=GIT, capture_output=True, check=True).stdout.decode()

    git("init", "-q")
    made = []
    for text in (b"LAMPS = 1\n", b"LAMPS = 2\n"):
        (folder / "lamps.py").write_bytes(text)
        git("add", "-A")
        git("commit", "-q", "-m", "lamps")
        made.append(git("rev-parse", "HEAD").strip())
    return made


def test_a_repository_task_starts_from_its_commit_with_its_own_files_laid_over_it(tmp_path, recipes):
    # the line of a corpus's tasks.jsonl names the task "<depot> <id>"; once
    # prepared, /work is the repository checked out at the task's commit on
    # BRANCH, its .git folder included, and the task's files win over the
    # repository's own
    first, _ = commits(tmp_path / "upstream")
    found = content({"lamps.py": b"LAMPS = 3\n", "notes.txt": b"count them\n"},
                    repository={"address": str(tmp_path / "upstream"), "commit": first})
    line = json.dumps({"depot": "https://depot.example.org", "id": "lamps/1", "content": found})
    task = from_line(line, recipes)
    assert task.id == "https://depot.example.org lamps/1"
    assert task.repository == (str(tmp_path / "upstream"), first)
    started = prepared(task)
    assert started.repository is None and prepared(started) == started
    assert started.files["lamps.py"] == (FILE, b"LAMPS = 3\n") and started.files["notes.txt"] == (FILE, b"count them\n")
    assert started.files[".git/HEAD"] == (FILE, f"ref: refs/heads/{BRANCH}\n".encode())
    assert started.files[f".git/refs/heads/{BRANCH}"] == (FILE, f"{first}\n".encode())
    assert fetch(str(tmp_path / "upstream"), first)["lamps.py"] == (FILE, b"LAMPS = 1\n")


def test_a_commit_the_repository_does_not_hold_is_refused_by_name(tmp_path):
    commits(tmp_path / "upstream")
    with pytest.raises(RepositoryFailed, match=f"at {'0' * 40}: git fetch failed"):
        fetch(str(tmp_path / "upstream"), "0" * 40)


@pytest.mark.parametrize("fields, files, said", [
    ({"container": {"disk": 2048}}, {}, "raise container.memory above 5120"),
    ({"terse": "one\ntwo"}, {}, "terse is the request in one line"),
    ({"absent": ["/work/docs"]}, {}, "absent names '/work/docs'"),
    ({"needs": [["ask"]]}, {}, "needs 'ask', which the task does not offer (ask is offered only with replies)"),
    ({"tool_files": ["pages"]}, {"notes.txt": b""}, "tool_files names 'pages', which is no file or folder"),
    ({"image": "missing"}, {}, "task d t: "),
])
def test_a_task_that_breaks_a_rule_across_its_fields_is_refused_naming_the_rule(recipes, fields, files, said):
    with pytest.raises(InvalidTask) as refused:
        from_item("d t", content(files, **fields), recipes)
    assert said in str(refused.value) and str(refused.value).startswith("task d t: ")
