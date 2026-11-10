"""The first agent image, data/images/agent/: its shared check library
(wpt_check.py), tested once here for every check that calls it, and the
programs that answer the tools of its pool (tools.json). The library and the programs are read from the image's
folder, as the image holds them; the last test runs them in the image
itself, built from its lock (skipped without Docker)."""

import ast
import importlib
import importlib.machinery
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wpt_agent import AGENT, Limits, Sandbox, attempt, images, offer, score
from wpt_agent.files import FILE
from wpt_agent.tasks import Task, Variant, looks
from wpt_recipes import Recipes
from wpt_tokenizer import Tokenizer, chat

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "data" / "images" / "agent"
LAMP = b"# The lighthouse\n\nThe keeper lights the lamp at dusk and writes the hour in his log.\n"
TOKENIZER = Tokenizer([])  # bytes only: any text encodes

# the tools that look, by the one rule (wpt_agent.tasks.looks): the built-in ones, the emulations, the pool's
LOOKING = {tool["name"] for tool in [{"name": "read"}, {"name": "bash"},
                                     *json.loads((FOLDER / "tools.json").read_text())["tools"]] if looks(tool)}


def load(name):
    """A file of the image's folder as a module, the library's or a tool's
    program, its folder on the path as in the image (the programs import
    wpt_tools) while it runs. The folder is listed and wpt_tools imported
    here, by the test, never by the program being loaded nor by a later
    import elsewhere: the service layer's audit hook refuses any other code
    that reads under data/, so the folder leaves the path once it is
    loaded."""
    path = FOLDER / (f"{name}.py" if name.startswith("wpt_") else f"tools/{name}")
    spec = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    writes = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # a __pycache__ in the image's folder would change the image
    sys.path.insert(0, str(FOLDER))
    try:
        importlib.import_module("wpt_tools")
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(FOLDER))
        sys.path_importer_cache.pop(str(FOLDER), None)
        sys.dont_write_bytecode = writes
    return module


def turn(*blocks) -> list[int]:
    """The tokens a model writes for an assistant message, <|im_end|> included."""
    ids = chat.encode(TOKENIZER, {"messages": [{"role": "assistant", "content": list(blocks)}]})[0]
    start = ids.index(chat.IM_START) + len(TOKENIZER.encode(b"assistant\n")) + 1
    return ids[start:ids.index(chat.IM_END) + 1]


class Scripted:
    """A model that writes the given turns, one per call, whatever it read."""

    def __init__(self, *turns):
        self.turns = list(turns)

    def __call__(self, ids, rows, stops):
        return self

    def write(self, inputs, limits):
        return {row: self.turns.pop(0)[:limit] for row, limit in limits.items()}


def call(n, name, arguments):
    return {"type": "tool_call", "id": f"c{n}", "name": name, "arguments": arguments}


def conversation(calls, tools=None, answer=None, answered_by=(), results=None, answered_in=()):
    """An attempt's conversation: each call (name, arguments) in a turn of
    its own with its result (results, one text each, or "ok"), then in
    metadata the offered tools that look (LOOKING), as the agent loop
    writes them, the answer (the final reply's text and the confidence,
    None without one), the task's answered_by and the starting files its
    answered_in matches."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "Fix calc.py."}]}]
    for n, (name, arguments) in enumerate(calls):
        messages += [{"role": "assistant", "content": [{"type": "tool_call", "id": f"c{n}", "name": name,
                                                        "arguments": arguments}]},
                     {"role": "tool", "content": [{"type": "tool_result", "call": f"c{n}", "content": [
                         {"type": "text", "text": results[n] if results else "ok"}]}]}]
    names = tools or {name: name for name, _ in calls}
    return {"messages": messages,
            "metadata": {"tools": names, "looking": [name for name in names.values() if name in LOOKING],
                         "answered_by": list(answered_by), "answered_in": list(answered_in), "answer": answer}}


def test_a_missing_answer_scores_0_and_the_confidence_gets_its_brier_score():
    # every check scores 0 an attempt without an answer (no final reply, or
    # a reply to the confidence question that is not a number from 0 to 100),
    # whatever the files hold, and gives a confidence its Brier score
    #
    check = load("wpt_check")
    given = conversation([], answer={"answer": "Fixed the sign.", "confidence": 0.8})
    missing = conversation([])
    assert check.scored(given, 0.7) == 0.7 and check.scored(missing, 1.0) == 0.0
    assert check.brier(given, True) == pytest.approx(0.04) and check.brier(given, False) == pytest.approx(0.64)
    assert check.brier(missing, True) == 1.0


def test_the_trace_rules_read_each_tool_by_its_real_name():
    # an attempt sees its tools under varied names; the rules judge the real
    # tools, through the conversation's map of the names the model saw
    check = load("wpt_check")
    names = {"view": "read", "str_replace": "edit", "shell": "bash", "question": "ask", "grep": "search_text"}
    edit = ("str_replace", {"path": "calc.py", "old_string": "+", "new_string": "-"})

    def trace(*calls):
        return conversation(calls, names)

    assert check.read_before_write(trace(("view", {"path": "/work/calc.py"}), edit))
    assert check.read_before_write(trace(("shell", {"command": "cat ./calc.py"}), edit))
    assert not check.read_before_write(trace(("shell", {"command": "cat old_calc.py"}), edit))
    assert not check.read_before_write(trace(edit))
    assert check.read_before_write(trace(("str_replace", {"path": "notes.md"})), new=["notes.md"])

    ask = ("question", {"question": "Which Python version?"})
    assert check.sources_first(trace(("grep", {"pattern": "python"}), ask, edit))
    assert not check.sources_first(trace(ask, ("grep", {"pattern": "python"})))


def test_looking_is_a_looking_tool_or_bash_running_a_reading_program():
    # every looking tool opens a source,
    # the pool's among them, and a bash call looks when its command, parsed,
    # runs one of the image's reading programs; a word that only names one
    # does not
    check = load("wpt_check")
    names = {"view": "read", "str_replace": "edit", "shell": "bash", "grep": "search_text", "question": "ask"}
    edit = ("str_replace", {"path": "calc.py", "old_string": "+", "new_string": "-"})

    def shell(command):
        return ("shell", {"command": command})

    def first(*calls):
        return check.sources_first(conversation(calls, names))

    for command in ("cat calc.py", "cd docs && /usr/bin/head -n 5 a.md", "ls | sort", "LC_ALL=C grep -rn x .",
                    "for f in *.md; do sed -n 1p \"$f\"; done", "(cd docs; tail a.md) 2>&1", "xargs -0 cat < list",
                    # a line break ends a command, as ; does; a # inside a word starts no comment
                    "cd docs\ncat a.md", "echo a#b; cat x", "echo '#'; cat x", "cd docs && \\\n  zless a.md.gz",
                    # a function's body, and the programs that run a program after their own words
                    "f() { cat x; }", "timeout 5 cat x", "timeout -s KILL 5s comm a b", "xargs -n 1 cat",
                    "env -u HOME nice -n 5 diff3 a b c"):
        assert first(shell(command), edit), command
    for command in ("echo cat calc.py", "python3 --version", "git status", "echo 'cat", "printf '%s' ls",
                    "echo done # cat calc.py", "python3 x.py\n# then cat it", "xargs -n 1 echo"):
        assert not first(shell(command), edit), command
    assert check.programs("xargs -n 1 cat") == ["cat"] and check.programs("cd a\nls # cat b") == ["cd", "ls"]
    assert first(("grep", {"pattern": "def "}), edit) and first(("view", {"path": "calc.py"}))
    assert not first(edit, ("view", {"path": "calc.py"})) and not first(shell("python3 calc.py"))
    # a tool looks only when the agent loop listed it as looking: here bash is not offered as such
    unlisted = conversation([shell("cat calc.py"), edit], names)
    unlisted["metadata"]["looking"] = ["read"]
    assert not check.sources_first(unlisted)


def test_asking_comes_after_every_file_where_the_answer_lives_was_opened():
    # a question counts
    # only after every file of the task's answered_in was opened through one
    # of the looking tools the task names as able to answer (a bash call
    # counts when it names the file and runs a program that shows what it
    # holds), and while no look showed the answer, or at once when no such
    # file exists: the answer is absent
    check = load("wpt_check")
    names = {"view": "read", "shell": "bash", "find": "find_files", "question": "ask"}  # search_text is not offered
    ask = ("question", {"question": "Where do I put it?"})
    view, cat, grep, run, find = ("view", {"path": "NOTES.md"}), ("shell", {"command": "cat /work/NOTES.md"}), \
        ("shell", {"command": "grep -n put TODO.md"}), ("shell", {"command": "python3 NOTES.md"}), \
        ("find", {"path": "NOTES.md"})
    listing = ("shell", {"command": "ls -l NOTES.md && wc -l NOTES.md"})
    where = "Put results in result-0a1b2c.txt."

    def last(*calls, answered_by=("read", "bash", "search_text"), answered_in=("NOTES.md",), results=None,
             found=""):
        return check.asked_last(conversation(calls, names, answered_by=answered_by, results=results,
                                             answered_in=answered_in), found)

    assert last(view, ask) and last(cat, ask, ask) and last(view, grep)
    assert not last(ask, view) and not last(grep, ask) and not last(run, ask)  # no look into NOTES.md
    assert not last(listing, ask)  # its name and size only: not read
    # the answer was found in the notes: a question no longer counts
    assert not last(view, ask, found=where, results=[f"# Notes\n\n{where}\n", "Put it there."])
    assert last(view, ask, found=where, results=["# Notes\n\nThe documents are in docs/.\n", "Put it there."])
    assert not last(view, ask, answered_by=("bash",)) and not last(find, ask)  # a tool that cannot answer
    assert not last(view, ask, answered_in=("NOTES.md", "TODO.md")) and last(view, grep, ask, answered_in=(
        "NOTES.md", "TODO.md"))  # every file where the answer lives
    failed = conversation([view, ask], names, answered_by=("read",), answered_in=("NOTES.md",))
    failed["messages"][2]["content"][0]["error"] = True  # the read failed: nothing was opened
    assert not check.asked_last(failed)
    assert last(ask, answered_in=())  # no file holds the answer
    assert last(ask, answered_by=("search_text",))  # no tool offered could find it


REST = "and wrote the hour, the wind and the state of the sea in his log."


def test_the_library_reads_a_file_the_attempt_left_and_holds_wants_words(tmp_path):
    # file_text gives a file's text and "" for a pipe or a missing file;
    # holds wants the expected words, and "..." holds none
    check = load("wpt_check")
    (tmp_path / "out.txt").write_text(REST)
    os.mkfifo(tmp_path / "pipe")
    assert check.file_text(str(tmp_path / "out.txt")) == REST
    assert check.file_text(str(tmp_path / "pipe")) == "" and check.file_text(str(tmp_path / "gone.txt")) == ""
    assert check.holds(REST.upper(), "the WIND and the state") and not check.holds("anything", "...")


def test_the_pool_s_tools_answer_as_the_tools_they_stand_for(tmp_path):
    # padding tools really work: a
    # listing, a glob, a grep and a calculator, links never followed
    (tmp_path / "src" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("import os\nx = 1\n")
    (tmp_path / "src" / "deep" / "b.py").write_text("y = 2\nx = 3\n")
    (tmp_path / "notes.md").write_text("x marks it\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\xff x = 9\n")
    (tmp_path / "loop").symlink_to(tmp_path)

    def at(path="."):
        return os.open(tmp_path / path, os.O_RDONLY)

    assert load("list_files").listing(at()) == ["image.png", "loop@", "notes.md", "src/"]
    assert load("find_files").found("**/*.py", at()) == ["src/a.py", "src/deep/b.py"]
    assert load("find_files").found("src/*", at()) == ["src/a.py", "src/deep"]
    grep = load("search_text")
    assert grep.lines(re.compile(r"^x = "), ".", at(), True) == ["src/a.py:2:x = 1", "src/deep/b.py:2:x = 3"]
    assert grep.lines(re.compile("y"), "src/deep/b.py", at("src/deep/b.py"), False) == ["src/deep/b.py:1:y = 2"]
    calculator = load("calculator")
    assert calculator.value(ast.parse("(3 + 4) * sqrt(4) - 2 ** 3", mode="eval")) == 6.0
    with pytest.raises(ValueError):
        calculator.value(ast.parse("__import__('os')", mode="eval"))


def test_a_pool_tool_never_reaches_outside_work(tmp_path):
    # the pool's programs read the hidden files' group: a path or a link
    # leading out of /work must fail, or the model would read through them
    tools = load("wpt_tools")
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "hidden").write_text("secret\n")
    (work / "link").symlink_to(tmp_path / "hidden")
    (work / "folder").symlink_to(tmp_path)
    with open(tools.in_work("src/a.py", root=str(work)), closefd=True) as file:
        assert file.read() == "x = 1\n"
    for path in ("../hidden", "link", "folder/hidden", str(tmp_path / "hidden")):
        with pytest.raises(SystemExit):
            tools.in_work(path, root=str(work))


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a folder whose mode is 0")
def test_a_pool_tool_names_the_real_cause_of_a_refusal(tmp_path, capsys):
    tools = load("wpt_tools")
    work = tmp_path / "work"
    (work / "shut").mkdir(parents=True)
    (tmp_path / "hidden").write_text("secret\n")
    (work / "link").symlink_to(tmp_path / "hidden")
    (work / "shut").chmod(0)
    try:
        for path, said in (("shut/a", "permission denied"), ("link", "symbolic link")):
            with pytest.raises(SystemExit):
                tools.in_work(path, root=str(work))
            assert said in capsys.readouterr().err
    finally:
        (work / "shut").chmod(0o755)


@pytest.fixture(scope="module")
def agent(docker_runs, tmp_path_factory):
    """(a repository holding a copy of the agent image's folder, the image
    built from it and its lock, at once when this machine has it): a Recipes
    of the real repository would make its audit hook refuse pytest's own
    clean-up of folders named data."""
    root = tmp_path_factory.mktemp("repository")
    shutil.copytree(FOLDER, root / "data" / "images" / "agent", ignore=shutil.ignore_patterns("__pycache__"))
    return root, images.build(root, "agent", log=lambda *_: None)


@pytest.mark.image_build
@pytest.mark.xdist_group("agent-image")  # one worker builds the image: two would share its snapshot layer
def test_the_reading_programs_are_programs_the_agent_image_holds(agent):
    # wpt_check.READERS names the image's reading programs: each of them is on the image's path
    _, image = agent
    readers = sorted(load("wpt_check").READERS)
    listed = subprocess.run(["docker", "run", "--rm", "--network", "none", image, "bash", "-c",
                             'for name in "$@"; do command -v "$name" >/dev/null && echo "$name"; done', "bash",
                             *readers], capture_output=True, text=True, check=True).stdout.split()
    assert listed == readers


@pytest.mark.image_build
@pytest.mark.xdist_group("agent-image")  # one worker builds the image: two would share its snapshot layer
def test_in_the_agent_image_the_tools_answer_and_the_check_library_scores(agent):
    # in the image itself: a pool tool works in /work without reaching the
    # task's hidden files, and a check started as python3 -I imports the
    # library and reads the conversation
    root, image = agent
    pool = Recipes(root).image_tools("agent")
    hidden = {"web/example.org/lighthouse": (FILE, LAMP)}
    made = Task("example/lamp", "When is the lamp lit?", "agent", 8, 10_000, 10, {"notes.txt": (FILE, b"start\n")},
                b"#!/usr/bin/python3 -I\nimport wpt_check\nc = wpt_check.load()\n"
                b"print(wpt_check.scored(c, float('at dusk' in wpt_check.answer(c)['answer'])))\n",
                needs=((Variant(("bash",)),),), tool_files=hidden)
    chosen = next(chosen for chosen in (offer(made, pool, random.Random(seed)) for seed in range(100))
                  if "list_files" in chosen.names.values())
    listing = next(shown for shown, actual in chosen.names.items() if actual == "list_files")
    turns = [turn(call(1, listing, {"path": "/tmp/wpt-tools"}), call(2, listing, {}),
                  call(3, "bash", {"command": "cat /tmp/wpt-tools/web/example.org/lighthouse"})),
             turn({"type": "text", "text": "The keeper lights it at dusk."}), turn({"type": "text", "text": "90"})]
    limits = Limits(memory=256, disk=16, processes=64, cpus=1.0)
    with Sandbox(image, made.files, user=AGENT, limits=limits, tool_files=made.tool_files) as box:
        done = attempt(made, box, TOKENIZER, Scripted(*turns), result_bytes=4000, offer=chosen)
        assert score(box, made.check, 30, done.conversation) == ({"correct": 1.0}, "")
    outside, inside, seen = [block["content"][0]["text"] for m in done.conversation["messages"]
                             if m["role"] == "tool" for block in m["content"]]
    assert outside == "/tmp/wpt-tools: outside /work" and inside == "notes.txt\n"
    assert "Permission denied" in seen and "keeper" not in seen
