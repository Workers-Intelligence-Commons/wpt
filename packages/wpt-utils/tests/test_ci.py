"""The scripts of ci/: which parts a change runs, which tests run, and how
pytest-testmon comes to see the programs a test starts."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from wpt_utils.lines import bare

ROOT = Path(__file__).resolve().parents[3]
CI = ROOT / "ci"
STARTUP = CI / "startup"


@pytest.fixture
def ci(monkeypatch):
    """The scripts of ci/ as modules, their names gone from sys.modules
    afterwards: {name: module}."""
    monkeypatch.syspath_prepend(str(CI))
    monkeypatch.syspath_prepend(str(STARTUP))
    names = ("common", "lock", "tests", "rust", "data", "commits", "all", "last_green", "child_files")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield SimpleNamespace(**{name: importlib.import_module(name) for name in names})
    for name in names:
        sys.modules.pop(name, None)


@pytest.fixture
def clone(tmp_path):
    """A git repository of one commit that ignores `ignored.txt`, and a
    function that runs git in it; returns (its folder, git, the commit)."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}

    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=tmp_path, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "conftest.py").write_text("one\n")
    (tmp_path / "other.py").write_text("one\n")
    git("add", ".")
    git("commit", "-qm", "one")
    return tmp_path, git, git("rev-parse", "HEAD")


def test_a_change_is_what_differs_from_the_base_committed_or_not(ci, clone, monkeypatch):
    root, git, base = clone
    monkeypatch.setattr(ci.common, "ROOT", root)
    (root / "conftest.py").write_text("two\n")
    (root / "new.py").write_text("new\n")
    (root / "ignored.txt").write_text("ignored\n")
    assert sorted(ci.common.changed_paths(base)) == ["conftest.py", "new.py"]
    git("add", "conftest.py")  # staged counts the same
    assert sorted(ci.common.changed_paths(base)) == ["conftest.py", "new.py"]
    git("commit", "-qm", "two")  # committed too
    assert sorted(ci.common.changed_paths(base)) == ["conftest.py", "new.py"]


@pytest.mark.parametrize("path, parts", [
    ("packages/wpt-tokenizer/rust/src/lib.rs", {"rust"}),
    ("packages/wpt-tokenizer/rust-toolchain.toml", {"rust"}),
    ("packages/wpt-model/pyproject.toml", {"lock"}),
    ("uv.lock", {"lock", "data"}),
    ("data/recipes/models/one.json", {"data"}),
    ("data/images/agent/Dockerfile", {"data"}),
    ("packages/wpt-recipes/src/wpt_recipes/recipes.py", {"data"}),
    ("packages/wpt-recipes/tests/test_recipes.py", set()),
    ("apps/wpt/src/wpt/__init__.py", set()),
    ("packages/wpt-model/src/wpt_model/kda.py", set()),
    ("docs/architecture.md", set()),
    ("ci/common.py", {"lock", "rust", "data"}),
    ("ci/rust.py", {"rust"}),
])
def test_a_changed_path_runs_the_parts_it_touches_and_tests_and_commits_always_run(ci, path, parts):
    concerned = {part.name for part in ci.all.PARTS if part.concerned_by(path)}
    assert concerned == parts | {"tests", "commits"}


def test_the_commits_part_names_each_commit_without_its_author_s_sign_off(ci, clone, monkeypatch):
    # merge commits aside, every commit of the range carries its author's
    # Signed-off-by line; another person's line does not count
    root, git, base = clone
    monkeypatch.setattr(ci.commits, "ROOT", root)
    git("commit", "-q", "--allow-empty", "-s", "-m", "signed")
    git("commit", "-q", "--allow-empty", "-m", "other\n\nSigned-off-by: U <u@example.org>")
    other = git("rev-parse", "HEAD")
    assert ci.commits.unsigned(f"{base}..HEAD") == [f"{other[:12]}: no 'Signed-off-by: T <t@example.org>' line "
                                                   "(commit with `git commit -s`)"]
    assert ci.commits.unsigned(f"{base}..HEAD~1") == []


def given(ci, *arguments):
    return ci.common.options(["--base", "BASE", *arguments])


def test_a_part_no_change_touches_is_left_out_and_all_runs_it(ci, capsys):
    ran = []
    part = ci.common.Part("rust", lambda path: False, lambda options: ran.append(options) or True)
    assert ci.common.perform(part, given(ci), ["docs/a.md"]) == ("left out", 0.0)
    assert ran == [] and "left out  rust" in capsys.readouterr().out
    result, took = ci.common.perform(part, given(ci, "--all"), ["docs/a.md"])
    assert result == "passed" and len(ran) == 1 and "passed  rust in 0 s" in capsys.readouterr().out


def test_a_part_that_always_runs_runs_with_no_change_at_all(ci):
    # the tests that failed run again, and the commits are read, even when the files equal the base
    part = ci.common.Part("tests", ci.common.always, lambda options: True)
    assert ci.common.perform(part, given(ci), [])[0] == "passed"


def test_every_part_runs_after_a_failed_one_and_the_end_lists_each_with_its_result(ci, monkeypatch, capsys):
    ran = []

    def part(name, passes):
        return ci.common.Part(name, ci.common.always, lambda options: ran.append(name) or passes)

    monkeypatch.setattr(ci.all, "PARTS", (part("lock", False), part("tests", True),
                                          ci.common.Part("rust", lambda path: False, lambda options: 1 / 0)))
    monkeypatch.setattr(ci.all, "changed_paths", lambda base: ["a.py"])
    monkeypatch.setattr(sys, "argv", ["all.py", "--base", "BASE"])
    with pytest.raises(SystemExit) as ended:
        ci.all.main()
    assert ended.value.code == 1 and ran == ["lock", "tests"]
    printed = bare(capsys.readouterr().out).splitlines()
    assert [text.split()[:2] for text in printed[-4:-1]] == [["FAILED", "lock"], ["passed", "tests"], ["left", "out"]]


def test_a_part_that_exits_or_whose_program_fails_counts_as_failed(ci):
    def leaves(options):
        sys.exit("gave up")

    def fails(options):
        subprocess.run([sys.executable, "-c", "raise SystemExit(3)"], check=True)

    for run in (leaves, fails):
        assert ci.common.perform(ci.common.Part("x", ci.common.always, run), given(ci))[0] == "FAILED"


def test_the_image_build_test_runs_only_when_the_change_touches_image_code(ci, clone, monkeypatch):
    root, git, base = clone
    monkeypatch.setattr(ci.common, "ROOT", root)
    monkeypatch.setattr(ci.tests, "images_changed", lambda base: False)
    assert ci.tests.image_tests(base)[0] is False
    (root / "conftest.py").write_text("two\n")
    runs, why = ci.tests.image_tests(base)
    assert runs and "touches conftest.py" in why
    (root / "conftest.py").write_text("one\n")
    monkeypatch.setattr(ci.tests, "images_changed", lambda base: True)
    runs, why = ci.tests.image_tests(base)
    assert runs and "an image's folder" in why


def test_a_new_file_of_an_image_s_folder_counts_as_a_change_of_that_folder(ci, clone):
    # what the service layer answers: git diff alone never lists a file git does not track
    root, git, base = clone
    image = root / "data" / "images" / "one"
    image.mkdir(parents=True)
    (image / "Dockerfile").write_text("FROM scratch\n")

    def answer():
        return subprocess.run([sys.executable, "-c", ci.tests.IMAGES_DIFFER, base], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()

    assert answer() == "True"
    (image / "Dockerfile").unlink()
    assert answer() == "False"


def test_every_test_runs_with_all_with_no_record_and_when_a_file_coverage_cannot_see_changed(ci, tmp_path):
    data, kept = tmp_path / "data", tmp_path / "kept"
    now = {"packages/a/schema.json": "one", "uv.lock": "two"}
    every, select = ci.tests.EVERY, ci.tests.SELECT
    assert ci.tests.selection(data, kept, now, False)[0] == every  # no record
    data.write_text("")
    assert ci.tests.selection(data, kept, now, False)[0] == every  # a record no run finished
    kept.write_text(json.dumps(now))
    assert ci.tests.selection(data, kept, now, False)[0] == select
    assert ci.tests.selection(data, kept, now, True)[0] == every
    chosen, why = ci.tests.selection(data, kept, {**now, "uv.lock": "other"}, False)
    assert chosen == every and "uv.lock" in why and "schema.json" not in why
    assert ci.tests.selection(data, kept, {"uv.lock": "two"}, False)[0] == every  # a file that went


def test_what_coverage_cannot_see_is_every_other_file_of_the_code_and_the_project(ci, clone, monkeypatch):
    root, git, base = clone
    monkeypatch.setattr(ci.tests, "ROOT", root)
    (root / "uv.lock").write_text("lock\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "docs.md").write_text("elsewhere\n")
    code = root / "packages" / "a" / "src"
    code.mkdir(parents=True)
    (code / "a.py").write_text("x = 1\n")
    (code / "schema.json").write_text("{}\n")
    (code / "ignored.txt").write_text("ignored\n")
    found = ci.tests.others()
    assert sorted(found) == ["packages/a/src/schema.json", "pyproject.toml", "uv.lock"]
    (code / "schema.json").write_text("[]\n")
    assert ci.tests.others()["packages/a/src/schema.json"] != found["packages/a/src/schema.json"]


def test_the_record_is_in_the_user_s_cache_folder_unless_a_setting_names_another(ci, tmp_path, monkeypatch):
    monkeypatch.delenv("WPT_TEST_RECORD", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert ci.tests.record_folder() == tmp_path / "wpt-test-record"
    monkeypatch.setenv("WPT_TEST_RECORD", str(tmp_path / "other"))
    assert ci.tests.record_folder() == tmp_path / "other"


def test_the_cpu_option_hides_the_gpu_and_forces_the_cpu_and_the_startup_file_is_found(ci, monkeypatch):
    monkeypatch.setattr(ci.tests, "project_prefix", lambda: "/the/project")
    monkeypatch.setattr(ci.tests, "ENVIRON", {})
    plain, cpu = ci.tests.environment(False), ci.tests.environment(True)
    assert str(STARTUP) in plain["PYTHONPATH"] and plain["WPT_CI_PYTHON"] == "/the/project" and "WPT_CPU" not in plain
    assert cpu["WPT_CPU"] == "1" and cpu["CUDA_VISIBLE_DEVICES"] == cpu["HIP_VISIBLE_DEVICES"] == ""


def test_the_sample_is_every_nth_test_in_collection_order(ci):
    ids = [f"t{i}.py::test_{i}" for i in range(10)]
    assert ci.tests.sample_of(ids, 4) == ["t0.py::test_0", "t4.py::test_4", "t8.py::test_8"]
    assert ci.tests.sample_of([], 4) == []


def test_the_test_processes_are_sized_by_the_average_memory_the_sample_held_alone_as_a_worker(ci, monkeypatch):
    ran, asked = [], []
    monkeypatch.setattr(ci.tests, "average", lambda command, **options: ran.append((command, options)) or (0, 5e9))

    def limit(memory, what):
        asked.append((memory, what))
        return 3, "3 test workers: sized"

    monkeypatch.setattr(ci.tests, "limit", limit)
    count, text = ci.tests.test_processes([f"a.py::t{i}" for i in range(100)], {"PATH": "x"})
    assert count == 3 and asked == [(5e9, "test workers")]
    command, used = ran[0]
    assert command[-3:] == ["a.py::t0", "a.py::t40", "a.py::t80"] and "-n" in command  # alone: no other processes
    assert used["env"]["HIP_VISIBLE_DEVICES"] == "" and used["env"]["OMP_NUM_THREADS"] == "1"
    assert "3 sample tests ran alone (passed); it held 5.0 GB on average" in text and "3 test workers: sized" in text


def test_there_are_never_more_test_processes_than_tests(ci, monkeypatch):
    monkeypatch.setattr(ci.tests, "average", lambda command, **options: (0, 1e9))
    monkeypatch.setattr(ci.tests, "limit", lambda memory, what: (12, "sized"))
    assert ci.tests.test_processes(["a.py::t", "b.py::t"], {})[0] == 2


def test_a_sample_whose_memory_cannot_be_measured_leaves_the_cores_to_decide(ci, monkeypatch):
    monkeypatch.setattr(ci.tests, "average", lambda command, **options: (1, None))
    count, text = ci.tests.test_processes(["a.py::t"], {})
    assert count >= 1 and "FAILED" in text and "its memory could not be measured" in text and "unknown" in text


def test_a_program_s_file_adds_all_its_lines_to_what_the_test_itself_ran(ci, tmp_path, monkeypatch):
    # an edit in a function only the program ran must select the test, even when the
    # test's own process ran other lines of the same file
    source = tmp_path / "packages" / "a" / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("one\ntwo\nthree\n")
    monkeypatch.setattr(ci.child_files, "FOLDER", tmp_path / "child-files")
    log = ci.child_files.folder_of("t.py::test")
    log.mkdir(parents=True)
    (log / "1").write_text("packages/a/src/a.py\npackages/gone/src/gone.py\n")
    nodes = {"t.py::test": {"packages/a/src/a.py": {2}, "t.py": {1}, "scripts/no-extension": {1}}}
    assert ci.child_files.with_child_files(nodes, tmp_path) == {
        "t.py::test": {"packages/a/src/a.py": {1, 2, 3}, "t.py": {1}}}


def started(folder, prefix, *command):
    """Runs a Python program under the startup file."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(STARTUP), str(folder / "packages/p/src")]),
           "WPT_CI_PYTHON": prefix, "WPT_CHILD_FILES": str(folder / "log")}
    env.pop("WPT_CPU", None)
    return subprocess.run([sys.executable, *command], cwd=folder, env=env, capture_output=True, text=True)


@pytest.fixture
def source(tmp_path):
    """A folder holding a source package, and its folder for what programs write."""
    package = tmp_path / "packages" / "p" / "src" / "p"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "imported.py").write_text("")
    (package / "by_path.py").write_text("")
    (package / "__main__.py").write_text("import p.imported\nimport importlib.util as u\n"
                                         "s = u.spec_from_file_location('by_path', 'packages/p/src/p/by_path.py')\n"
                                         "s.loader.exec_module(u.module_from_spec(s))\n")
    (tmp_path / "log").mkdir()
    return tmp_path


def test_a_program_writes_every_file_of_ours_it_loads_whichever_way_it_loads_it(source):
    assert started(source, sys.prefix, "-m", "p").returncode == 0
    written = {line for log in (source / "log").iterdir() for line in log.read_text().split()}
    assert written == {f"packages/p/src/p/{name}.py" for name in ("__init__", "__main__", "imported", "by_path")}


def test_only_the_project_s_python_acts_on_the_startup_file(source):
    # Mercurial is a Python program, which logging inside it broke
    assert started(source, "/another/python", "-m", "p").returncode == 0
    assert list((source / "log").iterdir()) == []


def test_triton_cannot_be_imported_under_the_cpu_setting_in_the_project_s_python():
    env = {**os.environ, "PYTHONPATH": str(STARTUP), "WPT_CI_PYTHON": sys.prefix, "WPT_CPU": "1"}
    code = "try:\n    import triton.language\nexcept ModuleNotFoundError as error:\n    print(error.name)"
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True).stdout.strip() == "triton"


def test_an_edit_only_a_started_program_runs_selects_the_test(tmp_path):
    # through pytest-testmon itself: the plugin extends one of its internals
    package = tmp_path / "packages" / "p" / "src" / "p"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("def used():\n    return 1\n")
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "test_it.py").write_text(
        "import subprocess, sys\n"
        "def test_started():\n"
        "    subprocess.run([sys.executable, '-c', 'import p; p.used()'], check=True)\n"
        "def test_alone():\n"
        "    pass\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(STARTUP), str(package.parent)]),
           "WPT_CI_PYTHON": sys.prefix, "TESTMON_DATAFILE": str(tmp_path / "record")}

    def tests_that_ran():
        done = subprocess.run([sys.executable, "-m", "pytest", "-p", "child_files", "--testmon-forceselect", "-v",
                               "-p", "no:cacheprovider", "--rootdir", str(tmp_path)], cwd=tmp_path, env=env,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        return sorted(text.split("::")[1].split()[0] for text in done.stdout.splitlines() if " PASSED" in text)

    assert tests_that_ran() == ["test_alone", "test_started"]
    assert tests_that_ran() == []
    module.write_text("def used():\n    return 2\n")
    assert tests_that_ran() == ["test_started"]


def test_the_last_green_commit_is_the_newest_successful_run_s_when_this_clone_holds_it(ci, clone, tmp_path_factory,
                                                                                       monkeypatch):
    root, git, commit = clone
    monkeypatch.setattr(ci.last_green, "ROOT", root)
    programs = tmp_path_factory.mktemp("programs")
    gh = programs / "gh"
    monkeypatch.setenv("PATH", f"{programs}{os.pathsep}{os.environ['PATH']}")

    def answers(text, code=0):
        gh.write_text(f"#!/bin/sh\necho '{text}'\nexit {code}\n")
        gh.chmod(0o755)
        return ci.last_green.last_green("o/r", "main", "push")

    if os.name == "nt":
        pytest.skip("the stand-in for gh is a shell script")
    assert answers(commit) == commit
    assert answers("") is None  # no green run
    assert answers("0" * 40) is None  # a commit this clone does not hold
    assert answers("refused", 1) is None
