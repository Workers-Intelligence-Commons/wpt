# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""The tests of every package, chosen by what changed. Options: common.py.

pytest-testmon keeps a record of the source lines each test ran and runs a test
again only when one of them changed since, or when it failed;
ci/startup/ adds what the programs a test starts load. Coverage sees only
Python, and not ci/startup/ itself, so a change to any other file under
packages/ or apps/, to any file of ci/startup/, to pyproject.toml or to uv.lock runs every test, and so does a missing record: the
hashes of those files stay beside the record, written by a run that passed.

The record stays outside the checkout: inside, its changes would change the
hash of uncommitted files that a test reads. It is one folder per machine (the
user's cache folder, or WPT_TEST_RECORD), shared by every checkout, so a new
checkout does not run everything. Runs with and without --cpu keep apart, since
their tests see different machines.

A sample of the chosen tests (every SAMPLE_EVERY-th) runs alone, as a worker
would, and its memory is measured; the tests then run in parallel on as many
processes as limit() allows for that much each, and the slowest are listed. The
image build tests (marked image_build) run apart and keep no record, since
what they cover is not Python: when the change touches IMAGE_CODE or an image's
folder, and with --all.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ENVIRON, ROOT, UV, Part, always, base_of, changed_paths, execute, main
from wpt_utils.lines import ERROR, line, say
from wpt_utils.parallel import average, limit

SAMPLE_EVERY = 40  # every this many-th chosen test is in the sample whose memory sizes the test processes
SLOWEST = 20  # how many of the slowest tests a run lists
NOT_IMAGES = ["-m", "not image_build"]
SELECT, EVERY = ["--testmon-forceselect"], ["--testmon-noselect"]  # the arguments that make testmon choose, or not
STARTUP_FOLDER = "ci/startup"  # its files run before coverage or only in a child pytest: hashed whole, whatever their kind
STARTUP = ROOT / STARTUP_FOLDER  # the startup file of the project's Python, and the pytest plugin beside it
NO_GPU = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
# what a pytest-xdist worker sets for itself (the root conftest.py), which the sample holds too
AS_A_WORKER = {**NO_GPU, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# what the image build test covers: a change to any of these, or to an image's folder, runs it
IMAGE_CODE = ("packages/wpt-agent/src/wpt_agent/images.py", "apps/wpt/tests/agent/test_images.py",
              "packages/wpt-agent/tests/test_agent_image.py",
              "conftest.py",  # the image fixture
              "packages/wpt-recipes/src/wpt_recipes/recipes.py",  # Recipes.image_files: what the image's tag covers
              "packages/wpt-recipes/src/wpt_recipes/schemas/image-lock.schema.json",
              "packages/wpt-recipes/src/wpt_recipes/schemas/image-licenses.schema.json")
# whether an image's folder differs from the revision: only the service layer names the data folder
IMAGES_DIFFER = "import sys; from wpt_recipes import Recipes; print(Recipes('.').images_differ(sys.argv[1]))"


def record_folder() -> Path:
    given = os.environ.get("WPT_TEST_RECORD")
    if given:
        return Path(given)
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return Path(base or Path.home() / ".cache") / "wpt-test-record"


def others() -> dict[str, str]:
    """{path: sha256} of the files coverage does not see, as they are."""
    found = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "packages", "apps",
                            "pyproject.toml", "uv.lock", STARTUP_FOLDER], cwd=ROOT, capture_output=True, text=True, check=True)
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in sorted(set(found.stdout.splitlines()))
            if (not path.endswith(".py") or path.startswith(f"{STARTUP_FOLDER}/")) and (ROOT / path).is_file()}


def selection(data, kept, now, everything) -> tuple[list[str], str]:
    """(pytest's arguments that choose the tests, the line that says what
    runs and why). data: the record's file; kept: the file of others() the
    last passing run left; now: others() as the files are; everything: --all."""
    if everything:
        return EVERY, "every test runs: --all"
    if not data.exists() or not kept.exists():
        return EVERY, "every test runs: there is no record"
    saved = json.loads(kept.read_text(encoding="utf-8"))
    changed = sorted(path for path in saved.keys() | now.keys() if saved.get(path) != now.get(path))
    if changed:
        more = f" and {len(changed) - 5} more" if len(changed) > 5 else ""
        return EVERY, f"every test runs: a file only they can cover changed: {', '.join(changed[:5])}{more}"
    return SELECT, "the tests a change touches run"


def images_changed(base) -> bool:
    """Whether an image's folder differs from base."""
    folders = subprocess.run([UV, "run", "python", "-c", IMAGES_DIFFER, base], cwd=ROOT, env=ENVIRON,
                             capture_output=True, text=True)
    if folders.returncode:
        sys.exit(line(f"cannot tell whether an image's folder changed since {base}: {folders.stderr.strip()}", ERROR))
    return folders.stdout.strip() == "True"


def image_tests(base) -> tuple[bool, str]:
    """(whether the image build test runs, the line that says why): it runs
    only when the paths changed since base hold a file of IMAGE_CODE or an
    image's folder."""
    touched = [path for path in changed_paths(base) if path in IMAGE_CODE]
    if images_changed(base):
        touched.append("an image's folder under data/images/")
    if touched:
        return True, f"the image build test runs: the change touches {', '.join(touched)}"
    return False, "the image build test is left out: the change touches no image code"


def project_prefix() -> str:
    found = subprocess.run([UV, "run", "python", "-c", "import sys; print(sys.prefix)"], cwd=ROOT, env=ENVIRON,
                           capture_output=True, text=True, check=True)
    return found.stdout.strip()


def environment(cpu) -> dict[str, str]:
    """What the tests run under: the startup file on the search path, told
    which Python is the project's; with cpu, the GPU hidden and the CPU
    forced, which also blocks Triton (ci/startup/sitecustomize.py). The two
    go together: ROCm's startup breaks with a visible GPU and no Triton."""
    path = os.pathsep.join(place for place in (str(STARTUP), ENVIRON.get("PYTHONPATH")) if place)
    env = {**ENVIRON, "PYTHONPATH": path, "WPT_CI_PYTHON": project_prefix()}
    return {**env, **NO_GPU, "WPT_CPU": "1"} if cpu else env


def collected(chosen, env) -> list[str]:
    """The ids of the tests pytest collects with these arguments, in order,
    as a worker collects them; no test is an empty list."""
    found = subprocess.run([UV, "run", "pytest", "--collect-only", "-q", "-n", "0", *NOT_IMAGES, *chosen], cwd=ROOT,
                           env={**env, **AS_A_WORKER}, capture_output=True, text=True)
    if found.returncode not in (0, 5):  # 5: no test
        sys.exit(line(f"pytest --collect-only failed:\n{found.stdout}{found.stderr}", ERROR))
    return [text for text in found.stdout.splitlines() if "::" in text]


def sample_of(test_ids, every=SAMPLE_EVERY):
    """The sample of test ids that runs alone first: every `every`-th, from
    the first, in collection order, so the same tests give the same sample."""
    return test_ids[::every]


def test_processes(test_ids, env) -> tuple[int, str]:
    """(how many test processes, the line that says why): a sample of the
    tests runs alone, its average memory is measured, and limit() sizes the processes
    from it, one test process being taken to hold that much; never more
    processes than tests."""
    sample = sample_of(test_ids)
    say(f"$ pytest {len(sample)} sample tests alone, to measure their memory")
    code, mean = average([UV, "run", "pytest", "-q", "-n", "0", *sample], cwd=ROOT, env={**env, **AS_A_WORKER})
    count, why = limit(mean, "test workers")
    held = "its memory could not be measured" if mean is None else f"it held {mean / 1e9:.1f} GB on average"
    return (min(count, len(test_ids)),
            f"{len(sample)} sample tests ran alone ({'passed' if code == 0 else 'FAILED'}); {held}\n{why}")


def pytest(count, *arguments) -> list[str]:
    """The parallel run of pytest, which lists what it skipped and its slowest tests."""
    return [UV, "run", "pytest", "-rs", "-n", str(count), "--dist", "loadgroup", f"--durations={SLOWEST}",
            "--durations-min=1", *arguments]


def run(given) -> bool:
    folder = record_folder()
    folder.mkdir(parents=True, exist_ok=True)
    name = "cpu" if given.cpu else "machine"
    data, kept = folder / f"{name}.testmondata", folder / f"{name}.others.json"
    env = {**environment(given.cpu), "TESTMON_DATAFILE": str(data)}
    now = others()
    chosen, why = selection(data, kept, now, given.all)
    say(why)
    images, image_why = (True, "the image build test runs: --all") if given.all else image_tests(base_of(given))
    say(image_why)

    test_ids = collected([*SELECT, "--testmon-nocollect"] if chosen == SELECT else [], env)
    say(f"{len(test_ids)} tests chosen")
    passed, count = True, None
    if test_ids:
        count, sizing = test_processes(test_ids, env)
        say(sizing)
        passed = execute([pytest(count, "-p", "child_files", *NOT_IMAGES, *chosen)], env=env)
    if passed:
        kept.write_text(json.dumps(now, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
    if images:
        count = count or limit(None, "test workers")[0]  # their memory is Docker's, not the test process's
        passed = execute([pytest(count, "-m", "image_build")], env=env) and passed
    return passed


PART = Part("tests", always, run)

if __name__ == "__main__":
    main(PART)
