"""What the parts of the checks share: their options, what a change touched,
and how a part runs, says why it was left out and prints its time.

A part is one script of this folder, and all.py runs every part in order. A
part's script holds a Part: its name, which changed paths concern it, and what
it runs. Every part takes the same options:

    --all         every part, whatever changed
    --cpu         the software conditions of CI: every GPU hidden, Triton
                  blocked, the CPU forced
    --base REV    what the change is compared with (default: where this branch
                  left its upstream branch, for example origin/main)
    --range R     the commits to check (default: BASE..HEAD)

A change is the difference from BASE to the files as they are, committed or
not, new files git does not track yet included. The tools come from the
machine, as in CI: uv, git, cargo, and mercurial and GnuPG for the sources that
need them.
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from wpt_utils.lines import ERROR, INFO, line, say

ROOT = Path(__file__).resolve().parent.parent
UV = os.environ.get("UV", "uv")  # uv tells the scripts it runs where it is
# these scripts run in their own environment; what they check runs in the project's
ENVIRON = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
PROJECT = ("pyproject.toml", "uv.lock")  # they decide which libraries every part's code runs on


@dataclass(frozen=True)
class Part:
    """One check: its name; touched(path), whether a changed path concerns
    it; run(options), whether it passed."""

    name: str
    touched: Callable[[str], bool]
    run: Callable[[argparse.Namespace], bool]

    def concerned_by(self, path) -> bool:
        """Whether a changed path concerns this part: by its own rule, or as
        the part's own code."""
        return path in (f"ci/{self.name}.py", "ci/common.py") or self.touched(path)


def always(path) -> bool:
    """The rule of a part that runs, whatever changed."""
    return True


def under(*places) -> Callable[[str], bool]:
    """The rule of a part that the files of PROJECT concern, and those whose
    path starts with one of places, tests aside."""
    return lambda path: path in PROJECT or (path.startswith(places) and "/tests/" not in path)


def options(arguments=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="every part, whatever changed")
    parser.add_argument("--cpu", action="store_true", help="every GPU hidden, Triton blocked, the CPU forced")
    parser.add_argument("--base", help="what the change is compared with (default: where HEAD left its upstream)")
    parser.add_argument("--range", dest="range_", help="the commits to check (default: BASE..HEAD)")
    return parser.parse_args(arguments)


def base_of(given) -> str:
    """The revision a change is compared with: --base, or where this branch
    left its upstream."""
    if given.base:
        return given.base
    for command in (["git", "merge-base", "HEAD", "@{upstream}"], ["git", "merge-base", "HEAD", "origin/main"]):
        found = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if found.returncode == 0:
            return found.stdout.strip()
    sys.exit(line("no upstream branch to compare with: pass --base", ERROR))


def changed_paths(base) -> list[str]:
    """The paths that changed since base, in the files as they are."""
    paths = []
    for command in (["git", "diff", "--name-only", "--no-renames", base],
                    ["git", "ls-files", "--others", "--exclude-standard"]):
        found = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if found.returncode:
            sys.exit(line(f"{' '.join(command)} failed: {found.stderr.strip()}", ERROR))
        paths += found.stdout.splitlines()
    return paths


def duration(seconds) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def execute(commands, folder=ROOT, env=None) -> bool:
    """Run commands in order, printing each; whether all of them passed."""
    for command in commands:
        say(f"$ {' '.join(command)}")
        try:
            if subprocess.run(command, cwd=folder, env=ENVIRON if env is None else env).returncode != 0:
                return False
        except FileNotFoundError:
            say(f"{command[0]}: not found on this machine", ERROR, sys.stdout)
            return False
    return True


def perform(part: Part, given, changed=None) -> tuple[str, float]:
    """(its result, its seconds) of one part: "passed", "FAILED" or "left
    out", each printed. changed: the changed paths, found here when None;
    --all leaves them unread."""
    say(f"\n== {part.name}")
    started = time.monotonic()
    if not given.all and part.touched is not always:
        changed = changed_paths(base_of(given)) if changed is None else changed
        if not any(part.concerned_by(path) for path in changed):
            say(f"left out  {part.name}: no change touches it")
            return "left out", 0.0
    try:
        passed = part.run(given)
    except SystemExit as ended:  # a part that gives up still counts as failed
        if isinstance(ended.code, str):
            print(ended.code, file=sys.stderr)  # already a line() of ERROR
        passed = False
    except subprocess.CalledProcessError as failed:
        say(f"{' '.join(map(str, failed.cmd))} failed with exit code {failed.returncode}", ERROR, sys.stdout)
        passed = False
    took = time.monotonic() - started
    say(f"{'passed' if passed else 'FAILED'}  {part.name} in {duration(took)}", INFO if passed else ERROR, sys.stdout)
    return "passed" if passed else "FAILED", took


def main(part: Part):
    """The entry of a part's own script: exits with 1 when the part failed."""
    sys.exit(1 if perform(part, options())[0] == "FAILED" else 0)
