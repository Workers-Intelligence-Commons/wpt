# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""A sign-off on every commit of the range, what a push sends: each commit
(merge commits aside) carries `Signed-off-by: <its author>`, by which the
author certifies the Developer Certificate of Origin (CONTRIBUTING.md). It
always runs, whatever changed. Options: common.py."""

import subprocess
import sys

from common import ROOT, Part, always, base_of, main
from wpt_utils.lines import ERROR, say


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def unsigned(revision_range: str) -> list[str]:
    """One line per commit of revision_range without its author's sign-off."""
    problems = []
    for line in git("rev-list", "--reverse", "--parents", revision_range).splitlines():
        commit, *parents = line.split()
        if len(parents) > 1:
            continue  # a merge commit carries no change of its own
        author = git("show", "-s", "--format=%an <%ae>", commit).strip()
        if f"Signed-off-by: {author}" not in git("show", "-s", "--format=%B", commit).splitlines():
            problems.append(f"{commit[:12]}: no 'Signed-off-by: {author}' line (commit with `git commit -s`)")
    return problems


def run(given) -> bool:
    revision_range = given.range_ or f"{base_of(given)}..HEAD"
    say(f"$ the sign-off of every commit of {revision_range}")
    try:
        problems = unsigned(revision_range)
    except subprocess.CalledProcessError as error:
        say(f"git {error.cmd[1]}: {error.stderr.strip()}", ERROR, sys.stdout)
        return False
    for problem in problems:
        say(problem, ERROR, sys.stdout)
    return not problems


PART = Part("commits", always, run)

if __name__ == "__main__":
    main(PART)
