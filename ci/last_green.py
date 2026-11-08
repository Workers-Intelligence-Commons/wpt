# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""The commit of the last successful ci run of a branch, which CI compares
with, written as `commit=SHA` into the file GITHUB_OUTPUT names; no green run
leaves the value empty.

Usage: ci/last_green.py REPOSITORY BRANCH EVENT

A run reads only the record a run of its own kind of event kept, so EVENT
narrows the search to that kind. A commit this clone does not hold (the branch
was rewritten) counts as no green run, and so does an API that does not answer:
everything then runs.
"""

import os
import subprocess
import sys

from common import ROOT
from wpt_utils.lines import WARNING, say


def last_green(repository, branch, event) -> str | None:
    found = subprocess.run(["gh", "api", "--method", "GET", f"repos/{repository}/actions/workflows/ci.yml/runs",
                            "-f", f"branch={branch}", "-f", f"event={event}", "-f", "status=success",
                            "-f", "per_page=1", "--jq", ".workflow_runs[0].head_sha // empty"],
                           capture_output=True, text=True)
    if found.returncode:
        say(f"cannot read the last green run: {found.stderr.strip()}", WARNING)
        return None
    commit = found.stdout.strip()
    if commit and subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
                                 capture_output=True).returncode:
        say(f"the last green commit {commit} is not in this clone", WARNING)
        return None
    return commit or None


def main():
    commit = last_green(*sys.argv[1:4])
    say(f"last green commit: {commit or 'none'}")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8", newline="\n") as output:
        output.write(f"commit={commit or ''}\n")


if __name__ == "__main__":
    main()
