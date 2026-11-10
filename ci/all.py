# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""Run every part, each one even when an earlier one failed, and list what
passed, what failed and what was left out, with the time each took. Exits with
1 when a part failed. Options: common.py."""

import sys

import commits
import data
import lock
import rust
import tests
from common import base_of, changed_paths, duration, options, perform
from wpt_utils.lines import ERROR, INFO, say

PARTS = (lock.PART, tests.PART, rust.PART, data.PART, commits.PART)


def main():
    given = options()
    changed = None if given.all else changed_paths(base_of(given))
    results = [(part.name, *perform(part, given, changed)) for part in PARTS]
    say("")
    for name, result, took in results:
        say(f"{result}  {name}{f' in {duration(took)}' if took else ''}", ERROR if result == "FAILED" else INFO, sys.stdout)
    say(f"every part together: {duration(sum(took for _, _, took in results))}")
    sys.exit(1 if any(result == "FAILED" for _, result, _ in results) else 0)


if __name__ == "__main__":
    main()
