# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""wpt dataset check-commits RANGE: sign-offs and decision signatures on the
commits of the range, what a push sends. It always runs, whatever changed. Options:
common.py."""

from common import UV, Part, always, base_of, execute, main


def run(given):
    return execute([[UV, "run", "wpt", "dataset", "check-commits", given.range_ or f"{base_of(given)}..HEAD"]])


PART = Part("commits", always, run)

if __name__ == "__main__":
    main(PART)
