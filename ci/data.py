# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""wpt dataset check: the data rules on every source, read, not downloaded.
Options: common.py."""

from common import UV, Part, execute, main, under

# data/, and the code the check runs: the command, wpt-dataset and the packages it depends on
TOUCHED = under("data/", "apps/wpt/src/wpt/__init__.py", "apps/wpt/src/wpt/dataset.py",
                "apps/wpt/src/wpt/sessions.py", "packages/wpt-dataset/", "packages/wpt-store/",
                "packages/wpt-telemetry/", "packages/wpt-tokenizer/", "packages/wpt-utils/")

PART = Part("data", TOUCHED, lambda given: execute([[UV, "run", "wpt", "dataset", "check"]]))

if __name__ == "__main__":
    main(PART)
