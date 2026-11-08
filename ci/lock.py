# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""uv.lock matches every pyproject.toml. Options: common.py."""

from common import PROJECT, UV, Part, execute, main

PART = Part("lock", lambda path: path in PROJECT or path.endswith("/pyproject.toml"),
            lambda given: execute([[UV, "lock", "--check"]]))

if __name__ == "__main__":
    main(PART)
