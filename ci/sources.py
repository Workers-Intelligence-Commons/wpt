# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""wpt dataset check --fetch SMALL_SOURCE: one small source fetched and checked,
to prove the fetch and its checks work. The sources a pull request adds or
changes are fetched by the maintainer. Options: common.py."""

from common import UV, Part, execute, main
from data import TOUCHED

SMALL_SOURCE = "standardebooks-joseph-conrad-ford-madox-ford-the-nature-of-a-crime-5366b0f3"  # a short book

PART = Part("sources", TOUCHED, lambda given: execute([[UV, "run", "wpt", "dataset", "check", "--fetch", SMALL_SOURCE]]))

if __name__ == "__main__":
    main(PART)
