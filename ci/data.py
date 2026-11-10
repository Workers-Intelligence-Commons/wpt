# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""The rules on every recipe, tokenizer lock and image of data/
(wpt_recipes.Recipes.check), read through the service layer. Options:
common.py."""

from common import UV, Part, execute, main, under

# data/, and the code the check runs
TOUCHED = under("data/", "packages/wpt-recipes/")
# one line per problem; exit code 1 when there is one
CHECK = ("import sys; from wpt_recipes import Recipes; problems = Recipes('.').check(); "
         "print(*problems, sep='\\n') if problems else None; sys.exit(1 if problems else 0)")

PART = Part("data", TOUCHED, lambda given: execute([[UV, "run", "python", "-c", CHECK]]))

if __name__ == "__main__":
    main(PART)
