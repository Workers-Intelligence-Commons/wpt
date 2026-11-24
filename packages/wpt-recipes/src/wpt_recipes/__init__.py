"""WPT's recipes and task images, read and checked.

A model recipe (data/recipes/models/<name>.json) says everything that
changes a model: its data (the address of a depot, a snapshot there with
its change and digest, the kinds and shares by tag it takes, its total
tokens and its vocabulary's sha256 and size; recipes written before
snapshots name a dataset of the depot instead, whose tokenizer recipe
gives the vocabulary), its architecture, its seed, its training steps and its
drafter (docs/model.md). A series (data/recipes/series/<name>.json) holds
the rules several sizes of one model share, and series.py derives each
size's model recipe from it, for the vocabulary's size. A task image
(data/images/<name>/) is the Dockerfile of the container agentic tasks run
in, its lock, its packages' licenses and its shared pool of tools.

Recipes (recipes.py) is the only code that opens these files; guard.py
refuses any other (docs/architecture.md, "The service layer"). A recipe
a person hands over as JSON, read elsewhere (a depot's build), is read
through data_section, which gives its data part.
"""

from . import series
from .errors import InvalidRecipe, NotFound, RecipeError
from .recipes import ASK, BUILT_IN_TOOLS, SIZE_OF, Recipes, data_section, dump_json, validate

__all__ = ["ASK", "BUILT_IN_TOOLS", "SIZE_OF", "InvalidRecipe", "NotFound", "RecipeError", "Recipes", "data_section",
           "dump_json", "series", "validate"]
