"""WPT's recipes and task images, read and checked.

A model recipe (data/recipes/models/<name>.json) says everything that
changes a model: its data (the address of a depot, the name of a dataset
there, its tokenizer recipe and its total tokens), its architecture, its
seed, its training steps and its drafter (docs/model.md). A series
(data/recipes/series/<name>.json) holds the rules several sizes of one model
share, and series.py derives each size's model recipe from it. A tokenizer
recipe (data/recipes/tokenizers/<name>.json) names a depot, a dataset there
and the bytes of its texts to sample, and its lock lists the texts the
vocabulary was trained on, by depot and id (docs/tokenizer.md). A task image
(data/images/<name>/) is the Dockerfile of the container agentic tasks run
in, its lock, its packages' licenses and its shared pool of tools.

Recipes (recipes.py) is the only code that opens these files; guard.py
refuses any other (docs/architecture.md, "The service layer").
"""

from . import series
from .errors import InvalidRecipe, NotFound, RecipeError
from .recipes import ASK, BUILT_IN_TOOLS, SIZE_OF, Recipes, dump_json, validate

__all__ = ["ASK", "BUILT_IN_TOOLS", "SIZE_OF", "InvalidRecipe", "NotFound", "RecipeError", "Recipes", "dump_json",
           "series", "validate"]
