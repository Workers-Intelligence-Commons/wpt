"""What reading or checking a recipe or a task image can fail with."""


class RecipeError(Exception):
    """A recipe or an image cannot be read as asked; nothing was written."""


class NotFound(RecipeError):
    """No such recipe, series, size of it, run of a size, image, or file of
    an image."""


class InvalidRecipe(RecipeError):
    """A file the check refuses: it does not parse, does not match its
    schema, or breaks a rule across files (a recipe naming a tokenizer
    recipe that does not exist, a depot or dataset not named yet, a series
    whose sizes cannot be derived, an image's pool of tools that takes a
    built-in tool's name)."""
