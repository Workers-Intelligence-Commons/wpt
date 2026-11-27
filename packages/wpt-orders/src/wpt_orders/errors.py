"""What reading or checking an order or a task image can fail with."""


class OrderError(Exception):
    """An order or an image cannot be read as asked; nothing was written."""


class NotFound(OrderError):
    """No such family, size of it, run of a size, image, or file of an
    image, or a name of an order that names no size."""


class InvalidOrder(OrderError):
    """A file the check refuses: it does not parse, does not match its
    schema, or breaks a rule (a family's table that writes a setting twice
    or leaves a value empty, a size whose order the model order's schema
    refuses, a data part that names a dataset, a depot or snapshot not named
    yet, shares that name a tag twice or sum above 1, an image's pool of
    tools that takes a built-in tool's name)."""
