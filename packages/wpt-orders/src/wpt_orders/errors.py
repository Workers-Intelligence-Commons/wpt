"""What reading or checking an order or a task image can fail with."""


class OrderError(Exception):
    """An order or an image cannot be read as asked; nothing was written."""


class NotFound(OrderError):
    """No such order, series, size of it, run of a size, image, or file of
    an image."""


class InvalidOrder(OrderError):
    """A file the check refuses: it does not parse, does not match its
    schema, or breaks a rule across files (a depot, snapshot or dataset not
    named yet, shares that name a tag twice or sum above 1, a series whose sizes cannot be derived, an image's pool of tools
    that takes a built-in tool's name)."""
