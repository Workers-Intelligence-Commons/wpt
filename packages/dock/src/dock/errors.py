"""The dock's failures (see dock)."""


class DockError(Exception):
    pass


class DockSettingError(DockError):
    pass


class MissingFile(DockError):
    pass


class CorruptFile(DockError):
    pass


class MixedRecord(DockError):
    """A record file is written whole or added to line by line, never both."""
