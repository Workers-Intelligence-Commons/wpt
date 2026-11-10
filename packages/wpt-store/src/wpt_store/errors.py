"""The store's failures (see wpt_store)."""


class StoreError(Exception):
    pass


class StoreSettingError(StoreError):
    pass


class MissingFile(StoreError):
    pass


class CorruptFile(StoreError):
    pass


class MixedRecord(StoreError):
    """A record file is written whole or added to line by line, never both."""
