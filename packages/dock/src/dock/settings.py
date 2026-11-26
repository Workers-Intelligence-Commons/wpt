"""What every part of the dock reads alike: .env files (wic_utils.env), and the time."""

from datetime import UTC, datetime

from wic_utils.env import read_env_file

__all__ = ["now", "read_env_file"]


def now() -> str:
    """This moment in UTC, as a dock's records write times."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
