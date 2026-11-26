"""The settings file every part reads beside its environment: a `.env`
file of KEY=VALUE lines."""

from pathlib import Path


def read_env_file(path) -> dict:
    """KEY=VALUE lines; blank lines and # comments are skipped."""
    pairs = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return pairs
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key.strip()] = value.strip().strip("'\"")
    return pairs
