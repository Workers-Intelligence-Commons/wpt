"""The kinds of structured items: the rows a new depot's kinds table starts
with, and the check of an item's content against its kind's schema as the
depot stores it.

Each starting kind's schema is a file of the folder kinds/ beside this
module: a conversation in the chat format (docs/chat-format.md), a task
with its starting files and its check, a person's rating of an attempt,
and a probe of the probes test (docs/evaluation.md).
"""

import json
from importlib.resources import files

from jsonschema import Draft202012Validator

from .errors import BadContent

MEANINGS = {
    "conversation": "a conversation in the chat format, which the answer-as-assistant step learns from",
    "task": "a task for agentic training: what it asks, its starting files and its check, which the finish-tasks "
            "steps play",
    "rating": "a person's pass or fail on one attempt of a wpt eval run, a measure only",
    "probe": "a question the probes test asks a model and the patterns that flag a reply as alarming, a measure only",
}


def starting_rows() -> list[dict]:
    """The kinds table's rows of a new depot."""
    folder = files(__package__) / "kinds"
    return [{"id": kind, "meaning": meaning, "schema": (folder / f"{kind}.schema.json").read_text(encoding="utf-8")}
            for kind, meaning in MEANINGS.items()]


def check_content(kind: str, schema: str, content: str):
    """BadContent unless content is JSON that matches schema, the JSON
    Schema the kind's row holds."""
    try:
        value = json.loads(content)
    except ValueError as error:
        raise BadContent(f"an item of kind {kind} holds JSON: {error}") from None
    problems = sorted(Draft202012Validator(json.loads(schema)).iter_errors(value), key=lambda e: list(e.path))
    if problems:
        where = "/".join(map(str, problems[0].path)) or "the content"
        raise BadContent(f"an item of kind {kind} does not match its schema at {where}: {problems[0].message}"
                         + (f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""))
