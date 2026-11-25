"""The messages of the exchange between the freight and a preparation
(docs/api/preparation.md), on byte streams: each one line of JSON in UTF-8
ended by a line feed. Only the freight's `weigh` and `take` carry content
(CONTENT): such a message names its length in `bytes`, and exactly that
many bytes follow the line. In any other message, `bytes` is a field like
the others.

    read(stream, carrying)            the next message and its content, None at the stream's end
    write(stream, message, content)   one message, with its content when given
"""

import json

CONTENT = ("weigh", "take")  # the messages that carry content, all of them the freight's


class BadMessage(Exception):
    """A line that is no message, or content cut short."""


def read(stream, carrying=()) -> tuple[dict, bytes | None] | None:
    """(message, its content or None) read from stream; None at the
    stream's end. Only a message whose type is in carrying has content.
    BadMessage for a line that is no JSON object, or content shorter than
    its `bytes` say."""
    line = stream.readline()
    if not line:
        return None
    try:
        message = json.loads(line)
    except ValueError:
        raise BadMessage(f"a line that is no message: {line[:200]!r}") from None
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise BadMessage(f"a line that is no message: {line[:200]!r}")
    if message["type"] not in carrying:
        return message, None
    size = message.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise BadMessage(f"a message whose bytes is no length: {line[:200]!r}")
    content = stream.read(size)
    if len(content) != size:
        raise BadMessage(f"a {message['type']} message names {size} bytes and the stream ended after {len(content)}")
    return message, content


def write(stream, message: dict, content: bytes | None = None):
    """Write message, and its content after it with its length in `bytes`."""
    if content is not None:
        message = {**message, "bytes": len(content)}
    stream.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    if content is not None:
        stream.write(content)
