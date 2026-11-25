"""The messages of the exchange between the freight and a preparation
(docs/api/preparation.md), on byte streams: each one line of JSON in UTF-8
ended by a line feed; a message that carries content names its length in
`bytes`, and exactly that many bytes follow the line.

    read(stream)                      the next message and its content, None at the stream's end
    write(stream, message, content)   one message, with its content when given
"""

import json


class BadMessage(Exception):
    """A line that is no message, or content cut short."""


def read(stream) -> tuple[dict, bytes | None] | None:
    """(message, its content or None) read from stream; None at the
    stream's end. BadMessage for a line that is no JSON object, or content
    shorter than its `bytes` say."""
    line = stream.readline()
    if not line:
        return None
    try:
        message = json.loads(line)
    except ValueError:
        raise BadMessage(f"a line that is no message: {line[:200]!r}") from None
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise BadMessage(f"a line that is no message: {line[:200]!r}")
    size = message.get("bytes")
    if size is None:
        return message, None
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
