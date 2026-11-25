"""One item of a corpus as its tokens (docs/data-selection.md, "Corpora").

A plain text, or a piece of one, becomes its tokens followed by END,
encoded a block at a time: each block ends right before a line's first
byte that is not whitespace, a place where the tokenizer's chunks always
break, so its tokens are those of the whole text (Encoder). A conversation
becomes the tokens of the chat template (wpt_tokenizer.chat), which start
with END; each excerpt is written as the passage it quotes. A task is kept
as one JSON line of tasks.jsonl, never tokenized, and weighs the tokens of
its content as that JSON text (task_text).

    tokens(tokenizer, kind, content)       a plain text's or a conversation's tokens, as an array
    weight(tokenizer, kind, content)       the tokens an item counts for a corpus's size
    task_text(content)                     a task's content as the JSON text its tokens count
"""

import json

import numpy as np
from wpt_tokenizer import END, Tokenizer, chat

PLAIN, CONVERSATION, TASK = "plain", "conversation", "task"
KINDS = (CONVERSATION, PLAIN, TASK)  # the kinds a corpus prepares
TOKEN_KINDS = (PLAIN, CONVERSATION)  # the kinds kept as token files, each apart
WHITESPACE = b" \t\n\r\x0b\x0c"  # the bytes the tokenizer's chunks treat as whitespace (wpt_tokenizer)
WHITESPACE_BYTES = np.frombuffer(WHITESPACE, np.uint8)


def tokens(tokenizer: Tokenizer, kind: str, content: bytes) -> np.ndarray:
    """The tokens of a plain text or a piece (END after it), or of a
    conversation (its JSON as UTF-8 bytes, through the chat template), as
    uint32."""
    if kind == PLAIN:
        encoder = Encoder(tokenizer)
        encoder.feed(content)
        return np.concatenate([encoder.finish(), np.array([END], np.uint32)])
    if kind == CONVERSATION:
        return np.asarray(chat.encode(tokenizer, json.loads(content))[0], np.uint32)
    raise ValueError(f"an item of kind {kind} is not tokenized")


def weight(tokenizer: Tokenizer, kind: str, content: bytes) -> int:
    """The tokens an item counts for a corpus's size: a plain text's and a
    conversation's tokens, a task's tokens of its content as JSON text."""
    if kind == TASK:
        return task_tokens(tokenizer, json.loads(content))
    return len(tokens(tokenizer, kind, content))


def task_text(content) -> str:
    """A task's content as JSON text: what a task counts for a corpus's
    size, and the form of its line in tasks.jsonl."""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def task_tokens(tokenizer: Tokenizer, content) -> int:
    """The tokens a task's content counts for a corpus (task_text)."""
    return len(tokenizer.encode(task_text(content).encode("utf-8")))


class Encoder:
    """The tokens of a plain text fed a part at a time (UTF-8 bytes or str):
    each part is encoded up to its last place where the tokenizer's chunks
    always break (boundary), so the tokens are those of the whole text, and
    only the bytes after that place are held, whatever the text's lines."""

    def __init__(self, tokenizer):
        self.tokenizer, self.pending, self.last, self.tokens = tokenizer, [], None, []

    def feed(self, value: bytes | str):
        data = value.encode("utf-8") if isinstance(value, str) else value
        if not data:
            return
        cut = boundary(data, self.last)
        if cut is None:
            self.pending.append(data)
        else:
            left = b"".join(self.pending) + data[:cut]
            if left:
                self.tokens.append(np.asarray(self.tokenizer.encode(left), np.uint32))
            self.pending = [data[cut:]]
        self.last = data[-1]

    def finish(self) -> np.ndarray:
        left = b"".join(self.pending)
        if left:
            self.tokens.append(np.asarray(self.tokenizer.encode(left), np.uint32))
        return np.concatenate(self.tokens) if self.tokens else np.zeros(0, np.uint32)


def boundary(data: bytes, last: int | None) -> int | None:
    """The last place of data, read right after the bytes that end with
    last (None at a text's start), where a byte that is not whitespace is
    followed by whitespace (WHITESPACE): the tokenizer's chunk of that byte
    ends there whatever follows, so the bytes before it encode as they would
    within the whole text. None for no such place."""
    spaces = np.isin(np.frombuffer(data, np.uint8), WHITESPACE_BYTES)
    found = np.flatnonzero(~spaces[:-1] & spaces[1:])
    if len(found):
        return int(found[-1]) + 1
    if last is not None and last not in WHITESPACE and spaces[0]:
        return 0
    return None
