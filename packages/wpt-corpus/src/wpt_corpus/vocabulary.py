"""The program wpt-vocabulary, which the freight runs to learn a
vocabulary (docs/api/preparation.md): it reads the freight's messages on
its standard input and answers on its standard output.

Its one setting, vocab_size, is the size of the vocabulary it learns. It
prepares plain texts only, has no unit and weighs nothing: a vocabulary's
sample is a size in bytes, which the freight measures itself. Each text or
piece it takes goes to the tokenizer's training as it comes
(wpt_tokenizer.Tokenizer.train, which reads its documents once, as a
stream, in the order sent), and is answered at once, with no weight. At the end it writes
vocabulary.json, the canonical vocabulary file, and answers {sha256, size,
tokenizer_version}. The same texts in the same order give the same file.

The vocabulary learns from every text at once, so a job cut short cannot go
on: the program refuses a start that resumes, and the freight starts the
job over. An erasure that reaches a text it read does not make its delivery
stale (on_erasure "note"): the merges learned from it stay
(docs/tokenizer.md).
"""

import hashlib
import sys
from pathlib import Path

from wpt_tokenizer import BASE_SIZE, Tokenizer
from wpt_tokenizer import VERSION as TOKENIZER_VERSION

from . import channel
from .encoding import PLAIN

NAME = "wpt-vocabulary"
VERSION = f"1+tokenizer-{TOKENIZER_VERSION}"  # raised whenever the same texts give another file
FILE = "vocabulary.json"
SETTINGS = {"vocab_size"}


class Failed(Exception):
    """The program cannot do its part: the reason goes to the freight."""


def wpt_vocabulary():
    """The program wpt-vocabulary."""
    sys.exit(run(sys.stdin.buffer, sys.stdout.buffer))


def run(stdin, stdout) -> int:
    """Answer the freight's messages read from stdin on stdout: the exit code."""
    def send(message: dict):
        channel.write(stdout, message)
        stdout.flush()

    try:
        first = channel.read(stdin, channel.CONTENT)
        if first is None:
            return 0
        start, _ = first
        vocab_size, folder = _start(start)
        send({"type": "ready", "name": NAME, "version": VERSION, "unit": None, "kinds": [PLAIN],
              "weighs": False, "on_erasure": "note"})
        read = {"texts": 0, "ended": False}

        def documents():
            while (found := channel.read(stdin, channel.CONTENT)) is not None:
                message, content = found
                if message["type"] == "end":
                    read["ended"] = True
                    return
                if message["type"] != "take":
                    raise Failed(f"{NAME} weighs nothing and answers no {message['type']} message")
                if message.get("kind") != PLAIN:
                    raise Failed(f"{NAME} learns from plain texts only, not from an item of kind {message.get('kind')}")
                send({"type": "taken", "ref": message["ref"]})
                read["texts"] += 1
                yield content

        tokenizer = Tokenizer.train(documents(), vocab_size)
        if not read["ended"]:
            return 0  # the freight closed the channel before end: it wants nothing more
        if not read["texts"]:
            raise Failed(f"{NAME} was sent no text: a vocabulary learns from at least one")
        data = tokenizer.to_bytes()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / FILE).write_bytes(data)
        send({"type": "file", "path": FILE, "items": read["texts"]})
        send({"type": "done", "result": {"sha256": hashlib.sha256(data).hexdigest(), "size": tokenizer.vocab_size,
                                         "tokenizer_version": TOKENIZER_VERSION}})
        return 0
    except (Failed, channel.BadMessage, OSError, ValueError, KeyError) as error:
        send({"type": "failed", "reason": f"{type(error).__name__}: {error}"})
        return 1


def _start(start: dict) -> tuple[int, Path]:
    """(vocab_size, the folder it writes into) of the freight's start message."""
    if start["type"] != "start":
        raise Failed(f"the first message is {start['type']}, not start")
    if start.get("resume"):
        raise Failed(f"{NAME} learns from every text at once and cannot go on after a stop: start the job over")
    settings = start.get("settings") or {}
    if set(settings) - SETTINGS:
        raise Failed(f"no setting {', '.join(sorted(set(settings) - SETTINGS))}: {NAME} takes vocab_size")
    vocab_size = settings.get("vocab_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < BASE_SIZE:
        raise Failed(f"vocab_size is a whole number of at least {BASE_SIZE}: {vocab_size!r}")
    return vocab_size, Path(start["out"])
