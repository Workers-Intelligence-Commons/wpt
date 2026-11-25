"""The exchange between the freight and one preparation, a program it
starts as a child process and talks to through its standard input and
output (docs/api/preparation.md; the messages: freight.messages).

Preparation starts the program found by its name (find: beside the Python
the freight runs on, else on the search path), sends `start` and reads
`ready`. Its messages go to the program from a thread of their own, so the
freight never waits on a full pipe while the program waits on it. Then:

    weigh(items)        the weight of each (kind, content), in order
    take(item, content) send one item to prepare
    taken()             the next answer to a take; the files said before it wait in files
    end()               the result, once every file was said
    close()             stop the program, whatever it was doing

A `log` message goes to the job's log as it comes. A `failed` message, a
line that is no message, or a program that ends without `done` raise
PreparationFailed.
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from wpt_utils.lines import say
from wpt_utils.programs import program

from . import messages

_STOP = object()
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class PreparationFailed(Exception):
    """The preparation could not do its part: the job fails with its reason."""


def find(name: str) -> str:
    """The program of that name: beside the Python the freight runs on
    (the programs installed with it), else on the search path."""
    folder = Path(sys.executable).parent
    for candidate in (folder / name, folder / f"{name}.exe"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return program(name)


class Preparation:
    """One run of the program name: started with its settings, its inputs
    ({name: local folder}), the folder out it writes into, the threads it
    may use and resume (the files a former run said, and the last item
    they hold) when it goes on. ready: its answer to start."""

    def __init__(self, name: str, settings: dict, inputs: dict[str, str], out: Path, threads: int,
                 resume: dict | None = None, log=say):
        self.name, self.log = name, log
        self.files = []  # the file messages read and not yet handed out
        self.process = subprocess.Popen([find(name)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # unbounded: a caller sends a batch before it reads the answers, which the program may write before it
        # reads the rest of the batch; the caller bounds what one batch holds
        self._waiting = queue.Queue()
        self._sent_error = None
        self._sender = threading.Thread(target=self._send_all, name=f"to-{name}", daemon=True)
        self._sender.start()
        start = {"type": "start", "settings": settings, "inputs": inputs, "out": str(out), "threads": threads}
        if resume is not None:
            start["resume"] = resume
        self._send(start)
        ready = self._next()
        if ready["type"] != "ready":
            self.close()
            raise PreparationFailed(f"{name} answered start with {ready['type']}, not ready")
        self.ready = ready

    # Sending

    def _send_all(self):
        stream = self.process.stdin
        try:
            while True:
                found = self._waiting.get()
                if found is _STOP:
                    break
                message, content = found
                messages.write(stream, message, content)
                if self._waiting.empty():
                    stream.flush()
            stream.close()
        except (OSError, ValueError) as error:  # the program ended: the reader says why
            self._sent_error = error

    def _send(self, message: dict, content: bytes | None = None):
        self._waiting.put((message, content))

    # Reading

    def _next(self) -> dict:
        """The next message of the program that is no log line (each log line written as it comes)."""
        while True:
            try:
                found = messages.read(self.process.stdout)
            except messages.BadMessage as error:
                self.close()
                raise PreparationFailed(f"{self.name} wrote {error}") from None
            if found is None:
                code = self.process.wait()
                raise PreparationFailed(f"{self.name} ended (exit code {code}) without saying it was done")
            message, _ = found
            if message["type"] == "log":
                level = message.get("level") if message.get("level") in LEVELS else "INFO"
                self.log(f"{self.name}: {message.get('text', '')}", level)
                continue
            if message["type"] == "failed":
                self.close()
                raise PreparationFailed(f"{self.name}: {message.get('reason', 'no reason given')}")
            return message

    def _answer(self, wanted: str) -> dict:
        """The next answer of that type, the file messages before it kept in files."""
        while True:
            message = self._next()
            if message["type"] == "file":
                self.files.append(message)
                continue
            if message["type"] != wanted:
                self.close()
                raise PreparationFailed(f"{self.name} answered {message['type']} where {wanted} was due")
            return message

    # The exchange

    def weigh(self, items: list[tuple[str, bytes]], first_ref: int) -> list[int]:
        """The weight of each (kind, content) of items, in order, asked with
        the refs first_ref, first_ref + 1, ...: the program answers them in
        any order."""
        for at, (kind, content) in enumerate(items):
            self._send({"type": "weigh", "ref": first_ref + at, "kind": kind}, content)
        found = {}
        while len(found) < len(items):
            message = self._answer("weight")
            found[message["ref"]] = message["weight"]
        return [found[first_ref + at] for at in range(len(items))]

    def take(self, item: dict, content: bytes):
        """Send one item to prepare: {ref, id, name, depot, piece, kind, pool}."""
        self._send({"type": "take", **item}, content)

    def taken(self) -> dict:
        """The next answer to a take: {ref, weight, at}."""
        return self._answer("taken")

    def end(self) -> dict:
        """Say there are no more items: the program's result, once its last file was said (in files)."""
        self._send({"type": "end"})
        self._waiting.put(_STOP)
        done = self._answer("done")
        self.process.wait()
        return done.get("result") or {}

    def close(self):
        """Stop the program, whatever it was doing."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self._waiting.put(_STOP)
        try:
            self.process.stdout.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
