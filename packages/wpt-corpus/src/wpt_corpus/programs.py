"""The programs the freight runs for WPT (docs/api/preparation.md): each
reads the freight's messages on its standard input and answers on its
standard output.

wpt-tokens (wpt_tokens) writes a corpus: each plain text, piece and
conversation it takes becomes its tokens (wpt_corpus.encoding) in a token
file of its kind (wpt_corpus.shards; plain and conversation apart), a row
for each item: its `source` the depot named in the take message, its
`document` the item's name, its `pool` the share's tag and its `pass` 0. A
task becomes one line of tasks.jsonl, {"depot", "id", "content"} as JSON
text (wpt_corpus.encoding.task_text). A token file closes once it holds
shard_tokens tokens (a setting, SHARD_TOKENS by default), and closes the
other kind's with it; each is said to the freight as it closes, and
tasks.jsonl at the end. An item's `at` is the position of its first token
over the token files of its kind in order (a task's: its line in
tasks.jsonl), as `tokens`. Its result: {total_tokens (of plain text),
items_tokens (of conversations and tasks), shards {kind: [{path,
token_sha256, rows, tokens, sources}]}, tasks ({path, count}, None
without one), fingerprints {plain, conversation, task: what a training
step reading that kind reads, the vocabulary and the files}, vocabulary
{sha256, size}}.

wpt-count (wpt_count) gives the same weights and writes nothing: its result is
{total, plain, items}, the tokens of what it took.

Both read the vocabulary from the input `vocabulary`, a folder holding
vocabulary.json, the canonical vocabulary file (wpt_tokenizer), and weigh
an item in tokens as wpt_corpus.encoding.weight does. The items are
encoded on `threads` threads and answered in the order they came, so the
files never depend on the threads.
"""

import hashlib
import json
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wpt_tokenizer import Tokenizer

from . import channel
from .encoding import KINDS, PLAIN, TASK, TOKEN_KINDS, task_text, tokens, weight
from .shards import write_shard

VERSION = "1"  # raised whenever the files or weights the same messages give change
SHARD_TOKENS = 1 << 26  # about 64M tokens (256 MB of uint32) per token file
WAITING = 4  # items read ahead of the one being answered, for each thread
TASKS = "tasks.jsonl"
SETTINGS = {"shard_tokens"}


class Failed(Exception):
    """The program cannot do its part: the reason goes to the freight."""


def wpt_tokens():
    """The program wpt-tokens."""
    sys.exit(_Program("wpt-tokens", files=True).run(sys.stdin.buffer, sys.stdout.buffer))


def wpt_count():
    """The program wpt-count."""
    sys.exit(_Program("wpt-count", files=False).run(sys.stdin.buffer, sys.stdout.buffer))


class _Program:
    def __init__(self, name: str, files: bool):
        self.name, self.files = name, files

    def run(self, stdin, stdout) -> int:
        self.out = stdout
        try:
            first = channel.read(stdin)
            if first is None:
                return 0
            start, _ = first
            if start["type"] != "start":
                raise Failed(f"the first message is {start['type']}, not start")
            self._start(start)
        except (Failed, channel.BadMessage, OSError, ValueError, KeyError) as error:
            self._send({"type": "failed", "reason": f"{type(error).__name__}: {error}"})
            return 1
        self._send({"type": "ready", "name": self.name, "version": VERSION, "unit": "tokens", "kinds": list(KINDS),
                    "weighs": True})
        stdout.flush()
        waiting = queue.Queue(maxsize=WAITING * self.threads)
        reader = threading.Thread(target=self._read, args=(stdin, waiting), daemon=True)
        with ThreadPoolExecutor(self.threads, thread_name_prefix=self.name) as self.pool:
            self.reading_failed = None
            reader.start()
            try:
                while True:
                    found = waiting.get()
                    if found is None:
                        if self.reading_failed is not None:
                            raise Failed(self.reading_failed)
                        return 0  # the freight closed the channel before end: it wants nothing more
                    message, done = found
                    kind = message["type"]
                    if kind == "weigh":
                        self._send({"type": "weight", "ref": message["ref"], "weight": done.result()[0]})
                    elif kind == "take":
                        self._take(message, done.result())
                    elif kind == "end":
                        self._end()
                        stdout.flush()
                        return 0
                    else:
                        raise Failed(f"no message {kind} comes from the freight")
                    if waiting.empty():
                        stdout.flush()
            except Exception as error:  # noqa: BLE001 every failure is told to the freight
                self._send({"type": "failed", "reason": f"{type(error).__name__}: {error}"})
                stdout.flush()
                return 1

    def _send(self, message: dict):
        channel.write(self.out, message)

    def _start(self, start: dict):
        settings = start.get("settings") or {}
        if set(settings) - SETTINGS:
            raise Failed(f"no setting {', '.join(sorted(set(settings) - SETTINGS))}: {self.name} takes "
                         f"{', '.join(sorted(SETTINGS))}")
        self.shard_tokens = settings.get("shard_tokens", SHARD_TOKENS)
        if not isinstance(self.shard_tokens, int) or isinstance(self.shard_tokens, bool) or self.shard_tokens < 1:
            raise Failed(f"shard_tokens is a whole number above 0: {self.shard_tokens!r}")
        folder = (start.get("inputs") or {}).get("vocabulary")
        if folder is None:
            raise Failed(f"{self.name} needs the input vocabulary, a folder holding vocabulary.json")
        self.vocabulary = (Path(folder) / "vocabulary.json").read_bytes()
        self.tokenizer = Tokenizer.from_bytes(self.vocabulary)
        self.threads = max(1, int(start.get("threads") or 1))
        self.folder = Path(start["out"])
        self.folder.mkdir(parents=True, exist_ok=True)
        self.total = {"plain": 0, "items": 0}
        self.writers = {kind: _Shards(self.folder, kind) for kind in TOKEN_KINDS}
        self.shards = {kind: [] for kind in TOKEN_KINDS}
        self.tasks = 0
        resume = start.get("resume")
        if self.files:
            tasks = self.folder / TASKS
            kept = 0
            if resume:
                for one in resume.get("files") or []:
                    if one.get("kind") in self.shards:
                        self.shards[one["kind"]].append({key: one[key] for key in (
                            "path", "token_sha256", "rows", "tokens", "sources")})
                        self.writers[one["kind"]].resumed(one)
                    if "so_far" in one:
                        self.tasks, kept = one["so_far"]["tasks"], one["so_far"]["tasks_bytes"]
                        self.total = {"plain": one["so_far"]["plain"], "items": one["so_far"]["items"]}
            with open(tasks, "ab") as held:
                if held.tell() < kept:
                    raise Failed(f"{TASKS} holds {held.tell()} bytes, fewer than the {kept} the files to resume "
                                 "after say: the folder of the former run is gone")
                held.truncate(kept)
            self.tasks_file = open(tasks, "ab")

    def _read(self, stdin, waiting: queue.Queue):
        """Read the freight's messages and start the work of each on the pool, in order."""
        try:
            while (found := channel.read(stdin)) is not None:
                message, content = found
                if message["type"] in ("weigh", "take"):
                    waiting.put((message, self.pool.submit(self._work, message, content)))
                else:
                    waiting.put((message, None))
                    if message["type"] == "end":
                        return
        except Exception as error:  # noqa: BLE001 the main thread tells the freight
            self.reading_failed = f"{type(error).__name__}: {error}"
        waiting.put(None)

    def _work(self, message: dict, content: bytes):
        """(the item's weight, its tokens or a task's content), worked out on a thread."""
        kind = message["kind"]
        if kind not in KINDS:
            raise Failed(f"{self.name} prepares no item of kind {kind}")
        if kind == TASK:
            return weight(self.tokenizer, kind, content), content
        found = tokens(self.tokenizer, kind, content)
        return len(found), found

    def _take(self, message: dict, done: tuple):
        size, data = done
        kind = message["kind"]
        self.total["plain" if kind == PLAIN else "items"] += size
        at = {}
        if self.files and kind == TASK:
            at["tokens"] = self.tasks
            self.tasks_file.write(task_text({"depot": message["depot"], "id": message["id"],
                                             "content": json.loads(data)}).encode("utf-8") + b"\n")
            self.tasks += 1
        elif self.files:
            at["tokens"] = self.writers[kind].add(data, message["depot"], message["name"], message.get("pool") or "")
        self._send({"type": "taken", "ref": message["ref"], "weight": size, **({"at": at} if self.files else {})})
        if self.files and any(writer.full(self.shard_tokens) for writer in self.writers.values()):
            self._close()

    def _close(self):
        """Close the token file of each kind begun, and say each to the freight."""
        self.tasks_file.flush()
        so_far = {"tasks": self.tasks, "tasks_bytes": self.tasks_file.tell(), **self.total}
        for kind in TOKEN_KINDS:
            done = self.writers[kind].close()
            if done is not None:
                self.shards[kind].append(done)
                self._send({"type": "file", "path": done["path"], "items": done["rows"], "kind": kind,
                            "tokens": done["tokens"], "token_sha256": done["token_sha256"], "rows": done["rows"],
                            "sources": done["sources"], "hash": done["token_sha256"], "so_far": so_far})

    def _end(self):
        if not self.files:
            self._send({"type": "done", "result": {"total": self.total["plain"] + self.total["items"],
                                                   "plain": self.total["plain"], "items": self.total["items"]}})
            return
        self._close()
        self.tasks_file.close()
        tasks = self.folder / TASKS
        tasks_sha256 = hashlib.sha256(tasks.read_bytes()).hexdigest()
        if self.tasks:
            self._send({"type": "file", "path": TASKS, "items": self.tasks, "kind": TASK})
        sha256 = hashlib.sha256(self.vocabulary).hexdigest()
        self._send({"type": "done", "result": {
            "total_tokens": sum(one["tokens"] for one in self.shards[PLAIN]), "items_tokens": self.total["items"],
            "shards": self.shards, "tasks": {"path": TASKS, "count": self.tasks} if self.tasks else None,
            "fingerprints": {**{kind: _fingerprint([sha256, [one["token_sha256"] for one in kept]])
                                for kind, kept in self.shards.items()},
                             TASK: _fingerprint([sha256, tasks_sha256])},
            "vocabulary": {"sha256": sha256, "size": self.tokenizer.vocab_size}}})


class _Shards:
    """Rows of one kind written as local token files, one at a time; the
    token positions go on over the files (position)."""

    def __init__(self, folder: Path, kind: str):
        self.folder, self.kind, self.position, self.number = folder, kind, 0, 0
        self.batch, self.batch_tokens = [], 0

    def resumed(self, file: dict):
        """Go on after a token file a former run wrote."""
        self.position += file["tokens"]
        self.number += 1

    def add(self, tokens, depot: str, text: str, pool: str) -> int:
        """Add one item's row: the position of its first token."""
        start = self.position
        self.batch.append((tokens, depot, str(text), pool, 0))
        self.batch_tokens += len(tokens)
        self.position += len(tokens)
        return start

    def full(self, size: int) -> bool:
        return self.batch_tokens >= size

    def close(self) -> dict | None:
        """The token file of the rows added since the last one, written:
        {path (under the folder), token_sha256, rows, tokens, sources}; None without a row."""
        if not self.batch:
            return None
        path = f"{self.kind}-{self.number:05d}.parquet"
        digest = write_shard(self.folder / path, self.batch)
        done = {"path": path, "token_sha256": digest, "rows": len(self.batch), "tokens": self.batch_tokens,
                "sources": sorted({row[1] for row in self.batch})}
        self.batch, self.batch_tokens = [], 0
        self.number += 1
        return done


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
