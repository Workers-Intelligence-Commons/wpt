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

A delivery cut into portions (start names them, each take its own) gets
the same files for each portion apart, under the folder of its name
(`<portion>/plain-NNNNN.parquet`, `<portion>/tasks.jsonl`): an item's
position counts within its portion, each file message names its portion,
a portion's files close when its last item is taken, and the result lists
them by portion, {portions: {name: {total_tokens, items_tokens, shards,
tasks, fingerprints}}, vocabulary}, every portion named there, one that took
nothing included. A delivery without portions gets the files and result
it always got.

wpt-count (wpt_count) gives the same weights and writes nothing: its result is
{total, plain, items}, the tokens of what it took, every portion together.

Both read the vocabulary from the input `vocabulary`, a folder holding
vocabulary.json, the canonical vocabulary file (wpt_tokenizer), and weigh
an item in tokens as wpt_corpus.encoding.weight does. The items are
encoded on `threads` threads and answered in the order they came, so the
files never depend on the threads.
"""

import hashlib
import json
import queue
import re
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
PORTION = re.compile(r"[A-Za-z0-9_-]{1,64}")  # a portion's name, as the freight names it; its files' folder


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
            first = channel.read(stdin, channel.CONTENT)
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
        names = start.get("portions") or None
        if names is not None and (not isinstance(names, list) or len(set(names)) != len(names)
                                  or not all(isinstance(one, str) and PORTION.fullmatch(one) for one in names)):
            raise Failed(f"the portions are names of 1 to 64 letters, digits, _ or -, each once: {names!r}")
        self.portioned = names is not None
        self.portions = {name: _Portion(self.folder, name) for name in (names or [None])}
        self.current = None
        resume = start.get("resume")
        files = (resume or {}).get("files") or []
        for one in files:
            portion = self._portion(one.get("portion"))
            if one.get("kind") in portion.shards:
                portion.shards[one["kind"]].append({key: one[key] for key in (
                    "path", "token_sha256", "rows", "tokens", "sources")})
                portion.writers[one["kind"]].resumed(one)
            if one.get("kind") == TASK:
                portion.tasks_said = True
            if "so_far" in one:
                portion.tasks, portion.kept = one["so_far"]["tasks"], one["so_far"]["tasks_bytes"]
                portion.total = {"plain": one["so_far"]["plain"], "items": one["so_far"]["items"]}
        if self.files:
            # the portion the last file uploaded belongs to goes on (its tasks kept up to then); the ones before it
            # are complete, and a later one starts with its first item
            self._enter(self._portion(files[-1].get("portion")) if files else next(iter(self.portions.values())))

    def _portion(self, name) -> "_Portion":
        if name not in self.portions:
            raise Failed(f"no portion {name!r}: the portions are {', '.join(map(str, self.portions))}")
        return self.portions[name]

    def _enter(self, portion: "_Portion"):
        """Make portion the one items go into: its tasks file opened, cut to what the files uploaded hold."""
        self.current = portion
        if portion.tasks_said:
            return
        tasks = self.folder / portion.tasks_path
        tasks.parent.mkdir(parents=True, exist_ok=True)
        with open(tasks, "ab") as held:
            if held.tell() < portion.kept:
                raise Failed(f"{portion.tasks_path} holds {held.tell()} bytes, fewer than the {portion.kept} the "
                             "files to resume after say: the folder of the former run is gone")
            held.truncate(portion.kept)
        portion.tasks_file = open(tasks, "ab")

    def _read(self, stdin, waiting: queue.Queue):
        """Read the freight's messages and start the work of each on the pool, in order."""
        try:
            while (found := channel.read(stdin, channel.CONTENT)) is not None:
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
        portion = self._portion(message.get("portion") if self.portioned else None)
        if self.files and portion is not self.current:
            if list(self.portions).index(portion.name) < list(self.portions).index(self.current.name):
                raise Failed(f"an item of the portion {portion.name} came after the portion {self.current.name}")
            self._finish(self.current)
            self._enter(portion)
        portion.total["plain" if kind == PLAIN else "items"] += size
        at = {}
        if self.files and kind == TASK:
            at["tokens"] = portion.tasks
            portion.tasks_file.write(task_text({"depot": message["depot"], "id": message["id"],
                                                "content": json.loads(data)}).encode("utf-8") + b"\n")
            portion.tasks += 1
        elif self.files:
            at["tokens"] = portion.writers[kind].add(data, message["depot"], message["name"],
                                                     message.get("pool") or "")
        self._send({"type": "taken", "ref": message["ref"], "weight": size, **({"at": at} if self.files else {})})
        if self.files and any(writer.full(self.shard_tokens) for writer in portion.writers.values()):
            self._close(portion)

    def _close(self, portion: "_Portion"):
        """Close the token file of each kind begun in portion, and say each to the freight."""
        portion.tasks_file.flush()
        so_far = {"tasks": portion.tasks, "tasks_bytes": portion.tasks_file.tell(), **portion.total}
        for kind in TOKEN_KINDS:
            done = portion.writers[kind].close()
            if done is not None:
                portion.shards[kind].append(done)
                self._send({"type": "file", "path": done["path"], "items": done["rows"], "kind": kind,
                            "tokens": done["tokens"], "token_sha256": done["token_sha256"], "rows": done["rows"],
                            "sources": done["sources"], "hash": done["token_sha256"], "so_far": so_far,
                            **portion.named})

    def _finish(self, portion: "_Portion"):
        """Close every file of portion and say them: its token files, then its tasks file when it holds a task."""
        if portion.tasks_said:
            return
        self._close(portion)
        so_far = {"tasks": portion.tasks, "tasks_bytes": portion.tasks_file.tell(), **portion.total}
        portion.tasks_file.close()
        portion.tasks_said = True
        if portion.tasks:
            self._send({"type": "file", "path": portion.tasks_path, "items": portion.tasks, "kind": TASK,
                        **({"so_far": so_far} if self.portioned else {}), **portion.named})

    def _end(self):
        if not self.files:
            plain = sum(one.total["plain"] for one in self.portions.values())
            items = sum(one.total["items"] for one in self.portions.values())
            self._send({"type": "done", "result": {"total": plain + items, "plain": plain, "items": items}})
            return
        self._finish(self.current)
        sha256 = hashlib.sha256(self.vocabulary).hexdigest()
        vocabulary = {"sha256": sha256, "size": self.tokenizer.vocab_size}
        if not self.portioned:
            self._send({"type": "done", "result": {**self.portions[None].result(sha256), "vocabulary": vocabulary}})
            return
        self._send({"type": "done", "result": {
            "portions": {name: portion.result(sha256) for name, portion in self.portions.items()},
            "vocabulary": vocabulary}})


class _Portion:
    """What wpt-tokens writes of one portion of the delivery (name None: a
    delivery without portions): its token files by kind and its tasks file,
    under the folder of its name."""

    def __init__(self, folder: Path, name: str | None):
        self.folder, self.name = folder, name
        prefix = "" if name is None else f"{name}/"
        self.named = {} if name is None else {"portion": name}  # what its file messages add
        self.tasks_path = f"{prefix}{TASKS}"
        self.writers = {kind: _Shards(folder, kind, prefix) for kind in TOKEN_KINDS}
        self.shards = {kind: [] for kind in TOKEN_KINDS}
        self.total = {"plain": 0, "items": 0}
        self.tasks = self.kept = 0
        self.tasks_file, self.tasks_said = None, False

    def result(self, vocabulary: str) -> dict:
        """{total_tokens, items_tokens, shards, tasks, fingerprints} of the portion, once its files are said."""
        tasks = self.folder / self.tasks_path
        tasks_sha256 = hashlib.sha256(tasks.read_bytes() if tasks.exists() else b"").hexdigest()
        return {"total_tokens": sum(one["tokens"] for one in self.shards[PLAIN]), "items_tokens": self.total["items"],
                "shards": self.shards, "tasks": {"path": self.tasks_path, "count": self.tasks} if self.tasks else None,
                "fingerprints": {**{kind: _fingerprint([vocabulary, [one["token_sha256"] for one in kept]])
                                    for kind, kept in self.shards.items()},
                                 TASK: _fingerprint([vocabulary, tasks_sha256])}}


class _Shards:
    """Rows of one kind written as local token files, one at a time; the
    token positions go on over the files (position)."""

    def __init__(self, folder: Path, kind: str, prefix: str = ""):
        self.folder, self.kind, self.prefix, self.position, self.number = folder, kind, prefix, 0, 0
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
        path = f"{self.prefix}{self.kind}-{self.number:05d}.parquet"
        (self.folder / path).parent.mkdir(parents=True, exist_ok=True)
        digest = write_shard(self.folder / path, self.batch)
        done = {"path": path, "token_sha256": digest, "rows": len(self.batch), "tokens": self.batch_tokens,
                "sources": sorted({row[1] for row in self.batch})}
        self.batch, self.batch_tokens = [], 0
        self.number += 1
        return done


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
