"""What every process of a gpu worker's job starts with: `python -m
wpt_worker.job`, run by the worker's server from the manager's checkout.

The process connects to the server (link.py, through its channel to the
manager), sends what it prints to the manager, which prints it (and, when
the worker is not the manager's local one, prints it in its own output too,
which is the console of a rented machine's container), and reads
the job's spec (JSON, the file WPT_WORKER_SPEC names); spec["task"] names
what it runs (TASKS). The worker has no tokenizer: wpt_tokenizer cannot be
imported here (NoTokenizer), so nothing on the worker turns text into
tokens or back by mistake; it deals only in the token ids the manager sends.

    LinkDock      the manager's dock as the job reads it: files the manager allows, fetched
                  part by part into the job's folder and checked against their sha256, and the
                  run's record, each write and line given to the manager's dock before the call
                  returns, and sent on from there when flush_records() returns (an S3 dock holds
                  lines back until then, as on the manager)
    download()    a file the manager offers, into a local file, checked against its sha256
    upload()      a local file to the manager, part by part: its upload id
"""

import hashlib
import importlib.abc
import io
import json
import os
import socket
import sys
from pathlib import Path

from .link import PART, Connection, LinkError, RemoteError, receive, send

TOKENIZER = "wpt_tokenizer"
ECHO = "WPT_WORKER_ECHO"  # 1: the job also writes its lines to the worker's own output (the server sets it)
TASKS = {"wpt-train": "wpt_worker.training", "wpt-eval": "wpt_worker.evaluation"}  # spec["task"]: the module whose run(spec, link, work) is the job


class NoTokenizer(importlib.abc.MetaPathFinder):
    """Refuses to import wpt_tokenizer."""

    def find_spec(self, name, path, target=None):
        if name == TOKENIZER or name.startswith(TOKENIZER + "."):
            raise ModuleNotFoundError(f"No module named {name!r}: the gpu worker has no tokenizer; it deals only in "
                                      "token ids, which the wpt manager sends", name=name)
        return None


class Relay(io.TextIOBase):
    """A text stream whose writes the manager prints (its standard output
    or error output). With echo, they also go to this process's own stream
    (fallback), so the console of the worker's container shows them; a
    failure to write there is ignored. Without echo, they go to it only
    when the link is gone."""

    def __init__(self, link, stream, fallback, echo=False):
        self.link, self.stream, self.fallback, self.echo = link, stream, fallback, echo

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def write(self, text):
        if text:
            if self.echo:
                try:
                    self.fallback.write(text)
                    self.fallback.flush()
                except Exception:
                    pass
            try:
                self.link.tell("print", stream=self.stream, text=text)
            except LinkError:
                if not self.echo:
                    self.fallback.write(text)
        return len(text)

    def reconfigure(self, **_):
        pass


def connect_socket(environ, hello) -> socket.socket:
    """A socket to the worker's server, on this machine, greeted with hello
    (its "kind" and more) and the job's key, and welcomed. LinkError when
    the server refuses it."""
    host, port = environ["WPT_WORKER_JOB"].rsplit(":", 1)
    sock = socket.create_connection((host, int(port)))
    send(sock, {**hello, "key": environ["WPT_WORKER_JOB_KEY"]})
    header, _ = receive(sock)
    if header.get("kind") != "welcome":
        sock.close()
        raise LinkError(f"the gpu worker's server refused this process: {header.get('message', '')}")
    return sock


def connect(environ=None) -> Connection:
    """This process's link to the manager, through the worker's server."""
    environ = os.environ if environ is None else environ
    return Connection(connect_socket(environ, {"kind": "hello"}), name="job")


def remote(error: RemoteError):
    """The exception of this side that a RemoteError stands for: the
    dock's own errors and RecordError as themselves (so a dock call fails
    here as it would on the manager), the rest as the RemoteError."""
    from dock.api import CorruptFile, MissingFile
    from wpt_telemetry import RecordError

    known = {"MissingFile": MissingFile, "CorruptFile": CorruptFile, "RecordError": RecordError}
    found = known.get(error.kind)
    return found(error.message) if found is not None else error


def call(link, kind, body=b"", **fields):
    """link.call, a RemoteError raised as remote() says."""
    try:
        return link.call(kind, body, **fields)
    except RemoteError as error:
        raise remote(error) from None


def download(link, offer, target: Path) -> Path:
    """The file the manager offers ({"file": its id, "size", "sha256"}) into
    target, part by part, checked against its sha256 before it takes
    target's name. CorruptFile when it differs."""
    from dock.api import CorruptFile

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    try:
        with open(partial, "wb") as out:
            for start in range(0, offer["size"], PART):
                _, data = call(link, "file.range", file=offer["file"], start=start,
                               length=min(PART, offer["size"] - start))
                digest.update(data)
                out.write(data)
        if digest.hexdigest() != offer["sha256"]:
            raise CorruptFile(f"{offer['file']} arrived as {digest.hexdigest()}, not {offer['sha256']}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def upload(link, path) -> str:
    """Send a local file to the manager, part by part; the id the manager
    gave it, which a later request names."""
    header, _ = call(link, "upload.start")
    upload_id = header["upload"]
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while data := file.read(PART):
            digest.update(data)
            call(link, "upload.part", data, upload=upload_id)
    call(link, "upload.end", upload=upload_id, sha256=digest.hexdigest())
    return upload_id


def _sha(what) -> str:
    """The sha256 of what names a file: a record's reference to one
    ({"key", "sha256"}), or the sha256."""
    return what["sha256"] if isinstance(what, dict) else what


class LinkDock:
    """The manager's dock, as a job's process reads it: a reader of the
    corpus's files (wpt_corpus.files), which wpt_corpus.shards and the
    environments take, and the run's records, which wpt_telemetry writes
    to, as wpt_telemetry.DockRecords answers on the manager. Files by
    their sha256, which only the manager holds and gives part by part; the
    record folder of the run, whose every write goes to the manager's dock
    before the call returns, and whose lines that dock holds back are sent
    when flush_records() returns. Copies of files go into folder/files
    (path(), until release()), files derived from them into folder/scratch.
    The manager answers only for the files of the run's corpus and
    states, and only for its record folder."""

    def __init__(self, link, folder):
        self.link = link
        self.cache, self.scratch = Path(folder) / "files", Path(folder) / "scratch"

    def __repr__(self):
        return "the wpt manager's dock"

    def path(self, what) -> Path:
        sha = _sha(what)
        local = self.cache / sha[:2] / sha
        if not local.is_file():
            header, _ = call(self.link, "file.offer", sha=sha)
            download(self.link, header, local)
        return local

    def get_bytes(self, what) -> bytes:
        return self.path(what).read_bytes()

    def release(self, what):
        sha = _sha(what)
        (self.cache / sha[:2] / sha).unlink(missing_ok=True)

    def read_record(self, folder, name) -> bytes:
        return call(self.link, "record.read", folder=folder, name=name)[1]

    def write_record(self, folder, name, data):
        call(self.link, "record.write", data, folder=folder, name=name)

    def append_record(self, folder, name, data):
        call(self.link, "record.append", data, folder=folder, name=name)

    def flush_records(self):
        call(self.link, "record.flush")


def main():
    sys.meta_path.insert(0, NoTokenizer())
    link = connect()
    echo = os.environ.get(ECHO) == "1"
    sys.stdout = Relay(link, "out", sys.__stdout__, echo)
    sys.stderr = Relay(link, "err", sys.__stderr__, echo)
    spec = json.loads(Path(os.environ["WPT_WORKER_SPEC"]).read_text(encoding="utf-8"))
    module = __import__(TASKS[spec["task"]], fromlist=["run"])
    code = module.run(spec, link, Path(os.environ["WPT_WORKER_WORK"]))
    sys.stdout.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
