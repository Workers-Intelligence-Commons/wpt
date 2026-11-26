"""What code outside the dock asks of it (docs/api/dock.md). Every call
names the dock by its URL, as DOCK gives it, and takes and gives only
values JSON can carry and bytes, alone or as a stream of pieces, so each
can become a route without its callers changing. No call hands out a dock
object or a path into the dock's own files: a caller that needs a file on
its disk names a path of its own (fetch_into, put_from).

    url(env_file, environ)                  the dock's URL that DOCK names
    ready(dock)                             nothing when the dock answers, else why not
    find(dock, prefix), folders(dock, prefix)
                                            the files under a prefix: [{key, bytes}]; the folders right under it
    has(dock, key), size(dock, key)         whether a file is there; its length
    fetch(dock, key, sha256)                a file's bytes, checked against its sha256 when it is given
    fetch_range(dock, key, start, length)   bytes of a file from start
    stream(dock, key, start, length)        a file's bytes, or a range of them, as a stream of pieces
    fetch_into(dock, key, path, sha256)     a file written into a local path of the caller's: {key, sha256, bytes}
    check(dock, key, sha256)                whether a file holds the bytes of that sha256
    put(dock, key, data), put_from(dock, key, path)
                                            bytes, a stream or a local file stored under key: {key, sha256, bytes}
    put_content(dock, data), put_content_from(dock, path), content_key(sha256)
                                            a file named by its sha256, under cache/files/
    delete(dock, key)                       a file or a record removed
    put_json(dock, key, record), get_json(dock, key)
                                            a record that holds the sha256 of its fields
    put_result, get_result                  a result kept under the fingerprint of what made it, and what made it
    write_record, append_record, flush_records, end_records, read_record, read_new, record_files,
    record_folders                          the run records, runs/<run>/
    NAME, check_name, check_record_name, as_record_name
                                            the names a dock allows
    DockError, DockSettingError, NoDockFolder, MissingFile, CorruptFile, MixedRecord
                                            what a dock refuses

The dock of a URL is opened once in each process, on its first call, and
kept: what it learned of its files (the names they are stored under, the
seek tables and frames of compressed ones, the reads requested ahead) and
the record lines it holds back serve every later call of the process. A
process started by fork opens it again.
"""

import hashlib
import os
import tempfile
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import urlparse

from wic_utils.env import read_env_file

from . import CACHE, FILES, RUNS, SHA256
from . import Dock as _Dock
from . import LocalDock as _LocalDock
from . import RecordReader as _RecordReader
from . import S3Dock as _S3Dock
from . import local_folder as _local_folder
from . import s3_settings as _s3_settings
from .errors import CorruptFile, DockError, DockSettingError, MissingFile, MixedRecord, NoDockFolder
from .names import NAME, as_record_name, check_name, check_record_name
from .names import File as _File

SETTING = "DOCK"
CHUNK = 1 << 20
__all__ = ["NAME", "SETTING", "CorruptFile", "DockError", "DockSettingError", "MissingFile", "MixedRecord",
           "NoDockFolder", "append_record", "as_record_name", "check", "check_name", "check_record_name", "content_key", "delete",
           "end_records", "fetch", "fetch_into", "fetch_range", "find", "flush_records", "folders", "get_json",
           "get_result", "has", "put", "put_content", "put_content_from", "put_from", "put_json", "put_result",
           "read_new", "ready", "read_record", "record_files", "record_folders", "size", "stream", "url",
           "write_record"]

_OPENED = {}  # a dock's URL: the dock opened for it, once in each process
_KEYS = {}  # an S3 dock's URL: its keys and endpoint (s3_settings) as url found them
_LOCK = threading.Lock()


def _forget_opened():
    _OPENED.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_opened)  # a forked process opens its docks again, threads and all


def url(env_file=".env", environ=None) -> str:
    """The dock's URL DOCK gives, in environ (default: the process's
    environment), else in env_file. DockSettingError when neither sets it,
    when it is neither a local file:// URL nor an s3:// one with a bucket,
    and for an S3 dock whose keys neither sets. The calls that follow open
    an S3 dock of that URL with the keys found there; one this process
    opened already keeps the keys it was opened with. Without a call of
    url, an S3 dock reads its keys from the process's environment, else
    from .env. Nothing is asked of the dock."""
    environ = os.environ if environ is None else environ
    found = environ.get(SETTING) or read_env_file(env_file).get(SETTING)
    if not found:
        raise DockSettingError(f"{SETTING} is not set: put {SETTING}=file:///path or s3://bucket/prefix in .env")
    if _local_folder(found) is None:
        parsed = urlparse(found)
        if parsed.scheme != "s3":
            raise DockSettingError(f"DOCK must start with file:// or s3://, got {found!r}")
        if not parsed.netloc:
            raise DockSettingError(f"s3 URLs need a bucket: {found!r}")
        _KEYS[found] = _s3_settings(environ, env_file)
    return found


def _open(dock: str) -> _Dock:
    folder = _local_folder(dock)
    if folder is not None:
        return _LocalDock(folder)
    parsed = urlparse(dock)
    if parsed.scheme != "s3":
        raise DockSettingError(f"DOCK must start with file:// or s3://, got {dock!r}")
    if not parsed.netloc:
        raise DockSettingError(f"s3 URLs need a bucket: {dock!r}")
    keys = _KEYS.get(dock)
    return _S3Dock(parsed.netloc, parsed.path, settings=keys if keys is not None else _s3_settings(None, ".env"))


def _dock(dock: str) -> _Dock:
    if not isinstance(dock, str):
        raise TypeError(f"a dock is named by its URL, not {type(dock).__name__}")
    with _LOCK:
        if dock not in _OPENED:
            opened = _open(dock)
            opened.copies_only = True  # every read goes to the dock: no copy is kept in its cache folder
            _OPENED[dock] = opened
        return _OPENED[dock]


def _file(key: str, sha256: str | None):
    return key if sha256 is None else _File(key, sha256)


def _stored(key, size, packed=".zst") -> tuple[str, int] | None:
    if size is None or key.rsplit("/", 1)[-1].startswith("."):
        return None
    return key.removesuffix(packed), size


def ready(dock: str):
    """Nothing when the dock answers: a local dock's folder exists and is
    listed, an S3 dock's bucket answers a listing. NoDockFolder for a local
    folder that does not exist (none is made), DockSettingError for a bad
    URL or missing keys, whatever the folder or the bucket raised
    otherwise."""
    folder = _local_folder(dock)
    if folder is not None and not folder.is_dir():
        raise NoDockFolder(f"{folder} does not exist")
    _dock(dock).kind.list(f"{RUNS}/", deep=False)


# Finding

def find(dock: str, prefix: str) -> list[dict]:
    """Every file of the dock whose key starts with prefix, in the order of
    their keys: {key, bytes}, bytes the size it takes on the dock (a file
    kept compressed counts its compressed size). A key is given without the
    `.zst` the dock adds to a file it keeps compressed."""
    found = {}
    for key, size in _dock(dock).kind.list(prefix).items():
        if (one := _stored(key, size)) is not None:
            found[one[0]] = one[1]
    return [{"key": key, "bytes": size} for key, size in sorted(found.items())]


def folders(dock: str, prefix: str) -> list[str]:
    """The names of the folders right under prefix, a key ending in '/'
    (corpora/, models/), sorted: each folder once, without listing the
    files below it."""
    return sorted(key[len(prefix):].removesuffix("/") for key in _dock(dock).kind.list(prefix, deep=False)
                  if key.endswith("/"))


def has(dock: str, key: str) -> bool:
    """Whether the dock holds a file under key now (asked each time, so a
    file another process deleted is not taken as there)."""
    return _dock(dock)._stored(key, fresh=True) is not None


def size(dock: str, key: str) -> int:
    """The file's length in bytes, without reading it. MissingFile when
    there is no such file."""
    held = _dock(dock)
    held.refresh(key)
    if held._stored(key, fresh=True) is None:
        raise MissingFile(f"{key} is not on the dock")
    return held.size(_File(key, "0" * 64))


# Fetching

def fetch(dock: str, key: str, sha256: str | None = None) -> bytes:
    """The bytes of the file under key, checked against sha256 when it is
    given (a record's reference to a file gives it). MissingFile when the
    dock holds no such file, CorruptFile when it does not match."""
    held = _dock(dock)
    held.refresh(key)
    if sha256 is not None:
        return held.get_bytes(_File(key, sha256))
    data = held.whole(key)
    if data is None:
        raise MissingFile(f"{key} is not on the dock")
    return data


def fetch_range(dock: str, key: str, start: int, length: int) -> bytes:
    """length bytes of the file under key from start, without fetching the
    rest of it: a compressed file's frames are read a few at a time and
    kept a while, and on S3 the frames after a read going through the file
    are requested ahead; a file kept as it is is read in place, one request
    per call. What the process learned of the file serves the
    reads that follow: a file another process replaced since is read anew
    after size, fetch, stream or fetch_into of its key. A range cannot be
    checked against the file's sha256: the caller checks it against a hash
    of its own. MissingFile, or CorruptFile when the file ends first."""
    return _dock(dock).read_range(_File(key, "0" * 64), start, length)


def stream(dock: str, key: str, start: int = 0, length: int | None = None) -> Iterator[bytes]:
    """The bytes of the file under key from start, length of them (None: up
    to its end), as pieces of at most 8 MiB, read as fetch_range reads.
    MissingFile when there is no such file."""
    held = _dock(dock)
    held.refresh(key)
    return held.stream(key, start, length)


def fetch_into(dock: str, key: str, path, sha256: str | None = None) -> dict:
    """The file under key written into path, a local file of the caller's,
    replaced whole (a file there before stays until the new one takes its
    place) and checked against sha256 when it is given: {key, sha256,
    bytes}. The file is the caller's own copy, which it may change or
    remove. MissingFile, CorruptFile (nothing is left at path)."""
    held = _dock(dock)
    held.refresh(key)
    found = held.copy_to(_file(key, sha256), Path(path))
    return {"key": key, "sha256": found, "bytes": Path(path).stat().st_size}


def check(dock: str, key: str, sha256: str) -> bool:
    """Whether the file under key holds the bytes of that sha256, read
    through once. MissingFile when there is no such file."""
    digest = hashlib.sha256()
    for piece in stream(dock, key):
        digest.update(piece)
    return digest.hexdigest() == sha256


# Putting

def put(dock: str, key: str, data: bytes | Iterable[bytes]) -> dict:
    """data, bytes or a stream of pieces of bytes, stored under key,
    replacing any earlier file: {key, sha256, bytes}, the reference that
    fetches it back checked. A stream is gathered in the dock's own folder
    first."""
    held = _dock(dock)
    if isinstance(data, (bytes, bytearray, memoryview)):
        stored = held.put_named(key, bytes(data))
        return {**stored.reference(), "bytes": len(data)}
    held.cache.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=held.cache, prefix=".gathered-")
    try:
        with os.fdopen(fd, "wb") as out:
            for piece in data:
                out.write(piece)
        return put_from(dock, key, name)
    finally:
        Path(name).unlink(missing_ok=True)


def put_from(dock: str, key: str, path) -> dict:
    """The local file at path (the caller's, left as it is) stored under
    key, replacing any earlier file: {key, sha256, bytes}."""
    stored = _dock(dock).put_named(key, Path(path))
    return {**stored.reference(), "bytes": Path(path).stat().st_size}


def content_key(sha256: str) -> str:
    """The key of the file put_content keeps for bytes of that sha256:
    cache/files/<sha256>."""
    return _content_key(sha256)


def _content_key(sha256: str) -> str:
    if not SHA256.fullmatch(sha256):
        raise ValueError(f"not a sha256: {sha256!r}")
    return f"{CACHE}/{FILES}/{sha256}"


def put_content(dock: str, data: bytes) -> dict:
    """data kept under the key its sha256 names (content_key), sent only
    when the dock does not hold it yet: {key, sha256, bytes}. For a result
    whose content is all that names it."""
    sha = _dock(dock).put_bytes(bytes(data))
    return {"key": _content_key(sha), "sha256": sha, "bytes": len(data)}


def put_content_from(dock: str, path) -> dict:
    """The local file at path kept as put_content keeps bytes: {key,
    sha256, bytes}."""
    sha = _dock(dock).put_file(Path(path))
    return {"key": _content_key(sha), "sha256": sha, "bytes": Path(path).stat().st_size}


def delete(dock: str, key: str):
    """Remove the file or the record under key; nothing when it is absent.
    Only for one that nothing else names."""
    held = _dock(dock)
    held.delete_json(key)


# Records

def put_json(dock: str, key: str, record: dict) -> dict:
    """Store a record, a JSON object, under key, replacing any earlier
    one, with the sha256 of its fields in `check`: the record as stored. It
    is on the disk, or held by S3, when this returns."""
    return _dock(dock).put_json(key, record)


def get_json(dock: str, key: str) -> dict | None:
    """The record under key, checked against its `check`, None without one.
    CorruptFile when it does not match."""
    return _dock(dock).get_json(key)


# Results

def put_result(dock: str, stage: str, fingerprint: str, data: bytes, made_by: str | None = None) -> str:
    """Keep a result as cache/<stage>/<fingerprint>, replacing any earlier
    one; made_by, one word, says what made it. Its sha256."""
    return _dock(dock).put_result(stage, fingerprint, data, made_by)


def get_result(dock: str, stage: str, fingerprint: str) -> tuple[bytes, str | None] | None:
    """(a result's bytes, checked against their sha256; what made it, as
    put_result got it, None when nothing was said), read once; None when
    there is no result."""
    data, made_by = _dock(dock).get_result(stage, fingerprint)
    return None if data is None else (data, made_by)


# Run records

def write_record(dock: str, run: str, name: str, data: bytes):
    """Write the file name of the run's record whole, replacing any earlier
    one; on the dock when it returns. MixedRecord for a file added to line
    by line."""
    _dock(dock).write_record(run, name, bytes(data))


def append_record(dock: str, run: str, name: str, data: bytes):
    """Add lines at the end of a file of the run's record, held back until
    flush_records. Several processes may add to one file. MixedRecord for a
    file written whole."""
    _dock(dock).append_record(run, name, bytes(data))


def flush_records(dock: str):
    """Send the lines append_record holds back in this process: on the
    dock when it returns. The lines not sent are kept, to send again."""
    _dock(dock).flush_records()


def end_records(dock: str, run: str):
    """The run's record ended: what a local dock kept as plain files while
    lines were added is compressed into parts."""
    _dock(dock).end_records(run)


def read_record(dock: str, run: str, name: str) -> bytes:
    """A file of the run's record, every line added in the order it was
    added. MissingFile when there is none."""
    return _dock(dock).read_record(run, name)


def read_new(dock: str, run: str, name: str, cursor: dict | None = None) -> tuple[bytes, dict]:
    """The bytes added to a file of the run's record since cursor (None:
    since it began), and the cursor to give next time, a JSON value: what
    a reader of a run still going reads as it grows. A read that raises
    moves nothing."""
    reader = _RecordReader(_dock(dock), run, name)
    if cursor is not None:
        reader.seen, reader.plain = set(cursor["parts"]), cursor["plain"]
    data = reader.read()
    return data, {"parts": sorted(reader.seen), "plain": reader.plain}


def record_files(dock: str, run: str) -> list[str]:
    """The names of the run's record files, sorted; [] for no such run."""
    return _dock(dock).record_files(run)


def record_folders(dock: str) -> list[str]:
    """Every run's name, oldest first."""
    return _dock(dock).record_folders()
