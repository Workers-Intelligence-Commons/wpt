"""Model folders on the dock: models/<name>/ holds the model's real
files, each under its own path, and FILES, {"files": {path inside the
folder: {"sha256", "size"}}}, which every read checks against.

    put_model(dock, name, folder)     every file of folder onto the dock; what names them, for
                                      the caller to keep (recorded first)
    keep_model(dock, name, files)     the model named: FILES written, last
    has_model(dock, name)             whether the dock holds it
    model_file(dock, name, path)      one file's bytes, None when it holds no such file
    copy_out(dock, name, target)      the folder into target (a new folder), each file checked

dock is the dock's URL, read and written through dock.api.

A named run's checkpoint lies beside them until the model is kept: CHECKPOINT,
a record that names the state's file, which lies in the same folder
(wpt.train.DockCheckpoint).

    checkpoint_key(name)              the key of that record
    state_key(name, sha256)           the key a checkpoint state of that sha256 is stored under
"""

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from dock import api
from dock.api import CorruptFile, MissingFile, check_name

MODELS = "models"  # the dock's folder of models
FILES = "files.json"  # the file of models/<name>/ that names the others (a model folder holds no file of that name)
CHECKPOINT = "checkpoint.json"  # the file of models/<name>/ that names a run's checkpoint state


def _prefix(name) -> str:
    check_name(name)
    return f"{MODELS}/{name}/"


def checkpoint_key(name) -> str:
    return _prefix(name) + CHECKPOINT


def state_key(name, sha256) -> str:
    return f"{_prefix(name)}state-{sha256}"


def put_model(dock, name, folder) -> dict:
    """Every file of folder under the model name: {path: {"sha256",
    "size"}}, for keep_model. The model is not there until then."""
    folder, prefix, files = Path(folder), _prefix(name), {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder).as_posix()
        files[relative] = {"sha256": api.put_from(dock, prefix + relative, path)["sha256"], "size": path.stat().st_size}
    return files


def keep_model(dock, name, files: dict):
    """The model name is files (put_model): FILES written."""
    api.put_json(dock, _prefix(name) + FILES, {"name": name, "time": datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"), "files": files})


def _record(dock, name) -> tuple[str, dict] | None:
    """(the key prefix of the model name, its FILES record); None when the
    dock does not hold it."""
    prefix = _prefix(name)
    record = api.get_json(dock, prefix + FILES)
    return None if record is None else (prefix, record)


def names(dock) -> list[str]:
    """The names of the models the dock keeps, sorted."""
    return [name for name in api.folders(dock, f"{MODELS}/") if _record(dock, name) is not None]


def has_model(dock, name) -> bool:
    return _record(dock, name) is not None


def model_file(dock, name, path) -> bytes | None:
    """One file of the model name, checked against its sha256; None when
    the dock holds no such model or file."""
    found = _record(dock, name)
    if found is None or path not in found[1]["files"]:
        return None
    return api.fetch(dock, found[0] + path, found[1]["files"][path]["sha256"])


def copy_out(dock, name, target) -> int:
    """The model folder kept under name, written into target, which must
    not exist yet: beside it first, then renamed, so a failure leaves no
    partial target. The number of its files. MissingFile when no model
    has that name; CorruptFile when a file does not match its sha256."""
    target = Path(target)
    found = _record(dock, name)
    if found is None:
        raise MissingFile(f"no model {name} in {dock}")
    if target.exists():
        raise FileExistsError(f"{target} exists: copy the model into a new folder")
    prefix, record = found
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for path, entry in sorted(record["files"].items()):
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise CorruptFile(f"model {name}: {path!r} is not a path inside its folder")
            local = staging / path
            api.fetch_into(dock, prefix + path, local, entry["sha256"])  # checked against its sha256
            if local.stat().st_size != entry["size"]:
                raise CorruptFile(f"model {name}: {path} has {local.stat().st_size} bytes, not {entry['size']}")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return len(record["files"])
