"""The audit hook that keeps files under data/ to the service layers.

Only a service layer opens files under the repository's data folder
(docs/architecture.md, "The service layer"): wpt_recipes/recipes.py, for
the recipes and the task images. Each service layer calls guard() with its data folder and its own
file before it hands out its first object. From then on, an audit hook
(sys.addaudithook), set once per process, refuses with DataAccessRefused
any opening, listing, removal or renaming of a path under a guarded
folder, symbolic links resolved, whose nearest caller in the project's
own code is neither a guarded service layer nor a test (under a tests/
folder of the project, or a conftest.py). The static half of the rule is
the scan of packages/wpt-recipes/tests/test_data_access.py.
"""

import os
import shutil
import sys
import sysconfig
import threading
from pathlib import Path


class DataAccessRefused(RuntimeError):
    """Code outside the service layers opened, listed, removed or renamed a
    path under the repository's data folder; the audit hook refuses it
    before it happens. A fault of the code, never a refusal of the data."""


# the guarded data folders and the service layers' files, as written and with links resolved; replaced, never
# changed in place, so the hook reads a whole tuple while another thread adds one
_GUARDED: tuple[str, ...] = ()
_SERVICES: tuple[str, ...] = ()
_GUARDING = threading.Lock()
_HOOKED = False
# the project's checkout: packages/wpt-recipes/src/wpt_recipes/guard.py lies four folders below it
_PROJECT = os.path.normcase(os.path.abspath(Path(__file__).parents[4]))
# which arguments of each audited event are paths it reads or changes (os.symlink's first is the link's text)
_EVENTS = {"open": (0,), "os.listdir": (0,), "os.scandir": (0,), "os.remove": (0,), "os.rename": (0, 1),
           "os.mkdir": (0,), "os.rmdir": (0,), "os.chmod": (0,), "os.symlink": (1,), "os.link": (0, 1),
           "os.truncate": (0,), "os.utime": (0,), "shutil.rmtree": (0,), "shutil.copyfile": (0, 1),
           "shutil.copytree": (0, 1), "shutil.move": (0, 1), "glob.glob": (0,), "glob.glob/2": (0, 2)}
_LIBRARIES = tuple(sorted({os.path.normcase(os.path.abspath(path)) + os.sep for name, path in sysconfig.get_paths().items()
                           if name in ("stdlib", "platstdlib", "purelib", "platlib", "scripts")}))
_RMTREE = shutil.rmtree.__code__


def guard(data: Path, service: str):
    """Guard data (a service layer's data folder) with the audit hook, and
    let service (the file of that service layer) reach it."""
    global _GUARDED, _SERVICES, _HOOKED
    folders = {os.path.normcase(os.path.abspath(data)), os.path.normcase(os.path.realpath(data))}
    with _GUARDING:
        _GUARDED = tuple(dict.fromkeys((*_GUARDED, *sorted(folders))))
        _SERVICES = tuple(dict.fromkeys((*_SERVICES, os.path.normcase(os.path.abspath(service)))))
        if not _HOOKED:
            sys.addaudithook(_audit)
            _HOOKED = True


def _audit(event, args):
    paths = _EVENTS.get(event)
    if paths is None:
        return
    for index in paths:
        if index < len(args) and _in_data(args[index]) and not (
                event != "shutil.rmtree" and _inside_rmtree(args[index])):
            caller = _project_caller()
            if caller is not None and caller not in _SERVICES and not _is_test(caller):
                raise DataAccessRefused(f"{caller} reaches {os.fsdecode(args[index])} ({event}): only the service "
                                        f"layers open files under data/ (docs/architecture.md, \"The service "
                                        f"layer\")")


def _in_data(path) -> bool:
    """Whether path lies under a guarded data folder, as written or with its
    symbolic links resolved."""
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False  # a file descriptor
    guarded = _GUARDED
    try:
        written = os.path.abspath(os.fsdecode(path))
        if _under(os.path.normcase(written), guarded):
            return True
        return _under(os.path.normcase(os.path.realpath(written)), guarded)
    except (TypeError, ValueError, OSError):
        return False


def _inside_rmtree(path) -> bool:
    """Whether path is relative and named inside shutil.rmtree, which opens
    and removes each child by its name relative to its parent folder's
    descriptor (dir_fd), not to the current folder; rmtree's own event
    already checked the folder it removes."""
    if os.path.isabs(os.fsdecode(path)):
        return False
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code is _RMTREE:
            return True
        frame = frame.f_back
    return False


def _under(found: str, folders) -> bool:
    return any(found == folder or found.startswith(folder + os.sep) for folder in folders)


def _project_caller() -> str | None:
    """The file of the nearest caller that is the project's own code (not
    Python's library, not an installed package); None when there is none."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if not name.startswith("<"):
            name = os.path.normcase(os.path.abspath(name))
            if not name.startswith(_LIBRARIES):
                return name
        frame = frame.f_back
    return None


def _is_test(name: str) -> bool:
    """Whether a file is test code: a conftest.py, or a file under a tests
    folder, inside the project's checkout (a folder named tests above the
    checkout, or a conftest.py outside it, does not count)."""
    if not name.startswith(_PROJECT + os.sep):
        return False
    if os.path.basename(name) == "conftest.py":
        return True
    return "tests" in Path(name[len(_PROJECT) + 1:]).parts
