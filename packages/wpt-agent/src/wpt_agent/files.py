"""Files as a task and its containers hold them: {path: (mode, content)},
the mode one of Git's (FILE, EXECUTABLE, SYMLINK; a link's content is its
target), and the glob patterns a task's area and answered_in name.

    glob_match_any(path, patterns)   whether path matches one of patterns: `*` within one
                                     folder, `**` any number of folders, hidden files included
    read_folder(folder)              every file and link under a local folder, as a task holds them
"""

import functools
import glob
import os
import re
import stat
from pathlib import Path, PurePosixPath

FILE, EXECUTABLE, SYMLINK = "100644", "100755", "120000"
WILDCARD = re.compile(r"[*?[]")  # what fnmatch reads as more than its own letter


def _normal(path: str) -> str:
    """A path or pattern as PurePosixPath.full_match sees it, with "" for an
    empty path rather than "."."""
    pure = PurePosixPath(path)
    return str(pure) if pure.parts else ""


class Globs:
    """A list of patterns compiled once, where `match` gives what
    PurePosixPath.full_match gives for one of them. A pattern with no `*`,
    `?` or `[` matches only its own spelling, so those are a set; the others
    become one regular expression, each the very one full_match compiles
    (glob.translate, recursive, hidden files included, "/" as separator,
    letter case kept)."""

    def __init__(self, patterns):
        self.exact, wild = set(), []
        for pattern in map(_normal, patterns):
            if WILDCARD.search(pattern):
                wild.append(pattern)
            else:
                self.exact.add(pattern)
        translated = (glob.translate(p, recursive=True, include_hidden=True, seps="/") for p in dict.fromkeys(wild))
        self.wild = re.compile("|".join(translated)).match if wild else None

    def match(self, path: str) -> bool:
        path = _normal(path)
        return path in self.exact or (self.wild is not None and self.wild(path) is not None)


@functools.cache  # each task's list of patterns, compiled on its first use
def _globs(patterns: tuple[str, ...]) -> Globs:
    return Globs(patterns)


def glob_match_any(path: str, patterns) -> bool:
    """Whether path matches any of patterns (Globs)."""
    return _globs(tuple(patterns)).match(path)


def read_folder(folder) -> dict[str, tuple[str, bytes]]:
    """{path under folder, with /: (mode, content)} of every file and
    symbolic link under folder, in path order; a file is EXECUTABLE when
    its owner may run it. Folders are left out: an empty one is not kept."""
    folder, found = Path(folder), {}
    for root, folders, names in os.walk(folder):
        folders.sort()
        for name in sorted(names + [one for one in folders if (Path(root) / one).is_symlink()]):
            path = Path(root) / name
            relative = path.relative_to(folder).as_posix()
            if path.is_symlink():
                found[relative] = (SYMLINK, os.readlink(path).encode())
            else:
                executable = path.stat().st_mode & stat.S_IXUSR
                found[relative] = (EXECUTABLE if executable else FILE, path.read_bytes())
    return dict(sorted(found.items()))
