"""Run by every Python that finds this folder on PYTHONPATH (ci/tests.py puts
it there), before any of its code. It acts only in the project's own Python,
named by WPT_CI_PYTHON: a Python program a test starts, such as Mercurial,
stays as it is.

With WPT_CPU, Triton cannot be imported, as in CI, where it is not installed.

With WPT_CHILD_FILES, each source file of ours this process loads is written,
the moment it is loaded, into that folder (child_files.py reads it) under this
process's id, since a program a test started may end any way. A file's path
counts from packages/ or apps/, which also maps the copied checkout a gpu
worker runs from back to the repository's paths. Every load of a source file
goes through the loader's get_code: an import, `python -m`, a file loaded by
its path.
"""

import os
import sys


def _block_triton():
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name == "triton" or name.startswith("triton."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    sys.meta_path.insert(0, Block())


def _write_loaded_files(folder):
    import re
    from importlib.machinery import SourceFileLoader

    ours = re.compile(r"((?:packages|apps)/[^/]+/src/.+\.py)$")
    get_code = SourceFileLoader.get_code

    def written(self, fullname):
        found = ours.search(self.get_filename(fullname).replace(os.sep, "/"))
        if found:
            with open(os.path.join(folder, str(os.getpid())), "a", encoding="utf-8") as out:
                out.write(found.group(1) + "\n")
        return get_code(self, fullname)

    SourceFileLoader.get_code = written


if os.environ.get("WPT_CI_PYTHON") == sys.prefix:
    if os.environ.get("WPT_CPU"):
        _block_triton()
    if os.environ.get("WPT_CHILD_FILES"):
        _write_loaded_files(os.environ["WPT_CHILD_FILES"])
