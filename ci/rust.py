# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""The tokenizer's Rust tests. Options: common.py."""

import os
import subprocess
import sys

from common import ENVIRON, ROOT, UV, Part, execute, main

TOKENIZER = "packages/wpt-tokenizer"
# the folder of the project's Python library (on Windows, its DLL is in the base folder)
PYTHON_LIBRARY = "import sys, sysconfig; print(sysconfig.get_config_var('LIBDIR') or sys.base_prefix)"
# the variable each system reads for the folders where a program's shared libraries are found
LIBRARY_PATH = {"darwin": "DYLD_LIBRARY_PATH", "win32": "PATH"}


def with_python_library():
    """ENVIRON with the project's Python library folder searched first: the
    Rust test program links the Python library and loads it when it starts,
    and the Python uv installs is in a folder no system searches."""
    found = subprocess.run([UV, "run", "python", "-c", PYTHON_LIBRARY], cwd=ROOT, env=ENVIRON, capture_output=True,
                           text=True, check=True)
    name = LIBRARY_PATH.get(sys.platform, "LD_LIBRARY_PATH")
    return {**ENVIRON, name: os.pathsep.join(f for f in (found.stdout.strip(), ENVIRON.get(name)) if f)}


def run(given):
    # inside the folder, so rustup applies rust-toolchain.toml; through uv, so the
    # Rust build finds the project's Python even where none is on the search path
    return execute([[UV, "run", "rustc", "--version"], [UV, "run", "cargo", "test"]], ROOT / TOKENIZER / "rust",
                   with_python_library())


PART = Part("rust", lambda path: path.startswith(f"{TOKENIZER}/rust/") or path == f"{TOKENIZER}/rust-toolchain.toml", run)

if __name__ == "__main__":
    main(PART)
