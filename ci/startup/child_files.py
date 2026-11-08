"""A pytest plugin that makes pytest-testmon see the programs a test starts.

Coverage measures only the test's own process. Each test gets its own folder,
named by WPT_CHILD_FILES, where sitecustomize.py makes the programs the test
starts write the source files they load, and the test then depends on every
line of those files, on top of what its own process ran. The plugin extends
TestmonCollector.get_nodes_files_lines, an internal of pytest-testmon, whose
version the root pyproject.toml pins.
"""

import functools
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest

FOLDER = None  # made by pytest_configure, removed by pytest_unconfigure: importing this file makes nothing


def folder_of(test) -> Path:
    return FOLDER / hashlib.sha256(test.encode()).hexdigest()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    folder = folder_of(item.nodeid)
    folder.mkdir(exist_ok=True)
    os.environ["WPT_CHILD_FILES"] = str(folder)
    try:
        return (yield)
    finally:
        del os.environ["WPT_CHILD_FILES"]


def loaded_files(test) -> set[str]:
    """The source files the programs of a test loaded, as paths from the root."""
    folder = folder_of(test)
    if not folder.exists():
        return set()
    return {name for log in folder.iterdir() for name in log.read_text(encoding="utf-8").split()}


@functools.cache
def every_line(path) -> frozenset[int]:
    return frozenset(range(1, len(path.read_bytes().splitlines()) + 1))


def with_child_files(nodes, root):
    """nodes, testmon's {test: {file: lines}}, with every line of the files
    the test's programs loaded added to what the test ran, a file that does
    not exist here left out, and without the files with no extension, on
    which testmon crashes."""
    for test, files in nodes.items():
        for name in [name for name in files if "." not in Path(name).name]:
            del files[name]
        for name in loaded_files(test):
            if (root / name).exists():
                files[name] = files.get(name, set()) | every_line(root / name)
    return nodes


def pytest_configure(config):
    global FOLDER
    FOLDER = Path(tempfile.mkdtemp(prefix="wpt-child-files-"))
    from testmon import testmon_core

    collect = testmon_core.TestmonCollector.get_nodes_files_lines

    def extended(self, dont_include):
        nodes, lines = collect(self, dont_include)
        return with_child_files(nodes, config.rootpath), lines

    testmon_core.TestmonCollector.get_nodes_files_lines = extended


def pytest_unconfigure(config):
    if FOLDER is not None:
        shutil.rmtree(FOLDER, ignore_errors=True)
