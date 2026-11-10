"""The Git repository a task works in, fetched when the task runs.

fetch(address, commit) gives the files of a new repository that holds that
one commit of the repository at address, fetched alone (a shallow fetch,
git fetch --depth 1) and checked out at it on the branch BRANCH: the
working tree's files and its .git folder, as a task holds files
(wpt_agent.files). A task's files hold no empty folder, and git knows a
.git folder only when its refs folder exists, so the branch's ref is what
keeps that folder.
Each commit is fetched once per process and kept in memory; it runs on the
machine that plays the attempts, since their containers have no network.
RepositoryFailed when git cannot fetch the commit or the checkout is not at
it.
"""

import functools
import subprocess
import tempfile

from wpt_utils.programs import program

from .files import read_folder

BRANCH = "main"  # the branch the fetched commit is checked out on


class RepositoryFailed(Exception):
    """The repository's commit could not be fetched or checked out."""


@functools.cache
def fetch(address: str, commit: str) -> dict[str, tuple[str, bytes]]:
    """{path: (mode, content)} of the repository at address checked out at
    commit, its .git folder included."""
    with tempfile.TemporaryDirectory(prefix="wpt-repository-") as folder:
        git = program("git")

        def run(*args):
            try:
                return subprocess.run([git, "-C", folder, *args], check=True,
                                      capture_output=True).stdout.decode().strip()
            except subprocess.CalledProcessError as error:
                raise RepositoryFailed(f"{address} at {commit}: git {args[0]} failed: "
                                       f"{error.stderr.decode(errors='replace').strip()}") from None

        run("init", "--quiet")
        run("fetch", "--quiet", "--depth", "1", address, commit)
        run("checkout", "--quiet", "-B", BRANCH, commit)
        if run("rev-parse", "HEAD") != commit:
            raise RepositoryFailed(f"{address}: the checkout is not at {commit}")
        return read_folder(folder)
