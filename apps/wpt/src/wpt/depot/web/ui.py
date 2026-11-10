"""Build the interface's files with Bun, only when they are stale.

The interface's sources live in apps/wpt/ui (a source checkout: uv
installs workspace members in place). It is a member of the Bun workspace
of the repository's root package.json, with the organization's design
system, the git submodule vendor/wic-design-system, whose packages the
install builds first. The install writes node_modules folders and the
design system's dist folders; the build writes ui/dist. All are ignored by
git.
"""

import subprocess
from pathlib import Path

from wpt_utils.lines import say
from wpt_utils.programs import program

UI = Path(__file__).resolve().parents[4] / "ui"
STAMP = ".built"  # in dist/: written last, so a failed build stays stale
DESIGN_SYSTEM = Path("vendor") / "wic-design-system"  # from the repository's root
SKIPPED = {"node_modules", "dist", ".git"}


class UiBuildFailed(Exception):
    """The design system's submodule is not checked out, or `bun install`
    or `bun run build` failed; its output was shown. dist/ may hold a
    partial build, which stays stale and is rebuilt next time."""


def _newest_source(folders: list[Path], files: list[Path]) -> float:
    found = [p.stat().st_mtime for p in files if p.is_file()]
    for folder in folders:
        found += [p.stat().st_mtime for p in folder.rglob("*")
                  if p.is_file() and not SKIPPED & set(p.relative_to(folder).parts)]
    return max(found)


def built(ui: Path = UI, log=say) -> Path:
    """The folder of built files, building it first when any source (the
    interface's, the workspace's manifest and lock, the design system's
    packages) is newer than the last build. MissingTool if Bun is not
    installed; UiBuildFailed when the build cannot be made."""
    root = ui.parents[2]
    packages = root / DESIGN_SYSTEM / "packages"
    if not packages.is_dir() or not any(packages.iterdir()):
        raise UiBuildFailed(f"{root / DESIGN_SYSTEM} is empty: run `git submodule update --init` in {root}")
    dist, stamp = ui / "dist", ui / "dist" / STAMP
    if stamp.is_file() and stamp.stat().st_mtime >= _newest_source(
            [ui, packages], [root / "package.json", root / "bun.lock"]):
        return dist
    bun = program("bun")
    log("building the interface (bun install, bun run build)")
    for command, where in (([bun, "install", "--frozen-lockfile"], root), ([bun, "run", "build"], ui)):
        if subprocess.run(command, cwd=where).returncode != 0:
            raise UiBuildFailed(f"`{' '.join(command[1:])}` failed in {where}")
    stamp.write_text("", encoding="utf-8")
    return dist
