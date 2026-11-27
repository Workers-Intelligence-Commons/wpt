"""The parts import one another only as the dependency rule allows
(docs/architecture.md, "The parts"):

- the depot imports none of the dock, the freight and WPT;
- the dock imports none of the depot, the freight and WPT;
- the freight imports the depot's and the dock's api modules, nothing else
  of them, and nothing of WPT;
- WPT imports the freight's and the dock's api modules, and nothing of the
  depot;
- wic_utils, which every part may import, imports none of them.

A part is its library and its controllers: the depot is `depot` and
`depot_app`, the dock `dock` and `dock_app`, the freight `freight` and
`freight_app`; a package's benchmarks and scripts (packages/*/bench,
packages/*/scripts) are its part's; every other module of packages/*/src,
apps/*/src and scripts/ is WPT's. A part's own controllers may import all of
it.

WPT reaches texts only through the freight's deliveries on the dock, so no
module of WPT imports the libraries a depot's SQLite file is opened with
(DEPOT_FILES): a read of a depot's file that imports nothing of the depot
fails too.

The test reads the source of every module and fails on any import that
breaks the rule and is not in EXCEPTIONS, each of which says why it stands,
and on an exception that no import needs.
It also fails on any name of dock.api that another part takes, imported
(`from dock.api import X`) or read from the module (`api.X` after `from dock
import api`), and that is neither a function, an error, nor a plain value
the module lists in __all__: the dock is named by its URL, and no part holds
a dock object (docs/api/dock.md, "The calls").
The tests themselves are left out: they build the depots, docks and
deliveries they check.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEPOT, DOCK, FREIGHT, WPT, SHARED = "depot", "dock", "freight", "WPT", "wic_utils"
PARTS = {"depot": DEPOT, "depot_app": DEPOT, "dock": DOCK, "dock_app": DOCK, "freight": FREIGHT,
         "freight_app": FREIGHT, "wic_utils": SHARED}
# what each part may import of each other part: None for nothing, else the modules allowed
ALLOWED = {
    DEPOT: {DOCK: None, FREIGHT: None, WPT: None},
    DOCK: {DEPOT: None, FREIGHT: None, WPT: None},
    FREIGHT: {DEPOT: {"depot.api"}, DOCK: {"dock.api"}, WPT: None},
    WPT: {DEPOT: None, DOCK: {"dock.api"}, FREIGHT: {"freight.api"}},
    SHARED: {DEPOT: None, DOCK: None, FREIGHT: None, WPT: None},
}
# the libraries that open an SQLite file, a depot's: no module of WPT imports them
DEPOT_FILES = ("sqlite3", "apsw")
# (module that imports, module imported): imports the rule refuses that stand, each with its reason
EXCEPTIONS = {
    ("depot_app", "dock.api"): "the depot's backups go to the dock until they have a place of their own, and an "
                               "upgrade from a format before snapshots reads the records of the dock's corpora",
    ("depot_app.corpora", "dock.api"): "an upgrade from a format before snapshots reads the records of the dock's "
                                       "corpora",
    ("depot_app.web.routes", "dock.api"): "the routes answer a refusal of the dock as such",
    ("depot_app.web.translations", "dock.api"): "the Translate button's models are kept on the dock under their "
                                                "sha256",
}


def module_files() -> dict[str, Path]:
    """{module name: its file} of every module of the parts and of WPT."""
    found = {}
    for source in [*ROOT.glob("packages/*/src"), *ROOT.glob("apps/*/src")]:
        for path in source.rglob("*.py"):
            parts = path.relative_to(source).with_suffix("").parts
            found[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    for folder in [*ROOT.glob("packages/*/bench"), *ROOT.glob("packages/*/scripts")]:  # a package's, its part's
        [package] = [one.name for one in (folder.parent / "src").iterdir() if (one / "__init__.py").is_file()]
        for path in folder.glob("*.py"):
            found[f"{package}.{folder.name}.{path.stem}"] = path
    for path in ROOT.glob("scripts/*.py"):
        found[f"scripts.{path.stem}"] = path
    return found


def part_of(module: str) -> str | None:
    """The part a module belongs to; None for the standard library and third-party packages."""
    top = module.split(".")[0]
    if top in PARTS:
        return PARTS[top]
    return WPT if top.startswith("wpt") or top == "scripts" else None


def imported(module: str, path: Path) -> list[tuple[int, str]]:
    """(line, module) of each module or name the file imports, relative imports made absolute:
    `from depot import api` gives depot.api, `from depot import Depot` gives depot.Depot."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - node.level + 1]
                name = ".".join([*base, node.module] if node.module else base)
            else:
                name = node.module
            found += [(node.lineno, f"{name}.{alias.name}") for alias in node.names]
    return found


def refused(importer: str, name: str) -> bool:
    mine, theirs = part_of(importer), part_of(name)
    if theirs is None or theirs == mine or theirs == SHARED:
        return False
    allowed = ALLOWED[mine][theirs]
    return allowed is None or not any(name == one or name.startswith(one + ".") for one in allowed)


def problems() -> list[str]:
    found = []
    for module, path in module_files().items():
        for line, name in imported(module, path):
            if refused(module, name) and (module, ".".join(name.split(".")[:2])) not in EXCEPTIONS:
                found.append(f"{path.relative_to(ROOT).as_posix()}:{line}: {module} ({part_of(module)}) imports "
                             f"{name} ({part_of(name)})")
    return sorted(found)


def test_every_part_imports_the_others_only_as_the_rule_allows():
    assert problems() == [], "imports between the parts that the rule refuses:\n" + "\n".join(problems())


def test_every_exception_stands_for_an_import_the_rule_refuses():
    needed = {(module, ".".join(name.split(".")[:2])) for module, path in module_files().items()
              for _, name in imported(module, path) if refused(module, name)}
    unneeded = sorted(f"{importer} imports {name}" for importer, name in EXCEPTIONS if (importer, name) not in needed)
    assert unneeded == [], "exceptions no import needs:\n" + "\n".join(unneeded)


def test_no_module_of_wpt_opens_an_sqlite_file():
    found = sorted(f"{path.relative_to(ROOT).as_posix()}:{line}: {module} imports {name}"
                   for module, path in module_files().items() if part_of(module) == WPT
                   for line, name in imported(module, path) if name.split(".")[0] in DEPOT_FILES)
    assert found == [], "modules of WPT that import what opens a depot's file:\n" + "\n".join(found)


def test_the_check_reads_a_package_s_scripts_as_its_part_s():
    # packages/depot/scripts/source_credits.py reads a depot's file through the depot's library: the depot's own
    assert "depot.scripts.source_credits" in module_files() and part_of("depot.scripts.source_credits") == DEPOT


def dock_values(name: str) -> bool:
    """Whether the name of dock.api is one other parts may import: one it
    lists in __all__ that is a function, an error, or a plain value (a
    string, a number, a pattern)."""
    import inspect
    import re

    from dock import api

    if name not in api.__all__:
        return False
    found = getattr(api, name)
    if inspect.isclass(found):
        return issubclass(found, Exception)
    return inspect.isfunction(found) or isinstance(found, (str, int, float, re.Pattern))


def dock_names(path: Path) -> list[tuple[int, str]]:
    """(line, name) of each name the file reads from the module dock.api
    through a name bound to it: `api.fetch` after `from dock import api`,
    `dock_api.X` after `... as dock_api`, `dock.api.X` after `import
    dock.api`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = set()  # the names the file binds to dock.api
    whole = False  # whether the file reads it as dock.api
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dock" and not node.level:
            bound |= {alias.asname or alias.name for alias in node.names if alias.name == "api"}
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dock.api":
                    if alias.asname:
                        bound.add(alias.asname)
                    else:
                        whole = True
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in bound:
            found.append((node.lineno, node.attr))
        elif (whole and isinstance(node.value, ast.Attribute) and node.value.attr == "api"
              and isinstance(node.value.value, ast.Name) and node.value.value.id == "dock"):
            found.append((node.lineno, node.attr))
    return found


def test_other_parts_take_from_the_dock_only_functions_errors_and_values():
    found = sorted(f"{path.relative_to(ROOT).as_posix()}:{line}: {module} imports {name}"
                   for module, path in module_files().items() if part_of(module) != DOCK
                   for line, name in imported(module, path)
                   if name.startswith("dock.api.") and not dock_values(name.removeprefix("dock.api.")))
    found += sorted(f"{path.relative_to(ROOT).as_posix()}:{line}: {module} reads dock.api.{name}"
                    for module, path in module_files().items() if part_of(module) != DOCK
                    for line, name in dock_names(path) if not dock_values(name))
    assert found == [], "names of dock.api that are no function, error or value:\n" + "\n".join(found)


def test_the_check_of_dock_api_sees_names_read_from_the_module(tmp_path):
    source = tmp_path / "caller.py"
    source.write_text("from dock import api as dock_api\nimport dock.api\n"
                      "dock_api._dock('file:///d').path('k')\ndock.api.open_dock('file:///d')\ndock_api.fetch\n")
    assert sorted(dock_names(source)) == [(3, "_dock"), (4, "open_dock"), (5, "fetch")]
    assert sorted(name for _, name in dock_names(source) if not dock_values(name)) == ["_dock", "open_dock"]


def test_the_rule_refuses_each_import_it_names():
    assert refused("freight.job", "depot.depot")
    assert not refused("freight.job", "depot.api")
    assert refused("wpt.train", "depot.api")
    assert not refused("wpt.train", "freight.api")
    assert refused("wpt.train", "freight.deliveries")
    assert refused("depot.backup", "dock.api")
    assert refused("dock", "wpt_utils.device")
    assert refused("wic_utils.lines", "depot.Depot")
    assert refused("wpt.docked", "depot")
    assert not refused("depot_app.web", "depot.errors")
    assert not refused("wpt.train", "wic_utils.lines")
    assert not refused("wpt.train", "json")
    assert dock_values("fetch") and dock_values("MissingFile") and dock_values("NAME")
    assert not dock_values("Dock") and not dock_values("File") and not dock_values("open_dock")
