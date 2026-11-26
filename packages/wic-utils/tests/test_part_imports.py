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
`freight_app`; every other module of packages/*/src, apps/*/src and
scripts/ is WPT's. A part's own controllers may import all of it.

The test reads the source of every module and fails on any import that
breaks the rule and is not in EXCEPTIONS, each of which says why it stands.
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
# (module that imports, module imported): imports the rule refuses that stand, each with its reason
EXCEPTIONS = {
    ("depot_app", "dock.api"): "the depot's backups go to the dock until they have a place of their own, and an "
                               "upgrade from a format before snapshots reads the records of the dock's corpora",
    ("depot_app.web.legacy", "dock.api"): "the frozen dataset routes send the corpora built from a dataset, which "
                                          "the dock holds",
    ("depot_app.shared", "dock.api"): "an erasure carried out marks stale the corpora on the dock that hold a text it "
                                      "reached, so that the frozen dataset routes never send one",
    ("depot_app.corpora", "dock.api"): "the depot's reading of those corpora",
    ("depot_app.web.routes", "dock.api"): "the routes answer a refusal of the dock as such",
    ("depot_app.web.translations", "dock.api"): "the Translate button's models are kept on the dock under their "
                                                "sha256",
    ("wpt.adopt", "depot.api"): "wpt adopt-vocabularies puts on the dock the vocabularies the depot's tokenizer "
                                "recipes trained, which only the depot keeps",
}


def module_files() -> dict[str, Path]:
    """{module name: its file} of every module of the parts and of WPT."""
    found = {}
    for source in [*ROOT.glob("packages/*/src"), *ROOT.glob("apps/*/src")]:
        for path in source.rglob("*.py"):
            parts = path.relative_to(source).with_suffix("").parts
            found[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
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


def test_the_rule_refuses_each_import_it_names():
    assert refused("freight.job", "depot.depot")
    assert not refused("freight.job", "depot.api")
    assert refused("wpt.train", "depot.api")
    assert not refused("wpt.train", "freight.api")
    assert refused("wpt.train", "freight.deliveries")
    assert refused("depot.backup", "dock.api")
    assert refused("dock", "wpt_utils.device")
    assert refused("wic_utils.lines", "depot.Depot")
    assert refused("wpt.store", "depot")
    assert not refused("depot_app.web", "depot.errors")
    assert not refused("wpt.train", "wic_utils.lines")
    assert not refused("wpt.train", "json")
