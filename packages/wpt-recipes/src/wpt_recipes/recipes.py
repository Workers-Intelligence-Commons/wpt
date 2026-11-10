"""The service layer of the recipes and the task images.

Every file of these kinds is opened here and nowhere else
(docs/architecture.md, "The service layer"):

    data/recipes/tokenizers/<name>.json   a tokenizer recipe
                          + <name>.lock.json (written by the command that trains the vocabulary)
    data/recipes/models/<name>.json       data, architecture and training steps of one model
    data/recipes/series/<name>.json       a model series: the rules its sizes share and the sizes
    data/images/<name>/Dockerfile         a container image of agentic tasks (and any file it copies)
                      + lock.json, licenses.json (written by `wpt agent lock-image`)
                      + tools.json, the shared pool of tools its programs answer (optional)

A Recipes hands out documents and bytes, never a path under data/; docker
gets an image's files as a tar archive made from image_files. People write
the recipes and tools.json themselves; the generated files (tokenizer
locks, image locks and image license lists) are written only through
write_generated, by the command that makes them. The audit hook of
guard.py refuses any other code that reaches these files.

Every read is checked: a recipe against its schema (InvalidRecipe), and
check() and problems() add the rules across files. A recipe whose depot
or dataset is still empty passes its schema, so it reads, and the check
names it for a person to fill in.
"""

import json
import os
import re
import shutil
import subprocess
from functools import cache
from importlib import resources
from pathlib import Path

import jsonschema

from . import series
from .errors import InvalidRecipe, NotFound, RecipeError
from .guard import guard

DATA_FOLDER = "data"  # the repository's data, a top-level folder of its root
NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
RECIPE_FOLDERS = {"tokenizer": "tokenizers", "model": "models", "series": "series"}
IMAGE_FILES = {"dockerfile": "Dockerfile", "image-lock": "lock.json", "image-licenses": "licenses.json",
               "image-tools": "tools.json"}
# every kind of file, by its name: its JSON schema (None for a text kind), and the command that writes a
# generated kind (None: people write it)
KINDS = {
    "tokenizer": ("tokenizer", None),
    "model": ("model", None),
    "series": ("series", None),
    "lock": ("lock", "the command that trains the vocabulary"),
    "dockerfile": (None, None),
    "image-lock": ("image-lock", "wpt agent lock-image"),
    "image-licenses": ("image-licenses", "wpt agent lock-image"),
    "image-tools": ("image-tools", None),
}
SIZE_OF = ":"  # SERIES:SIZE[:RUN] names one size of a series (or another run of it) wherever a model recipe's
#                name goes (Recipes.model_recipe)
BUILT_IN_TOOLS = ("read", "write", "edit", "bash")  # the tools every task can offer (wpt_agent.tools)
ASK = "ask"  # the tool a task with replies offers
# characters str.splitlines() splits on that JSON may hold raw inside a string
LINE_BREAKS = {" ": "\\u2028", " ": "\\u2029", "\x85": "\\u0085"}


@cache
def schema(name: str) -> dict:
    """The JSON schema of that name (a kind's, KINDS)."""
    return json.loads(resources.files(__package__).joinpath(f"schemas/{name}.schema.json").read_text())


@cache
def _validator(name: str):
    """One validator per schema: jsonschema.validate checks the schema itself
    on every call, which costs more than checking a small document."""
    document = schema(name)
    jsonschema.validators.validator_for(document).check_schema(document)
    return jsonschema.validators.validator_for(document)(document)


def validate(document, kind: str, where: str):
    """InvalidRecipe unless the document is a valid file of that kind; where
    names it in the message."""
    error = jsonschema.exceptions.best_match(_validator(KINDS[kind][0]).iter_errors(document))
    if error is not None:
        place = "/".join(map(str, error.absolute_path)) or "(top level)"
        raise InvalidRecipe(f"{where}: at {place}: {error.message}")


def dump_json(document) -> str:
    """How every file of these kinds is written: 2-space indent, UTF-8
    characters kept, U+2028, U+2029 and U+0085 as escapes so that no line
    splitter cuts a line of JSON, final newline."""
    text = json.dumps(document, indent=2, ensure_ascii=False)
    for character, escape in LINE_BREAKS.items():
        text = text.replace(character, escape)
    return text + "\n"


def check_declared(where: str, names: list[str]):
    """InvalidRecipe when a declared tool takes the name of a tool every task
    has, or is declared twice (a task's tools, an image's pool)."""
    for name in names:
        if name in BUILT_IN_TOOLS or name == ASK or names.count(name) > 1:
            raise InvalidRecipe(f"{where}: the tool {name!r} is declared twice, or takes the name of a tool every "
                                f"task has ({', '.join(BUILT_IN_TOOLS)}, {ASK})")


def _empty_data(where: str, data: dict, place: str):
    """InvalidRecipe naming the fields of a data part that a person has not
    filled in yet: the depot's address and the dataset's name."""
    empty = [key for key in ("depot", "dataset") if not data[key]]
    if empty:
        raise InvalidRecipe(f"{where}: {' and '.join(f'{place}{key}' for key in empty)} empty: a person names the "
                            f"depot's address and the name of its dataset")


class Recipes:
    """The recipes and task images of the repository at root."""

    def __init__(self, root):
        self.root = Path(root)
        self._data = self.root / DATA_FOLDER
        guard(self._data, __file__)

    @classmethod
    def find(cls, start=None) -> "Recipes":
        """The nearest folder at or above start (default: the current folder)
        holding data/recipes; NotFound when there is none."""
        here = Path(start or Path.cwd()).resolve()
        for folder in (here, *here.parents):
            if (folder / DATA_FOLDER / "recipes").is_dir():
                return cls(folder)
        raise NotFound(f"no data/recipes folder at or above {here}")

    def data_folder(self) -> str:
        """The data folder's path from the repository's root, with /: what a
        copy of the repository's code leaves out (wpt_worker.manager)."""
        return self._data.relative_to(self.root).as_posix()

    # -- where each kind lives ---------------------------------------------------
    def _path(self, kind: str, name: str) -> Path:
        """The file of that kind and name; NotFound for an unknown kind or a
        name that is not one (so no name reaches outside its folder)."""
        if kind not in KINDS:
            raise NotFound(f"no kind of file named {kind!r}; the kinds are {', '.join(KINDS)}")
        if not NAME.fullmatch(name):
            raise NotFound(f"no {kind} {name!r}: a name is lowercase a-z, 0-9, '.', '_' and '-'")
        if kind in RECIPE_FOLDERS:
            return self._data / "recipes" / RECIPE_FOLDERS[kind] / f"{name}.json"
        if kind == "lock":
            return self._data / "recipes" / "tokenizers" / f"{name}.lock.json"
        return self._data / "images" / name / IMAGE_FILES[kind]

    def relative(self, kind: str, name: str) -> str:
        """The file's place in the repository, for messages and records:
        data/... with forward slashes."""
        return self._path(kind, name).relative_to(self.root).as_posix()

    def exists(self, kind: str, name: str) -> bool:
        return self._path(kind, name).is_file()

    def names(self, kind: str) -> list[str]:
        """The names of the existing files of a kind, sorted."""
        if kind in RECIPE_FOLDERS:
            folder = self._data / "recipes" / RECIPE_FOLDERS[kind]
            return sorted(p.stem for p in folder.glob("*.json") if not p.name.endswith(".lock.json")) \
                if folder.is_dir() else []
        if kind == "lock":
            folder = self._data / "recipes" / "tokenizers"
            return sorted(p.name.removesuffix(".lock.json") for p in folder.glob("*.lock.json")) \
                if folder.is_dir() else []
        if kind not in KINDS:
            raise NotFound(f"no kind of file named {kind!r}; the kinds are {', '.join(KINDS)}")
        return [name for name in self.image_names() if (self._data / "images" / name / IMAGE_FILES[kind]).is_file()]

    def _load(self, kind: str, name: str) -> dict:
        """The JSON file of that kind, validated; NotFound when it does not exist."""
        path = self._path(kind, name)
        if not path.is_file():
            raise NotFound(f"no {kind} {name!r} ({self.relative(kind, name)} does not exist)")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidRecipe(f"{self.relative(kind, name)}: {error}") from error
        validate(document, kind, self.relative(kind, name))
        return document

    # -- recipes -----------------------------------------------------------------
    def recipe(self, kind: str, name: str) -> dict:
        """The tokenizer, model or series recipe of that name, validated
        against its schema; NotFound when it does not exist."""
        if kind not in RECIPE_FOLDERS:
            raise NotFound(f"no kind of recipe named {kind!r}; the kinds are {', '.join(RECIPE_FOLDERS)}")
        return self._load(kind, name)

    def model_recipe(self, given: str) -> dict:
        """The model recipe a name gives: NAME, the file of
        data/recipes/models/; SERIES:SIZE, that size of the series file
        data/recipes/series/SERIES.json, derived from it (series.py);
        SERIES:SIZE:RUN, another run of that size (a decay branch, a
        learning rate of the check: series.runs). NotFound when there is no
        such recipe, series, size or run; InvalidRecipe when the series'
        sizes cannot be derived."""
        name, separated, rest = given.partition(SIZE_OF)
        if not separated:
            return self.recipe("model", given)
        size, _, run = rest.partition(SIZE_OF)
        document = self.recipe("series", name)
        if size not in series.sizes(document):
            raise NotFound(f"no size {size!r} in the series {name}; its sizes are {', '.join(series.sizes(document))}")
        try:
            if run and run not in series.runs(document, size):
                listed = series.runs(document, size)
                raise NotFound(f"series {name}: no run {run!r} of the size {size}; "
                               + (f"its runs are {', '.join(listed)}" if listed else "it has no other run"))
            return series.run_recipe(document, series.plan(document, size, self.vocab_size(document)), run)
        except ValueError as error:
            raise InvalidRecipe(f"series {name}: {error}") from None

    def series_plans(self, name: str) -> list[series.Plan]:
        """Every size's numbers in the series file of that name (series.Plan)."""
        document = self.recipe("series", name)
        return series.plans(document, self.vocab_size(document))

    def series_total_tokens(self, name: str) -> int:
        """The training tokens of the one corpus of the series file of that name (series.data_part)."""
        return series.data_part(self.recipe("series", name))["total_tokens"]

    def series_runs(self, name: str, found: series.Plan) -> list[dict]:
        """What each other run of a size of the series file of that name does (series.run_numbers)."""
        return series.run_numbers(self.recipe("series", name), found)

    def vocab_size(self, recipe: dict) -> int:
        """The vocabulary size the tokenizer recipe of a model recipe's or a series' data part asks for."""
        return self.recipe("tokenizer", recipe["data"]["tokenizer"])["vocab_size"]

    def generated(self, kind: str, name: str) -> dict | None:
        """A generated file (lock, image-lock, image-licenses), validated; None
        when it does not exist."""
        if not KINDS.get(kind, (None, None))[1]:
            raise NotFound(f"{kind!r} is not a kind of generated file")
        return self._load(kind, name) if self.exists(kind, name) else None

    def write_generated(self, kind: str, name: str, document: dict) -> str:
        """Write a generated file (lock, image-lock, image-licenses) after its
        schema, the image's folder or the tokenizer recipe it belongs to
        existing; only the command that makes it calls this. Nothing is
        written when a check refuses it. Returns its place in the
        repository."""
        if not KINDS.get(kind, (None, None))[1]:
            raise NotFound(f"{kind!r} is not a kind of generated file: people write it")
        if kind == "lock":
            self.recipe("tokenizer", name)
        else:
            self._image_folder(name)
        validate(document, kind, self.relative(kind, name))
        self._write_file(self._path(kind, name), dump_json(document).encode("utf-8"))
        return self.relative(kind, name)

    def _write_file(self, path: Path, data: bytes):
        """Write data as path's bytes, through a temporary file renamed into
        place; InvalidRecipe when path, through a symbolic link, leads
        outside the repository: then nothing is written."""
        target = Path(os.path.realpath(path))
        if not target.is_relative_to(Path(os.path.realpath(self.root))):
            raise InvalidRecipe(f"{path.relative_to(self.root).as_posix()} leads to {target}, outside the repository "
                                f"(through a symbolic link)")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.writing")
        temporary.unlink(missing_ok=True)  # a link left there would be written through
        temporary.write_bytes(data)
        os.replace(temporary, target)

    # -- images ------------------------------------------------------------------
    def image_names(self) -> list[str]:
        folder = self._data / "images"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def _image_folder(self, name: str) -> Path:
        """data/images/<name>/; NotFound unless the name is valid and the
        folder holds a Dockerfile."""
        if not NAME.fullmatch(name):
            raise NotFound(f"no image {name!r}: an image name is lowercase a-z, 0-9, '.', '_' and '-'")
        folder = self._data / "images" / name
        if not (folder / IMAGE_FILES["dockerfile"]).is_file():
            raise NotFound(f"no image {name!r}: data/images/{name}/Dockerfile does not exist")
        return folder

    def check_image(self, name: str):
        """NotFound unless data/images/<name>/ holds a Dockerfile."""
        self._image_folder(name)

    def image_tools(self, name: str) -> list[dict]:
        """The declarations of data/images/<name>/tools.json, the image's
        shared pool of tools ([] without the file), validated; NotFound as
        check_image, InvalidRecipe for a pool the check refuses."""
        self._image_folder(name)
        if not self.exists("image-tools", name):
            return []
        tools = self._load("image-tools", name)["tools"]
        check_declared(self.relative("image-tools", name), [tool["name"] for tool in tools])
        return tools

    def image_files(self, name: str) -> dict[str, tuple[bool, bytes]]:
        """{path in the image's folder: (executable, content)} of every file
        there, the Dockerfile, its lock and licenses included, but none git
        ignores (a __pycache__ a Python run left): the image, and its tag,
        come from the files the repository can hold; NotFound as
        check_image."""
        folder = self._image_folder(name)
        found = [p for p in sorted(folder.rglob("*")) if p.is_file()]
        ignored = self._ignored([p.relative_to(self.root).as_posix() for p in found])
        return {p.relative_to(folder).as_posix(): (bool(p.stat().st_mode & 0o100), p.read_bytes())
                for p in found if p.relative_to(self.root).as_posix() not in ignored}

    def images_differ(self, revision: str) -> bool:
        """Whether the files under data/images/ as they are differ from the
        git revision, the new files git does not track yet included."""
        changed = self._git("diff", "--name-only", revision, "--", "data/images").stdout
        new = self._git("ls-files", "--others", "--exclude-standard", "--", "data/images").stdout
        return bool(changed.strip() or new.strip())

    def _git(self, *args) -> subprocess.CompletedProcess:
        git = shutil.which("git")
        if git is None:
            raise RecipeError("git is not installed")
        return subprocess.run([git, "-C", str(self.root), *args], capture_output=True, check=True)

    def _ignored(self, paths) -> set[str]:
        """Those of the paths (relative to the root) that git ignores and does
        not track; none when the repository is not a git working tree, or git
        is not installed, so nothing tells what it ignores."""
        git = shutil.which("git")
        if not paths or git is None:
            return set()
        result = subprocess.run([git, "-C", str(self.root), "check-ignore", "-z", "--stdin"],
                                input="\0".join(paths).encode() + b"\0", capture_output=True, check=False)
        if result.returncode not in (0, 1):  # 1: none ignored; otherwise not a git working tree
            return set()
        return {path.decode() for path in result.stdout.split(b"\0") if path}

    # -- the rules -----------------------------------------------------------------
    def problems(self, kind: str, name: str) -> list[str]:
        """What the check finds in that one file: [] or its problem."""
        try:
            if kind in RECIPE_FOLDERS:
                self._check_recipe(kind, name)
            elif kind == "dockerfile":
                self._image_folder(name)
            elif kind == "image-tools":
                self.image_tools(name)
            else:
                self._load(kind, name)
        except RecipeError as error:
            return [str(error)]
        return []

    def check(self) -> list[str]:
        """Every rule on every recipe, the locks of the tokenizer recipes and
        every image; one line per problem."""
        found = []
        for kind in RECIPE_FOLDERS:
            for name in self.names(kind):
                found += self.problems(kind, name)
        for name in self.names("lock"):
            if name in self.names("tokenizer"):
                found += self.problems("lock", name)
        for name in self.image_names():
            found += self.problems("dockerfile", name)
            for kind in ("image-lock", "image-licenses", "image-tools"):
                if self.exists(kind, name):
                    found += self.problems(kind, name)
        return found

    def _check_recipe(self, kind: str, name: str):
        document = self.recipe(kind, name)
        where = self.relative(kind, name)
        if kind == "tokenizer":
            _empty_data(where, document, "")
            return
        self.recipe("tokenizer", document["data"]["tokenizer"])
        if kind == "series":
            try:
                series.check(document)
            except ValueError as error:
                raise InvalidRecipe(f"{where}: {error}") from None
        _empty_data(where, document["data"], "data.")
