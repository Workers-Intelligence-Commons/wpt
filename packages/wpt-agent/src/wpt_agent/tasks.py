"""Tasks for agentic training (docs/training.md, "Tasks").

A task is the content of a depot item of kind `task` (the depot checks it
against the kind's JSON Schema when it is added):

    task     instructions (and, optionally, the same request in one line, terse), image,
             limits, container sizes, the tools it needs and offers, the user's replies to the
             ask tool and where its answers live, the paths it may change, the rules that gate
             its score, and the Git repository it works in, when it has one
    files    the starting files, by their path under files/: {mode, content in base64}; those
             task.tool_files names go where only the tools' programs read them
    check    any program, the same way; run after the attempt, it prints a score from 0 to 1

from_item reads one into a Task and applies the rules across fields that
a schema cannot: its image must be a folder of data/images/ (wpt_recipes).
The attempt's container, and the check's, take SIZES, each raised by the
task's `container` when it asks for more; /work and /tmp are kept in
memory and may each hold `disk` MiB, so 2 x disk + HEADROOM must stay
below `memory`. A task's tools are the built-in ones (BUILT_IN_TOOLS),
`ask` when it has replies, and those it declares, which may not take
those names; each name in `needs` is one of them, in one need only, and
`ask` needs replies. A variant is one tool, or tools offered together with
the paths under files/ that it hides (Variant). A task with replies names,
in `answered_by`, the looking tools that can answer what it leaves open
(looks: read, bash, or a declared tool of the task or its image's pool
marked `looks`); and, in `answered_in`, glob patterns of the paths under
/work where its answers live (wpt_check.asked_last). Every reply's pattern
must compile, and every tool_files entry, a variant's included, must name
a file of files/ or a folder holding some. Each `absent` entry is a folder
of the image, outside /work and /tmp (OWN), that the task's attempts see
empty. A task's terse form is one line. A task's `area` lists glob
patterns of the paths under /work an attempt may change (wpt_agent.play
scores it), and its `gates` the parts that must be 1 for an attempt to
score at all (wpt_agent.gated).

A task that names a Git repository (`repository`: its address and the full
id of a commit) works in it: when the task runs, prepared() fetches that
commit (repository.fetch) and /work starts as the repository checked out
at it, its .git folder included, with the task's files laid over it.

A corpus keeps each of its tasks as one line of its tasks.jsonl,
{"depot", "id", "content"} (wpt_depot.corpus); from_line reads one.
"""

import base64
import json
import posixpath
import re
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from wpt_recipes import ASK, BUILT_IN_TOOLS, RecipeError, Recipes

from . import repository as repositories

# the containers' sizes when a task asks for no more: MiB of memory (the files in /work and /tmp included),
# MiB that /work and /tmp may each hold, processes and threads at once, CPUs' worth of time
SIZES = {"memory": 4096, "disk": 1024, "processes": 256, "cpus": 2.0}
HEADROOM = 1024  # MiB of memory left to the processes when /work and /tmp are both full
LOOKING = ("read", "bash")  # the built-in tools that look at sources; a declared tool looks when marked `looks`
OWN = ("/work", "/tmp")  # the attempt's own folders, kept in memory, which `absent` may not name


class InvalidTask(ValueError):
    """A task that breaks a rule across its fields, or whose image or pool
    of tools cannot be read; the message names the task."""


class Variant(NamedTuple):
    """One way to meet a need: tools offered together, and the paths under
    files/ that go to the tools' programs in an attempt offered them."""
    tools: tuple[str, ...]
    tool_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    id: str  # "<depot address> <the item's id there>"
    instructions: str
    image: str  # a folder name in data/images/
    max_turns: int
    max_tokens: int
    command_seconds: float
    files: dict[str, tuple[str, bytes]]  # path under /work -> (mode, content; a link's is its target)
    check: bytes
    container: dict = field(default_factory=lambda: dict(SIZES))  # SIZES, raised by the task's `container`
    needs: tuple = ()  # each need's variants, as tuples of Variants
    replies: tuple = ()  # (pattern, reply) of each reply to the ask tool, in order
    tools: tuple = ()  # the task's own tool declarations (name, description, parameters, program, names)
    tool_files: dict = field(default_factory=dict)  # the files only the tools' programs read, as `files`
    answered_by: tuple = ()  # the looking tools that can answer what the task leaves open (wpt_check.asked_last)
    answered_in: tuple = ()  # the patterns of the paths under /work where its answers live (wpt_check.asked_last)
    absent: tuple = ()  # the image's folders the attempts see empty
    terse: str = ""  # the same request as instructions, in one line; "" when the task has none
    area: tuple | None = None  # the patterns of the paths under /work an attempt may change; None: no area
    gates: tuple = ()  # the parts of the check, or the harness's, that must be 1 for the attempt to score
    repository: tuple | None = None  # (address, commit) of the Git repository /work starts as; None: none


def from_line(line: str, recipes: Recipes | None = None, pools: dict | None = None) -> Task:
    """The Task of one line of a corpus's tasks.jsonl."""
    found = json.loads(line)
    return from_item(f"{found['depot']} {found['id']}", found["content"], recipes, pools)


def from_item(task_id: str, content: dict, recipes: Recipes | None = None, pools: dict | None = None) -> Task:
    """The Task of an item's content, named task_id; its image and pool of
    tools read from recipes (default: the repository holding the current
    folder), each pool once into pools when given. InvalidTask names a
    rule it breaks."""
    document = content["task"]
    where = f"task {task_id}"
    try:
        recipes = recipes or Recipes.find()
        recipes.check_image(document["image"])
        pools = {} if pools is None else pools
        if document["image"] not in pools:
            pools[document["image"]] = recipes.image_tools(document["image"])
    except RecipeError as error:
        raise InvalidTask(f"{where}: {error}") from None
    container = {name: max(size, document.get("container", {}).get(name, 0)) for name, size in SIZES.items()}
    if 2 * container["disk"] + HEADROOM >= container["memory"]:
        raise InvalidTask(f"{where}: /work and /tmp may hold {container['disk']} MiB each, which with {HEADROOM} MiB "
                          f"for the processes needs more than {container['memory']} MiB of memory; raise "
                          f"container.memory above {2 * container['disk'] + HEADROOM}")
    terse = document.get("terse", "")
    if terse and terse.splitlines() != [terse]:
        raise InvalidTask(f"{where}: terse is the request in one line, and it holds a line break")
    absent = tuple(document.get("absent", []))
    for place in absent:
        if posixpath.normpath(place) != place or any(place == own or place.startswith(own + "/") for own in OWN):
            raise InvalidTask(f"{where}: absent names {place!r}; it names a folder of the image, written without . "
                              f"or .., outside {' and '.join(OWN)}")
    files = {path: (entry["mode"], base64.b64decode(entry["content"])) for path, entry in sorted(content["files"].items())}
    replies = tuple((reply["pattern"], reply["reply"]) for reply in document.get("replies", []))
    tools = tuple(document.get("tools", []))
    needs = tuple(tuple(Variant((variant,)) if isinstance(variant, str) else
                        Variant(tuple(variant["tools"]), tuple(variant.get("tool_files", [])))
                        for variant in variants) for variants in document["needs"])
    answered_by, answered_in = tuple(document.get("answered_by", [])), tuple(document.get("answered_in", []))
    hidden = _check_tools(where, needs, replies, tools, document.get("tool_files", []), list(files), answered_by,
                          pools[document["image"]])
    limits, repository = document["limits"], document.get("repository")
    return Task(task_id, document["instructions"], document["image"], limits["max_turns"], limits["max_tokens"],
                limits["command_seconds"], {path: entry for path, entry in files.items() if path not in hidden},
                base64.b64decode(content["check"]["content"]), container, needs, replies, tools,
                {path: files[path] for path in sorted(hidden)}, answered_by, answered_in, absent, terse,
                tuple(document["area"]) if "area" in document else None, tuple(document.get("gates", [])),
                (repository["address"], repository["commit"]) if repository else None)


def prepared(task: Task) -> Task:
    """The task as an attempt starts it: with a repository, its files laid
    over the repository checked out at its commit (repository.fetch, once
    per process for each commit), its .git folder included, and the
    repository named no more; without one, the task as it is."""
    if task.repository is None:
        return task
    return replace(task, files={**repositories.fetch(*task.repository), **task.files}, repository=None)


def looks(tool: dict) -> bool:
    """Whether a tool, by its declaration, looks at sources (wpt_check's
    looking rule): read and bash (LOOKING), or a declared tool marked `looks`."""
    return tool["name"] in LOOKING or tool.get("looks") is True


def _check_tools(where, needs, replies, tools, tool_files, files, answered_by, pool) -> set[str]:
    """The paths of files that tool_files hides; InvalidTask when the
    task's tools, needs, answered_by, replies or tool_files (a variant's
    among them) break the rules above."""
    declared = [tool["name"] for tool in tools]
    for name in declared:
        if name in BUILT_IN_TOOLS or name == ASK or declared.count(name) > 1:
            raise InvalidTask(f"{where}: the tool {name!r} is declared twice, or takes the name of a tool every "
                              f"task has ({', '.join(BUILT_IN_TOOLS)}, {ASK})")
    offered = {*BUILT_IN_TOOLS, *declared, *([ASK] if replies else [])}
    looking = {*LOOKING, *(tool["name"] for tool in [*tools, *(t for t in pool if t["name"] not in declared)]
                           if looks(tool))}
    named = [{name for variant in variants for name in variant.tools} for variants in needs]
    for number, names in enumerate(named):
        for name in sorted(names):
            if name not in offered:
                raise InvalidTask(f"{where}: needs {name!r}, which the task does not offer"
                                  + (" (ask is offered only with replies)" if name == ASK else ""))
            if any(name in other for other in named[number + 1:]):
                raise InvalidTask(f"{where}: {name!r} is in two needs; a tool is the variant of one need only")
    if replies and not answered_by:
        raise InvalidTask(f"{where}: a task with replies names in answered_by the looking tools that can answer "
                          "before the user is asked")
    for name in answered_by:
        if name not in looking:
            raise InvalidTask(f"{where}: answered_by names {name!r}, which is no looking tool the task or its "
                              "image's pool offers (read, bash, or a declared tool marked looks)")
    for pattern, _ in replies:
        try:
            re.compile(pattern)
        except re.error as error:
            raise InvalidTask(f"{where}: the reply pattern {pattern!r} is not a regular expression: {error}") from None
    for variants in needs:
        for variant in variants:
            _hidden(where, variant.tool_files, files)
    return _hidden(where, tool_files, files)


def _hidden(where, tool_files, files) -> set[str]:
    """The paths of files that the tool_files entries name; InvalidTask
    for an entry that names none."""
    hidden = set()
    for entry in tool_files:
        found = {path for path in files if path == entry or path.startswith(entry + "/")}
        if not found:
            raise InvalidTask(f"{where}: tool_files names {entry!r}, which is no file or folder under files/")
        hidden |= found
    return hidden
