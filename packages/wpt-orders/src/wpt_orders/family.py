"""Model families: one file, the sizes of one model (docs/model.md,
"Model families").

A family file (data/orders/<name>.json, read through the service layer)
writes once what its sizes share, as the parts of a model order (data,
model, seed, steps, drafter, note) and the other runs every size has
(branches, learning_rate_check); then `sizes`, a table. Its `columns`
are "size" then the path of each setting that varies between sizes; its
`rows` are one per size: the size's name, then a value per column. A path
names a setting from the top of a model order, its names joined by '.':
data.total_tokens, model.d_model, steps.1.optimizer.settings.lr, the steps
numbered from 1.

A setting is written in one place: common, or a column, never both, and a
column every size fills. `check` refuses anything else with a sentence that
says what is wrong. Nothing computes a setting from another: a size's
values are written.

`resolve` makes a size's model order: the common parts with the size's
values in place. Its data part's total_tokens is the tokens the size reads.
Sizes whose data parts are the same but for total_tokens share one corpus,
of the most tokens any of them reads: the resolved data part asks for that
corpus, and every step that lists text windows' `stages` reads the size's
own tokens from its start (prefix_tokens, docs/training.md, "Prefix").
Sizes of other data parts, another vocabulary among them, read corpora of
their own.

A size has other runs besides its own, each named FAMILY:SIZE:RUN (`runs`,
`run_order`), when its order is one step whose settings list stages:

    branch-SHARE  a decay branch, one per share of the family's `branches`
                  (above 0, at most 1): SHARE times the size's updates.
                  Every branch of a size goes on from one trunk, the size's
                  step at its peak rate after the warm-up, with no drop; the
                  trunk is cut into consecutive steps that end at each
                  branch's start, the last update before its first drop
                  (last_update), so a branch's order is the trunk's steps up
                  to its start, each continuing the one before it, then its
                  decay, which continues the last of them to the branch's
                  updates. Two branches share the trunk's steps they both
                  hold, and wpt train reuses their kept results: the trunk
                  trains once.
    lr-FACTOR     at the smallest size (the fewest parameters) only, one per
                  factor of the family's `learning_rate_check`: the size's
                  own run at FACTOR times the learning rate it writes.

The updates of a step come from its environment (wpt_training), which
this package does not hold: run_order takes them from its caller.
"""

import copy
import json
import math
import re
from fractions import Fraction

from . import counts

SIZE = "size"                 # the first column: the size's name
NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
TOP = ("data", "model", "seed", "steps", "drafter")  # where a path starts: the parts of a model order
OWN = ("note",)               # a part of the order that stays common
RUNS = ("branches", "learning_rate_check")
DERIVED = "prefix_tokens"     # a step's setting resolve gives: the size's total_tokens
FIRST_DROP = 0.8              # by-part's first_drop (wpt-training's tests check): a branch's decay starts after it
BRANCH, LEARNING_RATE = "branch-", "lr-"  # the names of a size's other runs, FAMILY:SIZE:RUN


def _exact(number) -> Fraction:
    """A JSON number as the decimal it is written as (0.04 is 1/25)."""
    return Fraction(str(number))


def _half_up(value) -> int:
    """value rounded to the nearest whole number, halves up, computed exactly."""
    return math.floor(Fraction(value) + Fraction(1, 2))


def parse(path: str) -> tuple:
    """The names of a column's path, a step's number as an int; ValueError
    for a path that names no setting of a model order."""
    names = path.split(".")
    if any(not name for name in names) or names[0] not in TOP:
        raise ValueError(f"column {path!r}: a path starts with one of {', '.join(TOP)} and joins names with '.'")
    if names[0] == "steps":
        if len(names) < 3 or not names[1].isdigit() or int(names[1]) < 1:
            raise ValueError(f"column {path!r}: a step's setting is steps.NUMBER.NAME, the steps numbered from 1")
        return ("steps", int(names[1]) - 1, *names[2:])
    if names[0] == "seed" and len(names) > 1:
        raise ValueError(f"column {path!r}: the seed is one number")
    return tuple(names)


def _shown(path: tuple) -> str:
    """A parsed path as the family writes it."""
    return ".".join(str(name + 1) if isinstance(name, int) else name for name in path)


def sizes(family: dict) -> list[str]:
    """The family's size names, in its order."""
    return [row[0] for row in family["sizes"]["rows"]]


def _columns(family: dict) -> list[tuple]:
    return [parse(path) for path in family["sizes"]["columns"][1:]]


def _written(common, path: tuple) -> bool:
    """Whether the common parts write the setting at path, a part of it, or
    a value that would hold it."""
    node = common
    for name in path:
        if isinstance(node, list) and isinstance(name, int):
            if name >= len(node):
                return False
            node = node[name]
        elif isinstance(node, dict) and not isinstance(name, int):
            if name not in node:
                return False
            node = node[name]
        else:
            return True  # a value where the path goes on: the common parts write it whole
    return True


def _common(family: dict) -> dict:
    """The parts of a model order the family writes once."""
    return {key: value for key, value in family.items() if key in TOP or key in OWN}


def check(family: dict):
    """ValueError, with a sentence that says what is wrong, for a table
    that cannot be: a first column other than "size", a column named
    twice, or inside another, a path that names no setting, a setting
    written both as common and as a column, a step's prefix_tokens written
    at all, a size named twice or by a name that is not one, or a row
    with another number of values than the columns, or with an empty one
    (null); and, with branches, a step listing stages whose mtp_switch or
    by-part first_drop is not FIRST_DROP, so that a size's own run is the
    longest branch."""
    columns = family["sizes"]["columns"]
    if not columns or columns[0] != SIZE:
        raise ValueError(f"sizes.columns: the first column is {SIZE!r}, each row's name; the others name the "
                         "settings that vary between sizes")
    paths = _columns(family)
    if len(set(paths)) != len(paths):
        repeated = sorted({_shown(path) for path in paths if paths.count(path) > 1})
        raise ValueError(f"sizes.columns names {', '.join(repeated)} more than once: one column per setting")
    for path in paths:
        for other in paths:
            if other != path and other[:len(path)] == path:
                raise ValueError(f"the column {_shown(other)} is a part of the column {_shown(path)}: write the "
                                 "setting in one column")
    common = _common(family)
    for path in paths:
        if path[0] == "steps" and path[1] >= len(common.get("steps", [])):
            raise ValueError(f"the column {_shown(path)} names step {path[1] + 1}, and the family writes "
                             f"{len(common.get('steps', []))} steps")
        if _written(common, path):
            raise ValueError(f"{_shown(path)} is written both as a common setting and as a column: write it in one "
                             "place, the column when it varies between sizes, else common")
        if path[-1] == DERIVED:
            raise ValueError(f"the column {_shown(path)}: a size reads its data.total_tokens, which sets each step's "
                             f"{DERIVED}; write data.total_tokens instead")
    for number, entry in enumerate(common.get("steps", []), 1):
        if DERIVED in (entry.get("settings") or {}):
            raise ValueError(f"steps.{number}.settings.{DERIVED}: a size reads its data.total_tokens, which sets "
                             f"each step's {DERIVED}; write data.total_tokens instead")
    names = sizes(family)
    for row in family["sizes"]["rows"]:
        name = row[0] if row else None
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError(f"size {name!r}: a size's name, the first value of its row, is lowercase a-z, 0-9, "
                             "'.', '_' and '-'")
        if len(row) != len(columns):
            raise ValueError(f"size {name}: its row has {len(row) - 1} values for {len(columns) - 1} columns: "
                             "every size fills every column")
        for path, value in zip(paths, row[1:]):
            if value is None:
                raise ValueError(f"size {name} leaves the column {_shown(path)} empty: every size fills every "
                                 "column, and a setting the same at every size is written once, common")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"the sizes {', '.join(repeated)} are named more than once")
    for what in RUNS:
        found = family.get(what, [])
        if len({_exact(value) for value in found}) != len(found):
            raise ValueError(f"{what}: a value repeats")
    if family.get("branches"):
        for name in names:
            for number, entry in enumerate(resolve(family, name)["steps"], 1):
                settings = entry.get("settings") or {}
                drop = ((entry.get("optimizer") or {}).get("settings") or {}).get("first_drop", FIRST_DROP)
                if "stages" in settings and (_exact(settings.get("mtp_switch", 0)) != _exact(FIRST_DROP)
                                             or _exact(drop) != _exact(FIRST_DROP)):
                    raise ValueError(
                        f"size {name}, step {number}: a branch decays after {FIRST_DROP} of its updates, where "
                        f"by-part's first drop is, and its trunk keeps the multi-token prediction's early weight to "
                        f"there: a family with branches writes mtp_switch {FIRST_DROP} and leaves first_drop alone")


def _place(order: dict, path: tuple, value):
    """Put value at path in order, making the objects on its way."""
    node = order
    for name in path[:-1]:
        if isinstance(name, int):
            node = node[name]
        else:
            node = node.setdefault(name, {})
    node[path[-1]] = copy.deepcopy(value)


def _row(family: dict, size: str) -> list:
    for row in family["sizes"]["rows"]:
        if row[0] == size:
            return row
    raise KeyError(size)


def _own(family: dict, size: str) -> dict:
    """The size's order before its data part names a shared corpus: the
    common parts with its values in place."""
    order = copy.deepcopy(_common(family))
    for path, value in zip(_columns(family), _row(family, size)[1:]):
        _place(order, path, value)
    return order


def _corpus_key(data: dict) -> str:
    """What decides a corpus but its size: the data part without total_tokens."""
    return json.dumps({key: value for key, value in data.items() if key != "total_tokens"}, sort_keys=True)


def resolve(family: dict, size: str) -> dict:
    """The model order of that size (the module's docstring): its data part
    asks for the corpus it shares with the sizes of the same data part,
    and each of its steps that lists stages reads its own total_tokens.
    KeyError for a size the family does not have. The order is not
    checked against the model order's schema here: Orders does that."""
    order = _own(family, size)
    data = order.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("total_tokens"), int):
        return order  # the schema says what is missing
    own = data["total_tokens"]
    shared = [_own(family, other)["data"] for other in sizes(family)]
    data["total_tokens"] = max(found["total_tokens"] for found in shared
                               if _corpus_key(found) == _corpus_key(data) and isinstance(found["total_tokens"], int))
    for entry in order.get("steps", []):
        settings = entry.get("settings")
        if isinstance(settings, dict) and "stages" in settings:
            settings[DERIVED] = own
    return order


def smallest(family: dict) -> str:
    """The size with the fewest parameters without the vocabulary tables,
    where the learning-rate check runs."""
    return min(sizes(family), key=lambda size: counts.parameters(_own(family, size)["model"], 0)[0])


def runs(family: dict, size: str) -> list[str]:
    """The names of the size's other runs (FAMILY:SIZE:RUN), in order: its
    branches, then, at the smallest size, the learning-rate check's."""
    found = [f"{BRANCH}{share}" for share in family.get("branches", [])]
    if family.get("learning_rate_check") and size == smallest(family):
        found += [f"{LEARNING_RATE}{factor}" for factor in family["learning_rate_check"]]
    return found


def _one_step(order: dict) -> dict:
    """The order's one step that lists stages; ValueError otherwise."""
    steps = order["steps"]
    if len(steps) != 1 or "stages" not in (steps[0].get("settings") or {}):
        raise ValueError("a size's other runs need an order of one step whose settings list stages (read-text)")
    return steps[0]


def branch_numbers(family: dict, updates: int) -> list[dict]:
    """The decay branches of a size whose own run makes `updates` updates,
    the shortest first: each its run's name, its updates in all (the
    trunk's it goes on from included) and start, the trunk's update it goes
    on from, the last before its first drop. ValueError when two have the
    same updates or one has no update before its drop."""
    made = []
    for share in family.get("branches", []):
        total = max(1, _half_up(_exact(share) * updates))
        # the last update by-part runs at the peak: it drops after first_drop x updates, compared as floats
        made.append({"run": f"{BRANCH}{share}", "updates": total, "start": math.floor(FIRST_DROP * total)})
    made.sort(key=lambda branch: branch["updates"])
    if len({branch["updates"] for branch in made}) != len(made):
        raise ValueError(f"two branches have the same number of updates of the size's {updates:,}")
    for branch in made:
        if branch["start"] < 1:
            raise ValueError(f"{branch['run']} has {branch['updates']:,} updates, too few to decay after one")
    return made


def run_order(family: dict, size: str, run: str, updates=None) -> dict:
    """The model order of a run of the size: its own ('') or one of
    `runs`. updates(entry): the updates a step entry of a model order
    makes, which a branch needs. ValueError for a run the size does not
    have or cannot make."""
    order = resolve(family, size)
    if not run:
        return order
    listed = runs(family, size)
    if run not in listed:
        raise ValueError(f"no run {run!r} of the size {size}; "
                         + (f"its runs are {', '.join(listed)}" if listed else "it has no other run"))
    entry = _one_step(order)
    optimizer = entry.setdefault("optimizer", {"name": "by-part"}).setdefault("settings", {})
    if run.startswith(LEARNING_RATE):
        if "lr" not in optimizer:
            raise ValueError("the learning-rate check multiplies the learning rate the family writes, "
                             "steps.1.optimizer.settings.lr, and it writes none")
        factor = next(f for f in family["learning_rate_check"] if f"{LEARNING_RATE}{f}" == run)
        optimizer["lr"] = float(f"{optimizer['lr'] * factor:.4g}")
        return order
    if updates is None:
        raise ValueError(f"{run} is cut from the size's updates, and no count of them was given")
    made = branch_numbers(family, updates(entry))
    mine = next(branch for branch in made if branch["run"] == run)
    ends = sorted({branch["start"] for branch in made if branch["start"] <= mine["start"]})
    steps = []
    for number, end in enumerate(ends):
        piece = copy.deepcopy(entry)
        piece["settings"].update(last_update=end, mtp_switch=1.0)  # the trunk: the early weight to its end
        piece["optimizer"]["settings"].update(first_drop=1.0, second_drop=1.0)  # never within the step
        if number:
            piece["continues"] = True
        steps.append(piece)
    decay = copy.deepcopy(entry)
    decay["settings"]["last_update"] = mine["updates"]
    decay["continues"] = True
    steps.append(decay)
    return {**order, "steps": steps}


def numbers(model: dict, vocab_size: int, block_size: int) -> dict:
    """A size's counts (counts.py): its layers, parameters without and with
    the vocabulary tables, the parameters one token uses and the compute
    of training on one token in windows of block_size."""
    without, every = counts.parameters(model, vocab_size)
    return {"layers": counts.layer_count(model), "parameters": without, "total_parameters": every,
            "active_parameters": counts.active_parameters(model),
            "compute_per_token": counts.compute_per_token(model, vocab_size, block_size)}
