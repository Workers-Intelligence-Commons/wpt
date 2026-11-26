"""The roles a training step is made of, and how their files are found.

    Strategy     how the hardware is used (strategies/)
    Method       how a loss is made from a batch of examples (methods/)
    Optimizer    how the weights the loss reaches move, with its schedule (optimizers/)
    Environment  where the examples come from, batch by batch (environments/)
    a step       what is done to the model: a file (steps/) that chooses a method, an
                 environment, the kinds of texts it reads, an optimizer with its
                 settings, the parts it locks, and its settings' defaults; load_step
                 reads an order's entry into a Step

Each is a Python file, a built-in by its name or any path:

    strategy file:     def start() -> Strategy
    method file:       READS = "<kind of examples>"; SETTINGS = {...}
                       def start(settings) -> Method
    optimizer file:    SETTINGS = {...}; def start(model, settings, run) -> Optimizer;
                       optional check(settings)
    environment file:  GIVES = "<kind of examples>"; KINDS = ("<kind of texts>", ...);
                       SETTINGS = {...}
                       def start(settings, data) -> Environment, on the gpu worker; optional
                       check(settings), context(settings): the window length it trains at,
                       which a model's context comes from, check_corpus(settings, record,
                       dock, kinds): ValueError when the corpus holds nothing for it, before
                       any step runs, and manager(settings, context) -> its part on the wpt
                       manager (below)
    step file:         METHOD, ENVIRONMENT, KINDS = ("<kind of texts>", ...),
                       OPTIMIZER = {"name": ..., "settings": {...}}, LOCKED = (...),
                       SETTINGS = {...} (defaults of its method's and environment's
                       settings); optional check(settings)

A method reads one kind of examples and an environment gives one; a step
whose method reads what its environment does not give is refused, and so
is any optimizer whose start refuses the model. A step names the kinds of
texts it reads (depot.kinds: the format of a text, such as plain,
conversation or task), each one its environment's KINDS can read, and
its environment reads the corpus's texts of those kinds alone (data.kinds,
context.kinds, check_corpus's kinds); a step that names none, or one its
environment cannot read, is refused. A setting belongs to the
method or to the environment by its name, never both.

A step trains on a gpu worker (wpt_worker; docs/machines.md), which has no
tokenizer, no Docker and no dock of its own: everything it reads comes
from the wpt manager. An environment that needs what only the manager
holds (the tokenizer, the sandboxes) has a part there: manager(settings,
context), given a Context (the corpus's files, the corpus's record, the tokenizer,
the run's record, and writer(...), which has the model on the gpu worker
write tokens), returns an object whose methods the environment's part on
the worker calls through data.call(name, *arguments). Arguments and
results are what JSON holds (a tuple arrives as a list); a StepFailed the
manager's part raises arrives as StepFailed. The manager makes the part
when the step's first call reaches it and drops it when the step ends.
data.writing(model, generator), on the worker, is where that model writes
meanwhile, drawing from that generator; it gives back each row's routes.
"""

import contextlib
import hashlib
import importlib.util
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from wic_utils.files import sha256_file
from wpt_utils.settings import typed

HERE = Path(__file__).parent
KINDS = ("strategies", "methods", "optimizers", "environments", "steps")
WINDOWS = "windows"  # the kind of examples of text windows: a step that reads them sets the model's context


class PluginError(Exception):
    """A strategy, method, optimizer, environment or step cannot be used:
    unknown name, not a Python file, missing what its kind defines, start()
    returned the wrong type, an unknown or mistyped setting, settings it
    refuses, a method that does not read what the environment gives, an
    unknown part of the model to lock, or a strategy that cannot start
    here (ddp in one process). Nothing has been trained."""


class StepFailed(Exception):
    """A step ran and could not do its part (an example it cannot learn
    from, a loss that is not finite, every attempt of a round dropped...):
    .reason says why. The step's result is not kept, a checkpoint it wrote
    stays, and the steps before it stay kept."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Strategy:
    """One process training alone on one device. A strategy for several
    processes overrides what differs. With splits_windows, every process
    takes the same windows and trains its own slice of each (wpt_training's
    parts.py); its synchronize then adds the processes' gradients up."""

    splits_windows = False

    def __init__(self, device, rank=0, world_size=1, local_rank=0):
        self.device, self.rank, self.world_size, self.local_rank = device, rank, world_size, local_rank

    @property
    def is_main(self) -> bool:
        """The one process that logs and writes the model folder."""
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        """The one process per machine that prepares local files."""
        return self.local_rank == 0

    def wrap(self, model):
        """The model as the training steps call it."""
        return model

    def no_sync(self, wrapped):
        """A context for a micro-batch whose gradients need not be shared yet."""
        return contextlib.nullcontext()

    def synchronize(self, model):
        """Called once a step's gradients are complete in this process, before
        the update: share them, when wrap does not."""

    def mean(self, value: float) -> float:
        """The average of value over every process."""
        return value

    def reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """The elementwise sum (op "sum") or maximum ("max") of tensor over
        every process, in every process."""
        return tensor

    def broadcast(self, value):
        """The main process's value, in every process (value is ignored
        elsewhere). It must survive pickle."""
        return value

    def barrier(self):
        """Wait until every process reaches this point."""

    def close(self):
        """Release what start() took."""


@dataclass(frozen=True)
class Run:
    """What an optimizer may know about the step."""
    tokens: int      # the tokens of the examples the step reads (text: training plus validation tokens)
    parameters: int  # the parameters the step trains
    steps: int       # the maximum number of updates
    log: callable    # a no-op except in the main process


@dataclass(frozen=True)
class Verdict:
    best: bool       # keep these weights as the best so far
    stop: bool       # stop training now
    note: str = ""   # added to the evaluation's log line


class Method:
    """How a loss is made from a batch of examples. learn runs one pass's
    forward and backward passes; the runner adds the passes of an update
    up. unreached: the model parts (wpt_model.PARTS) this loss never gives
    a gradient, which the step leaves out of its optimizer and of the
    processes' gradient exchange."""

    unreached: tuple = ()

    def begin(self, model, environment, run, segment, strategy):
        """Called once before the step's first update, the step's Run known:
        ValueError or PluginError when the method cannot train this
        environment's batches as the options ask (windows in parts, split
        windows)."""
        if strategy.splits_windows:
            raise PluginError("only a step that reads text windows splits them across processes")

    def learn(self, model, wrapped, examples, passes, strategy, segment, quiet):
        """One pass on examples, one of the update's `passes` passes in this
        process: the loss's backward pass, under quiet(last) for its last
        backward call (the gradients are exchanged at the update's last).
        Returns its share of the update's loss, a float or a detached
        tensor."""
        raise NotImplementedError

    def update_loss(self, losses, strategy) -> float:
        """The update's loss from this process's passes' values, the same in
        every process: by default, the shares of every pass of every
        process added up."""
        total = strategy.reduce(torch.tensor([sum(float(value) for value in losses)], dtype=torch.float64,
                                             device=strategy.device))
        return total.item()

    def after_update(self, step: int):
        """Called after the optimizer steps, with the update's number."""

    def state_dict(self) -> dict:
        """Everything a resumed step needs: tensors, numbers, strings, lists, dicts."""
        return {}

    def load_state_dict(self, state: dict):
        pass


class TextMethod(Method):
    """A method on text windows, x and y the tokens after them: by default
    the next-token loss plus the multi-token prediction loss at
    mtp_weight. Windows
    longer than the options' segment, or split across processes, train in
    parts (parts.py), which need part_loss."""

    mtp_weight = 0.3  # DeepSeek-V3's first weight; a method may change it as training goes

    def begin(self, model, environment, run, segment, strategy):
        from . import parts
        parts.check(model, self, environment.stage_list, segment, strategy)

    def learn(self, model, wrapped, examples, passes, strategy, segment, quiet):
        from . import parts
        x, y = examples
        return parts.learn(self, model, wrapped, x, y, 1 / passes, strategy, segment, quiet)

    def update_loss(self, losses, strategy):
        """The loss over the whole batch, every micro-batch of every process
        (the sum of their shares with split windows)."""
        whole = torch.stack([torch.as_tensor(value).float() for value in losses]).mean().item()
        return strategy.mean(whole) * (strategy.world_size if strategy.splits_windows else 1)

    def loss(self, model, x, y):
        """The training loss of a batch, windows x and y, the tokens after
        them: what the backward pass starts from, the language-model loss
        that the log shows next to the validation loss."""
        return self.part_loss(model, x, y, None, x.shape[1])

    def part_loss(self, model, x, y, following, window):
        """The share of a part of the windows, x and y (B, t), in loss()
        over the whole windows, so that the parts' shares add up to it
        (parts.py): model computes the part where it sits in the windows;
        following is the token after y's last (B, 1), None at the windows'
        end; window is the windows' length. A method that defines its own
        loss defines this too, or trains whole windows only."""
        B, t = y.shape
        targets = y[:, 1:] if following is None else torch.cat([y[:, 1:], following], 1)  # the token after next
        # the model sums the two cross-entropies without holding every token's scores at once
        next_sum, ahead_sum = model(x, next_tokens=y, targets=(y, targets))
        return next_sum / (B * window) + self.mtp_weight * ahead_sum / (B * (window - 1))

    def validation_loss(self, model, x, y):
        """The next-token loss alone, which evaluations compare: the mean
        cross-entropy of the next-token scores, summed by the model a chunk
        of tokens at a time as in training, so that the scores of every
        token of the windows are never held at once; the multi-token
        prediction block does not run."""
        next_sum = model(x, targets=(y,))
        return next_sum / y.numel()


class Optimizer:
    """How a step's weights move: self.optimizer, a torch optimizer over
    the parameters the step trains, and what happens around each update.
    By default no schedule, every evaluation the best, and no stopping."""

    optimizer: torch.optim.Optimizer

    def before_update(self, step: int):
        """Called after the gradients of an update are complete, before the optimizer steps."""

    def after_update(self, step: int, reduce):
        """Called after the optimizer steps. reduce(tensor, op) sums ("sum")
        or takes the maximum ("max") of a tensor over every process."""

    def stage(self, number: int, block_size: int):
        """Called when a stage after the first starts (number from 1): the
        windows are now block_size tokens long, and the evaluations that
        follow are not comparable with earlier ones. Called too when a step
        that continues the one before it starts, with the stage it starts
        in: the evaluations before it are not its own."""

    def evaluated(self, step: int, val_loss: float) -> Verdict:
        return Verdict(best=True, stop=False)

    def kept(self) -> dict | None:
        """Called once the step's updates are done: the weights the step
        keeps (a whole state_dict of the model) in place of its best
        evaluation's or its last ones, such as an average of its last
        states; None keeps those."""
        return None

    def state_dict(self) -> dict:
        """Everything a resumed step needs: tensors, numbers, strings, lists, dicts."""
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict):
        self.optimizer.load_state_dict(state["optimizer"])


@dataclass(frozen=True)
class Stage:
    """Part of a step: `steps` updates of batch_size examples each, over
    every process together; block_size is the windows' length for text
    windows, None for other examples."""
    steps: int
    batch_size: int
    block_size: int | None = None


def step_seed(seed: int, number: int) -> int:
    """The order's seed mixed with the step's number: what the step's
    draws (conversations' order, tasks' order, tools offered, sampled
    tokens) are seeded by, so two identical steps of one order draw
    differently."""
    digest = hashlib.sha256(repr((seed, number)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


@dataclass
class Data:
    """What an environment's part on the gpu worker reads its examples
    from: the step's name, the corpus's files (dock: a reader of
    wpt_corpus.files, the manager's dock as the worker reads it,
    wpt_worker.job.LinkDock), the corpus's record, the model's
    configuration, the strategy, the run's telemetry (its record), log (a
    no-op except in the main process), the order's seed, the step's number
    in the order (from 1), which the step's draws read with the seed
    (step_seed) so that two steps of one order draw other windows,
    conversations and tasks, the options that change how the examples are
    read (stream_shards: with --stream, the shards in use at once;
    micro_batch: the examples per pass in one process, 0 for its whole
    share), call(name, *arguments), which calls a method of the
    environment's part on the wpt manager and gives its result,
    writing(model, generator), a context in which the manager's writers
    write with that model (module docstring), and the kinds of texts the
    step reads (Step.kinds, which Step.start_environment sets)."""
    step: str
    dock: object
    record: dict
    config: object
    strategy: Strategy
    telemetry: object
    log: callable
    seed: int
    number: int
    stream_shards: int | None = None
    micro_batch: int = 0
    call: callable = None
    writing: callable = None
    kinds: tuple = ()

    @property
    def step_seed(self) -> int:
        return step_seed(self.seed, self.number)


@dataclass
class Context:
    """What an environment's part on the wpt manager reads: the step's
    name, the corpus's files (dock: a reader of wpt_corpus.files,
    wpt_corpus.files.DockFiles on the manager), the corpus's record, the model's tokenizer and
    configuration, the run's telemetry (its record), log, the order's seed
    and the step's number, and writer(prompt, rows, *, temperature, top_k,
    stops, routes), a writer (wpt_inference.Writer's write and routes) whose
    tokens the model writes on the gpu worker, drawn from the generator the
    worker's part gave data.writing; its routes are each row's handle,
    which the worker's part turns back into the row's routes; and the kinds
    of texts the step reads (Step.kinds, which Step.start_manager sets)."""
    step: str
    dock: object
    record: dict
    tokenizer: object
    config: object
    telemetry: object
    log: callable
    seed: int
    number: int
    writer: callable = None
    kinds: tuple = ()

    @property
    def step_seed(self) -> int:
        return step_seed(self.seed, self.number)


def in_process(part) -> callable:
    """data.call for an environment whose manager's part runs in this
    process, as through the link: the arguments and the result go through
    JSON."""
    def call(name, *arguments):
        found = getattr(part, name)(*json.loads(json.dumps(arguments)))
        return json.loads(json.dumps(found))
    return call


class Environment:
    """Where a step's examples come from. stage_list: the step's stages
    (one for most environments; text windows have one per window length).
    tokens: the tokens of its examples. learned: what of the corpus it
    read, for the model's NOTICE, by kind of texts: {"plain": the tokens of
    the plain texts' token files it read from their start}, {"conversation":
    how many}, {"task": how many}. An environment whose stages have a
    window length may define per_pass(index, examples): stage index (from
    0) then takes that many examples per pass in each process, which the
    runner chooses for a stage whose whole share does not fit the device's
    memory (memory.py); without it, the runner leaves its passes as they
    are."""

    stage_list: tuple = ()
    tokens: int = 0
    learned: dict = {}

    def begin(self, model, strategy):
        """Called once before the step's first update, with the weights the
        step starts from, after a resumed step's state is loaded and before
        stage_list is read; may raise StepFailed."""

    def stage(self, index: int):
        """Called when stage index (from 0) starts or a resumed step continues inside it."""

    def batch(self, step, model, strategy) -> list | None:
        """This process's passes of update `step` (from 1): one list entry of
        examples per pass, as many in every process; None when the update
        has nothing to learn (it is then skipped). The model may write the
        examples (attempts) in eval mode; it is in training mode after."""
        raise NotImplementedError

    def after_update(self, step, loss):
        """Called after update `step`, with its loss (None when it was
        skipped); may raise StepFailed."""

    def finished(self, last):
        """Called once the step's last update `last` is done; may raise StepFailed."""

    def evaluates(self, step, first, last) -> bool:
        """Whether update `step` of the stage running from update `first` to
        `last` is followed by an evaluation."""
        return False

    def evaluate(self, step, method, model, wrapped, strategy, segment) -> float:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict):
        pass

    def close(self):
        """Release what start() took (files on disk)."""


def _cross_entropy(logits, targets, reduction="mean"):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)


@dataclass(frozen=True)
class Plugin:
    name: str
    file: str     # a built-in by its package path, another file by its name only
    sha256: str
    module: object
    given: str = None  # the path an order, a step's file or a command gave for it; None for a built-in

    def record(self) -> dict:
        return {"name": self.name, "file": self.file, "file_sha256": self.sha256}


# {a file an order or a command names: where this process reads it from}: the gpu worker's copies of
# the files outside the repository that the manager sent (wpt_worker); empty elsewhere
SENT = {}


def built_in(kind) -> list[str]:
    return sorted(p.stem for p in (HERE / kind).glob("*.py") if not p.name.startswith("_"))


def _load(kind, name_or_path, needs=("start",)) -> Plugin:
    folder = HERE / kind
    if name_or_path in built_in(kind):
        path, file = folder / f"{name_or_path}.py", f"wpt_training/{kind}/{name_or_path}.py"
    else:
        path = Path(SENT.get(name_or_path, name_or_path))
        if path.suffix != ".py" or not path.is_file():
            raise PluginError(f"{name_or_path}: not a built-in {kind[:-1].replace('ie', 'y')} "
                              f"({', '.join(built_in(kind))}) nor a Python file")
        file = path.name
    spec = importlib.util.spec_from_file_location(f"wpt_training_{kind}_{path.stem}".replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in needs:
        if not hasattr(module, name):
            raise PluginError(f"{path.name}: defines no {name}")
    return Plugin(path.stem, file, sha256_file(path), module, None if name_or_path in built_in(kind) else name_or_path)


def strategy_plugin(name_or_path) -> Plugin:
    """The strategy's file, read but not started (what the manager checks
    before its gpu worker starts it)."""
    return _load("strategies", name_or_path)


def load_strategy(name_or_path) -> tuple[Plugin, Strategy]:
    plugin = _load("strategies", name_or_path)
    strategy = plugin.module.start()
    if not isinstance(strategy, Strategy):
        raise PluginError(f"{plugin.file}: start() returned {type(strategy).__name__}, not wpt_training.Strategy")
    return plugin, strategy


def _typed(defaults, overrides, where):
    try:
        return typed(defaults, overrides, where)
    except ValueError as error:
        raise PluginError(str(error)) from None


@dataclass(frozen=True)
class Step:
    """An order's step, every file read and every setting typed: its file
    and name, its method, environment and optimizer (Plugins), its settings
    (all of them, then those of its method and of its environment), its
    optimizer's settings, the parts it locks, and the kinds of texts it
    reads (its file's KINDS). continues: the step goes
    on from the whole state the step before it left (its weights, the
    optimizer's, method's and environment's state, and its count of
    updates), as one step that stopped and was resumed, instead of from its
    weights alone (check_continuations, draw_numbers)."""
    name: str
    plugin: Plugin
    method: Plugin
    environment: Plugin
    optimizer: Plugin
    settings: dict
    method_settings: dict
    environment_settings: dict
    optimizer_settings: dict
    locked: tuple = field(default_factory=tuple)
    continues: bool = False
    kinds: tuple = ()

    @property
    def reads(self) -> str:
        """The kind of examples the step learns from: its method's READS,
        which its environment gives (WINDOWS for text windows)."""
        return self.method.module.READS

    def record(self) -> dict:
        """What the step is, as its fingerprint and provenance.json read it."""
        return {"step": self.plugin.record(), "method": self.method.record(),
                "environment": self.environment.record(), "reads": self.reads, "kinds": list(self.kinds),
                "optimizer": {**self.optimizer.record(), "settings": self.optimizer_settings},
                "settings": self.settings, "locked": list(self.locked), "continues": self.continues}

    def context(self) -> int | None:
        """The window length the step leaves the model reading at, from its
        environment file's context(settings); None when it has none."""
        return getattr(self.environment.module, "context", lambda settings: None)(self.environment_settings)

    def check_corpus(self, record: dict, dock):
        """ValueError when the corpus (its record, and its files, dock: a
        reader of wpt_corpus.files) holds nothing the step can learn from, from its environment
        file's check_corpus(settings, record, dock, kinds); nothing when it
        has none."""
        getattr(self.environment.module, "check_corpus", lambda settings, record, dock, kinds: None)(
            self.environment_settings, record, dock, self.kinds)

    def start_method(self) -> Method:
        method = self.method.module.start(self.method_settings)
        if not isinstance(method, Method):
            raise PluginError(f"{self.method.file}: start() returned {type(method).__name__}, not wpt_training.Method")
        return method

    def files(self) -> list[Plugin]:
        """The step's files that are not built-ins (its own, its method's,
        environment's and optimizer's), which the manager sends its gpu
        worker."""
        return [plugin for plugin in (self.plugin, self.method, self.environment, self.optimizer)
                if plugin.given is not None]

    def start_manager(self, context: Context):
        """The environment's part on the wpt manager (its file's
        manager(settings, context)), or None when it has none."""
        made = getattr(self.environment.module, "manager", None)
        return None if made is None else made(self.environment_settings, replace(context, kinds=self.kinds))

    def start_environment(self, data: Data) -> Environment:
        environment = self.environment.module.start(self.environment_settings, replace(data, kinds=self.kinds))
        if not isinstance(environment, Environment):
            raise PluginError(f"{self.environment.file}: start() returned {type(environment).__name__}, not "
                              "wpt_training.Environment")
        return environment

    def start_optimizer(self, model, run: Run) -> Optimizer:
        optimizer = self.optimizer.module.start(model, self.optimizer_settings, run)
        if not isinstance(optimizer, Optimizer):
            raise PluginError(f"{self.optimizer.file}: start() returned {type(optimizer).__name__}, not "
                              "wpt_training.Optimizer")
        return optimizer


def load_step(entry: dict, parts) -> Step:
    """The Step of an order's `steps` entry: {"step": NAME} with, optionally,
    "optimizer" ({"name", "settings"}), "settings" and "locked", each left
    out taking the step file's own (OPTIMIZER, SETTINGS, LOCKED). The kinds
    of texts it reads are its file's KINDS, each one its environment's
    KINDS names. The
    settings are the defaults of the step's method and environment, those
    the step file sets, then the entry's, each of its default's type
    (wpt_utils.settings.typed); the step's and its environment's
    check(settings), when they have one, then pass. An optimizer the entry
    names takes the step's settings for it when it is the step's own, else
    its own defaults, then the entry's, and its file's check(settings),
    when it has one, then passes. parts: the names the model gives
    its parts (wpt_model.PARTS); a locked name not among them is refused.
    PluginError says what cannot be."""
    plugin = _load("steps", entry["step"], needs=("METHOD", "ENVIRONMENT", "KINDS", "OPTIMIZER"))
    module = plugin.module
    method = _load("methods", module.METHOD, needs=("start", "READS"))
    environment = _load("environments", module.ENVIRONMENT, needs=("start", "GIVES", "KINDS"))
    if method.module.READS != environment.module.GIVES:
        raise PluginError(f"{plugin.file}: its method {method.name} reads {method.module.READS}, and its environment "
                          f"{environment.name} gives {environment.module.GIVES}")
    kinds = tuple(module.KINDS)
    if not kinds or len(set(kinds)) != len(kinds) or not all(isinstance(one, str) for one in kinds):
        raise PluginError(f"{plugin.file}: KINDS names the kinds of texts the step reads, at least one, each once")
    unread = [one for one in kinds if one not in environment.module.KINDS]
    if unread:
        raise PluginError(f"{plugin.file}: its environment {environment.name} reads texts of kind "
                          f"{', '.join(environment.module.KINDS)}, not {', '.join(unread)}")
    method_defaults = getattr(method.module, "SETTINGS", {})
    environment_defaults = getattr(environment.module, "SETTINGS", {})
    shared = sorted(set(method_defaults) & set(environment_defaults))
    if shared:
        raise PluginError(f"{method.file} and {environment.file} both have the setting {shared[0]!r}")
    defaults = _typed({**method_defaults, **environment_defaults}, getattr(module, "SETTINGS", {}), plugin.file)
    settings = _typed(defaults, entry.get("settings", {}), plugin.file)
    for owner in (module, environment.module):
        try:
            getattr(owner, "check", lambda settings: None)(settings)
        except ValueError as error:
            raise PluginError(f"{plugin.file}: {error}") from None
    own = module.OPTIMIZER
    chosen = entry.get("optimizer", {"name": own["name"]})
    optimizer = _load("optimizers", chosen["name"])
    base = own.get("settings", {}) if chosen["name"] == own["name"] else {}
    optimizer_defaults = _typed(getattr(optimizer.module, "SETTINGS", {}), base, optimizer.file)
    optimizer_settings = _typed(optimizer_defaults, chosen.get("settings", {}), optimizer.file)
    try:
        getattr(optimizer.module, "check", lambda settings: None)(optimizer_settings)
    except ValueError as error:
        raise PluginError(f"{optimizer.file}: {error}") from None
    locked = tuple(entry.get("locked", getattr(module, "LOCKED", ())))
    unknown = [name for name in locked if name not in parts]
    if unknown:
        raise PluginError(f"{plugin.file}: no part of the model is named {unknown[0]!r} (parts: {', '.join(parts)})")
    continues = entry.get("continues", False)
    if type(continues) is not bool:
        raise PluginError(f"{plugin.file}: continues {continues!r}: true or false")
    return Step(plugin.name, plugin, method, environment, optimizer, settings,
                {key: settings[key] for key in method_defaults}, {key: settings[key] for key in environment_defaults},
                optimizer_settings, locked, continues, kinds)


def _updates(step: Step) -> int | None:
    """The updates of a step whose settings list its stages (text windows), else None."""
    stages = step.environment_settings.get("stages")
    return sum(stage["steps"] for stage in stages) if isinstance(stages, list) else None


def window_tokens_left(steps: list[Step], number: int = 1, update: int = 0) -> tuple[int, list[str]]:
    """(the tokens of text windows the steps still read after update of
    step number (from 1), the names of the steps after that point whose
    settings list no window lengths, so nothing here counts them). A step
    that continues the one before it goes on past that step's updates."""
    left, unknown, done = 0, [], update
    for position, step in enumerate(steps[number - 1:], number):
        stages = step.environment_settings.get("stages")
        if not isinstance(stages, list):
            unknown.append(step.name)
            done = 0
            continue
        first = 1
        for stage in stages:
            updates = min(stage["steps"], max(first + stage["steps"] - 1 - done, 0))
            left += updates * stage["batch_size"] * stage["block_size"]
            first += stage["steps"]
        following = steps[position] if position < len(steps) else None
        done = first - 1 if following is not None and following.continues else 0
    return left, unknown


def check_continuations(steps: list[Step]):
    """ValueError for a step that continues the one before it (Step.continues)
    and cannot: it is the first, or its step, method, environment, optimizer
    or locked parts are not those of the step before it (the state it goes
    on from is theirs), or, both listing stages, it has no update left
    after the updates of the step before it."""
    for number, step in enumerate(steps, 1):
        if not step.continues:
            continue
        if number == 1:
            raise ValueError(f"step 1 {step.name} continues the step before it, and there is none")
        before = steps[number - 2]
        mine = (step.plugin.record(), step.method.record(), step.environment.record(), step.optimizer.record(),
                step.locked)
        theirs = (before.plugin.record(), before.method.record(), before.environment.record(),
                  before.optimizer.record(), before.locked)
        if mine != theirs:
            raise ValueError(f"step {number} {step.name} continues step {number - 1} {before.name}: it must have the "
                             "same step, method, environment, optimizer and locked parts")
        found, left = _updates(before), _updates(step)
        if found is not None and left is not None and left <= found:
            raise ValueError(f"step {number} {step.name} continues step {number - 1}, which makes {found:,} updates: "
                             f"it must have more than that, and it has {left:,}")


def draw_numbers(steps: list[Step]) -> list[int]:
    """The number each step's draws read with the seed (Data.number, the
    runner's dropout): its own place in the list from 1, or for a step that
    continues the one before it, that one's, so a step and its continuations
    draw as one step."""
    found = []
    for number, step in enumerate(steps, 1):
        found.append(found[-1] if step.continues and found else number)
    return found
