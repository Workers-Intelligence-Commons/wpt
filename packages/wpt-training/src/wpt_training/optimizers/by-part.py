"""Each kind of weight under its own update rule (docs/model.md,
"Training"), on a schedule, keeping an average of its last states or,
without it, its best evaluation, with an optional early stop.

The update rule is wpt_model.NorMuonAdamW over the model's optimizer_roles:
NorMuon for matrix weights (DeepSeek-V4's hybrid Newton-Schulz steps, each
output neuron normalized, updates of RMS 0.18, query and gate projections
head by head, routed experts one by one), AdamW for the embedding, the
output head, norms, the queries of Attention Residuals and attention sinks,
both with cautious weight decay (only where the update and the weight have
the same sign), and Adam at table_lr_factor times the rate, with no weight
decay, for Engram's tables. Only the parameters the step trains are in it:
those of the parts it locks, and of the parts its method never reaches,
are not.

Schedule: "steps" (DeepSeek LLM), a linear warm-up to lr over `warmup`
updates, then first_factor of it after first_drop of the step's updates
and second_factor after second_drop, the updates being those of every
stage of the step together; "constant", lr at every update.

The average (IMU-1, arXiv 2602.02522): with `average_states` above 0, the
step keeps an exponential moving average of the model's states (its whole
state_dict, every floating-point entry) taken at its last update and every
`average_every` updates before it, at most `average_states` of them, those
before update 1 left out. It starts from the oldest state and takes in each
later one with weight 1 - `average_beta`: with 10 states and 0.8, the
newest weighs 0.2 and the oldest 0.8^9. The average lives in FP32 on the
model's device from the oldest state on, one copy of the weights, in the
optimizer's state. Evaluations are then never the best: the step keeps the
average. A step that continues the one before it after its own oldest
state has no whole average and keeps its last weights. Early stop is
refused with the average, whose states count back from the step's last
update. `average_every` counted in updates and its default of 500 are
WPT's own, open to review: IMU-1 saved its states every 5,000 of 100,000
updates.

Without the average and without early stop (`patience` 0), the step keeps
the weights of the lowest validation loss of its last stage. With early
stop, a stage stops when the smoothed loss (the mean of the last
`smoothing` evaluations) has not improved by `min_improve` for `patience`
evaluations, and the step keeps the weights of the last improvement in its
last stage. The smaller the tokens-per-parameter ratio, the more
overfitting is tolerated: at Chinchilla balance (`chinchilla` tokens per
parameter) an improvement must be `min_improve`; a data-starved run relaxes
toward "never stop early, memorizing is the only available mode". An early
stop ends the stage: the next one starts, with its own count. A step
without evaluations never stops early and keeps its last weights.

lr and weight_decay default to Kimi Linear's smallest model (2.006e-3) and
Moonlight (0.1); a recipe sets its own.
"""

import torch

from wpt_model import NorMuonAdamW

from wpt_training import Optimizer, PluginError, Verdict

SETTINGS = {
    "lr": 0.002, "weight_decay": 0.1, "momentum": 0.95,
    "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
    "table_lr_factor": 5.0,
    "schedule": "steps",
    "warmup": 2000, "first_drop": 0.8, "first_factor": 0.316, "second_drop": 0.9, "second_factor": 0.1,
    "average_states": 10, "average_every": 500, "average_beta": 0.8,
    "patience": 0, "min_improve": 0.01, "smoothing": 3, "chinchilla": 20,
}
SCHEDULES = ("steps", "constant")


def check(settings):
    if settings["schedule"] not in SCHEDULES:
        raise ValueError(f"schedule {settings['schedule']!r}: one of {', '.join(SCHEDULES)}")
    if settings["patience"] < 0:
        raise ValueError(f"patience {settings['patience']}: 0 (no early stop) or more evaluations")
    if settings["average_states"] < 0 or settings["average_every"] < 1:
        raise ValueError(f"average_states {settings['average_states']} and average_every "
                         f"{settings['average_every']}: 0 or more states, every 1 or more updates")
    if not 0 <= settings["average_beta"] < 1:
        raise ValueError(f"average_beta {settings['average_beta']}: from 0 to less than 1")
    if settings["average_states"] and settings["patience"]:
        raise ValueError("average_states and patience: the average counts back from the step's last update, "
                         "which an early stop moves; set one of them to 0")


class ByPart(Optimizer):
    def __init__(self, model, settings, run):
        if not hasattr(model, "optimizer_roles"):
            raise PluginError("by-part: the model does not name its parameters' roles (optimizer_roles)")
        self.model, self.settings, self.steps = model, settings, run.steps
        self.optimizer = NorMuonAdamW(model.optimizer_roles(), settings["lr"], settings["weight_decay"],
                                      settings["momentum"], (settings["beta1"], settings["beta2"]), settings["eps"],
                                      settings["table_lr_factor"])
        counts = self.optimizer.counts
        run.log(f"by-part: {counts['matrix']} NorMuon matrices, {counts['vector']} AdamW tensors, "
                f"{counts['table']} Engram tables; schedule {settings['schedule']}")
        self.oldest = None  # the update of the average's oldest state
        if settings["average_states"]:
            every = settings["average_every"]
            self.oldest = run.steps - min(settings["average_states"] - 1, (run.steps - 1) // every) * every
            run.log(f"by-part: keeps the average of {(run.steps - self.oldest) // every + 1} states, every {every} "
                    f"updates from update {self.oldest}, beta {settings['average_beta']}")
        self.average, self.average_from, self.average_to = None, None, None
        self.min_improve = None
        if settings["patience"]:
            ratio = run.tokens / run.parameters
            self.min_improve = max(0.0005, settings["min_improve"] * min(1.0, ratio / settings["chinchilla"]))
            run.log(f"tokens: {run.tokens:,} | params: {run.parameters:,} | ratio: {ratio:.3f} tok/param "
                    f"(Chinchilla {settings['chinchilla']}) -> improvement threshold {self.min_improve * 100:.2f}%")
        self.best, self.without_improvement, self.history = float("inf"), 0, []

    def factor(self, step):
        """The share of lr an update uses."""
        s = self.settings
        if s["schedule"] == "constant":
            return 1.0
        if step <= s["warmup"]:
            return step / s["warmup"]
        if step > s["second_drop"] * self.steps:
            return s["second_factor"]
        if step > s["first_drop"] * self.steps:
            return s["first_factor"]
        return 1.0

    def before_update(self, step):
        for group in self.optimizer.param_groups:
            group["lr"] = self.settings["lr"] * group["lr_factor"] * self.factor(step)

    def after_update(self, step, reduce):
        if self.oldest is None or step < self.oldest or (self.steps - step) % self.settings["average_every"]:
            return
        state = {name: t.detach() for name, t in self.model.state_dict().items() if t.is_floating_point()}
        if step == self.oldest:
            self.average = {name: t.float().clone() for name, t in state.items()}
            self.average_from = step
        elif self.average is not None and self.average_from == self.oldest:
            torch._foreach_lerp_(list(self.average.values()), [state[name].float() for name in self.average],
                                 1 - self.settings["average_beta"])
        else:
            return  # the oldest state was another step's: no whole average
        self.average_to = step

    def kept(self):
        if self.average is None or self.average_from != self.oldest or self.average_to != self.steps:
            return None
        return {name: self.average[name].to(t.dtype) if name in self.average else t.detach().clone()
                for name, t in self.model.state_dict().items()}

    def stage(self, number, block_size):
        # the step keeps the best weights of its last stage, at its window length; longer windows: another scale
        self.best, self.without_improvement, self.history = float("inf"), 0, []

    def evaluated(self, step, val_loss):
        note = f" | lr x{self.factor(step):.3f}"
        if self.oldest is not None:  # the step keeps the average
            return Verdict(best=False, stop=False, note=note)
        if self.min_improve is None:
            if val_loss < self.best:
                self.best = val_loss
                return Verdict(best=True, stop=False, note=note + " <- best")
            return Verdict(best=False, stop=False, note=note)
        self.history.append(val_loss)
        recent = self.history[-self.settings["smoothing"]:]
        smoothed = sum(recent) / len(recent)
        if smoothed < self.best * (1 - self.min_improve):
            self.best, self.without_improvement = smoothed, 0
            return Verdict(best=True, stop=False, note=note + f" <- best (smoothed {smoothed:.4f})")
        self.without_improvement += 1
        if self.without_improvement >= self.settings["patience"]:
            return Verdict(best=False, stop=True,
                           note=note + f"\nearly stop: smoothed val loss has not improved by "
                                       f"{self.min_improve * 100:.2f}% for {self.settings['patience']} evaluations "
                                       f"(best {self.best:.4f})")
        return Verdict(best=False, stop=False, note=note)

    def state_dict(self):
        return {**super().state_dict(), "best": self.best, "without_improvement": self.without_improvement,
                "history": list(self.history), "average": self.average, "average_from": self.average_from,
                "average_to": self.average_to}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.best, self.without_improvement, self.history = state["best"], state["without_improvement"], state["history"]
        self.average_from, self.average_to = state["average_from"], state["average_to"]
        self.average = None
        if state["average"] is not None:  # a checkpoint's tensors come back on the CPU
            found = self.model.state_dict()
            self.average = {name: t.to(found[name].device) for name, t in state["average"].items()}


def start(model, settings, run):
    return ByPart(model, settings, run)
