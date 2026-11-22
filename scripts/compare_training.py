"""Compare the training of this checkout with another revision's, on
synthetic tokens, without a corpus or a depot: whether it trains the same
model, and how fast.

Usage:
    uv run scripts/compare_training.py --base REV [--out FOLDER] [--tiny]
        [--budget-minutes M] [--parts checks,sameness,speed,compiled,chain]

Run it from a checkout where `uv run scripts/setup.py` installed PyTorch;
the working tree is the revision measured, committed or not. The files of
every other revision are extracted (git archive) into a temporary folder,
and each revision's code runs in a child process of this script with the
same Python and packages, its own Triton and TorchInductor caches. Parts,
in this order:

    checks    the tests of packages/wpt-model that compare a GPU's fused
              kernels with the PyTorch reference (they skip off an NVIDIA
              GPU), through pytest, in this checkout
    sameness  about 20 updates of a small model with the recipes' layers,
              trained by REV, by every commit from REV to HEAD and by the
              working tree's changes, each compared with the one before it:
              the largest difference of their weights, training and
              validation losses. On the CPU training is bit for bit, so a
              change that keeps the model gives 0; there it runs with
              recomputation off, on, and on the longer windows only. On a
              GPU REV trains twice, and the difference of those two runs is
              the spread each difference is read against; there it runs
              with recomputation off and on, and compiled
    speed     tokens per second of training updates, REV against the
              working tree, for models of width 256 and 512 (11 layers,
              a vocabulary of 65,536), windows of 2,048, 8,192 and 32,768
              tokens, with recomputation off and on; every pass holds
              32,768 tokens (32,768 / window windows)
    compiled  the same with --compile's passes (WPT.compile_training), on
              a CUDA device only
    chain     the same speed for every commit between REV and HEAD, at
              width 512 and 8,192 tokens, with and without recomputation:
              what each change brings (REV and the last revision are the
              speed part's, when it ran)

An update is a forward and a backward pass of the model to its two summed
losses, an AdamW step and the model's balancing. The windows of one
revision, width and setting run in one process, shortest first, with one
model and optimizer, as a run's stages do: each measure records its first
update's time apart (it autotunes the kernels for that length, and with
compile it compiles), the tokens per second of the updates after the
warm-up ones, and on a GPU the peak memory of the first update and of the
timed ones. A measure or check that fails is written as failed with its
error (a process that dies fails its measure, and the next ones run in a
new process), and the next one runs. Once the budget is spent, the items
left are written as skipped.

It writes FOLDER/results.json, every item, and FOLDER/results.txt, a table
of them (FOLDER: compare-<time> under the current folder); the exit code
is 1 when an item failed. --tiny runs every part with a tiny model and
short windows: a check of this script itself, on any device. The CPU is
used when the environment hides every GPU or sets WPT_CPU=1.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "wpt-utils" / "src"))  # no library here
from wpt_utils.lines import ERROR, INFO, WARNING, say  # noqa: E402
SOURCES = ["apps/wpt/src", *[f"packages/{name}/src" for name in (
    "wpt-model", "wpt-training", "wpt-utils", "wpt-store", "wpt-telemetry", "wpt-recipes", "wpt-tokenizer",
    "wpt-inference", "wpt-depot", "wpt-worker", "wpt-agent")]]
CHECKS = ["packages/wpt-model/tests/test_kernels.py", "packages/wpt-model/tests/test_triton_kernels.py",
          "packages/wpt-model/tests/test_residuals.py", "packages/wpt-model/tests/test_model.py",
          "packages/wpt-model/tests/test_attention.py"]
WIDTHS = (256, 512)
WINDOWS = (2048, 8192, 32768)
PASS_TOKENS = 32768
WARMUP, TIMED = 3, 4  # updates before the timed ones (the first among them timed apart), and timed updates


def shape(width, vocab=65536):
    """A model of 11 layers (2 window layers each side of an exact layer,
    then an exact layer and 5 sparse ones) as the recipes of
    data/recipes/models/ make them: watery-tardigrade at width 512, and
    unimpaired-axolotl's settings (4 heads) scaled to width 256."""
    return dict(vocab_size=vocab, block_size=max(WINDOWS), window_layers=2, sparse_groups=1, sparse_layers=5,
                d_model=width, attention_heads=8 if width >= 512 else 4, attention_head_dim=128, n_experts=32,
                top_k=4, expert_width=width // 2, latent_width=width // 2, shared_width=width, engram_layers=[2],
                engram_order=3, engram_heads=8, engram_dim=16, engram_table_size=50021, residual_groups=22)


SMALL = dict(vocab_size=512, block_size=512, window_layers=2, sparse_groups=1, sparse_layers=5, d_model=64,
             attention_heads=2, attention_head_dim=128, n_experts=8, top_k=4, expert_width=32, latent_width=32,
             shared_width=64, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=8, engram_table_size=97,
             residual_groups=22)  # the sameness runs' model: the recipes' layers and full Attention Residuals, small


# -- the child: runs in a process of its own, with one revision's code first on sys.path --

def child(spec):
    """Runs spec, printing one JSON line per speed measure as it ends, and
    returns the last line's object."""
    sys.path[:0] = [str(Path(spec["root"]) / source) for source in SOURCES]
    import torch
    from wpt_model import WPT
    from wpt_model.config import ModelConfig
    from wpt_utils.device import select_device, train_in_tf32
    device = select_device()
    train_in_tf32(device)
    if device.type == "cuda":
        from wpt_model.kernels import kernels
        kernels(device, spec.get("kernels", "triton"))
    torch.manual_seed(0)
    model = WPT(ModelConfig(**spec["model"])).to(device).train()
    if spec["compile"]:
        model.compile_training(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    vocab = spec["model"]["vocab_size"]

    def windows(batch, length):
        tokens = torch.randint(vocab, (batch, length + 2), generator=generator).to(device)
        return tokens[:, :-2], tokens[:, 1:-1], tokens[:, 2:]

    def update(batch, length, recompute):
        model.recompute = recompute
        x, y, after = windows(batch, length)
        next_sum, ahead_sum = model(x, next_tokens=y, targets=(y, after))
        loss = next_sum / y.numel() + 0.3 * ahead_sum / after.numel()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.after_update(lambda tensor, op: tensor)
        return loss.detach()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def peak():
        return torch.cuda.max_memory_allocated(device) / 2**30

    if spec["kind"] == "sameness":
        from wpt_training.plugins import TextMethod
        validation, losses = [], []
        for number, (batch, length) in enumerate(spec["updates"], start=1):
            recompute = spec["recompute"] == "on" or (spec["recompute"] == "long" and length >= spec["long"])
            losses.append(float(update(batch, length, recompute)))
            if number % 5 == 0:
                model.eval()
                with torch.no_grad():
                    x, y, _ = windows(batch, length)
                    validation.append(float(TextMethod().validation_loss(model, x, y)))
                model.train()
        weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save({"weights": weights, "validation": validation, "losses": losses}, spec["out"])
        return {"device": str(device)}

    recompute = spec["recompute"] == "on"
    for batch, length in spec["measures"]:
        measured = {"device": str(device), "window": length, "batch": batch}
        try:
            if device.type == "cuda":
                measured["gpu"] = torch.cuda.get_device_name(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            update(batch, length, recompute)
            synchronize()
            measured["first_update_seconds"] = time.perf_counter() - started
            if device.type == "cuda":
                measured["first_update_peak_gib"] = peak()
            for _ in range(spec["warmup"] - 1):
                update(batch, length, recompute)
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(spec["timed"]):
                update(batch, length, recompute)
            synchronize()
            seconds = time.perf_counter() - started
            measured["seconds_per_update"] = seconds / spec["timed"]
            measured["tokens_per_second"] = batch * length * spec["timed"] / seconds
            if device.type == "cuda":
                measured["peak_gib"] = peak()
        except Exception as error:  # an out-of-memory error among them: the next, longer windows still run
            measured["error"] = f"{type(error).__name__}: {str(error)[:500]}"
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        sys.stdout.write(json.dumps(measured) + "\n")
        sys.stdout.flush()
    return {"done": True}


# -- the parent --

def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def named(revision):
    """A revision as a reader knows it: its short hash and subject, or the working tree."""
    return git("log", "-1", "--format=%h %s", revision)[:72] if revision else "the working tree"


class Checkouts:
    """The files of each revision asked for (git archive), in a temporary folder removed at the end;
    None is the working tree."""

    def __init__(self):
        self.folder = Path(tempfile.mkdtemp(prefix="wpt-compare-"))
        self.made = {}

    def of(self, revision):
        if revision is None:
            return ROOT
        sha = git("rev-parse", revision)
        if sha not in self.made:
            path = self.folder / sha[:12]
            path.mkdir()
            archive = subprocess.run(["git", "archive", sha], cwd=ROOT, check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", str(path)], input=archive, check=True)
            self.made[sha] = path
        return self.made[sha]

    def close(self):
        shutil.rmtree(self.folder, ignore_errors=True)


class Battery:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out or f"compare-{time.strftime('%Y%m%d-%H%M%S')}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + args.budget_minutes * 60
        self.items = []
        self.checkouts = Checkouts()
        self.caches = Path(tempfile.mkdtemp(prefix="wpt-compare-caches-"))

    def record(self, item):
        self.items.append(item)
        self.write()

    def write(self):
        (self.out / "results.json").write_text(json.dumps({"base": self.args.base, "tiny": self.args.tiny,
                                                           "items": self.items}, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")
        (self.out / "results.txt").write_text(table(self.items), encoding="utf-8", newline="\n")

    def spent(self, kind, label, **about):
        """Whether the budget is spent; if so, the item is written as skipped."""
        if time.monotonic() <= self.deadline:
            return False
        say(f"SKIPPED {kind} {label}", WARNING)
        self.record({"kind": kind, "label": label, **about, "status": "skipped: the budget is spent"})
        return True

    def item(self, kind, label, run, **about):
        """Runs one item: its result, or its error, recorded; skipped once the budget is spent."""
        if self.spent(kind, label, **about):
            return None
        entry = {"kind": kind, "label": label, **about}
        say(f"{kind} {label} ...")
        started = time.monotonic()
        try:
            entry["result"] = run()
            entry["status"] = "ok"
        except Exception as error:  # every failure is written and the next item runs
            entry["status"] = "failed"
            entry["error"] = f"{type(error).__name__}: {error}"
            say(f"FAILED {kind} {label}: {entry['error']}", ERROR)
        entry["seconds"] = round(time.monotonic() - started, 1)
        self.record(entry)
        return entry.get("result")

    def run_child(self, revision, spec, timeout):
        """The JSON lines the child printed, and its error (None when it ended well)."""
        root = self.checkouts.of(revision)
        name = git("rev-parse", "--short=12", revision) if revision else "working-tree"
        env = {**os.environ,
               "TRITON_CACHE_DIR": str(self.caches / name / "triton"),
               "TORCHINDUCTOR_CACHE_DIR": str(self.caches / name / "inductor")}
        spec = {**spec, "root": str(root)}
        try:
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)],
                                  capture_output=True, text=True, timeout=timeout, env=env, cwd=root)
        except subprocess.TimeoutExpired as expired:
            out = expired.stdout.decode() if isinstance(expired.stdout, bytes) else (expired.stdout or "")
            return [json.loads(line) for line in out.splitlines() if line.startswith("{")], \
                f"no end after {timeout} seconds"
        lines = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
        if done.returncode or not lines:
            tail = (done.stderr or done.stdout).strip().splitlines()[-12:]
            return lines, f"exit code {done.returncode}: " + " | ".join(tail)
        return lines, None

    def one_child(self, revision, spec, timeout):
        lines, error = self.run_child(revision, spec, timeout)
        if error:
            raise RuntimeError(error)
        return lines[-1]

    # -- parts --

    def checks(self):
        def run():
            report = self.out / "checks.xml"
            done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
                                   f"--junitxml={report.resolve()}", *CHECKS], cwd=ROOT, capture_output=True,
                                  text=True, timeout=1800)
            summary = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr[-500:]
            failed = [line for line in done.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
            if done.returncode not in (0, 5):
                raise RuntimeError(f"{summary}; " + "; ".join(failed[:20]))
            return {"summary": summary}
        self.item("checks", "fused kernels against the PyTorch reference", run)

    def commits(self):
        """Every revision after REV, in order: its commits, then the working tree's changes (None) if any."""
        commits = git("rev-list", "--reverse", f"{self.args.base}..HEAD").split()
        return commits + ([None] if git("status", "--porcelain", "--untracked-files=no") else [])

    def sameness(self):
        lengths = (128, 256) if self.args.tiny else (256, 512)
        updates = [(2, lengths[0])] * 10 + [(2, lengths[1])] * 10
        gpu = self.gpu()
        options = [("on", False), ("off", True), ("off", False)] if gpu else [("off", False), ("on", False),
                                                                             ("long", False)]
        base = git("rev-parse", self.args.base)
        for recompute, compiled in options:
            setting = f"recompute {recompute}" + (" | compiled" if compiled else "")

            def train(revision, recompute=recompute, compiled=compiled):
                path = self.out / f"sameness-{len(self.items)}.pt"
                self.one_child(revision, {"kind": "sameness", "model": SMALL, "updates": updates,
                                          "recompute": recompute, "long": lengths[1], "compile": compiled,
                                          "out": str(path.resolve())}, timeout=1800)
                try:
                    return _load(path)
                finally:
                    path.unlink()

            runs = {}

            def spread():
                runs[base] = train(base)
                return {"spread": difference(runs[base], train(base)) if gpu else 0.0}

            found = self.item("sameness", f"{named(base)} twice | {setting}", spread, revision=base,
                              recompute=recompute, compiled=compiled) if gpu else None
            limit = found["spread"] if found else 0.0
            previous = base
            for revision in self.commits():
                def compare(previous=previous, revision=revision):
                    if previous not in runs:
                        runs[previous] = train(previous)
                    runs[revision] = train(revision)
                    gap = difference(runs.pop(previous), runs[revision])
                    return {"difference": gap, "spread": limit, "same_bits": gap == 0.0, "within_spread": gap <= limit}
                self.item("sameness", f"{named(revision)} | {setting}", compare, revision=revision or "working tree",
                          parent=previous, recompute=recompute, compiled=compiled)
                previous = revision

    def speed(self, kind, compiled, revisions, widths, windows, recomputes):
        if compiled and not self.gpu():
            say("compiled measures need a CUDA device: none here", WARNING)
            return
        tokens = 1024 if self.args.tiny else PASS_TOKENS
        for width in widths:
            model = {**SMALL, "d_model": width // 8, "shared_width": width // 8} if self.args.tiny else shape(width)
            model["block_size"] = max(windows)
            for recompute in recomputes:
                for revision in revisions:
                    labels = [f"{named(revision)} | width {width} | window {window} | recompute {recompute} | "
                              + ("compiled" if compiled else "uncompiled") for window in windows]
                    about = [dict(revision=revision or "working tree", width=width, window=window,
                                  recompute=recompute, compiled=compiled) for window in windows]
                    self.measures(kind, revision, model, recompute, compiled, windows, tokens, labels, about)

    def measures(self, kind, revision, model, recompute, compiled, windows, tokens, labels, about):
        """The windows' measures in one child, shortest first; after a child that died, the rest in a new one."""
        left = list(range(len(windows)))
        while left:
            if self.spent(kind, labels[left[0]], **about[left[0]]):
                left.pop(0)
                continue
            say(f"{kind} {labels[left[0]]} ... (and {len(left) - 1} longer windows in the same process)")
            started = time.monotonic()
            spec = {"kind": "speed", "model": model, "recompute": recompute, "compile": compiled,
                    "measures": [(max(1, tokens // windows[i]), windows[i]) for i in left],
                    "warmup": WARMUP, "timed": 1 if self.args.tiny else TIMED}
            lines, error = self.run_child(revision, spec, timeout=600 * len(left))
            measured = [line for line in lines if "window" in line]
            for line in measured:
                i = left.pop(0)
                entry = {"kind": kind, "label": labels[i], **about[i], "seconds": round(time.monotonic() - started, 1)}
                if "error" in line:
                    entry.update(status="failed", error=line.pop("error"), result=line)
                    say(f"FAILED {kind} {labels[i]}: {entry['error']}", ERROR)
                else:
                    entry.update(status="ok", result=line)
                self.record(entry)
            if error and left:
                i = left.pop(0)
                say(f"FAILED {kind} {labels[i]}: {error}", ERROR)
                self.record({"kind": kind, "label": labels[i], **about[i], "status": "failed", "error": error,
                             "seconds": round(time.monotonic() - started, 1)})

    def gpu(self):
        if not hasattr(self, "_gpu"):
            probe = ("import sys; sys.path[:0] = [sys.argv[1]]; from wpt_utils.device import select_device; "
                     "print(select_device().type)")
            found = subprocess.run([sys.executable, "-c", probe, str(ROOT / "packages/wpt-utils/src")],
                                   capture_output=True, text=True)
            self._gpu = found.stdout.strip() == "cuda"
        return self._gpu

    def run(self):
        parts = self.args.parts.split(",")
        tiny = self.args.tiny
        widths, windows = ((256,), (256, 512)) if tiny else (WIDTHS, WINDOWS)
        ends = [self.args.base, None]
        try:
            for part in parts:
                if part == "checks":
                    self.checks()
                elif part == "sameness":
                    self.sameness()
                elif part == "speed":
                    self.speed("speed", False, ends, widths, windows, ("off", "on"))
                elif part == "compiled":
                    self.speed("speed", True, ends, widths, windows, ("off", "on"))
                elif part == "chain":
                    between = self.commits()
                    if "speed" in parts:
                        between = between[:-1]  # the last one is the speed part's
                    self.speed("chain", False, between, widths[-1:], windows[1:2] if not tiny else windows[:1],
                               ("off", "on"))
                else:
                    say(f"no part {part!r}: checks, sameness, speed, compiled, chain", ERROR)
                    self.record({"kind": part, "label": "no such part", "status": "failed",
                                 "error": f"no part {part!r}"})
        finally:
            self.checkouts.close()
            shutil.rmtree(self.caches, ignore_errors=True)
            self.write()
        failed = sum(item["status"] == "failed" for item in self.items)
        say(f"wrote {self.out / 'results.json'} and {self.out / 'results.txt'}: {len(self.items)} items, "
            f"{failed} failed", ERROR if failed else INFO)
        return 1 if failed else 0


def _load(path):
    import torch
    return torch.load(path, weights_only=True)


def difference(a, b):
    """The largest absolute difference between two sameness runs' weights,
    validation losses and training losses; inf when their names differ."""
    import math
    if a["weights"].keys() != b["weights"].keys():
        return float("inf")
    found = max((float((x.double() - b["weights"][name].double()).abs().max()) if x.numel() else 0.0)
                for name, x in a["weights"].items())
    for key in ("validation", "losses"):
        found = max(found, *(abs(x - y) for x, y in zip(a[key], b[key])))
    return found if math.isfinite(found) else float("inf")


def table(items):
    rows = ["kind | label | status | result"]
    for item in items:
        result = item.get("result") or {}
        if item["kind"] in ("speed", "chain") and "tokens_per_second" in result:
            text = (f"{result['tokens_per_second']:,.0f} tok/s, first update {result['first_update_seconds']:.1f} s"
                    + (f", peak {result['peak_gib']:.2f} GiB (first update {result['first_update_peak_gib']:.2f})"
                       if "peak_gib" in result else ""))
        elif item["kind"] == "sameness" and "difference" in result:
            text = f"difference {result['difference']:.3g}, spread {result['spread']:.3g}"
        elif item["kind"] == "sameness" and "spread" in result:
            text = f"spread {result['spread']:.3g}"
        elif item["kind"] == "checks" and result:
            text = result["summary"]
        else:
            text = item.get("error", "")
        rows.append(f"{item['kind']} | {item['label']} | {item['status']} | {text}")
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--base", help="the revision this checkout is compared with")
    parser.add_argument("--out", help="the folder the results go into (default: compare-<time>)")
    parser.add_argument("--tiny", action="store_true", help="a tiny model and short windows: checks this script")
    parser.add_argument("--budget-minutes", type=float, default=50, help="no item starts after this (default 50)")
    parser.add_argument("--parts", default="checks,sameness,speed,compiled,chain", help="the parts to run, in order")
    args = parser.parse_args()
    if args.child:
        try:
            sys.stdout.write(json.dumps(child(json.loads(args.child))) + "\n")
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return
    if not args.base:
        parser.error("give --base REV")
    sys.exit(Battery(args).run())


if __name__ == "__main__":
    main()
