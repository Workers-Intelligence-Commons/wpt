"""Run tests on a trained model and record their results in its model card.

Usage:
    wpt eval MODEL_FOLDER [EVAL ...] [--kernels FORM]

EVAL is the name of a built-in test (the files in wpt/eval/evals/) or the
path of any Python file; with none, every built-in test runs. A test file
defines

    def run(context: wpt.eval.Context) -> wpt.eval.Result

The context gives the model, its tokenizer, the folder's provenance and
the store named by WPT_STORE (environment or .env). The result carries data for programs and the same findings as text for
people. The model is on a gpu worker (wpt.eval.remote, docs/machines.md,
"Evaluation"): the first of WPT_GPU_WORKERS, or a local one; a test reaches
it through the context only: context.model.config, .device (where it runs,
in words), .generator(seed), .writer(prompt, rows, ...),
.greedy(prompts, steps) and .losses(windows). --kernels FORM (WPT_KERNELS:
tilelang, triton or pytorch) names what computes the model's attention on
the gpu worker, as for wpt train (docs/model.md, "Attention"); a form that
cannot work there refuses the evaluation before any test runs.

Each result goes to MODEL_FOLDER/evals/<name>.json, and the model card
(README.md) is written again to show it. A test that fails stops the run;
results already written stay.

The run is recorded in its own folder of the store's runs/ (wpt_telemetry; docs/observability.md): the weights it measured, each test
with its file's sha256 and its result, and every file written. A test can
record its own events through context.telemetry. When the record cannot be
written, the run stops (at its next record call, or sooner when a send
in the background was refused; docs/observability.md) and exits with code
3.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from wpt_model import DrafterMismatch, TokenizerMismatch, check_drafter, load_tokenizer, weight_hashes
from wpt_model.sparse import FORMS, named_form
from wpt_model.notice import CARD_NAME, EVALS_NAME, documents
from wpt_store import Store, StoreError, read_env_file, store_from_settings
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry, short
from wpt_tokenizer import Tokenizer
from wpt_utils.files import sha256_file
from wpt_utils.lines import ERROR, say
from wpt_utils.programs import code
from wpt_worker.link import LinkError

from .remote import Remote, WorkerError

BUILT_IN = Path(__file__).parent / "evals"


class EvalError(Exception):
    """A test cannot run: the folder is not a trained model, the test is
    unknown, has no run(context), or returned something other than a Result.
    Nothing is written for that test."""


@dataclass(frozen=True)
class Result:
    data: dict  # numbers and names that json can hold, for programs
    text: str   # the same findings for people; shown in the model card


@dataclass
class Context:
    folder: Path
    model: object  # on a gpu worker (wpt.eval.remote.Remote), or wpt_worker.evaluation.Served in this process
    tokenizer: Tokenizer
    device: str  # where the model runs, in words
    provenance: dict
    store: Store
    telemetry: Telemetry = field(default_factory=Telemetry.off)  # the run's record, for a test's own events


def built_in_names() -> list[str]:
    return sorted(path.stem for path in BUILT_IN.glob("*.py") if not path.name.startswith("_"))


def find(eval_) -> tuple[str, Path, str]:
    """(result name, file, how the result names the file). A built-in is
    named by its package path; another file by its name only, so the model
    folder never records where it lives on this machine."""
    if eval_ in built_in_names():
        return eval_, BUILT_IN / f"{eval_}.py", f"wpt/eval/evals/{eval_}.py"
    path = Path(eval_)
    if path.suffix != ".py" or not path.is_file():
        raise EvalError(f"{eval_}: not a built-in test ({', '.join(built_in_names())}) nor a Python file")
    return path.stem, path, path.name


def run_eval(context, eval_) -> dict:
    """Run one test, write its result into the folder and return it."""
    name, path, file_name = find(eval_)
    spec = importlib.util.spec_from_file_location(f"wpt_eval_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise EvalError(f"{path.name}: defines no run(context)")
    file_sha256 = sha256_file(path)
    telemetry = context.telemetry
    with telemetry.span("eval", name=name, file=file_name, file_sha256=file_sha256):
        result = module.run(context)
        if not isinstance(result, Result):
            raise EvalError(f"{path.name}: run(context) returned {type(result).__name__}, not wpt.eval.Result")
        record = {"name": name, "file": file_name, "file_sha256": file_sha256,
                  "code": code(), "device": context.device, "data": result.data, "text": result.text}
        telemetry.event("eval.result", name=name, data=result.data)
        folder = context.folder / EVALS_NAME
        text = json.dumps(record, indent=2) + "\n"
        with telemetry.span("eval.write", file=f"{EVALS_NAME}/{name}.json"):
            # each file is in the record, sent to the store, before it is written
            telemetry.output(**{f"eval_{name}": hashlib.sha256(text.encode()).hexdigest()})
            telemetry.irreversible("eval.file_written", file=f"{EVALS_NAME}/{name}.json")
            folder.mkdir(exist_ok=True)
            (folder / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
            _write_documents(context.folder, telemetry)  # the model card shows it
    return record


def _write_documents(folder, telemetry):
    """Write the folder's LICENSE, NOTICE and model card again from its
    files, results included: under hidden names beside them, then, once
    the model card's sha256 is recorded and sent, renamed into place. A
    record that fails leaves the earlier copies."""
    staged = {}
    try:
        for name, text in documents(folder).items():
            descriptor, path = tempfile.mkstemp(dir=folder, prefix=f".{name}-")
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as out:
                out.write(text)
            staged[name] = Path(path)
        telemetry.output(model_card=sha256_file(staged[CARD_NAME]))
        telemetry.irreversible("model.documents_written", files=sorted(staged))
        for name, path in staged.items():
            os.replace(path, folder / name)
    except BaseException:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise


def open_context(folder, store, telemetry=None) -> Context:
    """The context of the model folder, with no model yet (main puts the
    gpu worker's there). EvalError when the folder is not one wpt train
    wrote, its drafter was trained against other weights, or its tokenizer
    is not the weights'."""
    folder = Path(folder)
    try:
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvalError(f"{folder}: no provenance.json; give a folder that wpt train wrote") from None
    if "steps" not in provenance or "corpus" not in provenance:
        raise EvalError(f"{folder}: provenance.json names no steps or no corpus, so it is of another shape than "
                        "this wpt eval reads; train it again")
    try:
        tokenizer = Tokenizer.from_bytes(load_tokenizer(folder))
        check_drafter(folder)  # refuses a folder whose drafter is missing or of other weights
    # ValueError: a vocabulary of another tokenizer version; DrafterMismatch is one too
    except (TokenizerMismatch, DrafterMismatch, ValueError) as error:
        raise EvalError(str(error)) from None
    return Context(folder, None, tokenizer, "", provenance, store, telemetry or Telemetry.off())


def _setting(parser, name, parse, default):
    """The default of an option: the environment's name, parsed, else default."""
    if name not in os.environ:
        return default
    try:
        return parse(os.environ[name])
    except argparse.ArgumentTypeError as error:
        parser.error(f"{name}={os.environ[name]!r}: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wpt eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="model folder written by wpt train")
    parser.add_argument("evals", nargs="*", help=f"built-in test names or .py files (default: {', '.join(built_in_names())})")
    parser.add_argument("--kernels", type=named_form, default=_setting(parser, "WPT_KERNELS", named_form, None),
                        metavar="FORM",
                        help=f"what computes the attention on the gpu worker: {', '.join(FORMS)}; one that cannot "
                             "work there refuses the evaluation (WPT_KERNELS; default: the first kernels installed "
                             "for its GPU, else pytorch)")
    args = parser.parse_args(argv)
    environ = {**read_env_file(Path.cwd() / ".env"), **{key: value for key, value in os.environ.items() if value}}
    try:
        store = store_from_settings(Path.cwd() / ".env")
    except StoreError as error:  # no store, no record: nothing has started
        sys.exit(f"{type(error).__name__}: {error}")
    evals = args.evals or built_in_names()
    try:
        weights = weight_hashes(args.model)
    except OSError:  # no model there: open_context names the problem, in the record
        weights = {}
    try:
        with Telemetry.start(store, "wpt eval", sys.argv[1:] if argv is None else list(argv),
                             short(next(iter(weights.values()), "none")),
                             inputs={"model_folder": args.model, "evals": evals,
                                     "kernels": args.kernels},
                             env_file=Path.cwd() / ".env") as telemetry:
            try:
                with telemetry.span("inputs"):
                    for eval_ in evals:
                        find(eval_)  # an unknown test stops the run before any other has run
                    context = open_context(args.model, store, telemetry)
                    telemetry.input(weights=weights, corpus=context.provenance["corpus"]["id"])
                with telemetry.span("gpu_worker.model"), Remote(args.model, environ, telemetry.log, args.kernels) as model:
                    context.model, context.device = model, model.device
                    for eval_ in evals:
                        record = run_eval(context, eval_)
                        telemetry.log(f"\n{record['name']}\n{record['text'].rstrip()}")
            except (StoreError, EvalError, LinkError, WorkerError) as error:
                sys.exit(f"{type(error).__name__}: {error}")  # end.json records it
            telemetry.log(f"\nresults in {Path(args.model) / EVALS_NAME}, model card {Path(args.model) / 'README.md'}")
    except RecordError as error:
        say(f"RecordError: {error}\nstopped; nothing more was written", ERROR)
        sys.exit(RECORD_FAILED)
