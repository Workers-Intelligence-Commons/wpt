"""Train a model as a model order says, on the corpus its data part asks for.

Usage:
    wpt train ORDER [CORPUS_ID] [options]
    wpt train ORDER [CORPUS_ID] --strategy ddp --processes N [options]
    wpt train series NAME

ORDER is a file name in data/orders/models/, without .json, in the
repository holding the current folder, or SERIES:SIZE, one size of the
series file data/orders/series/SERIES.json, whose model order the
series derives (wpt_orders.series; docs/model.md, "Model series"): every
size of a series reads the same corpus, the smaller sizes its first part; or
SERIES:SIZE:RUN, another run of that size (a decay branch, a learning
rate of the check). Everything
that changes the resulting model comes from it: the architecture's
numbers, the seed, and its steps, which run in order on one model
(docs/training.md, "Steps"):
each a built-in step of wpt_training (read-text, answer-as-assistant,
answer-as-assistant-dft, answer-as-assistant-neftune, finish-tasks,
finish-tasks-with-less-writing) or a .py file, with its
method, environment, optimizer, settings and locked parts. The model's
context is the last window length of the last step that reads text
windows. The order's data part names a depot by its public address, a
snapshot of it with the change and digest it reads, the roles of its
corpus's portions (wpt_corpus.roles), its shares, the size of its
pretraining portion (total_tokens) and its vocabulary, named by its sha256
or learned for it from a sample of a number of bytes; `wpt build-resources ORDER` asks the
freight for its vocabulary and corpus, which it delivers on the dock, the
one named by DOCK (environment or .env) (wpt.resources); a size of a
series is derived for the size of the corpus's vocabulary. CORPUS_ID is a
corpus's id (or a start of it), of a corpus delivery or of a corpus the
depot built onto the dock's corpora/; without it, the corpus delivery
whose key the order's data part gives (wpt.resources.corpus_key), and
for a vocabulary named by its sha256 without one, the latest corpus the
depot built from that snapshot with that data part. No depot is read. No
fresh corpus refuses the run, saying to run `wpt build-resources`. The
vocabulary is the corpus's: a delivery's is the file of the vocabulary
delivery it names. A corpus given by its id must be made from that
snapshot with the order's portions, shares and vocabulary, at any size; a
corpus without portions (a delivery made without them, or one of the
dock's corpora/) is all of it its pretraining portion and needs the
order's shares and plain texts. A corpus of the dock's corpora/ built from
a dataset, before snapshots, is refused. The corpus holds everything the
steps read, by kind of texts, each step reading the kinds it names: the
token files of plain texts and of conversations, and the tasks. Of a corpus
cut into portions, each step reads its role's portion, the steps that read
text windows the pretraining one and those that read conversations and
tasks the fine-tuning one, and nothing reads the held-out portion: the
record training is given names none of its files
(wpt.resources.corpus_record), and the texts the model learned from never
name one of its texts. Training never builds anything and writes nothing to the depot: it
reads the corpus and checks every input against its recorded hash first. Every step's settings, the drafter's, the steps'
locked parts, and that the corpus holds something for each step to learn
from are checked before any step runs. The data part's total_tokens is the
size of the pretraining portion the order asks for and a ceiling on what it
reads: a corpus built at that size (its max_tokens) may hold fewer tokens of
plain text (what the snapshot leaves the pretraining portion; for a corpus
without portions, its conversations and tasks took their share of that
size), and training reads what it holds. A corpus that is stale (an admitted
erasure reached a text it was built from), or that holds fewer tokens of
plain text than total_tokens and was built at another size, is refused. A
step that reads text windows reads the first prefix_tokens of the corpus
(without them, total_tokens), at most what a corpus built at the order's
size holds; the run's record says how many it reads when that is fewer.

`wpt train series NAME` prints the numbers of every size of a series.

The training runs on gpu workers (wpt_worker; docs/machines.md): the
keys WPT_GPU_WORKERS lists, one gpu worker per machine, or with none a
local worker this command starts on this machine, reached over the same
link. On several workers, each runs --processes processes; workers in one
place meet through NCCL, workers in different places exchange gradients
over their TLS links (only --strategy ddp runs that way): the same synced
training either way. This command is the
wpt manager's side: it reads the order, the corpus, the tokenizer and
the dock, sends each worker its own checkout and everything the steps
read, answers the environments' parts that run here (the tokenizer's
work, the sandboxes of agentic tasks), keeps the checkpoints and the
steps' results, and writes the model folder. The worker deals only in
token ids. WPT_LINK_CHECKPOINTS says which checkpoints come back:
step-ends (the default), each step's last, with its end; every, each one
the worker writes. The worker keeps every checkpoint on its own disk too,
where its restarts find it.

The options say only how the hardware runs it. --strategy names how the
hardware is used (`single`: one device; `ddp`: several processes on the
gpu worker's machine, started by --processes), a built-in name or a .py
file (see wpt_training), sent to the worker when it is not a built-in;
`split` divides each
text window between the processes instead, for windows too long for one
GPU, and is refused for an order with a step that reads something else. On
one GPU, --segment runs a longer window as consecutive segments of that
many tokens; with `split`, each process runs its slice of a longer window
in segments of that many tokens. Each gives the same training as whole
windows. --kernels names what computes the attention on the gpu worker
(wpt_model.kernels; docs/model.md, "Attention"); one that cannot work there
refuses the run. The passes over whole text windows run through
torch.compile wherever they can (a CUDA device with Triton and a C++
compiler), and uncompiled elsewhere, which the log says in one line;
--compile makes a gpu worker where they cannot refuse the run, and
--no-compile runs them uncompiled; a checkpoint resumes either way
(docs/training.md, "Compiling the passes").

A step's batch is the number of examples per update over every process
together, so the same order trains the same way on any hardware. Each
process takes --micro-batch examples per pass (default: all of its share of
each batch) and adds passes up until the batch is complete. Without
--micro-batch, --recompute or --no-recompute, a stage of text windows whose
share does not fit the GPU's memory takes fewer windows per pass, and one
whose single window does not fit runs it in segments; the stage chooses
both from a measure when it starts, and a stage no pass fits is refused
(docs/training.md, "Recomputing layers"). The training does not change.

Every option can also come from an environment variable, WPT_ plus the
option name in capitals with '_' for '-' (--micro-batch ->
WPT_MICRO_BATCH; WPT_RESUME=1), or from the same line in the .env file of
the current folder; the command line wins, then the environment. The
order may come from WPT_ORDER and the corpus id from WPT_CORPUS.
WPT_GPU_WORKERS and WPT_LINK_CHECKPOINTS are read the same way.

After each step, the dock keeps its result (cache/steps/<fingerprint>): the
weights and the optimizer's state, under a fingerprint of the step before
it, the step itself (method, optimizer, environment, settings, locked
parts), the fingerprint of the part of the corpus it reads, the seed and
the code (wic_utils.programs.code: its commit and a hash of its
uncommitted changes, so any commit makes every kept result unknown); no
option is part of it. A run whose first steps
have a kept result starts from the result of the last of them and runs the
steps after it: an order that shares its first steps with an earlier run
reuses them, and a run that failed after some steps continues from the
first step not kept. Reuse means the same inputs, not the same bits.

Inside a step, every --checkpoint-every updates the main process writes
a checkpoint; with WPT_LINK_CHECKPOINTS=every it comes to MODEL/checkpoint/,
or with --run NAME to the dock's checkpoint of that name, which any
machine using the same dock can continue. Run the same command
with --resume to continue the step that stopped (on other hardware too);
the steps before it are reused. With step-ends (the default), a checkpoint
inside a step stays on the gpu worker, where only that job's restarts read
it. Once the step ends, its last checkpoint
becomes its kept result, as it is. Ctrl+C or SIGTERM finishes the update and
writes a checkpoint, whatever --checkpoint-every says, which comes back
whatever WPT_LINK_CHECKPOINTS says, and --resume continues from it.
--stop-after-minutes and --stop-after-updates stop the run the same way,
whichever comes first, counted over every step it trains from the end of
its first update, which also compiles the kernels. Such a run ends by
printing and recording what it measured from there: the tokens per second
and the time its text windows take at that speed.
--profile-update N runs update N of each step under torch.profiler and gives the
run's record profile.txt (docs/training.md, "A limit and a measure").
After a crash, every process on the gpu worker starts again, up
to --max-restarts times (default 3), and continues from the worker's own
checkpoint; 0 starts no launcher there.

This command records the run in its own folder of the dock's runs/
(wpt_telemetry; docs/observability.md), and the gpu worker's main
process adds to the same folder through the link: what it read, every
step, update, evaluation and checkpoint, every step result kept or reused,
and every file it wrote, one folder over the worker's restarts. When that
record cannot be written, the run finishes its update unrecorded, saves its
checkpoint (which the broken record does not name), and stops with code 3.

After the last step, the gpu worker trains the model's drafter for
speculative decoding against the final weights, the model frozen, with the
order's `drafter` section (wpt_inference.drafting; docs/model.md, "The
drafter"): its prompts are the beginnings of the plain texts the steps
read, and the model alone writes the answers, which stop at the end
of a text and, when a step trained on conversations or tasks, at
<|im_end|>, the end of a turn (the manager sends both tokens' ids). A
drafter that cannot be trained refuses the run (code 2) and writes nothing;
every step's result is kept, so the same command trains the drafter again
without training. --no-drafter (WPT_DRAFTER=0) saves the model without a
drafter, its weights the same: its provenance.json says so (options), and
the same command without --no-drafter later trains the drafter from the
steps the dock keeps, without training them. Drafter settings no
drafter can have (feature layers the model does not have, an unknown
setting) are refused before training starts. Then it measures the
scheduler's pass times once (wpt_inference.measure_costs) and writes them
into the run's record as scheduler-costs.json, with the setup they were
measured on (the model's config, the drafter's block, the device);
--scheduler-costs FOLDER replays those an earlier run recorded instead,
refused before training when they were measured on another setup. With
--no-drafter, nothing is measured.

Exit codes: 0 done, 2 refused (bad input, other settings than the
checkpoint's, a step that failed, a drafter that could not be trained;
never restarted), 3 the record could not be written (never restarted), 4
a gpu worker lost (it could not be reached, refused the link, or its link
broke; the record names it, and the same command with --resume continues
from the checkpoint the manager holds), 5 a crash of a training process on
one of several gpu workers (the jobs on every worker were stopped; the
same command with --resume continues from the checkpoint the manager holds,
as after a lost worker; on one gpu worker a crash is the process's own exit
code, after its restarts), 130 stopped by a signal or a limit, others a crash. A step that fails keeps no result;
a checkpoint it wrote stays, which --resume continues and deleting it
starts the step over; the results of the steps before it stay on the
dock.

The model folder gets config.json, the weights (their headers name the
licenses), tokenizer.json (the corpus's vocabulary, for wpt chat), the
drafter unless --no-drafter (drafter/: config.json, model.safetensors, and
provenance.json, which names the sha256 of the weights it was trained
against),
provenance.json (what produced it: the depot, the snapshot with its
change and digest, the order, the code's commit, the key, id and
preparation of the vocabulary's and the corpus's deliveries, each step,
its fingerprint and result, and what of the corpus each one learned from;
docs/training.md, "What a model records"), CREDITS (every
text the steps learned from, one line each, from the corpus's tables:
wpt.train.notice), and LICENSE, NOTICE and README.md (the model card). It is written beside the target as a hidden folder,
recorded, then renamed into place: a run that fails before leaves the
target as it was. A folder already there, with everything in it (an
earlier model, its checkpoint, the results wpt eval wrote for other
weights), is replaced as a whole. When the new folder cannot take the
target's place and the target is then missing, or holds only a checkpoint
of this run (without --run), the new folder is kept and named on the
error output, to be moved into place by hand.
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from freight import api as freight
from wpt_inference import COSTS_FILE
from wpt_inference.drafting import settings as drafter_settings
from wpt_model import PARTS, WPT, ModelConfig, canonical_ids, save_drafter, save_model, weight_files, weight_hashes
from wpt_model.drafter import Drafter
from wpt_model.kernels import FORMS, named_form
from wpt_model.notice import CREDITS_NAME, WEIGHT_METADATA, write_documents
from wpt_corpus import corpora, roles
from wpt_corpus.files import DockFiles
from wpt_orders import SIZE_OF, OrderError, Orders
from wpt_orders.series import layer_count
from dock.api import NAME as DOCK_NAME
from dock import api as dock_api
from dock.api import DockError, DockSettingError
from wic_utils.env import read_env_file
from wpt.models import checkpoint_key, state_key
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry, file_hashes, set_result, short
from wpt_tokenizer import RUSTC_VERSION, Tokenizer, chat
from wpt_training import (WINDOWS, Checkpoint, FolderCheckpoint, PluginError, check_continuations, load_state,
                          load_step, save_state, strategy_plugin, weights)
from wpt_training.launch import REFUSED, STOPPED
from wic_utils.files import sha256_file
from wic_utils.lines import ERROR, WARNING, say
from wic_utils.programs import code, fingerprint, repository
from wpt_worker.link import LinkError
from wpt_worker.manager import Files, NoCheckout, code_root, keys, machines, run_jobs, start_workers
from wpt_worker.training import CHECKPOINTS, STEP_ENDS

from .. import resources
from .notice import PLAIN, model_notice
from .remote import Training

SIZE_UNITS = {"": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9}
CHECKPOINT_FOLDER = "checkpoint"
LOST = 4  # a gpu worker could not be reached or its link broke: the manager may replace it and resume
CRASHED = 5  # a training process crashed while the run spans several gpu workers: the manager may resume
LINK_CHECKPOINTS = "WPT_LINK_CHECKPOINTS"  # every: each checkpoint comes back from the gpu worker; step-ends: each step's last


class Refused(Exception):
    """The run cannot start as asked; trying again would refuse again."""


class InputError(Exception):
    """The corpus is missing, it was built from another snapshot or
    selection than the order's, the order's steps cannot run as
    asked, or the batch does not split over the processes. Nothing has been
    trained. (A token file or vocabulary that does not match its hash is the
    dock's CorruptFile.)"""


def parse_size(text):
    """'500MB' -> 500_000_000. Units: KB, MB, GB, or none for bytes."""
    match = re.fullmatch(r"(\d+)\s*([KMG]B)?", text.strip().upper())
    if not match:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (examples: 500MB, 2GB, 1000000)")
    return int(match[1]) * SIZE_UNITS[match[2] or ""]


# option name, type, default, help
OPTIONS = [
    ("out", str, "runs/default", "model folder to write"),
    ("micro-batch", int, None, "examples per pass in one process (default: its whole share of each batch, or for "
                               "a stage of text windows left to choose, see --recompute, the windows that fit "
                               "the GPU's memory)"),
    ("checkpoint-every", int, 500, "write a checkpoint every N updates of a step (0: none during a step); each "
                                   "step's result is kept on the dock when it ends"),
    ("segment", int, 0, "run a window longer than this many tokens as consecutive segments (with --strategy split, "
                        "each process's slice): the same training with less memory, each segment but the last run "
                        "twice (a multiple of 128; 0: whole windows)"),
    ("recompute-from", int, 0, "with --recompute: only the stages of windows of at least this many tokens run "
                               "their layers again, and shorter ones keep every activation, faster where memory "
                               "allows (0: every stage)"),
    ("strategy", str, "single", "how the hardware is used: a built-in name or a .py file"),
    ("processes", int, 1, "start this many processes on this machine, one per GPU (with --strategy ddp or split)"),
    ("max-shard-size", parse_size, None, "split the weights into files of at most this size, e.g. 500MB"),
    ("stream-shards", int, 8, "with --stream: token files of the corpus in use at once (windows are drawn from them)"),
    ("run", str, None, "keep the checkpoint on the dock under this name, so any machine using the dock can "
                       "continue the run"),
    ("scheduler-costs", str, None, "replay the drafter scheduler's pass times that an earlier run recorded on the "
                                   f"same setup: the name of its record folder, which holds {COSTS_FILE} (default: "
                                   "measure them)"),
    ("max-restarts", int, 3, "after a crash, start every process again this many times at most, continuing from "
                             "the checkpoint (0: run in this process, never restart)"),
    ("stop-after-minutes", float, 0, "stop as a stop signal does after this many minutes of training, counted from "
                                     "the end of the first update, and print the speed measured and the time the "
                                     "whole run takes at that speed (0: no limit)"),
    ("stop-after-updates", int, 0, "stop as a stop signal does after this many updates, over every step this "
                                   "command trains, and print what --stop-after-minutes prints (0: no limit)"),
    ("profile-update", int, 0, "run the update with this number in its step (from 1) under torch.profiler, and give "
                               "the run's record profile.txt: where its time goes, the machine's facts and the "
                               "memory's (0: no profile)"),
    ("kernels", named_form, None, f"what computes the attention: {', '.join(FORMS)}; one that cannot work on the "
                                   "gpu worker stops the run (default: the first kernels installed for its GPU, "
                                   "else pytorch)"),
]


SHARD_FIELDS = ("key", "sha256", "token_sha256")  # what a model's provenance keeps of each token file of its corpus


def switch(parser, environ, variable):
    """An option the environment turns on (1, true, yes) or off (0, false,
    no), or None when it says nothing."""
    value = environ.get(variable, "").strip().lower()
    if not value:
        return None
    if value not in ("1", "true", "yes", "0", "false", "no"):
        parser.error(f"{variable}={environ[variable]!r}: 1 (on) or 0 (off)")
    return value in ("1", "true", "yes")


def parse_args(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(prog="wpt train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("order", nargs="?", default=environ.get("WPT_ORDER"),
                        help="model order: a file name in data/orders/models/, without .json, or SERIES:SIZE[:RUN], "
                             "a size of a series file of data/orders/series/ or another run of it (WPT_ORDER)")
    parser.add_argument("corpus", nargs="?", default=environ.get("WPT_CORPUS"),
                        help="a corpus's id, or a start of it: a corpus delivery's (`wpt build-resources` prints it) "
                             "or a corpus of the dock's corpora/; default: the order's corpus (WPT_CORPUS)")
    for name, kind, default, text in OPTIONS:
        variable = "WPT_" + name.upper().replace("-", "_")
        if variable in environ:
            try:
                default = kind(environ[variable])
            except (ValueError, argparse.ArgumentTypeError) as error:
                parser.error(f"{variable}={environ[variable]!r}: {error}")
        parser.add_argument(f"--{name}", type=kind, default=default, help=f"{text} ({variable})")
    parser.add_argument("--stream", action="store_true", default=environ.get("WPT_STREAM", "") in ("1", "true", "yes"),
                        help="read the corpus's token files a few at a time instead of copying them all to disk "
                             "first (WPT_STREAM)")
    parser.add_argument("--recompute", action=argparse.BooleanOptionalAction,
                        default=switch(parser, environ, "WPT_RECOMPUTE"),
                        help="keep only the input of each window layer's attention and of each experts sublayer, and "
                             "run it again for the backward pass: the same training with much less memory, slower. "
                             "Unset, each stage of text windows chooses from short probe passes that measure the "
                             "GPU's memory: every activation kept where its pass fits, else the layers recomputed, "
                             "then fewer windows per pass (without --micro-batch), then one window in segments "
                             "(without --segment), and the stage is refused when no pass fits; --recompute "
                             "recomputes in every stage, --no-recompute in none, and neither measures "
                             "(WPT_RECOMPUTE=1 or 0)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=switch(parser, environ, "WPT_COMPILE"),
                        help="the passes over whole text windows run through torch.compile: the same training "
                             "within rounding, faster once the first update has compiled it. Unset, they do "
                             "wherever they can (a CUDA device with Triton and a C++ compiler) and run uncompiled "
                             "elsewhere, which the log says; --compile stops the run where they cannot, "
                             "--no-compile runs them uncompiled (WPT_COMPILE=1 or 0)")
    parser.add_argument("--drafter", action=argparse.BooleanOptionalAction,
                        default=switch(parser, environ, "WPT_DRAFTER") is not False,
                        help="end the run by training the model's drafter for speculative decoding (the default); "
                             "--no-drafter saves the model without one, its weights the same, and the same command "
                             "without it later trains the drafter from the steps the dock keeps (WPT_DRAFTER=1 or "
                             "0)")
    parser.add_argument("--resume", action="store_true", default=environ.get("WPT_RESUME", "") in ("1", "true", "yes"),
                        help="continue the step that stopped from its checkpoint: the model folder's, or with --run "
                             "the dock's (WPT_RESUME)")
    args = parser.parse_args(argv)
    if not args.order:
        parser.error("give the model order (or set WPT_ORDER)")
    if args.stop_after_minutes < 0 or args.stop_after_updates < 0:
        parser.error("--stop-after-minutes and --stop-after-updates: 0 (no limit) or more")
    if args.recompute_from < 0:
        parser.error("--recompute-from: 0 (every stage) or a number of tokens")
    if args.recompute_from and args.recompute is not True:
        parser.error("--recompute-from narrows --recompute to the longer windows: give --recompute too")
    if args.profile_update < 0:
        parser.error("--profile-update: 0 (no profile) or an update's number")
    if args.scheduler_costs and not args.drafter:
        parser.error("--scheduler-costs replays the pass times of the drafter's scheduler, and --no-drafter trains "
                     "no drafter")
    if args.run is not None and not DOCK_NAME.fullmatch(args.run):
        parser.error(f"--run {args.run!r}: use letters, digits, '.', '_' and '-' only")
    return args


def load_data(name) -> dict:
    """The data part of the model order of that name (or SERIES:SIZE[:RUN]):
    its depot, snapshot and total_tokens, from the repository holding the current folder."""
    try:
        return Orders.find().data_part(name)
    except OrderError as error:
        raise InputError(str(error)) from None


def load_order(name, record):
    """The model order of that name (or SERIES:SIZE[:RUN]), checked against its
    schema, from the repository holding the current folder; a size of a
    series is derived for the vocabulary of the corpus record."""
    try:
        return Orders.find().model_order(name, record["tokenizer"]["vocab_size"])
    except OrderError as error:
        raise InputError(str(error)) from None


# how an order gets its corpus
BUILD = ("run `wpt build-resources ORDER` (SERIES: for a series), which asks the freight for the order's vocabulary "
         "and corpus")


def find_corpus(dock, given, data) -> dict:
    """The record of the corpus given (an id or a start of it: a corpus of
    the dock's corpora/, else a corpus delivery of the dock, by its id or
    its key), or without one the corpus of the order's data part: the
    corpus delivery whose key the data part gives (wpt.resources.corpus_key,
    its vocabulary's delivery wpt.resources.vocabulary_key), as
    wpt.resources.corpus_record gives it; without one, for a vocabulary
    named by its sha256, the latest corpus of the dock's corpora/ built from
    that snapshot (name, change, digest), depot, shares (taking plain
    texts), vocabulary and size. No depot is read. InputError when neither
    is there fresh, saying to run `wpt build-resources`; nothing is built or
    queued. InputError too for a corpus given that was built from a dataset,
    before snapshots, and when the freight cannot read the dock (a setting
    it needs is missing)."""
    try:
        return _find_corpus(dock, given, data)
    except (freight.SettingMissing, DockSettingError) as error:
        raise InputError(f"the deliveries of the dock {dock} cannot be read ({type(error).__name__}: "
                         f"{error})") from None


def _find_corpus(dock, given, data) -> dict:
    """find_corpus, its freight's and dock's errors aside."""
    if given is None:
        return _snapshot_corpus(dock, data)
    try:
        found = corpora.find(dock, given) or _delivered(dock, given)
    except ValueError as error:
        raise InputError(str(error)) from None
    if found is None:
        raise InputError(f"no corpus {given} in {dock}")
    if "build_key" not in found and "delivery" not in found:
        raise InputError(f"corpus {found['id']} was built from {_source(found)}, before snapshots; an order names "
                         f"a snapshot: {BUILD}")
    return found


def _delivered(dock, given) -> dict | None:
    """The record of the corpus delivery of the dock whose id or key
    starts with given, None when none does. ValueError when several do."""
    found = [note for note in freight.deliveries_on(dock=dock)
             if note.get("preparation", {}).get("name") == resources.TOKENS
             and (note["id"].startswith(given) or note["key"].startswith(given))]
    if len(found) > 1:
        raise ValueError(f"{given!r} starts the ids or keys of {len(found)} corpus deliveries: give more of it")
    return _record_of(dock, found[0]) if found else None


def _record_of(dock, note) -> dict:
    """A corpus delivery's note as training reads it, with the note of its vocabulary on the dock."""
    vocabulary = freight.delivery(note["preparation"]["inputs"]["vocabulary"], dock=dock)
    if vocabulary is None:
        raise InputError(f"corpus delivery {note['key']} names the vocabulary delivery "
                         f"{note['preparation']['inputs']['vocabulary']}, which the dock does not hold")
    return resources.corpus_record(note, vocabulary)


def _snapshot_corpus(dock, data) -> dict:
    """The record of the corpus an order's data part asks for (find_corpus)."""
    vocabulary = resources.vocabulary_key(data, dock)
    key = None if vocabulary is None else resources.corpus_key(data, vocabulary)
    note = None if key is None else freight.delivery(key, dock=dock)
    if note is not None and "stale" not in note and "deleted" not in note:
        return _record_of(dock, note)
    if note is not None:
        raise InputError(f"corpus delivery {note['key']} of the order is "
                         + (f"stale: the depot carried out the erasures {', '.join(note['stale']['erasures'])} on "
                            "texts it holds" if "stale" in note else f"deleted: {note['deleted']['why']}")
                         + f"; {BUILD}, which makes it again")
    legacy = _built(dock, data) if resources.vocabulary_named(data) else None
    if legacy is not None and not corpora.by_kind(legacy) and not {"stale", "deleted"} & legacy.keys():
        raise InputError(f"corpus {legacy['id']} does not list its token files by kind, which each training step "
                         f"reads: {BUILD}")
    if legacy is not None and corpora.fresh(legacy):
        return legacy
    vocabulary_said = (f"the vocabulary {data['vocabulary']['sha256']}" if resources.vocabulary_named(data)
                       else f"its vocabulary learned on {data['vocabulary']['sample_bytes']:,} bytes")
    missing = (f"the dock holds no vocabulary {data['vocabulary']['sha256']}" if vocabulary is None
               else f"the dock holds no corpus delivery {key}")
    raise InputError(f"no fresh corpus of snapshot {data['snapshot']} of the depot {data['depot']} with "
                     f"{vocabulary_said} at the size {data['total_tokens']:,} in {dock} ({missing}): {BUILD}")


def _built(dock, data) -> dict | None:
    """The latest corpus of the dock's corpora/ built from the snapshot a
    data part names (name, change, digest), of its depot, with its shares
    (taking plain texts) and vocabulary, at its size, under the selection rules the depot
    built with last (wpt_corpus.corpora.RULES, which its build key covered): a fresh one
    first. None without one."""
    found = [record for record in corpora.records(dock)
             if "build_key" in record and record["depot"] == data["depot"] and record.get("rules") == corpora.RULES
             and _same_build(record, data) and corpora.size(record) == data["total_tokens"]]
    return next((record for record in found if corpora.fresh(record)), found[0] if found else None)


def _source(record) -> str:
    """What a corpus was built from, as a refusal names it."""
    if "build_key" in record or "delivery" in record:
        snapshot = record["snapshot"]
        return (f"snapshot {snapshot['name']} (change {snapshot['change']}, digest {snapshot['digest']}) of the "
                f"depot {record['depot']}")
    return f"dataset {record['dataset']['name']!r} of the depot {record['depot']!r}"


def _same_build(record, data) -> bool:
    """Whether a corpus built from a snapshot is one the order's data part
    asks for, its sizes aside: the same snapshot (name, change and digest,
    which names its owner too) and vocabulary, and the same selection. A
    corpus delivery's vocabulary is the delivery the data part names (its
    sha256, or the key of the vocabulary it learns). A delivery cut into
    portions has the portions of the data part's roles (wpt_corpus.roles), in
    the freight's canonical form, each with its kinds and shares and the
    held-out one with its size. A corpus without portions (a delivery made
    without them, or one the depot built onto corpora/) is all of it the
    pretraining portion: it has the data part's shares and takes plain
    texts."""
    snapshot = record["snapshot"]
    try:
        if "build_key" in record:
            theirs = corpora.canonical(record["data"])
            same_selection = (theirs["shares"] == corpora.canonical({"shares": data.get("shares")})["shares"]
                              and (not theirs["kinds"] or PLAIN in theirs["kinds"]))
        elif record["request"]["selection"].get("portions"):
            same_selection = (_cut(freight.canonical(record["request"]["selection"]))
                              == _cut(freight.canonical({"portions": roles.portions(data)})))
        else:
            theirs = freight.canonical({**record["request"]["selection"], "size": None})
            same_selection = (theirs["shares"] == freight.canonical({"shares": data.get("shares")})["shares"]
                              and (not theirs["kinds"] or PLAIN in theirs["kinds"]))
    except (freight.Refused, corpora.Refused, roles.Refused):
        return False
    if "delivery" in record:
        same_vocabulary = (record["vocabulary"]["sha256"] == data["vocabulary"]["sha256"]
                           if resources.vocabulary_named(data)
                           else record["vocabulary_delivery"] == resources.vocabulary_key(data))
    else:
        same_vocabulary = (resources.vocabulary_named(data)
                           and record["vocabulary"]["sha256"] == data["vocabulary"]["sha256"])
    return ((snapshot["name"], snapshot["change"], snapshot["digest"]) == (data["snapshot"], data["change"],
                                                                         data["digest"])
            and same_selection and same_vocabulary)


def _cut(selection: dict) -> list[dict]:
    """The portions of a canonical selection with the sizes training lets vary set aside: every one's but the
    held-out portion's, which is the same for every corpus."""
    return [{**one, "size": one["size"] if one["name"] == roles.HELD_OUT else None} for one in selection["portions"]]


def load_corpus(dock, record, order):
    """The corpus's tokenizer, its vocabulary checked, once the corpus is
    known to be the order's: a corpus built or delivered from the order's
    snapshot (name, change, digest) with its portions, shares and
    vocabulary (_same_build), at any size (check_record checks the size).
    The record is one find_corpus gives, of a corpus built from a
    snapshot."""
    data = order["data"]
    if not _same_build(record, data):
        raise InputError(f"corpus {record['id']} was built from {_source(record)} with portions, shares or "
                         f"vocabulary other than the order's; this order names snapshot {data['snapshot']} (change "
                         f"{data['change']}, digest {data['digest']}) of the depot {data['depot']}")
    vocabulary = record["tokenizer"]["vocabulary"]
    return Tokenizer.from_bytes(dock_api.fetch(dock, vocabulary["key"], vocabulary["sha256"]))  # checked by the dock


def built_for(record, order) -> bool:
    """Whether the corpus was built at the size the order asks for: its
    max_tokens (wpt_corpus.corpora.size) is the data part's total_tokens."""
    return corpora.size(record) == order["data"]["total_tokens"]


def check_record(record, order):
    """InputError when the corpus cannot train the order: it is stale (an
    admitted erasure reached a text it was built from), or it holds fewer
    tokens of plain text than the data part's total_tokens and was not
    built at that size (built_for: such a corpus gives its conversations
    and tasks their share of the size, and training reads what it holds),
    or it does not list its token files by kind, which each step reads
    (wpt_corpus.corpora.by_kind). The message says to make it again with
    `wpt build-resources`."""
    if not corpora.by_kind(record):
        raise InputError(f"corpus {record['id']} does not list its token files by kind, which each training step "
                         f"reads: {BUILD}")
    if "stale" in record:
        raise InputError(f"corpus {record['id']} is stale: the depot carried out the erasures "
                         f"{', '.join(record['stale']['erasures'])} on texts it was built from; {BUILD}")
    wanted = order["data"]["total_tokens"]
    if record["total_tokens"] < wanted and not built_for(record, order):
        size = corpora.size(record)
        raise InputError(f"the order trains on {wanted:,} tokens (data.total_tokens); corpus {record['id']} holds "
                         f"{record['total_tokens']:,} and was built at "
                         f"{'no size' if size is None else f'the size {size:,}'}, not the order's {wanted:,}")


def plan(order) -> tuple[list, int]:
    """(the order's steps, each a wpt_training.Step, every setting typed
    and checked, the model's context: the last window length of the last
    step that reads text windows). PluginError for a step that cannot be;
    InputError for steps that cannot run together. (Whether they run with
    the strategy asked is the gpu worker's to check: it starts the
    strategy.)"""
    steps = [load_step(entry, PARTS) for entry in order["steps"]]
    try:
        check_continuations(steps)
    except ValueError as error:
        raise InputError(str(error)) from None
    contexts = [step.context() for step in steps if step.context() is not None]
    if not contexts:
        raise InputError("no step reads text windows, so nothing sets the model's context; start the order with "
                         "read-text")
    return steps, contexts[-1]


def with_prefixes(order, record, log=lambda text: None) -> dict:
    """The order with each step that reads text windows given the prefix
    it reads: its prefix_tokens, or without them the data part's
    total_tokens, and for a corpus built at the order's size (built_for)
    never more than the tokens of plain text the corpus holds. log gets a
    line for each step that reads fewer tokens than it asks for."""
    held = record["total_tokens"] if built_for(record, order) else None
    entries = []
    for number, entry in enumerate(order["steps"], 1):
        step = load_step(entry, PARTS)
        if step.reads == WINDOWS:
            asked = step.settings["prefix_tokens"] or order["data"]["total_tokens"]
            prefix = asked if held is None else min(asked, held)
            if prefix < asked:
                log(f"step {number} {step.name} reads the first {prefix:,} tokens of plain text, all that corpus "
                    f"{record['id']} holds, not the {asked:,} it asks for: built at the order's size "
                    f"{order['data']['total_tokens']:,} (data.total_tokens), the corpus holds no more: a corpus "
                    "without portions counts its conversations and tasks in that size, and a corpus with portions "
                    "holds what its snapshot leaves its pretraining portion")
            if prefix != step.settings["prefix_tokens"]:
                entry = {**entry, "settings": {**entry.get("settings", {}), "prefix_tokens": prefix}}
        entries.append(entry)
    return {**order, "steps": entries}


def check_corpus(steps, record, files):
    """InputError naming the first step whose environment finds nothing to
    learn from in the corpus (wpt_training.Step.check_corpus), its files
    read through files (wpt_corpus.files)."""
    for number, step in enumerate(steps, 1):
        try:
            step.check_corpus(record, files)
        except ValueError as error:
            raise InputError(f"step {number} {step.name}: {error}") from None


def fingerprints(steps, model_config, record, seed, code) -> list[str]:
    """Each step's fingerprint, in order: the previous step's (for the first,
    the initial weights': the model's shape and vocabulary size, without
    block_size, which no weight depends on and a later step may change),
    the step (wpt_training.Step.record: its files, method, environment,
    optimizer, settings, locked parts and kinds of texts), the fingerprints
    of what the corpus holds of each kind of texts it reads, in the order
    it names them (the record's `fingerprints`; InputError for a kind it
    does not record), the order's seed
    and the code (code, wic_utils.programs.code()). The step's position in the list, which its draws
    mix into the seed, is the length of the chain before it. No option is
    part of it."""
    shape = {key: value for key, value in asdict(model_config).items() if key != "block_size"}
    previous = fingerprint({"initial": shape})
    found = []
    held = record.get("fingerprints", {})
    for number, step in enumerate(steps, 1):
        missing = [kind for kind in step.kinds if kind not in held]
        if missing:
            raise InputError(f"step {number} {step.name} reads texts of the kind {', '.join(missing)}, of which corpus "
                             f"{record.get('id', '')} records no fingerprint: no corpus build keeps that kind as one "
                             f"a step reads (kinds of a corpus: {', '.join(sorted(held)) or 'none'})")
        previous = fingerprint({"previous": previous, "step": step.record(),
                                "data": [held[kind] for kind in step.kinds], "seed": seed,
                                "code": code})
        found.append(previous)
    return found


class StepResults:
    """The results of the steps runs made, on the dock: the result
    cache/steps/<the step's fingerprint> is a JSON record, {"about": what
    the run recorded of the step, "state": the file of its checkpoint
    state, {"key", "sha256"}} (the weights, those of its best evaluation, the optimizer's,
    method's and environment's state, and the step's fingerprint as its
    signature; wpt_training.train), the bytes of the step's last checkpoint
    as they are. The main process alone reads and writes them. dock: the
    dock's URL."""

    STAGE = "steps"

    def __init__(self, dock, telemetry):
        self.dock, self.telemetry = dock, telemetry

    def found(self, mark) -> dict | None:
        """The result of the step of fingerprint mark, its own sha256 in
        "result"; None when none is kept."""
        found = dock_api.get_result(self.dock, self.STAGE, mark)
        if found is None:
            return None
        data = found[0]
        return {**json.loads(data), "result": hashlib.sha256(data).hexdigest()}

    def keep(self, mark, state, about) -> str:
        """Keep a step's checkpoint state, the file state (a reference {"key",
        "sha256"}); the sha256 of its result."""
        return set_result(self.dock, self.STAGE, mark, json.dumps({"about": about, "state": state},
                                                                   sort_keys=True).encode(), self.telemetry)


def learned(done) -> dict:
    """What of the corpus the steps learned from, together, by kind of
    texts: the most tokens of plain text one read from the start, and the
    most conversations and tasks one read (wpt_training.Environment.learned)."""
    found = {}
    for step in done:
        for part, count in step["learned"].items():
            found[part] = max(found.get(part, 0), count)
    return found


def provenance(args, order, record, model_config, n_params, worker, strategy_file, done, texts):
    """What produced the model; worker: what the gpu worker said of itself
    when it was ready (its device, its processes, its Python and the
    versions of the libraries it trained with); texts: how many texts of
    the corpus the steps learned from. A corpus the freight delivered
    gives `data`, the order's data section (the depot's address, the
    snapshot's name, change and digest, the roles, shares, sizes and
    vocabulary), `deliveries`, the key, id and preparation of the
    vocabulary's delivery and of the corpus's, and `corpus` its id,
    delivery key, rules, the snapshot it read (with its owner) and
    `request`, the request that makes it again: a corpus given by its id
    may have been made at a larger size than the order's, and holds other
    texts past the first one a smaller one leaves out. A corpus built by
    the depot from a snapshot gives `data` and `corpus` its id, build key,
    rules, snapshot and `data`, the data section it was built with."""
    processes = worker["processes"]
    if "delivery" in record:
        corpus = {"id": record["id"], "delivery": record["delivery"], "rules": record["rules"],
                  "depot": record["depot"], "snapshot": record["snapshot"], "request": record["request"]}
        source = {"data": order["data"], "deliveries": record["deliveries"]}
    else:
        corpus = {"id": record["id"], "build_key": record["build_key"], "rules": record["rules"],
                  "depot": record["depot"], "snapshot": record["snapshot"], "data": record["data"]}
        source = {"data": order["data"]}
    return {
        **source,
        "order_name": args.order,
        "order": order,  # everything that changes the model: data, architecture, seed, steps, drafter
        "corpus": corpus,  # a cache: the snapshot, its data section and the code build it again
        "learned": learned(done),  # what of the corpus the steps learned from
        "texts": texts,
        "vocabulary": record["tokenizer"]["vocabulary"],
        "shards": [{name: s[name] for name in SHARD_FIELDS} for s in record["shards"][PLAIN]],
        # how the hardware ran it, and whether a drafter was trained (drafter: false with --no-drafter)
        "options": {k: v for k, v in vars(args).items() if k not in ("order", "corpus")},
        "strategy": {"name": strategy_file.name, "file": strategy_file.file, "file_sha256": strategy_file.sha256,
                     "processes": processes},
        # each step in order: what it is, its fingerprint and kept result, whether this run reused it, what it
        # reached, and what of the corpus it learned from
        "steps": list(done),
        "model": asdict(model_config),
        "parameters": n_params,
        **repository(),  # repository_commit, repository_dirty
        "device": worker["device"] + (f", {processes} processes" if processes > 1 else ""),
        # the code (wic_utils.programs.code), the gpu worker's Python and libraries, which trained it; the
        # tokenizer's compiler is the manager's
        "tools": {"code": code(), **worker["tools"], "rustc": RUSTC_VERSION},
    }


def print_series(name) -> int:
    """Print the numbers of every size of the series file of that name (wpt train series NAME)."""
    orders = Orders.find()
    try:
        data = orders.data_part(f"{name}{SIZE_OF}")
        vocab_size = data["vocabulary"]["size"]  # the series names its vocabulary, with its size
        said = (f"the vocabulary {data['vocabulary']['sha256'] or '(not named yet)'} its data part names"
                if resources.vocabulary_named(data) else
                f"the size of the vocabulary its data part learns on {data['vocabulary']['sample_bytes']:,} bytes")
        plans = orders.series_plans(name, vocab_size)
        say(f"series {name}: a vocabulary of {vocab_size:,} tokens, {said}")
        total = orders.series_total_tokens(name)
        say(f"series {name}: one corpus of {total:,} training tokens; wpt train {name}:SIZE reads its first prefix tokens")
        for plan in plans:
            model = plan.model
            say(f"{plan.size}: width {model['d_model']}, {layer_count(model)} layers ({model['window_layers']} "
                f"window layers on each side of an exact layer, {model['sparse_groups']} groups of an exact layer "
                f"and {model['sparse_layers']} sparse layers), Engram at {', '.join(map(str, model['engram_layers']))}; "
                f"{plan.parameters:,} parameters without the vocabulary tables, {plan.total_parameters:,} with them; "
                f"M {plan.flops_per_token:.4g} FLOPs per token, C {plan.compute:.4g}, D {plan.tokens:.4g} tokens; "
                f"learning rate {plan.learning_rate:g}, batch {plan.batch_size} windows of {plan.block_size:,} "
                f"(the law: {plan.batch_tokens:,.0f} tokens), {plan.updates:,} updates, warm-up {plan.warmup:,}, "
                f"prefix {plan.prefix_tokens:,} tokens")
            for run in orders.series_runs(name, plan):
                say(f"  {name}:{plan.size}:{run['run']}: {run['updates']:,} updates"
                    + (f", decaying after update {run['start']:,} of the trunk" if run["start"] is not None else "")
                    + f", learning rate {run['learning_rate']:g}")
    except OrderError as error:
        say(f"InputError: {error}", ERROR)
        return REFUSED
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # the sample is any text; Windows pipes default to a narrower code page
    given = sys.argv[1:] if argv is None else list(argv)
    if given[:1] == ["series"]:
        if len(given) != 2:
            say("usage: wpt train series NAME", ERROR)
            sys.exit(REFUSED)
        code = print_series(given[1])
        if code:
            sys.exit(code)
        return
    environ = settings()
    args = parse_args(argv, environ)
    arguments = sys.argv[1:] if argv is None else list(argv)
    try:
        dock = dock_api.url(Path.cwd() / ".env")
        record = find_corpus(dock, args.corpus, load_data(args.order))
        check_record(record, load_order(args.order, record))
    except (DockError, InputError) as error:  # no dock, no corpus: nothing has started, nothing is recorded
        say(f"{type(error).__name__}: {error}", ERROR)
        sys.exit(REFUSED)
    try:
        with Telemetry.start(dock, "wpt train", arguments, short(record["id"]),
                             inputs={"order": args.order, "corpus": record["id"]},
                             env_file=Path.cwd() / ".env") as telemetry:
            code = _run(args, record, dock, telemetry, environ)
            if code == RECORD_FAILED:  # the gpu worker's lines of the record could not be written
                raise RecordError("the dock refused a line of this run's record that the gpu worker sent")
            if code:
                sys.exit(code)
    except RecordError as error:
        say(f"RecordError: {error}; stopped, nothing more was recorded", ERROR)
        sys.exit(RECORD_FAILED)


def settings() -> dict:
    """The settings this command reads: the environment, and for what it
    leaves unset or empty, the .env file of the current folder (as
    DOCK)."""
    return {**read_env_file(Path.cwd() / ".env"), **{key: value for key, value in os.environ.items() if value}}


def _run(args, record, dock, telemetry, environ) -> int:
    """The run; its exit code."""
    try:
        with telemetry.span("inputs.order", order=args.order):
            order = with_prefixes(load_order(args.order, record), record,
                                   lambda text: telemetry.log(text, WARNING))
            strategy_file = strategy_plugin(args.strategy)
        with telemetry.span("inputs.steps", steps=len(order["steps"])):
            steps, context = plan(order)
    except (InputError, PluginError) as error:
        return _refuse(f"{type(error).__name__}: {error}", telemetry)
    try:
        return _train(args, record, order, strategy_file, steps, context, dock, telemetry, environ)
    except Refused as error:
        return _refuse(str(error), telemetry)


def _refuse(text, telemetry) -> int:
    telemetry.log(text, ERROR)
    return REFUSED


def fetched_state(dock, state: dict, folder, only=None) -> dict:
    """The checkpoint state the dock (its URL, dock) holds as the file state
    ({"key", "sha256"}), fetched into folder (a local folder of the
    caller's own) and checked against its sha256, loaded (only: as
    wpt_training.load_state takes it), and its local file removed."""
    folder = Path(folder)
    local = folder / f"state-{state['sha256']}.{os.getpid()}"
    try:
        dock_api.fetch_into(dock, state["key"], local, state["sha256"])
        return load_state(local, only=only)
    finally:
        local.unlink(missing_ok=True)


class DockCheckpoint(Checkpoint):
    """The checkpoint of a named run on the dock: a record beside the
    model NAME (wpt.models.CHECKPOINT) names the state's file
    ({"state": {"key", "sha256"}}), which lies in the same folder, so any
    machine using the same dock continues it and emptying the dock's
    cache folder loses nothing of it. Each save replaces the record and
    deletes the file it named. Every change of the record and every deletion are recorded in
    the run's record, naming the file, and that line is on the dock before
    the step (RecordError otherwise). When the record breaks, a save still
    writes the checkpoint's, unrecorded, then raises the RecordError;
    nothing is deleted then: the previous file stays. kept: the keys of the
    files that are also a kept step's state (StepResults), which only stop
    being named, and are never deleted. dock: the dock's URL; folder: a
    local folder of the caller's own, where a state is written before it
    goes to the dock and fetched before it is loaded (default: the system's
    temporary folder)."""

    def __init__(self, dock, name, telemetry, kept: set, folder=None):
        self.dock, self.name, self.telemetry, self.kept = dock, name, telemetry, kept
        self.folder = Path(folder) if folder is not None else Path(tempfile.gettempdir())
        self.key = checkpoint_key(name)

    def __str__(self):
        return f"{self.dock} as {self.key}"

    def _named(self, state):
        dock_api.put_json(self.dock, self.key, {"state": state})

    def exists(self):
        return self.state_file() is not None

    def load(self):
        state = self.state_file()
        if state is None:
            return None
        return fetched_state(self.dock, state, self.folder)

    def save(self, state):
        self.folder.mkdir(parents=True, exist_ok=True)
        partial = self.folder / f"checkpoint-{self.name}.{os.getpid()}.partial"
        try:
            save_state(state, partial)
            self.save_file(partial)
        finally:
            partial.unlink(missing_ok=True)

    def save_file(self, path):
        stored = dock_api.put_from(self.dock, state_key(self.name, sha256_file(path)), path)
        state = {"key": stored["key"], "sha256": stored["sha256"]}
        previous = self.state_file()
        try:
            self.telemetry.irreversible("training.checkpoint_saved", state=state["key"], checkpoint=str(self))
        except RecordError:
            self._named(state)  # the checkpoint is saved all the same, unrecorded
            raise
        self._named(state)  # the previous one stays until this
        if previous and previous != state and previous["key"] not in self.kept:
            self._record_deletion(previous)
            dock_api.delete(self.dock, previous["key"])

    def remove(self):
        state = self.state_file()
        deleted = state if state and state["key"] not in self.kept else None
        if deleted:
            self._record_deletion(deleted)
        dock_api.delete(self.dock, self.key)
        if deleted:
            dock_api.delete(self.dock, deleted["key"])

    def state_file(self) -> dict | None:
        """The reference of the file of the saved state, None when none is saved."""
        record = dock_api.get_json(self.dock, self.key)
        return None if record is None else record["state"]

    def _record_deletion(self, state):
        self.telemetry.irreversible("training.checkpoint_deleted", state=state["key"], checkpoint=str(self))


class RestoreFailed(Exception):
    """A folder moved aside to make room for a new model could not be moved
    back after the new one failed to take its place: the target is missing,
    the earlier folder is at `aside`, the new one at `new`."""

    def __init__(self, target, aside, new, cause):
        super().__init__(f"{target} could not be replaced ({cause}), and the earlier folder, moved to {aside}, could "
                         f"not be moved back; the new model is in {new}: move one of them to {target} by hand")
        self.target, self.aside, self.new = target, aside, new


def _replace_folder(new, target: Path, telemetry):
    """Rename the folder new to target. A folder already at target is moved
    aside first and removed once new is in place. Moving aside and putting
    new in place are one step: on any failure or interruption between them,
    the earlier folder is moved back (and new with it, to where it was)
    before the error goes on, or RestoreFailed names both folders. Once new
    is in place, an earlier folder that cannot be removed, or whose removal
    is interrupted, stays beside the target, with a warning that names it
    (an interruption then goes on)."""
    if not target.exists():
        os.replace(new, target)
        return
    aside = target.parent / f".{target.name}-replaced-{secrets.token_hex(4)}"
    try:
        os.replace(target, aside)
        os.replace(new, target)
    except BaseException as error:
        if aside.exists():  # moved aside: put it back, whatever happened since
            try:
                if target.exists():
                    os.replace(target, new)
                os.replace(aside, target)
            except BaseException as failed:
                raise RestoreFailed(target, aside, new, f"{type(error).__name__}: {error}") from failed
        raise
    try:
        shutil.rmtree(aside)
    except (OSError, KeyboardInterrupt) as error:
        telemetry.event("model.replaced_folder_left", folder=str(aside), error=_describe(error))
        telemetry.log(f"warning: the earlier {target}, moved to {aside}, could not be removed "
                      f"({_describe(error)}): remove it by hand", WARNING)
        if isinstance(error, KeyboardInterrupt):
            raise


def _describe(error):
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


@dataclass
class _Run:
    """What the manager knows of a run, which its answers to the gpu
    worker read (wpt.train.remote.Training)."""

    args: argparse.Namespace
    order: dict
    steps: list
    record: dict
    dock: str  # the dock's URL
    files: DockFiles  # the corpus's files and the states, read from the dock into the run's work folder
    telemetry: Telemetry
    tokenizer: Tokenizer
    model_config: ModelConfig
    checkpoint: Checkpoint  # a DockCheckpoint with --run, else a FolderCheckpoint in the model folder
    kept_states: set  # the keys of the kept steps' state files, which a dock checkpoint never deletes
    results: StepResults
    strategy_file: object
    replayed: bytes | None  # the scheduler's pass times an earlier run recorded, with --scheduler-costs
    log: object
    done: list = field(default_factory=list)
    setup: dict | None = None  # what the pass times are measured on, once the job is ready
    parameters: int | None = None

    def fingerprints(self, code) -> list[str]:
        return fingerprints(self.steps, self.model_config, self.record, self.order["seed"], code)

    def checkpoint_state(self) -> dict | None:
        """With a dock checkpoint, the reference of its saved state's file
        (None when none is saved); with a folder checkpoint, None: its file
        is checkpoint_file()."""
        return self.checkpoint.state_file() if isinstance(self.checkpoint, DockCheckpoint) else None

    def checkpoint_file(self) -> Path | None:
        """With a folder checkpoint, its saved file (None when none is
        saved); with a dock checkpoint, None."""
        if isinstance(self.checkpoint, DockCheckpoint) or not self.checkpoint.path.is_file():
            return None
        return self.checkpoint.path

    def kept_state(self) -> dict:
        """The reference of the file of the step's last checkpoint, which
        becomes its kept state: a dock checkpoint's own file, or a folder
        checkpoint's file put onto the dock's cache folder (where a state
        lies that nothing but the step's result names)."""
        if isinstance(self.checkpoint, DockCheckpoint):
            return self.checkpoint.state_file()
        stored = dock_api.put_content_from(self.dock, self.checkpoint.path)
        return {"key": stored["key"], "sha256": stored["sha256"]}


def link_checkpoints(environ) -> str:
    """WPT_LINK_CHECKPOINTS: every (each checkpoint the gpu worker writes
    comes to the manager's checkpoint) or step-ends (the default: only each
    step's last, with its end)."""
    found = environ.get(LINK_CHECKPOINTS, STEP_ENDS)
    if found not in CHECKPOINTS:
        raise Refused(f"{LINK_CHECKPOINTS}={found!r}: one of {', '.join(CHECKPOINTS)}")
    return found


def sent_files(steps, strategy_file) -> dict:
    """{the path an order or an option gave: its text} of every file of the
    run's parts that is not a built-in: what the gpu worker loads them from."""
    files = [plugin for step in steps for plugin in step.files()]
    if strategy_file.given is not None:
        files.append(strategy_file)
    return {plugin.given: Path(plugin.given).read_text(encoding="utf-8") for plugin in files}


def _train(args, record, order, strategy_file, steps, context, dock, telemetry, environ) -> int:
    """The run, with its work folder: beside the model folder's target, for
    the files it fetches from the dock (the corpus's files the gpu workers
    read, the states it loads) and the checkpoints it writes before they go
    to the dock; removed when the run ends."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{out.name}-work-", dir=out.parent))
    try:
        return _train_in(args, record, order, strategy_file, steps, context, dock, telemetry, environ, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _train_in(args, record, order, strategy_file, steps, context, dock, telemetry, environ, work) -> int:
    log = telemetry.log
    out = Path(args.out)
    out_was_there = out.exists()  # a failed step then leaves the target as it was
    mode = link_checkpoints(environ)
    corpus_files = DockFiles(dock, work)
    try:
        with telemetry.span("inputs.corpus", corpus=record["id"], stream=args.stream) as span:
            kept_states = set()  # the keys of the kept steps' state files, which a dock checkpoint never deletes
            checkpoint = (DockCheckpoint(dock, args.run, telemetry, kept_states, work) if args.run
                          else FolderCheckpoint(out / CHECKPOINT_FOLDER))
            if checkpoint.exists() and not args.resume:
                raise Refused(f"{checkpoint} holds an unfinished step: add --resume to continue it, or delete it")
            tokenizer = load_corpus(dock, record, order)
            check_corpus(steps, record, corpus_files)  # what every step reads is there before the first one trains
            span.set(shards=len(record["shards"][PLAIN]), conversations=record["conversations"],
                     tasks=record.get("tasks", {}).get("count", 0))
        telemetry.input(order=args.order, corpus=record["id"], vocabulary=record["tokenizer"]["vocabulary"]["sha256"])
    except (DockError, InputError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    log(f"corpus {record['id']}: vocabulary {tokenizer.vocab_size:,} tokens")

    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=context, **order["model"])
    try:  # the drafter trained at the end, and the scheduler's pass times replayed: refused now, not after training
        drafter_config, _ = drafter_settings(model_config, order["drafter"])  # its feature layers included
        replayed = dock_api.read_record(dock, args.scheduler_costs, COSTS_FILE) if args.scheduler_costs else None
        files = sent_files(steps, strategy_file)
    except (ValueError, DockError, OSError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    run = _Run(args=args, order=order, steps=steps, record=record, dock=dock, files=corpus_files, telemetry=telemetry,
               tokenizer=tokenizer, model_config=model_config, checkpoint=checkpoint, kept_states=kept_states,
               results=StepResults(dock, telemetry), strategy_file=strategy_file, replayed=replayed, log=log)
    spec = {"task": "wpt-train", "files": files, "steps": order["steps"], "record": record,
            "model": asdict(model_config), "seed": order["seed"], "drafter": order["drafter"],
            "canonical": canonical_ids(tokenizer.decode, tokenizer.vocab_size).tolist(),
            "tokens": {"end": Tokenizer.END, "im_end": chat.IM_END}, "checkpoints": mode,
            "checkpoint": str(checkpoint), "join": telemetry.child_environment(live=False),
            "options": {"strategy": args.strategy, "processes": args.processes, "max_restarts": args.max_restarts,
                        "micro_batch": args.micro_batch, "checkpoint_every": args.checkpoint_every,
                        "segment": args.segment, "stream": args.stream, "stream_shards": args.stream_shards,
                        "recompute": args.recompute, "recompute_from": args.recompute_from,
                        "compile": args.compile, "resume": args.resume, "drafter": args.drafter,
                        "scheduler_costs": args.scheduler_costs, "kernels": args.kernels,
                        "stop_after_minutes": args.stop_after_minutes,
                        "stop_after_updates": args.stop_after_updates, "profile_update": args.profile_update}}
    try:
        checkout_root = code_root()
        placed = machines(keys(environ))  # refused before any worker is reached
        with telemetry.span("gpu_worker.start") as span:
            workers = start_workers(environ, log=log)
            span.set(workers=[str(worker) for worker in workers])
    except (NoCheckout, ValueError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    except LinkError as error:
        return _lost(error, telemetry)
    answers = Training(run, workers, Files(Path(tempfile.mkdtemp(prefix="wpt-train-link-"))), Refused)
    try:
        with telemetry.span("gpu_worker.checkout", workers=len(workers)):
            for worker in workers:
                worker.send_checkout(Orders(checkout_root).data_folder(), checkout_root)
        with telemetry.span("gpu_worker.job", checkpoints=mode, workers=len(workers),
                            transport=placed[0]["transport"]), telemetry.relay():
            code = run_jobs(workers, [{**spec, "machines": where} for where in placed], answers.handlers)
        if len(workers) > 1 and code not in (0, REFUSED, RECORD_FAILED, STOPPED):
            code = _crashed(code, len(workers), telemetry)
        if code == 0:
            _finish(args, order, record, model_config, drafter_config, answers, tokenizer, checkpoint, out, dock,
                    telemetry, log, work)
    except LinkError as error:
        code = _lost(error, telemetry)
    finally:
        answers.close()
        shutil.rmtree(answers.files.folder, ignore_errors=True)
        for worker in workers:
            worker.close()
    if code == REFUSED and not out_was_there and out.is_dir() and not any(out.iterdir()):
        out.rmdir()  # the folder the checkpoint alone made goes with it
    return code


def _crashed(code, workers, telemetry) -> int:
    """A training process crashed on one of several gpu workers: recorded,
    and the exit code that says so."""
    telemetry.event("gpu_worker.crashed", exit_code=code, workers=workers)
    telemetry.log(f"a training process crashed (exit code {code}) and the jobs on every one of the {workers} gpu "
                  "workers were stopped; what the manager holds stays: --resume continues from its checkpoint",
                  ERROR)
    return CRASHED


def _lost(error, telemetry) -> int:
    """A gpu worker lost: recorded, and the exit code that says so."""
    telemetry.event("gpu_worker.lost", error=str(error))
    telemetry.log(f"gpu worker lost: {error}; what the manager holds stays: --resume continues from its checkpoint",
                  ERROR)
    return LOST


def _finish(args, order, record, model_config, drafter_config, answers, tokenizer, checkpoint, out, dock,
            telemetry, log, work):
    """The model folder, from the last step's kept state and what the gpu
    worker sent at the end (the drafter, none with --no-drafter, and a
    sample's token ids)."""
    with telemetry.span("model.load", state=answers.last_state):
        model = WPT(model_config)  # on this machine's CPU: only written to the folder
        state = fetched_state(dock, answers.last_state, work, only=("model", "best_model"))
        model.load_state_dict(weights(state))
        del state
        drafter = None
        if answers.finished["drafter"] is not None:
            found = load_state(answers.finished["drafter"])
            drafter = Drafter(model_config, drafter_config)
            drafter.load_state_dict(found["state"])
            drafter.provenance = {**found["provenance"], "corpus": record["id"],
                                  "trained_by": {"command": "wpt train", "order_name": args.order}}
            answers.finished["drafter"].unlink()
    n_params = sum(p.numel() for p in model.parameters())
    done = answers.done
    named = record["tokenizer"]
    # written beside the target, then renamed: a failure midway leaves the target as it was
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    renaming = False
    try:
        with telemetry.span("model.write", folder=str(out)):
            with telemetry.span("model.weights"):
                vocabulary = dock_api.fetch(dock, named["vocabulary"]["key"], named["vocabulary"]["sha256"])
                tokenizer_name = named["recipe"] + (f" version {named['recipe_version']}"
                                                    if named.get("recipe_version") is not None else "")
                save_model(model, staging, vocabulary, tokenizer_name, max_shard_bytes=args.max_shard_size,
                           metadata=WEIGHT_METADATA)
                if drafter is not None:
                    save_drafter(drafter, staging, tokenizer_name, hashlib.sha256(vocabulary).hexdigest(),
                                 metadata=WEIGHT_METADATA,
                                 provenance={**drafter.provenance, "target": {"weights": weight_hashes(staging)}})
            with telemetry.span("model.notice") as span:  # every text learned from, NOTICE, LICENSE, the model card
                credits = model_notice(dock, record, learned(done))
                (staging / CREDITS_NAME).write_text(credits, encoding="utf-8", newline="\n")
                texts = sum(1 for line in credits.split("\n\n", 1)[0].splitlines() if line)
                span.set(texts=texts)
            with telemetry.span("model.provenance"):
                written = provenance(args, order, record, model_config, n_params, answers.ready_found,
                                     answers.run.strategy_file, done, texts)
                (staging / "provenance.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8",
                                                         newline="\n")
            write_documents(staging)
            files = file_hashes(staging)
            telemetry.event("model.files", files=files)
            # recorded before the folder appears: a record that cannot be written leaves the target as it was
            telemetry.output(model_folder=str(out), weights={name: files[name] for name in weight_files(staging)},
                             provenance=files["provenance.json"], drafter=files.get("drafter/model.safetensors"))
            # the run is complete; recorded before the checkpoint goes (a FolderCheckpoint goes with the old folder)
            telemetry.event("training.checkpoint_removed", checkpoint=str(checkpoint))
            telemetry.irreversible("model.folder_replaced", folder=str(out))
            renaming = True
            _replace_folder(staging, out, telemetry)
    except BaseException as error:
        # decided by the folders, not by the error: a RestoreFailed may arrive inside a RecordError
        if renaming and not out.exists() and staging.exists():
            if not isinstance(error, RestoreFailed):  # its own message already names both folders
                earlier = sorted(out.parent.glob(f".{out.name}-replaced-*"))
                say(f"{out} is missing: the new model is kept in {staging}"
                    + "".join(f", the earlier one in {folder.absolute()}" for folder in earlier)
                    + f"; move one of them to {out} by hand", ERROR)
        elif (renaming and not args.run and staging.exists() and out.is_dir()
              and {path.name for path in out.iterdir()} <= {CHECKPOINT_FOLDER}):
            # the target holds only what this run's checkpoint left there: no other model to go back to
            say(f"the new model could not take the place of {out}, which holds no other model: it is kept in "
                f"{staging}; move it to {out} by hand", ERROR)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    checkpoint.remove()  # a checkpoint left by a step this run reused
    log(f"saved {out} (licenses and the texts it learned from: {out / 'NOTICE'})")
    # the sample the gpu worker wrote, so you immediately see what it learned
    log(tokenizer.decode(answers.finished["sample"]).decode("utf-8", errors="replace"))
