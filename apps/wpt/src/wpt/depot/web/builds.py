"""The routes of `wpt depot web` over what the depot builds: its tokenizer
recipes and their vocabularies, the jobs' queue (wpt.depot.jobs), a
dataset's corpus built, the dataset's downloads, and the sample route
another depot's vocabulary training reads.

Anyone may queue a vocabulary training or a corpus build; only the command
line cancels a job. "Build" on a dataset queues its corpus build, after a
vocabulary training of its tokenizer recipe when the recipe has no
vocabulary of its current version (the training already open for it, when
one is). A deleted corpus is built again from its own record, and the
build says whether its id changed.

A dataset's download (GET /api/datasets/{id}/download, the page
#/download/<id>) never changes its address: it gives the latest fresh
corpus built from the dataset's current revision, or a state with nothing
to download: "checking an erasure" (an erasure decided here is being
carried into the corpora's records, wpt.depot.downloads; no corpus is
sent meanwhile), "being built" (the job's progress), "stale" (an admitted
erasure reached a text of the latest corpus; nothing builds it again by
itself) or "not built". A built corpus gives each of its files, streamed
from the store with Range support, and its texts as numbered parts made
when asked (wpt_depot.exports), each part's size once it was sent whole.
At most downloads.EXPORTS text exports run at once in all, and
downloads.VISITOR_EXPORTS for one visitor (429 past them).

The job operations of this module (train_job, build_job) run in the
queue's process; each takes the depot, its arguments, progress and job.
"""

import hashlib
import queue
import re
import shutil
import threading
from contextlib import closing
from functools import wraps

from wpt_depot import Depot, corpora, errors, exports
from wpt_depot.corpus import sample, sample_lines
from wpt_store import StoreError, store_from_settings
from wpt_store.names import File
from wpt_utils.lines import say

from .. import shared
from ..downloads import EXPORTS, VISITOR_EXPORTS, Downloads
from ..jobs import Job, Jobs, work_folder
from .server import Request, Route, Stream

CHUNK = 8 << 20  # bytes of a stored file read at a time to send it
PROGRESS_SECONDS = 5  # seconds between two lines of progress of a sample being ordered
RANGE = re.compile(r"bytes=(\d*)-(\d*)")
TEXT_PART = re.compile(r"part-(\d{5})\.(jsonl\.zst|parquet)")


class _Stopped(Exception):
    """The stream a sample is made for was closed: the sample stops there."""


class Guarded:
    """The chunks of a stream whose guard (a context manager entered
    already) is left once, when the server closes the stream, sent or not;
    closed (default: nothing) is called then too."""

    def __init__(self, chunks, guard, closed=lambda: None):
        self.chunks, self.guard, self.closed, self.left = chunks, guard, closed, False

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        return next(self.chunks)

    def close(self):
        if not self.left:
            self.left = True
            try:
                self.closed()
                getattr(self.chunks, "close", lambda: None)()
            finally:
                self.guard.__exit__(None, None, None)


# The job operations, which the queue's process imports by name

def train_job(depot: Depot, name: str, *, progress, job: Job) -> dict:
    """Train the vocabulary of the current version of the tokenizer recipe named so: {vocabulary: its sha256}."""
    job.work.mkdir(parents=True, exist_ok=True)
    return {"vocabulary": shared.train_vocabulary(depot, name, say, work=job.work, progress=progress)}


def build_job(depot: Depot, dataset: str, corpus: str | None = None, *, progress, job: Job) -> dict:
    """Build the corpus of the dataset of that id as it is now, or, with
    corpus, the deleted corpus of that id again, into the store WPT_STORE
    names, under the cap WPT_CORPORA_CAP sets, going on from the
    dataset's checkpoint: {corpus, texts, tokens, changed}."""
    store = store_from_settings(shared.env_file())
    downloads = Downloads(job.path)
    cap = shared.corpora_cap()
    record = None
    if corpus is not None:
        record = corpora.find(store, corpus)
        if record is None:
            raise errors.NotFound(f"no corpus {corpus} in the store")
    work = job.work / f"corpus-{dataset}"
    built = shared.build_corpus(
        depot, dataset, say, store=store, record=record, work=work, checkpoint=job.checkpoint(), save=job.save,
        progress=progress,
        room=lambda folder: corpora.Room(store, cap, folder, downloads.last_download, downloads.busy, say))
    job.save(None)
    shutil.rmtree(work, ignore_errors=True)
    return {"corpus": built["id"], "texts": built["texts"], "tokens": built["total_tokens"],
            "changed": built.get("changed")}


def builds(depot: Depot, jobs: Jobs, refusals) -> list[Route]:
    """The routes over depot's recipes, jobs, builds and downloads;
    refusals wraps a handler so that it answers each failure with its
    status (wpt.depot.web.routes.refusals)."""
    downloads = Downloads(jobs.path)
    held = {}
    sampling = threading.Lock()
    shared.settle_marks(depot, downloads, say)  # the marking a stop of the web app cut short

    def store():
        if "store" not in held:
            held["store"] = store_from_settings(shared.env_file())
        return held["store"]

    def wrapped(handler):
        """refusals for a handler that gets the Request too."""
        @wraps(handler)
        def answer(values, body, request):
            return refusals(lambda values, body: handler(values, body, request))(values, body)
        return answer

    # Tokenizer recipes and their vocabularies

    def recipe_shown(recipe: dict) -> dict:
        dataset = None
        if recipe["depot"] is None:
            try:
                dataset = depot.dataset_by_id(recipe["dataset"])["name"]
            except errors.NotFound:
                pass
        return {**recipe, "dataset_name": dataset if recipe["depot"] is None else recipe["dataset"],
                "job": jobs.open_job("vocabulary", recipe=recipe["name"])}

    @refusals
    def tokenizers(values, body):
        return 200, shared.jsonable([recipe_shown(one) for one in depot.tokenizers()])

    @refusals
    def tokenizer(values, body):
        recipe = depot.tokenizer(values["name"])
        revisions = depot.rows("SELECT * FROM tokenizers_revisions WHERE id = ? ORDER BY revision DESC",
                               (recipe["id"],))
        vocabularies = depot.rows("SELECT recipe, version, change, sha256, texts, bytes, made_at FROM vocabularies "
                                  "WHERE recipe = ? ORDER BY version DESC", (recipe["id"],))
        upstream = bool(depot.upstream_owned([recipe["id"]]))
        return 200, shared.jsonable({"recipe": recipe_shown(recipe), "revisions": revisions,
                                     "vocabularies": vocabularies, "upstream": upstream,
                                     "trained": any(one["version"] == recipe["version"] for one in vocabularies)})

    def recipe_fields(body) -> tuple:
        return (body.get("depot") or None, body["dataset"], body["sample_bytes"], body["vocab_size"],
                body.get("note"))

    @refusals
    def new_tokenizer(values, body):
        return 200, {"id": depot.new_tokenizer(body["name"], *recipe_fields(body))}

    @refusals
    def edit_tokenizer(values, body):
        recipe = depot.tokenizer(values["name"])
        return 200, {"version": depot.edit_tokenizer(recipe["id"], *recipe_fields(body))}

    def vocabulary_row(values) -> tuple[dict, dict]:
        recipe = depot.tokenizer(values["name"])
        try:
            version = int(values["version"])
        except ValueError:
            raise errors.NotFound(f"no version {values['version']} of the tokenizer recipe {recipe['name']}") from None
        found = depot.vocabulary(recipe["id"], version)
        if found is None:
            raise errors.NotFound(f"the tokenizer recipe {recipe['name']} has no vocabulary of its version {version}")
        return recipe, found

    @refusals
    def vocabulary(values, body):
        recipe, found = vocabulary_row(values)
        return 200, shared.jsonable({key: value for key, value in found.items() if key != "merges"}
                                    | {"name": recipe["name"]})

    @refusals
    def vocabulary_file(values, body):
        _, found = vocabulary_row(values)
        data = found["merges"].encode("utf-8")
        return 200, Stream("application/octet-stream", iter([data]), len(data),
                           {"Content-Disposition": f'attachment; filename="vocabulary-{found["sha256"][:12]}"'})

    @wrapped
    def vocabulary_texts(values, body, request: Request):
        recipe, found = vocabulary_row(values)
        offset, limit = int(request.query.get("offset", 0)), min(int(request.query.get("limit", 50)), 1000)
        return 200, depot.vocabulary_texts(recipe["id"], found["version"], offset, limit)

    def queue_training(name: str) -> dict:
        """The vocabulary job of the recipe named so: the one open, else a new one."""
        return jobs.open_job("vocabulary", recipe=name) or jobs.add(
            "vocabulary", f"train the vocabulary of the tokenizer recipe {name}", train_job, name, recipe=name)

    @refusals
    def train(values, body):
        recipe = depot.tokenizer(values["name"])
        if depot.vocabulary(recipe["id"], recipe["version"]) is not None:
            raise errors.Refused(f"the tokenizer recipe {recipe['name']} has a vocabulary of its version "
                                 f"{recipe['version']} already")
        return 200, queue_training(recipe["name"])

    # Jobs

    @refusals
    def jobs_listed(values, body):
        return 200, jobs.listed()

    @refusals
    def job_state(values, body):
        return 200, shared.jsonable(jobs.state(values["id"]))

    # Building a corpus

    @refusals
    def build(values, body):
        row = depot.dataset_by_id(values["id"])
        corpus = (body or {}).get("corpus")
        record = None
        if corpus is not None:
            try:
                record = corpora.find(store(), corpus)
            except ValueError as error:  # a start of the ids of several corpora
                raise errors.BadQuery(str(error)) from None
            if record is None or record["dataset"]["id"] != row["id"]:
                raise errors.NotFound(f"no corpus {corpus} of the dataset {row['name']}")
            if "deleted" not in record:
                raise errors.Refused(f"the corpus {corpus} keeps its files: nothing to build again")
        name = row["tokenizer"] if record is None else record["tokenizer"]["recipe"]
        try:
            recipe = depot.tokenizer(name)
        except errors.NotFound:
            raise errors.Refused(f"the depot holds no tokenizer recipe {name}: make it on the Tokenizers page") from None
        after = None
        if record is None and depot.vocabulary(recipe["id"], recipe["version"]) is None:
            after = queue_training(name)["id"]
        elif record is not None and depot.vocabulary(recipe["id"], record["tokenizer"]["recipe_version"]) is None:
            raise errors.Refused(f"the vocabulary of version {record['tokenizer']['recipe_version']} of {name}, "
                                 "which built that corpus, is not in the depot")
        title = (f"build the corpus of the dataset {row['name']}" if record is None
                 else f"build the corpus {record['id'][:12]} of the dataset {row['name']} again")
        started = jobs.add("corpus", title, build_job, row["id"], corpus, dataset=row["id"], after=after)
        return 200, started

    # Downloads

    def latest(row: dict) -> tuple[dict | None, dict | None]:
        """(the latest fresh corpus of the dataset's current revision, the latest corpus of that revision)."""
        found = [one for one in corpora.built_from(store(), shared.corpus_depot(depot, row), row["id"])
                 if one["dataset"]["revision"] == row["revision"]]
        return next((one for one in found if corpora.fresh(one)), None), next(iter(found), None)

    def file_name(record: dict, one: dict) -> str:
        return one["key"].removeprefix(corpora.folder_of(record) + "/")

    @refusals
    def download(values, body):
        row = depot.dataset_by_id(values["id"])
        job = jobs.open_job("corpus", dataset=row["id"])
        before = None if job is None or job["after"] is None else jobs.state(job["after"])
        shown = {"dataset": shared.jsonable(row), "job": job, "before": before}
        marking = downloads.pending()
        if marking:
            return 200, {**shown, "state": "checking an erasure", "erasures": marking}
        try:
            fresh, last = latest(row)
        except (StoreError, OSError) as error:
            return 200, {**shown, "state": "no store", "problem": f"{type(error).__name__}: {error}"}
        if fresh is None:
            if job is not None:
                state = "being built"
            elif last is not None and "stale" in last:
                state = "stale"
            else:
                state = "not built"
            return 200, {**shown, "state": state, "corpus": None if last is None else {
                key: last.get(key) for key in ("id", "built_at", "stale", "deleted")}}
        files = [{"name": file_name(fresh, one), "bytes": one.get("bytes"), "sha256": one["sha256"]}
                 for one in corpora.files(fresh)]
        parts = []
        for number, one in enumerate(fresh["tables"]["texts"]):
            parts.append({"part": number, "texts": one.get("rows"), "text_bytes": one.get("text_bytes"),
                          "forms": {form: {"name": f"part-{number:05d}.{form}",
                                           **(downloads.export(fresh["id"], number, form) or
                                              {"bytes": None, "sha256": None})} for form in exports.FORMS}})
        return 200, {**shown, "state": "built",
                     "corpus": {key: fresh.get(key) for key in ("id", "built_at", "texts", "conversations",
                                                                "total_tokens", "tokenizer")},
                     "files": files, "parts": parts}

    def fresh_corpus(values) -> dict:
        row = depot.dataset_by_id(values["id"])
        marking = downloads.pending()
        if marking:
            raise errors.Refused(f"the corpora holding a text of the erasures {', '.join(marking)} are being marked "
                                 "stale: no corpus is sent until that ends")
        found, _ = latest(row)
        if found is None:
            raise errors.NotFound(f"the dataset {row['name']} has no corpus to download now: see its download page")
        return found

    @wrapped
    def download_file(values, body, request: Request):
        record = fresh_corpus(values)
        chosen = next((one for one in corpora.files(record) if file_name(record, one) == values["name"]), None)
        if chosen is None:
            raise errors.NotFound(f"corpus {record['id']} has no file {values['name']}")
        reference = File.of(chosen)
        size = store().size(reference)
        start, end, status = 0, size, 200
        asked = request.headers.get("Range")
        if asked:
            match = RANGE.fullmatch(asked.strip())
            if match is None or (not match[1] and not match[2]):
                return 416, {"problems": [f"the Range {asked} is not one range of bytes"]}
            if match[1]:
                start, end = int(match[1]), min(int(match[2]) + 1 if match[2] else size, size)
            else:
                start = max(size - int(match[2]), 0)
            if start >= end:
                return 416, {"problems": [f"the Range {asked} is outside the file's {size:,} bytes"]}
            status = 206

        guard = downloads.streaming(record["id"], request.visitor, "file")
        guard.__enter__()

        def chunks():
            at = start
            while at < end:
                length = min(CHUNK, end - at)
                yield store().read_range(reference, at, length)
                at += length
        headers = {"Accept-Ranges": "bytes", "ETag": f'"{chosen["sha256"]}"',
                   "Content-Disposition": f'attachment; filename="{values["name"]}"'}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{size}"
        return status, Stream("application/octet-stream", Guarded(chunks(), guard), end - start, headers)

    @wrapped
    def download_text(values, body, request: Request):
        record = fresh_corpus(values)
        match = TEXT_PART.fullmatch(values["name"])
        if match is None or int(match[1]) >= len(record["tables"]["texts"]):
            raise errors.NotFound(f"corpus {record['id']} has no part of texts {values['name']}: its parts are "
                                  f"part-00000 to part-{len(record['tables']['texts']) - 1:05d}, each .jsonl.zst "
                                  "or .parquet")
        number, form = int(match[1]), match[2]
        known = downloads.export(record["id"], number, form)
        guard = downloads.streaming(record["id"], request.visitor, "text")
        guard.__enter__()  # TooMany before anything is sent

        def chunks():
            digest, size = hashlib.sha256(), 0
            for chunk in exports.part(depot, store(), record, number, form):
                digest.update(chunk)
                size += len(chunk)
                yield chunk
            if known is None:
                downloads.exported(record["id"], number, form, size, digest.hexdigest())
        return 200, Stream(exports.FORMS[form], Guarded(chunks(), guard), None if known is None else known["bytes"],
                           {"Content-Disposition": f'attachment; filename="{values["name"]}"'})

    # The sample another depot's vocabulary training reads

    @wrapped
    def sample_route(values, body, request: Request):
        try:
            size = int(request.query["bytes"])
        except (KeyError, ValueError):
            raise errors.BadQuery("name the sample's size in bytes: ?bytes=N") from None
        if size < 1:
            raise errors.BadQuery("a sample holds at least one byte")
        try:
            row = depot.dataset_by_id(values["dataset"])
        except errors.NotFound:
            row = depot.dataset(values["dataset"])
        guard = downloads.streaming(None, request.visitor, "sample")
        guard.__enter__()  # TooMany before anything is sent
        waiting = queue.Queue(maxsize=64)
        stop = threading.Event()  # set once the server closes the stream: the client went away, or it ended

        def offer(item):
            """Hand item to the stream; _Stopped once the stream is closed, so the producer lets go of the
            sampling lock instead of waiting on a queue nobody reads."""
            while True:
                if stop.is_set():
                    raise _Stopped
                try:
                    waiting.put(item, timeout=PROGRESS_SECONDS)
                    return
                except queue.Full:
                    continue

        def produce():
            folder = work_folder(depot.address) / f"sample-{row['id']}"
            try:
                with sampling:  # one sample at a time keeps one order file per dataset whole
                    if stop.is_set():
                        return
                    texts = sample(Depot.view(depot.address), row["name"], size, folder,
                                   lambda state: offer((None, state["texts read"], b"")))
                    with closing(texts):
                        for found in texts:
                            offer(found)
                offer(StopIteration())
            except _Stopped:
                pass
            except Exception as error:
                try:
                    offer(error)
                except _Stopped:
                    pass

        threading.Thread(target=produce, daemon=True).start()

        def texts():
            while True:
                try:
                    found = waiting.get(timeout=PROGRESS_SECONDS)
                except queue.Empty:
                    yield None, 0, b""
                    continue
                if isinstance(found, StopIteration):
                    return
                if isinstance(found, Exception):
                    raise found
                yield found

        return 200, Stream("application/octet-stream", Guarded(sample_lines(texts()), guard, stop.set))

    recipe_body = {"type": "object", "required": ["dataset", "sample_bytes", "vocab_size"],
                   "properties": {"depot": {"type": ["string", "null"], "description": "absent or null: this depot; "
                                            "else the web address of another depot's web app"},
                                  "dataset": {"type": "string", "minLength": 1, "description": "this depot's dataset "
                                              "(its name or id), or the other depot's dataset, by its name"},
                                  "sample_bytes": {"type": "integer", "minimum": 1},
                                  "vocab_size": {"type": "integer", "minimum": 513},
                                  "note": {"type": ["string", "null"]}}}
    job_answer = {"type": "object", "required": ["id", "kind", "title", "state"]}
    return [
        Route("GET", "/api/tokenizers", tokenizers, "every tokenizer recipe of the depot: its current revision (name, "
              "version, depot and dataset its vocabulary trains on, sample_bytes, vocab_size, note), whether its "
              "version has a vocabulary (trained), and its vocabulary job queued or running", answer={"type": "array"}),
        Route("POST", "/api/tokenizers", new_tokenizer, "a tokenizer recipe at version 1 (wpt depot tokenizer new)",
              body={**recipe_body, "required": ["name", *recipe_body["required"]],
                    "properties": {**recipe_body["properties"], "name": {"type": "string", "minLength": 1}}},
              answer={"type": "object", "required": ["id"]}),
        Route("GET", "/api/tokenizers/{name}", tokenizer, "one tokenizer recipe: its current revision, every revision, "
              "its vocabularies (without their merges) and whether its version has one",
              answer={"type": "object", "required": ["recipe", "revisions", "vocabularies", "trained"]}),
        Route("PUT", "/api/tokenizers/{name}", edit_tokenizer, "change a tokenizer recipe as a new revision: a change "
              "of what trains its vocabulary raises its version, a change of its note alone does not (wpt depot "
              "tokenizer edit)", body=recipe_body, answer={"type": "object", "required": ["version"]}),
        Route("POST", "/api/tokenizers/{name}/train", train, "queue the training of the vocabulary of the recipe's "
              "version (the job already open, when one is); refused when that version has one", body=None,
              answer=job_answer),
        Route("GET", "/api/tokenizers/{name}/vocabularies/{version}", vocabulary, "the vocabulary of one version: its "
              "sha256, how many texts and bytes it trained on, when it was made",
              answer={"type": "object", "required": ["sha256", "texts"]}),
        Route("GET", "/api/tokenizers/{name}/vocabularies/{version}/file", vocabulary_file, "the vocabulary's file "
              "(wpt_tokenizer's format)"),
        Route("GET", "/api/tokenizers/{name}/vocabularies/{version}/texts", vocabulary_texts, "a page of the texts the "
              "vocabulary trained on, in the order they were read: position, depot ('' for this depot) and id",
              answer={"type": "array"}, request=True,
              query={"offset": "the first position (default 0)", "limit": "how many (default 50, at most 1000)"}),
        Route("GET", "/api/jobs", jobs_listed, "the jobs' queue: every job queued or running in the order they run, "
              "then the latest that ended; each with its kind, what it does, its state, its progress and its answer "
              "or problem; only `wpt depot jobs cancel` cancels one", answer={"type": "array"}),
        Route("GET", "/api/jobs/{id}", job_state, "one job's state", answer=job_answer),
        Route("POST", "/api/datasets/{id}/build", build, "queue the build of the dataset's corpus as it is now, after "
              "the training of its tokenizer recipe's vocabulary when its version has none; with corpus, the "
              "deleted corpus of that id built again (409 when the dataset has a corpus job open)",
              body={"type": ["object", "null"], "properties": {"corpus": {"type": "string", "minLength": 1}}},
              answer=job_answer),
        Route("GET", "/api/datasets/{id}/download", download, "the dataset's download, at an address that never "
              "changes: state built (the latest fresh corpus of the dataset's current revision, its files with "
              "their sizes and sha256, and its parts of texts, each in jsonl.zst and parquet, with its size once "
              "sent whole), checking an erasure (the erasures whose corpora are being marked stale; nothing is "
              "sent meanwhile), being built (the job and its progress), stale or not built",
              answer={"type": "object", "required": ["state"]}),
        Route("GET", "/api/datasets/{id}/download/files/{name}", download_file, "one file of the dataset's corpus, "
              "streamed from the store; one Range of bytes gives 206", request=True),
        Route("GET", "/api/datasets/{id}/download/text/{name}", download_text, "one part of the corpus's texts "
              "(part-NNNNN.jsonl.zst or part-NNNNN.parquet), made as it is sent, the same bytes every time; at "
              f"most {EXPORTS} at once in all and {VISITOR_EXPORTS} per visitor (429)", request=True),
        Route("GET", "/api/datasets/{dataset}/sample", sample_route, "the sample of the dataset (its id or name) that "
              "trains a vocabulary, in its corpus's seeded order, each share stopping at its first text that would "
              "pass its part of the bytes: lines {depot ('' for this depot), text, bytes} each followed by the "
              "text's bytes, {ordering} lines while the order is made, and a last line {end, texts, bytes}",
              request=True, query={"bytes": "the sample's size in bytes"}),
    ]
