"""The routes of `wpt depot web` over what the depot builds and gives to
download: its tokenizer recipes and their vocabularies, the jobs' queue
(wpt.depot.jobs), a dataset's corpora built with a vocabulary at a size,
a dataset's export of its texts, and the sample route another depot's
vocabulary training reads.

Anyone may queue a vocabulary training or a corpus build, each only with
an explicit confirmation (confirmed: true in its body), after the pages
show the estimate of what it takes; only the command line cancels a job.
A build names its dataset, a vocabulary the depot holds already (by its
sha256: a build never trains one) and a size (max_tokens, or "all"). The
estimate of a build (GET /api/datasets/{id}/estimate) tokenizes a small
sample (wpt_depot.estimate) and writes nothing; that of a training
(GET /api/tokenizers/{name}/estimate) is its sample's bytes at the speed of
TRAINING_MEASURE. A deleted corpus is built again from its own record, and
the build says whether its id changed.

A dataset's download of one vocabulary and size
(GET /api/datasets/{id}/download/{vocabulary}/{size}, the page
#/download/<id>/<vocabulary>/<size>) never changes its address: it gives
the latest fresh corpus of that triple built from the dataset's current
revision (or from one that differs from it only in its max_tokens or the
vocabulary it publishes, which a build never reads: a corpus is
identified by its own triple), or a state with nothing to download:
"checking an erasure" (an erasure decided here is being carried into the
corpora's records, wpt.depot.downloads; no corpus is sent meanwhile),
"being built" (the job's progress), "stale" (an admitted erasure reached a
text of the latest corpus of the dataset's current revision; nothing builds
it again by itself), "not built" or "no store". Reading it starts nothing.
A built corpus gives each of its files, streamed from the store with Range
support. GET /api/datasets/{id}/download lists the triples the store holds
corpora of, each with its address and state, and with the latest fresh
corpus of the triple built from any revision of the dataset (latest_fresh,
its revision named), the corpus training takes (wpt.train.recipe_depot).

A dataset's export (GET /api/datasets/{id}/export, the page
#/export/<id>) gives the texts of its current revision in numbered parts
(wpt_depot.exports), each made as it is sent, in JSONL compressed with zstd
or in Parquet, with its size once it was sent whole; it needs no
tokenizer, no build and no job. A part whose size is known takes one Range
of bytes. At most downloads.EXPORTS parts (and samples) are sent at once
in all, and downloads.VISITOR_EXPORTS for one visitor (429 past them).

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

from wpt_depot import Depot, corpora, errors
from wpt_depot.corpus import sample, sample_lines
from wpt_depot.estimate import estimate
from wpt_depot.exports import FORMS, Export, parts_for
from wpt_store import StoreError, store_from_settings
from wpt_store.names import File
from wpt_utils.lines import say
from wpt_utils.parallel import Sizing

from .. import shared
from ..downloads import EXPORTS, VISITOR_EXPORTS, Downloads
from ..jobs import Job, Jobs, work_folder
from .server import Request, Route, Stream

CHUNK = 8 << 20  # bytes of a stored file read at a time to send it
PROGRESS_SECONDS = 5  # seconds between two lines of progress of a sample being ordered
RANGE = re.compile(r"bytes=(\d*)-(\d*)")
TEXT_PART = re.compile(r"part-(\d{5})\.(jsonl\.zst|parquet)")
SHA256 = re.compile(r"[0-9a-f]{64}")
ALL = "all"  # the size of a build with no max_tokens
# the columns of a dataset's row a corpus of it depends on: never max_tokens nor vocabulary, which a build never reads
SAME = ("name", "tokenizer", "libraries", "shares")
# the one measure of a vocabulary training: (sample bytes, seconds) it took on the public depot's machine
TRAINING_MEASURE = (10_000_000_000, 49 * 60)


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


def build_job(depot: Depot, dataset: str, vocabulary: str, max_tokens: int | None, corpus: str | None = None, *,
              progress, job: Job) -> dict:
    """Build the corpus of the dataset of that id as it is now with the
    vocabulary of that sha256 at the size max_tokens (None: no limit), or,
    with corpus, the deleted corpus of that id again, into the store
    WPT_STORE names, under the cap WPT_CORPORA_CAP sets, going on from the
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
        depot, dataset, say, vocabulary=vocabulary, max_tokens=max_tokens, store=store, record=record, work=work,
        checkpoint=job.checkpoint(), save=job.save, progress=progress,
        room=lambda folder: corpora.Room(store, cap, folder, downloads.last_download, downloads.busy, say))
    job.save(None)
    shutil.rmtree(work, ignore_errors=True)
    return {"corpus": built["id"], "texts": built["texts"], "tokens": built["total_tokens"],
            "changed": built.get("changed")}


def size_of(given) -> int | None:
    """A build's size from a route: a whole number of tokens above 0, or "all" (None). BadQuery otherwise."""
    if given == ALL:
        return None
    try:
        size = int(given)
    except (TypeError, ValueError):
        size = 0
    if isinstance(given, bool) or size < 1 or str(size) != str(given).strip():
        raise errors.BadQuery(f"a size is a whole number of tokens above 0, or {ALL}: {given!r}")
    return size


def size_name(size: int | None) -> str:
    """A build's size as its address names it."""
    return ALL if size is None else str(size)


def vocabulary_given(given) -> str:
    """A vocabulary's sha256 from a route. BadQuery when it is not 64 lowercase hexadecimal characters."""
    if not isinstance(given, str) or not SHA256.fullmatch(given):
        raise errors.BadQuery(f"name a vocabulary by its sha256, 64 lowercase hexadecimal characters: {given!r}")
    return given


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

    @refusals
    def vocabularies(values, body):
        names = {one["id"]: one["name"] for one in depot.tokenizers()}
        found = depot.rows("SELECT recipe, version, sha256, texts, bytes, made_at FROM vocabularies "
                           "ORDER BY made_at DESC, recipe, version")
        return 200, shared.jsonable([{**one, "name": names.get(one["recipe"])} for one in found])

    @refusals
    def training_estimate(values, body):
        recipe = depot.tokenizer(values["name"])
        measured_bytes, measured_seconds = TRAINING_MEASURE
        return 200, {"sample_bytes": recipe["sample_bytes"], "version": recipe["version"],
                     "trained": depot.vocabulary(recipe["id"], recipe["version"]) is not None,
                     "seconds": round(recipe["sample_bytes"] * measured_seconds / measured_bytes),
                     "basis": f"approximate: {measured_bytes:,} bytes of sample took {measured_seconds // 60} minutes "
                              "to train on the public depot's machine, and the time is taken in proportion to the "
                              "sample's bytes"}

    @refusals
    def train(values, body):
        recipe = depot.tokenizer(values["name"])
        if body.get("confirmed") is not True:
            raise errors.Refused("a vocabulary training starts only once confirmed (confirmed: true), after its "
                                 "estimate: nothing was queued")
        if depot.vocabulary(recipe["id"], recipe["version"]) is not None:
            raise errors.Refused(f"the tokenizer recipe {recipe['name']} has a vocabulary of its version "
                                 f"{recipe['version']} already")
        return 200, jobs.open_job("vocabulary", recipe=recipe["name"]) or jobs.add(
            "vocabulary", f"train the vocabulary of the tokenizer recipe {recipe['name']} version {recipe['version']}",
            train_job, recipe["name"], recipe=recipe["name"])

    # Jobs

    @refusals
    def jobs_listed(values, body):
        return 200, jobs.listed()

    @refusals
    def job_state(values, body):
        return 200, shared.jsonable(jobs.state(values["id"]))

    # Building a corpus

    def vocabulary_named(sha256: str) -> dict:
        """{sha256, recipe, version} of the depot's vocabulary of that sha256; NoVocabulary when it holds none."""
        recipe, version, _ = shared.vocabulary_by_sha256(depot, sha256)
        return {"sha256": sha256, "recipe": recipe["name"], "version": version}

    @refusals
    def build(values, body):
        row = depot.dataset_by_id(values["id"])
        vocabulary, size = vocabulary_given(body["vocabulary"]), size_of(body["max_tokens"])
        if body["confirmed"] is not True:
            raise errors.Refused("a build starts only once confirmed (confirmed: true), after its estimate: nothing "
                                 "was queued")
        named = vocabulary_named(vocabulary)
        corpus = body.get("corpus")
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
            if corpora.triple(record)[1:] != (vocabulary, size):
                raise errors.Refused(f"the corpus {corpus} was built with another vocabulary or size than the ones "
                                     "named")
        what = (f"with the vocabulary {vocabulary[:12]} ({named['recipe']} version {named['version']}), "
                f"{'every token' if size is None else f'at most {size:,} tokens'}")
        title = (f"build the corpus of the dataset {row['name']} {what}" if record is None
                 else f"build the corpus {record['id'][:12]} of the dataset {row['name']} again, {what}")
        return 200, jobs.add("corpus", title, build_job, row["id"], vocabulary, size, corpus, dataset=row["id"],
                             target={"vocabulary": vocabulary, "max_tokens": size})

    estimating = threading.Lock()  # one estimate at a time: each tokenizes on one core of the web app

    @wrapped
    def estimate_route(values, body, request: Request):
        row = depot.dataset_by_id(values["id"])
        vocabulary = vocabulary_given(request.query.get("vocabulary"))
        size = size_of(request.query.get("max_tokens", ALL))
        recipe, version, tokenizer = shared.vocabulary_by_sha256(depot, vocabulary)
        with estimating:
            found = estimate(depot, row, tokenizer, size, shared.public_url(), Sizing("tokenizing workers").limit)
        cap, others, problem = shared.corpora_cap(), None, None
        try:
            others = corpora.Room(store(), cap, "").taken()
        except (StoreError, OSError) as error:
            problem = f"{type(error).__name__}: {error}"
        return 200, {**found, "vocabulary": {"sha256": vocabulary, "recipe": recipe["name"], "version": version},
                     "max_tokens": size, "cap": cap, "others": others,
                     "fits": cap is None or found["bytes"] <= cap,
                     "evicts": None if cap is None or others is None else max(0, others + found["bytes"] - cap),
                     "store": problem}

    # Downloads of corpora

    def file_name(record: dict, one: dict) -> str:
        return one["key"].removeprefix(corpora.folder_of(record) + "/")

    def target_of(vocabulary: str, size: int | None) -> dict:
        return {"vocabulary": vocabulary, "max_tokens": size}

    def current(row: dict, record: dict) -> bool:
        """Whether the corpus's record was built from the dataset's current
        revision (row), or from a revision that differs from it only in
        columns a build never reads (those SAME leaves out: max_tokens and
        vocabulary)."""
        revision = record["dataset"]["revision"]
        if revision == row["revision"]:
            return True
        built = depot.rows(f"SELECT {', '.join(SAME)} FROM datasets_revisions WHERE id = ? AND revision = ?",
                           (row["id"], revision))
        return bool(built) and built[0] == {column: row[column] for column in SAME}

    def state_of(row: dict, records: list[dict], vocabulary: str, size: int | None, job: dict | None) -> tuple:
        """(the state of the dataset's download of that vocabulary and size,
        its fresh corpus or None, the latest corpus of the triple or None),
        records being the triple's corpora, latest first."""
        built = [one for one in records if current(row, one)]
        fresh = next((one for one in built if corpora.fresh(one)), None)
        last = next(iter(records), None)
        if fresh is not None:
            return "built", fresh, last
        if job is not None and job["target"] == target_of(vocabulary, size):
            return "being built", None, last
        newest = next(iter(built), None)
        if newest is not None and "stale" in newest:
            return "stale", None, newest
        return "not built", None, last

    def summary(record: dict | None) -> dict | None:
        return None if record is None else {key: record.get(key) for key in (
            "id", "built_at", "texts", "conversations", "total_tokens", "stale", "deleted")} | {
            "revision": record["dataset"]["revision"]}

    def link(row: dict, vocabulary: str, size: int | None) -> str:
        return f"/api/datasets/{row['id']}/download/{vocabulary}/{size_name(size)}"

    def vocabulary_shown(sha256: str) -> dict:
        try:
            return vocabulary_named(sha256)
        except shared.NoVocabulary:
            return {"sha256": sha256, "recipe": None, "version": None}

    @refusals
    def downloads_listed(values, body):
        row = depot.dataset_by_id(values["id"])
        job = jobs.open_job("corpus", dataset=row["id"])
        shown = {"dataset": shared.jsonable(row), "job": job, "corpora": []}
        marking = downloads.pending()
        if marking:
            return 200, {**shown, "state": "checking an erasure", "erasures": marking}
        try:
            found = corpora.built_from(store(), shared.corpus_depot(depot, row), row["id"])
        except (StoreError, OSError) as error:
            return 200, {**shown, "state": "no store", "problem": f"{type(error).__name__}: {error}"}
        triples = {}
        for record in found:
            triples.setdefault(corpora.triple(record)[1:], []).append(record)
        if job is not None and job["target"] is not None:
            triples.setdefault((job["target"]["vocabulary"], job["target"]["max_tokens"]), [])
        for (vocabulary, size), records in triples.items():
            state, fresh, last = state_of(row, records, vocabulary, size, job)
            shown["corpora"].append({"vocabulary": vocabulary_shown(vocabulary), "max_tokens": size,
                                     "size": size_name(size), "link": link(row, vocabulary, size), "state": state,
                                     "corpus": summary(fresh or last),
                                     "latest_fresh": summary(next((one for one in records if corpora.fresh(one)),
                                                                  None))})
        return 200, {**shown, "state": "listed"}

    @refusals
    def download(values, body):
        row = depot.dataset_by_id(values["id"])
        vocabulary, size = vocabulary_given(values["vocabulary"]), size_of(values["size"])
        job = jobs.open_job("corpus", dataset=row["id"])
        shown = {"dataset": shared.jsonable(row), "vocabulary": vocabulary_shown(vocabulary), "max_tokens": size,
                 "size": size_name(size), "link": link(row, vocabulary, size),
                 "job": job if job is not None and job["target"] == target_of(vocabulary, size) else None}
        marking = downloads.pending()
        if marking:
            return 200, {**shown, "state": "checking an erasure", "erasures": marking}
        try:
            records = corpora.built_from(store(), shared.corpus_depot(depot, row), row["id"], vocabulary, size)
        except (StoreError, OSError) as error:
            return 200, {**shown, "state": "no store", "problem": f"{type(error).__name__}: {error}"}
        state, fresh, last = state_of(row, records, vocabulary, size, job)
        if fresh is None:
            return 200, {**shown, "state": state, "corpus": summary(last)}
        files = [{"name": file_name(fresh, one), "bytes": one.get("bytes"), "sha256": one["sha256"]}
                 for one in corpora.files(fresh)]
        return 200, {**shown, "state": "built",
                     "corpus": {**summary(fresh), "tokenizer": fresh["tokenizer"]}, "files": files}

    def fresh_corpus(values) -> dict:
        row = depot.dataset_by_id(values["id"])
        vocabulary, size = vocabulary_given(values["vocabulary"]), size_of(values["size"])
        marking = downloads.pending()
        if marking:
            raise errors.Refused(f"the corpora holding a text of the erasures {', '.join(marking)} are being marked "
                                 "stale: no corpus is sent until that ends")
        records = corpora.built_from(store(), shared.corpus_depot(depot, row), row["id"], vocabulary, size)
        _, found, _ = state_of(row, records, vocabulary, size, None)
        if found is None:
            raise errors.NotFound(f"the dataset {row['name']} has no corpus of the vocabulary {vocabulary[:12]} at "
                                  f"the size {size_name(size)} to download now: see its download page")
        return found

    def ranged(request: Request, size: int):
        """(start, end, status) of the bytes to send of size bytes: one Range of them (206), else all (200);
        an answer of 416 for a Range that cannot be met."""
        asked = request.headers.get("Range")
        if not asked:
            return 0, size, 200
        match = RANGE.fullmatch(asked.strip())
        if match is None or (not match[1] and not match[2]):
            return 416, {"problems": [f"the Range {asked} is not one range of bytes"]}, None
        if match[1]:
            start, end = int(match[1]), min(int(match[2]) + 1 if match[2] else size, size)
        else:
            start, end = max(size - int(match[2]), 0), size
        if start >= end:
            return 416, {"problems": [f"the Range {asked} is outside the file's {size:,} bytes"]}, None
        return start, end, 206

    @wrapped
    def download_file(values, body, request: Request):
        record = fresh_corpus(values)
        chosen = next((one for one in corpora.files(record) if file_name(record, one) == values["name"]), None)
        if chosen is None:
            raise errors.NotFound(f"corpus {record['id']} has no file {values['name']}")
        reference = File.of(chosen)
        size = store().size(reference)
        start, end, status = ranged(request, size)
        if status is None:
            return start, end

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

    # Exports of texts

    def export_of(row: dict) -> tuple[Export, int, int]:
        """(the export of the dataset's current revision, how many texts it counts, its parts)."""
        export = Export(depot, row, shared.public_url())
        texts = downloads.export_texts(export.identity)
        if texts is None:
            texts = export.texts()
            downloads.counted(export.identity, texts)
        return export, texts, parts_for(texts)

    @refusals
    def export_listed(values, body):
        row = depot.dataset_by_id(values["id"])
        export, texts, parts = export_of(row)
        known = downloads.export_sizes(export.identity, parts)
        listed = [{"part": number, "forms": {form: {"name": f"part-{number:05d}.{form}",
                                                    **known.get((number, form), {"bytes": None, "sha256": None})}
                                             for form in FORMS}} for number in range(parts)]
        return 200, {"dataset": shared.jsonable(row), "export": export.identity, "texts": texts,
                     "texts_per_part": texts / parts, "parts": listed}

    @wrapped
    def export_part(values, body, request: Request):
        row = depot.dataset_by_id(values["id"])
        export, _, parts = export_of(row)
        match = TEXT_PART.fullmatch(values["name"])
        if match is None or int(match[1]) >= parts:
            raise errors.NotFound(f"the export of the dataset {row['name']} has no part {values['name']}: its parts "
                                  f"are part-00000 to part-{parts - 1:05d}, each .jsonl.zst or .parquet")
        number, form = int(match[1]), match[2]
        known = downloads.export_sizes(export.identity, parts).get((number, form))
        start, end, status = (0, None, 200) if known is None else ranged(request, known["bytes"])
        if status is None:
            return start, end
        guard = downloads.streaming(None, request.visitor, "text")
        guard.__enter__()  # TooMany before anything is sent

        def chunks():
            digest, size = hashlib.sha256(), 0
            for chunk in export.part(number, parts, form):
                digest.update(chunk)
                size += len(chunk)
                if end is None:
                    yield chunk
                    continue
                piece = chunk[max(start - (size - len(chunk)), 0):max(end - (size - len(chunk)), 0)]
                if piece:
                    yield piece
                if size >= end:
                    return
            if known is None:
                downloads.exported(export.identity, parts, number, form, size, digest.hexdigest())
        headers = {"Content-Disposition": f'attachment; filename="{values["name"]}"'}
        if known is not None:
            headers.update({"Accept-Ranges": "bytes", "ETag": f'"{known["sha256"]}"'})
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{known['bytes']}"
        return status, Stream(FORMS[form], Guarded(chunks(), guard), None if known is None else end - start, headers)

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
        Route("GET", "/api/vocabularies", vocabularies, "every vocabulary the depot holds, the latest made first: its "
              "sha256 (what a build names it by), its recipe (id and name) and version, how many texts and bytes "
              "it trained on, when it was made", answer={"type": "array"}),
        Route("GET", "/api/tokenizers/{name}/estimate", training_estimate, "the estimate of the training of the "
              "vocabulary of the recipe's version: its sample's bytes and the seconds it takes, approximate (the "
              "time of the one training measured, in proportion to the bytes); nothing is queued",
              answer={"type": "object", "required": ["sample_bytes", "seconds", "basis"]}),
        Route("POST", "/api/tokenizers/{name}/train", train, "queue the training of the vocabulary of the recipe's "
              "version (the job already open, when one is), only with confirmed: true; refused when that version "
              "has one",
              body={"type": "object", "required": ["confirmed"], "properties": {"confirmed": {"type": "boolean"}}},
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
        Route("POST", "/api/datasets/{id}/build", build, "queue the build of the dataset's corpus as it is now with "
              "a vocabulary the depot holds (its sha256) at a size (max_tokens: a number of tokens, or \"all\"), only "
              "with confirmed: true; refused for a vocabulary not trained (422); with corpus, the deleted corpus of "
              "that id, of that vocabulary and size, built again (409 when the dataset has a corpus job open)",
              body={"type": "object", "required": ["vocabulary", "max_tokens", "confirmed"],
                    "properties": {"vocabulary": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                                   "max_tokens": {"oneOf": [{"type": "integer", "minimum": 1}, {"const": ALL}]},
                                   "confirmed": {"type": "boolean"},
                                   "estimate": {"type": "object", "description": "the estimate the client showed"},
                                   "corpus": {"type": "string", "minLength": 1}}},
              answer=job_answer),
        Route("GET", "/api/datasets/{id}/estimate", estimate_route, "the estimate of a build of the dataset's corpus "
              "with a vocabulary at a size, from a small sample tokenized (wpt_depot.estimate): its tokens, the "
              "bytes it takes in the store, the seconds its build takes here (an estimate), and whether it fits "
              "under the store's cap (WPT_CORPORA_CAP) beside the bytes the other corpora take, which the build "
              "would delete as it needs room (evicts); nothing is written or queued",
              answer={"type": "object", "required": ["tokens", "bytes", "seconds", "fits"]}, request=True,
              query={"vocabulary": "the vocabulary's sha256", "max_tokens": "a number of tokens, or all (default)"}),
        Route("GET", "/api/datasets/{id}/download", downloads_listed, "the corpora the store holds of the dataset, "
              "one row per vocabulary and size, each with the address of its download and its state (as that "
              "address gives it), and latest_fresh: the latest fresh corpus of that vocabulary and size built from "
              "any revision of the dataset, with its revision (null without one), the one training takes; checking "
              "an erasure or no store instead when nothing can be listed",
              answer={"type": "object", "required": ["state", "corpora"]}),
        Route("GET", "/api/datasets/{id}/download/{vocabulary}/{size}", download, "the dataset's download of one "
              "vocabulary (its sha256) and size (a number of tokens, or all), at an address that never changes: "
              "state built (the latest fresh corpus built from the dataset's current revision, its max_tokens and "
              "published vocabulary aside, its files with their sizes and sha256), checking an erasure (nothing is sent meanwhile), being built (the job and "
              "its progress), stale, not built or no store; it starts nothing",
              answer={"type": "object", "required": ["state"]}),
        Route("GET", "/api/datasets/{id}/download/{vocabulary}/{size}/files/{name}", download_file, "one file of that "
              "corpus, streamed from the store; one Range of bytes gives 206", request=True),
        Route("GET", "/api/datasets/{id}/export", export_listed, "the export of the texts of the dataset's current "
              "revision, with no tokenizer and no build: how many texts its libraries hold, and its parts, each in "
              "jsonl.zst and parquet, with its size and sha256 once sent whole (null before)",
              answer={"type": "object", "required": ["export", "texts", "parts"]}),
        Route("GET", "/api/datasets/{id}/export/{name}", export_part, "one part of the export (part-NNNNN.jsonl.zst "
              "or part-NNNNN.parquet), made as it is sent, the same bytes every time for the same revision; one "
              f"Range of bytes gives 206 once its size is known; at most {EXPORTS} at once in all and "
              f"{VISITOR_EXPORTS} per visitor (429)", request=True),
        Route("GET", "/api/datasets/{dataset}/sample", sample_route, "the sample of the dataset (its id or name) that "
              "trains a vocabulary, in its corpus's seeded order, each share stopping at its first text that would "
              "pass its part of the bytes: lines {depot ('' for this depot), text, bytes} each followed by the "
              "text's bytes, {ordering} lines while the order is made, and a last line {end, texts, bytes}",
              request=True, query={"bytes": "the sample's size in bytes"}),
    ]
