"""The routes of `wpt depot web` over the depot's tokenizer recipes and
their vocabularies, and the jobs' queue (wpt.depot.jobs). The snapshots
are wpt.depot.web.snapshots', the repository's recipes and the corpora
they build wpt.depot.web.recipes', and the frozen dataset routes, a
dataset's corpora and export included, wpt.depot.web.legacy's.

Anyone may queue a vocabulary training, only with an explicit
confirmation (confirmed: true in its body), after the pages show the
estimate of what it takes (GET /api/tokenizers/{name}/estimate: its
sample's bytes at the speed of TRAINING_MEASURE); only the command line
cancels a job.

The job operation of this module (train_job) runs in the queue's process;
it takes the depot, its arguments, progress and job.
"""

from functools import wraps

from wpt_depot import Depot, errors
from wpt_depot.legacy import Legacy
from wpt_utils.lines import say

from .. import shared
from ..downloads import Downloads
from ..jobs import Job, Jobs
from .recipes import recipe_routes, recipes_naming
from .server import Request, Route, Stream
from .snapshots import snapshot_routes

# the media type of a vocabulary file, the JSON of wpt_tokenizer's merges
JSON_TYPE = "application/json"
# the one measure of a vocabulary training: (sample bytes, seconds) it took on the public depot's machine
TRAINING_MEASURE = (10_000_000_000, 49 * 60)


# The job operations, which the queue's process imports by name

def train_job(depot: Depot, name: str, *, progress, job: Job) -> dict:
    """Train the vocabulary of the current version of the tokenizer recipe named so: {vocabulary: its sha256}."""
    job.work.mkdir(parents=True, exist_ok=True)
    return {"vocabulary": shared.train_vocabulary(depot, name, say, work=job.work, progress=progress)}


def builds(depot: Depot, jobs: Jobs, refusals) -> list[Route]:
    """The routes over depot's tokenizer recipes and jobs, then those of
    its snapshots and of the repository's recipes and their corpora;
    refusals wraps a handler so that it answers each failure with its
    status (wpt.depot.web.routes.refusals)."""
    downloads = Downloads(jobs.path)
    shared.settle_marks(depot, downloads, say)  # the marking a stop of the web app cut short
    legacy = Legacy(depot.address)

    def wrapped(handler):
        """refusals for a handler that gets the Request too."""
        @wraps(handler)
        def answer(values, body, request):
            return refusals(lambda values, body: handler(values, body, request))(values, body)
        return answer

    # Tokenizer recipes and their vocabularies

    def recipe_shown(recipe: dict) -> dict:
        """The recipe's row with snapshot_name, the name of the snapshot its
        sample reads (None when it names none, or a snapshot the depot no
        longer holds); dataset_name, for a revision written before
        snapshots, the name of the dataset it names (None otherwise), and
        snapshot_offer, {id, name} of the snapshot the upgrade made of
        that dataset's latest revision it converted (None without one),
        which an edit can name in one click; and its vocabulary job queued
        or running."""
        snapshot = dataset = offer = None
        if recipe["sample"] is not None:
            snapshot = {one["id"]: one["name"] for one in depot.snapshots()}.get(recipe["sample"]["snapshot"])
        elif recipe["depot"] is None:
            try:
                dataset = legacy.dataset_by_id(recipe["dataset"])["name"]
            except errors.NotFound:
                pass
            offer = snapshot_of(recipe["dataset"])
        else:
            dataset = recipe["dataset"]
        return {**recipe, "snapshot_name": snapshot, "dataset_name": dataset, "snapshot_offer": offer,
                "job": jobs.open_job("vocabulary", recipe=recipe["name"])}

    def snapshot_of(dataset: str) -> dict | None:
        """{id, name} of the snapshot legacy_snapshot_of names for the
        latest revision of the dataset it converted; None without one, and
        on a depot of a format before snapshots, which has no such table."""
        if not depot.rows("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'legacy_snapshot_of'"):
            return None
        found = depot.rows("SELECT s.id, s.name FROM legacy_snapshot_of l JOIN snapshots s ON s.id = l.snapshot "
                           "WHERE l.dataset = ? ORDER BY l.revision DESC LIMIT 1", (dataset,))
        return found[0] if found else None

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
        return body["snapshot"], body.get("kinds"), body["sample_bytes"], body["vocab_size"], body.get("note")

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
        return 200, Stream(JSON_TYPE, iter([data]), len(data),
                           {"Content-Disposition": f'attachment; filename="vocabulary-{found["sha256"][:12]}.json"'})

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
                     "basis": f"approximate: {measured_bytes / 1e9:g} GB of sample took {measured_seconds // 60} minutes "
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

    recipe_body = {"type": "object", "required": ["snapshot", "sample_bytes", "vocab_size"],
                   "properties": {"snapshot": {"type": "string", "minLength": 1, "description": "the snapshot the "
                                               "sample reads: its id, or the name of one of this depot's own"},
                                  "kinds": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1},
                                            "description": "the kinds of texts the sample reads; absent or null: "
                                            "plain, the one kind a sample takes for now"},
                                  "sample_bytes": {"type": "integer", "minimum": 1},
                                  "vocab_size": {"type": "integer", "minimum": 513},
                                  "note": {"type": ["string", "null"]}}}
    job_answer = {"type": "object", "required": ["id", "kind", "title", "state"]}
    return [
        Route("GET", "/api/tokenizers", tokenizers, "every tokenizer recipe of the depot: its current revision (name, "
              "version, sample, the snapshot and kinds its vocabulary trains on, with snapshot_name, sample_bytes, "
              "vocab_size, note; a revision written before snapshots names no sample but a dataset, with "
              "dataset_name, needs_snapshot is true: it trains nothing until an edit names a snapshot, and "
              "snapshot_offer names the snapshot the upgrade made of that dataset, or is null), whether "
              "its version has a vocabulary (trained), and its vocabulary job queued or running", answer={"type": "array"}),
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
              "then the latest that ended; each with its kind, its lane (write: the writes over many texts; heavy: "
              "snapshots, vocabularies and corpora; each lane runs one job at a time), what it does, its key (a "
              "corpus job's build key, a snapshot job's name), its state, its progress and its answer or problem; "
              "only `wpt depot jobs cancel` cancels one", answer={"type": "array"}),
        Route("GET", "/api/jobs/{id}", job_state, "one job's state", answer=job_answer),
        *snapshot_routes(depot, jobs, refusals, recipes_naming(depot)),
        *recipe_routes(depot, jobs, refusals, wrapped),
    ]
