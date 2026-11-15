"""The routes of `wpt depot web`: the operations of the `wpt depot`
subcommands, each a controller over wpt_depot that reads its request, calls
the depot and turns its result, or its failure, into an answer. The depot
applies every rule, so the pages and the subcommands cannot disagree. A
pull, a remote's import and a change of the depot's maintainer or takedown
contact have no route: they are commands only (docs/architecture.md, "Two
interfaces"). The routes over tokenizer recipes, jobs, builds and
downloads are builds.py's. The follow routes only read:
they answer a clone or a pull of another depot (wpt_depot.follow), a page
at a time.

An operation on many texts (a decision, a license change, a put, a
take-out, a fork, a problem reported) takes a selection, never a list of
ids from the browser: the filter the page shows, the change number the page
read it at, and the ids the person unticked by hand. The depot matches the
filter when it writes and leaves out every text changed after that change
number, and the answer says how many it left out (wpt_depot.filters). Such
an operation runs as a job (wpt.depot.jobs), in a process of its own, unless it
writes at most INLINE texts (its filter names at most INLINE ids; for a
fork, the library holds at most INLINE texts) and no job is queued or
running; either way
its answer is the job's state, which GET /api/jobs/{id} reads again while it runs. A recount of the
depot's counts is a command only."""

import itertools
import json
from functools import wraps

from wpt_depot import Declaration, Depot, NewText, counts as facets, errors, follow
from wpt_depot.filters import STATUSES as TEXT_STATUSES
from wpt_depot.depot import (BELONGS_TO_UPSTREAM, CHANGED_SINCE, COUNT_STEP, CREDITS, INDEXED, NO_ALLOWED_LICENSE,
                             ORDERS, SORTED)
from wpt_depot.erasure import CONTEXT
from wpt_depot.filters import Filter, condition
from wpt_depot.schema import ADMITTED, REFUSED
from wpt_depot.corpora import NoRoom, TooBig
from wpt_store import StoreError
from wpt_utils.lines import say

from .. import shared
from ..shared import CONFIRMATION, REQUIREMENTS, jsonable
from . import readers as limits
from ..downloads import Downloads, TooMany
from ..jobs import Busy, Jobs
from .builds import builds
from .readers import Readers
from .server import Route, Stream

# the HTTP status of each failure; a failure's status is that of the first of its classes listed here
STATUSES = {errors.NotFound: 404, errors.BadQuery: 400, Busy: 409, TooMany: 429, StoreError: 503,
            errors.DepotError: 422, shared.NoVocabulary: 422, shared.NoSample: 422, shared.BadSetting: 422,
            TooBig: 422, NoRoom: 422}

TEXT = {"type": "string", "minLength": 1}
NAMES = {"type": "array", "items": TEXT}
OBJECT, ROWS = {"type": "object"}, {"type": "array", "items": {"type": "object"}}
DECLARATION = {"type": "object", "required": ["name", "email", "accepted"],
               "description": f"who decides; accepted is the confirmation: {CONFIRMATION}",
               "properties": {"name": TEXT, "email": TEXT, "accepted": {"type": "boolean"}}}
ID = {"type": "object", "required": ["id"], "properties": {"id": TEXT}}
# ids named exactly: erasure requests decided together
IDS = {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True}
INLINE = 100  # a write over at most this many texts runs in the request, not as a job, while no job runs
WAIT = 1.0  # seconds a request that starts a job waits for its end before it answers the job running
# where a page of texts starts or ends: a row's values of the order, as a page gives them
PLACE = {"type": "array", "items": {"type": ["string", "integer", "number"]}, "minItems": 1}
NULLABLE_PLACE = {"oneOf": [PLACE, {"type": "null"}]}
# how many characters of text a page of passages holds at most, by default and at the most a request may ask
PASSAGE_CHARACTERS, MAX_PASSAGE_CHARACTERS = 200_000, 2_000_000


def refusals(handler):
    """The handler, answering each failure of STATUSES with its status and
    message."""
    @wraps(handler)
    def answer(values, body):
        try:
            return handler(values, body)
        except tuple(STATUSES) as error:
            status = next(code for kind, code in STATUSES.items() if isinstance(error, kind))
            return status, {"problems": [f"{type(error).__name__}: {error}"]}
    return answer


def filter_of(found: dict, change: int | None = None, unticked=()) -> Filter:
    """The filter a body's filter object stands for, with the change number
    its page was read at and the ids unticked by hand, when given."""
    ids = found.get("ids")
    return Filter(ids=None if ids is None else tuple(ids), tags=tuple(found.get("tags", [])),
                  licenses=tuple(found.get("licenses", [])), statuses=tuple(found.get("statuses", [])),
                  libraries=tuple(found.get("libraries", [])), locked=found.get("locked"), where=found.get("where"),
                  unticked=tuple(unticked), change=change)


def selected(selection: dict) -> Filter:
    """The filter a body's selection stands for (SELECTION)."""
    return filter_of(selection["filter"], selection.get("change"), selection.get("unticked", ()))


def small(selection: dict) -> bool:
    """Whether the selection's filter names at most INLINE texts, so its write may run in the request."""
    ids = selection["filter"].get("ids")
    return ids is not None and len(ids) <= INLINE


# The operations a job runs, by name of function, so that a job's process imports them: each takes the depot, its
# arguments and progress, and gives the JSON answer of its route. left: {reason: how many texts the operation left
# out for it} (CHANGED_SINCE, and for a decision with leave_blocked, the reasons of Depot.blocked).

def decide_texts(depot: Depot, status: str, selection: Filter, declaration: Declaration, reason: str | None,
                 leave_blocked: bool, progress) -> dict:
    decision, texts, left = depot.decide(status, selection, declaration, reason, leave_blocked, progress)
    return {"decision": decision, "texts": texts, "left": left}


def relicense_texts(depot: Depot, selection: Filter, licenses: list[str], declaration: Declaration,
                    progress) -> dict:
    left = {}
    found, skipped = depot.relicense(selection, licenses, declaration, progress, left)
    return {"decisions": [{"decision": decision, "texts": texts} for decision, texts in found],
            "skipped": skipped, "left": left}


def put_texts(depot: Depot, library: str, selection: Filter, progress) -> dict:
    left = {}
    joined, left_out = depot.put(library, selection, progress, left)
    return {"joined": joined, "left_out": left_out, "left": left}


def take_out_texts(depot: Depot, library: str, selection: Filter, progress) -> dict:
    left = {}
    taken_out, not_held = depot.take_out(library, selection, progress, left)
    return {"taken_out": taken_out, "not_held": not_held, "left": left}


def fork_library(depot: Depot, library: str, name: str, without: Filter | None, progress) -> dict:
    left = {}
    return {"id": depot.fork(library, name, without, progress, left), "left": left}


def report_problem(depot: Depot, library: str, problem: str, selection: Filter | None, declaration: Declaration,
                   progress) -> dict:
    left = {}
    decision, texts = depot.report(library, problem, selection, declaration, progress, left)
    return {"decision": decision, "texts": len(texts), "left": left}


def routes(depot: Depot, jobs: Jobs) -> list[Route]:
    """Every route over depot; a person's SQL runs in reader processes
    (wpt.depot.web.readers), and a write over many texts in a job of jobs,
    whose runner the caller starts."""
    readers = Readers(depot.address)

    def job(kind: str, few: bool, operation, *args) -> tuple[int, dict]:
        """Run operation over the depot with args, in the request when it
        writes few texts and no write job is open (a write waits for a
        running write job's write lock, which a job shows and a request does
        not; a vocabulary training or a corpus build holds none), as a job
        otherwise: the job's state once it ended (200) or after WAIT
        seconds (202, running), or its refusal with its status."""
        if few and not jobs.writing():
            state = jobs.record("write", kind, depot, operation, *args)
        else:
            state = jobs.wait(jobs.add("write", kind, operation, *args)["id"], WAIT)
        if state["state"] in ("refused", "failed", "cancelled"):
            return state["status"] or 409, {"problems": [state["problem"]], "job": state["id"]}
        return 202 if state["state"] in ("queued", "running") else 200, jsonable(state)

    def declared(body) -> Declaration:
        found = body["declaration"]
        return Declaration(found["name"], found["email"], found["accepted"])

    @refusals
    def about(values, body):
        return 200, {"address": shared.public_url() or depot.name(), "change": depot.change(), "about": depot.about(),
                     "confirmation": CONFIRMATION, "requirements": REQUIREMENTS, "sorted": SORTED}

    @refusals
    def sql(values, body):
        return 200, readers.run("sql", body["query"])

    @refusals
    def kinds(values, body):
        return 200, [{**one, "schema": json.loads(one["schema"])} for one in
                     depot.rows("SELECT * FROM kinds ORDER BY id")]

    @refusals
    def licenses(values, body):
        return 200, depot.licenses()

    @refusals
    def license(values, body):
        return 200, depot.license(values["id"])

    @refusals
    def add_license(values, body):
        depot.add_license(body["id"], body["name"], body["text"])
        return 200, {}

    @refusals
    def add(values, body):
        kind = body.get("kind")
        one = NewText(text=body.get("text"), kind=kind,
                      content=None if kind is None else json.dumps(body.get("content"), ensure_ascii=False),
                      licenses=tuple(body["licenses"]), notice=body.get("notice"), title=body.get("title"),
                      authors=None if body.get("authors") is None else tuple(body["authors"]),
                      tags=tuple(body.get("tags", [])), address=body.get("address"))
        [added] = depot.add([one], declared(body))
        return 200, {"id": added}

    @refusals
    def show(values, body):
        return 200, jsonable(depot.show(values["id"]))

    @refusals
    def credits(values, body):
        decision, reason = depot.edit_credits(values["id"], {name: body[name] for name in CREDITS if name in body},
                                              declared(body))
        return 200, {"decision": decision, "reason": reason}

    @refusals
    def scores(values, body):
        decision, reason, changed, left_out = depot.edit_scores(body["texts"], declared(body))
        return 200, {"decision": decision, "reason": reason, "changed": changed, "left_out": left_out}

    @refusals
    def content(values, body):
        changed = depot.change_content(values["id"], body.get("text"),
                                       json.dumps(body["content"], ensure_ascii=False) if "content" in body else None,
                                       declared(body))
        decision, version = changed if changed is not None else (None, None)
        return 200, {"decision": decision, "version": version}

    @refusals
    def search(values, body):
        return 200, readers.run("search", condition(filter_of(body["filter"])), body.get("after"),
                                body.get("before"), body.get("limit", 50), body.get("order", "id"),
                                body.get("descending", False))

    @refusals
    def passages(values, body):
        return 200, readers.run(
            "passages", condition(filter_of(body["filter"])), body.get("after"), body.get("limit", 50),
            body.get("characters", PASSAGE_CHARACTERS), body.get("order", "id"), body.get("descending", False))

    @refusals
    def count(values, body):
        found = filter_of(body["filter"], body.get("change"), body.get("unticked", ()))
        return 200, readers.run("count", condition(found, depot.change()), body.get("start"),
                                body.get("size", COUNT_STEP), body["filter"].get("ids"))

    @refusals
    def blocked(values, body):
        return 200, readers.run("blocked", selected(body["selection"]), body.get("status", ADMITTED),
                                body.get("after"), body.get("limit", 50))

    @refusals
    def depot_counts(values, body):
        found = depot.counts()
        return 200, {"texts": found[facets.TEXTS].get("", 0), "statuses": {
            status: found[facets.STATUS].get(status, 0) for status in TEXT_STATUSES},
            "tags": found[facets.TAG], "licenses": found[facets.LICENSE], "libraries": found[facets.LIBRARY]}

    @refusals
    def tags(values, body):
        return 200, depot.tags()

    @refusals
    def decide(values, body):
        return job("decision", small(body["selection"]), decide_texts, body["status"], selected(body["selection"]),
                   declared(body), body.get("reason"), body.get("leave_blocked", False))

    @refusals
    def relicense(values, body):
        return job("license change", small(body["selection"]), relicense_texts, selected(body["selection"]),
                   body["licenses"], declared(body))

    @refusals
    def latest(values, body):
        return 200, jsonable(depot.latest_decisions())

    @refusals
    def decision(values, body):
        return 200, jsonable(depot.decision(values["id"]))

    @refusals
    def libraries(values, body):
        return 200, jsonable(depot.rows(f"SELECT l.*, {depot.upstream_sql('l.id')} AS upstream FROM libraries l "
                                        "ORDER BY l.id"))

    @refusals
    def library_size(values, body):
        library = depot.rows("SELECT id FROM libraries WHERE id = ?", (values["id"],))
        if not library:
            raise errors.NotFound(f"no library {values['id']}")
        return 200, {"id": values["id"], "texts": depot.counts()[facets.LIBRARY].get(values["id"], 0)}

    @refusals
    def pinned(values, body):
        return 200, jsonable(depot.pinned(values["id"], body.get("offset", 0), body.get("limit", 50)))

    @refusals
    def new_library(values, body):
        return 200, {"id": depot.new_library(body["name"])}

    @refusals
    def rename_library(values, body):
        depot.rename_library(values["id"], body["name"])
        return 200, {}

    @refusals
    def put(values, body):
        return job("put", small(body["selection"]), put_texts, values["id"], selected(body["selection"]))

    @refusals
    def take_out(values, body):
        return job("take-out", small(body["selection"]), take_out_texts, values["id"], selected(body["selection"]))

    @refusals
    def lock(values, body):
        depot.lock(values["id"])
        return 200, {}

    @refusals
    def fork(values, body):
        without = body.get("without")
        # a fork writes a membership for every admitted text of the library, whatever without takes
        held = depot.counts()[facets.LIBRARY].get(values["id"], 0)
        return job("fork", held <= INLINE, fork_library, values["id"], body["name"],
                   None if without is None else selected(without))

    @refusals
    def report(values, body):
        chosen = body.get("selection")
        return job("problem", chosen is None or small(chosen), report_problem, values["id"], body["problem"],
                   None if chosen is None else selected(chosen), declared(body))

    def held_libraries():
        """{id: {id, name, locked_at}} of every library this depot holds."""
        return {one["id"]: one for one in depot.rows("SELECT id, name, locked_at FROM libraries")}

    def with_here(row, held):
        """A dataset's row with here: the ids of the libraries it reads from
        this depot, those it names so (wpt_depot.Depot.reads_here) that this
        depot holds (held: the ids it holds)."""
        return {**row, "here": list(dict.fromkeys(
            ref["library"] for ref in row["libraries"]
            if ref["library"] in held and depot.reads_here(ref["depot"], ref["library"])))}

    @refusals
    def datasets(values, body):
        found = depot.rows(f"SELECT *, {depot.upstream_sql()} AS upstream FROM datasets ORDER BY id")
        held = held_libraries()
        return 200, jsonable([with_here(row, held) for row in found])

    def own(libraries) -> list[tuple[str | None, str]]:
        """The libraries a dataset of the body names, as (depot address,
        library id). Refused, before any depot is opened, for a library
        named by another depot's address: the server answers any client, so
        it opens no address a client gives (`wpt depot dataset` names them)."""
        others = sorted({one["depot"] for one in libraries
                         if one.get("depot") is not None and not depot.is_here(one["depot"])})
        if others:
            raise errors.Refused(f"a dataset made or changed here reads only this depot's libraries, not those of "
                                 f"{', '.join(others)}: use wpt depot dataset new or edit")
        return [(one.get("depot"), one["library"]) for one in libraries]

    @refusals
    def new_dataset(values, body):
        shares = body.get("shares")
        return 200, {"id": depot.new_dataset(
            body["name"], body["tokenizer"], own(body["libraries"]),
            None if shares is None else [(one["tag"], one["share"]) for one in shares], body.get("max_tokens"))}

    @refusals
    def dataset(values, body):
        found = depot.dataset_by_id(values["id"])
        upstream = bool(depot.upstream_owned([found["id"]]))
        held = held_libraries()
        row = with_here({**found, "upstream": upstream}, held)
        libraries = []
        for ref in row["libraries"]:
            mine = ref["library"] in row["here"]
            libraries.append({**ref, "local": mine, "name": held[ref["library"]]["name"] if mine else None,
                              "locked_at": held[ref["library"]]["locked_at"] if mine else None})
        corpora, unreachable = shared.corpora_of(depot, row["id"])
        return 200, jsonable({"dataset": row, "upstream": upstream, "libraries": libraries, "remote": depot.export_dataset(row["name"]),
                              "corpora": None if corpora is None else [
                                  {**{key: one.get(key) for key in ("id", "built_at", "texts", "conversations",
                                                                    "total_tokens", "stale", "deleted")},
                                   "revision": one["dataset"]["revision"]} for one in corpora],
                              "store": unreachable})

    @refusals
    def edit_dataset(values, body):
        shares = body.get("shares")
        depot.edit_dataset(values["id"], body["name"], own(body["libraries"]),
                           None if shares is None else [(one["tag"], one["share"]) for one in shares],
                           body.get("max_tokens"))
        return 200, {"id": values["id"]}

    @refusals
    def export(values, body):
        return 200, depot.export_dataset(values["name"])

    @refusals
    def erasures(values, body):
        found = depot.erasures()
        theirs = set(depot.upstream_owned([one["id"] for one in found]
                                          + [needle["text"] for one in found for needle in one["needles"]]))
        return 200, jsonable([{**one, "upstream": one["id"] in theirs
                               or any(needle["text"] in theirs for needle in one["needles"])} for one in found])

    @refusals
    def erasure_shown(values, body):
        return 200, depot.erasure_shown(values["id"])

    @refusals
    def erase(values, body):
        needles = [(one["text"], one.get("start"), one.get("end"), one.get("field"), one.get("version"))
                   for one in body["needles"]]
        return 200, {"id": depot.ask_erasure(body["reason"], needles, declared(body))}

    @refusals
    def decide_erasures(values, body):
        return 200, {"carried_out": shared.decide_erasures(depot, body["erasures"], body["status"], declared(body),
                                                           body.get("reason"), say, Downloads(jobs.path))}

    def streamed(kind: str, chunks) -> Stream:
        """chunks as an answer sent as it is made, its first chunk made
        already, so a refusal is answered before anything is sent."""
        first = next(chunks)
        return Stream(kind, itertools.chain([first], chunks))

    @refusals
    def follow_head(values, body):
        return 200, follow.head(depot)

    @refusals
    def follow_rows(values, body):
        return 200, streamed("application/x-ndjson", follow.rows(
            depot, body["table"], body.get("ranges"), body.get("after"), body.get("texts")))

    @refusals
    def follow_parts(values, body):
        keys = body["keys"]
        if len(keys) > follow.PARTS_KEYS:
            raise errors.BadQuery(f"a request names at most {follow.PARTS_KEYS} parts, not {len(keys)}")
        return 200, streamed("application/octet-stream", follow.parts(depot, keys))

    status = {"enum": [ADMITTED, REFUSED]}
    library_ref = {"type": "object", "required": ["library"],
                   "properties": {"depot": {"type": ["string", "null"], "description": "this depot's address, in "
                                            "any form; absent or null: this depot; another depot's address is "
                                            "refused (422), only the commands name one"}, "library": TEXT}}
    share = {"type": "object", "required": ["tag", "share"], "properties": {"tag": TEXT, "share": {"type": "number"}}}
    max_tokens = {"type": ["integer", "null"], "minimum": 1, "description": "the most tokens of plain text the "
                  "corpus takes: the texts go in the corpus's order and each share stops at its first text that "
                  "would pass its part; absent or null: no limit"}
    needle = {"type": "object", "required": ["text"],
              "properties": {"text": TEXT, "start": {"type": ["integer", "null"]}, "end": {"type": ["integer", "null"]},
                             "field": {"type": ["string", "null"], "description": "for an item: the JSON Pointer "
                                       "(RFC 6901) of the string of its content that start and end count in"},
                             "version": {"type": ["integer", "null"], "description": "the content version of the "
                                         "text start and end count in; absent or null: its current one"}},
              "description": "characters [start, end) of a text, or of the field of an item, in one content version; "
                             "start and end absent or null: the whole text or item"}
    nullable = {"type": ["string", "null"]}
    text_filter = {"type": "object", "description": "a text matches when it matches every part given; no part: "
                   "every text",
                   "properties": {"ids": {"type": ["array", "null"], "items": TEXT,
                                          "description": "only these texts; null or absent: any text"},
                                  "tags": {**NAMES, "description": "the text has every one"},
                                  "licenses": {**NAMES, "description": "the text names one of them"},
                                  "statuses": {"type": "array", "items": {"enum": list(TEXT_STATUSES)},
                                               "description": "the text has one of them"},
                                  "libraries": {**NAMES, "description": "one of them holds the text"},
                                  "locked": {"type": ["boolean", "null"],
                                             "description": "whether a locked library holds the text"},
                                  "where": {**nullable, "description": "an SQL condition over the texts table, in "
                                            "SQLite's SQL, which only reads the depot's file (400) and, with the "
                                            f"rest of the request's reads, stops after {limits.SECONDS} seconds or "
                                            f"past {limits.MEMORY} bytes of memory (422)"}}}
    selection = {"type": "object", "required": ["filter"], "description": "the texts an operation takes: those the "
                 "filter matches when it runs, less the ids unticked and every text changed after the change "
                 "number change (its revision, a decision on it, its libraries, or a pull), which the answer counts "
                 f"in left under {CHANGED_SINCE!r}",
                 "properties": {"filter": text_filter, "change": {"type": "integer", "minimum": 0,
                                                                  "description": "the change number the page showing "
                                                                  "the filter was read at (a search's change)"},
                                "unticked": {**NAMES, "description": "ids the person left out by hand"}}}
    left = {"type": "object", "additionalProperties": {"type": "integer"},
            "description": "how many texts the operation left out, by reason"}
    job_answer = {"type": "object", "required": ["id", "kind", "state", "stages", "started", "ended", "answer",
                                                "problem", "status"],
                 "description": "a write's job: state queued or running (the answer 202), done (answer: the "
                 "operation's answer), refused, failed or cancelled (problem and status: the HTTP status of the "
                 "refusal; nothing was written); stages, each stage the operation started, with its time in seconds "
                 "since the epoch",
                 "properties": {"state": {"enum": ["queued", "running", "done", "refused", "failed", "cancelled"]},
                                "stages": {"type": "array", "items": {"type": "object", "required": ["stage", "at"]}}}}

    def written(answer: dict) -> dict:
        """The state of a job whose operation answers answer, the schema of a route that starts one."""
        return {**job_answer, "properties": {**job_answer["properties"], "answer": {
            "oneOf": [{"type": "null"}, {**answer, "properties": {**answer.get("properties", {}), "left": left}}]}}}
    return [
        *builds(depot, jobs, refusals),
        Route("GET", "/api/about", about, "the depot's public address (WPT_DEPOT_URL, else its name), "
              "which names it in the corpora it builds, its change number, its current about row, the "
              "words of the confirmation every decision needs, and sorted, how many texts a search matches at most "
              f"for an order other than {' or '.join(INDEXED)}",
              answer={"type": "object", "required": ["address", "change", "about", "confirmation", "requirements",
                                                     "sorted"]}),
        Route("POST", "/api/sql", sql, "the rows of one read query over the depot's tables (wpt depot sql): texts, "
              "libraries, datasets, about and erasures are the current state (texts with their text, kind and "
              "content, and libraries, the libraries holding them now), <name>_revisions every revision row of "
              "each (its id, revision and the change number it was made at; a text's revision names the content version "
              "it reads), contents every content version of each text with the change number it was made at, "
              "content_lists each content version as the list of its parts, parts each part of each text, "
              "memberships each library's texts with the change numbers they joined and left at, pins the "
              "revision each locked library holds of each of its texts, and pulls, for a depot that follows "
              "another, each pull's change number with the upstream's change number it brought; licenses, notices, kinds, decisions and "
              "changes (each change's number, kind and time) as they are; a list is JSON text (a tag test: EXISTS "
              "(SELECT 1 FROM json_each(tags) WHERE value = 'book')), a time ISO 8601 text and a byte string "
              "hexadecimal text; the query is one statement of SQLite's SQL that only reads the depot's file (a "
              "write, an attached file, a pragma or an extension is refused with 400), and "
              f"stops after {limits.SECONDS} seconds or past {limits.MEMORY} bytes of memory (422); an answer "
              f"of more than {limits.ANSWER_ROWS} rows or {limits.ANSWER_BYTES} bytes of JSON is refused (422)",
              body={"type": "object", "required": ["query"], "properties": {"query": TEXT}},
              answer={"type": "object", "required": ["columns", "rows"]}),
        Route("GET", "/api/kinds", kinds, "the kinds of items: each one's id, meaning and JSON Schema", answer=ROWS),
        Route("GET", "/api/licenses", licenses, "the licenses the depot holds, by id: each one's id and name, "
              "whether the allowed list accepts it (a combined row only if every part is), and how many texts name "
              "it", answer=ROWS),
        Route("GET", "/api/licenses/{id}", license, "one license: its id, name and official text, whether the "
              "allowed list accepts it, and how many texts name it",
              answer={"type": "object", "required": ["id", "name", "text", "allowed", "texts"]}),
        Route("POST", "/api/licenses", add_license, "a license with its official text, stored once (wpt depot "
              "license add)",
              body={"type": "object", "required": ["id", "name", "text"],
                    "properties": {"id": TEXT, "name": TEXT, "text": TEXT}}, answer=OBJECT),
        Route("POST", "/api/texts", add, "add a plain text (text) or an item (kind and its content), under review "
              "(wpt depot add)",
              body={"type": "object", "required": ["licenses", "declaration"],
                    "properties": {"text": {"type": "string"}, "kind": TEXT, "content": {},
                                   "licenses": {**NAMES, "minItems": 1}, "notice": nullable, "title": nullable,
                                   "authors": {"type": ["array", "null"], "items": TEXT}, "tags": NAMES,
                                   "address": nullable, "declaration": DECLARATION}}, answer=ID),
        Route("GET", "/api/texts/{id}", show, "one text or item as it is now, with its status, its notice's text, its "
              "decisions and libraries, how many revisions and content versions it has, the revision each locked "
              "library holding it pinned, and whether it belongs to the upstream this depot follows, which alone changes it (wpt depot "
              "show)",
              answer={"type": "object", "required": ["text", "status", "notice", "decisions", "libraries", "revisions",
                                                     "pins", "content_versions", "upstream"]}),
        Route("POST", "/api/texts/{id}/credits", credits, "change the credits named in the body (title, authors, "
              "tags, address, notice; null clears one) of one text as a new revision, its id kept (wpt depot "
              "credits): the depot records it as a decision under review whose reason names each credit's old and "
              "new value, and the decisions before it keep counting; a locked library keeps the revision it pinned; "
              "refused when nothing changes: the decision's id and its reason",
              body={"type": "object", "required": ["declaration"],
                    "properties": {"title": nullable, "authors": {"type": ["array", "null"], "items": TEXT},
                                   "tags": NAMES, "address": nullable, "notice": nullable,
                                   "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decision", "reason"]}),
        Route("POST", "/api/texts/scores", scores, "change the scores of many texts at once, each as a new "
              "revision, its id kept (wpt depot scores): for each text named once, the scores to set (a name it has "
              "already is replaced; the value a finite number, the content version one of the text's, the one the "
              "score was made from) and the names of the scores to remove; the depot records it as one decision "
              "under review on every text it changes, whose reason names the scores set and removed, and the "
              "decisions before it keep counting; a locked library keeps the revision it pinned; a text whose "
              "scores stay as they are is left out, and the change is refused when every text is: the decision's "
              "id, its reason, how many texts changed and how many were left out",
              body={"type": "object", "required": ["texts", "declaration"],
                    "properties": {"texts": {"type": "array", "minItems": 1, "items": {
                        "type": "object", "required": ["id"], "additionalProperties": False,
                        "properties": {"id": TEXT, "scores": {"type": "array", "items": {
                            "type": "object", "required": ["name", "value", "content_version"],
                            "additionalProperties": False,
                            "properties": {"name": TEXT, "value": {"type": "number"},
                                           "content_version": {"type": "integer", "minimum": 1}}}},
                                       "remove": NAMES}}},
                                   "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decision", "reason", "changed", "left_out"],
                      "properties": {"decision": TEXT, "reason": TEXT, "changed": {"type": "integer"},
                                     "left_out": {"type": "integer"}}}),
        Route("POST", "/api/texts/{id}/content", content, "change the content of one text, its plain text (text) "
              "or an item's content (content, which must match its kind), exactly one of them, the one that matches "
              "the text, as a new content version and a new revision, its id kept (wpt depot content): the depot "
              "records it as a decision under review whose reason names the old and new content versions, the "
              "decisions before it no longer count, and the text leaves every open library; a locked library keeps "
              "the content of the revision it pinned: the decision's id and the new content version, both null "
              "when the content is the one it has and nothing is written",
              body={"type": "object", "required": ["declaration"],
                    "oneOf": [{"required": ["text"]}, {"required": ["content"]}],
                    "properties": {"text": {"type": "string"}, "content": {}, "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decision", "version"]}),
        Route("POST", "/api/texts/search", search, "the texts a filter matches, a page at a time, with no count "
              "(POST /api/texts/counts counts them): the SQL condition over the texts table the filter stands for, "
              "the page of at most limit rows in the order asked (then by id, in the numeric order of ids; "
              "descending turns the whole order around), each a text's credits, status and length in characters, "
              "without its text; next, the place of the page's last row when more rows follow it, and previous, the "
              "place of its first row when rows come before it, each null otherwise; a page starts after a place "
              "(after) or ends before one (before), so a far page is read as fast as the first, and a page before a "
              "place with fewer than limit rows before it is the first page; an order other than "
              f"{' or '.join(INDEXED)} sorts every match, so it is refused (422) when more than {SORTED} texts "
              "match; change, the depot's change number the page was read at, which a selection names",
              body={"type": "object", "required": ["filter"], "not": {"required": ["after", "before"]},
                    "properties": {"filter": text_filter, "after": PLACE, "before": PLACE,
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                                   "order": {"enum": list(ORDERS)}, "descending": {"type": "boolean"}}},
              answer={"type": "object", "required": ["rows", "next", "previous", "change"], "properties": {
                  "next": NULLABLE_PLACE, "previous": NULLABLE_PLACE, "change": {"type": "integer"}}}),
        Route("POST", "/api/texts/passages", passages, "the texts a filter matches with their text, a page at a "
              "time in the order of POST /api/texts/search, the page bounded both by a count of texts (limit) and "
              "by characters: after the place after (from the first text without it), at most limit texts holding at "
              "most characters characters together, each its id, kind, title, length in characters and passage, its "
              "text or an item's content from its start, whole, or cut at characters for the page's first text when "
              "it is longer; a later text that does not fit whole starts the next page; next is the place of the "
              "page's last text, after which the next page starts, null when the page ends the texts matched",
              body={"type": "object", "required": ["filter"],
                    "properties": {"filter": text_filter, "after": PLACE,
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                                   "characters": {"type": "integer", "minimum": 1, "maximum": MAX_PASSAGE_CHARACTERS},
                                   "order": {"enum": list(ORDERS)}, "descending": {"type": "boolean"}}},
              answer={"type": "object", "required": ["rows", "next"], "properties": {
                  "rows": {"type": "array", "items": {"type": "object", "required": [
                      "id", "kind", "title", "length", "passage"]}},
                  "next": NULLABLE_PLACE}}),
        Route("POST", "/api/texts/counts", count, "one step of counting the texts a filter matches, so a page "
              "counts them in the background, a step after another, each bounded whatever the depot's size: the "
              f"slice is the size texts (at most {COUNT_STEP}) after the id start (from the first without it) in the "
              "order of their ids as text, or, for a filter that names its texts, exactly those, in one step; texts, "
              "how many texts the slice holds; total, how many texts every step goes over, in the first step only "
              "(no start), null after it; the matches of the slice by status (the status the depot stores), how "
              "many of them a locked library holds, how many an admission cannot cover by reason (as POST "
              "/api/texts/blocked), by tag and by license; next, the start of the next step, null after the last "
              "one; with change and unticked, the filter is counted as a selection (POST /api/decisions) takes it",
              body={"type": "object", "required": ["filter"], "properties": {
                  "filter": text_filter, "start": TEXT, "size": {"type": "integer", "minimum": 1,
                                                                 "maximum": COUNT_STEP},
                  "change": selection["properties"]["change"], "unticked": selection["properties"]["unticked"]}},
              answer={"type": "object", "required": ["texts", "total", "statuses", "locked", "blocked", "tags",
                                                     "licenses", "next"],
                      "properties": {"texts": {"type": "integer"}, "total": {"type": ["integer", "null"]},
                                     "next": {"type": ["string", "null"]}}}),
        Route("POST", "/api/texts/blocked", blocked, "of the texts a selection takes, those a decision of that "
              "status (default admitted) cannot cover, the check a decision applies, a page at a time in the "
              "numeric order of ids: each its id and its reasons (it belongs to the upstream this depot follows, "
              "for any decision; none of its licenses is allowed, for an admission); next, the place the next page "
              "starts after (after), null after the last page",
              body={"type": "object", "required": ["selection"],
                    "properties": {"selection": selection, "status": status, "after": PLACE,
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 500}}},
              answer={"type": "object", "required": ["rows", "next"], "properties": {"next": NULLABLE_PLACE,
                  "rows": {"type": "array", "items": {
                      "type": "object", "required": ["id", "reasons"],
                      "properties": {"id": TEXT, "reasons": {"type": "array", "items": {
                          "enum": [BELONGS_TO_UPSTREAM, NO_ALLOWED_LICENSE]}}}}}}}),
        Route("GET", "/api/tags", tags, "every tag a text has, with how many texts have it", answer=ROWS),
        Route("GET", "/api/counts", depot_counts, "how many texts the depot holds, and of them how many have each "
              "status, each tag and each license, and how many each library holds, read at once",
              answer={"type": "object", "required": ["texts", "statuses", "tags", "licenses", "libraries"]}),
        Route("POST", "/api/decisions", decide, "one decision on the texts a selection takes, with an optional "
              "public reason, which never starts with 'licenses changed from ' (the depot's own record of a "
              "license change) (wpt depot decide): admitting is refused when a text has no allowed license, and any "
              "decision when a text belongs to the upstream, unless leave_blocked leaves such texts out; a job "
              "whose answer is the decision's id, how many texts it is on and how many it left out, by reason",
              body={"type": "object", "required": ["status", "selection", "declaration"],
                    "properties": {"status": status, "selection": selection, "reason": TEXT,
                                   "leave_blocked": {"type": "boolean"}, "declaration": DECLARATION}},
              answer=written({"type": "object", "required": ["decision", "texts", "left"]})),
        Route("POST", "/api/texts/licenses", relicense, "change the licenses of the texts a selection takes, as a new "
              "revision, their ids kept (wpt depot relicense): for each distinct old list, a decision under review "
              "whose reason the depot writes, 'licenses changed from OLD to NEW', after which the decisions before it "
              "no longer count, so each text is under review again and leaves every open library (a locked library "
              "keeps the revision it pinned); a text that names these licenses already is left out: a job whose "
              "answer is each decision's id and how many texts it is on, how many were skipped, and how many were "
              "left out, by reason",
              body={"type": "object", "required": ["selection", "licenses", "declaration"],
                    "properties": {"selection": selection, "licenses": {**NAMES, "minItems": 1},
                                   "declaration": DECLARATION}},
              answer=written({"type": "object", "required": ["decisions", "skipped", "left"]})),
        Route("GET", "/api/decisions/latest", latest, "the ten latest decisions, the newest first, each with how "
              "many texts and erasure requests it decides and the library whose reported problem it records",
              answer=ROWS),
        Route("GET", "/api/decisions/{id}", decision, "one decision: its row (status, declared name and email, "
              "time, reason), how many texts it is on, the erasure requests it decides, and the reported problem it "
              "records with its library",
              answer={"type": "object", "required": ["decision", "texts", "erasures", "problem"]}),
        Route("GET", "/api/libraries", libraries, "every library as it is now, with whether it belongs to the "
              "upstream this depot follows, which alone changes it", answer=ROWS),
        Route("GET", "/api/libraries/{id}/size", library_size, "how many texts the library holds now, from the "
              "depot's counts", answer={"type": "object", "required": ["id", "texts"]}),
        Route("POST", "/api/libraries/{id}/pins", pinned, "the revisions a locked library pinned, a page at a time "
              "(offset and limit, default 0 and 50): how many texts it pinned, how many of them have a newer "
              "current revision, and the page of rows, each the text's id, the pinned revision's title, the "
              "revision pinned, the text's current revision and its status when the library was locked, in the "
              "numeric order of ids; an open library pins none",
              body={"type": "object", "properties": {"offset": {"type": "integer", "minimum": 0},
                                                     "limit": {"type": "integer", "minimum": 1, "maximum": 500}}},
              answer={"type": "object", "required": ["total", "newer", "rows"]}),
        Route("POST", "/api/libraries", new_library, "a new open library (wpt depot library new)",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT}}, answer=ID),
        Route("POST", "/api/libraries/{id}/name", rename_library, "a new name for the library, open or locked (wpt "
              "depot library rename)",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT}}, answer=OBJECT),
        Route("POST", "/api/libraries/{id}/texts", put, "every admitted text a selection takes joins the open "
              "library (wpt depot library put): a job whose answer is how many joined it, how many were left out "
              "because they are not admitted, and how many were left out, by reason",
              body={"type": "object", "required": ["selection"], "properties": {"selection": selection}},
              answer=written({"type": "object", "required": ["joined", "left_out", "left"]})),
        Route("POST", "/api/libraries/{id}/taken-out", take_out, "every text a selection takes that the open "
              "library holds leaves it and keeps its status, admitted included (wpt depot library take-out): a job "
              "whose answer is how many were taken out, how many it did not hold, and how many were left out, by "
              "reason",
              body={"type": "object", "required": ["selection"], "properties": {"selection": selection}},
              answer=written({"type": "object", "required": ["taken_out", "not_held", "left"]})),
        Route("POST", "/api/libraries/{id}/lock", lock, "lock the library for good: it pins the current revision of "
              "each text it holds (wpt depot library lock)",
              body=OBJECT, answer=OBJECT),
        Route("POST", "/api/libraries/{id}/forks", fork, "a new open library holding the same texts that are "
              "admitted now, less the texts the selection without takes (wpt depot library fork): a job whose "
              "answer is its id",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT, "without": selection}},
              answer=written({"type": "object", "required": ["id", "left"]})),
        Route("POST", "/api/libraries/{id}/problems", report, "a problem with the library, locked or not: the "
              "texts of the library a selection takes (none: the library as a whole) and the problem in words (wpt "
              "depot library report): a job whose answer is the decision that records the reporter and how many "
              "texts the problem names",
              body={"type": "object", "required": ["problem", "declaration"],
                    "properties": {"problem": TEXT, "selection": selection, "declaration": DECLARATION}},
              answer=written({"type": "object", "required": ["decision", "texts", "left"]})),
        Route("GET", "/api/datasets", datasets, "every dataset as it is now, with whether it belongs to the "
              "upstream this depot follows, which alone changes it, and here, the ids of the libraries it reads "
              "from this depot: those this depot holds, named by its address in any form or, for a library of "
              "that upstream, by the upstream's", answer=ROWS),
        Route("POST", "/api/datasets", new_dataset, "a dataset reading libraries, each at its depot's change number now, "
              "with optional ordered shares and an optional limit of its corpus's tokens of plain text, whose "
              "vocabulary is that of the depot's tokenizer recipe it names (wpt depot dataset new)",
              body={"type": "object", "required": ["name", "tokenizer", "libraries"],
                    "properties": {"name": TEXT, "tokenizer": TEXT,
                                   "libraries": {"type": "array", "minItems": 1, "items": library_ref},
                                   "shares": {"type": ["array", "null"], "items": share},
                                   "max_tokens": max_tokens}}, answer=ID),
        Route("GET", "/api/datasets/{id}", dataset, "one dataset by its id: its current row as GET /api/datasets gives it, "
              "whether it belongs to the upstream this depot follows (which alone changes it), each library it "
              "reads with its change number and local, whether it is read here as GET /api/datasets says (and "
              "for such a library its name and whether it is locked), its remote, and the corpora the store holds "
              "built from it, through its renames, each with the revision of the dataset it was built from, the "
              "last built first (null, with why in store, when the store cannot be read)",
              answer={"type": "object", "required": ["dataset", "upstream", "libraries", "remote", "corpora",
                                                     "store"]}),
        Route("POST", "/api/datasets/{id}", edit_dataset, "change the dataset of that id as a new revision, its id "
              "and tokenizer kept, with the checks of a new one (wpt depot dataset edit): a library it read already "
              "keeps its change number, a new one is read at its depot's change number now; a corpus built before "
              "keeps its own copy",
              body={"type": "object", "required": ["name", "libraries"],
                    "properties": {"name": TEXT, "libraries": {"type": "array", "minItems": 1, "items": library_ref},
                                   "shares": {"type": ["array", "null"], "items": share},
                                   "max_tokens": max_tokens}}, answer=ID),
        Route("GET", "/api/datasets/{name}/remote", export, "the dataset as a remote (wpt depot dataset export)",
              answer=OBJECT),
        Route("GET", "/api/erasures", erasures, "every erasure request, with its status and whether it belongs to "
              "the upstream this depot follows or names a text of it, which the upstream alone decides "
              "(wpt depot erasures)",
              answer=ROWS),
        Route("GET", "/api/erasures/{id}", erasure_shown, "what one erasure request asks to erase, never a whole "
              f"text: for each needle, its place, its text's title and kind, and at most {CONTEXT} characters on "
              "each side of its passage (before, from the position from, and after) with the passage itself (one "
              f"longer than {2 * CONTEXT} characters as its two ends, passage and passage_end, and how many "
              "characters between them are omitted), or once "
              "the request is carried out, its length (erased) at the place it left; for a whole text, its head; its "
              "state: passage, erased, gone (the text is no longer in the depot) or misplaced (another erasure cut "
              "the text since, so the place no longer fits)",
              answer={"type": "object", "required": ["id", "reason", "status", "carried_out", "needles"],
                      "properties": {"needles": {"type": "array", "items": {
                          "type": "object", "required": ["text", "state", "before", "passage", "omitted", "passage_end",
                                                         "after", "from", "length", "erased"],
                          "properties": {"state": {"enum": ["passage", "erased", "gone", "misplaced"]}}}}}}),
        Route("POST", "/api/erasures", erase, "ask to erase passages of texts, or whole texts or items, under review "
              "(wpt depot erase)",
              body={"type": "object", "required": ["reason", "needles", "declaration"],
                    "properties": {"reason": TEXT, "needles": {"type": "array", "minItems": 1, "items": needle},
                                   "declaration": DECLARATION}}, answer=ID),
        Route("POST", "/api/erasures/decisions", decide_erasures, "the same decision, admitted or refused, on each "
              "erasure request named, with an optional public reason, every request checked before any is decided; "
              "each one admitted is carried out at once, and every corpus holding a text it cut or removed is "
              "marked stale (wpt depot decide --erasure): whether each was carried out, by id",
              body={"type": "object", "required": ["erasures", "status", "declaration"],
                    "properties": {"erasures": IDS, "status": status, "reason": TEXT, "declaration": DECLARATION}},
              answer={"type": "object", "required": ["carried_out"]}),
        Route("POST", "/api/follow/head", follow_head, "for a clone or a pull of this depot (wpt depot clone, "
              "pull): its name, its format, its change number and what it holds of each depot's numbering then",
              body=OBJECT, answer={"type": "object", "required": ["name", "format", "change", "bounds"]}),
        Route("POST", "/api/follow/rows", follow_rows, "for a clone or a pull: one page of the rows of a table "
              f"that travels ({', '.join(follow.TABLES)}), in the order of its key, from the key after which the "
              f"last page ended, at most {follow.PAGE_ROWS} rows or about {follow.PAGE_BYTES} bytes: JSON lines, "
              "each row an object, then {\"next\": the key the next page starts after, or null}; ranges: {depot "
              "name: [after, up to]}, the change numbers chosen of each depot's rows; texts, for content_lists, "
              "the ids of the texts whose every version is wanted",
              body={"type": "object", "required": ["table"],
                    "properties": {"table": {"enum": list(follow.TABLES)},
                                   "ranges": {"type": ["object", "null"], "additionalProperties": {
                                       "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
                                   "after": {"type": ["array", "null"]}, "texts": {"type": "array", "items": TEXT}}}),
        Route("POST", "/api/follow/parts", follow_parts, "for a clone or a pull: the parts named ([text id, hash in "
              f"hexadecimal], at most {follow.PARTS_KEYS}), each a JSON line {{id, hash, change, size}} then its "
              "stored value (size bytes, compressed), or {id, hash, missing: true} for one an erasure cut",
              body={"type": "object", "required": ["keys"],
                    "properties": {"keys": {"type": "array", "items": {
                        "type": "array", "items": TEXT, "minItems": 2, "maxItems": 2}}}}),
    ]
