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

from wpt_depot import Declaration, Depot, NewText, counts as facets, crediting, errors, follow, rewriting
from wpt_depot.filters import STATUSES as TEXT_STATUSES
from wpt_depot.depot import (ALREADY, ALREADY_UNDER_REVIEW, BELONGS_TO_UPSTREAM, CHANGED_SINCE, COUNT_STEP, CREDITS, INDEXED,
                             NO_ALLOWED_LICENSE, ORDERS, SORTED, TEXT_WINDOW, WINDOW)
from wpt_depot.erasure import CONTEXT
from wpt_depot.filters import Filter
from wpt_depot.schema import ADMITTED, REFUSED, UNDER_REVIEW
from wpt_depot.corpora import NoRoom, TooBig
from wpt_depot.corpora import triple as corpora_triple
from wpt_depot.corpus import measure as corpus_measure
from wpt_depot.corpus import revision_figures
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
NAMED_TITLES = 3  # the texts a job's title names, of a write over named texts
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
                  unticked=tuple(unticked), change=change, outside=tuple(found.get("outside", [])),
                  decision=found.get("decision"), search=found.get("search"), in_text=found.get("in_text", False),
                  whole_words=found.get("whole_words", False))


def selected(selection: dict) -> Filter:
    """The filter a body's selection stands for (SELECTION)."""
    return filter_of(selection["filter"], selection.get("change"), selection.get("unticked", ()))


def small(selection: dict) -> bool:
    """Whether the selection's filter names at most INLINE texts, so its write may run in the request."""
    ids = selection["filter"].get("ids")
    return ids is not None and len(ids) <= INLINE


# The operations a job runs, by name of function, so that a job's process imports them: each takes the depot, its
# arguments and progress, and gives the JSON answer of its route. left: {reason: how many texts the operation left
# out for it} (CHANGED_SINCE, and for a decision with leave_blocked, the reasons of Depot.blocked, with leave_same,
# ALREADY's).

def decide_texts(depot: Depot, status: str, selection: Filter, declaration: Declaration, reason: str | None,
                 leave_blocked: bool, leave_same: bool, progress) -> dict:
    decision, texts, left = depot.decide(status, selection, declaration, reason, leave_blocked, progress, leave_same)
    return {"decision": decision, "texts": texts, "left": left}


def relicense_texts(depot: Depot, selection: Filter, licenses: list[str], declaration: Declaration,
                    progress) -> dict:
    left = {}
    found, skipped = depot.relicense(selection, licenses, declaration, progress, left)
    return {"decisions": [{"decision": decision, "texts": texts} for decision, texts in found],
            "skipped": skipped, "left": left}


def score_texts(depot: Depot, selection: Filter, name: str, value: float | None, declaration: Declaration,
                progress) -> dict:
    left = {}
    decision, changed, left_out = depot.score(selection, name, value, declaration, progress, left)
    return {"decision": decision, "changed": changed, "left_out": left_out, "left": left}


def put_texts(depot: Depot, library: str, selection: Filter, progress) -> dict:
    left = {}
    joined, left_out = depot.put(library, selection, progress, left)
    return {"library": library, "joined": joined, "left_out": left_out, "left": left}


def take_out_texts(depot: Depot, library: str, selection: Filter, progress) -> dict:
    left = {}
    taken_out, not_held = depot.take_out(library, selection, progress, left)
    return {"library": library, "taken_out": taken_out, "not_held": not_held, "left": left}


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

    def job(title: str, few: bool, operation, *args) -> tuple[int, dict]:
        """Run operation over the depot with args, titled by what it does (the
        Jobs page shows it), in the request when it
        writes few texts and no write job is open (a write waits for a
        running write job's write lock, which a job shows and a request does
        not; a vocabulary training or a corpus build holds none), as a job
        otherwise: the job's state once it ended (200) or after WAIT
        seconds (202, running), or its refusal with its status."""
        if few and not jobs.writing():
            state = jobs.record("write", title, depot, operation, *args)
        else:
            state = jobs.wait(jobs.add("write", title, operation, *args)["id"], WAIT)
        if state["state"] in ("refused", "failed", "cancelled"):
            return state["status"] or 409, {"problems": [state["problem"]], "job": state["id"]}
        return 202 if state["state"] in ("queued", "running") else 200, jsonable(state)

    def library_named(library: str) -> str:
        """The library's name, for a job's title; its id when the depot holds none of that id (the job refuses it)."""
        found = depot.rows("SELECT name FROM libraries WHERE id = ?", (library,))
        return found[0]["name"] if found else library

    def text_names(ids: list[str]) -> dict[str, str]:
        """Each text's name as the pages give it, by id: its title, else its
        kind with an article ("a conversation"), else "an untitled text";
        its id when the depot holds no such text."""
        rows = {row["id"]: row for row in depot.rows(
            f"SELECT id, title, kind FROM texts WHERE id IN ({', '.join('?' * len(ids))})", tuple(ids))} if ids else {}

        def name(one: str) -> str:
            row = rows.get(one)
            if row is None:
                return one
            if row["title"]:
                return row["title"]
            if row["kind"]:
                return f"{'an' if row['kind'][0] in 'aeiou' else 'a'} {row['kind']}"
            return "an untitled text"
        return {one: name(one) for one in ids}

    def subject(selection: dict) -> str:
        """What a write over a selection acts on, for a job's title: the one
        text's name (named) when the selection names exactly one text, else
        the texts its filter takes in words ("texts under review, tagged
        prose, in the library mix"; "3 named texts (fizz.py, note.txt, a
        conversation), refused", the first NAMED_TITLES by their name;
        "every text")."""
        found = selection["filter"]
        ids = found.get("ids")
        if ids is not None and len(ids) == 1 and not selection.get("unticked"):
            return text_names(ids)[ids[0]]
        names = held_libraries()

        def named(libraries) -> str:
            return " or ".join(names[one]["name"] if one in names else one for one in libraries)
        words = [part for part in (
            " or ".join(found.get("statuses", [])) or None,
            f"tagged {' and '.join(found['tags'])}" if found.get("tags") else None,
            f"under {' or '.join(found['licenses'])}" if found.get("licenses") else None,
            f"in the library {named(found['libraries'])}" if found.get("libraries") else None,
            f"not in the library {named(found['outside'])}" if found.get("outside") else None,
            None if found.get("locked") is None else "in a locked library" if found["locked"]
            else "in no locked library",
            f"on the decision {found['decision']}" if found.get("decision") else None,
            f"holding “{found['search'].strip()}”{' as whole words' if found.get('whole_words') else ''}"
            f"{' in their text' if found.get('in_text') else ''}"
            if (found.get("search") or "").strip() else None,
            "matching an SQL condition" if (found.get("where") or "").strip() else None,
            f"less {len(selection['unticked']):,} unticked" if selection.get("unticked") else None) if part]
        if ids is not None:
            first = ids[:NAMED_TITLES]
            called = text_names(first)
            shown = ", ".join(called[one] for one in first) + (", ..." if len(ids) > len(first) else "")
            return ", ".join([f"{len(ids):,} named texts ({shown})" if first else "0 named texts", *words])
        return f"texts {', '.join(words)}" if words else "every text"

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
    def credits_many(values, body):
        decision, reason, changed, left_out = depot.edit_credits_many(body["texts"], declared(body))
        return 200, {"decision": decision, "reason": reason, "changed": changed, "left_out": left_out}

    @refusals
    def scores(values, body):
        decision, reason, changed, left_out = depot.edit_scores(body["texts"], declared(body))
        return 200, {"decision": decision, "reason": reason, "changed": changed, "left_out": left_out}

    @refusals
    def score_selected(values, body):
        value = None if body.get("remove") else body["value"]
        verb = (f"remove the score {body['name']} from" if value is None
                else f"set the score {body['name']} to {value:g} on")
        return job(f"{verb} {subject(body['selection'])}", small(body["selection"]), score_texts,
                   selected(body["selection"]), body["name"], value, declared(body))

    @refusals
    def score_names(values, body):
        return 200, depot.score_names()

    @refusals
    def content(values, body):
        changed = depot.change_content(values["id"], body.get("text"),
                                       json.dumps(body["content"], ensure_ascii=False) if "content" in body else None,
                                       declared(body))
        decision, version = changed if changed is not None else (None, None)
        return 200, {"decision": decision, "version": version}

    @refusals
    def content_many(values, body):
        done = depot.change_content_many(body["texts"], declared(body))
        return 200, {"decisions": [{"decision": decision, "reason": reason, "texts": texts}
                                   for decision, reason, texts in done.decisions],
                     "changed": done.changed, "unchanged": done.unchanged, "stale": done.stale,
                     "stale_ids": done.stale_ids}

    @refusals
    def search(values, body):
        return 200, readers.run("search", filter_of(body["filter"]), body.get("after"),
                                body.get("before"), body.get("limit", 50), body.get("order", "id"),
                                body.get("descending", False), body.get("at"))

    @refusals
    def passages(values, body):
        return 200, readers.run(
            "passages", filter_of(body["filter"]), body.get("after"), body.get("limit", 50),
            body.get("characters", PASSAGE_CHARACTERS), body.get("order", "id"), body.get("descending", False))

    @refusals
    def count(values, body):
        found = filter_of(body["filter"], body.get("change"), body.get("unticked", ()))
        if "start" not in body and body.get("held", True):
            held = depot.held_count(found)
            if held is not None:
                return 200, held
        return 200, readers.run("count", found, body.get("start"),
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
        verb = {ADMITTED: "admit", REFUSED: "refuse", UNDER_REVIEW: "put back under review"}[body["status"]]
        return job(f"{verb} {subject(body['selection'])}", small(body["selection"]),
                   decide_texts, body["status"], selected(body["selection"]), declared(body), body.get("reason"),
                   body.get("leave_blocked", False), body.get("leave_same", False))

    @refusals
    def relicense(values, body):
        return job(f"change the licenses of {subject(body['selection'])}", small(body["selection"]), relicense_texts,
                   selected(body["selection"]), body["licenses"], declared(body))

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
    def measure(values, body):
        return 200, jsonable(corpus_measure(depot, body["libraries"], body.get("shares") or []))

    @refusals
    def change_times(values, body):
        found = depot.change_times(body["changes"])
        return 200, [{"change": number, "made_at": found.get(number)} for number in dict.fromkeys(body["changes"])
                     if number in found]

    @refusals
    def pinned(values, body):
        return 200, jsonable(depot.pinned(values["id"], body.get("offset", 0), body.get("limit", 50), body.get("after")))

    @refusals
    def new_library(values, body):
        return 200, {"id": depot.new_library(body["name"])}

    @refusals
    def rename_library(values, body):
        depot.rename_library(values["id"], body["name"])
        return 200, {}

    @refusals
    def put(values, body):
        return job(f"put texts in the library {library_named(values['id'])}", small(body["selection"]), put_texts,
                   values["id"], selected(body["selection"]))

    @refusals
    def take_out(values, body):
        return job(f"take texts out of the library {library_named(values['id'])}", small(body["selection"]),
                   take_out_texts, values["id"], selected(body["selection"]))

    @refusals
    def lock(values, body):
        depot.lock(values["id"])
        return 200, {}

    @refusals
    def fork(values, body):
        without = body.get("without")
        # a fork writes a membership for every admitted text of the library, whatever without takes
        held = depot.counts()[facets.LIBRARY].get(values["id"], 0)
        return job(f"fork the library {library_named(values['id'])} as {body['name']}", held <= INLINE, fork_library,
                   values["id"], body["name"],
                   None if without is None else selected(without))

    @refusals
    def report(values, body):
        chosen = body.get("selection")
        return job(f"report a problem with the library {library_named(values['id'])}", chosen is None or small(chosen),
                   report_problem, values["id"], body["problem"],
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

    def own(libraries, read: list[dict] = ()) -> list[tuple[str | None, str]]:
        """The libraries a dataset of the body names, as (depot address,
        library id). Refused, before any depot is opened, for a library
        named by another depot's address that is not one the dataset reads
        already (read: its stored libraries, which an edit keeps as they are,
        never opening their depot): the server answers any client, so it
        opens no address a client gives (`wpt depot dataset` names them)."""
        stored = {(one["depot"], one["library"]) for one in read}
        others = sorted({one["depot"] for one in libraries
                         if one.get("depot") is not None and not depot.is_here(one["depot"])
                         and (one["depot"], one["library"]) not in stored})
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
                              "figures": revision_figures(depot, found),
                              "read_again": depot.read_again_shows(found["id"]),
                              "corpora": None if corpora is None else [
                                  {**{key: one.get(key) for key in ("id", "built_at", "texts", "conversations",
                                                                    "total_tokens", "items_tokens", "stale", "deleted")},
                                   "revision": one["dataset"]["revision"], "vocabulary": corpora_triple(one)[1],
                                   "max_tokens": corpora_triple(one)[2]} for one in corpora],
                              "store": unreachable})

    @refusals
    def edit_dataset(values, body):
        shares = body.get("shares")
        depot.edit_dataset(values["id"], body["name"],
                           own(body["libraries"], depot.dataset_by_id(values["id"])["libraries"]),
                           None if shares is None else [(one["tag"], one["share"]) for one in shares],
                           body.get("max_tokens"), body.get("vocabulary"), body.get("revision"))
        return 200, {"id": values["id"]}

    @refusals
    def read_again(values, body):
        return 200, {"id": values["id"], "libraries": depot.read_again(values["id"], body.get("revision"))}

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
    text_status = {"enum": [ADMITTED, REFUSED, UNDER_REVIEW]}
    library_ref = {"type": "object", "required": ["library"],
                   "properties": {"depot": {"type": ["string", "null"], "description": "this depot's address, in "
                                            "any form; absent or null: this depot; another depot's address is "
                                            "refused (422), only the commands name one, but in an edit, a library "
                                            "of another depot the dataset reads already, as it stores it"}, "library": TEXT}}
    share = {"type": "object", "required": ["tag", "share"], "properties": {"tag": TEXT, "share": {"type": "number"}}}
    max_tokens = {"type": ["integer", "null"], "minimum": 1, "description": "the size the build form starts at "
                  "(nothing else reads it): the most tokens the corpus holds, conversations and tasks taking their "
                  "natural share; absent or null: no limit"}
    needle = {"type": "object", "required": ["text"],
              "properties": {"text": TEXT, "start": {"type": ["integer", "null"]}, "end": {"type": ["integer", "null"]},
                             "field": {"type": ["string", "null"], "description": "for an item: the JSON Pointer "
                                       "(RFC 6901) of the string of its content that start and end count in"},
                             "version": {"type": ["integer", "null"], "description": "the content version of the "
                                         "text start and end count in; absent or null: its current one"}},
              "description": "characters [start, end) of a text, or of the field of an item, in one content version; "
                             "start and end absent or null: the whole text or item"}
    nullable = {"type": ["string", "null"]}
    # the credits a line of a change of many texts' credits names, each optional, null clearing it
    credits_given = {"title": nullable, "authors": {"type": ["array", "null"], "items": TEXT},
                "tags": {**NAMES, "type": ["array", "null"]}, "address": nullable, "notice": nullable}
    text_filter = {"type": "object", "description": "a text matches when it matches every part given; no part: "
                   "every text",
                   "properties": {"ids": {"type": ["array", "null"], "items": TEXT,
                                          "description": "only these texts; null or absent: any text"},
                                  "tags": {**NAMES, "description": "the text has every one"},
                                  "licenses": {**NAMES, "description": "the text names one of them"},
                                  "statuses": {"type": "array", "items": {"enum": list(TEXT_STATUSES)},
                                               "description": "the text has one of them"},
                                  "libraries": {**NAMES, "description": "one of them holds the text"},
                                  "outside": {**NAMES, "description": "none of these libraries holds the text"},
                                  "locked": {"type": ["boolean", "null"],
                                             "description": "whether a locked library holds the text"},
                                  "decision": {**nullable, "description": "the id of a decision the text is on"},
                                  "search": {**nullable, "description": "words, split at spaces, each in the text's "
                                             "title (with in_text, in its text too), in any case, or its exact id; "
                                             "% and _ are characters, not wildcards; without in_text, a word of 3 "
                                             "characters or more reads the texts from the title index"},
                                  "in_text": {"type": "boolean", "description": "the search reads each text itself "
                                              "too, a plain text's text or an item's JSON content"},
                                  "whole_words": {"type": "boolean", "description": "each word of the search "
                                                  "matches only where no letter, digit or underscore stands right "
                                                  "before or after it"},
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
        Route("POST", "/api/texts/credits", credits_many, "change the credits of many texts at once, each as a "
              "new revision, its id kept (wpt depot credits-file): for each text named once, the credits to change "
              "(title, authors, tags, address, notice; null clears one, one left out stays), and under was the "
              "values its writer read of some credits, the text changing only while they still hold; the depot "
              "records it "
              "as one decision under review on every text it changes, whose reason names the credits changed, and "
              "the decisions before it keep counting; a locked library keeps the revision it pinned; a text whose "
              "credits stay as they are, or whose was no longer holds, is left out, and the change is refused when "
              "every text is, or when it names "
              f"more than {crediting.MOST:,} texts: the decision's id, its reason, how many texts changed and how "
              "many were left out",
              body={"type": "object", "required": ["texts", "declaration"],
                    "properties": {"texts": {"type": "array", "minItems": 1, "maxItems": crediting.MOST, "items": {
                        "type": "object", "required": ["id"], "additionalProperties": False,
                        "properties": {"id": TEXT, **credits_given,
                                       "was": {"type": "object", "additionalProperties": False,
                                               "properties": credits_given}}}},
                                   "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decision", "reason", "changed", "left_out"],
                      "properties": {"decision": TEXT, "reason": TEXT, "changed": {"type": "integer"},
                                     "left_out": {"type": "integer"}}}),
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
        Route("POST", "/api/texts/content", content_many, "change the content of many plain texts at once, each "
              "as POST /api/texts/{id}/content changes one (wpt depot content-file): for each text named once, its "
              "new plain text, and under was the content version its writer read, the text changing only while "
              "that is still its current one; the depot records one decision under review for each pair of old and "
              "new content versions, on the texts it moves, whose reason names them, the decisions before it no "
              "longer count, and the texts leave every open library; a locked library keeps the content of the "
              "revision it pinned; a text whose content is that already, or whose was no longer holds, is left "
              "out, and the change is refused when every text is, when it names an item or more than "
              f"{rewriting.MOST:,} texts, or when a new text holds a passage an erasure removed from it: each "
              "decision with its reason and how many texts it is on, how many texts changed, how many were left "
              "out unchanged, how many because their was no longer holds, and the first of those",
              body={"type": "object", "required": ["texts", "declaration"],
                    "properties": {"texts": {"type": "array", "minItems": 1, "maxItems": rewriting.MOST, "items": {
                        "type": "object", "required": ["id", "text"], "additionalProperties": False,
                        "properties": {"id": TEXT, "text": {"type": "string"},
                                       "was": {"type": "integer", "minimum": 1}}}},
                                   "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decisions", "changed", "unchanged", "stale", "stale_ids"],
                      "properties": {"decisions": {"type": "array", "items": {
                          "type": "object", "required": ["decision", "reason", "texts"],
                          "properties": {"decision": TEXT, "reason": TEXT, "texts": {"type": "integer"}}}},
                          "changed": {"type": "integer"}, "unchanged": {"type": "integer"},
                          "stale": {"type": "integer"}, "stale_ids": {"type": "array", "items": TEXT}}}),
        Route("POST", "/api/texts/search", search, "the texts a filter matches, a page at a time, with no count "
              "(POST /api/texts/counts counts them): the SQL condition over the texts table the filter stands for, "
              "the page of at most limit rows in the order asked (then by id, in the numeric order of ids; "
              "descending turns the whole order around), each a text's credits, status and length in characters, "
              "without its text; next, the place where the next page starts when rows may follow, and previous, the "
              "place of its first row when rows come before it, each null otherwise; a page starts after a place "
              "(after) or ends before one (before), so a far page is read as fast as the first, and a page before a "
              "place with fewer than limit rows before it is the first page; a page at a text's id (at) starts at "
              "that text's place in the order, with the text when the filter matches it (the first page for an id "
              "the depot does not hold); a page reads its candidates from the narrowest listing the filter names "
              "(one tag, license, library or status, in the order of ids from its index; the texts whose title "
              "the title index gives for the search's words of 3 characters or more, and those named by their id; "
              f"a decision, several licenses or libraries, or ids, sorted when at most {SORTED}; else every text), "
              "at most "
              f"{WINDOW} of them ({TEXT_WINDOW} with a search in the texts) in the order of ids or of statuses, so "
              "a page whose filter few texts match can hold fewer than limit rows: partial is then true and next "
              "names the last candidate read, where the next page reads on; an order other than "
              f"{' or '.join(INDEXED)} sorts every candidate, so it is refused (422) when the listing holds more "
              f"than {SORTED} texts, and sortable says whether the listing allows every order; change, the "
              "depot's change number the page was read at, which a selection names",
              body={"type": "object", "required": ["filter"],
                    "not": {"anyOf": [{"required": ["after", "before"]}, {"required": ["after", "at"]},
                                      {"required": ["before", "at"]}]},
                    "properties": {"filter": text_filter, "after": PLACE, "before": PLACE, "at": TEXT,
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                                   "order": {"enum": list(ORDERS)}, "descending": {"type": "boolean"}}},
              answer={"type": "object", "required": ["rows", "next", "previous", "partial", "sortable", "change"],
                      "properties": {"next": NULLABLE_PLACE, "previous": NULLABLE_PLACE, "partial": {"type": "boolean"},
                                     "sortable": {"type": "boolean"}, "change": {"type": "integer"}}}),
        Route("POST", "/api/texts/passages", passages, "the texts a filter matches with their text, a page at a "
              "time in the order of POST /api/texts/search, the page bounded both by a count of texts (limit) and "
              "by characters: after the place after (from the first text without it), at most limit texts holding at "
              "most characters characters together, each its id, kind, title, length in characters and passage, its "
              "text or an item's content from its start, whole, or cut at characters for the page's first text when "
              "it is longer; a later text that does not fit whole starts the next page; next is the place after which "
              "the next page starts, null when the page ends the texts matched; the candidates are read as POST "
              "/api/texts/search reads them, so partial is true for a page cut at its reading bound",
              body={"type": "object", "required": ["filter"],
                    "properties": {"filter": text_filter, "after": PLACE,
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                                   "characters": {"type": "integer", "minimum": 1, "maximum": MAX_PASSAGE_CHARACTERS},
                                   "order": {"enum": list(ORDERS)}, "descending": {"type": "boolean"}}},
              answer={"type": "object", "required": ["rows", "next", "partial"], "properties": {
                  "rows": {"type": "array", "items": {"type": "object", "required": [
                      "id", "kind", "title", "length", "passage"]}},
                  "next": NULLABLE_PLACE, "partial": {"type": "boolean"}}}),
        Route("POST", "/api/texts/counts", count, "one step of counting the texts a filter matches, so a page "
              "counts them in the background, a step after another, each bounded whatever the depot's size: the "
              f"slice is the size rows (at most {COUNT_STEP}) after the place start (from the first without it) of "
              "the listing a page reads (POST /api/texts/search), in the order of ids, or, for a filter that names "
              "its texts, exactly those, in one step; texts, how many rows the slice holds; total, how many rows "
              "every step goes over, in the first step only "
              f"(no start), null after it, and null too for a search's texts past {COUNT_STEP} titles; the "
              "matches of the slice by status (the status the depot stores), how "
              "many of them a locked library holds, how many an admission cannot cover by reason (as POST "
              "/api/texts/blocked), by tag and by license; next, the start of the next step, null after the last "
              "one; with change and unticked, the filter is counted as a selection (POST /api/decisions) takes it; "
              "held, false for such a step; a first step (no start) of a filter of every text, or of one status, "
              "tag, license or library, unless held is false in the body, is read at once from the depot's counts "
              "when they give how many texts it matches, their split by status and how many a locked library "
              "holds: then held is true, texts and total are every text, next is null, and tags, licenses and "
              "blocked are null where the counts do not give them, which steps with held false then count",
              body={"type": "object", "required": ["filter"], "properties": {
                  "filter": text_filter, "start": TEXT, "size": {"type": "integer", "minimum": 1,
                                                                 "maximum": COUNT_STEP},
                  "change": selection["properties"]["change"], "unticked": selection["properties"]["unticked"],
                  "held": {"type": "boolean"}}},
              answer={"type": "object", "required": ["texts", "total", "statuses", "locked", "blocked", "tags",
                                                     "licenses", "next", "held"],
                      "properties": {"texts": {"type": "integer"}, "total": {"type": ["integer", "null"]},
                                     "next": {"type": ["string", "null"]}, "held": {"type": "boolean"},
                                     "tags": {"type": ["array", "null"]}, "licenses": {"type": ["array", "null"]},
                                     "blocked": {"type": ["object", "null"]}}}),
        Route("POST", "/api/texts/blocked", blocked, "of the texts a selection takes, those a decision of that "
              "status (default admitted) cannot cover, the check a decision applies, a page at a time in the "
              "numeric order of ids: each its id and its reasons (it belongs to the upstream this depot follows, "
              "for any decision; none of its licenses is allowed, for an admission); next, the place the next page "
              "starts after (after), null after the last page; answered at once from the depot's counts when no "
              "upstream is followed and the decision is no admission or every license a text names is allowed; "
              "else read as POST /api/texts/search reads, from the narrowest of the selection's listing and the "
              "texts that name a license that is not allowed, partial true for a page cut at its reading bound",
              body={"type": "object", "required": ["selection"],
                    "properties": {"selection": selection, "status": text_status, "after": PLACE,
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 500}}},
              answer={"type": "object", "required": ["rows", "next", "partial"], "properties": {"next": NULLABLE_PLACE,
                  "rows": {"type": "array", "items": {
                      "type": "object", "required": ["id", "reasons"],
                      "properties": {"id": TEXT, "reasons": {"type": "array", "items": {
                          "enum": [BELONGS_TO_UPSTREAM, NO_ALLOWED_LICENSE]}}}}}}}),
        Route("GET", "/api/tags", tags, "every tag a text has, with how many texts have it", answer=ROWS),
        Route("GET", "/api/counts", depot_counts, "how many texts the depot holds, and of them how many have each "
              "status, each tag and each license, and how many each library holds, read at once",
              answer={"type": "object", "required": ["texts", "statuses", "tags", "licenses", "libraries"]}),
        Route("POST", "/api/decisions", decide, "one decision on the texts a selection takes, with an optional "
              "public reason, which never starts with 'licenses changed from ', 'content changed from version ' or "
              "'put back under review' (the depot's own records) (wpt depot decide): admitting is refused when a "
              "text has no allowed license, and any decision when a text belongs to the upstream, unless "
              "leave_blocked leaves such texts out; the status under review puts the texts back under review, "
              "undoing their admission or refusal: the depot writes its reason, 'put back under review' (then ': ' "
              "and the reason given), the decisions before it no longer count, the texts leave every open library, "
              f"and a text under review already is left out ({ALREADY_UNDER_REVIEW!r}); leave_same leaves out of an "
              f"admission or a refusal the texts admitted or refused already ({ALREADY[ADMITTED]!r}, "
              f"{ALREADY[REFUSED]!r}), so it records no decision that changes no status; a job whose answer is the "
              "decision's id, how many texts it is on and how many it left out, by reason",
              body={"type": "object", "required": ["status", "selection", "declaration"],
                    "properties": {"status": text_status, "selection": selection, "reason": TEXT,
                                   "leave_blocked": {"type": "boolean"}, "leave_same": {"type": "boolean"},
                                   "declaration": DECLARATION}},
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
        Route("POST", "/api/texts/score", score_selected, "set one score on the texts a selection takes, "
              "each made from its current content version, or remove it from them (remove true, no value), as a "
              "new revision, their ids kept (wpt depot score): the depot records it as one decision under review on "
              "every text it changes, whose reason names the score set or removed, and the decisions before it keep "
              "counting, so the texts keep their status and libraries; a locked library keeps the revision it "
              "pinned; a text whose scores stay as they are is left out: a job whose answer is the decision's id "
              "(null when no text changes and nothing is written), how many texts changed, how many were left out "
              "unchanged, and how many were left out, by reason",
              body={"type": "object", "required": ["selection", "name", "declaration"],
                    "oneOf": [{"required": ["value"]}, {"required": ["remove"]}],
                    "properties": {"selection": selection, "name": TEXT, "value": {"type": "number"},
                                   "remove": {"const": True}, "declaration": DECLARATION}},
              answer=written({"type": "object", "required": ["decision", "changed", "left_out", "left"]})),
        Route("GET", "/api/scores", score_names, "every name of a score a text has now, with how many texts have "
              "it, by name (wpt depot score-names), read at once from the depot's counts", answer=ROWS),
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
        Route("POST", "/api/libraries/measure", measure, "what libraries hold now, each text at its current "
              "revision, read at once from the depot's counts of each library, whatever its size: texts, characters "
              "and UTF-8 bytes of them all, and of the plain texts (plain), the conversations and tasks (items, the "
              "items a corpus takes) and the items of other kinds (other); each library counts apart and the figures "
              "are their sum, so a text several of them hold counts once for each; with shares, the plain texts that "
              "have each share's tag (a text with several of the shares' tags counts once for each, where a dataset's "
              "share takes it only for the first: exact when each text has at most one of them), those left once the "
              "shares are taken away (unshared), and the mix of the corpus of every token: kept_bytes and "
              "kept_characters of each share (the share that holds the least for its proportion keeps all its texts, "
              "and limits the others, each its proportion of it, at most all it holds; a share with no text takes "
              "nothing), limits on the share that limits the others, and mix, the bytes and characters of plain text "
              "kept and the share of the bytes the conversations and tasks hold then (items_share), the share they "
              "take of any size; 404 for a library the depot does not hold",
              body={"type": "object", "required": ["libraries"],
                    "properties": {"libraries": {**NAMES, "minItems": 1},
                                   "shares": {"type": ["array", "null"], "items": share}}},
              answer={"type": "object", "required": ["texts", "characters", "bytes", "plain", "items", "other",
                                                     "shares", "unshared", "mix"]}),
        Route("POST", "/api/changes/times", change_times, "the UTC time each of the depot's changes named was made "
              "at, null for a change made before format 4, in the order named; a change the depot does not hold is "
              "left out",
              body={"type": "object", "required": ["changes"],
                    "properties": {"changes": {"type": "array", "items": {"type": "integer", "minimum": 0},
                                               "maxItems": 1000}}},
              answer={"type": "array", "items": {"type": "object", "required": ["change", "made_at"]}}),
        Route("POST", "/api/libraries/{id}/pins", pinned, "the revisions a locked library pinned, a page at a time "
              "(after a row's place, or offset rows, and limit, default 0 and 50): how many texts it pinned and how "
              "many of them have a newer current revision, both from the depot's counts, and the page of rows, each "
              "the text's id, the pinned revision's title, the revision pinned, the text's current revision, its "
              "status when the library was locked, in the numeric order of ids, read from an index of "
              "the pins; next, the place of the page's last row when more follow; an open library pins none",
              body={"type": "object", "properties": {"offset": {"type": "integer", "minimum": 0},
                                                     "after": TEXT,
                                                     "limit": {"type": "integer", "minimum": 1, "maximum": 500}}},
              answer={"type": "object", "required": ["total", "newer", "rows", "next"]}),
        Route("POST", "/api/libraries", new_library, "a new open library (wpt depot library new)",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT}}, answer=ID),
        Route("POST", "/api/libraries/{id}/name", rename_library, "a new name for the library, open or locked (wpt "
              "depot library rename)",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT}}, answer=OBJECT),
        Route("POST", "/api/libraries/{id}/texts", put, "every admitted text a selection takes joins the open "
              "library (wpt depot library put): a job whose answer is the library's id, how many joined it, how many "
              "were left out because they are not admitted, and how many were left out, by reason",
              body={"type": "object", "required": ["selection"], "properties": {"selection": selection}},
              answer=written({"type": "object", "required": ["library", "joined", "left_out", "left"]})),
        Route("POST", "/api/libraries/{id}/taken-out", take_out, "every text a selection takes that the open "
              "library holds leaves it and keeps its status, admitted included (wpt depot library take-out): a job "
              "whose answer is the library's id, how many were taken out, how many it did not hold, and how many were left out, by "
              "reason",
              body={"type": "object", "required": ["selection"], "properties": {"selection": selection}},
              answer=written({"type": "object", "required": ["library", "taken_out", "not_held", "left"]})),
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
              "with optional ordered shares and an optional size the build form starts at, whose vocabulary is "
              "trained by the depot's tokenizer recipe it names; it publishes no vocabulary until an edit sets one "
              "(wpt depot dataset new)",
              body={"type": "object", "required": ["name", "tokenizer", "libraries"],
                    "properties": {"name": TEXT, "tokenizer": TEXT,
                                   "libraries": {"type": "array", "minItems": 1, "items": library_ref},
                                   "shares": {"type": ["array", "null"], "items": share},
                                   "max_tokens": max_tokens}}, answer=ID),
        Route("GET", "/api/datasets/{id}", dataset, "one dataset by its id: its current row as GET /api/datasets gives it, "
              "whether it belongs to the upstream this depot follows (which alone changes it), each library it "
              "reads with its change number and local, whether it is read here as GET /api/datasets says (and "
              "for such a library its name and whether it is locked), its remote, and the corpora the store holds "
              "built from it, through its renames, each with its counts (texts, conversations, total_tokens of "
              "plain text, items_tokens of conversations and tasks), the revision of the dataset it was built from, its "
              "vocabulary's sha256 and its size (max_tokens, null for none), the last built first (null, with why in store, when the store cannot be read); "
              "and figures: what its current revision reads, each library at the change it reads it at, as the "
              "revision stores it (measure, as POST /api/libraries/measure gives it for its libraries and shares; "
              "null when the figures of one of its libraries are not known: one of another depot, or one the "
              "revision's write or an erasure since left unproved) and since, how "
              "many texts its libraries hold now minus that (null when not known); and read_again, whether "
              "POST /api/datasets/{id}/read-again changes what the page shows of it: a library of this depot whose "
              "figures are unknown while a read now knows them, or that holds another number of texts now, or, in "
              "a clone, an upstream's library a pull moved",
              answer={"type": "object", "required": ["dataset", "upstream", "libraries", "remote", "corpora",
                                                     "store", "figures", "read_again"]}),
        Route("POST", "/api/datasets/{id}", edit_dataset, "change the dataset of that id as a new revision, its id "
              "and tokenizer kept, with the checks of a new one (wpt depot dataset edit): a library it read already "
              "keeps its change number (one of another depot included, named as the dataset stores it), a new one "
              "is read at its depot's change number now; a corpus built before keeps its own copy; with revision, "
              "the revision the change was made from, refused (422) when the dataset is at another one",
              body={"type": "object", "required": ["name", "libraries"],
                    "properties": {"name": TEXT, "libraries": {"type": "array", "minItems": 1, "items": library_ref},
                                   "shares": {"type": ["array", "null"], "items": share},
                                   "max_tokens": max_tokens,
                                   "vocabulary": {"type": ["string", "null"], "description": "the sha256 of the "
                                                  "vocabulary the dataset publishes, one this depot holds (422 "
                                                  "otherwise), whose corpus training takes; absent or null: none"},
                                   "revision": {"type": "integer", "minimum": 1}}},
              answer=ID),
        Route("POST", "/api/datasets/{id}/read-again", read_again, "a new revision of the dataset of that id, the "
              "same except that every library of this depot it reads is read at the change number now (a locked one "
              "as its lock leaves it), with the figures of what it reads; a library of another depot keeps its change "
              "number; what the next build takes changes (wpt depot dataset read-again): libraries, how many "
              "libraries' change numbers changed, 0 with nothing written when none did or when the depot's latest "
              "change is this dataset's own read again; with revision, the revision the change was made from, refused (422) when the dataset is "
              "at another one",
              body={"type": "object", "properties": {"revision": {"type": "integer", "minimum": 1}}},
              answer={"type": "object", "required": ["id", "libraries"],
                      "properties": {"id": TEXT, "libraries": {"type": "integer", "minimum": 0}}}),
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
