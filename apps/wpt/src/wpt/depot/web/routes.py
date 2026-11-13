"""The routes of `wpt depot web`: the operations of the `wpt depot`
subcommands, each a controller over wpt_depot that reads its request, calls
the depot and turns its result, or its failure, into an answer. The depot
applies every rule, so the pages and the subcommands cannot disagree. A
pull, a corpus build, a remote's import and a change of the depot's
maintainer or takedown contact have no route: they are commands only
(docs/architecture.md, "Two interfaces")."""

import json
from functools import wraps

from wpt_depot import Declaration, Depot, NewText, errors
from wpt_depot.filters import STATUSES as TEXT_STATUSES
from wpt_depot.depot import BELONGS_TO_UPSTREAM, CREDITS, NO_ALLOWED_LICENSE, ORDERS
from wpt_depot.erasure import CONTEXT
from wpt_depot.filters import Filter, condition
from wpt_depot.scoring import lines as score_lines
from wpt_depot.schema import ADMITTED, REFUSED
from wpt_recipes import InvalidRecipe, Recipes
from wpt_recipes import NotFound as RecipeNotFound
from wpt_store import StoreError
from wpt_utils.lines import say

from .. import shared
from ..shared import CONFIRMATION, REQUIREMENTS, jsonable
from . import readers as limits
from .readers import Readers
from .server import Route

# the HTTP status of each failure; a failure's status is that of the first of its classes listed here
STATUSES = {errors.NotFound: 404, RecipeNotFound: 404, errors.BadQuery: 400, errors.JournalUnfinished: 500,
            StoreError: 503, errors.DepotError: 422, InvalidRecipe: 422}

TEXT = {"type": "string", "minLength": 1}
NAMES = {"type": "array", "items": TEXT}
OBJECT, ROWS = {"type": "object"}, {"type": "array", "items": {"type": "object"}}
DECLARATION = {"type": "object", "required": ["name", "email", "accepted"],
               "description": f"who decides; accepted is the confirmation: {CONFIRMATION}",
               "properties": {"name": TEXT, "email": TEXT, "accepted": {"type": "boolean"}}}
ID = {"type": "object", "required": ["id"], "properties": {"id": TEXT}}
# the exact texts an operation works on, by id: those the page reviewed, never a condition matched again later
IDS = {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True}
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


def routes(depot: Depot) -> list[Route]:
    """Every route over depot; a person's SQL runs in reader processes
    (wpt.depot.web.readers)."""
    readers = Readers(depot.address, depot.settings)

    def declared(body) -> Declaration:
        found = body["declaration"]
        return Declaration(found["name"], found["email"], found["accepted"])

    @refusals
    def about(values, body):
        return 200, {"address": depot.address, "change": depot.change(), "about": depot.about(),
                     "confirmation": CONFIRMATION, "requirements": REQUIREMENTS}

    @refusals
    def sql(values, body):
        return 200, jsonable(readers.run("sql", body["query"]))

    @refusals
    def kinds(values, body):
        return 200, [{**one, "schema": json.loads(one["schema"])} for one in
                     depot.sql("SELECT * FROM kinds ORDER BY id").to_pylist()]

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
        decision, reason, changed, left_out = depot.edit_scores(score_lines(body["texts"]), declared(body))
        return 200, {"decision": decision, "reason": reason, "changed": changed, "left_out": left_out}

    @refusals
    def content(values, body):
        changed = depot.change_content(values["id"], body.get("text"),
                                       json.dumps(body["content"], ensure_ascii=False) if "content" in body else None,
                                       declared(body))
        decision, version = changed if changed is not None else (None, None)
        return 200, {"decision": decision, "version": version}

    def filtered(found) -> str:
        """The SQL condition over the texts table a filter of the body stands for."""
        ids = found.get("ids")
        return condition(Filter(ids=None if ids is None else tuple(ids), tags=tuple(found.get("tags", [])),
                                licenses=tuple(found.get("licenses", [])), statuses=tuple(found.get("statuses", [])),
                                libraries=tuple(found.get("libraries", [])), locked=found.get("locked"),
                                where=found.get("where")))

    @refusals
    def search(values, body):
        where = filtered(body["filter"])
        return 200, {"where": where, **jsonable(readers.run(
            "search", where, body.get("offset", 0), body.get("limit", 50), body.get("order", "id"),
            body.get("descending", False), body.get("every_id", False)))}

    @refusals
    def passages(values, body):
        return 200, jsonable(readers.run(
            "passages", filtered(body["filter"]), body.get("offset", 0), body.get("limit", 50),
            body.get("characters", PASSAGE_CHARACTERS), body.get("order", "id"), body.get("descending", False)))

    @refusals
    def blocked(values, body):
        return 200, {"texts": depot.blocked(body["texts"], body.get("status", ADMITTED))}

    @refusals
    def tags(values, body):
        return 200, depot.tags()

    @refusals
    def decide(values, body):
        decision, texts = depot.decide(body["status"], body["texts"], declared(body), body.get("reason"))
        return 200, {"decision": decision, "texts": texts}

    @refusals
    def relicense(values, body):
        found, skipped = depot.relicense(body["texts"], body["licenses"], declared(body))
        return 200, {"decisions": [{"decision": decision, "texts": texts} for decision, texts in found],
                     "skipped": skipped}

    @refusals
    def latest(values, body):
        return 200, jsonable(depot.latest_decisions())

    @refusals
    def decision(values, body):
        return 200, jsonable(depot.decision(values["id"]))

    @refusals
    def libraries(values, body):
        return 200, jsonable(depot.sql(
            "SELECT l.*, (SELECT count(*) FROM texts t WHERE list_contains(t.libraries, l.id)) AS texts, "
            f"{depot.upstream_sql('l.id')} AS upstream FROM libraries l ORDER BY l.id").to_pylist())

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
        joined, left_out = depot.put(values["id"], body["texts"])
        return 200, {"joined": joined, "left_out": left_out}

    @refusals
    def take_out(values, body):
        taken_out, not_held = depot.take_out(values["id"], body["texts"])
        return 200, {"taken_out": taken_out, "not_held": not_held}

    @refusals
    def lock(values, body):
        depot.lock(values["id"])
        return 200, {}

    @refusals
    def fork(values, body):
        return 200, {"id": depot.fork(values["id"], body["name"], body.get("without"))}

    @refusals
    def report(values, body):
        decision, texts = depot.report(values["id"], body["problem"], body.get("texts"), declared(body))
        return 200, {"decision": decision, "texts": texts}

    @refusals
    def tokenizers(values, body):
        return 200, Recipes.find().names("tokenizer")

    def held_libraries():
        """{id: {id, name, locked_at}} of every library this depot holds."""
        return {one["id"]: one for one in depot.sql("SELECT id, name, locked_at FROM libraries").to_pylist()}

    def with_here(row, held):
        """A dataset's row with here: the ids of the libraries it reads from
        this depot, those it names so (wpt_depot.Depot.reads_here) that this
        depot holds (held: the ids it holds)."""
        return {**row, "here": list(dict.fromkeys(
            ref["library"] for ref in row["libraries"]
            if ref["library"] in held and depot.reads_here(ref["depot"], ref["library"])))}

    @refusals
    def datasets(values, body):
        found = depot.sql(f"SELECT *, {depot.upstream_sql()} AS upstream FROM datasets ORDER BY id").to_pylist()
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
            None if shares is None else [(one["tag"], one["share"]) for one in shares])}

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
                                                                    "total_tokens", "stale")},
                                   "revision": one["dataset"]["revision"]} for one in corpora],
                              "store": unreachable})

    @refusals
    def edit_dataset(values, body):
        shares = body.get("shares")
        depot.edit_dataset(values["id"], body["name"], own(body["libraries"]),
                           None if shares is None else [(one["tag"], one["share"]) for one in shares])
        return 200, {"id": values["id"]}

    @refusals
    def export(values, body):
        return 200, depot.export_dataset(values["name"])

    @refusals
    def erasures(values, body):
        found = depot.erasures().to_pylist()
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
                                                           body.get("reason"), say)}

    status = {"enum": [ADMITTED, REFUSED]}
    library_ref = {"type": "object", "required": ["library"],
                   "properties": {"depot": {"type": ["string", "null"], "description": "this depot's address, in "
                                            "any form; absent or null: this depot; another depot's address is "
                                            "refused (422), only the commands name one"}, "library": TEXT}}
    share = {"type": "object", "required": ["tag", "share"], "properties": {"tag": TEXT, "share": {"type": "number"}}}
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
                                  "where": {**nullable, "description": "an SQL condition over the texts table, which "
                                            "reaches nothing outside the depot's folder (400) and, with the "
                                            f"rest of the request's reads, stops after {limits.SECONDS} seconds or "
                                            f"past {limits.MEMORY} bytes of memory (422)"}}}
    return [
        Route("GET", "/api/about", about, "the depot's address, its change number, its current about row, and the "
              "words of the confirmation every decision needs",
              answer={"type": "object", "required": ["address", "change", "about", "confirmation", "requirements"]}),
        Route("POST", "/api/sql", sql, "the rows of one read query over the depot's tables (wpt depot sql): texts, "
              "libraries, datasets, about and erasures are the current state (texts with their text, kind and "
              "content, and libraries, the libraries holding them now), <name>_revisions every revision row of "
              "each (its id, revision and the change number it was made at; a text's revision names the content version "
              "it reads), contents every content version of each text with the change number it was made at, "
              "content_lists each content version as the list of its parts, parts each part of each text, "
              "memberships each library's texts with the change numbers they joined and left at, pins the "
              "revision each locked library holds of each of its texts, and pulls, for a depot that follows "
              "another, each pull's change number with the upstream's change number it brought; licenses, notices, kinds, decisions and "
              "journal as they are; a time is ISO 8601 text and a byte string hexadecimal text; the query reaches "
              "nothing outside the depot's folder (a read of another file or address is refused with 400), and "
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
        Route("POST", "/api/texts/search", search, "the texts a filter matches, a page at a time: the SQL "
              "condition over the texts table the filter stands for, how many texts it matches, how many of them "
              "are under review, admitted and refused, how many a locked library holds, how many have each tag and "
              "name each license, the page of rows from offset in the order asked (then by id, in the numeric order "
              "of ids), each a text's credits, status and length in characters, without its text, and with "
              "every_id the id of every text it matches, which a decision, a put, a fork or a report then names",
              body={"type": "object", "required": ["filter"],
                    "properties": {"filter": text_filter, "offset": {"type": "integer", "minimum": 0},
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                                   "order": {"enum": list(ORDERS)}, "descending": {"type": "boolean"},
                                   "every_id": {"type": "boolean"}}},
              answer={"type": "object", "required": ["where", "total", "statuses", "locked", "tags", "licenses",
                                                     "rows"]}),
        Route("POST", "/api/texts/passages", passages, "the texts a filter matches with their text, a page at a "
              "time in the order of POST /api/texts/search, the page bounded both by a count of texts (limit) and "
              "by characters: from offset, at most limit texts holding at most characters characters together, each "
              "its id, kind, title, length in characters and passage, its text or an item's content from its start, "
              "whole, or cut at characters for the page's first text when it is longer; a later text that does not "
              "fit whole starts the next page; next is the offset of the text after the page, null when the page "
              "ends the texts matched",
              body={"type": "object", "required": ["filter"],
                    "properties": {"filter": text_filter, "offset": {"type": "integer", "minimum": 0},
                                   "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                                   "characters": {"type": "integer", "minimum": 1, "maximum": MAX_PASSAGE_CHARACTERS},
                                   "order": {"enum": list(ORDERS)}, "descending": {"type": "boolean"}}},
              answer={"type": "object", "required": ["rows", "next"], "properties": {
                  "rows": {"type": "array", "items": {"type": "object", "required": [
                      "id", "kind", "title", "length", "passage"]}},
                  "next": {"type": ["integer", "null"]}}}),
        Route("POST", "/api/texts/blocked", blocked, "of exactly the texts named, each one a decision of that status "
              "(default admitted) cannot cover, the check a decision applies: its id and its reasons (it belongs to "
              "the upstream this depot follows, for any decision; none of its licenses is allowed, for an "
              "admission), in the numeric order of ids",
              body={"type": "object", "required": ["texts"], "properties": {"texts": IDS, "status": status}},
              answer={"type": "object", "required": ["texts"], "properties": {"texts": {"type": "array", "items": {
                  "type": "object", "required": ["id", "reasons"],
                  "properties": {"id": TEXT, "reasons": {"type": "array", "items": {
                      "enum": [BELONGS_TO_UPSTREAM, NO_ALLOWED_LICENSE]}}}}}}}),
        Route("GET", "/api/tags", tags, "every tag a text has, with how many texts have it", answer=ROWS),
        Route("POST", "/api/decisions", decide, "one decision on exactly the texts named, with an optional public "
              "reason, which never starts with 'licenses changed from ' (the depot's own record of a license "
              "change); admitting is refused when a text has no allowed license (wpt depot decide): its id and the "
              "texts' ids",
              body={"type": "object", "required": ["status", "texts", "declaration"],
                    "properties": {"status": status, "texts": IDS, "reason": TEXT, "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decision", "texts"]}),
        Route("POST", "/api/texts/licenses", relicense, "change the licenses of exactly the texts named, as a new "
              "revision, their ids kept (wpt depot relicense): for each distinct old list, a decision under review "
              "whose reason the depot writes, 'licenses changed from OLD to NEW', after which the decisions before it "
              "no longer count, so each text is under review again and leaves every open library (a locked library "
              "keeps the revision it pinned); a text that names these licenses already is left out: each "
              "decision's id and its texts' ids, and the ids left out",
              body={"type": "object", "required": ["texts", "licenses", "declaration"],
                    "properties": {"texts": IDS, "licenses": {**NAMES, "minItems": 1}, "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decisions", "skipped"], "properties": {"decisions": {
                  "type": "array", "items": {"type": "object", "required": ["decision", "texts"],
                                             "properties": {"decision": TEXT, "texts": NAMES}}},
                  "skipped": NAMES}}),
        Route("GET", "/api/decisions/latest", latest, "the ten latest decisions, the newest first, each with how "
              "many texts and erasure requests it decides and the library whose reported problem it records",
              answer=ROWS),
        Route("GET", "/api/decisions/{id}", decision, "one decision: its row (status, declared name and email, "
              "time, reason), how many texts it is on, the erasure requests it decides, and the reported problem it "
              "records with its library",
              answer={"type": "object", "required": ["decision", "texts", "erasures", "problem"]}),
        Route("GET", "/api/libraries", libraries, "every library as it is now, with how many texts it holds and whether it "
              "belongs to the upstream this depot follows, which alone changes it",
              answer=ROWS),
        Route("POST", "/api/libraries/{id}/pins", pinned, "the revisions a locked library pinned, a page at a time "
              "(offset and limit, default 0 and 50): how many texts it pinned, how many of them have a newer "
              "current revision, and the page of rows, each the text's id, the pinned revision's title, the "
              "revision pinned and the text's current revision, in the numeric order of ids; an open library pins "
              "none",
              body={"type": "object", "properties": {"offset": {"type": "integer", "minimum": 0},
                                                     "limit": {"type": "integer", "minimum": 1, "maximum": 500}}},
              answer={"type": "object", "required": ["total", "newer", "rows"]}),
        Route("POST", "/api/libraries", new_library, "a new open library (wpt depot library new)",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT}}, answer=ID),
        Route("POST", "/api/libraries/{id}/name", rename_library, "a new name for the library, open or locked (wpt "
              "depot library rename)",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT}}, answer=OBJECT),
        Route("POST", "/api/libraries/{id}/texts", put, "every admitted text of those named joins the open library "
              "(wpt depot library put): the ids that joined it, and those left out, not admitted",
              body={"type": "object", "required": ["texts"], "properties": {"texts": IDS}},
              answer={"type": "object", "required": ["joined", "left_out"]}),
        Route("POST", "/api/libraries/{id}/taken-out", take_out, "every text of those named that the open library "
              "holds leaves it and keeps its status, admitted included (wpt depot library take-out): the ids taken "
              "out, and those named that it did not hold",
              body={"type": "object", "required": ["texts"], "properties": {"texts": IDS}},
              answer={"type": "object", "required": ["taken_out", "not_held"]}),
        Route("POST", "/api/libraries/{id}/lock", lock, "lock the library for good: it pins the current revision of "
              "each text it holds (wpt depot library lock)",
              body=OBJECT, answer=OBJECT),
        Route("POST", "/api/libraries/{id}/forks", fork, "a new open library holding the same texts that are "
              "admitted now, less the texts named in without (wpt depot library fork)",
              body={"type": "object", "required": ["name"], "properties": {"name": TEXT, "without": IDS}},
              answer=ID),
        Route("POST", "/api/libraries/{id}/problems", report, "a problem with the library, locked or not: the "
              "texts of the library named (none: the library as a whole) and the problem in words (wpt depot "
              "library report): the decision that records the reporter, and the texts' ids",
              body={"type": "object", "required": ["problem", "declaration"],
                    "properties": {"problem": TEXT, "texts": IDS, "declaration": DECLARATION}},
              answer={"type": "object", "required": ["decision", "texts"]}),
        Route("GET", "/api/tokenizers", tokenizers, "the names of the repository's tokenizer recipes, which a "
              "dataset names", answer={"type": "array", "items": {"type": "string"}}),
        Route("GET", "/api/datasets", datasets, "every dataset as it is now, with whether it belongs to the "
              "upstream this depot follows, which alone changes it, and here, the ids of the libraries it reads "
              "from this depot: those this depot holds, named by its address in any form or, for a library of "
              "that upstream, by the upstream's", answer=ROWS),
        Route("POST", "/api/datasets", new_dataset, "a dataset reading libraries, each at its depot's change number now, "
              "with optional ordered shares (wpt depot dataset new)",
              body={"type": "object", "required": ["name", "tokenizer", "libraries"],
                    "properties": {"name": TEXT, "tokenizer": TEXT,
                                   "libraries": {"type": "array", "minItems": 1, "items": library_ref},
                                   "shares": {"type": ["array", "null"], "items": share}}}, answer=ID),
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
                                   "shares": {"type": ["array", "null"], "items": share}}}, answer=ID),
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
    ]
