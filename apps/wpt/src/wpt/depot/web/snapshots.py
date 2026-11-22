"""The routes of `wpt depot web` over the depot's snapshots
(wpt_depot.snapshots): the list, the name and cost of the next one, making
one as a job, one snapshot with what an erasure did to it since and the
recipes that name it, its sets for anyone to check, and its texts a page at
a time.

Making a snapshot starts only once confirmed (confirmed: true) with the
estimate GET /api/snapshots/new showed (estimate), and only while that
estimate still holds (wpt.depot.shared.unchanged: refused with 409 and the
new estimate otherwise). It runs as a job of the heavy lane
(wpt.depot.jobs), its stages "reading texts", "catching up" and "writing";
a restart of the web app starts it over, so the snapshot reads the change
current when it finishes, which its answer gives (at_change).

How many of a snapshot's texts an erasure cut or removed since is worked
out once for each state of the depot's erasures (snapshots.last_erasure,
which rises with every erasure revision, pulled ones too), and kept in the
queue's file (wpt.depot.jobs, erased_since).

The sets travel as portable roaring bitmaps in base64 (the format CRoaring
and every roaring library read), with the digest, so anyone can check them
against the snapshot's digest without a copy of the depot
(wpt_depot.snapshots says how the digest is made).

    snapshot_job(depot, name, note, progress=..., job=...)   the job's operation
    snapshot_routes(depot, jobs, refusals, recipes_naming)   the routes
"""

import base64
from datetime import UTC, datetime

from wpt_depot import Depot, errors, snapshots

from .. import shared
from ..jobs import Job, Jobs
from .server import Route

LIMIT = 1000  # the texts a page of a snapshot's texts holds at most
COSTS = {"seconds": 5, "bytes": 1 << 20}  # the costs of making a snapshot a confirmation covers, with their slack


def snapshot_job(depot: Depot, name: str, note: str | None, *, progress, job: Job) -> dict:
    """Make the snapshot called name with the note, its stages and the texts
    read so far given to progress: {id, name, at_change, texts, digest}."""
    total = snapshots.estimate(depot)["texts"]
    row = depot.create_snapshot(name, note, progress=progress,
                                read=lambda read: progress({"stage": "reading texts", "texts": read, "of": total}))
    return {"id": row["id"], "name": row["name"], "at_change": row["at_change"], "texts": row["texts"],
            "digest": row["digest"]}


def snapshot_routes(depot: Depot, jobs: Jobs, refusals, recipes_naming) -> list[Route]:
    """The routes over depot's snapshots; refusals wraps a handler so that it
    answers each failure with its status; recipes_naming(row) gives the
    repository's recipes that name the snapshot (its row)."""

    def being_made() -> list[str]:
        return [one["key"] for one in jobs.listed(0) if one["kind"] == "snapshot"]

    def shown(row: dict) -> dict:
        return shared.jsonable({key: value for key, value in row.items() if key != "summary"})

    def estimate_now() -> dict:
        return {**snapshots.estimate(depot), "change": depot.change()}

    @refusals
    def listed(values, body):
        return 200, [shown(row) for row in depot.snapshots()]

    @refusals
    def new(values, body):
        made = being_made()
        return 200, {"name": snapshots.next_name(depot, datetime.now(UTC).date(), made), "being_made": made,
                     "estimate": estimate_now()}

    @refusals
    def create(values, body):
        snapshots.check_name(body["name"])
        if body.get("confirmed") is not True:
            raise errors.Refused("a snapshot is made only once confirmed (confirmed: true), after its estimate: "
                                 "nothing was queued")
        now = estimate_now()
        changed = shared.unchanged(body.get("estimate"), now, COSTS)
        if changed:
            return 409, {"problems": [f"the estimate changed since it was shown ({'; '.join(changed)}): confirm "
                                      "the new one"], "estimate": now}
        if any(row["name"] == body["name"] for row in depot.snapshots() if row["owner"] == depot.name()):
            raise errors.Refused(f"the depot has a snapshot named {body['name']} already")
        return 200, jobs.add("snapshot", f"make the snapshot {body['name']}", snapshot_job, body["name"],
                             body.get("note"), key=body["name"],
                             target={"name": body["name"], "estimate": now})

    def row_of(values) -> dict:
        return snapshots.snapshot_by_id(depot, values["id"])

    def erased(row: dict) -> dict:
        last = snapshots.last_erasure(depot)
        found = jobs.erased_since(row["id"], last)
        if found is None:
            found = snapshots.erased_since(depot, row["id"])
            jobs.erased_kept(row["id"], last, found)
        return found

    @refusals
    def one(values, body):
        row = row_of(values)
        return 200, {**shown(row), "summary": row["summary"], "erased_since": erased(row),
                     "recipes": recipes_naming(row)}

    @refusals
    def sets(values, body):
        row = row_of(values)
        return 200, {"id": row["id"], "name": row["name"], "owner": row["owner"], "at_change": row["at_change"],
                     "digest": row["digest"], "format": "portable roaring bitmap, base64",
                     "sets": [{"part": part, "owner": owner, "bound": bound, "texts": len(bitmap),
                               "bitmap": base64.b64encode(bitmap.serialize()).decode("ascii")}
                              for part, owner, bound, bitmap in snapshots.sets(depot, row["id"])]}

    @refusals
    def texts(values, body):
        row = row_of(values)
        kinds = body.get("kinds") or None
        return 200, depot.snapshot_texts(row["id"], kinds, body.get("after"), body.get("limit", 100))

    estimate = {"type": "object", "required": ["texts", "seconds", "bytes"],
                "description": "what making a snapshot now takes: texts (admitted now), seconds, bytes (its sets "
                               "and summary), at (the change it reads: the one current when it finishes), basis"}
    row = {"type": "object", "required": ["id", "name", "owner", "at_change", "texts", "digest"]}
    return [
        Route("GET", "/api/snapshots", listed, "every snapshot of the depot, those of each owner (the depot whose "
              "snapshot it is) in the order they were made: id, name, owner, at_change (the change its texts are "
              "read at), made_at, origin, note, texts, characters, bytes, digest; never its summary",
              answer={"type": "array", "items": row}),
        Route("GET", "/api/snapshots/new", new, "the name offered for the next snapshot (the year and month, then "
              "with b, c, ... once taken, a snapshot being made counting as taken), the names being made, and the "
              "estimate of making one now (texts, seconds, bytes, change: the depot's change now); nothing is "
              "written",
              answer={"type": "object", "required": ["name", "estimate"], "properties": {"estimate": estimate}}),
        Route("POST", "/api/snapshots", create, "make a snapshot of the texts admitted now, called name, with an "
              "optional note, as a job of the heavy lane (the job's answer: id, name, at_change, texts, digest), "
              "only with confirmed: true and the estimate GET /api/snapshots/new gave (409 with the new estimate "
              f"when it now costs more than {shared.TOLERANCE:.0%} beyond it); refused when the name is taken",
              body={"type": "object", "required": ["name", "confirmed", "estimate"],
                    "properties": {"name": {"type": "string", "minLength": 1}, "note": {"type": ["string", "null"]},
                                   "confirmed": {"type": "boolean"}, "estimate": {"type": "object"}}},
              answer={"type": "object", "required": ["id", "kind", "state"]}),
        Route("GET", "/api/snapshots/{id}", one, "one snapshot by its id: its row, its summary (by kind, tag, plain "
              "tag set, license and owner: texts, characters, bytes), erased_since (how many of its texts an "
              "erasure carried out after it cut or removed, and those erasures' ids) and recipes (the "
              "repository's model recipes that name it, each with its name and whether its change and digest "
              "match)",
              answer={**row, "required": [*row["required"], "summary", "erased_since", "recipes"]}),
        Route("GET", "/api/snapshots/{id}/sets", sets, "the snapshot's sets, to check it without a copy of the "
              "depot: each set's part, owner, bound (the change of the owner's numbering its texts are read at), "
              "how many texts and its numbers as a portable roaring bitmap in base64; with the snapshot's "
              "at_change and digest, which a digest worked out from the sets must give",
              answer={"type": "object", "required": ["at_change", "digest", "sets"]}),
        Route("POST", "/api/snapshots/{id}/texts", texts, "a page of the snapshot's texts in the order of (owner, "
              "number), each at its revision as of its part's bound: id, part, revision, change, content_version, "
              "kind, length, bytes, title, tags, licenses; erased, how many numbers the page went past that an "
              "erasure removed whole; after, the id the next page starts after (null after the last)",
              body={"type": "object", "properties": {
                  "kinds": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1},
                            "description": "only texts of these kinds; absent, null or empty: every kind"},
                  "after": {"type": ["string", "null"]},
                  "limit": {"type": "integer", "minimum": 1, "maximum": LIMIT}}},
              answer={"type": "object", "required": ["texts", "erased", "after"]}),
    ]
