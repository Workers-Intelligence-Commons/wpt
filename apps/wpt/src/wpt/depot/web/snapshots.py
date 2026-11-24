"""The routes of `wpt depot web` over the depot's snapshots
(wpt_depot.snapshots): the list, the name and cost of the next one, making
one as a job, one snapshot with what an erasure did to it since, its sets for anyone to check, and its texts a page at
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

A snapshot read whole as one strip, its longest texts first, reads its
order by length (wpt_depot.sizes), made once by a job of the read lane
(sizes_job) and kept in the jobs' folder, <depot file>.work/sizes/, one
file a snapshot and digest; the first read of the strip queues that job and
answers its state, which the page shows until the order is there.

The sets travel as portable roaring bitmaps in base64 (the format CRoaring
and every roaring library read), with the digest, so anyone can check them
against the snapshot's digest without a copy of the depot
(wpt_depot.snapshots says how the digest is made).

    snapshot_job(depot, name, note, progress=..., job=...)   the job's operation
    sizes_job(depot, snapshot_id, progress=..., job=...)     the order by length's
    snapshot_routes(depot, jobs, refusals)                   the routes
"""

import base64
from datetime import UTC, datetime
from pathlib import Path

from wpt_depot import Depot, errors, sizes, snapshots

from .. import shared
from ..jobs import Busy, Job, Jobs, work_folder
from .server import Route

LIMIT = 1000  # the texts a page of a snapshot's texts holds at most
STRIP_TEXTS, STRIP_CHARACTERS = 200, 1_000_000  # the texts and characters a page of the strip holds at most
COSTS = {"seconds": 5, "bytes": 1 << 20}  # the costs of making a snapshot a confirmation covers, with their slack


def snapshot_job(depot: Depot, name: str, note: str | None, *, progress, job: Job) -> dict:
    """Make the snapshot called name with the note, its stages and the texts
    read so far given to progress: {id, name, at_change, texts, digest}."""
    total = snapshots.estimate(depot)["texts"]
    row = depot.create_snapshot(name, note, progress=progress,
                                read=lambda read: progress({"stage": "reading texts", "texts": read, "of": total}))
    return {"id": row["id"], "name": row["name"], "at_change": row["at_change"], "texts": row["texts"],
            "digest": row["digest"]}


def sizes_file(work: Path, snapshot_id: str, digest: str) -> Path:
    """Where the order by length of the snapshot of that id and digest is kept in the jobs' folder work."""
    return work / "sizes" / f"{snapshot_id}-{digest[:16]}.npy"


def sizes_job(depot: Depot, snapshot_id: str, *, progress, job: Job) -> dict:
    """Make and keep the snapshot's order by length, the texts read so far given to progress: {texts}."""
    row = snapshots.snapshot_by_id(depot, snapshot_id)
    total = row["texts"]
    progress("reading lengths")
    made = sizes.build(depot, snapshot_id, lambda read: progress({"stage": "reading lengths", "texts": read,
                                                                 "of": total}))
    target = sizes_file(job.work, snapshot_id, row["digest"])
    target.parent.mkdir(parents=True, exist_ok=True)
    sizes.save(made, target)
    return {"texts": len(made)}


def snapshot_routes(depot: Depot, jobs: Jobs, refusals) -> list[Route]:
    """The routes over depot's snapshots; refusals wraps a handler so that it
    answers each failure with its status."""

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
        return 200, {**shown(row), "summary": row["summary"], "erased_since": erased(row)}

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

    kept: dict[str, sizes.Sizes] = {}  # the orders by length read, by file

    @refusals
    def strip(values, body):
        row = row_of(values)
        path = sizes_file(work_folder(depot.address), row["id"], row["digest"])
        order = kept.get(str(path)) or sizes.load(path)
        if order is None:
            job = jobs.open_job("sizes", key=row["id"])
            if job is None and not body.get("again"):
                # a job that ended without the order (refused, failed or cancelled) is answered as it is, so the page
                # says why and offers to try again, rather than a new job queued on every reading of its state
                last = jobs.last_ended("sizes", row["id"])
                if last is not None and last["state"] != "done":
                    return 200, {"ready": False, "job": last, "texts": row["texts"]}
            if job is None:
                try:
                    job = jobs.add("sizes", f"order the snapshot {row['name']} by length", sizes_job, row["id"],
                                   key=row["id"], target={"snapshot": row["id"]})
                except Busy:  # queued meanwhile by another request
                    job = jobs.open_job("sizes", key=row["id"])
            return 200, {"ready": False, "job": job, "texts": row["texts"]}
        kept[str(path)] = order
        found = sizes.page(depot, order, body.get("after"), body.get("limit", 50),
                           body.get("characters", 200_000) if body.get("passages", True) else None,
                           body.get("descending", True), body.get("each"))
        return 200, {"ready": True, **found}

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
              "tag set, license and owner: texts, characters, bytes) and erased_since (how many of its texts an "
              "erasure carried out after it cut or removed, and those erasures' ids)",
              answer={**row, "required": [*row["required"], "summary", "erased_since"]}),
        Route("GET", "/api/snapshots/{id}/sets", sets, "the snapshot's sets, to check it without a copy of the "
              "depot: each set's part, owner, bound (the change of the owner's numbering its texts are read at), "
              "how many texts and its numbers as a portable roaring bitmap in base64; with the snapshot's "
              "at_change and digest, which a digest worked out from the sets must give",
              answer={"type": "object", "required": ["at_change", "digest", "sets"]}),
        Route("POST", "/api/snapshots/{id}/strip", strip, "the snapshot read whole as one strip, its texts from "
              "the longest to the shortest (descending, the default) or the other way, a page at a time, each at its "
              "revision as of its bound: ready, whether its order by length is made; without it, the job of the "
              "read lane that makes it, queued by this request when none was (job: its state with its progress), "
              "or the last one when it ended without the order (refused, failed or cancelled: its state and problem), "
              "a new one queued only with again: true, "
              "and texts, the snapshot's texts; with it, rows after the position after (from the first without "
              "it), at most limit texts holding at most characters characters together, each its id, kind, title, "
              "revision, length in characters, bytes, status now and passage (unless passages is false), its text or an item's content from "
              "its start, whole, or cut at characters for the page's first text when it is longer; next, the "
              "position the next page starts after, null after the last; total, the texts in the order (those an "
              "erasure removed whole since are passed by)",
              body={"type": "object", "properties": {
                  "after": {"type": ["integer", "null"], "minimum": -1},
                  "limit": {"type": "integer", "minimum": 1, "maximum": STRIP_TEXTS},
                  "characters": {"type": "integer", "minimum": 1, "maximum": STRIP_CHARACTERS},
                  "descending": {"type": "boolean"},
                  "again": {"type": "boolean", "description": "true: a new job makes the order when the last one "
                            "ended without it"},
                  "each": {"type": "integer", "minimum": 1, "description": "the most characters each text shows, "
                           "its start"},
                  "passages": {"type": "boolean", "description": "false: the rows without their passage, and "
                               "no bound but limit (the biggest texts of a size map)"}}},
              answer={"type": "object", "required": ["ready"]}),
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
