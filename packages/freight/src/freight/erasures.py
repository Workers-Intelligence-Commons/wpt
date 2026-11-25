"""What the erasures a depot carried out did to the deliveries on a dock
(docs/api/freight.md, "Erasures"): the freight asks the depot, through its
api module, which erasures it carried out since the mark it saw last there,
kept in the queue's file, and writes what each did into the note of every
delivery whose items hold a text it reached (freight.deliveries.reached):
stale, or the erasure noted, as the delivery's on_erasure says. It asks
before every job and whenever `freight check` runs.

    check(queue, depot, dock)   {mark, erasures, stale, noted}
"""

from wpt_depot import api as depot_api

from . import deliveries


def check(queue, depot: str, dock) -> dict:
    """Ask the depot at the address depot which erasures it carried out
    since the queue's mark for it, write what each did into the deliveries
    of dock, and keep the depot's mark now: {mark, erasures (their ids),
    stale (the keys of the deliveries marked stale), noted (those whose note
    took an erasure in)}."""
    answer = depot_api.erasures(depot, queue.mark(depot))
    stale, noted = set(), set()
    for one in answer["erasures"]:
        marked = deliveries.mark_reached(dock, one["texts"], [one["id"]])
        stale.update(marked["stale"])
        noted.update(marked["noted"])
    queue.keep_mark(depot, answer["mark"])
    return {"mark": answer["mark"], "erasures": [one["id"] for one in answer["erasures"]],
            "stale": sorted(stale), "noted": sorted(noted)}
