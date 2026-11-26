"""The routes of `depot web` over the jobs' queue (depot_app.jobs):
the list of the jobs and the state of one. Only the command line cancels a
job. The routes over snapshots, whose jobs the queue runs, are
depot_app.web.snapshots'.

    queue_routes(depot, jobs, refusals)   the routes
"""

from depot import Depot
from wic_utils.lines import say

from .. import shared
from ..downloads import Downloads
from ..jobs import Jobs
from .server import Route
from .snapshots import snapshot_routes


def queue_routes(depot: Depot, jobs: Jobs, refusals) -> list[Route]:
    """The routes over the jobs' queue, then those of depot's snapshots;
    refusals wraps a handler so that it answers each failure with its
    status (depot_app.web.routes.refusals). The marking of the corpora an
    erasure reached that a stop of the web app cut short goes on first
    (depot_app.shared.settle_marks)."""
    shared.settle_marks(depot, Downloads(jobs.path), say)

    @refusals
    def jobs_listed(values, body):
        return 200, jobs.listed()

    @refusals
    def job_state(values, body):
        return 200, shared.jsonable(jobs.state(values["id"]))

    job_answer = {"type": "object", "required": ["id", "kind", "title", "state"]}
    return [
        Route("GET", "/api/jobs", jobs_listed, "the jobs' queue: every job queued or running in the order they run, "
              "then the latest that ended; each with its kind, its lane (write: the writes over many texts; heavy: "
              "the snapshots; read: a snapshot's order by length; each lane runs one job at a time), what it does, "
              "its key (a sizes job's snapshot id), its state, its progress and its answer or problem; only `wpt "
              "depot jobs cancel` cancels one", answer={"type": "array"}),
        Route("GET", "/api/jobs/{id}", job_state, "one job's state", answer=job_answer),
        *snapshot_routes(depot, jobs, refusals),
    ]
