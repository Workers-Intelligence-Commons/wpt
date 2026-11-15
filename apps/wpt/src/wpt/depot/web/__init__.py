"""wpt depot web [--port N]: build the interface if needed, then serve the
depot WPT_DEPOT names on http://127.0.0.1:N/ until Ctrl+C.

The web server is the second controller of `wpt depot`, beside its
subcommands:

    server.py   serves the built interface, the routes and their OpenAPI
                description (/api/openapi.json), on this machine only
    ui.py       builds the interface (apps/wpt/ui) with Bun when its
                sources are newer than the last build
    routes.py   the routes, one per operation of the subcommands it serves
    readers.py  the processes that run a person's SQL, under a time limit
                and a memory cap
    builds.py   the routes over tokenizer recipes, jobs, corpus builds and
                downloads, and the job operations they queue

A page is one TypeScript file in ui/src/pages/, over the routes.
"""

from wpt_depot.errors import DepotError
from wpt_utils.lines import ERROR, say
from wpt_utils.programs import MissingTool

from ..jobs import Jobs
from ..shared import opened, public_url
from . import ui
from .routes import STATUSES, routes
from .server import make_server


def serve(port: int) -> int:
    """Serve, and run the queued jobs (wpt.depot.jobs), until Ctrl+C: exit
    code 0, or 1 when the server cannot start (no depot, no Bun, a failed
    build of the interface, the port taken)."""
    try:
        depot = opened()
        public_url()  # BadDepotSetting before the server starts
        static = ui.built()
        jobs = Jobs(depot.address, STATUSES)
        server = make_server(routes(depot, jobs), static, port)
    except (DepotError, MissingTool, ui.UiBuildFailed, OSError) as error:
        say(f"wpt depot web: {type(error).__name__}: {error}", ERROR)
        return 1
    say(f"wpt depot web: {depot.address} on http://127.0.0.1:{server.server_address[1]}/ (Ctrl+C stops it)")
    jobs.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        jobs.stop()
        server.server_close()
    return 0
