"""freight: the freight's command line, a controller over freight.api
(docs/api/freight.md). It has no web page.

    freight request FILE [--wait]   queue the request in the JSON file FILE and print its job; with --wait,
                                    run the queue here while no other process does, until the job ends, and
                                    print its delivery note
    freight work                    run the queue until stopped
    freight jobs                    every job queued or running, then the latest that ended
    freight job ID [--follow]       one job's state; --follow prints it again as it changes, until it ends
    freight cancel ID               cancel a queued or running job
    freight deliveries              every delivery note on the dock, the last made first
    freight delivery KEY            one delivery's note
    freight check                   mark what the erasures carried out since the last check did to the
                                    deliveries (a job checks the same before it runs)

Settings (environment or .env): FREIGHT_HOME, the freight's own folder;
DEPOT, the depot it reads; DOCK, the dock it delivers on;
FREIGHT_CAP, the bytes its deliveries may take on the dock (none by
default). Exit codes: 1 refused or failed, 2 a wrong command line.
"""

import argparse
import json
import sys
import time

from freight import api
from wic_utils.lines import ERROR, say

FOLLOW = 1.0  # seconds between two looks at a job followed


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="freight", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    asked = commands.add_parser("request", help="queue a request")
    asked.add_argument("file")
    asked.add_argument("--wait", action="store_true")
    commands.add_parser("work", help="run the queue until stopped")
    commands.add_parser("jobs", help="every job")
    one = commands.add_parser("job", help="one job's state")
    one.add_argument("id")
    one.add_argument("--follow", action="store_true")
    cancelled = commands.add_parser("cancel", help="cancel a job")
    cancelled.add_argument("id")
    commands.add_parser("deliveries", help="every delivery note")
    found = commands.add_parser("delivery", help="one delivery's note")
    found.add_argument("key")
    commands.add_parser("check", help="mark stale what an erasure reached")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "request":
            with open(args.file, encoding="utf-8") as held:
                state = api.request(json.load(held))
            if args.wait:
                state = api.wait(state["id"])
            _print(state["answer"] if args.wait and state["state"] == "done" else state)
            return 0 if state["state"] in ("queued", "running", "done") else 1
        if args.command == "work":
            say("running the queue until stopped (Ctrl+C)")
            try:
                if not api.work():
                    say("another process runs the queue", ERROR)
                    return 1
            except KeyboardInterrupt:
                return 0
            return 0
        if args.command == "jobs":
            _print(api.jobs())
        elif args.command == "job":
            state = api.job(args.id)
            _print(state)
            while args.follow and state["state"] in ("queued", "running"):
                time.sleep(FOLLOW)
                now = api.job(args.id)
                if now != state:
                    _print(now)
                state = now
        elif args.command == "cancel":
            _print(api.cancel(args.id))
        elif args.command == "deliveries":
            _print(api.deliveries_on())
        elif args.command == "delivery":
            note = api.delivery(args.key)
            if note is None:
                say(f"the dock holds no delivery {args.key}", ERROR)
                return 1
            _print(note)
        elif args.command == "check":
            _print(api.check())
    except Exception as error:  # noqa: BLE001 a refusal's reason, said once
        say(f"freight {args.command}: {type(error).__name__}: {error}", ERROR)
        return 1
    return 0


def entry():
    sys.exit(main())
