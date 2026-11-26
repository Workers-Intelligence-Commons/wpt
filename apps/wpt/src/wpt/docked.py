"""What WPT keeps on the dock named by DOCK (environment or .env), four
wpt subcommands that read it.

    wpt list                    every corpus the dock keeps under corpora/ (its id, its build key, the
                                snapshot and depot and the order it was built for, or for one built
                                before snapshots its dataset; when, its tokens and texts, and whether it
                                is fresh) and every model it keeps; `freight deliveries` lists the
                                deliveries
    wpt copy-out NAME FOLDER    the model folder the dock keeps under the run's name NAME (what the wpt
                                manager prints at the end of a run) into the new FOLDER, each file
                                checked against its sha256
    wpt runs                    every run recorded on the dock: its folder, command, start and how it
                                ended
    wpt run FOLDER [--follow]   one run (FOLDER, or a part of its name only that folder holds): its
                                stage, its counts, its last events and its errors; --follow prints its
                                lines as they reach the dock until it ends
"""

import argparse
import sys
from pathlib import Path

from dock import api as dock_api
from dock.api import DockError
from wpt_corpus import corpora
from wpt.models import copy_out, names
from wpt_telemetry import runs
from wic_utils.lines import ERROR, say


def cmd_list(args, store):
    found = corpora.records(store)
    say(f"{len(found)} corpora:")
    for record in found:
        say(f"  {record['id']}: {_source(record)}, built {record['built_at']}, {record['total_tokens']:,} tokens of "
            f"plain text, {record['texts']:,} texts" + ("" if corpora.fresh(record) else ", not fresh"))
    models = names(store)
    say(f"{len(models)} models:")
    for name in models:
        say(f"  {name}")
    return 0


def _source(record) -> str:
    """What a corpus was built for: its build key, snapshot and order, or
    for a corpus built before snapshots, its dataset."""
    if "build_key" not in record:
        return f"dataset {record['dataset']['name']} of {record['depot']}"
    snapshot, size = record["snapshot"], record.get("max_tokens")
    return (f"build key {record['build_key']}, snapshot {snapshot['name']} (change {snapshot['change']}) of "
            f"{record['depot']}, order {record.get('recipe') or '(none)'}, "
            + ("every token" if size is None else f"size {size:,}"))


def cmd_copy_out(args, store):
    count = copy_out(store, args.name, args.folder)
    say(f"model {args.name}: {count} files in {args.folder}")
    return 0


def cmd_runs(args, store):
    found = runs.runs(store)
    for run in found:
        took = f" in {run.seconds:,.0f} s" if run.seconds is not None else ""
        say(f"{run.started or '?'} {run.command or '?'}: {run.status}{took}: {run.folder}")
    say(f"{len(found)} runs")
    return 0


def cmd_run(args, store):
    try:
        folder = runs.find(store, args.folder)
    except ValueError as error:
        say(f"wpt run: {error}", ERROR)
        return 2
    if args.follow:
        for text, level in runs.follow(store, folder):
            say(text, level, sys.stdout)
    found = runs.summary(store, folder)
    say(f"{found.run.folder}: {found.run.command or '?'} {' '.join(found.arguments)}")
    say(f"started {found.run.started or '?'}; {found.run.status}"
        + (f" in {found.run.seconds:,.0f} s" if found.run.seconds is not None else ""))
    if found.error:
        say(f"error: {found.error}")
    say(f"stage: {' > '.join(found.stage) or '(no span open)'}")
    say(f"{found.lines:,} printed lines; spans: {_counts(found.spans)}; events: {_counts(found.events)}")
    for title, lines in (("last events", found.last), ("errors", found.errors)):
        if lines:
            say(f"{title}:")
            for text, level in lines:
                say(f"  {text}", level, sys.stdout)
    return 0


def _counts(counted: dict) -> str:
    return ", ".join(f"{name} {count:,}" for name, count in sorted(counted.items())) or "none"


def main(argv=None) -> int:
    """Run one of the four subcommands, its name first in argv."""
    parser = argparse.ArgumentParser(prog="wpt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="the corpora the depot built under the dock's corpora/, and the models (freight deliveries lists the deliveries)").set_defaults(run=cmd_list)
    copying = commands.add_parser("copy-out", help="a model folder the dock keeps, into a new folder")
    copying.add_argument("name", help="the run's name, which the wpt manager printed")
    copying.add_argument("folder", type=Path, help="the folder to write, which must not exist yet")
    copying.set_defaults(run=cmd_copy_out)
    commands.add_parser("runs", help="every run recorded on the dock").set_defaults(run=cmd_runs)
    showing = commands.add_parser("run", help="one run: its stage, counts, last events and errors")
    showing.add_argument("folder", help="its record folder, or a part of the name only that folder holds")
    showing.add_argument("--follow", action="store_true", help="print its lines as they reach the dock until it ends")
    showing.set_defaults(run=cmd_run)
    args = parser.parse_args(argv)
    try:
        return args.run(args, dock_api.url(Path.cwd() / ".env"))
    except (DockError, FileExistsError) as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return 2


if __name__ == "__main__":
    sys.exit(main())
