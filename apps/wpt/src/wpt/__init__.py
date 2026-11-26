"""wpt: every WPT job, each a subcommand with its own options (`wpt
SUBCOMMAND --help`).

    wpt list                     every corpus and model the dock keeps
    wpt copy-out NAME FOLDER     a model the dock keeps, copied out
    wpt runs, wpt run FOLDER     the runs recorded on the dock, and one of them
    wpt agent ...                the images of agentic tasks, locked and built
    wpt build-resources ORDER    ask the freight for an order's vocabulary and corpus
    wpt train ...                train a model order on its corpus
    wpt eval ...                 run tests on a trained model, into its model card
    wpt chat ...                 continue text with a trained model, or chat with one
    wpt manager                  train a model order on its corpus on the dock, through rented machines
    wpt doctor                   check this machine: what is missing and how to fix it

The subcommands are controllers over the services, the libraries of
packages/. The depot, the freight and the dock have commands of their own:
`depot`, `freight` and `dock`. A subcommand's module is imported only when
it runs, so a light one (`wpt list`) never loads PyTorch.
"""

import importlib
import sys

from wic_utils.lines import ERROR, say

DOCKED = "wpt.docked"  # its main takes the subcommand's name first
# each subcommand: the module whose main(argv) runs it
COMMANDS = {"list": "wpt.docked", "copy-out": "wpt.docked", "runs": "wpt.docked", "run": "wpt.docked",
            "agent": "wpt.agent", "train": "wpt.train",
            "eval": "wpt.eval", "chat": "wpt.chat", "manager": "wpt.manager",
            "doctor": "wpt.doctor", "build-resources": "wpt.resources"}


def main(argv=None):
    """Run the subcommand argv names with the rest of argv; its exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if not argv or argv[0] not in COMMANDS:
        known = ", ".join(COMMANDS)
        say(f"wpt: {f'unknown subcommand {argv[0]!r}' if argv else 'no subcommand'}; one of {known} "
            "(wpt --help)", ERROR)
        return 2
    name, *rest = argv
    try:
        return importlib.import_module(COMMANDS[name]).main([name, *rest] if COMMANDS[name] == DOCKED else rest)
    except SystemExit as stopped:
        if isinstance(stopped.code, str):  # a refusal's text, which Python would print as it is
            say(stopped.code, ERROR)
            return 1
        raise
