"""wpt: every WPT job, each a subcommand with its own options (`wpt
SUBCOMMAND --help`).

    wpt depot ...     the depot: texts, decisions, libraries, datasets, erasures, corpora, and its web pages
    wpt store ...     what the store holds, its runs, a kept model copied out
    wpt agent ...     the images of agentic tasks, locked and built
    wpt train ...     train a model recipe on the corpus of its depot and dataset
    wpt eval ...      run tests on a trained model, into its model card
    wpt chat ...      continue text with a trained model, or chat with one
    wpt manager       train a model recipe on its corpus in the store, through rented machines
    wpt doctor        check this machine: what is missing and how to fix it

The subcommands are controllers over the services, the libraries of
packages/; `wpt depot web` serves the depot's operations as pages too. A
subcommand's module is imported only when it runs, so a light one (`wpt
store`) never loads PyTorch.
"""

import importlib
import sys

from wpt_utils.lines import ERROR, say

# each subcommand: the module whose main(argv) runs it
COMMANDS = {"depot": "wpt.depot", "store": "wpt.store", "agent": "wpt.agent", "train": "wpt.train",
            "eval": "wpt.eval", "chat": "wpt.chat", "manager": "wpt.manager",
            "doctor": "wpt.doctor"}


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
        return importlib.import_module(COMMANDS[name]).main(rest)
    except SystemExit as stopped:
        if isinstance(stopped.code, str):  # a refusal's text, which Python would print as it is
            say(stopped.code, ERROR)
            return 1
        raise
