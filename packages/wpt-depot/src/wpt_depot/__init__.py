"""WPT's depot: texts and structured items, the decisions on them, libraries,
erasures and datasets, in Delta Lake tables of one folder (wpt_depot.depot),
and the corpus a dataset builds into the store (wpt_depot.corpus).

    depot.py     Depot: every change to a depot's tables, under its rules
    policy.py    how decisions combine, and what they make of a text: its status
    filters.py   a filter of the texts, and the SQL condition it stands for
    erasure.py   cutting erased passages, naming them by public hashes, and showing a request's places
    ids.py       the ids of rows: the depot's name, a hyphen and a number
    licenses.py  the allowed licenses, which admitting a text checks
    kinds.py     the starting kinds of items and the check of an item's content
    journal.py   changes across tables, written first and finished on start
    tables.py    the tables on disk, written through delta-rs, read through DuckDB
    follow.py    clone a depot and pull from its upstream
    backup.py    a full copy of a depot as of its settled change, and its restore
    remote.py    a dataset written out as JSON
    corpus.py    a dataset's corpus in the store
    notice.py    a model's notice from its corpus
    settings.py  WPT_DEPOT, the depot to work on, and WPT_DEPOT_* for a depot in a bucket
"""

from .backup import backup, restore
from .depot import About, Declaration, Depot, NewText
from .erasure import Needle
from .follow import clone, pull

__all__ = ["About", "Declaration", "Depot", "Needle", "NewText", "backup", "clone", "pull", "restore"]
