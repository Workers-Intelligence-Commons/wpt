"""WPT's depot: texts and structured items, the decisions on them, libraries,
erasures and datasets, in one SQLite file (wpt_depot.depot), and the corpus
a dataset builds into the store (wpt_depot.corpus).

    depot.py     Depot: every change to a depot's tables, under its rules
    schema.py    the tables of the file, their keys and the views of the current state
    database.py  the connections to the file, the person's read-only SQL, and the change files of each write
    parts.py     where a content is cut into parts, and reading a text from its parts
    policy.py    how decisions combine, and what they make of a text: its status
    filters.py   a filter of the texts, and the SQL condition it stands for
    scoring.py   a change of the scores of many texts, checked and made in SQL
    erasure.py   cutting erased passages, naming them by public hashes, and showing a request's places
    ids.py       the ids of rows: the depot's name, a hyphen and a number
    licenses.py  the allowed licenses, which admitting a text checks
    kinds.py     the starting kinds of items and the check of an item's content
    follow.py    clone a depot and pull from its upstream, over the upstream's web app
    backup.py    the change files and full copies sent into the store, and a restore from them
    upgrade.py   bringing a depot of an older format up to the code's
    legacy.py    a depot of format 3 or older, a folder of Delta Lake tables, which only the upgrade reads
    format3.py   the step of such a folder from format 2 to format 3
    remote.py    a dataset written out as JSON
    corpus.py    a dataset's corpus in the store
    notice.py    a model's notice from its corpus
    settings.py  WPT_DEPOT, the depot to work on
"""

from .backup import backup, restore
from .depot import About, Declaration, Depot, NewText
from .erasure import Needle
from .follow import clone, pull

__all__ = ["About", "Declaration", "Depot", "Needle", "NewText", "backup", "clone", "pull", "restore"]
