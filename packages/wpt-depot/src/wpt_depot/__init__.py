"""WPT's depot: texts and structured items, the decisions on them, erasures,
snapshots and tokenizer recipes, in one SQLite file (wpt_depot.depot), and
the corpus a snapshot builds into the store (wpt_depot.corpus).

    depot.py     Depot: every change to a depot's tables, under its rules
    schema.py    the tables of the file, their keys and the views of the current state
    database.py  the connections to the file, the person's read-only SQL, and the change files of each write
    parts.py     where a content is cut into parts, and reading a text from its parts
    policy.py    how decisions combine, and what they make of a text: its status
    counts.py    the depot's counts of texts by status, tag, license and decision, and of what the texts of each
                 status hold by kind, tag set, tag and license
    snapshots.py a name on a change number: the texts admitted then as roaring bitmaps, their digest and summary
    filters.py   a filter of the texts, and the SQL condition it stands for
    scoring.py   a change of the scores of many texts, checked and made in SQL
    crediting.py a change of the credits of many texts, checked in SQL over a temporary table
    rewriting.py a change of the plain texts of many texts, checked in SQL over a temporary table
    erasure.py   cutting erased passages, naming them by public hashes, and showing a request's places
    ids.py       the ids of rows: the depot's name, a hyphen and a number
    licenses.py  the allowed licenses, which admitting a text checks
    kinds.py     the starting kinds of items and the check of an item's content
    follow.py    clone a depot and pull from its upstream, over the upstream's web app
    backup.py    the change files and full copies sent into the store, and a restore from them
    upgrade.py   bringing a depot of an older format up to the code's
    delta.py     a depot of format 3 or older, a folder of Delta Lake tables, which only the upgrade reads
    format3.py   the step of such a folder from format 2 to format 3
    corpus.py    a snapshot's corpus in the store, under its build key
    order.py     a snapshot's texts in the order of their export keys, each read at its part's bound
    groups.py    how a share splits its budget over the tags its texts carry
    estimate.py  what a corpus build would give, from a small sample
    corpora.py   the corpora the store holds: records, staleness, the store's cap
    notice.py    a model's notice from its corpus
    legacy.py    the libraries and datasets of format 12, read-only, for models trained before snapshots
    conversion.py the snapshots the upgrade to format 13 makes of locked libraries and dataset revisions, read
                 from the libraries' tables; with legacy.py, the only code that reads those tables
    settings.py  WPT_DEPOT, the depot to work on, and WPT_DEPOT_URL, its public address
"""

from .backup import backup, restore
from .depot import About, Declaration, Depot, NewText
from .erasure import Needle
from .follow import clone, pull

__all__ = ["About", "Declaration", "Depot", "Needle", "NewText", "backup", "clone", "pull", "restore"]
