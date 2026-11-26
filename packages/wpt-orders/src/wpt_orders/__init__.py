"""WPT's orders and task images, read and checked.

A model order (data/orders/models/<name>.json) says everything that
changes a model: its data (the address of a depot, a snapshot there with
its change and digest, the kinds and shares by tag it takes, its total
tokens and its vocabulary: its sha256 and size, or its size and the bytes
of the sample it is learned from; orders written before
snapshots name a dataset of the depot instead, whose tokenizer recipe
gives the vocabulary), its architecture, its seed, its training steps and its
drafter (docs/model.md). A series (data/orders/series/<name>.json) holds
the rules several sizes of one model share, and series.py derives each
size's model order from it, for the vocabulary's size. A task image
(data/images/<name>/) is the Dockerfile of the container agentic tasks run
in, its lock, its packages' licenses and its shared pool of tools.

Orders (orders.py) is the only code that opens these files; guard.py
refuses any other (docs/architecture.md, "The service layer").
"""

from . import series
from .errors import InvalidOrder, NotFound, OrderError
from .orders import ASK, BUILT_IN_TOOLS, SIZE_OF, Orders, dump_json, validate

__all__ = ["ASK", "BUILT_IN_TOOLS", "SIZE_OF", "InvalidOrder", "NotFound", "OrderError", "Orders", "dump_json",
           "series", "validate"]
