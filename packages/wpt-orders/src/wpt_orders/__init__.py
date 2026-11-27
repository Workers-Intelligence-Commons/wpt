"""WPT's orders and task images, read and checked.

A model family (data/orders/<name>.json) writes once what its sizes
share, then a table of its sizes, and family.py resolves each size to a
model order. A model order says everything that changes a model: its data
(the address of a depot, a snapshot there with its change and digest, the
roles of its corpus's portions, the shares by tag it takes, the sizes of its
pretraining portion (total_tokens) and of its finetune one, and its
vocabulary: its sha256 and size, or its size and the bytes of the sample it
is learned from), its architecture, its seed, its training steps and its
drafter (docs/model.md). counts.py counts what a model's numbers make: its
layers, its parameters and the compute of a token. A task image
(data/images/<name>/) is the Dockerfile of the container agentic tasks run
in, its lock, its packages' licenses and its shared pool of tools.

Orders (orders.py) is the only code that opens these files; guard.py
refuses any other (docs/architecture.md, "The service layer").
"""

from . import counts, family
from .errors import InvalidOrder, NotFound, OrderError
from .orders import ASK, BUILT_IN_TOOLS, SIZE_OF, Orders, dump_json, validate

__all__ = ["ASK", "BUILT_IN_TOOLS", "SIZE_OF", "InvalidOrder", "NotFound", "OrderError", "Orders", "counts",
           "dump_json", "family", "validate"]
