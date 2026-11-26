"""The roles of a corpus's portions: what each portion of a corpus is for,
and the rules each follows (docs/data-selection.md, "The portions of a
corpus"). The freight cuts a delivery into named portions and knows only
their names, kinds, shares and sizes (docs/api/freight.md, "A request");
what each one means is set here, never by an order. The order schemas of
wpt-orders, which cannot import this module, repeat the role names in
their order, that pretrain is always one and the default, and its check
refuses finetune_tokens without finetune and held_out with a total_tokens
under HELD_OUT_MIN_TOKENS; apps/wpt/tests/freight/test_portions.py holds
them to this module.

    held_out   plain texts no training reads, to measure a model on; always
               the first portion, with the shares of the pretraining
               portion, its size HELD_OUT_BYTES bytes
    pretrain   plain texts, what the steps that read text windows learn
               from; its size the data section's total_tokens, in tokens
    finetune   conversations and tasks, what the steps that read them learn
               from; its size the data section's finetune_tokens, in
               tokens (none: every conversation and task left)

A portion's name is its role's. A data section names the roles its corpus
has (`roles`, DEFAULT when it names none); pretrain is always one of them.
The portions fill in the order of ROLES, so the held-out portion comes
first: it reads nothing of the portions after it, and two corpora of one
snapshot and one set of shares hold exactly the same held-out texts,
whatever their vocabulary and their sizes.

HELD_OUT_BYTES is 64 MiB: a measure of bits per byte on 2 MiB is known to
about 1.7 percent, too coarse beside the gaps between the variants it
compares, and its error shrinks as the square root of the text measured,
so 32 times the text brings it to about 0.3 percent. 64 MiB is some 16
million tokens at about four bytes a token, and about a seventieth of the
text of the orders of a billion tokens.

A data section whose roles hold held_out asks for a total_tokens of at
least HELD_OUT_MIN_TOKENS, which is HELD_OUT_BYTES: the pretraining portion
then holds at least as many tokens as the held-out portion holds bytes, and
a token of text stands for at least one byte, so the held-out portion is
never the larger part of the plain text (at about four bytes a token, it is
at most a fifth of it). A smaller one is refused: its order names its roles
without held_out, as the small test orders do, and its models are measured
on nothing held out.

A corpus without portions (a delivery made without them, or one of the
corpora under the dock's corpora/) is all of it its pretraining portion,
read as it always was, and has no held-out portion.

    names(data)           the roles a data section's corpus has, in the order they fill
    portions(data)        those portions as a request to the freight names them
    TRAINING              the roles training reads
    role_of(kind)         the role whose portion training reads texts of that kind from
"""

from .encoding import CONVERSATION, PLAIN, TASK

HELD_OUT, PRETRAIN, FINETUNE = "held_out", "pretrain", "finetune"
ROLES = (HELD_OUT, PRETRAIN, FINETUNE)  # every role, in the order the portions fill
DEFAULT = (HELD_OUT, PRETRAIN)  # the roles of a data section that names none
KINDS = {HELD_OUT: (PLAIN,), PRETRAIN: (PLAIN,), FINETUNE: (CONVERSATION, TASK)}  # what each role's portion holds
TRAINING = (PRETRAIN, FINETUNE)  # the roles whose portions training reads; never HELD_OUT
HELD_OUT_BYTES = 64 * 1024 * 1024  # the size of the held-out portion, in bytes, the same for every corpus
HELD_OUT_MIN_TOKENS = HELD_OUT_BYTES  # the least total_tokens of a data section whose roles hold held_out
BYTES, TOKENS = "bytes", "tokens"  # the units of a portion's size


class Refused(ValueError):
    """A data section whose roles no corpus can have."""


def names(data: dict) -> list[str]:
    """The roles the corpus of a data section has, in the order its portions
    fill: its `roles`, else DEFAULT. Refused for a role that is none of
    ROLES, a role named twice, roles without pretrain, finetune_tokens
    without finetune, and held_out with a total_tokens under
    HELD_OUT_MIN_TOKENS."""
    given = data.get("roles")
    given = list(DEFAULT) if given is None else given
    if not isinstance(given, list) or not all(isinstance(one, str) for one in given):
        raise Refused(f"a data section's roles are a list of names among {', '.join(ROLES)}: {given!r}")
    unknown = [one for one in given if one not in ROLES]
    if unknown:
        raise Refused(f"no role {', '.join(unknown)}: the roles are {', '.join(ROLES)}")
    if len(set(given)) != len(given):
        raise Refused(f"a data section names a role twice: {', '.join(given)}")
    if PRETRAIN not in given:
        raise Refused(f"a data section's roles hold {PRETRAIN}, the portion its total_tokens sizes: {given!r}")
    if data.get("finetune_tokens") is not None and FINETUNE not in given:
        raise Refused(f"finetune_tokens sizes the {FINETUNE} portion, which the data section's roles do not hold")
    total = data.get("total_tokens")
    if HELD_OUT in given and total is not None and total < HELD_OUT_MIN_TOKENS:
        raise Refused(f"a corpus of {total:,} tokens of pretraining is too small for a held-out portion of "
                      f"{HELD_OUT_BYTES:,} bytes, which needs a total_tokens of at least {HELD_OUT_MIN_TOKENS:,}: "
                      f"name the roles without {HELD_OUT}, such as [\"{PRETRAIN}\"], or ask for more tokens")
    return [one for one in ROLES if one in given]


def portions(data: dict) -> list[dict]:
    """The portions of a data section's corpus, in order, each {name, kinds,
    shares, size} as a request to the freight names it: held_out on
    HELD_OUT_BYTES bytes of plain text with the section's shares, pretrain
    on total_tokens tokens of plain text with them, finetune on
    finetune_tokens tokens of conversations and tasks (None: all of them).
    Refused as names is."""
    shares = data.get("shares") or []
    sizes = {HELD_OUT: {"unit": BYTES, "amount": HELD_OUT_BYTES},
             PRETRAIN: None if data.get("total_tokens") is None else {"unit": TOKENS, "amount": data["total_tokens"]},
             FINETUNE: None if data.get("finetune_tokens") is None
             else {"unit": TOKENS, "amount": data["finetune_tokens"]}}
    return [{"name": role, "kinds": list(KINDS[role]), "shares": shares if PLAIN in KINDS[role] else [],
             "size": sizes[role]} for role in names(data)]


def role_of(kind: str) -> str:
    """The role whose portion training reads texts of that kind from."""
    return next(role for role in TRAINING if kind in KINDS[role])
