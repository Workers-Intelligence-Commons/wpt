"""wpt build-resources: what an order's data section asks of the freight,
sent to it, and found again on the dock.

    wpt build-resources ORDER    send the order's requests to the freight in order and print both delivery notes

ORDER is a model order's name, or SERIES: for the one corpus every size of
a series reads (its largest size's). Its data section names a snapshot of a
depot.

The data section makes two requests (docs/api/freight.md, "A request"):

- the vocabulary, when its `vocabulary` is `{"size", "sample_bytes"}`: the
  program wpt-vocabulary learns `size` entries from `sample_bytes` bytes of
  the plain texts of the section's snapshot, chosen with its shares. A
  `vocabulary` named `{"sha256", "size"}` makes no request: the dock must
  hold a delivery of that vocabulary already (vocabulary_key).
- the corpus: the program wpt-tokens tokenizes the portions of the
  section's roles (wpt_corpus.roles), in order: the held-out portion first,
  on a size in bytes fixed in WPT's code, then the pretraining portion at
  the size `total_tokens` in tokens, then the fine-tuning one at
  `finetune_tokens`, each with the section's shares where it holds plain
  texts; its input `vocabulary` the vocabulary's delivery.

Each delivery key is worked out from the order alone (freight.api.key,
with the version the program wpt_corpus holds answers that request), with no depot and no
freight at hand, which is how `wpt train` finds its corpus; a vocabulary
named by its sha256 is the one delivery of that vocabulary the dock holds,
found by its note. The command sends the requests one after the other,
waits for each job (running the freight's queue here while no other
process does; a job of the same delivery already queued or running, such
as one a stopped run left, is waited for, and goes on from its
checkpoint), and prints the two notes, `{"vocabulary": ..., "corpus":
...}`, as JSON. Settings (environment or .env): FREIGHT_HOME, DEPOT and
DOCK, as the freight reads them. Exit codes: 0 done, 1 refused or
failed.

    vocabulary_request(data)            the vocabulary's request, None for a vocabulary named by its sha256
    corpus_request(data, vocabulary)    the corpus's request, its input the vocabulary's delivery key
    vocabulary_key(data, dock)          the key of the vocabulary's delivery; None when the dock holds none of a sha256
    corpus_key(data, vocabulary)        the key of the corpus's delivery
    vocabulary_named(data)              whether the section names its vocabulary by its sha256
    label(note)                         a delivery shown by what made it
    corpus_record(corpus, vocabulary)   a corpus delivery as training reads a corpus: of its held-out portion, the
                                        token files its validation reads alone
    held_out(corpus)                    the held-out portion of a corpus delivery, None without one
"""

import argparse
import json
import sys

from freight import api
from wpt_corpus import programs, roles
from wpt_corpus import vocabulary as learning
from wpt_orders import Orders
from wic_utils.lines import ERROR, say

TOKENS = "wpt-tokens"  # the program that writes a corpus


class Unanswered(Exception):
    """The freight did not deliver what the order asks: the reason."""


def vocabulary_named(data: dict) -> bool:
    """Whether the data section names its vocabulary by its sha256 (else: learned for it)."""
    return "sha256" in data["vocabulary"]


def _snapshot(data: dict) -> dict:
    return {"name": data["snapshot"], "change": data["change"], "digest": data["digest"]}


def vocabulary_request(data: dict) -> dict | None:
    """The request of the vocabulary a data section learns: wpt-vocabulary on
    sample_bytes bytes of the plain texts of its snapshot, with its shares.
    None for a vocabulary named by its sha256, which no request makes."""
    if vocabulary_named(data):
        return None
    return {"depot": data["depot"], "snapshot": _snapshot(data),
            "selection": {"kinds": [learning.PLAIN], "shares": data.get("shares") or [],
                          "size": {"unit": "bytes", "amount": data["vocabulary"]["sample_bytes"]}},
            "preparation": {"name": learning.NAME, "settings": {"vocab_size": data["vocabulary"]["size"]}}}


def corpus_request(data: dict, vocabulary: str) -> dict:
    """The request of a data section's corpus: wpt-tokens on the portions of
    its roles, in order (wpt_corpus.roles.portions: the held-out one first),
    its input `vocabulary` the delivery of that key. roles.Refused for roles
    no corpus can have."""
    return {"depot": data["depot"], "snapshot": _snapshot(data), "selection": {"portions": roles.portions(data)},
            "preparation": {"name": TOKENS, "inputs": {"vocabulary": vocabulary}}}


def vocabulary_key(data: dict, dock: str | None = None) -> str | None:
    """The key of the delivery of a data section's vocabulary. A learned
    one's is worked out from the section alone. One named by its sha256 is
    the delivery of that vocabulary the dock holds, fresh: the adopted one
    (`adopted-<sha256>`) when there is one, else the vocabulary delivery
    of that sha256 whose key comes first; None when the dock holds none."""
    if not vocabulary_named(data):
        return api.key(vocabulary_request(data), learning.VERSION)
    sha256 = data["vocabulary"]["sha256"]
    adopted = api.delivery(f"{api.ADOPTED}{sha256}", dock=dock)
    if adopted is not None and _fresh(adopted):
        return adopted["key"]
    found = sorted(note["key"] for note in api.deliveries_on(dock=dock)
                   if _fresh(note) and (note.get("preparation") or {}).get("name") == learning.NAME
                   and (note.get("result") or {}).get("sha256") == sha256)
    return found[0] if found else None


def corpus_key(data: dict, vocabulary: str) -> str:
    """The key of the delivery of a data section's corpus, its vocabulary the delivery of that key."""
    request = corpus_request(data, vocabulary)
    return api.key(request, programs.version(True, bool(request["selection"].get("portions"))))


def _fresh(note: dict) -> bool:
    return "stale" not in note and "deleted" not in note


def label(note: dict) -> str:
    """A delivery shown by what made it (docs/tokenizer.md, "A vocabulary in
    an order"): its program, or for an adopted one where it comes from, then
    the snapshot, the shares and the size."""
    if "adopted" in note:
        about = note["adopted"]
        named = about.get("recipe")
        return (f"vocabulary {note['result'].get('size', 0):,} · adopted"
                + (f" · tokenizer recipe {named} version {about.get('version')}" if named else ""))
    request, preparation = note["request"], note["preparation"]
    selection = request.get("selection") or {}
    if preparation["name"] == learning.NAME:
        head = f"vocabulary {note['result']['size']:,}"
    else:
        head = preparation["name"]
    if selection.get("portions"):
        cut = [f"{one['name']} {_said(one)}" for one in selection["portions"]]
        return " · ".join([head, f"snapshot {note['snapshot']['name']}", *cut])
    return " · ".join([head if preparation["name"] == learning.NAME else
                       f"{head} {', '.join(selection.get('kinds') or []) or 'every kind'}",
                       f"snapshot {note['snapshot']['name']}", _said(selection)])


def _said(selection: dict) -> str:
    """A selection's, or a portion's, shares and size, as label shows them."""
    shares = " / ".join(f"{one['tag']} {one['share'] * 100:g}" for one in selection.get("shares") or [])
    size = selection.get("size")
    if "name" in selection:  # a portion: its kinds first
        shares = f"{', '.join(selection.get('kinds') or []) or 'every kind'}" + (f" ({shares})" if shares else "")
    elif not shares:
        shares = "no shares"
    amount = "every text" if size is None else f"{size['amount']:,} {size['unit']}"
    return f"{shares} · {amount}"


def corpus_record(corpus: dict, vocabulary: dict) -> dict:
    """A corpus delivery's note, with the note of the vocabulary it names,
    in the shape training reads a corpus by: its `id`,
    `delivery` (its key), `vocabulary_delivery`, its token files by kind
    (`shards`, each the dock's reference with what the program said of it),
    its tables (`texts`: the items tables), `tasks`, `tokenizer` (the
    vocabulary's file and size, named "RECIPE version N" for one adopted
    from a depot's tokenizer recipe, else by label), `max_tokens` (the size in
    tokens it was asked for), `total_tokens`, `items_tokens`,
    `fingerprints`, and `stale` or `deleted` as its note says.

    A delivery cut into portions gives training its portions of the roles
    training reads alone (wpt_corpus.roles.TRAINING): the token files, tasks
    and fingerprints of each kind those of the portion whose role holds
    that kind (plain texts the pretraining portion's, conversations and
    tasks the fine-tuning one's), `total_tokens` and `max_tokens` the
    pretraining portion's, `texts` and `conversations` counted over them,
    and `trains_on` their names, the only portions whose rows of the items
    tables training reads (wpt_corpus.corpora.rows). Of the held-out
    portion it names only the token files (`held_out`: {"shards",
    "total_tokens"}), which the steps that read text windows measure their
    validation loss on and never train on; no row of its texts is read, and
    `portions` counts each portion's texts and tokens. A delivery without
    portions is all of it its pretraining
    portion, read as a whole (no `trains_on`). ValueError for a delivery cut
    into portions whose result does not list its files by portion (a
    wpt-tokens older than programs.PORTIONED_VERSION wrote it)."""
    result = corpus["result"]
    by_path = {one["path"]: one for one in corpus["files"]}

    def placed(one: dict) -> dict:
        file = by_path[one["path"]]
        return {"key": file["key"], "sha256": file["sha256"], "bytes": file["bytes"], **one}

    (vocabulary_file,) = [one for one in vocabulary["files"] if one["path"] == learning.FILE]
    selection = corpus["request"].get("selection") or {}
    about = vocabulary.get("adopted") or {}
    record = {
        "id": corpus["id"], "delivery": corpus["key"], "vocabulary_delivery": vocabulary["key"],
        "depot": corpus["request"].get("depot"), "snapshot": corpus["snapshot"], "rules": corpus["rules"],
        "request": corpus["request"], "preparation": corpus["preparation"], "made_at": corpus["made_at"],
        "tokenizer": {"recipe": f"{about['recipe']} version {about['version']}" if about.get("recipe")
                      else label(vocabulary), "sha256": vocabulary_file["sha256"],
                      "vocab_size": vocabulary["result"]["size"],
                      "vocabulary": {"key": vocabulary_file["key"], "sha256": vocabulary_file["sha256"],
                                     "bytes": vocabulary_file["bytes"]}},
        "vocabulary": {"sha256": vocabulary_file["sha256"], "size": vocabulary["result"]["size"]},
        "tables": {"texts": corpus["tables"]["items"], "licenses": corpus["tables"]["licenses"],
                   "notices": corpus["tables"]["notices"]},
        "deliveries": {"vocabulary": _named(vocabulary), "corpus": _named(corpus)},
    }
    if selection.get("portions") and "portions" not in result:
        raise ValueError(f"corpus delivery {corpus['key']} was cut into portions by a wpt-tokens of version "
                         f"{corpus['preparation'].get('version')}, which wrote every portion into the same files: "
                         f"ask the freight for the order's corpus again (`wpt build-resources`)")
    if selection.get("portions"):
        asked = {one["name"]: one for one in selection["portions"]}
        counted = {one["name"]: one for one in corpus["portions"]}
        made = result["portions"]
        trains_on = [role for role in roles.TRAINING if role in made]
        size = asked[roles.PRETRAIN]["size"]

        def by_role(kind: str) -> dict | None:
            role = roles.role_of(kind)
            return made[role] if role in made else None

        kinds = [*made[roles.PRETRAIN]["shards"], roles.TASK]  # the kinds a step reads: each has a fingerprint
        record.update({
            "texts": sum(counted[role]["items"]["texts"] for role in trains_on),
            "pieces": sum(counted[role]["items"]["pieces"] for role in trains_on),
            "conversations": sum(counted[role]["items"]["kinds"].get("conversation", 0) for role in trains_on),
            "max_tokens": None if size is None else size["amount"],
            "shards": {kind: [placed(one) for one in by_role(kind)["shards"][kind]] if by_role(kind) else []
                       for kind in made[roles.PRETRAIN]["shards"]},
            "total_tokens": made[roles.PRETRAIN]["total_tokens"],
            "items_tokens": made[roles.FINETUNE]["items_tokens"] if roles.FINETUNE in made else 0,
            "fingerprints": {kind: by_role(kind)["fingerprints"][kind] for kind in kinds if by_role(kind)},
            "trains_on": trains_on,
            "portions": {name: {"texts": counted[name]["items"]["texts"], "tokens": made[name]["total_tokens"]
                                + made[name]["items_tokens"]} for name in made},
        })
        if roles.HELD_OUT in made:
            record["held_out"] = {"shards": [placed(one) for one in made[roles.HELD_OUT]["shards"][roles.PLAIN]],
                                  "total_tokens": made[roles.HELD_OUT]["total_tokens"]}
        tasks = (by_role(roles.TASK) or {}).get("tasks")
    else:
        size = selection.get("size")
        record.update({
            "texts": corpus["items"]["texts"], "pieces": corpus["items"]["pieces"],
            "conversations": corpus["items"]["kinds"].get("conversation", 0),
            "max_tokens": None if size is None else size["amount"],
            "shards": {kind: [placed(one) for one in kept] for kind, kept in result["shards"].items()},
            "total_tokens": result["total_tokens"], "items_tokens": result["items_tokens"],
            "fingerprints": result["fingerprints"],
        })
        tasks = result.get("tasks")
    if tasks:
        record["tasks"] = {"file": placed(tasks), "count": tasks["count"]}
    for mark in ("stale", "deleted"):
        if mark in corpus:
            record[mark] = corpus[mark]
    return record


def held_out(corpus: dict) -> dict | None:
    """The held-out portion of a corpus delivery's note
    (wpt_corpus.roles.HELD_OUT), what wpt eval's bits_per_byte measures a
    model on and no training reads: {"shards": its token files of plain text, each the
    dock's reference with what the program said of it, "total_tokens",
    "texts", "bytes" (what the depot's listing gives its texts)}. None for a
    delivery without one, made without portions or for an order whose roles
    leave it out."""
    made = (corpus.get("result") or {}).get("portions") or {}
    if roles.HELD_OUT not in made:
        return None
    by_path = {one["path"]: one for one in corpus["files"]}
    (counted,) = [one for one in corpus["portions"] if one["name"] == roles.HELD_OUT]
    found = made[roles.HELD_OUT]
    return {"shards": [{"key": by_path[one["path"]]["key"], "sha256": by_path[one["path"]]["sha256"],
                        "bytes": by_path[one["path"]]["bytes"], **one} for one in found["shards"][roles.PLAIN]],
            "total_tokens": found["total_tokens"], "texts": counted["items"]["texts"], "bytes": counted["bytes"]}


def _named(note: dict) -> dict:
    """What a model folder keeps of a delivery: its key, id and preparation (for an adopted one, where it comes
    from)."""
    made = ({"adopted": note["adopted"]} if "adopted" in note
            else {"preparation": {key: note["preparation"][key] for key in ("name", "version", "settings")}})
    return {"key": note["key"], "id": note["id"], **made}


def _asked(request: dict, key: str, what: str) -> dict:
    """The delivery note of the request, whose delivery key is key, its job
    run to its end. A job of that key already queued or running, such as
    one a stopped run of this command left, is the one waited for: a job
    cut short goes on from its checkpoint. Unanswered when the job did not
    end done."""
    try:
        state = api.request(request)
    except api.Busy:
        state = next((job for job in api.jobs() if job["key"] == key and job["state"] in ("queued", "running")),
                     None)
        if state is None:  # it ended meanwhile: ask again, which answers its delivery at once
            state = api.request(request)
    if state["state"] not in ("queued", "running", "done"):
        raise Unanswered(f"the {what}'s request was {state['state']}: {state.get('problem')}")
    say(f"{what}: job {state['id']} {state['state']}")
    if state["state"] != "done":
        state = api.wait(state["id"])
    if state["state"] != "done":
        raise Unanswered(f"the {what}'s job {state['id']} ended {state['state']}: {state.get('problem')}")
    return state["answer"]


def build(data: dict) -> dict:
    """Send a data section's requests to the freight, the vocabulary first
    when it is learned, then the corpus: {"vocabulary": its note, "corpus":
    its note}. roles.Refused, before any request, for roles no corpus of
    the section can have; Unanswered when the dock holds no vocabulary of a
    sha256 named, or a job does not end done."""
    roles.names(data)
    asking = vocabulary_request(data)
    if asking is None:
        key = vocabulary_key(data)
        if key is None:
            raise Unanswered(f"the dock holds no vocabulary {data['vocabulary']['sha256']}: name one the dock holds, "
                             "or let the order learn its vocabulary, {\"size\", \"sample_bytes\"}")
        vocabulary = api.delivery(key)
        say(f"vocabulary: delivery {key}, already on the dock")
    else:
        vocabulary = _asked(asking, vocabulary_key(data), "vocabulary")
    say(f"vocabulary {vocabulary['result']['sha256']}: {label(vocabulary)}")
    corpus = _asked(corpus_request(data, vocabulary["key"]), corpus_key(data, vocabulary["key"]), "corpus")
    made = corpus["result"].get("portions") or {None: corpus["result"]}
    held = {one["name"]: one["items"]["texts"] for one in corpus.get("portions") or []}
    say(f"corpus {corpus['id']} (delivery {corpus['key']}): {label(corpus)}; "
        + ", ".join(f"{'' if name is None else f'{name}: '}{one['total_tokens']:,} tokens of plain text, "
                    f"{one['items_tokens']:,} of conversations and tasks, "
                    f"{held.get(name, corpus['items']['texts']):,} texts" for name, one in made.items()))
    return {"vocabulary": vocabulary, "corpus": corpus}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wpt build-resources", description=__doc__.split("\n\n")[0])
    parser.add_argument("order", help="a model order's name, or SERIES: for a series")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        data = Orders.find().data_part(args.order)
        notes = build(data)
    except Exception as error:  # noqa: BLE001 a refusal of the order, the freight or the dock, said once
        say(f"wpt build-resources {args.order}: {type(error).__name__}: {error}", ERROR)
        return 1
    print(json.dumps(notes, indent=2, ensure_ascii=False))
    return 0
