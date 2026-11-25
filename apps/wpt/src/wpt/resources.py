"""wpt build-resources: what an order's data section asks of the freight,
sent to it, and found again on the dock.

    wpt build-resources ORDER    send the order's requests to the freight in order and print both delivery notes

ORDER is a model recipe's name, or SERIES: for the one corpus every size of
a series reads (its largest size's). Its data section names a snapshot of a
depot; the legacy shape, which names a dataset, makes no request.

The data section makes two requests (docs/api/freight.md, "A request"):

- the vocabulary, when its `vocabulary` is `{"size", "sample_bytes"}`: the
  program wpt-vocabulary learns `size` entries from `sample_bytes` bytes of
  the plain texts of the section's snapshot, chosen with its shares. A
  `vocabulary` named `{"sha256", "size"}` makes no request: the dock must
  hold a delivery of that vocabulary already (vocabulary_key).
- the corpus: the program wpt-tokens tokenizes the section's kinds with its
  shares, at the size `total_tokens` in tokens, its input `vocabulary` the
  vocabulary's delivery.

Each delivery key is worked out from the order alone (freight.api.key,
with the version of the program wpt_corpus holds), with no depot and no
freight at hand, which is how `wpt train` finds its corpus; a vocabulary
named by its sha256 is the one delivery of that vocabulary the dock holds,
found by its note. The command sends the requests one after the other,
waits for each job (running the freight's queue here while no other
process does), and prints the two notes, `{"vocabulary": ..., "corpus":
...}`, as JSON. Settings (environment or .env): FREIGHT_HOME, WPT_DEPOT and
WPT_STORE, as the freight reads them. Exit codes: 0 done, 1 refused or
failed.

    vocabulary_request(data)            the vocabulary's request, None for a vocabulary named by its sha256
    corpus_request(data, vocabulary)    the corpus's request, its input the vocabulary's delivery key
    vocabulary_key(data, dock)          the key of the vocabulary's delivery; None when the dock holds none of a sha256
    corpus_key(data, vocabulary)        the key of the corpus's delivery
    vocabulary_named(data)              whether the section names its vocabulary by its sha256
    label(note)                         a delivery shown by what made it
    corpus_record(corpus, vocabulary)   a corpus delivery as training reads a corpus
"""

import argparse
import json
import sys

from freight import api
from wpt_corpus import programs
from wpt_corpus import vocabulary as learning
from wpt_recipes import Recipes
from wpt_utils.lines import ERROR, say

TOKENS = "wpt-tokens"  # the program that writes a corpus
ADOPTED = "adopted-"  # the start of the key of a vocabulary put on the dock from elsewhere (wpt adopt-vocabularies)


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
    """The request of a data section's corpus: wpt-tokens on its kinds, with
    its shares, at total_tokens tokens, its input `vocabulary` the delivery of
    that key."""
    return {"depot": data["depot"], "snapshot": _snapshot(data),
            "selection": {"kinds": data.get("kinds") or [], "shares": data.get("shares") or [],
                          "size": {"unit": "tokens", "amount": data["total_tokens"]}},
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
    adopted = api.delivery(f"{ADOPTED}{sha256}", dock=dock)
    if adopted is not None and _fresh(adopted):
        return adopted["key"]
    found = sorted(note["key"] for note in api.deliveries_on(dock=dock)
                   if _fresh(note) and (note.get("preparation") or {}).get("name") == learning.NAME
                   and (note.get("result") or {}).get("sha256") == sha256)
    return found[0] if found else None


def corpus_key(data: dict, vocabulary: str) -> str:
    """The key of the delivery of a data section's corpus, its vocabulary the delivery of that key."""
    return api.key(corpus_request(data, vocabulary), programs.VERSION)


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
        head = f"{preparation['name']} {', '.join(selection.get('kinds') or []) or 'every kind'}"
    shares = " / ".join(f"{one['tag']} {one['share'] * 100:g}" for one in selection.get("shares") or [])
    size = selection.get("size")
    parts = [head, f"snapshot {note['snapshot']['name']}", shares or "no shares",
             "every text" if size is None else f"{size['amount']:,} {size['unit']}"]
    return " · ".join(parts)


def corpus_record(corpus: dict, vocabulary: dict) -> dict:
    """A corpus delivery's note, with the note of the vocabulary it names,
    in the shape training reads a corpus by (wpt_depot.corpus): its `id`,
    `delivery` (its key), `vocabulary_delivery`, its token files by kind
    (`shards`, each the dock's reference with what the program said of it),
    its tables (`texts`: the items tables), `tasks`, `tokenizer` (the
    vocabulary's file and size, named "RECIPE version N" for one adopted
    from a depot's tokenizer recipe, else by label), `max_tokens` (the size in
    tokens it was asked for), `total_tokens`, `items_tokens`,
    `fingerprints`, and `stale` or `deleted` as its note says."""
    result = corpus["result"]
    by_path = {one["path"]: one for one in corpus["files"]}

    def placed(one: dict) -> dict:
        file = by_path[one["path"]]
        return {"key": file["key"], "sha256": file["sha256"], "bytes": file["bytes"], **one}

    (vocabulary_file,) = [one for one in vocabulary["files"] if one["path"] == learning.FILE]
    size = (corpus["request"].get("selection") or {}).get("size")
    about = vocabulary.get("adopted") or {}
    record = {
        "id": corpus["id"], "delivery": corpus["key"], "vocabulary_delivery": vocabulary["key"],
        "depot": corpus["request"].get("depot"), "snapshot": corpus["snapshot"], "rules": corpus["rules"],
        "request": corpus["request"], "preparation": corpus["preparation"], "made_at": corpus["made_at"],
        "texts": corpus["items"]["texts"], "pieces": corpus["items"]["pieces"],
        "conversations": corpus["items"]["kinds"].get("conversation", 0),
        "tokenizer": {"recipe": f"{about['recipe']} version {about['version']}" if about.get("recipe")
                      else label(vocabulary), "sha256": vocabulary_file["sha256"],
                      "vocab_size": vocabulary["result"]["size"],
                      "vocabulary": {"key": vocabulary_file["key"], "sha256": vocabulary_file["sha256"],
                                     "bytes": vocabulary_file["bytes"]}},
        "vocabulary": {"sha256": vocabulary_file["sha256"], "size": vocabulary["result"]["size"]},
        "max_tokens": None if size is None else size["amount"],
        "shards": {kind: [placed(one) for one in kept] for kind, kept in result["shards"].items()},
        "tables": {"texts": corpus["tables"]["items"], "licenses": corpus["tables"]["licenses"],
                   "notices": corpus["tables"]["notices"]},
        "total_tokens": result["total_tokens"], "items_tokens": result["items_tokens"],
        "fingerprints": result["fingerprints"],
        "deliveries": {"vocabulary": _named(vocabulary), "corpus": _named(corpus)},
    }
    if result.get("tasks"):
        record["tasks"] = {"file": placed(result["tasks"]), "count": result["tasks"]["count"]}
    for mark in ("stale", "deleted"):
        if mark in corpus:
            record[mark] = corpus[mark]
    return record


def _named(note: dict) -> dict:
    """What a model folder keeps of a delivery: its key, id and preparation (for an adopted one, where it comes
    from)."""
    made = ({"adopted": note["adopted"]} if "adopted" in note
            else {"preparation": {key: note["preparation"][key] for key in ("name", "version", "settings")}})
    return {"key": note["key"], "id": note["id"], **made}


def _asked(request: dict, what: str) -> dict:
    """The delivery note of the request, its job run to its end. Unanswered
    when the job did not end done."""
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
    its note}. Unanswered when the dock holds no vocabulary of a sha256
    named, or a job does not end done."""
    asking = vocabulary_request(data)
    if asking is None:
        key = vocabulary_key(data)
        if key is None:
            raise Unanswered(f"the dock holds no vocabulary {data['vocabulary']['sha256']}: name one an earlier "
                             "order learned, or put a depot's vocabulary on the dock with `wpt adopt-vocabularies`")
        vocabulary = api.delivery(key)
        say(f"vocabulary: delivery {key}, already on the dock")
    else:
        vocabulary = _asked(asking, "vocabulary")
    say(f"vocabulary {vocabulary['result']['sha256']}: {label(vocabulary)}")
    corpus = _asked(corpus_request(data, vocabulary["key"]), "corpus")
    say(f"corpus {corpus['id']} (delivery {corpus['key']}): {label(corpus)}, {corpus['result']['total_tokens']:,} "
        f"tokens of plain text, {corpus['items']['texts']:,} texts")
    return {"vocabulary": vocabulary, "corpus": corpus}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wpt build-resources", description=__doc__.split("\n\n")[0])
    parser.add_argument("order", help="a model recipe's name, or SERIES: for a series")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        data = Recipes.find().data_part(args.order)
        if "snapshot" not in data:
            raise Unanswered(f"{args.order} names a dataset, the legacy shape: the freight chooses in a snapshot")
        notes = build(data)
    except Exception as error:  # noqa: BLE001 a refusal of the order, the freight or the dock, said once
        say(f"wpt build-resources {args.order}: {type(error).__name__}: {error}", ERROR)
        return 1
    print(json.dumps(notes, indent=2, ensure_ascii=False))
    return 0
