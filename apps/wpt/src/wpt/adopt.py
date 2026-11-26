"""wpt adopt-vocabularies: put every vocabulary the depot's tokenizer
recipes trained, which the depot keeps unwritten in its tables
tokenizers_revisions, vocabularies and vocabulary_texts, on the dock,
each as a delivery of its own, so that an order naming a vocabulary by its
sha256 finds it (wpt.resources).

    wpt adopt-vocabularies    adopt each vocabulary of the depot DEPOT names onto the dock DOCK names

Each vocabulary becomes the delivery `adopted-<sha256>` (freight.api.adopt):
its file `vocabulary.json`, the vocabulary file as the depot keeps it; the
result a wpt-vocabulary delivery gives, `{sha256, size,
tokenizer_version}`; what the depot recorded of it (`adopted`): the depot,
its tokenizer recipe's name and version, the dataset it named, its sample
(the snapshot and kinds) and sample bytes, the texts it read and their
bytes, and when it was made; and as its items table the texts it read, in
the order it read them, each `{depot, id, change}`, the depot by its name
for one of its own texts. An erasure that reaches one of those texts is
written into its note and leaves it fresh (`on_erasure` `note`,
docs/tokenizer.md). A vocabulary the dock holds already, fresh, stays as
it is, so the command may run again. It reads the depot without writing to
it and prints each delivery's key. Exit codes: 0 done, 1 refused.

It is the one module of WPT that reads a depot, through depot.api
(vocabularies, vocabulary_texts), since what it adopts is kept nowhere
else (docs/architecture.md, "The parts").
"""

import argparse
import json
import os
import sys
from pathlib import Path

from depot import api as depot_api
from freight import api
from wic_utils.env import read_env_file
from wic_utils.lines import ERROR, say
from wpt_tokenizer import Tokenizer

PAGE = 100_000  # the texts of a vocabulary read at a time
FILE = "vocabulary.json"
SETTING = "DEPOT"


def vocabularies(depot: str):
    """(files, result, about, items) of each vocabulary the depot at that address holds, as adopt takes them."""
    own = depot_api.about(depot)["name"]
    for row in depot_api.vocabularies(depot):
        merges = row["merges"].encode("utf-8")
        items, offset = [], 0
        while page := depot_api.vocabulary_texts(depot, row["recipe"], row["version"], offset, PAGE):
            items += page
            offset += len(page)
        result = {"sha256": row["sha256"], "size": Tokenizer.from_bytes(merges).vocab_size,
                  "tokenizer_version": json.loads(merges).get("tokenizer_version")}
        about = {"by": "wpt adopt-vocabularies", "depot": own, "recipe": row["name"], "version": row["version"],
                 "dataset": row["dataset"], "sample": row["sample"], "sample_bytes": row["sample_bytes"],
                 "texts": row["texts"], "bytes": row["bytes"], "made_at": row["made_at"]}
        yield {FILE: merges}, result, about, items


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wpt adopt-vocabularies", description=__doc__.split("\n\n")[0])
    parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        depot = os.environ.get(SETTING) or read_env_file(Path.cwd() / ".env").get(SETTING)
        if not depot:
            raise ValueError(f"set {SETTING} to the depot's file, in the environment or in .env")
        for files, result, about, items in vocabularies(depot):
            note = api.adopt(f"{api.ADOPTED}{result['sha256']}", files, result, about, on_erasure="note", items=items)
            say(f"vocabulary {result['sha256']} of the tokenizer recipe {about['recipe']} version "
                f"{about['version']}: delivery {note['key']}, {len(items):,} texts")
    except Exception as error:  # noqa: BLE001 a refusal of the depot or the dock, said once
        say(f"wpt adopt-vocabularies: {type(error).__name__}: {error}", ERROR)
        return 1
    return 0
