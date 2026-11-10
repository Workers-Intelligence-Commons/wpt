"""The notice of a model trained on a corpus, written from the corpus's
tables alone: one line per text it learned from, in the corpus's order,

    <depot> <id>: <title> by <authors>, <address>; <licenses, joined by OR>; notice <n>

each part present only when the text has it (a text without a title is
"untitled"), a line break inside a credit written as a space; then each
notice the lines name, under its number, and the text of each license
they name. A text's licenses are a choice: any one of them allows it.

What a model learned from is what its training steps read (`learned`, each
step's, added up: wpt_training.Environment.learned): the plain texts whose
first token lies in the first `text` tokens of the plain texts' token
files, every conversation when a step read conversations, every task when a
step read tasks. Without `learned`, every text of the corpus.
"""

from .corpus import read_table


def model_notice(store, record: dict, learned: dict | None = None) -> str:
    """The notice of a model trained on the corpus whose record is given,
    learning what `learned` says ({"text": tokens, "conversations": count,
    "tasks": count}, each part optional)."""
    texts = [one for one in read_table(store, record, "texts").to_pylist() if _learned(one, learned)]
    lines = [_line(one) for one in texts]
    named = {one["notice"] for one in texts}
    notices = [one for one in read_table(store, record, "notices").to_pylist() if one["id"] in named]
    used = {license_id for one in texts for license_id in one["licenses"]}
    licenses = [one for one in read_table(store, record, "licenses").to_pylist() if one["id"] in used]
    parts = ["\n".join(lines)]
    if notices:
        parts.append("Notices\n\n" + "\n\n".join(f"[{one['id']}]\n{one['text']}" for one in notices))
    if licenses:
        parts.append("Licenses\n\n" + "\n\n".join(f"{one['id']} ({one['name']})\n{one['text']}" for one in licenses))
    return "\n\n".join(parts) + "\n"


def _learned(text: dict, learned: dict | None) -> bool:
    if learned is None:
        return True
    if text["kind"] is None:
        return text["tokens"] < learned.get("text", 0)
    return bool(learned.get({"conversation": "conversations", "task": "tasks"}.get(text["kind"], ""), 0))


def _line(text: dict) -> str:
    credit = _flat(text["title"]) if text["title"] else "untitled"
    if text["authors"]:
        credit += " by " + ", ".join(map(_flat, text["authors"]))
    if text["address"]:
        credit += f", {_flat(text['address'])}"
    line = f"{text['depot']} {text['id']}: {credit}; {' OR '.join(text['licenses'])}"
    return line + (f"; notice {text['notice']}" if text["notice"] is not None else "")


def _flat(value: str) -> str:
    return " ".join(value.splitlines())
