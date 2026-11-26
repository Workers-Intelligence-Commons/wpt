"""The notice of a model trained on a corpus, written from the corpus's
tables alone: one line per text it learned from, in the corpus's order,
a text taken as pieces once, where the first piece it learned from comes,

    <depot> <id>: <title> by <authors>, <address>; <licenses, joined by OR>; notice <n>

each part present only when the text has it (a text without a title is
"untitled"), a line break inside a credit written as a space; then each
notice the lines name, under its number, and the text of each license
they name. A text's licenses are a choice: any one of them allows it.

What a model learned from is what its training steps read (`learned`, each
step's, added up, by kind of texts: wpt_training.Environment.learned): the
plain texts and pieces whose first token lies in the first `plain` tokens
of the plain texts' token files, and every text of another kind when a
step read texts of that kind. Without `learned`, every text of the corpus.
A conversation learned from also credits every text it quotes (its row's
`quotes`: the texts of its depot whose passages its excerpts put in its
tokens), each under its own credits, licenses and notice.
"""

from wpt_corpus import corpora

PLAIN = "plain"


def model_notice(dock, record: dict, learned: dict | None = None) -> str:
    """The notice of a model trained on the corpus whose record is given,
    learning what `learned` says ({"plain": tokens, "conversation": count,
    "task": count}, each part optional)."""
    rows = corpora.rows(dock, record, "texts")
    quoted = {(one["depot"], text) for one in rows if one.get("quotes") and _learned(one, learned)
              for text in one["quotes"]}
    texts, seen = [], set()
    for one in rows:
        if (_learned(one, learned) or (one["depot"], one["id"]) in quoted) and (one["depot"], one["id"]) not in seen:
            seen.add((one["depot"], one["id"]))
            texts.append(one)
    lines = [_line(one) for one in texts]
    named = {one["notice"] for one in texts}
    notices = [one for one in corpora.rows(dock, record, "notices") if one["id"] in named]
    used = {license_id for one in texts for license_id in one["licenses"]}
    licenses = [one for one in corpora.rows(dock, record, "licenses") if one["id"] in used]
    parts = ["\n".join(lines)]
    if notices:
        parts.append("Notices\n\n" + "\n\n".join(f"[{one['id']}]\n{one['text']}" for one in notices))
    if licenses:
        parts.append("Licenses\n\n" + "\n\n".join(f"{one['id']} ({one['name']})\n{one['text']}" for one in licenses))
    return "\n\n".join(parts) + "\n"


def _learned(text: dict, learned: dict | None) -> bool:
    if learned is None:
        return True
    if text["kind"] == PLAIN:
        return text["tokens"] < learned.get(PLAIN, 0)
    return bool(learned.get(text["kind"], 0))


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
