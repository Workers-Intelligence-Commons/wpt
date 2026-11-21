"""General knowledge: does the model know what everyone knows?

Each question of general_knowledge.json is a sentence to complete with one
answer that leaves no doubt ("The capital of Italy is" Rome), asked in
French and in English, with three wrong answers of the same kind. A
document starts before the sentence (Tokenizer.END), as wpt chat starts
one. Two measures, for each language and each topic:

- choice: the answer is the one of the four that costs the model the
  fewest nats after the sentence (the sum over its tokens); chance is 1 in
  4;
- written: the model's likeliest WRITTEN tokens after the sentence start
  with the answer, case and the spaces before it aside.

The answer follows the sentence after a space, except after an apostrophe
("l'" automne).
"""

import json
from pathlib import Path

from wpt_tokenizer import Tokenizer
from wpt.eval import Result

QUESTIONS = Path(__file__).with_suffix(".json")
LANGUAGES = ("fr", "en")
WRITTEN = 8  # tokens the model writes after the sentence


def joined(prompt: str, answer: str) -> str:
    """The answer as it follows the prompt: after a space, or right after an apostrophe."""
    return answer if prompt.endswith(("'", "’")) else " " + answer


def asked(context, questions: list[dict]) -> list[dict]:
    """Each question's answer: whether the right one costs the fewest nats
    of the four, and whether the model writes it. The model is asked once
    for every cost, and once for the writing of the sentences of each
    length (greedy reads rows of one length)."""
    tokenizer = context.tokenizer
    prompts = [[Tokenizer.END] + tokenizer.encode(one["prompt"].encode()) for one in questions]
    windows = [ids + tokenizer.encode(joined(one["prompt"], answer).encode())
               for ids, one in zip(prompts, questions) for answer in [one["answer"]] + one["wrong"]]
    found = context.model.losses(windows)
    lengths = {}
    for number, ids in enumerate(prompts):
        lengths.setdefault(len(ids), []).append(number)
    written = {}
    for numbers in lengths.values():
        written |= dict(zip(numbers, context.model.greedy([prompts[number] for number in numbers], WRITTEN)))
    answers = []
    for number, (ids, one) in enumerate(zip(prompts, questions)):
        costs = [float(found[4 * number + choice][len(ids) - 1:].sum()) for choice in range(4)]
        text = tokenizer.decode(written[number]).decode("utf-8", errors="replace")
        answers.append({"prompt": one["prompt"], "answer": one["answer"], "chosen": costs.index(min(costs)) == 0,
                        "written": text.strip().lower().startswith(one["answer"].lower()), "text": text})
    return answers


def share(found: list[dict], key: str) -> float:
    return sum(one[key] for one in found) / len(found) if found else 0.0


def run(context) -> Result:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    answers = {language: [dict(answer, topic=one["topic"])
                          for answer, one in zip(asked(context, [one[language] for one in questions]), questions)]
               for language in LANGUAGES}
    topics = list(dict.fromkeys(one["topic"] for one in questions))
    data = {"questions": len(questions), "chance": 0.25, "languages": {}}
    lines = [f"{len(questions)} questions, each in French and in English. choice: the right answer is the "
             "likeliest of four (chance 25%); written: the model writes it.", "",
             "| | " + " | ".join(f"{language} choice | {language} written" for language in LANGUAGES) + " |",
             "| --- |" + " --- | --- |" * len(LANGUAGES)]
    for topic in [None] + topics:
        row = []
        for language in LANGUAGES:
            found = [one for one in answers[language] if topic is None or one["topic"] == topic]
            figures = {"choice": share(found, "chosen"), "written": share(found, "written")}
            data["languages"].setdefault(language, {})[topic or "all"] = figures
            row += [f"{figures['choice']:.0%}", f"{figures['written']:.0%}"]
        lines.append(f"| {topic or '**all**'} | " + " | ".join(row) + " |")
    data["answers"] = answers
    return Result(data, "\n".join(lines))
