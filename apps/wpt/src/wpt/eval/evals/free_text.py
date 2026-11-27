"""Free text: what the model writes from an opening, and where it starts
repeating itself.

From each opening of OPENINGS, in French and in English, the model writes
ROWS texts of up to WRITTEN tokens, through its generation cache
(wpt_inference.Writer, on the gpu worker), with the sampling wpt chat uses
(wpt_inference.TEMPERATURE and TOP_K) and a generator seeded with SEED, so
the same weights write the same texts again on the same machine. A
document starts before the opening (Tokenizer.END), as wpt chat starts
one, and a text ends early where the model ends the document. Measures,
on the tokens written, for each text and then for each language:

- repeated: the share of the text's SHORT-token runs (each run of SHORT
  tokens in a row) that already appeared earlier in the text; text a
  person writes repeats few, a model caught in a loop most;
- first_repeat: the place, counted in tokens written from 1, of the last
  token of the first run of LONG tokens in a row that repeats an earlier
  one, where the text starts going over itself again; None when none
  does. A language gives the share of its texts that repeat such a run
  (repeating) and the median place among them.

The model card shows the first text written from each opening, for a
person to read.
"""

import statistics

from wpt_inference import TEMPERATURE, TOP_K
from wpt_tokenizer import Tokenizer
from wpt.eval import Result

SEED = 0
ROWS = 4  # texts written from each opening, side by side
WRITTEN = 200  # the most tokens of each text
SHORT, LONG = 4, 8  # the runs of tokens that repeated and first_repeat count
OPENINGS = {
    "fr": ["Il était une fois, dans un petit village,",
           "L'histoire de la ville commence au",
           "La photosynthèse est le processus par lequel les plantes transforment la lumière du soleil en énergie. "
           "Elle se déroule principalement dans les feuilles, où"],
    "en": ["Once upon a time, in a small village,",
           "The history of the city begins in",
           "The city council met on Tuesday to discuss the new bridge. The mayor explained that the old bridge, "
           "built in 1924, could no longer carry heavy trucks. Engineers proposed two options:"],
}


def runs(tokens: list[int], size: int) -> list[tuple]:
    return [tuple(tokens[start:start + size]) for start in range(len(tokens) - size + 1)]


def repeated(tokens: list[int], size: int = SHORT) -> float:
    """The share of the runs of size tokens in tokens that appeared earlier
    in them (0 when there is no run)."""
    seen, again = set(), 0
    found = runs(tokens, size)
    for run in found:
        again += run in seen
        seen.add(run)
    return again / len(found) if found else 0.0


def first_repeat(tokens: list[int], size: int = LONG) -> int | None:
    """The place, from 1, of the last token of the first run of size tokens
    that appeared earlier in tokens (overlapping it or not); None when none
    did."""
    seen = set()
    for start, run in enumerate(runs(tokens, size)):
        if run in seen:
            return start + size
        seen.add(run)
    return None


def fenced(text: str) -> str:
    """text as a block of code in Markdown, fenced by more backticks than
    any run of them inside it, so the model card shows it as written."""
    longest, current = 0, 0
    for character in text:
        current = current + 1 if character == "`" else 0
        longest = max(longest, current)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def written(context, opening: str, generator: int) -> list[list[int]]:
    """The ROWS texts the model writes after opening, each up to the end of
    a document (left out) or WRITTEN tokens."""
    prompt = [Tokenizer.END] + context.tokenizer.encode(opening.encode())
    writer = context.model.writer(prompt, ROWS, temperature=TEMPERATURE, top_k=TOP_K, stops=[Tokenizer.END],
                                  generator=generator)
    found = writer.write({}, {row: WRITTEN for row in range(ROWS)})
    return [[token for token in found[row] if token != Tokenizer.END] for row in range(ROWS)]


def run(context) -> Result:
    generator = context.model.generator(SEED)
    data = {"seed": SEED, "rows": ROWS, "written": WRITTEN, "temperature": TEMPERATURE, "top_k": TOP_K,
            "short": SHORT, "long": LONG, "languages": {}}
    lines = [f"From each opening, {ROWS} texts of up to {WRITTEN} tokens, sampled as wpt chat samples "
             f"(temperature {TEMPERATURE}, the {TOP_K} likeliest tokens; seed {SEED}). repeated: the share of runs "
             f"of {SHORT} tokens that already appeared earlier in their text; repeating: the share of texts where a "
             f"run of {LONG} tokens comes back, and first repeat: the median token where it first does, among them.",
             "", "| | tokens | repeated | repeating | first repeat |", "| --- | --- | --- | --- | --- |"]
    shown = []
    for language, openings in OPENINGS.items():
        texts = []
        for opening in openings:
            for tokens in written(context, opening, generator):
                texts.append({"opening": opening, "text": context.tokenizer.decode(tokens).decode("utf-8", "replace"),
                              "tokens": len(tokens), "repeated": repeated(tokens),
                              "first_repeat": first_repeat(tokens)})
            shown.append(texts[-ROWS])
        places = [one["first_repeat"] for one in texts if one["first_repeat"] is not None]
        figures = {"tokens":statistics.fmean(one["tokens"] for one in texts),
                   "repeated": statistics.fmean(one["repeated"] for one in texts),
                   "repeating": len(places) / len(texts),
                   "first_repeat": statistics.median(places) if places else None}
        data["languages"][language] = dict(figures, texts=texts)
        first = "none" if figures["first_repeat"] is None else f"{figures['first_repeat']:g}"
        lines.append(f"| {language} | {figures['tokens']:.0f} | {figures['repeated']:.0%} | "
                     f"{figures['repeating']:.0%} | {first} |")
    lines += ["", "The first text written from each opening, the opening first:"]
    for one in shown:
        lines += ["", fenced(one["opening"] + one["text"])]
    return Result(data, "\n".join(lines))
