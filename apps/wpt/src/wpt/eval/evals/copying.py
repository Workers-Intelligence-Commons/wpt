"""Copying: does the model reuse what its context says?

Every prompt is made here from a seed, so no prompt is in the corpus, and
the answer is in the prompt alone. A document starts before each one
(Tokenizer.END), as wpt chat starts one. The words are tokens of the
model's vocabulary that are one whole word (a space, then 3 to 10 lowercase
letters), drawn at random. Two measures:

- sequence: SEQUENCE words in a random order, written twice, then the first
  half a third time; the model writes the second half. tokens: the share of
  the written tokens that are the sequence's; whole: the share of prompts
  where all of them are. first and repeated: the mean cost in nats of a
  word of the sequence the first time and the third time it is read; with
  no copying the two are equal.
- stated: a sentence states a value, a word or a number of 4 digits, and
  the next sentence asks for it ("The code is 7391. I repeat, the code
  is"), in English and in French. choice: the stated value costs the model
  fewer nats than 3 other values of the same kind (chance is 1 in 4);
  written: the model's likeliest tokens after the prompt are the value.
"""

import random

from wpt_tokenizer import Tokenizer
from wpt.eval import Result

SEED = 0
PROMPTS = 200  # of each measure, and of each kind of stated value
SEQUENCE = 8  # words of a sequence
FEWEST_WORDS = 16  # whole-word tokens the vocabulary must hold for a measure to mean something
SENTENCES = {
    "en": ["The code is{}. I repeat, the code is", "My name is{}. As I said, my name is",
           "The answer is{}. Once more, the answer is", "The word of the day is{}. Again, the word of the day is"],
    "fr": ["Le code est{}. Je répète, le code est", "Mon nom est{}. Comme je l'ai dit, mon nom est",
           "La réponse est{}. Encore une fois, la réponse est", "Le mot du jour est{}. Je redis, le mot du jour est"],
}


def whole_words(tokenizer) -> list[int]:
    """The tokens that are one whole word: a space, then 3 to 10 lowercase letters."""
    found = []
    for token in range(tokenizer.vocab_size):
        if token == Tokenizer.END:
            continue
        text = tokenizer.decode([token])
        if text[:1] == b" " and 3 <= len(text) - 1 <= 10 and text[1:].isalpha() and text[1:].islower() \
                and text.isascii():
            found.append(token)
    return found


def written_after(model, prompts: list[list[int]], steps: int) -> list[list[int]]:
    """The tokens the model writes after each prompt; greedy reads rows of
    one length, so the prompts of each length are asked together."""
    lengths = {}
    for number, ids in enumerate(prompts):
        lengths.setdefault(len(ids), []).append(number)
    found = {}
    for numbers in lengths.values():
        found |= dict(zip(numbers, model.greedy([prompts[number] for number in numbers], steps)))
    return [found[number] for number in range(len(prompts))]


def sequences(context, words: list[int], draw) -> dict:
    half = SEQUENCE // 2
    drawn = [draw.sample(words, SEQUENCE) for _ in range(PROMPTS)]
    written = context.model.greedy([[Tokenizer.END] + one * 2 + one[:half] for one in drawn], SEQUENCE - half)
    right = [sum(a == b for a, b in zip(tokens, one[half:])) for tokens, one in zip(written, drawn)]
    costs = context.model.losses([[Tokenizer.END] + one * 3 for one in drawn])
    # the loss at index i is the cost of the window's token i + 1: the first copy's are 0 to SEQUENCE - 1
    first = sum(float(cost[:SEQUENCE].mean()) for cost in costs) / PROMPTS
    repeated = sum(float(cost[2 * SEQUENCE:].mean()) for cost in costs) / PROMPTS
    return {"prompts": PROMPTS, "tokens": sum(right) / (PROMPTS * (SEQUENCE - half)),
            "whole": sum(count == SEQUENCE - half for count in right) / PROMPTS,
            "first_nats": first, "repeated_nats": repeated}


def stated(context, sentences: list[str], values: list[list[bytes]]) -> dict:
    """values: for each prompt, the stated value then 3 others of its kind,
    each as it follows the sentence (a space first)."""
    tokenizer = context.tokenizer
    prompts, windows, answers = [], [], []
    for number, four in enumerate(values):
        sentence = sentences[number % len(sentences)]
        prompt = [Tokenizer.END] + tokenizer.encode(sentence.format(four[0].decode()).encode())
        prompts.append(prompt)
        answers.append(tokenizer.encode(four[0]))
        windows += [prompt + tokenizer.encode(value) for value in four]
    costs = context.model.losses(windows)
    written = written_after(context.model, prompts, max(len(answer) for answer in answers))
    chosen = right = 0
    for number, (prompt, answer) in enumerate(zip(prompts, answers)):
        four = [float(costs[4 * number + choice][len(prompt) - 1:].sum()) for choice in range(4)]
        chosen += four.index(min(four)) == 0
        right += written[number][:len(answer)] == answer
    return {"prompts": len(prompts), "choice": chosen / len(prompts), "written": right / len(prompts)}


def run(context) -> Result:
    draw = random.Random(SEED)
    words = whole_words(context.tokenizer)
    data = {"seed": SEED, "words": len(words), "chance": 0.25}
    if len(words) < FEWEST_WORDS:
        return Result(data, f"The vocabulary holds {len(words)} tokens that are one whole word, fewer than the "
                            f"{FEWEST_WORDS} the test needs: nothing measured.")
    data["sequence"] = found = sequences(context, words, draw)
    lines = [f"{found['prompts']} sequences of {SEQUENCE} words drawn from {len(words):,}, written twice, then "
             f"their first half: the model writes {found['tokens']:.0%} of the second half's tokens, and the "
             f"whole of it for {found['whole']:.0%} of the sequences. A word of a sequence costs "
             f"{found['first_nats']:.2f} nats the first time it is read and {found['repeated_nats']:.2f} the "
             "third time.", "",
             "A value stated in a sentence and asked for in the next. choice: the stated value is the likeliest "
             "of four (chance 25%); written: the model writes it.", "",
             "| | word choice | word written | number choice | number written |", "| --- | --- | --- | --- | --- |"]
    data["stated"] = {}
    for language, sentences in SENTENCES.items():
        kinds = {"word": [[context.tokenizer.decode([token]) for token in draw.sample(words, 4)]
                          for _ in range(PROMPTS)],
                 "number": [[f" {number}".encode() for number in draw.sample(range(1000, 10000), 4)]
                            for _ in range(PROMPTS)]}
        data["stated"][language] = found = {kind: stated(context, sentences, values) for kind, values in kinds.items()}
        lines.append(f"| {language} | " + " | ".join(f"{found[kind][key]:.0%}" for kind in kinds
                                                     for key in ("choice", "written")) + " |")
    return Result(data, "\n".join(lines))
