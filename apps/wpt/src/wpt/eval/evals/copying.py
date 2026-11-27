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
  is"), in English and in French. The tokenizer may join the period after
  the value to its last letters or digits (" 73" "91."), so the value is
  judged as text, never by its tokens. choice: the stated value followed
  by a period costs the model fewer nats than 3 other values of the same
  kind followed by a period (chance is 1 in 4); written: the text of the
  model's likeliest tokens after the prompt, leading spaces aside, is the
  value, then the end of the text or a character that is neither a letter
  nor a digit; first_rank and first_probability: the median rank (1: the
  likeliest) and the mean probability of the first token of the value
  followed by a period, at the place where the answer starts.
"""

import math
import random
import statistics

from wpt_tokenizer import Tokenizer
from wpt.eval import Result

SEED = 0
PROMPTS = 200  # of each measure, and of each kind of stated value
SEQUENCE = 8  # words of a sequence
MORE = 4  # tokens written past the longest stated value: what follows it decides whether it ended
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


def writes(text: str, value: str) -> bool:
    """Whether text, leading spaces aside, is value (leading spaces aside)
    ended there: by the end of the text or by a character that is neither a
    letter nor a digit."""
    text, value = text.lstrip(" "), value.lstrip(" ")
    return text.startswith(value) and not text[len(value):len(value) + 1].isalnum()


def stated(context, sentences: list[str], values: list[list[bytes]]) -> dict:
    """values: for each prompt, the stated value then 3 others of its kind,
    each as it follows the sentence (a space first). Every sentence ends the
    value with a period, which the tokenizer may join to the value's last
    letters or digits, so each value is scored followed by that period, cut
    as the prompt cuts the stated one."""
    tokenizer, model = context.tokenizer, context.model
    prompts, windows, longest = [], [], 0
    for number, four in enumerate(values):
        sentence = sentences[number % len(sentences)]
        prompt = [Tokenizer.END] + tokenizer.encode(sentence.format(four[0].decode()).encode())
        prompts.append(prompt)
        ended = [tokenizer.encode(value + b".") for value in four]
        longest = max(longest, *map(len, ended))
        windows += [prompt + one for one in ended]
    costs = model.losses(windows)
    ranks = model.ranks(windows[::4])
    written = written_after(model, prompts, longest + MORE)
    chosen = right = 0
    first_ranks, first_probabilities = [], []
    for number, prompt in enumerate(prompts):
        four = [float(costs[4 * number + choice][len(prompt) - 1:].sum()) for choice in range(4)]
        chosen += four.index(min(four)) == 0
        right += writes(tokenizer.decode(written[number]).decode("utf-8", "replace"), values[number][0].decode())
        first_ranks.append(int(ranks[number][len(prompt) - 1]))
        first_probabilities.append(math.exp(-float(costs[4 * number][len(prompt) - 1])))
    return {"prompts": len(prompts), "choice": chosen / len(prompts), "written": right / len(prompts),
            "first_rank": statistics.median(first_ranks), "first_probability": statistics.fmean(first_probabilities)}


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
             f"A value stated in a sentence and asked for in the next, {PROMPTS} prompts of each kind. choice: the "
             "stated value, followed by the sentence's period, is the likeliest of four (chance 25%); written: the "
             "text the model writes is the value, ended there; first token: the median rank (1: the likeliest) and "
             "the mean probability of the value's first token, as the prompt cuts it, where the answer starts.", "",
             "| | choice | written | first token's rank | first token's probability |",
             "| --- | --- | --- | --- | --- |"]
    data["stated"] = {}
    for language, sentences in SENTENCES.items():
        kinds = {"word": [[context.tokenizer.decode([token]) for token in draw.sample(words, 4)]
                          for _ in range(PROMPTS)],
                 "number": [[f" {number}".encode() for number in draw.sample(range(1000, 10000), 4)]
                            for _ in range(PROMPTS)]}
        data["stated"][language] = found = {kind: stated(context, sentences, values) for kind, values in kinds.items()}
        lines += [f"| {language}, {kind} | {one['choice']:.0%} | {one['written']:.0%} | {one['first_rank']:g} | "
                  f"{one['first_probability']:.3f} |" for kind, one in found.items()]
    return Result(data, "\n".join(lines))
