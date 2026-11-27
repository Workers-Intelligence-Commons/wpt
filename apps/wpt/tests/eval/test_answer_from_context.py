"""The items and the judging of the built-in test answer_from_context: items
made from templates with values of their own, an answer counted on the
first line the model writes, as a whole word, and refused when the line
names the other fact the text states."""

import importlib.util
import random

import pytest

from wpt.eval import BUILT_IN
from wpt_tokenizer import Tokenizer


@pytest.fixture(scope="module")
def answer_from_context():
    spec = importlib.util.spec_from_file_location("answer_from_context_under_test", BUILT_IN / "answer_from_context.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_format_gets_its_items_each_with_its_answer_in_the_text(answer_from_context):
    items = answer_from_context.items(random.Random(0))
    formats = [f"{language}, {name}" for language, found in answer_from_context.FORMATS.items() for name in found]
    assert [one["format"] for one in items] == [name for name in formats for _ in range(answer_from_context.PROMPTS)]
    for one in items:
        text, question = one["prompt"].split("\n\n")
        assert answer_from_context.named(text, one["answer"]) and "{" not in one["prompt"]
        assert not answer_from_context.named(question, one["answer"])  # the question does not give it away
        assert one["other"] is None or (one["other"] != one["answer"] and answer_from_context.named(text, one["other"]))
    assert len({one["prompt"] for one in items}) > 0.9 * len(items)  # values of their own, not a handful of items
    assert {one["kind"] for one in items} == {"city", "year", "job", "pet", "number", "color"}


@pytest.mark.parametrize("line, right", [
    (" Lyon.", True),
    (" Marie habite à lyon depuis dix ans.", True),  # case aside
    (" Lyonnais", False),  # a longer word
    (" Lyon ou Nantes", False),  # the other fact the text states
    (" Nantes", False),
    ("", False),
])
def test_the_answer_counts_as_a_whole_word_without_the_other_fact(answer_from_context, line, right):
    assert answer_from_context.answers(line, {"answer": "Lyon", "other": "Nantes"}) == right


def test_a_number_counts_only_as_itself(answer_from_context):
    item = {"answer": "1924", "other": None}
    assert answer_from_context.answers(" In 1924.", item)
    assert not answer_from_context.answers(" In 19245.", item) and not answer_from_context.answers(" 924", item)


def test_only_the_first_line_written_is_judged(answer_from_context):
    tokenizer = Tokenizer([])
    tokens = tokenizer.encode(b" Paris\nQuestion: Where? Answer: Lyon")
    assert answer_from_context.first_line(tokenizer, tokens) == " Paris"
    assert answer_from_context.first_line(tokenizer, tokenizer.encode(b" Lyon") + [Tokenizer.END, 76]) == " Lyon"
    stops = answer_from_context.line_breaks(tokenizer)
    assert Tokenizer.END in stops and tokenizer.encode(b"\n")[0] in stops and tokenizer.encode(b"a")[0] not in stops
