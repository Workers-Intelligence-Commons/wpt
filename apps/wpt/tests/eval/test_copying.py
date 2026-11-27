"""The judging of the built-in test copying, on a value stated in a sentence:
what the model writes counts by its text, whatever its tokens, and the four
values are scored as the prompt cuts the stated one, period included."""

import importlib.util

import numpy as np
import pytest

from wpt.eval import BUILT_IN
from wpt_tokenizer import Tokenizer

FIRST = 513  # the id of the first merge
# "91" and then "91." are merges: the period joins the value's last digits, as a trained vocabulary joins them
MERGES = [(ord("9"), ord("1")), (FIRST, ord("."))]
SENTENCE = "The code is{}. I repeat, the code is"
VALUES = [b" 7391", b" 2468", b" 1357", b" 8024"]


@pytest.fixture(scope="module")
def copying():
    spec = importlib.util.spec_from_file_location("copying_under_test", BUILT_IN / "copying.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Writes:
    """A model that writes the given tokens after every prompt, and costs
    every token the same: it records the windows it scores."""

    def __init__(self, tokens):
        self.tokens, self.scored = tokens, []

    def greedy(self, prompts, steps):
        return [self.tokens[:steps] for _ in prompts]

    def losses(self, windows):
        self.scored += windows
        return [np.ones(len(window) - 1, np.float32) for window in windows]

    def ranks(self, windows):
        return [np.ones(len(window) - 1, np.int32) for window in windows]


class Context:
    def __init__(self, model):
        self.model, self.tokenizer = model, Tokenizer(MERGES)


def written(copying, text: bytes) -> float:
    """The share written when the model writes text after the sentence that states 7391."""
    tokenizer = Tokenizer(MERGES)
    return copying.stated(Context(Writes(tokenizer.encode(text))), [SENTENCE], [VALUES])["written"]


def test_the_value_with_the_period_joined_to_its_last_digits_is_written(copying):
    tokenizer = Tokenizer(MERGES)
    assert tokenizer.encode(b" 7391.")[-1] != tokenizer.encode(b" 7391")[-1]  # other tokens than the value alone
    assert written(copying, b" 7391. I repeat") == 1


def test_the_value_cut_into_other_tokens_is_written(copying):
    tokenizer = Tokenizer(MERGES)
    pieces = [token for byte in b" 7391" for token in tokenizer.encode(bytes([byte]))]  # one token a character
    assert pieces != tokenizer.encode(b" 7391")
    assert copying.stated(Context(Writes(pieces)), [SENTENCE], [VALUES])["written"] == 1


@pytest.mark.parametrize("text, right", [
    (b" 7391", True),  # the text ends with the value
    (b" 7391\n", True),
    (b"7391.", True),  # no space before it: leading spaces are set aside
    (b"   7391,", True),
    (b" 73912.", False),  # a longer number that starts with the value
    (b" 739", False),  # the value cut short
    (b" 7392.", False),
    (b"\n7391.", False),  # only spaces are set aside
])
def test_the_text_written_is_the_value_only_when_it_ends_there(copying, text, right):
    assert written(copying, text) == right


@pytest.mark.parametrize("text, value, right", [
    (" pomme", " pom", False),  # a longer word that starts with the value
    (" Pom.", " pom", False),  # a capital is another text
    ("pom.", " pom", True),
    (" pom's", " pom", True),
    (" pomé", " pom", False),  # a letter outside ASCII is a letter
])
def test_a_word_is_written_only_when_it_ends_there(copying, text, value, right):
    assert copying.writes(text, value) == right


def test_the_four_values_are_scored_followed_by_the_period_cut_as_in_the_prompt(copying):
    tokenizer, model = Tokenizer(MERGES), Writes([])
    copying.stated(Context(model), [SENTENCE], [VALUES])
    prompt = [Tokenizer.END] + tokenizer.encode(SENTENCE.format(" 7391").encode())
    start = 1 + len(tokenizer.encode(b"The code is"))
    stated = prompt[start:start + len(tokenizer.encode(b" 7391."))]
    assert tokenizer.decode(stated) == b" 7391."
    assert model.scored[0] == prompt + stated
    assert [tokenizer.decode(window[len(prompt):]) for window in model.scored] == [value + b"." for value in VALUES]
