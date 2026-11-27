"""The measures of the built-in test free_text: the share of repeated runs
of tokens, the place where a text first goes over itself again, and the
texts shown as written in the model card."""

import importlib.util

import pytest

from wpt.eval import BUILT_IN


@pytest.fixture(scope="module")
def free_text():
    spec = importlib.util.spec_from_file_location("free_text_under_test", BUILT_IN / "free_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_text_that_never_repeats_a_run_has_no_repeat(free_text):
    tokens = list(range(50))
    assert free_text.repeated(tokens) == 0 and free_text.first_repeat(tokens) is None


def test_a_text_caught_in_a_loop_repeats_almost_every_run(free_text):
    tokens = list(range(10)) * 5  # 50 tokens, the same 10 again and again
    assert free_text.repeated(tokens, 4) == 37 / 47  # every run of 4 but the 10 that start in the first loop
    assert free_text.first_repeat(tokens, 8) == 18  # the run of tokens 10 to 17 is that of 0 to 7


def test_a_run_counts_once_seen_even_when_it_overlaps_the_earlier_one(free_text):
    assert free_text.first_repeat([7] * 9, 8) == 9
    assert free_text.repeated([7] * 9, 4) == 5 / 6


def test_a_text_shorter_than_a_run_has_nothing_to_repeat(free_text):
    assert free_text.repeated([1, 2], 4) == 0 and free_text.first_repeat([1, 2], 8) is None


def test_a_text_is_fenced_by_more_backticks_than_it_holds(free_text):
    assert free_text.fenced("plain") == "```\nplain\n```"
    assert free_text.fenced("a ```` b").startswith("`````\n")
