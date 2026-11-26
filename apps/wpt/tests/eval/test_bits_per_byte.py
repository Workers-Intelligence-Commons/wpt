"""The arithmetic of the built-in test bits_per_byte, on a stream small
enough to work out by hand: which texts it takes, the windows it cuts, the
tokens and bytes it counts, and the figures it gives."""

import importlib.util
import math

import numpy as np
import pytest

from wpt.eval import BUILT_IN
from wpt_tokenizer import Tokenizer

END = Tokenizer.END
BYTES = Tokenizer([])  # no merge: a token is a byte


@pytest.fixture(scope="module")
def bpb():
    spec = importlib.util.spec_from_file_location("bits_per_byte_under_test", BUILT_IN / "bits_per_byte.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(text: str, document: str, pool: str):
    return [*BYTES.encode(text.encode()), END], "s", document, pool, 0


def one_bit_each(windows):
    """Every token after a window's first costs 1 bit."""
    return [np.full(len(window) - 1, math.log(2), dtype=np.float32) for window in windows]


def test_bits_per_byte_counts_every_byte_of_the_texts_once_as_worked_out_by_hand(bpb):
    # in the order of their document ids, whatever the order of the rows; s/2 repeated comes once
    rows = [row("c€", "s/2", "de"), row("ab", "s/1", "en"), row("c€", "s/2", "de")]
    read = bpb.stream(rows, BYTES)
    # END, a b END, c and the three bytes of €, END
    assert read["tokens"].tolist() == [END, 97, 98, END, 99, 226, 130, 172, END]
    assert read["bytes"].tolist() == [0, 1, 1, 0, 1, 1, 1, 1, 0] and read["texts"] == 2
    # windows of block_size + 1 = 4 tokens from 0, 3, 6: each starts on the last token of the one before
    assert bpb.windows(9, 3) == [(0, 4), (3, 7), (6, 9)]
    data = bpb.measure(read, 3, one_bit_each)
    # 8 tokens have a loss (all but the first END), 1 bit each, over the 6 bytes of "ab" and "c€"
    assert (data["windows"], data["tokens"], data["bytes"]) == (3, 8, 6)
    assert data["bits_per_byte"] == pytest.approx(8 / 6)
    assert data["nats_per_token"] == pytest.approx(math.log(2))
    # window by window, bits over bytes: a b END 3/2, c E2 82 3/3, AC END 2/1
    assert data["bits_per_byte_spread"] == pytest.approx(float(np.std([1.5, 1.0, 2.0], ddof=1)))  # 0.5
    # sqrt(((3 - 8/6 * 2)^2 + (3 - 8/6 * 3)^2 + (2 - 8/6 * 1)^2) / (3 * 2)) / (6 / 3)
    assert data["bits_per_byte_error"] == pytest.approx(math.sqrt((1 / 9 + 1 + 4 / 9) / 6) / 2)
    assert data["by_share"] == {
        "de": {"tokens": 5, "bytes": 4, "bits_per_byte": pytest.approx(5 / 4), "nats_per_token": pytest.approx(math.log(2))},
        "en": {"tokens": 3, "bytes": 2, "bits_per_byte": pytest.approx(3 / 2), "nats_per_token": pytest.approx(math.log(2))}}


def test_the_losses_reach_the_tokens_they_stand_for(bpb):
    # each token's loss is its position in the stream, in bits: a token counted twice or in the wrong place shows
    read = bpb.stream([row("ab", "s/1", "en"), row("c€", "s/2", "en")], BYTES)

    def positions(windows):
        found, at = [], 0
        for window in windows:
            found.append(np.arange(at + 1, at + len(window)) * math.log(2))
            at += len(window) - 1
        return found

    data = bpb.measure(read, 3, positions)
    assert data["bits_per_byte"] == pytest.approx(sum(range(1, 9)) / 6)


def test_texts_past_the_most_bytes_are_left_out_whatever_their_tokens(bpb):
    rows = [row("abc", "s/1", "en"), row("defgh", "s/2", "en"), row("ij", "s/3", "en")]
    read = bpb.stream(rows, BYTES, max_bytes=6)
    # s/1 fits (3), s/2 would make 8 and is left out, s/3 fits (5)
    assert (read["texts"], read["left_out"], int(read["bytes"].sum())) == (2, 1, 5)
    merged = Tokenizer.train([b"abcdefghij" * 50], vocab_size=600)
    other = bpb.stream([([*merged.encode(text.encode()), END], *rest) for (_, *rest), text in
                        zip(rows, ("abc", "defgh", "ij"))], merged, max_bytes=6)
    assert len(other["tokens"]) < len(read["tokens"])
    assert (other["texts"], other["left_out"], int(other["bytes"].sum())) == (2, 1, 5)


def test_a_window_longer_than_the_stream_reads_it_whole_and_gives_no_error(bpb):
    read = bpb.stream([row("ab", "s/1", "en")], BYTES)
    data = bpb.measure(read, 16, one_bit_each)
    assert (data["windows"], data["tokens"], data["bytes"]) == (1, 3, 2)
    assert data["bits_per_byte"] == pytest.approx(3 / 2)
    assert data["bits_per_byte_error"] is None and data["bits_per_byte_spread"] is None
