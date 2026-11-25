"""The messages between the freight and a preparation, the program's
failures as the freight sees them, and the delivery key."""

import io
import json
import sys

import pytest

from freight import exchange, messages
from freight.exchange import Preparation, PreparationFailed
from freight.selection import Refused, delivery_key

SNAPSHOT = {"name": "s", "change": 12, "digest": "ab" * 32}
PREPARATION = {"name": "wpt-tokens", "version": "1", "settings": {}, "inputs": {"vocabulary": "adopted-x"}}


def test_a_message_and_its_content_read_back_as_written():
    stream = io.BytesIO()
    messages.write(stream, {"type": "take", "ref": 3}, "café\n".encode())
    messages.write(stream, {"type": "end"})
    stream.seek(0)
    assert messages.read(stream) == ({"type": "take", "ref": 3, "bytes": 6}, "café\n".encode())
    assert messages.read(stream) == ({"type": "end"}, None)
    assert messages.read(stream) is None


@pytest.mark.parametrize("written", [b"not json\n", b'{"no": "type"}\n', b'{"type": "take", "bytes": 9}\nshort'])
def test_a_line_that_is_no_message_is_refused(written):
    with pytest.raises(messages.BadMessage):
        messages.read(io.BytesIO(written))


def test_the_delivery_key_covers_what_makes_the_delivery_and_nothing_else():
    selection = {"kinds": ["plain"], "shares": [{"tag": "en", "share": 1.0}], "size": {"unit": "tokens", "amount": 9}}
    key = delivery_key(SNAPSHOT, selection, PREPARATION)
    assert key == delivery_key({**SNAPSHOT, "name": "another name"}, dict(reversed(selection.items())), PREPARATION)
    for changed in ({**selection, "size": {"unit": "tokens", "amount": 10}}, {**selection, "kinds": []},
                    {**selection, "shares": [{"tag": "de", "share": 1.0}]}):
        assert delivery_key(SNAPSHOT, changed, PREPARATION) != key
    assert delivery_key(SNAPSHOT, selection, {**PREPARATION, "version": "2"}) != key
    assert delivery_key({**SNAPSHOT, "change": 13}, selection, PREPARATION) != key
    with pytest.raises(Refused):
        delivery_key(SNAPSHOT, {"shares": [{"tag": "en", "share": 0}]}, PREPARATION)


def program(tmp_path, monkeypatch, body: str):
    """A preparation named `fake` whose program is the Python code body."""
    path = tmp_path / "fake"
    path.write_text(f"#!{sys.executable}\nimport json, sys\n{body}\n")
    path.chmod(0o755)
    monkeypatch.setattr(exchange, "find", lambda name: str(path))


def test_a_program_that_says_failed_fails_the_job_with_its_reason(tmp_path, monkeypatch):
    program(tmp_path, monkeypatch, 'sys.stdin.readline()\n'
                                   'print(json.dumps({"type": "failed", "reason": "no vocabulary"}), flush=True)')
    with pytest.raises(PreparationFailed, match="no vocabulary"):
        Preparation("fake", {}, {}, tmp_path / "out", 1)


def test_a_program_that_writes_no_message_or_ends_early_fails_the_job(tmp_path, monkeypatch):
    program(tmp_path, monkeypatch, 'sys.stdin.readline()\nprint("hello", flush=True)')
    with pytest.raises(PreparationFailed):
        Preparation("fake", {}, {}, tmp_path / "out", 1)
    ready = json.dumps({"type": "ready", "name": "fake", "version": "1", "unit": "tokens", "kinds": ["plain"],
                       "weighs": True})
    program(tmp_path, monkeypatch, f'sys.stdin.readline()\nprint({ready!r}, flush=True)')
    with Preparation("fake", {}, {}, tmp_path / "out", 1) as fake:
        with pytest.raises(PreparationFailed):
            fake.end()
