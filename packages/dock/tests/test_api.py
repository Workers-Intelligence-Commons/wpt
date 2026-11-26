"""dock.api as its callers use it: the dock named by its URL, values and
bytes in and out (docs/api/dock.md)."""

import hashlib
import json
import os

import boto3
import pytest
from moto import mock_aws

from dock import api, open_dock

MIB = 1 << 20


@pytest.fixture
def dock(tmp_path):
    (tmp_path / "dock").mkdir()
    return (tmp_path / "dock").as_uri()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_files_are_found_fetched_and_checked_by_their_key(dock):
    text = b"a line of text\n" * 10_000  # kept compressed
    noise = os.urandom(3000)  # kept as it is
    assert api.put(dock, "corpora/c/text", text) == {"key": "corpora/c/text", "sha256": sha(text), "bytes": len(text)}
    api.put(dock, "corpora/c/noise", noise)
    assert [one["key"] for one in api.find(dock, "corpora/")] == ["corpora/c/noise", "corpora/c/text"]
    assert api.folders(dock, "corpora/") == ["c"]
    assert api.has(dock, "corpora/c/text") and not api.has(dock, "corpora/c/none")
    assert api.size(dock, "corpora/c/text") == len(text) and api.size(dock, "corpora/c/noise") == len(noise)
    assert api.fetch(dock, "corpora/c/text", sha(text)) == text
    assert api.fetch_range(dock, "corpora/c/text", 15, 30) == text[15:45]
    assert api.fetch_range(dock, "corpora/c/noise", 100, 50) == noise[100:150]
    assert api.check(dock, "corpora/c/text", sha(text)) and not api.check(dock, "corpora/c/text", sha(noise))
    with pytest.raises(api.CorruptFile):
        api.fetch(dock, "corpora/c/text", sha(noise))
    with pytest.raises(api.MissingFile):
        api.fetch(dock, "corpora/c/none")
    with pytest.raises(api.MissingFile):
        api.size(dock, "corpora/c/none")
    with pytest.raises(api.MissingFile):
        api.check(dock, "corpora/c/none", sha(text))
    api.delete(dock, "corpora/c/noise")
    assert not api.has(dock, "corpora/c/noise")


@pytest.mark.parametrize("compressible", [True, False])
def test_a_stream_gives_the_bytes_in_pieces_of_at_most_8_mib(dock, compressible):
    data = (b"0123456789abcdef" * (MIB // 16 * 9)) if compressible else os.urandom(9 * MIB)
    api.put(dock, "big", data)
    pieces = list(api.stream(dock, "big"))
    assert [len(piece) for piece in pieces] == [8 * MIB, MIB] and b"".join(pieces) == data
    assert b"".join(api.stream(dock, "big", 5, 100)) == data[5:105]
    assert b"".join(api.stream(dock, "big", 9 * MIB - 10)) == data[-10:]
    with pytest.raises(api.MissingFile):
        list(api.stream(dock, "none"))


def test_a_stream_of_a_file_that_ends_early_is_refused(dock, tmp_path):
    data = os.urandom(9 * MIB)
    api.put(dock, "big", data)
    pieces = api.stream(dock, "big")
    assert next(pieces) == data[:8 * MIB]
    os.truncate(tmp_path / "dock" / "big", 8 * MIB)  # cut while it was read
    with pytest.raises(api.CorruptFile):
        next(pieces)


def test_a_stream_of_pieces_is_put_whole(dock):
    pieces = [b"first ", b"second ", b"third\n"]
    stored = api.put(dock, "gathered", iter(pieces))
    assert stored == {"key": "gathered", "sha256": sha(b"".join(pieces)), "bytes": len(b"".join(pieces))}
    assert api.fetch(dock, "gathered", stored["sha256"]) == b"".join(pieces)


def test_a_file_is_fetched_into_a_path_of_the_callers_and_a_bad_one_leaves_nothing(dock, tmp_path):
    text = b"a token file\n" * 10_000
    api.put(dock, "corpora/c/tokens", text)
    target = tmp_path / "mine" / "tokens"
    assert api.fetch_into(dock, "corpora/c/tokens", target, sha(text)) == {
        "key": "corpora/c/tokens", "sha256": sha(text), "bytes": len(text)}
    assert target.read_bytes() == text
    target.write_bytes(b"changed by its caller")  # the caller's own copy: the dock's file stays
    assert api.fetch(dock, "corpora/c/tokens", sha(text)) == text
    refused = tmp_path / "mine" / "refused"
    with pytest.raises(api.CorruptFile):
        api.fetch_into(dock, "corpora/c/tokens", refused, sha(b"other bytes"))
    assert list(refused.parent.iterdir()) == [target]  # neither the file nor a part of it
    with pytest.raises(api.MissingFile):
        api.fetch_into(dock, "corpora/c/none", refused)


def test_a_file_put_from_a_path_and_a_content_file(dock, tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights" * 1000)
    assert api.put_from(dock, "models/m/model.bin", source)["sha256"] == sha(source.read_bytes())
    assert source.read_bytes() == b"weights" * 1000  # left as it was
    kept = api.put_content(dock, b"a cut's output\n")
    assert kept["sha256"] == sha(b"a cut's output\n")
    assert kept["key"] == api.content_key(kept["sha256"]) == f"cache/files/{kept['sha256']}"
    assert api.fetch(dock, kept["key"], kept["sha256"]) == b"a cut's output\n"
    assert api.put_content_from(dock, source)["key"] == api.content_key(sha(source.read_bytes()))
    with pytest.raises(ValueError):
        api.content_key("not a sha256")


def test_records_and_results_are_values(dock):
    stored = api.put_json(dock, "corpora/c/corpus.json", {"id": "c", "shards": []})
    assert api.get_json(dock, "corpora/c/corpus.json") == stored and "check" in stored
    assert api.get_json(dock, "corpora/none.json") is None
    assert api.get_result(dock, "builds", "abc") is None
    api.put_result(dock, "builds", "abc", b"what the stage gave\n", made_by="the-commit")
    assert api.get_result(dock, "builds", "abc") == (b"what the stage gave\n", "the-commit")
    api.put_result(dock, "builds", "def", b"no maker\n")
    assert api.get_result(dock, "builds", "def") == (b"no maker\n", None)


def test_a_reader_of_a_growing_record_resumes_from_a_cursor_sent_as_json(dock):
    api.append_record(dock, "run-1", "events.jsonl", b"one\n")
    api.flush_records(dock)
    data, cursor = api.read_new(dock, "run-1", "events.jsonl")
    assert data == b"one\n"
    cursor = json.loads(json.dumps(cursor))  # a value: it crosses a route as it is
    api.append_record(dock, "run-1", "events.jsonl", b"two\n")
    api.append_record(dock, "run-1", "events.jsonl", b"three\n")
    api.flush_records(dock)
    data, cursor = api.read_new(dock, "run-1", "events.jsonl", cursor)
    assert data == b"two\nthree\n"
    cursor = json.loads(json.dumps(cursor))
    api.end_records(dock, "run-1")  # the plain file becomes a part that repeats what was read: left out
    assert api.read_new(dock, "run-1", "events.jsonl", cursor)[0] == b""
    api.write_record(dock, "run-1", "end.json", b"{}\n")
    assert api.read_record(dock, "run-1", "events.jsonl") == b"one\ntwo\nthree\n"
    assert api.record_files(dock, "run-1") == ["end.json", "events.jsonl"] and api.record_folders(dock) == ["run-1"]
    with pytest.raises(api.MixedRecord):
        api.write_record(dock, "run-1", "events.jsonl", b"whole\n")


def test_a_file_another_process_replaced_is_read_anew_after_its_size(dock, tmp_path):
    first, second = b"first version\n" * 100_000, b"other version\n" * 100_000  # compressed, the same length
    api.put(dock, "shard", first)
    assert api.fetch_range(dock, "shard", 0, 14) == b"first version\n"  # its frames are kept in this process
    open_dock(dock).put_named("shard", second)  # another process
    assert api.size(dock, "shard") == len(second)
    assert api.fetch_range(dock, "shard", 0, 14) == b"other version\n"
    api.put(dock, "shard", first)
    open_dock(dock).put_named("shard", second)
    assert b"".join(api.stream(dock, "shard", 0, 14)) == b"other version\n"
    open_dock(dock).put_named("shard", first)
    assert api.fetch_into(dock, "shard", tmp_path / "copy", sha(first))["bytes"] == len(first)


def test_a_dock_is_ready_only_when_its_folder_exists(tmp_path):
    missing = (tmp_path / "none").as_uri()
    with pytest.raises(api.NoDockFolder):
        api.ready(missing)
    assert not (tmp_path / "none").exists()  # none is made
    (tmp_path / "there").mkdir()
    assert api.ready((tmp_path / "there").as_uri()) is None
    with pytest.raises(TypeError):
        api.find(tmp_path, "")  # a dock is named by its URL


def test_the_url_comes_from_the_settings_and_an_s3_dock_opens_with_the_keys_found_there(tmp_path, monkeypatch):
    for name in ("DOCK", "DOCK_ACCESS_KEY_ID", "DOCK_SECRET_ACCESS_KEY", "DOCK_SESSION_TOKEN", "DOCK_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    none = tmp_path / "none.env"
    with pytest.raises(api.DockSettingError):
        api.url(env_file=none, environ={})
    with pytest.raises(api.DockSettingError):
        api.url(env_file=none, environ={"DOCK": "http://elsewhere"})
    with pytest.raises(api.DockSettingError):
        api.url(env_file=none, environ={"DOCK": "s3://bucket/prefix"})  # no keys
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="api-test")
        found = api.url(env_file=none, environ={"DOCK": "s3://api-test/prefix", "DOCK_ACCESS_KEY_ID": "a",
                                                "DOCK_SECRET_ACCESS_KEY": "s"})
        assert found == "s3://api-test/prefix"
        api.put(found, "a/file", b"bytes on s3\n")  # the keys of url, not of the process's environment
        assert api.fetch(found, "a/file") == b"bytes on s3\n" and api.ready(found) is None
