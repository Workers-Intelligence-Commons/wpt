import boto3
import pytest
from moto import mock_aws

from dock import api
from wpt_corpus.files import DockFiles


def test_a_token_file_holds_the_shards_tokens_and_is_rebuilt_when_changed(tmp_path, monkeypatch):
    # training reads this file instead of the shards, so it must hold exactly
    # their tokens, and a copy changed on disk (its size or its time) must
    # never be used; an unchanged one is reused without reading it again
    import os

    import numpy as np

    from wpt_corpus import shards as shards_module
    from wpt_corpus.shards import token_file, write_shard

    store = (tmp_path / "store").as_uri()
    rows = [[([1, 2, 3], "s", "s/a", "p", 0), ([4, 5], "s", "s/b", "p", 0)], [([6, 7, 8, 9], "s", "s/c", "p", 0)]]
    shards = []
    for i, part in enumerate(rows):
        token_sha256 = write_shard(tmp_path / f"{i}.parquet", part)
        file = api.put_from(store, f"corpora/d/{i}.parquet", tmp_path / f"{i}.parquet")
        shards.append({"key": file["key"], "sha256": file["sha256"], "token_sha256": token_sha256})
    store = DockFiles(store, tmp_path / "reader")

    path = token_file(store, shards)
    assert np.fromfile(path, dtype="<u4").tolist() == list(range(1, 10))
    hashing = shards_module.sha256_file
    monkeypatch.setattr(shards_module, "sha256_file", lambda _: pytest.fail("an unchanged file was read again"))
    assert token_file(store, shards) == path
    monkeypatch.setattr(shards_module, "sha256_file", hashing)
    written = path.stat().st_mtime_ns
    path.write_bytes(np.arange(9, dtype="<u4").tobytes())
    os.utime(path, ns=(written + 10**9, written + 10**9))  # a write, whatever the file system's clock grain
    assert np.fromfile(token_file(store, shards), dtype="<u4").tolist() == list(range(1, 10))
    os.utime(path, ns=(written + 2 * 10**9, written + 2 * 10**9))  # touched: read again, and reused
    assert np.fromfile(token_file(store, shards), dtype="<u4").tolist() == list(range(1, 10))


def test_a_shard_stream_keeps_only_the_shards_in_use_on_disk(tmp_path, monkeypatch):
    # a machine whose disk is smaller than the dataset trains from a remote
    # store: each step reads the shards in use at its place in the dataset,
    # and only those (and the next, prepared ahead) stay on its disk
    import numpy as np

    from wpt_corpus.shards import ShardStream, write_shard

    for name, value in {"DOCK_ACCESS_KEY_ID": "test", "DOCK_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        dock = "s3://wpt-test/stream"
        store = DockFiles(dock, tmp_path / "reader")
        shards = []
        for i in range(4):
            path = tmp_path / f"{i}.parquet"
            token_sha256 = write_shard(path, [(list(range(10 * i, 10 * i + 10)), "s", f"s/{i}", "p", 0)])
            file = api.put_from(dock, f"corpora/d/{i}.parquet", path)
            shards.append({"key": file["key"], "sha256": file["sha256"], "token_sha256": token_sha256, "tokens": 10})
        stream = ShardStream(store, shards, window=2)

        def on_disk():
            return sorted((store.scratch / "tokens").glob("*.u32"))

        assert len(stream) == 40
        assert stream.at(0)[0:20].tolist() == list(range(0, 20))
        assert stream.at(25)[0:20].tolist() == list(range(20, 40))
        # after the last shard comes the first again; a window may span two shards
        assert stream.at(35)[5:15].tolist() == [35, 36, 37, 38, 39, 0, 1, 2, 3, 4]
        assert len(on_disk()) <= 3
        assert not any((store.copies / s["sha256"][:2] / s["sha256"]).exists() for s in shards)  # copies released
        stream.close()
        assert on_disk() == []


def test_reading_documents_from_a_remote_dock_keeps_at_most_one_shard_on_disk(tmp_path, monkeypatch):
    # the drafter reads the training shards once at the end of a run: each
    # shard's local copy goes once its rows are read, as a shard stream's do
    from wpt_corpus.shards import documents, write_shard

    for name, value in {"DOCK_ACCESS_KEY_ID": "test", "DOCK_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        dock = "s3://wpt-test/documents"
        store = DockFiles(dock, tmp_path / "reader")
        shards = []
        for i in range(3):
            path = tmp_path / f"{i}.parquet"
            rows = [([i, 0], "s", f"s/{i}", "p", 0), ([i, 0], "s", "s/again", "p", 1)]
            token_sha256 = write_shard(path, rows)
            file = api.put_from(dock, f"corpora/d/{i}.parquet", path)
            shards.append({"key": file["key"], "sha256": file["sha256"], "token_sha256": token_sha256, "tokens": 4})

        def copies():
            return [s["key"] for s in shards if (store.copies / s["sha256"][:2] / s["sha256"]).exists()]

        seen = []
        for document, source, tokens in documents(store, shards):
            assert copies() == []  # the shard being read is already released
            seen.append((document, source, tokens.tolist()))
        assert seen == [("s/0", "s", [0, 0]), ("s/again", "s", [0, 0]), ("s/1", "s", [1, 0]), ("s/2", "s", [2, 0])]
