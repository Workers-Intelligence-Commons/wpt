import io
import random
from compression import zstd

import boto3
import pytest
from moto import mock_aws

from dock import CorruptFile, blocks, open_dock

TEXT = b"".join(f"line {i} of a text that compresses well\n".encode() for i in range(150_000))  # about 6 MB
NOISE = random.Random(1).randbytes(11 << 20)  # does not compress; on S3 it goes up in 3 parts


@pytest.fixture(params=["local", "s3"])
def dock(request, tmp_path, monkeypatch):
    """A local dock, or an S3 dock whose requests moto answers inside the test process."""
    if request.param == "local":
        yield open_dock((tmp_path / "dock").as_uri())
        return
    for name, value in {"DOCK_ACCESS_KEY_ID": "test", "DOCK_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        dock = open_dock("s3://wpt-test/compressed")
        dock.kind.part = 5 << 20  # S3's smallest part, so NOISE goes up in parts, several at once
        yield dock


def test_a_file_is_kept_compressed_only_when_that_saves_space_and_reads_back_alike(dock, tmp_path):
    (tmp_path / "text").write_bytes(TEXT)
    (tmp_path / "noise").write_bytes(NOISE)
    text, noise = dock.put_file(tmp_path / "text"), dock.put_file(tmp_path / "noise")
    small = dock.put_bytes(b"too short to shrink\n")
    stored = dock.kind.list("cache/files/")
    assert stored[f"cache/files/{text}.zst"] < len(TEXT) / 5
    assert stored[f"cache/files/{noise}"] == len(NOISE)
    assert stored[f"cache/files/{small}"] == 20
    assert len(stored) == 3

    # the compressed file is plain zstd: any zstd tool gives the bytes back
    assert zstd.decompress(dock.kind.read(f"cache/files/{text}.zst")[0]) == TEXT

    # sizes are known without copying anything into the cache
    assert (dock.size(text), dock.size(noise), dock.size(small)) == (len(TEXT), len(NOISE), 20)
    assert not any(dock.cache.rglob(text))

    # a range spanning two frames, read without the rest, then the whole file
    middle = blocks.FRAME * 3 - 100
    assert dock.read_range(text, middle, 300) == TEXT[middle:middle + 300]
    assert dock.read_range(noise, 10, 5) == NOISE[10:15]
    assert dock.get_bytes(text) == TEXT and dock.get_bytes(noise) == NOISE
    assert dock.path(text).read_bytes() == TEXT
    assert dock.path(small).read_bytes() == b"too short to shrink\n"
    dock.release(text)
    dock.delete(text)
    assert not dock.has(text)


def test_a_damaged_compressed_file_is_refused(tmp_path):
    dock = open_dock((tmp_path / "dock").as_uri())
    sha = dock.put_bytes(TEXT)
    packed = tmp_path / "dock" / "cache" / "files" / f"{sha}.zst"
    data = bytearray(packed.read_bytes())
    data[100] ^= 0xFF
    packed.write_bytes(bytes(data))
    with pytest.raises(CorruptFile):
        dock.get_bytes(sha)
    with pytest.raises(CorruptFile):
        dock.path(sha)


def test_frames_compressed_side_by_side_give_the_bytes_one_at_a_time_gives():
    alone, together = io.BytesIO(), io.BytesIO()
    writer = blocks.Writer(alone)
    writer.write(TEXT)
    writer.close()
    executor, sizing = blocks.parallel()
    with executor:
        writer = blocks.Writer(together, executor, sizing, batch=4)
        for at in range(0, len(TEXT), 300_000):  # written in pieces that do not match the frames
            writer.write(TEXT[at:at + 300_000])
        writer.close()
    assert together.getvalue() == alone.getvalue()


def test_each_file_is_looked_for_once_however_many_ranges_are_read(dock):
    noise = dock.put_bytes(NOISE[:1000])  # kept as it is: looked for under the compressed name first
    reader = open_dock(dock.root.as_uri() if hasattr(dock, "root") else f"s3://{dock.bucket}/{dock.prefix}")
    probes = []
    real = reader.kind.read

    def counted(key, start=0, length=None):
        if length == 0:
            probes.append(key)
        return real(key, start, length)

    reader.kind.read = counted
    for at in range(0, 1000, 100):
        assert reader.read_range(noise, at, 100) == NOISE[at:at + 100]
    assert len(probes) == 2  # its compressed name, then its own, once


def test_a_range_read_fetches_the_frames_after_it_and_the_next_range_sends_no_request(dock):
    key = dock.put_bytes(TEXT)
    reader = open_dock(dock.root.as_uri() if hasattr(dock, "root") else f"s3://{dock.bucket}/{dock.prefix}")
    sizes = []
    real = reader.kind.read

    def counted(key, start=0, length=None):
        sizes.append(length)
        return real(key, start, length)

    reader.kind.read = counted
    assert reader.read_range(key, 10, 100) == TEXT[10:110]
    sent = len(sizes)
    for at in (blocks.FRAME + 5, 3 * blocks.FRAME + 7, len(TEXT) - 50):
        assert reader.read_range(key, at, 40) == TEXT[at:at + 40]
        size = min(blocks.FRAME, len(TEXT) - at + 4)
        assert reader.read_range(key, at - 4, size) == TEXT[at - 4:at - 4 + size]
    assert len(sizes) == sent
