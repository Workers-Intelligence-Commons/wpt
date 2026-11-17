import hashlib
import io
import subprocess
import tarfile

from wpt.depot.web import archive
from wpt.depot.web.builds import ARCHIVE_FORMAT
from wpt_utils.programs import program

MTIME = 1_700_000_000


def whole(entries: list[tuple[str, bytes]]) -> bytes:
    """The archive of these (name, content) entries, made through layout and span."""
    pieces, length = archive.layout([(name, len(data)) for name, data in entries], MTIME)
    made = b"".join(archive.span(pieces, 0, length, lambda index, at, n: entries[index][1][at:at + n], 7))
    assert len(made) == length
    return made


def test_a_name_past_100_bytes_and_an_empty_file_read_back_with_tarfile_and_the_system_s_tar(tmp_path):
    entries = [("n" * 150 + ".parquet", b"some bytes"), ("empty.json", b""), ("short", b"x" * 1024)]
    made = whole(entries)
    with tarfile.open(fileobj=io.BytesIO(made)) as tar:
        assert [(one.name, tar.extractfile(one).read(), one.mtime) for one in tar] == [
            (name, data, MTIME) for name, data in entries]
    (tmp_path / "a.tar").write_bytes(made)
    (tmp_path / "out").mkdir()
    subprocess.run([program("tar"), "-xf", str(tmp_path / "a.tar"), "-C", str(tmp_path / "out")], check=True)
    assert {path.name: path.read_bytes() for path in (tmp_path / "out").iterdir()} == dict(entries)


def test_a_size_past_8_gib_takes_a_pax_header_that_tarfile_reads():
    size = 9 << 30
    pieces, length = archive.layout([("big.parquet", size), ("after", 3)], MTIME)
    assert length == pieces[2].offset + 512 + 512 + 1024  # the file whole, no padding, then a header, 3 bytes, the end
    head = b"".join(archive.span(pieces, 0, pieces[1].offset, lambda *_: b"", 1))
    with tarfile.open(fileobj=io.BytesIO(head + b"\0" * 1024)) as tar:
        member = tar.firstmember
    assert (member.name, member.size, member.mtime) == ("big.parquet", size, MTIME)


def test_any_span_is_that_slice_of_the_whole_archive():
    entries = [("a", b"abc" * 300), ("b" * 120, b"z" * 512), ("c", b"")]
    made = whole(entries)
    pieces, length = archive.layout([(name, len(data)) for name, data in entries], MTIME)
    for start, end in ((0, length), (5, 700), (511, 513), (1300, 2100), (length - 3, length)):
        part = b"".join(archive.span(pieces, start, end, lambda index, at, n: entries[index][1][at:at + n], 64))
        assert part == made[start:end]
    assert hashlib.sha256(made).hexdigest() == hashlib.sha256(whole(entries)).hexdigest()


def test_the_bytes_written_are_the_ones_the_archive_s_etag_names():
    """A resumed download joins bytes sent under the same ETag: a change to
    what archive.py writes must raise ARCHIVE_FORMAT, then this hash."""
    made = whole([("a" * 150, b"x" * 700), ("\u00e9t\u00e9.json", b""), ("short", b"abc")])
    made += archive.headers("big", 9 << 30, MTIME)
    assert (ARCHIVE_FORMAT, hashlib.sha256(made).hexdigest()) == (
        "wpt-tar-1", "2cfc7142e40e9ad461e5b33fc44d026fcf978bb9988fb77a6f4a04a717899680"), (
        "archive.py writes other bytes: raise ARCHIVE_FORMAT in wpt.depot.web.builds, then the hash here")
