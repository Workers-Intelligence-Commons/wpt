# The dock's api

The dock keeps what machines made: the freight's deliveries, models, run
records and cached results. Everything on it can be made again. It knows
nothing of the depot, the freight or WPT.

`dock.api` is what code outside the dock asks of it, and the only module of
the package `dock` that code outside it imports
(`packages/wic-utils/tests/test_part_imports.py` checks it). The command
`dock` (`apps/dock`) is a controller over the same functions. A change to
the module changes this document in the same commit.

## Settings

| Setting | What it names |
| --- | --- |
| `DOCK` | the dock, by its URL: `file:///path` for a local folder, `s3://bucket/prefix` for an S3-compatible bucket |
| `DOCK_ACCESS_KEY_ID`, `DOCK_SECRET_ACCESS_KEY` | an S3 dock's keys, both needed |
| `DOCK_SESSION_TOKEN`, `DOCK_ENDPOINT_URL`, `DOCK_REGION` | an S3 dock's session token, its endpoint when it is not AWS, its region (default `us-east-1`) |

Each is read from the environment, else from the `.env` file of the working
folder. No `AWS_*` setting and no AWS configuration file is read.

## The calls

Every call takes the dock by its URL, as `url` gives it, and takes and
gives only values JSON can carry and bytes, alone or as a stream of
pieces, so each can become a route without its callers changing. No call
hands out a dock object or a path into the dock's own files: a caller that
needs a file on its disk names a path in a folder of its own (`fetch_into`,
`put_from`), and the dock's cache and scratch folders stay its own.

The dock of a URL is opened once in each process, on its first call, and
kept: what it learned of its files (the names they are stored under, the
seek tables and frames of a compressed file, the reads requested ahead)
and the record lines it holds back serve the later calls of the process.
A process started by fork opens it again. Reads keep no copy of a file in
the dock's cache folder.

A reference is what a record names a file by: `{key, sha256}`. A key never
ends with the `.zst` the dock adds to a file it keeps compressed.

### Settings and state

| Call | What it does |
| --- | --- |
| `url(env_file=".env", environ=None)` | the URL `DOCK` gives, in `environ` (by default the process's environment), else in `env_file`; an S3 dock of that URL reads its keys from the same places. `DockSettingError` when nothing sets it, when it is neither a `file://` URL nor an `s3://` one with a bucket, and for an S3 dock without its keys. Nothing is asked of the dock |
| `ready(dock)` | nothing when the dock answers (a local folder that exists and lists, a bucket that answers a listing); `NoDockFolder` for a local folder that does not exist, which it never makes; else what the folder or the bucket raised |

### Finding

| Call | What it gives |
| --- | --- |
| `find(dock, prefix)` | every file whose key starts with `prefix`, in the order of their keys: `[{key, bytes}]`, `bytes` the size it takes on the dock (a file kept compressed counts its compressed size) |
| `folders(dock, prefix)` | the names of the folders right under `prefix`, a key ending in `/`, sorted, without listing the files below them |
| `has(dock, key)` | whether the dock holds a file under `key` now, asked each time |
| `size(dock, key)` | the file's length in bytes, without reading it; `MissingFile` |

### Fetching

| Call | What it gives |
| --- | --- |
| `fetch(dock, key, sha256=None)` | the file's bytes, checked against `sha256` when it is given; `MissingFile` when the dock holds no such file, `CorruptFile` when the bytes do not match |
| `fetch_range(dock, key, start, length)` | `length` bytes of the file from `start`, without fetching the rest: a compressed file's frames are read a few at a time and kept a while, and on S3 the frames after a read going through the file are requested ahead. A range cannot be checked against the file's sha256: the caller checks it with a hash of its own. `MissingFile`, or `CorruptFile` when the file ends first |
| `stream(dock, key, start=0, length=None)` | the file's bytes from `start`, `length` of them (`None`: to its end), as pieces of at most 8 MiB read as `fetch_range` reads; `MissingFile` |
| `fetch_into(dock, key, path, sha256=None)` | the file written into `path`, a local file of the caller's, replaced whole (a file there before stays until the new one takes its place) and checked against `sha256` when it is given: `{key, sha256, bytes}`. The file is the caller's own copy. `MissingFile`, `CorruptFile`, and nothing is left at `path` |
| `check(dock, key, sha256)` | whether the file holds the bytes of that sha256, read through once; `MissingFile` |

What a process learned of a file serves the reads that follow. When
another process replaced the file since, `size`, `fetch`, `stream` and
`fetch_into` of its key notice it (the file's version changed) and read it
anew, and the `fetch_range` calls after them read the new file.

### Putting and removing

| Call | What it does |
| --- | --- |
| `put(dock, key, data)` | `data`, bytes or an iterable of pieces of bytes, stored under `key`, replacing any earlier file: `{key, sha256, bytes}`, a reference that fetches it back checked. A stream is gathered in the dock's own folder first |
| `put_from(dock, key, path)` | the local file at `path`, left as it is, stored under `key`: `{key, sha256, bytes}` |
| `put_content(dock, data)`, `put_content_from(dock, path)` | bytes, or a local file, kept under the key their sha256 names and sent only when the dock does not hold them yet: `{key, sha256, bytes}`. For a file whose content is all that names it |
| `content_key(sha256)` | that key, `cache/files/<sha256>`; `ValueError` for anything but a sha256 |
| `delete(dock, key)` | the file or the record under `key` removed; nothing when it is absent. Only for one nothing else names |

### Records

| Call | What it does |
| --- | --- |
| `put_json(dock, key, record)` | a JSON object stored under `key`, replacing any earlier one, with the sha256 of its fields in `check`: the record as stored, on the disk or held by S3 when it returns |
| `get_json(dock, key)` | the record under `key`, checked against its `check`; `None` without one, `CorruptFile` when it does not match |

### Results

A result is kept under the fingerprint of what made it,
`cache/<stage>/<fingerprint>`, for a later run to reuse.

| Call | What it does |
| --- | --- |
| `put_result(dock, stage, fingerprint, data, made_by=None)` | the bytes kept, replacing any earlier result; `made_by`, one word, says what made them. Their sha256 |
| `get_result(dock, stage, fingerprint)` | the result's bytes, checked against their sha256; `None` when there is none |
| `result_maker(dock, stage, fingerprint)` | what made it, as `put_result` got it; `None` when there is none or nothing was said |

### Run records

Each command writes the record of its run under `runs/<run>/`
([observability.md](../observability.md)): files written whole, and files
added to line by line, by several processes at once if need be.

| Call | What it does |
| --- | --- |
| `write_record(dock, run, name, data)` | the file written whole, replacing any earlier one, on the dock when it returns; `MixedRecord` for a file added to line by line |
| `append_record(dock, run, name, data)` | lines added at the end of the file, held back in the process until `flush_records`; `MixedRecord` for a file written whole |
| `flush_records(dock)` | the lines the process holds back sent, on the dock when it returns; the lines not sent are kept, to send again |
| `end_records(dock, run)` | the run's record ended: what a local dock kept as plain files while lines were added is compressed into parts |
| `read_record(dock, run, name)` | the file's bytes, every line in the order it was added; `MissingFile` |
| `read_new(dock, run, name, cursor=None)` | `(bytes, cursor)`: the bytes added since `cursor` (`None`: since the file began) and the cursor to give next time, a JSON value, for a reader that follows a run still going; a read that raises moves nothing |
| `record_files(dock, run)` | the names of the run's record files, sorted; `[]` for no such run |
| `record_folders(dock)` | every run's name, oldest first |

### The command

`dock` (`apps/dock`) gives three of the calls: `dock find PREFIX` prints one
JSON line per file, `dock fetch KEY [--sha256 S] [--out FILE]` writes the
bytes to standard output or into a new file, `dock put KEY FILE` prints
`{key, sha256, bytes}`. Exit codes: 1 refused or failed, 2 a wrong command
line.

## Names and refusals

- `NAME`, `check_name(part)`, `check_record_name(part)`,
  `as_record_name(text)`: the names a dock allows for what it holds and for
  run records (letters, digits, `.`, `_` and `-`, not starting with `.`).
- `DockError`, and under it `DockSettingError` (no `DOCK` or a bad one;
  under it `NoDockFolder`, a local folder that does not exist),
  `MissingFile`, `CorruptFile` and `MixedRecord` (a whole write to a record
  file added to line by line, or the reverse).
- `SETTING`, the name of the setting, `DOCK`.

## The layout

```
deliveries/<delivery key>/   what the freight delivered for a request (freight.md)
corpora/<name>/              a corpus built before the freight, still read by wpt train
models/<name>/               a model's files
runs/<run>/                  each command's record of one run
cache/<stage>/<fingerprint>  results that can be made again
backups/                     the depot's backups
```
