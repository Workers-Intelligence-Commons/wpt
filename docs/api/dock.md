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

## The three gestures

Each takes the dock by its URL and gives only values JSON and byte streams
can carry, so each can become a route without its callers changing.

| Call | What it gives |
| --- | --- |
| `url(env_file=".env", environ=None)` | the URL `DOCK` names; `DockSettingError` when nothing sets it |
| `find(dock, prefix)` | every file whose key starts with `prefix`, in the order of their keys: `[{key, bytes}]`, `bytes` the size it takes on the dock (a file kept compressed counts its compressed size); a key never ends with the `.zst` the dock adds to a file it keeps compressed |
| `fetch(dock, key, sha256=None)` | the file's bytes, checked against `sha256` when it is given; `MissingFile` when the dock holds no such file, `CorruptFile` when the bytes do not match |
| `put(dock, key, data)` | `data` stored under `key`, replacing any earlier file: `{key, sha256, bytes}` |

The command gives the same three: `dock find PREFIX` prints one JSON line
per file, `dock fetch KEY [--sha256 S] [--out FILE]` writes the bytes to
standard output or into a new file, `dock put KEY FILE` prints
`{key, sha256, bytes}`. Exit codes: 1 refused or failed, 2 a wrong command
line.

## The dock as an object

Two calls hand out the dock as an object, `Dock`, and so break the rule of
the gestures: `open_dock(url, environ=None, env_file=None)` and
`dock_from_settings(env_file=".env", environ=None)`. The freight's
deliveries, WPT's training, models, records and evaluations, and the
depot's backups and frozen dataset routes read and write through it: its
results named by their fingerprints (`put_result`, `get_result`), files
named by their sha256 (`put_bytes`, `put_file`, `get_bytes`,
`read_range`, `path`), records written by key (`put_json`, `get_json`),
and run records (`write_record`, `append_record`, `read_record`,
`record_folders`). The module `dock` describes each.

`local_folder(url)` gives the folder a `file://` URL names, `None` for any
other.

## Names and refusals

- `File(key, sha256)`: one stored file as a record names it;
  `File.of({key, sha256})` and `reference()` turn it into that JSON and
  back. `PACKED` is `.zst`, the end of a key whose file is kept
  compressed.
- `NAME`, `check_name(part)`, `check_record_name(part)`,
  `as_record_name(text)`: the names a dock allows for what it holds and for
  run records (letters, digits, `.`, `_` and `-`, not starting with `.`).
- `DockError`, and under it `DockSettingError` (no `DOCK` or a bad one),
  `MissingFile`, `CorruptFile` and `MixedRecord` (a whole write to a record
  file added to line by line, or the reverse).

## The layout

```
deliveries/<delivery key>/   what the freight delivered for a request (freight.md)
corpora/<name>/              a corpus built before the freight, still read by wpt train
models/<name>/               a model's files
runs/<run>/                  each command's record of one run
cache/<stage>/<fingerprint>  results that can be made again
backups/                     the depot's backups
```
