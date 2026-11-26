"""The routes of `depot web` that models trained before snapshots
read, frozen: the dataset routes of format 12, and the two reads of a
tokenizer recipe a training of then made, answering what they answered
then, read-only, from the depot's legacy tables (depot.legacy) and the
corpora the dock holds of each dataset. Every route is a GET; none writes
the depot. Code at the commits such a model records reaches them
(docs/training.md, "Models trained before snapshots"): `wpt train` reads
the listing of datasets, a dataset's downloads, its tokenizer recipe and
that recipe's vocabulary file, and another depot's vocabulary training
reads a dataset's sample; a person reads the rest from the pages of then.

A dataset's download of one vocabulary and size
(GET /api/datasets/{id}/download/{vocabulary}/{size}) gives the latest
fresh corpus of that triple built from the dataset's current revision (or
from one that differs from it only in its max_tokens or the vocabulary it
publishes, which a build never read: SAME), or a state with nothing to
download: "stale" (the freight marked the latest corpus of the dataset's
current revision stale, since an admitted erasure reached one of its
texts), "not built" or "no store". The depot marks nothing: between an
erasure and the freight's next check (docs/erasure.md, "What it covers"),
a corpus that holds the erased text is still sent. A
built corpus gives each of its files, streamed from the dock with Range
support, and all of them in one tar archive (depot_app.web.archive), made
as it is sent, of a length known beforehand, with the same Range support,
each saved under a name made of the dataset's name, the vocabulary, the
size and what the file is. GET /api/datasets/{id}/download lists the
triples the dock holds corpora of, each with its address and state, and
with the latest fresh corpus of the triple built from any revision of the
dataset (latest_fresh), the corpus `wpt train` of then takes. No corpus of
a dataset is built any more, so no job is ever named (job: null).

A dataset's export (GET /api/datasets/{id}/export) gives the texts of its
current revision in numbered parts (depot.legacy.Export), each made as
it is sent, in JSONL compressed with zstd or in Parquet, with its size once
it was sent whole (kept in the web app's downloads file, never in the
depot); a part whose size is known takes one Range of bytes. At most
downloads.EXPORTS parts and samples are sent at once in all, and
downloads.VISITOR_EXPORTS for one visitor (429 past them).

    legacy(address, downloads, refusals)   the routes over the depot at address
"""

import hashlib
import json
import queue
import re
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import wraps

from depot import errors
from depot.legacy import FORMS, Export, Legacy, depot_name, parts_for, sample
from dock import api as dock_api
from dock.api import DockError, DockSettingError

from .. import corpora, shared
from ..downloads import EXPORTS, VISITOR_EXPORTS, Downloads
from . import archive
from .files import ARCHIVE, ARCHIVE_FORMAT, CHUNK, JSON_TYPE, SIZING, Guarded, ranged, roles, slug
from .server import Request, Route, Stream

PROGRESS_SECONDS = 5  # seconds between two lines {waiting: true} of a sample slow to find its next text
TEXT_PART = re.compile(r"part-(\d{5})\.(jsonl\.zst|parquet)")
SHA256 = re.compile(r"[0-9a-f]{64}")
ALL = "all"  # the size of a build with no max_tokens
# the columns of a dataset's row a corpus of it depends on: never max_tokens nor vocabulary, which a build never read
SAME = ("name", "tokenizer", "libraries", "shares")
OBJECT, ROWS = {"type": "object"}, {"type": "array", "items": {"type": "object"}}


class _Stopped(Exception):
    """The stream a sample is made for was closed: the sample stops there."""


def size_of(given) -> int | None:
    """A build's size from a route: a whole number of tokens above 0, or "all" (None). BadQuery otherwise."""
    if given == ALL:
        return None
    try:
        size = int(given)
    except (TypeError, ValueError):
        size = 0
    if isinstance(given, bool) or size < 1 or str(size) != str(given).strip():
        raise errors.BadQuery(f"a size is a whole number of tokens above 0, or {ALL}: {given!r}")
    return size


def size_name(size: int | None) -> str:
    """A build's size as its address names it."""
    return ALL if size is None else str(size)


def vocabulary_given(given) -> str:
    """A vocabulary's sha256 from a route. BadQuery when it is not 64 lowercase hexadecimal characters."""
    if not isinstance(given, str) or not SHA256.fullmatch(given):
        raise errors.BadQuery(f"name a vocabulary by its sha256, 64 lowercase hexadecimal characters: {given!r}")
    return given


def sample_lines(texts: Iterator[tuple[str, str, bytes]]) -> Iterator[bytes]:
    """The texts of a dataset's sample (depot.legacy.sample) as the
    frozen sample route of a depot's web app sends them: for each text one
    JSON line {"depot", "text", "bytes"}, depot as the sample gives it,
    then the text's bytes; a last line {"end": true, "texts",
    "bytes"}. A line {"waiting": true} may come between texts, for each
    (None, None, b"") of texts: the sample is slow to find its next text."""
    count = total = 0
    for address, text, data in texts:
        if address is None:
            yield (json.dumps({"waiting": True}) + "\n").encode()
            continue
        yield (json.dumps({"depot": address, "text": text, "bytes": len(data)}) + "\n").encode() + data
        count, total = count + 1, total + len(data)
    yield (json.dumps({"end": True, "texts": count, "bytes": total}) + "\n").encode()


def legacy(address: str, downloads: Downloads, refusals, dock: str | None = None,
           public_url=shared.public_url) -> list[Route]:
    """The frozen dataset routes over the depot whose file is at address,
    read through its own read-only connections (depot.legacy.Legacy),
    whatever its format; downloads: the web app's (depot_app.downloads),
    where the streams being sent and the sizes of export parts are kept;
    refusals wraps a handler so that it answers each failure with its
    status (depot_app.web.routes.refusals); dock: the URL of the dock of
    the corpora (dock.api), by default the one DOCK names, read from the
    settings when first asked;
    public_url: the depot's public address, None without one."""
    read = Legacy(address)
    held = {} if dock is None else {"dock": dock}

    def opened_dock():
        if "dock" not in held:
            held["dock"] = dock_api.url(shared.env_file())
        return held["dock"]

    def wrapped(handler):
        """refusals for a handler that gets the Request too."""
        @wraps(handler)
        def answer(values, body, request):
            return refusals(lambda values, body: handler(values, body, request))(values, body)
        return answer

    def corpus_depot(row: dict) -> str:
        """The name the corpora of the dataset (its row) recorded for the depot (depot.legacy.depot_name)."""
        return depot_name(read, row, public_url())

    # Datasets

    @refusals
    def datasets(values, body):
        return 200, shared.jsonable(read.datasets())

    @refusals
    def dataset(values, body):
        found = read.dataset_by_id(values["id"])
        upstream = bool(read.upstream_owned([found["id"]]))
        libraries_held = read.libraries()
        row = read.here({**found, "upstream": upstream}, libraries_held)
        libraries = []
        for ref in row["libraries"]:
            mine = ref["library"] in row["here"]
            libraries.append({**ref, "local": mine, "name": libraries_held[ref["library"]]["name"] if mine else None,
                              "locked_at": libraries_held[ref["library"]]["locked_at"] if mine else None})
        try:
            built, unreachable = corpora.built_from(opened_dock(), corpus_depot(found), found["id"]), None
        except (DockSettingError, DockError, OSError) as error:
            built, unreachable = None, f"{type(error).__name__}: {error}"
        return 200, shared.jsonable({
            "dataset": row, "upstream": upstream, "libraries": libraries, "remote": read.remote(row["name"]),
            "read_again": read.read_again_shows(found["id"]), "figures": read.revision_figures(found),
            "corpora": None if built is None else [
                {**{key: one.get(key) for key in ("id", "built_at", "texts", "conversations", "total_tokens",
                                                  "items_tokens", "stale", "deleted")},
                 "revision": one["dataset"]["revision"], "vocabulary": corpora.triple(one)[1],
                 "max_tokens": corpora.triple(one)[2]} for one in built],
            "store": unreachable})

    @refusals
    def remote(values, body):
        return 200, shared.jsonable(read.remote(values["name"]))

    # A tokenizer recipe and its vocabulary file, which a training of a dataset's order read

    @refusals
    def tokenizer(values, body):
        found = read.tokenizer(values["name"])
        recipe = found["recipe"]
        named = None
        if recipe.get("depot") is None and recipe.get("dataset"):
            try:
                named = read.dataset_by_id(recipe["dataset"])["name"]
            except errors.NotFound:
                pass
        elif recipe.get("dataset"):
            named = recipe["dataset"]
        return 200, shared.jsonable({**found, "recipe": {**recipe, "dataset_name": named, "job": None},
                                     "upstream": bool(read.upstream_owned([recipe["id"]]))})

    @refusals
    def vocabulary_file(values, body):
        try:
            version = int(values["version"])
        except ValueError:
            raise errors.NotFound(f"no version {values['version']} of the tokenizer recipe {values['name']}") from None
        sha256, data = read.vocabulary_file(values["name"], version)
        return 200, Stream(JSON_TYPE, iter([data]), len(data),
                           {"Content-Disposition": f'attachment; filename="vocabulary-{sha256[:12]}.json"'})

    # Downloads of corpora

    def file_name(record: dict, one: dict) -> str:
        return one["key"].removeprefix(corpora.folder_of(record) + "/")

    def download_start(row: dict, vocabulary: str, size: int | None) -> str:
        """What every download name of the corpus starts with: the dataset, the vocabulary and the size."""
        named = read.vocabulary_named(vocabulary)
        words = f"{named['recipe']}-v{named['version']}" if named["recipe"] else vocabulary[:12]
        return f"{slug(row['name'])}_{slug(words)}_{size_name(size)}"

    def download_names(row: dict, record: dict, vocabulary: str, size: int | None) -> dict:
        """{file name: the name a browser saves it under} of each file of the
        corpus: the dataset, the vocabulary, the size and what the file is
        (numbered when several files are the same thing), its own extension
        kept, and the vocabulary, stored with none, saved as the JSON it is."""
        start = download_start(row, vocabulary, size)
        role = roles(record)
        files = corpora.files(record)
        alike = Counter(role.get(one["key"], "file") for one in files)
        seen = Counter()
        found = {}
        for one in files:
            what = role.get(one["key"], "file")
            seen[what] += 1
            number = f"-{seen[what]}" if alike[what] > 1 else ""
            name = file_name(record, one)
            extension = name[name.index("."):] if "." in name else ".json" if what == "vocabulary" else ""
            found[name] = f"{start}_{slug(what.replace(' ', '-'))}{number}{extension}"
        return found

    sizes = {}  # {corpus id: the sizes of its files}: a corpus never changes once built
    sizing = threading.Lock()

    def sizes_of(record: dict, files: list[dict], references: list[str]) -> list[int]:
        """The length of each file of the corpus: the record's own when it names
        one, else the dock's, asked SIZING at a time once per corpus and kept."""
        with sizing:
            known = sizes.get(record["id"])
        if known is None:
            asked = [i for i, one in enumerate(files) if "bytes" not in one]
            with ThreadPoolExecutor(min(SIZING, len(asked) or 1)) as pool:
                found = dict(zip(asked, pool.map(lambda i: dock_api.size(opened_dock(), references[i]), asked)))
            known = [found[i] if i in found else one["bytes"] for i, one in enumerate(files)]
            with sizing:
                sizes[record["id"]] = known
        return known

    def archive_of(row: dict, record: dict, vocabulary: str, size: int | None) -> dict:
        """The tar archive of every file of the corpus, in the order
        corpora.files gives them, each under its download name, dated when
        the corpus was built: its files' keys on the dock (references), its pieces
        (depot_app.web.archive.layout), its length, its ETag and its download
        name. It reads the record and the dock's sizes of the files."""
        files = corpora.files(record)
        saved = download_names(row, record, vocabulary, size)
        references = [one["key"] for one in files]
        entries = list(zip((saved[file_name(record, one)] for one in files), sizes_of(record, files, references)))
        built = int(datetime.fromisoformat(record["built_at"]).timestamp()) if record.get("built_at") else 0
        pieces, length = archive.layout(entries, built)
        tag = hashlib.sha256("\n".join([ARCHIVE_FORMAT, str(built), *(f"{name} {one['sha256']}" for (name, _), one
                                                                       in zip(entries, files))]).encode()).hexdigest()
        return {"references": references, "pieces": pieces, "bytes": length, "etag": f'"{tag}"',
                "download": f"{download_start(row, vocabulary, size)}.tar"}

    def current(row: dict, record: dict) -> bool:
        """Whether the corpus's record was built from the dataset's current
        revision (row), or from one that differs from it only in columns a
        build never read (those SAME leaves out)."""
        revision = record["dataset"]["revision"]
        if revision == row["revision"]:
            return True
        built = read.revision(row["id"], revision, SAME)
        return built is not None and built == {column: row[column] for column in SAME}

    def state_of(row: dict, records: list[dict]) -> tuple:
        """(the state of the dataset's download of a vocabulary and size,
        its fresh corpus or None, the latest corpus of the triple or None),
        records being the triple's corpora, latest first."""
        built = [one for one in records if current(row, one)]
        fresh = next((one for one in built if corpora.fresh(one)), None)
        last = next(iter(records), None)
        if fresh is not None:
            return "built", fresh, last
        newest = next(iter(built), None)
        if newest is not None and "stale" in newest:
            return "stale", None, newest
        return "not built", None, last

    def summary(record: dict | None) -> dict | None:
        """What a page says of a corpus: its counts, the tokens of each share
        and of its items, what it left out, its state and the revision of the
        dataset it was built from."""
        if record is None:
            return None
        return {key: record.get(key) for key in (
            "id", "built_at", "texts", "conversations", "total_tokens", "items_tokens", "shares_tokens", "left_out",
            "stale", "deleted")} | {
            "revision": record["dataset"]["revision"], "tasks": (record.get("tasks") or {}).get("count", 0)}

    def link(row: dict, vocabulary: str, size: int | None) -> str:
        return f"/api/datasets/{row['id']}/download/{vocabulary}/{size_name(size)}"

    @refusals
    def downloads_listed(values, body):
        row = read.dataset_by_id(values["id"])
        shown = {"dataset": shared.jsonable(row), "job": None, "corpora": []}
        try:
            found = corpora.built_from(opened_dock(), corpus_depot(row), row["id"])
        except (DockSettingError, DockError, OSError) as error:
            return 200, {**shown, "state": "no store", "problem": f"{type(error).__name__}: {error}"}
        triples = {}
        for record in found:
            triples.setdefault(corpora.triple(record)[1:], []).append(record)
        for (vocabulary, size), records in triples.items():
            state, fresh, last = state_of(row, records)
            shown["corpora"].append({"vocabulary": read.vocabulary_named(vocabulary), "max_tokens": size,
                                     "size": size_name(size), "link": link(row, vocabulary, size), "state": state,
                                     "corpus": summary(fresh or last),
                                     "latest_fresh": summary(next((one for one in records if corpora.fresh(one)),
                                                                  None))})
        return 200, {**shown, "state": "listed"}

    @refusals
    def download(values, body):
        row = read.dataset_by_id(values["id"])
        vocabulary, size = vocabulary_given(values["vocabulary"]), size_of(values["size"])
        shown = {"dataset": shared.jsonable(row), "vocabulary": read.vocabulary_named(vocabulary), "max_tokens": size,
                 "size": size_name(size), "link": link(row, vocabulary, size), "job": None}
        try:
            records = corpora.built_from(opened_dock(), corpus_depot(row), row["id"], vocabulary, size)
        except (DockSettingError, DockError, OSError) as error:
            return 200, {**shown, "state": "no store", "problem": f"{type(error).__name__}: {error}"}
        state, fresh, last = state_of(row, records)
        if fresh is None:
            return 200, {**shown, "state": state, "corpus": summary(last)}
        role = roles(fresh)
        saved = download_names(row, fresh, vocabulary, size)
        files = [{"name": file_name(fresh, one), "download": saved[file_name(fresh, one)], "bytes": one.get("bytes"),
                  "sha256": one["sha256"], "role": role.get(one["key"], "other"), "tokens": one.get("tokens")}
                 for one in corpora.files(fresh)]
        try:
            whole = archive_of(row, fresh, vocabulary, size)
            together = {"name": ARCHIVE, "download": whole["download"], "bytes": whole["bytes"]}
        except (DockError, OSError):
            together = None  # a file the dock cannot size: the files are still listed, each failing on its own
        return 200, {**shown, "state": "built",
                     "corpus": {**summary(fresh), "tokenizer": fresh["tokenizer"]}, "files": files,
                     "archive": together}

    def fresh_corpus(values) -> tuple[dict, dict]:
        """(the dataset's row, the fresh corpus its download address names). Refused or NotFound without one."""
        row = read.dataset_by_id(values["id"])
        vocabulary, size = vocabulary_given(values["vocabulary"]), size_of(values["size"])
        records = corpora.built_from(opened_dock(), corpus_depot(row), row["id"], vocabulary, size)
        _, found, _ = state_of(row, records)
        if found is None:
            raise errors.NotFound(f"the dataset {row['name']} has no corpus of the vocabulary {vocabulary[:12]} at "
                                  f"the size {size_name(size)} to download now: see its download page")
        return row, found

    @wrapped
    def download_file(values, body, request: Request):
        row, record = fresh_corpus(values)
        chosen = next((one for one in corpora.files(record) if file_name(record, one) == values["name"]), None)
        if chosen is None:
            raise errors.NotFound(f"corpus {record['id']} has no file {values['name']}")
        reference = chosen["key"]
        size = dock_api.size(opened_dock(), reference)
        start, end, status = ranged(request, size, f'"{chosen["sha256"]}"')
        if status is None:
            return start, end

        guard = downloads.streaming(record["id"], request.visitor, "file")
        guard.__enter__()

        def chunks():
            at = start
            while at < end:
                length = min(CHUNK, end - at)
                yield dock_api.fetch_range(opened_dock(), reference, at, length)
                at += length
        saved = download_names(row, record, vocabulary_given(values["vocabulary"]), size_of(values["size"]))
        headers = {"Accept-Ranges": "bytes", "ETag": f'"{chosen["sha256"]}"',
                   "Content-Disposition": f'attachment; filename="{saved[values["name"]]}"'}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{size}"
        kind = JSON_TYPE if roles(record).get(chosen["key"]) == "vocabulary" else "application/octet-stream"
        return status, Stream(kind, Guarded(chunks(), guard), end - start, headers)

    @wrapped
    def download_archive(values, body, request: Request):
        row, record = fresh_corpus(values)
        whole = archive_of(row, record, vocabulary_given(values["vocabulary"]), size_of(values["size"]))
        start, end, status = ranged(request, whole["bytes"], whole["etag"])
        if status is None:
            return start, end

        guard = downloads.streaming(record["id"], request.visitor, "file")
        guard.__enter__()
        references = whole["references"]

        def read_piece(index: int, at: int, length: int) -> bytes:
            return dock_api.fetch_range(opened_dock(), references[index], at, length)
        headers = {"Accept-Ranges": "bytes", "ETag": whole["etag"],
                   "Content-Disposition": f'attachment; filename="{whole["download"]}"'}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{whole['bytes']}"
        chunks = archive.span(whole["pieces"], start, end, read_piece, CHUNK)
        return status, Stream("application/x-tar", Guarded(chunks, guard), end - start, headers)

    # Exports of texts

    def export_of(row: dict) -> tuple[Export, int, int]:
        """(the export of the dataset's current revision, how many texts it counts, its parts)."""
        export = Export(read, row, public_url())
        texts = downloads.export_texts(export.identity)
        if texts is None:
            texts = export.texts()
            downloads.counted(export.identity, texts)
        return export, texts, parts_for(texts)

    @refusals
    def export_listed(values, body):
        row = read.dataset_by_id(values["id"])
        export, texts, parts = export_of(row)
        known = downloads.export_sizes(export.identity, parts)
        listed = [{"part": number, "forms": {form: {"name": f"part-{number:05d}.{form}",
                                                    "download": f"{slug(row['name'])}_part-{number:05d}.{form}",
                                                    **known.get((number, form),
                                                                {"bytes": None, "sha256": None, "left_out": None})}
                                             for form in FORMS}} for number in range(parts)]
        return 200, {"dataset": shared.jsonable(row), "export": export.identity, "texts": texts,
                     "texts_per_part": texts / parts, "parts": listed,
                     "figures": shared.jsonable(read.revision_figures(row))}

    @wrapped
    def export_part(values, body, request: Request):
        row = read.dataset_by_id(values["id"])
        export, _, parts = export_of(row)
        match = TEXT_PART.fullmatch(values["name"])
        if match is None or int(match[1]) >= parts:
            raise errors.NotFound(f"the export of the dataset {row['name']} has no part {values['name']}: its parts "
                                  f"are part-00000 to part-{parts - 1:05d}, each .jsonl.zst or .parquet")
        number, form = int(match[1]), match[2]
        known = downloads.export_sizes(export.identity, parts).get((number, form))
        start, end, status = (0, None, 200) if known is None else ranged(request, known["bytes"])
        if status is None:
            return start, end
        guard = downloads.streaming(None, request.visitor, "text")
        guard.__enter__()  # TooMany before anything is sent

        def chunks():
            digest, size, left_out = hashlib.sha256(), 0, {}
            for chunk in export.part(number, parts, form, left_out):
                digest.update(chunk)
                size += len(chunk)
                if end is None:
                    yield chunk
                    continue
                piece = chunk[max(start - (size - len(chunk)), 0):max(end - (size - len(chunk)), 0)]
                if piece:
                    yield piece
                if size >= end:
                    return
            if known is None:
                downloads.exported(export.identity, parts, number, form, size, digest.hexdigest(), left_out)
        headers = {"Content-Disposition": f'attachment; filename="{slug(row["name"])}_{values["name"]}"'}
        if known is not None:
            headers.update({"Accept-Ranges": "bytes", "ETag": f'"{known["sha256"]}"'})
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{known['bytes']}"
        return status, Stream(FORMS[form], Guarded(chunks(), guard), None if known is None else end - start, headers)

    # The sample another depot's vocabulary training read

    @wrapped
    def sample_route(values, body, request: Request):
        try:
            size = int(request.query["bytes"])
        except (KeyError, ValueError):
            raise errors.BadQuery("name the sample's size in bytes: ?bytes=N") from None
        if size < 1:
            raise errors.BadQuery("a sample holds at least one byte")
        row = read.dataset_named(values["dataset"])
        guard = downloads.streaming(None, request.visitor, "sample")
        guard.__enter__()  # TooMany before anything is sent
        waiting = queue.Queue(maxsize=64)
        stop = threading.Event()  # set once the server closes the stream: the client went away, or it ended

        def offer(item):
            """Hand item to the stream; _Stopped once the stream is closed."""
            while True:
                if stop.is_set():
                    raise _Stopped
                try:
                    waiting.put(item, timeout=PROGRESS_SECONDS)
                    return
                except queue.Full:
                    continue

        def produce():
            try:
                texts = sample(read, row["id"], size)
                with closing(texts):
                    for found in texts:
                        offer(found)
                offer(StopIteration())
            except _Stopped:
                pass
            except Exception as error:
                try:
                    offer(error)
                except _Stopped:
                    pass

        threading.Thread(target=produce, daemon=True).start()

        def texts():
            while True:
                try:
                    found = waiting.get(timeout=PROGRESS_SECONDS)
                except queue.Empty:
                    yield None, None, b""
                    continue
                if isinstance(found, StopIteration):
                    return
                if isinstance(found, Exception):
                    raise found
                yield found

        return 200, Stream("application/octet-stream", Guarded(sample_lines(texts()), guard, stop.set))

    return [
        Route("GET", "/api/datasets", datasets, "frozen: every dataset as it was when datasets went, with whether it "
              "belongs to the upstream this depot follows, and here, the ids of the libraries it reads from this "
              "depot", answer=ROWS),
        Route("GET", "/api/datasets/{id}", dataset, "frozen: one dataset by its id: its row as GET /api/datasets "
              "gives it, whether the upstream owns it, each library it reads with its change number and local "
              "(with, for one read here, its name and whether it is locked), its remote, the corpora the dock "
              "holds built from it (null, with why in store, when the dock cannot be read), read_again (always "
              "false) and figures (measure: what its revision stored it reads, null when not known; since: null)",
              answer={"type": "object", "required": ["dataset", "upstream", "libraries", "remote", "corpora",
                                                     "store", "read_again", "figures"]}),
        Route("GET", "/api/tokenizers/{name}", tokenizer, "frozen: one tokenizer recipe as the depot kept it: its "
              "current revision (with dataset_name, the name of the dataset it named), every revision, its "
              "vocabularies without their files, whether its version has one (trained), and whether the upstream "
              "owns it", answer={"type": "object", "required": ["recipe", "revisions", "vocabularies", "trained"]}),
        Route("GET", "/api/tokenizers/{name}/vocabularies/{version}/file", vocabulary_file, "frozen: the file of "
              "the vocabulary of one version of a tokenizer recipe (wpt_tokenizer's format)"),
        Route("GET", "/api/datasets/{name}/remote", remote, "frozen: the dataset as a remote: the fields of its row "
              "that say what it reads (name, tokenizer, libraries, shares, max_tokens)", answer=OBJECT),
        Route("GET", "/api/datasets/{id}/download", downloads_listed, "frozen: the corpora the dock holds of the "
              "dataset, one row per vocabulary and size, each with the address of its download, its state and "
              "latest_fresh, the latest fresh corpus of that vocabulary and size built from any revision of the "
              "dataset (the one training of then takes); no store instead when nothing can be listed", answer={"type": "object", "required": ["state", "corpora"]}),
        Route("GET", "/api/datasets/{id}/download/{vocabulary}/{size}", download, "frozen: the dataset's download "
              "of one vocabulary (its sha256) and size (a number of tokens, or all): state built (the latest fresh "
              "corpus built from the dataset's current revision, its files and the tar archive of every file), "
              "stale, not built or no store", answer={"type": "object", "required": ["state"]}),
        Route("GET", "/api/datasets/{id}/download/{vocabulary}/{size}/files/{name}", download_file, "frozen: one "
              "file of that corpus, streamed from the dock and saved under its download name; one Range of bytes "
              "gives 206", request=True),
        Route("GET", f"/api/datasets/{{id}}/download/{{vocabulary}}/{{size}}/{ARCHIVE}", download_archive, "frozen: "
              "every file of that corpus in one tar archive, made as it is sent, the same bytes each time; one "
              "Range of bytes gives 206", request=True),
        Route("GET", "/api/datasets/{id}/export", export_listed, "frozen: the export of the texts of the dataset's "
              "current revision: how many texts its libraries hold, and its parts, each in jsonl.zst and parquet, "
              "with its download name and its size and sha256 once sent whole (null before), and the figures of "
              "that revision as GET /api/datasets/{id} gives them",
              answer={"type": "object", "required": ["export", "texts", "parts", "figures"]}),
        Route("GET", "/api/datasets/{id}/export/{name}", export_part, "frozen: one part of the export "
              "(part-NNNNN.jsonl.zst or part-NNNNN.parquet), made as it is sent, the same bytes every time; one "
              f"Range of bytes gives 206 once its size is known; at most {EXPORTS} at once in all and "
              f"{VISITOR_EXPORTS} per visitor (429)", request=True),
        Route("GET", "/api/datasets/{dataset}/sample", sample_route, "frozen: the sample of the dataset (its id or "
              "name) that trained a vocabulary, in its corpus's order, each share filling its part of the bytes: "
              "lines {depot ('' for this depot), text, bytes} each followed by the text's bytes, a line "
              "{waiting: true} while the next text is slow to come, and a last line {end, texts, bytes}",
              request=True, query={"bytes": "the sample's size in bytes"}),
    ]
