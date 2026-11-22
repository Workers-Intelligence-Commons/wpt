"""The routes of `wpt depot web` over the repository's model recipes and
the corpora they build: the recipes of the checkout WPT_REPOSITORY names
(read through wpt_recipes.Recipes, the only reader of data/), and its
series, each named SERIES: and read at its largest size, whose one corpus
every size of the series reads (recipe_names), each with
its snapshot, its measure, its estimate and the corpus of its build key; a
build as a job; a recipe's data section composed from clicks, nothing
written; and the downloads of a corpus by its build key.

A recipe reads here when its data section has the snapshot shape and names
this depot by its public address (or the upstream it follows, whose
snapshots a clone holds), a snapshot of it by name whose change and digest
are the ones it names, and a vocabulary the depot holds (by sha256, of the
size it names). Its build key (wpt_depot.corpus.build_key) never holds the
address the request came from, so a clone and its upstream key the same
corpus. A recipe of the dataset shape of before snapshots is listed and not
built (state "dataset").

The measure comes from the snapshot's summary alone
(wpt_depot.corpus.census); the estimate tokenizes a sample
(wpt_depot.estimate, a few seconds, one at a time, kept by build key since
a snapshot never changes), and says what the store's cap would delete for
it (wpt_depot.corpora.Room.evicted). A build starts only once confirmed
(confirmed: true) with the estimate the recipe's page showed, and only
while it still holds (wpt.depot.shared.unchanged: refused with 409 and the
new estimate otherwise; a build that would delete a corpus the shown
estimate did not name counts as changed). It runs as a corpus job of the
heavy lane, keyed by the build key, its checkpoint kept by that key, so a
restart of the web app goes on from its last token file.

    build_job(depot, snapshot, data, vocabulary, corpus, recipe, code_commit, progress=..., job=...)
    recipe_routes(depot, jobs, refusals, wrapped)   the routes
    recipes_naming(depot)                            what the snapshot page lists as the recipes naming one
"""

import hashlib
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from wpt_depot import Depot, corpora, errors, snapshots
from wpt_depot.corpus import build_key, canonical, census
from wpt_depot.estimate import estimate
from wpt_recipes import SIZE_OF
from wpt_recipes.errors import RecipeError
from wpt_store import StoreError, StoreSettingError, store_from_settings
from wpt_store.names import File
from wpt_utils.lines import say
from wpt_utils.parallel import Sizing
from wpt_utils.programs import code_name

from .. import shared
from ..downloads import Downloads
from ..jobs import Job, Jobs
from . import archive
from .files import ARCHIVE, ARCHIVE_FORMAT, CHUNK, JSON_TYPE, SIZING, Guarded, ranged, roles, slug
from .server import Request, Route, Stream

KEPT_ESTIMATES = 64  # estimates kept by build key in this process
# the costs of a build a confirmation covers, with their slack: an estimate is a sample's, so it moves a little
COSTS = {"seconds": 30, "bytes": 1 << 20}
SUMMARY = ("id", "built_at", "recipe", "texts", "conversations", "total_tokens", "items_tokens", "shares_tokens",
           "left_out", "max_tokens", "stale", "deleted", "code_commit")


def build_job(depot: Depot, snapshot: str | None, data: dict | None, vocabulary: str | None,
              corpus: str | None = None, recipe: str | None = None, code_commit: str | None = None, *, progress,
              job: Job) -> dict:
    """Build the corpus of the snapshot of that id with the data section
    data and the vocabulary of that sha256 (the model recipe named recipe's,
    the code at code_commit), or, with corpus, the deleted corpus of that id
    again, into the store WPT_STORE names, under the cap WPT_CORPORA_CAP
    sets, going on from the checkpoint of the job's build key: {corpus,
    build_key, texts (every text and item it took), conversations, tasks,
    tokens (of plain text), item_tokens (of the conversations and tasks),
    left_out, changed}."""
    store = store_from_settings(shared.env_file())
    downloads = Downloads(job.path)
    cap = shared.corpora_cap()
    record = None
    if corpus is not None:
        record = corpora.find(store, corpus)
        if record is None:
            raise errors.NotFound(f"no corpus {corpus} in the store")
    work = job.work / "corpus" / (job.key or "corpus")
    built = shared.build_corpus(
        depot, snapshot, data, say, vocabulary=vocabulary, store=store, record=record, work=work,
        checkpoint=job.checkpoint(), save=job.save, progress=progress, recipe=recipe, code_commit=code_commit,
        room=lambda folder: corpora.Room(store, cap, folder, downloads.last_download, downloads.busy, say))
    job.save(None)
    shutil.rmtree(work, ignore_errors=True)
    return {"corpus": built["id"], "build_key": built.get("build_key"), "texts": built["texts"],
            "conversations": built.get("conversations", 0), "tasks": (built.get("tasks") or {}).get("count", 0),
            "tokens": built["total_tokens"], "item_tokens": built.get("items_tokens") or 0,
            "left_out": built.get("left_out") or {}, "changed": built.get("changed")}


def recipe_names(recipes) -> list[str]:
    """The names the recipes page lists: each model recipe's, then each
    series' as SERIES: (wpt_recipes.SIZE_OF), whose one corpus every size of
    the series reads (the series' data part at its largest size,
    wpt_recipes.series.data_part)."""
    return [*recipes.names("model"), *(f"{name}{SIZE_OF}" for name in recipes.names("series"))]


def file_of(recipes, name: str) -> str:
    """The repository file of the model recipe or series (SERIES:) of that name."""
    series, separated, _ = name.partition(SIZE_OF)
    return recipes.relative("series", series) if separated else recipes.relative("model", name)


def recipes_naming(depot: Depot):
    """naming(row): the model recipes and series (recipe_names) of the
    repository whose data section names the snapshot (its row): [{name,
    matches: whether the change and digest it names are the snapshot's}];
    [] without a repository."""
    def naming(row: dict) -> list[dict]:
        try:
            recipes = shared.repository()
        except shared.NoRepository:
            return []
        found = []
        for name in recipe_names(recipes):
            try:
                data = recipes.data_part(name)
            except RecipeError:
                continue
            if "snapshot" not in data or data["snapshot"] != row["name"]:
                continue
            if snapshots.owner_named(depot, data["depot"], shared.public_url()) != row["owner"]:
                continue
            found.append({"name": name, "matches": data.get("change") == row["at_change"]
                          and data.get("digest") == row["digest"]})
        return found
    return naming


def measure_of(summary: dict, data: dict) -> dict:
    """What the snapshot of that summary holds for the data section, from
    the summary alone: {shares: [{tag, share, texts, bytes}] (the plain
    texts each share takes, a text counting for its first share's tag; one
    share "every plain text" without shares), plain: {texts, bytes}, items:
    {texts, bytes} (the conversations and tasks its kinds take), no_share
    (plain texts no share takes), other (texts of kinds it does not take),
    kinds: {kind: {texts, bytes}} of the whole snapshot}."""
    wanted = canonical(data)
    shares = wanted["shares"] or None
    found = census(summary, shares, wanted["kinds"] or None)
    listed = shares or [{"tag": None, "share": 1.0}]
    return {"shares": [{"tag": one["tag"], "share": one["share"], "texts": found["pool_texts"][at],
                        "bytes": found["pool_bytes"][at]} for at, one in enumerate(listed)],
            "plain": {"texts": found["plain"], "bytes": sum(found["pool_bytes"].values())},
            "items": {"texts": found["items"], "bytes": found["item_bytes"]},
            "no_share": found["no_share"], "other": found["other"], "kinds": found["kinds"]}


def recipe_routes(depot: Depot, jobs: Jobs, refusals, wrapped) -> list[Route]:
    """The routes over the repository's recipes and the corpora of build
    keys; refusals wraps a handler so that it answers each failure with its
    status, wrapped one that gets the Request too."""
    downloads = Downloads(jobs.path)
    kept: dict[str, dict] = {}  # {build key: its estimate}, the KEPT_ESTIMATES made last
    estimating = threading.Lock()  # one estimate at a time: each tokenizes on one core of the web app

    def opened_store():
        return store_from_settings(shared.env_file())

    def vocabulary_of(data: dict) -> tuple[dict, object]:
        """({sha256, size, recipe, version} of the vocabulary data names, its tokenizer). NoVocabulary without it."""
        sha256 = (data.get("vocabulary") or {}).get("sha256")
        recipe, version, tokenizer = shared.vocabulary_by_sha256(depot, sha256 or "")
        return {"sha256": sha256, "size": tokenizer.vocab_size, "recipe": recipe["name"], "version": version}, tokenizer

    def plan(name: str, data: dict) -> dict:
        """What the recipe named so (data, its data section) gives here:
        {name, data, shape, state, why, snapshot (its row with its summary),
        vocabulary, tokenizer, build_key}: state "ready" with each, or
        "dataset", "snapshot not here", "snapshot differs", "no vocabulary"
        and "vocabulary differs" with why."""
        found = {"name": name, "data": data, "shape": "dataset" if "dataset" in data else "snapshot",
                 "state": "ready", "why": None, "snapshot": None, "vocabulary": None, "tokenizer": None,
                 "build_key": None}
        if found["shape"] == "dataset":
            return {**found, "state": "dataset", "why": "a recipe of a dataset, written before snapshots: it is "
                                                        "not built here (docs/training.md)"}
        try:
            row = shared.snapshot_named(depot, data["depot"], data["snapshot"])
        except errors.NotFound as error:
            return {**found, "state": "snapshot not here", "why": str(error)}
        found["snapshot"] = row
        if data.get("change") != row["at_change"] or data.get("digest") != row["digest"]:
            return {**found, "state": "snapshot differs",
                    "why": f"snapshot {row['name']} of depot {row['owner']} reads change {row['at_change']} with "
                           f"digest {row['digest']}; the recipe says {data.get('change')}, {data.get('digest')}"}
        try:
            vocabulary, tokenizer = vocabulary_of(data)
        except shared.NoVocabulary as error:
            return {**found, "state": "no vocabulary", "why": str(error)}
        found |= {"vocabulary": vocabulary, "tokenizer": tokenizer}
        if (data.get("vocabulary") or {}).get("size") != vocabulary["size"]:
            return {**found, "state": "vocabulary differs",
                    "why": f"the vocabulary {vocabulary['sha256'][:12]} has {vocabulary['size']:,} tokens; the "
                           f"recipe says {data['vocabulary'].get('size')}"}
        return {**found, "build_key": build_key(row, data, vocabulary["sha256"])}

    def read_recipe(name: str) -> dict:
        return plan(name, shared.repository().data_part(name))

    def built(key: str | None) -> dict | None:
        """The summary of the corpus of the build key the store holds, None without one (or without a store)."""
        if key is None:
            return None
        try:
            record = corpora.of_key(opened_store(), key)
        except (StoreSettingError, StoreError, OSError):
            return None
        return None if record is None else summary(record)

    def summary(record: dict) -> dict:
        return {key: record.get(key) for key in SUMMARY} | {
            "build_key": record.get("build_key"), "fresh": corpora.fresh(record),
            "tasks": (record.get("tasks") or {}).get("count", 0), "link": f"/api/builds/{record.get('build_key')}"}

    def shown(found: dict) -> dict:
        """A plan as a route gives it: the snapshot without its summary, no tokenizer."""
        row = found["snapshot"]
        return {**{key: value for key, value in found.items() if key not in ("snapshot", "tokenizer")},
                "snapshot": None if row is None else shared.jsonable(
                    {key: value for key, value in row.items() if key != "summary"})}

    def costed(found: dict) -> dict:
        """The estimate of the build the plan (state ready) names, and what
        it does to the store: {**wpt_depot.estimate's, cap, taken (the bytes
        the other corpora take), evicted (what the cap deletes for it, None
        when it cannot fit), fits, built (the corpus of its key, fresh or
        not), store (the store's failure, when it cannot be read)}."""
        key = found["build_key"]
        with estimating:
            base = kept.get(key)
            if base is None:
                base = estimate(depot, found["snapshot"]["id"], found["data"], found["tokenizer"],
                                Sizing("tokenizing workers").limit)
                kept[key] = base
                while len(kept) > KEPT_ESTIMATES:
                    kept.pop(next(iter(kept)))
        cap, taken, evicted, problem = shared.corpora_cap(), None, None, None
        try:
            room = corpora.Room(opened_store(), cap, f"{corpora.FOLDER}/{key}", downloads.last_download,
                                downloads.busy)
            taken, evicted = room.taken(), room.evicted(base["bytes"])
        except (StoreSettingError, StoreError, OSError) as error:
            problem = f"{type(error).__name__}: {error}"
        return {**base, "cap": cap, "taken": taken, "evicted": evicted,
                "fits": problem is None and evicted is not None, "built": built(key), "store": problem}

    @refusals
    def listed(values, body):
        try:
            recipes = shared.repository()
        except shared.NoRepository as error:
            return 200, {"repository": None, "problem": str(error), "recipes": []}
        found = []
        for name in recipe_names(recipes):
            try:
                one = read_recipe(name)
            except RecipeError as error:
                found.append({"name": name, "file": file_of(recipes, name), "state": "invalid", "why": str(error)})
                continue
            found.append({**shown(one), "file": file_of(recipes, name), "corpus": built(one["build_key"]),
                          "job": jobs.open_job("corpus", key=one["build_key"]) if one["build_key"] else None})
        return 200, {"repository": shared.checkout(recipes), "problem": None, "recipes": found}

    @refusals
    def one(values, body):
        recipes = shared.repository()
        found = read_recipe(values["name"])
        series, separated, _ = values["name"].partition(SIZE_OF)
        answer = {**shown(found), "file": file_of(recipes, values["name"]), "repository": shared.checkout(recipes),
                  "problems": recipes.problems("series", series) if separated
                  else recipes.problems("model", values["name"]), "measure": None, "estimate": None,
                  "job": None}
        if found["snapshot"] is not None and found["state"] in ("ready", "no vocabulary", "vocabulary differs"):
            answer["measure"] = measure_of(found["snapshot"]["summary"], found["data"])
        if found["state"] == "ready":
            answer["estimate"] = costed(found)
            answer["job"] = jobs.open_job("corpus", key=found["build_key"])
        return 200, answer

    @refusals
    def build(values, body):
        recipes = shared.repository()
        found = read_recipe(values["name"])
        if found["state"] != "ready":
            raise errors.Refused(f"the recipe {values['name']} cannot be built here ({found['state']}): "
                                 f"{found['why']}")
        if body.get("confirmed") is not True:
            raise errors.Refused("a build starts only once confirmed (confirmed: true), after its estimate: nothing "
                                 "was queued")
        open_job = jobs.open_job("corpus", key=found["build_key"])
        if open_job is not None:
            return 200, open_job
        now = costed(found)
        shown_estimate = body.get("estimate")
        changed = shared.unchanged(shown_estimate, now, COSTS, same=("build_key",))
        if not changed and now["evicted"] is not None:
            named = {one["corpus"] for one in (shown_estimate.get("evicted") or [])}
            more = sorted(one["corpus"] for one in now["evicted"] if one["corpus"] not in named)
            if more:
                changed.append(f"the store's cap now deletes {', '.join(more)} too")
        if changed:
            return 409, {"problems": [f"the estimate changed since it was shown ({'; '.join(changed)}): confirm "
                                      "the new one"], "estimate": now}
        if now["built"] is not None and now["built"]["fresh"]:
            raise errors.Refused(f"the corpus of the recipe {values['name']} is built already: download it")
        if now["evicted"] is None:
            raise errors.Refused(now["store"] or "the corpus does not fit the store's cap, even once every other "
                                 "corpus it may delete is deleted: nothing was queued")
        commit = shared.checkout(recipes)["commit"]
        return 200, jobs.add(
            "corpus", f"build the corpus of the recipe {values['name']} (snapshot {found['snapshot']['name']}, "
                      f"{'every token' if found['data'].get('total_tokens') is None else format(found['data']['total_tokens'], ',') + ' tokens'})",
            build_job, found["snapshot"]["id"], found["data"], found["vocabulary"]["sha256"], None, values["name"],
            code_name() if commit is None else commit, key=found["build_key"], recipe=values["name"],
            target={"build_key": found["build_key"], "snapshot": found["snapshot"]["name"],
                    "recipe_commit": commit, "estimate": {key: now[key] for key in ("tokens", "bytes", "seconds")}})

    @refusals
    def compose(values, body):
        row = snapshots.snapshot_by_id(depot, body["snapshot"])
        own = row["owner"] == depot.name()
        address = (shared.public_url() or depot.name()) if own else depot.upstream()
        if address is None:
            raise errors.Refused(f"the snapshot {row['id']} belongs to {row['owner']}, which this depot does not "
                                 "follow: no recipe can name it")
        data = {"depot": address, "snapshot": row["name"], "change": row["at_change"], "digest": row["digest"],
                "kinds": body.get("kinds") or [], "shares": body.get("shares") or [],
                "total_tokens": body["total_tokens"], "vocabulary": {"sha256": body["vocabulary"], "size": None}}
        vocabulary, tokenizer = vocabulary_of(data)
        data["vocabulary"]["size"] = vocabulary["size"]
        canonical(data)
        found = {"name": None, "data": data, "shape": "snapshot", "state": "ready", "why": None, "snapshot": row,
                 "vocabulary": vocabulary, "tokenizer": tokenizer, "build_key": build_key(row, data,
                                                                                          vocabulary["sha256"])}
        return 200, {"recipe": {"data": data}, "build_key": found["build_key"], "vocabulary": vocabulary,
                     "measure": measure_of(row["summary"], data), "estimate": costed(found)}

    # Downloads of the corpus of a build key

    def record_of(key: str) -> dict:
        """The fresh corpus of the build key. Refused while an erasure is being carried into the corpora;
        NotFound without a fresh one."""
        marking = downloads.pending()
        if marking:
            raise errors.Refused(f"the corpora holding a text of the erasures {', '.join(marking)} are being marked "
                                 "stale: no corpus is sent until that ends")
        record = corpora.of_key(opened_store(), key)
        if record is None or not corpora.fresh(record):
            raise errors.NotFound(f"no corpus of the build {key} to download now: see its recipe's page")
        return record

    def file_name(record: dict, one: dict) -> str:
        return one["key"].removeprefix(corpora.folder_of(record) + "/")

    def download_start(record: dict) -> str:
        return f"{slug(record.get('recipe') or record['snapshot']['name'])}_{record['build_key'][:12]}"

    def download_names(record: dict) -> dict:
        """{file name: the name a browser saves it under}: the recipe, the start of the build key and what the
        file is (numbered when several are the same thing), its extension kept, the vocabulary saved as JSON."""
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
            found[name] = f"{download_start(record)}_{slug(what.replace(' ', '-'))}{number}{extension}"
        return found

    sizes, sizing = {}, threading.Lock()  # {corpus id: the sizes of its files}: a corpus never changes once built

    def sizes_of(record: dict, files: list[dict], references: list[File]) -> list[int]:
        with sizing:
            known = sizes.get(record["id"])
        if known is None:
            asked = [i for i, one in enumerate(files) if "bytes" not in one]
            with ThreadPoolExecutor(min(SIZING, len(asked) or 1)) as pool:
                found = dict(zip(asked, pool.map(lambda i: opened_store().size(references[i]), asked)))
            known = [found[i] if i in found else one["bytes"] for i, one in enumerate(files)]
            with sizing:
                sizes[record["id"]] = known
        return known

    def archive_of(record: dict) -> dict:
        files = corpora.files(record)
        saved = download_names(record)
        references = [File.of(one) for one in files]
        entries = list(zip((saved[file_name(record, one)] for one in files), sizes_of(record, files, references)))
        made = int(datetime.fromisoformat(record["built_at"]).timestamp()) if record.get("built_at") else 0
        pieces, length = archive.layout(entries, made)
        tag = hashlib.sha256("\n".join([ARCHIVE_FORMAT, str(made), *(f"{name} {one['sha256']}" for (name, _), one
                                                                      in zip(entries, files))]).encode()).hexdigest()
        return {"references": references, "pieces": pieces, "bytes": length, "etag": f'"{tag}"',
                "download": f"{download_start(record)}.tar"}

    @refusals
    def build_shown(values, body):
        key = values["key"]
        shown_job = jobs.open_job("corpus", key=key)
        marking = downloads.pending()
        if marking:
            return 200, {"build_key": key, "state": "checking an erasure", "erasures": marking, "job": shown_job}
        try:
            record = corpora.of_key(opened_store(), key)
        except (StoreSettingError, StoreError, OSError) as error:
            return 200, {"build_key": key, "state": "no store", "problem": f"{type(error).__name__}: {error}",
                         "job": shown_job}
        if record is None:
            return 200, {"build_key": key, "state": "being built" if shown_job else "not built", "job": shown_job}
        if not corpora.fresh(record):
            return 200, {"build_key": key, "state": "stale" if "stale" in record else "deleted",
                         "corpus": summary(record), "job": shown_job}
        role = roles(record)
        saved = download_names(record)
        files = [{"name": file_name(record, one), "download": saved[file_name(record, one)],
                  "bytes": one.get("bytes"), "sha256": one["sha256"], "role": role.get(one["key"], "other"),
                  "tokens": one.get("tokens")} for one in corpora.files(record)]
        try:
            whole = archive_of(record)
            together = {"name": ARCHIVE, "download": whole["download"], "bytes": whole["bytes"]}
        except (StoreError, OSError):
            together = None
        return 200, {"build_key": key, "state": "built", "job": shown_job,
                     "corpus": {**summary(record), "snapshot": record["snapshot"], "data": record.get("data"),
                                "tokenizer": record["tokenizer"], "depot": record.get("depot")},
                     "files": files, "archive": together}

    @wrapped
    def build_file(values, body, request: Request):
        record = record_of(values["key"])
        chosen = next((one for one in corpora.files(record) if file_name(record, one) == values["name"]), None)
        if chosen is None:
            raise errors.NotFound(f"corpus {record['id']} has no file {values['name']}")
        reference = File.of(chosen)
        size = opened_store().size(reference)
        start, end, status = ranged(request, size, f'"{chosen["sha256"]}"')
        if status is None:
            return start, end
        guard = downloads.streaming(record["id"], request.visitor, "file")
        guard.__enter__()

        def chunks():
            at = start
            while at < end:
                length = min(CHUNK, end - at)
                yield opened_store().read_range(reference, at, length)
                at += length
        headers = {"Accept-Ranges": "bytes", "ETag": f'"{chosen["sha256"]}"',
                   "Content-Disposition": f'attachment; filename="{download_names(record)[values["name"]]}"'}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{size}"
        kind = JSON_TYPE if roles(record).get(chosen["key"]) == "vocabulary" else "application/octet-stream"
        return status, Stream(kind, Guarded(chunks(), guard), end - start, headers)

    @wrapped
    def build_archive(values, body, request: Request):
        record = record_of(values["key"])
        whole = archive_of(record)
        start, end, status = ranged(request, whole["bytes"], whole["etag"])
        if status is None:
            return start, end
        guard = downloads.streaming(record["id"], request.visitor, "file")
        guard.__enter__()
        references = whole["references"]

        def read_piece(index: int, at: int, length: int) -> bytes:
            return opened_store().read_range(references[index], at, length)
        headers = {"Accept-Ranges": "bytes", "ETag": whole["etag"],
                   "Content-Disposition": f'attachment; filename="{whole["download"]}"'}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{whole['bytes']}"
        return status, Stream("application/x-tar", Guarded(archive.span(whole["pieces"], start, end, read_piece,
                                                                         CHUNK), guard), end - start, headers)

    estimate_body = {"type": "object", "description": "the estimate the recipe's page showed, as it gave it"}
    share = {"type": "object", "required": ["tag", "share"], "additionalProperties": False,
             "properties": {"tag": {"type": "string", "minLength": 1},
                            "share": {"type": "number", "minimum": 0, "maximum": 1}}}
    return [
        Route("GET", "/api/recipes", listed, "the model recipes of the repository checkout WPT_REPOSITORY names, "
              "then its series, each named SERIES: and read at its largest size, the one corpus every size reads, "
              "with the checkout's commit and whether it has changes not committed (repository: null and problem "
              "when it is not set): each recipe's name, its file, data section, shape (snapshot, or dataset for a "
              "recipe of before snapshots), state (ready, dataset, snapshot not here, snapshot differs, no vocabulary, "
              "vocabulary differs, invalid) and why, its snapshot, vocabulary and build key, the corpus of its "
              "key the store holds and its corpus job queued or running",
              answer={"type": "object", "required": ["repository", "problem", "recipes"]}),
        Route("POST", "/api/recipes/compose", compose, "a recipe's data section composed from a snapshot (its id), "
              "kinds, shares by tag, total_tokens and a vocabulary the depot holds (its sha256), with its build "
              "key, its measure (from the snapshot's summary) and its estimate, as GET /api/recipes/{name} gives "
              "them; nothing is written: the recipe file is the person's to commit",
              body={"type": "object", "required": ["snapshot", "total_tokens", "vocabulary"],
                    "properties": {"snapshot": {"type": "string", "minLength": 1},
                                   "kinds": {"type": "array", "items": {"type": "string", "minLength": 1},
                                             "uniqueItems": True},
                                   "shares": {"type": "array", "items": share},
                                   "total_tokens": {"type": "integer", "minimum": 1},
                                   "vocabulary": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}},
              answer={"type": "object", "required": ["recipe", "build_key", "measure", "estimate"]}),
        Route("GET", "/api/recipes/{name}", one, "one model recipe or series (SERIES:), as GET /api/recipes lists "
              "it, with the "
              "repository's commit, what the repository's check finds in it (problems), its measure (from its "
              "snapshot's summary: the plain texts and bytes each share takes, the items, what no share takes) "
              "and, when it can be built here, its estimate (tokens, bytes, seconds, from a sample) with what "
              "the store's cap would delete for it (evicted, null when it cannot fit; fits), the corpus of its "
              "build key (built) and its corpus job",
              answer={"type": "object", "required": ["name", "state", "data", "measure", "estimate", "job"]}),
        Route("POST", "/api/recipes/{name}/build", build, "queue the build of the recipe's corpus as a job of the "
              "heavy lane keyed by its build key (the job already open, when one is), only with confirmed: true "
              "and the estimate GET /api/recipes/{name} gave (409 with the new estimate when the build key "
              f"changed, it now costs more than {shared.TOLERANCE:.0%} beyond it, or the cap now deletes another "
              "corpus); refused when its corpus is built already or cannot fit",
              body={"type": "object", "required": ["confirmed", "estimate"],
                    "properties": {"confirmed": {"type": "boolean"}, "estimate": estimate_body}},
              answer={"type": "object", "required": ["id", "kind", "state"]}),
        Route("GET", "/api/builds/{key}", build_shown, "the corpus of a build key: its state (built, being built, "
              "not built, stale, deleted, checking an erasure, no store), its record's summary and, built, each "
              "file (name, download name, bytes, sha256, role, tokens) and the archive of them all",
              answer={"type": "object", "required": ["build_key", "state"]}),
        Route("GET", "/api/builds/{key}/files/{name}", build_file, "one file of the corpus of a build key, from "
              "the store, with one Range of bytes when asked", request=True),
        Route("GET", f"/api/builds/{{key}}/{ARCHIVE}", build_archive, "every file of the corpus of a build key in "
              "one tar archive, made as it is sent, of a length known beforehand, with one Range of bytes when "
              "asked", request=True),
    ]
