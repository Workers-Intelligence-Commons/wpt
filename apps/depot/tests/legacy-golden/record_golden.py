"""Record the answers the web app of the code it runs with gives to the dataset routes that code at earlier commits
calls (README.md beside this file lists them), on the format-12 fixture depot and its dock, as golden files.

    python record_golden.py FIXTURES OUT PORT

FIXTURES: packages/depot/tests/fixtures; OUT: this folder; PORT: a free port, 8500-8599. It copies the depot
and the dock into a temporary folder, serves them on 127.0.0.1:PORT with the routes `depot web` serves (the
interface's files left out: an empty folder stands for them), asks every route in the order record() asks them, and
writes each answer as OUT/answers/NNN.json: {method, route, asked_headers, status, headers (those of KEPT), body}, the
body as JSON, or, for a body that is not JSON or is a file to save, body_file and body_bytes, its bytes being
OUT/answers/NNN.bin. OUT/answers.json lists them in order. It stops the server at the end.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

KEPT = ("Content-Type", "Content-Length", "Content-Disposition", "Content-Range", "Accept-Ranges", "ETag")


def ask(base: str, route: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(base + route, headers={"User-Agent": "wpt-golden", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=60) as answer:
            return answer.status, dict(answer.headers), answer.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def main(fixtures: Path, out: Path, port: int):
    work = Path(tempfile.mkdtemp(prefix="wpt-golden-"))
    try:
        shutil.copy(fixtures / "format-12.sqlite", work / "depot.sqlite")
        shutil.copytree(fixtures / "format-12-dock", work / "dock")
        (work / "static").mkdir()
        os.chdir(work)  # no .env but this empty folder's
        os.environ.update({"DEPOT": str(work / "depot.sqlite"), "DOCK": (work / "dock").as_uri()})
        os.environ.pop("DEPOT_URL", None)

        from depot_app.jobs import Jobs
        from depot_app.shared import opened
        from depot_app.web.routes import STATUSES, routes
        from depot_app.web.server import make_server

        depot = opened()
        jobs = Jobs(depot.address, STATUSES)
        server = make_server(routes(depot, jobs), work / "static", port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        try:
            record(base, out)
        finally:
            server.shutdown()
            server.server_close()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def record(base: str, out: Path):
    answers = out / "answers"
    shutil.rmtree(answers, ignore_errors=True)
    answers.mkdir(parents=True)
    listed = []

    def get(route: str, headers: dict | None = None):
        status, got, body = ask(base, route, headers)
        number = len(listed)
        kept = {name: got[name] for name in KEPT if name in got}
        entry = {"method": "GET", "route": route, "asked_headers": headers or {}, "status": status, "headers": kept}
        try:
            shown = "json" in got.get("Content-Type", "") and "attachment" not in got.get("Content-Disposition", "")
            entry["body"] = json.loads(body) if shown else None  # a file to save is kept as its bytes
        except ValueError:
            entry["body"] = None
        if entry["body"] is None:
            entry["body_file"] = f"{number:03d}.bin"
            (answers / entry["body_file"]).write_bytes(body)
            entry["body_bytes"] = len(body)
        (answers / f"{number:03d}.json").write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n",
                                                   encoding="utf-8", newline="\n")
        listed.append({"file": f"{number:03d}.json", "route": route, "asked_headers": headers or {},
                       "status": status})
        return status, entry["body"], body

    _, datasets, _ = get("/api/datasets")
    by_name = {row["name"]: row["id"] for row in datasets}
    for name in ("ds-two", "ds-mixed"):
        get(f"/api/datasets/{by_name[name]}")
        get(f"/api/datasets/{quote(name)}/remote")
    get("/api/datasets/no-such-dataset")
    for name in ("ds-two", "ds-mixed"):
        get(f"/api/datasets/{by_name[name]}/download")
    get("/api/datasets/no-such-dataset/download")
    _, listing, _ = get(f"/api/datasets/{by_name['ds-two']}/download")  # asked twice: reading it changes nothing
    for corpus in sorted(listing["corpora"], key=lambda one: str(one["max_tokens"])):
        _, shown, _ = get(corpus["link"])
        if shown.get("state") != "built":
            continue
        for one in shown["files"]:
            get(f"{corpus['link']}/files/{quote(one['name'])}")
        first = shown["files"][0]["name"]
        get(f"{corpus['link']}/files/{quote(first)}", {"Range": "bytes=0-9"})
        get(f"{corpus['link']}/files/{quote(first)}", {"Range": "bytes=-5"})
        get(f"{corpus['link']}/files/{quote(first)}", {"Range": "bytes=999999-"})
        get(f"{corpus['link']}/files/no-such-file")
        get(f"{corpus['link']}/everything.tar")
        get(f"{corpus['link']}/everything.tar", {"Range": "bytes=100-1123"})
    vocabulary = listing["corpora"][0]["vocabulary"]["sha256"]
    get(f"/api/datasets/{by_name['ds-two']}/download/{vocabulary}/123")  # a size never built
    for name in ("ds-two", "ds-mixed"):
        _, export, _ = get(f"/api/datasets/{by_name[name]}/export")
        for part in export["parts"]:
            for form in part["forms"].values():
                get(f"/api/datasets/{by_name[name]}/export/{form['name']}")
        get(f"/api/datasets/{by_name[name]}/export")  # again, the sizes of the parts now known
    get(f"/api/datasets/{by_name['ds-two']}/export/part-99999.parquet")
    get(f"/api/datasets/{by_name['ds-two']}/sample?bytes=2000")
    get("/api/datasets/ds-two/sample?bytes=500")  # by name, as another depot's tokenizer recipe names it
    get(f"/api/datasets/{by_name['ds-two']}/sample")
    get("/api/tokenizers")
    get("/api/vocabularies")
    for name in ("tok", "tok-mixed", "no-such-recipe"):
        get(f"/api/tokenizers/{name}")
    get("/api/tokenizers/tok/vocabularies/1")
    get("/api/tokenizers/tok/vocabularies/1/file")
    get("/api/tokenizers/tok/vocabularies/1/texts")
    get("/api/tokenizers/tok/vocabularies/2/file")
    (out / "answers.json").write_text(json.dumps(listed, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(listed)} answers in {answers}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3]))
