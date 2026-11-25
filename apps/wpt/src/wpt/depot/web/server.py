"""Serve one local web interface: its built files, the JSON routes given,
and their OpenAPI description at /api/openapi.json.

The server listens on 127.0.0.1 and answers any client, like the commands:
no login and no header to send; it checks neither the Host header nor a
request's Content-Type. A query or a filter's condition only reads the
depot's file and runs in a process of its own, stopped at a time limit
and capped in memory (readers.py); the follow routes, which a clone or a
pull of this depot reads, and the downloads of corpora, send their
answers as they are made (Stream); the long work runs in the jobs' queue
(wpt.depot.jobs); and a pull and a change of the
depot's maintainer or takedown contact are commands only (what that leaves
open: docs/architecture.md, "Two interfaces"). Each route declares the JSON
Schema of the body it takes and
of its answer: the server refuses a body that does not match with 400, and
the OpenAPI description is made from the same declarations, so it cannot
drift from the routes. An answer of the API is never kept by a browser; of
the built files, those under ASSETS, named by a hash of their content, are
kept for good, and the others read again each time.
"""

import json
import mimetypes
import re
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterator, Mapping
from urllib.parse import parse_qsl, unquote, urlsplit

import jsonschema
from wpt_utils.lines import ERROR, say

PROBLEMS = {"type": "object", "required": ["problems"],
            "properties": {"problems": {"type": "array", "items": {"type": "string"}}}}
# the built interface's folder of files whose names carry a hash of their content: a browser keeps them for good, and
# reads again every other file (index.html, which names them) each time
ASSETS = "assets"
HASHED, REVALIDATED = "public, max-age=31536000, immutable", "no-cache"
REFUSALS = {"400": "the body does not match the route's schema, or a query is not one read of the depot",
            "404": "no such text, snapshot, erasure, legacy dataset or tokenizer recipe, or no such route",
            "422": "the depot's rules refuse it, or a person's SQL ran past its time or memory, or its answer past its "
                   "size; nothing was written",
            "500": "a failure the routes do not name, whose trace the server prints; nothing was written",
            "503": "what the route needs is missing on this machine (the store), or Mozilla's translation models "
                   "cannot be read",
            "409": "a snapshot is being made already, or the same snapshot ordered by length, or an estimate "
                   "changed since it was shown",
            "429": "as many legacy text exports and samples are being sent as the web app sends at once"}


@dataclass(frozen=True)
class Stream:
    """An answer sent as it is made, chunk by chunk, of the media type kind,
    so an answer of any size never sits whole in memory; with length, its
    bytes in all (Content-Length), and with headers, more headers to send
    (Cache-Control among them in place of no-store)."""
    kind: str
    chunks: Iterator[bytes]
    length: int | None = None
    headers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    """What a route that asks for it (Route.request) gets of its request:
    its headers, the values of its query string (the last of each name),
    and the address it came from: X-Forwarded-For's first address, which
    the web server in front sets to the visitor's, else the peer's."""
    headers: Mapping[str, str]
    query: dict[str, str]
    visitor: str


@dataclass(frozen=True)
class Encoded:
    """A JSON answer encoded already, sent as these bytes."""
    data: bytes


@dataclass(frozen=True)
class Route:
    """method and pattern ("/api/texts/{id}") select the handler, which
    gets the pattern's values and the request's JSON body (None for GET) and
    returns (HTTP status, JSON payload, an Encoded payload or a Stream). summary says what it does; body is
    the JSON Schema of the body it takes (None: no body); answer the JSON
    Schema of a successful answer; with request, the handler gets the
    Request too, as a third argument; query names the query string's
    values it reads, each with its description."""
    method: str
    pattern: str
    handler: Callable[..., tuple[int, object]]
    summary: str = ""
    body: dict | None = None
    answer: dict | None = None
    request: bool = False
    query: dict = field(default_factory=dict)

    def match(self, method, path):
        if method != self.method:
            return None
        regex = re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", re.escape(self.pattern))
        found = re.fullmatch(regex, path)
        return {k: unquote(v) for k, v in found.groupdict().items()} if found else None


def openapi(routes: list[Route]) -> dict:
    """The OpenAPI 3.1 description of the routes."""
    paths = {}
    for route in routes:
        operation = {"summary": route.summary,
                     "responses": {"200": {"description": "done",
                                           "content": {"application/json": {"schema": route.answer or {}}}},
                                   **{status: {"description": why,
                                               "content": {"application/json": {"schema": PROBLEMS}}}
                                      for status, why in REFUSALS.items()}}}
        names = re.findall(r"\{(\w+)\}", route.pattern)
        parameters = [{"name": name, "in": "path", "required": True, "schema": {"type": "string"},
                       "description": "percent-encoded; a path's '/' is %2F"} for name in names]
        parameters += [{"name": name, "in": "query", "required": False, "schema": {"type": "string"},
                        "description": why} for name, why in route.query.items()]
        if parameters:
            operation["parameters"] = parameters
        if route.body is not None:
            operation["requestBody"] = {"required": True, "content": {"application/json": {"schema": route.body}}}
        paths.setdefault(route.pattern, {})[route.method.lower()] = operation
    return {"openapi": "3.1.0",
            "info": {"title": "wpt depot web", "version": "1",
                     "description": "The depot WPT_DEPOT names, read and changed through wpt_depot, as the "
                                    "`wpt depot` commands do: the depot applies its rules, and a refused change "
                                    "writes nothing. docs/architecture.md, \"Two interfaces\"."},
            "paths": paths}


def make_server(routes: list[Route], static: Path, port: int = 0) -> ThreadingHTTPServer:
    """A server on 127.0.0.1:port (0: any free port), not started yet:
    call serve_forever(). OSError if the port is taken."""
    static = Path(static).resolve()
    routes = [*routes, Route("GET", "/api/openapi.json", lambda values, body: (200, description),
                             "this description of every route", answer={"type": "object"})]
    description = openapi(routes)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._answer("GET")

        def do_PUT(self):
            self._answer("PUT")

        def do_POST(self):
            self._answer("POST")

        def _answer(self, method):
            parts = urlsplit(self.path)
            path = parts.path
            if path.startswith("/api/"):
                return self._api(method, path, parts.query)
            if method != "GET":
                return self._json(405, {"problems": ["only GET outside /api/"]})
            self._file(path)

        def _api(self, method, path, query=""):
            body = None
            if method != "GET":
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
                except (ValueError, UnicodeDecodeError) as error:
                    return self._json(400, {"problems": [f"the body is not JSON: {error}"]})
            for route in routes:
                values = route.match(method, path)
                if values is not None:
                    if route.body is not None:
                        error = jsonschema.exceptions.best_match(
                            jsonschema.Draft202012Validator(route.body).iter_errors(body))
                        if error is not None:
                            where = "/".join(map(str, error.absolute_path)) or "(top level)"
                            return self._json(400, {"problems": [f"the body, at {where}: {error.message}"]})
                    try:
                        if route.request:
                            forwarded = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
                            answer = route.handler(values, body, Request(
                                self.headers, dict(parse_qsl(query)), forwarded or self.client_address[0]))
                        else:
                            answer = route.handler(values, body)
                    except Exception as error:  # answered, so that the page shows it, and its trace printed
                        say(traceback.format_exc().rstrip(), ERROR)
                        return self._json(500, {"problems": [f"{type(error).__name__}: {error}"]})
                    if isinstance(answer[1], Stream):
                        return self._stream(*answer)
                    if isinstance(answer[1], Encoded):
                        return self._send(answer[0], answer[1].data, "application/json")
                    return self._json(*answer)
            self._json(404, {"problems": [f"no route {method} {path}"]})

        def _file(self, path):
            target = (static / path.lstrip("/")).resolve()
            if target == static or target.is_dir():
                target = target / "index.html"
            if not target.is_relative_to(static) or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            hashed = target.parent == static / ASSETS
            self._send(200, target.read_bytes(), kind, HASHED if hashed else REVALIDATED)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

        def _stream(self, status, stream: Stream):
            """Send the stream's chunks as they come, the connection closed at its end."""
            self.close_connection = True
            try:
                self.send_response(status)
                self.send_header("Content-Type", stream.kind)
                if "Cache-Control" not in stream.headers:
                    self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                if stream.length is not None:
                    self.send_header("Content-Length", str(stream.length))
                for name, value in stream.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                for chunk in stream.chunks:
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):  # the client went away: the stream stops there
                pass
            except Exception:  # the headers are sent: the client sees the answer cut short
                say(traceback.format_exc().rstrip(), ERROR)
            finally:
                close = getattr(stream.chunks, "close", None)
                if close is not None:  # a generator's own cleanup runs now, not whenever it is collected
                    close()

        def _send(self, status, data, kind, caching="no-store"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", caching)
            self.end_headers()
            self.wfile.write(data)

        def log_request(self, code="-", size="-"):  # the terminal shows only refused requests
            if str(code).isdigit() and int(code) >= 400:
                super().log_request(code, size)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
