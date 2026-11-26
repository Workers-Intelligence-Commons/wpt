"""What the tests of the corpus build (support/corpus_build.py) start
from: a new depot, the decider's declaration, and the follow routes of a
depot served as its web app serves them, for a build over a clone."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from depot import About, Declaration, Depot, follow

ME = Declaration("Ada Lovelace", "ada@example.org", True)


@pytest.fixture
def me():
    return ME


@pytest.fixture
def make_depot(tmp_path):
    """make_depot(name): a new depot in the test's folder, with the license
    CC0-1.0."""
    def make(name="depot"):
        depot = Depot.new(str(tmp_path / f"{name}.sqlite"), About(name, "Ada Lovelace", "ada@example.org"))
        depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
        return depot
    return make


@pytest.fixture
def serve():
    """serve(depot): the web address of a server answering the follow routes
    over the depot, as the web app's do (depot.follow), stopped after
    the test."""
    servers = []

    def start(depot: Depot) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                route = self.path.rsplit("/", 1)[-1]
                if route == "head":
                    data = [json.dumps(follow.head(depot)).encode()]
                elif route == "rows":
                    data = follow.rows(depot, body["table"], body.get("ranges"), body.get("after"), body.get("texts"))
                else:
                    data = follow.parts(depot, body["keys"])
                self.send_response(200)
                self.end_headers()
                for chunk in data:
                    self.wfile.write(chunk)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()

