"""Write a small synthetic depot of the format of the depot it runs with (11 or 12), and the dock of its
corpora, for the upgrade tests and the legacy API's golden answers (README.md beside this file says how it is run).

    python make_fixtures.py OUT_FOLDER

writes OUT_FOLDER/format-<N>.sqlite and OUT_FOLDER/format-<N>-store/ (one corpus folder per build); neither may
exist yet. Every text is made of words drawn from a fixed list with a fixed seed; no text is anyone's writing. The
steps, in order:

    the upstream "up": 60 texts, 50 admitted; library up-a (40 texts) locked
    the depot "fixture" clones up (its follow routes served on 127.0.0.1:UPSTREAM_PORT)
    up then adds 10 texts, admits them, locks up-b; fixture pulls
    fixture adds 240 plain texts (tags, two licenses), 6 conversations (format 12: 2 quote a plain text), 2 tasks
    fixture admits most, refuses some, leaves the rest under review
    lib-a (60 texts) locked at L_A; then one of its texts gets a new content version (edited after the lock)
    lib-b (50 texts) locked at L_B > L_A
    lib-c (40 texts) stays open; a problem reported on lib-a (two texts) and on lib-c (whole library)
    lib-d forked from lib-a, open
    dataset ds-two reads lib-a and lib-b (two libraries locked at different changes), shares by tag
    dataset ds-mixed reads lib-a (its ref's change above L_A), lib-c (open) and up-b (the upstream's)
    tokenizer recipe tok (version 1 trained, edited to version 2 untrained), recipe tok-mixed untrained
    an erasure of a passage of a lib-a text, admitted and carried out; a second request left under review
    ds-two publishes tok's vocabulary; corpus built at SMALL tokens (ds-two revision 2)
    ds-two edited (shares changed, revision 3); corpus built at LARGE tokens (revision 3)
So the dock holds a corpus naming an older revision of ds-two and one naming its current revision.
"""

import base64
import json
import random
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import apsw

from depot import About, Declaration, Depot, NewText, follow
from corpus_build import build
from depot.schema import FORMAT
from dock import open_dock
from wpt_tokenizer import Tokenizer

ME = Declaration("Ada Lovelace", "ada@example.org", True)
UPSTREAM_PORT = 8531  # recorded in the clone's about.upstream, so the fixture names the same address each time
VOCAB_SIZE = 556
SMALL, LARGE = 2000, 6000  # the sizes of the two corpora, in tokens
WORDS = ("river stone lamp field orchard bread window harbor letter garden morning winter candle market bridge "
         "meadow forest kettle ladder pocket mirror thread basket valley cloud silver copper paper engine "
         "the a of and to in on under over with by from near walks sings reads keeps builds carries finds").split()
TAGS = ("en", "fr", "poem", "letter", "manual")
MIT = ("MIT", "MIT License", "Copyright (c) 2026 Ada Lovelace\n\nPermission is hereby granted, free of charge.\n")
CC0 = ("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.\n")


def words(rng: random.Random, count: int) -> str:
    sentences, left = [], count
    while left > 0:
        size = min(left, rng.randint(5, 12))
        sentence = " ".join(rng.choice(WORDS) for _ in range(size))
        sentences.append(sentence[0].upper() + sentence[1:] + ".")
        left -= size
    return " ".join(sentences)


def plain(rng: random.Random, number: int, license_id: str) -> NewText:
    tags = tuple(sorted(rng.sample(TAGS, rng.randint(1, 2))))
    return NewText(text=f"Text {number}. {words(rng, rng.randint(25, 70))}", licenses=(license_id,),
                   title=f"Synthetic text {number}", authors=("Ada Lovelace",), tags=tags)


def conversation(rng: random.Random, quoted: tuple[str, str] | None) -> NewText:
    asked = [{"type": "text", "text": words(rng, 8)}]
    if quoted is not None:  # an excerpt block (format 12): a passage of a plain text the depot holds
        source, exact = quoted
        asked.append({"type": "excerpt", "source": source, "selector": {"type": "TextQuoteSelector", "exact": exact}})
    content = {"messages": [{"role": "user", "content": asked},
                            {"role": "assistant", "content": [{"type": "text", "text": words(rng, 12)}]}]}
    return NewText(kind="conversation", content=json.dumps(content), licenses=("CC0-1.0",), tags=("chat",))


def task(rng: random.Random, number: int) -> NewText:
    check = {"mode": "100755", "content": base64.b64encode(b"#!/bin/sh\necho 1\n").decode()}
    content = {"task": {"instructions": f"Task {number}. {words(rng, 10)}", "image": "fixture", "needs": [],
                        "limits": {"max_turns": 2, "max_tokens": 6, "command_seconds": 5}},
               "files": {}, "check": check}
    return NewText(kind="task", content=json.dumps(content), licenses=("CC0-1.0",), tags=("task",))


def ids_in(found: list[str]) -> str:
    return f"id IN ({', '.join(repr(one) for one in found)})"


def serve(depot: Depot) -> ThreadingHTTPServer:
    """The follow routes over depot, as `depot web` serves them, on 127.0.0.1:UPSTREAM_PORT."""
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

    server = ThreadingHTTPServer(("127.0.0.1", UPSTREAM_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def make(out: Path):
    work = Path(tempfile.mkdtemp(prefix=f"wpt-fixture-{FORMAT}-"))  # the upstream, the clone and their change files
    rng = random.Random(FORMAT * 1000 + 7)

    up = Depot.new(str(work / "up.sqlite"), About("up", "Ada Lovelace", "ada@example.org"))
    up.add_license(*CC0)
    up_texts = up.add([plain(rng, n, "CC0-1.0") for n in range(60)], ME)
    up.decide("admitted", ids_in(up_texts[:50]), ME)
    up.decide("refused", ids_in(up_texts[50:55]), ME, reason="not clear enough")
    up_a = up.new_library("up-a")
    up.put(up_a, ids_in(up_texts[:40]))
    up.lock(up_a)

    server = serve(up)
    try:
        address = f"http://127.0.0.1:{UPSTREAM_PORT}"
        depot = follow.clone(address, str(work / "fixture.sqlite"),
                             About("fixture", "Ada Lovelace", "ada@example.org"))
        more = up.add([plain(rng, n, "CC0-1.0") for n in range(60, 70)], ME)
        up.decide("admitted", ids_in(more), ME)
        up_b = up.new_library("up-b")
        up.put(up_b, ids_in(up_texts[40:50] + more))
        up.lock(up_b)
        follow.pull(depot)
    finally:
        server.shutdown()
        server.server_close()

    depot.add_license(*MIT)
    own = depot.add([plain(rng, n, "MIT" if n % 3 else "CC0-1.0") for n in range(1000, 1240)], ME)
    admitted, refused = own[:200], own[200:215]
    depot.decide("admitted", ids_in(admitted), ME)
    depot.decide("refused", ids_in(refused), ME, reason="a license that does not fit")
    texts = {one["id"]: one for one in depot.sql(f"SELECT id FROM texts WHERE {ids_in(admitted[:6])}")}
    quoted = []
    if FORMAT >= 12:  # excerpts exist from format 12
        for text in list(texts)[:2]:
            body = depot.content(text)
            quoted.append((text, body[:body.index(".") + 1]))  # "Text N.", once in its text
    items = [conversation(rng, quoted[n] if n < len(quoted) else None) for n in range(6)]
    items += [task(rng, n) for n in range(2)]
    made = depot.add(items, ME)
    depot.decide("admitted", ids_in(made), ME)

    lib_a = depot.new_library("lib-a")
    depot.put(lib_a, ids_in(admitted[:60] + made[:3]))
    depot.lock(lib_a)
    edited = admitted[5]  # a text lib-a pinned, edited after the lock, then admitted again
    depot.change_content(edited, f"Text edited. {words(rng, 30)}", None, ME)
    depot.decide("admitted", ids_in([edited]), ME)

    lib_b = depot.new_library("lib-b")
    depot.put(lib_b, ids_in(admitted[60:110] + made[3:]))
    depot.lock(lib_b)

    lib_c = depot.new_library("lib-c")
    depot.put(lib_c, ids_in(admitted[110:150]))
    depot.report(lib_a, "two texts carry the same passage", ids_in(admitted[10:12]), ME)
    depot.report(lib_c, "the library mixes two languages", None, ME)
    depot.fork(lib_a, "lib-d", without=ids_in(admitted[:5]))

    two = depot.new_dataset("ds-two", "tok", [(None, lib_a), (None, lib_b)], [("en", 0.5), ("poem", None)])
    depot.new_dataset("ds-mixed", "tok-mixed", [(None, lib_a), (None, lib_c), (address, up_b)])

    tok = depot.new_tokenizer("tok", None, "ds-two", 10**6, VOCAB_SIZE, note="the fixture's vocabulary")
    trained = [one["id"] for one in depot.sql("SELECT id FROM texts WHERE status = 'admitted' AND kind = 'plain' "
                                               "ORDER BY id LIMIT 80")] if FORMAT >= 12 else admitted[:80]
    bodies = [depot.content(one).encode() for one in trained]
    vocabulary = Tokenizer.train(bodies, vocab_size=VOCAB_SIZE)
    sha = depot.add_vocabulary(tok, 1, vocabulary.to_bytes(), [("", one) for one in trained], sum(map(len, bodies)))
    depot.edit_tokenizer(tok, None, "ds-two", 2 * 10**6, VOCAB_SIZE, note="a larger sample, not trained")
    depot.new_tokenizer("tok-mixed", None, "ds-mixed", 10**6, VOCAB_SIZE)

    target = admitted[20]  # held by lib-a, so the erasure reaches a locked library's pin
    request = depot.ask_erasure("a passage its author asked to remove", [(target, 0, 7)], ME)
    depot.decide_erasure(request, "admitted", ME)
    depot.ask_erasure("asked, not yet decided", [(admitted[120], 0, 4)], ME)

    row = depot.dataset_by_id(two)
    libraries = [(None, one["library"]) for one in row["libraries"]]
    depot.edit_dataset(two, "ds-two", libraries, [("en", 0.5), ("poem", None)], vocabulary=sha)
    store = open_dock((out / f"format-{FORMAT}-store").resolve().as_uri())
    quiet = lambda *_: None  # noqa: E731
    build(depot, "ds-two", store, vocabulary, 1, max_tokens=SMALL, log=quiet)
    depot.edit_dataset(two, "ds-two", libraries, [("en", 0.3), ("letter", None)], vocabulary=sha)
    build(depot, "ds-two", store, vocabulary, 1, max_tokens=LARGE, log=quiet)

    final = out / f"format-{FORMAT}.sqlite"
    con = apsw.Connection(str(work / "fixture.sqlite"))
    con.execute(f"VACUUM INTO '{final}'")
    con.close()
    print(f"{final}: format {FORMAT}, change {depot.change()}, {len(depot.sql('SELECT id FROM texts'))} texts")
    shutil.rmtree(work)


if __name__ == "__main__":
    make(Path(sys.argv[1]))
