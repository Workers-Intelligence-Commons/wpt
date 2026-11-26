"""Shared setup for the freight's tests: a small depot with a snapshot,
a dock, and a preparation of the tests' own, `fake`, that the freight
finds in place of an installed program (freight.exchange.find)."""

import sys
from types import SimpleNamespace

import pytest

from freight import exchange
from depot import About, Declaration, Depot, NewText

ME = Declaration("Ada Writer", "ada@example.org", True)
LICENSE = ("MIT", "MIT License", "Copyright (c) 2026 Ada Writer\n\nPermission is hereby granted, free of charge, to "
           "any person.\n")

# The program fake: weighs an item by its bytes; writes each item taken as one line of a file part-NNNNN.txt,
# closing a file once it holds per_file lines (a setting, 3 by default); answers each take with `at` {at_name (a
# setting, "line" by default): the item's line}; answers on_erasure as its settings say. Its
# file messages carry `lines` (the lines written so far, which a resume reads back) and fields named as the freight
# names its own (`bytes`, `key`, `sha256`), which the freight keeps apart from the dock's.
FAKE = r'''
import json, sys
from pathlib import Path

stdin, stdout = sys.stdin.buffer, sys.stdout.buffer


def read():
    line = stdin.readline()
    if not line:
        return None
    message = json.loads(line)
    return message, stdin.read(message["bytes"]) if message["type"] in ("weigh", "take") else None


def send(message):
    stdout.write(json.dumps(message).encode() + b"\n")


start, _ = read()
settings = start.get("settings") or {}
per_file = settings.get("per_file", 3)
at_name = settings.get("at_name", "line")
folder = Path(start["out"])
folder.mkdir(parents=True, exist_ok=True)
said = (start.get("resume") or {}).get("files") or []
number, lines, batch = len(said), said[-1]["lines"] if said else 0, []
send({"type": "ready", "name": "fake", "version": "1", "unit": "bytes-of-text", "kinds": ["plain", "conversation",
      "task"], "weighs": True, "on_erasure": settings.get("on_erasure", "stale")})
stdout.flush()


def close():
    global number, batch
    if batch:
        path = f"part-{number:05d}.txt"
        (folder / path).write_bytes(b"".join(batch))
        send({"type": "file", "path": path, "items": len(batch), "lines": lines, "bytes": 1 << 20, "key": "mine",
              "sha256": "mine"})
        number, batch = number + 1, []


while (found := read()) is not None:
    message, content = found
    if message["type"] == "weigh":
        send({"type": "weight", "ref": message["ref"], "weight": len(content)})
    elif message["type"] == "take":
        batch.append(f"{message['name']} {message['pool']} ".encode() + content.replace(b"\n", b" ") + b"\n")
        send({"type": "taken", "ref": message["ref"], "weight": len(content), "at": {at_name: lines}})
        lines += 1
        if len(batch) >= per_file:
            close()
    elif message["type"] == "end":
        close()
        send({"type": "done", "result": {"lines": lines}})
        stdout.flush()
        break
    stdout.flush()
'''


@pytest.fixture
def freight(tmp_path, monkeypatch):
    """A depot at depot (its Depot; me, the declaration its decisions take), with one snapshot of 12 plain
    texts tagged en and 6 tagged de, of several lengths; a dock, by its URL (dock); a freight home; the program fake; and request(**settings),
    the request of that snapshot with shares en 0.7 / de 0.3 and a size of 1,500 bytes
    of text, prepared by fake with those settings."""
    monkeypatch.chdir(tmp_path)  # no .env of the checkout is read
    program = tmp_path / "fake"
    program.write_text(f"#!{sys.executable}\n{FAKE}", encoding="utf-8", newline="\n")
    program.chmod(0o755)
    monkeypatch.setattr(exchange, "find", lambda name: str(program))
    depot = Depot.new(str(tmp_path / "depot"), About("fox", "Ada Writer", "ada@example.org"))
    depot.add_license(*LICENSE)
    texts = [NewText(text=f"story {n}: the fox ran. " * (2 + n % 5), licenses=("MIT",), tags=("en",))
             for n in range(12)]
    texts += [NewText(text=f"Geschichte {n}: der Fuchs lief. " * (2 + n % 3), licenses=("MIT",), tags=("de",))
              for n in range(6)]
    added = depot.add(texts, ME)
    depot.decide("admitted", f"id IN ({', '.join(repr(one) for one in added)})", ME)
    name = depot.create_snapshot()["name"]
    dock = (tmp_path / "dock").as_uri()

    def request(**settings) -> dict:
        snapshot = depot.snapshot(name)
        return {"depot": depot.name(),
                "snapshot": {"name": name, "change": snapshot["at_change"], "digest": snapshot["digest"]},
                "selection": {"shares": [{"tag": "en", "share": 0.7}, {"tag": "de", "share": 0.3}],
                              "size": {"unit": "bytes-of-text", "amount": 1500}},
                "preparation": {"name": "fake", "settings": settings, "inputs": {}}}

    return SimpleNamespace(me=ME, depot=depot, texts=added, dock=dock,
                           home=str(tmp_path / "home"), request=request, tmp_path=tmp_path)
