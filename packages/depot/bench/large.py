"""A depot of format 12 far bigger than the memory the upgrade to format 13
gets, laid out as a depot that lived is: the faithful test of `depot
upgrade --check` and `--yes` where random reads cost what they cost on a
disk (bench/upgrade.py measures them on a depot the page cache holds whole).

It starts from the fixture depot of format 12 (tests/fixtures), keeps its
layout and about row, deletes its rows and writes, in plain SQL, as the
code of format 12 laid them out:

- `texts` texts, added in changes of ADD_BATCH, 90% at first and the rest
  later; every text then gets new revisions in WAVES, each wave a share of
  the texts in changes of WAVE_BATCH, the texts of a wave in the order of
  their numbers or shuffled (a scores file in id order, a review in any
  order), so the revisions of one text lie far apart in the file, about
  REVISIONS revisions a text in all; two waves give new content versions;
- every content version a row of content_lists and its parts, the parts'
  values of random bytes (`filler` bytes in all), written in the same
  transactions as the revisions, so the pages of texts_revisions,
  content_lists and parts interleave in the file as they do in a depot that
  grew; one text in LONG_EVERY has LONG_PARTS parts, so its row of
  content_lists overflows its page;
- two locked libraries, wic-1 (about 75% of the admitted texts) locked
  after the wave LOCK_1 and wic-2 (about 85%, most of wic-1) after the wave
  LOCK_2, their memberships and their pins at each text's revision then;
  an open library open-a (30%, a tenth of them left it); the dataset ds-both
  over wic-1 then wic-2, ds-one over wic-1 alone and ds-open over open-a.

    uv run python packages/depot/bench/large.py --texts 2500000 --out .bench-depots/large/base.sqlite

Run it outside any memory limit: it holds a few arrays of every text in
memory and gives SQLite a page cache of 4 GB. Then measure the check and
the upgrade on a copy, on a disk (not tmpfs), with the file dropped from
the page cache (`dd if=COPY iflag=nocache count=0`, or posix_fadvise) and a
memory limit far below the file, the temporary folder on the same disk:

    cp .bench-depots/large/base.sqlite .bench-depots/large/work.sqlite
    DEPOT=.bench-depots/large/work.sqlite SQLITE_TMPDIR=.bench-depots/tmp \\
      systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 uv run depot upgrade --yes

With --format 11 the depot is taken back to format 11 at the end
(bench/upgrade.py's older), its conversations' parts holding JSON whose
excerpts quote other texts, for the step to format 12. On 2,500,000 texts
(18,322,341 revisions, 21 GB) of format 11 the check took 79 to 87
seconds, and `--yes` after a saved check 294 (98 for the step to format
12, 194 for the step to format 13), each stage saying its seconds.
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import apsw
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from upgrade import older  # noqa: E402

from depot import ids
from depot.parts import pack

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "format-12.sqlite"
OWNER = "fixture"
SEED = 20261122
ADD_BATCH = 20_000
WAVE_BATCH = 100_000
# (share of the texts it changes, what it changes, whether its texts come shuffled)
WAVES = ((0.9, "decision", True), (0.8, "tags", False), (0.5, "content", True), (0.9, "scores", False),
         (0.6, "tags", True), (0.9, "scores", False), (0.4, "content", True), (0.6, "tags", False),
         (1.0, "scores", True))
LOCK_1, LOCK_2, LATE = 3, 5, 4  # after which wave wic-1 and wic-2 lock; before which wave the late texts come
REVISIONS = 1 + sum(one[0] for one in WAVES)
LONG_EVERY, LONG_PARTS = 1000, 150
TAGS = tuple(f"{word}-{n}" for word in ("wikisource", "gutenberg", "hansard", "europarl", "manual") for n in range(10))
KINDS = ("plain",) * 46 + ("conversation",) * 3 + ("task",)
STATUSES = ("under review", "admitted", "refused")
STATUS_COUNTS = ("INSERT INTO counts (facet, name, texts) SELECT 'status', status, count(*) FROM texts_revisions "
                 "WHERE latest GROUP BY status")
CLEARED = ("texts_revisions", "content_lists", "parts", "libraries_revisions", "memberships", "pins",
           "datasets_revisions", "tokenizers_revisions", "vocabularies", "vocabulary_texts", "erasures_revisions",
           "text_decisions", "decisions", "counts", "quotes", "text_tags", "text_licenses", "pulls", "library_moves",
           "notices", "title_owners")


def say(*parts):
    print(time.strftime("%H:%M:%S"), *parts, file=sys.stderr, flush=True)


def tags(rng: random.Random) -> list[str]:
    count = rng.choices((1, 2, 3, 4), weights=(50, 30, 15, 5))[0]
    return sorted(set(rng.choices(TAGS, weights=[1 / (1 + n) ** 0.8 for n in range(len(TAGS))], k=count)))


class Depot:
    """The rows of the depot file being made, each change its own transaction, and what each text is at."""

    def __init__(self, path: Path, n: int, filler: float):
        self.con = apsw.Connection(str(path))
        self.con.create_scalar_function("export_key", ids.export_key, 1, deterministic=True)
        for pragma in ("journal_mode = WAL", "synchronous = OFF", "cache_size = -4000000"):
            self.con.execute(f"PRAGMA {pragma}").fetchall()
        self.change = self.con.execute("SELECT max(id) FROM changes").fetchall()[0][0]
        self.rng = random.Random(SEED)
        versions = n * (1 + sum(share for share, what, _ in WAVES if what == "content"))
        self.part_bytes = max(64, int(filler / versions))
        self.revision = np.zeros(n + 1, np.int32)
        self.version = np.zeros(n + 1, np.int32)
        self.tag_epoch = np.zeros(n + 1, np.int32)
        self.scored = np.zeros(n + 1, bool)
        self.decided = np.zeros(n + 1, bool)

    def begin(self, kind: str) -> int:
        self.change += 1
        self.con.execute("BEGIN")
        self.con.execute("INSERT INTO changes (id, kind, made_at) VALUES (?, ?, '2026-01-01T00:00:00+00:00')",
                         (self.change, kind))
        return self.change

    def commit(self):
        self.con.execute("COMMIT")

    def content(self, number: int, change: int):
        text = f"{OWNER}-{number}"
        self.version[number] += 1
        many = LONG_PARTS if number % LONG_EVERY == 0 else 1
        parts, rows, length = [], [], 0
        kind = KINDS[number % len(KINDS)]
        for _ in range(many):
            size = max(16, int(self.rng.lognormvariate(0, 0.6) * self.part_bytes / many))
            digest = self.rng.randbytes(32)
            if kind == "conversation":  # read whole by the step to format 12, for the texts its excerpts quote
                quoted = [f"{OWNER}-{self.rng.randrange(1, number + 1)}" for _ in range(self.rng.randrange(3))]
                value = json.dumps({"messages": [{"role": "user", "content": [
                    {"type": "text", "text": self.rng.randbytes(size // 2).hex()},
                    *({"type": "excerpt", "source": one, "start": 0, "end": 10} for one in quoted)]}]})
                parts.append({"hash": digest.hex(), "length": len(value)})
                rows.append((text, digest, change, pack(value)))
                length += len(value)
                continue
            parts.append({"hash": digest.hex(), "length": size * 3})
            rows.append((text, digest, change, self.rng.randbytes(size)))
            length += size * 3
        self.con.execute("INSERT INTO content_lists (id, version, change, kind, parts, length, bytes) VALUES "
                         "(?, ?, ?, ?, ?, ?, ?)", (text, int(self.version[number]), change,
                                                    kind, json.dumps(parts), length, length))
        self.con.executemany("INSERT INTO parts (id, hash, change, value) VALUES (?, ?, ?, ?)", rows)

    def write(self, number: int, change: int, latest: bool, status: int):
        """A new revision of the text, as it is now."""
        self.revision[number] += 1
        one = random.Random(number * 1000 + int(self.tag_epoch[number]))
        scores = ([{"name": "symbols", "value": round(self.rng.random(), 4),
                    "content_version": int(self.version[number])},
                   {"name": "running-text", "value": round(self.rng.random(), 4),
                    "content_version": int(self.version[number])}] if self.scored[number] else [])
        decided = bool(self.decided[number])
        self.con.execute(
            "INSERT INTO texts_revisions (id, revision, change, latest, licenses, notice, title, authors, tags, "
            "address, decisions, content_version, scores, status) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, ?, "
            "?, ?)", (f"{OWNER}-{number}", int(self.revision[number]), change, int(latest),
                      json.dumps(["CC0-1.0"] if number % 9 else ["CC-BY-4.0"]),
                      f"Synthetic text {number} of the large benchmark", json.dumps([f"Author {number % 5000}"]),
                      json.dumps(tags(one)), json.dumps([f"{OWNER}-{number}"] if decided else []),
                      int(self.version[number]), json.dumps(scores), STATUSES[status] if decided else "under review"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--texts", type=int, default=2_500_000)
    parser.add_argument("--filler", type=float, default=8e9, help="bytes of the parts' values in all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format", type=int, choices=(11, 12), default=12,
                        help="11: taken back to format 11 at the end, as bench/upgrade.py does it")
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    for path in out.parent.glob(out.name + "*"):
        path.unlink()
    shutil.copyfile(FIXTURE, out)
    began = time.monotonic()
    n = args.texts
    late = int(n * 0.9)
    np_rng = np.random.default_rng(SEED)
    # which waves change each text (a text added late is changed only from the wave LATE on, decided then)
    changed = np.stack([np_rng.random(n) < share for share, _, _ in WAVES], axis=1)
    changed[late:, :LATE] = False
    changed[late:, LATE] = True
    status = np.where(np_rng.random(n) < 0.85, 1, np.where(np_rng.random(n) < 0.33, 2, 0))
    pick = np_rng.random(n)
    remaining = changed.sum(axis=1)  # waves left that change each text: its last revision is the latest
    d = Depot(out, n, args.filler)
    con = d.con
    con.execute("BEGIN")
    for table in CLEARED:
        con.execute(f"DELETE FROM {table}")
    con.execute("INSERT INTO text_titles(text_titles) VALUES ('delete-all')")
    con.execute("UPDATE about_revisions SET upstream = NULL")
    con.execute("COMMIT")

    def add(numbers):
        change = d.begin("add")
        for number in numbers:
            d.content(number, change)
            d.write(number, change, remaining[number - 1] == 0, status[number - 1])
        d.commit()

    def lock(name: str, number: int, share: float, joined: int) -> tuple[str, int]:
        library = f"{OWNER}-{number}"
        change = d.begin("lock")
        con.execute("INSERT INTO libraries_revisions (id, revision, change, latest, name, locked_at) VALUES "
                    "(?, 1, ?, 0, ?, NULL), (?, 2, ?, 1, ?, '2026-01-01T00:00:00Z')",
                    (library, joined, name, library, change, name))
        chosen = np.flatnonzero((d.revision[1:] > 0) & (status == 1) & (pick < share)) + 1
        con.executemany("INSERT INTO memberships (library, text, joined_change, export_key) VALUES "
                        "(?, ?, ?, export_key(?))", ((library, f"{OWNER}-{one}", joined, f"{OWNER}-{one}")
                                                     for one in chosen.tolist()))
        con.executemany("INSERT INTO pins (library, text, revision, change, export_key) VALUES (?, ?, ?, ?, "
                        "export_key(?))", ((library, f"{OWNER}-{one}", int(d.revision[one]), change, f"{OWNER}-{one}")
                                           for one in chosen.tolist()))
        d.commit()
        say(f"  {name} locked at change {change}: {len(chosen):,} pins")
        return library, change

    for first in range(1, late + 1, ADD_BATCH):
        add(range(first, min(late, first + ADD_BATCH - 1) + 1))
    say(f"  added {late:,} texts ({time.monotonic() - began:.0f} s)")
    joined = d.change
    locks, open_joined = {}, []
    for wave, (share, what, shuffled) in enumerate(WAVES):
        if wave == LATE:
            for first in range(late + 1, n + 1, ADD_BATCH):
                add(range(first, min(n, first + ADD_BATCH - 1) + 1))
            say(f"  added {n - late:,} late texts")
        numbers = (np.flatnonzero(changed[:, wave]) + 1).tolist()
        if shuffled:
            d.rng.shuffle(numbers)
        for first in range(0, len(numbers), WAVE_BATCH):
            change = d.begin("content" if what == "content" else "put")
            for number in numbers[first:first + WAVE_BATCH]:
                if what == "decision" or (wave == LATE and number > late):
                    d.decided[number] = True
                elif what == "content":
                    d.content(number, change)
                elif what == "tags":
                    d.tag_epoch[number] += 1
                else:
                    d.scored[number] = True
                remaining[number - 1] -= 1
                d.write(number, change, remaining[number - 1] == 0, status[number - 1])
            d.commit()
        say(f"  wave {wave + 1}: {len(numbers):,} revisions, change {d.change} ({time.monotonic() - began:.0f} s)")
        if wave + 1 == LOCK_1:
            locks["wic-1"] = lock("wic-1", 1, 0.75, joined)
        if wave + 1 == LOCK_2:
            locks["wic-2"] = lock("wic-2", 2, 0.85, locks["wic-1"][1])
        if wave + 1 in (2, 6):
            open_joined.append(d.change)
    library = f"{OWNER}-3"
    change = d.begin("library")
    con.execute("INSERT INTO libraries_revisions (id, revision, change, latest, name) VALUES (?, 1, ?, 1, 'open-a')",
                (library, open_joined[0]))
    con.executemany("INSERT INTO memberships (library, text, joined_change, left_change, export_key) VALUES "
                    "(?, ?, ?, ?, export_key(?))",
                    ((library, f"{OWNER}-{one}", open_joined[0] if one <= late else open_joined[1],
                      change if one % 10 == 0 else None, f"{OWNER}-{one}")
                     for one in (np.flatnonzero((pick >= 0.4) & (pick < 0.7)) + 1).tolist()))
    for number, (name, refs) in enumerate((("ds-both", [locks["wic-1"][0], locks["wic-2"][0]]),
                                           ("ds-one", [locks["wic-1"][0]]), ("ds-open", [library])), start=1):
        con.execute("INSERT INTO datasets_revisions (id, revision, change, latest, name, tokenizer, libraries) "
                    "VALUES (?, 1, ?, 1, ?, 'none', ?)",
                    (f"{OWNER}-{number}", change, name,
                     json.dumps([{"depot": None, "library": one, "change": change - 1} for one in refs])))
    found = {}
    for (tagged,) in con.execute("SELECT tags FROM texts_revisions INDEXED BY texts_latest WHERE latest"):
        for tag in json.loads(tagged):
            found[tag] = found.get(tag, 0) + 1
    con.executemany("INSERT INTO counts (facet, name, texts) VALUES ('tag', ?, ?)", list(found.items()))
    con.execute(STATUS_COUNTS)
    con.executemany("INSERT OR REPLACE INTO numbers (name, last) VALUES (?, ?)",
                    [("texts", n), ("libraries", 3), ("datasets", 3), ("decisions", n)])
    con.execute("UPDATE about_revisions SET change = ? WHERE revision = (SELECT max(revision) FROM about_revisions)",
                (change,))
    d.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    con.execute("PRAGMA journal_mode = DELETE").fetchall()
    con.close()
    if args.format == 11:
        older(out)
    say(f"made {out} in {time.monotonic() - began:.0f} s: {out.stat().st_size / 1e9:.1f} GB, "
        f"about {REVISIONS:.1f} revisions a text")


if __name__ == "__main__":
    main()
