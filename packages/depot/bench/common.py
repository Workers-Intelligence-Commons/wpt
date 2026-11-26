"""What every benchmark of packages/depot/bench shares: the synthetic
depot it measures, the timing of one operation warm and cold, and the
report.

The synthetic depot is made by this code from a seed, never from a real
depot: `texts` texts of the kinds plain (90%), conversation (8%) and task
(2%), lengths drawn from a log-normal law around 600 characters with a few
texts past schema.LONG, 50 tags (t00 to t49) given one to four at a time
with skewed weights and correlated pairs, so tag sets overlap, two
licenses, 85% of the texts admitted, 5% refused, 10% under review, and
the rows of the legacy tables a depot of format 12 held (legacy_library):
three open libraries, two locked ones, and a small library with a dataset
over it, whose texts a build reads alone. It is made once per size under ROOT/<texts>/base.sqlite
and kept (the file `made` beside it names GENERATOR, so a change of the
generator makes it again); each benchmark measures a fresh copy of it,
ROOT/<texts>/work.sqlite, so the base stays as it was made. A base made by
the code of an older format is kept as it is and each work copy is brought
up to the code's format first (brought_up), the time it takes reported.

A cold run closes every connection, writes the depot's files to disk
(fsync) and asks the kernel to drop their pages from the cache
(posix_fadvise DONTNEED, no root needed), then opens the depot again
before the timed operation. On a tmpfs (/tmp on many machines) the files
are memory and nothing can be dropped: the report says so, and cold runs
equal warm ones there. A warm run follows an untimed run of the same
operation. Each figure is the median of `runs` runs.
"""

import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from depot import About, Declaration, Depot, NewText

DEFAULT_ROOT = Path("/tmp/claude-1000/wpt-bench")
GENERATOR = 1  # bump when the synthetic depot changes, so a kept base is made again
NAME = "bench"
ME = Declaration("Bench Person", "bench@example.org", True)
TAGS = tuple(f"t{n:02d}" for n in range(50))
LICENSES = ("CC0-1.0", "CC-BY-4.0")
SEED = 20261121
BATCH = 2000  # texts per add while the depot is made
LONG_EVERY = 40_000  # one text in this many is past schema.LONG (1 MiB characters)


@dataclass
class Layout:
    """What the made depot holds, read back by the benchmarks (saved as JSON beside the base)."""
    texts: int
    open_libraries: dict = field(default_factory=dict)  # name -> id
    locked_libraries: dict = field(default_factory=dict)
    small_library: str = ""
    dataset: str = "small"
    seconds: float = 0.0


def text_id(number: int) -> str:
    return f"{NAME}-{number}"


def status_of(number: int) -> str:
    """The status the made depot gives text number (1-based)."""
    rest = number % 20
    return "refused" if rest == 0 else "under review" if rest in (1, 2) else "admitted"


def kind_of(number: int) -> str:
    rest = number % 50
    return "task" if rest == 7 else "conversation" if rest in (3, 13, 23, 33) else "plain"


class Words:
    """A long string of pseudo-words, sliced into texts of any length."""

    def __init__(self, rng: random.Random, size: int = 16 << 20):
        letters = "etaoinshrdlucmfwypvbgkjqxz"
        vocabulary = ["".join(rng.choices(letters, weights=range(26, 0, -1), k=rng.randint(1, 10)))
                      for _ in range(8000)]
        pieces, length = [], 0
        while length < size:
            line = " ".join(rng.choices(vocabulary, k=rng.randint(6, 16))) + ".\n"
            pieces.append(line)
            length += len(line)
        self.blob = "".join(pieces)
        self.rng = rng

    def text(self, length: int) -> str:
        if length >= len(self.blob):
            return (self.blob * (length // len(self.blob) + 1))[:length]
        start = self.rng.randrange(0, len(self.blob) - length)
        return self.blob[start:start + length]


def length_of(rng: random.Random, number: int) -> int:
    if number % LONG_EVERY == 0:
        return 1_100_000
    return max(20, min(200_000, int(rng.lognormvariate(math.log(600), 1.2))))


def tags_of(rng: random.Random) -> tuple[str, ...]:
    """One to four tags, skewed toward the first ones, with pairs that come together (t10 with t11, t20 with t21...)."""
    count = rng.choices((1, 2, 3, 4), weights=(50, 30, 15, 5))[0]
    chosen = set(rng.choices(TAGS, weights=[1 / (1 + n) ** 0.8 for n in range(len(TAGS))], k=count))
    for low in (10, 20, 30):
        if f"t{low}" in chosen and rng.random() < 0.7:
            chosen.add(f"t{low + 1}")
    return tuple(sorted(chosen))


def conversation(rng: random.Random, words: Words) -> str:
    messages = []
    for turn in range(rng.randint(1, 4)):
        messages.append({"role": "user", "content": [{"type": "text", "text": words.text(rng.randint(30, 400))}]})
        messages.append({"role": "assistant", "content": [{"type": "text",
                                                           "text": words.text(rng.randint(50, 1500))}]})
    return json.dumps({"messages": messages})


def task(rng: random.Random, words: Words) -> str:
    import base64
    encoded = base64.b64encode(words.text(rng.randint(50, 600)).encode()).decode("ascii")
    return json.dumps({"task": {"instructions": words.text(rng.randint(40, 800)), "image": "base",
                                "limits": {"max_turns": 4, "max_tokens": 512, "command_seconds": 10}, "needs": []},
                       "files": {"notes.txt": {"mode": "100644", "content": encoded}},
                       "check": {"mode": "100755", "content": "IyEvYmluL3NoCmVjaG8gMQo="}})


def new_text(rng: random.Random, words: Words, number: int) -> NewText:
    kind = kind_of(number)
    licenses = (LICENSES[1],) if number % 9 == 0 else (LICENSES[0],)
    credits = {"licenses": licenses, "title": f"Text {number} {words.text(20).strip()}",
               "authors": (f"Author {rng.randrange(2000)}",), "tags": tags_of(rng)}
    if kind == "plain":
        return NewText(text=words.text(length_of(rng, number)), **credits)
    return NewText(kind=kind, content=conversation(rng, words) if kind == "conversation" else task(rng, words),
                   **credits)


def say(*parts):
    print(*parts, file=sys.stderr, flush=True)


def make_base(folder: Path, texts: int) -> Layout:
    """The synthetic depot of `texts` texts in folder/base.sqlite, made unless it is there already from this
    GENERATOR: its layout."""
    base, made, layout_file = folder / "base.sqlite", folder / "made", folder / "layout.json"
    if made.is_file() and made.read_text().strip() == str(GENERATOR) and base.is_file():
        return Layout(**json.loads(layout_file.read_text()))
    for path in folder.glob("base.sqlite*"):
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    made.unlink(missing_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rng = random.Random(SEED)
    words = Words(rng)
    depot = Depot.new(str(base), About(NAME, "Bench Person", "bench@example.org"))
    depot.db.backup_off()  # no change files while it is made
    depot.add_license(LICENSES[0], "Creative Commons Zero v1.0 Universal", "The CC0 text.")
    depot.add_license(LICENSES[1], "Creative Commons Attribution 4.0", "The CC BY text.")
    for first in range(1, texts + 1, BATCH):
        numbers = range(first, min(texts, first + BATCH - 1) + 1)
        depot.add([new_text(rng, words, n) for n in numbers], ME)
        if first // BATCH % 20 == 0:
            say(f"  made {numbers[-1]:,} of {texts:,} texts ({time.perf_counter() - start:.0f} s)")
    number = "CAST(substr(id, length('bench-') + 1) AS INTEGER)"
    say("  deciding")
    depot.decide("admitted", f"{number} % 20 NOT IN (0, 1, 2)", ME)
    depot.decide("refused", f"{number} % 20 = 0", ME)
    layout = Layout(texts=texts)
    say("  libraries")
    for name, where in (("open-a", "status = 'admitted' AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = "
                                   "'t00')"),
                        ("open-b", "status = 'admitted' AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value IN "
                                   "('t01', 't02'))"),
                        ("open-c", "status = 'admitted' AND kind = 'conversation'")):
        layout.open_libraries[name] = legacy_library(depot, name, where)
    for name, where in (("locked-a", "status = 'admitted' AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = "
                                     "'t03')"),
                        ("locked-b", f"status = 'admitted' AND {number} % 7 = 0")):
        layout.locked_libraries[name] = legacy_library(depot, name, where, locked=True)
    small = legacy_library(depot, "small", f"status = 'admitted' AND {number} <= {min(texts, 4000)} AND length < 20000")
    layout.small_library = small
    with depot.db.write("legacy dataset") as (con, change):
        con.execute("INSERT INTO datasets_revisions (id, revision, change, latest, name, tokenizer, libraries) "
                    "VALUES (?, 1, ?, 1, ?, 'bench-tokenizer', ?)",
                    (depot._id(depot._take(con, "datasets")), change, layout.dataset,
                     json.dumps([{"depot": None, "library": small, "change": change}])))
    depot.db.backup_on()
    depot.db.checkpoint()
    depot.db.close()
    layout.seconds = time.perf_counter() - start
    layout_file.write_text(json.dumps(vars(layout)))
    made.write_text(str(GENERATOR))
    say(f"  made in {layout.seconds:.0f} s")
    return layout


def legacy_library(depot: Depot, name: str, where: str, locked: bool = False) -> str:
    """The rows of the legacy tables, which no write of the depot makes any more, of a library holding the texts
    the SQL condition where matches, locked at once when locked (its pins at each text's revision then), as a
    depot of format 12 held them: its id."""
    with depot.db.write("legacy library") as (con, change):
        library = depot._id(depot._take(con, "libraries"))
        con.execute("INSERT INTO libraries_revisions (id, revision, change, latest, name, locked_at) "
                    "VALUES (?, 1, ?, 1, ?, ?)", (library, change, name, "2026-01-01T00:00:00Z" if locked else None))
        con.execute(f"INSERT INTO memberships (library, text, joined_change, export_key) SELECT ?, id, ?, "
                    f"export_key(id) FROM texts WHERE {where}", (library, change))
        if locked:
            con.execute(f"INSERT INTO pins (library, text, revision, change, export_key) SELECT ?, id, revision, ?, "
                        f"export_key(id) FROM texts WHERE {where}", (library, change))
    return library


def fresh_copy(folder: Path) -> Path:
    """folder/work.sqlite, a new copy of the base (its waiting folder left behind: it records from scratch)."""
    work = folder / "work.sqlite"
    for path in folder.glob("work.sqlite*"):
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    shutil.copyfile(folder / "base.sqlite", work)
    return work


def brought_up(work: Path) -> float | None:
    """The work copy brought up to the code's format when its base is of an older one (a base made by older code,
    kept): the seconds the upgrade took, None when it was of the code's format already."""
    from depot import upgrade
    from depot.depot import format_of
    from depot.schema import FORMAT
    if format_of(str(work)) == FORMAT:
        return None
    start = time.perf_counter()
    upgrade.upgrade(str(work), log=lambda line: say(f"  {line}"))[0].db.close()
    return time.perf_counter() - start


def self_ignoring(root: Path):
    """root made, with a .gitignore that ignores everything in it, for a root inside a checkout."""
    root.mkdir(parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")


def on_tmpfs(path: Path) -> bool:
    found = subprocess.run(["stat", "-f", "-c", "%T", str(path)], capture_output=True, text=True)
    return found.stdout.strip() in ("tmpfs", "ramfs")


def depot_files(path: Path) -> list[Path]:
    return [one for one in (path, Path(f"{path}-wal")) if one.is_file()]


def drop_cache(path: Path):
    """The pages of the depot's files written to disk and dropped from the kernel's cache."""
    for one in depot_files(path):
        fd = os.open(one, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def cached_share(path: Path) -> float | None:
    """The share of the depot file's pages in the kernel's cache (fincore), None when it cannot be read."""
    try:
        found = subprocess.run(["fincore", "--bytes", "--noheadings", "--output", "RES,SIZE", str(path)],
                               capture_output=True, text=True, check=True)
        resident, size = (int(one) for one in found.stdout.split())
        return resident / size if size else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


class Bench:
    """Times operations on the work copy of a depot, warm and cold, and keeps the results for the report."""

    def __init__(self, path: Path, runs: int):
        self.path = path
        self.runs = runs
        self.depot = Depot.open(str(path))
        self.results = []  # dicts: name, warm, cold (seconds), note
        self.figures = []  # dicts: name, value, unit, note

    def reopen(self):
        self.depot.db.close()
        self.depot = Depot.open(str(self.path))

    def cold_open(self):
        self.reopen()
        drop_cache(self.path)

    def time(self, name: str, op, note: str = "", cold: bool = True):
        """op(depot, run) timed `runs` times warm (after one untimed run) and `runs` times cold; run counts every
        call from 0, so a write can pick new texts each time. What op returns last is kept as the note's detail."""
        say(f"  {name}")
        call = 0
        detail = None

        def once():
            nonlocal call, detail
            start = time.perf_counter()
            detail = op(self.depot, call)
            call += 1
            return time.perf_counter() - start

        once()
        warm = [once() for _ in range(self.runs)]
        colds = []
        if cold:
            for _ in range(self.runs):
                self.cold_open()
                colds.append(once())
        self.results.append({"name": name, "warm": statistics.median(warm),
                             "cold": statistics.median(colds) if colds else None,
                             "note": note if detail is None else f"{note} {detail}".strip()})

    def figure(self, name: str, value, unit: str, note: str = ""):
        self.figures.append({"name": name, "value": value, "unit": unit, "note": note})

    def close(self):
        self.depot.db.close()


def machine() -> str:
    cpu = "?"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory = "?"
    try:
        kib = int(Path("/proc/meminfo").read_text().split()[1])
        memory = f"{kib / (1 << 20):.0f} GiB"
    except (OSError, ValueError, IndexError):
        pass
    import apsw
    return (f"{cpu}, {os.cpu_count()} threads, {memory} RAM, {platform.system()} {platform.release()}, "
            f"Python {platform.python_version()}, SQLite {apsw.sqlite_lib_version()}")


def milliseconds(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds * 1000:,.1f}"


def report(title: str, bench: Bench, header: list[str]) -> str:
    """The results as Markdown: the header lines, a table of timings (ms, medians) and a table of figures."""
    lines = [f"### {title}", "", *header, "",
             "| Operation | warm ms | cold ms | note |", "|---|---:|---:|---|"]
    lines += [f"| {one['name']} | {milliseconds(one['warm'])} | {milliseconds(one['cold'])} | {one['note']} |"
              for one in bench.results]
    if bench.figures:
        lines += ["", "| Figure | value | unit | note |", "|---|---:|---|---|"]
        lines += [f"| {one['name']} | {one['value']} | {one['unit']} | {one['note']} |" for one in bench.figures]
    return "\n".join(lines) + "\n"
