"""Benchmark of the depot's daily work, run on demand (not by pytest):
`uv run python packages/wpt-depot/bench/daily.py [--texts N | --large] [--root DIR] [--runs 5] [--out FILE]`.

It makes its own synthetic depot (bench/common.py; 200,000 texts by
default, 2,000,000 with --large) under --root (default
/tmp/claude-1000/wpt-bench; never a real depot) and measures, on a fresh
copy of it, the medians of --runs runs warm and cold of: adding one text,
Depot.show, the first page of a search (no filter, a tag, a kind),
counting with a tag or a kind, a decision over 1,000 texts, a change of
one text's content, relicensing 1,000 texts, a bulk tag change through the
credits write (its time, and the rows and bytes it writes per text), a scan
of the index texts_status for the admitted texts, the number of distinct
tag sets of plain texts, the export-key sort of every admitted id, and a
snapshot's corpus build in tokens per second (bench/build.py). It prints a Markdown
report, and writes it to --out when given.

Cold figures need a root on a disk: on a tmpfs the files are memory and
cannot leave the cache (the report says so). A root inside a checkout gets
a .gitignore that ignores it.

Later benchmarks (snapshots, the upgrade, retagging) live beside this one and
reuse bench/common.py.
"""

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import LICENSES, ME, Bench, say, status_of, text_id  # noqa: E402

from wpt_depot import NewText, ids  # noqa: E402
from wpt_depot.filters import Filter  # noqa: E402
from wpt_depot.schema import tags_now  # noqa: E402

KIND_WHERE = "kind = '{kind}'"  # over the texts view, which joins content_lists on the current content version
KIND_JOIN = ("EXISTS (SELECT 1 FROM content_lists k WHERE k.id = texts.id AND k.version = texts.content_version "
             "AND k.kind = '{kind}')")
PROJECTED = 22_000_000  # the server's texts, about: the figures projected to it


class Picker:
    """Text numbers of one status, handed out without repeating, from a seeded shuffle."""

    def __init__(self, texts: int, status: str, seed: int, kinds=("plain", "conversation", "task")):
        numbers = [n for n in range(1, texts + 1) if status_of(n) == status and common.kind_of(n) in kinds
                   and n % common.LONG_EVERY]
        random.Random(seed).shuffle(numbers)
        self.numbers = numbers
        self.at = 0

    def take(self, count: int) -> list[str]:
        if self.at + count > len(self.numbers):
            raise SystemExit(f"the synthetic depot has too few texts left for this benchmark ({len(self.numbers)})")
        found = self.numbers[self.at:self.at + count]
        self.at += count
        return [text_id(n) for n in found]


def table_rows(depot) -> dict[str, int]:
    con = depot.db.reader()
    names = [name for (name,) in con.execute("SELECT name FROM sqlite_schema WHERE type = 'table' "
                                             "AND name NOT LIKE 'sqlite_%' "
                                             "AND sql NOT LIKE 'CREATE VIRTUAL%' ORDER BY name")]
    return {name: con.execute(f'SELECT count(*) FROM "{name}"').fetchall()[0][0] for name in names}


def used_bytes(depot) -> int:
    """Bytes of pages in use in the file once the WAL is folded into it."""
    depot.db.checkpoint()
    con = depot.db.reader()
    pages, free, size = (con.execute(f"PRAGMA {one}").fetchall()[0][0]
                         for one in ("page_count", "freelist_count", "page_size"))
    return (pages - free) * size


def change_file_bytes(depot) -> int:
    return sum(path.stat().st_size for path in depot.db.waiting_files())


def measure(bench: Bench, layout: common.Layout, root: Path):
    texts = layout.texts
    rng = random.Random(7)
    words = common.Words(random.Random(8), size=1 << 20)
    review = Picker(texts, "under review", 1, kinds=("plain",))
    admitted = Picker(texts, "admitted", 2, kinds=("plain",))
    shown = Picker(texts, "admitted", 3)

    bench.time("add one text", lambda d, _: d.add([NewText(text=words.text(1500), licenses=(LICENSES[0],),
                                                           title="A new text", tags=("t05", "t17"))], ME) and None,
               "1,500 characters, 2 tags")
    bench.time("show one text", lambda d, _: d.show(shown.take(1)[0]) and None)

    def search(selection, label):
        def op(d, _):
            page = d.search(selection, limit=50)
            return f"{len(page['rows'])} rows{', partial' if page['partial'] else ''}"
        bench.time(f"search first page, {label}", op)

    search(Filter(), "no filter")
    search(Filter(tags=("t00",)), "tag t00 (common)")
    search(Filter(tags=("t49",)), "tag t49 (rare)")
    search(Filter(statuses=("admitted",)), "status admitted")
    for kind, how in (("plain", "common"), ("task", "rare")):
        search(Filter(where=KIND_WHERE.format(kind=kind)), f"kind {kind} ({how}), view's join")
        search(Filter(where=KIND_JOIN.format(kind=kind)), f"kind {kind} ({how}), EXISTS on content_lists")

    def held(selection):
        def op(d, _):
            found = d.held_count(selection)
            return "not held" if found is None else f"{sum(found['statuses'].values()):,} texts match"
        return op

    def stepped(selection):
        def op(d, _):
            found = d.count(selection)
            return f"{found['texts']:,} rows read, {found['statuses']['admitted']:,} admitted match"
        return op

    bench.time("count, tag t00, from the counts (held_count)", held(Filter(tags=("t00",))))
    bench.time("count, tag t00, first count step", stepped(Filter(tags=("t00",))))
    bench.time("count, kind task, first count step", stepped(Filter(where=KIND_WHERE.format(kind="task"))))

    bench.time("decide admitted over 1,000 texts", lambda d, _: d.decide("admitted", review.take(1000), ME) and None,
               "texts under review, by id")
    bench.time("change content of one text",
               lambda d, _: d.change_content(admitted.take(1)[0], words.text(rng.randint(400, 3000)), None, ME)
               and None)
    bench.time("relicense 1,000 texts", lambda d, _: d.relicense(admitted.take(1000), [LICENSES[1]], ME) and None,
               "admitted, to CC-BY-4.0")

    def retag(d, chosen):
        tags = {row["id"]: row["tags"] for row in d.rows(
            "SELECT id, tags FROM texts WHERE id IN (SELECT value FROM json_each(?))",
            (common.json.dumps(chosen),))}
        lines = [{"id": one, "tags": sorted(set(tags[one]) | {"bench-tag"})} for one in chosen]
        return d.edit_credits_many(lines, ME)

    bench.time("bulk tag change, 1,000 texts (credits write)", lambda d, _: retag(d, admitted.take(1000)) and None,
               "adds one tag to each")

    # what one bulk tag change writes, per text: on 10,000 texts, measured outside the timings
    tagged = 10_000
    depot = bench.depot
    rows_before, bytes_before, files_before = table_rows(depot), used_bytes(depot), change_file_bytes(depot)
    retag(depot, admitted.take(tagged))
    rows_after, bytes_after, files_after = table_rows(depot), used_bytes(depot), change_file_bytes(depot)
    grown = {name: rows_after[name] - rows_before.get(name, 0) for name in rows_after
             if rows_after[name] != rows_before.get(name, 0)}
    bench.figure("bulk tag change: rows written per text", f"{sum(grown.values()) / tagged:.2f}", "rows",
                 ", ".join(f"{name} {count / tagged:g}" for name, count in sorted(grown.items())) +
                 " (net new rows; updated rows such as the old revision's latest flag not counted)")
    bench.figure("bulk tag change: file bytes per text", f"{(bytes_after - bytes_before) / tagged:,.0f}", "bytes",
                 f"pages in use after a checkpoint, {tagged:,} texts")
    bench.figure("bulk tag change: change-file bytes per text", f"{(files_after - files_before) / tagged:,.0f}",
                 "bytes", "the session's changeset beside the depot")

    # the index texts_status, admitted texts, read whole
    scan = (f"SELECT {ids.order_sql()} FROM texts_revisions INDEXED BY texts_status "
            "WHERE status = 'admitted' AND latest")
    plan = " ".join(str(row[-1]) for row in bench.depot.db.reader().execute(f"EXPLAIN QUERY PLAN {scan}"))
    resolved = any(row[1] == "Column" and row[2] == 0 for row in bench.depot.db.reader().execute(f"EXPLAIN {scan}"))
    plan += "; reads revision rows" if resolved else "; index only (its DeferredSeek is never resolved)"
    counted = {}

    def status_scan(d, _):
        start = time.perf_counter()
        n = sum(1 for _ in d.db.reader().execute(scan))
        counted["n"], counted["s"] = n, time.perf_counter() - start
        return f"{n:,} ids"

    bench.time("scan texts_status, admitted", status_scan, plan)
    cold_scan = bench.results[-1]["cold"] or bench.results[-1]["warm"]
    warm_scan = bench.results[-1]["warm"]
    bench.figure("texts_status scan rate, cold", f"{counted['n'] / cold_scan:,.0f}", "ids/s",
                 f"warm {counted['n'] / warm_scan:,.0f} ids/s; at 22M admitted: "
                 f"{PROJECTED / (counted['n'] / cold_scan):,.0f} s cold")

    con = bench.depot.db.reader()
    for label, where in (("every status", ""), ("admitted", "AND t.status = 'admitted'")):
        sets = {tuple(sorted(common.json.loads(tags))) for (tags,) in con.execute(
            f"SELECT DISTINCT {tags_now('t.id')} FROM texts_revisions t JOIN content_lists c ON c.id = t.id "
            f"AND c.version = t.content_version WHERE t.latest AND c.kind = 'plain' {where}")}
        bench.figure(f"distinct plain tag sets, {label}", f"{len(sets):,}", "sets", "50 tags, 1-4 a text")

    # export keys of every admitted id, then their sort
    import numpy as np
    admitted_ids = [row[0] for row in con.execute(
        "SELECT id FROM texts_revisions INDEXED BY texts_status WHERE status = 'admitted' AND latest")]
    keying, sorting = [], []
    for _ in range(bench.runs):
        start = time.perf_counter()
        keys = np.fromiter((ids.export_key(one) for one in admitted_ids), dtype=np.int64, count=len(admitted_ids))
        middle = time.perf_counter()
        np.argsort(keys, kind="stable")
        keying.append(middle - start)
        sorting.append(time.perf_counter() - middle)
    key_s, sort_s = statistics.median(keying), statistics.median(sorting)
    bench.figure("export keys of the admitted ids", f"{key_s * 1000:,.0f}", "ms",
                 f"{len(admitted_ids):,} ids, {len(admitted_ids) / key_s:,.0f} ids/s; 22M: "
                 f"{PROJECTED / (len(admitted_ids) / key_s):,.0f} s")
    bench.figure("argsort of those keys", f"{sort_s * 1000:,.0f}", "ms",
                 f"22M projected: {sort_s * PROJECTED / len(admitted_ids):,.1f} s")

    build(bench, layout, root)


def build(bench: Bench, layout: common.Layout, root: Path):
    """A snapshot of the depot's admitted texts, its corpus built at build.TOKENS tokens warm `runs` times into new
    stores: tokens per second (median, build.snapshot_build)."""
    import build as builds
    snapshot = bench.depot.create_snapshot("bench-daily")
    builds.snapshot_build(bench, root, snapshot, builds.tokenizer_of(bench.depot))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--texts", type=int, default=200_000)
    parser.add_argument("--large", action="store_true", help="2,000,000 texts")
    parser.add_argument("--root", type=Path, default=common.DEFAULT_ROOT)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    texts = 2_000_000 if args.large else args.texts
    root = args.root.resolve()
    common.self_ignoring(root)
    folder = root / str(texts)
    say(f"synthetic depot of {texts:,} texts in {folder}")
    layout = common.make_base(folder, texts)
    work = common.fresh_copy(folder)
    upgraded = common.brought_up(work)
    tmpfs = common.on_tmpfs(work)
    bench = Bench(work, args.runs)
    try:
        bench.cold_open()
        dropped = common.cached_share(work)
        measure(bench, layout, root)
    finally:
        bench.close()
    size = work.stat().st_size
    header = [f"- Command: `uv run python packages/wpt-depot/bench/daily.py --texts {texts} --root {args.root} "
              f"--runs {args.runs}`",
              f"- Machine: {common.machine()}",
              f"- Depot: {texts:,} synthetic texts, {size / (1 << 20):,.0f} MiB, made in {layout.seconds:,.0f} s; "
              f"{'tmpfs: cold runs cannot drop the cache, they equal warm ones' if tmpfs else 'on disk'}"
              + ("" if dropped is None else f"; after a cache drop {dropped:.1%} of the file stayed cached"),
              f"- Medians of {args.runs} runs; warm after one untimed run; cold after closing, fsync, "
              "posix_fadvise DONTNEED and reopening."]
    if upgraded is not None:
        header.append(f"- The base is of an older format: the work copy was upgraded first, in {upgraded:,.0f} s.")
    text = common.report(f"Daily work, format {bench_format(work)}, {texts:,} texts", bench, header)
    print(text)
    if args.out:
        args.out.write_text(text)


def bench_format(path: Path) -> int:
    from wpt_depot.depot import format_of
    return format_of(str(path))


if __name__ == "__main__":
    main()
