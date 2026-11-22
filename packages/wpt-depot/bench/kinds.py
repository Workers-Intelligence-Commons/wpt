"""The kind filter and retagging on a synthetic depot (bench/common.py): the
first page of a search with no filter and with a kind filter (a common kind,
plain, and a rare one, task), a count of a kind from the depot's counts
(Depot.held_count, and the read of the facet KIND it makes) and by a first
count step, a retag of 1,000 texts and its
estimate, and what retagging writes per text (rows, bytes of the file and of
the change file, seconds), measured adding one tag to TAGGED texts outside
the timings. It prints a Markdown report, and writes it to --out when given.

    uv run python packages/wpt-depot/bench/kinds.py --texts 200000 --root .bench-depots --runs 5
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import ME, Bench, say  # noqa: E402
from daily import PROJECTED, Picker, change_file_bytes, table_rows, used_bytes  # noqa: E402

from wpt_depot import counts  # noqa: E402
from wpt_depot.filters import STATUSES, Filter  # noqa: E402

TAGGED = 10_000  # texts retagged by the measure of what a retag writes per text


def measure(bench: Bench, layout: common.Layout):
    def search(selection, label):
        def op(d, _):
            page = d.search(selection, limit=50)
            return f"{len(page['rows'])} rows{', partial' if page['partial'] else ''}"
        bench.time(f"search first page, {label}", op)

    search(Filter(), "no filter")
    search(Filter(kinds=("plain",)), "kind plain (common)")
    search(Filter(kinds=("task",)), "kind task (rare)")
    search(Filter(kinds=("task",), statuses=("admitted",)), "kind task (rare), admitted")

    def held(selection):
        def op(d, _):
            found = d.held_count(selection)
            return "not held" if found is None else f"{sum(found['statuses'].values()):,} texts match"
        return op

    def stepped(selection):
        def op(d, _):
            found = d.count(selection)
            return f"{found['texts']:,} rows read, {sum(found['statuses'].values()):,} match"
        return op

    bench.time("count, kind task, from the counts (held_count)", held(Filter(kinds=("task",))))
    bench.time("count, kind task and admitted, from the counts (held_count)",
               held(Filter(kinds=("task",), statuses=("admitted",))))
    bench.time("count, kind task, first count step", stepped(Filter(kinds=("task",))))
    # the read held_count makes for a kind, timed on its own
    bench.time("count, kind task, from the facet KIND alone (counts.of_kinds)",
               lambda d, _: f"{sum(counts.of_kinds(d.db.reader(), STATUSES, ['task']).values()):,} texts match")

    admitted = Picker(layout.texts, "admitted", 21, kinds=("plain",))
    bench.time("retag estimate, 1,000 texts by id",
               lambda d, _: f"{d.retag_estimate(admitted.take(1000), ['bench-tag'], [])['changed']:,} would change")
    bench.time("retag estimate, tag t49 (rare), every text of it",
               lambda d, _: f"{d.retag_estimate(Filter(tags=('t49',)), ['bench-tag'], [])['changed']:,} would change")
    bench.time("retag, 1,000 texts by id", lambda d, _: f"{d.retag(admitted.take(1000), ['bench-tag'], [], ME)[1]:,} "
               "changed", "adds one tag to each")

    depot = bench.depot
    chosen = admitted.take(TAGGED)
    estimate = depot.retag_estimate(chosen, ["bench-tag"], [])
    rows_before, bytes_before, files_before = table_rows(depot), used_bytes(depot), change_file_bytes(depot)
    start = time.perf_counter()
    depot.retag(chosen, ["bench-tag"], [], ME)
    took = time.perf_counter() - start
    rows_after, bytes_after, files_after = table_rows(depot), used_bytes(depot), change_file_bytes(depot)
    grown = {name: rows_after[name] - rows_before.get(name, 0) for name in rows_after
             if rows_after[name] != rows_before.get(name, 0)}
    bench.figure("retag: rows added per text", f"{sum(grown.values()) / TAGGED:.2f}", "rows",
                 ", ".join(f"{name} {count / TAGGED:g}" for name, count in sorted(grown.items()))
                 + f"; the estimate said {estimate['written'] / TAGGED:.2f} ({estimate['rows']})")
    file_bytes = (bytes_after - bytes_before) / TAGGED
    change_bytes = (files_after - files_before) / TAGGED
    bench.figure("retag: file bytes per text", f"{file_bytes:,.0f}", "bytes",
                 f"pages in use after a checkpoint, {TAGGED:,} texts given one tag; 22M texts: "
                 f"{file_bytes * PROJECTED / 1e9:,.1f} GB")
    bench.figure("retag: change-file bytes per text", f"{change_bytes:,.0f}", "bytes",
                 f"the session's changeset beside the depot; 22M texts: {change_bytes * PROJECTED / 1e9:,.1f} GB")
    bench.figure("retag: seconds per text", f"{took / TAGGED * 1000:.3f}", "ms",
                 f"{TAGGED:,} texts in {took:,.1f} s, warm; 22M texts: {took / TAGGED * PROJECTED / 60:,.0f} min")


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
        measure(bench, layout)
    finally:
        bench.close()
    header = [f"- Command: `uv run python packages/wpt-depot/bench/kinds.py --texts {texts} --root {args.root} "
              f"--runs {args.runs}`",
              f"- Machine: {common.machine()}",
              f"- Depot: {texts:,} synthetic texts, {work.stat().st_size / (1 << 20):,.0f} MiB; "
              f"{'tmpfs: cold runs cannot drop the cache, they equal warm ones' if tmpfs else 'on disk'}"
              + ("" if dropped is None else f"; after a cache drop {dropped:.1%} of the file stayed cached"),
              f"- Medians of {args.runs} runs; warm after one untimed run; cold after closing, fsync, "
              "posix_fadvise DONTNEED and reopening."]
    if upgraded is not None:
        header.append(f"- The base is of an older format: the work copy was upgraded first, in {upgraded:,.0f} s.")
    text = common.report(f"Kind filter and retag, {texts:,} texts", bench, header)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
