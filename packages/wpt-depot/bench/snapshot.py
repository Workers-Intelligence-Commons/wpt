"""Snapshots on a synthetic depot (bench/common.py): making one, cold and
warm, its rate in texts per second and the longest read transaction it
held; the bytes its row and sets take; opening one (its rows, its bitmaps
deserialized and its digest checked); a count from its summary; reading its
texts in order at their revisions then; and a decision over 1,000 texts
made while a snapshot is being made in another thread, against the same
decision alone, the two measured in turn. Each figure is projected to PROJECTED texts where it grows
with them. It prints a Markdown report, and writes it to --out when given.

    uv run python packages/wpt-depot/bench/snapshot.py --texts 200000 --root .bench-depots --runs 5
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import ME, Bench, say  # noqa: E402
from daily import PROJECTED, Picker  # noqa: E402

from wpt_depot import Depot, snapshots  # noqa: E402

READ = 20_000  # texts read in order by the read measure


def measure(bench: Bench, layout: common.Layout):
    made = {"longest": []}
    names = iter(f"bench-{n}" for n in range(10_000))

    def create(d, _):
        measured = {}
        row = d.create_snapshot(next(names), measured=measured)
        made["row"], made["measured"] = row, measured
        made["longest"].append(measured["longest_read"])
        return f"{row['texts']:,} texts, {measured['pages']} pages, longest read {measured['longest_read'] * 1000:,.0f} ms"

    bench.time("create a snapshot", create)
    row, measured = made["row"], made["measured"]
    cold = bench.results[-1]["cold"] or bench.results[-1]["warm"]
    warm = bench.results[-1]["warm"]
    bench.figure("create rate, cold", f"{row['texts'] / cold:,.0f}", "texts/s",
                 f"warm {row['texts'] / warm:,.0f} texts/s; at 22M: {PROJECTED / (row['texts'] / cold):,.0f} s cold "
                 "(target >= 100,000 texts/s)")
    bench.figure("longest read transaction during create", f"{max(made['longest']) * 1000:,.0f}", "ms",
                 f"the longest of every run, cold ones included: a page of {snapshots.PAGE:,} ids or the catching up "
                 "(target <= 1,000 ms)")

    con = bench.depot.db.reader()
    sets_bytes = con.execute("SELECT sum(length(bitmap)) FROM snapshot_sets WHERE snapshot = ?",
                             (row["id"],)).fetchall()[0][0]
    summary_bytes, row_bytes = con.execute(
        "SELECT length(summary), length(summary) + length(origin) + length(digest) + length(name) + 64 FROM snapshots "
        "WHERE id = ?", (row["id"],)).fetchall()[0]
    bench.figure("storage: sets", f"{sets_bytes / row['texts']:.3f}", "bytes/text",
                 f"{sets_bytes:,} bytes for {row['texts']:,} texts (target <= 1.5)")
    bench.figure("storage: summary", f"{summary_bytes:,}", "bytes",
                 f"{len(row['summary']['plain_tagsets']):,} plain tag sets; row with summary about {row_bytes:,} bytes "
                 "(target <= 1 MB)")

    def opened(d, _):
        snapshots._kept.clear()
        d.snapshot(row["name"])
        parts = snapshots.sets(d, row["id"])
        return f"{sum(len(one[3]) for one in parts):,} numbers"

    bench.time("open a snapshot (rows, bitmaps, digest check)", opened)
    bench.figure("open a snapshot, 22M numbers", f"{projected_open() * 1000:,.0f}", "ms",
                 "deserialize and digest of a set of 22M random numbers below 30M, the server's size; its rows read "
                 "as above (target <= 300 ms)")
    bench.time("count from the summary", lambda d, _: d.snapshot_count(row["id"], ["plain"])["texts"] and None,
               "(target <= 50 ms)")

    def reading(start):
        def read(d, _):
            got, after = 0, start
            while got < READ:
                page = d.snapshot_texts(row["id"], after=after, limit=min(1000, READ - got))
                got += len(page["texts"])
                after = page["after"]
                if after is None:
                    break
            return f"{got:,} texts"
        return read

    # from the first text, whose short numbers lie far apart in the index of ids (bench-1, bench-10, ...), and from
    # the middle, where neighbouring numbers have as many digits and lie side by side in it, as most numbers do
    middle = common.text_id(layout.texts // 2)
    for label, start in (("from the first", None), (f"from {middle}", middle)):
        bench.time(f"read {READ:,} texts in order at their revisions then, {label}", reading(start))
        rate_cold = READ / (bench.results[-1]["cold"] or bench.results[-1]["warm"])
        bench.figure(f"read rate, cold, {label}", f"{rate_cold:,.0f}", "texts/s", "metadata only (target >= 20,000)")

    review = Picker(layout.texts, "under review", 11, kinds=("plain",))

    def alone(d):
        begin = time.perf_counter()
        d.decide("admitted", review.take(1000), ME)
        return time.perf_counter() - begin, None

    def during(d):
        other = Depot.open(d.address)
        started, done = threading.Event(), threading.Event()

        def job():
            other.create_snapshot(next(names), progress=lambda stage: started.set())
            done.set()

        worker = threading.Thread(target=job)
        worker.start()
        started.wait()
        begin = time.perf_counter()
        d.decide("admitted", review.take(1000), ME)
        took = time.perf_counter() - begin
        overlapped = not done.is_set()
        worker.join()
        other.db.close()
        return took, overlapped

    # the two in turn, so both meet the same state of the page cache and of the depot, which warms up as it goes
    say("  decide admitted over 1,000 texts, alone and while a snapshot is made, in turn")
    alone(bench.depot)
    times = {alone: [], during: []}
    overlaps = []
    for _ in range(bench.runs):
        for op in (alone, during):
            took, overlapped = op(bench.depot)
            times[op].append(took)
            if overlapped is not None:
                overlaps.append(overlapped)
    import statistics
    alone_s, during_s = statistics.median(times[alone]), statistics.median(times[during])
    bench.results.append({"name": "decide admitted over 1,000 texts, alone", "warm": alone_s, "cold": None,
                          "note": "texts under review, by id; warm only, in turn with the next line"})
    bench.results.append({"name": "decide admitted over 1,000 texts, while a snapshot is made", "warm": during_s,
                          "cold": None, "note": f"the snapshot in another thread of the same process; warm only; "
                                                f"the snapshot still running at the decision's end in "
                                                f"{sum(overlaps)} of {len(overlaps)} runs"})
    bench.figure("decide while a snapshot is made / alone", f"{during_s / alone_s:.2f}", "ratio",
                 "warm medians, the two measured in turn (target <= 1.05)")


def projected_open() -> float:
    """Seconds to deserialize a set of PROJECTED random numbers below 30M and check its digest, median of 3."""
    import random
    import statistics
    numbers = snapshots.BitMap(random.Random(1).sample(range(30_000_000), PROJECTED))
    numbers.run_optimize()
    stored = numbers.serialize()
    took = []
    for _ in range(3):
        start = time.perf_counter()
        snapshots.digest_of([(0, "bench", 1, snapshots.FrozenBitMap.deserialize(stored))], 1)
        took.append(time.perf_counter() - start)
    return statistics.median(took)


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
    header = [f"- Command: `uv run python packages/wpt-depot/bench/snapshot.py --texts {texts} --root {args.root} "
              f"--runs {args.runs}`",
              f"- Machine: {common.machine()}",
              f"- Depot: {texts:,} synthetic texts, {work.stat().st_size / (1 << 20):,.0f} MiB; "
              f"{'tmpfs: cold runs cannot drop the cache, they equal warm ones' if tmpfs else 'on disk'}"
              + ("" if dropped is None else f"; after a cache drop {dropped:.1%} of the file stayed cached"),
              f"- Medians of {args.runs} runs; warm after one untimed run; cold after closing, fsync, "
              "posix_fadvise DONTNEED and reopening."]
    if upgraded is not None:
        header.append(f"- The base is of an older format: the work copy was upgraded first, in {upgraded:,.0f} s.")
    text = common.report(f"Snapshots, {texts:,} texts", bench, header)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
