"""The order of a snapshot's texts on a synthetic depot (bench/common.py):
the order by export key (depot.order.of) that the depot's listing reads,
its time and the memory it holds, projected to PROJECTED texts. It prints
a Markdown report, and writes it to --out when given.

    uv run python packages/depot/bench/listing.py --texts 200000 --root .bench-depots --runs 5
"""

import argparse
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import Bench, say  # noqa: E402
from daily import PROJECTED  # noqa: E402

from depot import order  # noqa: E402


def measure(bench: Bench):
    say("  snapshot")
    snapshot = bench.depot.create_snapshot()
    took = []
    for _ in range(bench.runs):
        start = time.perf_counter()
        order.of(bench.depot, snapshot["id"])
        took.append(time.perf_counter() - start)
    tracemalloc.start()  # apart: tracing slows the order down
    made = order.of(bench.depot, snapshot["id"])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    held = made.numbers.nbytes + made.owner_sets.nbytes + made.piece.nbytes
    del made
    seconds = statistics.median(took)
    bench.figure("export-key order of the snapshot", f"{seconds * 1000:,.0f}", "ms",
                 f"{snapshot['texts']:,} texts, {snapshot['texts'] / seconds:,.0f} texts/s; 22M projected: "
                 f"{seconds * PROJECTED / snapshot['texts']:,.1f} s (target <= 60 s)")
    bench.figure("memory the order takes at its peak", f"{peak / 1e6:,.1f}", "MB",
                 f"traced allocations, {peak / snapshot['texts']:,.1f} bytes/text, then {held / 1e6:,.1f} MB held; "
                 f"22M projected: {peak * PROJECTED / snapshot['texts'] / 1e6:,.0f} MB (target <= 400 MB)")


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
    common.make_base(folder, texts)
    work = common.fresh_copy(folder)
    upgraded = common.brought_up(work)
    tmpfs = common.on_tmpfs(work)
    bench = Bench(work, args.runs)
    try:
        bench.cold_open()
        measure(bench)
    finally:
        bench.close()
    header = [f"- Command: `uv run python packages/depot/bench/listing.py --texts {texts} --root {args.root} "
              f"--runs {args.runs}`",
              f"- Machine: {common.machine()}",
              f"- Depot: {texts:,} synthetic texts, {work.stat().st_size / (1 << 20):,.0f} MiB; "
              f"{'tmpfs' if tmpfs else 'on disk'}",
              f"- Medians of {args.runs} runs, warm."]
    if upgraded is not None:
        header.append(f"- The base is of an older format: the work copy was upgraded first, in {upgraded:,.0f} s.")
    text = common.report(f"The order of a snapshot, {texts:,} texts", bench, header)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
