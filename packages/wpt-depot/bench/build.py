"""Corpus builds of a snapshot on a synthetic depot (bench/common.py): the
order of a snapshot's texts by export key (wpt_depot.order.of), its time and
the memory it holds, projected to PROJECTED texts; a build of the snapshot
cut at TOKENS tokens (about the tokens of the small dataset the baseline
built), in tokens per second, warm, into a new store each run; the estimate
the Build page would show for that build; and, to compare with the
baseline's dataset build, the whole corpus of a snapshot of the small
library's texts alone (every other admitted text of the work copy refused
first). It prints a Markdown
report, and writes it to --out when given.

    uv run python packages/wpt-depot/bench/build.py --texts 200000 --root .bench-depots --runs 5
"""

import argparse
import hashlib
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import Bench, say  # noqa: E402
from daily import PROJECTED  # noqa: E402

from wpt_depot import order  # noqa: E402

TOKENS = 1_940_000  # about the tokens of the baseline's dataset build (REDESIGN-NOTES.md, bench)
VOCAB_SIZE = 4096


def tokenizer_of(depot):
    """The bench's vocabulary: VOCAB_SIZE tokens trained on 200 plain texts, as the baseline's."""
    from wpt_tokenizer import Tokenizer
    sample = [depot.content(row["id"]).encode() for row in depot.rows(
        "SELECT id FROM texts WHERE kind = 'plain' LIMIT 200")]
    return Tokenizer.train(sample, vocab_size=VOCAB_SIZE)


def data_of(snapshot: dict, tokenizer, total_tokens: int | None = TOKENS) -> dict:
    return {"snapshot": snapshot["name"], "change": snapshot["at_change"], "digest": snapshot["digest"],
            "kinds": [], "shares": [], "total_tokens": total_tokens,
            "vocabulary": {"sha256": hashlib.sha256(tokenizer.to_bytes()).hexdigest(), "size": tokenizer.vocab_size}}


def snapshot_build(bench: Bench, root: Path, snapshot: dict, tokenizer, total_tokens: int | None = TOKENS,
                   name: str = "snapshot build, cut at TOKENS") -> dict:
    """The snapshot's corpus at total_tokens, built warm `runs` times into new stores: its figure, tokens per
    second (median), named so, and the record of the last build."""
    from wpt_depot.corpus import build
    from wpt_store import open_store
    say("  snapshot build")
    rates, record, tokens = [], None, 0
    with tempfile.TemporaryDirectory(prefix="stores-", dir=root) as stores:
        for run in range(bench.runs + 1):
            store = open_store((Path(stores) / f"store-{run}").as_uri())
            start = time.perf_counter()
            record = build(bench.depot, snapshot["id"], data_of(snapshot, tokenizer, total_tokens), store, tokenizer,
                           address="https://bench.example", log=lambda *_: None)
            seconds = time.perf_counter() - start
            tokens = record["total_tokens"] + sum(s["tokens"] for s in record["shards"].get("conversation", []))
            if run:
                rates.append(tokens / seconds)
    bench.figure(name.replace("TOKENS", f"{TOKENS:,}"), f"{statistics.median(rates):,.0f}", "tokens/s",
                 f"{record['texts']:,} texts, {tokens:,} tokens (plain and conversations) of a snapshot of "
                 f"{snapshot['texts']:,}, vocabulary {VOCAB_SIZE:,}, median of warm builds (the order of every "
                 "text of the snapshot included)")
    return record


def measure(bench: Bench, layout: common.Layout, root: Path):
    say("  snapshot")
    snapshot = bench.depot.create_snapshot("bench-build")
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
    tokenizer = tokenizer_of(bench.depot)
    snapshot_build(bench, root, snapshot, tokenizer)
    from wpt_depot.estimate import estimate
    took = []
    for _ in range(bench.runs):
        start = time.perf_counter()
        found = estimate(bench.depot, snapshot["id"], data_of(snapshot, tokenizer), tokenizer)
        took.append(time.perf_counter() - start)
    bench.figure("estimate of that build", f"{statistics.median(took) * 1000:,.0f}", "ms",
                 f"{found['sample']['tokenized']:,} texts tokenized, forecast {found['forecast']}; what a recipe "
                 "page would read (target for the page: <= 1,000 ms)")
    say("  refusing every admitted text outside the small library")
    bench.depot.decide("refused", "status = 'admitted' AND id NOT IN (SELECT text FROM memberships WHERE library = "
                       f"'{layout.small_library}')", common.ME)
    small = bench.depot.create_snapshot("bench-small")
    snapshot_build(bench, root, small, tokenizer, None, "snapshot build, the small library's texts alone, whole")


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
        measure(bench, layout, root)
    finally:
        bench.close()
    header = [f"- Command: `uv run python packages/wpt-depot/bench/build.py --texts {texts} --root {args.root} "
              f"--runs {args.runs}`",
              f"- Machine: {common.machine()}",
              f"- Depot: {texts:,} synthetic texts, {work.stat().st_size / (1 << 20):,.0f} MiB; "
              f"{'tmpfs' if tmpfs else 'on disk'}",
              f"- Medians of {args.runs} runs, warm."]
    if upgraded is not None:
        header.append(f"- The base is of an older format: the work copy was upgraded first, in {upgraded:,.0f} s.")
    text = common.report(f"Corpus builds of a snapshot, {texts:,} texts", bench, header)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
