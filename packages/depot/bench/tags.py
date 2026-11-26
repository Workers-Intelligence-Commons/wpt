"""Tags on a synthetic depot (bench/common.py): the first page of a search
by one tag (a common one, t00, and a rare one, t49), by a tag and a status
and by two tags, a count of a tag from the depot's counts (Depot.held_count)
and by a first count step, showing one text, a retag of 1,000 texts (adding
a tag, then removing it) and a change of 1,000 texts' tags through the
credits write (Depot.edit_credits_many), a snapshot's first page of texts
with their tags as of it (Depot.snapshot_texts) and the walk of a build's
first 20,000 entries with the snapshot's order made
(depot.order); and what one change of tags
writes per text, adding one tag to TAGGED texts outside the timings, by
retag and by the credits write: rows, bytes of the file and of the change
file a backup sends, seconds. It prints a Markdown report, and writes it to
--out when given.

    uv run python packages/depot/bench/tags.py --large --root .bench-depots --runs 5
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import ME, Bench, say  # noqa: E402
from daily import PROJECTED, Picker, change_file_bytes, table_rows, used_bytes  # noqa: E402

from depot import order  # noqa: E402
from depot.filters import Filter  # noqa: E402

TAGGED = 10_000  # texts given one tag by each measure of what a change of tags writes per text
WALKED = 20_000  # entries of a build's walk timed


def current_tags(depot, chosen: list[str]) -> dict[str, list[str]]:
    return {row["id"]: row["tags"] for row in depot.rows(
        "SELECT id, tags FROM texts WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(chosen),))}


def credited(depot, chosen: list[str], tag: str):
    """Give each text chosen the tag through the credits write."""
    held = current_tags(depot, chosen)
    return depot.edit_credits_many([{"id": one, "tags": sorted(set(held[one]) | {tag})} for one in chosen], ME)


def written(bench: Bench, label: str, write):
    """The figures of what write (a function of the depot) writes per text, on TAGGED texts."""
    depot = bench.depot
    rows_before, bytes_before, files_before = table_rows(depot), used_bytes(depot), change_file_bytes(depot)
    start = time.perf_counter()
    write(depot)
    took = time.perf_counter() - start
    rows_after, bytes_after, files_after = table_rows(depot), used_bytes(depot), change_file_bytes(depot)
    grown = {name: rows_after[name] - rows_before.get(name, 0) for name in rows_after
             if rows_after[name] != rows_before.get(name, 0)}
    file_bytes = (bytes_after - bytes_before) / TAGGED
    change_bytes = (files_after - files_before) / TAGGED
    bench.figure(f"{label}: rows added per text", f"{sum(grown.values()) / TAGGED:.2f}", "rows",
                 ", ".join(f"{name} {count / TAGGED:g}" for name, count in sorted(grown.items())))
    bench.figure(f"{label}: file bytes per text", f"{file_bytes:,.0f}", "bytes",
                 f"pages in use after a checkpoint, {TAGGED:,} texts given one tag; 22M texts: "
                 f"{file_bytes * PROJECTED / 1e9:,.2f} GB")
    bench.figure(f"{label}: change-file bytes per text", f"{change_bytes:,.0f}", "bytes",
                 f"the session's changeset beside the depot; 22M texts: {change_bytes * PROJECTED / 1e9:,.2f} GB")
    bench.figure(f"{label}: seconds per text", f"{took / TAGGED * 1000:.3f}", "ms",
                 f"{TAGGED:,} texts in {took:,.1f} s, warm; 22M texts: {took / TAGGED * PROJECTED / 60:,.0f} min")


def measure(bench: Bench, layout: common.Layout):
    def search(selection, label):
        def op(d, _):
            page = d.search(selection, limit=50)
            return f"{len(page['rows'])} rows{', partial' if page['partial'] else ''}"
        bench.time(f"search first page, {label}", op)

    search(Filter(tags=("t00",)), "tag t00 (common)")
    search(Filter(tags=("t49",)), "tag t49 (rare)")
    search(Filter(tags=("t00",), statuses=("admitted",)), "tag t00 and admitted")
    search(Filter(tags=("t00", "t01")), "tags t00 and t01")

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

    bench.time("count, tag t00, from the counts (held_count)", held(Filter(tags=("t00",))))
    bench.time("count, tag t00 and admitted, from the counts (held_count)",
               held(Filter(tags=("t00",), statuses=("admitted",))))
    bench.time("count, tag t00, first count step", stepped(Filter(tags=("t00",))))
    bench.time("count, tag t49, first count step", stepped(Filter(tags=("t49",))))
    shown = Picker(layout.texts, "admitted", 31)
    bench.time("show one text", lambda d, _: d.show(shown.take(1)[0]) and None)

    admitted = Picker(layout.texts, "admitted", 32, kinds=("plain",))
    removed = []

    def added(d, _):
        chosen = admitted.take(1000)
        removed.append(chosen)
        return f"{d.retag(chosen, ['bench-tag'], [], ME)[1]:,} changed"

    bench.time("retag, 1,000 texts by id, adding a tag", added)
    bench.time("retag, 1,000 texts by id, removing that tag",
               lambda d, _: f"{d.retag(removed.pop(), [], ['bench-tag'], ME)[1]:,} changed")
    bench.time("credits write, 1,000 texts given a tag", lambda d, _: f"{credited(d, admitted.take(1000), 'bench-tag')[2]:,}"
               " changed")

    snapshot = bench.depot.create_snapshot()
    bench.time("snapshot texts, first page of 1,000, tags as of it",
               lambda d, _: f"{len(d.snapshot_texts(snapshot['id'], limit=1000)['texts']):,} texts")

    def walked(d, _):
        count = sum(1 for _ in itertools.islice(order.walk(order.of(d, snapshot["id"])), WALKED))
        return f"{count:,} entries"

    bench.time(f"order of the snapshot and its walk's first {WALKED:,} entries", walked)

    written(bench, "retag", lambda d: d.retag(admitted.take(TAGGED), ["bench-tag-2"], [], ME))
    written(bench, "credits write", lambda d: credited(d, admitted.take(TAGGED), "bench-tag-3"))


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
    header = [f"- Command: `uv run python packages/depot/bench/tags.py --texts {texts} --root {args.root} "
              f"--runs {args.runs}`",
              f"- Machine: {common.machine()}",
              f"- Depot: {texts:,} synthetic texts, {work.stat().st_size / (1 << 20):,.0f} MiB; "
              f"{'tmpfs: cold runs cannot drop the cache, they equal warm ones' if tmpfs else 'on disk'}"
              + ("" if dropped is None else f"; after a cache drop {dropped:.1%} of the file stayed cached"),
              f"- Medians of {args.runs} runs; warm after one untimed run; cold after closing, fsync, "
              "posix_fadvise DONTNEED and reopening."]
    if upgraded is not None:
        header.append(f"- The base is of an older format: the work copy was upgraded first, in {upgraded:,.0f} s.")
    text = common.report(f"Tags, {texts:,} texts", bench, header)
    print(text)
    if args.out:
        args.out.write_text(text)
    for path in folder.glob("work.sqlite*"):
        common.shutil.rmtree(path) if path.is_dir() else path.unlink()


if __name__ == "__main__":
    main()
