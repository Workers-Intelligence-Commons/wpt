"""The upgrade the server runs (`wpt depot upgrade --yes`, formats 11 to 13
in one run) on a synthetic depot (bench/common.py): the check
(upgrade.check) and then the upgrade, each started cold, the time of each
step and stage, the rates the check takes for what it does not read itself
(upgrade.RATE_12, RATE_13_TABLES, RATE_13_TAGGINGS, TAGGING_BYTES, RATE_13_SIZES, SIZE_BYTES, COPY_BYTES and COPY_TAG_BYTES), the rate of each conversion of the
step to format 13 (pins or memberships read a second, its revisions
checked), and how far the check's estimate of the time was from the time
measured; each projected to PROJECTED texts. It prints a Markdown report,
and writes it to --out when given.

The base is of format 12 (made by common.make_base). The benchmark copies
it to ROOT/<texts>/upgrade-work.sqlite (its own name, so a benchmark
running beside it keeps its work copy) and takes it back to format 11 as
the code of format 11 laid it out, in plain SQL: no kind on the plain
content versions, the column kind nullable, no row plain in kinds, the
index content_lists_long of format 11, no table quotes, no counts of what
libraries hold of the kinds other than plain. One run each: an upgrade
writes the copy, so it cannot run again on it. The copy is deleted at the
end.

    uv run python packages/wpt-depot/bench/upgrade.py --large --root .bench-depots
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import apsw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import say  # noqa: E402
from daily import PROJECTED  # noqa: E402

from wpt_depot import conversion, upgrade  # noqa: E402


def older(path: Path):
    """The copy at path, of format 12, taken back to format 11 as the module says."""
    con = apsw.Connection(str(path))
    try:
        with con:
            statement = con.execute("SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'content_lists'"
                                    ).fetchall()[0][0]
            version = con.execute("PRAGMA schema_version").fetchall()[0][0]
            con.execute("PRAGMA writable_schema = ON")
            con.execute("UPDATE sqlite_schema SET sql = ? WHERE type = 'table' AND name = 'content_lists'",
                        (statement.replace(upgrade.NOT_NULL_12, upgrade.NULLABLE_12),))
            con.execute(f"PRAGMA schema_version = {int(version) + 1}")
            con.execute("PRAGMA writable_schema = OFF")
        with con:
            con.execute("UPDATE content_lists SET kind = NULL WHERE kind = 'plain'")
            con.execute("DROP INDEX IF EXISTS content_lists_long")
            con.execute(upgrade.LONG_11)
            con.execute("DELETE FROM kinds WHERE id = 'plain'")
            con.execute("DROP TABLE IF EXISTS quotes")
            con.execute(f"DELETE FROM counts WHERE facet = '{upgrade.LIBRARY_KIND_9}' AND "
                        "substr(name, instr(name, ' ') + 1) <> 'plain'")
            con.execute("UPDATE about_revisions SET format = 11 WHERE revision = (SELECT max(revision) FROM "
                        "about_revisions)")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()


def _table_bytes(con, table: str) -> int:
    """The bytes of the file the table's own pages take (its rows, not its indexes)."""
    return con.execute("SELECT coalesce(sum(pgsize), 0) FROM dbstat WHERE name = ?", (table,)).fetchall()[0][0]


def timed_stages(spent: dict, measured: list):
    """Wrap the step to format 12 and each stage of the step to format 13 so their seconds land in spent, and
    conversion.measure so each conversion's rows and seconds land in measured; the function that undoes it."""
    real_12, real_stages, real_measure = upgrade.FILE_STEPS[11], dict(upgrade.STAGES_13), conversion.measure

    def wrap(name, function):
        def run(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                spent[name] = time.perf_counter() - start
        return run

    def measure(con, planned, *args, **kwargs):
        found = real_measure(con, planned, *args, **kwargs)
        measured.append((planned.name, planned.origin["kind"], found.rows, found.texts, found.seconds))
        return found

    upgrade.FILE_STEPS[11] = wrap("11 to 12", real_12)
    for name, stage in real_stages.items():
        upgrade.STAGES_13[name] = wrap(f"12 to 13, {name}", stage)
    conversion.measure = measure

    def undo():
        upgrade.FILE_STEPS[11] = real_12
        upgrade.STAGES_13.update(real_stages)
        conversion.measure = real_measure
    return undo


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--texts", type=int, default=200_000)
    parser.add_argument("--large", action="store_true", help="2,000,000 texts")
    parser.add_argument("--root", type=Path, default=common.DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    texts = 2_000_000 if args.large else args.texts
    root = args.root.resolve()
    common.self_ignoring(root)
    folder = root / str(texts)
    say(f"synthetic depot of {texts:,} texts in {folder}")
    common.make_base(folder, texts)
    work = folder / "upgrade-work.sqlite"
    for path in folder.glob("upgrade-work.sqlite*"):
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    shutil.copyfile(folder / "base.sqlite", work)
    try:
        start = time.perf_counter()
        older(work)
        say(f"  taken back to format 11 in {time.perf_counter() - start:,.0f} s")
        con = apsw.Connection(f"file:{work.as_posix()}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
        contents, current = (con.execute("SELECT max(rowid) FROM content_lists").fetchall()[0][0],
                             con.execute("SELECT count(*) FROM texts_revisions WHERE latest").fetchall()[0][0])
        revisions = con.execute("SELECT max(rowid) FROM texts_revisions").fetchall()[0][0]
        revision_bytes = _table_bytes(con, "texts_revisions")
        con.close()
        size = work.stat().st_size

        common.drop_cache(work)
        dropped = common.cached_share(work)
        start = time.perf_counter()
        checked = upgrade.check(str(work))
        check_seconds = time.perf_counter() - start
        for line in checked.lines:
            say(f"  | {line}")

        spent, measured = {}, []
        undo = timed_stages(spent, measured)
        common.drop_cache(work)
        start = time.perf_counter()
        try:
            upgrade.upgrade(str(work), log=lambda line: say(f"  {line}"))[0].db.close()
        finally:
            undo()
        total = time.perf_counter() - start
        added = work.stat().st_size - size
        con = apsw.Connection(f"file:{work.as_posix()}?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
        tagged, tagging_bytes = con.execute("SELECT (SELECT count(*) FROM taggings), (SELECT sum(pgsize) FROM dbstat "
                                            "WHERE name IN ('taggings', 'taggings_change'))").fetchall()[0]
        sized_bytes = _table_bytes(con, "texts_revisions")
        copy_bytes, current_tags = con.execute(
            "SELECT (SELECT coalesce(sum(length(current_tags)), 0) FROM texts_revisions WHERE latest), "
            "(SELECT coalesce(sum(texts), 0) FROM counts WHERE facet = 'tag')").fetchall()[0]
        con.close()
    finally:
        for path in folder.glob("upgrade-work.sqlite*"):
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    scale = PROJECTED / texts
    rate_12 = contents / spent["11 to 12"]
    rate_tables = current / spent["12 to 13, tables"]
    rate_taggings = revisions / spent["12 to 13, taggings"]
    rate_sizes = revisions / spent["12 to 13, sizes"]
    rows = [f"| `--check`, cold | {check_seconds:,.1f} | {check_seconds * scale:,.0f} | the check's own estimate of "
            f"the upgrade: {sum(checked.cost['seconds'].values()):,.0f} s |",
            *(f"| {name} | {seconds:,.1f} | {seconds * scale:,.0f} | |" for name, seconds in spent.items()),
            f"| whole upgrade, cold | {total:,.1f} | {total * scale:,.0f} | |"]
    figures = [f"| content rows the step to 12 writes a second (RATE_12) | {rate_12:,.0f} | {contents:,} rows |",
               f"| current texts the stage tables counts a second (RATE_13_TABLES) | {rate_tables:,.0f} | "
               f"{current:,} texts |",
               f"| revisions the stage taggings reads a second (RATE_13_TAGGINGS) | {rate_taggings:,.0f} | "
               f"{revisions:,} revisions |",
               f"| bytes of a row of taggings, its index's entry included (TAGGING_BYTES) | "
               f"{tagging_bytes / max(tagged, 1):,.0f} | {tagged:,} rows in {tagging_bytes / 1e6:,.0f} MB |",
               f"| revisions the stage sizes writes a second (RATE_13_SIZES) | {rate_sizes:,.0f} | "
               f"{revisions:,} revisions |",
               f"| bytes texts_revisions grows by a revision in the stage sizes (SIZE_BYTES, the copy of the current "
               f"tags included) | {(sized_bytes - revision_bytes) / max(revisions, 1):,.1f} | "
               f"{revision_bytes / 1e6:,.0f} MB to {sized_bytes / 1e6:,.0f} MB, "
               f"{revision_bytes / max(revisions, 1):,.0f} bytes a revision before |",
               f"| bytes of the copy of a current text's tags, for each tag it has (COPY_TAG_BYTES, with COPY_BYTES "
               f"for the text) | {(copy_bytes - current * 2) / max(current_tags, 1):,.1f} | {copy_bytes / 1e6:,.1f} MB "
               f"of JSON for {current:,} texts and {current_tags:,} of their tags |"]
    for name, kind, read, held, seconds in measured:
        figures.append(f"| {kind} {name}: rows read and checked a second | {read / max(seconds, 1e-9):,.0f} | "
                       f"{read:,} rows, {held:,} texts, {seconds:,.2f} s (target >= 200,000) |")
    figures += [f"| disk added | {added / 1e6:,.1f} MB | the check said {checked.cost['added'] / 1e6:,.1f} MB |",
                f"| WAL at its largest, as the check says | {checked.cost['wal'] / 1e6:,.1f} MB | |"]
    text = "\n".join([
        f"### Upgrade 11 to 13, {texts:,} texts", "",
        "- Command: `uv run python packages/wpt-depot/bench/upgrade.py "
        + ("--large" if args.large else f"--texts {texts}") + f" --root {args.root}`",
        f"- Machine: {common.machine()}",
        f"- Depot: {texts:,} synthetic texts, {size / (1 << 20):,.0f} MiB, {contents:,} content rows; "
        f"{'tmpfs' if common.on_tmpfs(folder) else 'on disk'}"
        + ("" if dropped is None else f"; after a cache drop {dropped:.1%} of the file stayed cached"),
        "- One run each, started cold; the projection multiplies by "
        f"{PROJECTED:,} / {texts:,} (every stage reads each text or row a fixed number of times).", "",
        "| What | seconds | at 22M, s | note |", "|---|---:|---:|---|", *rows, "",
        "| Figure | value | note |", "|---|---:|---|", *figures, ""])
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
