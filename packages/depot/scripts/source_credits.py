"""Write the input files of `depot credits-file` that give titles to
the texts of a depot that have none, and fix authors stored as the text of
a Python dict, by the rules of docs/data-selection.md, "Titles from a
text's source".

Usage:
    uv run python packages/depot/scripts/source_credits.py --depot FILE --out FOLDER
        --ledger FILE [--ledger FILE ...] [--catalogue pg_catalog.csv]
        [--family NAME ...] [--range FIRST LAST]

The depot is read through the depot's library, opened read only
(depot.Depot.view), one text at a time by its id, so it may be read while
the depot's web app writes it.

Which family of sources a text comes from is read from the ledgers of the
run that added the texts: JSON lines whose "intent" lines name a source and give `before`
(how many texts the depot held just before) and `count` (how many texts it
added, 1 when absent), so the texts are the ids NAME-(before+1) to
NAME-(before+count) of the depot named NAME. The source's name says its
family (FAMILIES). --range keeps only the ids from FIRST to LAST (their
numbers), --family only the families named.

For each family it writes FOLDER/<family>-0001.jsonl, -0002, ..., at most
CHUNK lines each, one JSON object per text that changes ({"id", "title"}
or {"id", "authors"}, with "was": the value it read, so the depot changes
nothing set since), and FOLDER/<family>.json: how far it went and its
counts (texts read, titled, left empty, already titled, authors fixed,
authors dropped). A chunk is written to a .part file first and renamed
once whole, then the counts are saved, so a run stopped at any point and
started again goes on from the last whole chunk. --out - writes the lines
to the standard output instead, and the counts to the standard error.
Gutenberg titles come from the catalogue pg_catalog.csv
(https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv), given with
--catalogue.
"""

import argparse
import ast
import csv
import json
import re
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from urllib.parse import urlparse

from depot import Depot
from depot.errors import NotFound

CHUNK = 350_000  # lines per input file: what one `depot credits-file` takes at most (depot.crediting.MOST)

# a family by the start of its sources' names, the longest first; any other source has no rule
FAMILIES = (
    ("huggingface-common-pile-stackv2-edu-filtered-", "stackv2-edu"),
    ("huggingface-common-pile-stackv2-html-filtered-", "stackv2-html"),
    ("huggingface-common-pile-github-archive-filtered-", "github-archive"),
    ("huggingface-common-pile-pes2o-filtered-", "pes2o"),
    ("huggingface-common-pile-python-enhancement-proposals-filtered-", "pep"),
    ("huggingface-common-pile-pubmed-filtered-", "pubmed"),
    ("standardebooks-", "standard-ebooks"),
    ("gutenberg-", "gutenberg"),
    ("ebible-", "ebible"),
    ("github-", "github"),
)

# what a family's rule reads besides the text's row: nothing, its start up to its first blank line, its first part,
# or its whole text
READS = {"stackv2-edu": None, "stackv2-html": None, "github": None, "ebible": None, "gutenberg": None,
         "pubmed": None, "github-archive": "opening", "pes2o": "opening", "pep": "head",
         "standard-ebooks": "whole"}

# the fields of a PEP's header (PEP 1, "PEP Header Preamble"), which end the field before them
PEP_FIELDS = ("PEP", "Title", "Version", "Last-Modified", "Author", "Sponsor", "BDFL-Delegate", "PEP-Delegate",
              "Discussions-To", "Status", "Type", "Topic", "Content-Type", "Requires", "Created", "Python-Version",
              "Post-History", "Replaces", "Superseded-By", "Resolution")
PEP_TITLE = re.compile(r"\bTitle:(.*?)(?=\b(?:" + "|".join(re.escape(one) for one in PEP_FIELDS if one != "Title")
                       + r"):|\Z)", re.S)
# Markdown's escape of a punctuation character, and its hard line break: a backslash at the end of a line
ESCAPED = re.compile(r"\\([!-/:-@\[-`{-~])")
HARD_BREAK = re.compile(r"\\\n")
# the end of a text's first paragraph: a blank line, or only spaces left to the end of the text
BLANK = re.compile(r"\n[ \t\r]*(?:\n|\Z)")
LONGEST_TITLE = 300  # characters: a longer first line or paragraph is no title
DC = "{http://purl.org/dc/elements/1.1/}"
OPF_BLOCK = "=== src/epub/content.opf\n"


# The rules, one per family: each takes what the text's row and content give and returns the title, or None.

def unescaped(line: str) -> str:
    """A line of Markdown as its text: hard line breaks as spaces, each escaped punctuation character as itself."""
    return ESCAPED.sub(r"\1", HARD_BREAK.sub(" ", line))


def line_title(line: str) -> str:
    """A line of Markdown as its text, its spaces at both ends trimmed; an odd last backslash is a hard break."""
    line = line.rstrip(" \t\r")
    ending = len(line) - len(line.rstrip("\\"))
    return unescaped(line[:-1] if ending % 2 else line).strip()


def first_line_title(opening: str, joins_hard_breaks: bool = False) -> str | None:
    """The title of a text that starts with it on a line of its own: the first line when a blank line or the end
    of the text follows it, as its Markdown reads; with joins_hard_breaks, the first paragraph when every line of
    it but the last ends in a hard break, its lines joined by a space. None when there is no such line, when it is
    empty, longer than LONGEST_TITLE characters, or holds \\documentclass (the source of a LaTeX paper).
    opening: the text, or its start up to its first blank line at least."""
    found = BLANK.search(opening)
    lines = (opening[:found.start()] if found else opening).split("\n")
    if len(lines) > 1 and not (joins_hard_breaks and all(_hard_break(line) for line in lines[:-1])):
        return None
    title = " ".join(part for part in (line_title(line) for line in lines) if part)
    if not title or len(title) > LONGEST_TITLE or "\\documentclass" in title:
        return None
    return title


def _hard_break(line: str) -> bool:
    """The line ends in an odd number of backslashes: a Markdown hard break."""
    line = line.rstrip(" \t\r")
    return (len(line) - len(line.rstrip("\\"))) % 2 == 1


def code_file_title(address: str | None) -> str | None:
    """owner/repo/path from https://raw.githubusercontent.com/<owner>/<repo>/<commit>/<path>, as the address writes
    it (a raw path: '#' and '?' are part of a file's name)."""
    parts = _after(address, "https://raw.githubusercontent.com/")
    if parts is None or len(parts) < 4 or not all(parts[:4]):
        return None
    return "/".join((parts[0], parts[1], *parts[3:]))


def repository_title(address: str | None) -> str | None:
    """owner/name from https://github.com/<owner>/<name>."""
    parts = _after(address, "https://github.com/")
    if parts is None or len(parts) != 2 or not all(parts):
        return None
    return "/".join(parts)


def _after(address: str | None, start: str) -> list[str] | None:
    """The segments of the address after start, None when it does not begin so."""
    return address[len(start):].split("/") if address and address.startswith(start) else None


def pep_title(head: str) -> str | None:
    """The Title: field of the header that starts a PEP (its first paragraph), up to the next field."""
    header = unescaped(head.lstrip().split("\n\n", 1)[0])
    found = PEP_TITLE.search(header) if header.startswith("PEP:") else None
    title = " ".join(found.group(1).split()) if found else ""
    return title or None


def standard_ebooks_title(text: str) -> str | None:
    """The first dc:title of the text's `=== src/epub/content.opf` block."""
    start = text.find(OPF_BLOCK)
    if start < 0:
        return None
    start += len(OPF_BLOCK)
    end = text.find("\n=== ", start)
    block = text[start:] if end < 0 else text[start:end]
    try:
        root = ElementTree.fromstring(block.strip())
    except ElementTree.ParseError:
        return None
    found = root.find(f".//{DC}title")
    title = None if found is None else "".join(found.itertext()).strip()
    return title or None


def ebible_title(notice: str | None) -> str | None:
    """The notice's text before ', eBible.org'."""
    if not notice or ", eBible.org" not in notice:
        return None
    return notice.split(", eBible.org", 1)[0].strip() or None


def gutenberg_number(address: str | None) -> str | None:
    """The ebook number of a Project Gutenberg address: its last folder made of digits alone, which the name of
    what follows it starts with (7984/7984-h/7984-h.htm, 3352/3352.txt)."""
    if not address:
        return None
    parts = urlparse(address).path.strip("/").split("/")
    numbered = [at for at, one in enumerate(parts[:-1]) if one.isdigit()]
    if not numbered:
        return None
    number = parts[numbered[-1]]
    return number if re.fullmatch(rf"{number}([-.].*)?", parts[numbered[-1] + 1]) else None


def gutenberg_title(address: str | None, catalogue: dict[str, str]) -> str | None:
    """The title the catalogue gives the ebook number of the address, its lines trimmed, the empty ones skipped,
    and joined by ': '."""
    number = gutenberg_number(address)
    title = catalogue.get(number) if number else None
    if not title:
        return None
    lines = [" ".join(line.split()) for line in title.splitlines()]
    return ": ".join(line for line in lines if line) or None


def fixed_authors(authors: list[str]) -> tuple[list[str], int] | None:
    """Authors stored as the text of a Python dict {'first': F, 'last': L} as "F L", the parts that are not
    empty kept, an author whose parts are both empty dropped: (the new list, how many dropped), or None when no
    author is such a dict."""
    fixed, dropped, changed = [], 0, False
    for one in authors:
        value = None
        if one.startswith("{"):
            try:
                value = ast.literal_eval(one)
            except (ValueError, SyntaxError):
                value = None
        if isinstance(value, dict) and value and set(value) <= {"first", "last"}:
            changed = True
            name = " ".join(str(value[key]).strip() for key in ("first", "last")
                            if value.get(key) and str(value[key]).strip())
            if name:
                fixed.append(name)
            else:
                dropped += 1
        else:
            fixed.append(one)
    return (fixed, dropped) if changed else None


# Reading

def ledger_entries(ledgers: list[Path]):
    """The entries of the ledgers, each a JSON object, in order."""
    for path in ledgers:
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)


def families(entries, name: str) -> list[tuple[int, int, str]]:
    """(first, last, family) of every run of ids the ledgers' entries name for a source of a family, by first.
    SystemExit when two runs share an id."""
    found = []
    for entry in entries:
        if entry.get("kind") != "intent":
            continue
        family = next((family for start, family in FAMILIES if entry["source"].startswith(start)), None)
        if family is not None:
            first = entry["before"] + 1
            found.append((first, first + entry.get("count", 1) - 1, family))
    found.sort()
    for before, after in zip(found, found[1:]):
        if after[0] <= before[1]:
            raise SystemExit(f"the ledgers name the ids {name}-{after[0]} to {name}-{before[1]} twice")
    return found


def catalogue_titles(path: Path) -> dict[str, str]:
    """{ebook number: title} of pg_catalog.csv."""
    with path.open(encoding="utf-8", newline="") as file:
        return {row["Text#"]: row["Title"] for row in csv.DictReader(file)}


class Reader:
    """A depot, opened read only (depot.Depot.view), one text at a time."""

    def __init__(self, path: str):
        self.depot = Depot.view(path)

    def name(self) -> str:
        return self.depot.name()

    def row(self, text: str) -> dict | None:
        """The text's current credits and its notice's text (Depot.show, without reading its content); None
        without such a text."""
        try:
            shown = self.depot.show(text, characters=0)
        except NotFound:
            return None
        found = shown["text"]
        return {"id": text, "title": found["title"], "authors": found["authors"], "address": found["address"],
                "notice": shown["notice"]}

    def parts(self, row: dict):
        """The texts of the parts of the text's current content, in order, read one at a time (Depot.parted)."""
        return iter(self.depot.parted(row["id"]))

    def opening(self, row: dict) -> str:
        """The text's start, up to its first blank line at least, or the whole text when it has none."""
        read = ""
        for part in self.parts(row):
            read += part
            if re.search(r"\n[ \t\r]*\n", read):
                break
        return read

    def head(self, row: dict) -> str:
        return next(self.parts(row), "")

    def whole(self, row: dict) -> str:
        return "".join(self.parts(row))


def change_of(family: str, row: dict, reader: Reader, catalogue: dict[str, str], counts: dict) -> dict | None:
    """The line of a text's change by its family's rule, or None; counts moved by what it found."""
    counts["read"] += 1
    if family == "pubmed":
        fixed = fixed_authors(row["authors"]) if row["authors"] else None
        if fixed is None:
            return None
        counts["authors fixed"] += 1
        counts["authors dropped"] += fixed[1]
        return {"id": row["id"], "authors": fixed[0] or None, "was": {"authors": row["authors"]}}
    if row["title"] is not None:
        counts["already titled"] += 1
        return None
    reads = READS[family]
    given = (reader.opening(row) if reads == "opening" else reader.head(row) if reads == "head"
             else reader.whole(row) if reads == "whole" else None)
    title = {"stackv2-edu": lambda: code_file_title(row["address"]),
             "stackv2-html": lambda: code_file_title(row["address"]),
             "github": lambda: repository_title(row["address"]),
             "github-archive": lambda: first_line_title(given),
             "pes2o": lambda: first_line_title(given, joins_hard_breaks=True),
             "pep": lambda: pep_title(given),
             "standard-ebooks": lambda: standard_ebooks_title(given),
             "ebible": lambda: ebible_title(row["notice"]),
             "gutenberg": lambda: gutenberg_title(row["address"], catalogue)}[family]()
    if title is None:
        counts["left empty"] += 1
        return None
    counts["titled"] += 1
    return {"id": row["id"], "title": title, "was": {"title": None}}


# Writing

class Output:
    """A family's input files in chunks of CHUNK lines and its progress, or the standard output."""

    def __init__(self, folder: Path | None, family: str):
        self.folder, self.family = folder, family
        self.saved = folder / f"{family}.json" if folder else None
        state = json.loads(self.saved.read_text(encoding="utf-8")) if self.saved and self.saved.exists() else {}
        self.after = state.get("after", 0)  # the number of the last id read into a whole chunk
        self.chunk = state.get("chunk", 0)
        self.counts = state.get("counts") or {name: 0 for name in (
            "read", "titled", "left empty", "already titled", "authors fixed", "authors dropped")}
        self.lines, self.file = 0, None
        self.last = self.after  # the number of the last id read

    def write(self, line: dict):
        if self.folder is None:
            sys.stdout.write(json.dumps(line, ensure_ascii=False) + "\n")
            return
        if self.file is None:
            self.file = self._part().open("w", encoding="utf-8", newline="\n")
        self.file.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.lines += 1

    def reached(self, number: int):
        """Every id up to number is read: a whole chunk is closed and saved."""
        self.last = number
        if self.folder is not None and self.lines >= CHUNK:
            self.close(number)

    def close(self, number: int):
        """Close the chunk being written, whole or not, and save how far the family went: up to number."""
        if self.folder is None:
            self.after = number
            return
        if self.file is not None:
            self.file.close()
            self._part().replace(self.folder / f"{self.family}-{self.chunk + 1:04d}.jsonl")
            self.chunk += 1
            self.file, self.lines = None, 0
        self.after = number
        temporary = self.saved.with_suffix(".json.part")
        temporary.write_text(json.dumps({"after": self.after, "chunk": self.chunk, "counts": self.counts},
                                        indent=1) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(self.saved)

    def _part(self) -> Path:
        return self.folder / f"{self.family}-{self.chunk + 1:04d}.jsonl.part"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--depot", required=True, help="the depot's SQLite file, read only")
    parser.add_argument("--out", required=True, help="the folder of the input files, or - for the standard output")
    parser.add_argument("--ledger", action="append", required=True, type=Path, help="a ledger of the added texts")
    parser.add_argument("--catalogue", type=Path, help="Project Gutenberg's pg_catalog.csv")
    parser.add_argument("--family", action="append", choices=sorted(READS), help="only this family")
    parser.add_argument("--range", nargs=2, type=int, metavar=("FIRST", "LAST"), help="only these id numbers")
    args = parser.parse_args(argv)
    reader = Reader(args.depot)
    name = reader.name()
    wanted = set(args.family or READS)
    if "gutenberg" in wanted and args.catalogue is None:
        parser.error("the gutenberg family needs --catalogue")
    catalogue = catalogue_titles(args.catalogue) if "gutenberg" in wanted else {}
    folder = None if args.out == "-" else Path(args.out)
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for first, last, family in families(ledger_entries(args.ledger), name):
        if family not in wanted:
            continue
        if args.range:
            first, last = max(first, args.range[0]), min(last, args.range[1])
        output = outputs.get(family) or outputs.setdefault(family, Output(folder, family))
        for number in range(max(first, output.after + 1), last + 1):
            row = reader.row(f"{name}-{number}")
            if row is None:
                raise SystemExit(f"the ledgers name {name}-{number}, which the depot does not hold")
            line = change_of(family, row, reader, catalogue, output.counts)
            if line is not None:
                output.write(line)
            output.reached(number)
    for family, output in outputs.items():
        output.close(output.last)
        print(json.dumps({"family": family, "chunks": output.chunk, **output.counts}), file=sys.stderr)


if __name__ == "__main__":
    main()
