"""scripts/source_credits.py: the rule of each family on rows shaped as the depot's own, and a run over a depot that
writes the input files of `depot credits-file` and goes on after a stop."""

import importlib.util
import json
from pathlib import Path

import pytest

from depot import NewText, crediting

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "source_credits.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("wpt_source_credits_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PEP_732 = ("PEP: 732 Title: The Python Documentation Editorial Board Author: Joanna\\\nJablonski Sponsor: Mariatta "
           "Wijaya Discussions-To:\\\nhttps://discuss.python.org/t/pep-732-the-python-documentation-editorial-board/"
           "36710\\\nStatus: Active Type: Process Topic: Governance Created: 14-Oct-2023\\\nPost-History: "
           "20-Oct-2023 Resolution:\\\nhttps://discuss.python.org/t/pep-732-the-python-documentation-editorial-"
           "board/3671\n\nAbstract\n")
PEP_3123 = ("PEP: 3123 Title: Making PyObject\\_HEAD conform to standard C Version:\\\n$Revision$ Last-Modified: $Date$ "
            "Author: Martin von Löwis\\\n\\<EMAIL\\_ADDRESS>Status: Final Type: Standards Track Content-Type:\\\n"
            "text/x-rst Created: 27-Apr-2007 Python-Version: 3.0 Post-History:\n\nAbstract\n\nPython currently relies")
OPF = """=== LICENSE.md
The source text and artwork in this ebook are believed to be in the United States public domain.
=== src/epub/content.opf
<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" dir="ltr" unique-identifier="uid" version="3.0" xml:lang="en-GB">
\t<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
\t\t<dc:identifier id="uid">https://standardebooks.org/ebooks/a-e-w-mason/at-the-villa-rose</dc:identifier>
\t\t<dc:publisher id="publisher">Standard Ebooks</dc:publisher>
\t\t<dc:title id="title">At the Villa Rose</dc:title>
\t\t<dc:creator id="author">A. E. W. Mason</dc:creator>
\t</metadata>
\t<spine>
\t\t<itemref idref="uncopyright.xhtml"/>
\t</spine>
</package>
=== src/epub/css/core.css
@charset "utf-8";
"""


@pytest.mark.parametrize("address, title", [
    ("https://raw.githubusercontent.com/jfuerlinger/leocloud/3e863f262be0e391d7803cf6713c7242e5494a10/www/docker/"
     "readme.md", "jfuerlinger/leocloud/www/docker/readme.md"),
    ("https://raw.githubusercontent.com/sebastienboisvert/ZX81QuickLook/0123456789abcdef0123456789abcdef01234567/"
     "ZX81QuickLook/ZX81 Functions/README.md", "sebastienboisvert/ZX81QuickLook/ZX81QuickLook/ZX81 Functions/README.md"),
    ("https://raw.githubusercontent.com/mikewong-sfsu/GeneDive/d2f11eb9f1bf63c9e494f5039b098536f9d5aa96/frontend/"
     "docs/html/howtouse.html", "mikewong-sfsu/GeneDive/frontend/docs/html/howtouse.html"),
    ("https://github.com/callstack/linaria", None),
    (None, None),
])
def test_a_code_file_is_titled_owner_repo_and_path_from_its_address(script, address, title):
    assert script.code_file_title(address) == title


@pytest.mark.parametrize("address, title", [
    ("https://github.com/NationalSecurityAgency/ghidra", "NationalSecurityAgency/ghidra"),
    ("https://github.com/abpframework/abp", "abpframework/abp"),
    ("https://github.com/0no-co/GraphQLSP/issues/180", None),
])
def test_a_repository_is_titled_owner_and_name_from_its_address(script, address, title):
    assert script.repository_title(address) == title


@pytest.mark.parametrize("opening, title", [
    ("Field used but still warned as not used\n\nWhen I", "Field used but still warned as not used"),
    ("Field used but still warned as not used\n", "Field used but still warned as not used"),
    ("Production of charmonium $\\\\chi\\_{cJ}(2P)$ plus one $\\\\omega$ meson by $e^+e^-$ annihilation",
     "Production of charmonium $\\chi_{cJ}(2P)$ plus one $\\omega$ meson by $e^+e^-$ annihilation"),
    ("System.IO.DirectoryNotFoundException: Could not find a part of the path 'C:\\\\Actor\\\\Pack\\\\",
     "System.IO.DirectoryNotFoundException: Could not find a part of the path 'C:\\Actor\\Pack\\"),
    ("Issue Type: Bug\\\nDescription: the build fails\n\nSteps", None),
    ("\\[X\\] I can confirm", "[X] I can confirm"),
    ("The first line\nof a paragraph\n\nBody", None),
    ("x" * 300, "x" * 300),
    ("x" * 301, None),
    ("\\documentclass\\[12pt\\]{minimal}\n\nBody", None),
    ("   ", None),
    ("", None),
])
def test_an_issue_or_a_paper_is_titled_by_its_first_line_alone_as_its_markdown_reads(script, opening, title):
    assert script.first_line_title(opening) == title


@pytest.mark.parametrize("opening, title", [
    ("Adaptive Sampling of\\\nSparse Signals\\\nin Practice\n\nAbstract",
     "Adaptive Sampling of Sparse Signals in Practice"),
    ("A title\\\nthen a line\nwith no hard break\n\nAbstract", None),
    ("Issue Type: Bug\\\nDescription: the build fails", "Issue Type: Bug Description: the build fails"),
    (("word " * 40 + "\\\n") * 2 + "\n", None),
])
def test_a_paper_s_title_split_by_hard_breaks_in_its_first_paragraph_is_joined(script, opening, title):
    assert script.first_line_title(opening, joins_hard_breaks=True) == title


@pytest.mark.parametrize("head, title", [
    (PEP_732, "The Python Documentation Editorial Board"),
    (PEP_3123, "Making PyObject_HEAD conform to standard C"),
    ("Abstract\n\nPython currently relies", None),
])
def test_a_pep_is_titled_by_the_title_field_of_its_header(script, head, title):
    assert script.pep_title(head) == title


def test_a_standard_ebook_is_titled_by_the_dc_title_of_its_content_opf_block(script):
    assert script.standard_ebooks_title(OPF) == "At the Villa Rose"
    assert script.standard_ebooks_title(OPF.replace("=== src/epub/content.opf", "=== src/epub/other.opf")) is None


def test_an_ebible_text_is_titled_by_its_notice_before_ebible_org(script):
    assert script.ebible_title("الكتاب المقدس باللغة العربية، فان دايك, eBible.org, https://ebible.org/arb-vd/") \
        == "الكتاب المقدس باللغة العربية، فان دايك"
    assert script.ebible_title("Some other notice") is None


@pytest.mark.parametrize("address, number", [
    ("https://aleph.pglaf.org/2/3/5/1/23510/23510-h/23510-h.htm", "23510"),
    ("https://aleph.pglaf.org/7/9/793/793-h/793-h.htm", "793"),
    ("https://aleph.pglaf.org/2/20/20-h/20-h.htm", "20"),
    ("https://aleph.pglaf.org/3/3/5/3352/3352.txt", "3352"),
    ("https://aleph.pglaf.org/9/2/5/9259/9259-8.txt", "9259"),
    ("https://aleph.pglaf.org/9/2/5/9259/other.txt", None),
])
def test_a_gutenberg_address_gives_its_ebook_number(script, address, number):
    assert script.gutenberg_number(address) == number


def test_a_gutenberg_ebook_is_titled_by_the_catalogue_its_lines_joined(script, tmp_path):
    catalogue = tmp_path / "pg_catalog.csv"
    catalogue.write_text('Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\r\n'
                         '9999,Text,2006-02-01,"Harriet, the Moses of Her People",en,"Bradford, Sarah H.",,,\r\n'
                         '20,Text,1991-01-01,"Paradise Lost",en,"Milton, John",,,\r\n'
                         '2,Text,1972-12-01,"The United States Bill of Rights\r\nThe Ten Original Amendments to the '
                         'Constitution of the United States",en,United States,,,\r\n', encoding="utf-8", newline="")
    titles = script.catalogue_titles(catalogue)

    assert script.gutenberg_title("https://aleph.pglaf.org/9/9/9/9999/9999-h/9999-h.htm", titles) \
        == "Harriet, the Moses of Her People"
    assert script.gutenberg_title("https://aleph.pglaf.org/2/2.txt", titles) \
        == "The United States Bill of Rights: The Ten Original Amendments to the Constitution of the United States"
    assert script.gutenberg_title("https://aleph.pglaf.org/1/2/123/123.txt", titles) is None


def test_pubmed_authors_stored_as_python_dicts_become_names_and_empty_ones_go(script):
    assert script.fixed_authors(["{'first': 'Jean M', 'last': 'Feugang'}", "{'first': '', 'last': 'Lee'}",
                                 "{'first': '', 'last': ''}", "{'first': 'Manuel', 'last': 'Llinás'}"]) == (
        ["Jean M Feugang", "Lee", "Manuel Llinás"], 1)
    assert script.fixed_authors(["Ann Lee"]) is None


def test_the_script_s_files_are_as_long_as_one_change_of_the_depot_takes(script):
    assert script.CHUNK == crediting.MOST


def test_a_run_writes_each_family_s_changes_in_chunks_and_goes_on_after_a_stop_in_a_chunk(script, make_depot, me,
                                                                                          tmp_path, monkeypatch):
    depot = make_depot()
    code = "https://raw.githubusercontent.com/owner/repo/0123456789abcdef0123456789abcdef01234567/src/{}.py"
    depot.add([NewText(text="x", licenses=("CC0-1.0",), address="https://en.wikipedia.org/wiki/X", title="X")]
              + [NewText(text=f"file {n}", licenses=("CC0-1.0",), address=code.format(n)) for n in range(5)]
              + [NewText(text="kept", licenses=("CC0-1.0",), address=code.format("kept"), title="Kept")]
              + [NewText(text="Body", licenses=("CC0-1.0",), authors=("{'first': 'Jean M', 'last': 'Feugang'}",)),
                 NewText(text="Body", licenses=("CC0-1.0",), authors=("Ann Lee",)),
                 NewText(text="Body", licenses=("CC0-1.0",), authors=("{'first': '', 'last': ''}",))], me)
    ledger = tmp_path / "added.jsonl"
    ledger.write_text("".join(json.dumps(one) + "\n" for one in [
        {"kind": "start", "total": 0},
        {"kind": "intent", "source": "wikimedia-wikipedia-enwiki-00000000", "start": 0, "count": 1, "before": 0},
        {"kind": "intent", "source": "huggingface-common-pile-stackv2-edu-filtered-00000000", "start": 0,
         "count": 6, "before": 1},
        {"kind": "intent", "source": "huggingface-common-pile-pubmed-filtered-00000000", "start": 0, "count": 3,
         "before": 7}]), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(script, "CHUNK", 2)
    arguments = ["--depot", depot.address, "--out", str(out), "--ledger", str(ledger), "--family", "stackv2-edu",
                 "--family", "pubmed"]

    write, written = script.Output.write, []

    def stopping(output, line):
        if len(written) == 3:  # the second chunk begun, its first line written
            raise KeyboardInterrupt
        written.append(line)
        write(output, line)

    monkeypatch.setattr(script.Output, "write", stopping)
    with pytest.raises(KeyboardInterrupt):
        script.main(arguments)
    assert sorted(one.name for one in out.iterdir()) == [
        "stackv2-edu-0001.jsonl", "stackv2-edu-0002.jsonl.part", "stackv2-edu.json"]
    monkeypatch.setattr(script.Output, "write", write)
    script.main(arguments)

    assert sorted(one.name for one in out.iterdir()) == [
        "pubmed-0001.jsonl", "pubmed.json", "stackv2-edu-0001.jsonl", "stackv2-edu-0002.jsonl",
        "stackv2-edu-0003.jsonl", "stackv2-edu.json"]
    lines = [json.loads(line) for name in ("stackv2-edu-0001", "stackv2-edu-0002", "stackv2-edu-0003")
             for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines == [{"id": f"depot-{n + 2}", "title": f"owner/repo/src/{n}.py", "was": {"title": None}}
                     for n in range(5)]
    assert [json.loads(line) for line in (out / "pubmed-0001.jsonl").read_text(encoding="utf-8").splitlines()] == [
        {"id": "depot-8", "authors": ["Jean M Feugang"], "was": {"authors": ["{'first': 'Jean M', 'last': 'Feugang'}"]}},
        {"id": "depot-10", "authors": None, "was": {"authors": ["{'first': '', 'last': ''}"]}}]
    assert json.loads((out / "stackv2-edu.json").read_text(encoding="utf-8"))["counts"] == {
        "read": 6, "titled": 5, "left empty": 0, "already titled": 1, "authors fixed": 0, "authors dropped": 0}
    assert json.loads((out / "pubmed.json").read_text(encoding="utf-8"))["counts"]["authors fixed"] == 2

    # the lines are what the depot's change takes, and a title set after the run stays
    depot.edit_credits("depot-3", {"title": "Set by a person"}, me)
    assert depot.edit_credits_many(out / "stackv2-edu-0001.jsonl", me)[2:] == (1, 1)
    assert [depot.show(one)["text"]["title"] for one in ("depot-2", "depot-3")] == [
        "owner/repo/src/0.py", "Set by a person"]
    assert depot.edit_credits_many(out / "pubmed-0001.jsonl", me)[2] == 2
    assert depot.show("depot-10")["text"]["authors"] is None
