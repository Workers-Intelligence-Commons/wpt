import re
from pathlib import Path

import pytest

from wpt_depot import licenses

DOCUMENT = Path(__file__).parents[3] / "docs" / "licenses.md"


def listed(section_from: str, section_to: str) -> set[str]:
    """The licenses in the first column of the tables between two headings of docs/licenses.md."""
    text = DOCUMENT.read_text(encoding="utf-8")
    found = text[text.index(section_from):text.index(section_to)]
    return {name for line in found.splitlines() if line.startswith("| `")
            for name in re.findall(r"`([^`]+)`", line.split("|")[1])}


def test_the_allowed_licenses_are_those_of_the_tables_of_the_documentation():
    accepted = listed("### Public domain and permissive", "### Several licenses on one text")

    assert {name for name in accepted if licenses.allowed(name)} == accepted - {"EPL-2.0"}  # EPL-2.0 alone is refused
    assert licenses.ALLOWED == accepted - {"EPL-2.0", "GPL-2.0-or-later WITH Classpath-exception-2.0"}
    refused = listed("### Waiting for the linking test", "## Rules")
    assert not any(licenses.allowed(name) for name in refused)


@pytest.mark.parametrize("license_id, expected", [
    ("MIT", True), ("GPL-2.0-only", False), ("Apache-2.0 WITH LLVM-exception", True),
    ("MIT WITH LLVM-exception2", False), ("GPL-3.0-or-later WITH GCC-exception-3.1", True),
    ("MIT AND CC-BY-4.0", True), ("MIT AND GPL-2.0-only", False), ("MIT AND Apache-2.0 WITH LLVM-exception", True),
    ("EPL-2.0", False), ("", False)])
def test_a_combined_license_is_allowed_only_if_every_part_is(license_id, expected):
    assert licenses.allowed(license_id) is expected

