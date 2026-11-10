import re
from pathlib import Path

import pytest

from wpt_depot import licenses
from wpt_depot.errors import NoBucketSetting
from wpt_depot.settings import bucket_settings
from wpt_depot.tables import Tables

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


def test_a_depot_in_a_bucket_reads_its_own_settings_and_none_of_the_store_s(tmp_path):
    env = tmp_path / ".env"
    env.write_text("WPT_DEPOT_ACCESS_KEY_ID=key\nWPT_DEPOT_SECRET_ACCESS_KEY=secret\nWPT_DEPOT_REGION=eu-west-1\n"
                   "WPT_STORE_ACCESS_KEY_ID=store\n", encoding="utf-8")
    found = bucket_settings({"WPT_DEPOT_ENDPOINT_URL": "http://127.0.0.1:9100", "AWS_ACCESS_KEY_ID": "aws"}, env)

    assert found == {"access_key_id": "key", "secret_access_key": "secret", "region": "eu-west-1",
                     "endpoint_url": "http://127.0.0.1:9100"}
    options = Tables("s3://bucket/depot", found).options()
    assert options["AWS_ACCESS_KEY_ID"] == "key" and options["AWS_ENDPOINT_URL"] == "http://127.0.0.1:9100"
    assert options["AWS_ALLOW_HTTP"] == "true" and options["AWS_REGION"] == "eu-west-1"
    assert Tables("s3://bucket/depot", found).uri("texts") == "s3://bucket/depot/texts"
    with pytest.raises(NoBucketSetting):
        bucket_settings({"WPT_STORE_ACCESS_KEY_ID": "store", "WPT_STORE_SECRET_ACCESS_KEY": "secret"})
