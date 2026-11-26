"""The dock command: find, fetch and put on the dock DOCK names."""

import hashlib
import json

import pytest

from dock_app import main


@pytest.fixture
def dock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env of the checkout is read
    monkeypatch.setenv("DOCK", (tmp_path / "dock").as_uri())
    return tmp_path


def test_a_file_put_is_found_and_fetched_back_checked(dock, capsys):
    data = b"the fox " * 1000
    (dock / "fox.txt").write_bytes(data)
    assert main(["put", "deliveries/k/files/fox.txt", str(dock / "fox.txt")]) == 0
    stored = json.loads(capsys.readouterr().out)
    assert stored == {"key": "deliveries/k/files/fox.txt", "sha256": hashlib.sha256(data).hexdigest(),
                      "bytes": len(data)}

    assert main(["find", "deliveries/k/"]) == 0
    [found] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert found["key"] == "deliveries/k/files/fox.txt" and found["bytes"] > 0

    assert main(["fetch", "deliveries/k/files/fox.txt", "--sha256", stored["sha256"], "--out",
                 str(dock / "back.txt")]) == 0
    assert (dock / "back.txt").read_bytes() == data


def test_a_fetch_refuses_a_missing_file_a_wrong_sha256_and_an_existing_out_file(dock, capsys):
    (dock / "a.txt").write_bytes(b"a")
    assert main(["put", "a", str(dock / "a.txt")]) == 0
    assert main(["fetch", "nothing-here"]) == 1
    assert "MissingFile" in capsys.readouterr().err
    assert main(["fetch", "a", "--sha256", "0" * 64]) == 1
    assert main(["fetch", "a", "--out", str(dock / "a.txt")]) == 1
    assert (dock / "a.txt").read_bytes() == b"a"


def test_without_dock_the_command_says_what_to_set(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCK", raising=False)
    assert main(["find", ""]) == 1
    assert "DOCK is not set" in capsys.readouterr().err
