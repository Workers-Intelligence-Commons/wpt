import pytest

from wpt_depot import About, Declaration, Depot, tables

ME = Declaration("Ada Lovelace", "ada@example.org", True)


@pytest.fixture
def me():
    return ME


@pytest.fixture
def make_depot(tmp_path):
    """make_depot(name): a new depot in the test's folder, with the license
    CC0-1.0."""
    def make(name="depot"):
        depot = Depot.new(str(tmp_path / name), About(name, "Ada Lovelace", "ada@example.org"))
        depot.add_license("CC0-1.0", "Creative Commons Zero v1.0 Universal", "The CC0 text.")
        return depot
    return make


@pytest.fixture(params=["as set", "every second version"])
def compaction(request, monkeypatch):
    """Runs the test with compaction as the tables set it, then again with
    every table compacted at every second storage version, so the test
    meets merged files and checkpoints after a few writes."""
    if request.param == "every second version":
        monkeypatch.setattr(tables, "COMPACT_EVERY", 2)
    return request.param
