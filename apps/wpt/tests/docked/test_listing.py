"""`wpt list`, `runs` and `run`: what the dock holds, and its runs."""

from freight import corpora
from wpt.models import keep_model, put_model
from wpt_telemetry import Telemetry

from wpt.docked import main


def _record(store, message):
    with Telemetry.start(store, "wpt train", [], "x", env_file=store.root / ".env", environ={},
                         echo=lambda *_: None) as telemetry:
        telemetry.log(message)


def test_the_dock_lists_its_corpora_models_and_runs(built, tmp_path, capsys):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}\n")
    keep_model(built.store, "fox-1", put_model(built.store, "fox-1", tmp_path / "model"))
    _record(built.store, "a run")
    capsys.readouterr()

    assert main(["list"]) == 0
    listed = capsys.readouterr().out
    record = corpora.find(built.store, built.corpus_id)
    assert (f"{built.corpus_id}: build key {record['build_key']}, snapshot {built.names['d']} (change {record['snapshot']['change']}) "
            f"of {built.depot.name()}, order d, every token" in listed and "  fox-1" in listed)

    assert main(["runs"]) == 0
    assert "wpt train" in capsys.readouterr().out
    assert main(["run", "wpt-train"]) == 0  # a part of the name only one folder holds
    assert "wpt train" in capsys.readouterr().out


def test_copying_out_a_model_the_dock_does_not_keep_is_refused(built, tmp_path, capsys):
    assert main(["copy-out", "fox-1", str(tmp_path / "out")]) == 2
    assert "MissingFile" in capsys.readouterr().err and not (tmp_path / "out").exists()
