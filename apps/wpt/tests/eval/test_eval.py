import json
import threading

import pytest

import wpt_telemetry
from wpt.eval import main
from wpt_store import LocalStore
from wpt_telemetry import FLUSH_THREAD
from wpt_utils.lines import bare


def test_a_folder_whose_drafter_was_trained_against_other_weights_is_refused(trained):
    # a model and its drafter are one unit (docs/model.md, "The drafter")
    provenance = trained.folder / "drafter" / "provenance.json"
    provenance.write_text(json.dumps({"target": {"weights": {"model.safetensors": "0" * 64}}}))
    with pytest.raises(SystemExit, match="EvalError: .*its drafter was trained against weights"):
        main([str(trained.folder)])
    assert not (trained.folder / "evals").exists()


def test_any_python_file_can_be_a_test(trained, tmp_path):
    folder = trained.folder
    test = tmp_path / "count_tokens.py"
    test.write_text(
        "from wpt.eval import Result\n\n"
        "def run(context):\n"
        "    corpus = context.provenance['corpus']['id']\n"
        "    return Result({'corpus': corpus}, f'trained on {corpus}')\n")
    main([str(folder), str(test)])

    result = json.loads((folder / "evals" / "count_tokens.json").read_text())
    assert result["file"] == "count_tokens.py" and result["data"]["corpus"] == trained.corpus_id
    assert f"trained on {trained.corpus_id}" in (folder / "README.md").read_text()


def test_a_folder_holding_a_result_of_another_wpt_eval_version_is_evaluated_again(trained, tmp_path):
    # a result written by another version of wpt eval stays in the model card beside the new ones
    folder = trained.folder
    (folder / "evals").mkdir()
    (folder / "evals" / "earlier.json").write_text(json.dumps({
        "name": "earlier", "file": "earlier.py", "file_sha256": "0" * 64, "wpt_eval": "0.1.0",
        "device": "cpu", "data": {}, "text": "an earlier result"}))
    test = tmp_path / "count_tokens.py"
    test.write_text("from wpt.eval import Result\n\ndef run(context):\n    return Result({}, 'a new result')\n")
    main([str(folder), str(test)])

    card = (folder / "README.md").read_text()
    assert "an earlier result" in card and "wpt eval 0.1.0" in card and "a new result" in card
    assert "schema" not in json.loads((folder / "evals" / "count_tokens.json").read_text())


def test_a_test_that_returns_no_result_writes_nothing(trained, tmp_path):
    folder = trained.folder
    test = tmp_path / "broken.py"
    test.write_text("def run(context):\n    return 'all good'\n")
    with pytest.raises(SystemExit, match="returned str, not wpt.eval.Result"):
        main([str(folder), str(test)])
    assert not (folder / "evals" / "broken.json").exists()


def test_a_record_the_store_refuses_stops_the_evaluation_with_code_3(trained, monkeypatch, capsys, tmp_path):
    # every command stops without its record, with the one exit code
    # that says so, before writing the result it could not record
    real = LocalStore._raw_append_record

    def disk_full_at_the_result(self, folder, name, data):
        if b'"name": "eval.result"' in data:
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full_at_the_result)
    test = tmp_path / "constant.py"
    test.write_text("from wpt.eval import Result\n\ndef run(context):\n    return Result(data={'x': 1}, text='x is 1')\n")
    with pytest.raises(SystemExit) as stopped:
        main([str(trained.folder), str(test)])
    assert stopped.value.code == 3 and "No space left on device" in capsys.readouterr().err
    assert not (trained.folder / "evals").exists()


@pytest.mark.parametrize("refused, result_written", [(b'"eval_constant"', False), (b'"model_card"', True)])
def test_on_s3_a_result_and_the_model_card_are_written_only_once_their_record_is_in_the_bucket(
        trained, s3, tmp_path, refused, result_written):
    # S3 holds record lines back: each file's sha256 must be in the bucket
    # before the file is written, so a refused send leaves the earlier files
    test = tmp_path / "constant.py"
    test.write_text("from wpt.eval import Result\n\ndef run(context):\n    return Result(data={'x': 1}, text='x is 1')\n")
    before = {path.name: path.read_bytes() for path in trained.folder.iterdir() if path.is_file()}
    s3.refuse(refused)
    with pytest.raises(SystemExit) as stopped:
        main([str(trained.folder), str(test)])
    assert stopped.value.code == 3
    assert {path.name: path.read_bytes() for path in trained.folder.iterdir() if path.is_file()} == before
    assert (trained.folder / "evals" / "constant.json").exists() == result_written


def test_a_send_refused_in_the_background_stops_the_evaluation_before_its_next_record_call(trained, tmp_path,
                                                                                           monkeypatch, capsys):
    # the command must stop when the send fails, not at its next record call,
    # and exit as a record failure (3), not as a stop asked for (130)
    real = LocalStore._raw_flush_records

    def refused_in_the_background(self):
        if threading.current_thread().name == FLUSH_THREAD:
            raise OSError(28, "No space left on device")
        real(self)

    monkeypatch.setattr(wpt_telemetry, "FLUSH_SECONDS", 0.02)
    monkeypatch.setattr(LocalStore, "_raw_flush_records", refused_in_the_background)
    finished = tmp_path / "finished"
    test = tmp_path / "long.py"  # works without recording anything, far longer than a loaded machine delays the stop
    test.write_text("import pathlib, time\nfrom wpt.eval import Result\n\ndef run(context):\n"
                    "    started = time.monotonic()\n    while time.monotonic() - started < 120:\n        pass\n"
                    f"    pathlib.Path({str(finished)!r}).touch()\n"
                    "    return Result(data={}, text='')\n")
    with pytest.raises(SystemExit) as stopped:
        main([str(trained.folder), str(test)])
    assert stopped.value.code == 3
    assert not finished.exists()  # the end of the run would give the same code
    assert "No space left on device" in capsys.readouterr().err
    assert not (trained.folder / "evals").exists()
