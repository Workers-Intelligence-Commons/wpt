import json
import threading

import pytest

import wpt_telemetry
from wpt import resources
from wpt.eval import BUILT_IN, main
from dock import api as dock_api
from wpt_telemetry import FLUSH_THREAD
from wic_utils.lines import bare


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


def test_a_record_the_dock_refuses_stops_the_evaluation_with_code_3(trained, monkeypatch, capsys, tmp_path):
    # every command stops without its record, with the one exit code
    # that says so, before writing the result it could not record
    real = dock_api.append_record

    def disk_full_at_the_result(dock, folder, name, data):
        if b'"name": "eval.result"' in data:
            raise OSError(28, "No space left on device")
        real(dock, folder, name, data)

    monkeypatch.setattr(dock_api, "append_record", disk_full_at_the_result)
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
    real = dock_api.flush_records

    def refused_in_the_background(dock):
        if threading.current_thread().name == FLUSH_THREAD:
            raise OSError(28, "No space left on device")
        real(dock)

    monkeypatch.setattr(wpt_telemetry, "FLUSH_SECONDS", 0.02)
    monkeypatch.setattr(dock_api, "flush_records", refused_in_the_background)
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


def test_general_knowledge_asks_each_question_in_both_languages(trained, tmp_path):
    # the built-in test, beside a file of two of its questions (one ends with an apostrophe in French)
    built_in = BUILT_IN / "general_knowledge.py"
    questions = json.loads(built_in.with_suffix(".json").read_text(encoding="utf-8"))
    two = [questions[0], next(one for one in questions if one["fr"]["prompt"].endswith("'"))]
    test = tmp_path / "general_knowledge.py"
    test.write_text(built_in.read_text(encoding="utf-8"), encoding="utf-8")
    test.with_suffix(".json").write_text(json.dumps(two), encoding="utf-8")
    main([str(trained.folder), str(test)])
    data = json.loads((trained.folder / "evals" / "general_knowledge.json").read_text())["data"]
    assert data["questions"] == 2 and set(data["languages"]) == {"fr", "en"}
    assert [one["prompt"] for one in data["answers"]["fr"]] == [one["fr"]["prompt"] for one in two]
    assert all(0 <= figures["choice"] <= 1 for figures in data["languages"]["en"].values())
    assert all(one["rank"] >= 1 and 0 < one["probability"] <= 1 for found in data["answers"].values() for one in found)
    figures = data["languages"]["fr"]["all"]
    assert figures["rank"] >= 1 and 0 <= figures["top_1"] <= figures["top_10"] <= 1
    card = (trained.folder / "README.md").read_text()
    assert "| **all** |" in card and "| fr rank | fr top 1 | fr top 10 | fr probability |" in card


def test_free_text_shows_what_the_model_writes_from_each_opening(trained, tmp_path):
    # the built-in test, with fewer and shorter texts
    test = tmp_path / "free_text.py"
    test.write_text((BUILT_IN / "free_text.py").read_text(encoding="utf-8").replace("ROWS = 4", "ROWS = 2").replace(
        "WRITTEN = 200", "WRITTEN = 30"), encoding="utf-8")
    main([str(trained.folder), str(test)])
    data = json.loads((trained.folder / "evals" / "free_text.json").read_text())["data"]
    assert set(data["languages"]) == {"fr", "en"}
    for found in data["languages"].values():
        assert len(found["texts"]) == 3 * 2 and all(one["tokens"] <= 30 for one in found["texts"])
        assert 0 <= found["repeated"] <= 1 and 0 <= found["repeating"] <= 1
    card = (trained.folder / "README.md").read_text()
    assert "| fr |" in card
    assert all(one["opening"] + one["text"] in card for found in data["languages"].values()
               for one in found["texts"][::2])  # the first text of each opening
    main([str(trained.folder), str(test)])  # the seed gives the same texts again
    assert json.loads((trained.folder / "evals" / "free_text.json").read_text())["data"] == data


def test_answer_from_context_scores_each_format_greedy_and_sampled(trained, tmp_path):
    test = tmp_path / "answer_from_context.py"
    test.write_text((BUILT_IN / "answer_from_context.py").read_text(encoding="utf-8").replace(
        "PROMPTS = 50", "PROMPTS = 3"), encoding="utf-8")
    main([str(trained.folder), str(test)])
    data = json.loads((trained.folder / "evals" / "answer_from_context.json").read_text())["data"]
    assert list(data["formats"]) == ["fr, question", "fr, chat", "en, question", "en, chat"]
    assert len(data["items"]) == 12 and all("\n" not in one["greedy_line"] + one["sampled_line"]
                                            for one in data["items"])
    assert all(0 <= one[key] <= 1 for one in data["formats"].values() for key in ("greedy", "sampled", "probability"))
    assert all(one["rank"] >= 1 for one in data["kinds"].values())
    assert "| en, chat |" in (trained.folder / "README.md").read_text()


def test_copying_measures_what_the_model_reuses_from_its_prompt(trained, tmp_path):
    # the built-in test, with fewer prompts and a lower floor of words: the test vocabulary is small
    source = (BUILT_IN / "copying.py").read_text(encoding="utf-8")
    test = tmp_path / "copying.py"
    test.write_text(source.replace("PROMPTS = 200", "PROMPTS = 6").replace("FEWEST_WORDS = 16", "FEWEST_WORDS = 8"),
                    encoding="utf-8")
    main([str(trained.folder), str(test)])
    data = json.loads((trained.folder / "evals" / "copying.json").read_text())["data"]
    assert data["words"] >= 8 and data["sequence"]["prompts"] == 6
    assert 0 <= data["sequence"]["tokens"] <= 1 and data["sequence"]["first_nats"] > 0
    assert set(data["stated"]) == {"en", "fr"}
    assert all(0 <= figures[key] <= 1 for found in data["stated"].values() for figures in found.values()
               for key in ("choice", "written", "first_probability"))
    assert all(figures["first_rank"] >= 1 for found in data["stated"].values() for figures in found.values())
    assert "| en, word |" in (trained.folder / "README.md").read_text()


def test_copying_measures_nothing_with_a_vocabulary_of_too_few_whole_words(trained, tmp_path):
    test = tmp_path / "copying.py"
    test.write_text((BUILT_IN / "copying.py").read_text(encoding="utf-8").replace(
        "FEWEST_WORDS = 16", "FEWEST_WORDS = 10**6"), encoding="utf-8")
    main([str(trained.folder), str(test)])
    record = json.loads((trained.folder / "evals" / "copying.json").read_text())
    assert "sequence" not in record["data"] and "nothing measured" in record["text"]


def test_bits_per_byte_measures_the_model_on_the_held_out_portion_of_its_corpus(trained):
    main([str(trained.folder), "bits_per_byte"])
    record = json.loads((trained.folder / "evals" / "bits_per_byte.json").read_text())
    data, note = record["data"], trained.notes["corpus"]
    held = resources.held_out(note)
    # every text of the held-out portion, and its bytes as the depot's listing gives them
    assert data["delivery"] == note["key"] and (data["texts"], data["bytes"]) == (held["texts"], held["bytes"]) == (
        2, sum(one["bytes"] for one in note["portions"] if one["name"] == "held_out"))
    assert data["tokens"] == held["total_tokens"] and data["bits_per_byte"] > 0 and data["nats_per_token"] > 0
    # in windows of the model's context, 16 tokens here, with the error the spread between them gives
    assert data["block_size"] == 16 and data["windows"] == -(-data["tokens"] // 16) and data["texts_left_out"] == 0
    assert 0 < data["bits_per_byte_error"] < data["bits_per_byte_spread"]
    assert set(data["by_share"]) == {"en"} and data["by_share"]["en"]["bits_per_byte"] == pytest.approx(
        data["bits_per_byte"])
    assert f"{data['bits_per_byte']:.4f} ± {data['bits_per_byte_error']:.4f} bits per byte" in (
        trained.folder / "README.md").read_text()


NO_HELD_OUT = [
    (lambda provenance, notes: provenance.pop("deliveries"), "a corpus the freight did not deliver"),
    # a delivery without portions: the vocabulary's stands for one here
    (lambda provenance, notes: provenance["deliveries"]["corpus"].update(key=notes["vocabulary"]["key"]),
     "has no held-out portion"),
]


def changed(trained, change):
    """The model folder's provenance.json, changed by change(provenance, the delivery notes)."""
    path = trained.folder / "provenance.json"
    provenance = json.loads(path.read_text())
    change(provenance, trained.notes)
    path.write_text(json.dumps(provenance))


@pytest.mark.parametrize("change, refused", [
    *NO_HELD_OUT,
    (lambda provenance, notes: provenance["deliveries"]["corpus"].update(key="0" * 64), "holds no corpus delivery"),
])
def test_bits_per_byte_named_refuses_a_model_whose_corpus_has_no_held_out_portion(trained, change, refused):
    changed(trained, change)
    with pytest.raises(SystemExit, match=f"EvalError: .*{refused}"):
        main([str(trained.folder), "bits_per_byte"])
    assert not (trained.folder / "evals" / "bits_per_byte.json").exists()


@pytest.mark.parametrize("change, reason", NO_HELD_OUT)
def test_every_built_in_test_runs_on_a_model_whose_corpus_has_no_held_out_portion_which_is_not_measured(
        trained, change, reason):
    changed(trained, change)
    main([str(trained.folder)])
    records = {path.stem: json.loads(path.read_text()) for path in (trained.folder / "evals").glob("*.json")}
    assert set(records) == {"answer_from_context", "bits_per_byte", "copying", "free_text",
                            "general_knowledge"}  # the run went on past it
    data = records["bits_per_byte"]["data"]
    assert data["measured"] is False and reason in data["reason"] and set(data) == {"measured", "reason"}
    assert str(trained.folder) not in data["reason"]  # the folder never records where it lives
    card = (trained.folder / "README.md").read_text()
    shown = card[card.index("### bits_per_byte"):card.index("Data: [evals/bits_per_byte.json]")]
    assert "Not measured: " in shown and reason in shown and "bits per byte" not in shown
