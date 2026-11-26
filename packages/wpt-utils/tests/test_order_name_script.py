import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

from wic_utils.lines import bare

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "order_name.py"


def test_a_name_a_order_or_series_has_is_not_drawn_again(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("wpt_order_name_script", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(subprocess, "run", lambda *_, **__: SimpleNamespace(stdout="fluffy-axolotl\nold-owl\n"))
    monkeypatch.setattr(script, "words", lambda name: [name])
    draws = iter(["fluffy", "axolotl", "brave", "axolotl"])
    monkeypatch.setattr(script.secrets, "choice", lambda _: next(draws))
    script.main()
    assert bare(capsys.readouterr().out) == "brave-axolotl\n"
