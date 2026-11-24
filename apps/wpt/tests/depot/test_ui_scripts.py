import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[2] / "ui"


@pytest.mark.skipif(shutil.which("bun") is None, reason="the UI's own tests run with Bun")
def test_the_ui_s_own_tests_pass():
    # the calculations of the pages that need no browser (apps/wpt/ui/src/**/*.test.ts), such as the choices the
    # review panel offers for a score from the values the depot holds
    found = subprocess.run(["bun", "test"], cwd=UI, capture_output=True, text=True, timeout=120)
    assert found.returncode == 0, found.stdout + found.stderr
