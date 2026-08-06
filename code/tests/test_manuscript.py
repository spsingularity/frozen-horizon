"""The manuscripts must not drift from the code.

scripts/check_manuscript.py recomputes every number quoted in the prose of
both papers and asserts it appears in the .tex. Running it under pytest means
a code change that moves a published number fails the suite rather than
silently leaving a stale claim in the manuscript -- which is exactly how
N_tot = 74.403, a value belonging to the superseded p = 67 run, survived into
the first public release of Paper II.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "scripts" / "check_manuscript.py"


@pytest.mark.slow
def test_manuscript_numbers_match_the_pipeline():
    result = subprocess.run([sys.executable, str(CHECKER)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, (
        "manuscript prose disagrees with the code:\n" + result.stdout)
