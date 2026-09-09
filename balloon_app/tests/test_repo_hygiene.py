"""Guards against stray pytest scratch dirs (e.g. .pytest-fix-126) getting committed."""
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _in_git_checkout() -> bool:
    return (REPO_ROOT / ".git").exists()


@pytest.mark.skipif(not _in_git_checkout(), reason="not running inside a git checkout")
def test_no_pytest_scratch_dirs_are_tracked():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_scratch_files = [
        line for line in result.stdout.splitlines() if line.startswith(".pytest-")
    ]
    assert tracked_scratch_files == []
