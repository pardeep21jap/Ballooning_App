"""BalloonIQ entry point.

Run from the project root with:

    python main.py
"""

from __future__ import annotations

import sys

from balloon_app.app import main as run_app

if __name__ == "__main__":
    sys.exit(run_app())
