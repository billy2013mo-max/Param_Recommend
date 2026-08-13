#!/usr/bin/env python3
"""Refresh the immutable original 60-row report after resume-2 finishes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import ROOT


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_h800_final_memory_business_blind_v2.py"),
            "--allow-incomplete",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
