#!/usr/bin/env python
"""Convenience entrypoint: python scripts/run_solve.py -p \"...\""""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solve import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
