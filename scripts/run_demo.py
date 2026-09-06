#!/usr/bin/env python3
"""
Thin wrapper intended to become the single-command entry point for the
final screen-recording demo.

At this milestone this simply forwards to main.py — it exists now so the
repository structure and eventual demo command are stable, but it has no
additional logic yet. Later milestones will extend this to run the
"VERIFIED" and "TAMPER_DETECTED" scenarios back to back for the
recording, per the approved demo architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import main

if __name__ == "__main__":
    sys.exit(main())
