#!/usr/bin/env python3
"""Evaluate all currently terminal original blind jobs after exact resume."""

from __future__ import annotations

import sys

import evaluate_h800_final_memory_business_blind_v2 as evaluator

if __name__ == "__main__":
    if "--allow-incomplete" not in sys.argv:
        sys.argv.append("--allow-incomplete")
    evaluator.main()
