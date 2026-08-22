#!/usr/bin/env python3
"""Lightweight text filter for Nsight Compute raw exports."""

from __future__ import annotations

import argparse
from pathlib import Path


KEYWORDS = (
    "Kernel Name",
    "Duration",
    "SM",
    "Memory",
    "Occupancy",
    "Warp",
    "Stall",
    "Register",
    "Throughput",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter important lines from ncu raw text")
    parser.add_argument("raw_txt", type=Path)
    args = parser.parse_args()

    print(f"# ncu summary: {args.raw_txt}")
    for line in args.raw_txt.read_text(errors="replace").splitlines():
        if any(k.lower() in line.lower() for k in KEYWORDS):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
