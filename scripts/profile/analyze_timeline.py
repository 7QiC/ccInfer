#!/usr/bin/env python3
"""Extract a compact summary from nsys stats text output."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize nsys *_stats.txt")
    parser.add_argument("stats_txt", type=Path)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    text = args.stats_txt.read_text(errors="replace")
    print(f"# nsys summary: {args.stats_txt}")

    sections = {
        "CUDA Kernel Summary": "cuda_gpu_kern_sum",
        "CUDA Memcpy/Memset Summary": "cuda_gpu_mem_time_sum",
        "OS Runtime Summary": "osrt_sum",
    }
    for title, marker in sections.items():
        idx = text.find(marker)
        print(f"\n## {title}")
        if idx < 0:
            print("section not found")
            continue
        start = text.rfind("\n", 0, idx)
        start = 0 if start < 0 else start + 1
        next_processing = text.find("\nProcessing [", idx + 1)
        chunk = text[start : next_processing if next_processing > 0 else len(text)]
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        shown = 0
        for ln in lines:
            if re.search(r"\d", ln):
                print(ln)
                shown += 1
            if shown >= args.top + 3:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
