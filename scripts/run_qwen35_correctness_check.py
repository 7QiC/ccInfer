#!/usr/bin/env python3
"""Run the Qwen3.5 numpy reference and write C++ comparison artifacts.

Usage:
    python scripts/run_qwen35_correctness_check.py \
        --gguf models/qwen3_5-2B/Qwen3.5-2B-Q8_0.gguf \
        --prompt "Hello world"

This is a convenience wrapper around qwen35_reference.py. It writes
ref_logits_qwen35.bin under <gguf_parent>/ref for use by the M1 integration
tests (T1.8).
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", default="models/qwen3_5-2B/Qwen3.5-2B-Q8_0.gguf")
    p.add_argument("--prompt", default="Hello world")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Optionally self-check chunked prefill against one-shot")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default <gguf_parent>/ref)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gguf = Path(args.gguf).resolve()
    if not gguf.exists():
        print(f"error: GGUF not found: {gguf}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/qwen35_reference.py"),
        str(gguf),
        "--prompt",
        args.prompt,
    ]
    if args.chunk_size is not None:
        cmd += ["--chunk-size", str(args.chunk_size)]
    if args.output_dir is not None:
        cmd += ["--output-dir", str(Path(args.output_dir).resolve())]

    print(f"RUN: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
