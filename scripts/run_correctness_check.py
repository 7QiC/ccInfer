#!/usr/bin/env python3
"""One-command correctness check: ccInfer vs HuggingFace Transformers for Qwen3-0.6B.

Workflow:
  1. Generate (or reuse) HF reference artifacts per prompt
  2. Build and run the C++ logits-match test
  3. Report alignment metrics

Usage:
    python scripts/run_correctness_check.py --model models/qwen3-0.6B
    python scripts/run_correctness_check.py --skip-ref-gen --skip-build  # re-run only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ccInfer correctness check")
    p.add_argument("--model", default="models/qwen3-0.6B", help="Model directory")
    p.add_argument("--prompts", nargs="+",
                   default=["Hello", "Hello world"],
                   help="Prompts to test")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--build-dir", default="build", help="CMake build directory")
    p.add_argument("--skip-ref-gen", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, **kwargs)


def generate_refs(model_dir: Path, prompts: list[str], max_new_tokens: int) -> None:
    for prompt in prompts:
        safe = prompt.replace(" ", "_").replace("/", "_")[:32]
        out = model_dir / "ccinfer_correctness_ref" / safe
        run([sys.executable,
             str(PROJECT_ROOT / "scripts/qwen3_correctness_reference.py"),
             "--model", str(model_dir), "--prompt", prompt,
             "--max-new-tokens", str(max_new_tokens),
             "--output-dir", str(out)])

    # Legacy format consumed by the GTest binary
    run([sys.executable,
         str(PROJECT_ROOT / "scripts/save_ref_logits.py"),
         "--model", str(model_dir), "--prompt", "Hello",
         "--output", str(model_dir / "ref_logits_single.bin")])
    run([sys.executable,
         str(PROJECT_ROOT / "scripts/save_ref_logits.py"),
         "--model", str(model_dir), "--prompt", "Hello world",
         "--output", str(model_dir / "ref_logits.bin")])


def build_tests(build_dir: Path) -> None:
    nproc = os.cpu_count() or 4
    run(["make", "-C", str(build_dir), "-j", str(nproc), "test_logits_match"])


def find_binary(build_dir: Path) -> Optional[Path]:
    for c in [build_dir / "tests/integration/test_logits_match",
              build_dir / "tests/test_logits_match"]:
        if c.exists():
            return c
    return None


def run_test(build_dir: Path, model_dir: Path) -> tuple[bool, str]:
    binary = find_binary(build_dir)
    if binary is None:
        return False, f"binary not found under {build_dir}"
    env = os.environ.copy()
    env["CCINFER_TEST_MODEL_DIR"] = str(model_dir.resolve())
    p = subprocess.run([str(binary)], cwd=PROJECT_ROOT, env=env,
                       capture_output=True, text=True)
    return p.returncode == 0, p.stdout + p.stderr


def report(args: argparse.Namespace, test_ok: bool, test_output: str) -> int:
    model_dir = Path(args.model).resolve()
    print()
    print("=" * 68)
    print("  ccInfer Correctness Report")
    print("=" * 68)
    print(f"  Model:    {model_dir}")
    print(f"  C++ test: {'PASSED' if test_ok else 'FAILED'}")
    print()

    for line in test_output.splitlines():
        if "max_diff" in line:
            print(f"  {line.strip()}")

    print()
    for prompt in args.prompts:
        safe = prompt.replace(" ", "_")[:32]
        ref_dir = model_dir / "ccinfer_correctness_ref" / safe
        mf = ref_dir / "metadata.json"
        if not mf.exists():
            continue
        meta = json.loads(mf.read_text())
        top5 = meta["top5_token_ids"]
        print(f"  {prompt:<22s}  tokens={meta['prompt_length']}"
              f"  top5={top5}")

    print()
    print("  Thresholds:  max_diff < 0.15 (single) / < 0.25 (multi)"
          "  top-5 >= 3/5")
    print("=" * 68)
    return 0 if test_ok else 1


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model).resolve()
    build_dir = Path(args.build_dir).resolve()

    if not args.skip_ref_gen:
        print("[1/3] Generating HF references ...")
        generate_refs(model_dir, args.prompts, args.max_new_tokens)
    else:
        print("[1/3] Skipping ref generation")

    if not args.skip_build:
        print("[2/3] Building C++ test ...")
        build_tests(build_dir)
    else:
        print("[2/3] Skipping build")

    print("[3/3] Running test ...")
    test_ok, output = run_test(build_dir, model_dir)
    return report(args, test_ok, output)


if __name__ == "__main__":
    sys.exit(main())
