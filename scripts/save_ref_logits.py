#!/usr/bin/env python3
"""Save reference logits from HuggingFace Transformers for ccInfer alignment test.

Usage:
    python scripts/save_ref_logits.py --model test_models/Llama-3.2-1B --prompt "Hello world"
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Save HF reference logits")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt", default="Hello world", help="Input prompt")
    parser.add_argument("--output", default=None, help="Output path (default: <model>/ref_logits.bin)")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    output_path = args.output or os.path.join(model_path, "ref_logits.bin")

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    print(f"Encoding prompt: '{args.prompt}'")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()
    print(f"Token IDs: {token_ids}")

    print("Running forward pass ...")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    print(f"Saving {len(logits)} logits to {output_path}")
    logits.astype(np.float32).tofile(output_path)

    top5 = np.argsort(logits)[-5:][::-1]
    print(f"Top-5 token IDs: {top5.tolist()}")
    print(f"Top-5 logits: {logits[top5].tolist()}")
    print("Done.")


if __name__ == "__main__":
    main()
