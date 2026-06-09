#!/usr/bin/env python3
"""Generate HuggingFace reference artifacts for Qwen3 correctness checks."""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Qwen3 HF correctness references")
    parser.add_argument("--model", default="models/qwen3-0.6B", help="Model directory")
    parser.add_argument("--prompt", default="Hello world", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Greedy generation length")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model>/ccinfer_correctness_ref)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Reference device. Use cuda for the intended BF16 reference path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def save_npy(path: Path, array: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array.astype(np.float32))

    model_dir = Path(os.path.expanduser(args.model)).resolve()
    output_dir = (
        Path(os.path.expanduser(args.output_dir)).resolve()
        if args.output_dir
        else model_dir / "ccinfer_correctness_ref"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requested but torch.cuda.is_available() is false")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    token_ids = input_ids[0].detach().cpu().tolist()
    position_ids = list(range(len(token_ids)))

    forward_kwargs = {"input_ids": input_ids, "output_hidden_states": True}
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        forward_out = model(**forward_kwargs)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    logits_all = forward_out.logits[0].float().detach().cpu().numpy()
    logits_last = logits_all[-1]
    logits_all.astype(np.float32).tofile(output_dir / "logits_all_fp32.bin")
    logits_last.astype(np.float32).tofile(output_dir / "logits_last_fp32.bin")

    hidden_dir = output_dir / "hidden_states"
    hidden_state_files = []
    for idx, hidden in enumerate(forward_out.hidden_states):
        name = "embedding.npy" if idx == 0 else f"layer_{idx - 1}.npy"
        save_npy(hidden_dir / name, hidden[0].float().detach().cpu().numpy())
        hidden_state_files.append(str(Path("hidden_states") / name))

    generated_ids = generated[0].detach().cpu().tolist()
    generated_new_ids = generated_ids[len(token_ids) :]
    top5 = np.argsort(logits_last)[-5:][::-1]

    metadata = {
        "model": str(model_dir),
        "prompt": args.prompt,
        "input_token_ids": token_ids,
        "position_ids": position_ids,
        "prompt_length": len(token_ids),
        "reference_dtype": "torch.bfloat16",
        "logits_dtype": "float32",
        "logits_all_file": "logits_all_fp32.bin",
        "logits_last_file": "logits_last_fp32.bin",
        "hidden_state_files": hidden_state_files,
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "generated_token_ids": generated_ids,
            "generated_new_token_ids": generated_new_ids,
        },
        "top5_token_ids": top5.tolist(),
        "top5_logits": logits_last[top5].astype(np.float32).tolist(),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Saved HF reference artifacts to {output_dir}")
    print(f"input_token_ids={token_ids}")
    print(f"position_ids={position_ids}")
    print(f"generated_new_token_ids={generated_new_ids}")
    print(f"top5_token_ids={top5.tolist()}")


if __name__ == "__main__":
    main()
