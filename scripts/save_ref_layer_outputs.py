#!/usr/bin/env python3
"""Save per-layer hidden states from HF for layer-by-layer comparison."""
import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello world")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # Store hidden states after each layer
    hidden_states = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            # output[0] is the hidden state after this layer
            hidden_states[layer_idx] = output[0].detach().float().cpu().numpy()
        return hook

    # Register hooks on each decoder layer's output
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook_fn(i))

    # Also hook the final norm output (before lm_head)
    def final_norm_hook(module, input, output):
        hidden_states["final_norm"] = output.detach().float().cpu().numpy()
    model.model.norm.register_forward_hook(final_norm_hook)

    print(f"Running forward pass for '{args.prompt}'...")
    with torch.no_grad():
        model(**inputs)

    out_dir = os.path.join(args.model, "layer_outputs")
    os.makedirs(out_dir, exist_ok=True)
    for key, arr in hidden_states.items():
        path = os.path.join(out_dir, f"layer_{key}.npy")
        np.save(path, arr.astype(np.float32))
        print(f"  layer_{key}: shape={arr.shape} saved")

    print(f"All layer outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
