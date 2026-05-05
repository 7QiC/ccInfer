#!/usr/bin/env python3
"""Compare ccInfer layer outputs with HF reference."""
import argparse
import numpy as np
import os


def load_raw_bin(path, shape):
    """Load a raw float32 binary file."""
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def load_npy(path):
    """Load a numpy .npy file."""
    return np.load(path).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt", default="Hello", help="Prompt used")
    parser.add_argument("--start-layer", type=int, default=0,
                        help="Compare starting from this layer (0-indexed)")
    parser.add_argument("--max-layers", type=int, default=5,
                        help="Only show first N layers in detail")
    args = parser.parse_args()

    our_dir = os.path.join(args.model, "our_layer_outputs")
    hf_dir = os.path.join(args.model, "layer_outputs")

    # Determine T and D from the embedding file
    our_embed = load_raw_bin(os.path.join(our_dir, "embedding.bin"), (-1,))
    T = len(our_embed)
    # Get D from config
    import json
    with open(os.path.join(args.model, "config.json")) as f:
        cfg = json.load(f)
    D = cfg["hidden_size"]
    T = len(our_embed) // D  # Should be 1 for single token
    print(f"Config: D={D}, T={T}")

    # Compare layer outputs
    n_layers = cfg["num_hidden_layers"]
    print(f"\n{'Layer':<10} {'Max Diff':<14} {'Mean Diff':<14} {'RMSE':<14} {'First Bad Idx':<16} {'Our val':<14} {'HF val':<14}")
    print("-" * 110)

    max_per_layer = []
    for l in range(n_layers):
        our_path = os.path.join(our_dir, f"layer_{l}.bin")
        hf_path = os.path.join(hf_dir, f"layer_{l}.npy")

        our = load_raw_bin(our_path, (T, D))
        hf = load_npy(hf_path)
        hf = hf.reshape(T, D)  # handle 3D arrays like (1, 1, 1024)

        assert our.shape == hf.shape, f"Shape mismatch at layer {l}: {our.shape} vs {hf.shape}"

        diff = np.abs(our - hf)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        max_idx = np.argmax(diff)
        max_per_layer.append((l, max_diff, max_idx, our.flat[max_idx], hf.flat[max_idx]))

        if l < args.max_layers or l >= n_layers - 3:
            marker = " <--" if max_diff > 0.001 else ""
            print(f"layer_{l:<3}  {max_diff:<14.6f} {mean_diff:<14.6f} {rmse:<14.6f} "
                  f"{max_idx:<16} {our.flat[max_idx]:<14.6f} {hf.flat[max_idx]:<14.6f}{marker}")

    # Summary: find first layer with significant divergence
    print(f"\n--- Per-layer max diff summary ---")
    threshold = 0.001
    first_bad = None
    for l, max_diff, max_idx, our_val, hf_val in max_per_layer:
        bar = "#" * int(max_diff * 200) if max_diff < 0.5 else "#" * 100
        marker = ""
        if max_diff > threshold and first_bad is None:
            first_bad = l
            marker = " <-- FIRST DIVERGENCE"
        if l < args.max_layers or l >= n_layers - 3 or (first_bad is not None and abs(l - first_bad) <= 2):
            print(f"  layer_{l:>3}: max_diff={max_diff:.6f} {bar}{marker}")

    if first_bad is not None:
        print(f"\n*** First significant divergence (> {threshold}) at layer {first_bad} ***")
    else:
        print(f"\nNo layer exceeds threshold {threshold}")

    # Compare final norm
    print(f"\n--- Final norm ---")
    our_final = load_raw_bin(os.path.join(our_dir, "final_norm.bin"), (T, D))
    hf_final = load_npy(os.path.join(hf_dir, "final_norm.npy"))
    hf_final = hf_final.reshape(T, D)  # handle 3D arrays
    diff = np.abs(our_final - hf_final)
    max_idx = np.argmax(diff)
    print(f"  max_diff={np.max(diff):.6f} at idx {max_idx} "
          f"(our={our_final.flat[max_idx]:.6f} hf={hf_final.flat[max_idx]:.6f})")
    print(f"  mean_diff={np.mean(diff):.6f} rmse={np.sqrt(np.mean(diff**2)):.6f}")

    # Also show sampling result comparison
    print(f"\n--- Logits comparison ---")
    # Load our logits from the existing test
    our_logits_path = os.path.join(our_dir, "logits.bin")
    if os.path.exists(our_logits_path):
        our_logits = load_raw_bin(our_logits_path, (-1,))
        ref_path = os.path.join(args.model, "ref_logits_single.bin")
        if os.path.exists(ref_path):
            ref_logits = np.fromfile(ref_path, dtype=np.float32)
            diff = np.abs(our_logits - ref_logits)
            max_idx = np.argmax(diff)
            print(f"  max_diff={np.max(diff):.6f} at token {max_idx} "
                  f"(our={our_logits[max_idx]:.6f} hf={ref_logits[max_idx]:.6f})")


if __name__ == "__main__":
    main()
