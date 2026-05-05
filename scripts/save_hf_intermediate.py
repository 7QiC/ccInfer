#!/usr/bin/env python3
"""Save detailed intermediate states from HF for the first layer."""
import argparse, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    l0 = model.model.layers[0]
    l0_attn = l0.self_attn
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = l0_attn.head_dim
    B, T = input_ids.shape

    results = {}

    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        results["embed"] = hidden_states.detach().float().cpu().numpy()

        # --- Attention block ---
        residual = hidden_states
        normed = l0.input_layernorm(hidden_states)
        results["l0_attn_norm"] = normed.detach().float().cpu().numpy()

        # QKV projections
        q = l0_attn.q_proj(normed)
        k = l0_attn.k_proj(normed)
        v = l0_attn.v_proj(normed)
        results["l0_q"] = q.detach().float().cpu().numpy()
        results["l0_k"] = k.detach().float().cpu().numpy()
        results["l0_v"] = v.detach().float().cpu().numpy()

        # QK norm: reshape to [B, T, n_heads, head_dim], RMSNorm normalizes last dim
        q_4d = q.reshape(B, T, n_heads, head_dim)
        k_4d = k.reshape(B, T, n_kv_heads, head_dim)
        q_normed = l0_attn.q_norm(q_4d).reshape(T, -1)
        k_normed = l0_attn.k_norm(k_4d).reshape(T, -1)
        results["l0_q_normed"] = q_normed.detach().float().cpu().numpy()
        results["l0_k_normed"] = k_normed.detach().float().cpu().numpy()

        # Full self_attn (includes RoPE, attention, O-proj inside)
        cos_sin = l0_attn.rotary_emb.get_cos_sin(input_ids)
        attn_out = l0_attn(normed, position_embeddings=cos_sin)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        results["l0_attn_out"] = attn_out.detach().float().cpu().numpy()

        hidden_states = residual + attn_out
        results["l0_attn_residual"] = hidden_states.detach().float().cpu().numpy()

        # --- FFN block ---
        residual = hidden_states
        normed_ffn = l0.post_attention_layernorm(hidden_states)
        results["l0_ffn_norm"] = normed_ffn.detach().float().cpu().numpy()

        gate = l0.mlp.gate_proj(normed_ffn)
        up = l0.mlp.up_proj(normed_ffn)
        results["l0_gate"] = gate.detach().float().cpu().numpy()
        results["l0_up"] = up.detach().float().cpu().numpy()

        silu_out = torch.nn.functional.silu(gate) * up
        results["l0_silu_mul"] = silu_out.detach().float().cpu().numpy()

        down = l0.mlp.down_proj(silu_out)
        results["l0_down"] = down.detach().float().cpu().numpy()

        hidden_states = residual + down
        results["l0_output"] = hidden_states.detach().float().cpu().numpy()

    out_dir = os.path.join(args.model, "hf_intermediate")
    os.makedirs(out_dir, exist_ok=True)
    for key, arr in results.items():
        path = os.path.join(out_dir, f"{key}.npy")
        np.save(path, arr.astype(np.float32))
        print(f"  {key}: shape={arr.shape} saved")

    print(f"All intermediates saved to {out_dir}")


if __name__ == "__main__":
    main()
