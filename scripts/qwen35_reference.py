#!/usr/bin/env python3
"""Numpy Qwen3.5-2B GGUF Q8_0 reference forward.

This script is the Python correctness baseline for the M1 Qwen3.5 GGUF Q8_0
reference path.  It:

  * parses the GGUF with gguf_inventory.parse_gguf;
  * dequantizes Q8_0 to FP32 in numpy;
  * executes the hybrid decoder (18 Gated DeltaNet + 6 full-attention layers)
    in FP32;
  * supports one-shot prefill and chunked prefill through a persistent state
    object;
  * writes the last-token logits as raw FP32 binary for C++ integration tests.

The script intentionally has no CUDA/torch dependency.
"""

from __future__ import annotations

import argparse
import json
import mmap
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_inventory import parse_gguf

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
except ImportError:  # pragma: no cover - documented extra requirement
    HFTokenizer = None
    BPE = None
    ByteLevelDecoder = None
    ByteLevelPreTokenizer = None


def softplus(x: np.ndarray) -> np.ndarray:
    # Numerically stable softplus.
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(np.square(x.astype(np.float32)), axis=-1, keepdims=True)
    normalized = x.astype(np.float32) * np.reciprocal(np.sqrt(variance + eps))
    return normalized * weight.astype(np.float32)


def l2_norm(x: np.ndarray, eps: float) -> np.ndarray:
    inv_norm = np.reciprocal(np.sqrt(np.sum(np.square(x.astype(np.float32)), axis=-1, keepdims=True) + eps))
    return x.astype(np.float32) * inv_norm


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((-x2, x1), axis=-1)


class GgufTensorReader:
    """Reads individual GGUF tensors directly from the mmap'ed file."""

    def __init__(self, path: str | Path):
        self.info = parse_gguf(str(path))
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._tensors = {t["name"]: t for t in self.info["tensors"]}
        self.data_start = int(self.info["data_start"])

    def close(self) -> None:
        self._mm.close()
        self._file.close()

    def has(self, name: str) -> bool:
        return name in self._tensors

    def logical_shape(self, name: str) -> list[int]:
        info = self._tensors[name]
        return [int(d) for d in reversed(info["dims"])]

    def load(self, name: str) -> np.ndarray:
        if name not in self._tensors:
            raise KeyError(f"tensor not found: {name}")
        info = self._tensors[name]
        shape = [int(d) for d in reversed(info["dims"])]
        tensor_type = int(info["tensor_type"])
        offset = self.data_start + int(info["offset"])

        if tensor_type == 0:  # F32
            arr = np.ndarray(shape=shape, dtype="<f4", buffer=self._mm, offset=offset)
            return np.array(arr, dtype=np.float32)
        if tensor_type == 8:  # Q8_0
            if len(shape) != 2:
                raise ValueError(f"Q8_0 reference only supports 2D tensors, got {name}: {shape}")
            rows, cols = shape
            if cols % 32 != 0:
                raise ValueError(f"Q8_0 last logical dim must be block-aligned: {name}: {shape}")
            block_dtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
            n_blocks = cols // 32
            blocks = np.ndarray(shape=(rows, n_blocks), dtype=block_dtype, buffer=self._mm,
                                offset=offset)
            scales = blocks["d"].astype(np.float32)
            out = blocks["qs"].astype(np.float32)
            out *= scales[..., None]
            return out.reshape(rows, cols)
        raise ValueError(f"unsupported GGUF tensor type {tensor_type} for {name}")


class Qwen35Reference:
    """FP32 numpy implementation of the Qwen3.5 hybrid decoder."""

    def __init__(self, gguf_path: str | Path):
        self.reader = GgufTensorReader(gguf_path)
        md = self.reader.info["metadata"]
        self.metadata = md
        self.gguf_path = str(gguf_path)

        self.block_count = int(md["qwen35.block_count"])
        self.n_main_layers = int(md["qwen35.block_count"]) - int(md.get("qwen35.nextn_predict_layers", 0) or 0)
        self.d_model = int(md["qwen35.embedding_length"])
        self.d_ff = int(md["qwen35.feed_forward_length"])
        self.n_q_heads = int(md["qwen35.attention.head_count"])
        self.n_kv_heads = int(md["qwen35.attention.head_count_kv"])
        self.head_dim = int(md["qwen35.attention.key_length"])
        self.vocab_size = int(md.get("qwen35.vocab_size", self.reader.logical_shape("token_embd.weight")[0]))
        self.rotary_dim = int(md["qwen35.rope.dimension_count"])
        self.rope_theta = float(md["qwen35.rope.freq_base"])
        self.eps = float(md["qwen35.attention.layer_norm_rms_epsilon"])

        self.ssm_conv_kernel = int(md["qwen35.ssm.conv_kernel"])
        self.ssm_state_size = int(md["qwen35.ssm.state_size"])
        self.ssm_group_count = int(md["qwen35.ssm.group_count"])
        self.ssm_time_step_rank = int(md["qwen35.ssm.time_step_rank"])
        self.ssm_inner_size = int(md["qwen35.ssm.inner_size"])

        self.head_k_dim = self.ssm_state_size
        self.head_v_dim = self.ssm_inner_size // self.ssm_time_step_rank
        self.n_k_heads = self.ssm_group_count
        self.n_v_heads = self.ssm_time_step_rank
        self.key_dim = self.head_k_dim * self.n_k_heads
        self.value_dim = self.head_v_dim * self.n_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        if self.n_k_heads != self.n_v_heads:
            raise ValueError("Qwen35Reference only supports n_k_heads == n_v_heads for this artifact")

        interval = int(md["qwen35.full_attention_interval"])
        nextn = int(md.get("qwen35.nextn_predict_layers", 0) or 0)
        self.layer_types = []
        for i in range(self.n_main_layers):
            if interval > 0 and i % interval == interval - 1:
                self.layer_types.append("full_attention")
            else:
                self.layer_types.append("gdn")
        for _ in range(nextn):
            self.layer_types.append("mtp")

        self.gdn_layers = [i for i, t in enumerate(self.layer_types[: self.n_main_layers]) if t == "gdn"]
        self.attn_layers = [i for i, t in enumerate(self.layer_types[: self.n_main_layers]) if t == "full_attention"]

        # Token embedding is also the tied lm_head.
        self._embed: np.ndarray | None = None
        self.embed = self.load_embed()

    def close(self) -> None:
        self._embed = None
        self.reader.close()

    def load_embed(self) -> np.ndarray:
        if self._embed is None:
            self._embed = self.reader.load("token_embd.weight")
        return self._embed

    def _tensor_name(self, layer: int, kind: str) -> str:
        return f"blk.{layer}.{kind}"

    def load_gdn_weights(self, layer: int) -> dict[str, np.ndarray]:
        def load(kind: str) -> np.ndarray:
            return self.reader.load(self._tensor_name(layer, kind))

        w = {
            "attn_norm": load("attn_norm.weight"),
            "attn_qkv": load("attn_qkv.weight"),
            "attn_gate": load("attn_gate.weight"),
            "ssm_conv1d": load("ssm_conv1d.weight"),
            "ssm_a": load("ssm_a"),
            "ssm_dt_bias": load("ssm_dt.bias"),
            "ssm_norm": load("ssm_norm.weight"),
            "ssm_alpha": load("ssm_alpha.weight"),
            "ssm_beta": load("ssm_beta.weight"),
            "ssm_out": load("ssm_out.weight"),
            "ffn_gate": load("ffn_gate.weight"),
            "ffn_up": load("ffn_up.weight"),
            "ffn_down": load("ffn_down.weight"),
            "post_attn_norm": load("post_attention_norm.weight"),
        }
        return w

    def load_attn_weights(self, layer: int) -> dict[str, np.ndarray]:
        def load(kind: str) -> np.ndarray:
            return self.reader.load(self._tensor_name(layer, kind))

        qkv = load("attn_q.weight")  # [n_q_heads * 2 * head_dim, d_model]
        qkv = qkv.reshape(self.n_q_heads, 2, self.head_dim, self.d_model)
        w = {
            "attn_norm": load("attn_norm.weight"),
            "q": qkv[:, 0, :, :].reshape(self.n_q_heads * self.head_dim, self.d_model),
            "gate": qkv[:, 1, :, :].reshape(self.n_q_heads * self.head_dim, self.d_model),
            "k": load("attn_k.weight"),
            "v": load("attn_v.weight"),
            "q_norm": load("attn_q_norm.weight"),
            "k_norm": load("attn_k_norm.weight"),
            "o": load("attn_output.weight"),
            "ffn_gate": load("ffn_gate.weight"),
            "ffn_up": load("ffn_up.weight"),
            "ffn_down": load("ffn_down.weight"),
            "post_attn_norm": load("post_attention_norm.weight"),
        }
        return w

    def new_state(self) -> dict:
        conv_states = []
        recurrent_states = []
        for _ in self.gdn_layers:
            conv_states.append(np.zeros((self.conv_dim, self.ssm_conv_kernel - 1), dtype=np.float32))
            recurrent_states.append(np.zeros(
                (self.n_v_heads, self.head_k_dim, self.head_v_dim), dtype=np.float32))
        kv_cache = {}
        for layer in self.attn_layers:
            empty_k = np.zeros((0, self.n_kv_heads, self.head_dim), dtype=np.float32)
            empty_v = np.zeros((0, self.n_kv_heads, self.head_dim), dtype=np.float32)
            kv_cache[layer] = [empty_k, empty_v]
        return {
            "conv_states": conv_states,
            "recurrent_states": recurrent_states,
            "kv_cache": kv_cache,
            "processed_tokens": 0,
        }

    def _gdn_forward(self, hidden: np.ndarray, layer: int, state: dict) -> np.ndarray:
        gdn_index = self.gdn_layers.index(layer)
        w = self.load_gdn_weights(layer)
        T = hidden.shape[0]

        normed = rms_norm(hidden, w["attn_norm"], self.eps)
        qkv_mixed = normed @ w["attn_qkv"].T  # [T, key*2 + value]
        z = normed @ w["attn_gate"].T  # [T, value_dim]
        beta = sigmoid(normed @ w["ssm_beta"].T)  # [T, n_v]
        alpha = normed @ w["ssm_alpha"].T + w["ssm_dt_bias"]  # [T, n_v]
        # GGUF stores the already-negated per-head decay coefficient in ssm_a.
        decay = np.exp(softplus(alpha) * w["ssm_a"])  # [T, n_v]

        conv_state = state["conv_states"][gdn_index]  # [C, K-1], oldest -> newest
        conv_out = np.empty_like(qkv_mixed)
        for t in range(T):
            window = np.concatenate([conv_state, qkv_mixed[t][:, None]], axis=1)  # [C, K]
            conv_out[t] = silu(np.sum(w["ssm_conv1d"] * window, axis=1))
            conv_state = np.concatenate([conv_state[:, 1:], qkv_mixed[t][:, None]], axis=1)
        state["conv_states"][gdn_index] = conv_state

        q = conv_out[:, : self.key_dim].reshape(T, self.n_k_heads, self.head_k_dim)
        k = conv_out[:, self.key_dim : 2 * self.key_dim].reshape(T, self.n_k_heads, self.head_k_dim)
        v = conv_out[:, 2 * self.key_dim :].reshape(T, self.n_v_heads, self.head_v_dim)
        q = l2_norm(q, self.eps)
        k = l2_norm(k, self.eps)
        q = q * (self.head_k_dim ** -0.5)

        recurrent = state["recurrent_states"][gdn_index]  # [n_v, head_k, head_v]
        out = np.empty((T, self.n_v_heads, self.head_v_dim), dtype=np.float32)
        for t in range(T):
            decay_t = decay[t][:, None, None]  # [n_v, 1, 1]
            recurrent = recurrent * decay_t
            kv_mem = np.einsum("hkd,hk->hd", recurrent, k[t])
            delta = (v[t] - kv_mem) * beta[t][:, None]
            recurrent = recurrent + np.einsum("hk,hd->hkd", k[t], delta)
            out[t] = np.einsum("hkd,hk->hd", recurrent, q[t])
        state["recurrent_states"][gdn_index] = recurrent

        z = z.reshape(T, self.n_v_heads, self.head_v_dim)
        core = out.reshape(T, self.n_v_heads, self.head_v_dim)
        gated = rms_norm(core, w["ssm_norm"], self.eps) * silu(z)
        attn_out = gated.reshape(T, self.value_dim) @ w["ssm_out"].T
        del w
        return attn_out

    def _rope_embeddings(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dim = self.rotary_dim
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        emb = np.concatenate([angles, angles], axis=-1)  # [T, dim]
        return np.cos(emb), np.sin(emb)

    @staticmethod
    def _apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        # x: [T, heads, head_dim]; cos/sin: [T, rotary_dim] broadcast over heads.
        x_rot = x[..., : cos.shape[-1]]
        x_pass = x[..., cos.shape[-1] :]
        rotated = x_rot * cos[:, None, :] + rotate_half(x_rot) * sin[:, None, :]
        return np.concatenate([rotated, x_pass], axis=-1)

    def _attn_forward(self, hidden: np.ndarray, positions: np.ndarray, layer: int, state: dict) -> np.ndarray:
        attn_index = self.attn_layers.index(layer)
        w = self.load_attn_weights(layer)
        T = hidden.shape[0]
        n_rep = self.n_q_heads // self.n_kv_heads

        normed = rms_norm(hidden, w["attn_norm"], self.eps)
        qg = normed @ w["q"].T
        gate = normed @ w["gate"].T
        q = qg.reshape(T, self.n_q_heads, self.head_dim)
        k = (normed @ w["k"].T).reshape(T, self.n_kv_heads, self.head_dim)
        v = (normed @ w["v"].T).reshape(T, self.n_kv_heads, self.head_dim)

        q = rms_norm(q, w["q_norm"], self.eps)
        k = rms_norm(k, w["k_norm"], self.eps)
        cos, sin = self._rope_embeddings(positions)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        past_k, past_v = state["kv_cache"][layer]
        past_k = np.concatenate([past_k, k], axis=0) if past_k.shape[0] else k
        past_v = np.concatenate([past_v, v], axis=0) if past_v.shape[0] else v
        state["kv_cache"][layer] = [past_k, past_v]

        base = positions[0]
        total_len = past_k.shape[0]
        k_full = np.repeat(past_k, n_rep, axis=1)  # [total_len, n_q_heads, hd]
        v_full = np.repeat(past_v, n_rep, axis=1)
        scores = np.einsum("qhd,khd->hqk", q.astype(np.float32), k_full.astype(np.float32))
        scores *= self.head_dim ** -0.5
        row_global = base + np.arange(T)
        mask = np.zeros((T, total_len), dtype=np.float32)
        for r in range(T):
            mask[r, row_global[r] + 1 :] = -np.inf
        scores += mask[None, :, :]
        scores -= np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        attn_out = np.einsum("hqk,khd->qhd", attn_probs, v_full.astype(np.float32))

        attn_out = attn_out.reshape(T, self.n_q_heads * self.head_dim)
        attn_out = attn_out * sigmoid(gate)
        attn_out = attn_out @ w["o"].T
        del w
        return attn_out

    def forward_chunk(self, token_ids: list[int] | np.ndarray, state: dict) -> np.ndarray:
        """Process one prefill/decode chunk and return the last hidden vector."""
        tokens = np.asarray(token_ids, dtype=np.int64)
        T = tokens.shape[0]
        if T == 0:
            raise ValueError("empty chunk")
        hidden = self.embed[tokens].astype(np.float32)  # [T, d_model]
        positions = np.arange(state["processed_tokens"], state["processed_tokens"] + T, dtype=np.int64)

        for layer in range(self.n_main_layers):
            if self.layer_types[layer] == "gdn":
                out = self._gdn_forward(hidden, layer, state)
            elif self.layer_types[layer] == "full_attention":
                out = self._attn_forward(hidden, positions, layer, state)
            else:
                raise ValueError(f"unexpected MTP layer in main loop: {layer}")
            hidden = hidden + out

            residual = hidden
            if self.layer_types[layer] == "gdn":
                w = self.load_gdn_weights(layer)
            else:
                w = self.load_attn_weights(layer)
            normed = rms_norm(hidden, w["post_attn_norm"], self.eps)
            gate = normed @ w["ffn_gate"].T
            up = normed @ w["ffn_up"].T
            ffn_out = silu(gate) * up
            hidden = residual + (ffn_out @ w["ffn_down"].T)
            del w

        state["processed_tokens"] += T
        return hidden[-1].copy()

    def final_logits(self, last_hidden: np.ndarray) -> np.ndarray:
        return (self.embed @ last_hidden.astype(np.float32)).astype(np.float32)

    def run(self, token_ids: list[int], chunk_size: int | None = None) -> tuple[np.ndarray, dict]:
        state = self.new_state()
        if chunk_size is None or chunk_size <= 0 or chunk_size >= len(token_ids):
            last_hidden = self.forward_chunk(token_ids, state)
        else:
            last_hidden = None
            for start in range(0, len(token_ids), chunk_size):
                last_hidden = self.forward_chunk(token_ids[start : start + chunk_size], state)
        return self.final_logits(last_hidden), state


def load_tokenizer_from_gguf(gguf_path: str | Path) -> HFTokenizer:
    if HFTokenizer is None:
        raise RuntimeError("qwen35_reference.py requires the 'tokenizers' package")
    info = parse_gguf(str(gguf_path))
    tokens = info["metadata"]["tokenizer.ggml.tokens"]
    merges = info["metadata"]["tokenizer.ggml.merges"]
    if not isinstance(tokens, list) or not isinstance(merges, list):
        raise ValueError("GGUF does not contain a GPT-2/byte-level BPE tokenizer")
    vocab = {str(token): i for i, token in enumerate(tokens)}
    merge_list = []
    for merge in merges:
        if isinstance(merge, str):
            parts = merge.split()
        elif isinstance(merge, list) and len(merge) == 2:
            parts = [str(merge[0]), str(merge[1])]
        else:
            raise ValueError(f"unsupported GGUF merge entry: {merge!r}")
        if len(parts) != 2:
            raise ValueError(f"unsupported GGUF merge entry: {merge!r}")
        merge_list.append((parts[0], parts[1]))
    bpe = BPE(vocab=vocab,
              merges=merge_list,
              dropout=None,
              unk_token="<|endoftext|>")
    tokenizer = HFTokenizer(bpe)
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
        add_prefix_space=False, trim_offsets=True, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder(
        add_prefix_space=False, trim_offsets=True, use_regex=False)
    return tokenizer


def load_tokenizer(tokenizer_path: str | Path | None, gguf_path: Path) -> HFTokenizer:
    if tokenizer_path is not None:
        if HFTokenizer is None:
            raise RuntimeError("qwen35_reference.py requires the 'tokenizers' package")
        try:
            return HFTokenizer.from_file(str(tokenizer_path))
        except Exception as exc:  # fall back to the GGUF embedded tokenizer
            print(f"warning: could not load tokenizer file ({exc}); using GGUF metadata",
                  file=sys.stderr)
    return load_tokenizer_from_gguf(gguf_path)


def write_logits(path: Path, logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logits.astype(np.float32).tofile(path)


def write_metadata(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf", help="Path to Qwen3.5 GGUF file")
    p.add_argument("--prompt", default="Hello world")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer.json path")
    p.add_argument("--output-dir", default=None,
                   help="Output directory; default <gguf_parent>/ref")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="If set, process the prompt in chunks of this many tokens "
                        "(must be >=1) and check the last-token logits match the "
                        "one-shot result")
    p.add_argument("--save-state", action="store_true",
                   help="Write a metadata JSON describing the prompt and output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gguf_path = Path(args.gguf).resolve()
    if not gguf_path.exists():
        print(f"error: GGUF file not found: {gguf_path}", file=sys.stderr)
        return 1

    tokenizer = load_tokenizer(args.tokenizer, gguf_path)
    encoded = tokenizer.encode(args.prompt, add_special_tokens=False)
    token_ids = encoded.ids
    if not token_ids:
        print("error: prompt encodes to zero tokens", file=sys.stderr)
        return 1

    print(f"Loaded tokenizer: vocab={tokenizer.get_vocab_size()}")
    print(f"Prompt token ids ({len(token_ids)}): {token_ids}")

    model = Qwen35Reference(gguf_path)
    try:
        one_shot_logits, _ = model.run(token_ids, chunk_size=None)

        output_dir = Path(args.output_dir).resolve() if args.output_dir else gguf_path.parent / "ref"
        output_dir.mkdir(parents=True, exist_ok=True)
        logits_path = output_dir / "ref_logits_qwen35.bin"
        write_logits(logits_path, one_shot_logits)
        print(f"Wrote one-shot last-token logits to {logits_path}")

        chunked_logits = None
        if args.chunk_size is not None:
            if args.chunk_size <= 0:
                print("error: --chunk-size must be positive", file=sys.stderr)
                return 1
            chunked_logits, _ = model.run(token_ids, chunk_size=args.chunk_size)
            max_diff = float(np.max(np.abs(chunked_logits - one_shot_logits)))
            print(f"Chunked vs one-shot max_diff={max_diff:.6g}")
            if not np.allclose(chunked_logits, one_shot_logits, rtol=1e-4, atol=1e-4):
                print("error: chunked and one-shot logits differ", file=sys.stderr)
                return 1

        top5 = np.argsort(one_shot_logits)[-5:][::-1]
        print("Top-5 token ids:", top5.tolist())
        print("Top-5 logits:", one_shot_logits[top5].tolist())

        if args.save_state:
            meta = {
                "gguf": str(gguf_path),
                "prompt": args.prompt,
                "token_ids": token_ids,
                "logits_file": str(logits_path.relative_to(output_dir)),
                "top5_token_ids": top5.tolist(),
            }
            write_metadata(output_dir / "reference_meta.json", meta)
    finally:
        model.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
