#!/usr/bin/env python3
"""GGUF inventory: parse header/metadata/tensor infos and print a concise summary."""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

GGUF_VALUE_TYPES = {
    0: "u8",
    1: "i8",
    2: "u16",
    3: "i16",
    4: "u32",
    5: "i32",
    6: "f32",
    7: "bool",
    8: "string",
    9: "array",
    10: "u64",
    11: "i64",
    12: "f64",
}

GGUF_TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
}

_STRUCT_FORMATS = {
    0: ("u8", "<B"),
    1: ("i8", "<b"),
    2: ("u16", "<H"),
    3: ("i16", "<h"),
    4: ("u32", "<I"),
    5: ("i32", "<i"),
    6: ("f32", "<f"),
    7: ("bool", "<B"),
    10: ("u64", "<Q"),
    11: ("i64", "<q"),
    12: ("f64", "<d"),
}


def parse_generic_value(data: memoryview, offset: int, value_type: int):
    if value_type == 8:  # string
        (length,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        raw = bytes(data[offset : offset + length])
        if len(raw) != length:
            raise ValueError("string length exceeds file size")
        try:
            return raw.decode("utf-8"), offset + length
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace"), offset + length

    if value_type == 9:  # array
        (elem_type,) = struct.unpack_from("<I", data, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        values = []
        for _ in range(count):
            value, offset = parse_generic_value(data, offset, elem_type)
            values.append(value)
        return values, offset

    name, fmt = _STRUCT_FORMATS.get(value_type)
    if name is None:
        raise ValueError(f"unsupported metadata value type {value_type}")
    (value,) = struct.unpack_from(fmt, data, offset)
    return value, offset + struct.calcsize(fmt)


def read_string(data: memoryview, offset: int):
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    raw = bytes(data[offset : offset + length])
    if len(raw) != length:
        raise ValueError("string length exceeds file size")
    return raw.decode("utf-8", errors="replace"), offset + length


def parse_gguf(path: str):
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(data)

    if bytes(mv[0:4]) != GGUF_MAGIC:
        raise ValueError("bad magic: expected GGUF")

    version, n_tensors, n_metadata = struct.unpack_from("<IQQ", mv, 4)
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version}")

    offset = 4 + 4 + 8 + 8  # magic + version + n_tensors + n_metadata

    metadata = {}
    for _ in range(n_metadata):
        key, offset = read_string(mv, offset)
        (value_type,) = struct.unpack_from("<I", mv, offset)
        offset += 4
        value, offset = parse_generic_value(mv, offset, value_type)
        metadata[key] = value

    tensors = []
    for _ in range(n_tensors):
        name, offset = read_string(mv, offset)
        (n_dims,) = struct.unpack_from("<I", mv, offset)
        offset += 4
        dims = list(struct.unpack_from(f"<{'Q' * n_dims}", mv, offset))
        offset += 8 * n_dims
        (tensor_type,) = struct.unpack_from("<I", mv, offset)
        offset += 4
        (tensor_offset,) = struct.unpack_from("<Q", mv, offset)
        offset += 8
        tensors.append(
            {
                "name": name,
                "n_dims": n_dims,
                "dims": dims,
                "tensor_type": tensor_type,
                "tensor_type_name": GGUF_TENSOR_TYPES.get(tensor_type, f"UNKNOWN_{tensor_type}"),
                "offset": tensor_offset,
            }
        )

    if offset > len(data):
        raise ValueError("tensor table extends past end of file")

    return {
        "path": str(path),
        "version": version,
        "n_tensors": n_tensors,
        "n_metadata": n_metadata,
        "metadata": metadata,
        "tensors": tensors,
        "file_size": len(data),
    }


def summarize(info: dict):

    def meta(key, default=None):
        return info["metadata"].get(key, default)

    tensor_type_counts: dict[str, int] = {}
    for t in info["tensors"]:
        name = t["tensor_type_name"]
        tensor_type_counts[name] = tensor_type_counts.get(name, 0) + 1

    names = [t["name"] for t in info["tensors"]]
    has_nextn_tensors = any(".nextn." in n or "nextn_" in n for n in names)
    nextn_predict_layers = meta("qwen35.nextn_predict_layers")
    if nextn_predict_layers is None:
        nextn_predict_layers = meta("nextn_predict_layers")

    # Classify main-model blocks.
    block_count = meta("qwen35.block_count") or meta("block_count") or 0
    full_attention_interval = meta("qwen35.full_attention_interval")
    if full_attention_interval is None:
        full_attention_interval = meta("full_attention_interval")

    blocks_with_ssm = 0
    blocks_with_attn = 0
    mtp_blocks = 0
    for name in names:
        if not name.startswith("blk."):
            continue
        parts = name.split(".", 1)
        if len(parts) < 2:
            continue
        try:
            block_id = int(parts[1].split(".", 1)[0])
        except ValueError:
            continue
        if block_id >= block_count:
            mtp_blocks += 1
        elif "ssm_" in name:
            blocks_with_ssm = max(blocks_with_ssm, block_id + 1)
        elif "attn_" in name:
            blocks_with_attn = max(blocks_with_attn, block_id + 1)

    # The above block counters are not used for classification; layer types are
    # derived from the metadata + tensor-name pattern below.
    layer_types = []
    if block_count and isinstance(block_count, int):
        mtp_count = int(nextn_predict_layers or 0)
        interval = int(full_attention_interval or 0) if full_attention_interval is not None else 0
        for i in range(int(block_count)):
            if mtp_count > 0 and i >= int(block_count) - mtp_count:
                layer_types.append("MtpPredictor")
            elif interval > 0 and i % interval == interval - 1:
                layer_types.append("FullAttention")
            else:
                layer_types.append("GatedDeltaNet")
    else:
        layer_types = []

    tokenizer_embedded = "tokenizer.ggml.tokens" in info["metadata"]

    summary = {
        "path": info["path"],
        "gguf_version": info["version"],
        "file_size": info["file_size"],
        "tensor_count": info["n_tensors"],
        "metadata_count": info["n_metadata"],
        "tensor_type_counts": tensor_type_counts,
        "general_architecture": meta("general.architecture"),
        "block_count": block_count,
        "context_length": meta("qwen35.context_length") or meta("context_length"),
        "embedding_length": meta("qwen35.embedding_length") or meta("embedding_length"),
        "feed_forward_length": meta("qwen35.feed_forward_length") or meta("feed_forward_length"),
        "attention_head_count": meta("qwen35.attention.head_count") or meta("attention.head_count"),
        "attention_head_count_kv": meta("qwen35.attention.head_count_kv")
        or meta("attention.head_count_kv"),
        "attention_key_length": meta("qwen35.attention.key_length")
        or meta("attention.key_length"),
        "attention_value_length": meta("qwen35.attention.value_length")
        or meta("attention.value_length"),
        "rope_dimension_sections": meta("qwen35.rope.dimension_sections")
        or meta("rope.dimension_sections"),
        "rope_dimension_count": meta("qwen35.rope.dimension_count")
        or meta("rope.dimension_count"),
        "rope_freq_base": meta("qwen35.rope.freq_base") or meta("rope.freq_base"),
        "ssm_conv_kernel": meta("qwen35.ssm.conv_kernel") or meta("ssm.conv_kernel"),
        "ssm_state_size": meta("qwen35.ssm.state_size") or meta("ssm.state_size"),
        "ssm_group_count": meta("qwen35.ssm.group_count") or meta("ssm.group_count"),
        "ssm_time_step_rank": meta("qwen35.ssm.time_step_rank")
        or meta("ssm.time_step_rank"),
        "ssm_inner_size": meta("qwen35.ssm.inner_size") or meta("ssm.inner_size"),
        "full_attention_interval": full_attention_interval,
        "nextn_predict_layers": nextn_predict_layers,
        "has_nextn_tensors": has_nextn_tensors,
        "tokenizer_embedded": tokenizer_embedded,
        "tokenizer_model": meta("tokenizer.ggml.model"),
        "tokenizer_tokens_count": len(meta("tokenizer.ggml.tokens") or []),
        "tokenizer_merges_count": len(meta("tokenizer.ggml.merges") or []),
        "tokenizer_eos": meta("tokenizer.ggml.eos_token_id"),
        "tokenizer_pad": (
            meta("tokenizer.ggml.pad_token_id")
            if meta("tokenizer.ggml.pad_token_id") is not None
            else meta("tokenizer.ggml.padding_token_id")
        ),
        "tokenizer_add_bos": meta("tokenizer.ggml.add_bos_token"),
        "has_output_weight": any(n == "output.weight" for n in names),
        "has_token_embd_weight": any(n == "token_embd.weight" for n in names),
        "layer_types": layer_types,
        "main_layers": sum(1 for x in layer_types if x in ("FullAttention", "GatedDeltaNet")),
        "gated_delta_net_layers": sum(1 for x in layer_types if x == "GatedDeltaNet"),
        "full_attention_layers": sum(1 for x in layer_types if x == "FullAttention"),
        "mtp_layers": sum(1 for x in layer_types if x == "MtpPredictor"),
        "sample_tensors": names[:8],
    }
    return summary


def render_markdown(summary: dict) -> str:
    lines = [
        "# GGUF Inventory",
        "",
        f"- 文件: `{summary['path']}`",
        f"- GGUF 版本: {summary['gguf_version']}",
        f"- 文件大小: {summary['file_size']:,} bytes",
        f"- 张量数: {summary['tensor_count']}",
        f"- metadata 条数: {summary['metadata_count']}",
        f"- general.architecture: `{summary['general_architecture']}`",
        f"- block_count: {summary['block_count']}",
        f"- context_length: {summary['context_length']}",
        f"- embedding_length: {summary['embedding_length']}",
        f"- feed_forward_length: {summary['feed_forward_length']}",
        f"- attention.head_count / head_count_kv: {summary['attention_head_count']} / {summary['attention_head_count_kv']}",
        f"- attention.key_length / value_length: {summary['attention_key_length']} / {summary['attention_value_length']}",
        f"- rope.dimension_sections: {summary['rope_dimension_sections']}",
        f"- rope.dimension_count / freq_base: {summary['rope_dimension_count']} / {summary['rope_freq_base']}",
        f"- ssm.conv_kernel / state_size: {summary['ssm_conv_kernel']} / {summary['ssm_state_size']}",
        f"- ssm.group_count / time_step_rank / inner_size: {summary['ssm_group_count']} / {summary['ssm_time_step_rank']} / {summary['ssm_inner_size']}",
        f"- full_attention_interval: {summary['full_attention_interval']}",
        f"- nextn_predict_layers (MTP): {summary['nextn_predict_layers']}",
        f"- MTP tensor 检测: {summary['has_nextn_tensors']}",
        f"- tokenizer 内嵌: {summary['tokenizer_embedded']}",
        f"- tokenizer.model: {summary['tokenizer_model']}",
        f"- tokenizer tokens / merges: {summary['tokenizer_tokens_count']} / {summary['tokenizer_merges_count']}",
        f"- tokenizer eos / pad: {summary['tokenizer_eos']} / {summary['tokenizer_pad']}",
        f"- tokenizer add_bos: {summary['tokenizer_add_bos']}",
        f"- 权重绑定 (无 output.weight): {not summary['has_output_weight']}",
        f"- 主模型层: GDN {summary['gated_delta_net_layers']} + FullAttention {summary['full_attention_layers']} + MTP {summary['mtp_layers']}",
        "",
        "## Tensor type 分布",
        "",
    ]
    for name, count in sorted(summary["tensor_type_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## 层类型", ""])
    for idx, layer in enumerate(summary["layer_types"]):
        lines.append(f"- blk.{idx}: {layer}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--json", type=Path, default=None, help="Write JSON summary to this path")
    parser.add_argument("--markdown", type=Path, default=None, help="Write markdown summary to this path")
    args = parser.parse_args()

    try:
        info = parse_gguf(args.gguf_path)
    except Exception as exc:  # noqa: BLE001 - user-facing inventory tool
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = summarize(info)
    json_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.json:
        args.json.write_text(json_text, encoding="utf-8")
    else:
        print(json_text, end="")

    if args.markdown:
        args.markdown.write_text(render_markdown(summary), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
