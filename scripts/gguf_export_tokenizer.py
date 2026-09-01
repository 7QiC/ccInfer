#!/usr/bin/env python3
"""Export a GGUF-embedded GPT-2 tokenizer to a HF-style tokenizer.json.

This is a debugging/compatibility helper; the runtime obtains the tokenizer
directly from GGUF metadata. The exported file is tested for vocab/merges
equivalence with the GGUF metadata path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gguf_inventory import parse_gguf


def build_tokenizer_json(info: dict) -> dict:
    metadata = info["metadata"]

    tokens = metadata.get("tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError("GGUF does not contain tokenizer.ggml.tokens")
    merges = metadata.get("tokenizer.ggml.merges")
    if merges is None:
        raise ValueError("GGUF does not contain tokenizer.ggml.merges")

    token_types = metadata.get("tokenizer.ggml.token_type")
    if token_types is None:
        token_types = [0] * len(tokens)
    if len(token_types) != len(tokens):
        raise ValueError("tokenizer.ggml.token_type length does not match tokens")

    vocab = {}
    for i, token in enumerate(tokens):
        if not isinstance(token, str):
            raise ValueError("tokenizer.ggml.tokens must be strings")
        vocab[token] = i

    merge_list = []
    for merge in merges:
        if isinstance(merge, str):
            merge_list.append(merge)
        elif isinstance(merge, list) and len(merge) == 2:
            merge_list.append(f"{merge[0]} {merge[1]}")
        else:
            raise ValueError("tokenizer.ggml.merges has unsupported entry")

    added_tokens = []
    # GGUF token types: 0 undefined, 1 normal, 2 unknown, 3 control,
    # 4 user-defined, 5 unused, 6 byte. Mark control/user-defined as special.
    for token, token_type in zip(tokens, token_types):
        if isinstance(token_type, int) and token_type >= 3:
            added_tokens.append({
                "id": vocab[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            })

    eos = metadata.get("tokenizer.ggml.eos_token_id")
    pad = metadata.get("tokenizer.ggml.pad_token_id")
    if pad is None:
        pad = metadata.get("tokenizer.ggml.padding_token_id")
    bos = metadata.get("tokenizer.ggml.bos_token_id")
    add_bos = metadata.get("tokenizer.ggml.add_bos_token")

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": None}],
            "pair": [{"SpecialToken": None}, {"SpecialToken": None}],
            "special_tokens": {},
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merge_list,
        },
        "chat_template": None,
    }

    if bos is not None and 0 <= int(bos) < len(tokens):
        tokenizer_json["bos_token"] = tokens[int(bos)]
    if eos is not None and 0 <= int(eos) < len(tokens):
        tokenizer_json["eos_token"] = tokens[int(eos)]
    if pad is not None and 0 <= int(pad) < len(tokens):
        tokenizer_json["pad_token"] = tokens[int(pad)]
    # The current Qwen3.5 artifact has add_bos_token=false; the exported JSON is
    # used only for vocab/merges equivalence and must not invent a BOS.
    if add_bos is not None:
        tokenizer_json["add_bos_token"] = bool(add_bos)

    return tokenizer_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--output", type=Path, default="models/qwen3_5-2B/tokenizer.json",
                        help="Output tokenizer.json path")
    args = parser.parse_args()

    info = parse_gguf(str(args.gguf_path))
    tokenizer_json = build_tokenizer_json(info)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    vocab = tokenizer_json["model"]["vocab"]
    merges = tokenizer_json["model"]["merges"]
    print(f"Wrote {args.output}: vocab={len(vocab)}, merges={len(merges)}, "
          f"added_tokens={len(tokenizer_json['added_tokens'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
