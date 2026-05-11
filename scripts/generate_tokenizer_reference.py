#!/usr/bin/env python3
"""
Generate HF tokenizer reference data for ccInfer tokenizer alignment testing.

Usage:
  python scripts/generate_tokenizer_reference.py \
      --tokenizer /path/to/tokenizer.json \
      --output tests/unit/tokenizer_alignment_reference.json

The generated JSON is consumed by the C++ GTest test_tokenizer_alignment.
"""

import argparse
import json
import sys

try:
    from tokenizers import Tokenizer as HFTokenizer
except ImportError:
    print("ERROR: pip install tokenizers", file=sys.stderr)
    sys.exit(1)


TEST_PROMPTS = [
    "",
    "Hello",
    "Hello world",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    " Hello",
    "Hello ",
    "  Hello  world  ",
    "Hello\nworld",
    "Hello\n\nworld",
    "Hello\tworld",
    "Line 1\nLine 2\nLine 3",
    "a+b=c",
    "price: $12.50",
    "email@example.com",
    "https://example.com/path?a=1&b=2",
    "foo_bar-baz.qux",
    "1234567890",
    "3.1415926535",
    "-42 + 17 = -25",
    "2026-05-11 14:30:00",
    "你好",
    "你好，世界！",
    "我正在做大模型推理引擎。",
    "这是一个中文 tokenizer 对齐测试。",
    "南京大学建筑与城市规划学院",
    "Hello 世界",
    "Qwen3 模型推理 test",
    "GPU 上的 KV cache 管理",
    "我想找 AI Infra 的工作。",
    "😀",
    "Hello 😀 world",
    "CUDA 🚀 inference",
    "中文🙂英文🚀混合",
    "int main() { return 0; }",
    "std::vector<int32_t> tokens;",
    "if (!result) return std::unexpected(result.error());",
    "template <typename T>\nResult<void> foo(T x) {\n    return {};\n}",
    '{"name":"ccInfer","version":1,"ok":true}',
    "- item 1\n- item 2\n- item 3",
    "```cpp\nint x = 42;\n```",
    "# Title\n\nThis is a paragraph.",
    "<|endoftext|>",
    "<|im_start|>user\nHello<|im_end|>",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>",
    "<s>Hello</s>",
]


def main():
    parser = argparse.ArgumentParser(description="Generate HF tokenizer reference data")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--output", default="tests/unit/tokenizer_alignment_reference.json",
                        help="Output path")
    args = parser.parse_args()

    print(f"Loading: {args.tokenizer}")
    hf = HFTokenizer.from_file(args.tokenizer)

    reference = {
        "tokenizer_path": args.tokenizer,
        "vocab_size": hf.get_vocab_size(),
        "cases": [],
    }

    for prompt in TEST_PROMPTS:
        ids_ns = hf.encode(prompt, add_special_tokens=False).ids
        ids_s = hf.encode(prompt, add_special_tokens=True).ids
        dec_ns = hf.decode(ids_ns, skip_special_tokens=True)
        dec_s = hf.decode(ids_s, skip_special_tokens=False)

        reference["cases"].append({
            "prompt": prompt,
            "encode_no_special": ids_ns,
            "encode_special": ids_s,
            "decode_no_special": dec_ns,
            "decode_special": dec_s,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(TEST_PROMPTS)} cases to {args.output}")
    print("Next: rebuild and run test_tokenizer_alignment")


if __name__ == "__main__":
    main()
