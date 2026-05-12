# ccInfer

High-performance C++23 LLM inference framework.

## Dependencies

- CUDA Toolkit 11.8+
- GCC 13+ (C++23)
- CMake 3.20+
- Boost 1.83+
- nlohmann-json, fmt, spdlog

## Quick Start

### Build
```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
make -C build -j$(nproc)
```

### Download model (Qwen3-0.6B, ~1.5GB)
#### Option A: Python
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/qwen3-0.6B
```

#### Option B: Git LFS
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-0.6B models/qwen3-0.6B
```

### Run server
```bash
./build/src/ccinfer-server --port 8080 --model-path ./models/qwen3-0.6B
```

### Test
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
```
### Run tests
```bash
ctest --test-dir build
```

## Features

- PagedAttention KV cache with online softmax
- GPT-2 BPE tokenizer (ByteLevel)
- SSE streaming responses
- GQA (Grouped Query Attention)
- BF16 model weights, FP32 logits
- Graceful shutdown with two-phase drain