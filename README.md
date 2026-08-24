# ccInfer

ccInfer is a C++23 LLM inference framework for CUDA GPUs. The current server
supports Qwen3-0.6B with paged KV caching, prefix caching, and streaming
responses.

## Dependencies

- CUDA Toolkit 11.8+
- GCC 13+ (C++23)
- CMake 3.20+
- Boost 1.83+
- nlohmann-json, fmt, spdlog

## Quick Start

### Build
```bash
cmake -S . -B build -DBUILD_SERVER=ON -DCMAKE_CUDA_ARCHITECTURES=89
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

Optional scheduling limits are available through `--max-running-requests`,
`--max-concurrent-batches`, `--max-pending-requests`, `--max-token-budget`,
and `--max-seq-prefill-tokens`.

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

## Source Layout

- `src/common/` — shared errors, results, request types, and channels
- `src/backend/`, `src/core/`, and `src/cache/` — device access, tensors, and KV caches
- `src/model/` and `src/tokenizer/` — model loading, execution, and tokenization
- `src/scheduler/`, `src/executor/`, `src/worker/`, and `src/engine/` — request scheduling and execution
- `src/http/` and `src/facade/` — HTTP service and external operator-library adapters
- `tests/` — unit and integration tests

## Features

- Paged KV cache and prefix-cache reuse
- GPT-2 BPE tokenizer (ByteLevel)
- SSE streaming responses
- GQA (Grouped Query Attention)
- BF16 model weights, FP32 logits
- Asynchronous batch dispatch with bounded in-flight work
- Graceful shutdown with two-phase drain
