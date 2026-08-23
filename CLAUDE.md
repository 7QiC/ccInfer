# ccInfer

High-performance C++23 LLM inference framework targeting CUDA GPUs, with PagedAttention KV cache, continuous batching, prefix caching, and an OpenAI-compatible HTTP server.

## Build

```bash
cmake -S . -B build -DBUILD_SERVER=ON -DCMAKE_CUDA_ARCHITECTURES=89
make -C build -j$(nproc)
ctest --test-dir build
```

Requires: CUDA Toolkit 11.8+, GCC 13+, CMake 3.20+, Boost 1.83+, nlohmann-json, fmt, spdlog.

## Python env

```bash
conda activate llm-infer
```
Activate the Python environment by default and execute tasks in this environment by default.

## Architecture

- `src/base/`  — Common types, `Result<T>`, error codes, channels, request types
- `src/backend/`  — GPU backend abstraction (`Backend`, `Buffer`)
- `src/core/`  — Framework `Tensor` and dtype/traits
- `src/cache/`  — Paged KV cache and prefix cache
- `src/executor/`, `src/worker/`, `src/scheduler/`  — Execution pipeline
- `src/model/`  — Model config, loader, registry, Qwen3, rope cache
- `src/facade/`  — Unified third-party facade (`ops.h` for ccop, `log.h` for logging)
- `src/http/`  — HTTP/SSE server
- `src/tokenizer/`  — Tokenizer
- `tests/unit/`  — Unit tests (GTest)
- `tests/integration/`  — Integration tests

## Principles

- **Never lower standards to pass tests.** If a test doesn't pass, find and fix the root cause — don't relax tolerances, weaken assertions, or bypass checks.
- **Never sacrifice performance for convenience.** No host-side fallbacks that defeat GPU parallelism, no unnecessary allocations or copies.
- **Keep code clean and consistent.** Match existing patterns, avoid clutter, delete dead code. This applies to all files including CMakeLists.txt.
- **High cohesion, low coupling.** Keep related functionality together within a module; modules communicate through abstract interfaces. Avoid circular dependencies and implicit coupling.

## Code Style

- **C++23** with `ccinfer` namespace
- **Naming:** PascalCase for classes/enums/structs, snake_case for functions/variables/files
- **Headers:** `.h` extension, `#pragma once`, sorted includes (C std → C++ std → third-party → project)
- **IWYU:** explicit includes, forward-declare in headers, no indirect relies
- **Members:** snake_case with trailing underscore, e.g. `int count_;`
- **Error handling:** `Result<T>` = `std::expected<T, ErrorCode>`, no exceptions in hot paths
- **CUDA:** Device memory via `Buffer` RAII
- **Tests:** GTest, test file per module
