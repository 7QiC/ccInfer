# ccInfer

High-performance C++23 LLM inference framework targeting CUDA GPUs, with PagedAttention KV cache, continuous batching, prefix caching, and an OpenAI-compatible HTTP server.

## Build

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
make -C build -j$(nproc)
ctest --test-dir build
```

Requires: CUDA Toolkit 11.8+, GCC 13+, CMake 3.20+, Boost 1.83+, nlohmann-json, fmt, spdlog.

## Architecture

- `src/common/`  — Project-level utilities
- `src/engine/`  — Inference engine
- `src/server/`  — HTTP server, scheduler
- `tests/unit/`  — Unit tests (GTest)
- `tests/kernel` — Kernel tests

## Principles

- **Never lower standards to pass tests.** If a test doesn't pass, find and fix the root cause — don't relax tolerances, weaken assertions, or bypass checks.
- **Never sacrifice performance for convenience.** No host-side fallbacks that defeat GPU parallelism, no unnecessary allocations or copies.
- **Keep code clean and consistent.** Match existing patterns, avoid clutter, delete dead code. This applies to all files including CMakeLists.txt.

## Code Style

- **C++23** with `ccinfer` namespace
- **Naming:** PascalCase for classes/enums/structs, snake_case for functions/variables/files
- **Headers:** `.h` extension, `#pragma once`, sorted includes (C std → C++ std → third-party → project)
- **IWYU:** explicit includes, forward-declare in headers, no indirect relies
- **Members:** snake_case with trailing underscore, e.g. `int count_;`
- **Error handling:** `Result<T>` = `std::expected<T, ErrorCode>`, no exceptions in hot paths
- **CUDA:** Device memory via `DeviceBuffer<T>` RAII
- **Tests:** GTest, test file per module
