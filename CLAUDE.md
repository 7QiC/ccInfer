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

- `src/common/` — Project-level utilities (DType, ErrorCode, Result<T>, hash)
- `src/engine/core/` — GPU core types (DeviceBuffer<T>, Tensor views)
- `src/engine/backend/` — DeviceBackend interface + CUDA implementation
- `src/engine/model/` — Model config loading
- `src/server/` — HTTP server, scheduler, KV cache (Phase 4+)
- `tests/unit/` — Unit tests (GTest)

## Code Style

- **C++23** with `ccinfer` namespace
- **Naming:** PascalCase for classes/enums/structs, snake_case for functions/variables/files
- **Headers:** `.h` extension, `#pragma once`, sorted includes (C std → C++ std → third-party → project)
- **Members:** no `m_` prefix; accessors `get()` return raw pointers, no getter prefix
- **Error handling:** `Result<T>` = `std::expected<T, ErrorCode>`, no exceptions in hot paths
- **CUDA:** Device memory via `DeviceBuffer<T>` RAII, macros `CCINFER_CUDA_CHECK` / `CCINFER_CUBLAS_CHECK`
- **Tests:** GTest, test file per module, `TYPED_TEST` for dtype-generic tests
- **Format:** `.clang-format` (Google-based, 100col, 4-space indent), use clang-format before commit

## Design Spec

See `docs/superpowers/specs/2026-04-29-ccinfer-design.md`.
