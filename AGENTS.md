# Repository Guidelines

Contributor guide for ccInfer, a high-performance C++23 LLM inference framework targeting CUDA GPUs.

## Project Structure & Module Organization

- `src/` — Source code. `common/` holds shared types (`Result<T>`, error codes, request types, and channels); `backend/` abstracts the GPU backend (`Backend`, `Buffer`); `cache/` implements the paged KV cache and prefix cache; `core/` defines the framework `Tensor` and dtype traits; `engine/`, `executor/`, `worker/`, and `scheduler/` form the execution pipeline; `model/` handles model loading and Qwen3; `facade/` adapts external libraries such as ccop; `http/` and `tokenizer/` make up the server layer; `main.cpp` is the server entry point.
- `tests/` — `unit/` GTest tests and `integration/` end-to-end tests.
- `docs/`, `scripts/`, `models/`, `tools/` — Project notes, Python benchmark/profiling scripts, downloaded model weights, and helper utilities.

## Build, Test, and Development Commands

```bash
conda activate llm-infer
cmake -S . -B build -DBUILD_SERVER=ON -DCMAKE_CUDA_ARCHITECTURES=89
make -C build -j$(nproc)
ctest --test-dir build
```

Configure with `-DBUILD_SERVER=ON` to build the HTTP server (`BUILD_TESTS` defaults to ON). Run the server with `./build/src/ccinfer-server --port 8080 --model-path ./models/qwen3-0.6B`. Requires CUDA Toolkit 11.8+, GCC 13+, CMake 3.20+, Boost 1.83+, nlohmann-json, fmt, spdlog.

## Coding Style & Naming Conventions

C++23 with the `ccinfer` namespace. `.clang-format` enforces Google base style: 100-column limit, 4-space indentation, no tabs; run `clang-format-20` on changed files before committing. Classes/enums use PascalCase; functions, variables, and files use snake_case; members end with an underscore (e.g. `int count_;`). Parameter semantics: read-only parameters use `const T&`; parameters that are written (outputs/in-place) take a pointer `T*`, never a reference. Headers use `.h` with `#pragma once` and sorted includes (source-file header first, then C std → C++ std → third-party → project). Errors use `Result<T> = std::expected<T, ErrorCode>` from `common/error_code.h`; never throw in hot paths. Device memory is owned by the framework-side `Buffer` (`backend/buffer.h`).

## Comment Policy

- No unnecessary comments: add concise comments only where intent or constraints are not obvious from the code. Never restate code, add line-by-line noise, or copy conversational decisions into code comments.
- Prefer self-documenting code: naming and structure carry meaning; comments explain "why", not "what".

## Testing Guidelines

Tests use GTest, one test file per module, named `test_<module>.cpp`. Unit tests should run without a GPU; integration tests may require a GPU and/or a local Qwen3-0.6B model. Run everything with `ctest --test-dir build`, or the scheduler suite with `ctest --test-dir build -R SchedulerTest`. Never weaken assertions or relax tolerances to make a failing test pass — fix the root cause.

## Commit & Pull Request Guidelines

History follows Conventional Commits: `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `chore:`. Use a lowercase imperative subject under 72 characters, e.g. `feat: add prefix-cache LRU eviction`. Keep PRs to one logical change, describe what and why, link the relevant issue, and include benchmark or correctness evidence for performance changes.
