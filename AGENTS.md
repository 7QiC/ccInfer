# Repository Guidelines

ccInfer is a C++23 LLM inference framework targeting CUDA GPUs. Optimize for
correctness, explicit ownership, asynchronous execution, and measurable
end-to-end inference performance.

## Project Structure

- `src/common/` — shared types, errors, request types, channels.
- `src/backend/` — backend abstraction and device-memory `Buffer`.
- `src/cache/` — paged KV cache and prefix cache.
- `src/core/` — framework `Tensor` and dtype infrastructure.
- `src/engine/`, `src/executor/`, `src/worker/`, `src/scheduler/` — execution pipeline.
- `src/model/` — model loading and model implementations.
- `src/facade/` — adapters for external libraries such as ccop.
- `src/http/`, `src/tokenizer/` — serving layer.
- `tests/unit/`, `tests/integration/` — GTest and end-to-end tests.
- `third_party/ccop/` — independent CUDA operator-library submodule.

When working inside `third_party/ccop/`, follow its own `AGENTS.md`. Its
operator-learning rules do not apply to normal ccInfer framework code.

## Build & Test

```bash
conda activate llm-infer
cmake -S . -B build -DBUILD_SERVER=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
ctest --test-dir build
```

Run the server with:

```bash
./build/src/ccinfer-server --port 8080 --model-path ./models/qwen3-0.6B
```

Never weaken assertions or tolerances to make tests pass.

## Architecture & Engineering Rules

- Device memory is owned by framework-side `Buffer`; views/operators do not
  acquire ownership implicitly.
- Use `Result<T> = std::expected<T, ErrorCode>` at module/API boundaries.
  Internal invariants guaranteed by correct program logic use `assert`.
  Do not throw exceptions on hot paths.
- Preserve asynchronous execution semantics. Do not add host blocking,
  unnecessary stream synchronization, or implicit device synchronization.
- Keep ownership and lifetime explicit across scheduler, executor, worker,
  KV-cache, and asynchronous in-flight work.
- Prefer simple state ownership and explicit contracts over duplicated state,
  hidden synchronization, or defensive state copies.
- Changes to scheduling or execution flow must preserve established ordering,
  retirement, and ownership semantics; do not simplify away supported in-flight
  concurrency without an explicit design change.

## Performance Work

Treat inference performance as a cross-layer problem:

`request -> scheduler -> executor -> kernels -> KV/cache -> runtime -> GPU`

Before deep optimization, identify the actual bottleneck. Consider:

- scheduler and batching behavior;
- CPU launch gaps and synchronization;
- CUDA Graph applicability;
- KV-cache layout and memory traffic;
- kernel performance;
- communication/overlap when distributed execution is involved.

Use pipeline/system profiling before spending significant effort on isolated
kernels. Kernel-specific optimization belongs primarily in ccop.

Performance changes should record representative workload/shape information and
before/after measurements. Prefer explanations tied to the measured bottleneck,
not surface-level micro-optimizations.

## Coding Style

C++23, namespace `ccinfer`. Follow `.clang-format` (Google base, 100 columns,
4 spaces).

- Types/enums: `PascalCase`
- Functions/variables/files: `snake_case`
- Members: trailing `_`
- Read-only parameters: `const T&`
- Output/in-place parameters: `T*`
- Headers: `.h` + `#pragma once`
- Include order: matching header first, then standard, third-party, project

Comments explain non-obvious intent, invariants, ownership, synchronization, or
performance constraints. Do not restate code or preserve conversational history
in comments.

## Testing

Tests use GTest. Unit tests should avoid GPU requirements when possible;
integration tests may require CUDA and local model weights.

Run the smallest relevant test while iterating, then run `ctest --test-dir build`
before considering the change complete.

Never:
- relax correctness tolerances to hide a bug;
- weaken assertions to make a test pass;
- remove coverage for an existing behavior without an explicit design change.

## Commits & Git Safety

Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `perf:`, `test:`,
`docs:`, `chore:`) and keep each commit to one logical change.

Never perform mutating git operations without explicit user authorization.
Read-only commands such as `git status`, `git diff`, `git log`, and `git show`
are allowed.

`third_party/ccop` is a separate Git repository. Do not accidentally mix ccop
changes with unrelated ccInfer changes, and do not update the ccop submodule
pointer unless that is part of the requested work.
