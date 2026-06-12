#pragma once

#include <cstdint>

namespace ccinfer {

inline bool is_aligned_4(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 0x3u) == 0; }

inline bool is_aligned_8(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 0x7u) == 0; }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline int ceil_div_i64(int64_t a, int b) { return static_cast<int>((a + b - 1) / b); }

}  // namespace ccinfer
