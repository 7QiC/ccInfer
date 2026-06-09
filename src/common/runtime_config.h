#pragma once

#include <atomic>

namespace ccinfer {
namespace runtime {

inline std::atomic<int> g_prefill_chunk_size{512};

inline int prefill_chunk_size() noexcept {
    return g_prefill_chunk_size.load(std::memory_order_relaxed);
}

inline void set_prefill_chunk_size(int value) noexcept {
    g_prefill_chunk_size.store(value, std::memory_order_relaxed);
}

}  // namespace runtime
}  // namespace ccinfer
