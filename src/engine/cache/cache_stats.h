#pragma once

#include <cstddef>
#include <cstdint>

namespace ccinfer {
namespace engine {

struct PrefixCacheStats {
    uint64_t lookup_hits = 0;
    uint64_t lookup_misses = 0;
    uint64_t evictions = 0;
    std::size_t cached_blocks = 0;
};

struct KVCacheStats {
    int block_total = 0;
    int block_active = 0;
    int block_free = 0;
    int block_cached_idle = 0;
    int block_size = 0;

    PrefixCacheStats prefix;
};

}  // namespace engine
}  // namespace ccinfer
