#pragma once

#include <cstdint>

namespace ccinfer {

inline uint64_t hash_combine(uint64_t seed, uint64_t val) {
    seed ^= val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

}  // namespace ccinfer
