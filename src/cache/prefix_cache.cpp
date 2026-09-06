#include "cache/prefix_cache.h"

#include <cassert>

namespace ccinfer {

uint64_t PrefixCache::hash_combine(uint64_t seed, uint64_t val) noexcept {
    seed ^= val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

std::vector<uint64_t> PrefixCache::chain_hashes(const std::vector<int32_t>& tokens, int block_size,
                                                uint64_t seed) {
    return chain_hashes(tokens, static_cast<int>(tokens.size()), block_size, seed);
}

std::vector<uint64_t> PrefixCache::chain_hashes(const std::vector<int32_t>& tokens, int token_count,
                                                int block_size, uint64_t seed) {
    return chain_hashes(tokens, token_count, block_size, /*parent_hash=*/0, seed);
}

std::vector<uint64_t> PrefixCache::chain_hashes(const std::vector<int32_t>& tokens, int token_count,
                                                int block_size, uint64_t parent_hash,
                                                uint64_t seed) {
    assert(block_size > 0 && "block_size must be positive");
    int num_full_blocks = token_count / block_size;
    std::vector<uint64_t> hashes;
    hashes.reserve(num_full_blocks);

    uint64_t h = parent_hash;
    if (seed != 0) h = hash_combine(h, seed);
    for (int b = 0; b < num_full_blocks; ++b) {
        for (int t = 0; t < block_size; ++t) {
            h = hash_combine(h, static_cast<uint64_t>(tokens[b * block_size + t]));
        }
        hashes.push_back(h);
    }
    return hashes;
}

}  // namespace ccinfer
