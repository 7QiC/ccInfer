#pragma once

#include <cstdint>
#include <vector>

namespace ccinfer {

// Prefix hash semantics only. BlockCache/StateCache build their entries and LRU
// on top of these deterministic chain hashes; this class intentionally stores
// no cache metadata.
class PrefixCache {
public:
    PrefixCache() = delete;

    static std::vector<uint64_t> chain_hashes(const std::vector<int32_t>& tokens, int block_size,
                                              uint64_t namespace_salt = 0);

    static std::vector<uint64_t> chain_hashes(const std::vector<int32_t>& tokens, int token_count,
                                              int block_size, uint64_t namespace_salt);

    static std::vector<uint64_t> chain_hashes(const std::vector<int32_t>& tokens, int token_count,
                                              int block_size, uint64_t parent_hash, uint64_t seed);

private:
    static uint64_t hash_combine(uint64_t seed, uint64_t val) noexcept;
};

}  // namespace ccinfer
