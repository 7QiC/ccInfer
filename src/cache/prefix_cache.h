#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "base/result.h"
#include "cache/cache_stats.h"

namespace ccinfer {

// Single-threaded prefix cache. All access must come from the same worker thread.

class PrefixCache {
public:
    PrefixCache() = default;

    static std::vector<uint64_t> chain_hashes(const std::vector<int32_t>& tokens, int block_size,
                                              uint64_t namespace_salt = 0);

    static std::vector<uint64_t> chain_hashes(const std::vector<int32_t>& tokens, int token_count,
                                              int block_size, uint64_t namespace_salt);

    std::optional<int32_t> lookup(uint64_t hash) const;

    Result<void> insert(uint64_t hash, int32_t block_id);

    void remove(uint64_t hash);
    void remove_by_block(int32_t block_id);

    void record_eviction() { ++evictions_; }

    std::size_t size() const { return hash_to_block_.size(); }
    PrefixCacheStats stats() const;

private:
    static uint64_t hash_combine(uint64_t seed, uint64_t val) noexcept;

    std::unordered_map<uint64_t, int32_t> hash_to_block_;
    std::unordered_map<int32_t, uint64_t> block_to_hash_;
    // Stats counters. lookup_hits/misses are mutable because lookup() is const.
    mutable uint64_t lookup_hits_ = 0;
    mutable uint64_t lookup_misses_ = 0;
    uint64_t evictions_ = 0;
};

}  // namespace ccinfer
