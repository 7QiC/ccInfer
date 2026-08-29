#include "cache/prefix_cache.h"

#include <algorithm>
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

std::optional<int32_t> PrefixCache::lookup(uint64_t hash) const {
    auto it = hash_to_blocks_.find(hash);
    if (it == hash_to_blocks_.end()) {
        ++lookup_misses_;
        return std::nullopt;
    }
    assert(!it->second.empty());
    ++lookup_hits_;
    return it->second.front();
}

void PrefixCache::insert(uint64_t hash, int32_t block_id) {
    auto it = block_to_hash_.find(block_id);
    assert(it == block_to_hash_.end() || it->second == hash);
    if (it == block_to_hash_.end()) {
        hash_to_blocks_[hash].push_back(block_id);
        block_to_hash_[block_id] = hash;
    }
}

void PrefixCache::remove_by_block(int32_t block_id) {
    auto it = block_to_hash_.find(block_id);
    assert(it != block_to_hash_.end());
    auto bucket_it = hash_to_blocks_.find(it->second);
    assert(bucket_it != hash_to_blocks_.end());
    auto& bucket = bucket_it->second;
    auto block_it = std::find(bucket.begin(), bucket.end(), block_id);
    assert(block_it != bucket.end());
    *block_it = bucket.back();
    bucket.pop_back();
    if (bucket.empty()) hash_to_blocks_.erase(bucket_it);
    block_to_hash_.erase(it);
}

PrefixCacheStats PrefixCache::stats() const {
    std::size_t cached_blocks = 0;
    for (const auto& entry : hash_to_blocks_) cached_blocks += entry.second.size();
    return {.lookup_hits = lookup_hits_,
            .lookup_misses = lookup_misses_,
            .evictions = evictions_,
            .cached_blocks = cached_blocks};
}

}  // namespace ccinfer
