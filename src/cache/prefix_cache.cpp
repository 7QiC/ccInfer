#include "cache/prefix_cache.h"

#include <cassert>

#include "base/error_code.h"

namespace ccinfer {

uint64_t PrefixCache::hash_combine(uint64_t seed, uint64_t val) noexcept {
    seed ^= val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

std::vector<uint64_t> PrefixCache::chain_hashes(const std::vector<int32_t>& tokens, int block_size,
                                                uint64_t namespace_salt) {
    return chain_hashes(tokens, static_cast<int>(tokens.size()), block_size, namespace_salt);
}

std::vector<uint64_t> PrefixCache::chain_hashes(const std::vector<int32_t>& tokens, int token_count,
                                                int block_size, uint64_t namespace_salt) {
    assert(block_size > 0 && "block_size must be positive");
    int num_full_blocks = token_count / block_size;
    std::vector<uint64_t> hashes;
    hashes.reserve(num_full_blocks);

    uint64_t h = namespace_salt;
    for (int b = 0; b < num_full_blocks; ++b) {
        for (int t = 0; t < block_size; ++t) {
            h = hash_combine(h, static_cast<uint64_t>(tokens[b * block_size + t]));
        }
        hashes.push_back(h);
    }
    return hashes;
}

std::optional<int32_t> PrefixCache::lookup(uint64_t hash) const {
    auto it = hash_to_block_.find(hash);
    if (it == hash_to_block_.end()) {
        ++lookup_misses_;
        return std::nullopt;
    }
    ++lookup_hits_;
    return it->second;
}

Result<void> PrefixCache::insert(uint64_t hash, int32_t block_id) {
    // Consistency check: block_id must not already map to a different hash.
    auto rev = block_to_hash_.find(block_id);
    if (rev != block_to_hash_.end()) {
        if (rev->second == hash) return {};  // idempotent
        return std::unexpected(ErrorCode::KVBlockHashCollision);
    }

    // Consistency check: hash must not already map to a different block_id.
    auto it = hash_to_block_.find(hash);
    if (it != hash_to_block_.end()) {
        if (it->second == block_id) return {};  // idempotent
        return std::unexpected(ErrorCode::KVBlockHashCollision);
    }

    hash_to_block_[hash] = block_id;
    block_to_hash_[block_id] = hash;
    return {};
}

void PrefixCache::remove(uint64_t hash) {
    auto it = hash_to_block_.find(hash);
    if (it == hash_to_block_.end()) return;
    block_to_hash_.erase(it->second);
    hash_to_block_.erase(it);
}

void PrefixCache::remove_by_block(int32_t block_id) {
    auto it = block_to_hash_.find(block_id);
    if (it == block_to_hash_.end()) return;
    hash_to_block_.erase(it->second);
    block_to_hash_.erase(it);
}

PrefixCacheStats PrefixCache::stats() const {
    return {.lookup_hits = lookup_hits_,
            .lookup_misses = lookup_misses_,
            .evictions = evictions_,
            .cached_blocks = hash_to_block_.size()};
}

}  // namespace ccinfer
