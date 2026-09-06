#include "cache/block_cache.h"

#include <algorithm>
#include <cassert>
#include <iterator>

#include "block/block_pool.h"
#include "cache/prefix_cache.h"
#include "facade/log.h"

namespace ccinfer {

BlockCache::~BlockCache() {
    // Recycle idle entries so the pool is left clean if the cache outlives a
    // request-owner release. Active cached blocks still have request refs and
    // are simply dropped from the cache index.
    while (evict_one()) {
    }
    by_hash_.clear();
    by_block_.clear();
    entries_.clear();
}

BlockCache::PrefixLookup BlockCache::lookup_prefix_cache(const std::vector<int32_t>& tokens,
                                                         uint64_t namespace_salt) {
    assert(!tokens.empty());
    PrefixLookup result;
    const int block_size = pool_.block_size();
    assert(block_size > 0);
    const auto hashes = PrefixCache::chain_hashes(tokens, block_size, namespace_salt);
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        const auto bucket = by_hash_.find(hashes[i]);
        if (bucket == by_hash_.end() || bucket->second.empty()) {
            ++lookup_misses_;
            break;
        }
        const EntryIt entry_it = bucket->second.front();
        const BlockId id = entry_it->block_id;
        assert(by_block_.find(id) != by_block_.end());
        if (pool_.ref_count(id) == 0) {
            assert(entry_it->is_in_idle_lru());
            idle_lru_.erase(IdleLru::s_iterator_to(*entry_it));
        }
        pool_.retain_request(id);
        result.block_table.push_back(id);
        result.parent_hash = hashes[i];
        ++result.prefix_hit_blocks;
        ++lookup_hits_;
    }
    result.block_table.set_shared_count(result.prefix_hit_blocks);
    return result;
}

uint64_t BlockCache::publish_full_block(uint64_t parent_hash, const std::vector<int32_t>& tokens,
                                        BlockId block_id, uint64_t seed) {
    const int block_size = pool_.block_size();
    assert(block_size > 0);
    assert(tokens.size() >= static_cast<std::size_t>(block_size));
    const auto hashes =
        PrefixCache::chain_hashes(tokens, block_size, block_size, parent_hash, seed);
    assert(!hashes.empty());
    const uint64_t hash = hashes.front();

    auto existing = by_block_.find(block_id);
    if (existing != by_block_.end()) {
        assert(existing->second->prefix_hash == hash && "block cannot change prefix while cached");
        return hash;
    }

    assert(pool_.ref_count(block_id) > 0 && "publish requires a request-owned block");
    entries_.emplace_back(hash, block_id);
    const EntryIt entry_it = std::prev(entries_.end());
    by_block_.emplace(block_id, entry_it);
    by_hash_[hash].push_back(entry_it);
    ccLog::debug("block cache insert block={} hash={:x}", block_id, hash);
    return hash;
}

bool BlockCache::contains(BlockId id) const { return by_block_.contains(id); }

bool BlockCache::is_idle(BlockId id) const {
    auto it = by_block_.find(id);
    return it != by_block_.end() && it->second->is_in_idle_lru();
}

void BlockCache::mark_idle(BlockId id) {
    auto it = by_block_.find(id);
    assert(it != by_block_.end() && "mark_idle requires a cached block");
    auto& entry = *it->second;
    assert(pool_.ref_count(id) == 0);
    assert(!entry.is_in_idle_lru());
    idle_lru_.push_back(entry);
}

bool BlockCache::evict_one() {
    if (idle_lru_.empty()) return false;
    const BlockId victim = idle_lru_.front().block_id;
    assert(pool_.ref_count(victim) == 0);
    erase_entry(victim);
    pool_.recycle(victim);
    ++evictions_;
    return true;
}

void BlockCache::erase_entry(BlockId id) {
    auto block_it = by_block_.find(id);
    assert(block_it != by_block_.end());
    const EntryIt entry_it = block_it->second;

    if (entry_it->is_in_idle_lru()) {
        idle_lru_.erase(IdleLru::s_iterator_to(*entry_it));
    }

    const uint64_t hash = entry_it->prefix_hash;
    auto hash_it = by_hash_.find(hash);
    assert(hash_it != by_hash_.end());
    auto& bucket = hash_it->second;
    auto pos = std::find(bucket.begin(), bucket.end(), entry_it);
    assert(pos != bucket.end());
    *pos = bucket.back();
    bucket.pop_back();
    if (bucket.empty()) by_hash_.erase(hash_it);

    by_block_.erase(block_it);
    entries_.erase(entry_it);
}

BlockCacheStats BlockCache::stats() const {
    BlockCacheStats s;
    s.block_total = pool_.max_blocks();
    s.block_free = pool_.num_free_blocks();
    s.block_size = pool_.block_size();
    s.block_cached_idle = static_cast<int>(idle_lru_.size());
    s.block_active = s.block_total - s.block_free - s.block_cached_idle;
    s.lookup_hits = lookup_hits_;
    s.lookup_misses = lookup_misses_;
    s.evictions = evictions_;
    s.cached_blocks = entries_.size();
    return s;
}

}  // namespace ccinfer
