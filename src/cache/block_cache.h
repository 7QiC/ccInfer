#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <unordered_map>
#include <vector>

#include <boost/intrusive/list.hpp>

#include "block/block.h"
#include "block/block_table.h"
#include "cache/cache_stats.h"

namespace ccinfer {

class BlockPool;

using IdleLruHook = boost::intrusive::list_member_hook<>;

struct BlockCacheEntry {
    uint64_t prefix_hash = 0;
    BlockId block_id = kInvalidBlockId;
    IdleLruHook idle_lru_hook;

    BlockCacheEntry() = default;
    BlockCacheEntry(uint64_t hash, BlockId id) : prefix_hash(hash), block_id(id) {}
    BlockCacheEntry(const BlockCacheEntry&) = delete;
    BlockCacheEntry& operator=(const BlockCacheEntry&) = delete;
    BlockCacheEntry(BlockCacheEntry&&) = delete;
    BlockCacheEntry& operator=(BlockCacheEntry&&) = delete;

    bool is_in_idle_lru() const noexcept { return idle_lru_hook.is_linked(); }
};

using IdleLru = boost::intrusive::list<
    BlockCacheEntry,
    boost::intrusive::member_hook<BlockCacheEntry, boost::intrusive::list_member_hook<>,
                                  &BlockCacheEntry::idle_lru_hook>>;

// Cached KV-block subset + replacement policy. It owns cache entries only; the
// full resource set and ref counts remain in BlockPool.
class BlockCache {
public:
    explicit BlockCache(BlockPool& pool) : pool_(pool) {}
    ~BlockCache();

    struct PrefixLookup {
        BlockTable block_table;
        int prefix_hit_blocks = 0;
        uint64_t parent_hash = 0;
    };

    PrefixLookup lookup_prefix_cache(const std::vector<int32_t>& tokens,
                                     uint64_t namespace_salt = 0);

    // Publishes one full block into the cache. The request caller must already
    // own the block (BlockPool ref_count > 0). Returns the block's prefix hash.
    uint64_t publish_full_block(uint64_t parent_hash, const std::vector<int32_t>& tokens,
                                BlockId block_id, uint64_t seed = 0);

    bool contains(BlockId id) const;
    bool is_idle(BlockId id) const;

    // Moves a cached block with zero request refs into the idle LRU.
    void mark_idle(BlockId id);

    // Evicts the LRU idle block. Erases its cache entry and recycles the block
    // through BlockPool. Returns false when no idle block exists.
    bool evict_one();

    std::size_t num_cached_blocks() const { return entries_.size(); }
    std::size_t num_idle_blocks() const { return idle_lru_.size(); }
    BlockCacheStats stats() const;

private:
    using EntryList = std::list<BlockCacheEntry>;
    using EntryIt = EntryList::iterator;

    void erase_entry(BlockId id);

    BlockPool& pool_;
    std::list<BlockCacheEntry> entries_;
    IdleLru idle_lru_;
    std::unordered_map<uint64_t, std::deque<EntryIt>> by_hash_;
    std::unordered_map<BlockId, EntryIt> by_block_;
    mutable uint64_t lookup_hits_ = 0;
    mutable uint64_t lookup_misses_ = 0;
    uint64_t evictions_ = 0;
};

}  // namespace ccinfer
