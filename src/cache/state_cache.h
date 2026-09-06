#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>

#include <boost/intrusive/list.hpp>

#include "base/error.h"
#include "cache/cache_stats.h"
#include "state/state.h"

namespace ccinfer {

class StatePool;

using StateLruHook = boost::intrusive::list_member_hook<>;

struct StateCacheEntry {
    uint64_t prefix_hash = 0;
    int frontier_block_index = -1;
    StateId state_id = kInvalidStateId;

    StateLruHook lru_hook;

    StateCacheEntry() = default;
    StateCacheEntry(uint64_t hash, int frontier_block, StateId state)
        : prefix_hash(hash), frontier_block_index(frontier_block), state_id(state) {}
    StateCacheEntry(const StateCacheEntry&) = delete;
    StateCacheEntry& operator=(const StateCacheEntry&) = delete;
    StateCacheEntry(StateCacheEntry&&) = delete;
    StateCacheEntry& operator=(StateCacheEntry&&) = delete;

    bool is_in_lru() const noexcept { return lru_hook.is_linked(); }
};

using StateLru = boost::intrusive::list<
    StateCacheEntry,
    boost::intrusive::member_hook<StateCacheEntry, boost::intrusive::list_member_hook<>,
                                  &StateCacheEntry::lru_hook>>;

// State snapshot prefix cache. It owns the cached subset + LRU policy; physical
// snapshot resources and copy/restore primitives live in StatePool.
class StateCache {
public:
    StateCache(StatePool& pool, int frontier_depth);
    ~StateCache();

    // Copies active_state into a newly acquired snapshot and registers the
    // prefix entry. Existing prefix entries are reused (no copy).
    Result<void> snapshot(uint64_t prefix_hash, int frontier_block, StateId active_state);

    // Copies the cached snapshot into active_state and touches the LRU entry.
    Result<void> restore(uint64_t prefix_hash, StateId active_state);

    bool has(uint64_t prefix_hash) const;
    StateId state_id_of(uint64_t prefix_hash) const;

    // Evicts the LRU snapshot and releases its StatePool resource.
    bool evict_one();

    int frontier_depth() const { return frontier_depth_; }
    std::size_t size() const { return by_hash_.size(); }
    std::size_t num_idle() const { return lru_.size(); }
    StateCacheStats stats() const;

private:
    using EntryList = std::list<StateCacheEntry>;
    using EntryIt = EntryList::iterator;

    void erase_entry(const EntryIt& entry_it);

    StatePool& pool_;
    int frontier_depth_ = 0;
    std::list<StateCacheEntry> entries_;
    StateLru lru_;
    std::unordered_map<uint64_t, EntryIt> by_hash_;
    mutable uint64_t lookup_hits_ = 0;
    mutable uint64_t lookup_misses_ = 0;
    uint64_t evictions_ = 0;
};

}  // namespace ccinfer
