#include "cache/state_cache.h"

#include <cassert>
#include <iterator>

#include "state/state_pool.h"

namespace ccinfer {

StateCache::~StateCache() {
    for (auto& entry : entries_) {
        (void)pool_.release_snapshot(entry.state_id);
    }
    lru_.clear();
    entries_.clear();
}

StateCache::StateCache(StatePool& pool, int frontier_depth) : pool_(pool) {
    assert(frontier_depth >= 0);
    frontier_depth_ = frontier_depth;
}

Result<void> StateCache::snapshot(uint64_t prefix_hash, int frontier_block, StateId active_state) {
    if (frontier_block < 0 || frontier_block >= frontier_depth_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (has(prefix_hash)) return {};

    if (pool_.num_free_snapshots() == 0) {
        if (!evict_one()) return std::unexpected(ErrorCode::MaxSequencesReached);
    }
    auto snapshot_r = pool_.acquire_snapshot();
    if (!snapshot_r) return std::unexpected(snapshot_r.error());

    const StateId snapshot_state = *snapshot_r;
    if (auto r = pool_.copy_active_to_snapshot(active_state, snapshot_state); !r) {
        (void)pool_.release_snapshot(snapshot_state);
        return r;
    }

    entries_.emplace_back(prefix_hash, frontier_block, snapshot_state);
    const auto entry_it = std::prev(entries_.end());
    by_hash_.emplace(prefix_hash, entry_it);
    lru_.push_back(*entry_it);
    return {};
}

Result<void> StateCache::restore(uint64_t prefix_hash, StateId active_state) {
    auto it = by_hash_.find(prefix_hash);
    if (it == by_hash_.end()) {
        ++lookup_misses_;
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    ++lookup_hits_;
    EntryIt entry_it = it->second;
    if (entry_it->is_in_lru()) {
        lru_.splice(lru_.end(), lru_, StateLru::s_iterator_to(*entry_it));
    }
    return pool_.restore_snapshot_to_active(entry_it->state_id, active_state);
}

bool StateCache::has(uint64_t prefix_hash) const { return by_hash_.contains(prefix_hash); }

StateId StateCache::state_id_of(uint64_t prefix_hash) const {
    auto it = by_hash_.find(prefix_hash);
    if (it == by_hash_.end()) return kInvalidStateId;
    return it->second->state_id;
}

bool StateCache::evict_one() {
    if (lru_.empty()) return false;
    const StateId victim_state = lru_.front().state_id;
    const uint64_t victim_hash = lru_.front().prefix_hash;
    auto it = by_hash_.find(victim_hash);
    assert(it != by_hash_.end());
    erase_entry(it->second);
    (void)pool_.release_snapshot(victim_state);
    ++evictions_;
    return true;
}

void StateCache::erase_entry(const EntryIt& entry_it) {
    const uint64_t hash = entry_it->prefix_hash;
    if (entry_it->is_in_lru()) {
        lru_.erase(StateLru::s_iterator_to(*entry_it));
    }
    by_hash_.erase(hash);
    entries_.erase(entry_it);
}

StateCacheStats StateCache::stats() const {
    StateCacheStats s;
    s.lookup_hits = lookup_hits_;
    s.lookup_misses = lookup_misses_;
    s.evictions = evictions_;
    s.cached_snapshots = entries_.size();
    return s;
}

}  // namespace ccinfer
