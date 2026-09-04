#include "state/state_pool.h"

#include <cassert>
#include <utility>

namespace ccinfer {

Result<std::unique_ptr<StatePool>> StatePool::create(Backend& backend, const ModelConfig& config,
                                                     int max_active, int frontier_depth,
                                                     int cached_capacity) {
    if (max_active <= 0 || frontier_depth < 0 || cached_capacity < 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto pool = std::make_unique<StatePool>();
    pool->max_active_ = max_active;
    pool->frontier_depth_ = frontier_depth;
    pool->cached_capacity_ = cached_capacity;
    pool->cached_base_ = max_active;

    auto storage_r = StateStorage::create(backend, config, max_active + cached_capacity);
    if (!storage_r) return std::unexpected(storage_r.error());
    pool->storage_ = std::move(*storage_r);

    pool->free_active_.reserve(static_cast<std::size_t>(max_active));
    for (int slot = 0; slot < max_active; ++slot) {
        pool->free_active_.push_back(slot);
    }
    pool->free_cached_.reserve(static_cast<std::size_t>(cached_capacity));
    for (int i = 0; i < cached_capacity; ++i) {
        pool->free_cached_.push_back(pool->cached_base_ + i);
    }
    return pool;
}

Result<void> StatePool::acquire_active(SequenceId seq) {
    if (active_.contains(seq)) return {};
    if (free_active_.empty()) return std::unexpected(ErrorCode::MaxSequencesReached);

    const StateSlotId slot = free_active_.back();
    free_active_.pop_back();
    if (auto r = storage_->zero_slot(slot); !r) {
        free_active_.push_back(slot);
        return r;
    }
    active_.emplace(seq, slot);
    return {};
}

Result<void> StatePool::release_active(SequenceId seq) {
    auto it = active_.find(seq);
    if (it == active_.end()) return {};
    const StateSlotId slot = it->second;
    active_.erase(it);
    assert(slot >= 0 && slot < max_active_);
    free_active_.push_back(slot);
    return {};
}

std::optional<StateSlotId> StatePool::active_slot_of(SequenceId seq) const {
    auto it = active_.find(seq);
    if (it == active_.end()) return std::nullopt;
    return it->second;
}

Result<void> StatePool::snapshot(uint64_t prefix_hash, int frontier_block,
                                 StateSlotId active_slot) {
    if (active_slot < 0 || active_slot >= max_active_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (frontier_block < 0 || frontier_block >= frontier_depth_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (cached_by_hash_.contains(prefix_hash)) return {};  // Immutable reuse.

    if (free_cached_.empty()) return std::unexpected(ErrorCode::MaxSequencesReached);
    const StateSlotId cached_slot = free_cached_.back();
    free_cached_.pop_back();
    if (auto r = storage_->copy_slot(active_slot, cached_slot); !r) {
        free_cached_.push_back(cached_slot);
        return r;
    }
    cached_by_hash_.emplace(prefix_hash, cached_slot);
    return {};
}

Result<void> StatePool::restore(uint64_t prefix_hash, StateSlotId active_slot) {
    if (active_slot < 0 || active_slot >= max_active_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    auto it = cached_by_hash_.find(prefix_hash);
    if (it == cached_by_hash_.end()) return std::unexpected(ErrorCode::InvalidArgument);
    return storage_->copy_slot(it->second, active_slot);
}

bool StatePool::has_cached(uint64_t prefix_hash) const {
    return cached_by_hash_.contains(prefix_hash);
}

StateSlotId StatePool::cached_slot_of(uint64_t prefix_hash) const {
    auto it = cached_by_hash_.find(prefix_hash);
    if (it == cached_by_hash_.end()) return kInvalidStateSlot;
    return it->second;
}

}  // namespace ccinfer
