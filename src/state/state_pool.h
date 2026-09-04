#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "backend/backend.h"
#include "base/error.h"
#include "base/types.h"
#include "config/model_config.h"
#include "state/state.h"
#include "state/state_storage.h"

namespace ccinfer {

// Logical GDN state-slot manager (parallel to BlockPool).
//
// M1 implements a deliberately small correct subset:
//   - active slots: mutable, zero-on-acquire, free-list reuse, owned by seq_id;
//   - cached slots: immutable snapshots keyed by prefix identity (the same
//     uint64_t block-frontier hash used by PrefixCache), not by seq_id;
//   - frontier depth and cached slot capacity are separate concepts.
// no LRU/eviction/budgets/preemption are included (M2).
class StatePool {
public:
    // frontier_depth: how many leading full-block frontiers of a prefix may be
    // cached (e.g. 3 -> frontier indices 0/1/2). cached_capacity: total number
    // of cached state slots available across all prefixes. M1 callers may pass
    // equal values, but the StatePool treats them as distinct fields.
    static Result<std::unique_ptr<StatePool>> create(Backend& backend, const ModelConfig& config,
                                                     int max_active, int frontier_depth,
                                                     int cached_capacity);

    StateStorage& storage() { return *storage_; }
    int max_active() const { return max_active_; }
    int frontier_depth() const { return frontier_depth_; }
    int cached_capacity() const { return cached_capacity_; }

    // Acquires an active slot for seq. If seq already has an active slot this
    // is a no-op, which lets multiple outstanding batches for the same sequence
    // share one mutable slot while preserving FIFO state ordering.
    Result<void> acquire_active(SequenceId seq);

    // Releases the active slot. Unknown seq is a no-op (idempotent cleanup).
    Result<void> release_active(SequenceId seq);

    std::optional<StateSlotId> active_slot_of(SequenceId seq) const;

    // active -> cached, keyed by the PrefixCache frontier hash. frontier_block
    // is used only to enforce frontier_depth_ (0 <= frontier_block < depth).
    // If the prefix hash is already cached this is a no-op (immutable reuse).
    Result<void> snapshot(uint64_t prefix_hash, int frontier_block, StateSlotId active_slot);

    // cached -> active. active_slot must already be an acquired active slot.
    Result<void> restore(uint64_t prefix_hash, StateSlotId active_slot);

    bool has_cached(uint64_t prefix_hash) const;
    StateSlotId cached_slot_of(uint64_t prefix_hash) const;

private:
    std::unique_ptr<StateStorage> storage_;
    int max_active_ = 0;
    int frontier_depth_ = 0;
    int cached_capacity_ = 0;
    int cached_base_ = 0;
    std::vector<StateSlotId> free_active_;
    std::vector<StateSlotId> free_cached_;
    std::unordered_map<SequenceId, StateSlotId> active_;
    std::unordered_map<uint64_t, StateSlotId> cached_by_hash_;
};

}  // namespace ccinfer
