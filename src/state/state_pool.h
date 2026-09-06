#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

#include "backend/backend.h"
#include "base/error.h"
#include "base/types.h"
#include "config/model_config.h"
#include "state/state.h"
#include "state/state_storage.h"

namespace ccinfer {

// GDN state resource manager (parallel to BlockPool). It owns the full State
// metadata set plus mutable/snapshot free lists. Cache policy (prefix hashes,
// LRU, eviction) is layered on top by StateCache; StatePool knows no cache key.
class StatePool {
public:
    static Result<std::unique_ptr<StatePool>> create(Backend& backend, const ModelConfig& config,
                                                     int max_active, int max_snapshots);

    StateStorage& storage() { return *storage_; }
    int max_active() const { return max_active_; }
    int max_snapshots() const { return max_snapshots_; }

    // Acquires an active state for seq. If seq already owns an active state this
    // is a no-op, letting multiple outstanding batches share one mutable state.
    Result<void> acquire_active(SequenceId seq);
    Result<void> release_active(SequenceId seq);
    std::optional<StateId> active_state_of(SequenceId seq) const;

    // Snapshot states are owned by StateCache via StateId; StatePool only
    // supplies/frees the resource and performs device copies.
    Result<StateId> acquire_snapshot();
    Result<void> release_snapshot(StateId state);

    Result<void> copy_active_to_snapshot(StateId active_state, StateId snapshot_state);
    Result<void> restore_snapshot_to_active(StateId snapshot_state, StateId active_state);

    int num_free_snapshots() const { return static_cast<int>(snapshot_free_list_.size()); }

private:
    std::unique_ptr<StateStorage> storage_;
    std::unique_ptr<State[]> states_;
    int max_active_ = 0;
    int max_snapshots_ = 0;
    StateFreeList mutable_free_list_;
    StateFreeList snapshot_free_list_;
    std::unordered_map<SequenceId, StateId> active_by_seq_;
};

}  // namespace ccinfer
