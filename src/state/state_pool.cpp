#include "state/state_pool.h"

#include <cassert>

namespace ccinfer {

Result<std::unique_ptr<StatePool>> StatePool::create(Backend& backend, const ModelConfig& config,
                                                     int max_active, int max_snapshots) {
    if (max_active <= 0 || max_snapshots < 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto pool = std::make_unique<StatePool>();
    pool->max_active_ = max_active;
    pool->max_snapshots_ = max_snapshots;

    const int total_states = max_active + max_snapshots;
    auto storage_r = StateStorage::create(backend, config, total_states);
    if (!storage_r) return std::unexpected(storage_r.error());
    pool->storage_ = std::move(*storage_r);

    pool->states_ = std::make_unique<State[]>(total_states);
    for (int state = 0; state < total_states; ++state) {
        auto& meta = pool->states_[state];
        meta.id = state;
        meta.kind = state < max_active ? StateKind::Mutable : StateKind::Snapshot;
        meta.status = StateStatus::Free;
        if (meta.is_mutable_kind()) {
            pool->mutable_free_list_.push_back(meta);
        } else {
            pool->snapshot_free_list_.push_back(meta);
        }
    }
    return pool;
}

Result<void> StatePool::acquire_active(SequenceId seq) {
    if (active_by_seq_.contains(seq)) return {};
    if (mutable_free_list_.empty()) return std::unexpected(ErrorCode::MaxSequencesReached);

    auto& meta = mutable_free_list_.front();
    assert(meta.is_free() && meta.is_mutable_kind());
    mutable_free_list_.pop_front();
    const StateId state = meta.id;
    if (auto r = storage_->zero_state(state); !r) {
        mutable_free_list_.push_back(meta);
        return r;
    }
    meta.status = StateStatus::InUse;
    active_by_seq_.emplace(seq, state);
    return {};
}

Result<void> StatePool::release_active(SequenceId seq) {
    auto it = active_by_seq_.find(seq);
    if (it == active_by_seq_.end()) return {};
    const StateId state = it->second;
    active_by_seq_.erase(it);
    assert(state >= 0 && state < max_active_);
    auto& meta = states_[state];
    assert(meta.is_in_use() && meta.is_mutable_kind());
    meta.status = StateStatus::Free;
    mutable_free_list_.push_back(meta);
    return {};
}

std::optional<StateId> StatePool::active_state_of(SequenceId seq) const {
    auto it = active_by_seq_.find(seq);
    if (it == active_by_seq_.end()) return std::nullopt;
    return it->second;
}

Result<StateId> StatePool::acquire_snapshot() {
    if (snapshot_free_list_.empty()) return std::unexpected(ErrorCode::MaxSequencesReached);
    auto& meta = snapshot_free_list_.front();
    assert(meta.is_free() && meta.is_snapshot_kind());
    snapshot_free_list_.pop_front();
    const StateId state = meta.id;
    meta.status = StateStatus::InUse;
    return state;
}

Result<void> StatePool::release_snapshot(StateId state) {
    if (state < max_active_ || state >= max_active_ + max_snapshots_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    auto& meta = states_[state];
    assert(meta.is_in_use() && meta.is_snapshot_kind());
    meta.status = StateStatus::Free;
    snapshot_free_list_.push_back(meta);
    return {};
}

Result<void> StatePool::copy_active_to_snapshot(StateId active_state, StateId snapshot_state) {
    if (active_state < 0 || active_state >= max_active_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (snapshot_state < max_active_ || snapshot_state >= max_active_ + max_snapshots_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return storage_->copy_state(active_state, snapshot_state);
}

Result<void> StatePool::restore_snapshot_to_active(StateId snapshot_state, StateId active_state) {
    if (active_state < 0 || active_state >= max_active_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (snapshot_state < max_active_ || snapshot_state >= max_active_ + max_snapshots_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return storage_->copy_state(snapshot_state, active_state);
}

}  // namespace ccinfer
