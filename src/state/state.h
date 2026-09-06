#pragma once

#include <cstdint>

#include <boost/intrusive/list.hpp>

namespace ccinfer {

// GDN state resource identifier. State objects live permanently in StatePool
// metadata storage. Ranges are fixed at creation: [0, max_mutable) mutable
// states and [max_mutable, max_mutable + max_snapshots) snapshot states.
using StateId = std::int32_t;

inline constexpr StateId kInvalidStateId = -1;

enum class StateKind : std::uint8_t {
    Mutable,
    Snapshot,
};

enum class StateStatus : std::uint8_t {
    Free,
    InUse,
};

using FreeHook = boost::intrusive::list_member_hook<>;

// Resource metadata for one GDN state. Cache identity (prefix hash, frontier
// index, LRU) lives in StateCacheEntry, not here.
//
// State objects live permanently in StatePool's fixed metadata storage and must
// never be moved after being linked into a free list.
struct State {
    StateId id = kInvalidStateId;
    StateKind kind = StateKind::Mutable;
    StateStatus status = StateStatus::Free;

    FreeHook free_hook;

    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;

    bool is_free() const noexcept { return status == StateStatus::Free; }
    bool is_in_use() const noexcept { return status == StateStatus::InUse; }
    bool is_mutable_kind() const noexcept { return kind == StateKind::Mutable; }
    bool is_snapshot_kind() const noexcept { return kind == StateKind::Snapshot; }
};

using StateFreeList = boost::intrusive::list<
    State,
    boost::intrusive::member_hook<State, boost::intrusive::list_member_hook<>, &State::free_hook>>;

}  // namespace ccinfer
