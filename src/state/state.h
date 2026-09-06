#pragma once

#include <cstdint>

namespace ccinfer {

// GDN state slot identifier. Physical slot ranges:
//   [0, max_active)                 active slots
//   [max_active, max_active + N)    cached snapshot slots
using StateSlotId = std::int32_t;

inline constexpr StateSlotId kInvalidStateSlot = -1;

// Physical slot partitions are fixed at StatePool creation:
//   - Active slots are mutable sequence-owned state.
//   - Cached slots are immutable prefix snapshots.
// A slot is never both active and cached.
enum class StateSlotKind : std::uint8_t { Active, Cached };

enum class StateSlotStatus : std::uint8_t { Free, Occupied };

// StateSlot metadata + state machine (the state-side counterpart to Block).
// Unlike Block, a slot's kind is fixed; status only moves Free <-> Occupied.
struct StateSlot {
    StateSlotId slot_id = kInvalidStateSlot;
    StateSlotKind kind = StateSlotKind::Active;
    StateSlotStatus status = StateSlotStatus::Free;

    // Active slots record the owning sequence; cached slots record the prefix
    // hash. Only the field relevant to the slot's kind is meaningful.
    std::uint64_t owner_seq = 0;
    std::uint64_t prefix_hash = 0;

    bool is_free() const noexcept { return status == StateSlotStatus::Free; }
    bool is_occupied() const noexcept { return status == StateSlotStatus::Occupied; }
    bool is_active_kind() const noexcept { return kind == StateSlotKind::Active; }
    bool is_cached_kind() const noexcept { return kind == StateSlotKind::Cached; }
};

}  // namespace ccinfer
