#pragma once

#include <cstdint>

namespace ccinfer {

// GDN state slot identifier. Physical slot ranges:
//   [0, max_active)                 active slots
//   [max_active, max_active + N)    cached snapshot slots
using StateSlotId = std::int32_t;

inline constexpr StateSlotId kInvalidStateSlot = -1;

}  // namespace ccinfer
