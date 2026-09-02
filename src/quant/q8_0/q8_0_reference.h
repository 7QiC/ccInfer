#pragma once

#include <cstdint>
#include <span>

#include "base/error.h"
#include "ccop/quant.h"

namespace ccinfer {

// IEEE 754 binary16/binary32 helpers used by the CPU Q8_0 reference.
float q8_0_f16_to_float(std::uint16_t value) noexcept;
std::uint16_t q8_0_float_to_f16(float value) noexcept;

// CPU Q8_0 correctness reference. Uses the canonical ccop::Q8_0Block ABI.
// Both functions require block-aligned logical dimensions and do not silently
// pad partial blocks.
Result<void> dequantize_q8_0_reference(std::span<const ccop::Q8_0Block> src,
                                       std::span<float> dst) noexcept;

Result<void> quantize_q8_0_reference(std::span<const float> src,
                                     std::span<ccop::Q8_0Block> dst) noexcept;

}  // namespace ccinfer
