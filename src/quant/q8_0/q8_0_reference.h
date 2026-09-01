#pragma once

#include <span>

#include "base/error.h"
#include "quant/q8_0/q8_0.h"

namespace ccinfer {

// CPU correctness reference for Q8_0. Not used by the production inference hot
// path. M1 uses dequantize reference for the GGUF->dense->cuBLAS path and M3
// uses it as a correctness oracle for the developer-written ccop Q8_0 kernel.
//
// Both functions require block-aligned logical dimensions:
//   - quantize: src.size() % kQ8_0BlockSize == 0
//   - dequantize: dst.size() == src.size() * kQ8_0BlockSize
// Invalid inputs return ErrorCode::InvalidArgument instead of silently padding.
Result<void> dequantize_q8_0_reference(std::span<const Q8_0Block> src,
                                       std::span<float> dst) noexcept;

Result<void> quantize_q8_0_reference(std::span<const float> src,
                                     std::span<Q8_0Block> dst) noexcept;

}  // namespace ccinfer
