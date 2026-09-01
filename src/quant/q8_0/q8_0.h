#pragma once

#include <cstdint>
#include <span>

#include "base/error.h"

namespace ccinfer {

// Storage kind used by Weight/WeightSource. This module only defines Q8_0;
// dense is listed so callers can distinguish the two storage representations.
enum class QuantType : uint8_t { kDense, kQ8_0 };

// GGML/GGUF Q8_0 block format. The byte layout is fixed: 2-byte FP16 scale +
// 32 int8 quantized values. Do not reinterpret this as a general quant group.
constexpr int kQ8_0BlockSize = 32;
constexpr int kQ8_0BlockBytes = 34;

struct Q8_0Block {
    uint16_t d_f16;
    int8_t qs[kQ8_0BlockSize];
};

static_assert(sizeof(Q8_0Block) == kQ8_0BlockBytes,
              "Q8_0 block must be 2-byte fp16 scale + 32 int8 quantized values");

// IEEE 754 binary16/binary32 conversions. These are part of the Q8_0 scale
// interpretation and are small, independently tested helpers.
float q8_0_f16_to_float(uint16_t value) noexcept;
uint16_t q8_0_float_to_f16(float value) noexcept;

// Interprets the FP16 scale field of a Q8_0 block as float.
float block_scale(const Q8_0Block& block) noexcept;

// Storage size calculations. Both functions consume canonical ccInfer logical
// shapes (row-major, last dimension is the block-aligned inner dimension).
// GGUF raw dimension order is normalized by the GGUF/WeightSource layer before
// these functions are called.
//
// Non-multiple-of-32 dimensions are invalid Q8_0 rows and are rejected.
Result<int64_t> q8_0_row_bytes(int64_t n_elems) noexcept;
Result<int64_t> q8_0_tensor_bytes(std::span<const int64_t> shape) noexcept;

}  // namespace ccinfer
