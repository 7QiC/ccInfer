#include "quant/q8_0/q8_0.h"

#include <bit>
#include <cstdint>

namespace ccinfer {

float q8_0_f16_to_float(uint16_t h) noexcept {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;

    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal half: normalize to a float by finding the leading bit.
            uint32_t m = mant;
            int shift = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++shift;
            }
            const uint32_t float_exp = static_cast<uint32_t>(127 - 14 - shift);
            bits = sign | (float_exp << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t float_exp = exp + (127 - 15);
        bits = sign | (float_exp << 23) | (mant << 13);
    }

    return std::bit_cast<float>(bits);
}

uint16_t q8_0_float_to_f16(float value) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;

    if (exp == 0xFFu) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }

    const int32_t half_exp = static_cast<int32_t>(exp) - 127 + 15;
    if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<uint16_t>(sign);

        const uint32_t m = mant | 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
        uint32_t half_mant = m >> shift;

        // Round-to-nearest-even on the discarded bits.
        const uint32_t round_bit_mask = 1u << (shift - 1);
        if (m & round_bit_mask) {
            const uint32_t lower = m & (round_bit_mask - 1);
            if (lower != 0 || (half_mant & 1u)) ++half_mant;
        }

        if (half_mant >= 0x400u) {
            return static_cast<uint16_t>(sign | (1u << 10));
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint32_t half_mant = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant >= 0x400u) {
            half_mant = 0;
            const uint32_t next_exp = static_cast<uint32_t>(half_exp + 1);
            if (next_exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
            return static_cast<uint16_t>(sign | (next_exp << 10) | half_mant);
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10) | half_mant);
}

float block_scale(const Q8_0Block& block) noexcept { return q8_0_f16_to_float(block.d_f16); }

Result<int64_t> q8_0_row_bytes(int64_t n_elems) noexcept {
    if (n_elems <= 0 || n_elems % kQ8_0BlockSize != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return (n_elems / kQ8_0BlockSize) * kQ8_0BlockBytes;
}

Result<int64_t> q8_0_tensor_bytes(std::span<const int64_t> shape) noexcept {
    if (shape.empty()) return std::unexpected(ErrorCode::InvalidArgument);

    int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) {
        if (shape[i] <= 0) return std::unexpected(ErrorCode::InvalidArgument);
        rows *= shape[i];
    }

    const int64_t row_elems = shape.back();
    auto row_bytes = q8_0_row_bytes(row_elems);
    if (!row_bytes) return std::unexpected(row_bytes.error());
    return rows * *row_bytes;
}

}  // namespace ccinfer
