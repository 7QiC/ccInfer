#include "quant/q8_0/q8_0_reference.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace ccinfer {

namespace {

float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10) & 0x1Fu;
    const std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x3FFu;

    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            std::uint32_t m = mant;
            int shift = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++shift;
            }
            const std::uint32_t float_exp = static_cast<std::uint32_t>(127 - 14 - shift);
            bits = sign | (float_exp << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        const std::uint32_t float_exp = exp + (127 - 15);
        bits = sign | (float_exp << 23) | (mant << 13);
    }

    return std::bit_cast<float>(bits);
}

std::uint16_t float_to_half(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp = (bits >> 23) & 0xFFu;
    const std::uint32_t mant = bits & 0x7FFFFFu;

    if (exp == 0xFFu) {
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }

    const std::int32_t half_exp = static_cast<std::int32_t>(exp) - 127 + 15;
    if (half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);

    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t m = mant | 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - half_exp);
        std::uint32_t half_mant = m >> shift;
        const std::uint32_t round_bit_mask = 1u << (shift - 1);
        if (m & round_bit_mask) {
            const std::uint32_t lower = m & (round_bit_mask - 1);
            if (lower != 0 || (half_mant & 1u)) ++half_mant;
        }
        if (half_mant >= 0x400u) return static_cast<std::uint16_t>(sign | (1u << 10));
        return static_cast<std::uint16_t>(sign | half_mant);
    }

    std::uint32_t half_mant = mant >> 13;
    const std::uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant >= 0x400u) {
            half_mant = 0;
            const std::uint32_t next_exp = static_cast<std::uint32_t>(half_exp + 1);
            if (next_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
            return static_cast<std::uint16_t>(sign | (next_exp << 10) | half_mant);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exp) << 10) |
                                      half_mant);
}

}  // namespace

float q8_0_f16_to_float(std::uint16_t value) noexcept { return half_to_float(value); }

std::uint16_t q8_0_float_to_f16(float value) noexcept { return float_to_half(value); }

Result<void> dequantize_q8_0_reference(std::span<const ccop::Q8_0Block> src,
                                       std::span<float> dst) noexcept {
    if (dst.size() != src.size() * ccop::kQ8_0BlockSize) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    for (std::size_t block = 0; block < src.size(); ++block) {
        const float scale = q8_0_f16_to_float(src[block].d_f16);
        for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
            dst[block * ccop::kQ8_0BlockSize + static_cast<std::size_t>(i)] =
                scale * static_cast<float>(src[block].qs[i]);
        }
    }
    return {};
}

Result<void> quantize_q8_0_reference(std::span<const float> src,
                                     std::span<ccop::Q8_0Block> dst) noexcept {
    if (src.empty() || src.size() % ccop::kQ8_0BlockSize != 0 ||
        dst.size() != src.size() / ccop::kQ8_0BlockSize) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    for (std::size_t block = 0; block < dst.size(); ++block) {
        const auto* src_block = src.data() + block * ccop::kQ8_0BlockSize;
        float max_abs = 0.0f;
        for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
            max_abs = std::max(max_abs, std::fabs(src_block[i]));
        }

        float scale = 0.0f;
        if (max_abs > 0.0f) scale = max_abs / 127.0f;

        ccop::Q8_0Block out;
        out.d_f16 = q8_0_float_to_f16(scale);
        const float actual_scale = q8_0_f16_to_float(out.d_f16);

        for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
            float q = actual_scale != 0.0f ? std::round(src_block[i] / actual_scale) : 0.0f;
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            out.qs[i] = static_cast<std::int8_t>(q);
        }
        dst[block] = out;
    }
    return {};
}

}  // namespace ccinfer
