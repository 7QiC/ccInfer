#include "quant/q8_0/q8_0_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ccinfer {

Result<void> dequantize_q8_0_reference(std::span<const Q8_0Block> src,
                                       std::span<float> dst) noexcept {
    if (dst.size() != src.size() * kQ8_0BlockSize) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    for (std::size_t block = 0; block < src.size(); ++block) {
        const float scale = block_scale(src[block]);
        for (int i = 0; i < kQ8_0BlockSize; ++i) {
            dst[block * kQ8_0BlockSize + static_cast<std::size_t>(i)] =
                scale * static_cast<float>(src[block].qs[i]);
        }
    }
    return {};
}

Result<void> quantize_q8_0_reference(std::span<const float> src,
                                     std::span<Q8_0Block> dst) noexcept {
    if (src.empty() || src.size() % kQ8_0BlockSize != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (dst.size() != src.size() / kQ8_0BlockSize) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    for (std::size_t block = 0; block < dst.size(); ++block) {
        const auto* src_block = src.data() + block * kQ8_0BlockSize;
        float max_abs = 0.0f;
        for (int i = 0; i < kQ8_0BlockSize; ++i) {
            max_abs = std::max(max_abs, std::fabs(src_block[i]));
        }

        float scale = 0.0f;
        if (max_abs > 0.0f) {
            scale = max_abs / 127.0f;
        }

        Q8_0Block out;
        out.d_f16 = q8_0_float_to_f16(scale);
        const float actual_scale = q8_0_f16_to_float(out.d_f16);

        for (int i = 0; i < kQ8_0BlockSize; ++i) {
            float q = actual_scale != 0.0f ? std::round(src_block[i] / actual_scale) : 0.0f;
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            out.qs[i] = static_cast<int8_t>(q);
        }
        dst[block] = out;
    }
    return {};
}

}  // namespace ccinfer
