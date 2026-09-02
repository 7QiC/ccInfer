#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "ccop/quant.h"
#include "quant/q8_0/q8_0_reference.h"

namespace ccinfer {
namespace {

TEST(Q8_0Test, BlockLayoutAndSizes) {
    static_assert(sizeof(ccop::Q8_0Block) == ccop::kQ8_0BlockBytes);
    static_assert(ccop::kQ8_0BlockSize == 32);
    static_assert(offsetof(ccop::Q8_0Block, d_f16) == 0);
    static_assert(offsetof(ccop::Q8_0Block, qs) == 2);

    EXPECT_EQ(ccop::q8_0_row_bytes(32), ccop::kQ8_0BlockBytes);
    EXPECT_EQ(ccop::q8_0_row_bytes(64), 2 * ccop::kQ8_0BlockBytes);

    const std::vector<int64_t> shape2{2, 32};
    const std::vector<int64_t> shape3{3, 64};
    EXPECT_EQ(*ccop::q8_0_storage_bytes(shape2), 2 * ccop::kQ8_0BlockBytes);
    EXPECT_EQ(*ccop::q8_0_storage_bytes(shape3), 3 * 2 * ccop::kQ8_0BlockBytes);
}

TEST(Q8_0Test, RejectsNonBlockAlignedDimensions) {
    const std::vector<int64_t> bad_shape{2, 33};
    EXPECT_FALSE(ccop::q8_0_storage_bytes(bad_shape).has_value());
    const std::vector<int64_t> empty_shape{};
    EXPECT_FALSE(ccop::q8_0_storage_bytes(empty_shape).has_value());

    // Locked shape semantics: the LAST dimension is the block-aligned inner
    // dimension. {32, 2} is therefore invalid even though 32 is aligned.
    const std::vector<int64_t> reversed_shape{32, 2};
    EXPECT_FALSE(ccop::q8_0_storage_bytes(reversed_shape).has_value());
}

TEST(Q8_0Test, QuantTypeValues) {
    EXPECT_NE(ccop::QuantType::kUnknown, ccop::QuantType::kQ8_0);
    EXPECT_EQ(static_cast<int>(ccop::QuantType::kUnknown), 0);
    EXPECT_EQ(static_cast<int>(ccop::QuantType::kQ8_0), 1);
}

TEST(Q8_0Test, Fp16ConversionKnownValues) {
    EXPECT_EQ(q8_0_f16_to_float(0x3C00u), 1.0f);
    EXPECT_EQ(q8_0_f16_to_float(0xBC00u), -1.0f);
    EXPECT_EQ(q8_0_f16_to_float(0x0001u), 5.960464477539063e-8f);
    EXPECT_EQ(q8_0_f16_to_float(0x7BFFu), 65504.0f);

    EXPECT_EQ(q8_0_float_to_f16(0.0f), 0);
    EXPECT_EQ(q8_0_float_to_f16(-0.0f), 0x8000u);
    EXPECT_EQ(q8_0_float_to_f16(1.0f), 0x3C00u);
    EXPECT_EQ(q8_0_float_to_f16(-1.0f), 0xBC00u);
    EXPECT_EQ(q8_0_float_to_f16(65504.0f), 0x7BFFu);
    EXPECT_EQ(q8_0_float_to_f16(65520.0f), 0x7C00u);

    EXPECT_EQ(q8_0_float_to_f16(0.5f), 0x3800u);
    EXPECT_EQ(q8_0_float_to_f16(0.500244140625f), 0x3800u);
    EXPECT_EQ(q8_0_float_to_f16(0.500732421875f), 0x3802u);
}

TEST(Q8_0Test, DequantKnownRawBlock) {
    ccop::Q8_0Block block;
    block.d_f16 = 0x3C00;  // 1.0
    for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) block.qs[i] = static_cast<int8_t>(i + 1);

    std::vector<float> dst(ccop::kQ8_0BlockSize);
    auto r = dequantize_q8_0_reference(std::span<const ccop::Q8_0Block>(&block, 1), dst);
    ASSERT_TRUE(r.has_value());
    for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
        EXPECT_FLOAT_EQ(dst[static_cast<std::size_t>(i)], static_cast<float>(i + 1));
    }
}

TEST(Q8_0Test, QuantizeGoldenBlockBytes) {
    std::vector<float> src(ccop::kQ8_0BlockSize);
    for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
        src[static_cast<std::size_t>(i)] = static_cast<float>(i - 16);
    }

    std::vector<ccop::Q8_0Block> dst(1);
    auto r = quantize_q8_0_reference(src, dst);
    ASSERT_TRUE(r.has_value());

    const std::vector<int8_t> expected_qs = {
        -127, -119, -111, -103, -95, -87, -79, -71, -64, -56, -48, -40, -32, -24, -16, -8,
        0,    8,    16,   24,   32,   40,   48,   56,   64,   71,   79,   87,   95,   103,
        111,  119,
    };

    EXPECT_EQ(dst[0].d_f16, 0x3008u);
    for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
        EXPECT_EQ(dst[0].qs[i], expected_qs[static_cast<std::size_t>(i)]);
    }
}

TEST(Q8_0Test, AllZeroBlock) {
    std::vector<float> src(ccop::kQ8_0BlockSize, 0.0f);
    std::vector<ccop::Q8_0Block> dst(1);
    ASSERT_TRUE(quantize_q8_0_reference(src, dst).has_value());
    EXPECT_EQ(dst[0].d_f16, 0);
    for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) EXPECT_EQ(dst[0].qs[i], 0);

    std::vector<float> out(ccop::kQ8_0BlockSize, 1.0f);
    ASSERT_TRUE(dequantize_q8_0_reference(dst, out).has_value());
    for (float x : out) EXPECT_FLOAT_EQ(x, 0.0f);
}

TEST(Q8_0Test, PositiveAndNegativeExtremes) {
    std::vector<float> src(ccop::kQ8_0BlockSize);
    for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
        src[static_cast<std::size_t>(i)] = (i % 2 == 0) ? -127.0f : 127.0f;
    }
    std::vector<ccop::Q8_0Block> dst(1);
    ASSERT_TRUE(quantize_q8_0_reference(src, dst).has_value());
    std::vector<float> out(ccop::kQ8_0BlockSize);
    ASSERT_TRUE(dequantize_q8_0_reference(dst, out).has_value());

    for (int i = 0; i < ccop::kQ8_0BlockSize; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], src[static_cast<std::size_t>(i)], 0.02f);
    }
}

TEST(Q8_0Test, RejectsInvalidReferenceInputs) {
    std::vector<float> src(33, 1.0f);
    std::vector<ccop::Q8_0Block> dst(2);
    EXPECT_FALSE(quantize_q8_0_reference(src, dst).has_value());

    std::vector<float> good_src(32, 1.0f);
    EXPECT_FALSE(quantize_q8_0_reference(good_src, dst).has_value());

    ccop::Q8_0Block block{};
    std::vector<float> short_dst(31);
    EXPECT_FALSE(
        dequantize_q8_0_reference(std::span<const ccop::Q8_0Block>(&block, 1), short_dst).has_value());
}

TEST(Q8_0Test, RandomVectorErrorBound) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    constexpr int kLength = 256;
    std::vector<float> src(kLength);
    float max_abs = 0.0f;
    for (float& x : src) {
        x = dist(rng);
        max_abs = std::max(max_abs, std::fabs(x));
    }

    std::vector<ccop::Q8_0Block> blocks(kLength / ccop::kQ8_0BlockSize);
    std::vector<float> dst(kLength);
    ASSERT_TRUE(quantize_q8_0_reference(src, blocks).has_value());
    ASSERT_TRUE(dequantize_q8_0_reference(blocks, dst).has_value());

    const float bound = std::max(1.0f, max_abs) * 0.01f;
    for (std::size_t i = 0; i < src.size(); ++i) {
        ASSERT_NEAR(dst[i], src[i], bound);
    }
}

}  // namespace
}  // namespace ccinfer
