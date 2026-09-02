#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "model/weight.h"

namespace ccinfer {
namespace {

TEST(WeightTest, DenseWeightKindAndHelpers) {
    DenseWeight dw;
    Weight w = dw;

    EXPECT_EQ(weight_kind(w), WeightKind::kDense);
    EXPECT_TRUE(weight_logical_shape(w).empty());
    EXPECT_EQ(weight_byte_size(w), 0u);
}

TEST(WeightTest, QuantizedWeightKindAndHelpers) {
    QuantizedWeight qw;
    qw.logical_shape = {2, 32};
    qw.quant_type = ccop::QuantType::kQ8_0;
    Weight w = qw;

    EXPECT_EQ(weight_kind(w), WeightKind::kQuantized);
    EXPECT_EQ(weight_logical_shape(w), (std::vector<int64_t>{2, 32}));

    const std::vector<int64_t> shape{2, 32};
    const auto expected = ccop::q8_0_storage_bytes(shape);
    ASSERT_TRUE(expected.has_value());
    EXPECT_EQ(weight_byte_size(w), static_cast<uint64_t>(*expected));
}

TEST(WeightTest, Q8_0SizeIsNotDenseF32Size) {
    QuantizedWeight qw;
    qw.logical_shape = {2, 32};
    qw.quant_type = ccop::QuantType::kQ8_0;
    Weight w = qw;

    const uint64_t dense_f32_bytes = static_cast<uint64_t>(2 * 32) * sizeof(float);
    EXPECT_NE(weight_byte_size(w), dense_f32_bytes);
}

TEST(WeightTest, VariantAccess) {
    QuantizedWeight qw;
    qw.logical_shape = {4, 32};
    qw.quant_type = ccop::QuantType::kQ8_0;
    Weight w = qw;

    const auto* q = std::get_if<QuantizedWeight>(&w);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->logical_shape.size(), 2u);
    EXPECT_EQ(q->quant_type, ccop::QuantType::kQ8_0);
}

}  // namespace
}  // namespace ccinfer
