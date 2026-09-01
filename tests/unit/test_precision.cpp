#include <gtest/gtest.h>

#include "runtime/precision.h"

namespace ccinfer {
namespace {

static_assert(execution_traits_valid_v<Qwen3ExecutionTraits>);
static_assert(execution_traits_valid_v<Qwen3_5ExecutionTraits>);
static_assert(!execution_traits_valid_v<ExecutionTraits>);

TEST(PrecisionTest, Qwen3TraitsValues) {
    EXPECT_EQ(Qwen3ExecutionTraits::activation_dtype, ccop::DType::kBFloat16);
    EXPECT_EQ(Qwen3ExecutionTraits::kv_dtype, ccop::DType::kBFloat16);
    EXPECT_EQ(Qwen3ExecutionTraits::accum_dtype, ccop::DType::kFloat32);
    EXPECT_EQ(Qwen3ExecutionTraits::logits_dtype, ccop::DType::kFloat32);
    EXPECT_EQ(Qwen3ExecutionTraits::recurrent_state_dtype, ccop::DType::kFloat32);
    EXPECT_EQ(Qwen3ExecutionTraits::conv_state_dtype, ccop::DType::kBFloat16);
}

TEST(PrecisionTest, Qwen35TraitsValues) {
    EXPECT_EQ(Qwen3_5ExecutionTraits::activation_dtype, ccop::DType::kBFloat16);
    EXPECT_EQ(Qwen3_5ExecutionTraits::kv_dtype, ccop::DType::kBFloat16);
    EXPECT_EQ(Qwen3_5ExecutionTraits::accum_dtype, ccop::DType::kFloat32);
    EXPECT_EQ(Qwen3_5ExecutionTraits::logits_dtype, ccop::DType::kFloat32);
    EXPECT_EQ(Qwen3_5ExecutionTraits::recurrent_state_dtype, ccop::DType::kFloat32);
    EXPECT_EQ(Qwen3_5ExecutionTraits::conv_state_dtype, ccop::DType::kBFloat16);
}

TEST(PrecisionTest, BaseIsIncompleteByDesign) {
    // The base ExecutionTraits intentionally has kUnknown values; concrete model
    // traits must override all fields.
    EXPECT_EQ(ExecutionTraits::activation_dtype, ccop::DType::kUnknown);
    EXPECT_EQ(ExecutionTraits::kv_dtype, ccop::DType::kUnknown);
    EXPECT_EQ(ExecutionTraits::accum_dtype, ccop::DType::kUnknown);
    EXPECT_EQ(ExecutionTraits::logits_dtype, ccop::DType::kUnknown);
    EXPECT_EQ(ExecutionTraits::recurrent_state_dtype, ccop::DType::kUnknown);
    EXPECT_EQ(ExecutionTraits::conv_state_dtype, ccop::DType::kUnknown);
}

}  // namespace
}  // namespace ccinfer
