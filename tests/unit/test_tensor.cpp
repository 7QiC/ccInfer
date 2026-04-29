#include <gtest/gtest.h>

#include "core/tensor.h"

using namespace ccinfer;
using namespace ccinfer::engine;

TEST(TensorTest, MakeAndShape) {
    float data[24] = {};
    auto t = Tensor<>::make(data, DType::kFloat32, {2, 3, 4});
    EXPECT_EQ(t.rank, 3);
    EXPECT_EQ(t.shape[0], 2);
    EXPECT_EQ(t.shape[1], 3);
    EXPECT_EQ(t.shape[2], 4);
    EXPECT_EQ(t.numel(), 24);
}

TEST(TensorTest, DefaultStrideRowMajor) {
    float data[24] = {};
    auto t = Tensor<>::make(data, DType::kFloat32, {2, 3, 4});
    EXPECT_EQ(t.stride[2], 1);
    EXPECT_EQ(t.stride[1], 4);
    EXPECT_EQ(t.stride[0], 12);
}

TEST(TensorTest, Nbytes) {
    float data[10] = {};
    auto t = Tensor<>::make(data, DType::kFloat32, {10});
    EXPECT_EQ(t.nbytes(), 40);

    int8_t i8[8] = {};
    auto t2 = Tensor<>::make(i8, DType::kInt8, {8});
    EXPECT_EQ(t2.nbytes(), 8);
}

TEST(TensorTest, Slice) {
    float data[24] = {};
    auto t = Tensor<>::make(data, DType::kFloat32, {2, 3, 4});
    auto s = t.slice(0, 1, 2);
    EXPECT_EQ(s.rank, 3);
    EXPECT_EQ(s.shape[0], 1);
    EXPECT_EQ(s.shape[1], 3);
    EXPECT_EQ(s.shape[2], 4);
    EXPECT_EQ(static_cast<float*>(s.data), data + 12);
}

TEST(TensorTest, Select) {
    float data[24] = {};
    auto t = Tensor<>::make(data, DType::kFloat32, {2, 3, 4});
    auto s = t.select(0, 1);
    EXPECT_EQ(s.rank, 2);
    EXPECT_EQ(s.shape[0], 3);
    EXPECT_EQ(s.shape[1], 4);
    EXPECT_EQ(static_cast<float*>(s.data), data + 12);
}
