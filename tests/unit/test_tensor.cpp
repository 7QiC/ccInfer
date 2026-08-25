#include <gtest/gtest.h>

#include "backend/backend.h"
#include "core/tensor.h"

using namespace ccinfer;

TEST(TensorTest, EmptyAllocatesAndOwnsBuffer) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto t_r = Tensor::empty(backend, ccop::DType::kFloat32, {2, 3, 4});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    EXPECT_TRUE(t.valid());
    EXPECT_EQ(t.dtype(), ccop::DType::kFloat32);
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 3);
    EXPECT_EQ(t.shape(2), 4);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.nbytes(), 24 * sizeof(float));
    EXPECT_TRUE(t.is_contiguous());
    ASSERT_NE(t.buffer(), nullptr);
    EXPECT_EQ(t.data(), t.buffer()->data());
}

TEST(TensorTest, CopySharesOwnerAndView) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto t_r = Tensor::empty(backend, ccop::DType::kFloat32, {10});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    Tensor copy = t;
    EXPECT_EQ(copy.data(), t.data());
    EXPECT_EQ(copy.buffer(), t.buffer());
    EXPECT_EQ(copy.numel(), 10);
}

TEST(TensorTest, FlatSharesOwner) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto t_r = Tensor::empty(backend, ccop::DType::kFloat32, {2, 3, 4});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    Tensor flat = t.flat();
    EXPECT_EQ(flat.data(), t.data());
    EXPECT_EQ(flat.buffer(), t.buffer());
    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat.shape(0), 24);
}

TEST(TensorTest, FlatAndViewKeepOffsetViewDataPointer) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto buffer_r = backend.allocate_buffer(10 * sizeof(float));
    ASSERT_TRUE(buffer_r.has_value());
    auto buffer = std::move(*buffer_r);
    void* offset_data = static_cast<char*>(buffer->data()) + 2 * sizeof(float);

    Tensor t = Tensor::from_buffer(buffer, offset_data, ccop::DType::kFloat32, {2, 4});
    Tensor flat = t.flat();
    Tensor viewed = t.view({2, 4});

    EXPECT_EQ(flat.data(), t.data());
    EXPECT_EQ(viewed.data(), t.data());
    EXPECT_NE(t.data(), t.buffer()->data());
    EXPECT_EQ(flat.numel(), 8);
    EXPECT_EQ(viewed.numel(), 8);
}

TEST(TensorTest, SliceAdjustsViewOnly) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto t_r = Tensor::empty(backend, ccop::DType::kFloat32, {2, 3, 4});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    Tensor s = t.slice(0, 1, 2);
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s.shape(0), 1);
    EXPECT_EQ(s.shape(1), 3);
    EXPECT_EQ(s.shape(2), 4);
    EXPECT_EQ(s.buffer(), t.buffer());
    EXPECT_NE(s.data(), t.data());
}

TEST(TensorTest, SelectAdjustsViewOnly) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto t_r = Tensor::empty(backend, ccop::DType::kFloat32, {2, 3, 4});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    Tensor s = t.select(0, 1);
    EXPECT_EQ(s.rank(), 2);
    EXPECT_EQ(s.shape(0), 3);
    EXPECT_EQ(s.shape(1), 4);
    EXPECT_EQ(s.buffer(), t.buffer());
    EXPECT_NE(s.data(), t.data());
}
