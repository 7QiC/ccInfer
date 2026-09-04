#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "backend/backend.h"
#include "runtime/tensor.h"

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

TEST(TensorTest, QuantizedEmptyAllocatesPhysicalBytes) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto t_r = Tensor::empty(backend, ccop::kQ8_0QType, {2, 32});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    EXPECT_TRUE(t.valid());
    EXPECT_TRUE(t.is_quantized());
    EXPECT_FALSE(t.is_dense());
    EXPECT_EQ(t.qtype().quant_type, ccop::QuantType::kQ8_0);
    EXPECT_EQ(t.qtype().data_type, ccop::DType::kInt8);
    EXPECT_EQ(t.qtype().scale_type, ccop::DType::kFloat16);
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 32);
    EXPECT_EQ(t.numel(), 64);
    EXPECT_EQ(t.nbytes(), 68u);
    EXPECT_EQ(t.buffer()->bytes(), 68u);
    EXPECT_EQ(t.data(), t.buffer()->data());
}

TEST(TensorTest, QuantizedFromHostUploadsPackedBytes) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    std::vector<ccop::Q8_0Block> host(2);
    host[0].d_f16 = 0x3C00;  // fp16 1.0
    host[1].d_f16 = 0x3C00;
    for (std::int8_t& q : host[0].qs) q = 1;
    for (std::int8_t& q : host[1].qs) q = 2;

    auto t_r = Tensor::from_host(backend, host.data(), ccop::kQ8_0QType, {2, 32});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    EXPECT_EQ(t.nbytes(), 68u);
    EXPECT_EQ(t.buffer()->bytes(), 68u);

    std::vector<ccop::Q8_0Block> back(2);
    auto r = backend.memcpy_d2h(back.data(), t.data(), 68u);
    ASSERT_TRUE(r.has_value());
    auto sync = backend.synchronize();
    ASSERT_TRUE(sync.has_value());
    EXPECT_EQ(back[0].d_f16, 0x3C00u);
    EXPECT_EQ(back[1].d_f16, 0x3C00u);
    EXPECT_EQ(back[0].qs[0], 1);
    EXPECT_EQ(back[1].qs[31], 2);
}

TEST(TensorTest, QuantizedCopySharesOwnerAndView) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto t_r = Tensor::empty(backend, ccop::kQ8_0QType, {2, 32});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    Tensor copy = t;
    EXPECT_EQ(copy.data(), t.data());
    EXPECT_EQ(copy.buffer(), t.buffer());
    EXPECT_EQ(copy.numel(), 64);
    EXPECT_TRUE(copy.is_quantized());
    EXPECT_EQ(copy.qtype().quant_type, ccop::QuantType::kQ8_0);
}
