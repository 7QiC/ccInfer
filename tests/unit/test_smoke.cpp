#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "backend/backend.h"
#include "base/result.h"
#include "core/tensor.h"
#include "model/config.h"

using namespace ccinfer;

TEST(SmokeTest, GPUAccessible) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    ASSERT_EQ(err, cudaSuccess);
}

TEST(SmokeTest, AllocateAndZero) {
    auto backend_r = Backend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto buf_r = backend.allocate_buffer(256 * sizeof(float));
    ASSERT_TRUE(buf_r.has_value());
    auto buf = std::move(*buf_r);
    ASSERT_NE(buf->data(), nullptr);

    cudaMemset(buf->data(), 0, 256 * sizeof(float));
    std::vector<float> host(256, 42.0f);
    cudaMemcpy(host.data(), buf->data(), 256 * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 256; i++) {
        EXPECT_EQ(host[i], 0.0f);
    }
}

TEST(SmokeTest, TensorSmoke) {
    auto backend_r = Backend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto t_r = Tensor::empty(backend, ccop::DType::kFloat32, {2, 3});
    ASSERT_TRUE(t_r.has_value());
    Tensor t = std::move(*t_r);
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_NE(t.data(), nullptr);
}
