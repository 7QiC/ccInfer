#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "backend/device_buffer.h"
#include "core/tensor.h"
#include "core/dtype.h"
#include "model/config.h"
#include "base/result.h"

using namespace ccinfer;

TEST(SmokeTest, GPUAccessible) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    ASSERT_EQ(err, cudaSuccess);
}

TEST(SmokeTest, AllocateAndZero) {
    auto backend_r = CudaBackend::create(0);
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
    float data[6] = {};
    auto t = Tensor<>::make(data, DType::kFloat32, {2, 3});
    EXPECT_EQ(t.rank_, 2);
    EXPECT_EQ(t.numel(), 6);
}
