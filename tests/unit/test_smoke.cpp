#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "engine/backend/cuda/cuda_backend.h"
#include "engine/backend/device_buffer.h"
#include "core/tensor.h"
#include "engine/common/dtype.h"
#include "engine/model/config.h"
#include "common/result.h"

using namespace ccinfer;
using namespace ccinfer::engine;

TEST(SmokeTest, GPUAccessible) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    ASSERT_EQ(err, cudaSuccess);
}

TEST(SmokeTest, AllocateAndZero) {
    CudaBackend backend;
    ASSERT_TRUE(backend.init(0).has_value());
    auto buf = backend.allocate_buffer(256 * sizeof(float));
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
