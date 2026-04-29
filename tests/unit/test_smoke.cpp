#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "core/device_buffer.h"
#include "core/tensor.h"
#include "dtype.h"
#include "model/config.h"
#include "result.h"

using namespace ccinfer;
using namespace ccinfer::engine;

TEST(SmokeTest, GPUAccessible) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    ASSERT_EQ(err, cudaSuccess);
}

TEST(SmokeTest, AllocateAndZero) {
    DeviceBuffer<float> buf(256);
    ASSERT_NE(buf.get(), nullptr);

    cudaMemset(buf.get(), 0, 256 * sizeof(float));
    std::vector<float> host(256, 42.0f);
    cudaMemcpy(host.data(), buf.get(), 256 * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 256; i++) {
        EXPECT_EQ(host[i], 0.0f);
    }
}

TEST(SmokeTest, TensorSmoke) {
    float data[6] = {};
    auto t = Tensor<>::make(data, DType::kFloat32, {2, 3});
    EXPECT_EQ(t.rank, 2);
    EXPECT_EQ(t.numel(), 6);
}
