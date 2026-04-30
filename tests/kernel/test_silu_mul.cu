#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "engine/kernel/mlp/silu_mul.h"

using namespace ccinfer;
using namespace ccinfer::engine;

static std::vector<half> silu_mul_cpu(const half* gate, const half* up, int64_t n) {
    std::vector<half> out(n);
    for (int64_t i = 0; i < n; i++) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = __float2half_rn(silu * u);
    }
    return out;
}

class SiluMulTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_{};
};

TEST_F(SiluMulTest, EvenElements) {
    constexpr int64_t n = 256;
    std::vector<half> gate_h(n), up_h(n);
    for (int i = 0; i < n; i++) {
        gate_h[i] = __float2half((float)(i % 7 - 3));
        up_h[i] = __float2half((float)(i % 5 + 1) / 3.0f);
    }
    auto expected = silu_mul_cpu(gate_h.data(), up_h.data(), n);

    half *gate_d, *up_d, *out_d;
    cudaMalloc(&gate_d, n * sizeof(half));
    cudaMalloc(&up_d, n * sizeof(half));
    cudaMalloc(&out_d, n * sizeof(half));
    cudaMemcpy(gate_d, gate_h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(up_d, up_h.data(), n * sizeof(half), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_silu_mul(gate_d, up_d, out_d, n, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<half> out_h(n);
    cudaMemcpy(out_h.data(), out_d, n * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(__half2float(out_h[i]), __half2float(expected[i]), 0.02f);

    cudaFree(gate_d); cudaFree(up_d); cudaFree(out_d);
}

TEST_F(SiluMulTest, OddElements) {
    constexpr int64_t n = 127;
    std::vector<half> gate_h(n), up_h(n);
    for (int i = 0; i < n; i++) {
        gate_h[i] = __float2half((float)(i % 11 - 5));
        up_h[i] = __float2half((float)(i % 3));
    }
    auto expected = silu_mul_cpu(gate_h.data(), up_h.data(), n);

    half *gate_d, *up_d, *out_d;
    cudaMalloc(&gate_d, n * sizeof(half));
    cudaMalloc(&up_d, n * sizeof(half));
    cudaMalloc(&out_d, n * sizeof(half));
    cudaMemcpy(gate_d, gate_h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(up_d, up_h.data(), n * sizeof(half), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_silu_mul(gate_d, up_d, out_d, n, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<half> out_h(n);
    cudaMemcpy(out_h.data(), out_d, n * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(__half2float(out_h[i]), __half2float(expected[i]), 0.02f);

    cudaFree(gate_d); cudaFree(up_d); cudaFree(out_d);
}
