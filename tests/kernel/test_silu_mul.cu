#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "kernel/cuda_kernels.h"

using namespace ccinfer;

static std::vector<__nv_bfloat16> silu_mul_cpu(const __nv_bfloat16* gate, const __nv_bfloat16* up, int64_t n) {
    std::vector<__nv_bfloat16> out(n);
    for (int64_t i = 0; i < n; i++) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = __float2bfloat16_rn(silu * u);
    }
    return out;
}

class SiluMulTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override {
        auto sync_err = cudaStreamSynchronize(stream_);
        ASSERT_EQ(sync_err, cudaSuccess);
        auto last_err = cudaGetLastError();
        ASSERT_EQ(last_err, cudaSuccess);
        cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_{};
};

TEST_F(SiluMulTest, EvenElements) {
    constexpr int64_t n = 256;
    std::vector<__nv_bfloat16> gate_h(n), up_h(n);
    for (int i = 0; i < n; i++) {
        gate_h[i] = __float2bfloat16((float)(i % 7 - 3));
        up_h[i] = __float2bfloat16((float)(i % 5 + 1) / 3.0f);
    }
    auto expected = silu_mul_cpu(gate_h.data(), up_h.data(), n);

    __nv_bfloat16 *gate_d, *up_d, *out_d;
    cudaMalloc(&gate_d, n * sizeof(__nv_bfloat16));
    cudaMalloc(&up_d, n * sizeof(__nv_bfloat16));
    cudaMalloc(&out_d, n * sizeof(__nv_bfloat16));
    cudaMemcpy(gate_d, gate_h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(up_d, up_h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_silu_mul(gate_d, up_d, out_d, n, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<__nv_bfloat16> out_h(n);
    cudaMemcpy(out_h.data(), out_d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(__bfloat162float(out_h[i]), __bfloat162float(expected[i]), 0.02f);

    cudaFree(gate_d); cudaFree(up_d); cudaFree(out_d);
}

TEST_F(SiluMulTest, OddElements) {
    constexpr int64_t n = 127;
    std::vector<__nv_bfloat16> gate_h(n), up_h(n);
    for (int i = 0; i < n; i++) {
        gate_h[i] = __float2bfloat16((float)(i % 11 - 5));
        up_h[i] = __float2bfloat16((float)(i % 3));
    }
    auto expected = silu_mul_cpu(gate_h.data(), up_h.data(), n);

    __nv_bfloat16 *gate_d, *up_d, *out_d;
    cudaMalloc(&gate_d, n * sizeof(__nv_bfloat16));
    cudaMalloc(&up_d, n * sizeof(__nv_bfloat16));
    cudaMalloc(&out_d, n * sizeof(__nv_bfloat16));
    cudaMemcpy(gate_d, gate_h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(up_d, up_h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_silu_mul(gate_d, up_d, out_d, n, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<__nv_bfloat16> out_h(n);
    cudaMemcpy(out_h.data(), out_d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(__bfloat162float(out_h[i]), __bfloat162float(expected[i]), 0.02f);

    cudaFree(gate_d); cudaFree(up_d); cudaFree(out_d);
}
