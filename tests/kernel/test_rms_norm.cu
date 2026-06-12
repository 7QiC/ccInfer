#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernel/cuda_kernels.h"

using namespace ccinfer;

// CPU reference
static std::vector<__nv_bfloat16> rms_norm_cpu(const __nv_bfloat16* input, const __nv_bfloat16* weight, int rows, int dim,
                                      float eps) {
    std::vector<__nv_bfloat16> out(rows * dim);
    for (int r = 0; r < rows; r++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float v = __bfloat162float(input[r * dim + d]);
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
        for (int d = 0; d < dim; d++) {
            float v = __bfloat162float(input[r * dim + d]) * inv_rms;
            float w = __bfloat162float(weight[d]);
            out[r * dim + d] = __float2bfloat16(v * w);
        }
    }
    return out;
}

class RmsNormTest : public ::testing::Test {
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

TEST_F(RmsNormTest, SingleRow) {
    constexpr int rows = 1, dim = 64;
    std::vector<__nv_bfloat16> input_h(rows * dim);
    std::vector<__nv_bfloat16> weight_h(dim);
    for (int i = 0; i < dim; i++) {
        input_h[i] = __float2bfloat16(1.0f);
        weight_h[i] = __float2bfloat16(1.0f);
    }

    __nv_bfloat16 *in_d, *w_d, *out_d;
    cudaMalloc(&in_d, rows * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&w_d, dim * sizeof(__nv_bfloat16));
    cudaMalloc(&out_d, rows * dim * sizeof(__nv_bfloat16));
    cudaMemcpy(in_d, input_h.data(), rows * dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, weight_h.data(), dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_rms_norm(in_d, w_d, out_d, rows, dim, 1e-5f, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<__nv_bfloat16> out_h(rows * dim);
    cudaMemcpy(out_h.data(), out_d, rows * dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    auto expected = rms_norm_cpu(input_h.data(), weight_h.data(), rows, dim, 1e-5f);
    for (int i = 0; i < rows * dim; i++) {
        EXPECT_NEAR(__bfloat162float(out_h[i]), __bfloat162float(expected[i]), 0.01f);
    }

    cudaFree(in_d);
    cudaFree(w_d);
    cudaFree(out_d);
}

TEST_F(RmsNormTest, MultipleRows) {
    constexpr int rows = 4, dim = 128;
    std::vector<__nv_bfloat16> input_h(rows * dim);
    std::vector<__nv_bfloat16> weight_h(dim);
    for (int i = 0; i < rows * dim; i++) {
        input_h[i] = __float2bfloat16(static_cast<float>(i % 7 - 3));
    }
    for (int i = 0; i < dim; i++) {
        weight_h[i] = __float2bfloat16(static_cast<float>(i % 5 + 1) / 3.0f);
    }

    __nv_bfloat16 *in_d, *w_d, *out_d;
    cudaMalloc(&in_d, rows * dim * sizeof(__nv_bfloat16));
    cudaMalloc(&w_d, dim * sizeof(__nv_bfloat16));
    cudaMalloc(&out_d, rows * dim * sizeof(__nv_bfloat16));
    cudaMemcpy(in_d, input_h.data(), rows * dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, weight_h.data(), dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_rms_norm(in_d, w_d, out_d, rows, dim, 1e-5f, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<__nv_bfloat16> out_h(rows * dim);
    cudaMemcpy(out_h.data(), out_d, rows * dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    auto expected = rms_norm_cpu(input_h.data(), weight_h.data(), rows, dim, 1e-5f);
    for (int i = 0; i < rows * dim; i++) {
        EXPECT_NEAR(__bfloat162float(out_h[i]), __bfloat162float(expected[i]), 0.02f);
    }

    cudaFree(in_d);
    cudaFree(w_d);
    cudaFree(out_d);
}
