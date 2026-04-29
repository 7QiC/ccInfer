#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernel/norm/rms_norm.h"

using namespace ccinfer;
using namespace ccinfer::engine;

// CPU reference
static std::vector<half> rms_norm_cpu(const half* input, const half* weight, int rows, int dim,
                                      float eps) {
    std::vector<half> out(rows * dim);
    for (int r = 0; r < rows; r++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float v = __half2float(input[r * dim + d]);
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
        for (int d = 0; d < dim; d++) {
            float v = __half2float(input[r * dim + d]) * inv_rms;
            float w = __half2float(weight[d]);
            out[r * dim + d] = __float2half(v * w);
        }
    }
    return out;
}

class RmsNormTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }

    cudaStream_t stream_{};
};

TEST_F(RmsNormTest, SingleRow) {
    constexpr int rows = 1, dim = 64;
    std::vector<half> input_h(rows * dim);
    std::vector<half> weight_h(dim);
    for (int i = 0; i < dim; i++) {
        input_h[i] = __float2half(1.0f);
        weight_h[i] = __float2half(1.0f);
    }

    half *in_d, *w_d, *out_d;
    cudaMalloc(&in_d, rows * dim * sizeof(half));
    cudaMalloc(&w_d, dim * sizeof(half));
    cudaMalloc(&out_d, rows * dim * sizeof(half));
    cudaMemcpy(in_d, input_h.data(), rows * dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, weight_h.data(), dim * sizeof(half), cudaMemcpyHostToDevice);

    launch_rms_norm(in_d, w_d, out_d, rows, dim, 1e-5f, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> out_h(rows * dim);
    cudaMemcpy(out_h.data(), out_d, rows * dim * sizeof(half), cudaMemcpyDeviceToHost);

    auto expected = rms_norm_cpu(input_h.data(), weight_h.data(), rows, dim, 1e-5f);
    for (int i = 0; i < rows * dim; i++) {
        EXPECT_NEAR(__half2float(out_h[i]), __half2float(expected[i]), 0.01f);
    }

    cudaFree(in_d);
    cudaFree(w_d);
    cudaFree(out_d);
}

TEST_F(RmsNormTest, MultipleRows) {
    constexpr int rows = 4, dim = 128;
    std::vector<half> input_h(rows * dim);
    std::vector<half> weight_h(dim);
    for (int i = 0; i < rows * dim; i++) {
        input_h[i] = __float2half(static_cast<float>(i % 7 - 3));
    }
    for (int i = 0; i < dim; i++) {
        weight_h[i] = __float2half(static_cast<float>(i % 5 + 1) / 3.0f);
    }

    half *in_d, *w_d, *out_d;
    cudaMalloc(&in_d, rows * dim * sizeof(half));
    cudaMalloc(&w_d, dim * sizeof(half));
    cudaMalloc(&out_d, rows * dim * sizeof(half));
    cudaMemcpy(in_d, input_h.data(), rows * dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, weight_h.data(), dim * sizeof(half), cudaMemcpyHostToDevice);

    launch_rms_norm(in_d, w_d, out_d, rows, dim, 1e-5f, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> out_h(rows * dim);
    cudaMemcpy(out_h.data(), out_d, rows * dim * sizeof(half), cudaMemcpyDeviceToHost);

    auto expected = rms_norm_cpu(input_h.data(), weight_h.data(), rows, dim, 1e-5f);
    for (int i = 0; i < rows * dim; i++) {
        EXPECT_NEAR(__half2float(out_h[i]), __half2float(expected[i]), 0.02f);
    }

    cudaFree(in_d);
    cudaFree(w_d);
    cudaFree(out_d);
}
