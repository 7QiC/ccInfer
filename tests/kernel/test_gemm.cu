#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "backend/cuda/cuda_backend.h"

using namespace ccinfer;

// C = A @ B  with A[M][K], B[K][N], C[M][N] in row-major.
// Verifies cuBLAS produces the correct result via the column-major reinterpretation.

class GemmTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto b = CudaBackend::create(0);
        ASSERT_TRUE(b.has_value());
        backend_ = std::move(*b);
    }

    std::unique_ptr<CudaBackend> backend_;
};

TEST_F(GemmTest, SmallMatrices2x3Times3x2) {
    constexpr int M = 2;
    constexpr int K = 3;
    constexpr int N = 2;

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7,  8],
    //      [9,  10],
    //      [11, 12]]
    // C = A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12],
    //              [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //   = [[58,  64],
    //      [139, 154]]
    std::vector<float> a_host = {1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f};
    std::vector<float> b_host = {7.0f, 8.0f,
                                  9.0f, 10.0f,
                                  11.0f, 12.0f};
    float expected[4] = {58.0f, 64.0f, 139.0f, 154.0f};

    // Convert to bf16
    std::vector<__nv_bfloat16> a_bf16(M * K);
    std::vector<__nv_bfloat16> b_bf16(K * N);
    for (int i = 0; i < M * K; ++i) a_bf16[i] = __float2bfloat16(a_host[i]);
    for (int i = 0; i < K * N; ++i) b_bf16[i] = __float2bfloat16(b_host[i]);

    // Upload to GPU
    __nv_bfloat16 *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&b_d, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&c_d, M * N * sizeof(__nv_bfloat16));
    cudaMemcpy(a_d, a_bf16.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_bf16.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Call gemm through the backend public API (row-major convention).
    GemmParams p;
    p.a_ = a_d;
    p.b_ = b_d;
    p.c_ = c_d;
    p.m_ = M;
    p.n_ = N;
    p.k_ = K;
    p.lda_ = K;  // A_row[M][K] has K columns
    p.ldb_ = N;  // B_row[K][N] has N columns
    p.ldc_ = N;  // C_row[M][N] has N columns
    auto r = backend_->template gemm<__nv_bfloat16>(p);
    ASSERT_TRUE(r.has_value());

    // Download and verify.
    std::vector<__nv_bfloat16> c_bf16(M * N);
    cudaMemcpy(c_bf16.data(), c_d, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; ++i) {
        float got = __bfloat162float(c_bf16[i]);
        EXPECT_NEAR(got, expected[i], 0.5f) << " at index " << i;
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
