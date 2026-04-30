#include "engine/backend/cuda/cuda_backend.h"

#include <cuda_fp16.h>

#include <cstdlib>

#include "common/result.h"
#include "cuda_utils.h"
#include "engine/kernel/mlp/silu_mul.h"
#include "engine/kernel/norm/rms_norm.h"
#include "engine/kernel/pos/rope.h"

namespace ccinfer {
namespace engine {

CudaBackend::CudaBackend() {
    if (auto r = cublas_check(cublasCreate(&cublas_handle_)); !r) abort();
    if (auto r = cuda_check(cudaStreamCreate(&stream_)); !r) abort();
}

CudaBackend::~CudaBackend() {
    cublasDestroy(cublas_handle_);
    cudaStreamDestroy(stream_);
}

Result<void> CudaBackend::gemm(const GemmParams& p) {
    auto s = static_cast<cudaStream_t>(p.stream_);
    if (auto r = cublas_check(cublasSetStream(cublas_handle_, s)); !r) return r;

    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    return cublas_check(cublasGemmEx(
        cublas_handle_, p.trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
        p.trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, p.n_, p.m_, p.k_, &alpha, p.b_, CUDA_R_16F,
        p.ldb_, p.a_, CUDA_R_16F, p.lda_, &beta, p.c_, CUDA_R_16F, p.ldc_, CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

Result<void> CudaBackend::rms_norm(const RmsNormParams& p) {
    return launch_rms_norm(static_cast<const half*>(p.input_),
                           static_cast<const half*>(p.weight_),
                           static_cast<half*>(p.output_), p.rows_, p.dim_, p.eps_,
                           static_cast<cudaStream_t>(p.stream_));
}

Result<void> CudaBackend::rope(const RopeParams& p) {
    return launch_rope(static_cast<half*>(p.q_), static_cast<half*>(p.k_), p.positions_,
                       p.rope_cache_, p.num_tokens_, p.num_q_heads_, p.num_kv_heads_,
                       p.head_dim_, p.rotary_dim_, p.max_position_,
                       static_cast<cudaStream_t>(p.stream_));
}

Result<void> CudaBackend::silu_mul(const SiluMulParams& p) {
    return launch_silu_mul(static_cast<const half*>(p.gate_), static_cast<const half*>(p.up_),
                           static_cast<half*>(p.output_), p.n_,
                           static_cast<cudaStream_t>(p.stream_));
}

std::unique_ptr<DeviceBackend> DeviceBackend::create() { return std::make_unique<CudaBackend>(); }

}  // namespace engine
}  // namespace ccinfer
