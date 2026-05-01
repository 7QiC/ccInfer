#include "engine/backend/cuda/cuda_backend.h"

#include <cuda_bf16.h>

#include <cstdlib>

#include "common/result.h"
#include "cuda_utils.h"
#include "engine/kernel/attention/naive_attn.h"
#include "engine/kernel/mlp/silu_mul.h"
#include "engine/kernel/norm/rms_norm.h"
#include "engine/kernel/pos/rope.h"

namespace ccinfer {
namespace engine {

CudaBackend::CudaBackend() {
    if (auto r = cublas_check(cublasCreate(&cublas_handle_)); !r) {
        std::abort();
    }

    if (auto r = cuda_check(cudaStreamCreate(&stream_)); !r) {
        std::abort();
    }

    if (auto r = cublas_check(cublasSetStream(cublas_handle_, stream_)); !r) {
        std::abort();
    }
}

CudaBackend::~CudaBackend() {
    if (cublas_handle_ != nullptr) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }

    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

Result<void> CudaBackend::gemm(const GemmParams& p) {
    cudaStream_t s = p.stream_ ? static_cast<cudaStream_t>(p.stream_) : stream_;

    if (p.a_ == nullptr || p.b_ == nullptr || p.c_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (p.m_ <= 0 || p.n_ <= 0 || p.k_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (auto r = cublas_check(cublasSetStream(cublas_handle_, s)); !r) {
        return r;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    return cublas_check(cublasGemmEx(cublas_handle_, p.trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
                                     p.trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, p.m_, p.n_, p.k_,
                                     &alpha, p.a_, CUDA_R_16BF, p.lda_, p.b_, CUDA_R_16BF, p.ldb_,
                                     &beta, p.c_, CUDA_R_16BF, p.ldc_, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

Result<void> CudaBackend::rms_norm(const RmsNormParams& p) {
    cudaStream_t s = p.stream_ ? static_cast<cudaStream_t>(p.stream_) : stream_;

    return launch_rms_norm(static_cast<const __nv_bfloat16*>(p.input_),
                           static_cast<const __nv_bfloat16*>(p.weight_),
                           static_cast<__nv_bfloat16*>(p.output_), p.rows_, p.dim_, p.eps_, s);
}

Result<void> CudaBackend::rope(const RopeParams& p) {
    cudaStream_t s = p.stream_ ? static_cast<cudaStream_t>(p.stream_) : stream_;

    return launch_rope(static_cast<__nv_bfloat16*>(p.q_), static_cast<__nv_bfloat16*>(p.k_),
                       p.positions_, static_cast<const float2*>(p.rope_cache_), p.num_tokens_,
                       p.num_q_heads_, p.num_kv_heads_, p.head_dim_, p.rotary_dim_, p.max_position_,
                       s);
}

Result<void> CudaBackend::silu_mul(const SiluMulParams& p) {
    cudaStream_t s = p.stream_ ? static_cast<cudaStream_t>(p.stream_) : stream_;

    return launch_silu_mul(static_cast<const __nv_bfloat16*>(p.gate_),
                           static_cast<const __nv_bfloat16*>(p.up_),
                           static_cast<__nv_bfloat16*>(p.output_), p.n_, s);
}

Result<void> CudaBackend::naive_attention(const NaiveAttnParams& p) {
    cudaStream_t s = p.stream_ ? static_cast<cudaStream_t>(p.stream_) : stream_;

    return ::ccinfer::engine::naive_attention(
        static_cast<const __nv_bfloat16*>(p.q_), static_cast<const __nv_bfloat16*>(p.k_),
        static_cast<const __nv_bfloat16*>(p.v_), static_cast<__nv_bfloat16*>(p.output_),
        p.num_tokens_, p.num_q_heads_, p.num_kv_heads_, p.head_dim_, s);
}

std::unique_ptr<DeviceBackend> DeviceBackend::create() { return std::make_unique<CudaBackend>(); }

}  // namespace engine
}  // namespace ccinfer
