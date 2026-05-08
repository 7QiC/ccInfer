#include "engine/backend/cuda/cuda_backend.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdlib>

#include "common/result.h"
#include "cuda_utils.h"
#include "engine/backend/cuda/cuda_buffer.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {

CudaBackend::CudaBackend() {
    if (auto r = cublas_check(cublasCreate(&cublas_handle_)); !r) std::abort();
    if (auto r = cuda_check(cudaStreamCreate(&stream_)); !r) std::abort();
    if (auto r = cublas_check(cublasSetStream(cublas_handle_, stream_)); !r) std::abort();
}

CudaBackend::~CudaBackend() {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

template <>
Result<void> CudaBackend::gemm_impl<__nv_bfloat16>(const GemmParams& p) {
    if (auto r = cublas_check(cublasSetStream(cublas_handle_, stream_)); !r) {
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

template <>
Result<void> CudaBackend::rms_norm_impl<__nv_bfloat16>(const RmsNormParams& p) {
    return launch_rms_norm(
        static_cast<const __nv_bfloat16*>(p.input_), static_cast<const __nv_bfloat16*>(p.weight_),
        static_cast<__nv_bfloat16*>(p.output_), p.rows_, p.dim_, p.eps_, stream_);
}

template <>
Result<void> CudaBackend::rope_impl<__nv_bfloat16>(const RopeParams& p) {
    return launch_rope(static_cast<__nv_bfloat16*>(p.q_), static_cast<__nv_bfloat16*>(p.k_),
                       p.positions_, static_cast<const float2*>(p.rope_cache_), p.num_tokens_,
                       p.num_q_heads_, p.num_kv_heads_, p.head_dim_, p.rotary_dim_, p.max_position_,
                       stream_);
}

template <>
Result<void> CudaBackend::silu_mul_impl<__nv_bfloat16>(const SiluMulParams& p) {
    return launch_silu_mul(static_cast<const __nv_bfloat16*>(p.gate_),
                           static_cast<const __nv_bfloat16*>(p.up_),
                           static_cast<__nv_bfloat16*>(p.output_), p.n_, stream_);
}

template <>
Result<void> CudaBackend::element_add_impl<__nv_bfloat16>(const ElementAddParams& p) {
    return launch_element_add(static_cast<__nv_bfloat16*>(p.dst_),
                              static_cast<const __nv_bfloat16*>(p.src_), p.n_, stream_);
}

template <>
Result<void> CudaBackend::split_qkv_impl<__nv_bfloat16>(const SplitQkvParams& p) {
    return launch_split_qkv(static_cast<const __nv_bfloat16*>(p.qkv_),
                            static_cast<__nv_bfloat16*>(p.q_), static_cast<__nv_bfloat16*>(p.k_),
                            static_cast<__nv_bfloat16*>(p.v_), p.num_tokens_, p.num_q_heads_,
                            p.num_kv_heads_, p.head_dim_, stream_);
}

template <>
Result<void> CudaBackend::naive_attention_impl<__nv_bfloat16>(const NaiveAttnParams& p) {
    return launch_naive_attention(
        static_cast<const __nv_bfloat16*>(p.q_), static_cast<const __nv_bfloat16*>(p.k_),
        static_cast<const __nv_bfloat16*>(p.v_), static_cast<__nv_bfloat16*>(p.output_),
        p.num_tokens_, p.num_q_heads_, p.num_kv_heads_, p.head_dim_, stream_);
}

template <>
Result<void> CudaBackend::prefill_attention_impl<__nv_bfloat16>(const PrefillAttnParams& p) {
    return launch_prefill_attention(
        static_cast<const __nv_bfloat16*>(p.q_), static_cast<const __nv_bfloat16*>(p.k_cache_),
        static_cast<const __nv_bfloat16*>(p.v_cache_), p.block_table_, p.query_start_loc_,
        p.context_lens_, static_cast<__nv_bfloat16*>(p.output_), p.batch_size_,
        p.query_start_loc_[p.batch_size_], p.max_blocks_per_req_, p.num_q_heads_, p.num_kv_heads_,
        p.head_dim_, p.cache_block_size_, stream_);
}

template <>
Result<void> CudaBackend::decode_attention_impl<__nv_bfloat16>(const DecodeAttnParams& p) {
    return launch_decode_attention(
        static_cast<const __nv_bfloat16*>(p.q_), static_cast<const __nv_bfloat16*>(p.k_cache_),
        static_cast<const __nv_bfloat16*>(p.v_cache_), p.block_table_, p.context_lens_,
        static_cast<__nv_bfloat16*>(p.output_), p.batch_size_, p.max_blocks_per_req_,
        p.num_q_heads_, p.num_kv_heads_, p.head_dim_, p.cache_block_size_, stream_);
}

template <>
Result<void> CudaBackend::write_kv_cache_impl<__nv_bfloat16>(const WriteKVCacheParams& p) {
    return launch_write_kv_cache(
        static_cast<const __nv_bfloat16*>(p.k_new_), static_cast<const __nv_bfloat16*>(p.v_new_),
        static_cast<__nv_bfloat16*>(p.k_cache_), static_cast<__nv_bfloat16*>(p.v_cache_),
        p.slot_mapping_, p.total_tokens_, p.num_kv_heads_, p.head_dim_, p.max_slots_, stream_);
}

std::unique_ptr<DeviceBuffer> CudaBackend::allocate_buffer_impl(size_t bytes) {
    return std::make_unique<CudaBuffer>(bytes);
}

Result<void> CudaBackend::memcpy_h2d_impl(void* dst, const void* src, size_t count) {
    return cuda_check(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

Result<void> CudaBackend::memcpy_d2h_impl(void* dst, const void* src, size_t count) {
    return cuda_check(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

Result<void> CudaBackend::memcpy_d2d_impl(void* dst, const void* src, size_t count) {
    return cuda_check(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}

void* CudaBackend::stream_impl() {
    return stream_;
}

}  // namespace engine
}  // namespace ccinfer
