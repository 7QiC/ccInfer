#include "engine/backend/cuda/cuda_backend.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdlib>
#include <utility>

#include "common/result.h"
#include "engine/backend/cuda/cuda_buffer.h"
#include "engine/backend/cuda/cuda_utils.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {

// ---------------------------------------------------------------------------
// create / dtor / init / reset
// ---------------------------------------------------------------------------

Result<std::unique_ptr<CudaBackend>> CudaBackend::create(int device_id) {
    auto backend = std::unique_ptr<CudaBackend>(new CudaBackend());
    auto r = backend->init(device_id);
    if (!r) return std::unexpected(r.error());
    return std::move(backend);
}

CudaBackend::~CudaBackend() { reset(); }

void CudaBackend::reset() noexcept {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

Result<void> CudaBackend::init(int device_id) {
    reset();

    if (auto r = cuda_check(cudaSetDevice(device_id)); !r) return r;
    if (auto r = cublas_check(cublasCreate(&cublas_handle_)); !r) {
        reset();
        return r;
    }
    if (auto r = cuda_check(cudaStreamCreate(&stream_)); !r) {
        reset();
        return r;
    }
    if (auto r = cublas_check(cublasSetStream(cublas_handle_, stream_)); !r) {
        reset();
        return r;
    }
    return {};
}

// ---------------------------------------------------------------------------
// embed
// ---------------------------------------------------------------------------

Result<void> CudaBackend::embed_impl(const EmbedParams& p) {
    return launch_embed(static_cast<const __nv_bfloat16*>(p.embed_table_), p.token_ids_,
                        static_cast<__nv_bfloat16*>(p.input_embeds_), p.num_tokens_, p.d_model_,
                        stream_);
}

// ---------------------------------------------------------------------------
// gemm (BF16)
// ---------------------------------------------------------------------------
// Row-major convention:
//   GemmParams describes C = A @ B in natural row-major:
//     A: [m_][k_] with leading dimension lda_ (≥ k_)
//     B: [k_][n_] with leading dimension ldb_ (≥ n_)
//     C: [m_][n_] with leading dimension ldc_ (≥ n_)
//
//   cuBLAS is column-major, so we apply the classic swap trick:
//     cublasGemmEx(..., N, M, K, B, ..., A, ..., C, ...)
//   This computes the equivalent row-major product.

template <>
Result<void> CudaBackend::gemm_impl<__nv_bfloat16>(const GemmParams& p) {
    float alpha = 1.0f;
    float beta = 0.0f;

    const auto op_b = p.trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    return cublas_check(cublasGemmEx(cublas_handle_, op_b, CUBLAS_OP_N, p.n_, p.m_, p.k_, &alpha,
                                     p.b_, CUDA_R_16BF, p.ldb_, p.a_, CUDA_R_16BF, p.lda_, &beta,
                                     p.c_, CUDA_R_16BF, p.ldc_, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------------------
// gemm_logits (BF16 → FP32)
// ---------------------------------------------------------------------------
// Same row-major swap trick as gemm_impl.  Output C uses CUDA_R_32F.

Result<void> CudaBackend::gemm_logits_impl(const GemmParams& p) {
    float alpha = 1.0f;
    float beta = 0.0f;

    const auto op_b = p.trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    return cublas_check(cublasGemmEx(cublas_handle_, op_b, CUBLAS_OP_N, p.n_, p.m_, p.k_, &alpha,
                                     p.b_, CUDA_R_16BF, p.ldb_, p.a_, CUDA_R_16BF, p.lda_, &beta,
                                     p.c_, CUDA_R_32F, p.ldc_, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------------------
// rms_norm / rope / silu_mul / element_add / split_qkv
// ---------------------------------------------------------------------------

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
                       p.num_q_heads_, p.num_kv_heads_, p.head_dim_, p.rotary_dim_,
                       p.rope_cache_max_position_, stream_);
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

// ---------------------------------------------------------------------------
// attention (naive / prefill / decode) + write_kv_cache
// ---------------------------------------------------------------------------

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
        p.context_lens_, static_cast<__nv_bfloat16*>(p.output_), p.batch_size_, p.num_tokens_,
        p.max_blocks_per_req_, p.num_q_heads_, p.num_kv_heads_, p.head_dim_, p.cache_block_size_,
        stream_);
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

// ---------------------------------------------------------------------------
// sample
// ---------------------------------------------------------------------------

Result<void> CudaBackend::sample_impl(const SampleParams& p) {
    return launch_greedy_sample(p.logits_, p.tokens_out_, p.logits_indices_, p.batch_size_,
                                p.vocab_size_, p.num_tokens_, stream_);
}

// ---------------------------------------------------------------------------
// allocation
// ---------------------------------------------------------------------------

Result<std::unique_ptr<DeviceBuffer>> CudaBackend::allocate_buffer_impl(size_t bytes) {
    auto buf = CudaBuffer::create(bytes);
    if (!buf) return std::unexpected(buf.error());
    std::unique_ptr<DeviceBuffer> base = std::move(*buf);
    return std::move(base);
}

// ---------------------------------------------------------------------------
// memcpy
// ---------------------------------------------------------------------------
// h2d:  async enqueue on stream_.  Caller must keep src alive until the
//       stream reaches this point (or call synchronize() immediately after).
// d2h:  async enqueue + cudaStreamSynchronize, so host data is valid on return.
// d2d:  async enqueue on stream_.  Source and destination must remain valid
//       until the stream completes the copy (stream ordering is sufficient).
// ---------------------------------------------------------------------------

Result<void> CudaBackend::memcpy_h2d_impl(void* dst, const void* src, size_t count) {
    return cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream_));
}

Result<void> CudaBackend::memcpy_d2h_impl(void* dst, const void* src, size_t count) {
    if (auto r = cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream_)); !r)
        return r;
    return cuda_check(cudaStreamSynchronize(stream_));
}

Result<void> CudaBackend::memcpy_d2d_impl(void* dst, const void* src, size_t count) {
    return cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream_));
}

void* CudaBackend::stream_impl() { return stream_; }

Result<void> CudaBackend::synchronize_impl() { return cuda_check(cudaStreamSynchronize(stream_)); }

}  // namespace engine
}  // namespace ccinfer
