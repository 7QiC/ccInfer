#include "backend/backend.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "base/error_code.h"
#include "base/result.h"
#include "backend/cuda/cuda_utils.h"
#include "kernel/cuda_kernels.h"

namespace ccinfer {

namespace {

// CUDA-backed Buffer implementation. Framework code only sees Buffer;
// this class is an internal detail of Backend's Allocator implementation.
class CudaBuffer final : public Buffer {
public:
    static Result<std::shared_ptr<CudaBuffer>> create(std::size_t bytes, ops::Device device) {
        if (bytes == 0) return std::unexpected(ErrorCode::InvalidArgument);
        void* ptr = nullptr;
        auto r = cuda_check(cudaMalloc(&ptr, bytes));
        if (!r) return std::unexpected(r.error());
        return std::shared_ptr<CudaBuffer>(new CudaBuffer(bytes, ptr, device));
    }

    ~CudaBuffer() override {
        if (ptr_) cudaFree(ptr_);
    }

    void* data() noexcept override { return ptr_; }
    const void* data() const noexcept override { return ptr_; }
    std::size_t bytes() const noexcept override { return bytes_; }
    ops::Device device() const noexcept override { return device_; }

private:
    CudaBuffer(std::size_t bytes, void* ptr, ops::Device device)
        : bytes_(bytes), ptr_(ptr), device_(device) {}

    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    ops::Device device_{};
};

// CUDA kernel dispatch helpers. All framework kernels currently operate on
// BF16 activations; new dtypes are added here when kernels support them.

Result<void> gemm_bf16(const GemmParams& p, cublasHandle_t handle, cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = 0.0f;

    const auto op_b = p.trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    return cublas_check(cublasGemmEx(handle, op_b, CUBLAS_OP_N, p.n_, p.m_, p.k_, &alpha, p.b_,
                                     CUDA_R_16BF, p.ldb_, p.a_, CUDA_R_16BF, p.lda_, &beta, p.c_,
                                     CUDA_R_16BF, p.ldc_, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Mixed-precision GEMM: bf16 inputs → float output (for lm_head).
Result<void> gemm_logits_impl(const GemmParams& p, cublasHandle_t handle, cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = 0.0f;

    const auto op_b = p.trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    return cublas_check(cublasGemmEx(handle, op_b, CUBLAS_OP_N, p.n_, p.m_, p.k_, &alpha, p.b_,
                                     CUDA_R_16BF, p.ldb_, p.a_, CUDA_R_16BF, p.lda_, &beta, p.c_,
                                     CUDA_R_32F, p.ldc_, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

Result<void> rms_norm_bf16(const RmsNormParams& p, cudaStream_t stream) {
    return launch_rms_norm(static_cast<const __nv_bfloat16*>(p.input_),
                           static_cast<const __nv_bfloat16*>(p.weight_),
                           static_cast<__nv_bfloat16*>(p.output_), p.rows_, p.dim_, p.eps_,
                           stream);
}

Result<void> rope_bf16(const RopeParams& p, cudaStream_t stream) {
    return launch_rope(static_cast<__nv_bfloat16*>(p.q_), static_cast<__nv_bfloat16*>(p.k_),
                       p.positions_, static_cast<const float2*>(p.rope_cache_), p.num_tokens_,
                       p.num_q_heads_, p.num_kv_heads_, p.head_dim_, p.rotary_dim_,
                       p.rope_cache_max_position_, stream);
}

Result<void> silu_mul_bf16(const SiluMulParams& p, cudaStream_t stream) {
    return launch_silu_mul(static_cast<const __nv_bfloat16*>(p.gate_),
                           static_cast<const __nv_bfloat16*>(p.up_),
                           static_cast<__nv_bfloat16*>(p.output_), p.n_, stream);
}

Result<void> element_add_bf16(const ElementAddParams& p, cudaStream_t stream) {
    return launch_element_add(static_cast<__nv_bfloat16*>(p.dst_),
                              static_cast<const __nv_bfloat16*>(p.src_), p.n_, stream);
}

Result<void> split_qkv_bf16(const SplitQkvParams& p, cudaStream_t stream) {
    return launch_split_qkv(static_cast<const __nv_bfloat16*>(p.qkv_),
                            static_cast<__nv_bfloat16*>(p.q_), static_cast<__nv_bfloat16*>(p.k_),
                            static_cast<__nv_bfloat16*>(p.v_), p.num_tokens_, p.num_q_heads_,
                            p.num_kv_heads_, p.head_dim_, stream);
}

Result<void> naive_attention_bf16(const NaiveAttnParams& p, cudaStream_t stream) {
    return launch_naive_attention(static_cast<const __nv_bfloat16*>(p.q_),
                                  static_cast<const __nv_bfloat16*>(p.k_),
                                  static_cast<const __nv_bfloat16*>(p.v_),
                                  static_cast<__nv_bfloat16*>(p.output_), p.num_tokens_,
                                  p.num_q_heads_, p.num_kv_heads_, p.head_dim_, stream);
}

Result<void> prefill_attention_bf16(const PrefillAttnParams& p, cudaStream_t stream) {
    return launch_prefill_attention(
        static_cast<const __nv_bfloat16*>(p.q_), static_cast<const __nv_bfloat16*>(p.k_cache_),
        static_cast<const __nv_bfloat16*>(p.v_cache_), p.block_table_, p.query_start_loc_,
        p.context_lens_, static_cast<__nv_bfloat16*>(p.output_), p.batch_size_, p.num_tokens_,
        p.max_blocks_per_req_, p.num_q_heads_, p.num_kv_heads_, p.head_dim_,
        p.cache_block_size_, stream);
}

Result<void> decode_attention_bf16(const DecodeAttnParams& p, cudaStream_t stream) {
    return launch_decode_attention(static_cast<const __nv_bfloat16*>(p.q_),
                                   static_cast<const __nv_bfloat16*>(p.k_cache_),
                                   static_cast<const __nv_bfloat16*>(p.v_cache_),
                                   p.block_table_, p.context_lens_,
                                   static_cast<__nv_bfloat16*>(p.output_), p.batch_size_,
                                   p.max_blocks_per_req_, p.num_q_heads_, p.num_kv_heads_,
                                   p.head_dim_, p.cache_block_size_, stream);
}

Result<void> write_kv_cache_bf16(const WriteKVCacheParams& p, cudaStream_t stream) {
    return launch_write_kv_cache(
        static_cast<const __nv_bfloat16*>(p.k_new_), static_cast<const __nv_bfloat16*>(p.v_new_),
        static_cast<__nv_bfloat16*>(p.k_cache_), static_cast<__nv_bfloat16*>(p.v_cache_),
        p.slot_mapping_, p.total_tokens_, p.num_kv_heads_, p.head_dim_, p.max_slots_, stream);
}

Result<void> embed_impl(const EmbedParams& p, cudaStream_t stream) {
    return launch_embed(static_cast<const __nv_bfloat16*>(p.embed_table_), p.token_ids_,
                        static_cast<__nv_bfloat16*>(p.input_embeds_), p.num_tokens_, p.d_model_,
                        stream);
}

Result<void> sample_impl(const SampleParams& p, cudaStream_t stream) {
    return launch_greedy_sample(p.logits_, p.tokens_out_, p.logits_indices_, p.batch_size_,
                                p.vocab_size_, p.num_tokens_, stream);
}

}  // namespace

// Backend::Impl — device-specific state (CUDA in this translation unit).

struct Backend::Impl {
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    int device_id_ = 0;
};

Result<std::unique_ptr<Backend>> Backend::create(int device_id) {
    auto impl = std::make_unique<Impl>();

    if (auto r = cuda_check(cudaSetDevice(device_id)); !r) return std::unexpected(r.error());
    impl->device_id_ = device_id;
    if (auto r = cublas_check(cublasCreate(&impl->cublas_handle_)); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = cuda_check(cudaStreamCreate(&impl->stream_)); !r) {
        cublasDestroy(impl->cublas_handle_);
        return std::unexpected(r.error());
    }
    if (auto r = cublas_check(cublasSetStream(impl->cublas_handle_, impl->stream_)); !r) {
        cudaStreamDestroy(impl->stream_);
        cublasDestroy(impl->cublas_handle_);
        return std::unexpected(r.error());
    }

    return std::unique_ptr<Backend>(new Backend(std::move(impl)));
}

Backend::Backend(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Backend::~Backend() = default;
Backend::Backend(Backend&&) noexcept = default;
Backend& Backend::operator=(Backend&&) noexcept = default;

Result<std::shared_ptr<Buffer>> Backend::allocate_buffer(std::size_t bytes) {
    if (bytes == 0) return std::unexpected(ErrorCode::InvalidArgument);
    auto buf = CudaBuffer::create(bytes, ops::Device{ops::DeviceType::kCUDA, impl_->device_id_});
    if (!buf) return std::unexpected(buf.error());
    return std::move(*buf);
}

Result<void> Backend::memcpy_h2d(void* dst, const void* src, std::size_t count) {
    if (count == 0) return {};
    if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    return cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, impl_->stream_));
}

Result<void> Backend::memcpy_d2h(void* dst, const void* src, std::size_t count) {
    if (count == 0) return {};
    if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    if (auto r = cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                                            impl_->stream_));
        !r)
        return r;
    return cuda_check(cudaStreamSynchronize(impl_->stream_));
}

Result<void> Backend::memcpy_d2d(void* dst, const void* src, std::size_t count) {
    if (count == 0) return {};
    if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    return cuda_check(
        cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, impl_->stream_));
}

void* Backend::stream() const noexcept { return impl_->stream_; }

Result<void> Backend::synchronize() {
    return cuda_check(cudaStreamSynchronize(impl_->stream_));
}

ops::ExecutionContext Backend::context() const noexcept {
    return {impl_->stream_};
}

Result<void> Backend::embed(const EmbedParams& p) {
    if (p.embed_table_ == nullptr || p.token_ids_ == nullptr || p.input_embeds_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.num_tokens_ <= 0 || p.d_model_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return embed_impl(p, impl_->stream_);
}

Result<void> Backend::gemm(ops::DType dtype, const GemmParams& p) {
    if (p.a_ == nullptr || p.b_ == nullptr || p.c_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.m_ <= 0 || p.n_ <= 0 || p.k_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.lda_ <= 0 || p.ldb_ <= 0 || p.ldc_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.trans_a_) {
        return std::unexpected(ErrorCode::Unsupported);
    }
    {
        const int min_ldb = p.trans_b_ ? p.k_ : p.n_;
        if (p.lda_ < p.k_ || p.ldb_ < min_ldb || p.ldc_ < p.n_) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return gemm_bf16(p, impl_->cublas_handle_, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::gemm_logits(const GemmParams& p) {
    if (p.a_ == nullptr || p.b_ == nullptr || p.c_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.m_ <= 0 || p.n_ <= 0 || p.k_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.lda_ <= 0 || p.ldb_ <= 0 || p.ldc_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.trans_a_) {
        return std::unexpected(ErrorCode::Unsupported);
    }
    {
        const int min_ldb = p.trans_b_ ? p.k_ : p.n_;
        if (p.lda_ < p.k_ || p.ldb_ < min_ldb || p.ldc_ < p.n_) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
    }
    return gemm_logits_impl(p, impl_->cublas_handle_, impl_->stream_);
}

Result<void> Backend::rms_norm(ops::DType dtype, const RmsNormParams& p) {
    if (p.input_ == nullptr || p.weight_ == nullptr || p.output_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.rows_ <= 0 || p.dim_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.eps_ <= 0.0f) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return rms_norm_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::rope(ops::DType dtype, const RopeParams& p) {
    if (p.num_tokens_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.q_ == nullptr || p.k_ == nullptr || p.positions_ == nullptr || p.rope_cache_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.num_q_heads_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.rotary_dim_ <= 0 || p.rotary_dim_ > p.head_dim_ || (p.rotary_dim_ & 1)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.rope_cache_max_position_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return rope_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::silu_mul(ops::DType dtype, const SiluMulParams& p) {
    if (p.n_ < 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (p.n_ == 0) return {};
    if (p.gate_ == nullptr || p.up_ == nullptr || p.output_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return silu_mul_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::element_add(ops::DType dtype, const ElementAddParams& p) {
    if (p.n_ < 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (p.n_ == 0) return {};
    if (p.dst_ == nullptr || p.src_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return element_add_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::split_qkv(ops::DType dtype, const SplitQkvParams& p) {
    if (p.qkv_ == nullptr || p.q_ == nullptr || p.k_ == nullptr || p.v_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.num_tokens_ <= 0 || p.num_q_heads_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return split_qkv_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::naive_attention(ops::DType dtype, const NaiveAttnParams& p) {
    if (p.q_ == nullptr || p.k_ == nullptr || p.v_ == nullptr || p.output_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.num_tokens_ <= 0 || p.num_q_heads_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.num_q_heads_ % p.num_kv_heads_ != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return naive_attention_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::prefill_attention(ops::DType dtype, const PrefillAttnParams& p) {
    if (p.q_ == nullptr || p.k_cache_ == nullptr || p.v_cache_ == nullptr ||
        p.block_table_ == nullptr || p.query_start_loc_ == nullptr || p.context_lens_ == nullptr ||
        p.output_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.batch_size_ <= 0 || p.num_tokens_ <= 0 || p.max_blocks_per_req_ <= 0 ||
        p.num_q_heads_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0 ||
        p.cache_block_size_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.num_q_heads_ % p.num_kv_heads_ != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return prefill_attention_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::decode_attention(ops::DType dtype, const DecodeAttnParams& p) {
    if (p.q_ == nullptr || p.k_cache_ == nullptr || p.v_cache_ == nullptr ||
        p.block_table_ == nullptr || p.context_lens_ == nullptr || p.output_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.batch_size_ <= 0 || p.max_blocks_per_req_ <= 0 || p.num_q_heads_ <= 0 ||
        p.num_kv_heads_ <= 0 || p.head_dim_ <= 0 || p.cache_block_size_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.num_q_heads_ % p.num_kv_heads_ != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return decode_attention_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::write_kv_cache(ops::DType dtype, const WriteKVCacheParams& p) {
    if (p.k_new_ == nullptr || p.v_new_ == nullptr || p.k_cache_ == nullptr ||
        p.v_cache_ == nullptr || p.slot_mapping_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.total_tokens_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0 || p.max_slots_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    switch (dtype) {
        case ops::DType::kBFloat16:
            return write_kv_cache_bf16(p, impl_->stream_);
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<void> Backend::sample(const SampleParams& p) {
    if (p.logits_ == nullptr || p.logits_indices_ == nullptr || p.tokens_out_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.batch_size_ <= 0 || p.vocab_size_ <= 0 || p.num_tokens_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.top_k_ < 0 || p.top_p_ <= 0.0f || p.top_k_ > p.vocab_size_ || p.top_p_ > 1.0f ||
        p.temperature_ < 0.0f) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (p.top_k_ > 0 || p.top_p_ < 1.0f || p.temperature_ > 0.0f) {
        return std::unexpected(ErrorCode::Unsupported);
    }
    return sample_impl(p, impl_->stream_);
}

}  // namespace ccinfer
