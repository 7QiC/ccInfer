#pragma once

#include <cstddef>
#include <memory>

#include "base/error_code.h"
#include "base/result.h"
#include "backend/device_buffer.h"
#include "backend/params.h"

namespace ccinfer {

template <typename Derived>
class DeviceBackend {
public:
    DeviceBackend() = default;

    template <typename DType>
    Result<void> gemm(const GemmParams& p) {
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

        return self().template gemm_impl<DType>(p);
    }

    // Mixed-precision GEMM: bf16 inputs → float output (for lm_head).
    Result<void> gemm_logits(const GemmParams& p) {
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

        return self().gemm_logits_impl(p);
    }

    template <typename DType>
    Result<void> rms_norm(const RmsNormParams& p) {
        if (p.input_ == nullptr || p.weight_ == nullptr || p.output_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.rows_ <= 0 || p.dim_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.eps_ <= 0.0f) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template rms_norm_impl<DType>(p);
    }

    template <typename DType>
    Result<void> rope(const RopeParams& p) {
        if (p.num_tokens_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.q_ == nullptr || p.k_ == nullptr || p.positions_ == nullptr ||
            p.rope_cache_ == nullptr) {
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

        return self().template rope_impl<DType>(p);
    }

    template <typename DType>
    Result<void> silu_mul(const SiluMulParams& p) {
        if (p.n_ < 0) return std::unexpected(ErrorCode::InvalidArgument);
        if (p.n_ == 0) return {};

        if (p.gate_ == nullptr || p.up_ == nullptr || p.output_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template silu_mul_impl<DType>(p);
    }

    template <typename DType>
    Result<void> element_add(const ElementAddParams& p) {
        if (p.n_ < 0) return std::unexpected(ErrorCode::InvalidArgument);
        if (p.n_ == 0) return {};

        if (p.dst_ == nullptr || p.src_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template element_add_impl<DType>(p);
    }

    template <typename DType>
    Result<void> split_qkv(const SplitQkvParams& p) {
        if (p.qkv_ == nullptr || p.q_ == nullptr || p.k_ == nullptr || p.v_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.num_tokens_ <= 0 || p.num_q_heads_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template split_qkv_impl<DType>(p);
    }

    template <typename DType>
    Result<void> naive_attention(const NaiveAttnParams& p) {
        if (p.q_ == nullptr || p.k_ == nullptr || p.v_ == nullptr || p.output_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.num_tokens_ <= 0 || p.num_q_heads_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.num_q_heads_ % p.num_kv_heads_ != 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template naive_attention_impl<DType>(p);
    }

    template <typename DType>
    Result<void> prefill_attention(const PrefillAttnParams& p) {
        if (p.q_ == nullptr || p.k_cache_ == nullptr || p.v_cache_ == nullptr ||
            p.block_table_ == nullptr || p.query_start_loc_ == nullptr ||
            p.context_lens_ == nullptr || p.output_ == nullptr) {
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

        return self().template prefill_attention_impl<DType>(p);
    }

    template <typename DType>
    Result<void> decode_attention(const DecodeAttnParams& p) {
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

        return self().template decode_attention_impl<DType>(p);
    }

    template <typename DType>
    Result<void> write_kv_cache(const WriteKVCacheParams& p) {
        if (p.k_new_ == nullptr || p.v_new_ == nullptr || p.k_cache_ == nullptr ||
            p.v_cache_ == nullptr || p.slot_mapping_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.total_tokens_ <= 0 || p.num_kv_heads_ <= 0 || p.head_dim_ <= 0 || p.max_slots_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template write_kv_cache_impl<DType>(p);
    }

    Result<void> embed(const EmbedParams& p) {
        if (p.embed_table_ == nullptr || p.token_ids_ == nullptr || p.input_embeds_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.num_tokens_ <= 0 || p.d_model_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().embed_impl(p);
    }

    Result<void> sample(const SampleParams& p) {
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

        return self().sample_impl(p);
    }

    Result<std::unique_ptr<DeviceBuffer>> allocate_buffer(std::size_t bytes) {
        if (bytes == 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().allocate_buffer_impl(bytes);
    }

    Result<void> memcpy_h2d(void* dst, const void* src, std::size_t count) {
        if (count == 0) return {};
        if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
        return self().memcpy_h2d_impl(dst, src, count);
    }

    Result<void> memcpy_d2h(void* dst, const void* src, std::size_t count) {
        if (count == 0) return {};
        if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
        return self().memcpy_d2h_impl(dst, src, count);
    }

    Result<void> memcpy_d2d(void* dst, const void* src, std::size_t count) {
        if (count == 0) return {};
        if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
        return self().memcpy_d2d_impl(dst, src, count);
    }

    void* stream() { return self().stream_impl(); }

    Result<void> synchronize() { return self().synchronize_impl(); }

protected:
    ~DeviceBackend() = default;

private:
    Derived& self() { return static_cast<Derived&>(*this); }

    const Derived& self() const { return static_cast<const Derived&>(*this); }
};

}  // namespace ccinfer
