#pragma once

#include <cstddef>
#include <memory>

#include "common/error_code.h"
#include "common/result.h"
#include "engine/backend/device_buffer.h"
#include "engine/backend/params.h"

namespace ccinfer {
namespace engine {

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

        return self().template gemm_impl<DType>(p);
    }

    template <typename DType>
    Result<void> rms_norm(const RmsNormParams& p) {
        if (p.input_ == nullptr || p.weight_ == nullptr || p.output_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.rows_ <= 0 || p.dim_ <= 0) {
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

        if (p.max_position_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template rope_impl<DType>(p);
    }

    template <typename DType>
    Result<void> silu_mul(const SiluMulParams& p) {
        if (p.n_ <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (p.gate_ == nullptr || p.up_ == nullptr || p.output_ == nullptr) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        return self().template silu_mul_impl<DType>(p);
    }

    template <typename DType>
    Result<void> element_add(const ElementAddParams& p) {
        if (p.n_ <= 0) {
            return {};
        }

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

        int total_query_tokens = p.query_start_loc_[p.batch_size_];
        if (p.batch_size_ <= 0 || total_query_tokens <= 0 || p.max_blocks_per_req_ <= 0 || p.num_q_heads_ <= 0 ||
            p.num_kv_heads_ <= 0 || p.head_dim_ <= 0 || p.cache_block_size_ <= 0) {
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

    std::unique_ptr<DeviceBuffer> allocate_buffer(std::size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }

        return self().allocate_buffer_impl(bytes);
    }

    Result<void> memcpy_h2d(void* dst, const void* src, std::size_t count) {
        if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
        if (count == 0) return {};
        return self().memcpy_h2d_impl(dst, src, count);
    }

    Result<void> memcpy_d2h(void* dst, const void* src, std::size_t count) {
        if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
        if (count == 0) return {};
        return self().memcpy_d2h_impl(dst, src, count);
    }

    Result<void> memcpy_d2d(void* dst, const void* src, std::size_t count) {
        if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
        if (count == 0) return {};
        return self().memcpy_d2d_impl(dst, src, count);
    }

    void* stream() { return self().stream_impl(); }

protected:
    ~DeviceBackend() = default;

private:
    Derived& self() { return static_cast<Derived&>(*this); }

    const Derived& self() const { return static_cast<const Derived&>(*this); }
};

}  // namespace engine
}  // namespace ccinfer
