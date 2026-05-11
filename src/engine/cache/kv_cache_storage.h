#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "common/result.h"
#include "engine/backend/device_buffer.h"
#include "engine/cache/block.h"
#include "engine/common/backend_def.h"

namespace ccinfer {
namespace engine {
namespace detail {

inline bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    out = a * b;
    return true;
}

}  // namespace detail

class KVCacheStorage {
public:
    template <typename KVDType>
    Result<void> init(DefaultBackend& backend, int num_layers, int max_blocks, int block_size,
                      int num_kv_heads, int head_dim) {
        if (num_layers <= 0 || max_blocks <= 0 || block_size <= 0 || num_kv_heads <= 0 ||
            head_dim <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        if (block_size != kKVBlockSize) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        if (k_data_ || v_data_) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        using detail::checked_mul;

        std::size_t max_slots;
        if (!checked_mul(static_cast<std::size_t>(max_blocks), static_cast<std::size_t>(block_size),
                         max_slots)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        std::size_t token_stride;
        if (!checked_mul(static_cast<std::size_t>(num_kv_heads), static_cast<std::size_t>(head_dim),
                         token_stride)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        std::size_t elements_per_layer;
        if (!checked_mul(max_slots, token_stride, elements_per_layer)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        std::size_t total_elements;
        if (!checked_mul(static_cast<std::size_t>(num_layers), elements_per_layer,
                         total_elements)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        std::size_t total_bytes;
        if (!checked_mul(total_elements, sizeof(KVDType), total_bytes)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (max_slots > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        if (elements_per_layer > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        auto k_r = backend.allocate_buffer(total_bytes);
        if (!k_r) return std::unexpected(k_r.error());
        auto k_buf = std::move(*k_r);

        auto v_r = backend.allocate_buffer(total_bytes);
        if (!v_r) return std::unexpected(v_r.error());
        auto v_buf = std::move(*v_r);

        // Both buffers allocated — commit.
        k_data_ = std::move(k_buf);
        v_data_ = std::move(v_buf);

        layer_stride_ = static_cast<int64_t>(elements_per_layer);
        elem_size_ = sizeof(KVDType);
        max_slots_ = static_cast<int>(max_slots);
        block_size_ = block_size;
        num_layers_ = num_layers;
        return {};
    }

    void* k_data() { return k_data_->data(); }
    void* v_data() { return v_data_->data(); }
    const void* k_data() const { return k_data_->data(); }
    const void* v_data() const { return v_data_->data(); }

    // Pre: init() must have succeeded and 0 <= layer < num_layers_.
    void* k_layer(int layer) {
        const std::size_t offset =
            static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride_) * elem_size_;
        return buffer_data<char>(*k_data_) + offset;
    }
    void* v_layer(int layer) {
        const std::size_t offset =
            static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride_) * elem_size_;
        return buffer_data<char>(*v_data_) + offset;
    }

    int max_slots() const { return max_slots_; }
    int block_size() const { return block_size_; }

private:
    std::unique_ptr<DeviceBuffer> k_data_;
    std::unique_ptr<DeviceBuffer> v_data_;
    int64_t layer_stride_ = 0;
    std::size_t elem_size_ = 0;
    int max_slots_ = 0;
    int block_size_ = 0;
    int num_layers_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
