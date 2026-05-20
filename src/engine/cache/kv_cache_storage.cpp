#include "engine/cache/kv_cache_storage.h"

#include <cassert>
#include <limits>
#include <utility>

#include "common/error_code.h"
#include "engine/backend/device_buffer.h"

namespace ccinfer {
namespace engine {

namespace {

inline bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    out = a * b;
    return true;
}

}  // namespace

KVCacheStorage::~KVCacheStorage() = default;
KVCacheStorage::KVCacheStorage(KVCacheStorage&&) noexcept = default;
KVCacheStorage& KVCacheStorage::operator=(KVCacheStorage&&) noexcept = default;

void* KVCacheStorage::k_data() {
    assert(k_data_ && "KVCacheStorage not initialized");
    return k_data_->data();
}
void* KVCacheStorage::v_data() {
    assert(v_data_ && "KVCacheStorage not initialized");
    return v_data_->data();
}
const void* KVCacheStorage::k_data() const {
    assert(k_data_ && "KVCacheStorage not initialized");
    return k_data_->data();
}
const void* KVCacheStorage::v_data() const {
    assert(v_data_ && "KVCacheStorage not initialized");
    return v_data_->data();
}

void* KVCacheStorage::k_layer(int layer) {
    assert(k_data_ && "KVCacheStorage not initialized");
    assert(layer >= 0 && layer < num_layers_ && "layer out of range");
    const std::size_t offset =
        static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride_) * elem_size_;
    return buffer_data<char>(*k_data_) + offset;
}

void* KVCacheStorage::v_layer(int layer) {
    assert(v_data_ && "KVCacheStorage not initialized");
    assert(layer >= 0 && layer < num_layers_ && "layer out of range");
    const std::size_t offset =
        static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride_) * elem_size_;
    return buffer_data<char>(*v_data_) + offset;
}

const void* KVCacheStorage::k_layer(int layer) const {
    assert(k_data_ && "KVCacheStorage not initialized");
    assert(layer >= 0 && layer < num_layers_ && "layer out of range");
    const std::size_t offset =
        static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride_) * elem_size_;
    return buffer_data<char>(*k_data_) + offset;
}

const void* KVCacheStorage::v_layer(int layer) const {
    assert(v_data_ && "KVCacheStorage not initialized");
    assert(layer >= 0 && layer < num_layers_ && "layer out of range");
    const std::size_t offset =
        static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride_) * elem_size_;
    return buffer_data<char>(*v_data_) + offset;
}

Result<void> KVCacheStorage::init(DefaultBackend& backend, int num_layers, int max_blocks,
                                  int block_size, int num_kv_heads, int head_dim,
                                  std::size_t elem_size) {
    if (num_layers <= 0 || max_blocks <= 0 || block_size <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || elem_size == 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (k_data_ || v_data_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

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
    if (!checked_mul(static_cast<std::size_t>(num_layers), elements_per_layer, total_elements)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    std::size_t total_bytes;
    if (!checked_mul(total_elements, elem_size, total_bytes)) {
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

    k_data_ = std::move(k_buf);
    v_data_ = std::move(v_buf);

    layer_stride_ = static_cast<int64_t>(elements_per_layer);
    elem_size_ = elem_size;
    max_slots_ = static_cast<int>(max_slots);
    num_layers_ = num_layers;
    return {};
}

}  // namespace engine
}  // namespace ccinfer
