#include "cache/kv_cache_storage.h"

#include <cassert>
#include <limits>
#include <utility>

#include "backend/backend.h"
#include "common/error_code.h"

namespace ccinfer {

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

namespace {

void* layer_data(const std::shared_ptr<Buffer>& buffer, int layer, int num_layers,
                 int64_t layer_stride, std::size_t elem_size) {
    assert(buffer && "KVCacheStorage not initialized");
    assert(layer >= 0 && layer < num_layers && "layer out of range");
    const std::size_t offset =
        static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride) * elem_size;
    return static_cast<char*>(buffer->data()) + offset;
}

}  // namespace

Tensor KVCacheStorage::k_layer_tensor(int layer) {
    return Tensor::from_buffer(k_data_,
                               layer_data(k_data_, layer, num_layers_, layer_stride_, elem_size_),
                               dtype_, {max_slots_, num_kv_heads_, head_dim_});
}

Tensor KVCacheStorage::v_layer_tensor(int layer) {
    return Tensor::from_buffer(v_data_,
                               layer_data(v_data_, layer, num_layers_, layer_stride_, elem_size_),
                               dtype_, {max_slots_, num_kv_heads_, head_dim_});
}

Tensor KVCacheStorage::k_block_tensor(int layer) {
    return Tensor::from_buffer(k_data_,
                               layer_data(k_data_, layer, num_layers_, layer_stride_, elem_size_),
                               dtype_, {max_blocks_, block_size_, num_kv_heads_, head_dim_});
}

Tensor KVCacheStorage::v_block_tensor(int layer) {
    return Tensor::from_buffer(v_data_,
                               layer_data(v_data_, layer, num_layers_, layer_stride_, elem_size_),
                               dtype_, {max_blocks_, block_size_, num_kv_heads_, head_dim_});
}

Result<std::unique_ptr<KVCacheStorage>> KVCacheStorage::create(Backend& backend, int num_layers,
                                                               int max_blocks, int block_size,
                                                               int num_kv_heads, int head_dim,
                                                               ccop::DType dtype) {
    auto storage = std::make_unique<KVCacheStorage>();
    auto r =
        storage->init(backend, num_layers, max_blocks, block_size, num_kv_heads, head_dim, dtype);
    if (!r) return std::unexpected(r.error());
    return storage;
}

Result<void> KVCacheStorage::init(Backend& backend, int num_layers, int max_blocks, int block_size,
                                  int num_kv_heads, int head_dim, ccop::DType dtype) {
    if (num_layers <= 0 || max_blocks <= 0 || block_size <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || dtype == ccop::DType::kUnknown) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    const std::size_t elem_size = ccop::dtype_size(dtype);
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
    dtype_ = dtype;
    max_slots_ = static_cast<int>(max_slots);
    max_blocks_ = max_blocks;
    block_size_ = block_size;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    num_layers_ = num_layers;
    return {};
}

}  // namespace ccinfer
