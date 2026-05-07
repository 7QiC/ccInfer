#pragma once

#include <cstdint>
#include <memory>

#include "common/result.h"
#include "engine/backend/backend.h"
#include "engine/backend/device_buffer.h"

namespace ccinfer {
namespace engine {

class KVCacheStorageBase {
public:
    virtual ~KVCacheStorageBase() = default;
};

template <typename DType>
class KVCacheStorage final : public KVCacheStorageBase {
public:
    KVCacheStorage() = default;

    Result<void> init(DeviceBackend& backend, int num_layers, int max_blocks, int block_size,
                      int num_kv_heads, int head_dim) {
        int64_t token_stride = static_cast<int64_t>(num_kv_heads) * head_dim;
        int64_t elements_per_layer =
            static_cast<int64_t>(max_blocks) * block_size * token_stride;
        layer_stride_ = elements_per_layer;

        int64_t total_elements = static_cast<int64_t>(num_layers) * elements_per_layer;
        int64_t total_bytes = total_elements * sizeof(DType);

        k_data_ = backend.allocate_buffer(total_bytes);
        v_data_ = backend.allocate_buffer(total_bytes);

        if (!k_data_ || !v_data_) {
            return std::unexpected(ErrorCode::CudaOutOfMemory);
        }
        return {};
    }

    DType* k_data() { return static_cast<DType*>(k_data_->data()); }
    DType* v_data() { return static_cast<DType*>(v_data_->data()); }
    const DType* k_data() const { return static_cast<const DType*>(k_data_->data()); }
    const DType* v_data() const { return static_cast<const DType*>(v_data_->data()); }

    DType* k_layer(int layer) { return static_cast<DType*>(k_data_->data()) + layer * layer_stride_; }
    DType* v_layer(int layer) { return static_cast<DType*>(v_data_->data()) + layer * layer_stride_; }

private:
    std::unique_ptr<DeviceBuffer> k_data_;
    std::unique_ptr<DeviceBuffer> v_data_;
    int64_t layer_stride_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
