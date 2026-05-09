#pragma once

#include <concepts>
#include <cstdint>
#include <memory>

#include "common/result.h"
#include "engine/backend/device_buffer.h"
#include "engine/common/backend_def.h"

namespace ccinfer {
namespace engine {

class KVCacheStorage {
public:
    template <typename KVDType>
    Result<void> init(DefaultBackend& backend, int num_layers, int max_blocks, int block_size,
                      int num_kv_heads, int head_dim) {
        int64_t token_stride = static_cast<int64_t>(num_kv_heads) * head_dim;
        int64_t elements_per_layer = static_cast<int64_t>(max_blocks) * block_size * token_stride;
        layer_stride_ = elements_per_layer;

        int64_t total_elements = static_cast<int64_t>(num_layers) * elements_per_layer;
        int64_t total_bytes = total_elements * sizeof(KVDType);

        k_data_ = backend.allocate_buffer(total_bytes);
        v_data_ = backend.allocate_buffer(total_bytes);

        if (!k_data_ || !v_data_) {
            return std::unexpected(ErrorCode::CudaOutOfMemory);
        }

        elem_size_ = sizeof(KVDType);
        return {};
    }

    void* k_data() { return k_data_->data(); }
    void* v_data() { return v_data_->data(); }
    const void* k_data() const { return k_data_->data(); }
    const void* v_data() const { return v_data_->data(); }

    void* k_layer(int layer) {
        return buffer_data<char>(*k_data_) + layer * layer_stride_ * elem_size_;
    }
    void* v_layer(int layer) {
        return buffer_data<char>(*v_data_) + layer * layer_stride_ * elem_size_;
    }

private:
    std::unique_ptr<DeviceBuffer> k_data_;
    std::unique_ptr<DeviceBuffer> v_data_;
    int64_t layer_stride_ = 0;
    size_t elem_size_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
