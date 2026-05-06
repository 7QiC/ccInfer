#pragma once

#include <cstdint>

#include "common/result.h"
#include "engine/core/device_buffer.h"

namespace ccinfer {
namespace engine {

template <typename DType>
class KVCacheStorage {
public:
    KVCacheStorage() = default;

    Result<void> init(int num_layers, int max_blocks, int block_size,
                      int num_kv_heads, int head_dim) {
        int64_t token_stride = static_cast<int64_t>(num_kv_heads) * head_dim;
        int64_t elements_per_layer =
            static_cast<int64_t>(max_blocks) * block_size * token_stride;
        layer_stride_ = elements_per_layer;

        int64_t total_elements = static_cast<int64_t>(num_layers) * elements_per_layer;

        k_data_ = DeviceBuffer<DType>(total_elements);
        v_data_ = DeviceBuffer<DType>(total_elements);

        if (k_data_.get() == nullptr || v_data_.get() == nullptr) {
            return std::unexpected(ErrorCode::CudaOutOfMemory);
        }
        return {};
    }

    DType* k_data() { return k_data_.get(); }
    DType* v_data() { return v_data_.get(); }
    const DType* k_data() const { return k_data_.get(); }
    const DType* v_data() const { return v_data_.get(); }

    DType* k_layer(int layer) { return k_data_.get() + layer * layer_stride_; }
    DType* v_layer(int layer) { return v_data_.get() + layer * layer_stride_; }

private:
    DeviceBuffer<DType> k_data_;
    DeviceBuffer<DType> v_data_;
    int64_t layer_stride_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
