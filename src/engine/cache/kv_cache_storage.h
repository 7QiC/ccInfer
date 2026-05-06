#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
        num_layers_ = num_layers;
        max_blocks_ = max_blocks;
        block_size_ = block_size;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;

        int64_t elements_per_layer =
            static_cast<int64_t>(max_blocks) * block_size * num_kv_heads * head_dim;
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

    int64_t slot_offset(int block_id, int pos) const {
        return (static_cast<int64_t>(block_id) * block_size_ + pos) *
               num_kv_heads_ * head_dim_;
    }

    int num_layers() const { return num_layers_; }
    int max_blocks() const { return max_blocks_; }
    int block_size() const { return block_size_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    int64_t layer_stride() const { return layer_stride_; }

private:
    DeviceBuffer<DType> k_data_;
    DeviceBuffer<DType> v_data_;
    int num_layers_ = 0;
    int max_blocks_ = 0;
    int block_size_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int64_t layer_stride_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
