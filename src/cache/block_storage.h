#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "base/error.h"
#include "runtime/tensor.h"

namespace ccinfer {

class Backend;

class BlockStorage {
public:
    BlockStorage() = default;
    ~BlockStorage();
    BlockStorage(BlockStorage&&) noexcept;
    BlockStorage& operator=(BlockStorage&&) noexcept;

    BlockStorage(const BlockStorage&) = delete;
    BlockStorage& operator=(const BlockStorage&) = delete;

    // num_kv_layers is the number of layers that own KV cache (full-attention
    // layers for Qwen3.5; all decoder layers for Qwen3).
    static Result<std::unique_ptr<BlockStorage>> create(Backend& backend, int num_kv_layers,
                                                        int max_blocks, int block_size,
                                                        int num_kv_heads, int head_dim,
                                                        ccop::DType dtype);

    // Slot-major 3D view of one KV layer: [max_slots, num_kv_heads, head_dim].
    Tensor k_layer_tensor(int layer);
    Tensor v_layer_tensor(int layer);
    // Paged 4D view of one KV layer: [max_blocks, block_size, num_kv_heads, head_dim].
    Tensor k_block_tensor(int layer);
    Tensor v_block_tensor(int layer);

    // Number of KV layers allocated (not necessarily decoder layers).
    int num_layers() const { return num_layers_; }
    int max_blocks() const { return max_blocks_; }
    int block_size() const { return block_size_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    int max_slots() const { return max_slots_; }
    int64_t layer_stride() const { return layer_stride_; }
    std::size_t elem_size() const { return elem_size_; }
    ccop::DType dtype() const { return dtype_; }

private:
    Result<void> init(Backend& backend, int num_kv_layers, int max_blocks, int block_size,
                      int num_kv_heads, int head_dim, ccop::DType dtype);

    std::shared_ptr<Buffer> k_data_;
    std::shared_ptr<Buffer> v_data_;
    int64_t layer_stride_ = 0;
    std::size_t elem_size_ = 0;
    ccop::DType dtype_ = ccop::DType::kUnknown;
    int max_slots_ = 0;
    int max_blocks_ = 0;
    int block_size_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int num_layers_ = 0;
};

}  // namespace ccinfer
