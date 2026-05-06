#pragma once

#include <cuda_bf16.h>

#include <vector>

#include "common/result.h"
#include "engine/cache/block.h"
#include "engine/cache/kv_cache_storage.h"
#include "engine/model/config.h"

namespace ccinfer {
namespace engine {

class KVCacheManager {
public:
    struct AllocResult {
        BlockTable block_table;
        std::vector<int32_t> slot_mapping;
    };

    KVCacheManager() = default;

    Result<void> init(const ModelConfig& config, int max_blocks);

    Result<AllocResult> prepare_blocks(const std::vector<int32_t>& token_ids);
    Result<void> release_blocks(const BlockTable& table);

    __nv_bfloat16* k_cache(int layer) { return storage_.k_layer(layer); }
    __nv_bfloat16* v_cache(int layer) { return storage_.v_layer(layer); }

    int num_layers() const { return storage_.num_layers(); }
    int max_blocks() const { return storage_.max_blocks(); }
    int block_size() const { return storage_.block_size(); }
    int num_kv_heads() const { return storage_.num_kv_heads(); }
    int head_dim() const { return storage_.head_dim(); }
    int num_free_blocks() const { return static_cast<int>(free_list_.size()); }

private:
    KVCacheStorage<__nv_bfloat16> storage_;
    std::vector<Block> metadata_;
    FreeList free_list_;
    int max_blocks_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
