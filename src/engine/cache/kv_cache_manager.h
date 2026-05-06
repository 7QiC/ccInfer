#pragma once

#include <cstdint>
#include <vector>

#include "common/result.h"
#include "engine/cache/block.h"

namespace ccinfer {
namespace engine {

class KVCacheManager {
public:
    struct AllocResult {
        BlockTable block_table;
        std::vector<int32_t> slot_mapping;
    };

    KVCacheManager() = default;

    Result<void> init(int max_blocks);

    Result<AllocResult> prepare_blocks(const std::vector<int32_t>& token_ids);
    Result<void> release_blocks(const BlockTable& table);

    int max_blocks() const { return max_blocks_; }
    int num_free_blocks() const { return static_cast<int>(free_list_.size()); }

private:
    std::vector<Block> metadata_;
    FreeList free_list_;
    int max_blocks_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
