#pragma once

#include <cstdint>
#include <vector>

#include "common/result.h"
#include "engine/cache/block.h"

namespace ccinfer {
namespace engine {

// Owns the free list and block metadata.  allocate_blocks / release_blocks
// are the only mutating operations.  Slot mapping, positions, and block-table
// layout are computed by BatchTranslator.
class KVCacheManager {
public:
    KVCacheManager() = default;

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    KVCacheManager(KVCacheManager&&) = delete;
    KVCacheManager& operator=(KVCacheManager&&) = delete;

    Result<void> init(int max_blocks);

    Result<BlockTable> allocate_blocks(int num_blocks);
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
