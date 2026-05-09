#include "engine/cache/kv_cache_manager.h"

#include <unordered_set>

#include "common/error_code.h"

namespace ccinfer {
namespace engine {

Result<void> KVCacheManager::init(int max_blocks) {
    if (max_blocks_ != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (max_blocks <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    max_blocks_ = max_blocks;
    metadata_.resize(max_blocks);

    for (int i = 0; i < max_blocks; ++i) {
        metadata_[i].block_id = i;
        metadata_[i].ref_count = 0;
        metadata_[i].flags = static_cast<uint32_t>(BlockFlags::kInFreeList);
        free_list_.push_back(metadata_[i]);
    }

    return {};
}

Result<BlockTable> KVCacheManager::allocate_blocks(int num_blocks) {
    if (num_blocks <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (num_blocks > static_cast<int>(free_list_.size())) {
        return std::unexpected(ErrorCode::KVBlockExhausted);
    }

    BlockTable result;
    for (int b = 0; b < num_blocks; ++b) {
        auto& block = free_list_.front();
        free_list_.pop_front();

        block.flags &= ~static_cast<uint32_t>(BlockFlags::kInFreeList);
        block.ref_count = 1;
        block.block_hash = 0;

        result.push_back(block.block_id);
    }

    return result;
}

Result<void> KVCacheManager::release_blocks(const BlockTable& table) {
    // --- Phase 1: validate all entries before mutating state ---
    std::unordered_set<int32_t> seen;
    for (int i = 0; i < table.size(); ++i) {
        int32_t block_id = table[i];
        if (block_id < 0 || block_id >= max_blocks_) {
            return std::unexpected(ErrorCode::KVInvalidBlockTable);
        }
        if (!seen.insert(block_id).second) {
            return std::unexpected(ErrorCode::KVInvalidBlockTable);
        }

        auto& block = metadata_[block_id];
        if (block.is_free()) {
            return std::unexpected(ErrorCode::KVBlockDoubleFree);
        }
        if (block.ref_count <= 0) {
            return std::unexpected(ErrorCode::KVBlockDoubleFree);
        }
    }

    // --- Phase 2: mutate ---
    for (int i = 0; i < table.size(); ++i) {
        int32_t block_id = table[i];
        auto& block = metadata_[block_id];

        block.ref_count--;
        if (block.ref_count == 0) {
            block.flags |= static_cast<uint32_t>(BlockFlags::kInFreeList);
            block.block_hash = 0;
            free_list_.push_back(block);
        }
    }

    return {};
}

}  // namespace engine
}  // namespace ccinfer
