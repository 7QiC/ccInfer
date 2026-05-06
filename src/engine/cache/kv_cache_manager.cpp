#include "engine/cache/kv_cache_manager.h"

#include <algorithm>

namespace ccinfer {
namespace engine {

Result<void> KVCacheManager::init(const ModelConfig& config, int max_blocks) {
    if (max_blocks <= 0 || config.n_layers_ <= 0 || config.n_kv_heads_ <= 0 ||
        config.head_dim_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto r = storage_.init(config.n_layers_, max_blocks, kKVBlockSize,
                           config.n_kv_heads_, config.head_dim_);
    if (!r) return r;

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

Result<KVCacheManager::AllocResult> KVCacheManager::prepare_blocks(
    const std::vector<int32_t>& token_ids) {

    if (token_ids.empty()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    int num_tokens = static_cast<int>(token_ids.size());
    int num_blocks_needed = (num_tokens + kKVBlockSize - 1) / kKVBlockSize;

    if (num_blocks_needed > static_cast<int>(free_list_.size())) {
        return std::unexpected(ErrorCode::KVBlockExhausted);
    }

    AllocResult result;
    result.slot_mapping.resize(num_tokens, -1);

    for (int b = 0; b < num_blocks_needed; ++b) {
        auto& block = free_list_.front();
        free_list_.pop_front();

        block.flags &= ~static_cast<uint32_t>(BlockFlags::kInFreeList);
        block.ref_count = 1;
        block.block_hash = 0;

        result.block_table.push_back(block.block_id);

        int tokens_in_block = std::min(kKVBlockSize, num_tokens - b * kKVBlockSize);
        for (int p = 0; p < tokens_in_block; ++p) {
            int token_idx = b * kKVBlockSize + p;
            result.slot_mapping[token_idx] = block.block_id * kKVBlockSize + p;
        }
    }

    return result;
}

Result<void> KVCacheManager::release_blocks(const BlockTable& table) {
    for (int i = 0; i < table.size(); ++i) {
        int32_t block_id = table[i];
        if (block_id < 0 || block_id >= max_blocks_) {
            return std::unexpected(ErrorCode::KVInvalidBlockTable);
        }

        auto& block = metadata_[block_id];

        if (block.is_free()) {
            return std::unexpected(ErrorCode::KVBlockDoubleFree);
        }

        block.ref_count--;
        if (block.ref_count < 0) {
            return std::unexpected(ErrorCode::KVBlockDoubleFree);
        }

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
