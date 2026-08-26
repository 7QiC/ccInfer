#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cache/block.h"
#include "cache/cache_stats.h"
#include "common/error_code.h"
#include "core/tensor.h"

namespace ccinfer {

class KVCacheStorage;
class PrefixCache;

class KVCacheManager {
public:
    KVCacheManager();
    ~KVCacheManager();

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    KVCacheManager(KVCacheManager&&) = delete;
    KVCacheManager& operator=(KVCacheManager&&) = delete;

    Result<void> init(std::unique_ptr<KVCacheStorage> storage, int max_blocks, int block_size);

    // Slot-major 3D view of one layer: [max_slots, num_kv_heads, head_dim].
    Tensor k_cache(int layer);
    Tensor v_cache(int layer);
    // Paged 4D view of one layer: [max_blocks, block_size, num_kv_heads, head_dim].
    Tensor k_cache_blocks(int layer);
    Tensor v_cache_blocks(int layer);

    int max_slots() const;

    Result<BlockTable> allocate_blocks(int num_blocks);

    struct PrepareResult {
        BlockTable block_table;
        int prefix_hit_blocks = 0;
    };
    Result<PrepareResult> lookup_prefix_cache(const std::vector<int32_t>& tokens,
                                              uint64_t namespace_salt = 0);
    Result<void> cache_full_blocks(const BlockTable& table, const std::vector<int32_t>& tokens,
                                   int committed_tokens, uint64_t namespace_salt = 0);
    Result<uint64_t> cache_rolling_blocks(uint64_t parent_hash,
                                          const std::vector<int32_t>& pending_tokens,
                                          const std::vector<int32_t>& block_ids);

    Result<void> release_blocks(const BlockTable& table);

    int max_blocks() const { return max_blocks_; }
    int block_size() const { return block_size_; }
    int num_free_blocks() const { return static_cast<int>(free_list_.size()); }
    KVCacheStats stats() const;
    PrefixCacheStats prefix_stats() const;

private:
    Result<void> cache_block_hash(int32_t block_id, uint64_t hash);
    bool evict_one();
    void rollback_prefix_hits(const BlockTable& table, int count);

    std::unique_ptr<KVCacheStorage> kv_storage_;
    std::unique_ptr<Block[]> metadata_;  // pre-allocated, never resized
    FreeList free_list_;
    LruList lru_list_;
    std::unique_ptr<PrefixCache> prefix_cache_;
    int max_blocks_ = 0;
    int block_size_ = kKVBlockSize;
};

}  // namespace ccinfer
