#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "cache/block.h"
#include "cache/cache_stats.h"
#include "cache/prefix_cache.h"
#include "common/block_table.h"
#include "common/error_code.h"

namespace ccinfer {

// Scheduler-owned logical KV block allocator. It owns IDs, refcounts, the LRU
// and prefix index, but no device memory.
class BlockPool {
public:
    BlockPool() = default;
    BlockPool(int max_blocks, int block_size);
    ~BlockPool();

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    Result<void> init(int max_blocks, int block_size);

    Result<BlockTable> allocate_blocks(int num_blocks);
    void release_blocks(const BlockTable& table);

    struct PrefixLookup {
        BlockTable block_table;
        int prefix_hit_blocks = 0;
        uint64_t parent_hash = 0;
    };

    PrefixLookup lookup_prefix_cache(const std::vector<int32_t>& tokens,
                                     uint64_t namespace_salt = 0);

    // Publishes the requested block without changing the request block table.
    uint64_t publish_full_block(uint64_t parent_hash, const std::vector<int32_t>& tokens,
                                int32_t block_id, uint64_t seed = 0);

    int max_blocks() const noexcept { return max_blocks_; }
    int block_size() const noexcept { return block_size_; }
    int num_free_blocks() const noexcept;
    KVCacheStats stats() const;
    PrefixCacheStats prefix_stats() const;

private:
    bool evict_one();

    std::unique_ptr<Block[]> metadata_;
    FreeList free_list_;
    LruList lru_list_;
    PrefixCache prefix_cache_;
    int max_blocks_ = 0;
    int block_size_ = kKVBlockSize;
    mutable std::mutex mutex_;
};

}  // namespace ccinfer
