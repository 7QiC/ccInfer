#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "common/result.h"
#include "engine/cache/block.h"
#include "engine/cache/cache_stats.h"

namespace ccinfer {
namespace engine {

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

    // ---- Data pointers for kernel launches ----
    void* k_cache(int layer);
    void* v_cache(int layer);
    const void* k_cache(int layer) const;
    const void* v_cache(int layer) const;

    int max_slots() const;

    // ---- Block allocation ----
    Result<BlockTable> allocate_blocks(int num_blocks);

    struct PrepareResult {
        BlockTable block_table;
        int prefix_hit_blocks = 0;
    };
    Result<PrepareResult> lookup_prefix_cache(const std::vector<int32_t>& tokens,
                                              uint64_t namespace_salt = 0);
    Result<PrepareResult> prepare_blocks(const std::vector<int32_t>& tokens,
                                         uint64_t namespace_salt = 0);

    Result<void> cache_full_blocks(const BlockTable& table, const std::vector<int32_t>& tokens,
                                   int committed_tokens, uint64_t namespace_salt = 0);

    Result<void> release_blocks(const BlockTable& table);

    // ---- Capacity and stats ----
    int max_blocks() const { return max_blocks_; }
    int block_size() const { return block_size_; }
    int num_free_blocks() const { return static_cast<int>(free_list_.size()); }
    KVCacheStats stats() const;
    PrefixCacheStats prefix_stats() const;

private:
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

}  // namespace engine
}  // namespace ccinfer
