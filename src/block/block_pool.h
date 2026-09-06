#pragma once

#include <memory>
#include <mutex>

#include "base/error.h"
#include "block/block.h"
#include "block/block_table.h"

namespace ccinfer {

// Scheduler-owned logical KV block resource pool. It owns the full block
// metadata set plus the free list. Cache membership, LRU and eviction policy
// live in BlockCache (cache/), which coordinates with this pool at a higher
// level.
class BlockPool {
public:
    BlockPool() = default;
    BlockPool(int max_blocks, int block_size);
    ~BlockPool();

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    Result<void> init(int max_blocks, int block_size);

    // Allocates one block and gives the caller a request ref (ref_count = 1).
    // Returns KVBlockExhausted when the free list is empty. Eviction is the
    // caller's responsibility.
    Result<BlockId> allocate();
    Result<BlockTable> allocate_blocks(int num_blocks);

    // Retains a request ref. The block must already be InUse.
    void retain_request(BlockId id);

    // Releases one request ref and returns the remaining refs. Does not recycle
    // at zero: the caller decides whether a BlockCacheEntry still owns the block.
    int release_request(BlockId id);

    // Releases every block in a table by one request ref. Equivalent to calling
    // release_request per id; no automatic recycling is performed.
    void release_blocks(const BlockTable& table);

    // Moves a zero-ref InUse block back to Free and re-links it into the free
    // list. The caller must guarantee there is no BlockCacheEntry for this id.
    void recycle(BlockId id);

    int ref_count(BlockId id) const;
    int num_free_blocks() const noexcept;
    int max_blocks() const noexcept { return max_blocks_; }
    int block_size() const noexcept { return block_size_; }

private:
    std::unique_ptr<Block[]> blocks_;
    FreeList free_list_;
    int max_blocks_ = 0;
    int block_size_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace ccinfer
