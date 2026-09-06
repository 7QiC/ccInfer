#include "block/block_pool.h"

#include <cassert>
#include <limits>

namespace ccinfer {

BlockPool::BlockPool(int max_blocks, int block_size) {
    auto result = init(max_blocks, block_size);
    assert(result);
}

BlockPool::~BlockPool() = default;

Result<void> BlockPool::init(int max_blocks, int block_size) {
    assert(max_blocks_ == 0);
    if (max_blocks <= 0 || block_size <= 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (static_cast<int64_t>(max_blocks) * block_size > std::numeric_limits<int>::max())
        return std::unexpected(ErrorCode::InvalidArgument);

    max_blocks_ = max_blocks;
    block_size_ = block_size;
    blocks_ = std::make_unique<Block[]>(max_blocks_);
    for (int i = 0; i < max_blocks_; ++i) {
        blocks_[i].id = i;
        blocks_[i].status = BlockStatus::Free;
        free_list_.push_back(blocks_[i]);
    }
    return {};
}

Result<BlockId> BlockPool::allocate() {
    std::lock_guard lock(mutex_);
    if (free_list_.empty()) return std::unexpected(ErrorCode::KVBlockExhausted);
    auto& block = free_list_.front();
    assert(block.is_free() && block.ref_count == 0);
    free_list_.pop_front();
    block.status = BlockStatus::InUse;
    block.ref_count = 1;
    return block.id;
}

Result<BlockTable> BlockPool::allocate_blocks(int num_blocks) {
    assert(num_blocks > 0);
    BlockTable table;
    for (int i = 0; i < num_blocks; ++i) {
        auto id = allocate();
        if (!id) return std::unexpected(id.error());
        table.push_back(*id);
    }
    return table;
}

void BlockPool::retain_request(BlockId id) {
    std::lock_guard lock(mutex_);
    assert(id >= 0 && id < max_blocks_);
    auto& block = blocks_[id];
    assert(block.is_in_use() && block.ref_count > 0);
    ++block.ref_count;
}

int BlockPool::release_request(BlockId id) {
    std::lock_guard lock(mutex_);
    assert(id >= 0 && id < max_blocks_);
    auto& block = blocks_[id];
    assert(block.is_in_use() && block.ref_count > 0);
    --block.ref_count;
    return block.ref_count;
}

void BlockPool::release_blocks(const BlockTable& table) {
    for (int i = 0; i < table.size(); ++i) (void)release_request(table[i]);
}

void BlockPool::recycle(BlockId id) {
    std::lock_guard lock(mutex_);
    assert(id >= 0 && id < max_blocks_);
    auto& block = blocks_[id];
    assert(block.is_in_use() && block.ref_count == 0);
    block.status = BlockStatus::Free;
    free_list_.push_back(block);
}

int BlockPool::ref_count(BlockId id) const {
    std::lock_guard lock(mutex_);
    assert(id >= 0 && id < max_blocks_);
    return blocks_[id].ref_count;
}

int BlockPool::num_free_blocks() const noexcept {
    std::lock_guard lock(mutex_);
    return static_cast<int>(free_list_.size());
}

}  // namespace ccinfer
