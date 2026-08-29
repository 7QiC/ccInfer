#include "cache/block_pool.h"

#include <cassert>
#include <limits>
#include <unordered_set>

#include "facade/log.h"

namespace ccinfer {

BlockPool::BlockPool(int max_blocks, int block_size) {
    auto result = init(max_blocks, block_size);
    assert(result);
}

BlockPool::~BlockPool() {
    free_list_.clear();
    lru_list_.clear();
}

Result<void> BlockPool::init(int max_blocks, int block_size) {
    assert(max_blocks_ == 0);
    if (max_blocks <= 0 || block_size <= 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (static_cast<int64_t>(max_blocks) * block_size > std::numeric_limits<int>::max())
        return std::unexpected(ErrorCode::InvalidArgument);

    max_blocks_ = max_blocks;
    block_size_ = block_size;
    metadata_ = std::make_unique<Block[]>(max_blocks_);
    for (int i = 0; i < max_blocks_; ++i) {
        metadata_[i].block_id = i;
        metadata_[i].set_flag(BlockFlags::kInFreeList);
        free_list_.push_back(metadata_[i]);
    }
    return {};
}

Result<BlockTable> BlockPool::allocate_blocks(int num_blocks) {
    assert(num_blocks > 0);
    std::lock_guard lock(mutex_);
    if (num_blocks > max_blocks_) return std::unexpected(ErrorCode::KVBlockExhausted);

    if (static_cast<int>(free_list_.size()) + static_cast<int>(lru_list_.size()) < num_blocks)
        return std::unexpected(ErrorCode::KVBlockExhausted);
    while (static_cast<int>(free_list_.size()) < num_blocks) {
        if (!evict_one()) return std::unexpected(ErrorCode::KVBlockExhausted);
    }

    BlockTable result;
    for (int i = 0; i < num_blocks; ++i) {
        auto& block = free_list_.front();
        assert(block.is_free() && block.ref_count == 0 && !block.is_cached() &&
               !block.is_in_lru() && block.block_hash == 0);
        free_list_.pop_front();
        block.flags = static_cast<uint32_t>(BlockFlags::kNone);
        block.ref_count = 1;
        result.push_back(block.block_id);
    }
    return result;
}

void BlockPool::release_blocks(const BlockTable& table) {
    std::lock_guard lock(mutex_);
    std::unordered_set<int32_t> seen;
    for (int i = 0; i < table.size(); ++i) {
        const int32_t id = table[i];
        assert(id >= 0 && id < max_blocks_ && seen.insert(id).second);
        auto& block = metadata_[id];
        assert(!block.is_in_lru());
        assert(!block.is_free() && block.ref_count > 0);
    }
    (void)seen;
    for (int i = 0; i < table.size(); ++i) {
        auto& block = metadata_[table[i]];
        --block.ref_count;
        if (block.ref_count != 0) continue;
        if (block.is_cached()) {
            block.set_flag(BlockFlags::kInLRU);
            lru_list_.push_back(block);
        } else {
            block.flags = static_cast<uint32_t>(BlockFlags::kInFreeList);
            block.block_hash = 0;
            free_list_.push_back(block);
        }
    }
}

BlockPool::PrefixLookup BlockPool::lookup_prefix_cache(const std::vector<int32_t>& tokens,
                                                       uint64_t namespace_salt) {
    std::lock_guard lock(mutex_);
    assert(!tokens.empty() && block_size_ > 0);
    PrefixLookup result;
    const auto hashes = PrefixCache::chain_hashes(tokens, block_size_, namespace_salt);
    std::size_t hit_tokens = 0;
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        auto id = prefix_cache_.lookup(hashes[i]);
        if (!id) break;
        assert(*id >= 0 && *id < max_blocks_);
        auto& block = metadata_[*id];
        assert(block.is_cached() && !block.is_free() && block.block_hash == hashes[i]);
        if (block.is_cached_idle()) {
            lru_list_.erase(LruList::s_iterator_to(block));
            block.clear_flag(BlockFlags::kInLRU);
            block.ref_count = 1;
        } else {
            assert(!block.is_in_lru() && block.ref_count > 0);
            ++block.ref_count;
        }
        result.block_table.push_back(*id);
        result.parent_hash = hashes[i];
        ++result.prefix_hit_blocks;
        hit_tokens += static_cast<std::size_t>(block_size_);
    }
    result.block_table.set_shared_count(result.prefix_hit_blocks);
    result.pending_tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(hit_tokens),
                                 tokens.end());
    return result;
}

uint64_t BlockPool::publish_full_block(uint64_t parent_hash, const std::vector<int32_t>& tokens,
                                       int32_t block_id, uint64_t seed) {
    std::lock_guard lock(mutex_);
    assert(tokens.size() >= static_cast<std::size_t>(block_size_));
    assert(block_id >= 0 && block_id < max_blocks_);
    auto hashes = PrefixCache::chain_hashes(tokens, block_size_, block_size_, parent_hash, seed);
    assert(!hashes.empty());
    const uint64_t hash = hashes.front();
    auto& block = metadata_[block_id];
    assert(block.ref_count > 0 && !block.is_free() && !block.is_in_lru());
    if (block.is_cached()) {
        assert(block.block_hash == hash);
        return hash;
    }
    prefix_cache_.insert(hash, block_id);
    block.set_flag(BlockFlags::kCached);
    block.block_hash = hash;
    ccLog::debug("prefix insert block={} hash={:x}", block_id, hash);
    return hash;
}

bool BlockPool::evict_one() {
    if (lru_list_.empty()) return false;
    auto& block = lru_list_.front();
    assert(block.is_cached_idle());
    lru_list_.pop_front();
    prefix_cache_.record_eviction();
    prefix_cache_.remove_by_block(block.block_id);
    block.flags = static_cast<uint32_t>(BlockFlags::kInFreeList);
    block.block_hash = 0;
    free_list_.push_back(block);
    return true;
}

KVCacheStats BlockPool::stats() const {
    std::lock_guard lock(mutex_);
    KVCacheStats s;
    s.block_total = max_blocks_;
    s.block_size = block_size_;
    s.block_cached_idle = static_cast<int>(lru_list_.size());
    s.block_free = static_cast<int>(free_list_.size());
    s.block_active = max_blocks_ - s.block_free - s.block_cached_idle;
    s.prefix = prefix_cache_.stats();
    return s;
}

PrefixCacheStats BlockPool::prefix_stats() const {
    std::lock_guard lock(mutex_);
    return prefix_cache_.stats();
}

int BlockPool::num_free_blocks() const noexcept {
    std::lock_guard lock(mutex_);
    return static_cast<int>(free_list_.size());
}

}  // namespace ccinfer
