#include "cache/kv_cache_manager.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <unordered_set>
#include <utility>

#include "cache/cache_stats.h"
#include "cache/kv_cache_storage.h"
#include "cache/prefix_cache.h"
#include "common/error_code.h"
#include "facade/log.h"

namespace ccinfer {

// Lifecycle (complete types available here).

KVCacheManager::KVCacheManager() = default;

KVCacheManager::~KVCacheManager() {
    // Unlink all intrusive-list hooks before the Block array is destroyed.
    free_list_.clear();
    lru_list_.clear();
}

// Init.

Result<void> KVCacheManager::init(std::unique_ptr<KVCacheStorage> storage, int max_blocks,
                                  int block_size) {
    if (max_blocks_ != 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (!storage) return std::unexpected(ErrorCode::InvalidArgument);
    if (max_blocks <= 0 || block_size <= 0) return std::unexpected(ErrorCode::InvalidArgument);

    int64_t expected_slots = static_cast<int64_t>(max_blocks) * block_size;
    if (expected_slots <= 0 || expected_slots > std::numeric_limits<int>::max())
        return std::unexpected(ErrorCode::InvalidArgument);
    if (storage->max_slots() != static_cast<int>(expected_slots))
        return std::unexpected(ErrorCode::InvalidArgument);

    kv_storage_ = std::move(storage);
    max_blocks_ = max_blocks;
    block_size_ = block_size;
    metadata_ = std::make_unique<Block[]>(max_blocks);

    prefix_cache_ = std::make_unique<PrefixCache>();

    for (int i = 0; i < max_blocks; ++i) {
        metadata_[i].block_id = i;
        metadata_[i].ref_count = 0;
        metadata_[i].set_flag(BlockFlags::kInFreeList);
        free_list_.push_back(metadata_[i]);
    }

    return {};
}

// Data views.

Tensor KVCacheManager::k_cache(int layer) { return kv_storage_->k_layer_tensor(layer); }
Tensor KVCacheManager::v_cache(int layer) { return kv_storage_->v_layer_tensor(layer); }
Tensor KVCacheManager::k_cache_blocks(int layer) { return kv_storage_->k_block_tensor(layer); }
Tensor KVCacheManager::v_cache_blocks(int layer) { return kv_storage_->v_block_tensor(layer); }

int KVCacheManager::max_slots() const { return kv_storage_->max_slots(); }

// Block allocation.

Result<BlockTable> KVCacheManager::allocate_blocks(int num_blocks) {
    if (num_blocks <= 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (num_blocks > max_blocks_) return std::unexpected(ErrorCode::KVBlockExhausted);

    // Only evict when the remaining free + cacheable LRU blocks can satisfy the
    // request.  Avoids destroying prefix-cache entries on a failed allocation.
    const int available = static_cast<int>(free_list_.size()) + static_cast<int>(lru_list_.size());
    if (available < num_blocks) return std::unexpected(ErrorCode::KVBlockExhausted);

    while (static_cast<int>(free_list_.size()) < num_blocks) {
        if (!evict_one()) return std::unexpected(ErrorCode::KVBlockExhausted);
    }

    BlockTable result;
    for (int b = 0; b < num_blocks; ++b) {
        auto& block = free_list_.front();
        assert(block.is_free() && block.ref_count == 0 && !block.is_cached() &&
               !block.is_in_lru() && block.block_hash == 0);
        free_list_.pop_front();

        block.flags = static_cast<uint32_t>(BlockFlags::kNone);
        block.ref_count = 1;
        block.block_hash = 0;

        result.push_back(block.block_id);
    }
    return result;
}

KVCacheManager::PrepareResult KVCacheManager::lookup_prefix_cache(
    const std::vector<int32_t>& tokens, uint64_t namespace_salt) {
    assert(!tokens.empty());
    assert(block_size_ > 0);

    const int64_t total_tokens = static_cast<int64_t>(tokens.size());
    const int64_t blocks_needed = (total_tokens + block_size_ - 1) / block_size_;
    assert(blocks_needed <= max_blocks_);

    PrepareResult result;
    auto& table = result.block_table;

    auto hashes = PrefixCache::chain_hashes(tokens, block_size_, namespace_salt);
    if (!hashes.empty()) result.parent_hash = hashes.back();
    const std::size_t full_tokens =
        static_cast<std::size_t>(tokens.size() / static_cast<std::size_t>(block_size_)) *
        static_cast<std::size_t>(block_size_);
    result.pending_tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(full_tokens),
                                 tokens.end());

    for (std::size_t i = 0; i < hashes.size(); ++i) {
        auto opt_id = prefix_cache_->lookup(hashes[i]);
        if (!opt_id) break;

        const int32_t block_id = *opt_id;
        assert(block_id >= 0 && block_id < max_blocks_);

        auto& block = metadata_[block_id];
        assert(block.is_cached() && !block.is_free() && block.block_hash == hashes[i]);

        if (block.is_cached_idle()) {
            // CACHED_IDLE → ACTIVE_CACHED: remove from LRU, bump ref.
            lru_list_.erase(LruList::s_iterator_to(block));
            block.clear_flag(BlockFlags::kInLRU);
            block.ref_count = 1;
        } else {
            assert(!block.is_in_lru() && block.ref_count > 0);
            // ACTIVE_CACHED — already referenced, just bump ref.
            block.ref_count++;
        }
        table.push_back(block_id);
        result.prefix_hit_blocks++;
    }

    table.set_shared_count(result.prefix_hit_blocks);
    return result;
}

uint64_t KVCacheManager::cache_rolling_blocks(uint64_t parent_hash,
                                              const std::vector<int32_t>& pending_tokens,
                                              const std::vector<int32_t>& block_ids,
                                              const std::vector<int32_t>& table_indices,
                                              BlockTable& table, uint64_t seed) {
    assert(block_size_ > 0);
    assert(block_ids.size() == table_indices.size());
    assert(block_ids.size() * static_cast<std::size_t>(block_size_) <= pending_tokens.size());

    auto hashes = PrefixCache::chain_hashes(pending_tokens, static_cast<int>(pending_tokens.size()),
                                            block_size_, parent_hash, seed);
    assert(hashes.size() >= block_ids.size());

    uint64_t hash = parent_hash;
    for (std::size_t i = 0; i < block_ids.size(); ++i) {
        hash = hashes[i];
        const int32_t block_id = block_ids[i];
        const int32_t table_index = table_indices[i];
        assert(block_id >= 0 && block_id < max_blocks_);
        assert(table_index >= 0 && table_index < table.size());
        assert(table[table_index] == block_id);

        auto& block = metadata_[block_id];
        assert(block.ref_count > 0 && !block.is_free() && !block.is_in_lru());

        if (block.is_cached()) {
            assert(block.block_hash == hash);
            continue;
        }

        auto existing = prefix_cache_->find(hash);
        if (existing.has_value()) {
            // Same content is already cached: treat this full block as a late
            // cache hit. Release the duplicate block and point this sequence's
            // block table at the canonical cached block.
            const int32_t canonical = *existing;
            assert(canonical >= 0 && canonical < max_blocks_);
            assert(canonical != block_id && "cached hash must not point at an uncached block");

            auto& canonical_block = metadata_[canonical];
            assert(canonical_block.is_cached() && canonical_block.block_hash == hash);
            if (canonical_block.is_cached_idle()) {
                lru_list_.erase(LruList::s_iterator_to(canonical_block));
                canonical_block.clear_flag(BlockFlags::kInLRU);
                canonical_block.ref_count = 1;
            } else {
                assert(!canonical_block.is_in_lru() && canonical_block.ref_count > 0);
                canonical_block.ref_count++;
            }

            block.ref_count--;
            if (block.ref_count == 0) {
                block.flags = static_cast<uint32_t>(BlockFlags::kInFreeList);
                block.block_hash = 0;
                free_list_.push_back(block);
            }
            table.set(table_index, canonical);
            continue;
        }

        auto insert_r = prefix_cache_->insert(hash, block_id);
        assert(insert_r);

        block.set_flag(BlockFlags::kCached);
        block.block_hash = hash;
        ccLog::debug("prefix insert block={} hash={:x}", block_id, hash);
    }
    return hash;
}

Result<void> KVCacheManager::release_blocks(const BlockTable& table) {
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
        assert(!block.is_in_lru());
        if (block.is_free()) return std::unexpected(ErrorCode::KVBlockDoubleFree);
        if (block.ref_count <= 0) return std::unexpected(ErrorCode::KVBlockDoubleFree);
    }

    for (int i = 0; i < table.size(); ++i) {
        int32_t block_id = table[i];
        auto& block = metadata_[block_id];

        block.ref_count--;
        if (block.ref_count == 0) {
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
    return {};
}

bool KVCacheManager::evict_one() {
    if (lru_list_.empty()) return false;

    auto& block = lru_list_.front();
    if (!block.is_cached_idle()) {
        ccLog::warn("evict_one: block {} not cached_idle, flags={} ref={}", block.block_id,
                    block.flags, block.ref_count);
        return false;
    }
    lru_list_.pop_front();

    prefix_cache_->record_eviction();
    prefix_cache_->remove_by_block(block.block_id);

    block.clear_flag(BlockFlags::kCached);
    block.clear_flag(BlockFlags::kInLRU);
    block.set_flag(BlockFlags::kInFreeList);
    block.block_hash = 0;
    free_list_.push_back(block);

    ccLog::info("lru evict block={}", block.block_id);
    return true;
}

void KVCacheManager::rollback_prefix_hits(const BlockTable& table, int count) {
    for (int b = 0; b < count; ++b) {
        auto& blk = metadata_[table[b]];
        if (!blk.is_cached() || blk.ref_count <= 0) {
            ccLog::error("rollback_prefix_hits: block {} not cached/active, flags={} ref={}",
                         blk.block_id, blk.flags, blk.ref_count);
            continue;
        }
        blk.ref_count--;
        if (blk.ref_count == 0) {
            if (blk.is_in_lru()) {
                ccLog::error("rollback_prefix_hits: block {} already in LRU", blk.block_id);
            } else {
                blk.set_flag(BlockFlags::kInLRU);
                lru_list_.push_back(blk);
            }
        }
    }
}

KVCacheStats KVCacheManager::stats() const {
    KVCacheStats s;
    s.block_total = max_blocks_;
    s.block_size = block_size_;
    s.block_cached_idle = static_cast<int>(lru_list_.size());
    s.block_free = static_cast<int>(free_list_.size());
    s.block_active = max_blocks_ - s.block_free - s.block_cached_idle;
    s.prefix = prefix_cache_ ? prefix_cache_->stats() : PrefixCacheStats{};
    return s;
}

PrefixCacheStats KVCacheManager::prefix_stats() const {
    return prefix_cache_ ? prefix_cache_->stats() : PrefixCacheStats{};
}

}  // namespace ccinfer
