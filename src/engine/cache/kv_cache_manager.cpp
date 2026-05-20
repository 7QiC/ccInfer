#include "engine/cache/kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>

#include "common/error_code.h"
#include "engine/cache/cache_stats.h"
#include "engine/cache/kv_cache_storage.h"
#include "engine/cache/prefix_cache.h"
#include "spdlog/spdlog.h"

namespace ccinfer {
namespace engine {

// ---- Lifecycle (complete types available here) ----

KVCacheManager::KVCacheManager() = default;

KVCacheManager::~KVCacheManager() {
    // Unlink all intrusive-list hooks before the Block array is destroyed.
    free_list_.clear();
    lru_list_.clear();
}

// ---- Init ----

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

// ---- Data pointers ----

void* KVCacheManager::k_cache(int layer) { return kv_storage_->k_layer(layer); }
void* KVCacheManager::v_cache(int layer) { return kv_storage_->v_layer(layer); }
const void* KVCacheManager::k_cache(int layer) const { return kv_storage_->k_layer(layer); }
const void* KVCacheManager::v_cache(int layer) const { return kv_storage_->v_layer(layer); }

int KVCacheManager::max_slots() const { return kv_storage_->max_slots(); }

// ---- Block allocation ----

Result<BlockTable> KVCacheManager::allocate_blocks(int num_blocks) {
    if (num_blocks <= 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (num_blocks > max_blocks_) return std::unexpected(ErrorCode::KVBlockExhausted);

    while (static_cast<int>(free_list_.size()) < num_blocks) {
        if (!evict_one()) return std::unexpected(ErrorCode::KVBlockExhausted);
    }

    BlockTable result;
    for (int b = 0; b < num_blocks; ++b) {
        auto& block = free_list_.front();
        if (!block.is_free() || block.ref_count != 0 || block.is_cached() || block.is_in_lru() ||
            block.block_hash != 0)
            return std::unexpected(ErrorCode::InternalError);
        free_list_.pop_front();

        block.flags = static_cast<uint32_t>(BlockFlags::kNone);
        block.ref_count = 1;
        block.block_hash = 0;

        result.push_back(block.block_id);
    }
    return result;
}

Result<KVCacheManager::PrepareResult> KVCacheManager::prepare_blocks(
    const std::vector<int32_t>& tokens, uint64_t namespace_salt) {
    if (tokens.empty()) return std::unexpected(ErrorCode::InvalidArgument);

    PrepareResult result;
    auto& table = result.block_table;

    if (tokens.size() > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
        return std::unexpected(ErrorCode::InvalidArgument);
    int64_t total_tokens = static_cast<int64_t>(tokens.size());
    int64_t blocks64 = (total_tokens + block_size_ - 1) / block_size_;
    if (blocks64 > static_cast<int64_t>(max_blocks_))
        return std::unexpected(ErrorCode::KVBlockExhausted);
    if (blocks64 > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return std::unexpected(ErrorCode::InvalidArgument);
    int total_blocks_needed = static_cast<int>(blocks64);
    if (total_blocks_needed > max_blocks_) return std::unexpected(ErrorCode::KVBlockExhausted);

    // Phase 1: match prefix cache.
    auto hashes = PrefixCache::chain_hashes(tokens, block_size_, namespace_salt);
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        auto opt_id = prefix_cache_->lookup(hashes[i]);
        if (!opt_id) break;

        int32_t block_id = *opt_id;
        if (block_id < 0 || block_id >= max_blocks_) {
            rollback_prefix_hits(table, result.prefix_hit_blocks);
            return std::unexpected(ErrorCode::KVInvalidBlockTable);
        }

        auto& block = metadata_[block_id];
        if (!block.is_cached() || block.is_free() || block.block_hash != hashes[i]) {
            rollback_prefix_hits(table, result.prefix_hit_blocks);
            return std::unexpected(ErrorCode::InternalError);
        }

        if (block.is_cached_idle()) {
            // CACHED_IDLE → ACTIVE_CACHED: remove from LRU, bump ref.
            lru_list_.erase(LruList::s_iterator_to(block));
            block.clear_flag(BlockFlags::kInLRU);
            block.ref_count = 1;
        } else if (!block.is_in_lru() && block.ref_count > 0) {
            // ACTIVE_CACHED — already referenced, just bump ref.
            block.ref_count++;
        } else {
            rollback_prefix_hits(table, result.prefix_hit_blocks);
            return std::unexpected(ErrorCode::InternalError);
        }
        table.push_back(block_id);
        result.prefix_hit_blocks++;
    }

    // Phase 2: allocate remaining blocks. Rollback prefix hits on failure.
    int remaining = total_blocks_needed - table.size();
    if (remaining > 0) {
        auto alloc = allocate_blocks(remaining);
        if (!alloc) {
            rollback_prefix_hits(table, result.prefix_hit_blocks);
            return std::unexpected(alloc.error());
        }
        for (int b = 0; b < alloc->size(); ++b) {
            table.push_back((*alloc)[b]);
        }
    }

    table.set_shared_count(result.prefix_hit_blocks);
    return result;
}

Result<void> KVCacheManager::cache_full_blocks(const BlockTable& table,
                                               const std::vector<int32_t>& tokens,
                                               int committed_tokens, uint64_t namespace_salt) {
    // Only hash tokens whose KV has been written.
    if (committed_tokens < 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (tokens.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return std::unexpected(ErrorCode::InvalidArgument);
    if (committed_tokens > static_cast<int>(tokens.size()))
        return std::unexpected(ErrorCode::InvalidArgument);
    int64_t capacity = table.token_capacity(block_size_);
    if (committed_tokens > capacity) return std::unexpected(ErrorCode::InvalidArgument);

    auto hashes = PrefixCache::chain_hashes(tokens, committed_tokens, block_size_, namespace_salt);
    int num_full = std::min(static_cast<int>(hashes.size()), table.size());

    for (int i = table.shared_count(); i < num_full; ++i) {
        int32_t block_id = table[i];
        if (block_id < 0 || block_id >= max_blocks_)
            return std::unexpected(ErrorCode::KVInvalidBlockTable);

        auto& block = metadata_[block_id];
        if (block.ref_count <= 0) return std::unexpected(ErrorCode::InternalError);
        if (block.is_free()) return std::unexpected(ErrorCode::InternalError);
        if (block.is_in_lru()) return std::unexpected(ErrorCode::InternalError);

        if (block.is_cached()) {
            // Already cached — verify hash consistency.
            if (block.block_hash != hashes[i]) return std::unexpected(ErrorCode::InternalError);
            continue;
        }

        auto r = prefix_cache_->insert(hashes[i], block_id);
        if (!r) return std::unexpected(r.error());

        block.set_flag(BlockFlags::kCached);
        block.block_hash = hashes[i];
        spdlog::info("prefix insert block={} hash={:x}", block_id, hashes[i]);
    }
    return {};
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
        if (block.is_free()) return std::unexpected(ErrorCode::KVBlockDoubleFree);
        if (block.is_in_lru()) return std::unexpected(ErrorCode::InternalError);
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
        spdlog::warn("evict_one: block {} not cached_idle, flags={} ref={}", block.block_id,
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

    spdlog::info("lru evict block={}", block.block_id);
    return true;
}

void KVCacheManager::rollback_prefix_hits(const BlockTable& table, int count) {
    for (int b = 0; b < count; ++b) {
        auto& blk = metadata_[table[b]];
        if (!blk.is_cached() || blk.ref_count <= 0) {
            spdlog::error("rollback_prefix_hits: block {} not cached/active, flags={} ref={}",
                          blk.block_id, blk.flags, blk.ref_count);
            continue;
        }
        blk.ref_count--;
        if (blk.ref_count == 0) {
            if (blk.is_in_lru()) {
                spdlog::error("rollback_prefix_hits: block {} already in LRU", blk.block_id);
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

}  // namespace engine
}  // namespace ccinfer
