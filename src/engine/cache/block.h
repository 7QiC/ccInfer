#pragma once

#include <boost/intrusive/list.hpp>
#include <cstdint>
#include <vector>

namespace ccinfer {
namespace engine {

constexpr int kKVBlockSize = 16;

enum class BlockFlags : uint32_t {
    kNone = 0,
    kInFreeList = 1 << 0,
    kCached = 1 << 1,
    kInLRU = 1 << 2,
};

inline constexpr bool has_flag(uint32_t flags, BlockFlags f) noexcept {
    return (flags & static_cast<uint32_t>(f)) != 0;
}

// Block state machine (4 canonical states):
//   FREE          — kInFreeList,           ref_count=0
//   ACTIVE        — no flags,              ref_count>0
//   ACTIVE_CACHED — kCached,               ref_count>0
//   CACHED_IDLE   — kCached | kInLRU,      ref_count=0
//
// Blocks live in fixed-address storage owned by KVCacheManager, such as
// std::unique_ptr<Block[]>.
// They must never be moved after intrusive-list hooks are inserted.

struct Block {
    int32_t block_id = -1;
    int32_t ref_count = 0;
    uint64_t block_hash = 0;
    uint32_t flags = static_cast<uint32_t>(BlockFlags::kNone);

    boost::intrusive::list_member_hook<> free_hook;
    boost::intrusive::list_member_hook<> lru_hook;

    // Immovable — intrusive-list hooks require stable addresses.
    // Owning container must pre-allocate capacity (e.g. std::unique_ptr<Block[]>)
    // and never reallocate after hooks are inserted.
    Block() = default;
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&&) = delete;
    Block& operator=(Block&&) = delete;

    // ---- Queries ----
    bool is_free() const noexcept { return has_flag(flags, BlockFlags::kInFreeList); }
    bool is_cached() const noexcept { return has_flag(flags, BlockFlags::kCached); }
    bool is_in_lru() const noexcept { return has_flag(flags, BlockFlags::kInLRU); }
    bool is_active() const noexcept { return ref_count > 0; }
    bool is_cached_idle() const noexcept { return is_cached() && is_in_lru() && ref_count == 0; }

    // ---- Flag mutations ----
    void set_flag(BlockFlags f) noexcept { flags |= static_cast<uint32_t>(f); }
    void clear_flag(BlockFlags f) noexcept { flags &= ~static_cast<uint32_t>(f); }
};

using FreeList = boost::intrusive::list<
    Block,
    boost::intrusive::member_hook<Block, boost::intrusive::list_member_hook<>, &Block::free_hook>>;

using LruList = boost::intrusive::list<
    Block,
    boost::intrusive::member_hook<Block, boost::intrusive::list_member_hook<>, &Block::lru_hook>>;

class BlockTable {
public:
    BlockTable() = default;

    void push_back(int32_t block_id) { block_ids_.push_back(block_id); }
    void set_shared_count(int32_t n) { shared_count_ = n; }

    int32_t operator[](int i) const { return block_ids_[i]; }
    int32_t size() const { return static_cast<int32_t>(block_ids_.size()); }
    bool empty() const { return block_ids_.empty(); }
    void clear() {
        block_ids_.clear();
        shared_count_ = 0;
    }

    int32_t shared_count() const { return shared_count_; }
    const int32_t* data() const { return block_ids_.data(); }
    const std::vector<int32_t>& ids() const { return block_ids_; }

    int64_t token_capacity(int32_t block_size) const {
        return static_cast<int64_t>(size()) * block_size;
    }

private:
    std::vector<int32_t> block_ids_;
    // Number of prefix-hit shared blocks at the front of this table.
    // release_blocks() determines block lifetime via ref_count, not shared_count_.
    int32_t shared_count_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
