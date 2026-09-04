#pragma once

#include <cstdint>

#include <boost/intrusive/list.hpp>

namespace ccinfer {

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
// Blocks live in fixed-address metadata storage owned by BlockPool, such as
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

    bool is_free() const noexcept { return has_flag(flags, BlockFlags::kInFreeList); }
    bool is_cached() const noexcept { return has_flag(flags, BlockFlags::kCached); }
    bool is_in_lru() const noexcept { return has_flag(flags, BlockFlags::kInLRU); }
    bool is_active() const noexcept { return ref_count > 0; }
    bool is_cached_idle() const noexcept { return is_cached() && is_in_lru() && ref_count == 0; }

    void set_flag(BlockFlags f) noexcept { flags |= static_cast<uint32_t>(f); }
    void clear_flag(BlockFlags f) noexcept { flags &= ~static_cast<uint32_t>(f); }
};

using FreeList = boost::intrusive::list<
    Block,
    boost::intrusive::member_hook<Block, boost::intrusive::list_member_hook<>, &Block::free_hook>>;

using LruList = boost::intrusive::list<
    Block,
    boost::intrusive::member_hook<Block, boost::intrusive::list_member_hook<>, &Block::lru_hook>>;

}  // namespace ccinfer
