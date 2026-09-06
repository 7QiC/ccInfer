#pragma once

#include <cstdint>

#include <boost/intrusive/list.hpp>

namespace ccinfer {

constexpr int kKVBlockSize = 16;

using BlockId = std::int32_t;
inline constexpr BlockId kInvalidBlockId = -1;

enum class BlockStatus : std::uint8_t {
    Free,
    InUse,
};

using FreeHook = boost::intrusive::list_member_hook<>;

// Resource metadata for one KV block. Block objects live permanently in
// BlockPool's fixed metadata storage and must never be moved after being
// linked into the free list.
//
// Cache ownership is intentionally absent: it is represented by a
// BlockCacheEntry in cache/, never by Block fields.
struct Block {
    BlockId id = kInvalidBlockId;
    int32_t ref_count = 0;  // request / sequence refs only; cache does not count
    BlockStatus status = BlockStatus::Free;

    FreeHook free_hook;

    Block() = default;
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&&) = delete;
    Block& operator=(Block&&) = delete;

    bool is_free() const noexcept { return status == BlockStatus::Free; }
    bool is_in_use() const noexcept { return status == BlockStatus::InUse; }
};

using FreeList = boost::intrusive::list<
    Block,
    boost::intrusive::member_hook<Block, boost::intrusive::list_member_hook<>, &Block::free_hook>>;

}  // namespace ccinfer
