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

inline BlockFlags operator|(BlockFlags a, BlockFlags b) {
    return static_cast<BlockFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline bool operator&(BlockFlags a, uint32_t mask) {
    return (static_cast<uint32_t>(a) & mask) != 0;
}

struct Block {
    int32_t block_id = -1;
    int32_t ref_count = 0;
    uint64_t block_hash = 0;
    uint32_t flags = static_cast<uint32_t>(BlockFlags::kNone);

    boost::intrusive::list_member_hook<> free_hook;
    boost::intrusive::list_member_hook<> lru_hook;

    bool is_free() const { return flags & static_cast<uint32_t>(BlockFlags::kInFreeList); }
    bool is_cached() const { return flags & static_cast<uint32_t>(BlockFlags::kCached); }
};

using FreeList = boost::intrusive::list<
    Block, boost::intrusive::member_hook<Block, boost::intrusive::list_member_hook<>,
                                         &Block::free_hook>>;

using LruList = boost::intrusive::list<
    Block, boost::intrusive::member_hook<Block, boost::intrusive::list_member_hook<>,
                                         &Block::lru_hook>>;

class BlockTable {
public:
    BlockTable() = default;

    void push_back(int32_t block_id) { block_ids_.push_back(block_id); }
    void set_shared_count(int32_t n) { shared_count_ = n; }

    int32_t operator[](int i) const { return block_ids_[i]; }
    int32_t size() const { return static_cast<int32_t>(block_ids_.size()); }
    int32_t shared_count() const { return shared_count_; }

    const int32_t* data() const { return block_ids_.data(); }
    int32_t token_capacity() const { return size() * kKVBlockSize; }

private:
    std::vector<int32_t> block_ids_;
    int32_t shared_count_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
