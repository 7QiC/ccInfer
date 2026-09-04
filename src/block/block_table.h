#pragma once

#include <cstdint>
#include <vector>

namespace ccinfer {

// Logical KV block table shared by the Scheduler, Worker, and BlockPool. The
// BlockPool owns block metadata and reference counts; this table is the
// scheduler-owned mirror passed with each work item. Kept in its own header so
// base/types.h and block/block_pool.h can use it without pulling in the rest
// of either module.
//
// WorkItem stores this by value as a snapshot. The scheduler may append blocks
// to the canonical table while an earlier batch for the same sequence is still
// in flight, so a reference/shared_ptr into scheduler state would race with the
// worker thread. Do not "optimize" this into a shared view.
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

}  // namespace ccinfer
