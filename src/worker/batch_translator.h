#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "backend/backend.h"
#include "cache/block.h"
#include "common/error_code.h"
#include "common/physical_batch.h"
#include "common/types.h"
#include "worker/sequence_state.h"

namespace ccinfer {

class KVCacheManager;

// Prepares a scheduled batch for one physical execution step. Allocates
// KV-cache blocks via KVCacheManager, builds GPU-ready buffers, and records the
// logical state changes that become visible only after successful execution.
//
class BatchTranslator {
public:
    BatchTranslator(Backend& backend, KVCacheManager& kv_mgr, int block_size);

    enum class ExecutionPlanStatus : uint8_t { Prepared, Committed, RolledBack };

    // Per-item allocation metadata.  One entry per WorkItem, in batch order.
    struct PerItemAlloc {
        BlockTable new_blocks;              // blocks allocated by this translation
        std::vector<int32_t> slot_mapping;  // per-token physical slot [new_tokens]
        int kv_tokens_to_commit = 0;
        int prompt_tokens_to_commit = 0;
        bool deferred = false;        // item excluded from this physical batch
        bool force_no_write = false;  // decode ran but did not write KV
    };

    class BatchExecutionPlan {
    public:
        BatchExecutionPlan(const BatchExecutionPlan&) = delete;
        BatchExecutionPlan& operator=(const BatchExecutionPlan&) = delete;
        BatchExecutionPlan(BatchExecutionPlan&& other) noexcept;
        BatchExecutionPlan& operator=(BatchExecutionPlan&& other) noexcept;
        ~BatchExecutionPlan();

        PhysicalBatch physical_batch;

        // Original item indices that were excluded because no KV block was
        // available. The caller must synthesize deferred WorkItemResults for
        // these items.
        std::vector<std::size_t> deferred_indices;

        // Per original item: true when a decode was executed without writing KV.
        std::vector<bool> no_write_flags;

        // Makes the logical SequenceState changes visible after a successful
        // model execution. A plan that is not committed rolls back its newly
        // allocated KV blocks on destruction.
        void commit();

        // Explicitly abandons the plan. Safe to call more than once.
        void rollback() noexcept;

        ExecutionPlanStatus status() const noexcept { return status_; }

    private:
        friend class BatchTranslator;

        BatchExecutionPlan(BatchTranslator* translator, ScheduledBatch batch,
                           std::unordered_map<SequenceId, SequenceState>* sequences,
                           PhysicalBatch physical_batch, std::vector<PerItemAlloc> per_item);

        BatchTranslator* translator_ = nullptr;
        ScheduledBatch batch_;
        std::unordered_map<SequenceId, SequenceState>* sequences_ = nullptr;
        std::vector<PerItemAlloc> per_item_;
        ExecutionPlanStatus status_ = ExecutionPlanStatus::Prepared;
    };

    // Build a BatchExecutionPlan from the scheduled batch and current
    // SequenceState. The plan owns the transaction and is committed only after
    // the caller's physical execution succeeds.
    Result<BatchExecutionPlan> prepare(const ScheduledBatch& batch,
                                       std::unordered_map<SequenceId, SequenceState>& sequences);

private:
    struct AllocationPlan {
        std::vector<PerItemAlloc> per_item;
        int total_tokens = 0;
        int batch_size = 0;
        int max_blocks_per_req = 0;
    };

    Result<AllocationPlan> allocate_blocks(
        const ScheduledBatch& batch,
        const std::unordered_map<SequenceId, SequenceState>& sequences) const;
    Result<PhysicalBatch> build_physical_batch(
        const ScheduledBatch& batch, const std::unordered_map<SequenceId, SequenceState>& sequences,
        const AllocationPlan& plan) const;
    BatchExecutionPlan make_plan(ScheduledBatch batch,
                                 std::unordered_map<SequenceId, SequenceState>* sequences,
                                 AllocationPlan&& plan, PhysicalBatch physical_batch);
    static void assert_item_matches_state(const WorkItem& item, const SequenceState& seq);

    void commit_plan(const ScheduledBatch& batch,
                     std::unordered_map<SequenceId, SequenceState>& sequences,
                     const std::vector<PerItemAlloc>& per_item) const;
    void rollback_allocations(const std::vector<PerItemAlloc>& per_item) const noexcept;

    Backend& backend_;
    KVCacheManager& kv_mgr_;
    int block_size_;
};

}  // namespace ccinfer
