#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "base/result.h"
#include "base/types.h"
#include "cache/block.h"
#include "backend/default_backend.h"
#include "base/execution.h"

namespace ccinfer {

class KVCacheManager;

// Translates a ScheduledBatch (WorkItems with CPU metadata) into a
// PhysicalBatch (GPU-ready buffers).  Allocates KV-cache blocks via
// KVCacheManager and computes per-token slot mappings.
//
// Transactional semantics:
//   translate()  — allocate blocks, build PhysicalBatch, record pending changes
//   commit()     — update SequenceState (kv_written, prompt_processed, block_table)
//   rollback()   — release newly allocated blocks, SequenceState unchanged
class BatchTranslator {
public:
    BatchTranslator(DefaultBackend& backend, KVCacheManager& kv_mgr, int block_size);

    // Per-item allocation metadata.  One entry per WorkItem, in batch order.
    struct PerItemAlloc {
        BlockTable new_blocks;              // blocks allocated by this translation
        std::vector<int32_t> slot_mapping;  // per-token physical slot [new_tokens]
        int kv_tokens_to_commit = 0;
        int prompt_tokens_to_commit = 0;
        bool release_after_forward = false;
        bool prefix_cache_bootstrap = false;
    };

    struct TranslateResult {
        PhysicalBatch physical_batch;
        std::vector<PerItemAlloc> per_item;
    };

    // Build a GPU-ready PhysicalBatch from the scheduled batch and current
    // SequenceState.  Allocates additional KV blocks as needed.
    Result<TranslateResult> translate(
        ScheduledBatch& batch,
        const std::unordered_map<SequenceId, SequenceState>& sequences);

    // Persist changes to SequenceState after a successful forward pass.
    // Returns InvalidArgument if per_item size does not match batch items
    // or a referenced sequence is not found.
    Result<void> commit(const ScheduledBatch& batch,
                        std::unordered_map<SequenceId, SequenceState>& sequences,
                        const std::vector<PerItemAlloc>& per_item) const;

    // Release blocks allocated by translate() after a failed forward pass.
    void rollback(const std::vector<PerItemAlloc>& per_item) const;

private:
    DefaultBackend& backend_;
    KVCacheManager& kv_mgr_;
    int block_size_;
};

}  // namespace ccinfer
