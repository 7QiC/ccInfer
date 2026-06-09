#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/types.h"
#include "engine/backend/device_buffer.h"
#include "engine/cache/block.h"

namespace ccinfer {
namespace engine {

struct DeviceCapacity {
    int max_sequences = 0;
    int active_sequences = 0;
    int free_blocks = 0;
    int max_blocks = 0;
    int block_size = 0;
    int block_active = 0;
    int block_cached_idle = 0;
    uint64_t prefix_lookup_hits = 0;
    uint64_t prefix_lookup_misses = 0;
    uint64_t prefix_evictions = 0;
    uint64_t prefix_cached_blocks = 0;
};

// ---------------------------------------------------------------------------
// SequenceState — per-sequence state owned by the DeviceWorker.
// CUDA-free: uses only CPU-side types (BlockTable is CPU metadata).
// ---------------------------------------------------------------------------
struct SequenceState {
    SequenceId seq_id = 0;
    std::vector<int32_t> prompt_tokens;
    int max_context_len = 0;
    int kv_written = 0;        // tokens already written into KV cache
    int prompt_processed = 0;  // tokens already consumed from prompt
    BlockTable block_table;
    bool aborted = false;
};

// ---------------------------------------------------------------------------
// PhysicalBatch — GPU-ready data for ModelRunner::inference.
// ---------------------------------------------------------------------------
struct PhysicalBatch {
    int num_tokens = 0;
    std::unique_ptr<DeviceBuffer> token_ids;     // [num_tokens]
    std::unique_ptr<DeviceBuffer> positions;     // [num_tokens]
    std::unique_ptr<DeviceBuffer> slot_mapping;  // [num_tokens]

    int batch_size = 0;
    int max_blocks_per_req = 0;
    std::unique_ptr<DeviceBuffer> block_table;  // [batch_size, max_blocks_per_req]

    // Prefill: query_start_loc_[i] = cumulative token offset for request i,
    //          shape [batch_size + 1], last element = total_tokens.
    // Decode:  [0, 1, ..., batch_size], shape [batch_size + 1].
    std::unique_ptr<DeviceBuffer> query_start_loc;  // [batch_size + 1]
    std::unique_ptr<DeviceBuffer> context_lens;     // [batch_size]

    // logits_indices_[i] is the logits row sampled for request i.
    // Prefill: query_start_loc[i + 1] - 1 (last token of the chunk).
    // Decode:  i (the single token just computed).
    std::unique_ptr<DeviceBuffer> logits_indices;  // [batch_size], int32

    std::vector<std::size_t> item_indices;  // maps physical seq → WorkItem index
    ForwardMode mode = ForwardMode::Prefill;
    std::vector<SequenceId> item_seq_ids;
    std::vector<WorkKind> item_kinds;
};

}  // namespace engine
}  // namespace ccinfer
