#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "base/types.h"
#include "cache/block.h"
#include "core/tensor.h"

namespace ccinfer {

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

// SequenceState — worker-local execution view used by BatchTranslator.
// Logical fields are copied from Executor-owned SequenceSnapshot; block_table
// is worker/device-local metadata owned by the Worker.
struct SequenceState {
    SequenceId seq_id = 0;
    std::vector<int32_t> prompt_tokens;
    int max_context_len = 0;
    int kv_written = 0;        // tokens already written into KV cache
    int prompt_processed = 0;  // tokens already consumed from prompt
    BlockTable block_table;
    bool aborted = false;
};

// PhysicalBatch — GPU-ready data for ModelRunner::inference.
struct PhysicalBatch {
    int num_tokens = 0;
    Tensor token_ids;     // [num_tokens]
    Tensor positions;     // [num_tokens]
    Tensor slot_mapping;  // [num_tokens]

    int batch_size = 0;
    int max_blocks_per_req = 0;
    Tensor block_table;  // [batch_size, max_blocks_per_req]

    // Prefill: query_start_loc_[i] = cumulative token offset for request i,
    //          shape [batch_size + 1], last element = total_tokens.
    // Decode:  [0, 1, ..., batch_size], shape [batch_size + 1].
    Tensor query_start_loc;  // [batch_size + 1]
    Tensor context_lens;     // [batch_size]

    // logits_indices_[i] is the logits row sampled for request i.
    // Prefill: query_start_loc[i + 1] - 1 (last token of the chunk).
    // Decode:  i (the single token just computed).
    Tensor logits_indices;  // [batch_size], int32

    std::vector<std::size_t> item_indices;  // maps physical seq → WorkItem index
    ForwardMode mode = ForwardMode::Prefill;
    std::vector<SequenceId> item_seq_ids;
    std::vector<WorkKind> item_kinds;
};

}  // namespace ccinfer
