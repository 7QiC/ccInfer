#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/types.h"
#include "cache/block.h"
#include "core/tensor.h"

namespace ccinfer {

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
    SequenceStatus status = SequenceStatus::Active;
    int32_t last_token = -1;
    int tokens_generated = 0;
    int max_tokens = 0;
    bool finished = false;
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

    // CPU-side per-item metadata used by ModelRunner to build WorkItemResults,
    // avoiding D2H round-trips of data BatchTranslator already produced.
    std::vector<int32_t> item_token_counts;  // [batch_size] tokens per physical seq
    std::vector<bool> sample_flags;          // [batch_size] sample this seq's logits
    int max_position_id = 0;
};

}  // namespace ccinfer
