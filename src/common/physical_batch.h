#pragma once

#include <cstdint>
#include <vector>

#include "common/types.h"
#include "core/tensor.h"

namespace ccinfer {

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

    // logits_indices_[i] is the compact row into output.logits sampled for
    // request i; -1 means no sampling.
    Tensor logits_indices;  // [batch_size], int32

    ForwardMode mode = ForwardMode::Prefill;
    std::vector<SequenceId> item_seq_ids;
    std::vector<WorkKind> item_kinds;

    // CPU-side per-item metadata used by ModelRunner to build WorkItemResults,
    // avoiding D2H round-trips of data the worker already produced.
    std::vector<int32_t> item_token_counts;  // [batch_size] tokens per physical seq
    std::vector<bool> sample_flags;          // [batch_size] sample this seq's logits

    // Original row indices into the prefill/decode hidden tensor that need
    // logits; used by Qwen3Model to compute only the sampled rows.
    std::vector<int32_t> logits_rows_host;
    int num_logits = 0;

    int max_position_id = 0;
};

}  // namespace ccinfer
