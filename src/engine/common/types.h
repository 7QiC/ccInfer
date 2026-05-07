#pragma once

#include <cstdint>
#include <vector>

#include "common/types.h"
#include "engine/cache/block.h"
#include "engine/backend/device_buffer.h"

namespace ccinfer {
namespace engine {

// ---------------------------------------------------------------------------
// SequenceState — per-sequence state owned by the DeviceWorker.
// CUDA-free: uses only CPU-side types (BlockTable is CPU metadata).
// ---------------------------------------------------------------------------
struct SequenceState {
    SequenceId seq_id;
    std::vector<int32_t> prompt_tokens;
    int max_context_len = 0;
    int kv_written = 0;          // tokens already written into KV cache
    int prompt_processed = 0;    // tokens already consumed from prompt
    BlockTable block_table;
    bool aborted = false;
};

// ---------------------------------------------------------------------------
// PhysicalBatch — GPU-ready data for ModelRunner::execute.
// DeviceBuffer implies CUDA allocation, but the struct itself has no raw
// CUDA types in its interface.
// ---------------------------------------------------------------------------
struct PhysicalBatch {
    int num_tokens = 0;
    DeviceBuffer<int32_t> token_ids;       // [num_tokens]
    DeviceBuffer<int32_t> positions;        // [num_tokens]
    DeviceBuffer<int32_t> slot_mapping;     // [num_tokens]

    int batch_size = 0;
    int max_blocks_per_seq = 0;
    DeviceBuffer<int32_t> block_table;      // [batch_size, max_blocks_per_seq]

    DeviceBuffer<int32_t> query_start_loc;  // [batch_size + 1]
    DeviceBuffer<int32_t> context_lens;     // [batch_size]

    std::vector<int> item_indices;          // maps physical seq → WorkItem index
};

}  // namespace engine
}  // namespace ccinfer
