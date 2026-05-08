#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "common/types.h"
#include "engine/backend/device_buffer.h"
#include "engine/cache/block.h"

namespace ccinfer {
namespace engine {

class Model;

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
// ---------------------------------------------------------------------------
struct PhysicalBatch {
    int num_tokens = 0;
    std::unique_ptr<DeviceBuffer> token_ids;      // [num_tokens]
    std::unique_ptr<DeviceBuffer> positions;       // [num_tokens]
    std::unique_ptr<DeviceBuffer> slot_mapping;    // [num_tokens]

    int batch_size = 0;
    int max_blocks_per_seq = 0;
    std::unique_ptr<DeviceBuffer> block_table;     // [batch_size, max_blocks_per_seq]

    std::unique_ptr<DeviceBuffer> query_start_loc; // [batch_size + 1]
    std::unique_ptr<DeviceBuffer> context_lens;    // [batch_size]

    std::vector<size_t> item_indices;                 // maps physical seq → WorkItem index
};

}  // namespace engine
}  // namespace ccinfer
