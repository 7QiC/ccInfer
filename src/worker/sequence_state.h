#pragma once

#include <cstdint>
#include <vector>

#include "cache/block.h"
#include "common/types.h"

namespace ccinfer {

// SequenceState — worker-local execution view used by BatchTranslator.
// Logical fields are copied from Executor-owned SequenceSnapshot; block_table
// is worker/device-local metadata owned by the Worker.
struct SequenceState {
    SequenceId seq_id = 0;
    int prompt_len = 0;
    int max_context_len = 0;
    int kv_written = 0;        // tokens already written into KV cache
    int prompt_processed = 0;  // tokens already consumed from prompt
    BlockTable block_table;
    int32_t last_token = -1;
    int tokens_generated = 0;
    uint64_t parent_hash = 0;
    std::vector<int32_t> pending_tokens;
    int max_tokens = 0;
    bool finished = false;
};

}  // namespace ccinfer
