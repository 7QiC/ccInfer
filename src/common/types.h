#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/error_code.h"

namespace ccinfer {

// ---------------------------------------------------------------------------
// Shared types used by Engine, Executor, DeviceWorker, Scheduler, and HTTP.
// No CUDA or ASIO dependency.
// ---------------------------------------------------------------------------
using SequenceId = uint64_t;

enum class WorkKind : uint8_t { PrefillChunk, DecodeOneToken };

struct TokenSpan {
    int start;
    int length;
};

struct PrefillChunk {
    SequenceId seq_id;
    TokenSpan prompt_span;
    std::optional<int> expected_context_len;
};

struct DecodeOneToken {
    SequenceId seq_id;
    int32_t input_token;
    std::optional<int> expected_context_len;
};

using WorkItem = std::variant<PrefillChunk, DecodeOneToken>;

struct ScheduledBatch {
    uint64_t batch_id;
    std::vector<WorkItem> items;
};

struct WorkItemResult {
    int item_index;
    SequenceId seq_id;
    WorkKind kind;
    std::vector<int32_t> sampled_tokens;
    int tokens_consumed;
    bool eos = false;
};

struct BatchResult {
    uint64_t batch_id;
    std::vector<WorkItemResult> items;
};

struct GeneratedToken {
    int32_t token_id = -1;
    std::string text;
    bool has_token = false;
    bool finished = false;
    ErrorCode error = ErrorCode::Ok;
};

struct EngineCapacity {
    int max_sequences = 0;
    int active_sequences = 0;
    int free_blocks = 0;
    int max_blocks = 0;
};

}  // namespace ccinfer
