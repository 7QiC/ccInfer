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

enum class ForwardMode : uint8_t { Prefill, Decode };
enum class WorkKind : uint8_t { PrefillChunk, DecodeOneToken };

struct TokenSpan {
    int start = 0;
    int length = 0;
};

struct PrefillChunk {
    SequenceId seq_id = 0;
    TokenSpan prompt_span;
    std::optional<int> expected_context_len;
};

struct DecodeOneToken {
    SequenceId seq_id = 0;
    int32_t input_token = 0;
    std::optional<int> expected_context_len;
};

using WorkItem = std::variant<PrefillChunk, DecodeOneToken>;

// Sampling parameters — user-facing, per-request.
// Phase 4.1: shared by all sequences in a batch.
// Future: per-sequence via std::vector<SamplingParams>.
struct SamplingParams {
    int max_tokens = 256;      // scheduler-level generation limit
    int top_k = 0;             // 0 = disabled (greedy)
    float top_p = 1.0f;        // >= 1.0 = disabled
    float temperature = 0.0f;  // <= 0 = greedy
    uint32_t seed = 42;
};

struct ScheduledBatch {
    uint64_t batch_id = 0;
    std::vector<WorkItem> items;
    SamplingParams sampling;
};

struct WorkItemResult {
    int item_index = 0;
    SequenceId seq_id = 0;
    WorkKind kind = WorkKind::PrefillChunk;
    std::vector<int32_t> sampled_tokens;
    int tokens_consumed = 0;
    bool eos = false;
};

struct BatchResult {
    uint64_t batch_id = 0;
    std::vector<WorkItemResult> items;
};

struct GeneratedToken {
    int32_t token_id = -1;
    std::string text;
    bool has_token = false;
    bool finished = false;
};

struct EngineCapacity {
    int max_sequences = 0;
    int active_sequences = 0;
    int free_blocks = 0;
    int max_blocks = 0;
};

}  // namespace ccinfer
