#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "base/error_code.h"

namespace ccinfer {

// Shared types used by Executor, DeviceWorker, Scheduler, and HTTP.
// No CUDA or ASIO dependency.
using SequenceId = uint64_t;

struct SequenceInitialState {
    int32_t last_token = -1;
    int tokens_generated = 0;
    int max_tokens = 0;
};

struct AdmitSequenceResult {
    SequenceId seq_id = 0;
    int prompt_processed = 0;
};

struct SuspendSequenceResult {
    int prompt_processed = 0;
};

enum class ForwardMode : uint8_t { Prefill, Decode, Mixed };
enum class WorkKind : uint8_t { PrefillChunk, DecodeOneToken };

// Lifecycle of a request in the Scheduler. A waiting request has no executor
// sequence yet; admission materializes that sequence in its RequestState.
enum class RequestStatus : uint8_t { Active, Finished, Cancelled, Failed };

// Lifecycle of an executor-owned logical sequence. The map entry itself is the
// allocated state; release removes the entry and abort leaves a tombstone.
enum class SequenceStatus : uint8_t { Active, Suspended, Aborted };

struct TokenSpan {
    int start = 0;
    int length = 0;
};

struct PrefillChunk {
    SequenceId seq_id = 0;
    TokenSpan prompt_span;
    std::optional<int> expected_context_len;
    bool needs_sample = false;  // only true for the final chunk
};

struct DecodeOneToken {
    SequenceId seq_id = 0;
    int32_t input_token = 0;
    std::optional<int> expected_context_len;
    bool late_bind = false;
    // false = bootstrap decode: sample from the last cached prompt token and
    // write no new KV / allocate no new block.
    bool write_kv = true;
};

using WorkItem = std::variant<PrefillChunk, DecodeOneToken>;

inline SequenceId work_sequence_id(const WorkItem& item) noexcept {
    return std::visit([](const auto& work) { return work.seq_id; }, item);
}

inline WorkKind work_kind(const WorkItem& item) noexcept {
    return std::holds_alternative<PrefillChunk>(item) ? WorkKind::PrefillChunk
                                                      : WorkKind::DecodeOneToken;
}

inline int work_token_count(const WorkItem& item) noexcept {
    if (const auto* prefill = std::get_if<PrefillChunk>(&item)) return prefill->prompt_span.length;
    return 1;
}

// Sampling parameters — user-facing, per-request.
// Shared by all sequences in a batch.
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
    bool stale = false;
};

struct BatchResult {
    uint64_t batch_id = 0;
    std::vector<WorkItemResult> items;
};

// Executor -> Worker logical sequence snapshot.  This intentionally contains
// no physical KV block table: block ids are worker/device-local resources.
struct SequenceSnapshot {
    SequenceId seq_id = 0;
    std::vector<int32_t> prompt_tokens;
    int max_context_len = 0;
    int kv_written = 0;
    int prompt_processed = 0;
    SequenceStatus status = SequenceStatus::Active;
};

// Worker -> Executor logical progress delta.  Physical block-table mutations
// stay in the Worker; Executor only updates sequence progress.
struct SequenceDelta {
    SequenceId seq_id = 0;
    int kv_tokens_committed = 0;
    int prompt_tokens_committed = 0;
};

struct WorkerBatchResult {
    BatchResult batch;
    std::vector<SequenceDelta> deltas;
};

struct GeneratedToken {
    int32_t token_id = -1;
    std::string text;
    bool has_token = false;
    bool finished = false;
};

struct Capacity {
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

}  // namespace ccinfer
