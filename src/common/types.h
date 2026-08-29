#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/experimental/channel.hpp>
#include <boost/system/error_code.hpp>

#include "cache/block.h"
#include "common/error_code.h"

namespace ccinfer {

namespace asio = boost::asio;

// Shared types used by Executor, DeviceWorker, Scheduler, and HTTP. This file
// also contains the HTTP request boundary, so it depends on the lightweight
// Boost.Asio channel/executor types but not on CUDA or model implementations.
using SequenceId = uint64_t;

enum class ForwardMode : uint8_t { Prefill, Decode, Mixed };
enum class WorkKind : uint8_t { PrefillChunk, DecodeOneToken };

// Lifecycle of a request in the Scheduler. A waiting request has no executor
// sequence yet; admission materializes that sequence in its RequestState.
enum class RequestStatus : uint8_t { Active, Finished, Cancelled, Failed };

struct TokenSpan {
    int start = 0;
    int length = 0;
};

struct PrefillChunk {
    SequenceId seq_id = 0;
    TokenSpan prompt_span;
    std::optional<int> expected_context_len;
    bool needs_sample = false;    // only true for the final chunk
    std::vector<int32_t> tokens;  // actual token ids for this chunk
    BlockTable block_table;       // scheduler-owned logical table mirror
};

struct DecodeOneToken {
    SequenceId seq_id = 0;
    int32_t input_token = 0;
    std::optional<int> expected_context_len;
    // false = bootstrap decode: sample from the last cached prompt token and
    // write no new KV / allocate no new block.
    bool write_kv = true;
    bool late_bind = false;
    BlockTable block_table;  // scheduler-owned logical table mirror
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
    int32_t eos_token_id = -1;  // token that ends generation, -1 = disabled
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

struct GeneratedToken {
    int32_t token_id = -1;
    bool has_token = false;
    bool finished = false;
};

using TokenChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<GeneratedToken>)>;

// HTTP -> Scheduler request boundary.
// TokenSink lets Scheduler post tokens/events back to the HTTP io_context.
struct TokenSink {
    asio::any_io_executor executor;
    std::weak_ptr<TokenChannel> channel;
    std::function<void()> on_send_failed;
};

struct SchedulerRequest {
    std::string request_id;
    std::vector<int32_t> prompt_tokens;
    SamplingParams sampling;
    int max_context_len = 0;  // 0 means use EngineConfig::default_max_context_len.
    TokenSink sink;
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
