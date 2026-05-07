#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/result.h"

namespace ccinfer {
namespace engine {

// ---------------------------------------------------------------------------
// Public types
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

struct EngineCapacity {
    int max_sequences = 0;
    int active_sequences = 0;
    int free_blocks = 0;
    int max_blocks = 0;
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
class Executor;

namespace asio = boost::asio;

class Engine {
public:
    explicit Engine(asio::io_context& io);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Result<void> init(const std::string& model_path);
    void shutdown();

    asio::awaitable<SequenceId> create_sequence(std::vector<int32_t> prompt_tokens,
                                                        int max_context_len);
    asio::awaitable<void> release_sequence(SequenceId seq_id);
    asio::awaitable<void> abort_sequence(SequenceId seq_id);
    EngineCapacity capacity() const;
    asio::awaitable<BatchResult> execute_batch(ScheduledBatch batch);

private:
    std::unique_ptr<Executor> executor_;
};

}  // namespace engine
}  // namespace ccinfer