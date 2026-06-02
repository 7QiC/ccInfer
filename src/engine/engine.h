#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/result.h"
#include "common/types.h"

namespace ccinfer {
namespace engine {

namespace asio = boost::asio;

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
class Executor;

class Engine {
public:
    explicit Engine(asio::io_context& io);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Result<void> init(const std::string& model_path);
    void shutdown();

    asio::awaitable<Result<CreateSequenceResult>> create_sequence(std::vector<int32_t> prompt_tokens,
                                                                  int max_context_len);
    asio::awaitable<Result<void>> release_sequence(SequenceId seq_id);
    asio::awaitable<Result<void>> abort_sequence(SequenceId seq_id);
    EngineCapacity capacity() const;
    asio::awaitable<Result<BatchResult>> execute_batch(ScheduledBatch batch);

private:
    std::unique_ptr<Executor> executor_;
};

}  // namespace engine
}  // namespace ccinfer
