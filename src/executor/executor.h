#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/result.h"
#include "base/types.h"

namespace ccinfer {

namespace asio = boost::asio;

class Executor {
public:
    virtual ~Executor() = default;

    virtual Result<void> init(const std::string& model_path) = 0;
    virtual void shutdown() = 0;

    virtual asio::awaitable<Result<CreateSequenceResult>> create_sequence(
        std::vector<int32_t> prompt_tokens, int max_context_len) = 0;
    virtual asio::awaitable<Result<SuspendSequenceResult>> suspend_sequence(
        SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len) = 0;
    virtual asio::awaitable<Result<void>> release_sequence(SequenceId seq_id) = 0;
    virtual asio::awaitable<Result<void>> abort_sequence(SequenceId seq_id) = 0;
    virtual asio::awaitable<Result<BatchResult>> execute_batch(ScheduledBatch batch) = 0;

    virtual EngineCapacity capacity() const = 0;

    static std::unique_ptr<Executor> create(boost::asio::io_context& io);
};

}  // namespace ccinfer
