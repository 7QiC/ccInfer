#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>

#include "base/channel.h"
#include "base/result.h"
#include "base/types.h"

namespace ccinfer {

namespace asio = boost::asio;

struct Config;

class Executor {
public:
    virtual ~Executor() = default;

    virtual Result<void> init(const Config& config) = 0;
    virtual void shutdown() = 0;

    virtual asio::awaitable<Result<AdmitSequenceResult>> admit_sequence(
        std::vector<int32_t> prompt_tokens, int max_context_len,
        SequenceInitialState initial_state = {}) = 0;
    virtual asio::awaitable<Result<SuspendSequenceResult>> suspend_sequence(
        SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len) = 0;
    virtual asio::awaitable<Result<void>> release_sequence(SequenceId seq_id) = 0;
    virtual asio::awaitable<Result<void>> abort_sequence(SequenceId seq_id) = 0;
    // Dispatch is non-blocking: the returned channel is the completion future
    // for this batch. EngineCore owns the in-flight bookkeeping and decides
    // when to await the future.
    virtual Result<BatchFuture> execute_batch(ScheduledBatch batch) = 0;
    virtual asio::awaitable<Result<BatchResult>> collect_batch(BatchFuture future) = 0;

    virtual Capacity capacity() const = 0;

    static std::unique_ptr<Executor> create(boost::asio::io_context& io);
};

}  // namespace ccinfer
