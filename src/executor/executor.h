#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>

#include "base/channel.h"
#include "base/error.h"
#include "base/types.h"

namespace ccinfer {

namespace asio = boost::asio;

struct EngineConfig;
struct ModelConfig;

class Executor {
public:
    virtual ~Executor() = default;

    virtual Result<void> init(const std::string& model_path, const ModelConfig& model,
                              const EngineConfig& engine) = 0;
    virtual void shutdown() = 0;

    // Dispatch is non-blocking. Implementations execute queued batches in FIFO
    // order; the returned channel is the completion future for this batch.
    virtual Result<BatchFuture> execute_batch(ScheduledBatch batch) = 0;
    virtual asio::awaitable<Result<BatchResult>> collect_batch(BatchFuture future) = 0;

    // Releases sequence-owned execution resources (e.g. active GDN state).
    // Default no-op keeps test doubles source-compatible. Scheduler calls this
    // only after the sequence is terminal/failed and its authoritative
    // execution_leases has reached zero.
    virtual void release_sequence(SequenceId) {}

    static std::unique_ptr<Executor> create(boost::asio::io_context& io);
};

}  // namespace ccinfer
