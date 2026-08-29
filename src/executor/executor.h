#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>

#include "common/channel.h"
#include "common/error_code.h"
#include "common/types.h"

namespace ccinfer {

namespace asio = boost::asio;

struct Config;

class Executor {
public:
    virtual ~Executor() = default;

    virtual Result<void> init(const Config& config) = 0;
    virtual void shutdown() = 0;

    // Dispatch is non-blocking. Implementations execute queued batches in FIFO
    // order; the returned channel is the completion future for this batch.
    virtual Result<BatchFuture> execute_batch(ScheduledBatch batch) = 0;
    virtual asio::awaitable<Result<BatchResult>> collect_batch(BatchFuture future) = 0;

    static std::unique_ptr<Executor> create(boost::asio::io_context& io);
};

}  // namespace ccinfer
