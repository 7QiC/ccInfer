#pragma once

#include <cstddef>
#include <deque>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>

#include "base/channel.h"
#include "base/types.h"
#include "config/engine_config.h"

namespace ccinfer {

class Executor;
class Scheduler;

// EngineCore owns the execution loop and FIFO completion collection; Worker
// remains the owner of device execution state.
class EngineCore {
public:
    EngineCore(boost::asio::io_context& io, Scheduler& scheduler, Executor& executor,
               EngineConfig config);

    void start();
    void request_shutdown();

private:
    struct InFlightBatch {
        ScheduledBatch batch;
        BatchFuture future;
    };

    boost::asio::awaitable<void> run();
    boost::asio::awaitable<void> collect_oldest();
    boost::asio::awaitable<void> drain_in_flight();

    boost::asio::io_context& io_;
    Scheduler& scheduler_;
    Executor& executor_;
    EngineConfig config_;
    std::deque<InFlightBatch> in_flight_;
};

}  // namespace ccinfer
