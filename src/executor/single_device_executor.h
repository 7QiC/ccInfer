#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "executor/executor.h"

namespace ccinfer {

class Worker;

class SingleDeviceExecutor final : public Executor {
public:
    explicit SingleDeviceExecutor(boost::asio::io_context& io);
    ~SingleDeviceExecutor() override;

    Result<void> init(const Config& config) override;
    void shutdown() override;

    Result<BatchFuture> execute_batch(ScheduledBatch batch) override;
    asio::awaitable<Result<BatchResult>> collect_batch(BatchFuture future) override;

private:
    std::unique_ptr<Worker> worker_;
};

}  // namespace ccinfer
