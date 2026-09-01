#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "executor/executor.h"

namespace ccinfer {

class Worker;
struct EngineConfig;
struct ModelConfig;

class SingleDeviceExecutor final : public Executor {
public:
    explicit SingleDeviceExecutor(boost::asio::io_context& io);
    ~SingleDeviceExecutor() override;

    Result<void> init(const std::string& model_path, const ModelConfig& model,
                      const EngineConfig& engine) override;
    void shutdown() override;

    Result<BatchFuture> execute_batch(ScheduledBatch batch) override;
    asio::awaitable<Result<BatchResult>> collect_batch(BatchFuture future) override;

private:
    std::unique_ptr<Worker> worker_;
};

}  // namespace ccinfer
