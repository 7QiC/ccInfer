#pragma once

#include <memory>
#include <string>
#include <vector>

#include "engine/executor/executor.h"

namespace ccinfer {
namespace engine {

class Worker;

class SingleDeviceExecutor final : public Executor {
public:
    explicit SingleDeviceExecutor(boost::asio::io_context& io);
    ~SingleDeviceExecutor() override;

    Result<void> init(const std::string& model_path) override;
    void shutdown() override;

    asio::awaitable<Result<CreateSequenceResult>> create_sequence(
        std::vector<int32_t> prompt_tokens, int max_context_len) override;
    asio::awaitable<Result<SuspendSequenceResult>> suspend_sequence(
        SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len) override;
    asio::awaitable<Result<void>> release_sequence(SequenceId seq_id) override;
    asio::awaitable<Result<void>> abort_sequence(SequenceId seq_id) override;
    asio::awaitable<Result<BatchResult>> execute_batch(ScheduledBatch batch) override;

    EngineCapacity capacity() const override;

private:
    std::unique_ptr<Worker> worker_;
};

}  // namespace engine
}  // namespace ccinfer
